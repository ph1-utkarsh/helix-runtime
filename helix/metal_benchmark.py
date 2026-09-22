"""Matched NumPy CPU and explicit MLX Metal attention comparison."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from runtime import causal_attention

ROOT=Path(__file__).resolve().parents[1]


def metal_attention(q,k,v):
    length=q.shape[-2]
    scores=(q @ mx.swapaxes(k,-1,-2)) / np.sqrt(q.shape[-1])
    mask=mx.arange(length)[None,:] <= mx.arange(length)[:,None]
    return mx.softmax(mx.where(mask,scores,mx.array(-1e9)),axis=-1) @ v


def timed(fn, repeats=9):
    values=[]
    for _ in range(repeats):
        start=time.perf_counter(); value=fn(); mx.eval(value); mx.synchronize(); values.append(time.perf_counter()-start)
    return values


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True); args=parser.parse_args()
    out=ROOT/"helix/experiments"/args.run_id; out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(0); rows=[]
    for length in (16,64,256):
        arrays=[rng.normal(size=(1,8,length,64)).astype(np.float32) for _ in range(3)]
        expected=causal_attention(*arrays); metal=[mx.array(x) for x in arrays]
        actual=np.array(metal_attention(*metal)); error=float(np.max(np.abs(expected-actual)))
        cpu=[]
        for _ in range(9):
            start=time.perf_counter(); causal_attention(*arrays); cpu.append(time.perf_counter()-start)
        gpu=timed(lambda:metal_attention(*metal))
        rows.append({"length":length,"max_abs_error":error,"cpu_median_seconds":float(np.median(cpu)),
                     "metal_median_seconds":float(np.median(gpu)),"cpu_samples":cpu,"metal_samples":gpu})
    script=Path(__file__).read_bytes(); config={"seed":0,"repeats":9,"mlx":mx.__version__,"paid_spend":0,
        "script_sha256":hashlib.sha256(script).hexdigest(),"device":str(mx.default_device())}
    (out/"config.json").write_text(json.dumps(config,indent=2)); (out/"metal_benchmark.py").write_bytes(script)
    metrics={"rows":rows,"all_within_1e-5":all(r["max_abs_error"]<=1e-5 for r in rows),
             "interpretation":"single-machine explicit Metal-vs-CPU primitive comparison; no CUDA claim","paid_spend":0}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
    if not metrics["all_within_1e-5"]: raise SystemExit(1)


if __name__=="__main__": main()
