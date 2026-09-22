"""Reproducible CPU numerical and scheduler benchmark."""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
from runtime import causal_attention
from scheduler import Request, simulate

ROOT = Path(__file__).resolve().parents[1]


def percentile(values, q): return float(np.percentile(np.asarray(values), q))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); args=parser.parse_args()
    out=ROOT/"helix/experiments"/args.run_id; out.mkdir(parents=True, exist_ok=False)
    rng=np.random.default_rng(0); timings=[]
    for length in (16,64,256):
        q=rng.normal(size=(1,8,length,64)).astype(np.float32); k=rng.normal(size=q.shape).astype(np.float32); v=rng.normal(size=q.shape).astype(np.float32)
        causal_attention(q,k,v)
        samples=[]
        for _ in range(7):
            start=time.perf_counter(); causal_attention(q,k,v); samples.append(time.perf_counter()-start)
        timings.append({"length":length,"median_seconds":float(np.median(samples)),"samples_seconds":samples})
    jobs=[Request(f"r{i}",i//3,16+(i*37)%240,8+(i*13)%64) for i in range(30)]
    policies={p:simulate(jobs,p,budget=64,chunk=32) for p in ("static","adaptive")}
    for result in policies.values():
        result["p95_ttft"]=percentile([r["ttft"] for r in result["requests"]],95)
        result["p95_tpot"]=percentile([r["tpot"] for r in result["requests"]],95)
    config={"seed":0,"attention_shapes":[16,64,256],"scheduler_budget":64,"chunk":32,"python":platform.python_version(),"numpy":np.__version__,"paid_spend":0}
    script=Path(__file__).read_bytes(); config["script_sha256"]=hashlib.sha256(script).hexdigest()
    (out/"config.json").write_text(json.dumps(config,indent=2)); (out/"benchmark.py").write_bytes(script)
    metrics={"attention":timings,"policies":policies,"adaptive_throughput_gain_percent":100*(policies["adaptive"]["tokens_per_tick"]/policies["static"]["tokens_per_tick"]-1),"interpretation":"CPU kernel timing plus deterministic scheduler simulation; not production serving evidence","paid_spend":0}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))


if __name__=="__main__": main()
