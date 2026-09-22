"""Quantization and speculative-decoding ablations on the deterministic fixture."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from advanced import Int8Weight,speculative_generate
from engine import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]

def elapsed(fn,repeats=7):
    rows=[]; value=None
    for _ in range(repeats): start=time.perf_counter(); value=fn(); rows.append(time.perf_counter()-start)
    return value,rows

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True); args=parser.parse_args()
    out=ROOT/"helix/experiments"/args.run_id; out.mkdir(parents=True,exist_ok=False)
    model=TinyDecoder(vocab=64,width=32,heads=4,seed=7); matrices=("embed","q","k","v","out")
    float_bytes=sum(getattr(model,n).nbytes for n in matrices); errors=[]; quant_bytes=0
    for name in matrices:
        weight=getattr(model,name); packed=Int8Weight.from_float(weight); quant_bytes+=packed.bytes
        errors.append(float(np.max(np.abs(weight-packed.dequantize()))))
    prompt=[1,2,3]; expected=list(model.generate(prompt,32))
    baseline,base_times=elapsed(lambda:list(model.generate(prompt,32)))
    spec={}
    for label,draft in (("matched",TinyDecoder(64,32,4,7)),("mismatched",TinyDecoder(64,32,4,8))):
        (tokens,stats),times=elapsed(lambda:speculative_generate(model,draft,prompt,32,4))
        spec[label]={**stats,"exact_target_output":tokens==expected,"median_seconds":float(np.median(times)),"samples_seconds":times}
    script=Path(__file__).read_bytes(); config={"seed":7,"tokens":32,"depth":4,"repeats":7,"paid_spend":0,"script_sha256":hashlib.sha256(script).hexdigest()}
    metrics={"quantization":{"float_bytes":float_bytes,"int8_bytes":quant_bytes,"reduction_percent":100*(1-quant_bytes/float_bytes),"max_weight_error":max(errors)},
             "greedy_median_seconds":float(np.median(base_times)),"greedy_samples_seconds":base_times,"speculative":spec,"paid_spend":0,
             "interpretation":"tiny deterministic fixture; Python overhead and acceptance sensitivity, not production speed"}
    (out/"config.json").write_text(json.dumps(config,indent=2)); (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); (out/"advanced_benchmark.py").write_bytes(script)
    print(json.dumps(metrics,indent=2))
    if not all(row["exact_target_output"] for row in spec.values()): raise SystemExit(1)

if __name__=="__main__": main()
