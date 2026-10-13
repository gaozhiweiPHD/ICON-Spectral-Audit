#!/usr/bin/env python3
"""Regenerate one archived repeated-prompt row per PDE with all 16 Fourier pairs."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from models.revision_runtime import load_runtime,load_main_params,make_prompt,prompt_metrics
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/model_checks.json")
    parser.add_argument("--relative-tolerance",type=float,default=0.005)
    parser.add_argument("--absolute-tolerance",type=float,default=0.00005)
    args=parser.parse_args(); reports=[]; all_pass=True
    for pde in ("advection","burgers","allen_cahn"):
        start=time.perf_counter()
        path=ROOT/"results/revision/prompt_variability"/(pde+"_prompt_stats.csv")
        with path.open() as f: reference=next(csv.DictReader(f))
        r=load_runtime(pde); params=load_main_params(pde,r)
        parameter=float(reference["parameter"])
        _,data,label=make_prompt(r,pde,parameter,int(reference["prompt_seed"]),int(reference["query_seed"]),5)
        measured=prompt_metrics(r,pde,params,data,label,parameter,16)
        checks={}
        for key,value in measured.items():
            old=float(reference[key])
            passed=bool(np.isnan(old) and np.isnan(value) or np.isfinite(value) and np.isclose(value,old,rtol=args.relative_tolerance,atol=args.absolute_tolerance))
            checks[key]={"reference":None if np.isnan(old) else old,"measured":None if np.isnan(value) else value,"passed":passed}
            all_pass &= passed
        reports.append({"pde":pde,"parameter":parameter,"prompt_seed":int(reference["prompt_seed"]),"query_seed":int(reference["query_seed"]),
            "seconds":time.perf_counter()-start,"metrics":checks,"backend":r.jax.default_backend()})
        print(pde,checks,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"passed":all_pass,"rtol":args.relative_tolerance,"atol":args.absolute_tolerance,
        "scope":"one archived row per PDE; full 16-mode JVP diagnostics; no training or full-sweep certification","cases":reports},indent=2)+"\n")
    if not all_pass: raise SystemExit(1)
if __name__=="__main__": main()
