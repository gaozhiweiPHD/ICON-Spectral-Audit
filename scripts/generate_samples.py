#!/usr/bin/env python3
"""Small synthetic prompts from the preserved generators, without training."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from models.revision_runtime import load_runtime, default_parameters

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/samples")
    parser.add_argument("--seed",type=int,default=20261013)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    metadata={}
    for index,pde in enumerate(("advection","burgers","allen_cahn")):
        r=load_runtime(pde); parameter=default_parameters(pde)[2]
        path=args.output_dir/(pde+".npz")
        if path.exists(): raise FileExistsError(path)
        key=r.jax.random.PRNGKey(args.seed+index)
        if pde=="advection":
            raw=r.generate_shift_raw_data(key,1,6,100,16,(parameter,parameter+1e-8),
                r.dt,r.fourier_decay,r.initial_amp,r.bias_scale)
        elif pde=="burgers":
            raw=r.generate_burgers_raw_data(key,1,6,100,16,(parameter,parameter+1e-8),
                r.T_final,r.fourier_decay,r.bias_scale,r.initial_amp,r.n_solver_steps)
        else:
            raw=r.generate_allen_cahn_raw_data(key,1,6,100,16,(parameter,parameter+1e-8),
                r.T_final,r.fourier_decay,r.bias_scale,r.initial_amp,r.n_solver_steps)
        values={k:np.asarray(getattr(raw,k)) for k in ("cond_k","cond_v","qoi_k","qoi_v")}
        if not all(np.isfinite(v).all() for v in values.values()): raise ValueError(pde)
        np.savez(path,**values)
        metadata[pde]={"seed":args.seed+index,"parameter":parameter,"pairs":6,
            "n_grid":100,"jax_version":r.jax.__version__,"backend":r.jax.default_backend(),"source":"preserved evaluation generator","purpose":"new release sample; not historical training data"}
    (args.output_dir/"PROVENANCE.json").write_text(json.dumps(metadata,indent=2)+"\n")
if __name__=="__main__": main()
