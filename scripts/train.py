#!/usr/bin/env python3
"""Explicit training entry point. Full training has not been rerun for this release."""
import argparse,importlib.util,os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--experiment",required=True,choices=["advection","burgers","allen_cahn","advection_frequency8","burgers_corrupted"])
parser.add_argument("--run-root",type=Path,required=True,help="new empty directory for training checkpoints and outputs")
args=parser.parse_args()
output=args.run_root.resolve()
if output.exists() and any(output.iterdir()): raise FileExistsError("training run root must be empty")
output.mkdir(parents=True,exist_ok=True)
os.environ["ICON_RUN_ROOT"]=str(output)
os.environ["MPLBACKEND"]="Agg"
paths={"advection":"experiments/advection/training/train_advection.py","burgers":"experiments/burgers/training/train_burgers.py",
"allen_cahn":"experiments/allen_cahn/training/train_allen_cahn.py","advection_frequency8":"experiments/advection/bandwidth_ablation/train_frequency8.py",
"burgers_corrupted":"experiments/burgers/corrupted_prompt/train_corrupted_prompt.py"}
path=ROOT/paths[args.experiment]
sys.argv=[str(path)]
if args.experiment in {"burgers","burgers_corrupted"}:
    # Preserved scripts train at module level. Never import them for evaluation.
    runpy.run_path(str(path),run_name="__main__")
else:
    spec=importlib.util.spec_from_file_location("icon_training",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if args.experiment=="allen_cahn": module.train_allen_cahn()
    else: module.train_advection()
