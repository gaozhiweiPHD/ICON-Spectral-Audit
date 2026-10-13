#!/usr/bin/env python3
"""Run preserved comprehensive evaluators with a new figure/output directory."""
import argparse,importlib.util,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--experiment",required=True,choices=["advection","burgers","allen_cahn","advection_frequency8","burgers_corrupted"])
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args()
out=args.output_dir.resolve()
if out.exists() and any(out.iterdir()): raise FileExistsError("output directory must be empty")
out.mkdir(parents=True,exist_ok=True)
os.environ.pop("ICON_NUMERICS_ONLY",None); os.environ["MPLBACKEND"]="Agg"
paths={"advection":"experiments/advection/evaluation/evaluate_advection.py","burgers":"experiments/burgers/notebooks/icon_burgers.ipynb",
"allen_cahn":"experiments/allen_cahn/evaluation/evaluate_allen_cahn.py","advection_frequency8":"experiments/advection/bandwidth_ablation/evaluate_mode_ablation.py",
"burgers_corrupted":"experiments/burgers/corrupted_prompt/evaluate_corrupted_prompt.py"}
if args.experiment=="burgers":
    # The historical clean evaluator is a notebook. Use the verified runtime
    # for prediction/JVP diagnostics; do not execute the entire notebook.
    raise SystemExit("Use scripts/revision/run_prompt_variability.py --pde burgers for clean inference/JVPs and scripts/plot/plot_current_burgers.py for archived plots. The full notebook is preserved but not certified as a CLI.")
path=ROOT/paths[args.experiment]
spec=importlib.util.spec_from_file_location("icon_full_evaluator",path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.fig_dir=str(out)
if args.experiment=="advection": module.eval_advection(module.find_latest_checkpoint(module.ckpt_dir))
elif args.experiment=="allen_cahn": module.eval_allen_cahn_paper_style(module.CKPT_PATH)
elif args.experiment=="advection_frequency8": module.main()
else: module.plot_all_figures(module.compute_all_diagnostics())
