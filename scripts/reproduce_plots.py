#!/usr/bin/env python3
"""Redraw available archived arrays and CSVs; never train or load a checkpoint."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args(); out=args.output_dir.resolve()
if out.exists() and any(out.iterdir()): raise FileExistsError("output directory must be empty")
out.mkdir(parents=True,exist_ok=True)
os.environ["MPLBACKEND"]="Agg"
commands=[]
def run(path,*arguments):
    command=[sys.executable,str(ROOT/path),*map(str,arguments)]
    print(path,flush=True)
    proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    # Record portable command and bounded diagnostic output.
    command[0]="python"; command[1]=path
    commands.append({"command":command,"returncode":proc.returncode,"stdout":proc.stdout.replace(str(ROOT),".").replace(str(out),"OUTPUT"),"stderr":proc.stderr.replace(str(ROOT),".").replace(str(out),"OUTPUT")})
run("scripts/plot/replot_square_feature_maps.py","--output-dir",out/"feature_maps")
run("scripts/plot/plot_current_burgers.py","--output-dir",out/"burgers")
run("scripts/plot/render_regime_comparisons.py","--output-dir",out/"regime")
for pde in ("advection","burgers","allen_cahn"):
    run("scripts/revision/plot_prompt_variability.py","--pde",pde,"--output-dir",out/"summaries")
    run("scripts/revision/plot_context_size.py","--pde",pde,"--output",out/(pde+"_context_size.png"))
    run("scripts/revision/plot_training_proximity.py","--pde",pde,"--output",out/(pde+"_training_proximity.png"))
run("scripts/revision/plot_demo_parameter_noise.py","--output",out/"advection_demo_parameter_noise.png")
run("scripts/revision/plot_solver_convergence.py","--output",out/"solver_convergence.png")
# Array-preserving legacy redraws (including filtered manuscript spectra).
run("scripts/plot/redraw_detail_panels.py","--output-dir",out/"details")
for item in commands:
    item["command"]=[s.replace(str(out),"OUTPUT") for s in item["command"]]
report={"commands":commands,"failed":sum(c["returncode"]!=0 for c in commands),
        "scope":"redraws from archived arrays; exact raster/layout equality is not asserted"}
(out/"plot_verification.json").write_text(json.dumps(report,indent=2)+"\n")
print("Failed commands:",report["failed"],flush=True)
raise SystemExit(bool(report["failed"]))
