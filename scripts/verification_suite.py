#!/usr/bin/env python3
"""Run bounded representative checks; numeric and plotting interpreters are explicit."""
import argparse,csv,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--numerical-python",default=sys.executable)
parser.add_argument("--plot-python",default=sys.executable)
args=parser.parse_args()
os.environ["ICON_NUMERICS_ONLY"]="1"; os.environ["MPLBACKEND"]="Agg"
out=ROOT/"outputs/verification"; out.mkdir(parents=True,exist_ok=True)
reports=[]
def run(name,python,arguments,timeout=600):
    start=time.perf_counter()
    try:
        environment=os.environ.copy()
        if name=="solver_convergence": environment["JAX_PLATFORMS"]="cpu"
        proc=subprocess.run([python,*arguments],cwd=ROOT,text=True,capture_output=True,timeout=timeout,env=environment)
        rc=proc.returncode; stdout=proc.stdout; stderr=proc.stderr
    except subprocess.TimeoutExpired as e:
        rc=124; stdout=""; stderr="Bounded check timed out"
    # Do not export absolute interpreter or scratch/checkpoint paths.
    for old in (str(ROOT),str(out),args.numerical_python,args.plot_python):
        stdout=stdout.replace(old,"RELEASE"); stderr=stderr.replace(old,"RELEASE")
    stdout=re.sub(r'/(?:users|oscar/(?:data|home|scratch))/[^\s"]+',"ENVIRONMENT_PATH",stdout)
    stderr=re.sub(r'/(?:users|oscar/(?:data|home|scratch))/[^\s"]+',"ENVIRONMENT_PATH",stderr)
    report={"name":name,"returncode":rc,"seconds":time.perf_counter()-start,
            "arguments":[str(s).replace(str(ROOT),"RELEASE") for s in arguments],
            "stdout":stdout[-10000:],"stderr":stderr[-6000:]}
    reports.append(report); print(name,rc,round(report["seconds"],2),flush=True)
    return rc
run("numerical_environment",args.numerical_python,["scripts/environment_report.py",str(out/"numerical_environment.json")])
run("plot_environment",args.plot_python,["scripts/environment_report.py",str(out/"plot_environment.json")])
run("numerical_dependency_check",args.numerical_python,["scripts/check_project_dependencies.py","--profile","numerical"])
run("plot_dependency_check",args.plot_python,["scripts/check_project_dependencies.py","--profile","plot"])
run("solver_convergence",args.numerical_python,["scripts/revision/run_solver_convergence.py","--samples","4","--seed","20261004","--output",str(out/"solver_convergence.csv")],240)
run("model_and_spectral_reproduction",args.numerical_python,["scripts/verify_models.py","--output",str(out/"model_checks.json")],1000)
run("synthetic_samples",args.numerical_python,["scripts/generate_samples.py","--output-dir",str(out/"samples")],240)
if (out/"samples/PROVENANCE.json").exists():
    for p in (out/"samples").iterdir(): shutil.copy2(p,ROOT/"data/samples"/p.name)
run("archived_figure_redraws",args.plot_python,["scripts/reproduce_plots.py","--output-dir",str(out/"plots")],400)
run("figure_asset_export",args.plot_python,["scripts/export_figures.py","--output-dir",str(out/"exported")])
# Compare the targeted reference-solver check separately from model reproduction.
if (out/"solver_convergence.csv").exists():
    with (out/"solver_convergence.csv").open() as f: measured=list(csv.DictReader(f))
    with (ROOT/"experiments/revision/solver_convergence/solver_convergence.csv").open() as f: archived=list(csv.DictReader(f))
    comparisons=[]
    for old,new in zip(archived,measured):
        comparisons.append({"PDE":new["PDE"],"sample":new["sample"],"archived_time":float(old["baseline_vs_time_refined_rel_l2"]),
            "measured_time":float(new["baseline_vs_time_refined_rel_l2"]),"archived_space_time":float(old["baseline_vs_space_time_refined_rel_l2"]),
            "measured_space_time":float(new["baseline_vs_space_time_refined_rel_l2"])})
    (out/"solver_comparison.json").write_text(json.dumps({"comparisons":comparisons,"warning":"float32 backend differences; this check is terminal-state only, not tangent-map convergence"},indent=2)+"\n")
target=ROOT/"results/verification"
for name in ("numerical_environment.json","plot_environment.json","model_checks.json","solver_convergence.csv","solver_comparison.json"):
    if (out/name).exists(): shutil.copy2(out/name,target/name)
if (out/"plots/plot_verification.json").exists():
    shutil.copy2(out/"plots/plot_verification.json",target/"plot_verification.json")
report={"checks":reports,"failed_checks":[r["name"] for r in reports if r["returncode"]!=0],
    "scope":"representative reproduction only; no full retraining, full sweep, bandwidth control evaluation or reverse-Jacobian feature-map recertification"}
(target/"verification_report.json").write_text(json.dumps(report,indent=2)+"\n")
print("Failed checks",report["failed_checks"],flush=True)
# Run audit after all exported reports exist; its report is a separate deliverable.
run("security_and_manifest",args.plot_python,["scripts/audit_release.py"])
