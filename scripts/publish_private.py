#!/usr/bin/env python3
"""Create and push only a NEW PRIVATE repository after completed local review."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--owner",help="account/organization; defaults to authenticated user")
args=parser.parse_args()
def run(command,check=True):
    return subprocess.run(command,cwd=ROOT,text=True,capture_output=True,check=check)
if run(["gh","auth","status"],False).returncode: raise SystemExit("GitHub CLI is not authenticated. Run gh auth login in your own terminal; do not share tokens.")
user=json.loads(run(["gh","api","user"]).stdout)
name=(args.owner or user["login"])+"/ICON-Spectral-Audit"
if run(["gh","repo","view",name,"--json","nameWithOwner"],False).returncode==0: raise SystemExit("Name conflict: "+name+" exists. It was not modified.")
if run(["git","remote","get-url","origin"],False).returncode==0: raise SystemExit("Local origin already exists; review it manually instead of overwriting.")
if run(["git","status","--porcelain"]).stdout.strip(): raise SystemExit("Commit the reviewed candidate before uploading.")
if json.loads((ROOT/"docs/SECURITY_AUDIT.json").read_text())["findings"]: raise SystemExit("Unresolved security findings; upload aborted.")
proc=run(["gh","repo","create",name,"--private","--source",str(ROOT),"--remote","origin","--description","Reproducibility materials for Spectral Audit of In-Context Operator Networks"],False)
if proc.returncode: raise SystemExit("Creation failed. No existing repository was overwritten.")
meta=json.loads(run(["gh","repo","view",name,"--json","url,isPrivate,nameWithOwner"]).stdout)
if meta["isPrivate"] is not True: raise SystemExit("Visibility check failed; push aborted.")
run(["git","push","-u","origin","main"])
remote=run(["git","ls-remote","origin","refs/heads/main"]).stdout.split()[0]
local=run(["git","rev-parse","HEAD"]).stdout.strip()
if remote!=local: raise SystemExit("Remote commit verification failed.")
print(json.dumps({"url":meta["url"],"private":True,"commit":local,"nameWithOwner":meta["nameWithOwner"]},indent=2))
print("STOP: public visibility requires explicit author approval. This script cannot make the repository public.")
