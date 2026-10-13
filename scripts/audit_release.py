#!/usr/bin/env python3
"""Bounded release inventory, credential patterns, safe NPZs and SHA-256 manifests."""
import ast,csv,hashlib,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def main():
    paths=sorted(p for p in ROOT.rglob("*") if p.is_file() and not any(part in
        {".git","__pycache__","outputs","local",".venv"} for part in p.relative_to(ROOT).parts))
    findings=[]; dataset=[]; allfiles=[]
    rules={"github_token":rb"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{40,})",
           "aws_key":rb"(?:AKIA|ASIA)[A-Z0-9]{16}",
           "private_key":rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
           "api_assignment":rb"(?i)(?:api_key|access_token|auth_token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]"}
    for p in paths:
        rel=p.relative_to(ROOT).as_posix(); size=p.stat().st_size
        if p.is_symlink(): findings.append({"path":rel,"type":"symlink"})
        if size>100*1024**2: findings.append({"path":rel,"type":"GitHub size limit"})
        if p.suffix in {".py",".md",".json",".txt",".csv",".ipynb",".yml",".yaml"} or p.name in {"LICENSE","CITATION.cff"}:
            data=p.read_bytes()
            for label,pattern in rules.items():
                if re.search(pattern,data): findings.append({"path":rel,"type":label})
            if re.search(rb"/(?:oscar/(?:data|home|scratch)|users|home)/",data):
                findings.append({"path":rel,"type":"machine-specific absolute path"})
        if p.suffix==".py":
            try: ast.parse(p.read_text(),filename=rel)
            except SyntaxError: findings.append({"path":rel,"type":"Python syntax"})
        if p.suffix==".npz":
            try:
                with np.load(p,allow_pickle=False) as d:
                    for key in d.files:
                        a=d[key]
                        if a.dtype.kind in "US":
                            raw=" ".join(str(v) for v in a.flat)
                            if re.search(r"/(?:oscar/(?:data|home|scratch)|users|home)/",raw):
                                findings.append({"path":rel,"type":"absolute path in NPZ metadata","field":key})
            except ValueError: findings.append({"path":rel,"type":"pickle/object NPZ"})
        sha=digest(p)
        if p.suffix in {".npz",".msgpack",".csv"} and p.name not in {"FILE_INVENTORY.csv","SOURCE_INVENTORY.csv","DATASETS.csv"}:
            dataset.append({"path":rel,"bytes":size,"sha256":sha,
                "role":"trained model parameters" if p.suffix==".msgpack" else "new synthetic sample" if rel.startswith("data/samples/") else "archived synthetic experiment/diagnostics"})
        if p.name in {"FILE_INVENTORY.csv","SECURITY_AUDIT.json","MANIFEST.sha256","DATASETS.csv"}:
            allfiles.append({"path":rel,"bytes":"","sha256":""})
        else:
            allfiles.append({"path":rel,"bytes":size,"sha256":sha})
    docs=ROOT/"docs"; docs.mkdir(exist_ok=True)
    with (docs/"FILE_INVENTORY.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["path","bytes","sha256"]); w.writeheader(); w.writerows(allfiles)
    with (ROOT/"data/DATASETS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["path","bytes","sha256","role"]); w.writeheader(); w.writerows(dataset)
    (ROOT/"data/MANIFEST.sha256").write_text("".join(r["sha256"]+"  "+r["path"]+"\n" for r in dataset))
    report={"files_scanned":len(paths),"bytes":sum(p.stat().st_size for p in paths),"largest_file_bytes":max(p.stat().st_size for p in paths),
            "findings":findings,"scope":"Credential regexes on text; safe NPZ metadata; file sizes and syntax. No credential values are reported. Binary weights/PDF/images are not exhaustively analyzed; absence of regex findings is not a guarantee.",
            "manifest_note":"FILE_INVENTORY lists all scanned files; generated inventory/security/data manifest rows have blank size/hash to avoid circular hashes. Totals describe files before report regeneration."}
    (docs/"SECURITY_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if findings: raise SystemExit(1)
if __name__=="__main__": main()
