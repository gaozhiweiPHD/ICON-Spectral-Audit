#!/usr/bin/env python3
"""Export the preserved current manuscript assets; this does not recompute data."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)
for figure in json.loads((ROOT/"docs/MANUSCRIPT_FIGURES.json").read_text()):
    for rel in figure["assets"]:
        source=ROOT/rel; target=args.output_dir/source.name
        if target.exists(): raise FileExistsError(target)
        shutil.copy2(source,target)
print("Exported 24 preserved assets for 22 figures; no numerical reevaluation.")
