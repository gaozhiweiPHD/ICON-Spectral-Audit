#!/usr/bin/env python3
"""Report relevant package versions without credentials or machine paths."""
import importlib.metadata as metadata,json,platform,sys
from pathlib import Path
report={"python":platform.python_version(),"packages":{},"torch_used_by_project":False}
for name in ("jax","jaxlib","flax","optax","einshape","numpy","scipy","matplotlib","Pillow","torch","msgpack","orbax-checkpoint"):
    try: report["packages"][name]=metadata.version(name)
    except metadata.PackageNotFoundError: report["packages"][name]=None
try:
    import jax
    report["jax_backend"]=jax.default_backend()
    report["jax_devices"]=[{"platform":d.platform,"kind":d.device_kind} for d in jax.devices()]
except ImportError: pass
text=json.dumps(report,indent=2)+"\n"
if len(sys.argv)>1: Path(sys.argv[1]).write_text(text)
else: print(text)
