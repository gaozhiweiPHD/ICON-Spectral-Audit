#!/usr/bin/env python3
"""Validate installed dependency closure for a selected project profile."""
import argparse,importlib.metadata as metadata,json
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.utils import canonicalize_name
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile",choices=["numerical","plot"],required=True)
args=parser.parse_args()
todo=["jax","jaxlib","flax","optax","einshape","numpy","scipy","msgpack","orbax-checkpoint"] if args.profile=="numerical" else ["numpy","matplotlib","Pillow"]
visited={}; failures=[]; environment=default_environment(); environment["extra"]=""
while todo:
    name=canonicalize_name(todo.pop())
    if name in visited: continue
    try: dist=metadata.distribution(name)
    except metadata.PackageNotFoundError:
        failures.append({"package":name,"error":"missing"}); visited[name]=None; continue
    visited[name]=dist.version
    for raw in dist.requires or []:
        r=Requirement(raw)
        if r.marker and not r.marker.evaluate(environment): continue
        try: version=metadata.version(r.name)
        except metadata.PackageNotFoundError:
            failures.append({"package":name,"requirement":str(r),"error":"missing"}); continue
        if r.specifier and version not in r.specifier:
            failures.append({"package":name,"requirement":str(r),"installed":version})
        todo.append(r.name)
print(json.dumps({"profile":args.profile,"versions":visited,"failures":failures},indent=2))
raise SystemExit(bool(failures))
