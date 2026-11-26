#!/usr/bin/env python3
"""Write a scratch variant of a space file (never committed) for use with --spaces-dir.

  make_variant_space.py --src gauntlet/spaces/Op.space.json --out gauntlet/scratch_x \
      --axis 'd={"values":[64,72,80]}' --derive 'dv=128 if d == 192 else d' \
      --meta 'engine_variant={"id":"a16","knobs":{"hipkernel.sdpa.bwd.accumulator_type":"a16"}}'

--out must sit next to gauntlet/spaces/ (the template path is relative to it). The output file
keeps the source name, so the space id is unchanged and --filter works as before. Delete the
directory afterwards. Changes to tensors/derive that depend on the op need hand edits; copy this
script and add them (see lessons.md, "Widening the net").
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--axis", action="append", default=[], help='name=<json axis>')
ap.add_argument("--derive", action="append", default=[], help="name=<expression string>")
ap.add_argument("--meta", action="append", default=[], help='key=<json> into case.metadata')
a = ap.parse_args()
s = json.load(open(a.src))
for x in a.axis:
    k, v = x.split("=", 1); s["axes"][k] = json.loads(v)
for x in a.derive:
    k, v = x.split("=", 1); s.setdefault("derive", {})[k] = v
for x in a.meta:
    k, v = x.split("=", 1); s["case"].setdefault("metadata", {})[k] = json.loads(v)
s["description"] = "scratch variant (not committed): " + os.path.basename(a.src)
os.makedirs(a.out, exist_ok=True)
json.dump(s, open(os.path.join(a.out, os.path.basename(a.src)), "w"), indent=1)
print("wrote", os.path.join(a.out, os.path.basename(a.src)))
