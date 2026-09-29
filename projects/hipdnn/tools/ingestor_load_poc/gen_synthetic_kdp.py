# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Scale the packed gfx950_attention_dense bundle to N UKDs for descriptor-load timing.

Replica r of each shipped UKD gets a fresh UUID, a suffixed name, and
metadata.seqlen_q += r so its metadata tuple (the catalog key) stays unique per device.
Everything else -- provenance, signature, kpack toc_key -- is copied verbatim, so per-UKD
JSON size and shape match production. The kpack itself is copied unchanged: descriptor
enumeration never opens it. The output is serialized the way the source was (indented or
compact), so at N=840 it is byte-identical to the packer's output.

Usage: python gen_synthetic_kdp.py <packed arch_content/hip-kernel-provider> <N> <out root>
"""
import json
import shutil
import sys
import uuid
from pathlib import Path

BUNDLE = Path("gfx950/rocKE/gfx950_attention_dense")
KDP = "gfx950_attention_dense.kdp.json"


def main(src_root, n, out_root):
    src_root, out_root = Path(src_root), Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(src_root / "gfx950" / "kpack", out_root / "gfx950" / "kpack")
    (out_root / BUNDLE).mkdir(parents=True)
    for f in (src_root / BUNDLE).iterdir():
        # The provenance sidecar describes the shipped 840 only; nothing at load reads it.
        if f.name != KDP and not f.name.endswith(".provenance.json.gz"):
            shutil.copy2(f, out_root / BUNDLE / f.name)

    text = (src_root / BUNDLE / KDP).read_text()
    kdp = json.loads(text)
    base = kdp["kernelDescriptors"]
    out = []
    rng = uuid.UUID(int=0x5CA1E)  # deterministic namespace
    for i in range(n):
        r, src = divmod(i, len(base))
        u = base[src]
        if r:
            u = json.loads(json.dumps(u))
            u["id"] = str(uuid.uuid5(rng, f"{r}:{src}"))
            u["name"] = f"{u['name']}.r{r}"
            u["metadata"]["seqlen_q"] = u["metadata"]["seqlen_q"] + r
        out.append(u)
    kdp["kernelDescriptors"] = out
    # Same serializer settings as the packer's output.
    if "\n  " in text:
        body = json.dumps(kdp, indent=2)
    else:
        body = json.dumps(kdp, separators=(",", ":"))
    (out_root / BUNDLE / KDP).write_text(body + "\n")
    print(json.dumps({"n": n, "kdp_bytes": (out_root / BUNDLE / KDP).stat().st_size}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
