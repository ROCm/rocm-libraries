# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ABI section of a review's checks (lessons L-527, L-579): the abi.h counters, the C struct layouts and exported
functions in include/rocke/ (public and generated headers) that a PR head changes, relative to its merge base with its
base branch (BASE, a ref: develop's by default), and abi.h against portable_ir/src/abi.py. With MERGE (the merged
tree's commit; "-" for none), each counter's homes on the merge across all files: abi.h's #define, its newest history
line, abi.py's mirror and the highest layout canary (`ROCKE_ABI_VERSION >= N` static_asserts). Homes that disagree read
`HALF RENUMBER` (CI cannot see it: the shared engine then refuses to load, or a canary stops the build). Read-only.
Usage: python3 -I abi_scan.py REPO ROCKE_PATH HEAD [BASE [MERGE]]
Its last line is "ABI pass|fail|skip <detail>": fail for a HALF RENUMBER or any NOTE, a check flag (G9 also waits for
`rk abi-ok N` on a NOTE)."""
import pathlib
import subprocess
import sys

sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rkmon_lib import abi, gitops  # noqa: E402

repo, K, head = sys.argv[1:4]
base = sys.argv[4] if len(sys.argv) > 4 else gitops.DEVELOP_REF
merge = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] not in ("", "-") else ""
try:
    mb = gitops.git(repo, "merge-base", base, head).strip()
    c = abi.scan(repo, K, mb, head)
    dev = abi.develop_counters({"paths": {"repo": repo}, "rocke_path": K})
    homes = abi.homes(repo, K, merge) if merge else None
except (gitops.GitError, OSError, subprocess.SubprocessError) as exc:
    print(f"NOT RUN: {str(exc)[:200]}")
    print(f"ABI skip NOT RUN: {str(exc)[:160]}")
    sys.exit(0)
for lane, name in abi.DEFINES.items():
    cl = c["lanes"].get(lane)
    print(
        f"{name}: develop {dev.get(lane)}; "
        + (f"the PR bumps {cl['from']} -> {cl['to']}" if cl else "no bump")
    )
half = []
if homes is None:
    print("homes: not read (no merged tree)")
else:
    print(f"on the merge {merge[:10]}:")
    for line in abi.homes_lines(homes):
        print(line)
    half = [p for x in homes.values() for p in x["problems"]]
for line in c["notes"]:
    print("NOTE (G9 waits for `rk abi-ok N` after review):", line)
for line in c["info"]:
    print("INFO:", line)
for f in c["families"]:
    print(f"family {f['name']}: {f['kind']} {f.get('old', '-')} -> {f.get('new', '-')}")
if not (c["notes"] or c["info"]):
    print(
        "no struct layout or exported function change in include/rocke/ without a bump"
    )
if half:
    print(f"ABI fail HALF RENUMBER on the merge: {'; '.join(half)[:160]}")
elif c["notes"]:
    print(
        f"ABI fail {len(c['notes'])} note{'s' * (len(c['notes']) != 1)}: {c['notes'][0][:160]}"
    )
else:
    print(
        "ABI pass "
        + (
            ", ".join(
                f"{abi.DEFINES[k]} {v['from']} -> {v['to']}"
                for k, v in c["lanes"].items()
            )
            or "no ABI change without a bump"
        )
        + ("; homes agree on the merge" if homes else "")
    )
