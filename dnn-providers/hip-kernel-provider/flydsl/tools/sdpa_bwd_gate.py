# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Gates for one SDPA backward object (gfx950): resources, #1087, R-MIR. Prints one JSON object.

    python tools/sdpa_bwd_gate.py --hsaco OBJ --greedy-dump TEXT [--spec KEY]

``OBJ`` is the delivered code object; ``TEXT`` is what LLVM printed on stderr under
``print-after=greedy`` during the compile that produced it. The generators call this
once per object, in a subprocess, so a gate edit never restamps an object.

* ``resources``: the manifest's resource row (``analyze.py``).
* ``static_1087``: the #1087 detectors (``analyze.py``, ``undefscan_issue1087.py``).
  Recorded, never a verdict.
* ``rmir``: ``rmir.py``'s lane scan of the first SGPR greedy dump of each kernel.
  ``verdict`` is ``red`` iff a finding sits on a register restored from a spill slot
  (the ``rmRedundantSpills`` signature) and ``--spec`` is not in ``rmir_known.csv``,
  ``known`` when it is, else ``ok``. Findings on other registers are counted in
  ``other_findings`` (T4: an undefined descriptor lane of the merged objects).

The exit status is nonzero only when a gate could not run (no greedy dump, a target
other than gfx950): the caller treats that as an error, not a verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LLVM_BIN", "/opt/rocm/llvm/bin")
import analyze  # noqa: E402
import rmir  # noqa: E402

RESOURCE_COLUMNS = (
    "vgpr",
    "agpr",
    "vgpr_total",
    "sgpr",
    "sgpr_spill",
    "vgpr_spill",
    "scratch_bytes",
    "lds_bytes",
    "occupancy",
)
_FUNCTION = re.compile(r"(?m)^# Machine code for function (\S+?):")


def rmir_known() -> set[str]:
    with (HERE / "rmir_known.csv").open(newline="", encoding="utf-8") as fh:
        return {row["spec"] for row in csv.DictReader(fh)}


def first_dump_per_kernel(text: str) -> list[tuple[str, str]]:
    """[(function, MIR)] for the first greedy dump of each function: the SGPR allocation."""
    first: dict[str, str] = {}
    for dump in rmir.split_dumps(text):
        match = _FUNCTION.search(dump)
        first.setdefault(match.group(1) if match else "", dump)
    return list(first.items())


def gate(hsaco: Path, greedy_dump: Path, spec: str | None) -> dict:
    blob = hsaco.read_bytes()
    work = Path(tempfile.mkdtemp(prefix="sdpa_bwd_gate_"))
    try:
        (work / "obj.hsaco").write_bytes(blob)
        facts = analyze.analyze(work)
        if not str(facts["target"]).endswith("gfx950"):
            raise RuntimeError(
                f"the gates are gfx950-only; the object targets {facts['target']!r}"
            )
        dumps = first_dump_per_kernel(greedy_dump.read_text(errors="replace"))
        if not dumps:
            raise RuntimeError("no greedy dump in the compile's stderr")
        spill = other = 0
        findings = []
        for name, dump in dumps:
            s, o, rows = rmir.scan_text(dump, work / "mir.txt")
            spill, other = spill + s, other + o
            findings += [[name, *map(str, row)] for row in rows[:20]]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    verdict = "ok" if not spill else "known" if spec in rmir_known() else "red"
    return {
        "object_sha256": hashlib.sha256(blob).hexdigest(),
        "resources": {c: facts[c] for c in RESOURCE_COLUMNS},
        "static_1087": {
            "static_gate": facts["exposed_1087"],
            "undef_writelanes": facts["undef_writelanes"],
            "tainted_buffer_accesses": facts["tainted_buffer_accesses"],
        },
        "rmir": {
            "kernels": [name for name, _ in dumps],
            "spill_findings": spill,
            "other_findings": other,
            "verdict": verdict,
            "findings": findings[:20],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hsaco", required=True, type=Path)
    parser.add_argument("--greedy-dump", required=True, type=Path)
    parser.add_argument("--spec", default=None, help="the Spec key, for rmir_known.csv")
    args = parser.parse_args(argv)
    try:
        result = gate(args.hsaco, args.greedy_dump, args.spec)
    except Exception as exc:  # noqa: BLE001 - reported to the caller as one line
        print(
            f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500], file=sys.stderr
        )
        return 1
    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
