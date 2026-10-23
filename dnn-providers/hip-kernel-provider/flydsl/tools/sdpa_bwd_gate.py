# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Gates for one SDPA backward object (gfx950): resources, #1087, R-MIR, pins. Prints one JSON object.

    python tools/sdpa_bwd_gate.py --hsaco OBJ [--greedy-dump TEXT] [--spec KEY]
        [--reduce-symbol SYMBOL --reference-dir DIR [--reference-csv CSV]]

``OBJ`` is the delivered code object; ``TEXT`` is what LLVM printed on stderr under
``print-after=greedy`` during the compile that produced it (without it, ``rmir`` is
``not scanned``). The generators call this once per object, in a subprocess, so a gate
edit never restamps an object.

An object holds one kernel, or, with ``--reduce-symbol``, two: the merged kernel and the
dBias reduce kernel of a dB-storing ``bias`` merged object. ``per_kernel`` carries every kernel's
facts by symbol; ``resources`` and ``static_1087`` are the object's primary kernel's (the
one that is not the reduce).

* ``resources``: the manifest's resource row (``analyze.py``).
* ``static_1087``: the #1087 detectors (``analyze.py``, ``undefscan_issue1087.py``).
  Recorded, never a verdict.
* ``pin_gate`` (two-kernel objects, with ``--reference-dir``): the merged kernel against its
  standalone pin (``<reference-dir>/<spec hash>.hsaco``) and the reduce kernel against the
  standalone reduce pin of the dtype. ``verdict`` is ``red`` unless the merged kernel's
  function bytes (by ELF symbol) and resource row equal its pin's and the reduce kernel's
  resource row equals its pin's. The reduce bytes are not compared: they differ from the pin
  by 4 bytes under the merged kernel's process-wide ``llvm_options``.
* ``rmir``: ``rmir.py``'s lane scan of the first SGPR greedy dump of each kernel.
  ``verdict`` is ``red`` iff a finding sits on a register restored from a spill slot
  (the ``rmRedundantSpills`` signature) and ``--spec`` is not in ``rmir_known.csv``,
  ``known`` when it is, else ``ok``. Findings on other registers are counted in
  ``other_findings`` (T4: an undefined descriptor lane of the merged objects).

The exit status is nonzero only when a gate could not run (a greedy dump with no dump
in it, a target other than gfx950, an object with other kernels than declared, a pin that
is not the object its row names): the caller treats that as an error, not a verdict.
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

sys.path.insert(0, str(HERE.parent))
from generators._sdpa_bwd_spec import Spec, dbias_reduce_spec  # noqa: E402

REFERENCE_CSV = HERE.parent / "tests" / "data" / "sdpa_bwd_reference.csv"

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


def _row_differences(got: dict, row: dict) -> list[str]:
    """The resource columns where @p got and the reference row @p row differ."""
    return [
        f"{c} {got[c]} != {row[c]}"
        for c in RESOURCE_COLUMNS
        if str(got[c]) != str(row[c])
    ]


def pin_gate(
    kernels: dict[str, dict],
    merged: str,
    reduce: str,
    merged_pin: dict,
    reduce_pin: dict,
) -> dict:
    """The D6 verdict of a two-kernel object.

    @p kernels maps each symbol to ``function_sha256`` and ``resources``; @p merged_pin carries
    ``hash``, ``function_sha256`` and ``resources`` of the merged pin, @p reduce_pin ``hash`` and
    ``resources`` of the reduce pin. ``verdict`` is ``red`` iff ``problems`` is not empty.
    """
    problems = []
    function_equal = kernels[merged]["function_sha256"] == merged_pin["function_sha256"]
    if not function_equal:
        problems.append(
            f"merged function bytes differ from the pin {merged_pin['hash']}"
        )
    merged_diff = _row_differences(
        kernels[merged]["resources"], merged_pin["resources"]
    )
    if merged_diff:
        problems.append(
            f"merged resource row differs from the pin {merged_pin['hash']}: "
            + ", ".join(merged_diff)
        )
    reduce_diff = _row_differences(
        kernels[reduce]["resources"], reduce_pin["resources"]
    )
    if reduce_diff:
        problems.append(
            f"reduce resource row differs from the pin {reduce_pin['hash']}: "
            + ", ".join(reduce_diff)
        )
    return {
        "verdict": "red" if problems else "ok",
        "problems": problems,
        "merged_pin": merged_pin["hash"],
        "reduce_pin": reduce_pin["hash"],
        "merged_function_equal": function_equal,
        "merged_row_equal": not merged_diff,
        "reduce_row_equal": not reduce_diff,
    }


def pin_facts(
    reference_dir: Path, reference_csv: Path, spec: Spec
) -> tuple[dict, dict]:
    """(merged pin, reduce pin) of @p spec: the objects under @p reference_dir, checked against their CSV rows."""
    with reference_csv.open(newline="", encoding="utf-8") as fh:
        rows = {row["hash"]: row for row in csv.DictReader(fh)}

    def load(pin: Spec) -> tuple[dict, bytes]:
        row = rows.get(pin.hash())
        if row is None or row["spec"] != pin.key():
            raise RuntimeError(f"{reference_csv} has no row for {pin.key()}")
        blob = (reference_dir / f"{pin.hash()}.hsaco").read_bytes()
        if hashlib.sha256(blob).hexdigest() != row["sha256"]:
            raise RuntimeError(f"{pin.hash()}.hsaco is not the pin its row names")
        return (
            {"hash": pin.hash(), "resources": {c: row[c] for c in RESOURCE_COLUMNS}},
            blob,
        )

    merged_pin, blob = load(spec)
    (symbol,) = analyze.kernels(analyze.amdgpu_metadata(blob))
    merged_pin["function_sha256"] = hashlib.sha256(
        analyze.function_bytes(blob, symbol)
    ).hexdigest()
    reduce_pin, _ = load(dbias_reduce_spec(spec.dtype))
    return merged_pin, reduce_pin


def gate(
    hsaco: Path,
    greedy_dump: Path | None,
    spec: str | None,
    reduce_symbol: str = "",
    reference_dir: Path | None = None,
    reference_csv: Path = REFERENCE_CSV,
) -> dict:
    blob = hsaco.read_bytes()
    work = Path(tempfile.mkdtemp(prefix="sdpa_bwd_gate_"))
    try:
        (work / "obj.hsaco").write_bytes(blob)
        names = list(analyze.kernels(analyze.amdgpu_metadata(blob)))
        if len(names) != (2 if reduce_symbol else 1) or (
            reduce_symbol and reduce_symbol not in names
        ):
            raise RuntimeError(
                f"the object holds kernels {names}; expected "
                + (
                    f"the merged kernel and {reduce_symbol!r}"
                    if reduce_symbol
                    else "one"
                )
            )
        primary = next(name for name in names if name != reduce_symbol)
        per_kernel = {}
        for name in names:
            facts = analyze.analyze(work, name if reduce_symbol else None)
            if not str(facts["target"]).endswith("gfx950"):
                raise RuntimeError(
                    f"the gates are gfx950-only; the object targets {facts['target']!r}"
                )
            code = analyze.function_bytes(blob, name)
            per_kernel[name] = {
                "function_bytes": len(code),
                "function_sha256": hashlib.sha256(code).hexdigest(),
                "resources": {c: facts[c] for c in RESOURCE_COLUMNS},
                "static_1087": {
                    "static_gate": facts["exposed_1087"],
                    "undef_writelanes": facts["undef_writelanes"],
                    "tainted_buffer_accesses": facts["tainted_buffer_accesses"],
                },
            }
        if greedy_dump is None:
            rmir_result = {
                "kernels": names,
                "spill_findings": None,
                "other_findings": None,
                "verdict": "not scanned",
                "findings": [],
            }
        else:
            dumps = first_dump_per_kernel(greedy_dump.read_text(errors="replace"))
            if not dumps:
                raise RuntimeError("no greedy dump in the compile's stderr")
            spill = other = 0
            findings = []
            for name, dump in dumps:
                s, o, rows = rmir.scan_text(dump, work / "mir.txt")
                spill, other = spill + s, other + o
                findings += [[name, *map(str, row)] for row in rows[:20]]
            verdict = "ok" if not spill else "known" if spec in rmir_known() else "red"
            rmir_result = {
                "kernels": [name for name, _ in dumps],
                "spill_findings": spill,
                "other_findings": other,
                "verdict": verdict,
                "findings": findings[:20],
            }
        result = {
            "object_sha256": hashlib.sha256(blob).hexdigest(),
            "resources": per_kernel[primary]["resources"],
            "static_1087": per_kernel[primary]["static_1087"],
            "rmir": rmir_result,
            "per_kernel": per_kernel,
        }
        if reduce_symbol and reference_dir is not None:
            merged_pin, reduce_pin = pin_facts(
                reference_dir, reference_csv, Spec.parse(spec)
            )
            result["pin_gate"] = pin_gate(
                per_kernel, primary, reduce_symbol, merged_pin, reduce_pin
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hsaco", required=True, type=Path)
    parser.add_argument("--greedy-dump", default=None, type=Path)
    parser.add_argument(
        "--spec", default=None, help="the Spec key: rmir_known.csv, the merged pin"
    )
    parser.add_argument(
        "--reduce-symbol",
        default="",
        help="the object holds this dBias reduce kernel beside the merged one",
    )
    parser.add_argument(
        "--reference-dir", default=None, type=Path, help="the pins, <hash>.hsaco"
    )
    parser.add_argument("--reference-csv", default=REFERENCE_CSV, type=Path)
    args = parser.parse_args(argv)
    try:
        result = gate(
            args.hsaco,
            args.greedy_dump,
            args.spec,
            args.reduce_symbol,
            args.reference_dir,
            args.reference_csv,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the caller as one line
        print(
            f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500], file=sys.stderr
        )
        return 1
    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
