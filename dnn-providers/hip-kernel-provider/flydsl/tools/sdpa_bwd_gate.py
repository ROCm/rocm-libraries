# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Gates for one SDPA backward object (gfx950): resources, #1087, R-MIR, pins. Prints one JSON object.

    python tools/sdpa_bwd_gate.py --hsaco OBJ (--greedy-dump TEXT | --pin) [--spec KEY]
        [--reduce-symbol SYMBOL --reference-dir DIR [--reference-csv CSV]]

``OBJ`` is the delivered code object; ``TEXT`` is what LLVM printed on stderr under
``print-after=greedy`` during the compile that produced it. A freshly built object is
always scanned: without ``TEXT`` the gate fails, unless ``--pin`` says the object is a
stored pin (``rmir`` is then ``not scanned``). A stored pin passes only when its row of
``--reference-csv`` records the pin's own scan (``rmir_scan`` ``clean`` or ``exception``); a pin
with no row or no such value is an error. The
generators call this once per object, in a subprocess, so a gate edit never restamps an
object.

The gate covers gfx950 only: the detector reads gfx950 MIR, so a gfx11 object has no R-MIR
scan and is held to its pin byte for byte (``REGEN.md``).

An object holds one kernel, or, with ``--reduce-symbol``, two: the merged kernel and the
dBias reduce kernel of a dB-storing ``bias`` merged object. ``per_kernel`` carries every kernel's
facts by symbol; ``resources`` and ``static_1087`` are the object's primary kernel's (the
one that is not the reduce).

* ``resources``: the manifest's resource row (``analyze.py``).
* ``static_1087``: the #1087 detectors (``analyze.py``, ``undefscan_issue1087.py``).
  Recorded, never a verdict.
* ``pin_gate`` (two-kernel objects, with ``--reference-dir``): the merged kernel against its
  standalone pin (``<reference-dir>/<spec hash>.hsaco``) and the reduce kernel against the
  standalone reduce pin of the bias dtype. ``verdict`` is ``red`` unless the merged kernel's
  function bytes (by ELF symbol) and resource row equal its pin's and the reduce kernel's
  resource row equals its pin's. The reduce bytes are not compared: they differ from the pin
  by 4 bytes under the merged kernel's process-wide ``llvm_options``. A row of
  ``tests/data/sdpa_bwd_nondeterministic.csv`` also passes on the main-function sha of any of its
  ``variant_function_sha256s`` (the resource rows must still equal the pin's).
* ``rmir``: ``rmir.py``'s lane scan of the first SGPR greedy dump of each kernel.
  ``verdict`` is ``red`` iff a finding sits on a register restored from a spill slot
  (the ``rmRedundantSpills`` signature) and ``--spec`` is not in ``rmir_known.csv``,
  ``known`` when it is, else ``ok``. ``rmir_known.csv`` covers no merged spec: a merged row
  in it is an error. Findings on other registers are counted in
  ``other_findings`` (an undefined descriptor lane of the merged objects).

The exit status is nonzero only when a gate could not run (a greedy dump with no dump
in it, no greedy dump for an object that is not a stored pin, a stored pin with no recorded
clean scan, a target other than gfx950,
an object with other kernels than declared, a pin that is not the object its row names, a
merged spec in ``rmir_known.csv``): the caller treats that as an error, not a verdict.
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
import analyze  # noqa: E402
import rmir  # noqa: E402

os.environ.setdefault("LLVM_BIN", analyze.llvm_bin())

sys.path.insert(0, str(HERE.parent))
from generators import _sdpa_bwd_nondet as nondet  # noqa: E402
from generators._sdpa_bwd_rdna_build import RESOURCE_COLUMNS  # noqa: E402
from generators._sdpa_bwd_spec import Spec, dbias_reduce_spec  # noqa: E402
import sdpa_bwd_pins as pins  # noqa: E402

REFERENCE_CSV = HERE.parent / "tests" / "data" / "sdpa_bwd_reference.csv"
RMIR_SCANS = ("clean", "exception")
_FUNCTION = re.compile(r"(?m)^# Machine code for function (\S+?):")


def rmir_known(path: Path = HERE / "rmir_known.csv") -> set[str]:
    """The specs exempt from the R-MIR verdict; a merged spec is never exempt (``RuntimeError``)."""
    with path.open(newline="", encoding="utf-8") as fh:
        known = {row["spec"] for row in csv.DictReader(fh)}
    merged = sorted(key for key in known if Spec.parse(key).kernel == "merged")
    if merged:
        raise RuntimeError(
            f"{path} lists merged specs, which are never exempt: {merged}"
        )
    return known


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
    variants: tuple[str, ...] = (),
) -> dict:
    """The per-kernel verdict of a two-kernel object.

    @p kernels maps each symbol to ``function_sha256`` and ``resources``; @p merged_pin carries
    ``hash``, ``function_sha256`` and ``resources`` of the merged pin, @p reduce_pin ``hash`` and
    ``resources`` of the reduce pin. ``verdict`` is ``red`` iff ``problems`` is not empty.

    A listed row (``sdpa_bwd_nondeterministic.csv``) carries @p variants, the main-function shas of its recorded variants: the
    merged function bytes then equal the pin's or one of them.
    """
    problems = []
    function = kernels[merged]["function_sha256"]
    function_equal = function == merged_pin["function_sha256"]
    variant_equal = function in variants and not function_equal
    if not (function_equal or variant_equal):
        problems.append(
            f"merged function bytes differ from the pin {merged_pin['hash']}"
            + (" and from its recorded variants" if variants else "")
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
        "merged_function_equal": function_equal or variant_equal,
        "merged_function_variant": variant_equal,
        "merged_row_equal": not merged_diff,
        "reduce_row_equal": not reduce_diff,
    }


def pin_facts(
    reference_dir: Path, reference_csv: Path, spec: Spec
) -> tuple[dict, dict]:
    """(merged pin, reduce pin) of @p spec: the objects under @p reference_dir, checked against their CSV rows.

    A pin with no object there (the content never holds a combined object's main kernel or a standalone reduce) has its function bytes
    named by its row of ``sdpa_bwd_pin_functions.csv`` (``sdpa_bwd_pins.py``); a pin with neither stops the run.
    """
    with reference_csv.open(newline="", encoding="utf-8") as fh:
        rows = {row["hash"]: row for row in csv.DictReader(fh)}
    functions = pins.read_pin_functions()

    def load(pin: Spec) -> dict:
        row = rows.get(pin.hash())
        if row is None or row["spec"] != pin.key():
            raise RuntimeError(f"{reference_csv} has no row for {pin.key()}")
        facts = {
            "hash": pin.hash(),
            "resources": {c: row[c] for c in RESOURCE_COLUMNS},
        }
        path = reference_dir / f"{pin.hash()}.hsaco"
        if path.is_file():
            blob = path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != row["sha256"]:
                raise RuntimeError(f"{pin.hash()}.hsaco is not the pin its row names")
            facts["function_sha256"] = pins.function_sha256(blob)
        elif pin.hash() in functions and functions[pin.hash()]["spec"] == pin.key():
            facts["function_sha256"] = functions[pin.hash()]["function_sha256"]
        else:
            raise RuntimeError(
                f"{pin.key()}: no pin object under {reference_dir} and no row in {pins.PIN_FUNCTIONS_CSV.name}"
            )
        return facts

    return load(spec), load(dbias_reduce_spec(spec.bias_dtype))


def reference_scan(reference_csv: Path, spec: str | None) -> str | None:
    """The ``rmir_scan`` of @p spec's row in @p reference_csv (none: no @p spec, or no row for it)."""
    if spec is None:
        return None
    key = Spec.parse(spec)
    with reference_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["hash"] == key.hash() and row["spec"] == key.key():
                return row.get("rmir_scan")
    return None


def gate(
    hsaco: Path,
    greedy_dump: Path | None,
    spec: str | None,
    reduce_symbol: str = "",
    reference_dir: Path | None = None,
    reference_csv: Path = REFERENCE_CSV,
    pin: bool = False,
) -> dict:
    """The gates of @p hsaco. Without @p greedy_dump the object must be a stored @p pin (``RuntimeError`` otherwise)."""
    if greedy_dump is None and not pin:
        raise RuntimeError(
            "no greedy dump: a freshly built object must be scanned (--pin for a stored pin)"
        )
    if greedy_dump is None:
        pin_scan = reference_scan(reference_csv, spec)
        if pin_scan not in RMIR_SCANS:
            raise RuntimeError(
                f"the stored pin of {spec} has no recorded clean R-MIR scan (rmir_scan {pin_scan!r} in {reference_csv.name})"
            )
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
            known = rmir_known()
            verdict = "ok" if not spill else "known" if spec in known else "red"
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
            listed = nondet.read(arch="gfx950").get(merged_pin["hash"])
            result["pin_gate"] = pin_gate(
                per_kernel,
                primary,
                reduce_symbol,
                merged_pin,
                reduce_pin,
                listed.variant_function_sha256s if listed else (),
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hsaco", required=True, type=Path)
    parser.add_argument("--greedy-dump", default=None, type=Path)
    parser.add_argument(
        "--pin",
        action="store_true",
        help="the object is a stored pin: no greedy dump, rmir is `not scanned`",
    )
    parser.add_argument(
        "--spec",
        default=None,
        help="the Spec key: rmir_known.csv, the merged pin, the stored pin's recorded scan",
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
            args.pin,
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
