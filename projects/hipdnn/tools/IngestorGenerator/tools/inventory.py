#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Baseline inventory of an installed (or authored) descriptor tree, in one command.

extend.md's baseline inventory is the known-good installation plus the complete engine
tree. This prints the parts a reviewer compares before and after an extension:

    inventory.py <root> --engine hipkernel:Gfx950AttentionDense \\
        [--group-by head_size,num_query_heads,num_kv_heads] \\
        [--validator <build>/bin/hipdnn_validate_descriptors] [--json report.json]

  * every KDP under the engine's bundle directory, its kernel count (read from the
    `kernelDescriptors` key; a KDP without it is an error, never a silent zero) and
    its kernel source kinds;
  * optionally, distinct kernel-metadata tuples over `--group-by` fields with their
    kernel counts (for attention: the head configurations);
  * the catalog digest of the bundle directory, computed exactly as RUNBOOK stage 8's
    recipe does: sha256 of the `sha256sum` lines (`<sha256>  ./<path>`) of every
    regular `*.json` file (symlinks excluded, as `find -type f` does), in byte order
    of path;
  * optionally, `hipdnn_validate_descriptors <root> --expect-engine <engine> --json`:
    its success flag and its diagnostics counted by severity, with each distinct WARN
    or ERROR message (file path folded) and how often it occurred.

`<root>` is the tree the validator reads, for example an install's
`lib/hipdnn_plugins/engines/arch_content/hip-kernel-provider/<arch>`; the bundle is the
directory holding the UED named `--engine`.

Exit codes: 0 inventory taken (and validator succeeded, when run), 1 inventory failed
or the validator reported failure, 2 invalid invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

#: The KDP key holding the kernel entries.
KERNELS = "kernelDescriptors"


class InventoryError(Exception):
    pass


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise InventoryError(f"{path}: {e}") from e


def find_bundle(root: Path, engine: str | None) -> Path:
    """The directory holding the UED named `engine`, or `root` when none is named."""
    if engine is None:
        return root
    found = [
        p.parent
        for p in sorted(root.rglob("*.ued.json"))
        if _load(p).get("name") == engine
    ]
    if not found:
        raise InventoryError(f"no *.ued.json under {root} names engine {engine!r}")
    if len(found) > 1:
        raise InventoryError(
            f"engine {engine!r} is declared in {len(found)} UEDs: {found}"
        )
    return found[0]


def catalog_digest(bundle: Path) -> str:
    """RUNBOOK stage 8's catalog digest of `bundle`.

    The recipe lists files with `find -type f`, which leaves out symlinks, so a symlinked
    JSON file is left out here too.
    """
    files = sorted(
        (("./" + p.relative_to(bundle).as_posix()).encode(), p)
        for p in bundle.rglob("*.json")
        if p.is_file() and not p.is_symlink()
    )
    if not any(rel.endswith(b".kdp.json") for rel, _ in files):
        raise InventoryError(f"no *.kdp.json under {bundle}")
    lines = [
        hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"  " + rel
        for rel, p in files
    ]
    return hashlib.sha256(b"\n".join(lines) + b"\n").hexdigest()


def inventory(bundle: Path, group_by: list[str]) -> dict:
    kdps = []
    groups: Counter = Counter()
    kinds: Counter = Counter()
    for path in sorted(bundle.rglob("*.kdp.json")):
        doc = _load(path)
        if KERNELS not in doc:
            raise InventoryError(f"{path} has no {KERNELS!r} key")
        kernels = doc[KERNELS]
        file_kinds = Counter(
            (k.get("kernel_source") or {}).get("kind") for k in kernels
        )
        kinds.update(file_kinds)
        kdps.append(
            {
                "file": path.relative_to(bundle).as_posix(),
                "name": doc.get("name"),
                "kernels": len(kernels),
                "source_kinds": dict(file_kinds),
            }
        )
        if group_by:
            for k in kernels:
                metadata = k.get("metadata") or {}
                groups[tuple(metadata.get(f) for f in group_by)] += 1
    return {
        "bundle": str(bundle),
        "kdps": kdps,
        "kernels": sum(k["kernels"] for k in kdps),
        "source_kinds": dict(kinds),
        "group_by": group_by,
        "groups": [
            {"values": list(key), "kernels": n}
            for key, n in sorted(groups.items(), key=lambda kv: _sort_key(kv[0]))
        ],
        "catalog_digest": catalog_digest(bundle),
    }


def _sort_key(values: tuple) -> list:
    """Numbers in numeric order, ahead of anything else in string order."""
    return [
        (0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v)) for v in values
    ]


def _fold(message: str) -> str:
    """A diagnostic with its file path replaced, so one message per kernel collapses."""
    return re.sub(r"(?<=in )(?:/|[A-Za-z]:[\\/])[^\s;]+", "<path>", message)


def run_validator(validator: Path, root: Path, engine: str | None) -> dict:
    argv = [str(validator), str(root), "--json"]
    if engine:
        argv += ["--expect-engine", engine]
    try:
        result = subprocess.run(argv, capture_output=True, text=True)
    except OSError as e:
        raise InventoryError(f"cannot run validator {validator}: {e}") from e
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise InventoryError(
            f"validator exit {result.returncode} printed no JSON report: {e}\n{result.stderr}"
        ) from e
    diagnostics = report.get("diagnostics", [])
    notable = Counter(
        (d.get("severity"), _fold(d.get("message", "")))
        for d in diagnostics
        if d.get("severity") != "INFO"
    )
    return {
        "exit_code": result.returncode,
        "success": bool(report.get("success")) and result.returncode == 0,
        "engines": report.get("engines", []),
        "expected_engines_missing": report.get("expected_engines_missing", []),
        "severities": dict(Counter(d.get("severity") for d in diagnostics)),
        "messages": [
            {"severity": sev, "message": msg, "count": n}
            for (sev, msg), n in sorted(notable.items())
        ],
    }


def _print(report: dict) -> None:
    inv = report["inventory"]
    print(f"bundle: {inv['bundle']}")
    for kdp in inv["kdps"]:
        print(
            f"  {kdp['file']}: {kdp['kernels']} kernels, source kinds {kdp['source_kinds']}"
        )
    print(f"kernels: {inv['kernels']} (source kinds {inv['source_kinds']})")
    if inv["group_by"]:
        print(f"distinct ({', '.join(inv['group_by'])}): {len(inv['groups'])}")
        for g in inv["groups"]:
            print(f"  {tuple(g['values'])}: {g['kernels']}")
    print(f"catalog digest: {inv['catalog_digest']}")
    val = report.get("validator")
    if val is None:
        print("validator: not run (pass --validator)")
        return
    verdict = "success" if val["success"] else "FAILED"
    print(f"validator: {verdict} (exit {val['exit_code']}), engines {val['engines']}")
    if val["expected_engines_missing"]:
        print(f"  expected engines missing: {val['expected_engines_missing']}")
    print(f"  diagnostics by severity: {val['severities']}")
    for m in val["messages"]:
        print(f"  [{m['severity']}] x{m['count']} {m['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="descriptor tree the validator reads")
    parser.add_argument("--engine", help="engine name; selects the bundle by its UED")
    parser.add_argument(
        "--group-by",
        default="",
        help="comma-separated kernel metadata fields to count distinct tuples of",
    )
    parser.add_argument(
        "--validator", type=Path, help="hipdnn_validate_descriptors binary"
    )
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args(argv)

    group_by = [f for f in args.group_by.split(",") if f]
    try:
        if not args.root.is_dir():
            raise InventoryError(f"{args.root} is not a directory")
        report = {"inventory": inventory(find_bundle(args.root, args.engine), group_by)}
        if args.validator:
            report["validator"] = run_validator(args.validator, args.root, args.engine)
    except InventoryError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    _print(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report.get("validator", {}).get("success", True) else 1


if __name__ == "__main__":
    sys.exit(main())
