#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The megakernel family's lock: committed `manifest.json` against its DVC artifacts.

A megakernel bundle is a folder holding the small descriptors (KMD, UED, UDD, UMD) and
`manifest.json` in git, and `artifacts/` from DVC: one code object per lock entry and
`megakernel.kdp.json`, which this tool generates from the lock. The lock is the export
manifest of the megakernels exporter: a JSON list of entries

    {model, src, arch, cus, kv, block, M, tp, rt, lm_head, symbol, block_threads,
     lds_bytes, kernarg_bytes, sha256, size, file, flydsl_version, vllm_commit,
     megakernels_commit}

`generate` writes the KDP from the lock. `verify`, which the pack step runs, fails when
any code object's size or sha256 differs from the lock, when `artifacts/` holds a code
object the lock does not name, or when the KDP is not the one the lock generates. Only
then are the validated bytes the shipped bytes. When `artifacts/` is absent (no
`dvc pull`), configure has already warned and nothing is packed, so `verify` passes.

Standard library only: it runs under whichever interpreter the pack step has.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

KDP_NAME = "megakernel.kdp.json"
ARTIFACTS = "artifacts"

# Ids the small descriptors of the family declare (descriptors/megakernel/*.json).
KMD_ID = "8a6f2f0e-6b0b-5a57-9a3e-6d1c0f6f4b01"
UED_ID = "1ea37de2-144e-5c63-8e45-2d933f10e747"
UDD_ID = "5b6947c5-91dc-563e-b6a3-a292361ddf2e"
UMD_ID = "f94e96df-ef3a-540e-b6a6-ff22077e5943"

# uuid5 namespace of every generated id, so one lock always generates one KDP.
NAMESPACE = uuid.UUID("6c3c5f3e-2f7a-5d4e-9b61-0d6a8c1b7e22")

KEY_FIELDS = ("model", "src", "cus", "kv", "block", "M", "tp", "rt", "lm_head")
LAUNCH_FIELDS = ("block_threads", "lds_bytes")
ENTRY_TYPES = {
    "model": str,
    "src": str,
    "arch": str,
    "cus": int,
    "kv": str,
    "block": int,
    "M": int,
    "tp": int,
    "rt": int,
    "lm_head": bool,
    "symbol": str,
    "block_threads": int,
    "lds_bytes": int,
    "kernarg_bytes": int,
    "sha256": str,
    "size": int,
    "file": str,
    "flydsl_version": str,
    "vllm_commit": str,
    "megakernels_commit": str,
}
# kdec(tab: i64, glob: i64, nlayers: i32): the only ABI the native pack launches.
KERNARG_BYTES = 20


class LockError(Exception):
    pass


def load_lock(lock_path: Path) -> list[dict]:
    try:
        entries = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"{lock_path}: unreadable lock: {error}") from error
    if not isinstance(entries, list) or not entries:
        raise LockError(f"{lock_path}: the lock must be a non-empty JSON list")
    files, keys = set(), set()
    for index, entry in enumerate(entries):
        where = f"{lock_path} entry {index}"
        if not isinstance(entry, dict) or set(entry) != set(ENTRY_TYPES):
            raise LockError(f"{where}: keys must be exactly {sorted(ENTRY_TYPES)}")
        for name, kind in ENTRY_TYPES.items():
            # bool is an int subclass; an int field must not accept true/false.
            if type(entry[name]) is not kind:
                raise LockError(f"{where}: {name} must be {kind.__name__}")
        if entry["kernarg_bytes"] != KERNARG_BYTES:
            raise LockError(
                f"{where}: kernarg_bytes {entry['kernarg_bytes']}, the pack launches "
                f"kdec(i64, i64, i32) = {KERNARG_BYTES} bytes"
            )
        expected_file = f"{entry['model']}_M{entry['M']}_{entry['src'][:16]}.co"
        if entry["file"] != expected_file:
            raise LockError(f"{where}: file must be named {expected_file}")
        key = tuple(entry[k] for k in KEY_FIELDS) + (entry["arch"],)
        if entry["file"] in files or key in keys:
            raise LockError(f"{where}: duplicate file or binary key")
        files.add(entry["file"])
        keys.add(key)
    return entries


def kernel_descriptor(entry: dict) -> dict:
    metadata = {k: entry[k] for k in KEY_FIELDS + LAUNCH_FIELDS}
    return {
        "version": "1.0",
        "id": str(uuid.uuid5(NAMESPACE, f"{entry['model']}/{entry['arch']}/{entry['src']}")),
        "name": f"megakernel.{entry['model']}.M{entry['M']}.{entry['src'][:16]}",
        "arch": [entry["arch"]],
        "kernel_source": {"kind": "hsaco", "file": entry["file"], "symbol": entry["symbol"]},
        "metadata": metadata,
        "priority": 0,
    }


def generate_kdp(entries: list[dict]) -> dict:
    ordered = sorted(entries, key=lambda e: (e["model"], e["arch"], e["M"], e["src"]))
    return {
        "version": "1.0",
        "id": str(uuid.uuid5(NAMESPACE, "kdp:" + ",".join(e["sha256"] for e in ordered))),
        "name": "hipkernel:megakernel",
        "arch": sorted({e["arch"] for e in entries}),
        "matchers": [UMD_ID],
        "engine": UED_ID,
        "dispatch": UDD_ID,
        "provenance": {
            "specialization_contract": {
                "schema_version": 1,
                "consumers": [
                    {
                        "engine_id": UED_ID,
                        "kmd_id": KMD_ID,
                        "metadata_fields": [],
                        "matcher_only_fields": list(KEY_FIELDS + LAUNCH_FIELDS),
                        "bindings": {},
                        "vocabulary": {},
                    }
                ],
            }
        },
        "kernelDescriptors": [kernel_descriptor(e) for e in ordered],
    }


def render(kdp: dict) -> str:
    return json.dumps(kdp, indent=2) + "\n"


def verify(lock_path: Path) -> list[str]:
    """Every failure of the bundle at `lock_path` against its lock; empty when it holds."""
    entries = load_lock(lock_path)
    artifacts = lock_path.parent / ARTIFACTS
    if not artifacts.is_dir():
        return []
    failures = []
    for entry in entries:
        path = artifacts / entry["file"]
        if not path.is_file():
            failures.append(f"{path}: missing (the lock names it)")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != entry["size"] or digest != entry["sha256"]:
            failures.append(
                f"{path}: sha256 {digest} size {len(data)}, lock sha256 "
                f"{entry['sha256']} size {entry['size']}"
            )
    named = {e["file"] for e in entries}
    for path in sorted(artifacts.glob("*.co")):
        if path.name not in named:
            failures.append(f"{path}: not in the lock")
    kdp_path = artifacts / KDP_NAME
    expected = render(generate_kdp(entries))
    actual = kdp_path.read_text(encoding="utf-8") if kdp_path.is_file() else None
    if actual != expected:
        failures.append(
            f"{kdp_path}: not the KDP the lock generates; run "
            f"`{Path(__file__).name} generate --lock {lock_path}`"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("verify", "generate"))
    parser.add_argument("--lock", type=Path, required=True, help="the bundle's manifest.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            artifacts = args.lock.parent / ARTIFACTS
            artifacts.mkdir(exist_ok=True)
            (artifacts / KDP_NAME).write_text(
                render(generate_kdp(load_lock(args.lock))), encoding="utf-8"
            )
            return 0
        failures = verify(args.lock)
    except LockError as error:
        print(f"hkp megakernel lock: {error}", file=sys.stderr)
        return 1
    for failure in failures:
        print(f"hkp megakernel lock: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
