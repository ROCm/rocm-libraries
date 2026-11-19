#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Disposable raw-COMGR worker for the compiler-capability probe.

The controller starts one process per independently attributable probe module.
A compiler abort therefore destroys no other observation. The worker bypasses
rocKE's legacy flavor/profile guard on purpose; applying that guard here would
let the behavior under study decide the answer before COMGR sees the candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def _disable_core_dumps() -> None:
    """Keep expected compiler crashes from writing large local artifacts."""
    try:
        import resource
    except ImportError:  # Windows has no resource module.
        return
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _process_maps_path() -> Path | None:
    if not sys.platform.startswith("linux"):
        return None
    return Path(os.sep) / "proc" / "self" / "maps"


def _loaded_compiler_dependencies(maps: Path | None = None) -> list[str] | None:
    """Mapped shared LLVM libraries, or ``None`` when they cannot be enumerated."""
    selected_maps = maps if maps is not None else _process_maps_path()
    if selected_maps is None or not selected_maps.is_file():
        return None
    try:
        lines = selected_maps.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    dependencies = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        if path.name.startswith(("libLLVM", "libclang-cpp")) and path.is_file():
            dependencies.add(str(path.resolve()))
    return sorted(dependencies)


def _identity() -> dict[str, object]:
    from ..runtime import comgr

    info = comgr.loaded_compiler_info()
    if info is None:
        raise comgr.ComgrError("COMGR could not be loaded")
    return {
        "llvm_version": list(info.llvm_version) if info.llvm_version else None,
        "version_source": info.source,
        "requested_library": info.requested_comgr,
        "loaded_library": info.comgr_path,
        "query_library": info.query_library_path,
        "dependency_libraries": _loaded_compiler_dependencies(),
    }


def _probe(header: dict[str, object], ir_text: str) -> dict[str, object]:
    from ..runtime import comgr

    options = header.get("options", ["-O3"])
    if not isinstance(options, list) or not all(isinstance(x, str) for x in options):
        raise TypeError("COMGR probe options must be a list of strings")
    isa = header.get("isa")
    if not isinstance(isa, str) or not isa:
        raise ValueError("COMGR probe ISA must be a non-empty string")
    try:
        hsaco, timings = comgr._build_hsaco(
            (ir_text,),
            isa=isa,
            options=options,
            validate_profile=False,
        )
    except comgr.ComgrError as exc:
        diagnostic = str(exc)
        failed_stage = next(
            (
                stage
                for marker, stage in (
                    ("set_isa", "set_isa"),
                    ("COMPILE_SOURCE_TO_BC", "compile_source_to_bc"),
                    ("CODEGEN_BC_TO_RELOCATABLE", "codegen_bc_to_relocatable"),
                    (
                        "LINK_RELOCATABLE_TO_EXECUTABLE",
                        "link_relocatable_to_executable",
                    ),
                )
                if marker in diagnostic
            ),
            "setup",
        )
        return {
            "outcome": "rejected",
            "failed_stage": failed_stage,
            "diagnostic": diagnostic,
        }
    return {
        "outcome": "accepted",
        "evidence": {
            "hsaco_sha256": hashlib.sha256(hsaco).hexdigest(),
            "hsaco_size": len(hsaco),
            "completed_stages": [
                "compile_source_to_bc",
                "codegen_bc_to_relocatable",
                "link_relocatable_to_executable",
            ],
        },
        "timings": {
            "bc": timings.bc,
            "relocatable": timings.relocatable,
            "executable": timings.executable,
        },
    }


def main() -> int:
    _disable_core_dumps()
    payload = sys.stdin.buffer.read()
    head, separator, body = payload.partition(b"\n")
    if not separator:
        raise ValueError("COMGR probe worker expected one JSON header line")
    header = json.loads(head)
    command = header.get("command")
    if command == "identity":
        result = _identity()
    elif command == "probe":
        result = _probe(header, body.decode("utf-8"))
    else:
        raise ValueError(f"unknown COMGR probe worker command {command!r}")
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
