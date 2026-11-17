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
import sys


def _disable_core_dumps() -> None:
    """Keep expected compiler crashes from writing large local artifacts."""
    try:
        import resource
    except ImportError:  # Windows has no resource module.
        return
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _identity() -> dict[str, object]:
    from rocke.runtime import comgr

    info = comgr.loaded_compiler_info()
    if info is None:
        raise comgr.ComgrError("COMGR could not be loaded")
    return {
        "llvm_version": list(info.llvm_version) if info.llvm_version else None,
        "version_source": info.source,
        "requested_library": info.requested_comgr,
        "loaded_library": info.comgr_path,
        "query_library": info.query_library_path,
    }


def _probe(header: dict[str, object], ir_text: str) -> dict[str, object]:
    from rocke.runtime import comgr

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
