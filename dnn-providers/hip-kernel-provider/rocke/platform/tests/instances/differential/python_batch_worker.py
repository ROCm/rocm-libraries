#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run many configurations from one Python parity emitter import.

The differential parent needs only output presence and SHA-256 for LLVM mode.
Keeping that reduction in this worker avoids sending every full LLVM module
through a pipe while amortizing Python and rocKE imports across a family.
Emitter ``main()`` functions must therefore be reentrant: module state persists
between configurations, while ``sys.argv`` and captured streams are reset.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import traceback


def _exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    return 1


def _load_main(path: Path):
    spec = importlib.util.spec_from_file_location("_rocke_parity_batch_emitter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load parity emitter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise TypeError(f"parity emitter has no callable main(): {path}")
    return main


def _run_one(
    main, emitter: Path, index: int, mode: str, output_dir: Path | None
) -> dict:
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous_argv = sys.argv
    sys.argv = [str(emitter), str(index), mode]
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                result = main()
                returncode = int(result or 0)
            except SystemExit as exc:
                returncode = _exit_code(exc)
                if exc.code is not None and not isinstance(exc.code, int):
                    stderr.write(f"{exc.code}\n")
            except Exception:  # noqa: BLE001 -- preserve emitter failure as evidence
                returncode = 1
                traceback.print_exc(file=stderr)
    finally:
        sys.argv = previous_argv

    output = stdout.getvalue().encode("utf-8")
    if output_dir is not None:
        (output_dir / f"{index}.out").write_bytes(output)
    return {
        "idx": index,
        "returncode": returncode,
        "has_output": bool(output),
        "sha256": hashlib.sha256(output).hexdigest() if output else None,
        "stderr": stderr.getvalue(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("emitter", type=Path)
    parser.add_argument("mode", choices=("ll", "ir", "verify"))
    parser.add_argument("indices", nargs="+", type=int)
    args = parser.parse_args()

    emitter_main = _load_main(args.emitter)
    for index in args.indices:
        print(f"active config {index}", file=sys.stderr, flush=True)
        record = _run_one(emitter_main, args.emitter, index, args.mode, args.output_dir)
        # One flushed record per configuration lets the parent apply TIMEOUT to
        # each invocation while retaining this process and its imported module.
        print(json.dumps(record, separators=(",", ":")), flush=True)
        if not record["has_output"] and "unknown config" in record["stderr"].lower():
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
