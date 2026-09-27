# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Capture an ATT trace that WaveScope can map back to Python source.

Three things have to line up for the viewer's Source tab to be useful, and each
one fails quietly on its own:

1. The kernel must be **built** with source-location capture. A rocke kernel is
   Python that builds IR, so there is no C++ source for ``-g`` to point at and no
   ``debug=`` parameter on ``compile_kernel()``. The locations only exist while
   the kernel is being built, so ``ROCKE_DEBUG_LOC=1`` has to be set for the
   profiled process -- not passed to the compiler. Forget it and ``code.json``'s
   Source column comes back empty.
2. ``rocprofv3 --att`` must actually decode the dispatch you care about.
3. The inlining call stack has to be recovered afterwards. rocprofv3 flattens
   DWARF to the innermost frame, so on a kernel assembled out of helpers a single
   one-line loader appears to own most of the kernel's stalls.

This runs all three in order. Capture itself is delegated to
``stage2_capture/capture_att_trace.py`` (decoder preflight, kernel-name
discovery, per-dispatch reporting); this adds the environment for step 1 and the
sidecar for step 3, and says which folder to open.

Each invocation gets a fresh ``capture-<trace-id>`` generation below the output
directory. That boundary makes ``--no-source``, partial failure, and direct
``capture_att_trace.py`` use safe without deleting or confusing older captures.
Sidecar generation may enrich a completed generation, but cannot change whether
capture itself completed.

Usage:
    python3 capture_wavescope_trace.py -- python3 bench.py
    python3 capture_wavescope_trace.py --output-dir ./att_out -- python3 bench.py
    python3 capture_wavescope_trace.py --kernel-regex ugemm_gfx950 -- python3 bench.py

Any flag this does not recognize is forwarded to the capture script, so
``--target-cu 3`` or ``--iteration-range '[1, [2-4]]'`` work unchanged.

Open the reported folder with the **WaveScope: Open Trace Folder...** command,
then switch the Source tab to ``+ inlined`` to see each call site charged with
what was inlined into it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Capture itself is stage 2 of the stage1..stage5 profiling pipeline next door.
# Only the two steps either side of it are new, so delegating keeps one
# implementation of the rocprofv3 flag soup.
CAPTURE_SCRIPT = HERE.parent / "stage2_capture" / "capture_att_trace.py"
SIDECAR_SCRIPT = HERE / "emit_inline_frames.py"

# See IRBuilder(capture_loc=) and dsl_docs/reference/env_flags.md. Off by default
# because it costs a stack walk per op and changes the emitted .ll bytes; the
# generated ISA is unaffected, so a trace captured with it on is representative.
LOC_ENV = "ROCKE_DEBUG_LOC"

DISPATCH_GLOB = "ui_output_*_dispatch_*"

NO_COMMAND = (
    "no command given. Put the command to profile after --, e.g.\n"
    "  python3 capture_wavescope_trace.py -- python3 bench.py"
)


def split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split our own flags from the command to profile, on the first ``--``.

    Done by hand rather than with ``argparse.REMAINDER`` so that a flag meant for
    the capture script is never mistaken for one of the profiled command's.
    """
    if "--" not in argv:
        raise SystemExit(NO_COMMAND)
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1 :]


def run_capture(
    command: list[str], out_dir: Path, forward: list[str], *, with_source: bool
) -> Path:
    env = os.environ.copy()
    if with_source:
        # Inherited by rocprofv3 and through it by the profiled process, which is
        # where the kernel is built and therefore where it has to be set.
        env[LOC_ENV] = "1"
        print(f"[source] {LOC_ENV}=1 for the profiled command", flush=True)
    else:
        # An explicit opt-out must not be defeated by an inherited value.
        env.pop(LOC_ENV, None)
        print(f"[source] {LOC_ENV} unset -- Source column will be empty", flush=True)

    capture_id = str(uuid.uuid4())
    argv = [
        sys.executable,
        str(CAPTURE_SCRIPT),
        *forward,
        "--output-dir",
        str(out_dir),
        "--capture-id",
        capture_id,
        "--",
        *command,
    ]
    proc = subprocess.run(argv, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"capture failed (exit {proc.returncode})")
    return out_dir / f"capture-{capture_id}"


def run_sidecar(out_dir: Path, code_object: Path | None) -> bool:
    """Write inline_frames.json. Returns whether it succeeded.

    A failure here is reported but not fatal: the trace is the expensive part and
    is perfectly usable without the sidecar -- the Source tab just falls back to
    the innermost frame, which is how it behaved before the sidecar existed.
    """
    argv = [sys.executable, str(SIDECAR_SCRIPT), str(out_dir)]
    if code_object:
        argv += ["--code-object", str(code_object)]
    proc = subprocess.run(argv)
    if proc.returncode != 0:
        print(
            "\n[warn] no inline_frames.json was written, so the Source tab will "
            "attribute\n"
            "       every instruction to its innermost frame only. The trace "
            "itself is fine.\n"
            f"       Re-run just this step with:\n"
            f"         python3 {SIDECAR_SCRIPT} {out_dir}",
            file=sys.stderr,
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the trace (default: ./att_out)",
    )
    p.add_argument(
        "--code-object",
        type=Path,
        default=None,
        help="code object with DWARF; defaults to the one rocprofv3 dumped",
    )
    p.add_argument(
        "--no-source",
        action="store_true",
        help=f"skip {LOC_ENV} and the sidecar (ISA-level trace only)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    p = build_parser()
    if "--" not in raw:
        # Let the parser answer -h/--help before complaining about the command.
        p.parse_known_args(raw)
        raise SystemExit(NO_COMMAND)

    own, command = split_command(raw)
    args, forward = p.parse_known_args(own)
    if not command:
        raise SystemExit(NO_COMMAND)

    for script in (CAPTURE_SCRIPT, SIDECAR_SCRIPT):
        if not script.is_file():
            raise SystemExit(f"missing helper script: {script}")

    out_dir = (args.output_dir or Path.cwd() / "att_out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generation = run_capture(command, out_dir, forward, with_source=not args.no_source)

    if not args.no_source:
        print(
            "\n[inline] recovering the call stack rocprofv3 flattened away", flush=True
        )
        run_sidecar(generation, args.code_object)

    dispatches = sorted(generation.glob(DISPATCH_GLOB))
    if not dispatches:
        # capture_att_trace.py already explains this case; don't claim success.
        return 1
    print("\nOpen in WaveScope: run 'WaveScope: Open Trace Folder...' and pick")
    print(f"  {dispatches[0]}")
    if not args.no_source:
        print("Then switch the Source tab to '+ inlined'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
