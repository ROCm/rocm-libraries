# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both engines lower every parity kernel to the same HIP source.

``tools/check_byte_identity.py`` holds the LLVM-IR route byte-identical between
the Python and C++ engines. The HIP-source route has a lowerer in each engine as
well -- ``rocke.core.lower_hip.lower_kernel_to_hip`` and
``rocke_engine.lower_serialized_ir_hip`` -- and this test holds it to the same
rule: for every kernel the parity emitters build, both lowerers return the same
text, or both refuse the kernel with the same error type and message.

The input is the Python-built kernel, serialized. Whether the two engines'
builders produce the same IR is the differential gate's ``ir`` mode; here one
kernel goes through both HIP lowerers, so a difference can only come from them.

Each family's emitter runs in its own process through ``python_batch_worker.py``
(the gate's worker) in the ``target-ir`` mode, which prints a ``target <gfx>``
line and the serialized kernel for every kernel a config builds. A config the
Python builder rejects prints nothing and has nothing to lower.

Needs the ``rocke_engine`` extension; skipped without it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("rocke_engine")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_diff  # noqa: E402 -- the differential harness is not a package

from rocke.tests.engines import (  # noqa: E402
    HipRefusal,
    hip_outcomes_of_both_engines,
)

# Emitters that build no kernel, so they have no target-ir output.
NO_KERNEL_FAMILIES = {
    "conv_abi": "prints kernel argument lists, not a kernel",
}

# One family's emitter run: every config, one process. The slowest families
# (the library attention emitters) take seconds; this bounds a hung emitter.
EMITTER_TIMEOUT_S = 900

FAMILIES = run_diff.find_families(python_only=True)


def _target_ir_outputs(
    name: str, src_dir: Path
) -> list[tuple[int, str, run_diff.Emission]]:
    """Run one family's emitter in the target-ir mode over its configs; return
    ``(config index, stdout, emission)`` for every config it ran."""
    emitter = src_dir / f"{name}_emit.py"
    with tempfile.TemporaryDirectory(prefix=f"hip_identity_{name}_") as out_dir:
        proc = subprocess.run(
            [
                sys.executable,
                str(run_diff.PYTHON_BATCH_WORKER),
                "--output-dir",
                out_dir,
                str(emitter),
                "target-ir",
                *map(str, range(run_diff.config_limit(name))),
            ],
            env=run_diff.python_emitter_env(src_dir),
            check=False,  # the assert below reports the worker's stderr
            capture_output=True,
            text=True,
            timeout=EMITTER_TIMEOUT_S,
        )
        assert (
            proc.returncode == 0
        ), f"{name}: batch worker failed:\n{proc.stderr[-4000:]}"
        runs = []
        for line in proc.stdout.splitlines():
            record = json.loads(line)
            out = Path(out_dir) / f"{record['idx']}.out"
            text = out.read_text(encoding="utf-8") if out.exists() else ""
            runs.append((record["idx"], text, run_diff.emission_from_record(record)))
        return runs


def _kernels(text: str) -> list[tuple[str, str]]:
    """Split target-ir output into ``(arch, serialized kernel)`` blocks."""
    blocks: list[tuple[str, list[str]]] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("target "):
            blocks.append((line.split()[1], []))
        elif blocks:
            blocks[-1][1].append(line)
        else:
            raise ValueError(
                f"target-ir output does not start with a target line: {line!r}"
            )
    return [(arch, "".join(lines)) for arch, lines in blocks]


def _first_difference(python_outcome: str, native_outcome: str) -> str:
    python_lines, native_lines = (
        python_outcome.splitlines(),
        native_outcome.splitlines(),
    )
    for number, (python_line, native_line) in enumerate(
        zip(python_lines, native_lines), start=1
    ):
        if python_line != native_line:
            return (
                f"line {number}: Python {python_line.strip()!r}, "
                f"C++ {native_line.strip()!r}"
            )
    return f"Python has {len(python_lines)} lines, C++ {len(native_lines)}"


def _disagreement(
    python_outcome: str | HipRefusal, native_outcome: str | HipRefusal
) -> str | None:
    """How the two engines' outcomes for one kernel differ, or None if they agree."""
    if python_outcome == native_outcome:
        return None
    python_refuses = isinstance(python_outcome, HipRefusal)
    native_refuses = isinstance(native_outcome, HipRefusal)
    if python_refuses and native_refuses:
        return (
            f"both refuse, differently: Python {python_outcome}, C++ {native_outcome}"
        )
    if python_refuses:
        return f"only Python refuses: {python_outcome}"
    if native_refuses:
        return f"only C++ refuses: {native_outcome}"
    return _first_difference(python_outcome, native_outcome)


@pytest.mark.parametrize(
    ("name", "src_dir"), FAMILIES, ids=[name for name, _ in FAMILIES]
)
def test_both_engines_emit_the_same_hip_source(name: str, src_dir: Path) -> None:
    runs = _target_ir_outputs(name, src_dir)
    if name in NO_KERNEL_FAMILIES:
        assert runs and all(
            run_diff.mode_unsupported(emission) for _, _, emission in runs
        ), (
            f"{name} is listed as building no kernel "
            f"({NO_KERNEL_FAMILIES[name]}) but accepted the target-ir mode"
        )
        return
    assert not any(
        run_diff.mode_unsupported(emission) for _, _, emission in runs
    ), f"{name}_emit.py does not accept the target-ir mode"

    compared = 0
    disagreements = []
    for index, text, _ in runs:
        for block, (arch, ir) in enumerate(_kernels(text)):
            where = f"{name} config {index} kernel {block} ({arch})"
            python_outcome, native_outcome = hip_outcomes_of_both_engines(ir, arch)
            compared += 1
            difference = _disagreement(python_outcome, native_outcome)
            if difference is not None:
                disagreements.append(f"{where}: {difference}")
    assert compared, f"{name}: the emitter built no kernel in any config"
    assert not disagreements, (
        f"{len(disagreements)} of {compared} {name} kernels lower to different "
        "HIP source:\n" + "\n".join(disagreements[:20])
    )
