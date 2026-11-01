# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both engines lower every parity kernel to the same HIP source.

``tools/check_byte_identity.py`` holds the LLVM-IR route byte-identical between
the Python and C++ engines. The HIP-source route has a lowerer in each engine as
well -- ``rocke.core.lower_hip.lower_kernel_to_hip`` and
``rocke_engine.lower_serialized_ir_hip`` -- and this test holds it to the same
rule: for every kernel the parity emitters build, both lowerers return the same
text, or both refuse the kernel.

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
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

rocke_engine = pytest.importorskip("rocke_engine")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_diff  # noqa: E402 -- the differential harness is not a package

from rocke.core.ir_serialize import parse  # noqa: E402
from rocke.core.lower_hip import lower_kernel_to_hip  # noqa: E402

# Emitters that build no kernel, so they have no target-ir output.
NO_KERNEL_FAMILIES = {
    "conv_abi": "prints kernel argument lists, not a kernel",
}

# One family's emitter run: every config, one process. The slowest families
# (the library attention emitters) take seconds; this bounds a hung emitter.
EMITTER_TIMEOUT_S = 900

FAMILIES = run_diff.find_families(python_only=True)


def _emitter_env(src_dir: Path) -> dict[str, str]:
    """The environment the gate gives a Python emitter: the rocke package, the
    library, and the parity directories (for ``_emit_common``) on the path."""
    env = dict(os.environ)
    roots = [
        str(run_diff.PYROOT),
        str(run_diff.LIB_ROOT),
        str(src_dir),
        str(run_diff.PARITY),
    ]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(roots))
    return env


def _target_ir_outputs(name: str, src_dir: Path) -> list[tuple[int, str, str]]:
    """Run one family's emitter in the target-ir mode over its configs; return
    ``(config index, stdout, stderr)`` for every config it ran."""
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
            env=_emitter_env(src_dir),
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
            runs.append((record["idx"], text, record["stderr"]))
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


def _lower_both(ir: str, arch: str) -> tuple[str | None, str | None, str, str]:
    """Each engine's HIP source for one serialized kernel, or its refusal."""
    py = cpp = None
    py_error = cpp_error = ""
    try:
        py = lower_kernel_to_hip(parse(ir), arch=arch)
    except Exception as exc:  # noqa: BLE001 -- a refusal is a result
        py_error = f"{type(exc).__name__}: {exc}"
    try:
        cpp = rocke_engine.lower_serialized_ir_hip(ir, arch)
    except Exception as exc:  # noqa: BLE001 -- a refusal is a result
        cpp_error = str(exc)
    return py, cpp, py_error, cpp_error


def _first_difference(py: str, cpp: str) -> str:
    py_lines, cpp_lines = py.splitlines(), cpp.splitlines()
    for number, (a, b) in enumerate(zip(py_lines, cpp_lines), start=1):
        if a != b:
            return f"line {number}: Python {a.strip()!r}, C++ {b.strip()!r}"
    return f"Python has {len(py_lines)} lines, C++ {len(cpp_lines)}"


@pytest.mark.parametrize(
    ("name", "src_dir"), FAMILIES, ids=[name for name, _ in FAMILIES]
)
def test_both_engines_emit_the_same_hip_source(name: str, src_dir: Path) -> None:
    runs = _target_ir_outputs(name, src_dir)
    if name in NO_KERNEL_FAMILIES:
        assert runs and all("unknown mode" in err for _, _, err in runs), (
            f"{name} is listed as building no kernel "
            f"({NO_KERNEL_FAMILIES[name]}) but accepted the target-ir mode"
        )
        return
    assert not any(
        "unknown mode" in err for _, _, err in runs
    ), f"{name}_emit.py does not accept the target-ir mode"

    compared = 0
    disagreements = []
    for index, text, _ in runs:
        for block, (arch, ir) in enumerate(_kernels(text)):
            where = f"{name} config {index} kernel {block} ({arch})"
            py, cpp, py_error, cpp_error = _lower_both(ir, arch)
            compared += 1
            if py is not None and cpp is not None:
                if py != cpp:
                    disagreements.append(f"{where}: {_first_difference(py, cpp)}")
            elif py is not None:
                disagreements.append(f"{where}: only C++ refuses: {cpp_error}")
            elif cpp is not None:
                disagreements.append(f"{where}: only Python refuses: {py_error}")
    assert compared, f"{name}: the emitter built no kernel in any config"
    assert not disagreements, (
        f"{len(disagreements)} of {compared} {name} kernels lower to different "
        "HIP source:\n" + "\n".join(disagreements[:20])
    )
