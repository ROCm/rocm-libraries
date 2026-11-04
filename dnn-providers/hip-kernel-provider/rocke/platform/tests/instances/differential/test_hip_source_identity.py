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
Python builder rejects prints nothing and has nothing to lower; the test checks
that the ``ir`` mode rejects it with the same error, so a failure of the
target-ir path cannot pass as a rejection, and it fails any config that exits
non-zero after printing output.

The family test needs the ``rocke_engine`` extension and is skipped without it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from rocke.tests.engines import (
    REFUSAL_ERROR_TYPES,
    HipRefusal,
    hip_outcomes_of_both_engines,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_diff  # noqa: E402 -- the differential harness is not a package

# Emitters that build no kernel, so they have no target-ir output.
NO_KERNEL_FAMILIES = {
    "conv_abi": "prints kernel argument lists, not a kernel",
}

# One family's emitter run: every config, one process. The slowest families
# (the library attention emitters) take seconds; this bounds a hung emitter.
EMITTER_TIMEOUT_S = 900

# How a builder refusing a config ends: the exception line of one of these.
_REFUSAL_TYPE_NAMES = frozenset(
    error_type.__name__ for error_type in REFUSAL_ERROR_TYPES
)

FAMILIES = run_diff.find_families(python_only=True)


def _emitter_runs(
    name: str, src_dir: Path, mode: str, indices: list[int]
) -> list[tuple[int, str, run_diff.Emission]]:
    """Run one family's emitter in ``mode`` over the configs ``indices``, in one
    process; return ``(config index, stdout, emission)`` for every config it ran."""
    emitter = src_dir / f"{name}_emit.py"
    with tempfile.TemporaryDirectory(prefix=f"hip_identity_{name}_") as out_dir:
        proc = subprocess.run(
            [
                sys.executable,
                str(run_diff.PYTHON_BATCH_WORKER),
                "--output-dir",
                out_dir,
                str(emitter),
                mode,
                *map(str, indices),
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


def _unexpected_failures(
    name: str, src_dir: Path, runs: list[tuple[int, str, run_diff.Emission]]
) -> list[str]:
    """The configs of ``runs`` (target-ir mode) that failed for any reason but
    the end of the config list or the builder refusing the config.

    A config that exits non-zero after printing output, or exits zero with
    none, is a failure: its kernels would silently drop out of the comparison.
    A config that fails with no output is the builder's refusal only if it
    raised one of the refusal types (``REFUSAL_ERROR_TYPES``), so a builder
    crashing with, say, an AttributeError is a failure, and only if the ``ir``
    mode, which builds the same kernel and only serializes it, refuses it with
    the same error; otherwise the target-ir path itself failed."""
    failures = []
    refused: dict[int, str] = {}
    for index, _, emission in runs:
        if run_diff.is_end_emission(emission):
            continue
        if emission.returncode == 0 and emission.has_output:
            continue
        if emission.has_output:
            failures.append(
                f"config {index}: exited {emission.returncode} after printing "
                f"output: {run_diff.last_error_line(emission)}"
            )
        elif emission.returncode == 0:
            failures.append(f"config {index}: exited 0 without printing a kernel")
        else:
            error = run_diff.last_error_line(emission)
            if error.split(":", 1)[0] in _REFUSAL_TYPE_NAMES:
                refused[index] = error
            else:
                failures.append(f"config {index}: failed, not refused: {error}")
    if refused:
        for index, _, emission in _emitter_runs(name, src_dir, "ir", list(refused)):
            ir_error = run_diff.last_error_line(emission)
            if emission.returncode == 0 or emission.has_output:
                failures.append(
                    f"config {index}: failed in target-ir ({refused[index]}) but "
                    "builds in ir"
                )
            elif ir_error != refused[index]:
                failures.append(
                    f"config {index}: failed in target-ir ({refused[index]}) and "
                    f"in ir with another error ({ir_error})"
                )
    return failures


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


# An emitter with one config per way a config can end, for the check below.
_OUTCOME_EMITTER = """
import sys

def main():
    index, mode = int(sys.argv[1]), sys.argv[2]
    if index == 0:
        print("kernel")
    elif index == 1:
        raise ValueError("the builder refuses this config")
    elif index == 2:
        if mode == "target-ir":
            raise ValueError("the target-ir path broke")
        print("kernel")
    elif index == 3:
        print("half a kernel")
        raise RuntimeError("failed after printing")
    elif index == 4:
        return 0
    elif index == 5:
        raise ValueError(f"refused as {mode}")
    elif index == 6:
        raise AttributeError("the builder broke")
    else:
        sys.stderr.write(f"unknown config {index}\\n")
        return 1
"""


def test_only_builder_refusals_and_the_end_of_the_list_pass(tmp_path: Path) -> None:
    (tmp_path / "outcomes_emit.py").write_text(_OUTCOME_EMITTER, encoding="utf-8")
    runs = _emitter_runs("outcomes", tmp_path, "target-ir", list(range(9)))
    assert [index for index, _, _ in runs] == [0, 1, 2, 3, 4, 5, 6, 7]
    failures = _unexpected_failures("outcomes", tmp_path, runs)
    assert failures == [
        "config 3: exited 1 after printing output: RuntimeError: failed after printing",
        "config 4: exited 0 without printing a kernel",
        "config 6: failed, not refused: AttributeError: the builder broke",
        (
            "config 2: failed in target-ir (ValueError: the target-ir path "
            "broke) but builds in ir"
        ),
        (
            "config 5: failed in target-ir (ValueError: refused as target-ir) "
            "and in ir with another error (ValueError: refused as ir)"
        ),
    ]


@pytest.mark.parametrize(
    ("name", "src_dir"), FAMILIES, ids=[name for name, _ in FAMILIES]
)
def test_both_engines_emit_the_same_hip_source(name: str, src_dir: Path) -> None:
    pytest.importorskip("rocke_engine")
    runs = _emitter_runs(
        name, src_dir, "target-ir", list(range(run_diff.config_limit(name)))
    )
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
    failures = _unexpected_failures(name, src_dir, runs)
    assert not failures, (
        f"{name}: {len(failures)} configs failed other than by the builder "
        "refusing them:\n" + "\n".join(failures[:20])
    )

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
