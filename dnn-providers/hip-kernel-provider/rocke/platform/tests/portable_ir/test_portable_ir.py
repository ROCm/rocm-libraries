# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CI wiring for the portable-IR replay path.
#
# The implementation and its drivers live in python/rocke/portable_ir/; this
# module is what makes them run under the normal pytest invocation instead of
# only when someone remembers the driver command line.
#
# The layering, weakest gate to strongest:
#
#   unit tests      the roller, the CBOR bundle codec, and the recorder, in
#                   isolation. Pure Python, always run.
#   record coverage every production kernel the parity emitters can build is
#                   recorded, and the live recording is checked against an
#                   independent post-hoc walk of the same KernelDef. Catches a
#                   recorder that silently drops ops. Pure Python, always run.
#   parity matrix   the real gate: for every kernel and arch, the .ll produced
#                   by replaying through the C++ engine must be BYTE-IDENTICAL
#                   to the Python lowerer's. Needs a shared librocke.
#   standalone CLI  the same byte-identity, but through a binary with no Python
#                   in the process at all -- the shape this actually deploys in.
#                   Needs the CMake target to have been built.
#
# The last two are skipped (never silently passed) when their artifact is
# missing, and the skip reason names the command that produces it.

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_PLATFORM = _HERE.parents[1]  # tests/portable_ir -> tests -> platform
_PYROOT = _PLATFORM / "python"
_LIBRARY = _PLATFORM.parent / "library"


def _env():
    """Subprocess env with the engine package (and the library tree) importable."""
    env = dict(os.environ)
    parts = [str(_PYROOT)]
    if (_LIBRARY / "kernels").is_dir():
        parts.append(str(_LIBRARY))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # The oracle for every byte-identity claim below is the Python lowerer, and
    # the C++ pybind extension is not on this path; say so rather than emitting
    # a fallback warning per kernel.
    env["ROCKE_BACKEND"] = "python"
    return env


def _online_lib():
    """Path to a prebuilt shared librocke, or None."""
    import tempfile

    for cand in (
        os.environ.get("ROCKE_ONLINE_LIB"),
        os.path.join(tempfile.gettempdir(), "rocke_online", "librocke.so"),
        str(_PLATFORM / "build" / "librocke.so"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


def _replay_cli():
    """Path to the built standalone replay CLI, or None."""
    env = os.environ.get("ROCKE_REPLAY_CLI")
    if env and os.path.exists(env):
        return env
    import tempfile

    roots = [
        Path(tempfile.gettempdir()) / "rocke_verify",
        Path(tempfile.gettempdir()) / "rocke_online" / "core",
        _PLATFORM / "build",
    ]
    for root in roots:
        cand = root / "tests" / "rocke_portable_ir_replay_cli"
        if cand.exists():
            return str(cand)
    return None


def _run(args, **kw):
    return subprocess.run(
        args, cwd=str(_PLATFORM), env=_env(), capture_output=True, text=True, **kw
    )


# ---------------------------------------------------------------------
# Always-on lanes (pure Python, no engine binary)
# ---------------------------------------------------------------------
_UNIT_TEST_DIR = _PLATFORM / "python" / "rocke" / "portable_ir" / "tests"
# Discovered rather than listed. A hardcoded list silently stops covering new
# test modules -- test_roll_nd and test_roll_regimes were both missing from it --
# and a test that never runs is worse than no test, because it reads as covered.
_UNIT_MODULES = sorted(p.stem for p in _UNIT_TEST_DIR.glob("test_*.py"))


@pytest.mark.parametrize("mod", _UNIT_MODULES)
def test_unit(mod):
    """The in-package unit tests, which pytest does not otherwise collect
    (they live under python/rocke/, not under tests/)."""
    r = _run([sys.executable, "-m", "unittest", "-v", f"rocke.portable_ir.tests.{mod}"])
    assert r.returncode == 0, r.stderr[-4000:] or r.stdout[-4000:]


def test_record_coverage():
    """Every buildable production kernel records faithfully.

    This is what keeps the recorder honest as new kernels land: a builder that
    uses an IRBuilder call the recorder does not intercept shows up here as a
    recorder gap rather than as a mysterious parity failure later.
    """
    r = _run([sys.executable, "-m", "rocke.portable_ir.drivers.record_coverage"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]


# ---------------------------------------------------------------------
# Engine-binary lanes
# ---------------------------------------------------------------------
def test_parity_matrix():
    """Replayed .ll == Python .ll, byte for byte, for every kernel x arch.

    Covers both replay paths: the concrete-graph importer (ir_export -> C
    import) and the recipe VM (record -> CBOR -> C VM). Byte-identity, not
    just equivalence -- concrete recipes carry the SSA names, so the VM
    reproduces Python's naming exactly.
    """
    lib = _online_lib()
    if lib is None:
        pytest.skip(
            "no shared librocke; build one with "
            "`python -c 'from rocke.portable_ir.src import online; online.build_lib()'` "
            "or point ROCKE_ONLINE_LIB at it"
        )
    env_lib = dict(ROCKE_ONLINE_LIB=lib)
    r = subprocess.run(
        [sys.executable, "-m", "rocke.portable_ir.drivers.parity_matrix"],
        cwd=str(_PLATFORM),
        env={**_env(), **env_lib},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-4000:]


def test_standalone_cli_matches_python(tmp_path):
    """A binary with no Python in it reproduces the Python lowerer's .ll.

    parity_matrix proves the C++ engine agrees, but it drives that engine over
    ctypes from inside a Python process, so it cannot by itself rule out a
    dependency on something the interpreter set up. This runs the artifacts
    through a standalone executable instead, which is how the runtime ships.
    Both artifact forms are checked: the concrete IR JSON and the CBOR recipe.
    """
    cli = _replay_cli()
    if cli is None:
        pytest.skip(
            "replay CLI not built; "
            "`cmake --build <build> --target rocke_portable_ir_replay_cli` "
            "or point ROCKE_REPLAY_CLI at it"
        )

    # Author the artifacts out-of-process so this test does not import the IR
    # stack into the pytest process (the recorder patches core.ir.IRBuilder).
    author = f"""
import sys
from rocke.core import ir_export
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.elementwise import ElementwiseSpec, build_elementwise
from rocke.portable_ir.src import recipe_bundle
from rocke.portable_ir.src.recording_builder import record_kernel

out = sys.argv[1]
spec = ElementwiseSpec(op="add")
kernel, recipe = record_kernel(lambda: build_elementwise(spec))
open(out + "/k.ir.json", "w").write(ir_export.export_kernel_ir_json(kernel))
open(out + "/k.recipe.cbor", "wb").write(recipe_bundle.cbor_encode(recipe))
open(out + "/py.ll", "w").write(
    lower_kernel_to_llvm(kernel, llvm_flavor="llvm20", arch="gfx950"))
"""
    r = _run([sys.executable, "-c", author, str(tmp_path)])
    assert r.returncode == 0, r.stderr[-4000:]

    expected = (tmp_path / "py.ll").read_text()
    for args, label in (
        (["--ir", str(tmp_path / "k.ir.json")], "portable IR JSON"),
        (["--recipe", str(tmp_path / "k.recipe.cbor"), "--cbor"], "CBOR recipe"),
    ):
        got = subprocess.run(
            [cli, *args, "--arch", "gfx950", "--flavor", "llvm20"],
            capture_output=True,
            text=True,
        )
        assert got.returncode == 0, f"{label}: {got.stderr[-2000:]}"
        assert (
            got.stdout == expected
        ), f"{label}: standalone replay diverged from the Python lowerer"
