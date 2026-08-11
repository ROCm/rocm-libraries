# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CI wiring for the ROLLED (parametric) recipe path.
#
# test_portable_ir.py gates the concrete path: one recorded trace replays to the
# same .ll as the Python lowerer. That says nothing about rolling, which is the
# step that turns many concrete traces into ONE parametric recipe -- and rolling
# is where a bug is expensive, because a recipe that is subtly wrong away from
# the sampled points ships a wrong kernel rather than failing to build.
#
# So the claim these lanes gate is: for a rolled recipe, Python and the C replay
# produce .ll with the SAME SHA-256, at axis values the roller sampled AND at
# held-out values it never saw. Sampled points only prove the roller can replay
# what it recorded; the held-out points are what distinguish that from having
# actually generalized over the axis.
#
# The digest is usable as the primary gate because the rolled path reproduces
# Python's SSA names (the recipe carries each op's result-name prefix and the
# roller keeps Python's lane naming for loop-carry fans). Before that it could
# only be compared modulo renaming, which forced a comgr compile to get a
# trustworthy artifact-level answer.
#
# The layering, weakest gate to strongest:
#
#   sha parity     rolled .ll digest matches Python's at sampled + held-out
#                  points, over four kernel families and seven axes. Needs a
#                  shared librocke; no comgr, so it runs in a couple of seconds.
#   standalone     the same digest claim, but produced by a binary with no
#                  Python in the process -- the shape this deploys in, and the
#                  only lane that rules out a dependency on interpreter state.
#   hsaco parity   the artifact itself is byte-identical. Strongest, slowest;
#                  needs comgr.
#
# Every lane skips (never silently passes) when its artifact is missing, and the
# skip reason names the command that produces it.

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The concrete-path harness next door already owns artifact discovery (the shared
# librocke, the replay CLI, the subprocess environment); reuse it instead of
# keeping a second copy that can drift. The explicit path insert keeps the import
# working under pytest's importlib mode as well as the default prepend mode.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from test_portable_ir import _env, _online_lib, _replay_cli, _run  # noqa: E402

_DRIVER = "rocke.portable_ir.drivers.roll_hsaco_parity"
_PLATFORM = _HERE.parents[1]

# Rolled on tile_n from these two samples; 128 and 256 are never recorded, so a
# roller that merely replays its samples fails on them.
_GEMM_SAMPLES = (32, 64)
_GEMM_HOLDOUTS = (128, 256)
_ARCH = "gfx950"


def _skip_without_lib():
    lib = _online_lib()
    if lib is None:
        pytest.skip(
            "no shared librocke; build one with "
            "`python -c 'from rocke.portable_ir.src import online; online.build_lib()'` "
            "or point ROCKE_ONLINE_LIB at it"
        )
    return lib


def _have_comgr():
    """Whether a comgr the driver can load is present."""
    probe = (
        "from rocke.runtime.comgr import resolved_lib_rocm_version;"
        "raise SystemExit(0 if resolved_lib_rocm_version() else 1)"
    )
    return _run([sys.executable, "-c", probe]).returncode == 0


def _run_driver(*extra):
    lib = _skip_without_lib()
    return subprocess.run(
        [sys.executable, "-u", "-m", _DRIVER, *extra],
        cwd=str(_PLATFORM),
        env={**_env(), "ROCKE_ONLINE_LIB": lib, "ROCKE_CPP_QUIET_FALLBACK": "1"},
        capture_output=True,
        text=True,
    )


def test_roll_sha_parity():
    """Rolled .ll digest == Python .ll digest, sampled and held-out.

    The primary rolled-path gate: four families, seven axes, 22 points. Stops at
    .ll so it needs no comgr and stays fast enough to run on every change.
    """
    r = _run_driver("--no-hsaco")
    assert r.returncode == 0, r.stdout[-8000:] + r.stderr[-4000:]
    # The driver exits non-zero on a bad point, but it also exits zero if it
    # rolled nothing at all -- which would make this lane vacuous.
    assert "axes rolled          : 7/7" in r.stdout, (
        "the roller declined an axis it used to handle (compression regression, "
        "not a correctness one):\n" + r.stdout[-8000:]
    )


def test_roll_hsaco_parity():
    """The rolled recipe compiles to a byte-identical HSACO.

    The digest lane above compares IR text; this compares the object code the
    device actually runs, which is what closes the gap between "the two engines
    agree" and "the two engines ship the same kernel".
    """
    _skip_without_lib()
    if not _have_comgr():
        pytest.skip("no loadable comgr; install ROCm or set ROCM_PATH")
    r = _run_driver()
    assert r.returncode == 0, r.stdout[-8000:] + r.stderr[-4000:]


def test_standalone_cli_replays_rolled_recipe(tmp_path):
    """A binary with no Python in it replays the PARAMETRIC recipe to Python's .ll.

    test_portable_ir.py runs a concrete recipe through this CLI. A parametric one
    additionally exercises spec binding (`--int tile_n=...`) and the VM's
    static_for/intexpr expansion in a process that never initialized CPython --
    including at a held-out value, so the artifact is doing real work rather than
    replaying a recorded trace.
    """
    cli = _replay_cli()
    if cli is None:
        pytest.skip(
            "replay CLI not built; "
            "`cmake --build <build> --target rocke_portable_ir_replay_cli` "
            "or point ROCKE_REPLAY_CLI at it"
        )

    # Author out-of-process: the recorder rebinds core.ir.IRBuilder, which should
    # not happen inside the pytest process.
    author = f"""
import json, sys
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.portable_ir.drivers.roll_hsaco_parity import _gemm
from rocke.portable_ir.src import recipe_bundle
from rocke.portable_ir.src.roll import roll

out, flavor = sys.argv[1], sys.argv[2]
r = roll(build_at=lambda v: _gemm(tile_n=v), axis="tile_n",
         sample_points={list(_GEMM_SAMPLES)})
if not r.ok:
    raise SystemExit("roll failed: " + str(r.reason))
open(out + "/gemm.recipe.cbor", "wb").write(recipe_bundle.cbor_encode(r.recipe))
shas = {{}}
for v in {list(_GEMM_SAMPLES + _GEMM_HOLDOUTS)}:
    ll = lower_kernel_to_llvm(_gemm(tile_n=v), llvm_flavor=flavor, arch="{_ARCH}")
    open(out + "/py_%d.ll" % v, "w").write(ll)
    shas[v] = __import__("hashlib").sha256(ll.encode()).hexdigest()
json.dump(shas, open(out + "/shas.json", "w"))
"""
    # Pin the flavor: the CLI and the Python lowerer must lower at the same LLVM
    # generation or every comparison fails on the datalayout line alone.
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm20")
    r = _run([sys.executable, "-c", author, str(tmp_path), flavor])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]

    recipe = tmp_path / "gemm.recipe.cbor"
    for v in _GEMM_SAMPLES + _GEMM_HOLDOUTS:
        got = subprocess.run(
            [
                cli,
                "--recipe",
                str(recipe),
                "--cbor",
                "--int",
                f"tile_n={v}",
                "--arch",
                _ARCH,
                "--flavor",
                flavor,
            ],
            capture_output=True,
            text=True,
        )
        assert got.returncode == 0, f"tile_n={v}: {got.stderr[-2000:]}"
        want = (tmp_path / f"py_{v}.ll").read_text()
        held = "held-out" if v in _GEMM_HOLDOUTS else "sampled"
        assert (
            hashlib.sha256(got.stdout.encode()).hexdigest()
            == hashlib.sha256(want.encode()).hexdigest()
        ), (
            f"tile_n={v} ({held}): standalone replay of the rolled recipe "
            f"diverged from the Python lowerer"
        )
