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


# --------------------------------------------------------------------------
# multi-axis: one recipe per family covering an axis CROSS PRODUCT
# --------------------------------------------------------------------------
_ND_DRIVER = "rocke.portable_ir.drivers.roll_nd_coverage"

# Two shape axes, fitted from 3 traces and verified at 7 points. 64/512 is 8x the
# base on both axes at once, which no fitted trace comes near.
_CONV_GRID = (
    {"N": 8, "K": 64},
    {"N": 8, "K": 128},
    {"N": 16, "K": 64},
    {"N": 16, "K": 128},
)
_CONV_HOLDOUTS = ({"N": 32, "K": 256}, {"N": 8, "K": 256}, {"N": 64, "K": 512})

# conv C drives a strength-reduced `n // C`, so its constants are a log2 shift and
# a magic multiplier keyed on C's odd part. The holdouts deliberately cover odd
# parts 3, 5 and 7 (192, 160, 224) -- values whose multipliers differ from every
# sampled one, so a recipe that merely froze the sampled constants cannot pass.
_CONV_C_SAMPLES = [64, 96, 128]
_CONV_C_HOLDOUTS = [192, 160, 224, 384]


def test_roll_nd_cross_product():
    """One recipe covers a family's whole non-reduction axis cross product.

    The lanes above roll ONE axis per recipe with the others pinned, so k axes
    cost k recipes and none of them move together. This gates the combined
    model: every grid point AND every extrapolated point reproduces its own
    concrete recording, and the kernel name reconstructs at each one.

    It needs neither librocke nor comgr (the oracle is the Python expander), so
    unlike the lanes above it cannot skip -- a regression here always shows up.
    """
    r = subprocess.run(
        [sys.executable, "-u", "-m", _ND_DRIVER, "--ll"],
        cwd=str(_PLATFORM),
        env=_env(),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout[-8000:] + r.stderr[-4000:]
    assert "families rolled       : 4/4" in r.stdout, (
        "a family stopped rolling over its axis cross product:\n" + r.stdout[-8000:]
    )
    # The refusal list is part of the contract: if an axis starts rolling, the
    # driver trips too, so the frontier in the docs cannot drift silently.
    assert "NOW ROLLS" not in r.stdout, r.stdout[-4000:]


def test_standalone_cli_replays_multi_axis_recipe(tmp_path):
    """A binary with no Python replays a TWO-AXIS recipe to Python's exact .ll.

    This is the claim that matters for the C stack: multi-axis rolling needed no
    VM change, because a multi-axis constant is still just an intexpr over spec
    values. Both axes are bound at the command line (`--int N=.. --int K=..`),
    including at a point 8x the base on both axes at once.
    """
    cli = _replay_cli()
    if cli is None:
        pytest.skip(
            "replay CLI not built; "
            "`cmake --build <build> --target rocke_portable_ir_replay_cli` "
            "or point ROCKE_REPLAY_CLI at it"
        )
    points = list(_CONV_GRID + _CONV_HOLDOUTS)
    author = f"""
import json, sys
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.portable_ir.drivers.roll_hsaco_parity import _conv
from rocke.portable_ir.src import recipe_bundle
from rocke.portable_ir.src.roll_nd import roll_nd

out, flavor = sys.argv[1], sys.argv[2]
r = roll_nd(_conv, axes={{"N": [8, 16], "K": [64, 128]}},
            holdout_points={list(_CONV_HOLDOUTS)})
if not r.ok:
    raise SystemExit("roll_nd failed: " + str(r.reason))
open(out + "/conv.recipe.cbor", "wb").write(recipe_bundle.cbor_encode(r.recipe))
shas = {{}}
for pt in {points}:
    ll = lower_kernel_to_llvm(_conv(**pt), llvm_flavor=flavor, arch="{_ARCH}")
    shas[json.dumps(pt, sort_keys=True)] = __import__("hashlib").sha256(
        ll.encode()).hexdigest()
json.dump(shas, open(out + "/shas.json", "w"))
"""
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm20")
    r = _run([sys.executable, "-c", author, str(tmp_path), flavor])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]

    import json

    want = json.loads((tmp_path / "shas.json").read_text())
    recipe = tmp_path / "conv.recipe.cbor"
    for pt in points:
        cmd = [
            cli,
            "--recipe",
            str(recipe),
            "--cbor",
            "--arch",
            _ARCH,
            "--flavor",
            flavor,
        ]
        for axis, v in pt.items():
            cmd += ["--int", f"{axis}={v}"]
        got = subprocess.run(cmd, capture_output=True, text=True)
        assert got.returncode == 0, f"{pt}: {got.stderr[-2000:]}"
        held = "held-out" if pt in _CONV_HOLDOUTS else "sampled"
        assert (
            hashlib.sha256(got.stdout.encode()).hexdigest()
            == want[json.dumps(pt, sort_keys=True)]
        ), (
            f"{pt} ({held}): standalone replay of the multi-axis recipe "
            f"diverged from the Python lowerer"
        )


def test_standalone_cli_regenerates_magic_division_constants(tmp_path):
    """The C VM regenerates magic-division constants it never recorded.

    conv `C` is the axis that no curve fits: the kernel strength-reduces `n // C`
    into `(umul_hi(n, M) + n) >> s`, where `s` is `ceil(log2 C)` and `M` depends on
    `C`'s odd part. The recipe carries the generating formula instead
    (`magic_multiplier` / `magic_shift`), so this test is really asking whether the
    C VM's arithmetic matches Python's bit for bit at divisors it never saw --
    including odd parts 3, 5 and 7, whose multipliers share no value with any
    sampled point.
    """
    cli = _replay_cli()
    if cli is None:
        pytest.skip(
            "replay CLI not built; "
            "`cmake --build <build> --target rocke_portable_ir_replay_cli` "
            "or point ROCKE_REPLAY_CLI at it"
        )
    points = _CONV_C_SAMPLES + _CONV_C_HOLDOUTS
    author = f"""
import json, sys
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.portable_ir.drivers.roll_hsaco_parity import _conv
from rocke.portable_ir.src import recipe_bundle
from rocke.portable_ir.src.roll_nd import roll_nd

out, flavor = sys.argv[1], sys.argv[2]
r = roll_nd(_conv, axes={{"C": {_CONV_C_SAMPLES}}},
            holdout_points=[{{"C": c}} for c in {_CONV_C_HOLDOUTS}])
if not r.ok:
    raise SystemExit("roll_nd failed: " + str(r.reason))
prog = json.dumps(r.recipe["program"])
for fn in ("magic_multiplier", "magic_shift"):
    if fn not in prog:
        raise SystemExit("expected " + fn + " in the rolled recipe")
open(out + "/convc.recipe.cbor", "wb").write(recipe_bundle.cbor_encode(r.recipe))
shas = {{}}
for c in {points}:
    ll = lower_kernel_to_llvm(_conv(C=c), llvm_flavor=flavor, arch="{_ARCH}")
    shas[str(c)] = __import__("hashlib").sha256(ll.encode()).hexdigest()
json.dump(shas, open(out + "/shas.json", "w"))
"""
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm20")
    r = _run([sys.executable, "-c", author, str(tmp_path), flavor])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]

    import json

    want = json.loads((tmp_path / "shas.json").read_text())
    recipe = tmp_path / "convc.recipe.cbor"
    seen = set()
    for c in points:
        got = subprocess.run(
            [
                cli,
                "--recipe",
                str(recipe),
                "--cbor",
                "--arch",
                _ARCH,
                "--flavor",
                flavor,
                "--int",
                f"C={c}",
            ],
            capture_output=True,
            text=True,
        )
        assert got.returncode == 0, f"C={c}: {got.stderr[-2000:]}"
        sha = hashlib.sha256(got.stdout.encode()).hexdigest()
        held = "held-out" if c in _CONV_C_HOLDOUTS else "sampled"
        assert sha == want[str(c)], (
            f"C={c} ({held}): the C VM's magic-division constants diverged from "
            f"Python's -- the two mirrors of calculate_magic_numbers disagree"
        )
        seen.add(sha)
    # Negative control: if every C produced the same .ll, the comparison above
    # would be vacuous.
    assert len(seen) == len(points), "expected a distinct .ll per C value"


def test_regimes_do_not_oversplit_a_uniform_real_axis():
    """Specializing must stay a last resort, not a default.

    `roll_regimes` splits an axis wherever a recipe stops verifying, which is the
    right rule but also a quiet way to lose compression: split a uniform axis and
    you ship four recipes where one would do, with every check still passing. So
    the lane gates the negative on a real kernel -- gemm `tile_n` rolls as one
    recipe today and must keep coming back as exactly one regime."""
    from rocke.portable_ir.drivers.roll_hsaco_parity import _gemm
    from rocke.portable_ir.src.roll_regimes import roll_regimes

    vals = [32, 64, 128, 256]
    r = roll_regimes(lambda v: _gemm(tile_n=v), axis="tile_n", values=vals)
    assert r.n_recipes == 1, f"uniform axis was split into {r.n_recipes} regimes"
    assert r.regimes[0].values == vals
    # Two traces inferred it; the other two values were verified, not recorded.
    assert r.regimes[0].sampled == vals[:2]


def test_axis_domains_come_from_the_kernels_own_validation():
    """What an axis is worth is a property of the kernel, not of a driver table.

    Rolling an axis saves one concrete recipe per value the axis legally takes, so
    these counts are what should order the work. They are asserted loosely (an
    order of magnitude, not an exact set) because the point is the ranking: the
    axes already rolled have large domains, while `block_n` is capped by having to
    divide `seqlen_kv` and `head_size` accepts two values in total."""
    from kernels.gfx950.attention_dense import AttentionDenseSpec

    from rocke.portable_ir.src.roll_regimes import legal_values

    base = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=128,
        num_kv_heads=8,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_n=64,
        waves_per_eu=2,
    )
    make = lambda **kw: AttentionDenseSpec(**{**base, **kw})  # noqa: E731
    cands = list(range(16, 2049, 16))
    n = {
        a: len(legal_values(a, cands, make))
        for a in ("num_query_heads", "seqlen_kv", "block_n", "head_size")
    }
    assert n["head_size"] == 2, f"head_size domain changed: {n['head_size']}"
    assert n["block_n"] <= 8, f"block_n is bounded by divisors of seqlen_kv: {n}"
    assert (
        n["num_query_heads"] >= 10 * n["block_n"]
    ), f"the rolled axes should dominate the refused ones by domain size: {n}"
