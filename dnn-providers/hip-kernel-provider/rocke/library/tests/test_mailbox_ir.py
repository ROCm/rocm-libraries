# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``kernels.gfx950.qwen38_mono.mailbox_ir``: the Python-engine-only ``ir``
mailbox implementation (``buffer_store_cp`` + ``tile.tagged_poll``,
``int_to_ptr``), driven through the platform example microkernels
(``rocke.examples.gfx950.persistent_mailbox``).

Builds using it are excluded from cpp parity: every one must compile with the
Python engine, chosen explicitly (never a silent cpp->python fallback); the
LLVM text checks here use the Python lowerer directly, so they also pass
under ``ROCKE_CPP_STRICT=1``.

CPU: registration, engine requirement, emitted cache bits / pause / fences.
GPU (gfx942 / gfx950): ping-pong for every knob value, NUMA dual copy,
tag wrap, bulk hand-off, the full-grid all-to-all in a subprocess with a
timeout. Environment as in ``platform/tests/test_mailbox_helpers.py``
(``ROCKE_HIP_LIB`` must match ``ROCKE_COMGR_LIB``).
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from kernels.gfx950.qwen38_mono import mailbox_ir as MI
from rocke.core.ir import I32, I64, IRBuilder, PtrType, uses_python_only_ops
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.gfx950.persistent_mailbox import kernels as K
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

FLAVOR = "llvm22"
IR_KNOBS = [
    M.MailboxKnobs(impl="ir", scope=s, pause=p)
    for s, p in itertools.product(("agent", "system"), ("s_nop", "s_sleep1", "none"))
]
_LIB_ROOT = str(Path(__file__).resolve().parents[1])


def _pp(kn, pairs=1):
    return K.build_pingpong(K.PingPongSpec(knobs=kn, pairs=pairs))


# --------------------------------------------------------------------- CPU


def test_ir_impl_registered_and_named():
    assert "ir" in M.registered_impls()
    assert M.impl_for("ir") is MI.IR_IMPL
    assert K.load_ir_impl()
    assert M.MailboxKnobs(impl="ir", pause="s_sleep1").suffix() == "mbir_mbs_sleep1"


def test_ptr_from_i64_ir():
    b = IRBuilder("p")
    a = b.param("a", I64)
    p = MI.ptr_from_i64_ir(b, a)
    assert p.type == PtrType(I32, "global")
    with pytest.raises(TypeError):
        MI.ptr_from_i64_ir(b, b.param("x", I32))
    b.ret()
    assert uses_python_only_ops(b.kernel) == ["tile.int_to_ptr"]


def test_mailbox_from_i64_uses_int_to_ptr():
    b = IRBuilder("m")
    a = b.param("a", I64)
    kn = M.MailboxKnobs(impl="ir")
    mb = M.mailbox_from_i64(b, a, b.const_i32(64), kn)
    M.put(b, mb, 0, b.const_i32(1), b.const_i32(2), kn)
    b.ret()
    assert uses_python_only_ops(b.kernel) == [
        "tile.buffer_store_cp",
        "tile.int_to_ptr",
    ]


@pytest.mark.parametrize("kn", IR_KNOBS, ids=lambda k: k.suffix())
def test_engine_requirement_and_backend(kn):

    k = _pp(kn)
    used = uses_python_only_ops(k)
    assert "tile.tagged_poll" in used and "tile.buffer_store_cp" in used
    assert MI.backend_for(k) == "python"
    assert MI.backend_for(_pp(M.MailboxKnobs())) is None
    assert k.name.endswith(kn.suffix())


@pytest.mark.parametrize("kn", IR_KNOBS, ids=lambda k: k.suffix())
def test_lowering_carries_cache_bits_and_pause(kn):
    ll = _lower_kernel_to_llvm_python(_pp(kn, 2), arch="gfx950", llvm_flavor=FLAVOR)
    pause = D.PAUSES[kn.pause]
    assert re.search(rf"buffer\.store\.v4i32\(.*i32 {kn.cpol}\)", ll)
    assert re.search(rf"buffer\.load\.v4i32\(.*i32 0, i32 {kn.cpol}\)", ll)
    assert f'call void asm sideeffect "{pause}", ""()' in ll


def test_put_nt_sets_nt_bit():
    kn = M.MailboxKnobs(impl="ir", put_nt=True)
    ll = _lower_kernel_to_llvm_python(_pp(kn), arch="gfx950", llvm_flavor=FLAVOR)
    assert re.search(r"buffer\.store\.v2i32\(.*i32 18\)", ll)


def test_release_acquire_ir_paths_never_use_sync():
    for fence_impl in ("asm", "ir"):
        kn = M.MailboxKnobs(impl="ir", fence_impl=fence_impl)
        b = IRBuilder(f"rel_ir_{fence_impl}")
        box = b.param("box", PtrType(I32, "global"))
        tag = b.readfirstlane(b.param("tag", I32))
        mb = M.mailbox_from_ptr(b, box, b.const_i32(4096), kn)
        M.release_bulk(b, mb, 0, tag, kn)
        M.acquire_bulk(b, mb, 0, tag, kn)
        M.pre_poll(b, lambda m, mb=mb: (mb, m), 100, tag, kn)
        b.ret()
        ll = _lower_kernel_to_llvm_python(b.kernel, arch="gfx950", llvm_flavor=FLAVOR)
        assert "tile.sync" not in ll
        if fence_impl == "asm":
            assert "buffer_wbl2 sc1" in ll and "buffer_inv sc1" in ll
        else:
            assert 'fence syncscope("agent") release' in ll
            assert 'fence syncscope("agent") acquire' in ll


def test_every_microkernel_verifies():
    from rocke.core.verify import verify

    kn = M.MailboxKnobs(impl="ir")
    kn2 = M.MailboxKnobs(impl="ir", numa_copies=2)
    ks = [K.build_tag_wrap_kernel(kn)]
    ks += [_pp(k, p) for p in (1, 2) for k in (kn, kn2)]
    ks += [
        K.build_all_to_all(K.AllToAllSpec(knobs=k, consume=c, check=ch))
        for c in K.A2A_CONSUME
        for ch in (False, True)
        for k in (kn, kn2)
    ]
    ks += [K.build_bulk_kernel(K.BulkSpec(knobs=kn, rows=r)) for r in (1, 8)]
    for k in ks:
        assert verify(k) == [], k.name
        assert MI.backend_for(k) == "python", k.name


# --------------------------------------------------------------------- GPU


@pytest.fixture(scope="module")
def arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        target = get_device_arch(0)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no HIP device: {str(e)[:120]}")
    if target not in ("gfx942", "gfx950"):
        pytest.skip(f"requires a gfx942 or gfx950 device; detected {target}")
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    why = H.env_problem()
    if why:
        pytest.skip(why)
    return target


@pytest.fixture()
def dev(arch):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    d = H.Dev(watchdog_s=120)
    yield d
    d.close()


@pytest.mark.parametrize("kn", IR_KNOBS, ids=lambda k: k.suffix())
@pytest.mark.parametrize("pairs", [1, 2])
def test_gpu_pingpong(arch, dev, kn, pairs):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    for dist in (1, 8):
        rs = H.run_pingpong(
            dev,
            K.PingPongSpec(knobs=kn, pairs=pairs),
            arch,
            a=0,
            bb=dist,
            iters=200,
            reps=2,
        )
        for r in rs:
            assert r.backend == "python"
            assert r.ok and r.monotone, r


@pytest.mark.parametrize("pairs", [1, 2])
def test_gpu_pingpong_numa_copies(arch, dev, pairs):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    kn = M.MailboxKnobs(impl="ir", numa_copies=2)
    for dist in (1, 4, 8):
        for r in H.run_pingpong(
            dev,
            K.PingPongSpec(knobs=kn, pairs=pairs),
            arch,
            a=0,
            bb=dist,
            iters=200,
            reps=2,
        ):
            assert r.ok and r.monotone, r


def test_gpu_tag_wrap(arch, dev):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    wrap = 1 << 26
    recs = H.run_tag_wrap(
        dev,
        arch,
        [wrap - 2, wrap - 1, wrap, wrap + 1],
        [0, 47, 48, 62],
        knobs=M.MailboxKnobs(impl="ir"),
    )
    for r in recs:
        assert r["ok"] and r["tag"] != 0 and r["backend"] == "python", r


@pytest.mark.parametrize("fence_impl", ["asm", "ir"])
@pytest.mark.parametrize("rows", [1, 8])
def test_gpu_bulk_handoff(arch, dev, fence_impl, rows):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    r = H.run_bulk(
        dev,
        K.BulkSpec(knobs=M.MailboxKnobs(impl="ir", fence_impl=fence_impl), rows=rows),
        arch,
        steps=12,
    )
    assert r["ok"] and r["backend"] == "python", r


_A2A = """
import sys
from rocke.examples.gfx950.persistent_mailbox import harness as H, kernels as K
from rocke.helpers import mailbox as M
assert K.load_ir_impl(), "ir impl not importable"
kn = M.MailboxKnobs(impl="ir", pause=sys.argv[2], numa_copies=int(sys.argv[3]))
dev = H.Dev()
spec = K.AllToAllSpec(knobs=kn, consume=sys.argv[1], check=True, barrier=sys.argv[4])
rs = H.run_all_to_all(dev, spec, "gfx950", phases=20, reps=2)
dev.close()
print("A2A_OK" if all(r.ok for r in rs) else "A2A_BAD", rs[0].backend)
"""


def _a2a(consume, pause, numa, barrier="bare", timeout=240):
    env = dict(os.environ)
    env["PYTHONPATH"] = _LIB_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run(
            [sys.executable, "-c", _A2A, consume, pause, str(numa), barrier],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"ir all-to-all {consume}/{pause}/numa{numa}/{barrier} hung")
    assert "A2A_OK python" in p.stdout, (p.stdout[-2000:], p.stderr[-2000:])


@pytest.mark.parametrize("consume", K.A2A_CONSUME)
@pytest.mark.parametrize("pause", ["s_nop", "s_sleep1"])
def test_gpu_all_to_all_full_grid_never_hangs(arch, consume, pause):
    _a2a(consume, pause, 1, timeout=180)


@pytest.mark.parametrize("consume", K.A2A_CONSUME)
def test_gpu_all_to_all_numa_copies(arch, consume):
    _a2a(consume, "s_nop", 2)


def test_gpu_all_to_all_sync_barrier(arch):
    _a2a("wave0", "s_nop", 1, barrier="sync", timeout=180)
