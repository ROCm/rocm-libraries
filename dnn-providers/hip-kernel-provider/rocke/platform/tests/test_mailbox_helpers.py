# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``rocke.helpers.mailbox`` / ``rocke.helpers.device_ops`` (persistent kernels).

These helpers ship the inline-asm mailbox implementation only (both engines).
The Python-engine-only ``ir`` implementation lives in the rocKE library
(``kernels.gfx950.qwen38_mono.mailbox_ir``) and is tested in
``library/tests/test_mailbox_ir.py``.

CPU: knob validation, the tag rule and its wrap guard, the asm poll template,
that the helpers emit no Python-only opcode, the emitted LLVM, and the day-1
gate's IR half (multi-line ``${:uid}`` asm survives ``ir_serialize`` and, with
the binding built, the C++ engine emits the same bytes). With COMGR and
``llvm-objdump``: the ISA of the release / acquire / pre-poll / payload-wait
paths has no ``vmcnt`` drain ahead of a barrier beyond the intended release
one (and a ``b.sync()`` control shows the check sees one).

GPU (gfx942 / gfx950): the day-1 gate (``ptr_from_i64`` in every form, an
asm-built buffer descriptor), ping-pong hand-offs for every knob value with
payload and cross-XCD stamp-order checks, the bulk hand-off (rows, bf16
pairs, scoped atomics, payload_wait), the device-op probe (wave max, fresh
copies, clocks), the full-grid 160-producer all-to-all in a subprocess with a
timeout (a hang fails, never wedges the run), and the tag rule across the u32
wrap.

Environment for the GPU half: the HIP runtime must match the COMGR rocKE
compiles with. In the q38 venv job mode set
``ROCKE_HIP_LIB=<torch>/lib/libamdhip64.so`` (``ROCKE_COMGR_LIB`` already
points at torch's COMGR); otherwise allocations / code objects fail. The GPU
fixtures skip with that message when they detect the mismatch
(``harness.env_problem``).
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from rocke.core.ir import I32, I64, IRBuilder, PtrType, uses_python_only_ops
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx950.persistent_mailbox import kernels as K
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

FLAVOR = "llvm22"
ALL_KNOBS = [
    M.MailboxKnobs(scope=s, pause=p)
    for s, p in itertools.product(("agent", "system"), ("s_nop", "s_sleep1", "none"))
]
# Builder methods that emit an opcode in ``rocke.core.ir.PYTHON_ONLY_OPCODES``
_PY_ONLY_BUILDERS = (
    "int_to_ptr",
    "buffer_load_cp",
    "buffer_store_cp",
    "global_load_atomic",
    "global_store_atomic",
    "tagged_poll",
)


def _lower(k, backend="python"):
    os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")
    from rocke.helpers.compile import _lower_llvm_via_backend

    return _lower_llvm_via_backend(k, arch="gfx950", backend=backend, spec=None)


def _op_names(region):
    for op in region.ops:
        yield op.name
        for r in op.regions:
            yield from _op_names(r)


# --------------------------------------------------------------------- CPU


def test_knobs_validate_and_name_deviations_only():
    assert M.MailboxKnobs().suffix() == ""
    assert M.MailboxKnobs(pause="s_sleep1").suffix() == "mbs_sleep1"
    assert M.MailboxKnobs(numa_copies=2, put_nt=True).suffix() == "mbnt_mbnuma2"
    assert M.MailboxKnobs(scope="system").cpol == 17
    assert M.MailboxKnobs().cpol == 16
    for bad in (
        {"impl": "x"},
        {"scope": "cluster"},
        {"pause": "spin"},
        {"poll_max": 0},
        {"fence_impl": "y"},
        {"numa_copies": 3},
    ):
        with pytest.raises(ValueError):
            M.MailboxKnobs(**bad)


def test_helpers_ship_only_the_asm_impl():
    """``rocke.helpers`` alone registers ``asm`` only; ``impl="ir"`` names the
    library module that provides it (fresh interpreter: no library import)."""
    code = (
        "from rocke.helpers import mailbox as M\n"
        "assert M.registered_impls() == ('asm',), M.registered_impls()\n"
        "try:\n"
        "    M.MailboxKnobs(impl='ir')\n"
        "except ValueError as e:\n"
        "    assert 'kernels.gfx950.qwen38_mono.mailbox_ir' in str(e), e\n"
        "else:\n"
        "    raise SystemExit('impl=ir accepted without the library')\n"
        "print('OK')\n"
    )
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert "OK" in p.stdout, (p.stdout, p.stderr[-2000:])


def test_register_impl_rejects_name_clash():
    class Other(M.MailboxImpl):
        name = "asm"

    with pytest.raises(ValueError):
        M.register_impl(Other())
    assert M.register_impl(type(M.ASM_IMPL)()) is M.ASM_IMPL
    with pytest.raises(ValueError):
        M.impl_for("nope")


def test_helper_sources_use_no_python_only_builder():
    """``platform/AGENTS.md``: no helper without a C++ port path. The helper
    modules never call a builder method that emits a Python-only opcode."""
    for mod in (M, D):
        src = Path(mod.__file__).read_text()
        for name in _PY_ONLY_BUILDERS:
            assert not re.search(rf"\b\w+\.{name}\(", src), (mod.__name__, name)


def test_layer_guard_keeps_every_tag_nonzero():
    slots = M.LAYER_SLOTS
    # host model of the tag rule over the whole u32 period near the wrap
    for step in (0, 1, (1 << 26) - 2, (1 << 26) - 1, 1 << 26, (1 << 27) - 1):
        for layer in range(slots - 1):
            assert M.check_layer(layer) == layer
            assert (step * slots + layer + 1) & 0xFFFFFFFF != 0
    # the excluded layer is exactly the one whose tag wraps to 0
    assert (((1 << 26) - 1) * slots + (slots - 1) + 1) & 0xFFFFFFFF == 0
    with pytest.raises(ValueError):
        M.check_layer(slots - 1)
    with pytest.raises(ValueError):
        M.check_layer(-1)
    b = IRBuilder("t")
    with pytest.raises(ValueError):
        M.launch_tag(b, b.const_i32(0), slots - 1)


def test_runtime_layer_launcher_checks_layers():
    """The tag-wrap launcher checks runtime layer arguments on the host
    (``launch_tag`` cannot check a ``Value`` layer); it raises before any
    device work, so this needs no GPU."""
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    with pytest.raises(ValueError):
        H.run_tag_wrap(None, "gfx950", [0], [0, 63])


def test_asm_poll_template_operand_numbering():
    tmpl, cons = M.asm_poll_template([1, 2], 2, [0, 1], "sc1", "s_nop 0")
    # 3 pair outputs ($0..$2), tmp $3, mask $4, voffs $5 $6, bases $7 $8, tag $9
    assert cons.split(",") == [
        "=&v",
        "=&v",
        "=&v",
        "=&v",
        "=&s",
        "v",
        "v",
        "s",
        "s",
        "s",
        "~{vcc}",
        "~{scc}",
    ]
    lines = tmpl.split("\n")
    assert lines[0] == "${:private}poll${:uid}_top:"
    assert "global_load_dwordx2 $0, $5, $7 sc1" in lines
    assert "global_load_dwordx2 $1, $6, $8 sc1" in lines
    assert "global_load_dwordx2 $2, $6, $8 offset:8 sc1" in lines
    assert lines.count("s_waitcnt vmcnt(0)") == 1
    assert "v_cmp_ne_u64_e64 $4, $3, $9" in lines
    assert lines.count("s_or_b64 $4, $4, vcc") == 2
    assert lines[-1] == "${:private}poll${:uid}_done:"
    assert "s_cbranch_scc0 ${:private}poll${:uid}_done" in lines


def _pp(kn, pairs=1):
    return K.build_pingpong(K.PingPongSpec(knobs=kn, pairs=pairs))


def test_a2a_host_model_matches_mapping():
    """Every consume mode reads every producer at least once per round."""
    for consume in K.A2A_CONSUME:
        spec = K.AllToAllSpec(consume=consume)
        tot = sum(K.a2a_expected_thread_sum(spec, 0, t) for t in range(spec.threads))
        assert tot >= sum(range(spec.producers)), consume


def _helper_kernels():
    kn = M.MailboxKnobs()
    kn2 = M.MailboxKnobs(numa_copies=2)
    ks = [K.build_tag_wrap_kernel(kn), K.build_stamp_kernel()]
    ks += [_pp(k, p) for p in (1, 2) for k in (kn, kn2)]
    ks += [
        K.build_all_to_all(K.AllToAllSpec(knobs=k, consume=c, check=ch))
        for c in K.A2A_CONSUME
        for ch in (False, True)
        for k in (kn, kn2)
    ]
    ks += [K.build_all_to_all(K.AllToAllSpec(barrier="sync", check=True))]
    ks += [
        K.build_bulk_kernel(K.BulkSpec(knobs=M.MailboxKnobs(fence_impl=f), rows=r))
        for f in ("asm", "ir")
        for r in (1, 8)
    ]
    ks += [K.build_bulk_kernel(K.BulkSpec(fence="norelease"))]
    ks += [K.build_device_ops_kernel()]
    ks += [K.build_gate_kernel(f) for f in K.GATE_FORMS if f != "ir"]
    return ks


def test_every_microkernel_verifies_and_needs_no_python_only_op():
    from rocke.core.verify import verify

    for k in _helper_kernels():
        assert verify(k) == [], k.name
        assert uses_python_only_ops(k) == [], k.name
    # the example's ir gate form is the one deliberate Python-only build
    g = K.build_gate_kernel("ir")
    assert verify(g) == []
    assert uses_python_only_ops(g) == ["tile.int_to_ptr"]


def test_put_nt_knob_sets_nt_bit():
    kn = M.MailboxKnobs(put_nt=True)
    assert kn.suffix().endswith("mbnt")
    ll = lower_kernel_to_llvm(_pp(kn), arch="gfx950", llvm_flavor=FLAVOR)
    assert "global_store_dwordx2 $0, $1, $2 sc1 nt" in ll


@pytest.mark.parametrize("kn", ALL_KNOBS, ids=lambda k: k.suffix() or "default")
def test_lowering_carries_cache_bits_and_pause(kn):
    ll = lower_kernel_to_llvm(_pp(kn, 2), arch="gfx950", llvm_flavor=FLAVOR)
    pause = D.PAUSES[kn.pause]
    assert f"global_store_dwordx4 $0, $1, $2 {kn.cpol_asm}" in ll
    assert re.search(rf"global_load_dwordx2 \$\d+, \$\d+, \$\d+ {kn.cpol_asm}\\0A", ll)
    assert (
        ("\\0A" + pause + "\\0As_branch") in ll
        if pause
        else ("s_cbranch_scc0 ${:private}poll${:uid}_done\\0As_branch" in ll)
    )


def test_device_op_lowering():
    b = IRBuilder("dops")
    a = b.readfirstlane(b.param("a", I64))
    x = b.param("x", I32)
    D.atomic_add_scoped(b, a, x, x)
    D.atomic_add_scoped(b, a, x, x, scope="system")
    r = D.atomic_add_scoped(b, a, x, x, scope="workgroup", returns=True)
    r2 = D.atomic_add_scoped(b, a, x, r, scope="system", returns=True)
    D.fresh_v(b, r2)
    D.fresh_s(b, b.readfirstlane(x))
    b.ret()
    ll = lower_kernel_to_llvm(b.kernel, arch="gfx950", llvm_flavor=FLAVOR)
    assert '"global_atomic_add $0, $1, $2", "v,v,s,~{memory}"' in ll
    assert '"global_atomic_add $0, $1, $2 sc1", "v,v,s,~{memory}"' in ll
    assert "global_atomic_add $0, $1, $2, $3 sc0\\0As_waitcnt vmcnt(0)" in ll
    assert "global_atomic_add $0, $1, $2, $3 sc0 sc1\\0As_waitcnt" in ll
    assert '"; fresh $0", "=v,0"' in ll and '"; fresh $0", "=s,0"' in ll
    for bad in (
        lambda: D.atomic_add_scoped(b, a, x, x, scope="cluster"),
        lambda: D.fresh(b, x, "a"),
    ):
        with pytest.raises(ValueError):
            bad()


def test_payload_wait_is_barrier_or_pre_poll():
    for rows, want_bars, want_poll in ((1, 1, False), (4, 1, False), (5, 1, True)):
        b = IRBuilder(f"pw{rows}")
        box = b.param("box", PtrType(I32, "global"))
        tag = b.readfirstlane(b.param("tag", I32))
        mb = M.mailbox_from_ptr(b, box, b.const_i32(4096))
        M.payload_wait(b, lambda m, mb=mb: (mb, m), 100, tag, rows)
        b.ret()
        ops = list(_op_names(b.kernel.body))
        assert ops.count("tile.s_barrier_bare") == want_bars
        assert "tile.sync" not in ops
        assert ("tile.inline_asm" in ops) == want_poll, ops
    with pytest.raises(ValueError):
        M.payload_wait(b, lambda m: (mb, m), 1, tag, 0)


def test_release_acquire_and_pre_poll_never_use_sync():
    """``b.sync()`` drains vmcnt (and so the weight prefetch); the bulk
    hand-off and the pre-poll throttle must use the bare barrier."""
    for fence_impl in ("asm", "ir"):
        kn = M.MailboxKnobs(fence_impl=fence_impl)
        b = IRBuilder(f"rel_{fence_impl}")
        box = b.param("box", PtrType(I32, "global"))
        tag = b.readfirstlane(b.param("tag", I32))
        mb = M.mailbox_from_ptr(b, box, b.const_i32(4096), kn)
        M.release_bulk(b, mb, 0, tag, kn)
        M.acquire_bulk(b, mb, 0, tag, kn)
        M.pre_poll(b, lambda m, mb=mb: (mb, m), 100, tag, kn)
        b.ret()
        ops = list(_op_names(b.kernel.body))
        assert "tile.sync" not in ops
        assert ops.count("tile.s_barrier_bare") == 3  # release, acquire, pre_poll
        ll = lower_kernel_to_llvm(b.kernel, arch="gfx950", llvm_flavor=FLAVOR)
        if fence_impl == "asm":
            assert "buffer_wbl2 sc1" in ll and "buffer_inv sc1" in ll
        else:
            assert 'fence syncscope("agent") release' in ll
            assert 'fence syncscope("agent") acquire' in ll


# ---------------------------------------------------------- ISA (COMGR)


def _objdump():
    for c in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if c and os.path.isfile(c):
            return c
    return None


def _isa(kernel, arch="gfx950") -> str:
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import comgr

    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")
    hsaco = compile_kernel(kernel, arch=arch, backend="python").hsaco
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "k.hsaco"
        p.write_bytes(hsaco)
        return subprocess.run(
            [objdump, "-d", f"--mcpu={arch}", str(p)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout


def _insns(isa: str) -> list[str]:
    out = []
    for ln in isa.splitlines():
        t = ln.strip()
        if not t or t.endswith(">:") or t.startswith(("Disassembly", ";")):
            continue
        t = t.split("//")[0].strip()
        if t and not t.endswith(":"):
            out.append(t)
    return out


def _drained_barriers(insns: list[str]) -> tuple[int, int]:
    """(barriers, barriers whose previous instruction waits vmcnt(0))."""
    n = drained = 0
    for i, t in enumerate(insns):
        if t.split()[0] == "s_barrier":
            n += 1
            prev = insns[i - 1] if i else ""
            if prev.startswith("s_waitcnt") and "vmcnt(0)" in prev:
                drained += 1
    return n, drained


def _prefetch_kernel(body, name):
    """Weight loads issued first, ``body`` (the sync path under test), then a
    use of the weights: a vmcnt drain ahead of a barrier would stall them."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 256
    w = b.param("w", PtrType(I32, "global"))
    box = b.param("box", PtrType(I32, "global"))
    out = b.param("out", PtrType(I32, "global"))
    tag = b.readfirstlane(b.param("tag", I32))
    tid = b.thread_id_x()
    ws = [
        b.global_load_i32(w, b.add(b.mul(tid, b.const_i32(8)), b.const_i32(k)))
        for k in range(8)
    ]
    body(b, M.mailbox_from_ptr(b, box, b.const_i32(1 << 16)), tag)
    s = ws[0]
    for v in ws[1:]:
        s = b.add(s, v)
    b.global_store(out, tid, s)
    b.ret()
    return b.kernel


def test_isa_no_vmcnt_drain_before_mailbox_barriers():
    """ISA gate (T3): pre_poll, acquire_bulk and payload_wait (both forms)
    put no ``s_waitcnt vmcnt(0)`` right before their ``s_barrier``; the only
    drained barrier is release_bulk's intended one. Control: ``b.sync()``
    in the same kernel is seen as drained (the check is not blind)."""

    def body(b, mb, tag):
        M.pre_poll(b, lambda m: (mb, m), 100, tag)
        M.acquire_bulk(b, mb, 200, tag)
        M.payload_wait(b, lambda m: (mb, m), 100, tag, rows=1)
        M.payload_wait(b, lambda m: (mb, m), 100, tag, rows=8)

    n, drained = _drained_barriers(_insns(_isa(_prefetch_kernel(body, "isa_sync"))))
    assert n >= 4 and drained == 0, (n, drained)

    def rel(b, mb, tag):
        M.release_bulk(b, mb, 300, tag)

    insns = _insns(_isa(_prefetch_kernel(rel, "isa_rel")))
    n, drained = _drained_barriers(insns)
    assert (n, drained) == (1, 1), (n, drained)
    assert sum(t.startswith("buffer_wbl2") for t in insns) == 1
    assert sum(t.startswith("buffer_inv") for t in insns) == 0

    def ctl(b, mb, tag):
        b.sync()

    n, drained = _drained_barriers(_insns(_isa(_prefetch_kernel(ctl, "isa_ctl"))))
    assert n == 1 and drained == 1, (n, drained)


# -------------------------------------------------------- day-1 IR half


def test_day1_uid_multiline_asm_round_trips_ir_serialize():
    """Gate (IR half): the asm poll's multi-line ``${:uid}`` template, the
    multi-output struct and the ``ptr``-typed asm result survive
    serialize -> parse byte-exactly, and lower identically."""
    for form in ("asm_s", "asm_v", "rsrc_vec"):
        k = K.build_gate_kernel(form)
        t = serialize(k)
        assert serialize(parse(t)) == t
        assert _lower(parse(t)) == _lower(k)
    k = _pp(M.MailboxKnobs(), 2)
    t = serialize(k)
    k2 = parse(t)
    assert serialize(k2) == t
    assert "${:uid}" in t
    assert _lower(k2) == _lower(k)


_CPP_CASES = {
    "gate_asm_s": lambda: K.build_gate_kernel("asm_s"),
    "gate_asm_v": lambda: K.build_gate_kernel("asm_v"),
    "gate_rsrc_vec": lambda: K.build_gate_kernel("rsrc_vec"),
    "pingpong": lambda: _pp(M.MailboxKnobs(), 2),
    "pingpong_numa": lambda: _pp(M.MailboxKnobs(numa_copies=2), 2),
    "a2a": lambda: K.build_all_to_all(K.AllToAllSpec(check=True)),
    "a2a_sync": lambda: K.build_all_to_all(K.AllToAllSpec(barrier="sync")),
    "tagwrap": lambda: K.build_tag_wrap_kernel(M.MailboxKnobs()),
    "bulk_r1": lambda: K.build_bulk_kernel(K.BulkSpec()),
    "bulk_r8_fir": lambda: K.build_bulk_kernel(
        K.BulkSpec(knobs=M.MailboxKnobs(fence_impl="ir"), rows=8)
    ),
    "device_ops": K.build_device_ops_kernel,
}


@pytest.mark.parametrize("which", sorted(_CPP_CASES))
def test_day1_cpp_engine_byte_identical_for_asm_impl(which):
    """Gate (engine half): every helper-built kernel lowers to the same bytes
    in both engines (``backend="both"`` raises on any difference). This is
    also the guard for risk 5 (the C++ lowering reads op attributes with
    defaults, ``ll_attr_int(op, key, dflt)`` in ``cpp/core/lower_llvm/mem.cpp``,
    so an attribute only the Python engine reads is dropped silently): the
    helpers carry their cache bits in asm text, never in op attributes."""
    if importlib.util.find_spec("rocke_engine") is None:
        pytest.skip("cpp engine needs the rocke_engine binding")
    assert _lower(_CPP_CASES[which](), backend="both")


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

    # A hand-off bug hangs the kernel: fail the test after a watchdog instead
    # of blocking the run (the hung queue is released when the process exits).
    d = H.Dev(watchdog_s=120)
    yield d
    d.close()


def _backends(form):
    out = ["python"]
    if form != "ir" and importlib.util.find_spec("rocke_engine") is not None:
        out.append("cpp")
    return out


@pytest.mark.parametrize("form", K.GATE_FORMS)
def test_gpu_day1_gate(arch, dev, form):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    for be in _backends(form):
        r = H.run_gate(dev, form, arch, be)
        assert r["ok"], r


def test_gpu_uid_labels_unique_in_isa(arch):
    """Two asm polls in one kernel assemble (distinct ``${:uid}`` labels)."""
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    art, _ = H.compile_for(_pp(M.MailboxKnobs()), arch, "python")
    assert art.hsaco


@pytest.mark.parametrize("kn", ALL_KNOBS, ids=lambda k: k.suffix() or "default")
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
            assert r.ok, r
            assert r.monotone, r


@pytest.mark.parametrize("fence_impl", ["asm", "ir"])
@pytest.mark.parametrize("rows", [1, 8])
def test_gpu_bulk_handoff(arch, dev, fence_impl, rows):
    """release_bulk / acquire_bulk rows across XCDs over many steps (stale
    rows are mismatches), put_bf + bf2_f32 bit-exact vs numpy RNE,
    payload_wait (barrier and pre-poll forms), atomic_add_scoped counters and
    returned tickets."""
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    r = H.run_bulk(
        dev,
        K.BulkSpec(knobs=M.MailboxKnobs(fence_impl=fence_impl), rows=rows),
        arch,
        steps=12,
    )
    assert r["ok"], r


def test_gpu_bulk_without_release_reads_stale_rows(arch, dev):
    """Mutation check: without the producer's L2 write-back before its ready
    pair, consumers on other XCDs read stale rows, so the bulk test above
    discriminates. Shift 1 puts producer and consumer on different XCDs
    under round-robin dispatch; shift 128 (same XCD) is the control."""
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    spec = K.BulkSpec(fence="norelease")
    cross = H.run_bulk(dev, spec, arch, shifts=(1,), steps=4)
    assert cross["row_mismatch"] > 0, cross
    same = H.run_bulk(dev, spec, arch, shifts=(128,), steps=4)
    assert same["row_mismatch"] == 0, same


def test_gpu_device_ops(arch, dev):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    for seed in range(2):
        r = H.run_device_ops(dev, arch, seed=seed)
        assert r["ok"], r


def test_gpu_tag_wrap(arch, dev):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    wrap = 1 << 26
    steps = [wrap - 2, wrap - 1, wrap, wrap + 1]
    recs = H.run_tag_wrap(dev, arch, steps, [0, 47, 48, 62])
    for r in recs:
        assert r["ok"] and r["tag"] != 0, r


def test_gpu_tag_wrap_hazard_is_real(arch, dev):
    """Mutation check of the wrap guard: layer 63 at step 2**26 - 1 has tag 0,
    which a never-written (zeroed) mailbox matches at once, so the consumer
    returns 0 instead of the producer's payload."""
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    recs = H.run_tag_wrap(dev, arch, [(1 << 26) - 1], [63], delay=8, check_layers=False)
    assert recs[0]["tag"] == 0
    assert not recs[0]["ok"]


_A2A = """
import sys
from rocke.examples.gfx950.persistent_mailbox import harness as H, kernels as K
from rocke.helpers import mailbox as M
if sys.argv[1] != "asm":
    assert K.load_ir_impl(), "ir impl not importable"
kn = M.MailboxKnobs(impl=sys.argv[1], pause=sys.argv[3], numa_copies=int(sys.argv[4]))
dev = H.Dev()
spec = K.AllToAllSpec(knobs=kn, consume=sys.argv[2], check=True,
                      barrier=sys.argv[5] if len(sys.argv) > 5 else "bare")
rs = H.run_all_to_all(dev, spec, "gfx950", phases=20, reps=2)
dev.close()
print("A2A_OK" if all(r.ok for r in rs) else "A2A_BAD", rs[0].backend)
"""


def run_a2a_subprocess(impl, consume, pause, numa, barrier="bare", timeout=240):
    """Shared with ``library/tests/test_mailbox_ir.py``."""
    try:
        p = subprocess.run(
            [sys.executable, "-c", _A2A, impl, consume, pause, str(numa), barrier],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"all-to-all {impl}/{consume}/{pause}/numa{numa} hung")
    assert "A2A_OK" in p.stdout, (p.stdout[-2000:], p.stderr[-2000:])


@pytest.mark.parametrize("pairs", [1, 2])
def test_gpu_pingpong_numa_copies(arch, dev, pairs):
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    kn = M.MailboxKnobs(numa_copies=2)
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


@pytest.mark.parametrize("consume", K.A2A_CONSUME)
def test_gpu_all_to_all_numa_copies(arch, consume):
    """Dual-copy layout, calibrated flip table: every round's payloads exact."""
    run_a2a_subprocess("asm", consume, "s_nop", 2)


@pytest.mark.parametrize("consume", K.A2A_CONSUME)
@pytest.mark.parametrize("pause", ["s_nop", "s_sleep1"])
def test_gpu_all_to_all_full_grid_never_hangs(arch, consume, pause):
    run_a2a_subprocess("asm", consume, pause, 1, timeout=180)


def test_gpu_all_to_all_sync_barrier(arch):
    run_a2a_subprocess("asm", "wave0", "s_nop", 1, barrier="sync", timeout=180)
