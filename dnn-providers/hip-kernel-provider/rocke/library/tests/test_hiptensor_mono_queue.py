# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the hipTensor task-queue monokernel (IR inliner, plans, knobs).

These pin the host-side invariants and that every knob builds; the GPU
numerics are checked against an f64 reference outside the unit suite.
"""

import pytest
from kernels.gfx950.hiptensor_mono.inline import (
    InlineError,
    body_op_census,
    inline_kernel_body,
)
from kernels.gfx950.hiptensor_mono.queue_contraction import (
    Contraction,
    coalesce_modes,
    contraction_ops,
)
from kernels.gfx950.hiptensor_mono.queue_plan import contractions_plan
from kernels.gfx950.hiptensor_mono.task_queue import (
    QueueKnobs,
    QueuePlan,
    TaskQueueSpec,
    build_task_queue,
    queue_counter_words,
)
from rocke.core.ir import F16, IRBuilder, PtrType


def _toy(name="toy", early_return=False):
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    X = b.param("X", PtrType(F16, "global"))
    Y = b.param("Y", PtrType(F16, "global"))
    i = b.add(b.mul(b.block_id_x(), b.const_i32(64)), b.thread_id_x())
    with b.scf_for(b.const_i32(0), b.const_i32(2), b.const_i32(1), iv_name="j") as j:
        k = b.add(i, b.mul(j, b.const_i32(4096)))
        b.global_store(Y, k, b.global_load_f16(X, k), align=2)
        if early_return:
            b.ret()
    b.ret()
    return b.kernel


def P1():
    return Contraction(
        "p1",
        ("M0", "K0", "M1", "K1"),
        ("N0", "K0", "N1", "K1"),
        ("M0", "N0", "M1", "N1"),
        {"M0": 48, "K0": 24, "M1": 32, "K1": 24, "N0": 40, "N1": 36},
    )


def CANON():
    return Contraction(
        "cn",
        ("M0", "M1", "K0", "K1"),
        ("N0", "N1", "K0", "K1"),
        ("M0", "M1", "N0", "N1"),
        {"M0": 48, "K0": 24, "M1": 32, "K1": 24, "N0": 40, "N1": 36},
    )


def test_inline_renames_and_maps_block_ids():
    toy = _toy()
    b = IRBuilder("host")
    T0 = b.param("T0", PtrType(F16, "global"))
    T1 = b.param("T1", PtrType(F16, "global"))
    bx = b.const_i32(5)
    inline_kernel_body(
        b, toy, params={"X": T0, "Y": T1}, block_ids={"x": bx}, prefix="i0"
    )
    inline_kernel_body(
        b, toy, params={"X": T1, "Y": T0}, block_ids={"x": bx}, prefix="i1"
    )
    c = body_op_census(b.kernel)
    assert "gpu.block_id" not in c and "cf.return" not in c
    assert c["scf.for"] == 2
    names = []

    def walk(reg):
        for op in reg.ops:
            names.extend(r.name for r in op.results)
            if op.name == "scf.for":
                names.append(op.attrs["iv"])
            for r in op.regions:
                walk(r)

    walk(b.kernel.body)
    assert len(names) == len(set(names)), "inlined instances must not share value names"


def _value_names(kdef):
    names = []

    def walk(reg):
        for op in reg.ops:
            names.extend(r.name for r in op.results)
            if op.name == "scf.for":
                names.append(op.attrs["iv"])
                names.extend(m["name"] for m in op.attrs.get("iter_args", []))
            for r in op.regions:
                walk(r)

    walk(kdef.body)
    return names


def test_inline_names_come_from_the_host_builder():
    """Inlined names are allocated by the host IRBuilder's counter: unique
    against host values created before AND after the inline, and the same
    on every build (no id()-derived or caller-composed names)."""

    def build():
        b = IRBuilder("host")
        T0 = b.param("T0", PtrType(F16, "global"))
        T1 = b.param("T1", PtrType(F16, "global"))
        bx = b.add(b.const_i32(5), b.const_i32(1))
        before = b._counter
        inline_kernel_body(
            b, _toy(), params={"X": T0, "Y": T1}, block_ids={"x": bx}, prefix="i"
        )
        inline_kernel_body(
            b, _toy(), params={"X": T1, "Y": T0}, block_ids={"x": bx}, prefix="i"
        )
        after = b._counter
        b.add(bx, b.const_i32(2))
        return b.kernel, before, after

    k, before, after = build()
    names = _value_names(k)
    assert len(names) == len(set(names))
    inlined = [n for n in names if n.startswith("%i_")]
    assert inlined and after - before == len(inlined)
    k2, _, _ = build()
    assert _value_names(k2) == names


def test_permute_rank_cap_is_never_rebound(monkeypatch):
    """Bodies above the C++-mirrored permute rank cap get an explicit per-call
    cap; the module-level cap every other build reads is never rebound while
    the body is built (it was, in a try/finally, which a concurrent build in
    another thread could observe)."""
    from kernels.gfx950.hiptensor_mono.queue_contraction import (
        _build_permute_any_rank,
    )
    from rocke.instances.common import permute_nd as pn

    cap = pn.MAX_RANK
    seen = []
    real = pn._build_offset_descriptor

    def spy(spec):
        seen.append(pn.MAX_RANK)  # the cap a concurrent build would read now
        return real(spec)

    monkeypatch.setattr(pn, "_build_offset_descriptor", spy)
    big = pn.PermuteSpec(x_shape=(2,) * (cap + 2), perm=tuple(range(cap + 2))[::-1])
    k = _build_permute_any_rank(big, "gfx950")
    assert [p.name for p in k.params] == ["X", "Y"]
    assert seen == [cap] and pn.MAX_RANK == cap
    over = pn.PermuteSpec(x_shape=(2,) * (cap + 1), perm=tuple(range(cap + 1)))
    assert not pn.is_valid_spec(over, "gfx950")[0]
    with pytest.raises(ValueError, match="rank"):
        pn.build_permute(over, "gfx950")


def test_inline_refuses_early_return_and_unbound_param():
    b = IRBuilder("host")
    T0 = b.param("T0", PtrType(F16, "global"))
    with pytest.raises(InlineError):
        inline_kernel_body(
            b,
            _toy(early_return=True),
            params={"X": T0, "Y": T0},
            block_ids={"x": b.const_i32(0)},
            prefix="e",
        )
    with pytest.raises(InlineError):
        inline_kernel_body(
            IRBuilder("h2"), _toy(), params={}, block_ids={"x": T0}, prefix="u"
        )


def test_coalesce_modes():
    src, dst, ext = coalesce_modes(
        ("a", "b", "c", "d"), ("c", "d", "a", "b"), {"a": 2, "b": 3, "c": 4, "d": 5}
    )
    assert (
        len(src) == 2
        and ext[src[0]] == 6
        and ext[src[1]] == 20
        and dst == (src[1], src[0])
    )


def test_contraction_ops_shapes():
    pre, g, post, scr = contraction_ops(P1(), a="A", b="B", d="D")
    assert (
        [o.name for o in pre] == ["p1.pA", "p1.pB"]
        and len(post) == 1
        and set(scr) == {"p1_Ap", "p1_Bp", "p1_E"}
    )
    assert (
        g[0].bind["M"] == ("i32", 1440)
        and g[0].bind["N"] == ("i32", 1536)
        and g[0].bind["K"] == ("i32", 576)
    )
    pre, g, post, scr = contraction_ops(CANON(), a="A", b="B", d="D")
    assert post == [] and g[0].bind["C"] == ("buf", "D")
    pre, g, post, scr = contraction_ops(P1(), a="A", b="B", d="D", fuse_out="on")
    assert (
        post == []
        and g[0].out_map is not None
        and g[0].bind["C"] == ("buf", "D")
        and "p1_E" not in scr
    )
    # K-fastest operands are read in place (no permute op, no scratch)
    tn = Contraction(
        "tn", ("k", "m"), ("k", "n"), ("m", "n"), {"m": 256, "n": 384, "k": 512}
    )
    pre, g, post, scr = contraction_ops(tn, a="A", b="B", d="D")
    assert pre == [] and post == [] and scr == {} and g[0].deps == ()


def test_plan_topological_and_shared_bodies():
    plan, _ = contractions_plan("t", [P1(), CANON()])
    plan.validate()
    gem = [o for o in plan.ops if o.kind == "gemm"]
    assert len({o.body_key for o in gem}) == 1  # unify: one inlined GEMM body
    bad = QueuePlan(
        "bad", [gem[0], plan.ops[0], plan.ops[1]], plan.tensors, 512
    )  # consumer before producers
    with pytest.raises(ValueError):
        bad.validate()


@pytest.mark.parametrize(
    "knobs",
    [
        QueueKnobs(),
        QueueKnobs(sched="static"),
        QueueKnobs(first_static=True, slot_parity=True),
        QueueKnobs(acquire_once=True),
        QueueKnobs(stamps=True),
        QueueKnobs(publish_terminal=True),
        QueueKnobs(sched="phase"),
    ],
)
def test_build_mono_knobs(knobs):
    plan, _ = contractions_plan("t", [P1(), CANON()], fuse_out="all")
    k = build_task_queue(TaskQueueSpec(plan, knobs), arch="gfx950")
    assert k.name.endswith(knobs.suffix())
    assert [p.name for p in k.params][-2 if knobs.stamps else -1] == "ctr"
    assert (
        queue_counter_words(plan) == 3 + len(plan.ops) + 2 * 2
    )  # (ticket, arrive) per dependency level
    c = body_op_census(k)
    assert c.get("gpu.block_id", 0) == 1  # only the monokernel's own workgroup id


def test_static_refuses_dynamic_levers():
    plan, _ = contractions_plan("t", [CANON()])
    with pytest.raises(ValueError):
        build_task_queue(
            TaskQueueSpec(plan, QueueKnobs(sched="static", first_static=True)),
            arch="gfx950",
        )


def test_permute_bodies_build_and_refuse():
    from kernels.gfx950.hiptensor_mono.permute_gather import (
        PermuteGatherSpec,
        build_permute_gather,
    )
    from kernels.gfx950.hiptensor_mono.transpose_lds import (
        TransposeLdsSpec,
        build_transpose_lds,
    )
    from rocke.instances.common.permute_nd import PermuteSpec

    k = build_transpose_lds(TransposeLdsSpec(rows=128, cols=64), arch="gfx950")
    assert [p.name for p in k.params] == ["X", "Y"] and k.attrs[
        "max_workgroup_size"
    ] == 512
    with pytest.raises(ValueError):
        build_transpose_lds(TransposeLdsSpec(rows=96, cols=64), arch="gfx950")
    ps = PermuteSpec(x_shape=(8, 16, 32), perm=(2, 0, 1), block_size=256)
    k = build_permute_gather(PermuteGatherSpec(permute=ps, unroll=4), arch="gfx950")
    assert [p.name for p in k.params] == ["X", "Y"] and k.name.endswith("_gu4")
    with pytest.raises(ValueError):
        build_permute_gather(PermuteGatherSpec(permute=ps, unroll=0), arch="gfx950")
    # the gather body is validated like permute_nd (minus the rank cap)
    for bad, kw in (
        (PermuteSpec(x_shape=(8, 16, 32), perm=(2, 0, 1), dtype="f32"), {}),
        (PermuteSpec(x_shape=(8, 16, 32), perm=(2, 0, 1), block_size=96), {}),
        (ps, {"arch": "gfx000"}),
    ):
        with pytest.raises(ValueError, match="invalid permute spec"):
            build_permute_gather(
                PermuteGatherSpec(permute=bad, unroll=4), **{"arch": "gfx950", **kw}
            )


@pytest.mark.parametrize("body", ["auto", "nd", "tlds"])
def test_permute_body_choices(body):
    cn = Contraction(
        "c64",
        ("M0", "M1", "K0", "K1"),
        ("N0", "N1", "K0", "K1"),
        ("M0", "M1", "N0", "N1"),
        {"M0": 16, "K0": 16, "M1": 32, "K1": 32, "N0": 16, "N1": 16},
    )
    plan, _ = contractions_plan("pb", [cn], permute_body=body)
    names = [o.kdef.name for o in plan.ops if o.kind == "permute"]
    assert len(names) == 2
    if body in ("auto", "tlds"):
        assert all("tlds" in n for n in names)
    k = build_task_queue(TaskQueueSpec(plan, QueueKnobs()), arch="gfx950")
    assert body_op_census(k).get("gpu.block_id", 0) == 1


def test_lds_budget_refuses_two_gemm_bodies():
    from kernels.gfx950.hiptensor_mono.task_queue import kernel_lds_bytes

    cn = Contraction(
        "c64",
        ("M0", "M1", "K0", "K1"),
        ("N0", "N1", "K0", "K1"),
        ("M0", "M1", "N0", "N1"),
        {"M0": 16, "K0": 16, "M1": 32, "K1": 32, "N0": 16, "N1": 16},
    )
    # fuse_out="on" with a canonical and a permuted output: two inlined GEMM bodies, LDS adds up
    plan, _ = contractions_plan("two", [P1(), cn], fuse_out="on")
    with pytest.raises(ValueError, match="LDS"):
        build_task_queue(TaskQueueSpec(plan, QueueKnobs()), arch="gfx950")
    plan, _ = contractions_plan("one", [P1(), cn], fuse_out="all")
    k = build_task_queue(TaskQueueSpec(plan, QueueKnobs()), arch="gfx950")
    assert 0 < kernel_lds_bytes(k) <= 160 * 1024
