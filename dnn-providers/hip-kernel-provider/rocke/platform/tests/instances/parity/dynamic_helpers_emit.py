#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/dynamic_helpers_emit.py -- Python reference emitter for the
# runtime-shape (AOT) helpers: SoftwarePipeline.run_ping_pong_dynamic and the
# dynamic coordinate transforms (EmbedDynamic, UnmergeMagicDynamic, PadDynamic,
# DynamicTensorDescriptor).
#
# Those helpers are otherwise only byte-compared through the conv families,
# which drive a narrow slice of their argument space. Each config here builds
# the smallest kernel that reaches one branch of a helper, so a divergence
# between the two engines shows up against a one-screen kernel instead of a
# full convolution. Byte-compared against dynamic_helpers_emit.c.
#
# Every builder call is its own statement in the same order the C emitter
# issues it: each call consumes an SSA id, so a reordering emits the same IR
# under different names.
#
# arch = gfx950, llvm_flavor = AUTO, matching the C side.
from rocke.core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.pipeline import SoftwarePipeline
from rocke.helpers.schedule import SchedulePolicy
from rocke.helpers.transforms import (
    DynamicTensorDescriptor,
    embed_dynamic,
    pad_dynamic,
    unmerge_magic_dynamic,
)

from _emit_common import run_emit

BLOCK_K = 32
LDS_ELEMS = 256
BUFFER_BYTES = 1 << 20
# Byte offset the buffer resource drops (reads zero) for an invalid coord.
OOB_OFFSET = 0x7FFF0000


def _ptr(b, name, elem, *, readonly=False):
    if readonly:
        return b.param(
            name, PtrType(elem, "global"), noalias=True, readonly=True, align=16
        )
    return b.param(name, PtrType(elem, "global"), noalias=True, align=16)


# ---------------------------------------------------------------------------
# run_ping_pong_dynamic
# ---------------------------------------------------------------------------


def _pingpong(
    *, with_lo, with_zero_fill, wait_vmcnt, overlap_vmcnt, schedule, mask=False
):
    """Ping-pong over a runtime K extent.

    issue_load stages one f16 per thread from X[k + tid] into both halves of
    the buffer pair; compute reads the ``a`` half back and folds it into an
    f32 accumulator, and folds the tile offset into an i32 counter so the
    offset each phase sees is observable in the output.
    """

    def build(b: IRBuilder) -> None:
        x = _ptr(b, "X", F16, readonly=True)
        y = _ptr(b, "Y", F32)
        c_out = _ptr(b, "C", I32)
        k_extent = b.param("K", I32)
        k_lo = b.param("k_lo", I32) if with_lo else None
        k_zero = b.param("k_zero", I32) if with_zero_fill else None
        tid = b.thread_id_x()
        nbytes = b.const_i32(BUFFER_BYTES)
        rsrc = b.buffer_rsrc(x, nbytes)
        soff = b.const_i32(0)
        two = b.const_i32(2)
        a0 = b.smem_alloc(F16, [LDS_ELEMS], name_hint="a0")
        b0 = b.smem_alloc(F16, [LDS_ELEMS], name_hint="b0")
        a1 = b.smem_alloc(F16, [LDS_ELEMS], name_hint="a1")
        b1 = b.smem_alloc(F16, [LDS_ELEMS], name_hint="b1")

        def issue(k, buf):
            elem = b.add(k, tid)
            voff = b.mul(elem, two)
            v = b.buffer_load_f16(rsrc, voff, soff)
            b.smem_store_f16(buf[0], [tid], v)
            b.smem_store_f16(buf[1], [tid], v)

        def compute(k, buf, state):
            acc, cnt = state
            h = b.smem_load_vN_f16(buf[0], tid, n=1)
            e = b.vec_extract(h, 0)
            f = b.cast_to_f32(e)
            acc2 = b.fadd(acc, f)
            cnt2 = b.add(cnt, k)
            return [acc2, cnt2]

        acc0 = b.const_f32(0.0)
        cnt0 = b.const_i32(0)
        pipe = SoftwarePipeline(
            num_iters=0,
            wait_vmcnt=wait_vmcnt,
            overlap_vmcnt=overlap_vmcnt,
        )
        results = pipe.run_ping_pong_dynamic(
            b,
            k_extent=k_extent,
            block_k=BLOCK_K,
            k_lo=k_lo,
            k_zero_fill=k_zero,
            mask_tail_state=mask,
            buffers=[(a0, b0), (a1, b1)],
            iter_args=[("acc", acc0), ("cnt", cnt0)],
            issue_load_fn=issue,
            compute_fn=compute,
            schedule=schedule,
        )
        b.global_store(y, tid, results[0])
        b.global_store(c_out, tid, results[1])
        b.ret()

    return build


# ---------------------------------------------------------------------------
# Dynamic descriptors
# ---------------------------------------------------------------------------


def _store_through_valid(b, x, y, tid, off, valid):
    """Load X[off] (zero when !valid) and store it to Y[tid]."""
    nbytes = b.const_i32(BUFFER_BYTES)
    rsrc_x = b.buffer_rsrc(x, nbytes)
    two = b.const_i32(2)
    off_b = b.mul(off, two)
    oob = b.const_i32(OOB_OFFSET)
    voff = b.select(valid, off_b, oob)
    soff = b.const_i32(0)
    v = b.buffer_load_f16(rsrc_x, voff, soff)
    rsrc_y = b.buffer_rsrc(y, nbytes)
    tid_b = b.mul(tid, two)
    b.buffer_store_f16(rsrc_y, tid_b, soff, v)


def build_unmerge_embed(b: IRBuilder) -> None:
    """unmerge_magic_dynamic + embed_dynamic over a 2-D dynamic descriptor.

    m -> (ho, wo) by a runtime magic triple; h = ho*conv_stride + dy - 1 with
    a runtime stride (Value) next to an int stride of 1 (multiply skipped),
    bounded by [0, p_H); w = wo*2 with an int offset of 0 (constant built, no
    add), bounded by [0, p_W). The base strides are kernel args too.
    """
    x = _ptr(b, "X", F16, readonly=True)
    y = _ptr(b, "Y", F16)
    o = _ptr(b, "O", I32)
    p_h = b.param("p_H", I32)
    p_w = b.param("p_W", I32)
    p_sh = b.param("p_sH", I32)
    p_sw = b.param("p_sW", I32)
    p_cs = b.param("p_conv_stride", I32)
    p_mult = b.param("p_wo_mult", I32)
    p_shift = b.param("p_wo_shift", I32)
    p_wo = b.param("p_Wo", I32)
    tid = b.thread_id_x()
    dy = b.block_id_x()
    desc = DynamicTensorDescriptor.create(
        "x_hw", coord_names=("h", "w"), strides=(p_sh, p_sw)
    )
    desc = desc.transform(
        unmerge_magic_dynamic("m", ("ho", "wo"), [(p_mult, p_shift, p_wo)]),
        embed_dynamic(("ho", "dy"), "h", strides=[p_cs, 1], offset=-1, lo=0, hi=p_h),
        embed_dynamic(("wo",), "w", strides=[2], offset=0, lo=0, hi=p_w),
    )
    off, valid = desc.offset(b, m=tid, dy=dy)
    b.global_store(o, tid, off)
    _store_through_valid(b, x, y, tid, off, valid)
    b.ret()


def build_unmerge_mixed(b: IRBuilder) -> None:
    """unmerge_magic_dynamic with int and Value triple members mixed.

    m -> (n, h, w): the h triple is all ints with dim 1 (no division: the
    remainder is a const 0 and the quotient passes through, though its int
    members are still materialised); the w triple mixes a Value multiplier and
    dim with an int shift. pad_dynamic on w supplies the validity.
    """
    x = _ptr(b, "X", F16, readonly=True)
    y = _ptr(b, "Y", F16)
    o = _ptr(b, "O", I32)
    p_sn = b.param("p_sN", I32)
    p_sh = b.param("p_sH", I32)
    p_sw = b.param("p_sW", I32)
    p_mult = b.param("p_w_mult", I32)
    p_w = b.param("p_W", I32)
    tid = b.thread_id_x()
    desc = DynamicTensorDescriptor.create(
        "x_nhw", coord_names=("n", "h", "w"), strides=(p_sn, p_sh, p_sw)
    )
    desc = desc.transform(
        unmerge_magic_dynamic("m", ("n", "h", "w"), [(7, 2, 1), (p_mult, 5, p_w)]),
        pad_dynamic("w", hi=p_w),
    )
    off, valid = desc.offset(b, m=tid)
    b.global_store(o, tid, off)
    _store_through_valid(b, x, y, tid, off, valid)
    b.ret()


def build_pad_dynamic(b: IRBuilder) -> None:
    """embed_dynamic with every argument a Value, plus pad_dynamic variants.

    r = i*p_s + p_off bounded by [p_lo, p_M) (all runtime); c padded by
    [2, p_N) (int lo, Value hi); e padded by [-, p_E) (no lo side). The last
    base stride is a materialized constant 1 -- still a multiply, because the
    dynamic descriptor never special-cases its strides.
    """
    x = _ptr(b, "X", F16, readonly=True)
    y = _ptr(b, "Y", F16)
    o = _ptr(b, "O", I32)
    p_m = b.param("p_M", I32)
    p_n = b.param("p_N", I32)
    p_e = b.param("p_E", I32)
    p_ld = b.param("p_ld", I32)
    p_le = b.param("p_le", I32)
    p_s = b.param("p_s", I32)
    p_off = b.param("p_off", I32)
    p_lo = b.param("p_lo", I32)
    tid = b.thread_id_x()
    bx = b.block_id_x()
    by = b.block_id_y()
    one = b.const_i32(1)
    desc = DynamicTensorDescriptor.create(
        "x_rce", coord_names=("r", "c", "e"), strides=(p_ld, p_le, one)
    )
    desc = desc.transform(
        embed_dynamic(("i",), "r", strides=[p_s], offset=p_off, lo=p_lo, hi=p_m),
        pad_dynamic("c", lo=2, hi=p_n),
        pad_dynamic("e", hi=p_e),
    )
    off, valid = desc.offset(b, i=tid, c=bx, e=by)
    b.global_store(o, tid, off)
    _store_through_valid(b, x, y, tid, off, valid)
    b.ret()


CONFIGS = [
    # 0: k_lo=None (const 0 lower bound), no zero-fill select, default flags.
    _pingpong(
        with_lo=False,
        with_zero_fill=False,
        wait_vmcnt=False,
        overlap_vmcnt=False,
        schedule=None,
    ),
    # 1: split-K shape: runtime k_lo and the k_zero_fill select, partial vmcnt
    #    drain with LDS-only barriers.
    _pingpong(
        with_lo=True,
        with_zero_fill=True,
        wait_vmcnt=True,
        overlap_vmcnt=True,
        schedule=None,
    ),
    # 2: interwave SchedulePolicy: s_setprio bookends around each compute.
    _pingpong(
        with_lo=True,
        with_zero_fill=False,
        wait_vmcnt=True,
        overlap_vmcnt=False,
        schedule=SchedulePolicy.for_pipeline("interwave"),
    ),
    # 3: DynamicTensorDescriptor + unmerge_magic_dynamic + mixed embed_dynamic.
    build_unmerge_embed,
    # 4: DynamicTensorDescriptor + all-Value embed_dynamic + pad_dynamic.
    build_pad_dynamic,
    # 5: mask_tail_state without zero-fill: the in-range compare is emitted for
    #    the Phase B select alone (the counter must not see the stray tile).
    _pingpong(
        with_lo=True,
        with_zero_fill=False,
        wait_vmcnt=True,
        overlap_vmcnt=False,
        schedule=None,
        mask=True,
    ),
    # 6: mask_tail_state with zero-fill: one compare feeds both selects.
    _pingpong(
        with_lo=True,
        with_zero_fill=True,
        wait_vmcnt=True,
        overlap_vmcnt=True,
        schedule=None,
        mask=True,
    ),
    # 7: unmerge_magic_dynamic with mixed int/Value triples and a dim-1 triple.
    build_unmerge_mixed,
]


def _spec(idx: int):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    return CONFIGS[idx]


def _build(build_fn, *, arch: str = "gfx950") -> KernelDef:
    b = IRBuilder("dynamic_helpers")
    b.kernel.attrs["max_workgroup_size"] = LDS_ELEMS
    build_fn(b)
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: dynamic_helpers_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
