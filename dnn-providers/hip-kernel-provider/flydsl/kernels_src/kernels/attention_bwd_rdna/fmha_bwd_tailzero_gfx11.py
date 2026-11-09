# Not vendored: written for this provider; it has no upstream.
"""FlyDSL padded-row zeroing for the gfx11-generic (RDNA) SDPA backward: zero the logical rows `seq_len <= row < S` of the dense gradient outputs.

A graph with `padding_mask` has per-batch lengths `seq_len_q[b]` and `seq_len_kv[b]`. The backward kernels write only the rows below
them, so the rows past them of a dense dQ, dK or dV keep whatever the allocation held; the reference and cuDNN define them as zero.
This kernel runs after the last gradient (and dBias reduce) launch of the graph and writes zero to exactly those rows, over each output's
own `(batch, head, seq)` element strides (a shared `[B, S, 3, H, D]` gradient view included) and a contiguous last dimension. A ragged
output is not written: its storage past a length belongs to a gap or to the next sequence, and a packed extent has no padded rows.

    dQ[b, h, r, :] = 0     for seq_len_q[b]  <= r < Sq     (bit 0 of `mask`)
    dK[b, h, r, :] = 0     for seq_len_kv[b] <= r < Skv    (bit 1)
    dV[b, h, r, :] = 0     for seq_len_kv[b] <= r < Skv    (bit 2)

A bit of `mask` is set only for a dense output. dQ and dK have `Dqk` columns, dV has `Dv`. No output is read and no workspace is used.

**Layout.** Block 256 threads; grid `(blocks, B * (Hq + 2 * Hkv), 1)` (`fmha_bwd_tailzero_addressing`: the plane order, the unit
a thread owns and the grid size are defined there, and the host tests run that arithmetic). A thread owns one 16-byte vector of columns
of one tail row. When every output stride is a multiple of the vector width and the vector lies inside the row, the thread takes one
vector store; otherwise it takes element stores masked at the row's column count. Both write the same elements.

**ABI** (one object per dtype; every size and stride is a launch value; strides in elements; positional):
    DQ, DK, DV                    the gradient outputs, 16-byte aligned base pointers
    seq_len_q, seq_len_kv         INT32 `[B]` lengths
    B, Hq, Hkv, Sq, Skv, Dqk, Dv  the problem's extents
    stride_{dq,dk,dv}_{batch,head,seq}
    mask                          bit 0 / 1 / 2: write the dQ / dK / dV tail
"""

from kernels.attention_bwd_cdna import fmha_bwd_tailzero_addressing as addr

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "fmha_bwd_tailzero_gfx11"
BLOCK_THREADS = addr.BLOCK_THREADS

# No arithmetic beyond index math: nothing to hint.
_COMPILE_HINTS = {}

_ELEM = {
    "f16": (fx.Float16, 2),
    "bf16": (fx.BFloat16, 2),
    "fp32": (fx.Float32, 4),
}


def tailzero_arg_values(outputs, seq_lens, dims, strides, mask):
    """Launch values keyed by kernel argument name.

    `outputs` is the device addresses `(dq, dk, dv)`, `seq_lens` the addresses `(seq_len_q, seq_len_kv)`, `dims` is
    `(B, Hq, Hkv, Sq, Skv, Dqk, Dv)` and `strides` is `((b, h, s) of dQ, of dK, of dV)` in elements.
    """
    v = dict(zip(("DQ", "DK", "DV"), outputs))
    v.update(zip(("seq_len_q", "seq_len_kv"), seq_lens))
    v.update(zip(("B", "Hq", "Hkv", "Sq", "Skv", "Dqk", "Dv"), dims))
    for name, triple in zip(("dq", "dk", "dv"), strides):
        for axis, x in zip(("batch", "head", "seq"), triple):
            v[f"stride_{name}_{axis}"] = x
    v["mask"] = mask
    return v


def build_fmha_bwd_tailzero_gfx11_module_primary(dtype_str):
    if dtype_str not in _ELEM:
        raise ValueError(dtype_str)
    elem, elem_bytes = _ELEM[dtype_str]
    vec = addr.vec_of(elem_bytes)
    narrow = elem_bytes == 2
    _cache_tag = (KERNEL_NAME, dtype_str)

    def _ptr_ty(ty, align):
        return fx.PointerType.get(
            elem_ty=ty.ir_type, address_space=fx.AddressSpace.Global, alignment=align
        )

    def _zero_vec():
        zero = Vec.filled(vec, 0.0, fx.Float32)
        return zero.to(elem) if narrow else zero

    def _zero_elem():
        zero = fx.Float32(0.0)
        return zero.to(elem) if narrow else zero

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def fmha_bwd_tailzero_gfx11_kernel(
        DQ: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        seq_len_q: fx.Pointer,
        seq_len_kv: fx.Pointer,
        B: fx.Int32,
        Hq: fx.Int32,
        Hkv: fx.Int32,
        Sq: fx.Int32,
        Skv: fx.Int32,
        Dqk: fx.Int32,
        Dv: fx.Int32,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        mask: fx.Int32,
    ):
        tid = fx.Index(fx.thread_idx.x)
        plane = fx.Index(fx.block_idx.y)
        hq = fx.Index(Hq)
        hkv = fx.Index(Hkv)
        heads = addr.heads_total(hq, hkv)
        b = addr.batch_of(plane, heads)
        r = plane - b * heads

        # the plane's tensor: dQ heads, then dK heads, then dV heads
        in_q = r < hq
        in_v = r >= (hq + hkv)
        in_k = (r >= hq) & (r < (hq + hkv))
        first = in_q.select(fx.Index(0), in_k.select(hq, hq + hkv))
        head = r - first
        seq = in_q.select(fx.Index(Sq), fx.Index(Skv))
        dim = in_v.select(fx.Index(Dv), fx.Index(Dqk))
        s_batch = in_q.select(
            fx.Index(stride_dq_batch),
            in_k.select(fx.Index(stride_dk_batch), fx.Index(stride_dv_batch)),
        )
        s_head = in_q.select(
            fx.Index(stride_dq_head),
            in_k.select(fx.Index(stride_dk_head), fx.Index(stride_dv_head)),
        )
        s_seq = in_q.select(
            fx.Index(stride_dq_seq),
            in_k.select(fx.Index(stride_dk_seq), fx.Index(stride_dv_seq)),
        )
        bit = in_q.select(
            fx.Int32(addr.MASK_DQ),
            in_k.select(fx.Int32(addr.MASK_DK), fx.Int32(addr.MASK_DV)),
        )
        selected = (mask & bit) != fx.Int32(0)

        len_q_p = fx.recast_iter(_ptr_ty(fx.Int32, 4), seq_len_q)
        len_kv_p = fx.recast_iter(_ptr_ty(fx.Int32, 4), seq_len_kv)
        length = in_q.select(
            fx.Index(fx.ptr_load(len_q_p + fx.Int64(b), fx.Int32)),
            fx.Index(fx.ptr_load(len_kv_p + fx.Int64(b), fx.Int32)),
        )

        nvec = addr.units_per_row(dim, vec)
        index = fx.Index(fx.block_idx.x) * fx.Index(BLOCK_THREADS) + tid
        q, col0 = addr.row_of_unit(index, nvec, vec)
        row = length + q
        take = selected & (row < seq)
        off = addr.element_offset(b * s_batch + head * s_head, row, col0, s_seq)

        strides_al = (
            (
                stride_dq_batch
                | stride_dq_head
                | stride_dq_seq
                | stride_dk_batch
                | stride_dk_head
                | stride_dk_seq
                | stride_dv_batch
                | stride_dv_head
                | stride_dv_seq
            )
            & fx.Int64(vec - 1)
        ) == fx.Int64(0)
        vec_ok = strides_al & ((col0 + fx.Index(vec)) <= dim)

        outs = ((DQ, take & in_q), (DK, take & in_k), (DV, take & in_v))
        for t in range_constexpr(3):
            base, hit = outs[t]
            p_vec = fx.recast_iter(_ptr_ty(elem, 16), base)
            p_elem = fx.recast_iter(_ptr_ty(elem, elem_bytes), base)
            if hit:
                if vec_ok:
                    fx.ptr_store(_zero_vec(), p_vec + fx.Int64(off))
                else:
                    for j in range_constexpr(vec):
                        if (col0 + fx.Index(j)) < dim:
                            fx.ptr_store(
                                _zero_elem(), p_elem + fx.Int64(off + fx.Index(j))
                            )

    @flyc.jit
    def launch_fmha_bwd_tailzero_gfx11(
        DQ: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        seq_len_q: fx.Pointer,
        seq_len_kv: fx.Pointer,
        B: fx.Int32,
        Hq: fx.Int32,
        Hkv: fx.Int32,
        Sq: fx.Int32,
        Skv: fx.Int32,
        Dqk: fx.Int32,
        Dv: fx.Int32,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        mask: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        units_q = addr.units_of(fx.Index(Sq), fx.Index(Dqk), vec)
        units_k = addr.units_of(fx.Index(Skv), fx.Index(Dqk), vec)
        units_v = addr.units_of(fx.Index(Skv), fx.Index(Dv), vec)
        units = (units_q > units_k).select(units_q, units_k)
        units = (units > units_v).select(units, units_v)
        planes = fx.Index(B) * addr.heads_total(fx.Index(Hq), fx.Index(Hkv))
        fmha_bwd_tailzero_gfx11_kernel(
            DQ,
            DK,
            DV,
            seq_len_q,
            seq_len_kv,
            B,
            Hq,
            Hkv,
            Sq,
            Skv,
            Dqk,
            Dv,
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_dk_batch,
            stride_dk_head,
            stride_dk_seq,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_seq,
            mask,
        ).launch(
            grid=(addr.grid_blocks(units), planes, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(*args, **kwargs):
        return flyc.compile(launch_fmha_bwd_tailzero_gfx11, *args, **kwargs)

    _launch.kernel = fmha_bwd_tailzero_gfx11_kernel
    return _launch


# V-head variant: dV has its own head count `Hv` (`Hq + Hk + Hv` planes per batch); the argument list is the one above
# with `Hv` after `Hkv`.
KERNEL_NAME_HV = "fmha_bwd_tailzero_hv_gfx11"


def tailzero_hv_arg_values(outputs, seq_lens, dims, strides, mask):
    """Launch values keyed by kernel argument name.

    `outputs` is the device addresses `(dq, dk, dv)`, `seq_lens` the addresses `(seq_len_q, seq_len_kv)`, `dims` is
    `(B, Hq, Hkv, Hv, Sq, Skv, Dqk, Dv)` and `strides` is `((b, h, s) of dQ, of dK, of dV)` in elements.
    """
    v = dict(zip(("DQ", "DK", "DV"), outputs))
    v.update(zip(("seq_len_q", "seq_len_kv"), seq_lens))
    v.update(zip(("B", "Hq", "Hkv", "Hv", "Sq", "Skv", "Dqk", "Dv"), dims))
    for name, triple in zip(("dq", "dk", "dv"), strides):
        for axis, x in zip(("batch", "head", "seq"), triple):
            v[f"stride_{name}_{axis}"] = x
    v["mask"] = mask
    return v


def build_fmha_bwd_tailzero_hv_gfx11_module_primary(dtype_str):
    if dtype_str not in _ELEM:
        raise ValueError(dtype_str)
    elem, elem_bytes = _ELEM[dtype_str]
    vec = addr.vec_of(elem_bytes)
    narrow = elem_bytes == 2
    _cache_tag = (KERNEL_NAME_HV, dtype_str)

    def _ptr_ty(ty, align):
        return fx.PointerType.get(
            elem_ty=ty.ir_type, address_space=fx.AddressSpace.Global, alignment=align
        )

    def _zero_vec():
        zero = Vec.filled(vec, 0.0, fx.Float32)
        return zero.to(elem) if narrow else zero

    def _zero_elem():
        zero = fx.Float32(0.0)
        return zero.to(elem) if narrow else zero

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def fmha_bwd_tailzero_hv_gfx11_kernel(
        DQ: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        seq_len_q: fx.Pointer,
        seq_len_kv: fx.Pointer,
        B: fx.Int32,
        Hq: fx.Int32,
        Hkv: fx.Int32,
        Hv: fx.Int32,
        Sq: fx.Int32,
        Skv: fx.Int32,
        Dqk: fx.Int32,
        Dv: fx.Int32,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        mask: fx.Int32,
    ):
        tid = fx.Index(fx.thread_idx.x)
        plane = fx.Index(fx.block_idx.y)
        hq = fx.Index(Hq)
        hkv = fx.Index(Hkv)
        hv = fx.Index(Hv)
        heads = addr.heads_total(hq, hkv, hv)
        b = addr.batch_of(plane, heads)
        r = plane - b * heads

        # the plane's tensor: dQ heads, then dK heads, then dV heads (`Hv` of them)
        in_q = r < hq
        in_v = r >= (hq + hkv)
        in_k = (r >= hq) & (r < (hq + hkv))
        first = in_q.select(fx.Index(0), in_k.select(hq, hq + hkv))
        head = r - first
        seq = in_q.select(fx.Index(Sq), fx.Index(Skv))
        dim = in_v.select(fx.Index(Dv), fx.Index(Dqk))
        s_batch = in_q.select(
            fx.Index(stride_dq_batch),
            in_k.select(fx.Index(stride_dk_batch), fx.Index(stride_dv_batch)),
        )
        s_head = in_q.select(
            fx.Index(stride_dq_head),
            in_k.select(fx.Index(stride_dk_head), fx.Index(stride_dv_head)),
        )
        s_seq = in_q.select(
            fx.Index(stride_dq_seq),
            in_k.select(fx.Index(stride_dk_seq), fx.Index(stride_dv_seq)),
        )
        bit = in_q.select(
            fx.Int32(addr.MASK_DQ),
            in_k.select(fx.Int32(addr.MASK_DK), fx.Int32(addr.MASK_DV)),
        )
        selected = (mask & bit) != fx.Int32(0)

        len_q_p = fx.recast_iter(_ptr_ty(fx.Int32, 4), seq_len_q)
        len_kv_p = fx.recast_iter(_ptr_ty(fx.Int32, 4), seq_len_kv)
        length = in_q.select(
            fx.Index(fx.ptr_load(len_q_p + fx.Int64(b), fx.Int32)),
            fx.Index(fx.ptr_load(len_kv_p + fx.Int64(b), fx.Int32)),
        )

        nvec = addr.units_per_row(dim, vec)
        index = fx.Index(fx.block_idx.x) * fx.Index(BLOCK_THREADS) + tid
        q, col0 = addr.row_of_unit(index, nvec, vec)
        row = length + q
        take = selected & (row < seq)
        off = addr.element_offset(b * s_batch + head * s_head, row, col0, s_seq)

        strides_al = (
            (
                stride_dq_batch
                | stride_dq_head
                | stride_dq_seq
                | stride_dk_batch
                | stride_dk_head
                | stride_dk_seq
                | stride_dv_batch
                | stride_dv_head
                | stride_dv_seq
            )
            & fx.Int64(vec - 1)
        ) == fx.Int64(0)
        vec_ok = strides_al & ((col0 + fx.Index(vec)) <= dim)

        outs = ((DQ, take & in_q), (DK, take & in_k), (DV, take & in_v))
        for t in range_constexpr(3):
            base, hit = outs[t]
            p_vec = fx.recast_iter(_ptr_ty(elem, 16), base)
            p_elem = fx.recast_iter(_ptr_ty(elem, elem_bytes), base)
            if hit:
                if vec_ok:
                    fx.ptr_store(_zero_vec(), p_vec + fx.Int64(off))
                else:
                    for j in range_constexpr(vec):
                        if (col0 + fx.Index(j)) < dim:
                            fx.ptr_store(
                                _zero_elem(), p_elem + fx.Int64(off + fx.Index(j))
                            )

    @flyc.jit
    def launch_fmha_bwd_tailzero_hv_gfx11(
        DQ: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        seq_len_q: fx.Pointer,
        seq_len_kv: fx.Pointer,
        B: fx.Int32,
        Hq: fx.Int32,
        Hkv: fx.Int32,
        Hv: fx.Int32,
        Sq: fx.Int32,
        Skv: fx.Int32,
        Dqk: fx.Int32,
        Dv: fx.Int32,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        mask: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        units_q = addr.units_of(fx.Index(Sq), fx.Index(Dqk), vec)
        units_k = addr.units_of(fx.Index(Skv), fx.Index(Dqk), vec)
        units_v = addr.units_of(fx.Index(Skv), fx.Index(Dv), vec)
        units = (units_q > units_k).select(units_q, units_k)
        units = (units > units_v).select(units, units_v)
        planes = fx.Index(B) * addr.heads_total(
            fx.Index(Hq), fx.Index(Hkv), fx.Index(Hv)
        )
        fmha_bwd_tailzero_hv_gfx11_kernel(
            DQ,
            DK,
            DV,
            seq_len_q,
            seq_len_kv,
            B,
            Hq,
            Hkv,
            Hv,
            Sq,
            Skv,
            Dqk,
            Dv,
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_dk_batch,
            stride_dk_head,
            stride_dk_seq,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_seq,
            mask,
        ).launch(
            grid=(addr.grid_blocks(units), planes, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(*args, **kwargs):
        return flyc.compile(launch_fmha_bwd_tailzero_hv_gfx11, *args, **kwargs)

    _launch.kernel = fmha_bwd_tailzero_hv_gfx11_kernel
    return _launch
