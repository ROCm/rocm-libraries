# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.
"""FlyDSL dSink reduction for the split backward (gfx950): the per-head sum of the prep twin's sink term.

The prep twin (`fmha_bwd_prep_gfx950`, `sink=1`) writes `t[b, h, i] = -exp(sink_h - LSE_i) * Delta_i` to an fp32 workspace `SinkTerm`
([B, Hq, Sq], addressed like LSE and Delta through `fmha.lse_row_addressing` / `lse_token_pitch`: dense, CUMULATIVE and
INDIVIDUAL+ARRAY varlen, HT / TH; zero-filled by the caller, rows past a sequence never read). dSink is the sum over every batch and row:

    dSink[h] = round_to_io_dtype( sum_b sum_i t[b, h, i] )                       fp32 accumulation, one rounding (RNE)

**Deterministic.** One thread owns one output element and sums its terms itself in a fixed order, batch-major then row (sequence z, then
row 0, 1, ...). No atomics, no cross-thread combine, no dependence on the launch geometry, so the result is bitwise identical run to run.
The loop reads UNROLL rows per trip as independent loads and adds them to the one accumulator in row order, so the unrolled sum is the
sequential sum. Grid = (num_head_q, 1, 1); the 64 lanes of the block run the same work and lane 0 stores.

**ABI** (one object per dtype; every size is a launch value; names as the prep kernel's varlen descriptors):
    SinkTerm    fp32 workspace, 16-byte aligned
    DSink       [Hq] in the I/O dtype, unit stride, 16-byte aligned
    batch_size, seqinfo_q0, seqinfo_q1, varlen_bits, num_seqlens, max_seqlen_q, num_head_q
"""

from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

KERNEL_NAME = "fmha_bwd_dsink_reduce_gfx950"
BLOCK_THREADS = 64
UNROLL = 8  # rows per loop trip

# IEEE adds, no fast-math: the sum is the contract.
_COMPILE_HINTS = {}


def dsink_reduce_arg_values(vals, sinkterm, dsink):
    """Launch values keyed by kernel argument name. `vals` is the dict of launch values keyed by kernel argument name (it holds batch_size, the varlen
    descriptors, max_seqlen_q, num_head_q); `sinkterm` / `dsink` are device addresses.
    """
    keys = (
        "batch_size",
        "seqinfo_q0",
        "seqinfo_q1",
        "varlen_bits",
        "num_seqlens",
        "max_seqlen_q",
        "num_head_q",
    )
    out = {k: vals[k] for k in keys}
    out["SinkTerm"], out["DSink"] = sinkterm, dsink
    return out


def _carried(values):
    """The one value an `scf.for` with a single carried value hands back (it is unwrapped, not a one-element list)."""
    if const_expr(isinstance(values, (list, tuple))):
        return values[0]
    return values


def _elem(dtype_str):
    return {"f16": fx.Float16, "bf16": fx.BFloat16}[dtype_str]


def build_fmha_bwd_dsink_reduce_gfx950_module_primary(dtype_str):
    if dtype_str not in ("f16", "bf16"):
        raise ValueError(dtype_str)
    elem = _elem(dtype_str)
    _cache_tag = (KERNEL_NAME, dtype_str)

    def _ptr_ty(ty, align):
        return fx.PointerType.get(
            elem_ty=ty.ir_type, address_space=fx.AddressSpace.Global, alignment=align
        )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def fmha_bwd_dsink_reduce_gfx950_kernel(
        SinkTerm: fx.Pointer,
        DSink: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        num_head_q: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        head = fx.Index(fx.block_idx.x)
        nseq = fx.Index((num_seqlens != fx.Int32(0)).select(num_seqlens, batch_size))
        tokens = fmha.lse_token_pitch(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens
        )
        st_p = fx.recast_iter(_ptr_ty(fx.Float32, 4), SinkTerm)

        init_args = [fx.Float32(0.0)]
        for z, z_args in range(fx.Index(0), nseq, fx.Index(1), init=init_args):
            seqlen, row_off, batch = fmha.decode_addressing(
                varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, fx.Int32(z)
            )
            base, pitch = fmha.lse_row_addressing(
                varlen_bits,
                batch,
                head,
                fx.Index(num_head_q),
                fx.Index(tokens),
                row_off,
            )
            sl = fx.Index(seqlen)
            nchunk = (sl + fx.Index(UNROLL - 1)) // fx.Index(UNROLL)
            for c, c_args in range(fx.Index(0), nchunk, fx.Index(1), init=z_args):
                acc = fx.Float32(_carried(c_args))
                r0 = c * fx.Index(UNROLL)
                for u in range_constexpr(UNROLL):
                    r = r0 + fx.Index(u)
                    ok = r < sl
                    rc = ok.select(r, fx.Index(0))
                    v = fx.ptr_load(st_p + fx.Int64(base + rc * pitch), fx.Float32)
                    acc = acc + ok.select(v, fx.Float32(0.0))
                c_results = yield [acc]
            z_results = yield c_results

        total = fx.Float32(_carried(z_results))
        if tid == fx.Int32(0):
            fx.ptr_store(
                total.to(elem), fx.recast_iter(_ptr_ty(elem, 2), DSink) + fx.Int64(head)
            )

    @flyc.jit
    def launch_fmha_bwd_dsink_reduce_gfx950(
        SinkTerm: fx.Pointer,
        DSink: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        num_head_q: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        fmha_bwd_dsink_reduce_gfx950_kernel(
            SinkTerm,
            DSink,
            batch_size,
            seqinfo_q0,
            seqinfo_q1,
            varlen_bits,
            num_seqlens,
            max_seqlen_q,
            num_head_q,
        ).launch(
            grid=(fx.Index(num_head_q), 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(*args, **kwargs):
        return flyc.compile(launch_fmha_bwd_dsink_reduce_gfx950, *args, **kwargs)

    return _launch
