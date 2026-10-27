# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 MLA prefill live benchmark — rocke vs aiter comparison.

Three modes:

``--mode resolve`` (default)
    Shape -> ``MLARequest`` -> ``dispatch_mla`` -> spec, grid, block, kernel name.
    CPU only; no GPU, no comgr, no timing.

``--mode build``
    Resolve + build each selection's IR (still CPU-only). Catches lowering
    failures before touching a device.

``--mode time``
    Full live timing: builds the rocke MLA kernel, allocates torch tensors,
    runs warmup + timed iterations with ``time_launches``, and compares against
    the aiter ``mla_prefill_fwd`` absorb-mode baseline (when available).

    FlyDSL note: FlyDSL only ships an MLA *decode* kernel
    (``kernels/attention/mla_fwd_decode.py``). There is no FlyDSL MLA prefill
    implementation as of September 2026, so no ``--flydsl`` flag is provided.
    When FlyDSL adds prefill MLA support, wire it in the same way as
    ``benchmark_prefill2d_live.py`` does.

    aiter baseline: ``aiter.mla.mla_prefill_fwd`` (absorb mode, gfx942 ASM).
    It operates in *latent space*: its query is ``[total_q, H, r_kv+d_rope]``
    (576-wide) while the rocke kernel takes a post-W_UQ query
    ``[total_q, H, d_nope+d_rope]`` (192-wide). The aiter inputs are derived
    from the rocke inputs (absorbed query ``Q_nope @ W_UKᵀ`` and ``Q_rope``; a
    packed ``[c_kv, k_rope]`` cache), and both arms get the same softmax scale,
    ``1/sqrt(d_nope + d_rope)``: absorption changes the contraction width, not
    the temperature. aiter's timed region excludes the two head projections the
    fused rocke kernel performs, so the ratio compares unequal work.
    aiter only supports ``nhead in {16, 128}`` on gfx942 (hard constraint from
    its ASM kernel); shapes with other head counts skip the aiter lane.

    Numeric gate: every timed arm's output is checked against the fp32 oracle
    :func:`builders.mla.ref_mla_attn.ref_mla_prefill` -- rocke's ``out`` and
    ``lse``; aiter's latent output projected through ``W_UV`` (its gfx942 ASM
    prefill kernel leaves the LSE buffer unwritten) -- with the bars of
    :mod:`builders.mla.verify_score_probe_gfx942`. Both arms: ``out``
    ``max|got - ref| <= --tol`` (default ``BF16_ATTN_ATOL``, 4e-2) and, per gate
    window, ``out`` ``max_abs / peak(|ref|) <= --rel-tol`` (default
    ``OUT_REL_PEAK_TOL``, 2e-2). The rocke arm also: ``lse``
    ``max|got - ref| <= --lse-tol`` (default ``LSE_ATOL``, 1.7e-2); the aiter
    arm has no LSE to check. The oracle runs on a subset: up
    to four evenly spaced head pairs ``(h, h + 1)`` -- both waves of a two-head
    workgroup -- plus the last head, and the first and last ``_GATE_ROWS``
    query rows of each sequence. No ratio is reported for a shape unless both
    arms pass, and any failure -- including an oracle, projection or gate that
    raises after timing -- makes the run exit non-zero.

Run, from ``rocke/library``:

    export AITER_PATH=<path/to/aiter>
    PYTHONPATH="python:${AITER_PATH}" \\
      python -m benchmarks.gfx942.attention.prefill.benchmark_mla_prefill_live \\
        --mode time \\
        --shapes benchmarks/gfx942/attention/prefill/mla_prefill_shapes.json \\
        --limit 8

    # resolve/build only (no GPU):
    python -m ... --mode resolve
    python -m ... --mode build --regime chunked --limit 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from benchmarks.common.mla_shapes import MLAShape, filter_shapes, load_mla_shapes
from builders.mla.verify_score_probe_gfx942 import (
    BF16_ATTN_ATOL,
    LSE_ATOL,
    OUT_REL_PEAK_TOL,
)

_ARCH = "gfx942"
_DEFAULT_SHAPES = Path(__file__).resolve().parent / "mla_prefill_shapes.json"

# aiter nhead constraint for the ASM prefill kernel on gfx942
_AITER_SUPPORTED_NHEADS = frozenset({16, 128})

# aiter's ASM prefill kernel is only numerically correct at page_size == 1 (see
# _make_aiter_inputs). Independent of rocke's block_size, which is pinned to 16.
_AITER_PAGE_SIZE = 1

# Numeric gate subset: query rows checked at each end of every sequence, and
# the most heads checked per shape. The fp32 oracle at full size would need a
# [S_q, H, S_k] score tensor per sequence.
_GATE_ROWS = 32
_GATE_HEADS = 8


# ---------------------------------------------------------------------------
# Resolve / build path (CPU-only)
# ---------------------------------------------------------------------------


def _resolve(
    shape: MLAShape, *, arch: str, build: bool, absorbed_query: bool = False
) -> dict:
    """Resolve one shape; never raise, so one bad shape does not hide the rest."""
    from dispatch.mla import dispatch_mla

    row: dict = {"shape": shape.signature, "regime": shape.regime}
    try:
        result = dispatch_mla(
            shape.to_request(arch=arch, absorbed_query=absorbed_query)
        )
    except ValueError as exc:
        row["status"] = "unsupported"
        row["detail"] = str(exc)
        return row
    row.update(
        status="ok",
        candidate=result.candidate.name,
        kernel=result.spec.fwd_kernel_name(),
        grid=result.grid,
        block=result.block,
        nargs=len(result.signature),
    )
    if build:
        try:
            kernel = result.candidate.built(result.spec, arch)
            row["built"] = kernel.name
        except Exception as exc:  # pragma: no cover - diagnostic  # noqa: BLE001
            row["status"] = "build_failed"
            row["detail"] = f"{type(exc).__name__}: {exc}"
    return row


def _report_resolve(rows: list[dict]) -> int:
    width = max((len(r["shape"]) for r in rows), default=0)
    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row['shape']:<{width}}  {row['regime']:<11}  "
                f"grid={row['grid']} block={row['block']}  {row['kernel']}"
            )
        else:
            print(f"{row['shape']:<{width}}  {row['status'].upper()}: {row['detail']}")
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nSUMMARY {ok}/{len(rows)} shapes resolved on {_ARCH}")
    return 0 if ok == len(rows) else 1


# ---------------------------------------------------------------------------
# Timing path — tensor allocation
# ---------------------------------------------------------------------------


def _bench_stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


def _make_rocke_inputs(shape: MLAShape, *, seed: int = 0) -> dict[str, Any]:
    """Allocate torch tensors matching the rocke MLA kernel ABI.

    rocke takes *projected* query ``[total_q, H, d_nope+d_rope]`` plus
    separate ``c_kv``, ``k_rope``, and ``w_uk`` tensors. W_UQ projection
    happens on the host before the kernel call (or in a fused pre-kernel that
    is out of scope here); the timing region covers only the attention kernel.
    """
    import torch

    g = shape.geometry
    rng = torch.Generator(device="cuda")
    rng.manual_seed(seed)

    page = shape.block_size
    total_q = shape.total_q
    max_k = shape.seqlen_k
    # The pool to_request reports to dispatch's paged extent admission.
    num_pages = shape.num_kv_blocks
    max_bt_cols = (max_k + page - 1) // page

    dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16

    def randn(*s, std=1.0):
        x = torch.randn(*s, dtype=torch.float32, device="cuda", generator=rng)
        return (x * std).to(dtype)

    # Unit-variance regime of the verifier's make_problem: Q, c_kv, k_rope ~
    # N(0, 1) and W_UK ~ N(0, 1/r_kv), so K_nope, V and scale * q.k are ~ N(0, 1)
    # -- the regime the absolute bf16 gate tolerance is calibrated in.
    q = randn(total_q, shape.num_query_heads, g.qk_nope_dim + g.qk_rope_dim)
    c_kv = randn(num_pages, page, g.kv_lora_rank)
    k_rope = randn(num_pages, page, g.qk_rope_dim)
    # W_UK: [H, r_kv, d_nope + d_v]
    w_uk = randn(
        shape.num_query_heads,
        g.kv_lora_rank,
        g.qk_nope_dim + g.v_head_dim,
        std=g.kv_lora_rank**-0.5,
    )

    # packed varlen offsets
    seqlen_q = shape.seqlen_q
    seqlen_k = shape.seqlen_k
    batch = shape.batch
    cu_q = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * seqlen_q
    cu_k = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * seqlen_k

    # block table: each sequence owns its own pages in order
    block_table = torch.zeros(batch, max_bt_cols, dtype=torch.int32, device="cuda")
    for bi in range(batch):
        for bk in range(max_bt_cols):
            block_table[bi, bk] = bi * max_bt_cols + bk

    scale = float(1.0 / math.sqrt(g.qk_nope_dim + g.qk_rope_dim))

    return {
        "q": q,
        "c_kv": c_kv,
        "k_rope": k_rope,
        "w_uk": w_uk,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "block_table": block_table,
        "scale": scale,
        "total_q": total_q,
        "max_k": seqlen_k,
    }


def _dense_kv(shape: MLAShape, rocke_data: dict):
    """``(c_kv [B, S_k, r_kv], k_rope [B, S_k, d_rope])`` views of the rocke cache.

    ``_make_rocke_inputs`` gives each sequence its own pages in order, so the
    paged cache is the dense one reshaped.
    """
    b, s_k = shape.batch, shape.seqlen_k
    c_kv = rocke_data["c_kv"].reshape(b, -1, rocke_data["c_kv"].shape[-1])[:, :s_k]
    k_rope = rocke_data["k_rope"].reshape(b, -1, rocke_data["k_rope"].shape[-1])
    return c_kv, k_rope[:, :s_k]


def _make_aiter_inputs(shape: MLAShape, rocke_data: dict) -> dict[str, Any]:
    """Derive the aiter ``mla_prefill_fwd`` (absorb mode) inputs from rocke's.

    aiter's absorb-mode kernel takes ``q [total_q, H, r_kv+d_rope]`` (the
    query projected into the combined latent+rope space) and a packed
    ``kv_buffer [num_page, page_size, 1, r_kv+d_rope]``. Both are built from
    the rocke arm's tensors -- ``Qa = [Q_nope @ W_UK[:, :, :d_nope]ᵀ, Q_rope]``
    and ``kv = [c_kv, k_rope]`` -- so the two arms compute the same operator and
    aiter's output, projected through ``W_UV``, is checked against the same
    oracle. The softmax scale is rocke's ``1/sqrt(d_nope + d_rope)``, not
    ``1/sqrt(r_kv + d_rope)``.

    aiter uses FlashInfer-style indirections:
        qo_indptr  [batch+1] int32 — prefix sums of per-seq query lengths
        kv_indptr  [batch+1] int32 — prefix sums of per-seq KV page counts
        kv_indices [total_kv_pages] int32 — physical page indices
        kv_last_page_lens [batch] int32 — valid tokens in each last page
    """
    import torch

    g = shape.geometry

    # aiter's ASM prefill kernel only produces correct results at page_size == 1.
    # Verified in isolation against aiter's own oracle
    # (aiter_meta/csrc/cpp_itfs/mla/asm_mla_decode_fwd_test.py::test_absorb_prefill,
    # which itself pins block_size=1): page=1 -> rel_err 1.9e-03, page=16 -> NaN
    # with 65796 output elements left unwritten. rocke's own block_size is pinned
    # to block_k=16 by its spec, so the two paging schemes must NOT be shared.
    page = _AITER_PAGE_SIZE
    batch = shape.batch
    seqlen_q = shape.seqlen_q
    seqlen_k = shape.seqlen_k
    total_q = shape.total_q

    # aiter absorb qk_head_dim = r_kv + d_rope (not d_nope + d_rope)
    aiter_qk = g.kv_lora_rank + g.qk_rope_dim  # 576 for DeepSeek

    dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16
    d_nope, r_kv = g.qk_nope_dim, g.kv_lora_rank
    q_r, w_uk = rocke_data["q"], rocke_data["w_uk"]

    # absorbed query: [Q_nope @ W_UK_nopeᵀ, Q_rope], fp32 accumulate then bf16
    q = torch.empty(
        total_q, shape.num_query_heads, aiter_qk, dtype=dtype, device="cuda"
    )
    q[:, :, :r_kv] = torch.einsum(
        "thd,hrd->thr", q_r[:, :, :d_nope].float(), w_uk[:, :, :d_nope].float()
    ).to(dtype)
    q[:, :, r_kv:] = q_r[:, :, d_nope:]

    # packed KV buffer: [num_page, page_size, 1, r_kv + d_rope]; page_size 1, so
    # page i is token i of the batch-major dense cache
    assert page == 1, "the dense-cache reshape below assumes page_size == 1"
    pages_per_seq = (seqlen_k + page - 1) // page
    total_pages = batch * pages_per_seq
    c_kv, k_rope = _dense_kv(shape, rocke_data)
    kv_buffer = torch.cat([c_kv, k_rope], dim=-1).reshape(
        total_pages, page, 1, aiter_qk
    )

    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * seqlen_q
    kv_indptr = (
        torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.full(
        (batch,),
        seqlen_k - (pages_per_seq - 1) * page,
        dtype=torch.int32,
        device="cuda",
    )
    # Absorb mode rebinds v_head_dim := kv_lora_rank (512), NOT the 128-wide
    # post-W_UV output rocke produces. Allocating g.v_head_dim here under-sizes
    # the buffer the kernel writes.
    out = torch.zeros(
        total_q, shape.num_query_heads, g.kv_lora_rank, dtype=dtype, device="cuda"
    )
    # The softmax temperature is 1/sqrt(d_nope + d_rope) whatever the absorbed
    # width; 1/sqrt(aiter_qk) would be a different operator.
    sm_scale = rocke_data["scale"]

    return {
        "q": q,
        "kv_buffer": kv_buffer,
        "out": out,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "max_seqlen_q": seqlen_q,
        "sm_scale": sm_scale,
    }


# ---------------------------------------------------------------------------
# aiter baseline runner
# ---------------------------------------------------------------------------


def _aiter_supported(shape: MLAShape) -> tuple[bool, str]:
    """Return (supported, reason) for aiter mla_prefill_fwd on gfx942."""
    if shape.dtype not in ("bf16",):
        return False, f"dtype={shape.dtype!r} not supported (bf16 only on gfx942)"
    if shape.num_query_heads not in _AITER_SUPPORTED_NHEADS:
        return (
            False,
            (
                f"nhead={shape.num_query_heads} not in aiter gfx942 supported set "
                f"{sorted(_AITER_SUPPORTED_NHEADS)}"
            ),
        )
    return True, ""


def _import_aiter_mla():
    """Import ``aiter.mla.mla_prefill_fwd``; raise descriptively on failure."""
    try:
        import aiter.mla as _aiter_mla  # type: ignore

        return _aiter_mla.mla_prefill_fwd
    except ImportError as exc:
        raise RuntimeError(
            "aiter not importable. Set AITER_PATH and add it to PYTHONPATH: "
            "PYTHONPATH=python:${AITER_PATH} python -m ..."
        ) from exc


def _run_aiter_live(data: dict, *, warmup: int, iters: int) -> tuple[Any, float]:
    """Time aiter ``mla_prefill_fwd``; return ``(latent_out [total_q, H, r_kv], ms)``."""
    from rocke.runtime import synchronize_and_release, time_launches

    mla_prefill_fwd = _import_aiter_mla()
    hip_stream = _bench_stream_handle()

    q = data["q"]
    kv_buffer = data["kv_buffer"]
    out = data["out"].zero_()
    qo_indptr = data["qo_indptr"]
    kv_indptr = data["kv_indptr"]
    kv_indices = data["kv_indices"]
    kv_last_page_lens = data["kv_last_page_lens"]
    max_seqlen_q = data["max_seqlen_q"]
    sm_scale = data["sm_scale"]

    def call_once():
        return mla_prefill_fwd(
            q,
            kv_buffer,
            out,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_q,
            sm_scale,
        )

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    # NaN-seeded gated call. The returned LSE is not gated: the gfx942 ASM
    # prefill kernel never writes it (a NaN-filled LSE buffer stays NaN).
    out.fill_(float("nan"))
    call_once()
    synchronize_and_release(hip_stream)
    return out.clone(), ms


# ---------------------------------------------------------------------------
# rocke MLA kernel runner
# ---------------------------------------------------------------------------


def parse_spec_overrides(pairs) -> dict:
    """``KEY=VAL`` strings as typed :class:`MlaPrefillSpec` field overrides.

    Each value is parsed by the field's declared type, so it reaches the
    spec's exact-type checks as that type: a bool field takes only
    ``true``/``false`` (any case), an int field takes what ``int()`` parses,
    and a str field takes the text. An unknown field, a missing ``=`` or an
    unparseable value raises :class:`ValueError` naming it.
    """
    import dataclasses
    import typing

    from kernels.mla.mla_prefill_gfx942 import MlaPrefillSpec

    types = typing.get_type_hints(MlaPrefillSpec)
    names = [f.name for f in dataclasses.fields(MlaPrefillSpec)]
    overrides = {}
    for kv in pairs:
        key, sep, val = kv.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"--spec-override {kv!r}: expected KEY=VAL")
        if key not in types:
            raise ValueError(
                f"--spec-override {key!r}: not an MlaPrefillSpec field "
                f"(fields: {', '.join(names)})"
            )
        if key == "absorbed_query":
            # The request flag selects the absorbed kernel, and the report's
            # unfused split keys on it; an override would desynchronize them.
            raise ValueError(
                "--spec-override absorbed_query: use --absorbed-query instead"
            )
        kind = types[key]
        if kind is bool:
            if val.strip().lower() not in ("true", "false"):
                raise ValueError(
                    f"--spec-override {key}={val!r}: {key} is a bool, "
                    "expected true or false"
                )
            overrides[key] = val.strip().lower() == "true"
        elif kind is int:
            try:
                overrides[key] = int(val)
            except ValueError:
                raise ValueError(
                    f"--spec-override {key}={val!r}: {key} is an int, "
                    "expected an integer"
                ) from None
        else:
            overrides[key] = val
    return overrides


class SpecRefused(ValueError):
    """The ``--absorbed-query`` request or ``--spec-override`` spec was refused."""


def _build_rocke_kernel(
    shape: MLAShape,
    *,
    arch: str,
    absorbed_query: bool = False,
    spec_overrides: dict | None = None,
):
    """Build (compile to HSACO) the rocke MLA prefill kernel for ``shape``.

    Returns ``(DispatchResult, Artifact)`` where ``Artifact`` has ``.hsaco``
    and ``.kernel_name`` attributes. ``absorbed_query`` goes on the request,
    so dispatch selects the unfused kernel (see ``MLARequest.absorbed_query``)
    and its result carries that kernel's signature and grid; a dispatch
    refusal of it raises :class:`SpecRefused`. ``spec_overrides`` replaces
    further fields of the dispatched spec, which the dispatcher never selects,
    so that spec is re-admitted here -- the kernel's admission and the
    request's 32-bit extents -- and a refusal raises :class:`SpecRefused`.
    """
    import dataclasses

    from dispatch.mla import dispatch_mla
    from kernels.mla.mla_prefill_gfx942 import (
        build_mla_prefill_fwd,
        check_mla_prefill_extents,
        mla_prefill_fwd_signature,
        supports_mla_prefill,
    )
    from rocke import compile_kernel

    req = shape.to_request(arch=arch, absorbed_query=absorbed_query)
    try:
        result = dispatch_mla(req)
    except ValueError as exc:
        if not absorbed_query:
            raise
        raise SpecRefused(f"absorbed_query dispatch refused: {exc}") from None
    if spec_overrides:
        try:
            spec = dataclasses.replace(result.spec, **spec_overrides)
        except (TypeError, ValueError) as exc:
            raise SpecRefused(f"spec rejected: {exc}") from None
        ok, why = supports_mla_prefill(spec, arch=arch)
        if ok:
            ok, why = check_mla_prefill_extents(
                spec, total_q=req.total_q, num_kv_blocks=req.num_kv_blocks
            )
        if not ok:
            raise SpecRefused(why)
        result = dataclasses.replace(
            result,
            spec=spec,
            signature=tuple(mla_prefill_fwd_signature(spec)),
            grid=result.candidate.grid(spec, req),
            block=result.candidate.block(spec),
        )
        kernel_def = build_mla_prefill_fwd(spec, arch=arch)
    else:
        kernel_def = result.candidate.built(result.spec, arch)
    artifact = compile_kernel(kernel_def, arch=arch, capture_ir_text=False)
    return result, artifact


def _run_rocke_live(
    shape: MLAShape,
    data: dict,
    result,
    kernel_obj,
    *,
    warmup: int,
    iters: int,
) -> tuple[Any, Any, float, str]:
    """Time the rocke MLA prefill kernel; return ``(out, lse, ms, kernel_name)``."""
    import torch
    from rocke.runtime import (
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
        time_launches,
    )

    hip_stream = _bench_stream_handle()
    g = shape.geometry

    dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16
    out = torch.zeros(
        shape.total_q, shape.num_query_heads, g.v_head_dim, dtype=dtype, device="cuda"
    )
    lse = torch.zeros(
        shape.total_q, shape.num_query_heads, dtype=torch.float32, device="cuda"
    )

    q = data["q"]
    c_kv = data["c_kv"]
    k_rope = data["k_rope"]
    w_uk = data["w_uk"]
    cu_q = data["cu_q"]
    cu_k = data["cu_k"]
    block_table = data["block_table"]
    scale = data["scale"]
    max_k = data["max_k"]
    n_seqs = shape.batch
    bt_stride = int(block_table.shape[1])

    launcher = KernelLauncher(
        hsaco=kernel_obj.hsaco,
        kernel_name=kernel_obj.kernel_name,
        signature=tuple(result.signature),
        cache_key=("mla_prefill_live_gfx942", shape.signature),
    )

    # pack_args expects a dict keyed by the signature argument names
    # (see mla_prefill_fwd_signature: out_ptr, lse_ptr, q_ptr, c_kv_ptr,
    #  k_rope_ptr, w_uk_ptr, cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    #  block_table_ptr, scale, num_seqs, block_table_stride, max_k)
    vals = {
        "out_ptr": out,
        "lse_ptr": lse,
        "q_ptr": q,
        "c_kv_ptr": c_kv,
        "k_rope_ptr": k_rope,
        "w_uk_ptr": w_uk,
        "cu_seqlens_q_ptr": cu_q,
        "cu_seqlens_k_ptr": cu_k,
        "block_table_ptr": block_table,
        "scale": scale,
        "num_seqs": n_seqs,
        "block_table_stride": bt_stride,
        "max_k": max_k,
    }

    cfg = LaunchConfig(grid=result.grid, block=result.block, stream=hip_stream)

    if not getattr(result.spec, "absorbed_query", False):

        def call_once():
            launcher(vals, config=cfg)

        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        # NaN-seeded gated call: a slot the kernel never writes fails the gate.
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        call_once()
        synchronize_and_release(hip_stream)
        return out.clone(), lse.clone(), ms, kernel_obj.kernel_name

    # Unfused: both head projections run as batched GEMMs around the kernel,
    # and the timed region covers all of them. Every operand is a strided view
    # of the same buffers the fused kernel reads, laid out [head] x [token] for
    # torch.bmm:
    #   Qa[:, :, :r_kv] = Q_nope @ W_UK[:, :, :d_nope]ᵀ   (absorb)
    #   Qa[:, :, r_kv:] = Q_rope                          (copy)
    #   latent          = kernel(Qa, c_kv, k_rope)        (attention)
    #   out             = latent @ W_UK[:, :, d_nope:]    (W_UV)
    d_nope, r_kv = g.qk_nope_dim, g.kv_lora_rank
    qa = torch.empty(
        shape.total_q,
        shape.num_query_heads,
        r_kv + g.qk_rope_dim,
        dtype=dtype,
        device="cuda",
    )
    latent = torch.empty(
        shape.total_q, shape.num_query_heads, r_kv, dtype=dtype, device="cuda"
    )
    q_nope_t = q[:, :, :d_nope].transpose(0, 1)
    w_uk_nope_t = w_uk[:, :, :d_nope].transpose(1, 2)
    w_uv = w_uk[:, :, d_nope:]
    qa_nope_t = qa[:, :, :r_kv].transpose(0, 1)
    latent_t = latent.transpose(0, 1)
    out_t = out.transpose(0, 1)
    vals.pop("q_ptr")
    vals.pop("w_uk_ptr")
    vals["qa_ptr"] = qa
    vals["out_ptr"] = latent

    def absorb():
        torch.bmm(q_nope_t, w_uk_nope_t, out=qa_nope_t)
        qa[:, :, r_kv:].copy_(q[:, :, d_nope:])

    def attend():
        launcher(vals, config=cfg)

    def project():
        torch.bmm(latent_t, w_uv, out=out_t)

    def call_once():
        absorb()
        attend()
        project()

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    # Per-stage split, timed the same way; the stages are not overlapped in
    # call_once either, so they should roughly sum to ``ms``.
    _UNFUSED_SPLIT.clear()
    for name, fn in (("absorb", absorb), ("attend", attend), ("project", project)):
        _UNFUSED_SPLIT[name] = time_launches(
            fn, warmup=warmup, iters=iters, stream=hip_stream
        )
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    call_once()
    synchronize_and_release(hip_stream)
    return out.clone(), lse.clone(), ms, kernel_obj.kernel_name


# Per-stage times of the last unfused run, in ms (see ``_run_rocke_live``).
_UNFUSED_SPLIT: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Numeric gate against the fp32 oracle
# ---------------------------------------------------------------------------


def _gate_heads(num_heads: int) -> list[int]:
    """Evenly spaced head pairs ``(h, h + 1)``, plus the last head.

    The shipped layout packs two heads per workgroup, one wave each, so a pair
    starting on an even head gates both waves of one workgroup. ``h`` steps by
    an even stride giving at most ``_GATE_HEADS // 2`` pairs; ``H - 1`` is
    always gated so the tail workgroup is covered too.
    """
    pairs = max(1, _GATE_HEADS // 2)
    step = max(2, (num_heads // pairs) // 2 * 2)
    heads = {num_heads - 1}
    for h in range(0, num_heads, step):
        heads.update(x for x in (h, h + 1) if x < num_heads)
    return sorted(heads)


def _gate_windows(shape: MLAShape) -> list[tuple[int, int, int, int]]:
    """``(seq, row_lo, row_hi, k_len)`` query-row windows the gate checks.

    Bottom-right causal masking makes a window exact on its own: the last rows
    of a sequence see every key, and its first ``m`` rows see exactly the
    ``S_k - S_q + m`` key prefix, so each window is a valid smaller problem.
    """
    s_q, s_k = shape.seqlen_q, shape.seqlen_k
    if s_k < s_q:
        raise ValueError(f"gate windows assume S_k >= S_q; got S_q={s_q} S_k={s_k}")
    m = min(_GATE_ROWS, s_q)
    wins = []
    for b in range(shape.batch):
        wins.append((b, 0, m, s_k - s_q + m))
        if s_q > m:
            wins.append((b, s_q - m, s_q, s_k))
    return wins


def _reference(shape: MLAShape, rocke_data: dict) -> list[tuple[slice, Any, Any]]:
    """fp32 ``ref_mla_prefill`` ``(out, lse)`` on the gate windows and heads.

    The oracle reads the same bf16 values the arms read. The rocke kernel takes
    the query after W_UQ and RoPE, so the oracle is given that query as its
    latent with a per-head identity W_UQ and identity rotary tables.
    """
    import numpy as np

    from builders.mla.ref_mla_attn import ref_mla_prefill

    g = shape.geometry
    heads = _gate_heads(shape.num_query_heads)
    h, dqk = len(heads), g.qk_nope_dim + g.qk_rope_dim
    q = rocke_data["q"][:, heads].float().cpu().numpy()
    w_uk = rocke_data["w_uk"][heads].float().cpu().numpy()
    c_kv, k_rope = (t.float().cpu().numpy() for t in _dense_kv(shape, rocke_data))
    w_uq = np.zeros((h, h * dqk, dqk), dtype=np.float32)
    for i in range(h):
        w_uq[i, i * dqk : (i + 1) * dqk] = np.eye(dqk, dtype=np.float32)
    half = g.qk_rope_dim // 2
    cos_table = np.ones((1, half), dtype=np.float32)
    sin_table = np.zeros((1, half), dtype=np.float32)

    refs = []
    for b, lo, hi, k_len in _gate_windows(shape):
        m = hi - lo
        rows = slice(b * shape.seqlen_q + lo, b * shape.seqlen_q + hi)
        out, lse = ref_mla_prefill(
            q_latent=q[rows].reshape(m, h * dqk),
            # page size 1: block i of the window is key i
            c_kv=c_kv[b, :k_len][:, None, :],
            k_rope=k_rope[b, :k_len][:, None, :],
            w_uq=w_uq,
            w_uk=w_uk,
            cu_seqlens_q=[0, m],
            cu_seqlens_k=[0, k_len],
            block_table=np.arange(k_len, dtype=np.int32)[None, :],
            positions=np.zeros(m, dtype=np.int32),
            scale=rocke_data["scale"],
            cos_table=cos_table,
            sin_table=sin_table,
            causal=True,
        )
        refs.append((rows, out, lse))
    return refs


def _gate(
    out,
    lse,
    refs,
    *,
    tol: float = BF16_ATTN_ATOL,
    rel_tol: float = OUT_REL_PEAK_TOL,
    lse_tol: float = LSE_ATOL,
) -> dict:
    """Gate ``out [total_q, h, d_v]`` and, if given, ``lse [total_q, h]``.

    The verifier's bars (:mod:`builders.mla.verify_score_probe_gfx942`):
    ``out`` ``max_abs <= tol`` and, per gate window, ``max_abs / peak(|ref|)
    <= rel_tol`` -- ``out`` shrinks as S_k grows, so a lost key tile can stay
    under the absolute bar -- and ``lse`` ``max_abs <= lse_tol``. ``out`` and
    ``lse`` are already restricted to the gate heads. All rows are live
    (``S_k >= S_q``), so every gated slot must be finite. ``lse=None`` gates
    ``out`` alone, for an arm that does not produce an LSE.
    """
    import numpy as np

    worst_o = 0.0
    worst_rel = 0.0
    worst_l = 0.0 if lse is not None else None
    finite = True
    for rows, ref_out, ref_lse in refs:
        o = out[rows].float().cpu().numpy()
        if not np.all(np.isfinite(o)):
            finite = False
            continue
        err = float(np.max(np.abs(o - ref_out)))
        peak = float(np.max(np.abs(ref_out))) if ref_out.size else 0.0
        worst_o = max(worst_o, err)
        worst_rel = max(worst_rel, err / (peak if peak > 0 else 1.0))
        if lse is not None:
            lv = lse[rows].float().cpu().numpy()
            if not np.all(np.isfinite(lv)):
                finite = False
                continue
            worst_l = max(worst_l, float(np.max(np.abs(lv - ref_lse))))
    return {
        "ok": finite
        and worst_o <= tol
        and worst_rel <= rel_tol
        and (worst_l is None or worst_l <= lse_tol),
        "finite": finite,
        "max_abs_out": worst_o,
        "max_rel_peak_out": worst_rel,
        "max_abs_lse": worst_l,
    }


def _gate_error(exc: BaseException) -> dict:
    """A gate verdict for an arm whose oracle, projection or gate raised.

    The arm was timed but never checked, so it counts as a gate failure rather
    than as a missing gate a caller could read as "not applicable".
    """
    return {
        "ok": False,
        "finite": False,
        "max_abs_out": None,
        "max_rel_peak_out": None,
        "max_abs_lse": None,
        "error": repr(exc),
    }


def _gate_str(name: str, gate: dict | None) -> str:
    if gate is None:
        return ""
    if gate.get("error"):
        return f" {name}_gate=FAIL(error)"
    if not gate["finite"]:
        return f" {name}_gate=FAIL(non-finite)"
    verdict = "ok" if gate["ok"] else "FAIL"
    lse = gate["max_abs_lse"]
    lse_part = f" lse {lse:.1e}" if lse is not None else " lse n/a"
    return (
        f" {name}_gate={verdict}(out {gate['max_abs_out']:.1e}"
        f" rel {gate['max_rel_peak_out']:.1e}{lse_part})"
    )


# ---------------------------------------------------------------------------
# TFLOP count for MLA prefill (absorb mode, fused)
# ---------------------------------------------------------------------------


def _mla_prefill_flops(shape: MLAShape) -> float:
    """Approximate arithmetic intensity (TFLOP) for one MLA prefill shape.

    Counts the dominant GEMM work inside the kernel (score + softmax-weighted V
    accumulation). W_UQ and W_UK projections (done outside or fused) are excluded
    so the number is comparable across implementations with different fusion
    boundaries.
    """
    g = shape.geometry
    total_q = shape.total_q
    seqlen_k = shape.seqlen_k
    H = shape.num_query_heads
    # Score: total_q × seqlen_k × head_dim_qk (×2 for multiply-add)
    score_flops = 2.0 * total_q * seqlen_k * g.head_dim_qk * H
    # Value accumulation: total_q × seqlen_k × v_head_dim (×2)
    value_flops = 2.0 * total_q * seqlen_k * g.v_head_dim * H
    return (score_flops + value_flops) / 1e12


def _gm(vals: list[float]) -> float:
    vals = [v for v in vals if v > 0]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", type=Path, default=_DEFAULT_SHAPES)
    parser.add_argument(
        "--mode", choices=("resolve", "build", "time"), default="resolve"
    )
    parser.add_argument("--regime", choices=("full_prompt", "chunked"), default=None)
    parser.add_argument("--model", default=None, help="substring match on model name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arch", default=_ARCH)
    parser.add_argument(
        "--list-shapes", action="store_true", help="print the shapes and exit"
    )
    # timing-mode args
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="write timing results to this JSON file (--mode time only)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=BF16_ATTN_ATOL,
        help="absolute max|got - ref| bar for out against the fp32 oracle, "
        f"both arms (default: the bf16 attention bar, {BF16_ATTN_ATOL:g})",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=OUT_REL_PEAK_TOL,
        help="max|got - ref| / peak|ref| bar for out per gate window, both arms "
        f"(default {OUT_REL_PEAK_TOL:g})",
    )
    parser.add_argument(
        "--lse-tol",
        type=float,
        default=LSE_ATOL,
        help=f"absolute bar for lse, rocke arm only (default {LSE_ATOL:g})",
    )
    parser.add_argument(
        "--no-aiter",
        action="store_true",
        help="skip the aiter mla_prefill_fwd baseline even if available",
    )
    parser.add_argument(
        "--absorbed-query",
        action="store_true",
        help="time the unfused path end to end: torch.bmm absorb + the "
        "absorbed_query kernel + torch.bmm W_UV (rocke_ms covers all three)",
    )
    parser.add_argument(
        "--spec-override",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="override a dispatched MlaPrefillSpec field, parsed by its type "
        "(int, or true/false for a bool), repeatable",
    )
    args = parser.parse_args(argv)
    try:
        spec_overrides = parse_spec_overrides(args.spec_override)
    except ValueError as exc:
        parser.error(str(exc))

    if args.mode in ("resolve", "build"):
        shapes = filter_shapes(
            load_mla_shapes(args.shapes),
            regime=args.regime,
            model=args.model,
            limit=args.limit,
        )
        if not shapes:
            print("no shapes matched the filters", file=sys.stderr)
            return 1
        if args.list_shapes:
            for shape in shapes:
                print(f"{shape.signature}  regime={shape.regime}")
            return 0

        rows = [
            _resolve(
                shape,
                arch=args.arch,
                build=args.mode == "build",
                absorbed_query=args.absorbed_query,
            )
            for shape in shapes
        ]
        return _report_resolve(rows)

    # --mode time
    import torch

    if not torch.cuda.is_available():
        print("no GPU available", file=sys.stderr)
        return 1

    shapes = filter_shapes(
        load_mla_shapes(args.shapes),
        regime=args.regime,
        model=args.model,
        limit=args.limit,
    )
    if not shapes:
        print("no shapes matched the filters", file=sys.stderr)
        return 1
    if args.list_shapes:
        for shape in shapes:
            print(f"{shape.signature}  regime={shape.regime}")
        return 0

    print(f"device:  {torch.cuda.get_device_name(0)}")
    print(f"arch:    {args.arch}")
    print(f"shapes:  {len(shapes)}")
    print(f"warmup:  {args.warmup}  iters: {args.iterations}")
    print(
        f"aiter:   {'disabled (--no-aiter)' if args.no_aiter else 'enabled (bf16, nhead in {16,128})'}"
    )
    print("flydsl:  no MLA prefill kernel in FlyDSL (decode-only as of Sep 2026)")

    bars = {"tol": args.tol, "rel_tol": args.rel_tol, "lse_tol": args.lse_tol}
    results = []
    n_ok = 0
    n_aiter_ran = 0
    n_aiter_supported = 0
    n_gate_fail = 0

    for i, shape in enumerate(shapes, 1):
        tag = f"[{i}/{len(shapes)}] {shape.signature}"
        # TFLOP (not FLOP) -- _mla_prefill_flops already scales by 1e12.
        tflop = _mla_prefill_flops(shape)

        # -- build rocke kernel (once per shape)
        try:
            result, kernel_obj = _build_rocke_kernel(
                shape,
                arch=args.arch,
                absorbed_query=args.absorbed_query,
                spec_overrides=spec_overrides,
            )
        except SpecRefused as exc:
            print(f"{tag}  ROCKE SPEC REFUSED: {exc}")
            results.append(
                {
                    "signature": shape.signature,
                    "regime": shape.regime,
                    "status": "unsupported",
                    "detail": str(exc),
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}  ROCKE BUILD FAIL: {exc!r}")
            traceback.print_exc()
            results.append(
                {
                    "signature": shape.signature,
                    "regime": shape.regime,
                    "status": "build_fail",
                    "detail": repr(exc),
                }
            )
            continue

        rocke_ms: float | None = None
        rocke_kname: str | None = None
        rocke_gate: dict | None = None
        heads = _gate_heads(shape.num_query_heads)
        refs = None
        rocke_data = None
        try:
            rocke_data = _make_rocke_inputs(shape, seed=args.seed)
            r_out, r_lse, rocke_ms, rocke_kname = _run_rocke_live(
                shape,
                rocke_data,
                result,
                kernel_obj,
                warmup=args.warmup,
                iters=args.iterations,
            )
            refs = _reference(shape, rocke_data)
            rocke_gate = _gate(r_out[:, heads], r_lse[:, heads], refs, **bars)
            del r_out, r_lse
            if rocke_gate["ok"]:
                n_ok += 1
            else:
                n_gate_fail += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}  ROCKE TIME/GATE FAIL: {exc!r}")
            traceback.print_exc()
            if rocke_ms is not None:
                # Timed but never checked: a gate failure, not a missing gate.
                rocke_gate = _gate_error(exc)
                n_gate_fail += 1

        rocke_tflops = (
            tflop / (rocke_ms * 1e-3) if (rocke_ms and rocke_ms > 0) else None
        )

        # -- aiter baseline, on inputs derived from the rocke arm's
        aiter_ms: float | None = None
        aiter_gate: dict | None = None
        aiter_ok, aiter_skip_reason = _aiter_supported(shape)
        if not args.no_aiter and aiter_ok:
            n_aiter_supported += 1
            try:
                if rocke_data is None or refs is None:
                    raise RuntimeError("no rocke inputs/oracle to derive from")
                aiter_data = _make_aiter_inputs(shape, rocke_data)
                a_latent, aiter_ms = _run_aiter_live(
                    aiter_data, warmup=args.warmup, iters=args.iterations
                )
                del aiter_data
                n_aiter_ran += 1
                # W_UV outside the timed region, on the gated heads only
                w_uv = rocke_data["w_uk"][heads, :, shape.geometry.qk_nope_dim :]
                a_out = torch.einsum(
                    "thr,hrv->thv", a_latent[:, heads].float(), w_uv.float()
                )
                aiter_gate = _gate(a_out, None, refs, **bars)
                del a_latent, a_out
                if not aiter_gate["ok"]:
                    n_gate_fail += 1
            except Exception as exc:  # noqa: BLE001
                print(f"{tag}  AITER FAIL: {exc!r}")
                traceback.print_exc()
                if aiter_ms is not None and aiter_gate is None:
                    # Timed, but the W_UV projection or the gate raised.
                    aiter_gate = _gate_error(exc)
                    n_gate_fail += 1
        del rocke_data, refs

        aiter_tflops = (
            tflop / (aiter_ms * 1e-3) if (aiter_ms and aiter_ms > 0) else None
        )
        # A ratio is only reported when both arms passed the numeric gate.
        both_gated = bool(
            rocke_gate and rocke_gate["ok"] and aiter_gate and aiter_gate["ok"]
        )
        speedup_vs_aiter = (
            aiter_ms / rocke_ms
            if (both_gated and aiter_ms and rocke_ms and rocke_ms > 0)
            else None
        )

        if rocke_ms is None:
            status = "fail"
        elif not (rocke_gate and rocke_gate["ok"]) or (
            aiter_gate is not None and not aiter_gate["ok"]
        ):
            status = "gate_fail"
        else:
            status = "ok"
        rec = {
            "signature": shape.signature,
            "regime": shape.regime,
            "model": shape.model,
            "batch": shape.batch,
            "seqlen_q": shape.seqlen_q,
            "seqlen_k": shape.seqlen_k,
            "num_query_heads": shape.num_query_heads,
            "dtype": shape.dtype,
            "status": status,
            "rocke_ms": rocke_ms,
            "rocke_tflops": rocke_tflops,
            "rocke_kernel": rocke_kname,
            "rocke_gate": rocke_gate,
            "aiter_ms": aiter_ms,
            "aiter_tflops": aiter_tflops,
            "aiter_gate": aiter_gate,
            "aiter_skip_reason": aiter_skip_reason if not aiter_ok else None,
            "speedup_rocke_vs_aiter": speedup_vs_aiter,
            "gate_tol": args.tol,
            "gate_rel_tol": args.rel_tol,
            "gate_lse_tol": args.lse_tol,
            "tflop": tflop,
            "unfused_split_ms": dict(_UNFUSED_SPLIT) if args.absorbed_query else None,
            # FlyDSL: no MLA prefill kernel available (decode-only)
            "flydsl_ms": None,
            "flydsl_skip_reason": "no MLA prefill kernel in FlyDSL (decode-only as of Sep 2026)",
        }
        results.append(rec)

        # per-shape line
        rocke_str = f"rocke={rocke_ms * 1000:.1f}us" if rocke_ms else "rocke=FAIL"
        tf_str = f" {rocke_tflops:.1f}TF" if rocke_tflops else ""
        rocke_str += tf_str + _gate_str("rocke", rocke_gate)
        aiter_str = ""
        if args.no_aiter:
            aiter_str = " aiter=disabled"
        elif not aiter_ok:
            aiter_str = f" aiter=SKIP({aiter_skip_reason})"
        elif aiter_ms:
            spd = (
                f" ({speedup_vs_aiter:.2f}x)"
                if speedup_vs_aiter
                else " (ratio withheld: gate)"
            )
            aiter_str = (
                f" aiter={aiter_ms * 1000:.1f}us{_gate_str('aiter', aiter_gate)}{spd}"
            )
        else:
            aiter_str = " aiter=FAIL"
        print(f"{tag}  {rocke_str}{aiter_str}")

    # summary
    print(
        f"\n=== SUMMARY (arch={args.arch}, gate out {args.tol:g} abs / "
        f"{args.rel_tol:g} of peak, lse {args.lse_tol:g}) ==="
    )
    # Gated shapes only: a gate_fail shape's timing is not a result.
    ok_recs = [r for r in results if r.get("status") == "ok"]
    if ok_recs:
        by_regime: dict[str, list] = {}
        for r in ok_recs:
            by_regime.setdefault(r["regime"], []).append(r)
        for regime, rs in sorted(by_regime.items()):
            rocke_us = [r["rocke_ms"] * 1000 for r in rs]
            tf_vals = [r["rocke_tflops"] for r in rs if r.get("rocke_tflops")]
            aiter_spds = [
                r["speedup_rocke_vs_aiter"]
                for r in rs
                if r.get("speedup_rocke_vs_aiter")
            ]
            lat_part = f"rocke_lat_gm={_gm(rocke_us):.1f}us (n={len(rocke_us)})"
            tf_part = f"  tflops_gm={_gm(tf_vals):.1f}" if tf_vals else ""
            aiter_part = (
                f"  vs_aiter={_gm(aiter_spds):.3f}x (n={len(aiter_spds)}, both gated)"
                if aiter_spds
                else ""
            )
            print(f"  {regime:<12}  {lat_part}{tf_part}{aiter_part}")
    print(f"\nrocke gated ok:    {n_ok}/{len(shapes)}")
    print(f"aiter supported:   {n_aiter_supported}  ran: {n_aiter_ran}")
    print(f"gate failures:     {n_gate_fail}")
    print("flydsl:            N/A (no MLA prefill kernel)")
    print(
        "note: the rocke arm includes the W_UK absorb and W_UV projection; the "
        "aiter arm's timed region does not"
    )

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.output_json}  ({len(results)} shapes)")

    return 0 if (n_ok == len(shapes) and n_gate_fail == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
