#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side builder for the gfx942 dense flash-attention prefill kernel
(``kernels/gfx942/attention_dense.py``).

Mirrors the gfx950 builder (``builders/gfx950/attention/prefill/
attention_dense_prefill.py``): spec construction, compilation, ABI signature,
launch, a torch/SDPA parity check, and a TFLOPS benchmark. Targets ``arch="gfx942"``.

This is the single-shape smoke path. Full-cohort parity + perf is driven by
``benchmarks/gfx942/attention/prefill/benchmark_dense_prefill_live.py``, which is the
numeric gate for this kernel.

THE SPEC COMES FROM DISPATCH, NOT FROM THIS FILE
------------------------------------------------
Both harnesses (this one and the benchmark) used to HAND-BUILD an
``AttentionDenseSpec`` from CLI defaults. That is a silent-staleness machine: the
tuning that actually ships lives in ``dispatch.attention.gfx942._dense_spec``
(per-config ``waves_per_eu``, the 304-CTA persistent grid, the auto persistent
decision, the ragged path), and a hardcoded CLI default freezes whatever that
policy happened to be on the day the flag was written. A harness that measures a
config nobody ships is worse than no harness: it once reported a real speedup as a
regression.

So the flow here is:

1. build an :class:`AttentionRequest` from the CLI *shape* arguments,
2. resolve the shipped spec through
   :func:`dispatch.attention.gfx942.dense_spec_for_request` (import it from the
   ARCH module — the ``dispatch.attention`` package re-export of that name is
   gfx950's and would hand back the untuned spec),
3. apply only the tuning flags the user *explicitly passed* as a
   ``dataclasses.replace`` override on top of the resolved spec.

Every tuning flag therefore defaults to ``None`` meaning "whatever dispatch
ships": an omitted flag TRACKS future policy changes instead of pinning today's
value. :func:`resolve_dense_spec` re-derives the shipped spec and raises if any
non-overridden field drifted, so a future edit cannot quietly reintroduce a
hand-built spec.

These helpers (:func:`add_dense_tuning_args`, :func:`dense_request`,
:func:`dense_spec_overrides`, :func:`resolve_dense_spec`,
:func:`describe_dense_spec`) are the SINGLE copy — the live benchmark imports them
from here rather than keeping a second, drifting resolver.

NOTE: ``--sw`` (sliding window, P1+) still builds a spec that
``supports_attention_dense`` rejects, so ``run`` raises a ``ValueError`` naming the
reason. ``--persistent`` is NOT in that category any more: the persistent grid
ships and dispatch turns it on automatically for large-Sq prefill. ``--bn`` must
divide the 256-row query tile and keep ``K_lds+V_lds`` inside the 64 KB gfx942 LDS
-- ``--bn 128`` exceeds it at D128 and is rejected (it fits at D64).

Usage:
    python attention_dense_prefill.py                 # parity + bench, dispatch spec
    python attention_dense_prefill.py --dtype fp16 --d 64
    python attention_dense_prefill.py --persistent off   # force the default grid
    python attention_dense_prefill.py --wpe 2            # override the shipped wpe
"""
import argparse
import dataclasses
import math
import os
import sys

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from dispatch.attention.common import AttentionRequest  # noqa: E402

# Import from the ARCH module, never from the ``dispatch.attention`` package: the
# package-level re-export of this name is gfx950's and would silently hand a
# gfx942 request the untuned spec (256 CTAs, no waves-per-eu bump).
from dispatch.attention.gfx942 import dense_spec_for_request  # noqa: E402
from kernels.gfx942.attention_dense import (  # noqa: E402
    AttentionDenseSpec,
    attention_dense_block,
    attention_dense_grid,
    attention_dense_signature,
    build_attention_dense,
    gfx942_kernel_name,
    supports_attention_dense,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig  # noqa: E402

_ARCH = "gfx942"
_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Spec fields a harness may override on top of the dispatch-resolved spec. Every
# one of these is a real ``AttentionDenseSpec`` field with no ``AttentionRequest``
# counterpart, so dispatch cannot resolve it and an explicit flag is the only way
# to reach it. The persistent knobs are deliberately NOT here: they have request
# fields (``dense_persistent`` / ``dense_num_persistent`` /
# ``dense_persist_decode``), so they go through dispatch and get its gfx942
# normalization (e.g. the shared 256-CTA default -> 304) instead of bypassing it.
_OVERRIDE_FIELDS = (
    "block_n",
    "waves_per_eu",
    "interleave",
    "lds_k_group_pad",
    "sliding_window",
)


# --------------------------------------------------------------------------- #
# CLI -> AttentionRequest -> dispatch-resolved AttentionDenseSpec
# --------------------------------------------------------------------------- #
def add_dense_tuning_args(ap: argparse.ArgumentParser) -> None:
    """Add the dense tuning flags, every one defaulting to ``None``.

    ``None`` means "whatever dispatch ships for this request". That is the whole
    point of the default: a hardcoded number here would freeze today's policy into
    the harness and the harness would then measure a kernel that never ships.
    """
    ap.add_argument(
        "--bn",
        dest="block_n",
        type=int,
        default=None,
        help="block_n (KV tile) override; default = whatever dispatch ships",
    )
    ap.add_argument(
        "--wpe",
        "--waves-per-eu",
        dest="waves_per_eu",
        type=int,
        default=None,
        help="waves_per_eu override; default = the per-config tuned value dispatch ships",
    )
    ap.add_argument(
        "--persistent",
        nargs="?",
        const="on",
        choices=["auto", "on", "off"],
        default=None,
        help=(
            "persistent grid-stride kernel: 'auto' (dispatch decides, the default), "
            "'on' (bare --persistent), or 'off'"
        ),
    )
    ap.add_argument(
        "--np",
        "--num-persistent",
        dest="num_persistent",
        type=int,
        default=None,
        help="num_persistent CTAs; default = the gfx942 CU count dispatch ships",
    )
    ap.add_argument(
        "--persist-decode",
        dest="persist_decode",
        choices=["auto", "qb_major", "hkv_major"],
        default=None,
        help="persistent work-item decode; default = dispatch's 'auto'",
    )
    ap.add_argument(
        "--interleave",
        dest="interleave",
        action="store_const",
        const=True,
        default=None,
        help="boustrophedon qb order (persistent only)",
    )
    ap.add_argument(
        "--no-interleave",
        dest="interleave",
        action="store_const",
        const=False,
        help="force boustrophedon qb order off",
    )
    ap.add_argument(
        "--kpad",
        "--lds-k-group-pad",
        dest="lds_k_group_pad",
        type=int,
        default=None,
        help="K row-GROUP LDS pad in elements (D64 path); 0 reproduces the unpadded A/B",
    )
    ap.add_argument(
        "--sw",
        dest="sliding_window",
        type=int,
        default=None,
        help="sliding_window (0=off; multiple of block_n). P1+: currently rejected",
    )


def dense_request(
    args: argparse.Namespace,
    *,
    batch: int,
    seqlen_q: int,
    seqlen_kv: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    causal: bool,
    dtype: str,
) -> AttentionRequest:
    """The :class:`AttentionRequest` a production caller would submit.

    Shape comes from the caller; the three persistent knobs come from the CLI when
    explicitly passed and otherwise keep the request defaults, so dispatch applies
    its own gfx942 normalization to them.
    """
    req_kwargs = {}
    if getattr(args, "persistent", None) is not None:
        req_kwargs["dense_persistent"] = args.persistent
    if getattr(args, "num_persistent", None) is not None:
        req_kwargs["dense_num_persistent"] = int(args.num_persistent)
    if getattr(args, "persist_decode", None) is not None:
        req_kwargs["dense_persist_decode"] = args.persist_decode
    return AttentionRequest(
        batch=int(batch),
        nhead_q=int(num_query_heads),
        nhead_k=int(num_kv_heads),
        seqlen_q=int(seqlen_q),
        seqlen_k=int(seqlen_kv),
        hdim_q=int(head_size),
        hdim_v=int(head_size),
        arch=_ARCH,
        mask_type=1 if causal else 0,
        dtype=str(dtype).lower(),
        # Opt-in selector: this is the candidate whose spec we are measuring.
        algorithm="attention_dense",
        spec_id="gfx942_attention_dense",
        **req_kwargs,
    )


def dense_spec_overrides(args: argparse.Namespace) -> dict:
    """Spec-field overrides for the tuning flags the user EXPLICITLY passed."""
    return {
        name: getattr(args, name)
        for name in _OVERRIDE_FIELDS
        if getattr(args, name, None) is not None
    }


def assert_tracks_dispatch(
    spec: AttentionDenseSpec,
    req: AttentionRequest,
    overrides: "dict | None" = None,
) -> None:
    """Raise unless ``spec`` is what dispatch would ship for ``req``, modulo overrides.

    The anti-staleness mechanism, not decoration. It resolves the shipped spec
    ITSELF (rather than trusting a value the caller already had) and compares every
    field the caller did not explicitly override. A future edit that hand-sets a
    field after resolution -- the exact regression this rewrite removes -- fails
    loudly here instead of silently measuring a kernel that never ships.

    A ``raise``, not an ``assert``: ``python -O`` strips asserts and this guards a
    wrong number, which is worse than a crash.
    """
    overrides = dict(overrides or {})
    shipped = dense_spec_for_request(req)
    drift = [
        f"{f.name}: harness={getattr(spec, f.name)!r} dispatch={getattr(shipped, f.name)!r}"
        for f in dataclasses.fields(AttentionDenseSpec)
        if f.name not in overrides and getattr(spec, f.name) != getattr(shipped, f.name)
    ]
    if drift:
        raise RuntimeError(
            "harness spec does not match the spec dispatch would ship for this "
            "request; fields not explicitly overridden must track dispatch "
            f"(overrides={sorted(overrides)}): " + "; ".join(drift)
        )


def resolve_dense_spec(
    req: AttentionRequest, overrides: "dict | None" = None
) -> AttentionDenseSpec:
    """The spec dispatch ships for ``req``, plus the caller's explicit overrides."""
    overrides = dict(overrides or {})
    known = {f.name for f in dataclasses.fields(AttentionDenseSpec)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"unknown AttentionDenseSpec override field(s): {unknown}; "
            f"known fields: {sorted(known)}"
        )

    spec = dense_spec_for_request(req)
    if overrides:
        spec = dataclasses.replace(spec, **overrides)
    assert_tracks_dispatch(spec, req, overrides)
    return spec


def describe_dense_spec(
    spec: AttentionDenseSpec, overrides: "dict | None" = None
) -> str:
    """One-line identity of what is actually about to be built and measured."""
    tag = gfx942_kernel_name(spec)
    if overrides:
        tag += (
            "  [OVERRIDES "
            + " ".join(f"{k}={v!r}" for k, v in sorted(overrides.items()))
            + "]"
        )
    return tag


# --------------------------------------------------------------------------- #
# compile + launch
# --------------------------------------------------------------------------- #
def _make_launcher(spec: AttentionDenseSpec):
    """kernel-spec generation + compilation + ABI signature -> cached launcher."""
    ok, why = supports_attention_dense(spec, arch=_ARCH)
    if not ok:
        raise ValueError(f"unsupported spec: {why}")
    art = compile_kernel(
        build_attention_dense(spec, arch=_ARCH),
        arch=_ARCH,
        backend="python",
        capture_ir_text=False,
    )
    return KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=attention_dense_signature(spec),
    )


def _launch_config(spec: AttentionDenseSpec, stream) -> LaunchConfig:
    # Geometry is owned by the kernel module (it also handles the persistent grid),
    # never re-derived here -- a third copy of BLOCK_M would defeat the import-time
    # binding in attention_dense.py.
    return LaunchConfig(
        grid=attention_dense_grid(spec),
        block=attention_dense_block(spec),
        stream=stream,
    )


def run(
    spec: AttentionDenseSpec,
    *,
    warmup: int = 15,
    iters: int = 50,
    check: bool = True,
    overrides: "dict | None" = None,
):
    dev = "cuda"
    dt = _TORCH_DT[spec.dtype]
    B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    Hq, Hkv, D = spec.num_query_heads, spec.num_kv_heads, spec.head_size
    torch.manual_seed(0)
    q = (torch.randn(B, Sq, Hq, D, dtype=dt, device=dev) * 0.2).contiguous()
    k = (torch.randn(B, Skv, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
    v = (torch.randn(B, Skv, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
    out = torch.zeros(B, Sq, Hq, D, dtype=dt, device=dev)
    scale = 1.0 / math.sqrt(D)

    launcher = _make_launcher(spec)
    stream = torch.cuda.current_stream().cuda_stream
    cfg = _launch_config(spec, stream)
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": scale}

    def call():
        launcher(vals, config=cfg)

    call()
    torch.cuda.synchronize()

    err = float("nan")
    if check:
        qh = q.transpose(1, 2).float()
        rep = Hq // Hkv
        kh = k.transpose(1, 2).repeat_interleave(rep, 1).float()
        vh = v.transpose(1, 2).repeat_interleave(rep, 1).float()
        W = spec.sliding_window
        if spec.causal and W > 0:
            qi = torch.arange(Sq, device=dev).view(-1, 1)
            ki = torch.arange(Skv, device=dev).view(1, -1)
            allowed = (ki <= qi) & (ki > qi - W)
            ref = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=allowed
            ).transpose(1, 2)
        else:
            ref = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=spec.causal
            ).transpose(1, 2)
        err = (out.float() - ref).abs().max().item()

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call()
    e.record()
    e.synchronize()
    ms = s.elapsed_time(e) / iters
    W = spec.sliding_window
    if spec.causal and W > 0:
        pairs = W * Sq - W * (W - 1) // 2 if Sq >= W else Sq * (Sq + 1) // 2
        flops = 4 * B * Hq * D * pairs
    elif spec.causal:
        flops = 4 * B * Hq * D * (Sq * (Sq + 1) // 2)
    else:
        flops = 2 * 2 * B * Hq * D * Sq * Skv
    tf = flops / (ms * 1e-3) / 1e12
    status = "PASS" if (not check or err < 2e-2) else "FAIL"
    print(
        f"[{describe_dense_spec(spec, overrides)}] {ms:.4f} ms  {tf:.1f} TFLOPS  "
        f"max_abs={err:.2e}  {status}"
    )
    return ms, tf, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--hq", type=int, default=128)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve + print the dispatch spec for every shape, build nothing",
    )
    add_dense_tuning_args(ap)
    args = ap.parse_args()

    overrides = dense_spec_overrides(args)
    for sq in (256, 512, 2048, 8192):
        req = dense_request(
            args,
            batch=args.batch,
            seqlen_q=sq,
            seqlen_kv=sq,
            num_query_heads=args.hq,
            num_kv_heads=args.hkv,
            head_size=args.d,
            causal=bool(args.causal),
            dtype=args.dtype,
        )
        spec = resolve_dense_spec(req, overrides)
        if args.dry_run:
            print(
                f"Sq={sq:<6d} {describe_dense_spec(spec, overrides)}  "
                f"(persistent={spec.persistent} np={spec.num_persistent} "
                f"decode={spec.persist_decode} wpe={spec.waves_per_eu} "
                f"bn={spec.block_n} kpad={spec.lds_k_group_pad})"
            )
            continue
        run(spec, overrides=overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
