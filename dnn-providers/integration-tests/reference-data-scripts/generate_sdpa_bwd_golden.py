#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
SDPA Backward Golden Reference Bundle Generator

Generates pre-computed reference data for SDPA backward (`SdpaBwdNode`)
validation, with PyTorch as the independent reference source (RFC 0011).

The CPU and GPU backward references are checked mainly against each other, so a
shared misunderstanding passes those gate tests. Golden data from an independent
source catches it: `TestCpuReferenceUsingGoldenValues` and
`TestGpuReferenceUsingGoldenValues` run both references against these bundles.

One pipeline run, two bundles
-----------------------------
RFC 0011 §4.2 ("Forward-Backward Generation Constraint") requires that the saved
tensors a backward bundle consumes are the actual outputs of a mathematically
consistent forward pass, not arbitrary values. So a single invocation:

  1. draws Q, K, V and dO from one seeded RNG stream;
  2. runs the forward reference -> O, LSE;
  3. writes the forward bundle (inputs Q/K/V, outputs O/LSE);
  4. runs the backward reference on *those* O and LSE -> dQ, dK, dV;
  5. writes the backward bundle, recording `forward_source` in its metadata.

At test time each bundle is standalone; the coupling exists only here.

Reference strategy
------------------
`matched_precision` (RFC 0011): BF16/FP16 inputs with FP32 intermediates and FP32
gradient accumulation, converted to the IO type once at the end. This mirrors the
DUT's a32 accumulation contract. A higher-precision (e.g. FP64) reference would
diverge from the DUT's rounding trajectory at every stage and force artificially
loose tolerances that hide real regressions.

Kernels that round intermediates such as P and dS to the IO dtype before their
matrix products are not modelled yet; the metadata records
`intermediate_rounding: "none"` so bundles generated before and after such a mode
is added are distinguishable without diffing tensors.

Supported operation axes (deliberately narrow -- see "Not supported" below)
--------------------------------------------------------------------------
  - dtype:        bf16, fp16
  - layout:       bhsd (contiguous [B, H, S, D])
  - head dim:     D_qk and D_v may differ (e.g. hd192 Q/K with hd128 V)
  - masking:      diagonal band via --left-bound / --right-bound /
                  --diagonal-alignment, matching the graph's non-deprecated
                  bound attributes (a causal mask is --right-bound 0)
  - GQA / MQA:    H_q a multiple of H_kv; dK and dV sum over each query group
  - padding:      --seq-lens-q / --seq-lens-kv (padding_mask). Rows past a
                  batch's query length, and rows the band leaves empty, are
                  fully masked: O = 0, LSE = -inf, zero gradients (cuDNN)
  - bias + dBias: --additive-mask with --bias-dims (any broadcast of
                  [B, H_q, S_q, S_kv]) and --dbias (the unscaled dS reduced to
                  the bias shape, as cuDNN)
  - dropout:      --dropout-p draws an explicit keep mask; dropout_scale =
                  1 / (1 - p) and dropout_scale_inv = 1 - p, where cuDNN's
                  composition is the exact gradient

The padding, GQA, dBias and dropout features are always cross-checked against an
independent FP64 autograd evaluation of the forward definition. They run as
single-graph bundles, or as sweep cases of a template that expresses them (the
generator checks that the template expands to the graph it emits).

  - ragged:       --ragged-sides {q,kv,qkv} with --seq-lens-q/--seq-lens-kv (zero
                  lengths allowed): cuDNN ragged operands, each with its own INT32
                  offset tensor and ragged_offset_multiplier. A batch starts at its
                  offset (its origin) and holds seq_len rows; the file holds exactly
                  the elements the sequences address (elementSpace, RFC 0014).
                  --ragged-blocks sets a role's per-batch block rows (gaps past the
                  lengths), --ragged-origins places the batches at explicit origins,
                  --ragged-units tokens binds token starts with multiplier = the
                  tensor's sequence stride, --ragged-token-stride pads the token
                  stride, --ragged-head-major lays a role out [H, T] with head pitch
                  --ragged-head-pitch, --stats-layout picks packed [T, H] or [H, T]
                  Stats. Input gaps hold finite poison, output gaps NaN; gap rows of a
                  ragged output are not iterated (the reference and the comparison
                  visit [origin, origin + length) per batch); the storage written
                  is read back and cross-checked against FP64 autograd and the LSE.
  - graph-only:   --graph-only registers a sweep case with no golden data.
  - templates:    --write-template DIR --template-kind K writes graph.template.json.

Not supported yet, rejected with an explicit error rather than silently
mis-generated: FP8, BSHD layouts without ragged, Philox dropout, sinks and ALiBi,
ragged bias, dBias or dropout.

Output:
  backward bundle: {base_filename}.json + .tensor{uid}.bin + .meta.json
  forward bundle:  {forward_base_filename}.json + .tensor{uid}.bin + .meta.json

Usage:
    python generate_sdpa_bwd_golden.py \
        --base-filename out/SdpaBackward/Small/Small \
        --forward-base-filename out/SdpaFwd/Small/Small \
        --q-dims 1 1 256 128 --v-dims 1 1 256 128 \
        --attn-scale 0.08838834764831843 --seed 42

    python generate_sdpa_bwd_golden.py \
        --base-filename out/SdpaBackward/SmallCausal/SmallCausal \
        --forward-base-filename out/SdpaFwd/SmallCausal/SmallCausal \
        --q-dims 1 1 256 128 --v-dims 1 1 256 128 \
        --right-bound 0 --diagonal-alignment bottom_right --seed 42
"""

import argparse
import datetime
import json
import os
import re
import sys

# Find files in utilities folder without requiring a package to be created
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + "/utilities/")
import torch
from sdpa_common import (  # noqa: E402  (needs the sys.path entry above)
    DTYPE_MAP,
    build_bound_mask,
    combined_attn_mask,
    compute_contiguous_strides,
    compute_forward,
    compute_lse,
    compute_strides,
    save_tensor_bin,
    script_sha256,
)

# Bump when generator logic changes in a way that affects emitted output: the
# tensors (different reference semantics, precision handling, RNG draw order) or
# the .meta.json schema, since consumers key off both.
# 1.0.0 — Initial backward generator. One pipeline run emits the paired forward
#         bundle (Q/K/V -> O/LSE) and the backward bundle
#         (Q/K/V/O/LSE/dO -> dQ/dK/dV) with forward_source provenance.
#         RNG draw order is Q, K, V, dO from a single seeded generator; changing
#         that order changes every tensor, so it is part of the reproducibility
#         contract.
# 1.1.0 — Added --additive-mask (an explicit attention-mask tensor, added to the
#         scores before the diagonal band) and template-sweep output
#         (--sweep-bundle-dir/--case-id). Bundles without --additive-mask are
#         byte-identical to 1.0.0; the mask is drawn after dO, so it extends the
#         RNG draw order rather than disturbing it.
# 1.2.0 — --left-bound is cuDNN's left_bound: it keeps that many keys including the
#         diagonal (SdpaSemanticsRegister SEM-001), so it must be >= 1. Windowed
#         bundles from earlier versions keep one more key and must be regenerated.
# 1.3.0 — GQA, padding (--seq-lens-q/--seq-lens-kv), broadcast bias with dBias
#         (--bias-dims/--dbias) and explicit-mask dropout (--dropout-p), each
#         checked against FP64 autograd. Bundles without these flags are
#         byte-identical to 1.2.0.
# 1.4.0 — Ragged operands (--ragged-sides, --ragged-blocks, --ragged-units,
#         --stats-layout), zero sequence lengths, graph-only sweep cases
#         (--graph-only) and sweep-template writing (--write-template); each ragged
#         bundle's stored tensors are read back and cross-checked against FP64
#         autograd. Bundles without these flags are byte-identical to 1.3.0.
# 1.5.0 — A ragged tensor is a per-batch origin plus a length and ordinary strides;
#         its file holds exactly elementSpace elements (a trailing gap is not
#         written) and unwritten elements hold the gap value. Adds --ragged-origins,
#         --ragged-head-major, --ragged-head-pitch, --ragged-token-stride and
#         --ragged-sides kv; [H, T] Stats may have H > 1; a block may be longer than
#         S_max. Ragged bundles from 1.4.0 differ in payload size and gap values.
GENERATOR_VERSION = "1.5.0"

# Backward tensor UID assignment. These match the checked-in graph-only
# SdpaBackward template topology (integration-test-bundles/*/SdpaBackward/), so
# bundles generated here can be adopted as golden data for those cases without
# renumbering.
UID_K = 0
UID_O = 1
UID_Q = 2
UID_DK = 3
UID_DQ = 4
UID_DV = 5
UID_V = 6
UID_DO = 7
UID_STATS = 8
# The additive attention-mask tensor, present only with --additive-mask. It
# takes the first free UID so the mask-free topology keeps the UIDs above.
UID_ATTN_MASK = 9
# The same tensor in the paired forward bundle, which uses the forward
# generator's UID map. The forward generator assigns 0-11 and 12-19 (fp8 scaling
# and Max/Sum_exp), but the paired forward graph carries none of 12-19, so the
# mask and dropout tensors below reuse 12-14. Pick other UIDs before pairing a
# forward graph that uses those features.
FWD_UID_ATTN_MASK = 12

# Padding, dropout and dBias tensors, present only when their flags are given.
# Backward: the first UIDs free after UID_ATTN_MASK.
UID_SEQ_LEN_Q = 10
UID_SEQ_LEN_KV = 11
UID_DROPOUT_MASK = 12
UID_DROPOUT_SCALE = 13
UID_DROPOUT_SCALE_INV = 14
UID_DBIAS = 15
# Forward: the sequence lengths use the forward generator's own UIDs (5, 6); the
# dropout tensors take the first UIDs free after FWD_UID_ATTN_MASK.
FWD_UID_SEQ_LEN_Q = 5
FWD_UID_SEQ_LEN_KV = 6
FWD_UID_DROPOUT_MASK = 13
FWD_UID_DROPOUT_SCALE = 14

# FP64 autograd must agree with the matched-precision backward to this relative
# deviation. The backward is anchored to the bundle's IO-dtype O, autograd to the
# exact O, so they differ by about the IO rounding of O.
AUTOGRAD_TOLERANCE = 2e-2

# Forward tensor UID assignment -- the contract generate_sdpa_fwd_golden.py uses.
FWD_UID_Q = 0
FWD_UID_K = 1
FWD_UID_V = 2
FWD_UID_O = 3
FWD_UID_LSE = 4

# The reference backend names recorded in metadata. The forward bundle's value is
# what a backward bundle's forward_source.method must match.
FORWARD_METHOD = "pytorch_math_backend"
BACKWARD_METHOD = "pytorch_explicit_fp32_backward"

REFERENCE_STRATEGY = "matched_precision"

LAYOUT = "bhsd"


def expand_kv_heads(tensor, num_heads_q):
    """Repeat each K/V head over its query group (GQA/MQA); a no-op without GQA."""
    num_heads_kv = tensor.shape[1]
    if num_heads_kv == num_heads_q:
        return tensor
    return tensor.repeat_interleave(num_heads_q // num_heads_kv, dim=1)


def sum_kv_groups(tensor, num_heads_kv):
    """Sum per-query-head gradients onto their K/V head; a no-op without GQA."""
    batch, num_heads_q, seq, dim = tensor.shape
    if num_heads_q == num_heads_kv:
        return tensor
    return tensor.view(batch, num_heads_kv, num_heads_q // num_heads_kv, seq, dim).sum(
        dim=2
    )


def reduce_to_shape(tensor, dims):
    """Sum a [B, H_q, S_q, S_kv] gradient over every axis the bias broadcasts."""
    axes = [i for i, d in enumerate(dims) if d == 1 and tensor.shape[i] != 1]
    return tensor.sum(dim=axes, keepdim=True) if axes else tensor


def build_padded_keep_mask(
    s_q, s_kv, seq_lens_q, seq_lens_kv, left_bound, right_bound, top_left
):
    """[B, 1, S_q, S_kv] keep-mask for padded variable sequence lengths.

    Batch b attends only within its first seq_lens_q[b] queries and
    seq_lens_kv[b] keys, and the band's bottom-right offset uses those live
    lengths, as the CPU and GPU references do (SdpaSemanticsRegister SEM-008).
    """
    masks = []
    for live_q, live_kv in zip(seq_lens_q, seq_lens_kv):
        keep = torch.zeros(s_q, s_kv, dtype=torch.bool)
        band = build_bound_mask(live_q, live_kv, left_bound, right_bound, top_left)
        keep[:live_q, :live_kv] = True if band is None else band
        masks.append(keep)
    return torch.stack(masks).unsqueeze(1)


def compute_forward_explicit(
    Q, K, V, scale, keep_mask=None, attn_bias=None, dropout_mask=None, dropout_scale=1.0
):
    """FP32 forward for what PyTorch's SDPA cannot express.

    Fully-masked rows give O = 0 and LSE = -inf (cuDNN), and an explicit dropout
    mask gives O = (P * mask * dropout_scale) V with the LSE taken before
    dropout. Returns (O in FP32, LSE in FP32 [B, H_q, S_q, 1]).
    """
    num_heads_q = Q.shape[1]
    Q_f = Q.float()
    K_f = expand_kv_heads(K.float(), num_heads_q)
    V_f = expand_kv_heads(V.float(), num_heads_q)
    scores = torch.matmul(Q_f, K_f.transpose(-2, -1)) * scale
    if attn_bias is not None:
        scores = scores + attn_bias.float()
    if keep_mask is not None:
        scores = scores.masked_fill(~keep_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    # A fully-masked row computes exp(-inf - -inf) = NaN; its probabilities are 0.
    probs = torch.nan_to_num(torch.exp(scores - lse), nan=0.0)
    if dropout_mask is not None:
        probs = probs * dropout_mask * dropout_scale
    return torch.matmul(probs, V_f), lse


def autograd_fp64_gradients(
    Q,
    K,
    V,
    dO,
    scale,
    keep_mask=None,
    attn_bias=None,
    dropout_mask=None,
    dropout_scale=1.0,
):
    """The gradients of the forward definition, by PyTorch autograd in FP64.

    Shares no code with compute_backward: the softmax goes through
    torch.softmax, GQA through repeat_interleave (autograd sums each group) and
    a broadcast bias through broadcasting (autograd reduces dBias). Fully-masked
    rows are multiplied out, so their gradients are exactly zero.
    Returns (dQ, dK, dV, dBias or None) in FP64.
    """
    num_heads_q = Q.shape[1]
    Q_d = Q.double().requires_grad_(True)
    K_d = K.double().requires_grad_(True)
    V_d = V.double().requires_grad_(True)
    bias_d = None if attn_bias is None else attn_bias.double().requires_grad_(True)
    scores = (
        torch.matmul(Q_d, expand_kv_heads(K_d, num_heads_q).transpose(-2, -1)) * scale
    )
    if bias_d is not None:
        scores = scores + bias_d
    if keep_mask is not None:
        alive = keep_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~keep_mask, float("-inf"))
        scores = torch.where(alive, scores, torch.zeros_like(scores))
        probs = torch.softmax(scores, dim=-1) * alive
    else:
        probs = torch.softmax(scores, dim=-1)
    if dropout_mask is not None:
        probs = probs * dropout_mask.double() * dropout_scale
    O = torch.matmul(probs, expand_kv_heads(V_d, num_heads_q))
    O.backward(dO.double())
    return Q_d.grad, K_d.grad, V_d.grad, None if bias_d is None else bias_d.grad


def worst_relative_deviation(pairs):
    """max |a - b| / max |b| over (name, a, b) pairs, printing each."""
    worst = 0.0
    for name, ours, reference in pairs:
        reference = reference.double()
        diff = (ours.double() - reference).abs()
        ref_max = reference.abs().max().item()
        max_abs = diff.max().item()
        worst = max(worst, max_abs / ref_max if ref_max > 0 else max_abs)
        print(f"  fp64 autograd {name}: max_abs={max_abs:.3e} ref_max={ref_max:.3e}")
    return worst


def compute_backward(
    Q,
    K,
    V,
    O,
    lse,
    dO,
    scale,
    keep_mask=None,
    attn_bias=None,
    dropout_mask=None,
    dropout_scale=1.0,
    dropout_scale_inv=1.0,
    dbias_dims=None,
):
    """SDPA backward reference: dQ, dK, dV from the saved forward O and LSE.

    Matched-precision (RFC 0011): inputs arrive in the bundle's IO dtype, every
    intermediate and accumulation is FP32, and the caller converts the returned
    FP32 gradients to the IO dtype exactly once.

    Softmax probabilities are recovered from the forward LSE rather than
    re-softmaxed, so the backward is anchored to the saved forward statistics the
    way a real flash-attention kernel is:

        P  = exp(scale * Q @ K^T + bias + mask - LSE)
        D  = rowsum(dO * O)
        dP = dO @ V^T
        dS = P * (dP - D)
        dQ = (dS * scale) @ K
        dK = (dS * scale)^T @ Q
        dV = P^T @ dO

    O and dO must be the tensors that are written to the bundles (i.e. already
    rounded to the IO dtype), not higher-precision pre-conversion values --
    otherwise the golden gradients would not be the gradients of the bundle's
    own inputs.

    GQA: K and V are repeated over each query group and dK, dV are summed back.
    Dropout follows cuDNN's backward: dP is masked, D is scaled by
    dropout_scale_inv, dS by dropout_scale, and dV uses P * mask * dropout_scale.
    dBias is the unscaled dS reduced to dbias_dims.
    Returns (dQ, dK, dV, dBias or None) in FP32.
    """
    num_heads_q = Q.shape[1]
    num_heads_kv = K.shape[1]
    Q_f = Q.float()
    K_f = expand_kv_heads(K.float(), num_heads_q)
    V_f = expand_kv_heads(V.float(), num_heads_q)
    O_f = O.float()
    dO_f = dO.float()

    scores = torch.matmul(Q_f, K_f.transpose(-2, -1)) * scale
    if attn_bias is not None:
        # The graph's attention-mask tensor is added to the scores before the
        # diagonal band is applied, as the CPU reference does.
        scores = scores + attn_bias.float()
    if keep_mask is not None:
        # Masked positions contribute nothing; set P to exactly zero rather than
        # relying on exp(-inf - LSE), which is NaN for a fully-masked row.
        probs = torch.where(
            keep_mask, torch.exp(scores - lse), torch.zeros_like(scores)
        )
    else:
        probs = torch.exp(scores - lse)

    row_sum = (dO_f * O_f).sum(dim=-1, keepdim=True)  # D[b, h, sq]
    d_probs = torch.matmul(dO_f, V_f.transpose(-2, -1))
    if dropout_mask is not None:
        d_probs = d_probs * dropout_mask
        row_sum = row_sum * dropout_scale_inv
    d_scores = probs * (d_probs - row_sum)
    if dropout_mask is not None:
        d_scores = d_scores * dropout_scale
    if keep_mask is not None:
        d_scores = torch.where(keep_mask, d_scores, torch.zeros_like(d_scores))
    d_scores_scaled = d_scores * scale

    dQ = torch.matmul(d_scores_scaled, K_f)
    dK = sum_kv_groups(
        torch.matmul(d_scores_scaled.transpose(-2, -1), Q_f), num_heads_kv
    )
    kept_probs = probs if dropout_mask is None else probs * dropout_mask * dropout_scale
    dV = sum_kv_groups(torch.matmul(kept_probs.transpose(-2, -1), dO_f), num_heads_kv)
    dBias = None if dbias_dims is None else reduce_to_shape(d_scores, dbias_dims)
    return dQ, dK, dV, dBias


def validate_against_autograd(
    Q, K, V, dO, scale, keep_mask, dQ, dK, dV, attn_bias=None
):
    """Cross-check the explicit backward against PyTorch autograd (FP32).

    Autograd differentiates its own FP32 softmax rather than reusing the forward
    LSE, so small differences are expected and are the matched-precision gap. A
    large difference means the explicit backward above is wrong.
    """
    Q_g = Q.float().detach().requires_grad_(True)
    K_g = K.float().detach().requires_grad_(True)
    V_g = V.float().detach().requires_grad_(True)
    attn_mask = combined_attn_mask(keep_mask, attn_bias)
    O_g = torch.nn.functional.scaled_dot_product_attention(
        Q_g, K_g, V_g, attn_mask=attn_mask, scale=scale
    )
    O_g.backward(dO.float())

    worst = 0.0
    for name, reference, autograd in [
        ("dQ", dQ, Q_g.grad),
        ("dK", dK, K_g.grad),
        ("dV", dV, V_g.grad),
    ]:
        diff = (reference - autograd).abs()
        scale_ref = autograd.abs().max().item()
        max_abs = diff.max().item()
        worst = max(worst, max_abs / scale_ref if scale_ref > 0 else max_abs)
        print(
            f"  autograd {name}: max_abs={max_abs:.3e} "
            f"mean_abs={diff.mean().item():.3e} ref_max={scale_ref:.3e}"
        )
    return worst


def _tensor_entry(uid, name, dims, data_type):
    return {
        "uid": uid,
        "name": name,
        "dims": dims,
        "strides": compute_strides(dims, LAYOUT),
        "data_type": data_type,
        "virtual": False,
    }


def _seq_len_entry(uid, name, batch):
    dims = [batch, 1, 1, 1]
    return {
        "uid": uid,
        "name": name,
        "dims": dims,
        "strides": compute_contiguous_strides(dims),
        "data_type": "int32",
        "virtual": False,
    }


def _float_entry(uid, name, dims):
    return {
        "uid": uid,
        "name": name,
        "dims": dims,
        "strides": compute_contiguous_strides(dims),
        "data_type": "float",
        "virtual": False,
    }


def build_backward_graph_json(
    q_dims,
    k_dims,
    v_dims,
    o_dims,
    scale,
    dtype_str,
    left_bound,
    right_bound,
    diagonal_alignment,
    attn_mask_dims=None,
    padded=False,
    dropout=False,
    dbias=False,
    ragged=None,
):
    """Graph JSON for a single SdpaBackwardAttributes node.

    Every key the JSON reader looks up is required (it uses `.at()`), so all
    optional tensor UIDs and parameters are emitted explicitly as null/false.
    See flatbuffers_sdk/schemas/sdpa_backward_attributes.fbs.
    """
    B, H_q, S_q = q_dims[0], q_dims[1], q_dims[2]
    lse_dims = [B, H_q, S_q, 1]

    tensors = [
        _tensor_entry(UID_K, "K", k_dims, dtype_str),
        _tensor_entry(UID_O, "O", o_dims, dtype_str),
        _tensor_entry(UID_Q, "Q", q_dims, dtype_str),
        _tensor_entry(UID_DK, "SdpaBwd_0::dK", k_dims, dtype_str),
        _tensor_entry(UID_DQ, "SdpaBwd_0::dQ", q_dims, dtype_str),
        _tensor_entry(UID_DV, "SdpaBwd_0::dV", v_dims, dtype_str),
        _tensor_entry(UID_V, "V", v_dims, dtype_str),
        _tensor_entry(UID_DO, "dO", o_dims, dtype_str),
        {
            "uid": UID_STATS,
            "name": "stats",
            "dims": lse_dims,
            "strides": compute_contiguous_strides(lse_dims),
            "data_type": "float",
            "virtual": False,
        },
    ]
    if attn_mask_dims is not None:
        tensors.append(
            _tensor_entry(UID_ATTN_MASK, "attn_mask", attn_mask_dims, dtype_str)
        )
    if padded:
        tensors.append(_seq_len_entry(UID_SEQ_LEN_Q, "seq_len_q", B))
        tensors.append(_seq_len_entry(UID_SEQ_LEN_KV, "seq_len_kv", B))
    if dropout:
        tensors.append(
            _float_entry(UID_DROPOUT_MASK, "dropout_mask", [B, H_q, S_q, k_dims[2]])
        )
        tensors.append(_float_entry(UID_DROPOUT_SCALE, "dropout_scale", [1, 1, 1, 1]))
        tensors.append(
            _float_entry(UID_DROPOUT_SCALE_INV, "dropout_scale_inv", [1, 1, 1, 1])
        )
    if dbias:
        tensors.append(
            _tensor_entry(UID_DBIAS, "SdpaBwd_0::dBias", attn_mask_dims, dtype_str)
        )
    if ragged:
        by_uid = {t["uid"]: t for t in tensors}
        for role, spec in ragged.items():
            tensor = by_uid[ROLE_UID[role]]
            tensor["strides"] = spec["strides"]
            tensor["ragged_offset_tensor_uid"] = spec["offset_uid"]
            tensor["ragged_offset_multiplier"] = spec["multiplier"]
            offset_dims = [B + 1, 1, 1, 1]
            tensors.append(
                {
                    "uid": spec["offset_uid"],
                    "name": f"ragged_offset_{role}",
                    "dims": offset_dims,
                    "strides": compute_contiguous_strides(offset_dims),
                    "data_type": "int32",
                    "virtual": False,
                }
            )

    return {
        "nodes": [
            {
                "type": "SdpaBackwardAttributes",
                "compute_data_type": "float",
                "name": "SdpaBwd_0",
                "inputs": {
                    "q_tensor_uid": UID_Q,
                    "k_tensor_uid": UID_K,
                    "v_tensor_uid": UID_V,
                    "o_tensor_uid": UID_O,
                    "do_tensor_uid": UID_DO,
                    "stats_tensor_uid": UID_STATS,
                    "scale_tensor_uid": None,
                    "attn_mask_tensor_uid": (
                        None if attn_mask_dims is None else UID_ATTN_MASK
                    ),
                    "seq_len_q_tensor_uid": UID_SEQ_LEN_Q if padded else None,
                    "seq_len_kv_tensor_uid": UID_SEQ_LEN_KV if padded else None,
                    "seed_tensor_uid": None,
                    "offset_tensor_uid": None,
                    "dropout_mask_tensor_uid": UID_DROPOUT_MASK if dropout else None,
                    "dropout_scale_tensor_uid": UID_DROPOUT_SCALE if dropout else None,
                    "dropout_scale_inv_tensor_uid": (
                        UID_DROPOUT_SCALE_INV if dropout else None
                    ),
                },
                "outputs": {
                    "dq_tensor_uid": UID_DQ,
                    "dk_tensor_uid": UID_DK,
                    "dv_tensor_uid": UID_DV,
                    "dbias_tensor_uid": UID_DBIAS if dbias else None,
                },
                "parameters": {
                    "alibi_mask": False,
                    "padding_mask": padded,
                    # The deprecated causal flags stay false: masking is expressed
                    # through the bound + alignment attributes.
                    "causal_mask": False,
                    "causal_mask_bottom_right": False,
                    "dropout_probability": None,
                    "attn_scale_value": scale,
                    "left_bound": left_bound,
                    "right_bound": right_bound,
                    "diagonal_alignment": diagonal_alignment,
                },
            }
        ],
        "tensors": tensors,
        "io_data_type": dtype_str,
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }


def build_forward_graph_json(
    q_dims,
    k_dims,
    v_dims,
    o_dims,
    scale,
    dtype_str,
    left_bound,
    right_bound,
    diagonal_alignment,
    attn_mask_dims=None,
    padded=False,
    dropout=False,
):
    """Graph JSON for the paired forward bundle (always with the LSE output).

    The topology is produced by generate_sdpa_fwd_golden.build_graph_json so the
    forward half of a backward run is schema-identical to a bundle the forward
    generator would emit. Only the diagonal-band attributes are restated, because
    this generator expresses masking as explicit bounds rather than as the
    forward generator's causal/window arguments.
    """
    import generate_sdpa_fwd_golden as fwd

    graph = fwd.build_graph_json(
        q_dims,
        k_dims,
        v_dims,
        o_dims,
        scale,
        dtype_str=dtype_str,
        stats=True,
        emit_seq_lens=padded,
        layout=LAYOUT,
    )
    attributes = graph["nodes"][0]["attributes"]
    attributes["left_bound"] = left_bound
    attributes["right_bound"] = right_bound
    attributes["diagonal_alignment"] = diagonal_alignment
    if attn_mask_dims is not None:
        graph["tensors"].append(
            {
                "uid": FWD_UID_ATTN_MASK,
                "name": "attn_mask",
                "dims": attn_mask_dims,
                "strides": compute_contiguous_strides(attn_mask_dims),
                "data_type": dtype_str,
                "virtual": False,
            }
        )
        graph["nodes"][0]["inputs"]["attn_mask_tensor_uid"] = FWD_UID_ATTN_MASK
    if dropout:
        B, H_q, S_q = q_dims[0], q_dims[1], q_dims[2]
        graph["tensors"].append(
            _float_entry(FWD_UID_DROPOUT_MASK, "dropout_mask", [B, H_q, S_q, k_dims[2]])
        )
        graph["tensors"].append(
            _float_entry(FWD_UID_DROPOUT_SCALE, "dropout_scale", [1, 1, 1, 1])
        )
        graph["nodes"][0]["inputs"]["dropout_mask_tensor_uid"] = FWD_UID_DROPOUT_MASK
        graph["nodes"][0]["inputs"]["dropout_scale_tensor_uid"] = FWD_UID_DROPOUT_SCALE
    return graph


def _base_meta_json(config, direction, reference_backend):
    pytorch_version = torch.__version__
    rocm_version = ""
    if "+rocm" in pytorch_version:
        rocm_version = pytorch_version.split("+rocm")[1]

    return {
        # Required by RFC 0011 §4.1. The harness rejects metadata without it, and
        # a rejected .meta.json next to golden .bin blobs is a hard load failure.
        "format_version": 1,
        "generator": "generate_sdpa_bwd_golden.py",
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": script_sha256(os.path.abspath(__file__)),
        "generated_at": config["generated_at"],
        "reference_source": f"PyTorch {pytorch_version}",
        "reference_backend": reference_backend,
        "reference_strategy": REFERENCE_STRATEGY,
        "rocm_version": rocm_version,
        "direction": direction,
        "seed": config["seed"],
        "input_range": [config["min_val"], config["max_val"]],
        "deterministic": True,
        # Rounding of intermediates (P, dS) to the IO dtype is not modelled yet;
        # bundles generated once it is will say so here.
        "intermediate_rounding": "none",
        "config": {
            "batch": config["q_dims"][0],
            "num_heads_q": config["q_dims"][1],
            "num_heads_kv": config["v_dims"][1],
            "seq_q": config["q_dims"][2],
            "seq_kv": config["v_dims"][2],
            "head_dim_qk": config["q_dims"][3],
            "head_dim_v": config["v_dims"][3],
            "dtype": config["dtype"],
            "layout": LAYOUT,
            "scale": config["scale"],
            "left_bound": config["left_bound"],
            "right_bound": config["right_bound"],
            "diagonal_alignment": config["diagonal_alignment"],
            "additive_mask": config["additive_mask"],
            **config["extensions"],
        },
    }


def build_forward_meta_json(config):
    meta = _base_meta_json(config, "forward", FORWARD_METHOD)
    meta["generation_precision"] = (
        f"like-for-like: {config['dtype'].upper()} inputs, FP32 intermediates"
    )
    meta["notes"] = (
        "Forward half of a single SDPA forward+backward generation run; its O "
        "and LSE are the inputs of the paired backward bundle."
    )
    return meta


def build_backward_meta_json(config, forward_meta):
    meta = _base_meta_json(config, "backward", BACKWARD_METHOD)
    meta["generation_precision"] = (
        f"matched-precision: {config['dtype'].upper()} inputs, FP32 intermediates "
        "and FP32 gradient accumulation, converted to the IO type once at the end"
    )
    # RFC 0011 §4.2: a backward bundle must name the forward computation that
    # produced the O and LSE it consumes, so a forward bundle regenerated without
    # its backward partner can be detected instead of silently diverging.
    meta["forward_source"] = {
        "method": forward_meta["reference_backend"],
        "version": torch.__version__,
        "generated_at": forward_meta["generated_at"],
        "bundle": config["forward_bundle"],
    }
    return meta


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def _check_finite(name, tensor):
    if torch.isnan(tensor).any():
        print(f"ERROR: {name} contains NaN", file=sys.stderr)
        sys.exit(1)
    if torch.isinf(tensor).any():
        print(f"ERROR: {name} contains Inf", file=sys.stderr)
        sys.exit(1)


def _write_bundle(base_filename, graph_json, meta_json, tensors):
    os.makedirs(os.path.dirname(base_filename) or ".", exist_ok=True)
    for name, tensor, uid in tensors:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        size_kb = os.path.getsize(bin_path) / 1024
        print(
            f"  {name} (uid={uid}): {list(tensor.shape)} {tensor.dtype} "
            f"-> {size_kb:.1f} KB"
        )
    _write_json(f"{base_filename}.json", graph_json)
    print(f"  Graph JSON: {base_filename}.json")
    _write_json(f"{base_filename}.meta.json", meta_json)
    print(f"  Meta JSON: {base_filename}.meta.json")


def build_sweep_case(case_id, graph_json, meta_json, golden_path):
    """A template-sweep case entry describing the bundle just generated.

    The per-tensor dims, strides and data types are read back out of the graph
    JSON this run produced rather than restated, so a case can never disagree
    with the tensors sitting next to it in its golden directory. The topology
    itself stays in the checked-in graph.template.json; a case only supplies the
    values its ${case.*} placeholders ask for.
    """
    node = graph_json["nodes"][0]
    parameters = node["parameters"]
    case = {
        "id": case_id,
        "values": {
            "io_data_type": graph_json["io_data_type"],
            "tensors": [_sweep_tensor_values(t) for t in graph_json["tensors"]],
            "attributes": {
                "parameters__diagonal_alignment": parameters["diagonal_alignment"],
                "parameters__left_bound": parameters["left_bound"],
                "parameters__right_bound": parameters["right_bound"],
            },
        },
        "golden": {"path": golden_path},
        "metadata": meta_json,
    }
    if golden_path is None:
        del case["golden"]
    return case


def upsert_sweep_case(sweep_path, case):
    """Add or replace one case in a sweep.json, leaving every other case alone.

    Regenerating a single case must not disturb the rest of the sweep, and the
    graph-only cases captured from the C++ suite must survive untouched.
    """
    if os.path.isfile(sweep_path):
        with open(sweep_path) as f:
            sweep = json.load(f)
    else:
        sweep = {"version": 1, "cases": []}

    cases = sweep.setdefault("cases", [])
    for i, existing in enumerate(cases):
        if existing.get("id") == case["id"]:
            cases[i] = case
            break
    else:
        cases.append(case)

    _write_json(sweep_path, sweep)
    return sweep_path


def write_sweep_golden(sweep_bundle_dir, case_id, graph_json, meta_json, tensors):
    """Write a sweep case's golden tensors and register the case.

    The harness resolves a sweep case's golden directory as the parent of its
    golden.path and looks for `tensor{uid}.bin` inside it
    (IntegrationTestBundle.hpp, resolveSweepGoldenDirectory), so the layout here
    is golden/<case id>/tensor<uid>.bin -- unqualified names, unlike the
    `<Name>.tensor<uid>.bin` a single-graph bundle uses.
    """
    golden_dir = os.path.join(sweep_bundle_dir, "golden", case_id)
    os.makedirs(golden_dir, exist_ok=True)
    for name, tensor, uid in tensors:
        bin_path = os.path.join(golden_dir, f"tensor{uid}.bin")
        save_tensor_bin(tensor, bin_path)
        size_kb = os.path.getsize(bin_path) / 1024
        print(
            f"  {name} (uid={uid}): {list(tensor.shape)} {tensor.dtype} "
            f"-> {size_kb:.1f} KB"
        )

    case = build_sweep_case(
        case_id, graph_json, meta_json, f"golden/{case_id}/tensors.dvc"
    )
    sweep_path = upsert_sweep_case(os.path.join(sweep_bundle_dir, "sweep.json"), case)
    print(f"  Sweep case '{case_id}' registered in {sweep_path}")
    return golden_dir


# --- Ragged layouts (cuDNN ragged offsets, RFC 0014) ---------------------------------
#
# A ragged tensor keeps logical dims [B, H, S_max, D] and its own INT32 [B+1, 1, 1, 1]
# offset tensor. Batch b starts at element offset[b] * ragged_offset_multiplier (its
# origin) and holds seq_len[b] token rows; offset[B] is read with the table but neither
# used nor validated (origins come from entries 0..B-1, lengths from seq_len). Strides are ordinary:
# token-major (token stride H*D, or --ragged-token-stride) or head-major (token stride
# D, head stride = head pitch rows * D). The file holds exactly the elements the
# sequences address (elementSpace): up to the last row of the furthest sequence,
# including the gaps between sequences and between head pitches, and nothing past it.
# Batch b's block is the rows it owns (>= its length); blocks place the default origins
# (cumulative) and set offset[B]. An element no sequence addresses is a gap: an output
# holds NaN there (what the harness's sentinel fill leaves) and gap rows are not
# iterated, since the reference and the comparison visit [origin, origin + length) per
# batch (RFC 0014 §4.11.4); an input holds finite poison no kernel may read.
RAGGED_ROLES_Q = ("q", "o", "do", "dq", "stats")
RAGGED_ROLES_KV = ("k", "v", "dk", "dv")
ROLE_UID = {
    "q": UID_Q,
    "k": UID_K,
    "v": UID_V,
    "o": UID_O,
    "do": UID_DO,
    "dq": UID_DQ,
    "dk": UID_DK,
    "dv": UID_DV,
    "stats": UID_STATS,
}
ROLE_OUTPUT = ("dq", "dk", "dv")
# Each ragged tensor has its own offset tensor, so every role can be laid out
# independently.
UID_RAGGED_OFFSET = {
    "q": 16,
    "k": 17,
    "v": 18,
    "o": 19,
    "do": 20,
    "dq": 21,
    "dk": 22,
    "dv": 23,
    "stats": 24,
}
SDPA_SEQ_AXIS = 2
INT32_MAX = 2**31 - 1


def ragged_footprint(dims, strides, origin_rows, lens):
    """(elementCount, elementSpace) of a ragged tensor whose batch b holds the token
    rows [origin_rows[b], origin_rows[b] + lens[b]) (RFC 0014 §4.5). Raises
    ValueError naming the violated rule: negative stride, overlapping rows, or
    aliasing axes (sorted-stride rule on the dense view [maxRow+1, H, D]; the batch
    axis is excluded and zero-stride axes are exempt)."""
    _, heads, _, depth = dims
    stride_s, stride_h, stride_d = strides[SDPA_SEQ_AXIS], strides[1], strides[3]
    if stride_s <= 0 or stride_h < 0 or stride_d < 0:
        raise ValueError(
            "ragged strides must be non-negative with a positive sequence stride "
            f"(negative-stride rule; got {strides})"
        )
    live = sorted((o, n) for o, n in zip(origin_rows, lens) if n > 0)
    for (start, length), (next_start, _) in zip(live, live[1:]):
        if next_start < start + length:
            raise ValueError(
                f"ragged sequences overlap: rows from {next_start} start inside rows "
                f"{start}..{start + length - 1} (overlap rule)"
            )
    if not live:
        return 0, 0
    max_row = max(o + n - 1 for o, n in live)
    span = 1
    axes = sorted(
        (s, e)
        for e, s in ((max_row + 1, stride_s), (heads, stride_h), (depth, stride_d))
        if e > 1 and s > 0
    )
    for stride, extent in axes:
        if stride < span:
            raise ValueError(
                f"ragged strides {strides} alias: stride {stride} is below the span "
                f"{span} of the axes under it (sorted-stride rule)"
            )
        span += (extent - 1) * stride
    count = sum(lens) * heads * depth
    space = 1 + max_row * stride_s + (heads - 1) * stride_h + (depth - 1) * stride_d
    return count, space


def plan_ragged(
    q_dims,
    v_dims,
    seq_lens_q,
    seq_lens_kv,
    sides,
    blocks,
    units,
    stats_layout,
    origins=None,
    head_major=(),
    head_pitch=None,
    token_stride=None,
):
    """Layout of every ragged tensor: role -> dict(dims, strides, blocks, offsets,
    multiplier, offset_uid, origin_rows, count, space). Raises ValueError for a layout
    the harness's ragged tensor cannot hold.

    sides: "qkv" (every operand ragged), "q" (Q, O, dO, dQ and Stats ragged; K, V, dK
    and dV dense) or "kv" (K, V, dK and dV ragged; the rest dense). blocks maps a role
    to its per-batch block rows; a role without an entry uses its side's sequence
    lengths (compact). origins maps a role to its per-batch origins, in elements or
    tokens as `units` says; a role without an entry places its blocks one after the
    other. units: "elements" (multiplier 1) or "tokens" (offsets count tokens,
    multiplier = the tensor's sequence stride). stats_layout: "th" (token-major,
    stride [S*H, 1, H, 1]) or "ht" (head-major [H, T]). head_major lists further
    head-major roles: head stride = head pitch rows * D, token stride D. A head-major
    role's pitch is head_pitch[role] rows, else maxRow+1 with origins, else the sum of
    its blocks. token_stride maps a token-major role to its token stride in elements
    (>= H*D).
    """
    origins = origins or {}
    head_pitch = head_pitch or {}
    token_stride = token_stride or {}
    B, H_q, S_q, D_qk = q_dims
    _, H_kv, S_kv, D_v = v_dims
    if sides not in ("q", "kv", "qkv"):
        raise ValueError(f"--ragged-sides must be q, kv or qkv (got {sides!r})")
    if units not in ("elements", "tokens"):
        raise ValueError(f"--ragged-units must be elements or tokens (got {units!r})")
    if stats_layout not in ("th", "ht"):
        raise ValueError(f"--stats-layout must be th or ht (got {stats_layout!r})")
    shapes = {
        "q": [B, H_q, S_q, D_qk],
        "dq": [B, H_q, S_q, D_qk],
        "o": [B, H_q, S_q, D_v],
        "do": [B, H_q, S_q, D_v],
        "stats": [B, H_q, S_q, 1],
        "k": [B, H_kv, S_kv, D_qk],
        "dk": [B, H_kv, S_kv, D_qk],
        "v": [B, H_kv, S_kv, D_v],
        "dv": [B, H_kv, S_kv, D_v],
    }
    roles = (RAGGED_ROLES_Q if sides in ("q", "qkv") else ()) + (
        RAGGED_ROLES_KV if sides in ("kv", "qkv") else ()
    )
    for flag, named in (
        ("--ragged-blocks", blocks),
        ("--ragged-origins", origins),
        ("--ragged-head-major", head_major),
        ("--ragged-head-pitch", head_pitch),
        ("--ragged-token-stride", token_stride),
    ):
        unknown = sorted(set(named) - set(roles))
        if unknown:
            raise ValueError(f"{flag} names roles that are not ragged: {unknown}")
    if stats_layout == "ht" and "stats" not in roles:
        raise ValueError(
            "--stats-layout ht needs Stats to be ragged (--ragged-sides q or qkv)"
        )
    major = set(head_major)
    if stats_layout == "ht":
        major.add("stats")
    for role in sorted(set(head_pitch) - major):
        raise ValueError(f"--ragged-head-pitch names {role}, which is not head-major")
    for role in sorted(major & set(token_stride)):
        raise ValueError(
            f"--ragged-token-stride names {role}, which is head-major (token stride D)"
        )
    plan = {}
    for role in roles:
        dims = shapes[role]
        _, heads, s_max, depth = dims
        lens = seq_lens_q if role in RAGGED_ROLES_Q else seq_lens_kv
        rows = list(blocks.get(role, lens))
        if len(rows) != B:
            raise ValueError(f"role {role}: need one block size per batch ({B})")
        if any(r < n for r, n in zip(rows, lens)):
            raise ValueError(
                f"role {role}: block rows {rows} must be at least seq_len {lens} "
                "per batch"
            )
        token = depth if role in major else token_stride.get(role, heads * depth)
        if token < heads * depth and role not in major:
            raise ValueError(
                f"role {role}: token stride {token} must be >= H*D = {heads * depth}"
            )
        if role in origins:
            given = origins[role]
            if len(given) != B or any(o < 0 for o in given):
                raise ValueError(
                    f"role {role}: need one origin >= 0 per batch ({B}), got {given}"
                )
            if units == "elements" and any(o % token for o in given):
                raise ValueError(
                    f"role {role}: origins {given} must be whole tokens "
                    f"(multiples of the token stride {token}; whole-token origin rule)"
                )
            origin_rows = [o // token if units == "elements" else o for o in given]
        else:
            origin_rows = [sum(rows[:b]) for b in range(B)]
        max_row = max(
            (o + n - 1 for o, n in zip(origin_rows, lens) if n > 0), default=-1
        )
        if role in major:
            least = max(max_row + 1, 1)
            pitch = head_pitch.get(
                role, least if role in origins else max(sum(rows), 1)
            )
            if pitch < least:
                raise ValueError(
                    f"role {role}: head pitch {pitch} rows is below maxRow+1 = {least}"
                )
            strides = [INT32_MAX, pitch * depth, depth, 1]
        else:
            strides = [s_max * token, depth, token, 1]
        count, space = ragged_footprint(dims, strides, origin_rows, lens)
        tokens = origin_rows + [
            max((o + r for o, r in zip(origin_rows, rows) if r > 0), default=0)
        ]
        if units == "tokens":
            offsets, multiplier = tokens, token
        else:
            offsets, multiplier = [t * token for t in tokens], 1
        if max(offsets) > INT32_MAX:
            raise ValueError(f"role {role}: offsets overflow INT32")
        plan[role] = {
            "dims": dims,
            "strides": strides,
            "blocks": rows,
            "offsets": offsets,
            "multiplier": multiplier,
            "offset_uid": UID_RAGGED_OFFSET[role],
            "origin_rows": origin_rows,
            "count": count,
            "space": space,
        }
    return plan


def pack_ragged_role(dense, seq_lens, spec, pad):
    """Physical storage of a dense [B, H, S, D] tensor: exactly spec["space"] elements,
    batch b's first seq_len rows at its origin by the planned strides. Every other
    element is NaN (pad None) or finite poison cycled from `pad`."""
    space = spec["space"]
    if pad is None:
        storage = torch.full((space,), float("nan"), dtype=dense.dtype)
    else:
        flat = pad.reshape(-1)
        storage = flat.repeat(-(-space // flat.numel()))[:space]
    _, heads, _, depth = dense.shape
    _, stride_h, stride_s, stride_d = spec["strides"]
    for b, length in enumerate(seq_lens):
        if length > 0:
            window = storage.as_strided(
                (length, heads, depth),
                (stride_s, stride_h, stride_d),
                spec["origin_rows"][b] * stride_s,
            )
            window.copy_(dense[b, :, :length, :].transpose(0, 1))
    return storage


def ragged_offset_tensor(offsets):
    """INT32 [B+1, 1, 1, 1] offset table."""
    return torch.tensor(offsets, dtype=torch.int32).view(len(offsets), 1, 1, 1)


def gather_ragged(flat, entry, offsets, seq_lens):
    """Dense [B, H, S, D] (zeros outside the sequences) read from ragged storage by the
    tensor's declared strides, offsets and multiplier, with no code shared with
    pack_ragged_role. Returns (dense, covered) where covered marks the storage
    elements the sequences occupy; raises if two sequences share an element."""
    dims, strides = entry["dims"], entry["strides"]
    multiplier = entry["ragged_offset_multiplier"]
    storage = flat.reshape(-1)
    dense = torch.zeros(dims, dtype=flat.dtype)
    covered = torch.zeros(storage.numel(), dtype=torch.bool)
    _, heads, _, depth = dims
    for b, length in enumerate(seq_lens):
        if length == 0:
            continue
        base = int(offsets[b]) * multiplier
        s = torch.arange(length).view(-1, 1, 1)
        h = torch.arange(heads).view(1, -1, 1)
        d = torch.arange(depth).view(1, 1, -1)
        index = base + s * strides[2] + h * strides[1] + d * strides[3]
        if bool(covered[index].any()):
            raise ValueError(f"tensor {entry['uid']}: batch {b} aliases another batch")
        covered[index] = True
        dense[b, :, :length, :] = storage[index].permute(1, 0, 2)
    return dense, covered


def check_ragged_storage(
    graph_json, stored, plan, seq_lens_q, seq_lens_kv, scale, keep_mask, lse
):
    """Read every tensor back from the storage written to the bundle and check what
    the harness will see: sequences do not alias, each file holds exactly the elements
    its sequences address, input gaps are finite, output gaps are NaN, dense gradients
    are zero past their lengths, the Stats read back equal the LSE, and the gradients
    stored for the sequences agree with FP64 autograd of the stored Q, K, V and dO.
    Returns the worst relative deviation from autograd."""
    entries = {t["uid"]: t for t in graph_json["tensors"]}
    gathered = {}
    for role, uid in ROLE_UID.items():
        lens = seq_lens_q if role in RAGGED_ROLES_Q else seq_lens_kv
        if role in plan:
            spec = plan[role]
            dense, covered = gather_ragged(
                stored[uid], entries[uid], stored[spec["offset_uid"]], lens
            )
            addressed = int(covered.nonzero().max()) + 1 if bool(covered.any()) else 0
            if addressed != covered.numel():
                raise ValueError(
                    f"{role}: the file holds {covered.numel()} elements but the "
                    f"sequences address {addressed} (file-size vs elementSpace)"
                )
            gap = stored[uid].reshape(-1)[~covered]
            if role in ROLE_OUTPUT:
                if not bool(torch.isnan(gap).all()):
                    raise ValueError(f"{role}: a gap element is not NaN")
            elif not bool(torch.isfinite(gap.float()).all()):
                raise ValueError(f"{role}: a gap element is not finite")
        else:
            dense = stored[uid]
            if role in ROLE_OUTPUT:
                for b, length in enumerate(lens):
                    if bool((dense[b, :, length:, :] != 0).any()):
                        raise ValueError(f"{role}: a padded row is not zero")
        gathered[role] = dense
    for b, length in enumerate(seq_lens_q):
        if not torch.equal(gathered["stats"][b, :, :length], lse[b, :, :length]):
            raise ValueError(f"stats: batch {b} read back differs from the LSE")
    aq, ak, av, _ = autograd_fp64_gradients(
        gathered["q"], gathered["k"], gathered["v"], gathered["do"], scale, keep_mask
    )
    return worst_relative_deviation(
        [
            ("dQ", gathered["dq"], aq),
            ("dK", gathered["dk"], ak),
            ("dV", gathered["dv"], av),
        ]
    )


def _sweep_tensor_values(tensor):
    """The per-tensor values a sweep case supplies for the template placeholders."""
    values = {
        "uid": tensor["uid"],
        "dims": tensor["dims"],
        "strides": tensor["strides"],
        "data_type": tensor["data_type"],
    }
    if tensor.get("ragged_offset_tensor_uid") is not None:
        values["ragged_offset_multiplier"] = tensor["ragged_offset_multiplier"]
    return values


RAGGED_TEMPLATE_SIDES = {"ragged_qkv": "qkv", "ragged_q": "q", "ragged_kv": "kv"}

TEMPLATE_KINDS = {
    "dense": "SdpaBackwardTest",
    "dense_dbias": "SdpaBackwardBroadcastDBiasTest",
    "ragged_qkv": "SdpaBackwardRaggedTest",
    "ragged_q": "SdpaBackwardRaggedQTest",
    "ragged_kv": "SdpaBackwardRaggedKvTest",
}


def build_sweep_template(kind, scale):
    """graph.template.json for a sweep of one topology: the graph the generator emits
    for `kind`, with dims, strides, data types, bounds and the ragged multipliers left
    to the cases."""
    dims = [2, 2, 8, 8]
    mask_dims = [1, 1, 8, 8] if kind == "dense_dbias" else None
    ragged = None
    if kind.startswith("ragged"):
        ragged = plan_ragged(
            dims,
            dims,
            [8, 8],
            [8, 8],
            RAGGED_TEMPLATE_SIDES[kind],
            {},
            "elements",
            "th",
        )
    graph = build_backward_graph_json(
        dims,
        dims,
        dims,
        dims,
        scale,
        "bfloat16",
        None,
        None,
        "TOP_LEFT",
        mask_dims,
        padded=ragged is not None,
        dbias=kind == "dense_dbias",
        ragged=ragged,
    )
    graph["io_data_type"] = "${case.io_data_type}"
    graph["name"] = TEMPLATE_KINDS[kind]
    graph["is_override_shape_enabled"] = False
    graph["min_required_engine_api_version"] = {"major": 1, "minor": 0, "patch": 0}
    parameters = graph["nodes"][0]["parameters"]
    parameters["diagonal_alignment"] = (
        "${case.attributes.parameters__diagonal_alignment}"
    )
    parameters["left_bound"] = "${case.attributes.parameters__left_bound}"
    parameters["right_bound"] = "${case.attributes.parameters__right_bound}"
    for tensor in graph["tensors"]:
        tensor["alignment"] = 16
        tensor["is_runtime_pass_by_value"] = False
        for field in ("dims", "strides", "data_type"):
            tensor[field] = f"${{case.{field}}}"
        if tensor.get("ragged_offset_tensor_uid") is not None:
            tensor["ragged_offset_multiplier"] = "${case.ragged_offset_multiplier}"
    return graph


def write_sweep_template(bundle_dir, kind, scale):
    os.makedirs(bundle_dir, exist_ok=True)
    path = os.path.join(bundle_dir, "graph.template.json")
    with open(path, "w") as f:
        json.dump(build_sweep_template(kind, scale), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _lookup(value, path):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def expand_sweep_template(template, case_values):
    """The graph the harness builds from a template and a case's values: `${case.x}`
    resolves from the enclosing tensor's own values first, then from the case."""
    tensors = {t["uid"]: t for t in case_values.get("tensors", [])}

    def walk(node, uid):
        if isinstance(node, str) and node.startswith("${case.") and node.endswith("}"):
            path = node[len("${case.") : -1]
            found = _lookup(tensors.get(uid), path) if uid is not None else None
            return found if found is not None else _lookup(case_values, path)
        if isinstance(node, list):
            return [walk(item, uid) for item in node]
        if isinstance(node, dict):
            inner = node["uid"] if isinstance(node.get("uid"), int) else uid
            return {key: walk(value, inner) for key, value in node.items()}
        return node

    return walk(template, None)


def sweep_graph_mismatches(expanded, generated):
    """Differences between the template's expansion and the generator's own graph, in
    everything the engines read: the node and every tensor's shape, type and ragged
    binding."""
    problems = []
    for section in ("inputs", "outputs", "parameters"):
        if expanded["nodes"][0][section] != generated["nodes"][0][section]:
            problems.append(f"node {section} differ")
    fields = (
        "dims",
        "strides",
        "data_type",
        "ragged_offset_tensor_uid",
        "ragged_offset_multiplier",
    )

    def project(graph):
        return {
            t["uid"]: {f: t.get(f) for f in fields if t.get(f) is not None}
            for t in graph["tensors"]
        }

    if project(expanded) != project(generated):
        problems.append("tensors differ")
    if expanded["io_data_type"] != generated["io_data_type"]:
        problems.append("io_data_type differs")
    return problems


def ensure_template_expresses(sweep_bundle_dir, case_id, graph_json):
    """Exit unless the sweep's graph.template.json expands, for this case's values, to
    the graph the generator emitted."""
    template_path = os.path.join(sweep_bundle_dir, "graph.template.json")
    if not os.path.isfile(template_path):
        print(f"ERROR: no graph.template.json in {sweep_bundle_dir}", file=sys.stderr)
        sys.exit(1)
    with open(template_path) as f:
        template = json.load(f)
    values = build_sweep_case(case_id, graph_json, {}, "")["values"]
    problems = sweep_graph_mismatches(
        expand_sweep_template(template, values), graph_json
    )
    if problems:
        print(
            f"ERROR: {template_path} does not express this case: {problems}",
            file=sys.stderr,
        )
        sys.exit(1)


def write_graph_only_case(sweep_bundle_dir, case_id, graph_json, config):
    """Register a sweep case with no golden data: the harness fills the inputs and
    checks the engine against its reference executor."""
    meta = {
        "format_version": 1,
        "generator": "generate_sdpa_bwd_golden.py",
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": script_sha256(os.path.abspath(__file__)),
        "generated_at": config["generated_at"],
        "reference_source": "hipDNN reference executor (graph-only bundle, no golden data)",
        "notes": "Graph-only case: no golden data; inputs are filled by the harness.",
        "config": config["summary"],
    }
    ensure_template_expresses(sweep_bundle_dir, case_id, graph_json)
    case = build_sweep_case(case_id, graph_json, meta, None)
    sweep_path = upsert_sweep_case(os.path.join(sweep_bundle_dir, "sweep.json"), case)
    print(f"  Graph-only sweep case '{case_id}' registered in {sweep_path}")


def seq_len_tensor(lengths):
    """INT32 [B, 1, 1, 1] per-batch sequence lengths (cuDNN padding layout)."""
    return torch.tensor(lengths, dtype=torch.int32).view(len(lengths), 1, 1, 1)


def scalar_tensor(value):
    """A one-element FLOAT [1, 1, 1, 1] tensor (dropout scale / scale_inv)."""
    return torch.tensor([value], dtype=torch.float32).view(1, 1, 1, 1)


def generate_bundles(
    base_filename,
    forward_base_filename,
    q_dims,
    v_dims,
    dtype="bf16",
    left_bound=None,
    right_bound=None,
    diagonal_alignment="top_left",
    seed=42,
    min_val=-1.0,
    max_val=1.0,
    attn_scale=None,
    validate=False,
    additive_mask=False,
    sweep_bundle_dir=None,
    case_id=None,
    seq_lens_q=None,
    seq_lens_kv=None,
    bias_dims=None,
    dbias=False,
    dropout_p=None,
    ragged_sides=None,
    ragged_blocks=None,
    ragged_units="elements",
    stats_layout="th",
    ragged_origins=None,
    ragged_head_major=(),
    ragged_head_pitch=None,
    ragged_token_stride=None,
    graph_only=False,
):
    # Validated before anything else: the remaining arguments mean different
    # things in sweep mode and in single-graph mode.
    sweep_errors = []
    if (sweep_bundle_dir is None) != (case_id is None):
        sweep_errors.append(
            "--sweep-bundle-dir and --case-id must be given together (sweep mode) "
            "or both omitted (single-graph mode)"
        )
    if case_id is not None and not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", case_id):
        # verify_golden_bundles.py requires lowercase_snake_case case ids.
        sweep_errors.append(f"--case-id must be lowercase_snake_case (got {case_id!r})")
    if sweep_errors:
        for e in sweep_errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if sweep_bundle_dir is not None and case_id is not None:
        # In sweep mode the backward tensors live in the case's golden directory
        # instead of under a base filename; point the paired forward bundle's
        # recorded path at that directory so forward_source stays resolvable.
        base_filename = os.path.join(sweep_bundle_dir, "golden", case_id, case_id)
    elif base_filename is None:
        print(
            "ERROR: --base-filename is required unless --sweep-bundle-dir and "
            "--case-id are given",
            file=sys.stderr,
        )
        sys.exit(1)

    B, H_q, S_q, D_qk = q_dims
    B_v, H_kv, S_kv, D_v = v_dims
    k_dims = [B, H_kv, S_kv, D_qk]
    o_dims = [B, H_q, S_q, D_v]

    errors = []
    if dtype not in DTYPE_MAP:
        errors.append(
            f"--dtype must be one of {list(DTYPE_MAP.keys())} (got {dtype!r})"
        )
    for name, dims in [("q", q_dims), ("v", v_dims)]:
        for i, d in enumerate(dims):
            if d <= 0:
                errors.append(f"--{name}-dims[{i}] must be positive (got {d})")
    if B_v != B:
        errors.append(f"Batch mismatch: Q batch={B}, V batch={B_v}")
    if H_q % H_kv != 0:
        errors.append(f"H_q ({H_q}) must be a multiple of H_kv ({H_kv})")
    padded = seq_lens_q is not None or seq_lens_kv is not None
    if padded:
        if seq_lens_q is None or seq_lens_kv is None:
            errors.append("--seq-lens-q and --seq-lens-kv must be given together")
        elif len(seq_lens_q) != B or len(seq_lens_kv) != B:
            errors.append(
                f"--seq-lens-q and --seq-lens-kv need one value per batch ({B})"
            )
        elif not all(0 <= v <= S_q for v in seq_lens_q) or not all(
            0 <= v <= S_kv for v in seq_lens_kv
        ):
            errors.append("sequence lengths must be in [0, S_q] and [0, S_kv]")
    if bias_dims is not None:
        full = [B, H_q, S_q, S_kv]
        if not additive_mask:
            errors.append("--bias-dims requires --additive-mask")
        elif len(bias_dims) != 4 or any(
            d not in (1, f) for d, f in zip(bias_dims, full)
        ):
            errors.append(f"--bias-dims must broadcast to {full} (each dim 1 or full)")
    if dbias and not additive_mask:
        errors.append("--dbias requires --additive-mask")
    if dropout_p is not None and not 0.0 < dropout_p < 1.0:
        errors.append(f"--dropout-p must be in (0, 1) (got {dropout_p})")
    ragged = ragged_sides is not None
    extended = padded or ragged or H_q != H_kv or dbias or dropout_p is not None
    ragged_plan = None
    if stats_layout == "ht" and not ragged:
        errors.append(
            "--stats-layout ht needs Stats to be ragged (--ragged-sides q or qkv)"
        )
    if not ragged:
        for flag, given in (
            ("--ragged-blocks", ragged_blocks),
            ("--ragged-origins", ragged_origins),
            ("--ragged-head-major", ragged_head_major),
            ("--ragged-head-pitch", ragged_head_pitch),
            ("--ragged-token-stride", ragged_token_stride),
            ("--ragged-units", ragged_units != "elements"),
        ):
            if given:
                errors.append(f"{flag} requires --ragged-sides")
    if ragged:
        if not padded:
            errors.append("--ragged-sides requires --seq-lens-q and --seq-lens-kv")
        if additive_mask or dbias or dropout_p is not None:
            errors.append(
                "a ragged bundle carries no bias, dBias or dropout: the engines "
                "decline them beside a ragged operand"
            )
        if not errors:
            try:
                ragged_plan = plan_ragged(
                    q_dims,
                    v_dims,
                    seq_lens_q,
                    seq_lens_kv,
                    ragged_sides,
                    ragged_blocks or {},
                    ragged_units,
                    stats_layout,
                    ragged_origins,
                    ragged_head_major,
                    ragged_head_pitch,
                    ragged_token_stride,
                )
            except ValueError as error:
                errors.append(str(error))
    if graph_only:
        if sweep_bundle_dir is None:
            errors.append("--graph-only requires --sweep-bundle-dir and --case-id")
        if padded or ragged or dropout_p is not None:
            errors.append("--graph-only is for dense graphs without padding or dropout")
    if min_val >= max_val:
        errors.append(f"--min ({min_val}) must be less than --max ({max_val})")
    if left_bound is not None and left_bound < 1:
        errors.append(
            "--left-bound must be >= 1 when given (cuDNN: it counts the diagonal key)"
        )
    if right_bound is not None and right_bound < 0:
        errors.append("--right-bound must be >= 0 when given (omit it for unbounded)")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    torch_dtype = DTYPE_MAP[dtype]["torch"]
    json_dtype = DTYPE_MAP[dtype]["json"]

    if attn_scale is None:
        # An unset attn_scale_value means no scaling in hipDNN, as in cuDNN.
        attn_scale = 1.0

    if graph_only:
        mask_dims = None
        if additive_mask:
            mask_dims = [1, 1, S_q, S_kv] if bias_dims is None else list(bias_dims)
        write_graph_only_case(
            sweep_bundle_dir,
            case_id,
            build_backward_graph_json(
                q_dims,
                k_dims,
                v_dims,
                o_dims,
                attn_scale,
                json_dtype,
                left_bound,
                right_bound,
                diagonal_alignment.upper(),
                mask_dims,
                False,
                False,
                dbias,
            ),
            {
                "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "summary": {
                    "batch": B,
                    "num_heads_q": H_q,
                    "num_heads_kv": H_kv,
                    "seq_q": S_q,
                    "seq_kv": S_kv,
                    "head_dim_qk": D_qk,
                    "head_dim_v": D_v,
                    "dtype": dtype,
                    "scale": attn_scale,
                    "left_bound": left_bound,
                    "right_bound": right_bound,
                    "diagonal_alignment": diagonal_alignment,
                    "additive_mask": additive_mask,
                    "dbias": dbias,
                },
            },
        )
        return

    top_left = diagonal_alignment == "top_left"
    if padded:
        keep_mask = build_padded_keep_mask(
            S_q, S_kv, seq_lens_q, seq_lens_kv, left_bound, right_bound, top_left
        )
    else:
        keep_mask = build_bound_mask(S_q, S_kv, left_bound, right_bound, top_left)
    # Padding defines fully-masked rows (O = 0, LSE = -inf, zero gradients, as in
    # cuDNN), so they are part of what a padded bundle checks.
    if not padded and keep_mask is not None and not bool(keep_mask.any(dim=-1).all()):
        # A fully-masked query row has LSE = -inf and no defined softmax, so its
        # gradients are not a meaningful reference. Reject the configuration
        # rather than emitting zeros or NaNs.
        print(
            "ERROR: the requested diagonal band leaves at least one query row "
            "fully masked; choose bounds that keep every row attending to at "
            "least one key",
            file=sys.stderr,
        )
        sys.exit(1)
    if ragged:
        # Stats rows inside a sequence must be finite (the verifier rejects +inf and
        # the engines read them), so every live query row needs a live key.
        for b, live_q in enumerate(seq_lens_q):
            if live_q > 0 and not bool(keep_mask[b, 0, :live_q, :].any(dim=-1).all()):
                print(
                    f"ERROR: batch {b} has a live query row with no live key",
                    file=sys.stderr,
                )
                sys.exit(1)

    print(f"Generating SDPA forward+backward bundles (seed {seed})")
    print(f"  Q: {q_dims}, K: {k_dims}, V: {v_dims}, O/dO: {o_dims}")
    print(f"  dtype: {dtype}, layout: {LAYOUT}")
    print(
        f"  left_bound={left_bound}, right_bound={right_bound}, "
        f"diagonal_alignment={diagonal_alignment}"
    )
    print(f"  Scale: {attn_scale:.10f}")

    # One RNG stream, fixed draw order (Q, K, V, dO). Reproducing a bundle means
    # reproducing this order as well as the seed.
    rng = torch.Generator().manual_seed(seed)
    Q = torch.empty(q_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)
    K = torch.empty(k_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)
    V = torch.empty(v_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)
    dO = torch.empty(o_dims, dtype=torch_dtype).uniform_(
        min_val, max_val, generator=rng
    )
    attn_mask_dims = None
    attn_mask_tensor = None
    if additive_mask:
        # A per-(query, key) bias shared across batch and heads. The leading unit
        # dims exercise the reference's right-aligned broadcast while keeping the
        # payload independent of B and H.
        attn_mask_dims = [1, 1, S_q, S_kv] if bias_dims is None else list(bias_dims)
        attn_mask_tensor = torch.empty(attn_mask_dims, dtype=torch_dtype).uniform_(
            min_val, max_val, generator=rng
        )

    # Drawn last, so bundles without dropout keep their RNG draw order.
    dropout_mask = None
    dropout_scale = 1.0
    dropout_scale_inv = 1.0
    if dropout_p is not None:
        dropout_mask = (
            torch.empty([B, H_q, S_q, S_kv]).uniform_(0.0, 1.0, generator=rng)
            >= dropout_p
        ).float()
        dropout_scale = 1.0 / (1.0 - dropout_p)
        dropout_scale_inv = 1.0 - dropout_p

    # Drawn after everything else: the finite values the gap rows of the ragged O and
    # Stats inputs hold.
    poison_o = poison_stats = None
    if ragged:
        poison_o = torch.empty(o_dims, dtype=torch_dtype).uniform_(
            min_val, max_val, generator=rng
        )
        poison_stats = torch.empty([B, H_q, S_q, 1]).uniform_(
            min_val, max_val, generator=rng
        )

    # The bundles store the mask in the IO dtype, so the reference must see the
    # rounded values the executors will read back, not the FP32 draw.
    attn_bias = None if attn_mask_tensor is None else attn_mask_tensor.float()

    # Step 1: forward reference -> O, LSE.
    if padded or dropout_mask is not None:
        O_f, lse = compute_forward_explicit(
            Q, K, V, attn_scale, keep_mask, attn_bias, dropout_mask, dropout_scale
        )
        O = O_f.to(torch_dtype)
        _check_finite("O", O)
        # Fully-masked rows carry LSE = -inf by contract; nothing else may.
        alive = keep_mask.any(dim=-1, keepdim=True).expand_as(lse)
        _check_finite("LSE", lse[alive])
    else:
        O = compute_forward(Q, K, V, attn_scale, H_q, H_kv, keep_mask, attn_bias).to(
            torch_dtype
        )
        lse = compute_lse(Q, K, attn_scale, H_q, H_kv, keep_mask, attn_bias)
        _check_finite("O", O)
        _check_finite("LSE", lse)

    # Step 2: backward reference on the O and LSE just computed -- specifically on
    # the IO-dtype O that is written to the bundles, so the gradients are the
    # gradients of the bundle's own inputs.
    dQ, dK, dV, dBias = compute_backward(
        Q,
        K,
        V,
        O,
        lse,
        dO,
        attn_scale,
        keep_mask,
        attn_bias,
        dropout_mask,
        dropout_scale,
        dropout_scale_inv,
        attn_mask_dims if dbias else None,
    )
    if validate and not extended:
        worst = validate_against_autograd(
            Q, K, V, dO, attn_scale, keep_mask, dQ, dK, dV, attn_bias
        )
        print(f"  autograd cross-check worst relative deviation: {worst:.3e}")
    if extended:
        # The new features always get the independent FP64 check.
        aQ, aK, aV, aBias = autograd_fp64_gradients(
            Q, K, V, dO, attn_scale, keep_mask, attn_bias, dropout_mask, dropout_scale
        )
        pairs = [("dQ", dQ, aQ), ("dK", dK, aK), ("dV", dV, aV)]
        if dbias:
            pairs.append(("dBias", dBias, reduce_to_shape(aBias, attn_mask_dims)))
        worst = worst_relative_deviation(pairs)
        print(f"  fp64 autograd worst relative deviation: {worst:.3e}")
        if worst > AUTOGRAD_TOLERANCE:
            print(
                f"ERROR: the backward reference deviates from FP64 autograd by "
                f"{worst:.3e} (> {AUTOGRAD_TOLERANCE})",
                file=sys.stderr,
            )
            sys.exit(1)

    dQ = dQ.to(torch_dtype)
    dK = dK.to(torch_dtype)
    dV = dV.to(torch_dtype)
    for name, tensor in [("dQ", dQ), ("dK", dK), ("dV", dV)]:
        _check_finite(name, tensor)
    if dbias:
        dBias = dBias.to(torch_dtype)
        _check_finite("dBias", dBias)

    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    ragged_meta = None
    if ragged:
        ragged_meta = {
            "sides": ragged_sides,
            "units": ragged_units,
            "stats_layout": stats_layout,
            "blocks": {role: spec["blocks"] for role, spec in ragged_plan.items()},
            # Only the options a bundle uses, so earlier bundles' metadata is unchanged.
            **{
                key: value
                for key, value in [
                    ("origins", ragged_origins),
                    ("head_major", sorted(ragged_head_major)),
                    ("head_pitch", ragged_head_pitch),
                    ("token_stride", ragged_token_stride),
                ]
                if value
            },
        }
    config = {
        "q_dims": q_dims,
        "v_dims": v_dims,
        "dtype": dtype,
        "seed": seed,
        "min_val": min_val,
        "max_val": max_val,
        "scale": attn_scale,
        "left_bound": left_bound,
        "right_bound": right_bound,
        "diagonal_alignment": diagonal_alignment,
        "additive_mask": additive_mask,
        # Only the features a bundle uses, so earlier bundles' metadata is unchanged.
        "extensions": {
            k: v
            for k, v in [
                ("seq_lens_q", seq_lens_q),
                ("seq_lens_kv", seq_lens_kv),
                ("bias_dims", attn_mask_dims if bias_dims is not None else None),
                ("dbias", True if dbias else None),
                ("dropout_p", dropout_p),
                ("ragged", ragged_meta),
            ]
            if v is not None
        },
        "generated_at": generated_at,
        "forward_bundle": os.path.relpath(
            f"{forward_base_filename}.json",
            start=os.path.dirname(os.path.abspath(base_filename)),
        ),
    }

    print(f"Forward bundle: {forward_base_filename}")
    forward_meta = build_forward_meta_json(config)
    _write_bundle(
        forward_base_filename,
        build_forward_graph_json(
            q_dims,
            k_dims,
            v_dims,
            o_dims,
            attn_scale,
            json_dtype,
            left_bound,
            right_bound,
            diagonal_alignment.upper(),
            attn_mask_dims,
            padded,
            dropout_mask is not None,
        ),
        forward_meta,
        [
            ("Q", Q, FWD_UID_Q),
            ("K", K, FWD_UID_K),
            ("V", V, FWD_UID_V),
            ("O", O, FWD_UID_O),
            ("LSE", lse, FWD_UID_LSE),
        ]
        + (
            []
            if attn_mask_tensor is None
            else [("attn_mask", attn_mask_tensor, FWD_UID_ATTN_MASK)]
        )
        + (
            []
            if not padded
            else [
                ("seq_len_q", seq_len_tensor(seq_lens_q), FWD_UID_SEQ_LEN_Q),
                ("seq_len_kv", seq_len_tensor(seq_lens_kv), FWD_UID_SEQ_LEN_KV),
            ]
        )
        + (
            []
            if dropout_mask is None
            else [
                ("dropout_mask", dropout_mask, FWD_UID_DROPOUT_MASK),
                ("dropout_scale", scalar_tensor(dropout_scale), FWD_UID_DROPOUT_SCALE),
            ]
        ),
    )

    backward_graph = build_backward_graph_json(
        q_dims,
        k_dims,
        v_dims,
        o_dims,
        attn_scale,
        json_dtype,
        left_bound,
        right_bound,
        diagonal_alignment.upper(),
        attn_mask_dims,
        padded,
        dropout_mask is not None,
        dbias,
        ragged_plan,
    )
    backward_meta = build_backward_meta_json(config, forward_meta)
    backward_tensors = [
        ("K", K, UID_K),
        ("O", O, UID_O),
        ("Q", Q, UID_Q),
        ("dK", dK, UID_DK),
        ("dQ", dQ, UID_DQ),
        ("dV", dV, UID_DV),
        ("V", V, UID_V),
        ("dO", dO, UID_DO),
        ("stats", lse, UID_STATS),
    ] + (
        []
        if attn_mask_tensor is None
        else [("attn_mask", attn_mask_tensor, UID_ATTN_MASK)]
    )
    if padded:
        backward_tensors += [
            ("seq_len_q", seq_len_tensor(seq_lens_q), UID_SEQ_LEN_Q),
            ("seq_len_kv", seq_len_tensor(seq_lens_kv), UID_SEQ_LEN_KV),
        ]
    if dropout_mask is not None:
        backward_tensors += [
            ("dropout_mask", dropout_mask, UID_DROPOUT_MASK),
            ("dropout_scale", scalar_tensor(dropout_scale), UID_DROPOUT_SCALE),
            (
                "dropout_scale_inv",
                scalar_tensor(dropout_scale_inv),
                UID_DROPOUT_SCALE_INV,
            ),
        ]
    if dbias:
        backward_tensors.append(("dBias", dBias, UID_DBIAS))

    if ragged:
        dense_by_role = {
            "q": Q,
            "k": K,
            "v": V,
            "o": O,
            "do": dO,
            "dq": dQ,
            "dk": dK,
            "dv": dV,
            "stats": lse,
        }
        gap_source = {
            "q": Q,
            "k": K,
            "v": V,
            "do": dO,
            "o": poison_o,
            "stats": poison_stats,
        }
        packed = {}
        for role, spec in ragged_plan.items():
            lens = seq_lens_q if role in RAGGED_ROLES_Q else seq_lens_kv
            packed[ROLE_UID[role]] = pack_ragged_role(
                dense_by_role[role],
                lens,
                spec,
                None if role in ROLE_OUTPUT else gap_source[role],
            )
        backward_tensors = [
            (name, packed.get(uid, tensor), uid)
            for name, tensor, uid in backward_tensors
        ]
        for role, spec in ragged_plan.items():
            backward_tensors.append(
                (
                    f"ragged_offset_{role}",
                    ragged_offset_tensor(spec["offsets"]),
                    spec["offset_uid"],
                )
            )
        worst = check_ragged_storage(
            backward_graph,
            {uid: tensor for _, tensor, uid in backward_tensors},
            ragged_plan,
            seq_lens_q,
            seq_lens_kv,
            attn_scale,
            keep_mask,
            lse,
        )
        print(
            f"  ragged storage vs fp64 autograd worst relative deviation: {worst:.3e}"
        )
        if worst > AUTOGRAD_TOLERANCE:
            print(
                f"ERROR: the stored ragged gradients deviate from FP64 autograd by "
                f"{worst:.3e} (> {AUTOGRAD_TOLERANCE})",
                file=sys.stderr,
            )
            sys.exit(1)

    if sweep_bundle_dir is not None and extended:
        ensure_template_expresses(sweep_bundle_dir, case_id, backward_graph)

    if sweep_bundle_dir is None:
        print(f"Backward bundle: {base_filename}")
        _write_bundle(base_filename, backward_graph, backward_meta, backward_tensors)
    else:
        print(f"Backward sweep case: {sweep_bundle_dir} [{case_id}]")
        write_sweep_golden(
            sweep_bundle_dir, case_id, backward_graph, backward_meta, backward_tensors
        )


def role_values(parser, flag, items, count=None):
    """ROLE=N,N,.. items as {role: [int]}; `count`, when given, is the required number
    of values per role."""
    values = {}
    for item in items:
        role, _, numbers = item.partition("=")
        try:
            parsed = [int(n) for n in numbers.split(",")]
        except ValueError:
            parser.error(f"{flag} expects ROLE=N,N,.. (got {item!r})")
        if count is not None and len(parsed) != count:
            parser.error(f"{flag} expects ROLE=N (got {item!r})")
        values[role] = parsed
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired SDPA forward and backward golden reference "
        "bundles in a single pipeline run (RFC 0011)"
    )
    parser.add_argument(
        "--base-filename",
        help="Path prefix for the backward bundle's files (no extension). "
        "Required unless --sweep-bundle-dir and --case-id select sweep mode.",
    )
    parser.add_argument(
        "--forward-base-filename",
        help="Path prefix for the paired forward bundle's files (no extension). "
        "Required except with --graph-only or --write-template: RFC 0011 mandates that the forward pass whose O and LSE the "
        "backward bundle consumes is itself saved as a bundle.",
    )
    parser.add_argument(
        "--q-dims",
        nargs=4,
        type=int,
        metavar=("B", "H_Q", "S_Q", "D_QK"),
        help="Query tensor dims: batch, heads_q, seq_q, head_dim_qk",
    )
    parser.add_argument(
        "--v-dims",
        nargs=4,
        type=int,
        metavar=("B", "H_KV", "S_KV", "D_V"),
        help="Value tensor dims: batch, heads_kv, seq_kv, head_dim_v",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=list(DTYPE_MAP.keys()),
        help="Input/output dtype (default: bf16)",
    )
    parser.add_argument(
        "--left-bound",
        type=int,
        default=None,
        help="cuDNN left_bound: KV positions kept up to and including the diagonal "
        "(must be >= 1); omit for unbounded",
    )
    parser.add_argument(
        "--right-bound",
        type=int,
        default=None,
        help="KV positions kept to the right of the diagonal; omit for unbounded. "
        "A causal mask is --right-bound 0.",
    )
    parser.add_argument(
        "--diagonal-alignment",
        default="top_left",
        choices=["top_left", "bottom_right"],
        help="Where the diagonal is anchored when bounds apply (default: top_left)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min", type=float, default=-1.0, dest="min_val")
    parser.add_argument("--max", type=float, default=1.0, dest="max_val")
    parser.add_argument(
        "--attn-scale",
        type=float,
        default=None,
        help="Attention scale (default: 1.0, no scaling, as for an unset "
        "attn_scale_value)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Cross-check the backward reference against PyTorch autograd (CPU)",
    )
    parser.add_argument(
        "--additive-mask",
        action="store_true",
        help="Add an explicit additive attention-mask tensor of shape "
        "--bias-dims (default [1, 1, S_q, S_kv]) in the IO data type. It is added "
        "to the scores before the diagonal band, matching the CPU reference. "
        "dBias is produced only with --dbias.",
    )
    parser.add_argument(
        "--sweep-bundle-dir",
        help="Write the backward tensors as a template-sweep case of this bundle "
        "directory (the one holding graph.template.json and sweep.json) instead "
        "of as a standalone single-graph bundle. Requires --case-id.",
    )
    parser.add_argument(
        "--case-id",
        help="Sweep case id, lowercase_snake_case. Requires --sweep-bundle-dir.",
    )
    parser.add_argument(
        "--seq-lens-q",
        type=int,
        nargs="+",
        default=None,
        help="Per-batch query lengths (padding); requires --seq-lens-kv",
    )
    parser.add_argument(
        "--seq-lens-kv",
        type=int,
        nargs="+",
        default=None,
        help="Per-batch key/value lengths (padding); requires --seq-lens-q",
    )
    parser.add_argument(
        "--bias-dims",
        type=int,
        nargs=4,
        default=None,
        help="Shape of the --additive-mask tensor, any broadcast of "
        "[B, H_q, S_q, S_kv] (default: [1, 1, S_q, S_kv])",
    )
    parser.add_argument(
        "--dbias",
        action="store_true",
        help="Emit dBias (requires --additive-mask): the unscaled dS reduced to "
        "the bias shape",
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=None,
        help="Explicit-mask dropout with this drop probability: a FLOAT keep mask "
        "[B, H_q, S_q, S_kv], dropout_scale = 1/(1-p), dropout_scale_inv = 1-p",
    )
    parser.add_argument(
        "--ragged-sides",
        choices=["q", "kv", "qkv"],
        default=None,
        help="Make operands ragged: q = Q, O, dO, dQ and Stats (K, V, dK, dV dense); "
        "kv = K, V, dK and dV (the rest dense); qkv = every operand. Requires "
        "--seq-lens-q and --seq-lens-kv.",
    )
    parser.add_argument(
        "--ragged-blocks",
        action="append",
        default=[],
        metavar="ROLE=N,N,..",
        help="Per-batch block rows of one ragged role (q, k, v, o, do, dq, dk, dv, "
        "stats); rows past the sequence length are the gap. Default: compact. "
        "Repeatable.",
    )
    parser.add_argument(
        "--ragged-units",
        choices=["elements", "tokens"],
        default="elements",
        help="elements: offsets are element offsets, multiplier 1 (cuDNN's default). "
        "tokens: offsets are token starts and each tensor's multiplier is its "
        "sequence stride.",
    )
    parser.add_argument(
        "--stats-layout",
        choices=["th", "ht"],
        default="th",
        help="Packed Stats of a ragged Q: th = [T, H], ht = [H, T] (head-major; "
        "the same as --ragged-head-major stats)",
    )
    parser.add_argument(
        "--ragged-origins",
        action="append",
        default=[],
        metavar="ROLE=N,N,..",
        help="Per-batch origins of one ragged role, in --ragged-units, each a whole "
        "token. The batches need not be in order; the file spans to the last row "
        "any sequence addresses. Default: the blocks one after the other. "
        "Repeatable.",
    )
    parser.add_argument(
        "--ragged-head-major",
        action="append",
        default=[],
        metavar="ROLE,ROLE,..",
        help="Ragged roles laid out head-major [H, T]: token stride D, head stride "
        "= head pitch rows * D. Repeatable.",
    )
    parser.add_argument(
        "--ragged-head-pitch",
        action="append",
        default=[],
        metavar="ROLE=N",
        help="Head pitch in token rows of a head-major role, >= maxRow+1. Default: "
        "maxRow+1 with --ragged-origins, else the sum of the blocks. Repeatable.",
    )
    parser.add_argument(
        "--ragged-token-stride",
        action="append",
        default=[],
        metavar="ROLE=N",
        help="Token stride in elements of a token-major role, >= H*D (a fused or "
        "padded token). Repeatable.",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Register a sweep case with no golden data (needs --sweep-bundle-dir)",
    )
    parser.add_argument(
        "--write-template",
        metavar="DIR",
        help="Write DIR/graph.template.json for --template-kind and --attn-scale, "
        "then exit",
    )
    parser.add_argument(
        "--template-kind",
        choices=sorted(TEMPLATE_KINDS),
        help="Topology for --write-template",
    )
    args = parser.parse_args()

    if args.write_template is not None:
        if args.template_kind is None or args.attn_scale is None:
            parser.error("--write-template needs --template-kind and --attn-scale")
        print(
            write_sweep_template(
                args.write_template, args.template_kind, args.attn_scale
            )
        )
        return
    for flag, value in (("--q-dims", args.q_dims), ("--v-dims", args.v_dims)):
        if value is None:
            parser.error(f"{flag} is required")
    if args.forward_base_filename is None and not args.graph_only:
        parser.error("--forward-base-filename is required")
    ragged_blocks = role_values(parser, "--ragged-blocks", args.ragged_blocks)
    ragged_origins = role_values(parser, "--ragged-origins", args.ragged_origins)
    ragged_head_pitch = {
        role: values[0]
        for role, values in role_values(
            parser, "--ragged-head-pitch", args.ragged_head_pitch, count=1
        ).items()
    }
    ragged_token_stride = {
        role: values[0]
        for role, values in role_values(
            parser, "--ragged-token-stride", args.ragged_token_stride, count=1
        ).items()
    }
    ragged_head_major = [
        role for item in args.ragged_head_major for role in item.split(",")
    ]

    generate_bundles(
        base_filename=args.base_filename,
        forward_base_filename=args.forward_base_filename,
        q_dims=args.q_dims,
        v_dims=args.v_dims,
        dtype=args.dtype,
        left_bound=args.left_bound,
        right_bound=args.right_bound,
        diagonal_alignment=args.diagonal_alignment,
        seed=args.seed,
        min_val=args.min_val,
        max_val=args.max_val,
        attn_scale=args.attn_scale,
        validate=args.validate,
        additive_mask=args.additive_mask,
        sweep_bundle_dir=args.sweep_bundle_dir,
        case_id=args.case_id,
        seq_lens_q=args.seq_lens_q,
        seq_lens_kv=args.seq_lens_kv,
        bias_dims=args.bias_dims,
        dbias=args.dbias,
        dropout_p=args.dropout_p,
        ragged_sides=args.ragged_sides,
        ragged_blocks=ragged_blocks,
        ragged_origins=ragged_origins,
        ragged_head_major=ragged_head_major,
        ragged_head_pitch=ragged_head_pitch,
        ragged_token_stride=ragged_token_stride,
        ragged_units=args.ragged_units,
        stats_layout=args.stats_layout,
        graph_only=args.graph_only,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
