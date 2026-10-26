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

The padding, GQA, dBias and dropout features are single-graph only (the sweep
templates cannot express them) and are always cross-checked against an
independent FP64 autograd evaluation of the forward definition.

Not supported yet, rejected with an explicit error rather than silently
mis-generated: FP8, BSHD/ragged layouts, Philox dropout, sinks and ALiBi.

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
GENERATOR_VERSION = "1.3.0"

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
    return {
        "id": case_id,
        "values": {
            "io_data_type": graph_json["io_data_type"],
            "tensors": [
                {
                    "uid": t["uid"],
                    "dims": t["dims"],
                    "strides": t["strides"],
                    "data_type": t["data_type"],
                }
                for t in graph_json["tensors"]
            ],
            "attributes": {
                "parameters__diagonal_alignment": parameters["diagonal_alignment"],
                "parameters__left_bound": parameters["left_bound"],
                "parameters__right_bound": parameters["right_bound"],
            },
        },
        "golden": {"path": golden_path},
        "metadata": meta_json,
    }


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
        elif not all(1 <= v <= S_q for v in seq_lens_q) or not all(
            1 <= v <= S_kv for v in seq_lens_kv
        ):
            errors.append("sequence lengths must be in [1, S_q] and [1, S_kv]")
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
    extended = padded or H_q != H_kv or dbias or dropout_p is not None
    if extended and sweep_bundle_dir is not None:
        errors.append(
            "GQA, padding, dBias and dropout are single-graph only: the sweep "
            "templates cannot express them"
        )
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

    if sweep_bundle_dir is None:
        print(f"Backward bundle: {base_filename}")
        _write_bundle(base_filename, backward_graph, backward_meta, backward_tensors)
    else:
        print(f"Backward sweep case: {sweep_bundle_dir} [{case_id}]")
        write_sweep_golden(
            sweep_bundle_dir, case_id, backward_graph, backward_meta, backward_tensors
        )


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
        required=True,
        help="Path prefix for the paired forward bundle's files (no extension). "
        "Required: RFC 0011 mandates that the forward pass whose O and LSE the "
        "backward bundle consumes is itself saved as a bundle.",
    )
    parser.add_argument(
        "--q-dims",
        nargs=4,
        type=int,
        required=True,
        metavar=("B", "H_Q", "S_Q", "D_QK"),
        help="Query tensor dims: batch, heads_q, seq_q, head_dim_qk",
    )
    parser.add_argument(
        "--v-dims",
        nargs=4,
        type=int,
        required=True,
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
    args = parser.parse_args()

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
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
