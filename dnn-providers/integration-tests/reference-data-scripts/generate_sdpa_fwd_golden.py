#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
SDPA Forward Golden Reference Bundle Generator

Generates pre-computed reference data for SDPA forward kernel validation.
Uses PyTorch's SDPBackend.MATH (or an explicit FP32 reference) as the golden
source. Bundles are provider-agnostic: they encode the SDPA *operation* (graph
attributes + reference tensors), not a specific provider's kernel.

Supported, fully composable operation axes:
  - dtype:        bf16, fp16, fp8 (FP8_E4M3 inputs, BF16 output, per-tensor descale)
  - head dim:     hd128, hd192 (via --q-dims / --v-dims)
  - causal mask:  none, top_left, bottom_right
  - sliding window: --window-left / --window-right
  - GROUP mode:   --variable-seq-lens + --seq-lens-q / --seq-lens-kv
  - LSE output:   --stats (adds tensor uid=4)

Any combination of the above may be applied in a single invocation (e.g. fp8 +
bottom_right causal + group + stats), matching the 12 unique AITER forward tuples.

Output: {base_filename}.json + {base_filename}.tensor{uid}.bin + {base_filename}.meta.json

Usage:
    python generate_sdpa_fwd_golden.py \
        --base-filename golden_data/SdpaFwd/bf16/hd128_nomask_batch/Small \
        --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    python generate_sdpa_fwd_golden.py \
        --base-filename golden_data/SdpaFwd/bf16/hd128_causal_batch/Small \
        --q-dims 2 4 256 128 --v-dims 2 4 256 128 --causal bottom_right --seed 42
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Bump when generator logic changes in a way that affects output data.
# (e.g., different reference backend, precision handling, tensor layout)
# 1.0.0 — Initial forward generator (Q, K, V, O tensors)
# 1.0.1 — Added optional LSE output tensor (uid=4) via --stats flag
# 1.1.0 — Added causal/window masking, FP8 inputs, and GROUP (variable-seq-len) mode
GENERATOR_VERSION = "1.1.0"

DTYPE_MAP = {
    "bf16": {"torch": torch.bfloat16, "json": "bfloat16", "bytes": 2},
    "fp16": {"torch": torch.float16, "json": "half", "bytes": 2},
}

# FP8 operation: FP8_E4M3 inputs (Q/K/V), BF16 output. AITER's fmha fp8 path uses
# the OCP e4m3 encoding (float8_e4m3fn) with a per-tensor scalar descale; mirror it.
# The "fp8_e4m3" json name matches the flatbuffers DataType enum string map.
FP8_TORCH_DTYPE = torch.float8_e4m3fn
FP8_JSON_DTYPE = "fp8_e4m3"
FP8_OUTPUT_TORCH = torch.bfloat16
FP8_OUTPUT_JSON = "bfloat16"
FP8_E4M3_MAX = 448.0  # max representable magnitude of float8_e4m3fn

# Tensor UID assignment (stable contract with the JSON consumer / loader).
UID_Q = 0
UID_K = 1
UID_V = 2
UID_O = 3
UID_LSE = 4
UID_SEQ_LEN_Q = 5
UID_SEQ_LEN_KV = 6
UID_DESCALE_Q = 7
UID_DESCALE_K = 8
UID_DESCALE_V = 9


def compute_contiguous_strides(dims):
    strides = []
    stride = 1
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.reverse()
    return strides


def build_mask(s_q, s_kv, causal, window_left, window_right):
    """Construct a boolean keep-mask [S_q, S_kv] for the given causal/window config.

    Returns None when no masking applies (the dense attention case), so callers
    can take the fast PyTorch path. The mask is True where attention is allowed.
    The bound math matches the CPU reference executor's leftBound/rightBound +
    topLeftAlignment semantics so golden output is reproducible on the CPU side.

    causal:
      - "none":          no causal constraint
      - "top_left":      diagonal anchored at (0,0); keep skv <= sq
      - "bottom_right":  diagonal anchored bottom-right; keep skv <= sq + (S_kv - S_q)
    """
    has_causal = causal != "none"
    has_window = window_left >= 0 or window_right >= 0
    if not has_causal and not has_window:
        return None

    row = torch.arange(s_q).unsqueeze(1)  # [S_q, 1]
    col = torch.arange(s_kv).unsqueeze(0)  # [1, S_kv]
    # Bottom-right alignment shifts the diagonal by (S_kv - S_q).
    offset = 0 if causal != "bottom_right" else (s_kv - s_q)

    mask = torch.ones(s_q, s_kv, dtype=torch.bool)
    if has_causal:
        mask &= col <= (row + offset)
    if window_left >= 0:
        mask &= (row + offset - col) <= window_left
    if window_right >= 0:
        mask &= (col - (row + offset)) <= window_right
    return mask


def _additive_mask(keep_mask):
    """Convert a boolean keep-mask to an additive FP32 mask (0 keep, -inf drop)."""
    additive = torch.zeros(keep_mask.shape, dtype=torch.float32)
    additive.masked_fill_(~keep_mask, float("-inf"))
    return additive


def compute_forward(Q, K, V, scale, H_q, H_kv, keep_mask=None):
    """SDPA forward via PyTorch Math backend (half->FP32->half).

    keep_mask is an optional boolean [S_q, S_kv] (True = attend); applied as an
    additive mask so causal/window compose uniformly with GQA.
    """
    attn_mask = _additive_mask(keep_mask) if keep_mask is not None else None
    with sdpa_kernel(SDPBackend.MATH):
        O = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            scale=scale,
            enable_gqa=(H_q != H_kv),
        )
    return O


def compute_lse(Q, K, scale, H_q, H_kv, keep_mask=None):
    """Compute Log-Sum-Exp reference: logsumexp(Q @ K^T * scale, dim=-1).

    Returns shape [B, H_q, S_q, 1] in FP32. For fully-masked rows the LSE is
    -inf (log of zero), matching the CPU reference semantics.
    """
    Q_f = Q.float()
    K_f = K.float()

    if H_q != H_kv:
        gqa_ratio = H_q // H_kv
        K_f = K_f.repeat_interleave(gqa_ratio, dim=1)

    scores = torch.matmul(Q_f, K_f.transpose(-2, -1)) * scale
    if keep_mask is not None:
        scores = scores + _additive_mask(keep_mask)
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # [B, H_q, S_q, 1]
    return lse


def quantize_fp8_per_tensor(t):
    """Quantize a float tensor to FP8_E4M3 with a single per-tensor descale.

    Returns (fp8_tensor, descale_scalar, dequantized_fp32). descale maps FP8 codes
    back to the original scale: x ~= fp8_code * descale (AITER convention).
    """
    amax = t.abs().max().float()
    scale = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
    quantized = (t.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    fp8 = quantized.to(FP8_TORCH_DTYPE)
    dequant = fp8.float() * scale
    return fp8, scale.reshape(1), dequant


def cumulative_seqlens(seq_lens):
    """Build the cumulative seq-start pointer array [0, s0, s0+s1, ...] (int32)."""
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    return torch.tensor(cu, dtype=torch.int32)


def save_tensor_bin(tensor, path):
    t = tensor.contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16, FP8_TORCH_DTYPE):
        raw = t.view(torch.uint8).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)


def build_graph_json(
    q_dims,
    k_dims,
    v_dims,
    o_dims,
    scale,
    dtype_str="bfloat16",
    stats=False,
    *,
    causal="none",
    window_left=-1,
    window_right=-1,
    group_mode=False,
    fp8=False,
    o_dtype_str=None,
):
    B, H_q, S_q = q_dims[0], q_dims[1], q_dims[2]
    qkv_dtype = FP8_JSON_DTYPE if fp8 else dtype_str
    o_dtype = o_dtype_str if o_dtype_str is not None else dtype_str

    tensors = []
    for uid, name, dims, dt in [
        (UID_Q, "Q", q_dims, qkv_dtype),
        (UID_K, "K", k_dims, qkv_dtype),
        (UID_V, "V", v_dims, qkv_dtype),
        (UID_O, "O", o_dims, o_dtype),
    ]:
        tensors.append(
            {
                "uid": uid,
                "name": name,
                "dims": dims,
                "strides": compute_contiguous_strides(dims),
                "data_type": dt,
                "virtual": False,
            }
        )

    if stats:
        lse_dims = [B, H_q, S_q, 1]
        tensors.append(
            {
                "uid": UID_LSE,
                "name": "LSE",
                "dims": lse_dims,
                "strides": compute_contiguous_strides(lse_dims),
                "data_type": "float",
                "virtual": False,
            }
        )

    if group_mode:
        for uid, name in [(UID_SEQ_LEN_Q, "SeqLenQ"), (UID_SEQ_LEN_KV, "SeqLenKv")]:
            tensors.append(
                {
                    "uid": uid,
                    "name": name,
                    "dims": [B + 1],
                    "strides": [1],
                    "data_type": "int32",
                    "virtual": False,
                }
            )

    if fp8:
        for uid, name in [
            (UID_DESCALE_Q, "DescaleQ"),
            (UID_DESCALE_K, "DescaleK"),
            (UID_DESCALE_V, "DescaleV"),
        ]:
            tensors.append(
                {
                    "uid": uid,
                    "name": name,
                    "dims": [1],
                    "strides": [1],
                    "data_type": "float",
                    "virtual": False,
                }
            )

    # Causal / window -> bounds + diagonal alignment (non-deprecated path).
    diagonal_alignment = "BOTTOM_RIGHT" if causal == "bottom_right" else "TOP_LEFT"
    left_bound = None
    right_bound = None
    if causal != "none":
        left_bound = -1
        right_bound = 0
    if window_left >= 0:
        left_bound = window_left
    if window_right >= 0:
        right_bound = window_right

    graph = {
        "nodes": [
            {
                "type": "SdpaAttributes",
                "compute_data_type": "float",
                "name": "",
                "inputs": {
                    "q_tensor_uid": UID_Q,
                    "k_tensor_uid": UID_K,
                    "v_tensor_uid": UID_V,
                    "attn_mask_tensor_uid": None,
                    "scale_tensor_uid": None,
                    "seq_len_q_tensor_uid": UID_SEQ_LEN_Q if group_mode else None,
                    "seq_len_kv_tensor_uid": UID_SEQ_LEN_KV if group_mode else None,
                    "seed_tensor_uid": None,
                    "offset_tensor_uid": None,
                    "dropout_mask_tensor_uid": None,
                    "dropout_scale_tensor_uid": None,
                    "page_table_k_tensor_uid": None,
                    "page_table_v_tensor_uid": None,
                    "block_mask_tensor_uid": None,
                    "sink_token_tensor_uid": None,
                    "descale_q_tensor_uid": UID_DESCALE_Q if fp8 else None,
                    "descale_k_tensor_uid": UID_DESCALE_K if fp8 else None,
                    "descale_v_tensor_uid": UID_DESCALE_V if fp8 else None,
                    "descale_s_tensor_uid": None,
                    "scale_s_tensor_uid": None,
                    "scale_o_tensor_uid": None,
                },
                "outputs": {
                    "o_tensor_uid": UID_O,
                    "stats_tensor_uid": UID_LSE if stats else None,
                    "max_tensor_uid": None,
                    "sum_exp_tensor_uid": None,
                    "rng_dump_tensor_uid": None,
                    "amax_s_tensor_uid": None,
                    "amax_o_tensor_uid": None,
                },
                "attributes": {
                    "generate_stats": True if stats else None,
                    "alibi_mask": False,
                    "padding_mask": False,
                    "causal_mask": False,
                    "causal_mask_bottom_right": False,
                    "dropout_probability": None,
                    "attn_scale_value": scale,
                    "left_bound": left_bound,
                    "right_bound": right_bound,
                    "max_seq_len_kv": None,
                    "diagonal_alignment": diagonal_alignment,
                    "mma_core_mode": "float",
                    "implementation": "AUTO",
                },
            }
        ],
        "tensors": tensors,
        "io_data_type": dtype_str,
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }
    return graph


def _get_generator_sha256():
    """SHA-256 of this script's contents — git-independent version marker."""
    script_path = os.path.abspath(__file__)
    with open(script_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_meta_json(config, pytorch_version):
    rocm_ver = ""
    if "+rocm" in pytorch_version:
        rocm_ver = pytorch_version.split("+rocm")[1]

    return {
        "generator": "generate_sdpa_fwd_golden.py",
        "generator_sha256": _get_generator_sha256(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": f"PyTorch {pytorch_version}",
        "reference_backend": "pytorch_math_backend",
        "rocm_version": rocm_ver,
        "generator_version": GENERATOR_VERSION,
        "generation_precision": config["precision_note"],
        "direction": "forward",
        "seed": config["seed"],
        "input_range": [config["min_val"], config["max_val"]],
        "deterministic": True,
        "config": {
            "batch": config["q_dims"][0],
            "num_heads_q": config["q_dims"][1],
            "num_heads_kv": config["v_dims"][1],
            "seq_q": config["q_dims"][2],
            "seq_kv": config["v_dims"][2],
            "head_dim_qk": config["q_dims"][3],
            "head_dim_v": config["v_dims"][3],
            "dtype": config["dtype"],
            "causal": config["causal"],
            "window_left": config["window_left"],
            "window_right": config["window_right"],
            "group_mode": config["group_mode"],
            "seq_lens_q": config["seq_lens_q"],
            "seq_lens_kv": config["seq_lens_kv"],
            "stats": config["stats"],
            "scale": config["scale"],
            "gqa_ratio": config["q_dims"][1] // config["v_dims"][1],
        },
    }


AITER_VERSION = "0.1.13"
AITER_WHEEL_BASE = f"https://github.com/ROCm/aiter/releases/download/v{AITER_VERSION}"


def _get_rocm_version():
    """Extract ROCm version (e.g. '7.2') from the installed PyTorch build."""
    version = torch.__version__
    if "+rocm" in version:
        return version.split("+rocm")[1]
    return ""


def _get_python_tag():
    """Return cpython tag like 'cp310' or 'cp312'."""
    v = sys.version_info
    return f"cp{v.major}{v.minor:02d}"


def install_aiter():
    """Install the aiter wheel matching the system's ROCm version and Python."""
    rocm_ver = _get_rocm_version()
    if not rocm_ver:
        print(
            "ERROR: Cannot detect ROCm version from PyTorch. "
            "Install a ROCm-enabled PyTorch first.",
            file=sys.stderr,
        )
        sys.exit(1)

    py_tag = _get_python_tag()
    wheel_name = (
        f"amd_aiter-{AITER_VERSION}+rocm{rocm_ver}.manylinux.2.28"
        f"-{py_tag}-{py_tag}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    wheel_url = f"{AITER_WHEEL_BASE}/{wheel_name}"

    print(f"Installing aiter v{AITER_VERSION} for ROCm {rocm_ver} (Python {py_tag})")
    print(f"  URL: {wheel_url}")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", wheel_url],
        capture_output=False,
    )
    if result.returncode != 0:
        print(
            f"ERROR: pip install failed (exit code {result.returncode})",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  aiter v{AITER_VERSION} installed successfully.")


def validate_against_aiter(
    Q_bhsd,
    K_bhsd,
    V_bhsd,
    golden_O,
    scale,
    causal=False,
    window_left=-1,
    window_right=-1,
    stats=False,
):
    """Cross-check golden output against AITER fmha_v3_fwd GPU kernel.

    Layout:
      - Golden data uses BHSD (batch, heads, seq, dim) — PyTorch SDPA convention
      - AITER fmha_v3_fwd uses BSHD (batch, seq, heads, dim) — transpose dims 1,2

    Why transpose instead of stride manipulation:
      AITER's Python fmha_v3_fwd API requires contiguous BSHD tensors — it does
      not accept arbitrary strides. The hip-kernel-provider's C++ path, by contrast,
      passes raw per-axis strides (s_Seqs, s_Hs, s_Bs) to the kernel and handles
      any layout without transposing. These are two different interfaces to the same
      ASM kernel: Python API enforces BSHD shape, C++ path is stride-agnostic.
    """

    try:
        import aiter
        from aiter import fmha_v3_fwd

        try:
            from importlib.metadata import version as pkg_version

            aiter_ver = pkg_version("amd-aiter")
        except Exception:
            aiter_ver = getattr(aiter, "__version__", "unknown")
        print(f"  AITER version: {aiter_ver}")
        print(
            f"  Args: is_causal={causal}, window=({window_left},{window_right}), "
            f"return_softmax_lse={stats}"
        )

        Q_bshd = Q_bhsd.cuda().transpose(1, 2).contiguous()
        K_bshd = K_bhsd.cuda().transpose(1, 2).contiguous()
        V_bshd = V_bhsd.cuda().transpose(1, 2).contiguous()

        O_aiter_bshd, _, _, _ = fmha_v3_fwd(
            Q_bshd,
            K_bshd,
            V_bshd,
            dropout_p=0.0,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=window_left,
            window_size_right=window_right,
            return_softmax_lse=stats,
            return_dropout_randval=False,
            how_v3_bf16_cvt=0,  # RTNE — matches hip-kernel-provider
        )
        O_aiter_bhsd = O_aiter_bshd.transpose(1, 2).contiguous().cpu()

        diff_aiter = (golden_O.float() - O_aiter_bhsd.float()).abs()
        max_aiter = diff_aiter.max().item()
        print(
            f"  AITER v3 fwd:  max_abs={max_aiter:.6f}  "
            f"p99={diff_aiter.quantile(0.99).item():.6f}  "
            f"mean={diff_aiter.mean().item():.6f}"
        )
        if max_aiter > 0.01:
            print(f"  WARNING: Golden vs AITER exceeds 0.01 (got {max_aiter:.6f})")
            return False
        return True
    except ImportError:
        print("  AITER:         SKIPPED (aiter not installed)")
        return True


def _validate_group_args(B, seq_lens_q, seq_lens_kv, S_q, S_kv):
    errors = []
    if seq_lens_q is None or seq_lens_kv is None:
        errors.append(
            "--variable-seq-lens requires both --seq-lens-q and --seq-lens-kv"
        )
        return errors
    if len(seq_lens_q) != B or len(seq_lens_kv) != B:
        errors.append(
            f"--seq-lens-q/--seq-lens-kv must have exactly B={B} entries "
            f"(got {len(seq_lens_q)} and {len(seq_lens_kv)})"
        )
    for i, s in enumerate(seq_lens_q):
        if s <= 0 or s > S_q:
            errors.append(f"--seq-lens-q[{i}]={s} must be in (0, S_q={S_q}]")
    for i, s in enumerate(seq_lens_kv):
        if s <= 0 or s > S_kv:
            errors.append(f"--seq-lens-kv[{i}]={s} must be in (0, S_kv={S_kv}]")
    return errors


def _compute_group_forward(
    Q,
    K,
    V,
    scale,
    H_q,
    H_kv,
    causal,
    window_left,
    window_right,
    seq_lens_q,
    seq_lens_kv,
    o_torch_dtype,
    want_lse,
):
    """GROUP-mode reference: run SDPA independently per batch sequence.

    Each batch b attends only over its valid [seq_lens_q[b], seq_lens_kv[b]]
    region; padding positions are zero-filled in O (and -inf in LSE).
    """
    B, _, _, _ = Q.shape
    _, _, _, D_v = V.shape
    O = torch.zeros((B, H_q, Q.shape[2], D_v), dtype=o_torch_dtype)
    lse = (
        torch.full((B, H_q, Q.shape[2], 1), float("-inf"), dtype=torch.float32)
        if want_lse
        else None
    )
    for b in range(B):
        sq = seq_lens_q[b]
        skv = seq_lens_kv[b]
        q_b = Q[b : b + 1, :, :sq, :]
        k_b = K[b : b + 1, :, :skv, :]
        v_b = V[b : b + 1, :, :skv, :]
        keep = build_mask(sq, skv, causal, window_left, window_right)
        o_b = compute_forward(q_b, k_b, v_b, scale, H_q, H_kv, keep)
        O[b : b + 1, :, :sq, :] = o_b.to(o_torch_dtype)
        if want_lse:
            lse[b : b + 1, :, :sq, :] = compute_lse(q_b, k_b, scale, H_q, H_kv, keep)
    return O, lse


def generate_forward_bundle(
    base_filename,
    q_dims,
    v_dims,
    dtype="bf16",
    causal="none",
    window_left=-1,
    window_right=-1,
    variable_seq_lens=False,
    seq_lens_q=None,
    seq_lens_kv=None,
    stats=False,
    seed=42,
    min_val=-1.0,
    max_val=1.0,
    attn_scale=None,
    validate=False,
):
    B, H_q, S_q, D_qk = q_dims
    B_v, H_kv, S_kv, D_v = v_dims
    k_dims = [B, H_kv, S_kv, D_qk]
    o_dims = [B, H_q, S_q, D_v]

    is_fp8 = dtype == "fp8"
    if not is_fp8 and dtype not in DTYPE_MAP:
        print(
            f"ERROR: --dtype must be one of {list(DTYPE_MAP.keys()) + ['fp8']} "
            f"(got '{dtype}')",
            file=sys.stderr,
        )
        sys.exit(1)

    if is_fp8:
        gen_torch_dtype = torch.float32  # quantize FP8 from FP32
        o_torch_dtype = FP8_OUTPUT_TORCH
        io_json = FP8_OUTPUT_JSON  # io_data_type describes the float O dtype
        o_json = FP8_OUTPUT_JSON
    else:
        info = DTYPE_MAP[dtype]
        gen_torch_dtype = info["torch"]
        o_torch_dtype = info["torch"]
        io_json = info["json"]
        o_json = info["json"]

    errors = []
    for name, dims in [("q", q_dims), ("v", v_dims)]:
        for i, d in enumerate(dims):
            if d <= 0:
                errors.append(f"--{name}-dims[{i}] must be positive (got {d})")
    if B_v != B:
        errors.append(f"Batch mismatch: Q batch={B}, V batch={B_v}")
    if H_q % H_kv != 0:
        errors.append(f"H_q ({H_q}) must be divisible by H_kv ({H_kv})")
    if min_val >= max_val:
        errors.append(f"--min ({min_val}) must be less than --max ({max_val})")
    if variable_seq_lens:
        errors += _validate_group_args(B, seq_lens_q, seq_lens_kv, S_q, S_kv)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(D_qk)

    os.makedirs(os.path.dirname(base_filename) or ".", exist_ok=True)

    print(f"Generating forward bundle: {base_filename}")
    print(f"  Q: {q_dims}, K: {k_dims}, V: {v_dims}, O: {o_dims}")
    print(f"  dtype: {dtype} (output {o_torch_dtype})")
    print(f"  H_q={H_q}, H_kv={H_kv}, GQA ratio={H_q // H_kv}")
    print(
        f"  causal={causal}, window=({window_left},{window_right}), "
        f"group={variable_seq_lens}, stats={stats}"
    )
    print(f"  Scale: {attn_scale:.10f}, Seed: {seed}")

    rng = torch.Generator().manual_seed(seed)
    Q = torch.empty(q_dims, dtype=gen_torch_dtype).uniform_(
        min_val, max_val, generator=rng
    )
    K = torch.empty(k_dims, dtype=gen_torch_dtype).uniform_(
        min_val, max_val, generator=rng
    )
    V = torch.empty(v_dims, dtype=gen_torch_dtype).uniform_(
        min_val, max_val, generator=rng
    )

    descale_q = descale_k = descale_v = None
    if is_fp8:
        Q_fp8, descale_q, Q_deq = quantize_fp8_per_tensor(Q)
        K_fp8, descale_k, K_deq = quantize_fp8_per_tensor(K)
        V_fp8, descale_v, V_deq = quantize_fp8_per_tensor(V)
        Q_compute, K_compute, V_compute = Q_deq, K_deq, V_deq
        Q_store, K_store, V_store = Q_fp8, K_fp8, V_fp8
    else:
        Q_compute, K_compute, V_compute = Q, K, V
        Q_store, K_store, V_store = Q, K, V

    keep_mask = build_mask(S_q, S_kv, causal, window_left, window_right)

    try:
        if variable_seq_lens:
            O, lse = _compute_group_forward(
                Q_compute,
                K_compute,
                V_compute,
                attn_scale,
                H_q,
                H_kv,
                causal,
                window_left,
                window_right,
                seq_lens_q,
                seq_lens_kv,
                o_torch_dtype,
                want_lse=stats,
            )
        else:
            O = compute_forward(
                Q_compute, K_compute, V_compute, attn_scale, H_q, H_kv, keep_mask
            ).to(o_torch_dtype)
            lse = (
                compute_lse(Q_compute, K_compute, attn_scale, H_q, H_kv, keep_mask)
                if stats
                else None
            )
    except RuntimeError as e:
        print(f"ERROR: PyTorch SDPA failed: {e}", file=sys.stderr)
        sys.exit(1)

    assert not torch.isnan(O).any(), "NaN in O"
    assert not torch.isinf(O).any(), "Inf in O"

    # Write raw tensor data as .bin files (one per tensor UID)
    tensor_list = [
        ("Q", Q_store, UID_Q),
        ("K", K_store, UID_K),
        ("V", V_store, UID_V),
        ("O", O, UID_O),
    ]
    if lse is not None:
        tensor_list.append(("LSE", lse, UID_LSE))
    if variable_seq_lens:
        tensor_list.append(("SeqLenQ", cumulative_seqlens(seq_lens_q), UID_SEQ_LEN_Q))
        tensor_list.append(
            ("SeqLenKv", cumulative_seqlens(seq_lens_kv), UID_SEQ_LEN_KV)
        )
    if is_fp8:
        tensor_list.append(("DescaleQ", descale_q, UID_DESCALE_Q))
        tensor_list.append(("DescaleK", descale_k, UID_DESCALE_K))
        tensor_list.append(("DescaleV", descale_v, UID_DESCALE_V))

    for name, tensor, uid in tensor_list:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        size_kb = os.path.getsize(bin_path) / 1024
        print(
            f"  {name} (uid={uid}): {list(tensor.shape)} {tensor.dtype} -> {size_kb:.1f} KB"
        )

    # Write graph JSON (operation definition: node type, tensor metadata, attributes)
    graph_json = build_graph_json(
        q_dims,
        k_dims,
        v_dims,
        o_dims,
        attn_scale,
        dtype_str=io_json,
        stats=stats,
        causal=causal,
        window_left=window_left,
        window_right=window_right,
        group_mode=variable_seq_lens,
        fp8=is_fp8,
        o_dtype_str=o_json,
    )
    json_path = f"{base_filename}.json"
    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=4)
        f.write("\n")
    print(f"  Graph JSON: {json_path}")

    precision_note = (
        "fp8: FP8_E4M3 inputs (per-tensor descale), FP32 intermediates, BF16 output"
        if is_fp8
        else f"like-for-like: {dtype.upper()} inputs, FP32 intermediates"
    )
    config = {
        "q_dims": q_dims,
        "v_dims": v_dims,
        "dtype": dtype,
        "causal": causal,
        "window_left": window_left,
        "window_right": window_right,
        "group_mode": variable_seq_lens,
        "seq_lens_q": seq_lens_q,
        "seq_lens_kv": seq_lens_kv,
        "stats": stats,
        "seed": seed,
        "min_val": min_val,
        "max_val": max_val,
        "scale": attn_scale,
        "precision_note": precision_note,
    }
    meta_json = build_meta_json(config, torch.__version__)
    meta_path = f"{base_filename}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=4)
        f.write("\n")
    print(f"  Meta JSON: {meta_path}")

    # Optional: cross-check golden output against AITER GPU kernel (dense float only)
    if validate:
        if variable_seq_lens or is_fp8:
            print(
                "  Validation: SKIPPED (AITER cross-check only covers dense float "
                "BATCH mode; GROUP/FP8 are validated by the CPU reference)"
            )
        else:
            ok = validate_against_aiter(
                Q_store,
                K_store,
                V_store,
                O,
                attn_scale,
                causal=(causal != "none"),
                window_left=window_left,
                window_right=window_right,
                stats=stats,
            )
            print("  Validation: PASSED" if ok else "  Validation: FAILED")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SDPA forward golden reference bundles"
    )
    parser.add_argument(
        "--base-filename",
        required=True,
        help="Path prefix for output files (no extension)",
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
        choices=list(DTYPE_MAP.keys()) + ["fp8"],
        help="Input dtype. fp8 = FP8_E4M3 inputs, BF16 output, per-tensor descale "
        "(default: bf16)",
    )
    parser.add_argument(
        "--causal",
        default="none",
        choices=["none", "top_left", "bottom_right"],
        help="Causal mask type. bottom_right maps to AITER mask=2; top_left is "
        "reference-only (no AITER kernel) (default: none)",
    )
    parser.add_argument(
        "--window-left",
        type=int,
        default=-1,
        help="Sliding window left bound, -1 = unbounded (default: -1)",
    )
    parser.add_argument(
        "--window-right",
        type=int,
        default=-1,
        help="Sliding window right bound, -1 = unbounded (default: -1)",
    )
    parser.add_argument(
        "--variable-seq-lens",
        action="store_true",
        help="Enable GROUP mode (variable per-batch sequence lengths). Requires "
        "--seq-lens-q and --seq-lens-kv.",
    )
    parser.add_argument(
        "--seq-lens-q",
        nargs="+",
        type=int,
        default=None,
        help="Per-batch query lengths (GROUP mode). Must have B entries.",
    )
    parser.add_argument(
        "--seq-lens-kv",
        nargs="+",
        type=int,
        default=None,
        help="Per-batch key/value lengths (GROUP mode). Must have B entries.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Enable LSE output tensor (uid=4, shape [B, H_q, S_q, 1], dtype FP32)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min", type=float, default=-1.0, dest="min_val")
    parser.add_argument("--max", type=float, default=1.0, dest="max_val")
    parser.add_argument(
        "--attn-scale",
        type=float,
        default=None,
        help="Attention scale (default: 1/sqrt(D_qk))",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Cross-check golden output against AITER fmha_v3_fwd (requires GPU)",
    )
    parser.add_argument(
        "--install-aiter",
        action="store_true",
        help=f"Install aiter v{AITER_VERSION} wheel for the detected ROCm version",
    )
    args = parser.parse_args()

    if args.install_aiter:
        install_aiter()

    generate_forward_bundle(
        base_filename=args.base_filename,
        q_dims=args.q_dims,
        v_dims=args.v_dims,
        dtype=args.dtype,
        causal=args.causal,
        window_left=args.window_left,
        window_right=args.window_right,
        variable_seq_lens=args.variable_seq_lens,
        seq_lens_q=args.seq_lens_q,
        seq_lens_kv=args.seq_lens_kv,
        stats=args.stats,
        seed=args.seed,
        min_val=args.min_val,
        max_val=args.max_val,
        attn_scale=args.attn_scale,
        validate=args.validate,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
