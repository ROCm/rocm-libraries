#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 02: All Convolution Directions with NumPy CPU Reference

Demonstrates forward 2D/3D, backward-data, and backward-weight
config generation and validation, with NumPy CPU reference
implementations for each direction.

Usage:
    python3 02_all_directions.py
    python3 02_all_directions.py --arch gfx950
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codegen"))

from ctypes_utils import detect_gpu_arch
from grouped_conv_utils import (
    validate_grouped_conv_config,
    auto_correct_grouped_conv_config,
    get_grouped_conv_default_config,
    format_grouped_conv_summary,
)


# =============================================================================
# NumPy CPU Reference Implementations
# =============================================================================


def reference_conv2d_fwd(input_nhwc, weight_kyxc, stride=(1, 1), padding=(0, 0)):
    """CPU reference: 2D convolution forward (NHWC layout).

    input_nhwc: (N, Hi, Wi, C)
    weight_kyxc: (K, Y, X, C)
    returns: (N, Ho, Wo, K)
    """
    N, Hi, Wi, C = input_nhwc.shape
    K, Y, X, C_w = weight_kyxc.shape
    assert C == C_w, f"Channel mismatch: input {C} vs weight {C_w}"

    pad_h, pad_w = padding
    stride_h, stride_w = stride

    if pad_h > 0 or pad_w > 0:
        input_nhwc = np.pad(
            input_nhwc, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0))
        )

    Ho = (Hi + 2 * pad_h - Y) // stride_h + 1
    Wo = (Wi + 2 * pad_w - X) // stride_w + 1
    output = np.zeros((N, Ho, Wo, K), dtype=np.float32)

    for n in range(N):
        for ho in range(Ho):
            for wo in range(Wo):
                for k in range(K):
                    acc = 0.0
                    for y in range(Y):
                        for x in range(X):
                            for c in range(C):
                                hi = ho * stride_h + y
                                wi = wo * stride_w + x
                                acc += float(input_nhwc[n, hi, wi, c]) * float(
                                    weight_kyxc[k, y, x, c]
                                )
                    output[n, ho, wo, k] = acc

    return output


def reference_conv3d_fwd(input_ndhwc, weight_kzyxc, stride=1, padding=0):
    """CPU reference: 3D convolution forward (NDHWC layout).

    input_ndhwc: (N, Di, Hi, Wi, C)
    weight_kzyxc: (K, Z, Y, X, C)
    returns: (N, Do, Ho, Wo, K)
    """
    N, Di, Hi, Wi, C = input_ndhwc.shape
    K, Z, Y, X, C_w = weight_kzyxc.shape
    assert C == C_w

    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    pd, ph, pw = padding
    sd, sh, sw = stride

    if pd > 0 or ph > 0 or pw > 0:
        input_ndhwc = np.pad(
            input_ndhwc, ((0, 0), (pd, pd), (ph, ph), (pw, pw), (0, 0))
        )

    Do = (Di + 2 * pd - Z) // sd + 1
    Ho = (Hi + 2 * ph - Y) // sh + 1
    Wo = (Wi + 2 * pw - X) // sw + 1
    output = np.zeros((N, Do, Ho, Wo, K), dtype=np.float32)

    for n in range(N):
        for do_ in range(Do):
            for ho in range(Ho):
                for wo in range(Wo):
                    for k in range(K):
                        acc = 0.0
                        for z in range(Z):
                            for y in range(Y):
                                for x in range(X):
                                    for c in range(C):
                                        di = do_ * sd + z
                                        hi = ho * sh + y
                                        wi = wo * sw + x
                                        acc += float(
                                            input_ndhwc[n, di, hi, wi, c]
                                        ) * float(weight_kzyxc[k, z, y, x, c])
                        output[n, do_, ho, wo, k] = acc

    return output


def reference_conv2d_bwd_data(grad_output, weight_kyxc, Hi, Wi, stride=(1, 1), padding=(0, 0)):
    """CPU reference: 2D convolution backward data (NHWC layout).

    Computes gradient w.r.t. input: dX = ConvBwdData(dY, W)

    grad_output: (N, Ho, Wo, K)
    weight_kyxc: (K, Y, X, C)
    returns: (N, Hi, Wi, C)
    """
    N, Ho, Wo, K = grad_output.shape
    K_w, Y, X, C = weight_kyxc.shape
    assert K == K_w

    stride_h, stride_w = stride
    pad_h, pad_w = padding

    grad_input = np.zeros((N, Hi, Wi, C), dtype=np.float32)

    for n in range(N):
        for hi in range(Hi):
            for wi in range(Wi):
                for c in range(C):
                    acc = 0.0
                    for y in range(Y):
                        for x in range(X):
                            h_tmp = hi + pad_h - y
                            w_tmp = wi + pad_w - x
                            if h_tmp % stride_h == 0 and w_tmp % stride_w == 0:
                                ho = h_tmp // stride_h
                                wo = w_tmp // stride_w
                                if 0 <= ho < Ho and 0 <= wo < Wo:
                                    for k in range(K):
                                        acc += float(
                                            grad_output[n, ho, wo, k]
                                        ) * float(weight_kyxc[k, y, x, c])
                    grad_input[n, hi, wi, c] = acc

    return grad_input


def reference_conv2d_bwd_weight(input_nhwc, grad_output, Y, X, stride=(1, 1), padding=(0, 0)):
    """CPU reference: 2D convolution backward weight (NHWC layout).

    Computes gradient w.r.t. weight: dW = ConvBwdWeight(X, dY)

    input_nhwc: (N, Hi, Wi, C)
    grad_output: (N, Ho, Wo, K)
    returns: (K, Y, X, C)
    """
    N, Hi, Wi, C = input_nhwc.shape
    N_g, Ho, Wo, K = grad_output.shape
    assert N == N_g

    stride_h, stride_w = stride
    pad_h, pad_w = padding

    if pad_h > 0 or pad_w > 0:
        input_nhwc = np.pad(
            input_nhwc, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0))
        )

    grad_weight = np.zeros((K, Y, X, C), dtype=np.float32)

    for k in range(K):
        for y in range(Y):
            for x in range(X):
                for c in range(C):
                    acc = 0.0
                    for n in range(N):
                        for ho in range(Ho):
                            for wo in range(Wo):
                                hi = ho * stride_h + y
                                wi = wo * stride_w + x
                                acc += float(input_nhwc[n, hi, wi, c]) * float(
                                    grad_output[n, ho, wo, k]
                                )
                    grad_weight[k, y, x, c] = acc

    return grad_weight


# =============================================================================
# Validation helper
# =============================================================================


def validate(result, reference, name, rtol=1e-2, atol=1e-3):
    """Compare result vs reference, print stats, return pass/fail."""
    result_f32 = result.astype(np.float32)
    reference_f32 = reference.astype(np.float32)

    abs_diff = np.abs(result_f32 - reference_f32)
    max_abs = float(abs_diff.max())

    nonzero = np.abs(reference_f32) > 1e-6
    if np.any(nonzero):
        max_rel = float((abs_diff[nonzero] / np.abs(reference_f32[nonzero])).max())
    else:
        max_rel = max_abs

    passed = np.allclose(result_f32, reference_f32, rtol=rtol, atol=atol)

    status = "PASS" if passed else "FAIL"
    print(f"  {name}: max_abs={max_abs:.6f}, max_rel={max_rel:.6f} -> {status}")
    return passed


# =============================================================================
# Direction tests
# =============================================================================


def test_forward_2d():
    """2D forward conv with known-answer test (fp16).
    All-ones input (1,4,4,2) * all-ones weight (1,3,3,2) with padding=1 =>
    center pixel sees full 3x3 receptive field: sum = 3*3*2 = 18.0."""
    N, C, K, Hi, Wi, Y, X = 1, 2, 1, 4, 4, 3, 3
    inp = np.ones((N, Hi, Wi, C), dtype=np.float16)
    wei = np.ones((K, Y, X, C), dtype=np.float16)

    result = reference_conv2d_fwd(inp, wei, stride=(1, 1), padding=(1, 1))

    expected_center = float(Y * X * C)  # 18.0
    expected_corner = 4.0 * C            # 8.0

    center_ok = abs(result[0, 1, 1, 0] - expected_center) < 0.5
    corner_ok = abs(result[0, 0, 0, 0] - expected_corner) < 0.5

    print(f"  fwd_2d: center={result[0,1,1,0]:.1f} (expect {expected_center:.1f}), "
          f"corner={result[0,0,0,0]:.1f} (expect {expected_corner:.1f}) "
          f"-> {'PASS' if center_ok and corner_ok else 'FAIL'}")
    return center_ok and corner_ok


def test_forward_2d_random():
    """2D forward conv with random fp16 data, cross-checked against im2col+matmul."""
    np.random.seed(42)
    N, C, K, Hi, Wi, Y, X = 1, 4, 8, 6, 6, 3, 3
    inp = np.random.randn(N, Hi, Wi, C).astype(np.float16)
    wei = np.random.randn(K, Y, X, C).astype(np.float16)

    result = reference_conv2d_fwd(inp, wei, stride=(1, 1), padding=(0, 0))

    Ho = Hi - Y + 1  # 4
    Wo = Wi - X + 1  # 4
    patches = np.zeros((N, Ho, Wo, Y * X * C), dtype=np.float16)
    for ho in range(Ho):
        for wo in range(Wo):
            patches[0, ho, wo, :] = inp[0, ho:ho+Y, wo:wo+X, :].ravel()
    wei_mat = wei.reshape(K, -1).T  # (Y*X*C, K)
    expected = patches[0].reshape(-1, Y*X*C).astype(np.float32) @ wei_mat.astype(np.float32)
    expected = expected.reshape(N, Ho, Wo, K)

    return validate(result, expected, "fwd_2d_random_fp16", rtol=5e-2, atol=5e-2)


def test_forward_3d():
    """3D forward conv with known-answer test (fp16)."""
    N, C, K, Di, Hi, Wi = 1, 1, 1, 3, 3, 3
    Z, Y, X = 3, 3, 3
    inp = np.ones((N, Di, Hi, Wi, C), dtype=np.float16)
    wei = np.ones((K, Z, Y, X, C), dtype=np.float16)

    result = reference_conv3d_fwd(inp, wei, stride=1, padding=1)

    center_val = result[0, 1, 1, 1, 0]
    center_ok = abs(center_val - 27.0) < 0.5
    corner_val = result[0, 0, 0, 0, 0]
    corner_ok = abs(corner_val - 8.0) < 0.5

    print(f"  fwd_3d: center={center_val:.1f} (expect 27.0), "
          f"corner={corner_val:.1f} (expect 8.0) "
          f"-> {'PASS' if center_ok and corner_ok else 'FAIL'}")
    return center_ok and corner_ok


def test_bwd_data_2d():
    """2D backward data (fp16): fwd then bwd_data, verify adjoint relationship."""
    np.random.seed(44)
    N, C, K, Hi, Wi, Y, X = 1, 4, 8, 6, 6, 3, 3
    pad, stride = (1, 1), (1, 1)

    x = np.random.randn(N, Hi, Wi, C).astype(np.float16)
    w = np.random.randn(K, Y, X, C).astype(np.float16)
    dy = np.random.randn(N, Hi, Wi, K).astype(np.float16)

    fwd_out = reference_conv2d_fwd(x, w, stride=stride, padding=pad)
    bwd_out = reference_conv2d_bwd_data(dy, w, Hi, Wi, stride=stride, padding=pad)

    # Adjoint test: <dy, A*x> ~= <A^T*dy, x>  (fp16 accumulation -> looser tol)
    lhs = np.sum(dy.astype(np.float32) * fwd_out.astype(np.float32))
    rhs = np.sum(bwd_out.astype(np.float32) * x.astype(np.float32))
    rel_err = abs(float(lhs - rhs)) / (abs(float(lhs)) + 1e-6)
    ok = rel_err < 0.1  # 10% for fp16 accumulation

    print(f"  bwd_data_2d: <dy,Ax>={float(lhs):.4f}, <A^T dy,x>={float(rhs):.4f}, "
          f"rel_err={rel_err:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_bwd_weight_2d():
    """2D backward weight (fp16): known-answer with all-ones.
    dW[k,1,1,c] = Ho*Wo = 16, dW[k,0,0,c] = (Ho-1)*(Wo-1) = 9."""
    N, C, K, Hi, Wi, Y, X = 1, 2, 3, 4, 4, 3, 3
    Ho, Wo = Hi, Wi  # stride=1, pad=1

    inp = np.ones((N, Hi, Wi, C), dtype=np.float16)
    grad_out = np.ones((N, Ho, Wo, K), dtype=np.float16)

    grad_weight = reference_conv2d_bwd_weight(
        inp, grad_out, Y, X, stride=(1, 1), padding=(1, 1)
    )

    center_val = grad_weight[0, 1, 1, 0]
    expected = float(Ho * Wo * N)
    center_ok = abs(center_val - expected) < 0.5

    corner_val = grad_weight[0, 0, 0, 0]
    expected_corner = float((Ho - 1) * (Wo - 1) * N)
    corner_ok = abs(corner_val - expected_corner) < 0.5

    print(f"  bwd_weight_2d: center_dW={center_val:.1f} (expect {expected:.1f}), "
          f"corner_dW={corner_val:.1f} (expect {expected_corner:.1f}) "
          f"-> {'PASS' if center_ok and corner_ok else 'FAIL'}")
    return center_ok and corner_ok


def test_fwd_bwd_consistency():
    """Cross-check adjoint property with fp16: <dY, fwd(X, W)> ~= <bwd_data(dY, W), X>."""
    np.random.seed(46)
    N, C, K, Hi, Wi, Y, X = 1, 4, 8, 6, 6, 3, 3
    pad = (1, 1)
    stride = (1, 1)

    x = np.random.randn(N, Hi, Wi, C).astype(np.float16)
    w = np.random.randn(K, Y, X, C).astype(np.float16)
    dy = np.random.randn(N, Hi, Wi, K).astype(np.float16)

    fwd_out = reference_conv2d_fwd(x, w, stride=stride, padding=pad)
    bwd_out = reference_conv2d_bwd_data(dy, w, Hi, Wi, stride=stride, padding=pad)

    lhs = float(np.sum(dy.astype(np.float32) * fwd_out.astype(np.float32)))
    rhs = float(np.sum(bwd_out.astype(np.float32) * x.astype(np.float32)))
    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    ok = rel_err < 0.1  # fp16 accumulation tolerance

    print(f"  fwd_bwd_adjoint: <dy,Ax>={lhs:.4f}, <A^T dy,x>={rhs:.4f}, "
          f"rel_err={rel_err:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="All Convolution Directions with NumPy Reference")
    parser.add_argument(
        "--arch", default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 02: All Convolution Directions with NumPy CPU Reference")
    print("=" * 70)
    print(f"\n  Arch: {args.arch}\n")

    # =========================================================================
    # Config validation for all directions
    # =========================================================================
    print("--- Config Validation ---")
    test_cases = [
        ("forward", 2), ("forward", 3),
        ("bwd_data", 2), ("bwd_data", 3),
        ("bwd_weight", 2), ("bwd_weight", 3),
    ]

    print(f"  {'Direction':<20} {'Dims':<6} {'Valid':<8}")
    print("  " + "-" * 40)

    config_results = []
    for variant, ndim in test_cases:
        config = get_grouped_conv_default_config(
            variant=variant, ndim_spatial=ndim, arch=args.arch, dtype="fp16",
        )
        result = validate_grouped_conv_config(config)
        if not result.is_valid:
            config, result = auto_correct_grouped_conv_config(config)
        config_results.append(result.is_valid)
        status = "OK" if result.is_valid else "FAIL"
        print(f"  {variant:<20} {ndim}D    {status:<8}")

    # =========================================================================
    # NumPy CPU Reference Tests
    # =========================================================================
    print("\n--- NumPy CPU Reference Tests ---")

    ref_results = []

    t0 = time.time()
    ref_results.append(test_forward_2d())
    ref_results.append(test_forward_2d_random())
    ref_results.append(test_forward_3d())
    ref_results.append(test_bwd_data_2d())
    ref_results.append(test_bwd_weight_2d())
    ref_results.append(test_fwd_bwd_consistency())
    elapsed = time.time() - t0

    print(f"\n  Reference tests completed in {elapsed:.3f}s")

    # =========================================================================
    # Summary
    # =========================================================================
    configs_ok = sum(config_results)
    refs_ok = sum(ref_results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Config validation:    {configs_ok}/{len(config_results)}")
    print(f"  CPU reference tests:  {refs_ok}/{len(ref_results)}")
    print(f"\n  Directions covered:")
    print(f"    forward    (Y = Conv(X, W))          - 2D, 3D")
    print(f"    bwd_data   (dX = ConvBwdData(dY, W)) - 2D")
    print(f"    bwd_weight (dW = ConvBwdWt(X, dY))   - 2D")
    print(f"    fwd<->bwd adjoint consistency check")

    all_ok = configs_ok == len(config_results) and refs_ok == len(ref_results)
    print(f"\n  Status: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
