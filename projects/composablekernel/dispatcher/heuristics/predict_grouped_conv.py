#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Command-line tool to predict best grouped convolution kernel for a given problem.

Usage:
    python3 predict_grouped_conv.py --N 1 --C 256 --K 512 --G 1 --Hi 14 --Wi 14 --Y 1 --X 1 --stride_h 2 --stride_w 2

Examples:
    # ResNet-50 Layer (stride 2)
    python3 predict_grouped_conv.py --N 1 --C 256 --K 512 --G 1 --Hi 56 --Wi 56 --Y 1 --X 1 --stride_h 2 --stride_w 2

    # 3x3 convolution
    python3 predict_grouped_conv.py --N 1 --C 128 --K 256 --G 1 --Hi 32 --Wi 32 --Y 3 --X 3

    # Depthwise convolution (G=C)
    python3 predict_grouped_conv.py --N 1 --C 128 --K 128 --G 128 --Hi 32 --Wi 32 --Y 3 --X 3
"""

import argparse
from pathlib import Path

from predict import Predictor
from feature_engine_grouped_conv import GroupedConvFeatureEngine


# Available kernel configurations (from training data)
AVAILABLE_KERNELS = [
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv4'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv5'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv3'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv4'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv5'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv4'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv5'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv3'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv4'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv5'},
    {'block_size': 32, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 32, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv4'},
    {'block_size': 32, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv5'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv4'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv5'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv3'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv4'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv5'},
    {'block_size': 64, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 64, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv4'},
    {'block_size': 64, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv5'},
    {'block_size': 128, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv3'},
    {'block_size': 128, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv4'},
    {'block_size': 128, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv5'},
    {'block_size': 128, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 128, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv4'},
    {'block_size': 128, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv5'},
]


def main():
    parser = argparse.ArgumentParser(
        description="Predict best grouped convolution kernel using ML heuristic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Problem parameters
    parser.add_argument("--N", type=int, required=True, help="Batch size")
    parser.add_argument("--C", type=int, required=True, help="Input channels")
    parser.add_argument("--K", type=int, required=True, help="Output channels")
    parser.add_argument("--G", type=int, default=1, help="Number of groups (default: 1)")
    parser.add_argument("--Hi", type=int, required=True, help="Input height")
    parser.add_argument("--Wi", type=int, required=True, help="Input width")
    parser.add_argument("--Y", type=int, default=1, help="Filter height (default: 1)")
    parser.add_argument("--X", type=int, default=1, help="Filter width (default: 1)")
    parser.add_argument("--stride_h", type=int, default=1, help="Stride height (default: 1)")
    parser.add_argument("--stride_w", type=int, default=1, help="Stride width (default: 1)")
    parser.add_argument("--pad_h", type=int, default=0, help="Padding height (default: 0)")
    parser.add_argument("--pad_w", type=int, default=0, help="Padding width (default: 0)")

    # Model parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/grouped_conv_forward_bf16_gfx950",
        help="Model directory (default: models/grouped_conv_forward_bf16_gfx950)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top kernels to show (default: 5)",
    )

    args = parser.parse_args()

    # Build problem dictionary
    problem = {
        'N': args.N,
        'C': args.C,
        'K': args.K,
        'G': args.G,
        'Hi': args.Hi,
        'Wi': args.Wi,
        'Y': args.Y,
        'X': args.X,
        'stride_h': args.stride_h,
        'stride_w': args.stride_w,
        'pad_h': args.pad_h,
        'pad_w': args.pad_w,
        'dtype': 'bf16',
    }

    # Compute output dimensions
    Ho = (args.Hi + 2 * args.pad_h - args.Y) // args.stride_h + 1
    Wo = (args.Wi + 2 * args.pad_w - args.X) // args.stride_w + 1

    # Compute GEMM dimensions
    gemm_m = args.N * Ho * Wo
    gemm_n = args.K
    gemm_k = (args.C // args.G) * args.Y * args.X

    print("=" * 80)
    print("GROUPED CONVOLUTION - ML HEURISTIC KERNEL SELECTION")
    print("=" * 80)
    print()
    print(f"Problem:")
    print(f"  N={args.N}, C={args.C}, K={args.K}, G={args.G}")
    print(f"  Input: {args.Hi}x{args.Wi}, Filter: {args.Y}x{args.X}")
    print(f"  Stride: {args.stride_h}x{args.stride_w}, Padding: {args.pad_h}x{args.pad_w}")
    print(f"  Output: {Ho}x{Wo}")
    print()
    print(f"GEMM Equivalent:")
    print(f"  M={gemm_m:,} (N*Ho*Wo), N={gemm_n:,} (K), K={gemm_k:,} (C/G*Y*X)")
    print()

    # Load model
    model_dir = Path(args.model_dir)
    feature_engine = GroupedConvFeatureEngine()
    predictor = Predictor(model_dir, feature_engine=feature_engine)

    print(f"✓ Loaded model from {model_dir}")
    print()

    # Predict for all available kernels
    predictions = []
    for kernel in AVAILABLE_KERNELS:
        pred_tflops = predictor.predict_tflops(problem, kernel)
        predictions.append({
            'tflops': pred_tflops,
            'block_size': kernel['block_size'],
            'gemm_m': kernel['gemm_m_per_block'],
            'gemm_n': kernel['gemm_n_per_block'],
            'pipeline': kernel['pipeline'],
        })

    # Sort by predicted TFLOPS
    predictions.sort(key=lambda x: x['tflops'], reverse=True)

    # Show top K
    print(f"Top {args.top_k} Predicted Kernels:")
    print()
    print(f"{'Rank':<6} {'Block':<8} {'Tile M':<8} {'Tile N':<8} {'Pipeline':<12} {'Predicted TFLOPS':<20}")
    print("-" * 80)

    for i, pred in enumerate(predictions[:args.top_k], 1):
        print(f"{i:<6} {pred['block_size']:<8} {pred['gemm_m']:<8} {pred['gemm_n']:<8} "
              f"{pred['pipeline']:<12} {pred['tflops']:<20.2f}")

    print()
    best = predictions[0]
    print(f"✓ Best Kernel: block={best['block_size']}, tile={best['gemm_m']}x{best['gemm_n']}, {best['pipeline']}")
    print(f"  Predicted TFLOPS: {best['tflops']:.2f}")
    print()


if __name__ == "__main__":
    main()
