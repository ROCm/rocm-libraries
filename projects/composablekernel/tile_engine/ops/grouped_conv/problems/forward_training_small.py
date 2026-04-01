#!/usr/bin/env python3

# Small training problem set for forward grouped convolution
# 20 problems that are known to work without GPU memory issues

from grouped_conv_utils import GroupedConvProblem

TRAINING_PROBLEMS_FORWARD = [
    # ResNet-style problems (first 20 from forward_training.py)
    GroupedConvProblem(N=1, C=64, K=64, G=1, Hi=56, Wi=56, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),
    GroupedConvProblem(N=1, C=64, K=64, G=1, Hi=56, Wi=56, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),
    GroupedConvProblem(N=1, C=64, K=256, G=1, Hi=56, Wi=56, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),
    GroupedConvProblem(N=1, C=256, K=64, G=1, Hi=56, Wi=56, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),
    GroupedConvProblem(N=1, C=256, K=256, G=1, Hi=56, Wi=56, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),

    GroupedConvProblem(N=1, C=128, K=128, G=1, Hi=28, Wi=28, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),
    GroupedConvProblem(N=1, C=128, K=512, G=1, Hi=28, Wi=28, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),
    GroupedConvProblem(N=1, C=512, K=128, G=1, Hi=28, Wi=28, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),
    GroupedConvProblem(N=1, C=512, K=512, G=1, Hi=28, Wi=28, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),

    GroupedConvProblem(N=1, C=256, K=256, G=1, Hi=14, Wi=14, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),
    GroupedConvProblem(N=1, C=256, K=1024, G=1, Hi=14, Wi=14, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),
    GroupedConvProblem(N=1, C=1024, K=256, G=1, Hi=14, Wi=14, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),
    GroupedConvProblem(N=1, C=1024, K=1024, G=1, Hi=14, Wi=14, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),

    GroupedConvProblem(N=1, C=512, K=512, G=1, Hi=7, Wi=7, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),
    GroupedConvProblem(N=1, C=512, K=2048, G=1, Hi=7, Wi=7, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),
    GroupedConvProblem(N=1, C=2048, K=512, G=1, Hi=7, Wi=7, Y=1, X=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, direction="forward"),

    # Batch size variations
    GroupedConvProblem(N=2, C=64, K=64, G=1, Hi=56, Wi=56, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),
    GroupedConvProblem(N=4, C=64, K=64, G=1, Hi=56, Wi=56, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),
    GroupedConvProblem(N=8, C=64, K=64, G=1, Hi=56, Wi=56, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),
    GroupedConvProblem(N=16, C=64, K=64, G=1, Hi=56, Wi=56, Y=3, X=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, direction="forward"),
]

assert len(TRAINING_PROBLEMS_FORWARD) == 20, f"Expected 20 problems, got {len(TRAINING_PROBLEMS_FORWARD)}"
