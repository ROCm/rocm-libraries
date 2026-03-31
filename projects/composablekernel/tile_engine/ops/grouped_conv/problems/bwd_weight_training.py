#!/usr/bin/env python3

# Auto-generated from ALL_CONFIGS_FULL.txt
# Training problem set for bwd_weight
# Note: dtype will be expanded by benchmark (fp16, bf16, fp32)

from grouped_conv_utils import GroupedConvProblem

TRAINING_PROBLEMS_BWD_WEIGHT = [
    # Problem 1
    GroupedConvProblem(
        N=16, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 2
    GroupedConvProblem(
        N=32, C=1, K=1, G=1,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 3
    GroupedConvProblem(
        N=64, C=1, K=1, G=1,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 4
    GroupedConvProblem(
        N=8, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 5
    GroupedConvProblem(
        N=32, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 6
    GroupedConvProblem(
        N=64, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 7
    GroupedConvProblem(
        N=1, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 8
    GroupedConvProblem(
        N=16, C=1, K=1, G=1,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 9
    GroupedConvProblem(
        N=128, C=1, K=1, G=1,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 10
    GroupedConvProblem(
        N=256, C=1, K=1, G=1,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 11
    GroupedConvProblem(
        N=2, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 12
    GroupedConvProblem(
        N=256, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 13
    GroupedConvProblem(
        N=128, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 14
    GroupedConvProblem(
        N=4, C=1, K=16, G=1,
        Hi=48, Wi=480, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 15
    GroupedConvProblem(
        N=1, C=32, K=32, G=32,
        Hi=112, Wi=112, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 16
    GroupedConvProblem(
        N=1, C=64, K=64, G=64,
        Hi=113, Wi=113, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 17
    GroupedConvProblem(
        N=1, C=128, K=128, G=128,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 18
    GroupedConvProblem(
        N=1, C=128, K=128, G=128,
        Hi=57, Wi=57, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 19
    GroupedConvProblem(
        N=1, C=256, K=256, G=256,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 20
    GroupedConvProblem(
        N=1, C=256, K=256, G=256,
        Hi=29, Wi=29, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 21
    GroupedConvProblem(
        N=128, C=1024, K=2048, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 22
    GroupedConvProblem(
        N=128, C=1024, K=256, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 23
    GroupedConvProblem(
        N=128, C=1024, K=512, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 24
    GroupedConvProblem(
        N=128, C=128, K=512, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 25
    GroupedConvProblem(
        N=128, C=2048, K=512, G=1,
        Hi=7, Wi=7, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 26
    GroupedConvProblem(
        N=128, C=256, K=1024, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 27
    GroupedConvProblem(
        N=128, C=256, K=128, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 28
    GroupedConvProblem(
        N=128, C=256, K=512, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 29
    GroupedConvProblem(
        N=128, C=256, K=64, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 30
    GroupedConvProblem(
        N=128, C=512, K=1024, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 31
    GroupedConvProblem(
        N=128, C=512, K=128, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 32
    GroupedConvProblem(
        N=128, C=512, K=256, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 33
    GroupedConvProblem(
        N=128, C=512, K=2048, G=1,
        Hi=7, Wi=7, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 34
    GroupedConvProblem(
        N=128, C=64, K=256, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 35
    GroupedConvProblem(
        N=128, C=64, K=64, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 36
    GroupedConvProblem(
        N=32, C=1024, K=1024, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 37
    GroupedConvProblem(
        N=32, C=1024, K=2048, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 38
    GroupedConvProblem(
        N=32, C=1024, K=512, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 39
    GroupedConvProblem(
        N=32, C=1024, K=2048, G=1,
        Hi=7, Wi=7, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 40
    GroupedConvProblem(
        N=32, C=128, K=256, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 41
    GroupedConvProblem(
        N=128, C=3, K=64, G=1,
        Hi=230, Wi=230, Y=7, X=7,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 42
    GroupedConvProblem(
        N=32, C=1024, K=1024, G=32,
        Hi=14, Wi=14, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 43
    GroupedConvProblem(
        N=32, C=256, K=256, G=32,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 44
    GroupedConvProblem(
        N=32, C=256, K=512, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 45
    GroupedConvProblem(
        N=32, C=3, K=64, G=1,
        Hi=224, Wi=224, Y=7, X=7,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=3, pad_w=3,
    ),
    # Problem 46
    GroupedConvProblem(
        N=64, C=2048, K=256, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=12, dilation_w=12,
        pad_h=12, pad_w=12,
    ),
    # Problem 47
    GroupedConvProblem(
        N=64, C=2048, K=256, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=24, dilation_w=24,
        pad_h=24, pad_w=24,
    ),
    # Problem 48
    GroupedConvProblem(
        N=64, C=2048, K=256, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=36, dilation_w=36,
        pad_h=36, pad_w=36,
    ),
    # Problem 49
    GroupedConvProblem(
        N=64, C=256, K=256, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=2, dilation_w=2,
        pad_h=2, pad_w=2,
    ),
    # Problem 50
    GroupedConvProblem(
        N=64, C=512, K=512, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=2, dilation_w=2,
        pad_h=2, pad_w=2,
    ),
    # Problem 51
    GroupedConvProblem(
        N=64, C=512, K=512, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=4, dilation_w=4,
        pad_h=4, pad_w=4,
    ),
    # Problem 52
    GroupedConvProblem(
        N=32, C=2048, K=256, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=12, dilation_w=12,
        pad_h=12, pad_w=12,
    ),
    # Problem 53
    GroupedConvProblem(
        N=32, C=2048, K=256, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=24, dilation_w=24,
        pad_h=24, pad_w=24,
    ),
    # Problem 54
    GroupedConvProblem(
        N=32, C=2048, K=256, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=36, dilation_w=36,
        pad_h=36, pad_w=36,
    ),
    # Problem 55
    GroupedConvProblem(
        N=32, C=256, K=256, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=2, dilation_w=2,
        pad_h=2, pad_w=2,
    ),
    # Problem 56
    GroupedConvProblem(
        N=128, C=128, K=128, G=1,
        Hi=17, Wi=17, Y=1, X=7,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=3,
    ),
    # Problem 57
    GroupedConvProblem(
        N=128, C=16, K=48, G=1,
        Hi=14, Wi=14, Y=5, X=5,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=2, pad_w=2,
    ),
    # Problem 58
    GroupedConvProblem(
        N=128, C=16, K=32, G=1,
        Hi=28, Wi=28, Y=5, X=5,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=2, pad_w=2,
    ),
    # Problem 59
    GroupedConvProblem(
        N=128, C=24, K=64, G=1,
        Hi=14, Wi=14, Y=5, X=5,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=2, pad_w=2,
    ),
    # Problem 60
    GroupedConvProblem(
        N=128, C=3, K=64, G=1,
        Hi=225, Wi=225, Y=7, X=7,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=2, pad_w=2,
    ),
    # Problem 61
    GroupedConvProblem(
        N=128, C=32, K=128, G=1,
        Hi=14, Wi=14, Y=5, X=5,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=2, pad_w=2,
    ),
    # Problem 62
    GroupedConvProblem(
        N=128, C=32, K=64, G=1,
        Hi=14, Wi=14, Y=5, X=5,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=2, pad_w=2,
    ),
    # Problem 63
    GroupedConvProblem(
        N=128, C=32, K=96, G=1,
        Hi=28, Wi=28, Y=5, X=5,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=2, pad_w=2,
    ),
    # Problem 64
    GroupedConvProblem(
        N=128, C=128, K=128, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 65
    GroupedConvProblem(
        N=128, C=256, K=256, G=1,
        Hi=14, Wi=14, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 66
    GroupedConvProblem(
        N=128, C=512, K=512, G=1,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 67
    GroupedConvProblem(
        N=32, C=1024, K=1024, G=32,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 68
    GroupedConvProblem(
        N=32, C=2048, K=1024, G=1,
        Hi=7, Wi=7, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 69
    GroupedConvProblem(
        N=32, C=256, K=256, G=32,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 70
    GroupedConvProblem(
        N=32, C=256, K=512, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 71
    GroupedConvProblem(
        N=32, C=512, K=1024, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 72
    GroupedConvProblem(
        N=32, C=512, K=512, G=32,
        Hi=14, Wi=14, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 73
    GroupedConvProblem(
        N=32, C=512, K=1024, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 74
    GroupedConvProblem(
        N=32, C=512, K=256, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 75
    GroupedConvProblem(
        N=32, C=512, K=512, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 76
    GroupedConvProblem(
        N=32, C=512, K=512, G=32,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 77
    GroupedConvProblem(
        N=32, C=1024, K=1024, G=64,
        Hi=14, Wi=14, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 78
    GroupedConvProblem(
        N=32, C=1024, K=2048, G=1,
        Hi=14, Wi=14, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 79
    GroupedConvProblem(
        N=32, C=1024, K=1024, G=64,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 80
    GroupedConvProblem(
        N=32, C=2048, K=2048, G=64,
        Hi=14, Wi=14, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 81
    GroupedConvProblem(
        N=32, C=2048, K=2048, G=1,
        Hi=7, Wi=7, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 82
    GroupedConvProblem(
        N=32, C=2048, K=2048, G=64,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 83
    GroupedConvProblem(
        N=32, C=512, K=1024, G=1,
        Hi=28, Wi=28, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 84
    GroupedConvProblem(
        N=32, C=512, K=512, G=64,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 85
    GroupedConvProblem(
        N=32, C=1024, K=1024, G=32,
        Hi=14, Wi=14, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 86
    GroupedConvProblem(
        N=32, C=1024, K=1024, G=32,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 87
    GroupedConvProblem(
        N=32, C=2048, K=2048, G=32,
        Hi=14, Wi=14, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 88
    GroupedConvProblem(
        N=32, C=2048, K=2048, G=32,
        Hi=7, Wi=7, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 89
    GroupedConvProblem(
        N=32, C=512, K=512, G=32,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 90
    GroupedConvProblem(
        N=128, C=256, K=512, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 91
    GroupedConvProblem(
        N=128, C=512, K=512, G=1,
        Hi=14, Wi=14, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 92
    GroupedConvProblem(
        N=128, C=512, K=512, G=1,
        Hi=28, Wi=28, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 93
    GroupedConvProblem(
        N=128, C=64, K=64, G=1,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 94
    GroupedConvProblem(
        N=32, C=128, K=128, G=32,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 95
    GroupedConvProblem(
        N=32, C=256, K=128, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 96
    GroupedConvProblem(
        N=32, C=256, K=256, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 97
    GroupedConvProblem(
        N=32, C=64, K=128, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 98
    GroupedConvProblem(
        N=32, C=64, K=256, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 99
    GroupedConvProblem(
        N=32, C=256, K=256, G=64,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 100
    GroupedConvProblem(
        N=32, C=256, K=512, G=1,
        Hi=56, Wi=56, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 101
    GroupedConvProblem(
        N=32, C=512, K=512, G=64,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 102
    GroupedConvProblem(
        N=32, C=256, K=256, G=32,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 103
    GroupedConvProblem(
        N=32, C=512, K=512, G=32,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 104
    GroupedConvProblem(
        N=128, C=128, K=128, G=1,
        Hi=112, Wi=112, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 105
    GroupedConvProblem(
        N=128, C=128, K=256, G=1,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 106
    GroupedConvProblem(
        N=128, C=256, K=256, G=1,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 107
    GroupedConvProblem(
        N=128, C=64, K=128, G=1,
        Hi=112, Wi=112, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 108
    GroupedConvProblem(
        N=128, C=64, K=192, G=1,
        Hi=56, Wi=56, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 109
    GroupedConvProblem(
        N=128, C=192, K=32, G=1,
        Hi=35, Wi=35, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 110
    GroupedConvProblem(
        N=128, C=192, K=48, G=1,
        Hi=35, Wi=35, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 111
    GroupedConvProblem(
        N=128, C=192, K=64, G=1,
        Hi=35, Wi=35, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 112
    GroupedConvProblem(
        N=128, C=256, K=48, G=1,
        Hi=35, Wi=35, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 113
    GroupedConvProblem(
        N=128, C=256, K=64, G=1,
        Hi=35, Wi=35, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 114
    GroupedConvProblem(
        N=128, C=288, K=384, G=1,
        Hi=35, Wi=35, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 115
    GroupedConvProblem(
        N=128, C=288, K=48, G=1,
        Hi=35, Wi=35, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 116
    GroupedConvProblem(
        N=128, C=288, K=64, G=1,
        Hi=35, Wi=35, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 117
    GroupedConvProblem(
        N=128, C=48, K=64, G=1,
        Hi=35, Wi=35, Y=5, X=5,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=2, pad_w=2,
    ),
    # Problem 118
    GroupedConvProblem(
        N=128, C=64, K=96, G=1,
        Hi=35, Wi=35, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 119
    GroupedConvProblem(
        N=128, C=64, K=80, G=1,
        Hi=73, Wi=73, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 120
    GroupedConvProblem(
        N=128, C=80, K=192, G=1,
        Hi=73, Wi=73, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 121
    GroupedConvProblem(
        N=128, C=96, K=96, G=1,
        Hi=35, Wi=35, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 122
    GroupedConvProblem(
        N=128, C=32, K=32, G=1,
        Hi=149, Wi=149, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 123
    GroupedConvProblem(
        N=128, C=3, K=64, G=1,
        Hi=224, Wi=224, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 124
    GroupedConvProblem(
        N=128, C=64, K=64, G=1,
        Hi=224, Wi=224, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 125
    GroupedConvProblem(
        N=128, C=3, K=32, G=1,
        Hi=299, Wi=299, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 126
    GroupedConvProblem(
        N=128, C=32, K=64, G=1,
        Hi=147, Wi=147, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 127
    GroupedConvProblem(
        N=64, C=3, K=32, G=1,
        Hi=299, Wi=299, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 128
    GroupedConvProblem(
        N=64, C=32, K=64, G=1,
        Hi=147, Wi=147, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 129
    GroupedConvProblem(
        N=64, C=32, K=32, G=1,
        Hi=149, Wi=149, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 130
    GroupedConvProblem(
        N=64, C=64, K=96, G=1,
        Hi=147, Wi=147, Y=3, X=3,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 131
    GroupedConvProblem(
        N=2, C=128, K=128, G=1,
        Hi=128, Wi=128, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 132
    GroupedConvProblem(
        N=2, C=128, K=512, G=1,
        Hi=128, Wi=128, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 133
    GroupedConvProblem(
        N=2, C=256, K=256, G=1,
        Hi=128, Wi=128, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 134
    GroupedConvProblem(
        N=2, C=256, K=512, G=1,
        Hi=128, Wi=128, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 135
    GroupedConvProblem(
        N=2, C=256, K=128, G=1,
        Hi=256, Wi=256, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 136
    GroupedConvProblem(
        N=2, C=256, K=256, G=1,
        Hi=256, Wi=256, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 137
    GroupedConvProblem(
        N=2, C=256, K=256, G=1,
        Hi=256, Wi=256, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 138
    GroupedConvProblem(
        N=2, C=256, K=512, G=1,
        Hi=256, Wi=256, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 139
    GroupedConvProblem(
        N=2, C=256, K=512, G=1,
        Hi=256, Wi=256, Y=3, X=3,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=1, pad_w=1,
    ),
    # Problem 140
    GroupedConvProblem(
        N=2, C=256, K=64, G=1,
        Hi=256, Wi=256, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 141
    GroupedConvProblem(
        N=2, C=512, K=1024, G=1,
        Hi=128, Wi=128, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 142
    GroupedConvProblem(
        N=2, C=512, K=12, G=1,
        Hi=128, Wi=128, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 143
    GroupedConvProblem(
        N=2, C=512, K=128, G=1,
        Hi=128, Wi=128, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 144
    GroupedConvProblem(
        N=2, C=512, K=256, G=1,
        Hi=128, Wi=128, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 145
    GroupedConvProblem(
        N=2, C=512, K=256, G=1,
        Hi=128, Wi=128, Y=1, X=1,
        stride_h=2, stride_w=2,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 146
    GroupedConvProblem(
        N=2, C=512, K=6, G=1,
        Hi=128, Wi=128, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 147
    GroupedConvProblem(
        N=2, C=512, K=12, G=1,
        Hi=256, Wi=256, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 148
    GroupedConvProblem(
        N=2, C=512, K=6, G=1,
        Hi=256, Wi=256, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 149
    GroupedConvProblem(
        N=2, C=64, K=256, G=1,
        Hi=256, Wi=256, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
    # Problem 150
    GroupedConvProblem(
        N=2, C=64, K=64, G=1,
        Hi=256, Wi=256, Y=1, X=1,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1,
        pad_h=0, pad_w=0,
    ),
]
