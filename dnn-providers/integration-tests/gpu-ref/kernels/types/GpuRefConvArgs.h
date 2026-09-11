// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shared argument structs for GPU reference convolution kernels.
// Included by both device code (HipRTC) and host launch code.
// Only POD types allowed — no host or device includes.

#pragma once

// --- Stride structs for stride-based indexing ---

// NOLINTBEGIN(modernize-avoid-c-arrays)
struct Strides3
{
    long long s[3];
};

struct Strides4
{
    long long s[4];
};

struct Strides5
{
    long long s[5];
};
// NOLINTEND(modernize-avoid-c-arrays)

// --- Convolution forward argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming)
struct ConvFwdArgs1d
{
    const void* x;
    const void* w;
    void* y;
    Strides3 xStr;
    Strides3 wStr;
    Strides3 yStr;
    long long N, C, Wi;
    long long K, Wo;
    long long Kw;
    long long strideW;
    long long dilW;
    long long padW;
    long long groups;
    double alpha, beta;
};

struct ConvFwdArgs2d
{
    const void* x;
    const void* w;
    void* y;
    Strides4 xStr;
    Strides4 wStr;
    Strides4 yStr;
    long long N, C, Hi, Wi;
    long long K, Ho, Wo;
    long long Kh, Kw;
    long long strideH, strideW;
    long long dilH, dilW;
    long long padH, padW;
    long long groups;
    double alpha, beta;
};

struct ConvFwdArgs3d
{
    const void* x;
    const void* w;
    void* y;
    Strides5 xStr;
    Strides5 wStr;
    Strides5 yStr;
    long long N, C, Di, Hi, Wi;
    long long K, Do, Ho, Wo;
    long long Kd, Kh, Kw;
    long long strideD, strideH, strideW;
    long long dilD, dilH, dilW;
    long long padD, padH, padW;
    long long groups;
    double alpha, beta;
};

// --- Convolution backward data (dgrad) argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct ConvBwdArgs1d
{
    void* dx;
    const void* w;
    const void* dy;
    Strides3 dxStr;
    Strides3 wStr;
    Strides3 dyStr;
    long long N, C, Wi;
    long long K, Wo;
    long long Kw;
    long long strideW;
    long long dilW;
    long long padW;
    long long groups;
    double alpha, beta;
};

struct ConvBwdArgs2d
{
    void* dx;
    const void* w;
    const void* dy;
    Strides4 dxStr;
    Strides4 wStr;
    Strides4 dyStr;
    long long N, C, Hi, Wi;
    long long K, Ho, Wo;
    long long Kh, Kw;
    long long strideH, strideW;
    long long dilH, dilW;
    long long padH, padW;
    long long groups;
    double alpha, beta;
};

struct ConvBwdArgs3d
{
    void* dx;
    const void* w;
    const void* dy;
    Strides5 dxStr;
    Strides5 wStr;
    Strides5 dyStr;
    long long N, C, Di, Hi, Wi;
    long long K, Do, Ho, Wo;
    long long Kd, Kh, Kw;
    long long strideD, strideH, strideW;
    long long dilD, dilH, dilW;
    long long padD, padH, padW;
    long long groups;
    double alpha, beta;
};

// --- Convolution backward weight (wgrad) argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct ConvWrwArgs1d
{
    const void* x;
    void* dw;
    const void* dy;
    Strides3 xStr;
    Strides3 dwStr;
    Strides3 dyStr;
    long long N, C, Wi;
    long long K, Wo;
    long long Kw;
    long long strideW;
    long long dilW;
    long long padW;
    long long groups;
    double alpha, beta;
};

struct ConvWrwArgs2d
{
    const void* x;
    void* dw;
    const void* dy;
    Strides4 xStr;
    Strides4 dwStr;
    Strides4 dyStr;
    long long N, C, Hi, Wi;
    long long K, Ho, Wo;
    long long Kh, Kw;
    long long strideH, strideW;
    long long dilH, dilW;
    long long padH, padW;
    long long groups;
    double alpha, beta;
};

struct ConvWrwArgs3d
{
    const void* x;
    void* dw;
    const void* dy;
    Strides5 xStr;
    Strides5 dwStr;
    Strides5 dyStr;
    long long N, C, Di, Hi, Wi;
    long long K, Do, Ho, Wo;
    long long Kd, Kh, Kw;
    long long strideD, strideH, strideW;
    long long dilD, dilH, dilW;
    long long padD, padH, padW;
    long long groups;
    double alpha, beta;
};

// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming)
