// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shared argument structs for GPU reference convolution kernels.
// Included by both device code (HipRTC) and host launch code.
// Only POD types allowed — no host or device includes.

#pragma once

#define POINTWISE_UNARY_OP_IDENTITY 0
#define POINTWISE_UNARY_OP_ABS 1
#define POINTWISE_UNARY_OP_NEG 2
#define POINTWISE_UNARY_OP_RELU_FWD 3
#define POINTWISE_UNARY_OP_SIGMOID_FWD 4
#define POINTWISE_UNARY_OP_TANH_FWD 5
#define POINTWISE_UNARY_OP_GELU_FWD 6
#define POINTWISE_UNARY_OP_GELU_APPROX_TANH_FWD 7
#define POINTWISE_UNARY_OP_SWISH_FWD 8
#define POINTWISE_BINARY_OP_ADD 9
#define POINTWISE_BINARY_OP_SUB 10
#define POINTWISE_BINARY_OP_MUL 11
#define POINTWISE_BINARY_OP_SIGMOID_BWD 12
#define POINTWISE_BINARY_OP_TANH_BWD 13
#define POINTWISE_BINARY_OP_RELU_BWD 14

struct PointwiseUnaryArgs
{
    // In PointwiseBinaryArgs
    const void* input;
    void* output;

    // Number of elements in output buffer
    long long size;

    // Broadcasting metadata, max 5 dimensions supported
    int nDim;
    long long outputDims[5];
    long long inputStrides[5];
    long long outputStrides[5];

    // Activation operation parameters
    float lowerClip;
    float upperClip;
    float lowerSlope;
    float swishBeta;
};

struct PointwiseBinaryArgs
{
    // IO buffers
    const void* input0;
    const void* input1;
    void* output;

    // Number of elements in output buffer
    long long size;

    // Broadcasting metadata, max 5 dimensions supported
    int nDim;
    long long outputDims[5];
    long long input0Strides[5];
    long long input1Strides[5];
    long long outputStrides[5];

    // Activation operation parameters
    float lowerClip;
    float upperClip;
    float lowerSlope;
};
