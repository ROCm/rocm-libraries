// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shared argument structs for GPU reference pointwise kernels.
// Included by both device code (HipRTC) and host launch code.
// Only POD types allowed — no host or device includes.

#pragma once

// We don't use the `hipdnn_flatbuffers_sdk::data_objects::PointwiseMode` enum values directly
// because that definition isn't visible of device, and if we copied the enum values directly
// here we'd be at risk of breaking if those values changed as part of a major SDK version bump.
enum PointwiseOps
{
    POINTWISE_UNARY_OP_IDENTITY = 0,
    POINTWISE_UNARY_OP_ABS = 1,
    POINTWISE_UNARY_OP_NEG = 2,
    POINTWISE_UNARY_OP_RELU_FWD = 3,
    POINTWISE_UNARY_OP_SIGMOID_FWD = 4,
    POINTWISE_UNARY_OP_TANH_FWD = 5,
    POINTWISE_UNARY_OP_GELU_FWD = 6,
    POINTWISE_UNARY_OP_GELU_APPROX_TANH_FWD = 7,
    POINTWISE_UNARY_OP_SWISH_FWD = 8,
    POINTWISE_BINARY_OP_ADD = 9,
    POINTWISE_BINARY_OP_SUB = 10,
    POINTWISE_BINARY_OP_MUL = 11,
    POINTWISE_BINARY_OP_SIGMOID_BWD = 12,
    POINTWISE_BINARY_OP_TANH_BWD = 13,
    POINTWISE_BINARY_OP_RELU_BWD = 14,
};

struct PointwiseUnaryArgs
{
    // IO tensors
    const void* input;
    void* output;

    // Number of elements in output buffer
    long long size;

    // Broadcasting metadata, max 5 dimensions supported
    int nDim;
    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long outputDims[5];
    long long inputStrides[5];
    long long outputStrides[5];
    // NOLINTEND(modernize-avoid-c-arrays)

    // Activation operation parameters
    float lowerClip;
    float upperClip;
    float lowerSlope;
    float swishBeta;
};

struct PointwiseBinaryArgs
{
    // IO tensors
    const void* input0;
    const void* input1;
    void* output;

    // Number of elements in output buffer
    long long size;

    // Broadcasting metadata, max 5 dimensions supported
    int nDim;
    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long outputDims[5];
    long long input0Strides[5];
    long long input1Strides[5];
    long long outputStrides[5];
    // NOLINTEND(modernize-avoid-c-arrays)

    // Activation operation parameters
    float lowerClip;
    float upperClip;
    float lowerSlope;
};
