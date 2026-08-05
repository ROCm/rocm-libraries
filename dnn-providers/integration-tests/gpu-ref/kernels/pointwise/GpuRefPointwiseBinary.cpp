// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuRefTypes.h"

using namespace gpu_ref;

template <int Op, typename ComputeType, typename OutputType>
struct BinaryOp
{
    static_assert(false, "This Op is not supported.");
};

template <typename ComputeType, typename OutputType>
struct BinaryOp<POINTWISE_BINARY_OP_ADD, ComputeType, OutputType>
{
    __device__ BinaryOp(float, float, float) {}

    template <typename X0, typename X1>
    static __forceinline__ __device__ OutputType impl(const X0& x0, const X1& x1)
    {
        ComputeType x0Compute = static_cast<ComputeType>(x0);
        ComputeType x1Compute = static_cast<ComputeType>(x1);
        return static_cast<OutputType>(x0Compute + x1Compute);
    }
};

template <typename ComputeType, typename OutputType>
struct BinaryOp<POINTWISE_BINARY_OP_SUB, ComputeType, OutputType>
{
    __device__ BinaryOp(float, float, float) {}

    template <typename X0, typename X1>
    static __forceinline__ __device__ OutputType impl(const X0& x0, const X1& x1)
    {
        ComputeType x0Compute = static_cast<ComputeType>(x0);
        ComputeType x1Compute = static_cast<ComputeType>(x1);
        return static_cast<OutputType>(x0Compute - x1Compute);
    }
};

template <typename ComputeType, typename OutputType>
struct BinaryOp<POINTWISE_BINARY_OP_MUL, ComputeType, OutputType>
{
    __device__ BinaryOp(float, float, float) {}

    template <typename X0, typename X1>
    static __forceinline__ __device__ OutputType impl(const X0& x0, const X1& x1)
    {
        ComputeType x0Compute = static_cast<ComputeType>(x0);
        ComputeType x1Compute = static_cast<ComputeType>(x1);
        return static_cast<OutputType>(x0Compute * x1Compute);
    }
};

template <typename ComputeType, typename OutputType>
struct BinaryOp<POINTWISE_BINARY_OP_SIGMOID_BWD, ComputeType, OutputType>
{
    __device__ BinaryOp(float, float, float) {}

    template <typename Dy, typename X>
    static __forceinline__ __device__ OutputType impl(const Dy& dy, const X& x)
    {
        ComputeType dyCompute = static_cast<ComputeType>(dy);
        ComputeType xCompute = static_cast<ComputeType>(x);

        ComputeType sigmoidVal = ComputeType{1} / (ComputeType{1} + exp(-xCompute));
        auto localGradient = sigmoidVal * (ComputeType{1} - sigmoidVal);

        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType, typename OutputType>
struct BinaryOp<POINTWISE_BINARY_OP_TANH_BWD, ComputeType, OutputType>
{
    __device__ BinaryOp(float, float, float) {}

    template <typename Dy, typename X>
    static __forceinline__ __device__ OutputType impl(const Dy& dy, const X& x)
    {
        ComputeType dyCompute = static_cast<ComputeType>(dy);
        ComputeType xCompute = static_cast<ComputeType>(x);

        ComputeType tanhVal = tanh(xCompute);
        auto localGradient = ComputeType{1} - (tanhVal * tanhVal);
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType, typename OutputType>
struct BinaryOp<POINTWISE_BINARY_OP_RELU_BWD, ComputeType, OutputType>
{
    ComputeType _lowerClip;
    ComputeType _upperClip;
    ComputeType _lowerSlope;

    __device__ BinaryOp(float lowerClip, float upperClip, float lowerSlope)
        : _lowerClip(lowerClip)
        , _upperClip(upperClip)
        , _lowerSlope(lowerSlope)
    {
    }

    template <typename Dy, typename X>
    __forceinline__ __device__ OutputType impl(const Dy& dy, const X& x)
    {
        ComputeType dyCompute = static_cast<ComputeType>(dy);
        ComputeType xCompute = static_cast<ComputeType>(x);

        ComputeType localGradient;
        if(xCompute <= _lowerClip)
        {
            localGradient = _lowerSlope;
        }
        else if(xCompute > _upperClip)
        {
            localGradient = ComputeType{0};
        }
        else
        {
            localGradient = ComputeType{1};
        }
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

extern "C" __global__ void PointwiseBinaryRef(PointwiseBinaryArgs args)
{
    auto* input0 = static_cast<const INPUT0_TYPE*>(args.input0);
    auto* input1 = static_cast<const INPUT1_TYPE*>(args.input1);
    auto* output = static_cast<OUTPUT_TYPE*>(args.output);
    long long totalSize = args.size;

    constexpr long long localSize = LOCAL_SIZE;
    long long lid = threadIdx.x;
    long long gid = blockIdx.x;

    long long index = lid + localSize * gid;
    if(index >= totalSize)
    {
        return;
    }

    // Decompose the float output index into strided broadcasted accesses
    long long in0Idx = 0;
    long long in1Idx = 0;
    long long outIdx = 0;

    // This loop could be unrolled using macros set at compile time for ndim and strides,
    // however prefer save that implementation for a hipdnn backend plugin implementation
    // rather than the reference.
    long long remaining = index;
    for(int d = args.nDim - 1; d >= 0; --d)
    {
        long long coord = remaining % args.outputDims[d];
        remaining /= args.outputDims[d];
        in0Idx += coord * args.input0Strides[d];
        in1Idx += coord * args.input1Strides[d];
        outIdx += coord * args.outputStrides[d];
    }

    output[outIdx]
        = BinaryOp<OP, COMPUTE_TYPE, OUTPUT_TYPE>(args.lowerClip, args.upperClip, args.lowerSlope)
              .impl(input0[in0Idx], input1[in1Idx]);
}
