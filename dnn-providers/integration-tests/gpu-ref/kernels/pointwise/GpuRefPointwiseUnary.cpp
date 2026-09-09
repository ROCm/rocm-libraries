// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuRefTypes.h"

using namespace gpu_ref;

template <int Op, typename ComputeType, typename OutputType>
struct UnaryOp
{
    static_assert(false, "This Op is not supported.");
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_IDENTITY, ComputeType, OutputType>
{
    __device__ UnaryOp(float, float, float, float) {}

    template <typename X>
    static __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);
        return static_cast<OutputType>(xCompute);
    }
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_ABS, ComputeType, OutputType>
{
    __device__ UnaryOp(float, float, float, float) {}

    template <typename X>
    static __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);
        ComputeType result = fabs(xCompute);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_NEG, ComputeType, OutputType>
{
    __device__ UnaryOp(float, float, float, float) {}

    template <typename X>
    static __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);
        ComputeType result = -xCompute;
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_RELU_FWD, ComputeType, OutputType>
{
    const ComputeType _lowerClip;
    const ComputeType _upperClip;
    const ComputeType _lowerSlope;

    __device__ UnaryOp(float lowerClip, float upperClip, float lowerSlope, float)
        : _lowerClip(lowerClip)
        , _upperClip(upperClip)
        , _lowerSlope(lowerSlope)
    {
    }

    template <typename X>
    __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);

        ComputeType result;
        if(xCompute <= _lowerClip)
        {
            result = (_lowerSlope * (xCompute - _lowerClip)) + _lowerClip;
        }
        else if(xCompute >= _upperClip)
        {
            result = _upperClip;
        }
        else
        {
            result = xCompute;
        }
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_SIGMOID_FWD, ComputeType, OutputType>
{
    __device__ UnaryOp(float, float, float, float) {}

    template <typename X>
    static __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);
        auto result = ComputeType{1} / (ComputeType{1} + exp(-xCompute));
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_TANH_FWD, ComputeType, OutputType>
{
    __device__ UnaryOp(float, float, float, float) {}

    template <typename X>
    static __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);
        ComputeType result = tanh(xCompute);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_GELU_FWD, ComputeType, OutputType>
{
    __device__ UnaryOp(float, float, float, float) {}

    static constexpr ComputeType s_kSqrt2 = ComputeType{1.4142135623731};

    template <typename X>
    static __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);

        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        ComputeType result
            = ComputeType{0.5} * xCompute * (ComputeType{1} + erf(xCompute / s_kSqrt2));
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_GELU_APPROX_TANH_FWD, ComputeType, OutputType>
{
    __device__ UnaryOp(float, float, float, float) {}

    static constexpr ComputeType s_kCoeff = ComputeType{0.044715};
    static constexpr ComputeType s_kSqrt2OverPi = ComputeType{0.797884561258723};

    template <typename X>
    static __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);
        ComputeType inner = s_kSqrt2OverPi * (xCompute + s_kCoeff * xCompute * xCompute * xCompute);
        ComputeType result = ComputeType{0.5} * xCompute * (ComputeType{1} + tanh(inner));
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType>
struct UnaryOp<POINTWISE_UNARY_OP_SWISH_FWD, ComputeType, OutputType>
{
    const ComputeType _beta;

    __device__ UnaryOp(float, float, float, float beta)
        : _beta(beta)
    {
    }

    template <typename X>
    __forceinline__ __device__ OutputType impl(const X& x)
    {
        ComputeType xCompute = static_cast<ComputeType>(x);
        ComputeType result = xCompute / (ComputeType{1} + exp(-_beta * xCompute));
        return static_cast<OutputType>(result);
    }
};

extern "C" __global__ void PointwiseUnaryRef(PointwiseUnaryArgs args)
{
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
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
    long long inIdx = 0;
    long long outIdx = 0;

    // This loop could be unrolled using macros set at compile time for ndim and strides,
    // however prefer save that implementation for a hipdnn backend plugin implementation
    // rather than the reference.
    long long remaining = index;
    for(int d = args.nDim - 1; d >= 0; --d)
    {
        long long coord = remaining % args.outputDims[d];
        remaining /= args.outputDims[d];
        inIdx += coord * args.inputStrides[d];
        outIdx += coord * args.outputStrides[d];
    }

    output[outIdx] = UnaryOp<OP, COMPUTE_TYPE, OUTPUT_TYPE>(
                         args.lowerClip, args.upperClip, args.lowerSlope, args.swishBeta)
                         .impl(input[inIdx]);
}
