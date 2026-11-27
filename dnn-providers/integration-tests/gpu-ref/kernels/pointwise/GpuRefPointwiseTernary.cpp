// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuRefTypes.h"

using namespace gpu_ref;

template <int Op, typename OutputType>
struct TernaryOp
{
    static_assert(false, "This Op is not supported.");
};

// output = mask ? input0 : input1
template <typename OutputType>
struct TernaryOp<POINTWISE_TERNARY_OP_BINARY_SELECT, OutputType>
{
    template <typename X0, typename X1, typename Mask>
    static __forceinline__ __device__ OutputType impl(const X0& x0, const X1& x1, const Mask& mask)
    {
        return static_cast<bool>(mask) ? static_cast<OutputType>(x0) : static_cast<OutputType>(x1);
    }
};

extern "C" __global__ void PointwiseTernaryRef(PointwiseTernaryArgs args)
{
    auto* input0 = static_cast<const INPUT0_TYPE*>(args.input0);
    auto* input1 = static_cast<const INPUT1_TYPE*>(args.input1);
    auto* mask = static_cast<const MASK_TYPE*>(args.mask);
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

    // Decompose the flat output index into strided broadcasted accesses
    long long in0Idx = 0;
    long long in1Idx = 0;
    long long maskIdx = 0;
    long long outIdx = 0;

    long long remaining = index;
    for(int d = args.nDim - 1; d >= 0; --d)
    {
        long long coord = remaining % args.outputDims[d];
        remaining /= args.outputDims[d];
        in0Idx += coord * args.input0Strides[d];
        in1Idx += coord * args.input1Strides[d];
        maskIdx += coord * args.maskStrides[d];
        outIdx += coord * args.outputStrides[d];
    }

    output[outIdx]
        = TernaryOp<OP, OUTPUT_TYPE>::impl(input0[in0Idx], input1[in1Idx], mask[maskIdx]);
}
