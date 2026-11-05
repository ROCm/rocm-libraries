#pragma once

#include "hipconv/conv_params.hpp"
#include "hip_util.h"
#include "mathutil.h"
#include "packed_ops.h"
#include "types.h"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace hipconv::cdna4
{

// The grouped wgrad kernels sum dW with fp32 atomics. A 16-bit dW is summed in the workspace
// and rounded once, so it carries one rounding rather than one per atomic.
// n4 counts groups of four elements; dW's element count is a multiple of 4 as C/g is.
template <typename T>
__global__ void grouped_dw_round_cdna4(T* __restrict__ dw, const float* __restrict__ sum, size_t n4)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n4)
        reinterpret_cast<packed_vec_t<T, 4>*>(dw)[i] =
            packed_convert<T>(reinterpret_cast<const packed_vec_t<float, 4>*>(sum)[i]);
}

inline size_t grouped_dw_elems(const ConvParams& par)
{
    return (size_t)par.k * par.kd * par.kh * par.kw * par.channels_per_group();
}

inline size_t grouped_dw_workspace(const ConvParams& par)
{
    return par.weight_grad_type == DataType::fp32 ? 0 : grouped_dw_elems(par) * sizeof(float);
}

// Zero the fp32 sum, have run() accumulate into it, and round it into dW if dW is 16-bit.
template <typename Run>
void run_grouped_wgrad(const ConvParams& par,
                       void* out,
                       void* workspace,
                       hipStream_t stream,
                       Run&& run)
{
    const size_t n  = grouped_dw_elems(par);
    const bool fp32 = par.weight_grad_type == DataType::fp32;
    if(!fp32 && workspace == nullptr)
        throw HipError(hipErrorInvalidValue, "grouped_wgrad's 16-bit dW needs a workspace");
    float* const sum = static_cast<float*>(fp32 ? out : workspace);
    HIP_CHECK(hipMemsetAsync(sum, 0, n * sizeof(float), stream));
    run(sum);
    if(fp32)
        return;
    constexpr unsigned threads = 256;
    const size_t n4            = n / 4;
    const unsigned blocks      = static_cast<unsigned>((n4 + threads - 1) / threads);
    if(par.weight_grad_type == DataType::bf16)
        grouped_dw_round_cdna4<<<blocks, threads, 0, stream>>>(static_cast<bf16_t*>(out), sum, n4);
    else
        grouped_dw_round_cdna4<<<blocks, threads, 0, stream>>>(static_cast<fp16_t*>(out), sum, n4);
}

} // namespace hipconv::cdna4
