#include "reduce_generator.hpp"

#include "miopen/tensor.hpp"
#include "../device_prng.hpp"
#include "../gtest_hip_utilities.hpp"

namespace test::gtest {

template <typename T>
__device__ __forceinline__ T CastFromDouble(double x)
{
    if constexpr(std::is_same_v<T, __half> || std::is_same_v<T, hip_bfloat16>)
        return static_cast<T>(static_cast<float>(x));
    else
        return static_cast<T>(x);
}

template <typename T>
__device__ __forceinline__ T MakeReduceValue(uint32_t primary_raw,
                                             uint32_t secondary_raw,
                                             size_t index,
                                             const TensorShapeInfo& shape,
                                             miopenReduceTensorOp_t op,
                                             uint64_t max_val)
{
    const double sign = checkerboard_sign(index, shape);
    const double rand_val =
        static_cast<double>(max_val > 0 ? (static_cast<uint64_t>(primary_raw) % max_val) : 0ULL);

    double final_val = 0.0;

    switch(op)
    {
    // Data generation used by ADD/AVG, data is distributed around 1.0 rather than 0.0,
    // very low probability to get a reduced result of zero-value.
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_AVG:
        if(max_val > 0)
            final_val = sign * (rand_val / static_cast<double>(max_val)) + 0.01;
        else
            final_val = 0.01;
        break;

    // Special data generation for MUL, to avoid all-zero and large accumulative error in the
    // reduced result.
    case MIOPEN_REDUCE_TENSOR_MUL:
        if(max_val > 0)
        {
            const double mv = static_cast<double>(max_val);
            final_val       = (sign > 0.0) ? (rand_val + mv) / (rand_val + mv + 1.0)
                                           : (rand_val + mv + 1.0) / (rand_val + mv);
        }
        else
        {
            final_val = 1.0;
        }
        break;

    // Special data generation for AMAX, no zero value used.
    case MIOPEN_REDUCE_TENSOR_AMAX:
        final_val = (sign > 0.0) ? (rand_val + 0.5) : (-1.0 * rand_val - 0.5);
        break;

    // Special data generation for NORM1 and NORM2 using a large value space.
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_NORM2: {
        const double rand_ratio =
            0.1 + 0.9 * static_cast<double>(secondary_raw) * (1.0 / 4294967296.0); // [0.1, 1.0)
        final_val = rand_val * sign * rand_ratio;
        break;
    }

    case MIOPEN_REDUCE_TENSOR_MIN:
    case MIOPEN_REDUCE_TENSOR_MAX:
    default: final_val = rand_val * sign; break;
    }

    return CastFromDouble<T>(final_val);
}

template <typename T>
__global__ void ReduceGeneratorKernel(T* output,
                                      size_t n,
                                      uint64_t seed,
                                      miopenReduceTensorOp_t op,
                                      uint64_t max_val,
                                      TensorShapeInfo shape)
{
    const size_t logical_tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t logical_stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for(size_t chunk = logical_tid;; chunk += logical_stride)
    {
        const size_t base = chunk * 4;
        if(base >= n)
            break;

        rocrand_state_philox4x32_10 state;
        rocrand_init(seed, static_cast<uint64_t>(chunk), 0ULL, &state);

        const uint4 primary = rocrand4(&state);
        uint4 secondary{};
        if(op == MIOPEN_REDUCE_TENSOR_NORM1 || op == MIOPEN_REDUCE_TENSOR_NORM2)
            secondary = rocrand4(&state);

#pragma unroll
        for(int lane = 0; lane < 4; ++lane)
        {
            const size_t i = base + static_cast<size_t>(lane);
            if(i >= n)
                break;

            output[i] = MakeReduceValue<T>(
                uint4_get(primary, lane), uint4_get(secondary, lane), i, shape, op, max_val);
        }
    }
}

template <typename T>
void ReduceGenerator(T* output,
                     size_t n,
                     uint64_t seed,
                     miopenReduceTensorOp_t op,
                     uint64_t max_val,
                     const int* lengths,
                     int ndims,
                     hipStream_t stream)
{
    if(n == 0)
        return;

    constexpr int threadsPerBlock = 256;
    // 4 because we generate 4 values via rocrand4
    const size_t chunk_count    = (n + 3) / 4;
    const int numBlocks         = ComputeNumBlocks<T>(chunk_count);
    const TensorShapeInfo shape = MakeTensorShapeInfo(lengths, ndims);

    using DeviceT = typename ToDeviceType<T>::type;
    ReduceGeneratorKernel<DeviceT><<<numBlocks, threadsPerBlock, 0, stream>>>(
        reinterpret_cast<DeviceT*>(output), n, seed, op, max_val, shape);

    MIOPEN_GTEST_HIP_ERROR(hipGetLastError(), "ReduceGeneratorKernel launch failed");
}

template <typename T>
void ReduceGenerator(T* output,
                     size_t n,
                     uint64_t seed,
                     miopenReduceTensorOp_t op,
                     uint64_t max_val,
                     const miopen::TensorDescriptor& desc,
                     hipStream_t stream)
{
    const auto shape = MakeTensorShapeInfo(desc);
    ReduceGenerator(output, n, seed, op, max_val, shape.lens, shape.ndims, stream);
}

template void ReduceGenerator<double>(double* output,
                                      size_t n,
                                      uint64_t seed,
                                      miopenReduceTensorOp_t op,
                                      uint64_t max_val,
                                      const miopen::TensorDescriptor& desc,
                                      hipStream_t stream);
template void ReduceGenerator<float>(float* output,
                                     size_t n,
                                     uint64_t seed,
                                     miopenReduceTensorOp_t op,
                                     uint64_t max_val,
                                     const miopen::TensorDescriptor& desc,
                                     hipStream_t stream);
template void ReduceGenerator<bfloat16>(bfloat16* output,
                                        size_t n,
                                        uint64_t seed,
                                        miopenReduceTensorOp_t op,
                                        uint64_t max_val,
                                        const miopen::TensorDescriptor& desc,
                                        hipStream_t stream);
template void ReduceGenerator<half_float::half>(half_float::half* output,
                                                size_t n,
                                                uint64_t seed,
                                                miopenReduceTensorOp_t op,
                                                uint64_t max_val,
                                                const miopen::TensorDescriptor& desc,
                                                hipStream_t stream);
template void ReduceGenerator<int8_t>(int8_t* output,
                                      size_t n,
                                      uint64_t seed,
                                      miopenReduceTensorOp_t op,
                                      uint64_t max_val,
                                      const miopen::TensorDescriptor& desc,
                                      hipStream_t stream);
template void ReduceGenerator<int32_t>(int32_t* output,
                                       size_t n,
                                       uint64_t seed,
                                       miopenReduceTensorOp_t op,
                                       uint64_t max_val,
                                       const miopen::TensorDescriptor& desc,
                                       hipStream_t stream);

} // namespace test::gtest
