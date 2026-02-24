#include "device_prng.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

namespace test::gtest {

/**
 * @brief SplitMix64 step function for robust seeding.
 */
__device__ __forceinline__ static uint64_t splitmix64_step(uint64_t& state)
{
    // 0x9E37... is the Golden Ratio constant (2^64 / phi).
    // It provides a uniform distribution for the Weyl sequence to prevent zero-state lockups.
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);

    // Mixing constants (0xBF58... and 0x94D0...) and shift values (30, 27, 31).
    // These are specific to SplitMix64 to ensure sufficient bit avalanche.
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct sfc64_state
{
    uint64_t a, b, c, counter;

    /**
     * @brief Seeds the SFC64 state.
     * @param seed_val The global seed.
     * @param index The unique element index (ensures grid-invariance).
     */
    __device__ __forceinline__ void seed(uint64_t seed_val, uint64_t index)
    {
        // 0x9E37... used again here to mix the index into the seed cleanly.
        uint64_t x = seed_val ^ (index * 0x9E3779B97F4A7C15ull);
        a          = splitmix64_step(x);
        b          = splitmix64_step(x);
        c          = splitmix64_step(x);
        counter    = 1;
        // 12 rounds: The required warm-up for SFC64 to escape low-entropy states.
        for(int i = 0; i < 12; ++i)
            (void)next();
    }

    __device__ __forceinline__ uint64_t next()
    {
        const uint64_t tmp = a + b + counter++;
        // Rotation constants specific to SFC64 state transition (11, 3, 24, 40).
        a = b ^ (b >> 11);
        b = c + (c << 3);
        c = ((c << 24) | (c >> 40)) + tmp;
        return tmp;
    }
};

template <typename T>
__global__ void sfc64_kernel(T* out, size_t n, uint64_t seed)
{
    size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for(size_t i = tid; i < n; i += stride)
    {
        sfc64_state rng;
        rng.seed(seed, static_cast<uint64_t>(i));
        uint64_t raw = rng.next();

        if constexpr(std::is_same_v<T, float>)
        {
            // raw >> 40 keeps the top 24 bits (FP32 mantissa size).
            // 1.0f / 16777216.0f is 2^-24, normalizing the integer to [0, 1).
            out[i] = static_cast<float>(raw >> 40) * (1.0f / 16777216.0f);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            // raw >> 11 keeps the top 53 bits (FP64 mantissa size).
            // 1.0 / 9007... is 2^-53, normalizing the integer to [0, 1).
            out[i] = static_cast<double>(raw >> 11) * (1.0 / 9007199254740992.0);
        }
        // We convert to float first, then use the specific GPU-to-type conversion
        else if constexpr(std::is_same_v<T, hip_bfloat16>)
        {
            // raw >> 57 keeps top 7 bits (BFloat16 mantissa size).
            // 1.0f / 128.0f is 2^-7.
            float f_val = static_cast<float>(raw >> 57) * (1.0f / 128.0f);
            out[i]      = static_cast<T>(f_val);
        }
        else if constexpr(std::is_same_v<T, __half>)
        {
            // raw >> 54 keeps top 10 bits (FP16 mantissa size).
            // 1.0f / 1024.0f is 2^-10.
            float f_val = static_cast<float>(raw >> 54) * (1.0f / 1024.0f);
            out[i]      = static_cast<T>(f_val);
        }
        else if constexpr(std::is_integral_v<T>)
        {
            constexpr int shift = 64 - (sizeof(T) * 8);
            out[i]              = static_cast<T>(raw >> shift);
        }
        else
        {
            out[i] = static_cast<T>(raw);
        }
    }
}

template <typename T>
struct ToDeviceType
{
    using type = T;
};

// Map half_float::half -> __half
template <>
struct ToDeviceType<half_float::half>
{
    using type = __half;
};

// Map miopen::bfloat16 -> __bfloat16
template <>
struct ToDeviceType<bfloat16>
{
    using type = hip_bfloat16;
};

template <typename T>
void RandomizeBuffer(T* dev_ptr, size_t size, uint64_t seed, hipStream_t stream)
{
    if(size == 0)
        return;

    int deviceId;
    hipGetDevice(&deviceId);

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, deviceId);

    const int threadsPerBlock = 256; // Standard block size for occupancy.

    // Heuristic: 4 blocks per CU allows enough warps to hide memory latency.
    int numBlocks = props.multiProcessorCount * 4;

    // Guard for small buffers
    if((size_t)numBlocks * threadsPerBlock > size)
    {
        numBlocks = (static_cast<int>(size) + threadsPerBlock - 1) / threadsPerBlock;
    }

    using DeviceT = typename ToDeviceType<T>::type;
    sfc64_kernel<DeviceT><<<numBlocks, threadsPerBlock, 0, stream>>>(
        reinterpret_cast<DeviceT*>(dev_ptr), size, seed);
}

template void
RandomizeBuffer<double>(double* dev_ptr, size_t size, uint64_t seed, hipStream_t stream);
template void
RandomizeBuffer<float>(float* dev_ptr, size_t size, uint64_t seed, hipStream_t stream);
template void
RandomizeBuffer<bfloat16>(bfloat16* dev_ptr, size_t size, uint64_t seed, hipStream_t stream);
template void RandomizeBuffer<half_float::half>(half_float::half* dev_ptr,
                                                size_t size,
                                                uint64_t seed,
                                                hipStream_t stream);
template void
RandomizeBuffer<int8_t>(int8_t* dev_ptr, size_t size, uint64_t seed, hipStream_t stream);
template void
RandomizeBuffer<int32_t>(int32_t* dev_ptr, size_t size, uint64_t seed, hipStream_t stream);

template <typename T>
__global__ void ReduceGeneratorKernel(
    T* output, size_t n, uint64_t seed, miopenReduceTensorOp_t op, uint64_t max_val)
{
    size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for(size_t i = tid; i < n; i += stride)
    {
        sfc64_state s;
        s.seed(seed, static_cast<uint64_t>(i));
        uint64_t raw = s.next();

        // Checkboard sign pattern
        double sign = (i % 2 == 0) ? 1.0 : -1.0;

        double rand_val  = static_cast<double>(max_val > 0 ? (raw % max_val) : 0);
        double final_val = 0.0;

        switch(op)
        {
        // data generation used by ADD/AVG, data is distributed around 1.0 rather than 0.0,
        // very low probability to get a reduced result of zero-value
        case MIOPEN_REDUCE_TENSOR_ADD:
        case MIOPEN_REDUCE_TENSOR_AVG:
            if(max_val > 0)
                // 0.01: Offset to avoid zero-values which can hide bugs in reduction.
                final_val = sign * (rand_val / static_cast<double>(max_val)) + 0.01;
            else
                final_val = 0.01;
            break;

        // Special data generation for MUL, to avoid all-zero and large accumulative error in the
        // reduced result
        case MIOPEN_REDUCE_TENSOR_MUL:
            if(max_val > 0)
            {
                double mv = static_cast<double>(max_val);
                // 1.0: Center values around 1.0 to prevent underflow/overflow in large products.
                final_val = (sign > 0.0) ? (rand_val + mv) / (rand_val + mv + 1.0)
                                         : (rand_val + mv + 1.0) / (rand_val + mv);
            }
            else
            {
                final_val = 1.0;
            }
            break;

        // Special data generation for AMAX, no zero value used
        case MIOPEN_REDUCE_TENSOR_AMAX:
            // 0.5: Offset to ensure no zero values.
            final_val = (sign > 0.0) ? (rand_val + 0.5) : (-1.0 * rand_val - 0.5);
            break;

        case MIOPEN_REDUCE_TENSOR_NORM1:
        case MIOPEN_REDUCE_TENSOR_NORM2:
            // 0.1 and 10: Generates values in range [0.1, 1.0] in discrete steps.
            final_val = rand_val * sign * (0.1 * (1 + (raw % 10)));
            break;

        case MIOPEN_REDUCE_TENSOR_MIN:
        case MIOPEN_REDUCE_TENSOR_MAX:
        default: final_val = rand_val * sign; break;
        }
        output[i] = static_cast<T>(final_val);
    }
}

template <typename T>
void ReduceGenerator(T* output,
                     size_t n,
                     uint64_t seed,
                     miopenReduceTensorOp_t op,
                     uint64_t max_val,
                     hipStream_t stream)
{
    // int deviceId;
    // hipGetDevice(&deviceId);
    // hipDeviceProp_t props;
    // hipGetDeviceProperties(&props, deviceId);

    const int threadsPerBlock = 256;
    int numBlocks             = (n + threadsPerBlock - 1) / threadsPerBlock;

    using DeviceT = typename ToDeviceType<T>::type;

    ReduceGeneratorKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        reinterpret_cast<DeviceT*>(output), n, seed, op, max_val);
}

template void ReduceGenerator<double>(double* output,
                                      size_t n,
                                      uint64_t seed,
                                      miopenReduceTensorOp_t op,
                                      uint64_t max_val,
                                      hipStream_t stream);
template void ReduceGenerator<float>(float* output,
                                     size_t n,
                                     uint64_t seed,
                                     miopenReduceTensorOp_t op,
                                     uint64_t max_val,
                                     hipStream_t stream);
template void ReduceGenerator<bfloat16>(bfloat16* output,
                                        size_t n,
                                        uint64_t seed,
                                        miopenReduceTensorOp_t op,
                                        uint64_t max_val,
                                        hipStream_t stream);
template void ReduceGenerator<half_float::half>(half_float::half* output,
                                                size_t n,
                                                uint64_t seed,
                                                miopenReduceTensorOp_t op,
                                                uint64_t max_val,
                                                hipStream_t stream);
template void ReduceGenerator<int8_t>(int8_t* output,
                                      size_t n,
                                      uint64_t seed,
                                      miopenReduceTensorOp_t op,
                                      uint64_t max_val,
                                      hipStream_t stream);

} // namespace test::gtest
