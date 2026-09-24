// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef W4A16_GROUP_SIZE
#define W4A16_GROUP_SIZE 128
#endif
#ifndef W4A16_LOAD_WIDTH
#define W4A16_LOAD_WIDTH 8
#endif
#ifndef W4A16_UNROLL
#define W4A16_UNROLL 4
#endif
#ifndef W4A16_ACCUMULATORS
#define W4A16_ACCUMULATORS 1
#endif
#ifndef W4A16_NATIVE_PERMUTE
#define W4A16_NATIVE_PERMUTE 0
#endif
#ifndef W4A16_KERNEL_NAME
#define W4A16_KERNEL_NAME Custom_W4A16_Decode_G128_UnsignedBias8_gfx1151
#endif

// The assembly generated from this source uses the Tensile universal argument
// layout (version 3). Selection requires N = batch = 1 and positive K divisible
// by 256. Each wave reduces one row; consecutive lanes read consecutive chunks of packed weights.
namespace
{
    static_assert(W4A16_ACCUMULATORS == 1 || W4A16_ACCUMULATORS == 4,
                  "decode supports one or four accumulation chains");
#if W4A16_NATIVE_PERMUTE
    using NativeHalf2 = _Float16 __attribute__((ext_vector_type(2)));

    __device__ float dot2(unsigned a, unsigned b, float c)
    {
        return __builtin_amdgcn_fdot2(__builtin_bit_cast(NativeHalf2, a),
                                     __builtin_bit_cast(NativeHalf2, b), c, false);
    }

    __device__ unsigned dequantize(unsigned q, unsigned z, unsigned scale)
    {
        // Keep the exact lift subtraction separate from the FP16 rounding
        // in the scale multiply, while letting the compiler schedule both.
        #pragma clang fp contract(off)
        NativeHalf2 result = __builtin_bit_cast(NativeHalf2, q)
                             + __builtin_bit_cast(NativeHalf2, z);
        result *= __builtin_bit_cast(NativeHalf2, scale);
        return __builtin_bit_cast(unsigned, result);
    }
#else
    __device__ float dot2(unsigned a, unsigned b, float c)
    {
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
        return c;
    }

    __device__ unsigned dequantize(unsigned q, unsigned z, unsigned scale)
    {
        unsigned result;
        asm("v_pk_add_f16 %0, %1, %2\n\t"
            "v_pk_mul_f16 %0, %0, %3"
            : "=&v"(result)
            : "v"(q), "v"(z), "v"(scale));
        return result;
    }

#endif

    template<unsigned Width>
    __device__ float accumulate(const unsigned* __restrict__ a,
                                const half* __restrict__ b,
                                const half* __restrict__ scaleA,
                                const unsigned char* __restrict__ zeroA,
                                unsigned row, unsigned lane, unsigned k, unsigned lda)
    {
        float sums[W4A16_ACCUMULATORS] = {};
        constexpr unsigned chunk = 256 * Width;
        const unsigned offset = (row % (k / 256)) * 256;
        // Distribute simultaneous reads across memory channels even when lda is a
        // power of two. The wrap changes traversal order, not the packed layout.
        #pragma unroll W4A16_UNROLL
        for(unsigned iter = 0; iter < k / chunk; ++iter)
        {
            unsigned kk = iter * chunk + lane * (8 * Width) + offset;
            if(kk >= k)
                kk -= k;
            using Packed = unsigned __attribute__((ext_vector_type(Width)));
            const Packed packed = __builtin_nontemporal_load(
                reinterpret_cast<const Packed*>(a + size_t(row) * (lda / 8) + kk / 8));
#if W4A16_GROUP_SIZE == 128
            unsigned scale = __half_as_ushort(scaleA[size_t(row) * (k / 128) + kk / 128]);
            scale |= scale << 16;
            unsigned z = (zeroA[size_t(row / 2) * (k / 128) + kk / 128]
                          >> ((row % 2) * 4)) & 15;
            z |= z << 16;
#endif
            #pragma unroll
            for(unsigned j = 0; j < Width; ++j)
            {
#if W4A16_GROUP_SIZE != 128
                constexpr unsigned groupSize = W4A16_GROUP_SIZE;
                const unsigned group = kk / groupSize + j / (groupSize / 8);
                unsigned scale = __half_as_ushort(scaleA[size_t(row) * (k / groupSize) + group]);
                scale |= scale << 16;
                unsigned z = (zeroA[size_t(row / 2) * (k / groupSize) + group]
                              >> ((row % 2) * 4)) & 15;
                z |= z << 16;
#endif
                // Independent chains hide dot-product latency without moving
                // scaling past the required FP16 dequantization rounding.
                float& sum = sums[j % W4A16_ACCUMULATORS];
                unsigned q = packed[j];
                const uint4 packedActivation = *reinterpret_cast<const uint4*>(b + kk + j * 8);
                // Sequential nibbles pair K offsets (0,4), (1,5), (2,6), (3,7)
                // in the dequantizer below. Match the activation pairs to them.
#if W4A16_NATIVE_PERMUTE
                const uint4 activation = {
                    __builtin_amdgcn_perm(packedActivation.z, packedActivation.x, 0x05040100),
                    __builtin_amdgcn_perm(packedActivation.z, packedActivation.x, 0x07060302),
                    __builtin_amdgcn_perm(packedActivation.w, packedActivation.y, 0x05040100),
                    __builtin_amdgcn_perm(packedActivation.w, packedActivation.y, 0x07060302)};
#else
                const uint4 activation = {
                    (packedActivation.x & 0xffff) | (packedActivation.z << 16),
                    (packedActivation.x >> 16) | (packedActivation.z & 0xffff0000),
                    (packedActivation.y & 0xffff) | (packedActivation.w << 16),
                    (packedActivation.y >> 16) | (packedActivation.w & 0xffff0000)};
#endif

                // Extract pairs across the two halves of each packed dword.
                // The integer lifts/subtractions are exact; scaling rounds to fp16
                // before the fp32 dot product, matching the matrix kernel's contract.
                const unsigned p0 = dequantize((q & 0x000f000f) | 0x64006400,
                                               z | 0xe400e400, scale);
                const unsigned p1 = dequantize((q & 0x00f000f0) | 0x54005400,
                                               (z << 4) | 0xd400d400, scale);
                q >>= 8;
                const unsigned p2 = dequantize((q & 0x000f000f) | 0x64006400,
                                               z | 0xe400e400, scale);
                const unsigned p3 = dequantize((q & 0x00f000f0) | 0x54005400,
                                               (z << 4) | 0xd400d400, scale);
                sum = dot2(p0, activation.x, sum);
                sum = dot2(p1, activation.y, sum);
                sum = dot2(p2, activation.z, sum);
                sum = dot2(p3, activation.w, sum);
            }
        }
        if constexpr(W4A16_ACCUMULATORS == 4)
            return (sums[0] + sums[1]) + (sums[2] + sums[3]);
        else
            return sums[0];
    }
}

extern "C" __global__ __launch_bounds__(128)
void W4A16_KERNEL_NAME(
    unsigned info, unsigned kernelInfo0, unsigned kernelInfo1, unsigned numWorkgroups,
    unsigned m, unsigned n, unsigned batch, unsigned k,
    const unsigned* __restrict__ a, const half* __restrict__ b,
    unsigned lda, unsigned strideA, unsigned ldb, unsigned strideB,
    float alpha, float beta, half* d, const half* c,
    unsigned ldd, unsigned strideD, unsigned ldc, unsigned strideC,
    const half* __restrict__ scaleA, const half* scaleB,
    const unsigned char* __restrict__ zeroA,
    unsigned long offsetD, unsigned long offsetC, unsigned long offsetA, unsigned long offsetB)
{
    const unsigned lane = threadIdx.x % 32;
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32;
    if(row >= m)
        return;

    // Vector loads stream weights; metadata follows quantization groups.
    // Preserve the dword path when K or the row stride cannot support this width.
    float sum = 0;
    if(alpha != 0)
    {
        if(k % (256 * W4A16_LOAD_WIDTH) == 0 && lda % (8 * W4A16_LOAD_WIDTH) == 0)
            sum = accumulate<W4A16_LOAD_WIDTH>(a, b, scaleA, zeroA, row, lane, k, lda);
        else
            sum = accumulate<1>(a, b, scaleA, zeroA, row, lane, k, lda);
    }
    for(unsigned distance = 16; distance; distance /= 2)
        sum += __shfl_xor(sum, distance);
    if(lane == 0)
    {
        float result = alpha * sum;
        if(beta != 0)
            result += beta * __half2float(c[row]);
        d[row] = __float2half(result);
    }
}
