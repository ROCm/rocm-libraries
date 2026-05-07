#ifndef CK_AMD_SWMMA_HPP
#define CK_AMD_SWMMA_HPP

#include "data_type.hpp"

namespace ck {

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_f16f16_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_f16f16_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const half8_t& reg_a, const half16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x32_f16_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_f16f16_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_f16f16_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const half8_t& reg_a, const half16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x32_f16_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_bf16bf16_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_bf16bf16_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bhalf8_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x32_bf16_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_bf16_16x16_bf16bf16_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_bf16_16x16_bf16bf16_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bhalf8_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_bf16_16x16x32_bf16_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<bhalf8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_f8f8_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_f8f8_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const f8x8_t& reg_a, const f8x16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x32_fp8_fp8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool clamp, bool sparseSel>
struct intrin_swmma_f32_16x16_f8f8_w32<16, 16, clamp, 2, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const f8x16_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x64_fp8_fp8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_f8f8_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_f8f8_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const f8x8_t& reg_a, const f8x16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x32_fp8_fp8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool clamp, bool sparseSel>
struct intrin_swmma_f16_16x16_f8f8_w32<16, 16, clamp, 2, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const f8x16_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x64_fp8_fp8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_bf8bf8_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_bf8bf8_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bf8x8_t& reg_a, const bf8x16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x32_bf8_bf8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool clamp, bool sparseSel>
struct intrin_swmma_f32_16x16_bf8bf8_w32<16, 16, clamp, 2, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bf8x16_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x64_bf8_bf8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_bf8bf8_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_bf8bf8_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bf8x8_t& reg_a, const bf8x16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x32_bf8_bf8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool clamp, bool sparseSel>
struct intrin_swmma_f16_16x16_bf8bf8_w32<16, 16, clamp, 2, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bf8x16_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x64_bf8_bf8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_f8bf8_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_f8bf8_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const f8x8_t& reg_a, const bf8x16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x32_fp8_bf8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool clamp, bool sparseSel>
struct intrin_swmma_f32_16x16_f8bf8_w32<16, 16, clamp, 2, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const f8x16_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x64_fp8_bf8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_f8bf8_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_f8bf8_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const f8x8_t& reg_a, const bf8x16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x32_fp8_bf8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool clamp, bool sparseSel>
struct intrin_swmma_f16_16x16_f8bf8_w32<16, 16, clamp, 2, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const f8x16_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x64_fp8_bf8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_bf8f8_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f16_16x16_bf8f8_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bf8x8_t& reg_a, const f8x16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x32_bf8_fp8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool clamp, bool sparseSel>
struct intrin_swmma_f16_16x16_bf8f8_w32<16, 16, clamp, 2, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bf8x16_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f16_16x16x64_bf8_fp8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<half8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_bf8f8_w32;

template <bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_bf8f8_w32<16, 16, clamp, kMultiplier, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bf8x8_t& reg_a, const f8x16_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x32_bf8_fp8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool clamp, bool sparseSel>
struct intrin_swmma_f32_16x16_bf8f8_w32<16, 16, clamp, 2, sparseSel>
{
    template <class FloatC>
    __device__ static void
    Run(const bf8x16_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x64_bf8_fp8_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier,
          bool sparseSel>
struct intrin_swmma_i32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_i32_16x16x32_iu8_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp, bool sparseSel>
struct intrin_swmma_i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_i32_16x16x64_iu8_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier,
          bool sparseSel>
struct intrin_swmma_f32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x32_iu8_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp, bool sparseSel>
struct intrin_swmma_f32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x64_iu8_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier,
          bool sparseSel>
struct intrin_swmma_f32i32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC, class FloatD>
    __device__ static void Run(const FloatA& reg_a,
                               const FloatB& reg_b,
                               const FloatC& reg_c,
                               FloatD& reg_d,
                               index_t index_val)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32i32_16x16x32_iu8_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
        ignore = index_val;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp, bool sparseSel>
struct intrin_swmma_f32i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC, class FloatD>
    __device__ static void Run(const FloatA& reg_a,
                               const FloatB& reg_b,
                               const FloatC& reg_c,
                               FloatD& reg_d,
                               index_t index_val)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32i32_16x16x64_iu8_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier,
          bool sparseSel>
struct intrin_swmma_i32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_i32_16x16x32_iu4_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp, bool sparseSel>
struct intrin_swmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_i32_16x16x64_iu4_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp, bool sparseSel>
struct intrin_swmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 4, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, int32x2_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_i32_16x16x128_iu4_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier,
          bool sparseSel>
struct intrin_swmma_f32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x32_iu4_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp, bool sparseSel>
struct intrin_swmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, index_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x64_iu4_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp, bool sparseSel>
struct intrin_swmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 4, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, int32x2_t index_val)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32_16x16x128_iu4_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = index_val;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier,
          bool sparseSel>
struct intrin_swmma_f32i32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier, bool sparseSel>
struct intrin_swmma_f32i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC, class FloatD>
    __device__ static void Run(const FloatA& reg_a,
                               const FloatB& reg_b,
                               const FloatC& reg_c,
                               FloatD& reg_d,
                               index_t index_val)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32i32_16x16x32_iu4_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                sparseSel,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
        ignore = index_val;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp, bool sparseSel>
struct intrin_swmma_f32i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2, sparseSel>
{
    template <class FloatA, class FloatB, class FloatC, class FloatD>
    __device__ static void Run(const FloatA& reg_a,
                               const FloatB& reg_b,
                               const FloatC& reg_c,
                               FloatD& reg_d,
                               index_t index_val)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_swmma_f32i32_16x16x64_iu4_clamp(
                neg_a,
                reg_a,
                neg_b,
                reg_b,
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                index_val,
                false,
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
        ignore = index_val;
#endif
    }
};

} // namespace ck

#endif
