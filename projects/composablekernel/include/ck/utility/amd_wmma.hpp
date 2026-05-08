// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_AMD_WMMA_HPP
#define CK_AMD_WMMA_HPP

#include "ck/utility/amd_inline_asm.hpp"
#include "data_type.hpp"
#include "dtype_fp64.hpp"
// TODO: Add arch limitation
namespace ck {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx11_generic__)
#define __gfx11__
#endif

#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define __gfx120__
#endif

#if defined(__gfx1250__) || defined(__gfx1251__)
#define __gfx125__
#endif

/********************************WAVE32 MODE***********************************************/

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void
    Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void
    Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w32;

template <index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w32<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<half16_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<half16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void
    Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_f16_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w32;

template <index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<bhalf16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<bhalf16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void
    Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_clamp(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}], 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        if constexpr(sizeof(reg_a) == sizeof(int32x4_t))
        {
            reg_c.template AsType<int32x8_t>()(Number<0>{}) =
                __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    neg_a,
                    bit_cast<int32x4_t>(reg_a),
                    neg_b,
                    bit_cast<int32x4_t>(reg_b),
                    reg_c.template AsType<int32x8_t>()[Number<0>{}],
                    clamp);
        }
        else
#elif defined(__gfx12__)
        if constexpr(sizeof(reg_a) == sizeof(int32x2_t))
        {
            reg_c.template AsType<int32x8_t>()(Number<0>{}) =
                __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    neg_a,
                    bit_cast<int32x2_t>(reg_a),
                    neg_b,
                    bit_cast<int32x2_t>(reg_b),
                    reg_c.template AsType<int32x8_t>()[Number<0>{}],
                    clamp);
        }
        else
#endif
        {
            ignore = reg_a;
            ignore = reg_b;
            ignore = reg_c;
        }
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_clamp(
                neg_a,
                bit_cast<int32x2_t>(reg_a),
                neg_b,
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
    }
};

/**********************************GFX13***************************************************/
// for gfx13 usage
template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f16f16_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f16f16_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf16bf16_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf16bf16_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f16f16_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f16f16_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_f16_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_bf16_16x16_bf16bf16_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_bf16_16x16_bf16bf16_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_clamp(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_i32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_clamp(
                neg_a,
                bit_cast<int32x2_t>(reg_a),
                neg_b,
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x32_iu8_clamp(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for Grimlock
template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x128_iu8_clamp(
                neg_a,
                bit_cast<int32x16_t>(reg_a),
                neg_b,
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_f32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_iu8_clamp(
                neg_a,
                bit_cast<int32x2_t>(reg_a),
                neg_b,
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_iu8_clamp(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for Grimlock
template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_iu8_clamp(
                neg_a,
                bit_cast<int32x16_t>(reg_a),
                neg_b,
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f8f8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f8f8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_clamp(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_f8f8_w32<16, 16, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_fp8_fp8_clamp(
                bit_cast<int32x4_t>(reg_a),
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_f8f8_w32<16, 16, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8_clamp(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f8bf8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f8bf8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_clamp(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_f8bf8_w32<16, 16, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_fp8_bf8_clamp(
                bit_cast<int32x4_t>(reg_a),
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for grimlock
template <bool clamp>
struct intrin_wmma_f32_16x16_f8bf8_w32<16, 16, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8_clamp(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf8f8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf8f8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_clamp(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_bf8f8_w32<16, 16, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_bf8_fp8_clamp(
                bit_cast<int32x4_t>(reg_a),
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for grimlock
template <bool clamp>
struct intrin_wmma_f32_16x16_bf8f8_w32<16, 16, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8_clamp(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf8bf8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf8bf8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_clamp(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_bf8bf8_w32<16, 16, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_bf8_bf8_clamp(
                bit_cast<int32x4_t>(reg_a),
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for grimlock
template <bool clamp>
struct intrin_wmma_f32_16x16_bf8bf8_w32<16, 16, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8_clamp(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f8f8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f8f8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_fp8_fp8_clamp(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f16_16x16_f8f8_w32<16, 16, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x32_fp8_fp8_clamp(
                bit_cast<int32x4_t>(reg_a),
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for grimlock
template <bool clamp>
struct intrin_wmma_f16_16x16_f8f8_w32<16, 16, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8_clamp(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f8bf8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f8bf8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_fp8_bf8_clamp(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f16_16x16_f8bf8_w32<16, 16, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x32_fp8_bf8_clamp(
                bit_cast<int32x4_t>(reg_a),
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for grimlock
template <bool clamp>
struct intrin_wmma_f16_16x16_f8bf8_w32<16, 16, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8_clamp(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_bf8f8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_bf8f8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_bf8_fp8_clamp(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f16_16x16_bf8f8_w32<16, 16, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x32_bf8_fp8_clamp(
                bit_cast<int32x4_t>(reg_a),
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for grimlock
template <bool clamp>
struct intrin_wmma_f16_16x16_bf8f8_w32<16, 16, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8_clamp(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_bf8bf8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_bf8bf8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_bf8_bf8_clamp(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f16_16x16_bf8bf8_w32<16, 16, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x32_bf8_bf8_clamp(
                bit_cast<int32x4_t>(reg_a),
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// Todo for grimlock
template <bool clamp>
struct intrin_wmma_f16_16x16_bf8bf8_w32<16, 16, clamp, 8>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8_clamp(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                reg_c.template AsType<half8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

constexpr index_t VOP5M_MOD1_OFFSET = 6;

template <typename SrcAType, typename SrcBType>
constexpr int32_t get_f8f6f4_data_format()
{
    // Use generic lambda with a dummy parameter instead of explicit template
    constexpr auto get_src_type = [](auto dummy) constexpr {
        using T = decltype(dummy);
        if constexpr(is_same_v<T, f8_t>)
        {
            return 0;
        }
        else if constexpr(is_same_v<T, bf8_t>)
        {
            return 1;
        }
        else if constexpr(is_same_v<T, f6x16_pk_t> || is_same_v<T, f6x32_pk_t>)
        {
            return static_cast<int32_t>(MTX_FMT::MTX_FMT_FP6_E2M3);
        }
        else if constexpr(is_same_v<T, bf6x16_pk_t> || is_same_v<T, bf6x32_pk_t>)
        {
            return static_cast<int32_t>(MTX_FMT::MTX_FMT_FP6_E3M2);
        }
        else if constexpr(is_same_v<T, f4x2_pk_t>)
        {
            return static_cast<int32_t>(MTX_FMT::MTX_FMT_FP4_E2M1);
        }
        else
        {
            return static_cast<int32_t>(T::RawType);
        }
    };

    constexpr int32_t srcAType = get_src_type(SrcAType{});
    constexpr int32_t srcBType = get_src_type(SrcBType{});

    return (srcAType | (srcBType << 3)) << VOP5M_MOD1_OFFSET;
}

// this field in mod1
template <index_t ABlockSel, index_t BBlockSel>
constexpr int32_t scale_select()
{
    return ((ABlockSel) | (BBlockSel << 2)) << (VOP5M_MOD1_OFFSET + 8);
}

// src: f8f6f4, dst: fp32
template <index_t MPerWave,
          index_t NPerWave,
          typename SrcAType,
          typename SrcBType,
          index_t ABlockSel,
          index_t BBlockSel,
          bool clamp>
struct intrin_wmma_f32_16x16_f8f6f4_w32;

template <typename SrcAType, typename SrcBType, index_t ABlockSel, index_t BBlockSel, bool clamp>
struct intrin_wmma_f32_16x16_f8f6f4_w32<16, 16, SrcAType, SrcBType, ABlockSel, BBlockSel, clamp>
{
    template <class FloatC>
    __device__ static void Run(const f4x32_t& reg_a,
                               const f4x32_t& reg_b,
                               const int32_t& a_scale,
                               const int32_t& b_scale,
                               FloatC& reg_c)
    {
#if defined(__gfx13__)
        int32x4_t arg_a = bit_cast<int32x4_t>(reg_a);
        int32x4_t arg_b = bit_cast<int32x4_t>(reg_b);
        using arg_type  = int32x8_t;

        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x64_f8f6f4_clamp(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], 0, 0, 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], 0, 0, 0, 0},
                reg_c.template AsType<float8_t>()[Number<0>{}],
                a_scale,
                b_scale,
                get_f8f6f4_data_format<SrcAType, SrcBType>() | scale_select<ABlockSel, BBlockSel>(),
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = a_scale;
        ignore = b_scale;
#endif
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a,
                               const FloatB& reg_b,
                               const int32_t& a_scale,
                               const int32_t& b_scale,
                               FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x64_f8f6f4_clamp(
                bit_cast<int32x8_t>(reg_a),
                bit_cast<int32x8_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                a_scale,
                b_scale,
                get_f8f6f4_data_format<SrcAType, SrcBType>() | scale_select<ABlockSel, BBlockSel>(),
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = a_scale;
        ignore = b_scale;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_i32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const int32_t& reg_a, const int32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const int32x2_t& reg_a, const int32x2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x32_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 4>
{
    template <class FloatC>
    __device__ static void Run(const int32x4_t& reg_a, const int32x4_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x64_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_f32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const int32_t& reg_a, const int32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const int32x2_t& reg_a, const int32x2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 4>
{
    template <class FloatC>
    __device__ static void Run(const int32x4_t& reg_a, const int32x4_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x64_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_f32i32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC, class FloatD>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x16_iu8_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC, class FloatD>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x32_iu8_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_f32i32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const int32_t& reg_a, const int32_t& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x16_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const int32x2_t& reg_a, const int32x2_t& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x32_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 4>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const int32x4_t& reg_a, const int32x4_t& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x64_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};
/********************************WAVE64 MODE***********************************************/

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w64;

template <>
struct intrin_wmma_f32_16x16x16_f16_w64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w64;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64(
                reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w64;

template <index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w64<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w64;

template <index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w64<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w64;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w64<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x4_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// gfx12
/********************************WAVE32 MODE***********************************************/

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32_gfx12;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32_gfx12<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                neg_a,
                bit_cast<int32x2_t>(reg_a),
                neg_b,
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: f8, f8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f8f8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f8f8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: f8, bf8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf8, f8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf8, bf8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// gfx125x
/********************************WAVE32 MODE***********************************************/
// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x32_f16;

template <>
struct intrin_wmma_f16_16x16x32_f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x32_f16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<half8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_bf16_16x16x32_bf16;

template <>
struct intrin_wmma_bf16_16x16x32_bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx125__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<bhalf8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x32_f16;

template <>
struct intrin_wmma_f32_16x16x32_f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x32_f16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x32_bf16;

template <>
struct intrin_wmma_f32_16x16x32_bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_bf16f32_16x16x32_bf16;

template <>
struct intrin_wmma_bf16f32_16x16x32_bf16<16, 16>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx125__)
        reg_d
            .template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b>
struct intrin_wmma_i32_16x16x64_iu8;

template <bool neg_a, bool neg_b>
struct intrin_wmma_i32_16x16x64_iu8<16, 16, neg_a, neg_b>
{
    template <class FloatC>
    __device__ static void Run(const int8x32_t& reg_a, const int8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x64_iu8(neg_a,
                                                   bit_cast<int32x8_t>(reg_a),
                                                   neg_b,
                                                   bit_cast<int32x8_t>(reg_b),
                                                   reg_c.template AsType<int32x8_t>()[Number<0>{}],
                                                   false,
                                                   false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_f8f8;
template <>
struct intrin_wmma_f32_16x16x64_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_f8bf8;
template <>
struct intrin_wmma_f32_16x16x64_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_bf8f8;
template <>
struct intrin_wmma_f32_16x16x64_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_bf8bf8;
template <>
struct intrin_wmma_f32_16x16x64_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_f8f8;
template <>
struct intrin_wmma_f16_16x16x64_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_f8bf8;
template <>
struct intrin_wmma_f16_16x16x64_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_bf8f8;
template <>
struct intrin_wmma_f16_16x16x64_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_bf8bf8;
template <>
struct intrin_wmma_f16_16x16x64_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_f8f8;
template <>
struct intrin_wmma_f32_16x16x128_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_f8bf8;
template <>
struct intrin_wmma_f32_16x16x128_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_bf8f8;
template <>
struct intrin_wmma_f32_16x16x128_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_bf8bf8;
template <>
struct intrin_wmma_f32_16x16x128_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_f8f8;
template <>
struct intrin_wmma_f16_16x16x128_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_f8bf8;
template <>
struct intrin_wmma_f16_16x16x128_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_bf8f8;
template <>
struct intrin_wmma_f16_16x16x128_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_bf8bf8;
template <>
struct intrin_wmma_f16_16x16x128_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x4_f32;

template <>
struct intrin_wmma_f32_16x16x4_f32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float2_t& reg_a, const float2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x4_f32(0,
                                                  bit_cast<float2_t>(reg_a),
                                                  0,
                                                  bit_cast<float2_t>(reg_b),
                                                  0,
                                                  reg_c.template AsType<float8_t>()[Number<0>{}],
                                                  false,
                                                  false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f64_16x16x4_f64;

template <>
struct intrin_wmma_f64_16x16x4_f64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const double2_t& reg_a, const double2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx1251__)
        reg_c.template AsType<double8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f64_16x16x4_f64(0,
                                                  bit_cast<double2_t>(reg_a),
                                                  0,
                                                  bit_cast<double2_t>(reg_b),
                                                  0,
                                                  reg_c.template AsType<double8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

namespace wmma_impl {
#ifndef CK_CODE_GEN_RTC
// utils for f8f6f4 instructions
template <typename T>
struct ScaleTypeSelector
{
};

// use int32_t for backward compatibility
template <>
struct ScaleTypeSelector<int32_t>
{
    static constexpr int value = 0x0;
};

template <>
struct ScaleTypeSelector<e8m0x4_bexp_t>
{
    static constexpr int value = 0x0;
};

template <>
struct ScaleTypeSelector<e8m0x8_bexp_t>
{
    static constexpr int value = 0x0;
};

template <>
struct ScaleTypeSelector<e5m3x4_scale_t>
{
    static constexpr int value = 0x1;
};

template <>
struct ScaleTypeSelector<e5m3x8_scale_t>
{
    static constexpr int value = 0x1;
};

template <>
struct ScaleTypeSelector<e4m3x4_scale_t>
{
    static constexpr int value = 0x2;
};

template <>
struct ScaleTypeSelector<e4m3x8_scale_t>
{
    static constexpr int value = 0x2;
};

enum InputFormat : uint8_t
{
    E4M3 = 0x0,
    E5M2 = 0x1,
    E2M3 = 0x2,
    E3M2 = 0x3,
    E2M1 = 0x4
};
#endif // #ifndef CK_CODE_GEN_RTC
} // namespace wmma_impl

template <index_t MPerWave,
          index_t NPerWave,
          index_t ScaleOpselA,
          index_t ScaleOpselB,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct intrin_wmma_scale_f32_16x16x128_f8f6f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselA, index_t ScaleOpselB, typename ScaleTypeA, typename ScaleTypeB>
struct intrin_wmma_scale_f32_16x16x128_f8f6f4<16,
                                              16,
                                              ScaleOpselA,
                                              ScaleOpselB,
                                              ScaleTypeA,
                                              ScaleTypeB>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f8x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> || is_same_v<ScaleTypeA, int32_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t or int32_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x4_bexp_t> || is_same_v<ScaleTypeB, int32_t>,
                      "ScaleTypeB must be e8m0x4_bexp_t or int32_t");
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E4M3, // OPSEL
                reg_a,
                wmma_impl::InputFormat::E4M3, // OPSEL_HI
                reg_b,
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                // M=laneId % 16 [7:0] K=0..31; [15:8] K=32..63; [23:16] K=64..95; [31:24] K=96..127
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                // N=laneId % 16 [7:0] K=0..31; [15:8] K=32..63; [23:16] K=64..95; [31:24] K=96..127
                bit_cast<int32_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const bf8x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> || is_same_v<ScaleTypeA, int32_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t or int32_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x4_bexp_t> || is_same_v<ScaleTypeB, int32_t>,
                      "ScaleTypeB must be e8m0x4_bexp_t or int32_t");
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E5M2, // OPSEL
                reg_a,
                wmma_impl::InputFormat::E5M2, // OPSEL_HI
                reg_b,
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f6x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> || is_same_v<ScaleTypeA, int32_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t or int32_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x4_bexp_t> || is_same_v<ScaleTypeB, int32_t>,
                      "ScaleTypeB must be e8m0x4_bexp_t or int32_t");
#if defined(__gfx125__)
        // f6x64_t is a vector of 2 f6x32_pk_t, so we have to repack and cast them to int32x6_t
        int32x6_t arg_a_0 = bit_cast<int32x6_t>(reg_a.AsType<f6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_a_1 = bit_cast<int32x6_t>(reg_a.AsType<f6x32_pk_t>()[Number<1>{}]);
        int32x6_t arg_b_0 = bit_cast<int32x6_t>(reg_b.AsType<f6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_b_1 = bit_cast<int32x6_t>(reg_b.AsType<f6x32_pk_t>()[Number<1>{}]);
        using arg_type    = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E2M3, // OPSEL
                arg_type{arg_a_0[0],
                         arg_a_0[1],
                         arg_a_0[2],
                         arg_a_0[3],
                         arg_a_0[4],
                         arg_a_0[5],
                         arg_a_1[0],
                         arg_a_1[1],
                         arg_a_1[2],
                         arg_a_1[3],
                         arg_a_1[4],
                         arg_a_1[5],
                         0,
                         0,
                         0,
                         0},
                wmma_impl::InputFormat::E2M3, // OPSEL_HI
                arg_type{arg_b_0[0],
                         arg_b_0[1],
                         arg_b_0[2],
                         arg_b_0[3],
                         arg_b_0[4],
                         arg_b_0[5],
                         arg_b_1[0],
                         arg_b_1[1],
                         arg_b_1[2],
                         arg_b_1[3],
                         arg_b_1[4],
                         arg_b_1[5],
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    // Overload for f6x16x4_t (4 vectors of 16-packed FP6)
    template <class FloatC>
    __device__ static void Run(const f6x16x4_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f6x16x4_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {

        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> || is_same_v<ScaleTypeA, int32_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t or int32_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x4_bexp_t> || is_same_v<ScaleTypeB, int32_t>,
                      "ScaleTypeB must be e8m0x4_bexp_t or int32_t");

#if defined(__gfx125__)
        // f6x16x4_t = 4 x f6x16_pk_t, each f6x16_pk_t has data_ as uint32_t[3]
        // Extract the 12 uint32_t values (4 packs x 3 uint32_t each)
        auto a0 = reg_a.template AsType<f6x16_pk_t>()[Number<0>{}].data_;
        auto a1 = reg_a.template AsType<f6x16_pk_t>()[Number<1>{}].data_;
        auto a2 = reg_a.template AsType<f6x16_pk_t>()[Number<2>{}].data_;
        auto a3 = reg_a.template AsType<f6x16_pk_t>()[Number<3>{}].data_;

        auto b0 = reg_b.template AsType<f6x16_pk_t>()[Number<0>{}].data_;
        auto b1 = reg_b.template AsType<f6x16_pk_t>()[Number<1>{}].data_;
        auto b2 = reg_b.template AsType<f6x16_pk_t>()[Number<2>{}].data_;
        auto b3 = reg_b.template AsType<f6x16_pk_t>()[Number<3>{}].data_;

        using arg_type = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E2M3, // A data format
                arg_type{static_cast<int32_t>(a0[0]),
                         static_cast<int32_t>(a0[1]),
                         static_cast<int32_t>(a0[2]),
                         static_cast<int32_t>(a1[0]),
                         static_cast<int32_t>(a1[1]),
                         static_cast<int32_t>(a1[2]),
                         static_cast<int32_t>(a2[0]),
                         static_cast<int32_t>(a2[1]),
                         static_cast<int32_t>(a2[2]),
                         static_cast<int32_t>(a3[0]),
                         static_cast<int32_t>(a3[1]),
                         static_cast<int32_t>(a3[2]),
                         0,
                         0,
                         0,
                         0},
                wmma_impl::InputFormat::E2M3, // B data format
                arg_type{static_cast<int32_t>(b0[0]),
                         static_cast<int32_t>(b0[1]),
                         static_cast<int32_t>(b0[2]),
                         static_cast<int32_t>(b1[0]),
                         static_cast<int32_t>(b1[1]),
                         static_cast<int32_t>(b1[2]),
                         static_cast<int32_t>(b2[0]),
                         static_cast<int32_t>(b2[1]),
                         static_cast<int32_t>(b2[2]),
                         static_cast<int32_t>(b3[0]),
                         static_cast<int32_t>(b3[1]),
                         static_cast<int32_t>(b3[2]),
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const bf6x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> || is_same_v<ScaleTypeA, int32_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t or int32_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x4_bexp_t> || is_same_v<ScaleTypeB, int32_t>,
                      "ScaleTypeB must be e8m0x4_bexp_t or int32_t");
#if defined(__gfx125__)
        // bf6x64_t is a vector of 2 bf6x32_pk_t, so we have to repack and cast them to int32x6_t
        int32x6_t arg_a_0 = bit_cast<int32x6_t>(reg_a.AsType<bf6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_a_1 = bit_cast<int32x6_t>(reg_a.AsType<bf6x32_pk_t>()[Number<1>{}]);
        int32x6_t arg_b_0 = bit_cast<int32x6_t>(reg_b.AsType<bf6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_b_1 = bit_cast<int32x6_t>(reg_b.AsType<bf6x32_pk_t>()[Number<1>{}]);
        using arg_type    = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E3M2, // OPSEL
                arg_type{arg_a_0[0],
                         arg_a_0[1],
                         arg_a_0[2],
                         arg_a_0[3],
                         arg_a_0[4],
                         arg_a_0[5],
                         arg_a_1[0],
                         arg_a_1[1],
                         arg_a_1[2],
                         arg_a_1[3],
                         arg_a_1[4],
                         arg_a_1[5],
                         0,
                         0,
                         0,
                         0},
                wmma_impl::InputFormat::E3M2, // OPSEL_HI
                arg_type{arg_b_0[0],
                         arg_b_0[1],
                         arg_b_0[2],
                         arg_b_0[3],
                         arg_b_0[4],
                         arg_b_0[5],
                         arg_b_1[0],
                         arg_b_1[1],
                         arg_b_1[2],
                         arg_b_1[3],
                         arg_b_1[4],
                         arg_b_1[5],
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    // Overload for bf6x16x4_t (4 vectors of 16-packed BF6)
    template <class FloatC>
    __device__ static void Run(const bf6x16x4_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const bf6x16x4_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> || is_same_v<ScaleTypeA, int32_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t or int32_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x4_bexp_t> || is_same_v<ScaleTypeB, int32_t>,
                      "ScaleTypeB must be e8m0x4_bexp_t or int32_t");
#if defined(__gfx125__)
        // bf6x16x4_t = 4 x bf6x16_pk_t, each bf6x16_pk_t has data_ as uint32_t[3]
        // Extract the 12 uint32_t values (4 packs x 3 uint32_t each)
        auto a0 = reg_a.template AsType<bf6x16_pk_t>()[Number<0>{}].data_;
        auto a1 = reg_a.template AsType<bf6x16_pk_t>()[Number<1>{}].data_;
        auto a2 = reg_a.template AsType<bf6x16_pk_t>()[Number<2>{}].data_;
        auto a3 = reg_a.template AsType<bf6x16_pk_t>()[Number<3>{}].data_;

        auto b0 = reg_b.template AsType<bf6x16_pk_t>()[Number<0>{}].data_;
        auto b1 = reg_b.template AsType<bf6x16_pk_t>()[Number<1>{}].data_;
        auto b2 = reg_b.template AsType<bf6x16_pk_t>()[Number<2>{}].data_;
        auto b3 = reg_b.template AsType<bf6x16_pk_t>()[Number<3>{}].data_;

        using arg_type = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E3M2, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6
                                              // E3M2; 4-FP4 E2M1
                arg_type{static_cast<int32_t>(a0[0]),
                         static_cast<int32_t>(a0[1]),
                         static_cast<int32_t>(a0[2]),
                         static_cast<int32_t>(a1[0]),
                         static_cast<int32_t>(a1[1]),
                         static_cast<int32_t>(a1[2]),
                         static_cast<int32_t>(a2[0]),
                         static_cast<int32_t>(a2[1]),
                         static_cast<int32_t>(a2[2]),
                         static_cast<int32_t>(a3[0]),
                         static_cast<int32_t>(a3[1]),
                         static_cast<int32_t>(a3[2]),
                         0,
                         0,
                         0,
                         0},
                wmma_impl::InputFormat::E3M2, // OPSEL_HI
                arg_type{static_cast<int32_t>(b0[0]),
                         static_cast<int32_t>(b0[1]),
                         static_cast<int32_t>(b0[2]),
                         static_cast<int32_t>(b1[0]),
                         static_cast<int32_t>(b1[1]),
                         static_cast<int32_t>(b1[2]),
                         static_cast<int32_t>(b2[0]),
                         static_cast<int32_t>(b2[1]),
                         static_cast<int32_t>(b2[2]),
                         static_cast<int32_t>(b3[0]),
                         static_cast<int32_t>(b3[1]),
                         static_cast<int32_t>(b3[2]),
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f4x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(
            is_same_v<ScaleTypeA, e8m0x4_bexp_t> || is_same_v<ScaleTypeA, int32_t> ||
                is_same_v<ScaleTypeA, e5m3x4_scale_t> || is_same_v<ScaleTypeA, e4m3x4_scale_t>,
            "ScaleTypeA must be e8m0x4_bexp_t, int32_t, e5m3x4_scale_t, or e4m3x4_scale_t");
        static_assert(
            is_same_v<ScaleTypeB, e8m0x4_bexp_t> || is_same_v<ScaleTypeB, int32_t> ||
                is_same_v<ScaleTypeB, e5m3x4_scale_t> || is_same_v<ScaleTypeB, e4m3x4_scale_t>,
            "ScaleTypeB must be e8m0x4_bexp_t, int32_t, e5m3x4_scale_t, or e4m3x4_scale_t");
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E2M1, // OPSEL
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                wmma_impl::InputFormat::E2M1, // OPSEL_HI
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f4x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> || is_same_v<ScaleTypeA, int32_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t or int32_t");
        static_assert(
            is_same_v<ScaleTypeB, e8m0x4_bexp_t> || is_same_v<ScaleTypeB, int32_t> ||
                is_same_v<ScaleTypeB, e5m3x4_scale_t> || is_same_v<ScaleTypeB, e4m3x4_scale_t>,
            "ScaleTypeB must be e8m0x4_bexp_t, int32_t, e5m3x4_scale_t, or e4m3x4_scale_t");
#if defined(__gfx125__)
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E4M3, // OPSEL
                reg_a,
                wmma_impl::InputFormat::E2M1, // OPSEL_HI
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

template <index_t MPerWave,
          index_t NPerWave,
          index_t ScaleOpselA,
          index_t ScaleOpselB,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct intrin_wmma_scale16_f32_16x16x128_f8f6f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselA, index_t ScaleOpselB, typename ScaleTypeA, typename ScaleTypeB>
struct intrin_wmma_scale16_f32_16x16x128_f8f6f4<16,
                                                16,
                                                ScaleOpselA,
                                                ScaleOpselB,
                                                ScaleTypeA,
                                                ScaleTypeB>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f8x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        static_assert(is_same_v<ScaleTypeA, e8m0x8_bexp_t>, "ScaleTypeA must be e8m0x8_bexp_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x8_bexp_t>, "ScaleTypeB must be e8m0x8_bexp_t");
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E4M3, // OPSEL
                reg_a,
                wmma_impl::InputFormat::E4M3, // OPSEL_HI
                reg_b,
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const bf8x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        static_assert(is_same_v<ScaleTypeA, e8m0x8_bexp_t>, "ScaleTypeA must be e8m0x8_bexp_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x8_bexp_t>, "ScaleTypeB must be e8m0x8_bexp_t");
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E5M2, // OPSEL
                reg_a,
                wmma_impl::InputFormat::E5M2, // OPSEL_HI
                reg_b,
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f6x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        static_assert(is_same_v<ScaleTypeA, e8m0x8_bexp_t>, "ScaleTypeA must be e8m0x8_bexp_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x8_bexp_t>, "ScaleTypeB must be e8m0x8_bexp_t");
#if defined(__gfx125__)
        // f6x64_t is a vector of 2 f6x32_pk_t, so we have to repack and cast them to int32x6_t
        int32x6_t arg_a_0 = bit_cast<int32x6_t>(reg_a.AsType<f6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_a_1 = bit_cast<int32x6_t>(reg_a.AsType<f6x32_pk_t>()[Number<1>{}]);
        int32x6_t arg_b_0 = bit_cast<int32x6_t>(reg_b.AsType<f6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_b_1 = bit_cast<int32x6_t>(reg_b.AsType<f6x32_pk_t>()[Number<1>{}]);
        using arg_type    = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E2M3, // OPSEL
                arg_type{arg_a_0[0],
                         arg_a_0[1],
                         arg_a_0[2],
                         arg_a_0[3],
                         arg_a_0[4],
                         arg_a_0[5],
                         arg_a_1[0],
                         arg_a_1[1],
                         arg_a_1[2],
                         arg_a_1[3],
                         arg_a_1[4],
                         arg_a_1[5],
                         0,
                         0,
                         0,
                         0},
                wmma_impl::InputFormat::E2M3, // OPSEL_HI
                arg_type{arg_b_0[0],
                         arg_b_0[1],
                         arg_b_0[2],
                         arg_b_0[3],
                         arg_b_0[4],
                         arg_b_0[5],
                         arg_b_1[0],
                         arg_b_1[1],
                         arg_b_1[2],
                         arg_b_1[3],
                         arg_b_1[4],
                         arg_b_1[5],
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const bf6x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        static_assert(is_same_v<ScaleTypeA, e8m0x8_bexp_t>, "ScaleTypeA must be e8m0x8_bexp_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x8_bexp_t>, "ScaleTypeB must be e8m0x8_bexp_t");
#if defined(__gfx125__)
        // bf6x64_t is a vector of 2 bf6x32_pk_t, so we have to repack and cast them to int32x6_t
        int32x6_t arg_a_0 = bit_cast<int32x6_t>(reg_a.AsType<bf6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_a_1 = bit_cast<int32x6_t>(reg_a.AsType<bf6x32_pk_t>()[Number<1>{}]);
        int32x6_t arg_b_0 = bit_cast<int32x6_t>(reg_b.AsType<bf6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_b_1 = bit_cast<int32x6_t>(reg_b.AsType<bf6x32_pk_t>()[Number<1>{}]);
        using arg_type    = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E3M2, // OPSEL
                arg_type{arg_a_0[0],
                         arg_a_0[1],
                         arg_a_0[2],
                         arg_a_0[3],
                         arg_a_0[4],
                         arg_a_0[5],
                         arg_a_1[0],
                         arg_a_1[1],
                         arg_a_1[2],
                         arg_a_1[3],
                         arg_a_1[4],
                         arg_a_1[5],
                         0,
                         0,
                         0,
                         0},
                wmma_impl::InputFormat::E3M2, // OPSEL_HI
                arg_type{arg_b_0[0],
                         arg_b_0[1],
                         arg_b_0[2],
                         arg_b_0[3],
                         arg_b_0[4],
                         arg_b_0[5],
                         arg_b_1[0],
                         arg_b_1[1],
                         arg_b_1[2],
                         arg_b_1[3],
                         arg_b_1[4],
                         arg_b_1[5],
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f4x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        static_assert(is_same_v<ScaleTypeA, e8m0x8_bexp_t> ||
                          is_same_v<ScaleTypeA, e5m3x8_scale_t> ||
                          is_same_v<ScaleTypeA, e4m3x8_scale_t>,
                      "ScaleTypeA must be e8m0x8_bexp_t, e5m3x8_scale_t, or e4m3x8_scale_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x8_bexp_t> ||
                          is_same_v<ScaleTypeB, e5m3x8_scale_t> ||
                          is_same_v<ScaleTypeB, e4m3x8_scale_t>,
                      "ScaleTypeB must be e8m0x8_bexp_t, e5m3x8_scale_t, or e4m3x8_scale_t");
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E2M1, // OPSEL
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                wmma_impl::InputFormat::E2M1, // OPSEL_HI
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f4x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        static_assert(is_same_v<ScaleTypeA, e8m0x8_bexp_t>, "ScaleTypeA must be e8m0x8_bexp_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x8_bexp_t> ||
                          is_same_v<ScaleTypeB, e5m3x8_scale_t> ||
                          is_same_v<ScaleTypeB, e4m3x8_scale_t>,
                      "ScaleTypeB must be e8m0x8_bexp_t, e5m3x8_scale_t, or e4m3x8_scale_t");
#if defined(__gfx125__)
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                wmma_impl::InputFormat::E4M3, // OPSEL
                reg_a,
                wmma_impl::InputFormat::E2M1, // OPSEL_HI
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                                     // SCALE_OPSEL[0]
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value, // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,                                     // SCALE_OPSEL[1]
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value, // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b),
                0,  // NEG
                0); // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

template <index_t MPerWave,
          index_t NPerWave,
          index_t ScaleOpselB,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct intrin_wmma_scale_f32_32x16x128_f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselB, typename ScaleTypeA, typename ScaleTypeB>
struct intrin_wmma_scale_f32_32x16x128_f4<32, 16, ScaleOpselB, ScaleTypeA, ScaleTypeB>
{
    template <class FloatC>
    __device__ static void Run(const f4x128_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f4x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        // keep int32_t for backward compatibility
        static_assert(is_same_v<ScaleTypeA, e8m0x4_bexp_t> ||
                          is_same_v<ScaleTypeA, e5m3x4_scale_t> ||
                          is_same_v<ScaleTypeA, e4m3x4_scale_t>,
                      "ScaleTypeA must be e8m0x4_bexp_t, e5m3x4_scale_t, or e4m3x4_scale_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x4_bexp_t> ||
                          is_same_v<ScaleTypeB, e5m3x4_scale_t> ||
                          is_same_v<ScaleTypeB, e4m3x4_scale_t>,
                      "ScaleTypeB must be e8m0x4_bexp_t, e5m3x4_scale_t, or e4m3x4_scale_t");
#if defined(__gfx125__)
        int32x16_t arg_a = bit_cast<int32x16_t>(reg_a);
        int32x8_t arg_b  = bit_cast<int32x8_t>(reg_b);
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(
                arg_a,
                arg_b,
                0,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1, // fix ScaleOpselA as 1
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value,
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value,
                bit_cast<int32_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

template <index_t MPerWave,
          index_t NPerWave,
          index_t ScaleOpselB,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct intrin_wmma_scale16_f32_32x16x128_f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselB, typename ScaleTypeA, typename ScaleTypeB>
struct intrin_wmma_scale16_f32_32x16x128_f4<32, 16, ScaleOpselB, ScaleTypeA, ScaleTypeB>
{
    template <class FloatC>
    __device__ static void Run(const f4x128_t& reg_a,
                               const ScaleTypeA& scale_a,
                               const f4x64_t& reg_b,
                               const ScaleTypeB& scale_b,
                               FloatC& reg_c)
    {
        static_assert(is_same_v<ScaleTypeA, e8m0x8_bexp_t> ||
                          is_same_v<ScaleTypeA, e5m3x8_scale_t> ||
                          is_same_v<ScaleTypeA, e4m3x8_scale_t>,
                      "ScaleTypeA must be e8m0x8_bexp_t, e5m3x8_scale_t, or e4m3x8_scale_t");
        static_assert(is_same_v<ScaleTypeB, e8m0x8_bexp_t> ||
                          is_same_v<ScaleTypeB, e5m3x8_scale_t> ||
                          is_same_v<ScaleTypeB, e4m3x8_scale_t>,
                      "ScaleTypeB must be e8m0x8_bexp_t, e5m3x8_scale_t, or e4m3x8_scale_t");
#if defined(__gfx125__)
        int32x16_t arg_a = bit_cast<int32x16_t>(reg_a);
        int32x8_t arg_b  = bit_cast<int32x8_t>(reg_b);
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(
                arg_a,
                arg_b,
                0,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1, // fix ScaleOpselA as 1
                wmma_impl::ScaleTypeSelector<ScaleTypeA>::value,
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,
                wmma_impl::ScaleTypeSelector<ScaleTypeB>::value,
                bit_cast<int64_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

} // namespace ck
#endif
