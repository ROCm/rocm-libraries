// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_wmma.hpp"

namespace ck {

enum struct WmmaInstr
{
    // gfx11
    wmma_f32_16x16x16_f16 = 0,
    wmma_f32_16x16x16_bf16,
    wmma_f16_16x16x16_f16,
    wmma_bf16_16x16x16_bf16,
    wmma_i32_16x16x16_iu8,
    wmma_i32_16x16x16_iu4,
    // gfx120x
    wmma_f32_16x16x16_f16_gfx12,
    wmma_f32_16x16x16_bf16_gfx12,
    wmma_f16_16x16x16_f16_gfx12,
    wmma_bf16_16x16x16_bf16_gfx12,
    wmma_i32_16x16x16_iu8_gfx12,
    wmma_f32_16x16x16_f8f8_gfx12,
    wmma_f32_16x16x16_f8bf8_gfx12,
    wmma_f32_16x16x16_bf8f8_gfx12,
    wmma_f32_16x16x16_bf8bf8_gfx12,
    // gfx125x
    wmma_f16_16x16x32_f16_gfx12,
    wmma_bf16_16x16x32_bf16_gfx12,
    wmma_f32_16x16x32_f16_gfx12,
    wmma_f32_16x16x32_bf16_gfx12,
    wmma_i32_16x16x64_iu8_gfx12,
    wmma_f32_16x16x64_f8f8_gfx12,
    wmma_f32_16x16x64_f8bf8_gfx12,
    wmma_f32_16x16x64_bf8f8_gfx12,
    wmma_f32_16x16x64_bf8bf8_gfx12,

    // gfx13
    wmma_f32_16x16_f16_gfx13,
    wmma_f32_16x16_bf16_gfx13,
    wmma_f16_16x16_f16_gfx13,
    wmma_bf16_16x16_bf16_gfx13,
    wmma_i32_16x16_iu8_gfx13,
    wmma_f32_16x16_iu8_gfx13,
    wmma_f16_16x16_f8f8_gfx13,
    wmma_f16_16x16_f8bf8_gfx13,
    wmma_f16_16x16_bf8f8_gfx13,
    wmma_f16_16x16_bf8bf8_gfx13,
    wmma_f32_16x16_f8f8_gfx13,
    wmma_f32_16x16_f8bf8_gfx13,
    wmma_f32_16x16_bf8f8_gfx13,
    wmma_f32_16x16_bf8bf8_gfx13,
    wmma_f32_16x16x64_f8f6f4_gfx13,
};

/*
 *  WMMA Wave Tile Always MxNxK = 16x16x16
 *  WAVE32
        -----------------------------------
        |RC0| | | | | | | | | | | | | | | |	   SubGroup 0
        |RC1| | | | | | | | | | | | | | | |
        |RC2| | | | | | | | | | | | | | | |
        |RC3|T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|
        |RC4|0|0|0|0|0|0|0|0|0|1|1|1|1|1|1|
        |RC5|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5|
        |RC6| | | | | | | | | | | | | | | |
        |RC7| | | | | | | | | | | | | | | |
        -----------------------------------
        |   | | | | | | | | | | | | | | | |	   SubGroup 1
        |   | | | | | | | | | | | | | | | |
        | T |T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|
        | 1 |1|1|1|2|2|2|2|2|2|2|2|2|2|3|3|
        | 6 |7|8|9|0|1|2|3|4|5|6|7|8|9|0|1|
        |   | | | | | | | | | | | | | | | |
        |   | | | | | | | | | | | | | | | |
        |   | | | | | | | | | | | | | | | |
        -----------------------------------


 *  WAVE64
        -----------------------------------
        |RC0|T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|	   SubGroup 0
        |RC1|0|0|0|0|0|0|0|0|0|1|1|1|1|1|1|
        |RC2|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5|
        |RC3|T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|
        -----------------------------------
        | T |T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|    SubGroup 1
        | 1 |1|1|1|2|2|2|2|2|2|2|2|2|2|3|3|
        | 6 |7|8|9|0|1|2|3|4|5|6|7|8|9|0|1|
        |   | | | | | | | | | | | | | | | |
        -----------------------------------
        | T |T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|	   SubGroup 2
        | 3 |3|3|3|3|3|3|3|4|4|4|4|4|4|4|4|
        | 2 |3|4|5|6|7|8|9|0|1|2|3|4|5|6|7|
        |   | | | | | | | | | | | | | | | |
        -----------------------------------
        | T |T|T|T|T|T|T|T|T|T|T|T|T|T|T|T|    SubGroup 3
        | 4 |4|5|5|5|5|5|5|5|5|5|5|6|6|6|6|
        | 8 |9|0|1|2|3|4|5|6|7|8|9|0|1|2|3|
        |   | | | | | | | | | | | | | | | |
        -----------------------------------

*   RC = Register for storing accumalted result
*	T  = Thread ID
*/

template <WmmaInstr Instr, index_t WaveSize, typename = void>
struct wmma_type
{
};

// A-swizzled
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_f16,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    // * Data Pixel
    static constexpr index_t m_per_wmma      = 16;
    static constexpr index_t n_per_wmma      = 16;
    static constexpr index_t k_per_wmma      = 16;
    static constexpr index_t k_per_blk       = 8;
    static constexpr index_t src_a_data_size = 2;
    static constexpr index_t src_b_data_size = 2;
    static constexpr index_t acc_data_size   = 4;
    static constexpr index_t acc_pack_number = 1;
    // * Thread mapping inside wave, num_thread_per_subgroups always alone N direction
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // * Fixed on gfx11, Will be wave mode dependent for future architectures
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    // * num_acc_vgprs_per_wave alone M direction
    // * num_subgroups alone M direction
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16x16_f16_w32<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_f32_16x16x16_f16_w64<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_bf16,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t k_per_blk                = 8;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16x16_bf16_w32<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_f32_16x16x16_bf16_w64<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16x16_f16,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t k_per_blk                = 8;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 2;
    static constexpr index_t acc_pack_number          = 2;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16x16_f16_w32<MPerWmma, NPerWmma, false>::Run(a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_f16_16x16x16_f16_w64<MPerWmma, NPerWmma, false>::Run(a, b, reg_c);
        }
    }
};
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_bf16_16x16x16_bf16,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t k_per_blk                = 8;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 2;
    static constexpr index_t acc_pack_number          = 2;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t Opsel,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_bf16_16x16x16_bf16_w32<MPerWmma, NPerWmma, false>::Run(a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_bf16_16x16x16_bf16_w64<MPerWmma, NPerWmma, false>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_i32_16x16x16_iu8,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t k_per_blk                = 8;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool neg_a = true,
              bool neg_b = true,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        if constexpr(wave_size == 32)
        {
            intrin_wmma_i32_16x16x16_iu8_w32<MPerWmma, NPerWmma, neg_a, neg_b, clamp>::Run(
                a, b, reg_c);
        }
        else if constexpr(wave_size == 64)
        {
            intrin_wmma_i32_16x16x16_iu8_w64<MPerWmma, NPerWmma, neg_a, neg_b, clamp>::Run(
                a, b, reg_c);
        }
    }
};

// gfx12

// A-swizzled
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_f16_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    // * Data Pixel
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size = 2;
    // static constexpr index_t src_b_data_size = 2;
    // static constexpr index_t acc_data_size   = 4;
    // * Thread mapping inside wave, num_thread_per_subgroups always alone N direction
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // * Fixed for gfx11, Will be wave mode dependent on gfx12
    // static constexpr index_t num_src_a_vgprs_per_wave = k_per_wmma / 2 * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = k_per_wmma / 2 * src_b_data_size / 4;
    // * num_acc_vgprs_per_wave alone M direction
    // * num_subgroups alone M direction
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16x16_f16_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_bf16_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16x16_bf16_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16x16_f16_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    // * Data Pixel
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    // static constexpr index_t src_a_data_size = 2;
    // static constexpr index_t src_b_data_size = 2;
    // static constexpr index_t acc_data_size   = 4;
    // * Thread mapping inside wave, num_thread_per_subgroups always alone N direction
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // * Fixed in Navi3x, Will be wave mode dependent on Navi4x
    // static constexpr index_t num_src_a_vgprs_per_wave = k_per_wmma / 2 * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = k_per_wmma / 2 * src_b_data_size / 4;
    // * num_acc_vgprs_per_wave alone M direction
    // * num_subgroups alone M direction
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16x16_f16_w32<MPerWmma, NPerWmma, 0>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_bf16_16x16x16_bf16_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_bf16_16x16x16_bf16_w32<MPerWmma, NPerWmma, 0>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_i32_16x16x16_iu8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool neg_a = true,
              bool neg_b = true,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_i32_16x16x16_iu8_w32_gfx12<MPerWmma, NPerWmma, neg_a, neg_b, clamp>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_f8f8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t k_per_blk                = 8;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size              = Number<WaveSize>{};
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
#ifdef __gfx12__
            intrin_wmma_f32_16x16x16_f8f8_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
#else
            ignore = a;
            ignore = b;
            ignore = reg_c;
#endif
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_f8bf8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t k_per_blk                = 8;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size              = Number<WaveSize>{};
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
#ifdef __gfx12__
            intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
#else
            ignore = a;
            ignore = b;
            ignore = reg_c;
#endif
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_bf8f8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t k_per_blk                = 8;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size              = Number<WaveSize>{};
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
#ifdef __gfx12__
            intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
#else
            ignore = a;
            ignore = b;
            ignore = reg_c;
#endif
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_bf8bf8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 16;
    static constexpr index_t k_per_blk                = 8;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size              = Number<WaveSize>{};
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
#ifdef __gfx12__
            intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
#else
            ignore = a;
            ignore = b;
            ignore = reg_c;
#endif
        }
    }
};

// gfx125x

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16x32_f16_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 32;
    static constexpr index_t k_per_blk                = 16;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 2;
    static constexpr index_t acc_pack_number          = 2;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16x32_f16<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_bf16_16x16x32_bf16_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 32;
    static constexpr index_t k_per_blk                = 16;
    static constexpr index_t src_a_data_size          = 2;
    static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 2;
    static constexpr index_t acc_pack_number          = 2;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size                = Number<WaveSize>{};
    static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma * acc_data_size * acc_pack_number / wave_size / 4;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t Opsel,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_bf16_16x16x32_bf16<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};
// A-swizzled
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x32_f16_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    // * Data Pixel
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 32;
    static constexpr index_t k_per_blk  = 16;
    // static constexpr index_t src_a_data_size = 2;
    // static constexpr index_t src_b_data_size = 2;
    // static constexpr index_t acc_data_size   = 4;
    // * Thread mapping inside wave, num_thread_per_subgroups always alone N direction
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // * Fixed in Navi3x, Will be wave mode dependent on Navi4x
    // static constexpr index_t num_src_a_vgprs_per_wave = k_per_wmma / 2 * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = k_per_wmma / 2 * src_b_data_size / 4;
    // * num_acc_vgprs_per_wave alone M direction
    // * num_subgroups alone M direction
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16x32_f16<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x32_bf16_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 32;
    static constexpr index_t k_per_blk  = 16;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16x32_bf16<MPerWmma, NPerWmma>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_i32_16x16x64_iu8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 64;
    static constexpr index_t k_per_blk  = 32;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool neg_a = true,
              bool neg_b = true>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_i32_16x16x64_iu8<MPerWmma, NPerWmma, neg_a, neg_b>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x64_f8f8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 64;
    static constexpr index_t k_per_blk                = 32;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size              = Number<WaveSize>{};
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
#ifdef __gfx125__
            intrin_wmma_f32_16x16x64_f8f8<MPerWmma, NPerWmma>::Run(a, b, reg_c);
#else
            ignore = a;
            ignore = b;
            ignore = reg_c;
#endif
        }
    }
};
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x64_bf8f8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 64;
    static constexpr index_t k_per_blk                = 32;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size              = Number<WaveSize>{};
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
#ifdef __gfx125__
            intrin_wmma_f32_16x16x64_bf8f8<MPerWmma, NPerWmma>::Run(a, b, reg_c);
#else
            ignore = a;
            ignore = b;
            ignore = reg_c;
#endif
        }
    }
};
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x64_bf8bf8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 64;
    static constexpr index_t k_per_blk                = 32;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size              = Number<WaveSize>{};
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
#ifdef __gfx125__
            intrin_wmma_f32_16x16x64_bf8bf8<MPerWmma, NPerWmma>::Run(a, b, reg_c);
#else
            ignore = a;
            ignore = b;
            ignore = reg_c;
#endif
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x64_f8bf8_gfx12,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32 || WaveSize == 64>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma               = 16;
    static constexpr index_t n_per_wmma               = 16;
    static constexpr index_t k_per_wmma               = 64;
    static constexpr index_t k_per_blk                = 32;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size              = Number<WaveSize>{};
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx12 wmma");
        if constexpr(wave_size == 32)
        {
#ifdef __gfx125__
            intrin_wmma_f32_16x16x64_f8bf8<MPerWmma, NPerWmma>::Run(a, b, reg_c);
#else
            ignore = a;
            ignore = b;
            ignore = reg_c;
#endif
        }
    }
};

// gfx13
template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16_f16_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    // * Data Pixel
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size = 2;
    // static constexpr index_t src_b_data_size = 2;
    // static constexpr index_t acc_data_size   = 4;
    // * Thread mapping inside wave, num_thread_per_subgroups always alone N direction
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // * Fixed in Navi3x, Will be wave mode dependent on Navi4x
    // static constexpr index_t num_src_a_vgprs_per_wave = k_per_wmma / 2 * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = k_per_wmma / 2 * src_b_data_size / 4;
    // * num_acc_vgprs_per_wave alone M direction
    // * num_subgroups alone M direction
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 2;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;

    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16_f16f16_w32<MPerWmma, NPerWmma, false, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16_bf16_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 2;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16_bf16bf16_w32<MPerWmma, NPerWmma, false, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16_f16_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    // * Data Pixel
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size = 2;
    // static constexpr index_t src_b_data_size = 2;
    // static constexpr index_t acc_data_size   = 4;
    // * Thread mapping inside wave, num_thread_per_subgroups always alone N direction
    static constexpr index_t acc_data_size            = 2;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // * Fixed in Navi3x, Will be wave mode dependent on Navi4x
    // static constexpr index_t num_src_a_vgprs_per_wave = k_per_wmma / 2 * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = k_per_wmma / 2 * src_b_data_size / 4;
    // * num_acc_vgprs_per_wave alone M direction
    // * num_subgroups alone M direction
    static constexpr index_t num_acc_vgprs_per_wave =
        m_per_wmma * n_per_wmma / wave_size / acc_data_size;
    static constexpr index_t num_subgroups = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16_f16f16_w32<MPerWmma, NPerWmma, false, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_bf16_16x16_bf16_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;

    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_bf16_16x16_bf16bf16_w32<MPerWmma, NPerWmma, false, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_i32_16x16_iu8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool neg_a = false,
              bool neg_b = false,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_i32_16x16_iu8iu8_w32<MPerWmma,
                                             NPerWmma,
                                             neg_a,
                                             neg_b,
                                             clamp,
                                             KPerWmma / 16>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16_iu8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool neg_a = false,
              bool neg_b = false,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16_iu8iu8_w32<MPerWmma,
                                             NPerWmma,
                                             neg_a,
                                             neg_b,
                                             clamp,
                                             KPerWmma / 16>::Run(a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16_f8f8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16_f8f8_w32<MPerWmma, NPerWmma, clamp, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16_f8bf8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16_f8bf8_w32<MPerWmma, NPerWmma, clamp, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16_bf8f8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16_bf8f8_w32<MPerWmma, NPerWmma, clamp, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16_bf8bf8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16_bf8bf8_w32<MPerWmma, NPerWmma, clamp, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f32_16x16x64_f8f6f4_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 64;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              typename AType,
              typename BType,
              index_t ABlockSel,
              index_t BBlockSel,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a,
                        const FloatB& b,
                        const int32_t& a_scale,
                        const int32_t& b_scale,
                        FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f32_16x16_f8f6f4_w32<MPerWmma,
                                             NPerWmma,
                                             AType,
                                             BType,
                                             ABlockSel,
                                             BBlockSel,
                                             false>::Run(a, b, a_scale, b_scale, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16_f8f8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16_f8f8_w32<MPerWmma, NPerWmma, clamp, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16_f8bf8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16_f8bf8_w32<MPerWmma, NPerWmma, clamp, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16_bf8f8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16_bf8f8_w32<MPerWmma, NPerWmma, clamp, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <index_t WaveSize>
struct wmma_type<WmmaInstr::wmma_f16_16x16_bf8bf8_gfx13,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 32>>
{
    // Absolute fixing property
    static constexpr index_t m_per_wmma = 16;
    static constexpr index_t n_per_wmma = 16;
    static constexpr index_t k_per_wmma = 16;
    static constexpr index_t k_per_blk  = 8;
    // static constexpr index_t src_a_data_size          = 2;
    // static constexpr index_t src_b_data_size          = 2;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t acc_pack_number          = 1;
    static constexpr index_t num_thread_per_subgroups = n_per_wmma;

    // Wave mode dependent propety
    static constexpr index_t wave_size = Number<WaveSize>{};
    // static constexpr index_t num_src_a_vgprs_per_wave = m_per_wmma * src_a_data_size / 4;
    // static constexpr index_t num_src_b_vgprs_per_wave = n_per_wmma * src_b_data_size / 4;
    static constexpr index_t num_acc_vgprs_per_wave = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_subgroups          = wave_size / num_thread_per_subgroups;
    // * num_consecutive_vgprs means how many vgprs in consecutive
    static constexpr index_t num_acc_per_thread  = m_per_wmma * n_per_wmma / wave_size;
    static constexpr index_t num_consecutive_acc = 4;
    static constexpr index_t loop_of_consecutive = num_acc_per_thread / num_consecutive_acc;
    template <index_t MPerWmma,
              index_t NPerWmma,
              index_t KPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        static_assert(wave_size == 32, "only support wave32 for gfx13 wmma");
        if constexpr(wave_size == 32)
        {
            intrin_wmma_f16_16x16_bf8bf8_w32<MPerWmma, NPerWmma, clamp, KPerWmma / 16>::Run(
                a, b, reg_c);
        }
    }
};

template <typename src_type_a,
          typename src_type_b,
          typename dst_type,
          index_t MPerWmma,
          index_t NPerWmma>
struct WmmaSelector
{
    template <typename src_type_a_,
              typename src_type_b_,
              typename dst_type_,
              index_t MPerWmma_,
              index_t NPerWmma_>
    static constexpr auto GetWmma();

    template <>
    constexpr auto GetWmma<half_t, half_t, float, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_f32_16x16_f16_gfx13;
#elif defined(__gfx120__)
        return WmmaInstr::wmma_f32_16x16x16_f16_gfx12;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_f32_16x16x32_f16_gfx12;
#else
        return WmmaInstr::wmma_f32_16x16x16_f16;
#endif
    }

    template <>
    constexpr auto GetWmma<bhalf_t, bhalf_t, float, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_f32_16x16_bf16_gfx13;
#elif defined(__gfx120__)
        return WmmaInstr::wmma_f32_16x16x16_bf16_gfx12;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_f32_16x16x32_bf16_gfx12;
#else
        return WmmaInstr::wmma_f32_16x16x16_bf16;
#endif
    }

    template <>
    constexpr auto GetWmma<half_t, half_t, half_t, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_f16_16x16_f16_gfx13;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_f16_16x16x32_f16_gfx12;
#else
        return WmmaInstr::wmma_f16_16x16x16_f16;
#endif
    }

    template <>
    constexpr auto GetWmma<bhalf_t, bhalf_t, bhalf_t, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_bf16_16x16_bf16_gfx13;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_bf16_16x16x32_bf16_gfx12;
#else
        return WmmaInstr::wmma_bf16_16x16x16_bf16;
#endif
    }

    template <>
    constexpr auto GetWmma<int8_t, int8_t, int, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_i32_16x16_iu8_gfx13;
#elif defined(__gfx120__)
        return WmmaInstr::wmma_i32_16x16x16_iu8_gfx12;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_i32_16x16x64_iu8_gfx12;
#else
        return WmmaInstr::wmma_i32_16x16x16_iu8;
#endif
    }

    template <>
    constexpr auto GetWmma<uint8_t, uint8_t, int, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_i32_16x16_iu8_gfx13;
#elif defined(__gfx12__)
        return WmmaInstr::wmma_i32_16x16x16_iu8_gfx12;
#else
        return WmmaInstr::wmma_i32_16x16x16_iu8;
#endif
    }

#if defined(__gfx13__)
    template <>
    constexpr auto GetWmma<int8_t, int8_t, float, 16, 16>()
    {
        return WmmaInstr::wmma_f32_16x16_iu8_gfx13;
    }

    template <>
    constexpr auto GetWmma<uint8_t, uint8_t, float, 16, 16>()
    {
        return WmmaInstr::wmma_f32_16x16_iu8_gfx13;
    }
#endif

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    constexpr auto GetWmma<int4_t, int4_t, int, 16, 16>()
    {
        return WmmaInstr::wmma_i32_16x16x16_iu4;
    }

    template <>
    constexpr auto GetWmma<uint4_t, uint4_t, int, 16, 16>()
    {
        return WmmaInstr::wmma_i32_16x16x16_iu4;
    }
#endif

    template <>
    constexpr auto GetWmma<f8_t, f8_t, float, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_f32_16x16_f8f8_gfx13;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_f32_16x16x64_f8f8_gfx12; // A0
#else
        // note: host side will select this instruction, which is not consistent if target is
        // gfx1250 potential issue may occur when host side uses this struct.
        return WmmaInstr::wmma_f32_16x16x16_f8f8_gfx12;
#endif
    }

    template <>
    constexpr auto GetWmma<f8_t, bf8_t, float, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_f32_16x16_f8bf8_gfx13;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_f32_16x16x64_f8bf8_gfx12;
#else
        return WmmaInstr::wmma_f32_16x16x16_f8bf8_gfx12;
#endif
    }

    template <>
    constexpr auto GetWmma<bf8_t, f8_t, float, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_f32_16x16_bf8f8_gfx13;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_f32_16x16x64_bf8f8_gfx12;
#else
        return WmmaInstr::wmma_f32_16x16x16_bf8f8_gfx12;
#endif
    }

    template <>
    constexpr auto GetWmma<bf8_t, bf8_t, float, 16, 16>()
    {
#if defined(__gfx13__)
        return WmmaInstr::wmma_f32_16x16_bf8bf8_gfx13;
#elif defined(__gfx125__)
        return WmmaInstr::wmma_f32_16x16x64_bf8bf8_gfx12;
#else
        return WmmaInstr::wmma_f32_16x16x16_bf8bf8_gfx12;
#endif
    }

#if defined(__gfx13__)
    template <>
    constexpr auto GetWmma<f8_t, f8_t, half_t, 16, 16>()
    {
        return WmmaInstr::wmma_f16_16x16_f8f8_gfx13;
    }

    template <>
    constexpr auto GetWmma<f8_t, bf8_t, half_t, 16, 16>()
    {
        return WmmaInstr::wmma_f16_16x16_f8bf8_gfx13;
    }

    template <>
    constexpr auto GetWmma<bf8_t, f8_t, half_t, 16, 16>()
    {
        return WmmaInstr::wmma_f16_16x16_bf8f8_gfx13;
    }

    template <>
    constexpr auto GetWmma<bf8_t, bf8_t, half_t, 16, 16>()
    {
        return WmmaInstr::wmma_f16_16x16_bf8bf8_gfx13;
    }
#endif

    // get_warp_size do not return the correct wavesize, hardcode to 32 as workaround
    static constexpr auto selected_wmma_instr = []() {
        if constexpr(is_mx_type_t_v<src_type_a> && is_mx_type_t_v<src_type_b>)
            return WmmaInstr::wmma_f32_16x16x64_f8f6f4_gfx13;
        else
            return GetWmma<src_type_a, src_type_b, dst_type, MPerWmma, NPerWmma>();
    }();
    static constexpr auto selected_wmma = wmma_type<selected_wmma_instr, Number<32>{}>{};

    __host__ __device__ constexpr WmmaSelector()
    {
        static_assert(selected_wmma.m_per_wmma == 16, "WRONG! WMMA_M must equal to 16");
        static_assert(selected_wmma.n_per_wmma == 16, "WRONG! WMMA_N must equal to 16");
        static_assert(selected_wmma.k_per_wmma % 16 == 0, "WRONG! WMMA_K must mod(16) = 0");
        static_assert(selected_wmma.wave_size * selected_wmma.num_acc_vgprs_per_wave *
                              selected_wmma.acc_data_size * selected_wmma.acc_pack_number ==
                          selected_wmma.m_per_wmma * selected_wmma.n_per_wmma * 4,
                      "WRONG! Invalid Number of Accumulator Register");
    }
};

#if defined(__gfx13__)
template <typename src_type_a,
          typename src_type_b,
          typename dst_type,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t KPerWmma,
          index_t KPack,
          bool TransposeC      = false,
          bool WaveGroup       = false,
          bool AssemblyBackend = false>
#else
template <typename src_type_a,
          typename src_type_b,
          typename dst_type,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t KPack,
          bool TransposeC      = false,
          bool AssemblyBackend = false>
#endif
struct WmmaGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using CIndex   = MultiIndex<2>;
    using CIndex3D = MultiIndex<3>;

    __host__ __device__ constexpr WmmaGemm()
    {
        static_assert(NPerWmma == 16 && MPerWmma == 16,
                      "Only support GemmNPerWmma == 16 and GemmMPerWmma == 16 for wmma");
#if defined(__gfx13__)
        static_assert(KPack % KPerWmma == 0, "KPack should be multiple of k_per_wmma");
#endif
    }

    // WMMA output supporting C = A * B
    // Vector Write
    // MPerWMMA_NPerWMMA -> MSubGroup_..._NPerWMMA_MAccVgprPerWave
    template <typename CDesc_MBlockxRepeat_MWave_MPerWMMA_NBlockxRepeat_NWave_NPerWMMA>
    __host__ __device__ static constexpr auto
    MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
        const CDesc_MBlockxRepeat_MWave_MPerWMMA_NBlockxRepeat_NWave_NPerWMMA&
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma)
    {
        const auto MBlockxRepeat =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I0);
        const auto NBlockxRepeat =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I3);
        const auto MWave =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I1);
        const auto NWave =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I4);
#if defined(__gfx13__)
        // for gfx13 the layout for MPerWMMA is divided to 3 components because of layout change.
        // the 3 components are: consecutive accumulators-> how many consecutive accumulators ->
        // number of subgroups; because in C's layout there have some stride between one subgroup's
        // consecutive accumulators.
        return transform_tensor_descriptor(
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma,
            make_tuple(make_pass_through_transform(MBlockxRepeat),
                       make_pass_through_transform(MWave),
                       make_unmerge_transform(make_tuple(Number<wmma_instr.num_subgroups>{},
                                                         Number<wmma_instr.loop_of_consecutive>{},
                                                         Number<wmma_instr.num_consecutive_acc>{})),
                       make_pass_through_transform(NBlockxRepeat),
                       make_pass_through_transform(NWave),
                       make_pass_through_transform(Number<wmma_instr.num_thread_per_subgroups>{})),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2, 6, 7>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}));
#else
        return transform_tensor_descriptor(
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma,
            make_tuple(
                make_pass_through_transform(MBlockxRepeat),
                make_pass_through_transform(MWave),
                make_unmerge_transform(make_tuple(Number<wmma_instr.num_subgroups>{},
                                                  Number<wmma_instr.num_acc_vgprs_per_wave>{})),
                make_pass_through_transform(NBlockxRepeat),
                make_pass_through_transform(NWave),
                make_pass_through_transform(Number<wmma_instr.num_thread_per_subgroups>{})),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2, 6>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}));
#endif
    }

    // Transposed WMMA Output C' = B' * A'
    template <typename CDesc_MBlockxRepeat_MWave_MPerWMMA_NBlockxRepeat_NWave_NPerWMMA>
    __host__ __device__ static constexpr auto
    MakeCDesc_MBlockxRepeat_MWave_MThreadPerSubGroup_NBlockxRepeat_NWave_NSubGroup_NAccVgprs(
        const CDesc_MBlockxRepeat_MWave_MPerWMMA_NBlockxRepeat_NWave_NPerWMMA&
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma)
    {
        const auto MBlockxRepeat =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I0);
        const auto NBlockxRepeat =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I3);
        const auto MWave =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I1);
        const auto NWave =
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma.GetLength(I4);
#if defined(__gfx13__)
        return transform_tensor_descriptor(
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma,
            make_tuple(
                make_pass_through_transform(MBlockxRepeat),
                make_pass_through_transform(MWave),
                make_pass_through_transform(Number<wmma_instr.num_thread_per_subgroups>{}),
                make_pass_through_transform(NBlockxRepeat),
                make_pass_through_transform(NWave),
                make_unmerge_transform(make_tuple(Number<wmma_instr.loop_of_consecutive>{},
                                                  Number<wmma_instr.num_subgroups>{},
                                                  Number<wmma_instr.num_consecutive_acc>{}))),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6, 7>{}));
#else
        return transform_tensor_descriptor(
            c_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma,
            make_tuple(
                make_pass_through_transform(MBlockxRepeat),
                make_pass_through_transform(MWave),
                make_pass_through_transform(Number<wmma_instr.num_thread_per_subgroups>{}),
                make_pass_through_transform(NBlockxRepeat),
                make_pass_through_transform(NWave),
                make_unmerge_transform(make_tuple(Number<wmma_instr.num_subgroups>{},
                                                  Number<wmma_instr.num_acc_vgprs_per_wave>{}))),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6>{}));
#endif
    }

    __device__ static constexpr index_t GetRegSizePerWmma()
    {
        return wmma_instr.num_acc_vgprs_per_wave * wmma_instr.acc_pack_number;
    }

    __device__ static constexpr index_t GetWaveSize() { return wmma_instr.wave_size; }

    __device__ static constexpr index_t GetKPerWaveBlk() { return wmma_instr.k_per_blk; }

#if defined(__gfx13__)
    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(
            (is_same<src_type_a, half_t>::value && is_same<src_type_b, half_t>::value &&
             is_same<dst_type, float>::value) ||
                (is_same<src_type_a, bhalf_t>::value && is_same<src_type_b, bhalf_t>::value &&
                 is_same<dst_type, float>::value) ||
                (is_same<src_type_a, half_t>::value && is_same<src_type_b, half_t>::value &&
                 is_same<dst_type, half_t>::value) ||
                (is_same<src_type_a, bhalf_t>::value && is_same<src_type_b, bhalf_t>::value &&
                 is_same<dst_type, bhalf_t>::value) ||
                ((is_same<src_type_a, int8_t>::value || is_same<src_type_a, uint8_t>::value) &&
                 (is_same<src_type_b, int8_t>::value || is_same<src_type_b, uint8_t>::value) &&
                 (is_same<dst_type, int32_t>::value || is_same<dst_type, float>::value)) ||
                ((is_same<src_type_a, f8_t>::value || is_same<src_type_a, bf8_t>::value) &&
                 (is_same<src_type_b, f8_t>::value || is_same<src_type_b, bf8_t>::value) &&
                 (is_same<dst_type, float>::value || is_same<dst_type, half_t>::value)) ||
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                ((is_same<src_type_a, int4_t>::value || is_same<src_type_a, uint4_t>::value) &&
                 (is_same<src_type_b, int4_t>::value || is_same<src_type_b, uint4_t>::value) &&
                 is_same<dst_type, int32_t>::value) ||
#endif
                false,
            "base type couple must be (half, float), (bhalf, float), (half, half), (bhalf, bhalf), "
            "((f8 or bf8, f8 or bf8), float), (int8, int32) or (int4, int32)!");
        if constexpr(is_same<src_type_a, int8_t>::value || is_same<src_type_a, uint8_t>::value
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                     || is_same<src_type_a, int4_t>::value || is_same<src_type_a, uint4_t>::value
#endif
        )
        {
            constexpr bool neg_a = is_same<src_type_a, int8_t>::value
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                                   || is_same<src_type_a, int4_t>::value
#endif
                ;
            constexpr bool neg_b = is_same<src_type_b, int8_t>::value
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                                   || is_same<src_type_b, int4_t>::value
#endif
                ;
            static_for<0, KPack / KPerWmma, 1>{}([&](auto k) {
                if constexpr(!TransposeC)
                {
                    wmma_instr.template run<MPerWmma,
                                            NPerWmma,
                                            KPerWmma,
                                            remove_cvref_t<decltype(p_a_wave[k])>,
                                            remove_cvref_t<decltype(p_b_wave[k])>,
                                            remove_cvref_t<decltype(p_c_thread)>,
                                            neg_a,
                                            neg_b>(p_a_wave[k], p_b_wave[k], p_c_thread);
                }
                else
                {
                    wmma_instr.template run<MPerWmma,
                                            NPerWmma,
                                            KPerWmma,
                                            remove_cvref_t<decltype(p_b_wave[k])>,
                                            remove_cvref_t<decltype(p_a_wave[k])>,
                                            remove_cvref_t<decltype(p_c_thread)>,
                                            neg_b,
                                            neg_a>(p_b_wave[k], p_a_wave[k], p_c_thread);
                }
            });
        }
        else
        {
            static_for<0, KPack / KPerWmma, 1>{}([&](auto k) {
                if constexpr(!TransposeC)
                {
                    wmma_instr.template run<MPerWmma, NPerWmma, KPerWmma>(
                        p_a_wave[k], p_b_wave[k], p_c_thread);
                }
                else
                {
                    wmma_instr.template run<MPerWmma, NPerWmma, KPerWmma>(
                        p_b_wave[k], p_a_wave[k], p_c_thread);
                }
            });
        }
    }
#else
    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(
            (is_same<src_type_a, half_t>::value && is_same<src_type_b, half_t>::value &&
             is_same<dst_type, float>::value) ||
                (is_same<src_type_a, bhalf_t>::value && is_same<src_type_b, bhalf_t>::value &&
                 is_same<dst_type, float>::value) ||
                (is_same<src_type_a, half_t>::value && is_same<src_type_b, half_t>::value &&
                 is_same<dst_type, half_t>::value) ||
                (is_same<src_type_a, bhalf_t>::value && is_same<src_type_b, bhalf_t>::value &&
                 is_same<dst_type, bhalf_t>::value) ||
                (is_same<src_type_a, int8_t>::value && is_same<src_type_b, int8_t>::value &&
                 is_same<dst_type, int32_t>::value) ||
                ((is_same<src_type_a, f8_t>::value || is_same<src_type_a, bf8_t>::value) &&
                 (is_same<src_type_b, f8_t>::value || is_same<src_type_b, bf8_t>::value) &&
                 is_same<dst_type, float>::value) ||
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                (is_same<src_type_a, int4_t>::value && is_same<src_type_b, int4_t>::value &&
                 is_same<dst_type, int32_t>::value) ||
#endif
                false,
            "base type couple must be (half, float), (bhalf, float), (half, half), (bhalf, bhalf), "
            "((f8 or bf8, f8 or bf8), float), (int8, int32) or (int4, int32)!");
        static_for<0, KPack / wmma_instr.k_per_wmma, 1>{}([&](auto k) {
            // Integer wmma operators need extra input flags to indicate if the input is signed or
            // unsigned. At the moment CK supports only signed integer inputs, so these flags are
            // hardcoded.
            if constexpr(!TransposeC)
            {
                wmma_instr.template run<MPerWmma, NPerWmma>(p_a_wave[k], p_b_wave[k], p_c_thread);
            }
            else
            {
                wmma_instr.template run<MPerWmma, NPerWmma>(p_b_wave[k], p_a_wave[k], p_c_thread);
            }
        });
    }
#endif

#ifdef CK_EXTENSION_MX_TYPE
    template <int AScaleSel, int BScaleSel, class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave,
                        const FloatB& p_b_wave,
                        const int32_t& a_scale_value,
                        const int32_t& b_scale_value,
                        FloatC& p_c_thread) const
    {
        static_for<0, KPack / wmma_instr.k_per_wmma, 1>{}([&](auto k) {
            if constexpr(!TransposeC)
            {
                wmma_instr
                    .template run<MPerWmma, NPerWmma, src_type_a, src_type_b, AScaleSel, BScaleSel>(
                        p_a_wave[k], p_b_wave[k], a_scale_value, b_scale_value, p_c_thread);
            }
            else
            {
                wmma_instr
                    .template run<MPerWmma, NPerWmma, src_type_a, src_type_b, AScaleSel, BScaleSel>(
                        p_b_wave[k], p_a_wave[k], b_scale_value, a_scale_value, p_c_thread);
            }
        });
    }
#endif

    __device__ static auto GetLaneId()
    {
#if(defined(__gfx13__))
        if constexpr(WaveGroup == false)
        {
            return get_thread_local_1d_id() % wmma_instr.wave_size;
        }
        else
#endif
        {
            return get_lane_id();
        }
    }

    __device__ static auto GetSubGroupId()
    {
#if(defined(__gfx13__))
        // subgroup in gfx13. each row needs 2 threads to load and the layout between these two
        // threads will use this function to get
        return GetLaneId() & 1;
#else
        static_assert(wmma_instr.num_thread_per_subgroups * wmma_instr.num_subgroups ==
                          wmma_instr.wave_size,
                      "");
        return (GetLaneId() / wmma_instr.num_thread_per_subgroups) % wmma_instr.num_subgroups;
#endif
    }

    __device__ static auto GetLaneIdUnderSubGroup()
    {
        return GetLaneId() % wmma_instr.num_thread_per_subgroups;
    }
    __device__ static auto GetSwizzledLaneIdLow()
    {
        return ((GetLaneIdUnderSubGroup() & 1) << 3) | (GetLaneIdUnderSubGroup() >> 1);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex()
    {
#if(defined(__gfx12__))
        return GetLaneIdUnderSubGroup();
#elif(defined(__gfx13__))
        // the M dimension in gfx13; where each row needs to 2 threads to load
        // this function is used to map lane id to row id
        return GetLaneId() >> 1;
#else
        return TransposeC ? GetLaneIdUnderSubGroup() : GetSwizzledLaneIdLow();
#endif
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex()
    {
#if(defined(__gfx12__))
        return GetLaneIdUnderSubGroup();
#elif(defined(__gfx13__))
        // the N dimension in gfx13; where each row needs to 2 threads to load
        // this function is used to map lane id to row id
        return GetLaneId() >> 1;
#else
        return TransposeC ? GetSwizzledLaneIdLow() : GetLaneIdUnderSubGroup();
#endif
    }

    __device__ static CIndex GetBeginOfThreadBlk()
    {
#if(defined(__gfx13__))
        // the M, N offset in C matrix in gfx13; one column in gfx13 needs 2 threads to load
        // in row dimension needs 16 threads to load. details check gfx13 shader programming guide
        index_t n_offset = GetLaneId() / wmma_instr.num_subgroups;
        index_t m_offset =
            (GetLaneId() % wmma_instr.num_subgroups) * wmma_instr.num_consecutive_acc;
#else
        index_t n_offset = GetLaneIdUnderSubGroup();
        index_t m_offset = GetSubGroupId() * wmma_instr.num_acc_vgprs_per_wave;
#endif
        return TransposeC ? CIndex{n_offset, m_offset} : CIndex{m_offset, n_offset};
    }

    __device__ static CIndex3D GetBeginOfThreadBlk3D()
    {
#if(defined(__gfx13__))
        index_t n_offset = GetLaneId() / wmma_instr.num_subgroups;
        index_t m_offset = GetLaneId() % wmma_instr.num_subgroups;
#else
        index_t n_offset = GetLaneIdUnderSubGroup();
        index_t m_offset = GetSubGroupId();
#endif
        return TransposeC ? CIndex3D{n_offset, m_offset, I0} : CIndex3D{m_offset, n_offset, I0};
    }
    static constexpr auto wmma =
        WmmaSelector<src_type_a, src_type_b, dst_type, MPerWmma, NPerWmma>{};
    static constexpr auto wmma_instr = wmma.selected_wmma;

    __host__ __device__ static constexpr auto
    GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths()
    {
#if defined(__gfx13__)
        // Loop | Concecutive | AccStride
        return make_tuple(I1,
                          Number<wmma_instr.loop_of_consecutive>{},
                          Number<wmma_instr.num_consecutive_acc>{},
                          Number<wmma_instr.acc_pack_number>{});
#else
        return make_tuple(I1,
                          I1,
                          Number<wmma_instr.num_acc_vgprs_per_wave>{},
                          Number<wmma_instr.acc_pack_number>{});
#endif
    }
};

} // namespace ck
