// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Shared boilerplate for grouped convolution CK Tile kernel variants.
//
// Each variant (4c v3, 16c v2, …) provides:
//   - A Config struct with variant-specific wave-count fields; all common
//     fields (kh, kw, n_fold, direction, swizzle_type, epilogue, group_size())
//     are inherited from ConfigBase.
//   - A TileConstants<cfg> struct that inherits TileConstantsBase<cfg> and
//     adds only the variant-specific tile distributions (Mfma,
//     Weight::MakeLdsReadTileDistribution,
//     Output::MakeDramWriteTileDistributionNarrow).
//   - A WeightLoader struct — only the Dgrad TransposeLayout<> specialisation
//     differs; the Fprop path is provided by weight_read_fprop<TC>() here.
//
// Everything else lives here and is shared verbatim.

#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_descriptors.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_weight_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_compute_loop.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/ops/direct_convolution/utils/launch_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/kernel_variant.hpp"
#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include <hip/hip_runtime.h>
#include <string>

namespace ck_tile::direct_conv {

// ======================================================================
// TileConstantsBase — all constexpr members derivable from cfg.
//
// cfg must provide: block_c(), block_q(), num_waves(), block_size(),
//                   group_size(), block_groups(), kh, kw, swizzle_type.
//
// Variant-specific nested structs (Mfma, and the *TileDistribution*
// methods on Weight and Output) are added in each variant's TileConstants.
// ======================================================================
template <auto cfg>
struct TileConstantsBase
{
    // Channels per group (input channel must be equal to output channel for fp16 grouped conv).
    static constexpr int GROUP_SIZE   = cfg.group_size();

    // Channels per group in units of 4 (unit2)
    static constexpr int GROUP_SIZE_4 = GROUP_SIZE / 4;

    // Channel per group in units of 8 (unit4)
    static constexpr int GROUP_SIZE_8 = GROUP_SIZE / 8;

    // Number of conv groups processed by each thread block (block_group() * group_size() = total channels).
    static constexpr int BLOCK_GROUPS = cfg.block_groups();

    static constexpr int BLOCK_Q  = cfg.block_q();
    static constexpr int BLOCK_W  = BLOCK_Q + (cfg.kw - 1);

    static constexpr int BLOCK_C8 = cfg.block_c() / 8;
    static constexpr int BLOCK_C  = BLOCK_C8 * 8;
    static constexpr int BLOCK_C4 = BLOCK_C / 4;

    static constexpr int STORE_VECS = BLOCK_Q * BLOCK_C8;

    static constexpr int NUM_INPUT_LDS_BUFFERS      = 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C8   = BLOCK_C8 * BLOCK_W;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C4   = INPUT_LDS_BUFFER_SIZE_C8 * 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_FP16 = INPUT_LDS_BUFFER_SIZE_C8 * 8;

    static constexpr int NUM_WAVES     = cfg.num_waves();
    static constexpr int LANES_PER_ROW = 64 / BLOCK_C8;
    static constexpr int TOTAL_SPATIAL = cfg.block_size() / BLOCK_C8;

    static constexpr int KH_KW = cfg.kh * cfg.kw;
    static constexpr int KW    = cfg.kw;

    static constexpr SwizzleType SWIZZLE_TYPE = cfg.swizzle_type;

    // -----------------------------------------------------------------------
    // Weight — LDS sizing and descriptor factories (shared formula).
    // The variant-specific MakeLdsReadTileDistribution() is added by the
    // derived TileConstants.
    // -----------------------------------------------------------------------
    struct Weight
    {
        using Shared = typename SharedDescriptors<TileConstantsBase<cfg>>::Weight;

        static constexpr int WEIGHT_LDS_SIZE_UINT2 =
            cfg.kh * cfg.kw * cfg.block_groups() * GROUP_SIZE * GROUP_SIZE_4;
        static constexpr int WEIGHT_LDS_SIZE_UINT4   = WEIGHT_LDS_SIZE_UINT2 / 2;
        static constexpr int NUM_WEIGHT_PASSES =
            (WEIGHT_LDS_SIZE_UINT4 + cfg.block_size() - 1) / cfg.block_size();
        static constexpr int WEIGHT_LDS_PADDED_UINT4 = NUM_WEIGHT_PASSES * cfg.block_size();
        static constexpr int WEIGHT_LDS_READ_K = cfg.block_c();

        static constexpr auto MakeDramReadDescriptor()       { return Shared::MakeDramReadDescriptor(); }
        static constexpr auto MakeDramReadTileDistribution() { return Shared::MakeDramReadTileDistribution(); }
        static constexpr auto MakeLdsWriteDescriptor()       { return Shared::MakeLdsWriteDescriptor(); }
        static constexpr auto MakeLdsReadDescriptor()        { return Shared::MakeLdsReadDescriptor(); }
        
        template <int VectorSize>
        static CK_TILE_DEVICE auto MakeDramReadDescriptorPadded(int k_per_group, int c_per_group)
        {
            return Shared::template MakeDramReadDescriptorPadded<VectorSize>(k_per_group, c_per_group);
        }
    };

    // -----------------------------------------------------------------------
    // Input — fully delegated to SharedDescriptors.
    // -----------------------------------------------------------------------
    struct Input
    {
        using Shared = typename SharedDescriptors<TileConstantsBase<cfg>>::Input;

        static CK_TILE_DEVICE auto MakeDramReadDescriptor(int hi, int wi, int C_total, int px)
        {
            return Shared::MakeDramReadDescriptor(hi, wi, C_total, px);
        }
        static constexpr auto MakeDramReadTileDistribution() { return Shared::MakeDramReadTileDistribution(); }
        static constexpr auto MakeLdsWriteDescriptor()       { return Shared::MakeLdsWriteDescriptor(); }
        static constexpr auto MakeLdsReadDescriptor()        { return Shared::MakeLdsReadDescriptor(); }
    };

    // -----------------------------------------------------------------------
    // Output — shared descriptors; variant-specific distributions added below.
    // -----------------------------------------------------------------------
    struct Output
    {
        using Shared = typename SharedDescriptors<TileConstantsBase<cfg>>::Output;

        static constexpr int OUTPUT_LDS_BUFFER_SIZE = BLOCK_C8 * BLOCK_Q;

        static constexpr auto MakeLdsWriteDescriptor() { return Shared::MakeLdsWriteDescriptor(); }

        static CK_TILE_DEVICE auto MakeDramWriteDescriptorNarrow(int ho, int wo, int C)
        {
            return Shared::MakeDramWriteDescriptorNarrow(ho, wo, C);
        }

        static constexpr int STORE_Q = TOTAL_SPATIAL;

        static constexpr auto MakeLdsReadDescriptorWide()
        {
            return Shared::template MakeLdsReadDescriptorWide<STORE_Q>();
        }

        static CK_TILE_DEVICE auto MakeDramWriteDescriptorWide(int wo, int C)
        {
            return Shared::MakeDramWriteDescriptorWide(wo, C);
        }

        // Wide store distribution (identical for all variants).
        static constexpr auto MakeDramWriteTileDistributionWide()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<NUM_WAVES, LANES_PER_ROW>,
                                   ck_tile::sequence<BLOCK_C8>,
                                   ck_tile::sequence<8>>,
                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<3>,
                    ck_tile::sequence<0>,
                    ck_tile::number<STORE_VECS>>{});
        }
    };
};

// ======================================================================
// BlockCoords — workgroup-level coordinates (shared by all variants).
// ======================================================================
template <auto cfg>
struct BlockCoords
{
    int block_n;
    int block_q;
    int block_group;
    int block_k;
    int block_c8;
    int C;
    int C8;
    int K;

    __device__ BlockCoords(int groups)
        : C(groups * cfg.group_size()), C8(C / 8), K(C)
    {
        const int block_q_n_idx = blockIdx.x;
        block_n     = static_cast<int>(blockIdx.z) * cfg.n_fold + block_q_n_idx % cfg.n_fold;
        block_q     = (block_q_n_idx / cfg.n_fold) * cfg.block_q();
        block_group = static_cast<int>(blockIdx.y) * cfg.block_groups();
        block_k     = block_group * cfg.group_size();
        block_c8    = block_k / 8;
    }
};

// ======================================================================
// weight_read_fprop<TC> — shared Fprop weight-register read from LDS.
//
// Loads the weight tile via load_tile() and stores each filter position's
// fp16x4_t into the WeightAccessor using get_as<fp16x4_t>() which
// provides a zero-copy reinterpret view over the thread buffer.
// ======================================================================
template <typename TC, int KH, int KW>
__device__ void weight_read_fprop(WeightAccessor<KH, KW>& wa, uint4* weight_lds)
{
    constexpr auto weight_lds_read_desc = TC::Weight::MakeLdsReadDescriptor();
    auto weight_lds_view =
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(weight_lds), weight_lds_read_desc);

    constexpr auto weight_lds_read_dist = TC::Weight::MakeLdsReadTileDistribution();
    auto weight_lds_read_window = ck_tile::make_tile_window(
        weight_lds_view,
        ck_tile::make_tuple(ck_tile::number<TC::Weight::WEIGHT_LDS_READ_K>{},
                            ck_tile::number<TC::KH_KW>{},
                            ck_tile::number<TC::GROUP_SIZE>{}),
        {0, 0, 0},
        weight_lds_read_dist);

    const auto weight_tile = ck_tile::load_tile(weight_lds_read_window);

    // get_as<fp16x4_t>() reinterprets the thread_buffer<_Float16, KH_KW*4>
    // as thread_buffer<fp16x4_t, KH_KW> — a zero-copy view where each
    // element is one fp16x4_t per filter position.
    const auto& vec_buf = weight_tile.get_thread_buffer().template get_as<ck_tile::fp16x4_t>();
    static_for<TC::KH_KW>(
        [&]<int khw>()
        {
            wa.weights[khw] = vec_buf[ck_tile::number<khw>{}];
        });
}

// ======================================================================
// is_applicable_base — data-type, layout, and geometry checks shared
// by all grouped-conv fp16 variants.  Each variant additionally checks
// channels_per_group() and c_tot alignment.
// ======================================================================
inline bool is_applicable_base(const Conv2dParams& par)
{
    if(par.in_type  != DataType::fp16)   return false;
    if(par.wei_type != DataType::fp16)   return false;
    if(par.out_type != DataType::fp16)   return false;
    if(par.order    != TensorOrder::NHWC) return false;
    if(par.direction != Direction::Fprop &&
       par.direction != Direction::Dgrad) return false;
    if(par.kh != 3 || par.kw != 3)       return false;
    if(par.k_tot != par.c_tot)           return false;
    if(par.stride_h != 1 || par.stride_w != 1)       return false;
    if(par.dilation_h != 1 || par.dilation_w != 1)   return false;
    if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1) return false;
    return true;
}

// ======================================================================
// get_launch_params_impl — shared launch-parameter computation.
// ======================================================================
template <typename Config>
LaunchParams get_launch_params_impl(const Config& cfg, const Conv2dParams& par)
{
    const int out_q    = (cfg.direction == Direction::Dgrad) ? par.w : par.q;
    auto blocks_w      = ck_tile::integer_divide_ceil(out_q, cfg.block_q());
    auto blocks_w_n    = blocks_w * cfg.n_fold;
    auto blocks_c      = ck_tile::integer_divide_ceil(par.c_tot, cfg.block_c());
    auto blocks_n_fold = ck_tile::integer_divide_ceil(par.n, cfg.n_fold);

    LaunchParams launch;
    launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold);
    launch.block_size = dim3(cfg.block_size(), 1, 1);
    return launch;
}

// ======================================================================
// xor_config_valid — shared XOR swizzle alignment check.
// ======================================================================
template <typename Config>
bool xor_config_valid(const Config& cfg, const Conv2dParams& par)
{
    const int block_c8 = cfg.block_c() / 8;
    const int out_q    = (par.direction == Direction::Dgrad) ? par.w : par.q;
    return (cfg.block_q() % block_c8 == 0) || (out_q <= cfg.block_q());
}

} // namespace ck_tile::direct_conv
