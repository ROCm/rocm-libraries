// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Descriptor factories specific to the dense (non-grouped) 32c v3 convolution.
//
// The dense kernel's layout differs from the grouped one in three ways that
// prevent reuse of grouped_conv_descriptors.hpp directly (wave->C-group mapping,
// c_slices_per_wave chunking, and the global-wi swizzle coordinate). This header
// provides the small set of descriptor transforms the dense input path needs to
// express its swizzle as CK Tile tensor-descriptor transformations rather than
// hand-rolled integer math.
//
// Two swizzle helpers are provided, both keyed on TC::SWIZZLE_TYPE:
//   - MakeLdsReadDescriptor(spatial_len): [spatial, BLOCK_C8, 8] element-strided
//     LDS view with the INVERSE swizzle on (spatial, BLOCK_C8). The dense kernel
//     applies the swizzle on the GLOBAL wi coordinate (block_q + spatial_pos), so
//     callers evaluate at the global spatial index and subtract the block_q base
//     offset to recover the tile-local LDS element offset.
//   - MakeForwardSwizzleDescriptor(spatial_len): [spatial, BLOCK_C8] view with
//     spatial stride 0 and BLOCK_C8 stride 1, with the FORWARD swizzle. Evaluating
//     at (global_spatial, c8) yields the swizzled DRAM channel-8 index directly
//     (the overflow DRAM read path).

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"

namespace ck_tile {
namespace direct_conv {

// Dense-specific descriptor factories. TC is the dense TileConstants type and
// must provide BLOCK_W, BLOCK_C8 and SWIZZLE_TYPE.
template <typename TC>
struct DenseSharedDescriptors
{
    struct Input
    {
        // LDS read view: [spatial, BLOCK_C8, 8] with element strides
        // (BLOCK_C8*8, 8, 1) and the INVERSE swizzle applied on (spatial, c8).
        //
        // Evaluating make_tensor_coordinate at upper index (global_spatial,
        // c8_logical, 0) gives:
        //   global_spatial * BLOCK_C8 * 8 + inverse_swizzle(global_spatial,
        //   c8_logical) * 8
        // The caller subtracts block_q * BLOCK_C8 * 8 to obtain the tile-local
        // LDS element offset, matching the hand-rolled swizzle_c8_inverse path.
        //
        // spatial_len is runtime (it must exceed the largest global spatial index
        // evaluated, i.e. block_q + BLOCK_W); it only bounds the descriptor and
        // does not affect the swizzle (whose modulus is the compile-time BLOCK_C8).
        static CK_TILE_DEVICE auto MakeLdsReadDescriptor(int spatial_len)
        {
            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(
                    spatial_len, ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C8 * 8>{},
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});

            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::XOR)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_xor_transform(ck_tile::make_tuple(
                            spatial_len, ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
            }
            else if constexpr(TC::SWIZZLE_TYPE == SwizzleType::CyclicShift)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_inverse_cyclic_shift_transform(ck_tile::make_tuple(
                            spatial_len, ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
            }
            else
            {
                return desc_raw;
            }
        }

        // Forward-swizzle view: [spatial, BLOCK_C8] with strides (0, 1) and the
        // FORWARD swizzle on (spatial, c8). Evaluating make_tensor_coordinate at
        // upper index (global_spatial, c8) returns forward_swizzle(global_spatial,
        // c8) directly (spatial contributes nothing to the offset). Used to derive
        // the overflow DRAM channel-8 index.
        static CK_TILE_DEVICE auto MakeForwardSwizzleDescriptor(int spatial_len)
        {
            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(spatial_len, ck_tile::number<TC::BLOCK_C8>{}),
                ck_tile::make_tuple(ck_tile::number<0>{}, ck_tile::number<1>{}),
                ck_tile::number<1>{},
                ck_tile::number<1>{});

            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::XOR)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(ck_tile::make_xor_transform(
                        ck_tile::make_tuple(spatial_len, ck_tile::number<TC::BLOCK_C8>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}));
            }
            else if constexpr(TC::SWIZZLE_TYPE == SwizzleType::CyclicShift)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(ck_tile::make_cyclic_shift_transform(
                        ck_tile::make_tuple(spatial_len, ck_tile::number<TC::BLOCK_C8>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}));
            }
            else
            {
                return desc_raw;
            }
        }
    };

    // Output-channel padding expressed as a CK Tile pad transform.
    //
    // The dense output writer stores K-output channels contiguously. When
    // K_real is not a multiple of block_k_size (BLOCK_K), the tail block holds
    // out-of-range channels that must not be written. This factory returns a
    // 1-D descriptor over the K axis whose upper (padded) length is K_real
    // rounded up to a BLOCK_K multiple, with a right pad of the difference.
    //
    // Evaluating make_tensor_coordinate at upper index k and querying
    // coordinate_has_valid_offset_assuming_top_index_is_valid yields exactly
    // (k < K_real): the pad transform marks the right-padded tail invalid.
    // Callers evaluate the contiguous K-group indices and count the in-range
    // prefix (valid_k_count).
    struct Output
    {
        static CK_TILE_DEVICE auto MakeChannelPadDescriptor(int K_real)
        {
            constexpr int BLOCK_K = TC::BLOCK_K;
            const int K_padded    = ((K_real + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;

            const auto desc_real =
                ck_tile::make_naive_tensor_descriptor_packed(ck_tile::make_tuple(K_real));

            return ck_tile::transform_tensor_descriptor(
                desc_real,
                ck_tile::make_tuple(
                    ck_tile::make_pad_transform(K_real, ck_tile::number<0>{}, K_padded - K_real)),
                ck_tile::make_tuple(ck_tile::sequence<0>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}));
        }
    };
};

} // namespace direct_conv
} // namespace ck_tile
