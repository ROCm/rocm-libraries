// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"

namespace ck_tile {
namespace direct_conv {

// Shared descriptor and distribution factories for grouped convolution kernels.
//
// Template parameter TC is a TileConstants type providing these static constexpr members:
//   BLOCK_C8, BLOCK_C4, BLOCK_W, BLOCK_Q, NUM_WAVES, LANES_PER_ROW, TOTAL_SPATIAL,
//   GROUP_SIZE, GROUP_SIZE_4, KH_KW, KW, SWIZZLE_TYPE,
//   Weight::WEIGHT_LDS_SIZE_UINT4, Weight::WEIGHT_LDS_PADDED_UINT4,
//   Weight::NUM_WEIGHT_PASSES, Weight::WEIGHT_LDS_READ_K
template <typename TC>
struct SharedDescriptors
{
    // ===================================================================
    // Input — descriptors and distributions for input activation tensor.
    // ===================================================================
    struct Input
    {
        // DRAM descriptor: [hi, wi_padded, BLOCK_C8, 8] with optional XOR.
        static CK_TILE_DEVICE auto MakeDramDescriptor(int hi, int wi, int C_total, int px)
        {
            constexpr int right_pad_w = TC::KW - 1;

            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(hi, wi,
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(wi * C_total, C_total,
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});

            const auto desc_padded = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(hi),
                    ck_tile::make_pad_transform(wi, px, right_pad_w),
                    ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_C8>{}),
                    ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}));

            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::XOR)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_padded,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(hi),
                        ck_tile::make_xor_transform(ck_tile::make_tuple(
                            wi + px + right_pad_w, ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}));
            }
            else if constexpr (TC::SWIZZLE_TYPE == SwizzleType::CyclicShift)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_padded,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(hi),
                        ck_tile::make_cyclic_shift_transform(ck_tile::make_tuple(
                            wi + px + right_pad_w, ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}));
            }
            else
            {
                return desc_padded;
            }
        }

        // Tile distribution for DRAM async loads: [1, {NUM_WAVES, LANES_PER_ROW}, BLOCK_C8, 8].
        static constexpr auto MakeDramDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<1>,
                                   ck_tile::sequence<TC::NUM_WAVES, TC::LANES_PER_ROW>,
                                   ck_tile::sequence<TC::BLOCK_C8>,
                                   ck_tile::sequence<8>>,
                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 3>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<1, 4>,
                    ck_tile::sequence<0, 0>>{});
        }

        // LDS store descriptor: [1, TOTAL_SPATIAL, BLOCK_C8, 8] contiguous.
        static constexpr auto MakeLdsStoreDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<TC::TOTAL_SPATIAL>{},
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<TC::TOTAL_SPATIAL * TC::BLOCK_C8 * 8>{},
                                    ck_tile::number<TC::BLOCK_C8 * 8>{},
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        }

        // LDS read descriptor: [BLOCK_W, BLOCK_C4, 4] with optional XOR.
        static constexpr auto MakeLdsReadDescriptor()
        {
            constexpr auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_W>{},
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C8 * 8>{},
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});

            auto make_desc = [](auto desc_3d) constexpr {
                constexpr auto desc_merged = ck_tile::transform_tensor_descriptor(
                    desc_3d,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_W>{}),
                        ck_tile::make_merge_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C8>{},
                                                ck_tile::number<8>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

                return ck_tile::transform_tensor_descriptor(
                    desc_merged,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_W>{}),
                        ck_tile::make_unmerge_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C4>{},
                                                ck_tile::number<4>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}));
            };

            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::XOR)
            {
                constexpr auto desc_xor = ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_xor_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_W>{},
                                                ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
                return make_desc(desc_xor);
            }
            else if constexpr (TC::SWIZZLE_TYPE == SwizzleType::CyclicShift)
            {
                constexpr auto desc_cyclic_shift = ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_inverse_cyclic_shift_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_W>{},
                                                ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
                return make_desc(desc_cyclic_shift);
            }
            else
            {
                return make_desc(desc_raw);
            }
        }
    };

    // ===================================================================
    // Weight — descriptors and distributions for filter weight tensor.
    // ===================================================================
    struct Weight
    {
        // DRAM descriptor: [WEIGHT_LDS_SIZE_UINT4, 8] padded to [WEIGHT_LDS_PADDED_UINT4, 8].
        static constexpr auto MakeDramDescriptor()
        {
            constexpr auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<TC::Weight::WEIGHT_LDS_SIZE_UINT4>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<8>{}, ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});

            constexpr int right_pad =
                TC::Weight::WEIGHT_LDS_PADDED_UINT4 - TC::Weight::WEIGHT_LDS_SIZE_UINT4;
            constexpr auto desc_padded = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pad_transform(
                        ck_tile::number<TC::Weight::WEIGHT_LDS_SIZE_UINT4>{}, 0, right_pad),
                    ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
            return desc_padded;
        }

        // Tile distribution for weight async loads: linear tid → row.
        static constexpr auto MakeDramDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<TC::NUM_WAVES, 64>,
                                   ck_tile::sequence<8>>,
                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1>>,
                    ck_tile::sequence<2>,
                    ck_tile::sequence<0>>{});
        }

        // LDS store descriptor: [WEIGHT_LDS_PADDED_UINT4, 8] contiguous.
        static constexpr auto MakeLdsStoreDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<TC::Weight::WEIGHT_LDS_PADDED_UINT4>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<8>{}, ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        }

        // LDS read descriptor (Fprop): [block_c, kh*kw, GROUP_SIZE] row-major.
        static constexpr auto MakeLdsReadDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<TC::Weight::WEIGHT_LDS_READ_K>{},
                                    ck_tile::number<TC::KH_KW>{},
                                    ck_tile::number<TC::GROUP_SIZE>{}),
                ck_tile::make_tuple(ck_tile::number<TC::KH_KW * TC::GROUP_SIZE>{},
                                    ck_tile::number<TC::GROUP_SIZE>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});
        }
    };

    // ===================================================================
    // Output — descriptors for output activation tensor.
    // ===================================================================
    struct Output
    {
        // LDS write descriptor: [BLOCK_Q, BLOCK_C4, 4] with optional swizzle on (Q, C8).
        static constexpr auto MakeLdsWriteDescriptor()
        {
            constexpr auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C8 * 8>{},
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});

            auto make_desc = [](auto desc_3d) constexpr {
                constexpr auto desc_merged = ck_tile::transform_tensor_descriptor(
                    desc_3d,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_Q>{}),
                        ck_tile::make_merge_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C8>{},
                                                ck_tile::number<8>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

                return ck_tile::transform_tensor_descriptor(
                    desc_merged,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_Q>{}),
                        ck_tile::make_unmerge_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C4>{},
                                                ck_tile::number<4>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}));
            };

            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::XOR)
            {
                constexpr auto desc_xor = ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_xor_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                                ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
                return make_desc(desc_xor);
            }
            else if constexpr (TC::SWIZZLE_TYPE == SwizzleType::CyclicShift)
            {
                constexpr auto desc_cs = ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_cyclic_shift_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                                ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
                return make_desc(desc_cs);
            }
            else
            {
                return make_desc(desc_raw);
            }
        }

        // LDS read descriptor: [1, BLOCK_Q, BLOCK_C4, 4] with optional swizzle on (Q, C8).
        // Both write and read use the SAME swizzle (cyclic_shift / xor) because they
        // both map logical (q, c8) → the same physical LDS position. The input LDS path
        // uses inverse_cyclic_shift for reads because its LDS is written contiguously
        // (no swizzle in store descriptor), but here both descriptors apply the swizzle.
        static constexpr auto MakeLdsReadDescriptor()
        {
            constexpr auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C8 * 8>{},
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});

            auto make_desc = [](auto desc_3d) constexpr {
                constexpr auto desc_merged = ck_tile::transform_tensor_descriptor(
                    desc_3d,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_Q>{}),
                        ck_tile::make_merge_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C8>{},
                                                ck_tile::number<8>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

                constexpr auto desc_3d_out = ck_tile::transform_tensor_descriptor(
                    desc_merged,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_Q>{}),
                        ck_tile::make_unmerge_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_C4>{},
                                                ck_tile::number<4>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}));

                // Wrap with row dimension: [Q, C4, 4] → [1, Q, C4, 4]
                return ck_tile::transform_tensor_descriptor(
                    desc_3d_out,
                    ck_tile::make_tuple(
                        ck_tile::make_unmerge_transform(
                            ck_tile::make_tuple(ck_tile::number<1>{},
                                                ck_tile::number<TC::BLOCK_Q>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_C4>{}),
                        ck_tile::make_pass_through_transform(ck_tile::number<4>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                        ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{},
                                        ck_tile::sequence<3>{}));
            };

            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::XOR)
            {
                constexpr auto desc_xor = ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_xor_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                                ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
                return make_desc(desc_xor);
            }
            else if constexpr (TC::SWIZZLE_TYPE == SwizzleType::CyclicShift)
            {
                constexpr auto desc_cs = ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_cyclic_shift_transform(
                            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                                ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
                return make_desc(desc_cs);
            }
            else
            {
                return make_desc(desc_raw);
            }
        }

        // DRAM descriptor: [ho, wo_padded, BLOCK_C4, 4].
        static CK_TILE_DEVICE auto MakeDramDescriptor(int ho, int wo, int C)
        {
            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ho, wo,
                                    ck_tile::number<TC::BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                ck_tile::make_tuple(wo * C, C,
                                    ck_tile::number<4>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});

            constexpr int right_pad_w = TC::BLOCK_Q;
            const auto desc_padded = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(ho),
                    ck_tile::make_pad_transform(wo, 0, right_pad_w),
                    ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_C4>{}),
                    ck_tile::make_pass_through_transform(ck_tile::number<4>{})),
                ck_tile::make_tuple(
                    ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}),
                ck_tile::make_tuple(
                    ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}));

            return desc_padded;
        }
    };
};

} // namespace direct_conv
} // namespace ck_tile
