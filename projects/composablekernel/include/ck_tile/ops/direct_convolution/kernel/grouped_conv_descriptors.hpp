// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
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
        // DRAM read descriptor: [hi_padded, wi_padded, BLOCK_C8, 8] with optional XOR.
        static CK_TILE_DEVICE auto MakeDramReadDescriptor(int hi, int wi, int C_total, int px, int py, int dx, int dy, int sx, int sy)
        {
            const int hi_padded_size = hi + 2 * py;
            const int wi_padded_size = wi + 2 * px;

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
                    ck_tile::make_pad_transform(hi, py, py),
                    ck_tile::make_pad_transform(wi, px, px),
                    ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_C8>{}),
                    ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}));


            if (sx != 1 || sy != 1 || dx != 1 || dy != 1)
            {
                // TODO: implement the striding and dilation.
            }

            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::XOR)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_padded,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(hi_padded_size),
                        ck_tile::make_xor_transform(ck_tile::make_tuple(
                            wi_padded_size, ck_tile::number<TC::BLOCK_C8>{})),
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
                        ck_tile::make_pass_through_transform(hi_padded_size),
                        ck_tile::make_cyclic_shift_transform(ck_tile::make_tuple(
                            wi_padded_size, ck_tile::number<TC::BLOCK_C8>{})),
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

        // Padded DRAM descriptor for input loading when c_per_group < GROUP_SIZE.
        //
        // Builds a 4D [hi_padded, wi_padded, BLOCK_C8, 8] descriptor (same shape as
        // MakeDramReadDescriptor) so the caller can reuse the tile distribution
        // and LDS write descriptor unchanged.
        //
        // Transform chain:
        //   1. Raw DRAM: [hi, wi, BLOCK_GROUPS, c_per_group] with real strides.
        //   2. Pad spatial (hi, wi): [hi + 2 * py, wi + 2 * px, BLOCK_GROUPS, c_per_group].
        //   3. Pad channel (c_per_group → GROUP_SIZE):
        //      [hi_padded, wi_padded, BLOCK_GROUPS, GROUP_SIZE] (OOB reads as zero).
        //   4. Merge channel dims: [hi_padded, wi_padded, BLOCK_C].
        //   5. Unmerge to C8 layout: [hi_padded, wi_padded, BLOCK_C8, 8].
        //   6. (Optional) XOR/CyclicShift swizzle.
        //
        // The buffer base pointer must be set to:
        //   in + block_n * hi * wi * C_in + block_k_in
        template <int GuaranteedVectorLoadSize = 1>
        static CK_TILE_DEVICE auto MakeDramReadDescriptorPadded(
            int hi, int wi, int C_in, int c_per_group, int px, int py, int dx, int dy, int sx, int sy)
        {
            constexpr int GROUP_SIZE   = TC::GROUP_SIZE;
            constexpr int BLOCK_GROUPS = TC::BLOCK_GROUPS;
            constexpr int BLOCK_C8     = TC::BLOCK_C8;
            const int hi_padded_size = hi + 2 * py;
            const int wi_padded_size = wi + 2 * px;

            // Step 1: raw 4D descriptor with real DRAM strides.
            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(hi, wi,
                                    ck_tile::number<BLOCK_GROUPS>{}, c_per_group),
                ck_tile::make_tuple(wi * C_in, C_in, c_per_group, 1),
                ck_tile::number<GuaranteedVectorLoadSize>{},
                ck_tile::number<1>{});

            // Step 2+3: pad spatial + pad channel → GROUP_SIZE.
            const auto desc_padded_4d = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pad_transform(hi, py, py),
                    ck_tile::make_pad_transform(wi, px, px),
                    ck_tile::make_pass_through_transform(ck_tile::number<BLOCK_GROUPS>{}),
                    ck_tile::make_pad_transform(c_per_group, ck_tile::number<0>{},
                                               ck_tile::number<GROUP_SIZE>{} - c_per_group)),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}));

            if (sx != 1 || sy != 1 || dx != 1 || dy != 1)
            {
                // TODO: implement the striding and dilation.
            }

            // Step 4: merge channel dims → [hi_padded, wi_padded, BLOCK_C].
            const auto desc_merged = ck_tile::transform_tensor_descriptor(
                desc_padded_4d,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(hi_padded_size),
                    ck_tile::make_pass_through_transform(wi_padded_size),
                    ck_tile::make_merge_transform(
                        ck_tile::make_tuple(ck_tile::number<BLOCK_GROUPS>{},
                                            ck_tile::number<GROUP_SIZE>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2, 3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}));

            // Step 5: unmerge → [hi_padded, wi_padded, BLOCK_C8, 8].
            const auto desc_4d = ck_tile::transform_tensor_descriptor(
                desc_merged,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(hi_padded_size),
                    ck_tile::make_pass_through_transform(wi_padded_size),
                    ck_tile::make_unmerge_transform(
                        ck_tile::make_tuple(ck_tile::number<BLOCK_C8>{},
                                            ck_tile::number<8>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2, 3>{}));

            // Step 6: optional swizzle on [wi_padded, BLOCK_C8].
            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::XOR)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_4d,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(hi_padded_size),
                        ck_tile::make_xor_transform(ck_tile::make_tuple(
                            wi_padded_size, ck_tile::number<BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}));
            }
            else if constexpr(TC::SWIZZLE_TYPE == SwizzleType::CyclicShift)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_4d,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(hi_padded_size),
                        ck_tile::make_cyclic_shift_transform(ck_tile::make_tuple(
                            wi_padded_size, ck_tile::number<BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}));
            }
            else
            {
                return desc_4d;
            }
        }

        // Tile distribution for DRAM async loads: [1, {NUM_WAVES, LANES_PER_ROW}, BLOCK_C8, 8].
        static constexpr auto MakeDramReadTileDistribution()
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

        // LDS write descriptor: [1, BLOCK_W, BLOCK_C8, 8] contiguous.
        static constexpr auto MakeLdsWriteDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<TC::BLOCK_W>{},
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_W * TC::BLOCK_C8 * 8>{},
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
        // DRAM read descriptor: [WEIGHT_LDS_SIZE_UINT4, 8].
        static constexpr auto MakeDramReadDescriptor()
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

        // Padded DRAM descriptor for weight loading when c_per_group < GROUP_SIZE
        // or k_per_group < GROUP_SIZE.
        //
        // Builds a flat 2D [WEIGHT_LDS_PADDED_UINT4, 8] descriptor (same shape as
        // MakeDramReadDescriptor) so the caller can reuse MakeDramReadTileDistribution
        // and MakeLdsWriteDescriptor unchanged.
        //
        // The chain of transforms is:
        //   1. Raw DRAM: [BLOCK_GROUPS, k_per_group, KH_KW, c_per_group] with real strides.
        //   2. Pad K: [BLOCK_GROUPS, GROUP_SIZE, KH_KW, c_per_group]
        //      (k ∈ [k_per_group, GROUP_SIZE) → OOB, reads as zero).
        //   3. Pad C: [BLOCK_GROUPS, GROUP_SIZE, KH_KW, GROUP_SIZE]
        //      (c ∈ [c_per_group, GROUP_SIZE) → OOB, reads as zero).
        //   4. Merge all 4 dims → [BLOCK_GROUPS * GROUP_SIZE * KH_KW * GROUP_SIZE].
        //   5. Unmerge → [WEIGHT_LDS_SIZE_UINT4, 8].
        //   6. Pad rows → [WEIGHT_LDS_PADDED_UINT4, 8].
        //
        // The buffer base pointer must be set to the start of bc.block_group in the
        // global weight tensor (see weight_load_to_lds).
        template <int GuaranteedVectorLoadSize = 1>
        static CK_TILE_DEVICE auto MakeDramReadDescriptorPadded(int k_per_group, int c_per_group)
        {
            constexpr int filter_size   = TC::KH_KW;
            constexpr int GROUP_SIZE    = TC::GROUP_SIZE;
            constexpr int BLOCK_GROUPS  = TC::BLOCK_GROUPS;
            constexpr int LDS_SIZE_UINT4  = TC::Weight::WEIGHT_LDS_SIZE_UINT4;
            constexpr int LDS_PADDED_UINT4 = TC::Weight::WEIGHT_LDS_PADDED_UINT4;
            constexpr int right_pad_rows = LDS_PADDED_UINT4 - LDS_SIZE_UINT4;

            // Step 1: raw 4D descriptor with real DRAM strides (fp16 elements).
            const int CStride  = 1;
            const int XYStride = c_per_group;
            const int KStride  = filter_size * XYStride;
            const int GStride  = k_per_group * KStride;

            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<BLOCK_GROUPS>{}, k_per_group,
                                    ck_tile::number<filter_size>{}, c_per_group),
                ck_tile::make_tuple(GStride, KStride, XYStride, CStride),
                ck_tile::number<GuaranteedVectorLoadSize>{}, 
                ck_tile::number<1>{});

            // Step 2+3: pad K → GROUP_SIZE, pad C → GROUP_SIZE.
            // Use number<0>{} for left pads and compute right pads as
            // number<GROUP_SIZE> - runtime length, expressed via make_pad_transform
            // with the target upper length number<GROUP_SIZE>{}.
            // This makes the padded extents compile-time (required for merge).
            const auto desc_padded_4d = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(ck_tile::number<BLOCK_GROUPS>{}),
                    ck_tile::make_pad_transform(k_per_group, number<0>{},
                                               number<GROUP_SIZE>{} - k_per_group),
                    ck_tile::make_pass_through_transform(ck_tile::number<filter_size>{}),
                    ck_tile::make_pad_transform(c_per_group, number<0>{},
                                               number<GROUP_SIZE>{} - c_per_group)),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}));

            // Step 4: merge all 4 dims → [BLOCK_GROUPS * GROUP_SIZE * KH_KW * GROUP_SIZE].
            const auto desc_merged = ck_tile::transform_tensor_descriptor(
                desc_padded_4d,
                ck_tile::make_tuple(ck_tile::make_merge_transform(
                    ck_tile::make_tuple(ck_tile::number<BLOCK_GROUPS>{},
                                        ck_tile::number<GROUP_SIZE>{},
                                        ck_tile::number<filter_size>{},
                                        ck_tile::number<GROUP_SIZE>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0, 1, 2, 3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}));

            // Step 5: unmerge → [WEIGHT_LDS_SIZE_UINT4, 8].
            const auto desc_2d = ck_tile::transform_tensor_descriptor(
                desc_merged,
                ck_tile::make_tuple(ck_tile::make_unmerge_transform(
                    ck_tile::make_tuple(ck_tile::number<LDS_SIZE_UINT4>{},
                                        ck_tile::number<8>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}));

            // Step 6: pad rows → [WEIGHT_LDS_PADDED_UINT4, 8].
            const auto desc_final = ck_tile::transform_tensor_descriptor(
                desc_2d,
                ck_tile::make_tuple(
                    ck_tile::make_pad_transform(ck_tile::number<LDS_SIZE_UINT4>{}, 0, right_pad_rows),
                    ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

            return desc_final;
        }

        // Tile distribution for weight async loads: linear tid → row.
        static constexpr auto MakeDramReadTileDistribution()
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

        // LDS write descriptor: [WEIGHT_LDS_PADDED_UINT4, 8] contiguous.
        static constexpr auto MakeLdsWriteDescriptor()
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

        // LDS read descriptor (wide): [StoreQ, BLOCK_C8, 8] with optional swizzle.
        // StoreQ covers the full thread distribution range (typically 2 * BLOCK_Q).
        // Only threads with Q < BLOCK_Q read valid data; threads with Q >= BLOCK_Q
        // are inactive (guarded by is_thread_active() from the distribution).
        template <int StoreQ>
        static constexpr auto MakeLdsReadDescriptorWide()
        {
            constexpr auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<StoreQ>{},
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
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
                        ck_tile::make_xor_transform(
                            ck_tile::make_tuple(ck_tile::number<StoreQ>{},
                                                ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
            }
            else if constexpr(TC::SWIZZLE_TYPE == SwizzleType::CyclicShift)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_cyclic_shift_transform(
                            ck_tile::make_tuple(ck_tile::number<StoreQ>{},
                                                ck_tile::number<TC::BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
            }
            else
            {
                return desc_raw;
            }
        }

        // DRAM write descriptor (wide): [wo_padded, BLOCK_C8, 8].
        // Uses C8-aligned addressing (8 fp16 per uint4) for 16B stores.
        // The row dimension (ho) is handled externally via p_out * row_stride_elems.
        static CK_TILE_DEVICE auto MakeDramWriteDescriptorWide(int wo, int C)
        {
            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(wo,
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(C,
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});

            constexpr int right_pad_w = TC::BLOCK_Q;
            return ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pad_transform(wo, 0, right_pad_w),
                    ck_tile::make_pass_through_transform(ck_tile::number<TC::BLOCK_C8>{}),
                    ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                ck_tile::make_tuple(
                    ck_tile::sequence<0>{},
                    ck_tile::sequence<1>{}, ck_tile::sequence<2>{}),
                ck_tile::make_tuple(
                    ck_tile::sequence<0>{},
                    ck_tile::sequence<1>{}, ck_tile::sequence<2>{}));
        }

        // DRAM write descriptor (narrow): [ho, wo_padded, BLOCK_C4, 4].
        static CK_TILE_DEVICE auto MakeDramWriteDescriptorNarrow(int ho, int wo, int C)
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

        // Padded DRAM write descriptor (narrow): [ho, wo_padded, BLOCK_C4, 4].
        //
        // For k_per_group < GROUP_SIZE, builds a descriptor where the channel
        // dimension is padded so that positions >= k_per_group within each group
        // are marked OOB. store_tile will skip writes to those positions.
        //
        // Transform chain:
        //   1. Raw: [ho, wo, BLOCK_GROUPS, k_per_group] with real strides.
        //   2. Pad spatial: [ho, wo_padded, BLOCK_GROUPS, k_per_group].
        //   3. Pad channel (k_per_group → GROUP_SIZE): OOB for invalid k.
        //   4. Merge channel: [ho, wo_padded, BLOCK_C].
        //   5. Unmerge to C4: [ho, wo_padded, BLOCK_C4, 4].
        template <int VectorSize = 1>
        static CK_TILE_DEVICE auto MakeDramWriteDescriptorNarrowPadded(
            int ho, int wo, int K_total, int k_per_group)
        {
            constexpr int right_pad_w  = TC::BLOCK_Q;
            constexpr int GROUP_SIZE   = TC::GROUP_SIZE;
            constexpr int BLOCK_GROUPS = TC::BLOCK_GROUPS;
            constexpr int BLOCK_C4     = TC::BLOCK_C4;

            // Step 1: raw 4D with real DRAM strides.
            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ho, wo,
                                    ck_tile::number<BLOCK_GROUPS>{}, k_per_group),
                ck_tile::make_tuple(wo * K_total, K_total, k_per_group, 1),
                ck_tile::number<VectorSize>{},
                ck_tile::number<1>{});

            // Step 2+3: pad spatial + pad channel → GROUP_SIZE.
            const auto desc_padded_4d = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(ho),
                    ck_tile::make_pad_transform(wo, 0, right_pad_w),
                    ck_tile::make_pass_through_transform(ck_tile::number<BLOCK_GROUPS>{}),
                    ck_tile::make_pad_transform(k_per_group, ck_tile::number<0>{},
                                               ck_tile::number<GROUP_SIZE>{} - k_per_group)),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}));

            // Step 4: merge channel → [ho, wo_padded, BLOCK_C].
            const auto desc_merged = ck_tile::transform_tensor_descriptor(
                desc_padded_4d,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(ho),
                    ck_tile::make_pass_through_transform(wo + right_pad_w),
                    ck_tile::make_merge_transform(
                        ck_tile::make_tuple(ck_tile::number<BLOCK_GROUPS>{},
                                            ck_tile::number<GROUP_SIZE>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2, 3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}));

            // Step 5: unmerge to C4 → [ho, wo_padded, BLOCK_C4, 4].
            return ck_tile::transform_tensor_descriptor(
                desc_merged,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(ho),
                    ck_tile::make_pass_through_transform(wo + right_pad_w),
                    ck_tile::make_unmerge_transform(
                        ck_tile::make_tuple(ck_tile::number<BLOCK_C4>{},
                                            ck_tile::number<4>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2, 3>{}));
        }

        // Padded DRAM write descriptor (wide): [wo_padded, BLOCK_C8, 8].
        //
        // Same as MakeDramWriteDescriptorWide but with channel padding for
        // k_per_group < GROUP_SIZE.
        template <int VectorSize = 1>
        static CK_TILE_DEVICE auto MakeDramWriteDescriptorWidePadded(
            int wo, int K_total, int k_per_group)
        {
            constexpr int right_pad_w  = TC::BLOCK_Q;
            constexpr int GROUP_SIZE   = TC::GROUP_SIZE;
            constexpr int BLOCK_GROUPS = TC::BLOCK_GROUPS;
            constexpr int BLOCK_C8     = TC::BLOCK_C8;

            // Step 1: raw 3D (no ho dim for wide path).
            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(wo,
                                    ck_tile::number<BLOCK_GROUPS>{}, k_per_group),
                ck_tile::make_tuple(K_total, k_per_group, 1),
                ck_tile::number<VectorSize>{},
                ck_tile::number<1>{});

            // Step 2+3: pad spatial + pad channel.
            const auto desc_padded = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pad_transform(wo, 0, right_pad_w),
                    ck_tile::make_pass_through_transform(ck_tile::number<BLOCK_GROUPS>{}),
                    ck_tile::make_pad_transform(k_per_group, ck_tile::number<0>{},
                                               ck_tile::number<GROUP_SIZE>{} - k_per_group)),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}));

            // Step 4: merge channel → [wo_padded, BLOCK_C].
            const auto desc_merged = ck_tile::transform_tensor_descriptor(
                desc_padded,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(wo + right_pad_w),
                    ck_tile::make_merge_transform(
                        ck_tile::make_tuple(ck_tile::number<BLOCK_GROUPS>{},
                                            ck_tile::number<GROUP_SIZE>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

            // Step 5: unmerge to C8 → [wo_padded, BLOCK_C8, 8].
            return ck_tile::transform_tensor_descriptor(
                desc_merged,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(wo + right_pad_w),
                    ck_tile::make_unmerge_transform(
                        ck_tile::make_tuple(ck_tile::number<BLOCK_C8>{},
                                            ck_tile::number<8>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}));
        }
    };
};

} // namespace direct_conv
} // namespace ck_tile
