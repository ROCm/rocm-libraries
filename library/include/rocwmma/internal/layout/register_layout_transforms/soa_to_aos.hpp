/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_IMPL_HPP
#define ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_IMPL_HPP

#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    namespace RegisterTransforms_impl
    {
        template <uint32_t BlockDim, uint32_t VectorWidth>
        struct soa_to_aos
        {
            constexpr static uint32_t VW      = VectorWidth;
            constexpr static uint32_t VecSize = VW * (BlockDim / Constants::AMDGCN_WAVE_SIZE);

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                return v;
            }
        };

        // soa_to_aos
        template <>
        struct soa_to_aos<16, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Step 1 : UnpackLoHi1
                auto unpacked_data = unpackLoHi1(v);

                // Step 2 : UnpackLoHi2
                unpacked_data = unpackLoHi2(unpacked_data);

                // Step 3 : UnpackLoHi4
                unpacked_data = unpackLoHi4(unpacked_data);

                // Step 4 : UnpackLoHi8
                unpacked_data = unpackLoHi8(unpacked_data);

                return unpacked_data;
            }
        };

        template <>
        struct soa_to_aos<32, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Scatter
                auto packed = Permute::Scatter32<16, 0>::exec(PackUtil::paddedPack(v));

                auto unpacked_data = PackUtil::template paddedUnpack<16>(packed);

                // Step 2 : UnpackLoHi2
                unpacked_data = unpackLoHi2(unpacked_data);

                // Step 3 : UnpackLoHi4
                unpacked_data = unpackLoHi4(unpacked_data);

                // Step 4 : UnpackLoHi8
                unpacked_data = unpackLoHi8(unpacked_data);

                // Step 4 : UnpackLoHi16
                unpacked_data = unpackLoHi16(unpacked_data);

                return unpacked_data;
            }
        };

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<64, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Scatter
                auto packed = Permute::ScatterWave<16, 0>::exec(PackUtil::paddedPack(v));

                auto unpacked_data = PackUtil::template paddedUnpack<16>(packed);

                // Step 2 : Unpack groups of 4
                unpacked_data = unpackLoHi4(unpacked_data);

                // Step 3 : Unpack groups of 8
                unpacked_data = unpackLoHi8(unpacked_data);

                // Step 4 : Unpack groups of 16
                unpacked_data = unpackLoHi16(unpacked_data);

                // Step 5 : Unpack groups of 32
                unpacked_data = unpackLoHi32(unpacked_data);

                return unpacked_data;
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<64, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<128, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<128, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 64;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<256, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 64;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<256, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 128;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo  = extractLo(v);
                auto hi  = extractHi(v);
                auto lo0 = extractLo(lo);
                auto lo1 = extractHi(lo);
                auto hi0 = extractLo(hi);
                auto hi1 = extractHi(hi);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
                auto v4 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
                auto v5 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
                auto v6 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
                auto v7 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

                return concat(concat(concat(v0, v1), concat(v2, v3)),
                              concat(concat(v4, v5), concat(v6, v7)));
            }
        };

#endif

        template <>
        struct soa_to_aos<16, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Scatter
                auto unpacked_data = PackUtil::template paddedUnpack<8>(
                    Permute::Scatter16<8, 0>::exec(PackUtil::paddedPack(v)));

                // Step 2 : UnpackLoHi2
                unpacked_data = unpackLoHi2(unpacked_data);

                // Step 3 : UnpackLoHi4
                unpacked_data = unpackLoHi4(unpacked_data);

                // Step 4 : UnpackLoHi8
                unpacked_data = unpackLoHi8(unpacked_data);

                return unpacked_data;
            }
        };

        template <>
        struct soa_to_aos<32, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                if constexpr(PackUtil::Traits::PackRatio >= VW)
                {
                    // Small types more efficient to do full rotation.
                    // Step 1 : Scatter
                    auto packed_data = Permute::Scatter32<8, 0>::exec(PackUtil::paddedPack(v));

                    // Step 2 : UnpackLoHi4
                    auto unpacked_data
                        = unpackLoHi4(PackUtil::template paddedUnpack<VW>(packed_data));

                    // Step 3 : UnpackLoHi8
                    unpacked_data = unpackLoHi8(unpacked_data);

                    // Step 4 : UnpackLoHi16
                    unpacked_data = unpackLoHi16(unpacked_data);
                    return unpacked_data;
                }
                else
                {
                    // Step 1 : Scatter (half-rotate offset)
                    auto hi = (Permute::Scatter32<8, 16>::exec(PackUtil::paddedPack(extractHi(v))));
                    auto lo = (Permute::Scatter32<8, 0>::exec(PackUtil::paddedPack(extractLo(v))));
                    auto unpacked_data = concat(PackUtil::template paddedUnpack<4>(lo),
                                                PackUtil::template paddedUnpack<4>(hi));

                    // Step 2 : UnpackLoHi4
                    unpacked_data = unpackLoHi4(unpacked_data);

                    // Step 3 : UnpackLoHi8
                    unpacked_data = unpackLoHi8(unpacked_data);

                    // Step 4 : UnpackLoHi16 (half-rotate offset)
                    lo = PackUtil::paddedPack(extractEven(unpacked_data));
                    hi = PackUtil::paddedPack(extractOdd(unpacked_data));

                    auto lo_final = Dpp::Driver<DppImpl::Ops::MaskMove, 0x5, 0xF>::exec(lo, hi);
                    hi            = Dpp::Driver<DppImpl::Ops::MaskMove, 0x5, 0xF>::exec(hi, lo);

                    hi = Swizzle::RotateR32<16>::exec(hi);

                    return concat(PackUtil::template paddedUnpack<4u>(lo_final),
                                  PackUtil::template paddedUnpack<4u>(hi));
                }
            }
        };

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<64, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                if constexpr(PackUtil::Traits::PackRatio >= VW)
                {
                    // Small types more efficient to do full rotation.
                    // Step 1 : Scatter
                    auto packed_data = Permute::ScatterWave<8, 0>::exec(PackUtil::paddedPack(v));

                    // Step 2 : UnpackLoHi8
                    auto unpacked_data = unpackLoHi8(PackUtil::paddedUnPack<VW>(packed_data));

                    // Step 3 : UnpackLoHi16
                    unpacked_data = unpackLoHi16(unpacked_data);

                    // Step 4 : UnpackLoHi32
                    unpacked_data = unpackLoHi32(unpacked_data);
                    return unpacked_data;
                }
                else
                {
                    // Step 1 : Scatter (half-rotate offset)
                    auto hi
                        = (Permute::ScatterWave<8, 32>::exec(PackUtil::paddedPack(extractHi(v))));
                    auto lo
                        = (Permute::ScatterWave<8, 0>::exec(PackUtil::paddedPack(extractLo(v))));
                    auto unpacked_data = concat(PackUtil::template paddedUnpack<4>(lo),
                                                PackUtil::template paddedUnpack<4>(hi));

                    // Step 2 : UnpackLoHi8
                    unpacked_data = unpackLoHi8(unpacked_data);

                    // Step 3 : unpackLoHi16
                    unpacked_data = unpackLoHi16(unpacked_data);

                    // Step 4 : UnpackLoHi32 (half-rotate offset)
                    lo = PackUtil::paddedPack(extractEven(unpacked_data));
                    hi = PackUtil::paddedPack(extractOdd(unpacked_data));

                    auto lo_final = Dpp::Driver<DppImpl::Ops::MaskMove, 0x3, 0xF>::exec(lo, hi);
                    hi            = Dpp::Driver<DppImpl::Ops::MaskMove, 0x3, 0xF>::exec(hi, lo);

                    hi = Permute::RotateWaveR<32>::exec(hi);

                    return concat(PackUtil::template paddedUnpack<VecSize / 2u>(lo_final),
                                  PackUtil::template paddedUnpack<VecSize / 2u>(hi));
                }
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<64, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<128, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<128, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<256, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<256, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 64;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo  = extractLo(v);
                auto hi  = extractHi(v);
                auto lo0 = extractLo(lo);
                auto lo1 = extractHi(lo);
                auto hi0 = extractLo(hi);
                auto hi1 = extractHi(hi);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
                auto v4 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
                auto v5 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
                auto v6 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
                auto v7 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

                return concat(concat(concat(v0, v1), concat(v2, v3)),
                              concat(concat(v4, v5), concat(v6, v7)));
            }
        };

#endif

        template <>
        struct soa_to_aos<16, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Scatter
                auto unpacked_data = PackUtil::template paddedUnpack<VW>(
                    Permute::Scatter16<4, 0>::exec(PackUtil::paddedPack(v)));

                // Step 2 : UnpackLoHi4
                unpacked_data = unpackLoHi4(unpacked_data);

                // Step 3 : UnpackLoHi8
                unpacked_data = unpackLoHi8(unpacked_data);

                return unpacked_data;
            }
        };

        template <>
        struct soa_to_aos<32, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                if constexpr(PackUtil::Traits::PackRatio >= VW)
                {
                    // Small types more efficient to do full rotation.
                    // Step 1 : Scatter
                    auto packed_data = Permute::Scatter32<4, 0>::exec(PackUtil::paddedPack(v));

                    // Step 2 : UnpackLoHi8
                    auto unpacked_data
                        = unpackLoHi8(PackUtil::template paddedUnpack<VW>(packed_data));

                    // Step 3 : UnpackLoHi16
                    unpacked_data = unpackLoHi16(unpacked_data);
                    return unpacked_data;
                }
                else
                {
                    // Step 1 : Scatter (half-rotate offset)
                    auto hi = (Permute::Scatter32<4, 16>::exec(PackUtil::paddedPack(extractHi(v))));
                    auto lo = (Permute::Scatter32<4, 0>::exec(PackUtil::paddedPack(extractLo(v))));
                    auto unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                                PackUtil::template paddedUnpack<VW / 2>(hi));

                    // Step 2 : UnpackLoHi8
                    unpacked_data = unpackLoHi8(unpacked_data);

                    // Step 3 : UnpackLoHi16 (half-rotate offset)
                    lo             = PackUtil::paddedPack(extractEven(unpacked_data));
                    hi             = PackUtil::paddedPack(extractOdd(unpacked_data));
                    auto zipped_lo = Dpp::Zip16::exec(lo, hi);
                    auto zipped_hi = Dpp::Zip16::exec(hi, lo);
                    auto rot_hi    = Swizzle::RotateR32<16>::exec(zipped_hi);
                    unpacked_data  = concat(PackUtil::template paddedUnpack<VW / 2>(zipped_lo),
                                           PackUtil::template paddedUnpack<VW / 2>(rot_hi));

                    return unpacked_data;
                }
            }
        };

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<64, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                if constexpr(PackUtil::Traits::PackRatio >= VW)
                {
                    // Small types more efficient to do full rotation.
                    // Step 1 : Scatter
                    auto packed_data = Permute::ScatterWave<4, 0>::exec(PackUtil::paddedPack(v));

                    // Step 2 : UnpackLoHi16
                    auto unpacked_data
                        = unpackLoHi16(PackUtil::template paddedUnpack<VW>(packed_data));

                    // Step 3 : UnpackLoHi32
                    unpacked_data = unpackLoHi32(unpacked_data);
                    return unpacked_data;
                }
                else
                {
                    // Step 1 : Scatter (half-rotate offset)
                    auto lo = Permute::ScatterWave<4, 0>::exec(PackUtil::paddedPack(extractLo(v)));
                    auto hi = Permute::ScatterWave<4, 32>::exec(PackUtil::paddedPack(extractHi(v)));
                    auto unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                                PackUtil::template paddedUnpack<VW / 2>(hi));

                    // Step 2 : UnpackLoHi16
                    unpacked_data = unpackLoHi16(unpacked_data);

                    // Step 3 : UnpackLoHi32 (half-rotate offset)
                    lo = PackUtil::paddedPack(extractEven(unpacked_data));
                    hi = PackUtil::paddedPack(extractOdd(unpacked_data));

                    auto zip_lo = Dpp::Zip32::exec(lo, hi);
                    auto zip_hi = Dpp::Zip32::exec(hi, lo);

                    auto rot_hi = Permute::RotateWaveR<32>::exec(zip_hi);

                    unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(zip_lo),
                                           PackUtil::template paddedUnpack<VW / 2>(rot_hi));

                    return unpacked_data;
                }
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<64, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<128, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<128, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<256, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<256, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo  = extractLo(v);
                auto hi  = extractHi(v);
                auto lo0 = extractLo(lo);
                auto lo1 = extractHi(lo);
                auto hi0 = extractLo(hi);
                auto hi1 = extractHi(hi);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
                auto v4 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
                auto v5 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
                auto v6 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
                auto v7 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

                return concat(concat(concat(v0, v1), concat(v2, v3)),
                              concat(concat(v4, v5), concat(v6, v7)));
            }
        };

#endif

        template <>
        struct soa_to_aos<16, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 2;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Scatter
                auto unpacked_data = PackUtil::template paddedUnpack<2>(
                    Permute::Scatter16<2, 0>::exec(PackUtil::paddedPack(v)));

                // Step 2 : UnpackLoHi8
                unpacked_data = unpackLoHi8(unpacked_data);

                return unpacked_data;
            }
        };

        template <>
        struct soa_to_aos<32, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 2;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Scatter
                auto unpacked_data = PackUtil::template paddedUnpack<2>(
                    Permute::Scatter32<2, 0>::exec(PackUtil::paddedPack(v)));

                // Step 2 : UnpackLoHi16
                unpacked_data = unpackLoHi16(unpacked_data);

                return unpacked_data;
            }
        };

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<64, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 2;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Scatter
                auto unpacked_data = PackUtil::template paddedUnpack<2>(
                    Permute::ScatterWave<2, 0>::exec(PackUtil::paddedPack(v)));

                // Step 2 : UnpackLoHi32
                unpacked_data = unpackLoHi32(unpacked_data);

                return unpacked_data;
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<64, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<128, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<128, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct soa_to_aos<256, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct soa_to_aos<256, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo  = extractLo(v);
                auto hi  = extractHi(v);
                auto lo0 = extractLo(lo);
                auto lo1 = extractHi(lo);
                auto hi0 = extractLo(hi);
                auto hi1 = extractHi(hi);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
                auto v1 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
                auto v2 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
                auto v3 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
                auto v4 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
                auto v5 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
                auto v6 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
                auto v7 = soa_to_aos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

                return concat(concat(concat(v0, v1), concat(v2, v3)),
                              concat(concat(v4, v5), concat(v6, v7)));
            }
        };
#endif

    } // namespace RegisterTransforms_impl

} // namespace rocWMMA

#endif // ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_IMPL_HPP
