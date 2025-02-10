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
        template <uint32_t GroupSize, typename DataT, uint32_t VecSize>
        struct aos_to_soa_int
        {
            constexpr static uint32_t VW      = VectorWidth;
            constexpr static uint32_t VecSize = VW * (BlockDim / Constants::AMDGCN_WAVE_SIZE);

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                return v;
            }
        };

        template <uint32_t BlockDim, uint32_t VectorWidth>
        struct aos_to_soa
        {
            constexpr static uint32_t VW      = VectorWidth;
            constexpr static uint32_t VecSize = VW * (BlockDim / Constants::AMDGCN_WAVE_SIZE);

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                return v;
            }
        };

        template <>
        struct aos_to_soa<16, 16>
        {
            constexpr static uint32_t VW       = 16;
            constexpr static uint32_t VecSize  = 16;
            constexpr static uint32_t BlockDim = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Unpack groups of 1
                auto result = unpackLoHi1(v);

                // Step 2 : Unpack groups of 2
                result = unpackLoHi2(result);

                // Step 3 : Unpack groups of 4
                result = unpackLoHi4(result);

                // Step 4 : Unpack groups of 8
                result = unpackLoHi8(result);

                return result;
            }
        };

        template <>
        struct aos_to_soa<32, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Unpack groups of 2
                auto result = unpackLoHi2(v);

                // Step 2 : Unpack groups of 4
                result = unpackLoHi4(result);

                // Step 3 : Unpack groups of 8
                result = unpackLoHi8(result);

                // Step 4 : Unpack groups of 16 (half-rotate offset)
                auto evens = PackUtil::paddedPack(extractEven(result));
                auto odds  = PackUtil::paddedPack(extractOdd(result));

                auto rot = Swizzle::RotateR32<16>::exec(odds);
                auto lo  = Dpp::Zip16::exec(evens, rot);
                auto hi  = Dpp::Zip16::exec(rot, evens);

                // Step 5 : Gather (half-rotate offset)
                // Note the offset of 16 in hi
                lo = Permute::Gather32<VW, 0>::exec(lo);
                hi = Permute::Gather32<VW, 16>::exec(hi);

                return PackUtil::template paddedUnpack<VecSize>(concat(lo, hi));
            }
        };

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<64, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Unpack groups of 4
                auto result = unpackLoHi4(v);

                // Step 2 : Unpack groups of 8
                result = unpackLoHi8(result);

                // Step 3 : Unpack groups of 16
                result = unpackLoHi16(result);

                // Step 4 : Unpack groups of 32 (half-rotate offset)
                auto lo = PackUtil::paddedPack(extractEven(result));
                auto hi = PackUtil::paddedPack(extractOdd(result));

                auto rot_hi = Permute::RotateWaveR<32>::exec(hi);
                hi          = Dpp::Zip32::exec(rot_hi, lo);
                lo          = Dpp::Zip32::exec(lo, rot_hi);

                // Step 5 : Gather (half-rotate offset)
                // Note the offset of 32 in hi
                lo = Permute::GatherWave<VW, 0>::exec(lo);
                hi = Permute::GatherWave<VW, 32>::exec(hi);

                return PackUtil::template paddedUnpack<VecSize>(concat(lo, hi));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<64, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<128, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // There are TWO sets of VW = 16 registers (because this case BlockDim / 64 = 2):
                // 1. Vecs 0-15
                // 2. Vecs 16-31
                //
                // Register/ |          VW = 16                |
                //     Tidx  |___0___|___1___|___...___|___15__|
                //         0 |   0   |   1   |   ...   |   15  |
                //         1 |   16  |   9   |   ...   |   31  |
                //       ... |   ... |   ... |   ...   |  ...  |
                //        63 |__1968_|__1969_|___...___|__1983_|
                //
                // Register/ |          VW = 16                |
                //     Tidx  |___16__|___17__|___...___|___31__|
                //         0 |   64  |   65  |   ...   |   79  |
                //         1 |   80  |   81  |   ...   |   95  |
                //       ... |   ... |   ... |   ...   |  ...  |
                //        63 |__2032_|__2033_|___...___|__2047_|

                // Subdivide work to each batch of WAVE_SIZE
                auto result_b0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
                auto result_b1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

                return concat(result_b0, result_b1);
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<128, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 64;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<256, 16>
        {
            constexpr static uint32_t VW      = 16;
            constexpr static uint32_t VecSize = 64;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // There are FOUR sets of VW = 8 registers (because this case BlockDim / 64 = 4):
                // 1. Vecs 0-7
                // 2. Vecs 8-15
                // 3. Vecs 16-23
                // 4. Vecs 24-31
                //
                // Register/ |          VW = 8                 |
                //     Tidx  |___0___|___1___|___...___|___15__|
                //         0 |   0   |   1   |   ...   |   15  |
                //         1 |   16  |   17  |   ...   |   31  |
                //       ... |   ... |   ... |   ...   |  ...  |
                //        63 |__3888_|__3889_|___...___|__3903_|
                //
                // Register/ |          VW = 8                 |
                //     Tidx  |___16__|___9___|___...___|___31__|
                //         0 |  64   |  65   |   ...   |   79  |
                //         1 |  80   |  81   |   ...   |   95  |
                //       ... |   ... |   ... |   ...   |  ...  |
                //        63 |__3952_|__3953_|___...___|__3967_|
                //
                // Register/ |          VW = 8                    |
                //     Tidx  |___32___|___17___|___...___|___47___|
                //         0 |  128   |  129   |   ...   |  143   |
                //         1 |  144   |  145   |   ...   |  159   |
                //       ... |   .... |   .... |   ...   |  ....  |
                //        63 |__4016__|__4017__|___...___|__4031__|
                //
                // Register/ |          VW = 8                    |
                //     Tidx  |___48___|___25___|___...___|___63___|
                //         0 |  192   |  193   |   ...   |  207   |
                //         1 |  208   |  209   |   ...   |  223   |
                //       ... |   ...  |   ...  |   ...   |  ...   |
                //        63 |__4080__|__4081__|___...___|__4095 _|

                // Extract each batch of registers and put them through the 64 size
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<256, 16>
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
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
                auto v4 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
                auto v5 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
                auto v6 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
                auto v7 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

                return concat(concat(concat(v0, v1), concat(v2, v3)),
                              concat(concat(v4, v5), concat(v6, v7)));
            }
        };

#endif

        template <>
        struct aos_to_soa<16, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Unpack groups of 2
                auto result = unpackLoHi2(v);

                // Step 2 : Unpack groups of 4
                result = unpackLoHi4(result);

                // Step 3 : Unpack groups of 8
                result = unpackLoHi8(result);

                // Step 4 : Gather
                return PackUtil::template paddedUnpack<VecSize>(
                    Permute::Gather16<VW, 0>::exec(PackUtil::paddedPack(result)));
            }
        };

        template <>
        struct aos_to_soa<32, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Unpack groups of 4
                auto result = unpackLoHi4(v);

                // Step 2 : Unpack groups of 8
                result = unpackLoHi8(result);

                if constexpr(PackUtil::Traits::PackRatio >= VW)
                {
                    // Step 3 : UnpackLoHi16
                    // Small types more efficient to do full rotation.
                    result = unpackLoHi16(result);

                    // Step 4 : Gather
                    auto packed_data = Permute::Gather32<VW, 0>::exec(PackUtil::paddedPack(result));

                    return PackUtil::template paddedUnpack<VecSize>(packed_data);
                }
                else
                {
                    // Step 3 : Unpack groups of 16 (half-rotate offset)
                    // Larger types more efficient to rotate half block and
                    // make up the offset in Gather
                    auto evens = PackUtil::paddedPack(extractEven(result));
                    auto odds  = PackUtil::paddedPack(extractOdd(result));

                    auto rot = Swizzle::RotateR32<16>::exec(odds);
                    auto lo  = Dpp::Zip16::exec(evens, rot);
                    auto hi  = Dpp::Zip16::exec(rot, evens);

                    // Step 4 : Gather (half-rotate offset)
                    // Note the offset of 16 in hi
                    lo = Permute::Gather32<VW, 0>::exec(lo);
                    hi = Permute::Gather32<VW, 16>::exec(hi);

                    return PackUtil::template paddedUnpack<VecSize>(concat(lo, hi));
                }
            }
        };

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<64, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : Unpack groups of 8
                auto result = unpackLoHi8(v);

                // Step 2 : Unpack groups of 16
                result = unpackLoHi16(result);

                if constexpr(PackUtil::Traits::PackRatio >= VW)
                {
                    // Step 2 : UnpackLoHi32
                    // Small types more efficient to do full rotation.
                    result = unpackLoHi32(result);

                    // Step 3 : Gather
                    auto packed_data
                        = Permute::GatherWave<VW, 0>::exec(PackUtil::paddedPack(result));

                    return PackUtil::template paddedUnpack<VecSize>(packed_data);
                }
                else
                {
                    // Step 3 : Unpack groups of 32 (half-rotate offset)
                    // In order to save some operations, we can
                    // rotate the odds components only and make up the
                    // offset later in gather.
                    auto lo = PackUtil::paddedPack(extractEven(result));
                    auto hi = PackUtil::paddedPack(extractOdd(result));

                    // TODO: label as rotateR64 for consistency?
                    auto rot_hi = Permute::RotateWaveR<32>::exec(hi);
                    hi          = Dpp::Zip32::exec(rot_hi, lo);
                    lo          = Dpp::Zip32::exec(lo, rot_hi);

                    // Step 4 : Gather (half-rotate offset)
                    // Note the offset of 32 in hi
                    lo = Permute::GatherWave<VW, 0>::exec(lo);
                    hi = Permute::GatherWave<VW, 32>::exec(hi);

                    return PackUtil::template paddedUnpack<VecSize>(concat(lo, hi));
                }
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<64, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<128, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // There are TWO sets of VW = 8 registers (because this case BlockDim / 64 = 2):
                // 1. Vecs 0-7
                // 2. Vecs 8-15
                //
                // Register/ |          VW = 8                 |
                //     Tidx  |___0___|___1___|___...___|___7___|
                //         0 |   0   |   1   |   ...   |   7   |
                //         1 |   8   |   9   |   ...   |   15  |
                //       ... |   ... |   ... |   ...   |  ...  |
                //        63 |__952__|__953__|___...___|__959__|
                //
                // Register/ |          VW = 8                 |
                //     Tidx  |___8___|___9___|___...___|___15__|
                //         0 |  64   |  65   |   ...   |   71  |
                //         1 |  72   |  73   |   ...   |   79  |
                //       ... |   ... |   ... |   ...   |  ...  |
                //        63 |__1016_|__1017_|___...___|__1023_|

                // Subdivide work to each batch of WAVE_SIZE
                auto result_b0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
                auto result_b1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

                return concat(result_b0, result_b1);
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<128, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<256, 8>
        {
            constexpr static uint32_t VW      = 8;
            constexpr static uint32_t VecSize = 32;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // There are FOUR sets of VW = 8 registers (because this case BlockDim / 64 = 4):
                // 1. Vecs 0-7
                // 2. Vecs 8-15
                // 3. Vecs 16-23
                // 4. Vecs 24-31
                //
                // Register/ |          VW = 8                 |
                //     Tidx  |___0___|___1___|___...___|___7___|
                //         0 |   0   |   1   |   ...   |   7   |
                //         1 |   8   |   9   |   ...   |   15  |
                //       ... |   ... |   ... |   ...   |  ...  |
                //        63 |__1848_|__1849_|___...___|__1855_|
                //
                // Register/ |          VW = 8                 |
                //     Tidx  |___8___|___9___|___...___|___15__|
                //         0 |  64   |  65   |   ...   |   71  |
                //         1 |  72   |  73   |   ...   |   79  |
                //       ... |   ... |   ... |   ...   |  ...  |
                //        63 |__1912_|__1913_|___...___|__1919_|
                //
                // Register/ |          VW = 8                    |
                //     Tidx  |___16___|___17___|___...___|___23___|
                //         0 |  128   |  129   |   ...   |  135   |
                //         1 |  136   |  137   |   ...   |  143   |
                //       ... |   .... |   .... |   ...   |  ....  |
                //        63 |__1976__|__1977__|___...___|__1983__|
                //
                // Register/ |          VW = 8                    |
                //     Tidx  |___24___|___25___|___...___|___31___|
                //         0 |  192   |  193   |   ...   |  199   |
                //         1 |  200   |  201   |   ...   |  207   |
                //       ... |   ...  |   ...  |   ...   |  ...   |
                //        63 |__2040__|__2041__|___...___|__2047 _|

                // Extract each batch of registers and put them through the 64 size
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<256, 8>
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
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
                auto v4 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
                auto v5 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
                auto v6 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
                auto v7 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

                return concat(concat(concat(v0, v1), concat(v2, v3)),
                              concat(concat(v4, v5), concat(v6, v7)));
            }
        };

#endif

        template <>
        struct aos_to_soa<16, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : UnpackLoHi4
                auto unpacked_data = unpackLoHi4(v);

                // Step 2 : UnpackLoHi8
                unpacked_data = unpackLoHi8(unpacked_data);

                // Step 3 : Gather
                unpacked_data = PackUtil::template paddedUnpack<4>(
                    Permute::Gather16<4, 0>::exec(PackUtil::paddedPack(unpacked_data)));

                return unpacked_data;
            }
        };

        template <>
        struct aos_to_soa<32, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : UnpackLoHi8
                auto unpacked_data = unpackLoHi8(v);

                if constexpr(PackUtil::Traits::PackRatio >= VW)
                {
                    // Step 2 : UnpackLoHi16
                    // Small types more efficient to do full rotation.
                    unpacked_data = unpackLoHi16(unpacked_data);

                    // Step 3 : Gather
                    auto packed_data
                        = Permute::Gather32<4, 0>::exec(PackUtil::paddedPack(unpacked_data));

                    return PackUtil::template paddedUnpack<VW>(packed_data);
                }
                else
                {
                    // Step 2 : UnpackLoHi16 (half-rotate offset)
                    // Larger types more efficient to rotate half block and
                    // make up the offset in Gather
                    auto lo       = PackUtil::paddedPack(extractEven(unpacked_data));
                    auto hi       = PackUtil::paddedPack(extractOdd(unpacked_data));
                    auto rot_hi   = Swizzle::RotateR32<16>::exec(hi);
                    hi            = Dpp::Zip16::exec(rot_hi, lo);
                    lo            = Dpp::Zip16::exec(lo, rot_hi);
                    unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                           PackUtil::template paddedUnpack<VW / 2>(hi));

                    // Step 3 : Gather (half-rotate offset)
                    hi = Permute::Gather32<4, 16>::exec(
                        PackUtil::paddedPack(extractHi(unpacked_data)));
                    lo = Permute::Gather32<4, 0>::exec(
                        PackUtil::paddedPack(extractLo(unpacked_data)));
                    unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                           PackUtil::template paddedUnpack<VW / 2>(hi));

                    return unpacked_data;
                }
            }
        };

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<64, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : UnpackLohi16
                auto unpacked_data = unpackLoHi16(v);

                if constexpr(PackUtil::Traits::PackRatio >= VW)
                {
                    // Small types more efficient to do full rotation.
                    // Step 2 : UnpackLoHi32
                    unpacked_data = unpackLoHi32(unpacked_data);

                    // Step 3 : Gather
                    auto packed_data
                        = Permute::GatherWave<4, 0>::exec(PackUtil::paddedPack(unpacked_data));

                    return PackUtil::template paddedUnpack<VW>(packed_data);
                }
                else
                {
                    // Larger types more efficient to rotate half block and
                    // make up the offset in Gather
                    // Step 2 : UnpackLoHi32 (half-rotate offset)
                    auto lo = PackUtil::paddedPack(extractEven(unpacked_data));
                    auto hi = PackUtil::paddedPack(extractOdd(unpacked_data));

                    hi = Permute::RotateWaveR<32>::exec(hi);

                    auto zip_lo = Dpp::Zip32::exec(lo, hi);
                    auto zip_hi = Dpp::Zip32::exec(hi, lo);

                    // Step 3 : Gather (half-rotate offset)
                    lo = Permute::GatherWave<4, 0>::exec(zip_lo);
                    hi = Permute::GatherWave<4, 32>::exec(zip_hi);

                    return concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                  PackUtil::template paddedUnpack<VW / 2>(hi));
                }
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<64, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<128, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<128, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<256, 4>
        {
            constexpr static uint32_t VW      = 4;
            constexpr static uint32_t VecSize = 16;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<256, 4>
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
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
                auto v4 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
                auto v5 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
                auto v6 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
                auto v7 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

                return concat(concat(concat(v0, v1), concat(v2, v3)),
                              concat(concat(v4, v5), concat(v6, v7)));
            }
        };

#endif

        template <>
        struct aos_to_soa<16, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 2;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : UnpackLoHi8
                auto unpacked_data = unpackLoHi8(v);

                // Step 2 : Gather
                unpacked_data = PackUtil::template paddedUnpack<2>(
                    Permute::Gather16<2, 0>::exec(PackUtil::paddedPack(unpacked_data)));

                return unpacked_data;
            }
        };

        template <>
        struct aos_to_soa<32, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 2;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : UnpackLoHi16
                auto unpacked_data = unpackLoHi16(v);

                // Step 2 : Gather
                unpacked_data = PackUtil::template paddedUnpack<2>(
                    Permute::Gather32<2, 0>::exec(PackUtil::paddedPack(unpacked_data)));

                return unpacked_data;
            }
        };

#if ROCWMMA_WAVE64_MODE
        template <>
        struct aos_to_soa<64, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 2;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                using PackUtil = PackUtil<DataT>;

                // Step 1 : UnpackLoHi32
                auto unpacked_data = unpackLoHi32(v);

                // Step 2 : Gather
                unpacked_data = PackUtil::template paddedUnpack<2>(
                    Permute::GatherWave<2, 0>::exec(PackUtil::paddedPack(unpacked_data)));

                return unpacked_data;
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<64, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<128, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 4;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

                return concat(v0, v1);
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<128, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#endif

#if ROCWMMA_WAVE64_MODE

        template <>
        struct aos_to_soa<256, 2>
        {
            constexpr static uint32_t VW      = 2;
            constexpr static uint32_t VecSize = 8;

            template <typename DataT>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto lo = extractLo(v);
                auto hi = extractHi(v);

                // Subdivide work to each batch of WAVE_SIZE
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

                return concat(concat(v0, v1), concat(v2, v3));
            }
        };

#elif ROCWMMA_WAVE32_MODE

        template <>
        struct aos_to_soa<256, 2>
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
                auto v0 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
                auto v1 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
                auto v2 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
                auto v3 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
                auto v4 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
                auto v5 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
                auto v6 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
                auto v7 = aos_to_soa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

                return concat(concat(concat(v0, v1), concat(v2, v3)),
                              concat(concat(v4, v5), concat(v6, v7)));
            }
        };

#endif

        template <class Transform>
        struct Driver
        {
            using Func = Transform;

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto result = VecT<DataT, VecSize>{};

                auto rIt = makeVectorIterator<Func::VecSize>(v).begin();
                auto wIt = makeVectorIterator<Func::VecSize>(result).begin();

#pragma unroll
                for(uint32_t i = 0; i < decltype(rIt)::range(); i++)
                {
                    *wIt = Func::exec(*rIt);
                    rIt++;
                    wIt++;
                }
                return result;
            }
        };

    } // namespace RegisterTransforms_impl

} // namespace rocWMMA

#endif // ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_IMPL_HPP
