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

#ifndef ROCWMMA_DEVICE_PACK_UTIL_TEST_HPP
#define ROCWMMA_DEVICE_PACK_UTIL_TEST_HPP

#include <rocwmma/rocwmma.hpp>

#include "common.hpp"
#include "unit_test_macros.hpp"

static constexpr bool DebugOnFail = false;

#define PACK_UTIL_EXPECT_EQ(LHS, RHS) \
    ROCWMMA_EXPECT_EQUAL(LHS, RHS, DebugOnFail);

namespace rocwmma
{


    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline DataT get(VecT<DataT, VecSize> const& v, uint32_t idx)
    {
        return v.data[idx];
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto generateSeqVec()
    {
        auto buildSeq = [](auto&& idx) {
            constexpr auto Index = std::decay_t<decltype(idx)>::value;
            return static_cast<DataT>(Index);
        };

        return vector_generator<DataT, VecSize>()(buildSeq);
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool packUtilTestBasic()
    {
        bool err = false;

        auto res = generateSeqVec<DataT, VecSize>();

        for(uint32_t i = 0; i < VecSize; i++)
        {
            err |= !PACK_UTIL_EXPECT_EQ((get(res, i)), static_cast<DataT>(i));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool padTest()
    {
        /**
         * 8 bits [0x01, 0x02, 0x03, 0x04] -> 32 bits [0x00000001, 0x00000002, 0x00000003, 0x00000004]
         * 16 bits [0x0001, 0x0002, 0x0003, 0x0004] -> 32 bits [0x00000001, 0x00000002, 0x00000003, 0x00000004]
         * 32 bits, return the argument itself
         * 64 bits, return the argument itself
         **/
        using PackUtil = rocwmma::PackUtil<DataT>;

        bool err = false;
        auto res = generateSeqVec<DataT, VecSize>();

        if constexpr(PackUtil::Traits::PackRatio == 1)
        {
            err |= !PACK_UTIL_EXPECT_EQ(res, PackUtil::pad(res));
        }
        else
        {
            using PackedT   = typename PackUtil::Traits::PackedT;
            using UnpackedT = typename PackUtil::Traits::UnpackedT;

            auto paddedData = PackUtil::pad(res);
            err |= !PACK_UTIL_EXPECT_EQ((std::is_same_v<decltype(paddedData), VecT<PackedT, VecSize>>), true);
            for(uint32_t i = 0; i < VecSize; i++)
            {
                auto value = get(paddedData, i);

                // unpadded data is at idx 0 in the padded data by default
                err |= !PACK_UTIL_EXPECT_EQ(*reinterpret_cast<DataT*>(&value), static_cast<DataT>((float)i));
            }
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool padWithIdxTest()
    {
        /**
         * 8 bits [0x01, 0x02, 0x03, 0x04] -> 32 bits [0x00000100, 0x00000200, 0x00000300, 0x00000400]
         * 16 bits [0x0001, 0x0002, 0x0003, 0x0004] -> 32 bits [0x00010000, 0x00020000, 0x00030000, 0x00040000]
         * 32 bits, return the argument itself
         * 64 bits, return the argument itself
         **/
        using PackUtil = rocwmma::PackUtil<DataT>;

        bool err = false;
        auto res = generateSeqVec<DataT, VecSize>();

        if constexpr(PackUtil::Traits::PackRatio == 1)
        {
            err |= !PACK_UTIL_EXPECT_EQ(res, PackUtil::pad(res));
        }
        else
        {
            using PackedT   = typename PackUtil::Traits::PackedT;
            using UnpackedT = typename PackUtil::Traits::UnpackedT;

            // padIdx == 1 which is valid for 8 bits and 16 bits
            auto paddedData = PackUtil::template pad<1>(res);
            err |= !PACK_UTIL_EXPECT_EQ((std::is_same_v<decltype(paddedData), VecT<PackedT, VecSize>>), true);
            for(uint32_t i = 0u; i < VecSize; i++)
            {
                PackedT value = get(paddedData, i);
                err |= !PACK_UTIL_EXPECT_EQ(*(reinterpret_cast<UnpackedT*>(&value) + 1u), static_cast<UnpackedT>(i));
            }
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool unpadTest()
    {
        /**
         * 32 bits [0x00000001, 0x00000002, 0x00000003, 0x00000004] -> 8 bits [0x01, 0x02, 0x03, 0x04]
         * 32 bits [0x00000001, 0x00000002, 0x00000003, 0x00000004] -> 16 bits [0x0001, 0x0002, 0x0003, 0x0004]
         * 32 bits, return the argument itself
         * 64 bits, return the argument itself
         **/
        using PackUtil  = rocwmma::PackUtil<DataT>;
        using PackedT   = typename PackUtil::Traits::PackedT;
        using UnpackedT = typename PackUtil::Traits::UnpackedT;

        bool err = false;

        if constexpr(PackUtil::Traits::PackRatio == 1)
        {
            auto res = generateSeqVec<PackedT, VecSize>();
            err |= !PACK_UTIL_EXPECT_EQ(res, PackUtil::unpad(res));
        }
        else
        {
            auto res = VecT<PackedT, VecSize>(0);
            for(uint32_t i = 0; i < VecSize; i++)
            {
                *reinterpret_cast<UnpackedT*>(reinterpret_cast<PackedT*>(&res.data) + i)
                    = static_cast<UnpackedT>((float)i);
            }

            auto unpaddedData = PackUtil::unpad(res);
            err |= !PACK_UTIL_EXPECT_EQ((std::is_same_v<decltype(unpaddedData), VecT<UnpackedT, VecSize>>), true);
            err |= !PACK_UTIL_EXPECT_EQ(unpaddedData, (generateSeqVec<DataT, VecSize>()));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool unpadWithIdxTest()
    {
        /**
         * 32 bits [0x00000100, 0x00000200, 0x00000300, 0x00000400] -> 8 bits [0x01, 0x02, 0x03, 0x04]
         * 32 bits [0x00010000, 0x00020000, 0x00030000, 0x00040000] -> 16 bits [0x0001, 0x0002, 0x0003, 0x0004]
         * 32 bits, return the argument itself
         * 64 bits, return the argument itself
         **/
        using PackUtil  = rocwmma::PackUtil<DataT>;
        using PackedT   = typename PackUtil::Traits::PackedT;
        using UnpackedT = typename PackUtil::Traits::UnpackedT;

        bool err = false;

        if constexpr(PackUtil::Traits::PackRatio == 1)
        {
            auto res = generateSeqVec<PackedT, VecSize>();
            err |= !PACK_UTIL_EXPECT_EQ(res, PackUtil::unpad(res));
        }
        else
        {
            auto res = VecT<PackedT, VecSize>(0);
            for(uint32_t i = 0; i < VecSize; i++)
            {
                *(reinterpret_cast<UnpackedT*>(reinterpret_cast<PackedT*>(&res.data) + i) + 1)
                    = static_cast<UnpackedT>((float)i);
            }

            auto unpaddedData = PackUtil::template unpad<1>(res);
            err |= !PACK_UTIL_EXPECT_EQ((std::is_same_v<decltype(unpaddedData), VecT<UnpackedT, VecSize>>), true);
            err |= !PACK_UTIL_EXPECT_EQ(unpaddedData, (generateSeqVec<DataT, VecSize>()));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool packTest()
    {
        using PackUtil  = rocwmma::PackUtil<DataT>;
        using PackedT   = typename PackUtil::Traits::PackedT;
        using UnpackedT = typename PackUtil::Traits::UnpackedT;

        bool err = false;

        // required by pack(): VecSize % PackUtil::Traits::PackRatio == 0
        if constexpr(VecSize % PackUtil::Traits::PackRatio == 0)
        {
            auto res = generateSeqVec<UnpackedT, VecSize>();

            // pack() cast VecT<UnpackedT, VecSize> const &  to VecT<PackedT, VecSize / PackRatio> const &
            err |= !PACK_UTIL_EXPECT_EQ((std::is_same_v<decltype(PackUtil::pack(res)),
                                   VecT<PackedT, VecSize / PackUtil::Traits::PackRatio> const&>), true);

            // argument and return of pack() point to the same addr
            err |= !PACK_UTIL_EXPECT_EQ(reinterpret_cast<void const*>(&PackUtil::pack(res)), reinterpret_cast<void const*>(&res));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool unpackTest()
    {
        using PackUtil  = rocwmma::PackUtil<DataT>;
        using PackedT   = typename PackUtil::Traits::PackedT;
        using UnpackedT = typename PackUtil::Traits::UnpackedT;

        bool err = false;

        auto res = generateSeqVec<PackedT, VecSize>();

        // pack() cast VecT<PackedT, VecSize> const &  to VecT<UnpackedT, VecSize * PackRatio> const &
        err |= !PACK_UTIL_EXPECT_EQ((std::is_same_v<decltype(PackUtil::unpack(res)),
                               VecT<UnpackedT, VecSize * PackUtil::Traits::PackRatio> const&>), true);

        // argument and return of pack() point to the same addr
        err |= !PACK_UTIL_EXPECT_EQ(reinterpret_cast<void const*>(&PackUtil::unpack(res)),
               reinterpret_cast<void const*>(&res));

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool paddedPackTest()
    {
        using PackUtil  = rocwmma::PackUtil<DataT>;
        using PackedT   = typename PackUtil::Traits::PackedT;
        using UnpackedT = typename PackUtil::Traits::UnpackedT;

        bool err = false;
        auto res = generateSeqVec<UnpackedT, VecSize>();

        if constexpr(VecSize % PackUtil::Traits::PackRatio == 0)
        {
            err |= !PACK_UTIL_EXPECT_EQ((std::is_same_v<decltype(PackUtil::paddedPack(res)),
                                   VecT<PackedT, VecSize / PackUtil::Traits::PackRatio> const&>), true);
            err |= !PACK_UTIL_EXPECT_EQ(reinterpret_cast<void const*>(&PackUtil::paddedPack(res)),
                    reinterpret_cast<void const*>(&res));
        }
        else if constexpr(VecSize * 2 == PackUtil::Traits::PackRatio)
        {
            auto expectedResult = VecT<PackedT, 1>(0);
            for(uint32_t i = 0; i < 2; i++)
            {
                *(reinterpret_cast<VecT<UnpackedT, VecSize>*>(&expectedResult) + i) = res;
            }

            err |= !PACK_UTIL_EXPECT_EQ(expectedResult, PackUtil::paddedPack(res));
        }
        else if constexpr(VecSize == 1)
        {
            if constexpr(PackUtil::Traits::PackRatio == 1)
            {
                err |= !PACK_UTIL_EXPECT_EQ(res, PackUtil::paddedPack(res));
            }
            else
            {
                auto paddedData = PackUtil::paddedPack(res);
                err |= !std::is_same_v<decltype(paddedData), VecT<PackedT, VecSize>>;
                for(uint32_t i = 0; i < VecSize; i++)
                {
                    auto value = get(paddedData, i);

                    // unpadded data is at idx 0 in the padded data by default
                    err |= !PACK_UTIL_EXPECT_EQ(*reinterpret_cast<DataT*>(&value), static_cast<DataT>((float)i));
                }
            }
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool paddedUnpackNoPaddingTest()
    {
        using PackUtil  = rocwmma::PackUtil<DataT>;
        using PackedT   = typename PackUtil::Traits::PackedT;
        using UnpackedT = typename PackUtil::Traits::UnpackedT;

        bool err = false;

        if constexpr(VecSize % PackUtil::Traits::PackRatio == 0)
        {
            auto res = generateSeqVec<PackedT, VecSize / PackUtil::Traits::PackRatio>();
            err |= !PACK_UTIL_EXPECT_EQ((std::is_same_v<decltype(PackUtil::template paddedUnpack<VecSize>(res)),
                                   VecT<UnpackedT, VecSize> const&>), true);
            err |= !PACK_UTIL_EXPECT_EQ(reinterpret_cast<void const*>(&PackUtil::template paddedUnpack<VecSize>(res)),
                    reinterpret_cast<void const*>(&res));
        }
        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool paddedUnpackWithLowHalfTest()
    {
        using PackUtil  = rocwmma::PackUtil<DataT>;
        using PackedT   = typename PackUtil::Traits::PackedT;
        using UnpackedT = typename PackUtil::Traits::UnpackedT;

        bool err = false;

        if constexpr(VecSize * 2 == PackUtil::Traits::PackRatio)
        {
            auto constexpr packSize = std::max(1u, VecSize / PackUtil::Traits::PackRatio);
            auto res                = VecT<PackedT, packSize>(0);
            auto expectedData       = generateSeqVec<UnpackedT, VecSize>();
            for(uint32_t i = 0; i < VecSize; i++)
            {
                *(reinterpret_cast<VecT<UnpackedT, VecSize>*>(&res.data) + i) = expectedData;
            }
            err |= !PACK_UTIL_EXPECT_EQ(PackUtil::template paddedUnpack<VecSize>(res), expectedData);
        }
        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool paddedUnpackWithSizeOneTest()
    {
        using PackUtil  = rocwmma::PackUtil<DataT>;
        using PackedT   = typename PackUtil::Traits::PackedT;
        using UnpackedT = typename PackUtil::Traits::UnpackedT;

        bool err = false;

        if constexpr(VecSize == 1)
        {
            auto res          = VecT<PackedT, 1>(0);
            auto expectedData = VecT<UnpackedT, 1>(static_cast<UnpackedT>(1.0F));
            (*(reinterpret_cast<VecT<UnpackedT, VecSize>*>(&res.data))).data[0]
                = expectedData.data[0];
            err |= !PACK_UTIL_EXPECT_EQ(PackUtil::template paddedUnpack<VecSize>(res), expectedData);
        }
        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_KERNEL void packUtilTest(uint32_t     m,
                                     uint32_t     n,
                                     DataT const* in,
                                     DataT*       out,
                                     uint32_t     ld,
                                     DataT        param1,
                                     DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        // Add tests here
        err = err ? err : packUtilTestBasic<DataT, VecSize>();
        err = err ? err : padTest<DataT, VecSize>();
        err = err ? err : padWithIdxTest<DataT, VecSize>();
        err = err ? err : unpadTest<DataT, VecSize>();
        err = err ? err : unpadWithIdxTest<DataT, VecSize>();
        err = err ? err : packTest<DataT, VecSize>();
        err = err ? err : unpackTest<DataT, VecSize>();
        err = err ? err : paddedPackTest<DataT, VecSize>();
        err = err ? err : paddedUnpackNoPaddingTest<DataT, VecSize>();
        err = err ? err : paddedUnpackWithLowHalfTest<DataT, VecSize>();
        err = err ? err : paddedUnpackWithSizeOneTest<DataT, VecSize>();

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_PACK_UTIL_TEST_HPP
