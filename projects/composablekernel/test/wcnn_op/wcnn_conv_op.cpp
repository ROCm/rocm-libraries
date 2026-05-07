// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "common_wcnn.hpp"

#define USE_ABSOLUTE_SIZE 1
#ifdef USE_ABSOLUTE_SIZE
#define DEFAULT_K 16
#define DEFAULT_C 32
#define DEFAULT_W 8
#define DEFAULT_H 16
#define DEFAULT_W_1K 48
#define DEFAULT_H_1K 32
#else
#define DEFAULT_C_REPEAT 1
#define DEFAULT_K_REPEAT 1
#define DEFAULT_W_REPEAT 1
#define DEFAULT_H_REPEAT 1
#endif

namespace ck {

namespace conv_op_util {

// #define LOAD_DATA_PER_TILE 0

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          bool UseF32I32,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Width,
          index_t Height,
          index_t UnpackedInputChannels,
          index_t UnpackedOutputChannels>
__global__ void __launch_bounds__(64, 1)
    conv_fwd(const InDataType* in_, const WeiDataType* wei_, AccDataType* c_)
{
    constexpr auto wcnn_conv = ck::WcnnConv<WeiDataType,
                                            InDataType,
                                            AccDataType,
                                            HPerWcnn,
                                            WPerWcnn,
                                            FilterSize,
                                            DilationX,
                                            DilationY,
                                            1,
                                            false,
                                            false,
                                            UseF32I32>();

    auto in  = reinterpret_cast<const typename decltype(wcnn_conv)::KernelInDataType*>(in_);
    auto wei = reinterpret_cast<const typename decltype(wcnn_conv)::KernelWeightDataType*>(wei_);

    static_assert(Width % WPerWcnn == 0, "");
    static_assert(Height % HPerWcnn == 0, "");
    static_assert(UnpackedInputChannels % wcnn_conv.GetUnpackedNumInputChannels() == 0, "");
    static_assert(UnpackedOutputChannels % wcnn_conv.GetUnpackedNumOutputChannels() == 0, "");

    constexpr index_t DataTileHeight = 4;
    constexpr index_t InputChannels  = UnpackedInputChannels * wcnn_conv.GetNumInputChannels() /
                                      wcnn_conv.GetUnpackedNumInputChannels();
    constexpr index_t OutputChannels = UnpackedOutputChannels * wcnn_conv.GetNumOutputChannels() /
                                       wcnn_conv.GetUnpackedNumOutputChannels();

    constexpr index_t HRepeat = Height / HPerWcnn;
    constexpr index_t WRepeat = Width / WPerWcnn;
    constexpr index_t CRepeat = UnpackedInputChannels / wcnn_conv.GetUnpackedNumInputChannels();
    constexpr index_t KRepeat = UnpackedOutputChannels / wcnn_conv.GetUnpackedNumOutputChannels();
#ifdef LOAD_DATA_PER_TILE
    constexpr index_t AccVectorCount = 1;
#else
    constexpr index_t AccVectorCount = HRepeat * WRepeat * KRepeat;
#endif

    using InDataVec      = typename decltype(wcnn_conv)::InDataVec::type;
    using InDataTileVec  = typename decltype(wcnn_conv)::InDataTileVec::type;
    using WeiDataVec     = typename decltype(wcnn_conv)::WeiDataVec::type;
    using WeiDataTileVec = typename decltype(wcnn_conv)::WeiDataTileVec::type;
    using AccDataVec     = typename decltype(wcnn_conv)::AccDataVec;
    using AccVec         = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                                     AccDataType,
                                                     AccVectorCount,
                                                     wcnn_conv.GetNumAccumComponents(),
                                                     true>;

    const int lIdx = threadIdx.x;
#if !defined(LOAD_DATA_PER_TILE)
    InDataVec inData[HRepeat * WRepeat * CRepeat] = {};
    AccVec c_thread_buf_                          = {};
#endif
    WeiDataVec weiData[KRepeat * CRepeat] = {};

    // Data layout: HWC, unit: InDataType
    constexpr index_t data_H_stride = Width * InputChannels;
    constexpr index_t data_W_stride = InputChannels;
    constexpr index_t data_C_stride = wcnn_conv.GetNumInputChannels();

    // Filter layout: KYXC, unit: WeiDataType
    constexpr index_t weight_stride = InputChannels;

    // Acc layout: HWK, unit: AccDataType
    constexpr index_t acc_H_stride = Width * OutputChannels;
    constexpr index_t acc_W_stride = OutputChannels;
    constexpr index_t acc_K_stride = wcnn_conv.GetNumOutputChannels();

    // Load inData
    const index_t compIdx =
        lIdx * wcnn_conv.GetNumDataComponents() / wcnn_conv.GetNumImageSubTilesInVertical();

    const index_t accCompIdx =
        lIdx * wcnn_conv.GetNumAccumComponents() / wcnn_conv.GetNumSubTilesPerImageTile();

    auto load_in_data = [&](index_t h, index_t w, index_t c) {
        const index_t subC = compIdx % wcnn_conv.GetNumInputChannels();
        const index_t subW = (compIdx / wcnn_conv.GetNumInputChannels()) % WPerWcnn;
        const index_t subH = (compIdx / wcnn_conv.GetNumInputChannels()) / WPerWcnn;

        const index_t offset = (h * HPerWcnn + subH) * data_H_stride +
                               (w * WPerWcnn + subW) * data_W_stride + (c * data_C_stride + subC);
        if constexpr(wcnn_conv.GetNumImageSubTilesInVertical() > 1)
        {
            // shape 8x4
            typename decltype(wcnn_conv)::InDataVec inDataVec;

            static_for<0, wcnn_conv.GetNumImageSubTilesInVertical(), 1>{}([&](auto tileId) {
                inDataVec.template AsType<InDataTileVec>()(Number<tileId>{}) =
                    *reinterpret_cast<const InDataTileVec*>(
                        in + offset + tileId * DataTileHeight * data_H_stride);
            });
            return inDataVec.template AsType<InDataVec>()(Number<0>{});
        }
        else
        {
            return *reinterpret_cast<const InDataVec*>(in + offset);
        }
    };

    auto store_acc_data = [&](index_t h, index_t w, index_t k, AccDataVec& c_vec) {
        const index_t subW = (accCompIdx / wcnn_conv.GetNumOutputChannels()) % WPerWcnn;
        const index_t subH = (accCompIdx / wcnn_conv.GetNumOutputChannels()) / WPerWcnn;

        if constexpr(wcnn_conv.GetNumAccumComponents() == 4)
        {
            const index_t subK   = accCompIdx % wcnn_conv.GetNumOutputChannels();
            const index_t offset = (h * HPerWcnn + subH) * acc_H_stride +
                                   (w * WPerWcnn + subW) * acc_W_stride + (k * acc_K_stride + subK);
            *reinterpret_cast<typename AccDataVec::type*>(c_ + offset) =
                c_vec.template AsType<typename AccDataVec::type>()(Number<0>{});
        }
        else
        {
            static_assert(wcnn_conv.GetNumAccumComponents() == 8, "unexpected value");
            // ACO = 0, do swizzle after 4 channels.
            using AccSwizzleVec = typename vector_type<AccDataType, 4>::type;
            const index_t subK  = accCompIdx % wcnn_conv.GetNumOutputChannels() /
                                 (wcnn_conv.GetNumAccumComponents() * 2) *
                                 (wcnn_conv.GetNumAccumComponents() * 2);
            const index_t offset = (h * HPerWcnn + subH) * acc_H_stride +
                                   (w * WPerWcnn + subW) * acc_W_stride + (k * acc_K_stride + subK);
            index_t secOffset = 8;
            if constexpr(wcnn_conv.GetNumSubTilesPerImageTile() > 1)
            {
                secOffset = HPerWcnn / wcnn_conv.GetNumSubTilesPerImageTile() * acc_H_stride;
            }
            const index_t swizzleOffset = (lIdx & 1) * 4;
            *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset) =
                c_vec.template AsType<AccSwizzleVec>()(Number<0>{});
            *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + secOffset) =
                c_vec.template AsType<AccSwizzleVec>()(Number<1>{});
        }
    };

#if !defined(LOAD_DATA_PER_TILE)
    static_for<0, HRepeat, 1>{}([&](auto h) {
        static_for<0, WRepeat, 1>{}([&](auto w) {
            static_for<0, CRepeat, 1>{}([&](auto c) {
                constexpr index_t tileOffset = h * WRepeat * CRepeat + w * CRepeat + c;
                inData[tileOffset]           = load_in_data(h, w, c);
            });
        });
    });
#endif

    // Load weight
    constexpr index_t numWeightTile = wcnn_conv.GetWeightRegSize();
    const index_t weiCompIdx =
        lIdx / numWeightTile * numWeightTile * wcnn_conv.GetNumWeightComponents();
    static_for<0, KRepeat, 1>{}([&](auto k) {
        static_for<0, CRepeat, 1>{}([&](auto c) {
            constexpr index_t tileOffset = k * CRepeat + c;

            const index_t subC   = weiCompIdx % wcnn_conv.GetNumInputChannels();
            const index_t subK   = weiCompIdx / wcnn_conv.GetNumInputChannels();
            const index_t offset = (k * wcnn_conv.GetNumOutputChannels() + subK) * weight_stride +
                                   c * wcnn_conv.GetNumInputChannels() + subC;
            if constexpr(wcnn_conv.GetNumWeightComponents() * 32 >
                         wcnn_conv.GetNumInputChannels() * wcnn_conv.GetNumOutputChannels())
            {
                // only shape 8x4 consume 16 lanes
                static_assert(numWeightTile == 1, "");
                static_assert(HPerWcnn == 8 && WPerWcnn == 4, "unexpected shape!");
                if(weiCompIdx < wcnn_conv.GetNumInputChannels() * wcnn_conv.GetNumOutputChannels())
                {
                    weiData[tileOffset] = *reinterpret_cast<const WeiDataVec*>(wei + offset);
                }
            }
            else if constexpr(numWeightTile == 2)
            {
                // shape 4x2
                typename decltype(wcnn_conv)::WeiDataVec weiDataTmp;

                const index_t offsetBase =
                    offset + (lIdx % 2) * wcnn_conv.GetNumWeightComponents() / 2;
                weiDataTmp.template AsType<WeiDataTileVec>()(Number<0>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(wei + offsetBase);

                weiDataTmp.template AsType<WeiDataTileVec>()(Number<1>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(wei + offsetBase +
                                                             wcnn_conv.GetNumWeightComponents());

                weiData[tileOffset] = weiDataTmp.template AsType<WeiDataVec>()(Number<0>{});
            }
            else
            {
                // shape 4x4
                static_assert(numWeightTile == 1, "");
                weiData[tileOffset] = *reinterpret_cast<const WeiDataVec*>(wei + offset);
            }
        });
    });

    __syncthreads();

    // Do convolution
    static_for<0, KRepeat, 1>{}([&](auto k) {
        static_for<0, HRepeat, 1>{}([&](auto h) {
            static_for<0, WRepeat, 1>{}([&](auto w) {
#ifdef LOAD_DATA_PER_TILE
                AccVec c_thread_buf_ = {};
                auto& c_vec          = c_thread_buf_.GetVectorTypeReference(Number<0>{});
#else
                constexpr index_t tileOffset = h * WRepeat * KRepeat + w * KRepeat + k;
                auto& c_vec                  = c_thread_buf_.GetVectorTypeReference(
                    Number<tileOffset * wcnn_conv.GetNumAccumComponents()>{});
#endif
                static_for<0, CRepeat, 1>{}([&](auto c) {
#ifdef LOAD_DATA_PER_TILE
                    InDataVec in_tile_data = load_in_data(h, w, c);
#else
                    InDataVec& in_tile_data = inData[h * WRepeat * CRepeat + w * CRepeat + c];
#endif
                    const InDataVec* p_in_tile_data[1] = {&in_tile_data};
                    if constexpr(UseF32I32 && c + 1 == CRepeat)
                    {
                        wcnn_conv.conv_instr.Run(
                            weiData[k * CRepeat + c], p_in_tile_data, c_vec, Number<2>{});
                    }
                    else
                    {
                        wcnn_conv.conv_instr.Run(
                            weiData[k * CRepeat + c], p_in_tile_data, c_vec, Number<0>{});
                    }
                });
#ifdef LOAD_DATA_PER_TILE
                store_acc_data(h, w, k, c_vec);
#endif
            });
        });
    });

    __syncthreads();

#if !defined(LOAD_DATA_PER_TILE)
    // Output accum data
    static_for<0, HRepeat, 1>{}([&](auto h) {
        static_for<0, WRepeat, 1>{}([&](auto w) {
            static_for<0, KRepeat, 1>{}([&](auto k) {
                constexpr index_t tileOffset = h * WRepeat * KRepeat + w * KRepeat + k;
                auto& c_vec                  = c_thread_buf_.GetVectorTypeReference(
                    Number<tileOffset * wcnn_conv.GetNumAccumComponents()>{});
                store_acc_data(h, w, k, c_vec);
            });
        });
    });
#endif
}

template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          bool UseF32I32,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Width,
          index_t Height,
          index_t UnpackedInputChannels,
          index_t UnpackedOutputChannels>
__global__ void __launch_bounds__(64, 1)
    conv3_fwd(const InDataType* in_, const WeiDataType* wei_, AccDataType* c_)
{
    constexpr index_t DataTileHeight = 4;

    // conv3 always LOAD_DATA_PER_TILE. so marco LOAD_DATA_PER_TILE is ignored.
    constexpr auto wcnn_conv = ck::WcnnConv<WeiDataType,
                                            InDataType,
                                            AccDataType,
                                            HPerWcnn,
                                            WPerWcnn,
                                            FilterSize,
                                            DilationX,
                                            DilationY,
                                            1,
                                            false,
                                            false,
                                            UseF32I32>();

    auto in  = reinterpret_cast<const typename decltype(wcnn_conv)::KernelInDataType*>(in_);
    auto wei = reinterpret_cast<const typename decltype(wcnn_conv)::KernelWeightDataType*>(wei_);

    static_assert(Width % WPerWcnn == 0, "");
    static_assert(Height % HPerWcnn == 0, "");
    static_assert(UnpackedInputChannels % wcnn_conv.GetUnpackedNumInputChannels() == 0, "");
    static_assert(UnpackedOutputChannels % wcnn_conv.GetUnpackedNumOutputChannels() == 0, "");

    constexpr index_t InputChannels = UnpackedInputChannels * wcnn_conv.GetNumInputChannels() /
                                      wcnn_conv.GetUnpackedNumInputChannels();
    constexpr index_t OutputChannels = UnpackedOutputChannels * wcnn_conv.GetNumOutputChannels() /
                                       wcnn_conv.GetUnpackedNumOutputChannels();
    constexpr index_t HRepeat = Height / HPerWcnn;
    constexpr index_t WRepeat = Width / WPerWcnn;
    constexpr index_t CRepeat = UnpackedInputChannels / wcnn_conv.GetUnpackedNumInputChannels();
    constexpr index_t KRepeat = UnpackedOutputChannels / wcnn_conv.GetUnpackedNumOutputChannels();

    using InDataVec     = typename decltype(wcnn_conv)::InDataVec::type;
    using InDataTileVec = typename decltype(wcnn_conv)::InDataTileVec::type;
    using WeiDataVec    = typename decltype(wcnn_conv)::WeiDataVec::type;

    using AccDataVec = typename decltype(wcnn_conv)::AccDataVec;
    using AccVec     = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                                 AccDataType,
                                                 1,
                                                 wcnn_conv.GetNumAccumComponents(),
                                                 true>;
    const int lIdx   = threadIdx.x;

    // Data layout: HWC, unit: InDataType
    constexpr index_t data_H_stride = Width * InputChannels;
    constexpr index_t data_W_stride = InputChannels;
    constexpr index_t data_C_stride = wcnn_conv.GetNumInputChannels();

    // Filter layout: KYXC, unit: WeiDataType
    constexpr index_t weight_X_stride = InputChannels;
    constexpr index_t weight_Y_stride = FilterSize * InputChannels;
    constexpr index_t weight_K_stride = FilterSize * FilterSize * InputChannels;

    // Acc layout: HWK, unit: AccDataType
    constexpr index_t acc_H_stride = Width * OutputChannels;
    constexpr index_t acc_W_stride = OutputChannels;
    constexpr index_t acc_K_stride = wcnn_conv.GetNumOutputChannels();

    // Load inData
    const index_t compIdx =
        lIdx * wcnn_conv.GetNumDataComponents() / wcnn_conv.GetNumImageSubTilesInVertical();

    auto load_in_data = [&](index_t h, index_t w, index_t c, index_t tileOffset) {
        const index_t subC = compIdx % wcnn_conv.GetNumInputChannels();
        const index_t subW = (compIdx / wcnn_conv.GetNumInputChannels()) % WPerWcnn;
        const index_t subH = (compIdx / wcnn_conv.GetNumInputChannels()) / WPerWcnn;

        const index_t offset = (h * HPerWcnn - DataTileHeight + subH) * data_H_stride +
                               (w * WPerWcnn + tileOffset * WPerWcnn + subW) * data_W_stride +
                               (c * data_C_stride + subC);

        static_assert(wcnn_conv.GetNumImageSubTilesInVertical() > 1);

        typename decltype(wcnn_conv)::InDataVec inData;

        static_for<0, wcnn_conv.GetNumImageSubTilesInVertical(), 1>{}([&](auto tileId) {
            const index_t tileH = h * HPerWcnn - DataTileHeight + tileId * DataTileHeight;
            const index_t tileW = w * WPerWcnn + tileOffset * WPerWcnn;

            if((tileH >= 0) && (tileH + DataTileHeight <= Height) && (tileW >= 0) &&
               (tileW + WPerWcnn <= Width))
            {
                inData.template AsType<InDataTileVec>()(Number<tileId>{}) =
                    *reinterpret_cast<const InDataTileVec*>(
                        in + offset + tileId * DataTileHeight * data_H_stride);
            }
            else
            {
                inData.template AsType<InDataTileVec>()(Number<tileId>{}) = {};
            }
        });
        return inData.template AsType<InDataVec>()(Number<0>{});
    };

    constexpr index_t num_acc_tile = wcnn_conv.GetNumSubTilesPerImageTile();
    const index_t accCompIdx       = lIdx * wcnn_conv.GetNumAccumComponents() / num_acc_tile;
    auto store_acc_data            = [&](index_t h, index_t w, index_t k, AccDataVec& c_vec) {
        const index_t subW = (accCompIdx / wcnn_conv.GetNumOutputChannels()) % WPerWcnn;
        const index_t subH = (accCompIdx / wcnn_conv.GetNumOutputChannels()) / WPerWcnn;

        if constexpr(wcnn_conv.GetNumAccumComponents() == 4)
        {
            const index_t subK   = accCompIdx % wcnn_conv.GetNumOutputChannels();
            const index_t offset = (h * HPerWcnn + subH) * acc_H_stride +
                                   (w * WPerWcnn + subW) * acc_W_stride + (k * acc_K_stride + subK);
            *reinterpret_cast<typename AccDataVec::type*>(c_ + offset) =
                c_vec.template AsType<typename AccDataVec::type>()(Number<0>{});
        }
        else
        {
            static_assert(wcnn_conv.GetNumAccumComponents() == 8, "unexpected value");
            using acc_swizzle_vec = typename vector_type<AccDataType, 4>::type;
            const index_t subK    = accCompIdx % wcnn_conv.GetNumOutputChannels() /
                                 (wcnn_conv.GetNumAccumComponents() * 2) *
                                 (wcnn_conv.GetNumAccumComponents() * 2);
            const index_t offset = (h * HPerWcnn + subH) * acc_H_stride +
                                   (w * WPerWcnn + subW) * acc_W_stride + (k * acc_K_stride + subK);

            constexpr index_t secOffset =
                (num_acc_tile > 1) ? HPerWcnn / num_acc_tile * acc_H_stride : 8;
            const index_t swizzleOffset = (lIdx % 2) * 4;
            *reinterpret_cast<acc_swizzle_vec*>(c_ + offset + swizzleOffset) =
                c_vec.template AsType<acc_swizzle_vec>()(Number<0>{});
            *reinterpret_cast<acc_swizzle_vec*>(c_ + offset + swizzleOffset + secOffset) =
                c_vec.template AsType<acc_swizzle_vec>()(Number<1>{});
        }
    };

    // Load weight
    constexpr index_t numWeightTile = wcnn_conv.GetWeightRegSize();
    using WeiDataTileVec            = typename decltype(wcnn_conv)::WeiDataTileVec::type;

    auto load_weight = [&](index_t k, index_t c) {
        constexpr index_t tapeIdx[]                      = {4, 1, 0, 3, 6, 7, 8, 5, 2};
        typename decltype(wcnn_conv)::WeiDataVec weiData = {};
        if constexpr(HPerWcnn == 8)
        {
            // shape 8 x 4
            const index_t weiCompIdx =
                (lIdx % 16) * wcnn_conv.GetNumWeightComponents() / numWeightTile;
            const index_t subC = weiCompIdx % wcnn_conv.GetNumInputChannels();
            const index_t subK = weiCompIdx / wcnn_conv.GetNumInputChannels();

            if(lIdx < 16)
            {
                static_for<0, 5, 1>{}([&](auto id) {
                    constexpr index_t tapeId = 2 * id;
                    constexpr index_t subX   = tapeIdx[tapeId] % 3;
                    constexpr index_t subY   = tapeIdx[tapeId] / 3;
                    const index_t offset =
                        (k * wcnn_conv.GetNumOutputChannels() + subK) * weight_K_stride +
                        subY * weight_Y_stride + subX * weight_X_stride + subC +
                        c * wcnn_conv.GetNumInputChannels();

                    weiData.template AsType<WeiDataTileVec>()(Number<tapeId / 2>{}) =
                        *reinterpret_cast<const WeiDataTileVec*>(wei + offset);
                });
            }
            else
            {
                static_for<0, 4, 1>{}([&](auto id) {
                    constexpr index_t tapeId = 2 * id + 1;
                    constexpr index_t subX   = tapeIdx[tapeId] % 3;
                    constexpr index_t subY   = tapeIdx[tapeId] / 3;
                    const index_t offset =
                        (k * wcnn_conv.GetNumOutputChannels() + subK) * weight_K_stride +
                        subY * weight_Y_stride + subX * weight_X_stride + subC +
                        c * wcnn_conv.GetNumInputChannels();

                    weiData.template AsType<WeiDataTileVec>()(Number<tapeId / 2>{}) =
                        *reinterpret_cast<const WeiDataTileVec*>(wei + offset);
                });
            }
        }
        else if constexpr(WPerWcnn == 2)
        {
            // shape 4 x2
            static_assert(wcnn_conv.GetNumWeightComponents() % numWeightTile == 0, "");
            const index_t weiCompIdx = lIdx / 2 * 2 * wcnn_conv.GetNumWeightComponents() / 9;
            const index_t subC       = weiCompIdx % wcnn_conv.GetNumInputChannels();
            const index_t subK       = weiCompIdx / wcnn_conv.GetNumInputChannels();

            static_for<0, 9, 1>{}([&](auto tapeId) {
                constexpr index_t subX = tapeIdx[tapeId] % 3;
                constexpr index_t subY = tapeIdx[tapeId] / 3;
                const index_t offset =
                    (k * wcnn_conv.GetNumOutputChannels() + subK) * weight_K_stride +
                    subY * weight_Y_stride + subX * weight_X_stride + subC +
                    c * wcnn_conv.GetNumInputChannels() +
                    (lIdx % 2) * wcnn_conv.GetNumWeightComponents() / 9 / 2;

                weiData.template AsType<WeiDataTileVec>()(Number<tapeId * 2>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(wei + offset);
                weiData.template AsType<WeiDataTileVec>()(Number<tapeId * 2 + 1>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(
                        wei + offset + wcnn_conv.GetNumWeightComponents() / numWeightTile * 2);
            });
        }
        else
        {
            const index_t weiCompIdx = lIdx * wcnn_conv.GetNumWeightComponents() / numWeightTile;
            const index_t subC       = weiCompIdx % wcnn_conv.GetNumInputChannels();
            const index_t subK       = weiCompIdx / wcnn_conv.GetNumInputChannels();

            static_for<0, 9, 1>{}([&](auto tapeId) {
                constexpr index_t subX = tapeIdx[tapeId] % 3;
                constexpr index_t subY = tapeIdx[tapeId] / 3;
                const index_t offset =
                    (k * wcnn_conv.GetNumOutputChannels() + subK) * weight_K_stride +
                    subY * weight_Y_stride + subX * weight_X_stride + subC +
                    c * wcnn_conv.GetNumInputChannels();

                weiData.template AsType<WeiDataTileVec>()(Number<tapeId>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(wei + offset);
            });
        }

        return weiData.template AsType<WeiDataVec>()(Number<0>{});
    };

    __syncthreads();

    // Do convolution
    static_for<0, KRepeat, 1>{}([&](auto k) {
        WeiDataVec weiData[CRepeat];
        static_for<0, CRepeat, 1>{}([&](auto c) { weiData[c] = load_weight(k, c); });
        static_for<0, HRepeat, 1>{}([&](auto h) {
            static_for<0, WRepeat, 1>{}([&](auto w) {
                AccVec c_thread_buf_ = {};
                auto& c_vec          = c_thread_buf_.GetVectorTypeReference(Number<0>{});
                static_for<0, CRepeat, 1>{}([&](auto c) {
                    InDataVec inData[3];
                    inData[0]                         = load_in_data(h, w, c, -1);
                    inData[1]                         = load_in_data(h, w, c, 0);
                    inData[2]                         = load_in_data(h, w, c, 1);
                    const InDataVec* p_in_data_vec[3] = {&inData[0], &inData[1], &inData[2]};
                    if constexpr(UseF32I32 && c + 1 == CRepeat)
                    {
                        wcnn_conv.conv_instr.Run(weiData[c], p_in_data_vec, c_vec, Number<2>{});
                    }
                    else
                    {
                        wcnn_conv.conv_instr.Run(weiData[c], p_in_data_vec, c_vec, Number<0>{});
                    }
                });
                store_acc_data(h, w, k, c_vec);
            });
        });
    });
}
#pragma clang diagnostic pop
}; // namespace conv_op_util
}; // namespace ck

template <typename InDataType,
          typename WeiDataType,
          typename GPUAccType,
          bool UseF32I32,
          ShapeType Shape,
          FilterType Filter,
          bool Dilation,
          int32_t TestMask>
bool run_test()
{
    if((config.test_mask & 0xFFFF0000 & TestMask) == 0)
    {
        return true;
    }
    constexpr ck::index_t FilterSize   = (Filter == Filter_1X1) ? 1 : 3;
    constexpr ck::index_t HPerWcnn     = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWcnn     = (Shape == Shape_4X2) ? 2 : 4;
    constexpr ck::index_t DilationSize = Dilation ? 2 : 1;

#ifdef USE_ABSOLUTE_SIZE
    constexpr bool CheckVgpr1024         = (TestMask & 0xC000) != 0;
    constexpr ck::index_t Width          = CheckVgpr1024 ? DEFAULT_W_1K : DEFAULT_W;
    constexpr ck::index_t Height         = CheckVgpr1024 ? DEFAULT_H_1K : DEFAULT_H;
    constexpr ck::index_t InputChannels  = DEFAULT_C;
    constexpr ck::index_t OutputChannels = DEFAULT_K;
#else
    constexpr auto wcnn_conv     = ck::WcnnConv<WeiDataType,
                                                InDataType,
                                                GPUAccType,
                                                HPerWcnn,
                                                WPerWcnn,
                                                FilterSize,
                                                DilationSize,
                                                DilationSize,
                                                1,
                                                false,
                                                false,
                                                UseF32I32>();
    constexpr ck::index_t Width  = WPerWcnn * DEFAULT_W_REPEAT;
    constexpr ck::index_t Height = HPerWcnn * DEFAULT_H_REPEAT;
    constexpr ck::index_t InputChannels =
        wcnn_conv.GetUnpackedNumInputChannels() * DEFAULT_C_REPEAT;
    constexpr ck::index_t OutputChannels =
        wcnn_conv.GetUnpackedNumOutputChannels() * DEFAULT_K_REPEAT;
#endif
    constexpr ck::index_t n_dim          = 2;
    constexpr ck::index_t group_count    = 1;
    constexpr ck::index_t n_batch        = 1;
    constexpr ck::index_t n_out_channels = OutputChannels;
    constexpr ck::index_t n_in_channels  = InputChannels;

    const std::vector<ck::index_t> filters_1x1{1, 1};
    const std::vector<ck::index_t> filters_3x3{3, 3};
    const std::vector<ck::index_t> dilations_1{1, 1};
    const std::vector<ck::index_t> dilations_2{2, 2};
    const std::vector<ck::index_t> pads_0{0, 0};
    const std::vector<ck::index_t> pads_1{1, 1};
    const std::vector<ck::index_t> pads_2{2, 2};

    const std::vector<ck::index_t>& filters_len =
        (Filter == Filter_1X1) ? filters_1x1 : filters_3x3;
    const std::vector<ck::index_t> input_len = {Height, Width};
    const std::vector<ck::index_t> strides{1, 1};
    const std::vector<ck::index_t>& dilations  = Dilation ? dilations_2 : dilations_1;
    const std::vector<ck::index_t>& left_pads  = (Filter == Filter_1X1) ? pads_0
                                                 : Dilation             ? pads_2
                                                                        : pads_1;
    const std::vector<ck::index_t>& right_pads = (Filter == Filter_1X1) ? pads_0
                                                 : Dilation             ? pads_2
                                                                        : pads_1;

    ck::utils::conv::ConvParam conv_param{n_dim,
                                          group_count,
                                          n_batch,
                                          n_out_channels,
                                          n_in_channels,
                                          filters_len,
                                          input_len,
                                          strides,
                                          dilations,
                                          left_pads,
                                          right_pads};

    constexpr auto NDimSpatial = ck::Number<n_dim>{};
    const auto in_element_op   = InElementOp{};
    const auto wei_element_op  = WeiElementOp{};

    namespace ctc   = ck::tensor_layout::convolution;
    auto in_layout  = ctc::GNHWC{};
    auto wei_layout = ctc::GKYXC{};
    auto out_layout = ctc::GNHWK{};
    using InLayout  = decltype(in_layout);
    using WeiLayout = decltype(wei_layout);
    using OutLayout = decltype(out_layout);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<GPUAccType> out_host(out_g_n_k_wos_desc);
    Tensor<GPUAccType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        break;
    }

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                 InDataType,
                                                                 WeiDataType,
                                                                 GPUAccType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 PassThrough>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in,
                                              wei,
                                              out_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              in_element_op,
                                              wei_element_op,
                                              PassThrough{});

    if(config.do_verification)
    {
        ref_invoker.Run(ref_argument);
    }

    dump_tensor(in, "Input");
    dump_tensor(wei, "Weight");
    dump_tensor(out_host, "Accum");

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUAccType) * out_device.mDesc.GetElementSpaceSize());

    if constexpr(std::is_same<ck::int4_t, InDataType>::value)
    {
        std::vector<uint8_t> in_packed;
        std::vector<uint8_t> wei_packed;
        in_packed.resize(in.mData.size());
        wei_packed.resize(wei.mData.size());
        for(size_t i = 0; i < in.mData.size(); i += 2)
        {
            uint8_t val0 = (in.mData[i] & 0xf);
            uint8_t val1 = (in.mData[i + 1] & 0xf);

            in_packed[i / 2] = val0 | (val1 << 4);
        }
        for(size_t i = 0; i < wei.mData.size(); i += 2)
        {
            uint8_t val0      = (wei.mData[i] & 0xf);
            uint8_t val1      = (wei.mData[i + 1] & 0xf);
            wei_packed[i / 2] = val0 | (val1 << 4);
        }
        in_device_buf.ToDevice(in_packed.data());
        wei_device_buf.ToDevice(wei_packed.data());
    }
    else
    {
        in_device_buf.ToDevice(in.mData.data());
        wei_device_buf.ToDevice(wei.mData.data());
    }

    if constexpr(FilterSize == 1)
    {
        auto conv_kernel = ck::conv_op_util::conv_fwd<InDataType,
                                                      WeiDataType,
                                                      GPUAccType,
                                                      UseF32I32,
                                                      HPerWcnn,
                                                      WPerWcnn,
                                                      FilterSize,
                                                      DilationSize,
                                                      DilationSize,
                                                      Width,
                                                      Height,
                                                      InputChannels,
                                                      OutputChannels>;
        conv_kernel<<<1, 32>>>(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                               static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()));
    }
    else
    {
        auto conv_kernel = ck::conv_op_util::conv3_fwd<InDataType,
                                                       WeiDataType,
                                                       GPUAccType,
                                                       UseF32I32,
                                                       HPerWcnn,
                                                       WPerWcnn,
                                                       FilterSize,
                                                       DilationSize,
                                                       DilationSize,
                                                       Width,
                                                       Height,
                                                       InputChannels,
                                                       OutputChannels>;
        conv_kernel<<<1, 32>>>(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                               static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()));
    }

    out_device_buf.FromDevice(out_device.mData.data());

    dump_tensor(out_device, "Accum_Device");
    std::cout << "wcnn_conv_op<In/Wei:" << get_string<InDataType>()
              << ", Out:" << get_string<GPUAccType>() << ", " << get_string(Shape) << ", "
              << get_string(Filter) << ", Dilation:" << DilationSize << ", Id:0x" << std::hex
              << TestMask << ">: Status: ";

    if(config.do_verification)
    {
        bool ret = ck::utils::check_err(out_device,
                                        out_host,
                                        "Error: incorrect results!",
                                        get_rtol<GPUAccType>(),
                                        get_atol<GPUAccType>());
        if(ret)
        {
            std::cout << "Passed\n";
        }
        else
        {
            std::cout << "Failed\n";
        }

        return ret;
    }
    else
    {
        return true;
    }
}

template <typename InDataType,
          typename WeiDataType,
          typename GPUAccType,
          bool UseF32I32,
          int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;
    // clang-format off
    //                                                                  |ShapeType |FilterType |Dilation |TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_1X1, false, TestMask | 0x10000  >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_3X3, false, TestMask | 0x20000  >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_3X3, true,  TestMask | 0x40000  >();
    }
    else
    {
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_1X1, false, TestMask | 0x80000  >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X4, Filter_1X1, false, TestMask | 0x100000 >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_8X4, Filter_1X1, false, TestMask | 0x200000 >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_3X3, false, TestMask | 0x400000 >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X4, Filter_3X3, false, TestMask | 0x800000 >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_8X4, Filter_3X3, false, TestMask | 0x1000000>();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_3X3, true,  TestMask | 0x2000000>();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X4, Filter_3X3, true,  TestMask | 0x4000000>();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_8X4, Filter_3X3, true,  TestMask | 0x8000000>();
    }
    // clang-format on

    return pass;
}

int main(int argc, char* argv[])
{
    bool pass = true;
    if(parse_cmd_args_fix_size(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    //                  |InDataType  |WeiDataType |GPUAccType F32I32
    pass &= run_test_fmt<ck::half_t,  ck::half_t, float,       false, 0x1   >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t,float,       false, 0x2   >();
    pass &= run_test_fmt<ck::f8_t,    ck::f8_t,   float,       false, 0x4   >();
    pass &= run_test_fmt<ck::bf8_t,   ck::f8_t,   float,       false, 0x4   >();
    pass &= run_test_fmt<ck::bf8_t,   ck::bf8_t,  float,       false, 0x8   >();
    pass &= run_test_fmt<ck::f8_t,    ck::bf8_t,  float,       false, 0x8   >();
    pass &= run_test_fmt<int8_t,      int8_t,     float,       false, 0x10  >();
    pass &= run_test_fmt<int8_t,      int8_t,     float,       true,  0x10  >();
    pass &= run_test_fmt<int8_t,      int8_t,     int32_t,     false, 0x20  >();

    pass &= run_test_fmt<ck::half_t,  ck::half_t, ck::half_t,  false, 0x40  >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t,ck::bhalf_t, false, 0x80  >();
    pass &= run_test_fmt<ck::f8_t,    ck::f8_t,   ck::half_t,  false, 0x100 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::f8_t,   ck::half_t,  false, 0x100 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::bf8_t,  ck::half_t,  false, 0x200 >();
    pass &= run_test_fmt<ck::f8_t,    ck::bf8_t,  ck::half_t,  false, 0x200 >();
    pass &= run_test_fmt<int8_t,      int8_t,     ck::half_t,  false, 0x400 >();
    pass &= run_test_fmt<ck::half_t,  ck::half_t, float,       false, 0x4000>();
    pass &= run_test_fmt<ck::half_t,  ck::half_t, ck::half_t,  false, 0x8000>();
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    pass &= run_test_fmt<ck::int4_t,  ck::int4_t, float,       false, 0x800 >();
    pass &= run_test_fmt<ck::int4_t,  ck::int4_t, float,       true,  0x800 >();
    pass &= run_test_fmt<ck::int4_t,  ck::int4_t, int32_t,     false, 0x1000>();
    pass &= run_test_fmt<ck::int4_t,  ck::int4_t, ck::half_t,  false, 0x2000>();
#endif
    // clang-format on

    std::cout << "wcnn_conv_op: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
