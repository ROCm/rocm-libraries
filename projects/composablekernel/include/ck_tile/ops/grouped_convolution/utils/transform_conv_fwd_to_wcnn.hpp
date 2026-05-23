// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"

namespace ck_tile {

template <index_t NDimSpatial,
          ConvolutionSpecialization ConvSpecialization,
          index_t VectorSizeA,
          index_t VectorSizeB,
          index_t VectorSizeC,
          typename IndexType = index_t>
struct TransformConvFwdToHWCWcnn
{
    private:
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};
    static constexpr auto I5 = number<5>{};

    public:
    CK_TILE_HOST constexpr TransformConvFwdToHWCWcnn() {}
    // ======================== 1D constructor ========================
    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST TransformConvFwdToHWCWcnn(const ConvDimsType& a_g_n_c_wis_lengths,
                                           const ConvDimsType& b_g_k_c_xs_lengths,
                                           const ConvDimsType& c_g_n_k_wos_lengths,
                                           const ConvSpatialDimsType& conv_filter_strides,
                                           const ConvSpatialDimsType& conv_filter_dilations,
                                           const ConvSpatialDimsType& input_left_pads,
                                           const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
          N_{c_g_n_k_wos_lengths[I1]},
          Hi_{I1},
          Wi_{a_g_n_c_wis_lengths[I3]},
          Ho_{I1},
          Wo_{c_g_n_k_wos_lengths[I3]},
          Y_{I1},
          X_{b_g_k_c_xs_lengths[I3]},
          K_{c_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          ConvStrideH_{I1},
          ConvStrideW_{conv_filter_strides[I0]},
          ConvDilationH_{I1},
          ConvDilationW_{conv_filter_dilations[I0]},
          InLeftPadH_{I0},
          InLeftPadW_{input_left_pads[I0]},
          InRightPadH_{I0},
          InRightPadW_{input_right_pads[I0]},
          YX_{X_}
    {
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);
    }

    // ======================== 2D constructor ========================
    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST TransformConvFwdToHWCWcnn(const ConvDimsType& a_g_n_c_wis_lengths,
                                           const ConvDimsType& b_g_k_c_xs_lengths,
                                           const ConvDimsType& c_g_n_k_wos_lengths,
                                           const ConvSpatialDimsType& conv_filter_strides,
                                           const ConvSpatialDimsType& conv_filter_dilations,
                                           const ConvSpatialDimsType& input_left_pads,
                                           const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
          N_{c_g_n_k_wos_lengths[I1]},
          Hi_{a_g_n_c_wis_lengths[I3]},
          Wi_{a_g_n_c_wis_lengths[I4]},
          Ho_{c_g_n_k_wos_lengths[I3]},
          Wo_{c_g_n_k_wos_lengths[I4]},
          Y_{b_g_k_c_xs_lengths[I3]},
          X_{b_g_k_c_xs_lengths[I4]},
          K_{c_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          ConvStrideH_{conv_filter_strides[I0]},
          ConvStrideW_{conv_filter_strides[I1]},
          ConvDilationH_{conv_filter_dilations[I0]},
          ConvDilationW_{conv_filter_dilations[I1]},
          InLeftPadH_{input_left_pads[I0]},
          InLeftPadW_{input_left_pads[I1]},
          InRightPadH_{input_right_pads[I0]},
          InRightPadW_{input_right_pads[I1]},
          YX_{Y_ * X_}
    {
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);
    }

    // ======================== Descriptor builders ========================

    // Input descriptor: H × W × C
    // H = N * Ho (1D: H = N), W = Wo (1D: W = Wo), C = C
    template <typename ALayout,
              typename std::enable_if<NDimSpatial == 1 &&
                                          std::is_same_v<ALayout, tensor_layout::convolution::NWGC>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_H_W_C() const
    {
        // NWGC: stride_N = W*G*C, stride_W = G*C, stride_C = 1
        const IndexType NStride  = Wi_ * G_ * C_;
        const IndexType WiStride = G_ * C_;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            // For 1x1s1p0, H = N, W = Wi (no tiling along H, all spatial dim handled by W)
            return make_naive_tensor_descriptor(make_tuple(N_, Wi_, C_),
                                                make_tuple(NStride, WiStride, I1),
                                                number<VectorSizeA>{},
                                                number<1>{});
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            const auto in_n_wi_c_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Wi_, C_),
                                             make_tuple(NStride, WiStride, I1),
                                             number<VectorSizeA>{},
                                             number<1>{});

            return transform_tensor_descriptor(
                in_n_wi_c_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
        }
        else
        {
            static_assert(0, "Unsupported convolution specialization for 1D convolution");
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 2 && std::is_same_v<ALayout, tensor_layout::convolution::NHWGC>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_H_W_C() const
    {
        // NHWGC: stride_N = Hi*Wi*G*C, stride_H = Wi*G*C, stride_W = G*C, stride_C = 1
        const IndexType NStride  = Hi_ * Wi_ * G_ * C_;
        const IndexType HiStride = Wi_ * G_ * C_;
        const IndexType WiStride = G_ * C_;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {

            const auto in_n_hi_wi_c_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Hi_, Wi_, C_),
                                             make_tuple(NStride, HiStride, WiStride, I1),
                                             number<VectorSizeA>{},
                                             number<1>{});

            // H = N * Ho, W = Wo
            return transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Hi_)),
                           make_pass_through_transform(Wi_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            const auto in_n_hi_wi_c_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Hi_, Wi_, C_),
                                             make_tuple(NStride, HiStride, WiStride, I1),
                                             number<VectorSizeA>{},
                                             number<1>{});

            const auto in_n_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                           make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            return transform_tensor_descriptor(
                in_n_ho_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_)),
                           make_pass_through_transform(Wo_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
        }
        else
        {
            static_assert(0, "Unsupported convolution specialization for 2D convolution");
        }
    }

    // Weight descriptor: K × YX × C
    template <typename BLayout,
              typename std::enable_if<NDimSpatial == 1 &&
                                          std::is_same_v<BLayout, tensor_layout::convolution::GKXC>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_K_YX_C() const
    {
        // GKXC: stride_K = X*C, stride_X = C, stride_C = 1

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(K_, X_, C_),
                                                make_tuple(YX_ * C_, C_, I1),
                                                number<VectorSizeB>{},
                                                number<1>{});
        }
        else
        {
            const auto b_k_x_c_desc = make_naive_tensor_descriptor(make_tuple(K_, X_, C_),
                                                                   make_tuple(YX_ * C_, C_, I1),
                                                                   number<VectorSizeB>{},
                                                                   number<1>{});
            return transform_tensor_descriptor(
                b_k_x_c_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_insert_transform(I0),
                           make_merge_transform(make_tuple(X_, C_))),
                make_tuple(sequence<0>{}, sequence<>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
        }
    }

    template <typename BLayout,
              typename std::enable_if<
                  NDimSpatial == 2 && std::is_same_v<BLayout, tensor_layout::convolution::GKYXC>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_K_YX_C() const
    {
        const index_t KStride  = YX_ * C_;
        const index_t YXStride = C_;
        const index_t CStride  = 1;
        return make_naive_tensor_descriptor(make_tuple(K_, YX_, C_),
                                            make_tuple(KStride, YXStride, CStride),
                                            number<VectorSizeB>{},
                                            number<1>{});
    }

    // Output descriptor: H × W × K
    template <typename CLayout,
              typename std::enable_if<NDimSpatial == 1 &&
                                          std::is_same_v<CLayout, tensor_layout::convolution::NWGK>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_H_W_K() const
    {
        // NWGK: stride_N = Wo*G*K, stride_W = G*K, stride_K = 1
        const IndexType NStride  = Wo_ * G_ * K_;
        const IndexType WoStride = G_ * K_;

        const auto out_n_wo_k_desc = make_naive_tensor_descriptor(make_tuple(N_, Wo_, K_),
                                                                  make_tuple(NStride, WoStride, I1),
                                                                  number<VectorSizeC>{},
                                                                  number<1>{});

        return out_n_wo_k_desc;
    }

    template <typename CLayout,
              typename std::enable_if<
                  NDimSpatial == 2 && std::is_same_v<CLayout, tensor_layout::convolution::NHWGK>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_H_W_K() const
    {
        // NHWGK: stride_N = Ho*Wo*G*K, stride_H = Wo*G*K, stride_W = G*K, stride_K = 1
        const IndexType NStride  = Ho_ * Wo_ * G_ * K_;
        const IndexType HoStride = Wo_ * G_ * K_;
        const IndexType WoStride = G_ * K_;

        const auto out_n_ho_wo_k_desc =
            make_naive_tensor_descriptor(make_tuple(N_, Ho_, Wo_, K_),
                                         make_tuple(NStride, HoStride, WoStride, I1),
                                         number<VectorSizeC>{},
                                         number<1>{});

        return transform_tensor_descriptor(
            out_n_ho_wo_k_desc,
            make_tuple(make_merge_transform(make_tuple(N_, Ho_)),
                       make_pass_through_transform(Wo_),
                       make_pass_through_transform(K_)),
            make_tuple(sequence<0, 1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
    }

    // ======================== Member variables ========================
    IndexType G_, N_;
    IndexType Hi_, Wi_;
    IndexType Ho_, Wo_;
    IndexType Y_, X_;
    IndexType K_, C_;
    IndexType ConvStrideH_, ConvStrideW_;
    IndexType ConvDilationH_, ConvDilationW_;
    IndexType InLeftPadH_, InLeftPadW_;
    IndexType InRightPadH_, InRightPadW_;
    IndexType YX_;
};

} // namespace ck_tile
