// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"

TEST(TestConvBwdWeightTransform, Fused2DMatchesLegacy)
{
    using namespace ck;
    static_assert(ConvBwdWeightImplicitGemmInTransform::GetNumOfLowerDimension() == 4);
    using Specialization = tensor_operation::device::ConvolutionBackwardWeightSpecialization;
    using Fused          = tensor_operation::
        TransformConvBwdWeightToGemmV2<2, 16, 16, 8, 4, 1, Specialization::Default>;
    using Legacy = tensor_operation::
        TransformConvBwdWeightToGemmV2<2, 16, 16, 8, 4, 1, Specialization::Default, long_index_t>;

    for(index_t variant = 0; variant < 3; ++variant)
    {
        const index_t stride_h   = variant + 1;
        const index_t stride_w   = 2;
        const index_t dilation_h = variant == 2 ? 2 : 1;
        const index_t dilation_w = variant == 1 ? 2 : 1;
        const index_t left_h     = variant;
        const index_t left_w     = variant == 0 ? 0 : 1;
        const index_t right_h    = 1;
        const index_t right_w    = 2;
        const index_t output_h   = (7 + left_h + right_h - 2 * dilation_h - 1) / stride_h + 1;
        const index_t output_w   = (9 + left_w + right_w - dilation_w - 1) / stride_w + 1;

        for(index_t batch_k : {1, 2, 5})
            for(bool split_offset : {false, true})
                for(bool full_batch : {false, true})
                {
                    SCOPED_TRACE(::testing::Message() << variant << ',' << batch_k << ','
                                                      << split_offset << ',' << full_batch);
                    const auto make_descriptors = [&](auto index_type) {
                        using Index = decltype(index_type);
                        using Transform =
                            std::conditional_t<std::is_same_v<Index, index_t>, Fused, Legacy>;
                        return Transform::template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                            2>(2,
                               5,
                               3,
                               std::array<Index, 2>{7, 9},
                               std::array<Index, 2>{3, 2},
                               std::array<Index, 2>{output_h, output_w},
                               std::array<Index, 5>{10000, 1000, 2, 100, 7},
                               std::array<Index, 5>{90, 18, 1, 6, 3},
                               std::array<Index, 5>{
                                   10000, output_h * output_w * 5, 1, output_w * 5, 5},
                               std::array<Index, 2>{stride_h, stride_w},
                               std::array<Index, 2>{dilation_h, dilation_w},
                               std::array<Index, 2>{left_h, left_w},
                               std::array<Index, 2>{right_h, right_w},
                               batch_k,
                               split_offset,
                               full_batch);
                    };
                    const auto fused  = make_descriptors(index_t{})[Number<1>{}];
                    const auto legacy = make_descriptors(long_index_t{})[Number<1>{}];
                    static_for<0, 3, 1>{}([&](auto dimension) {
                        ASSERT_EQ(fused.GetLength(dimension), legacy.GetLength(dimension));
                    });

                    for(index_t k0 = 0; k0 < fused.GetLength(Number<0>{}); ++k0)
                        for(index_t column = 0; column < fused.GetLength(Number<1>{}); ++column)
                            for(index_t k1 = 0; k1 < fused.GetLength(Number<2>{}); ++k1)
                            {
                                const auto upper    = make_multi_index(k0, column, k1);
                                const auto actual   = make_tensor_coordinate(fused, upper);
                                const auto expected = make_tensor_coordinate(legacy, upper);
                                const bool valid    = coordinate_has_valid_offset(legacy, expected);
                                ASSERT_EQ(coordinate_has_valid_offset(fused, actual), valid);
                                if(valid)
                                    ASSERT_EQ(actual.GetOffset(), expected.GetOffset());

                                auto moved = actual;
                                const auto forward =
                                    make_tensor_coordinate_step(fused, make_multi_index(1, 0, 0));
                                move_tensor_coordinate(fused, moved, forward);
                                const auto next = make_tensor_coordinate(
                                    fused, make_multi_index(k0 + 1, column, k1));
                                ASSERT_EQ(moved.GetOffset(), next.GetOffset());
                                ASSERT_EQ(coordinate_has_valid_offset(fused, moved),
                                          coordinate_has_valid_offset(fused, next));
                                const auto backward =
                                    make_tensor_coordinate_step(fused, make_multi_index(-1, 0, 0));
                                move_tensor_coordinate(fused, moved, backward);
                                ASSERT_EQ(moved.GetOffset(), actual.GetOffset());
                            }
                }
    }
}

namespace {

class TestConvUtil : public ::testing::Test
{
    public:
    void SetNDParams(std::size_t ndims, std::size_t s, std::size_t d, std::size_t p)
    {
        conv_params = ck::utils::conv::ConvParam(ndims,
                                                 2,
                                                 128,
                                                 192,
                                                 256,
                                                 std::vector<ck::long_index_t>(ndims, 3),
                                                 std::vector<ck::long_index_t>(ndims, 71),
                                                 std::vector<ck::long_index_t>(ndims, s),
                                                 std::vector<ck::long_index_t>(ndims, d),
                                                 std::vector<ck::long_index_t>(ndims, p),
                                                 std::vector<ck::long_index_t>(ndims, p));
    }

    protected:
    // -------  default 2D -------
    // input GNCHW {2, 128, 192, 71, 71},
    // weights GKCYX {2, 256, 192, 3, 3},
    // stride {s, s},
    // dilations {d, d},
    // padding {{p, p}, {p, p}
    ck::utils::conv::ConvParam conv_params;
};

} // namespace

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths1D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(1, 2, 1, 1);
    std::vector<ck::long_index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::long_index_t>{36}, "Error: ConvParams 1D."));

    // stride 1, dilation 1, pad 1
    SetNDParams(1, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::long_index_t>{71}, "Error: ConvParams 1D stride {1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(1, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{37},
                                     "Error: ConvParams 1D padding left/right {2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(1, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::long_index_t>{36}, "Error: ConvParams 1D dilation {2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(1, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::long_index_t>{23},
                             "Error: ConvParams 1D strides{3}, padding {1}, dilations {2}."));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths2D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(2, 2, 1, 1);
    std::vector<ck::long_index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{36, 36},
                                     "Error: ConvParams 2D default constructor."));

    // stride 1, dilation 1, pad 1
    SetNDParams(2, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{71, 71},
                                     "Error: ConvParams 2D stride {1,1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(2, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{37, 37},
                                     "Error: ConvParams 2D padding left/right {2,2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(2, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{36, 36},
                                     "Error: ConvParams 2D dilation {2,2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(2, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::long_index_t>{23, 23},
                             "Error: ConvParams 2D strides{3,3}, padding {1,1}, dilations {2,2}."));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths3D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(3, 2, 1, 1);
    std::vector<ck::long_index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::long_index_t>{36, 36, 36}, "Error: ConvParams 3D."));

    // stride 1, dilation 1, pad 1
    SetNDParams(3, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{71, 71, 71},
                                     "Error: ConvParams 3D stride {1, 1, 1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(3, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{37, 37, 37},
                                     "Error: ConvParams 3D padding left/right {2, 2, 2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(3, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{36, 36, 36},
                                     "Error: ConvParams 3D dilation {2, 2, 2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(3, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len,
        std::vector<ck::long_index_t>{23, 23, 23},
        "Error: ConvParams 3D strides{3, 3, 3}, padding {1, 1, 1}, dilations {2, 2, 2}."));
}
