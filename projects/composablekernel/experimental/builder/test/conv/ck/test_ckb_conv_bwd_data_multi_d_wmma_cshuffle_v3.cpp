// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "gmock/gmock.h"
#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"
#include "ck_tile/builder/testing/conv/bwd_data_ck.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "testing_utils.hpp"

namespace {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

constexpr auto SIGNATURE =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::BACKWARD_DATA,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNHWK}}};

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle_V3{}
                               .with_thread_block(cku::ThreadBlock_64_32x32x32)
                               .with_gemm_config(cku::GemmParamsABK1_Wmma_16x16_2x1_per_wave)
                               .with_transfer(cku::BwdTransfer_4x16x1_4x16x1_v3)
                               .with_bwd_data_specialization(ckb::ConvSpecialization::DEFAULT)
                               .with_prefetch_config(1, ckb::PipelineScheduler::DEFAULT)
                               .with_gemm_pad_params(0, 0)
                               .with_block_gemm(cku::BlockGemmDesc_v1_intrawave)
                               .with_transpose_params(2, 2);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

using Reference = ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;

} // namespace

TEST(BwdData_2DFp16_MultiD_Wmma_CShuffle_V3_GNHWC, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    std::cout << "Expected Transfer Parameters: " << expected_transfer_parameters << std::endl;
    cku::run_test<Builder>({"DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3",
                            expected_transfer_parameters,
                            "Default",
                            "GNHWK,GKYXC,EmptyTuple,GNHWC",
                            "PassThrough,PassThrough,PassThrough",
                            "fp16,fp16"}); // check compute types
}

TEST(BwdData_2DFp16_MultiD_Wmma_CShuffle_V3_GNHWC, Exec)
{
    if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
    {
        // Note: See device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp
        // `IsSupportedArgument`
        GTEST_SKIP() << "unsupported architecture";
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = 2,
                .groups          = 4,
                .input_channels  = 128,
                .output_channels = 48,
                .image           = {.width = 128, .height = 32},
                .filter          = {.width = 3, .height = 3},
            },
        .filter_strides     = {.width = 1, .height = 1},
        .filter_dilation    = {.width = 1, .height = 1},
        .input_left_pad     = {.width = 0, .height = 0},
        .input_right_pad    = {.width = 0, .height = 0},
        .a_elementwise_op   = {},
        .b_elementwise_op   = {},
        .cde_elementwise_op = {},
    };

    auto inputs    = ckt::alloc_inputs(args);
    auto outputs   = ckt::alloc_outputs(args);
    auto reference = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    using namespace ck_tile::test;
    auto conv = Instance{};
    EXPECT_THAT(ckt::run(conv, args, inputs.get(), outputs.get()), SuccessfulRun());

    auto ref_conv = Reference{};
    EXPECT_THAT(ckt::run(ref_conv, args, inputs.get(), reference.get()), SuccessfulRun());

    EXPECT_THAT(outputs.get(), MatchesReference(args, reference.get()));
}
