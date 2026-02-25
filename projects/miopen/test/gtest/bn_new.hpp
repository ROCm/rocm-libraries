/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#pragma once

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/solver/ck_utility_common.hpp>
#include "gtest_common.hpp"

#include "bn_test_data.hpp"
#include "test_operations.hpp"

// MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_ENFORCE)
// #define WORKAROUND_SWDEV_547301 1

enum BNApiType
{
    testBNAPIV1,
    testBNAPIV2,
    testBNAPIV3,
    testBNAPIInvVar,
};

// Assuming miopenTensorLayout_t and testAPI_t are the types of your enums
static std::string LayoutToString(int tensor_format)
{
    switch(tensor_format)
    {
    case miopenTensorNCHW: return "NCHW";
    case miopenTensorNCDHW: return "NCDHW";
    case miopenTensorNHWC: return "NHWC";
    case miopenTensorNDHWC: return "NDHWC";
    default: return "UnknownTensorFormat";
    }
}

static std::string ApiVerisonToString(int api_version)
{
    switch(api_version)
    {
    case testBNAPIV1: return "testBNAPIV1";
    case testBNAPIV2: return "testBNAPIV2";
    case testBNAPIV3: return "testBNAPIV3";
    case testBNAPIInvVar: return "testBNAPIInvVar";
    default: return "UnknownAPIVersion";
    }
}

static std::string BNModeToString(int bn_mode)
{
    switch(bn_mode)
    {
    case miopenBNPerActivation: return "BNPerActivation";
    case miopenBNSpatial: return "BNSpatial";
    default: return "UnknownBNMode";
    }
}

// Custom test name generator to handle enums
template <typename TestCase>
struct TestNameGenerator
{
    std::string
    operator()(const testing::TestParamInfo<std::tuple<TestCase,
                                                       miopenTensorLayout_t,
                                                       miopenBatchNormMode_t,
                                                       BNApiType,
                                                       miopenActivationMode_t>>& info) const
    {
        constexpr int dimension = std::is_same<TestCase, BN2DTestCase>::value   ? 2
                                  : std::is_same<TestCase, BN3DTestCase>::value ? 3
                                                                                : -1;
        static_assert(dimension > 0);

        const auto& layout_type    = std::get<1>(info.param);
        const auto& batchnorm_mode = std::get<2>(info.param);
        const auto& api_type       = std::get<3>(info.param);

        std::string tensor_name  = LayoutToString(layout_type);
        std::string bn_mode_name = BNModeToString(batchnorm_mode);
        std::string api_name     = ApiVerisonToString(api_type);

        std::ostringstream oss;
        oss << tensor_name + "_" + bn_mode_name + "_" + api_name + "_Dim_" +
                   std::to_string(dimension) + "_test_id_" + std::to_string(info.index);
        return oss.str();
    }
};

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename RunSaveDataType,
          typename AccDataType,
          typename TestCase,
          typename TVerify,
          test::adaptive::UnitUnderTest UUT    = test::adaptive::UnitUnderTest::naiveGPU,
          test::adaptive::TestReference REF    = test::adaptive::TestReference::naiveCPU,
          test::adaptive::AfterTestFailure ATF = test::adaptive::AfterTestFailure::none,
          test::adaptive::VerifyOption VER     = test::adaptive::VerifyOption::validateAndRMS>
struct BNFwdTrainTestNew
    : public test::adaptive::AdaptiveTest<XDataType, TVerify, UUT, REF, ATF, VER>,
      public ::testing::TestWithParam<std::tuple<TestCase,
                                                 miopenTensorLayout_t,
                                                 miopenBatchNormMode_t,
                                                 BNApiType,
                                                 miopenActivationMode_t>>
{
protected:
    static void SetUpTestSuite()
    {
        if constexpr(!CheckTestConfiguration(UUT, REF))
        {
            GTEST_SKIP() << "Test configuration is incorrect";
        }
        test::adaptive::SetUpSharedVerifyData<XDataType, TVerify, UUT, REF, ATF, VER>();
    }

    static void TearDownTestSuite()
    {
        test::adaptive::TearDownSharedVerifyData<XDataType, TVerify, UUT, REF, ATF, VER>();
    }

    void SetUp() override
    {
        std::tie(bn_config, tensor_layout, bn_mode, api_type, bn_fwd_train_test_data.activ_mode) =
            this->GetParam();
        bn_fwd_train_test_data.SetUpImpl(bn_config, bn_mode, tensor_layout);

        bn_fwd_train_test_data.activ_alpha = 0.1;
        bn_fwd_train_test_data.activ_beta  = 0.3;

        std::fill(bn_fwd_train_test_data.output.begin(),
                  bn_fwd_train_test_data.output.end(),
                  std::numeric_limits<YDataType>::quiet_NaN());
        std::fill(bn_fwd_train_test_data.saveMean_ref.begin(),
                  bn_fwd_train_test_data.saveMean_ref.end(),
                  std::numeric_limits<YDataType>::quiet_NaN());
        std::fill(bn_fwd_train_test_data.saveVariance_ref.begin(),
                  bn_fwd_train_test_data.saveVariance_ref.end(),
                  std::numeric_limits<YDataType>::quiet_NaN());
    }

    miopenStatus_t RunOptimizedGPU() override { return miopenStatusNotImplemented; }

    miopenStatus_t RunNaiveGPU() override
    {
        auto&& handle      = get_handle();
        miopenStatus_t res = miopenStatusUnknownError;
        if(bn_fwd_train_test_data.activ_mode > 0)
        {
            miopenCreateActivationDescriptor(&activ_desc);
            miopenSetActivationDescriptor(activ_desc,
                                          bn_fwd_train_test_data.activ_mode,
                                          bn_fwd_train_test_data.activ_alpha,
                                          bn_fwd_train_test_data.activ_beta,
                                          0.0);
            res = miopenBatchNormForwardTrainingActivation(
                &handle,
                bn_mode,
                &bn_fwd_train_test_data.alpha,
                &bn_fwd_train_test_data.beta,
                &bn_fwd_train_test_data.input.desc,
                bn_fwd_train_test_data.in_dev.get(),
                &bn_fwd_train_test_data.output.desc,
                bn_fwd_train_test_data.out_dev.get(),
                &bn_fwd_train_test_data.scale.desc,
                &bn_fwd_train_test_data.shift.desc,
                &bn_fwd_train_test_data.saveMean.desc,
                &bn_fwd_train_test_data.saveVariance.desc,
                bn_fwd_train_test_data.scale_dev.get(),
                bn_fwd_train_test_data.shift_dev.get(),
                bn_fwd_train_test_data.averageFactor,
                bn_fwd_train_test_data.runMean_dev.get(),
                bn_fwd_train_test_data.runVariance_dev.get(),
                bn_fwd_train_test_data.epsilon,
                bn_fwd_train_test_data.saveMean_dev.get(),
                bn_fwd_train_test_data.saveVariance_dev.get(),
                activ_desc);
            miopenDestroyActivationDescriptor(activ_desc);
        }
        else
        {
            if(api_type == BNApiType::testBNAPIV1)
            {
                res = miopenBatchNormalizationForwardTraining(
                    &handle,
                    bn_mode,
                    &bn_fwd_train_test_data.alpha,
                    &bn_fwd_train_test_data.beta,
                    &bn_fwd_train_test_data.input.desc,
                    bn_fwd_train_test_data.in_dev.get(),
                    &bn_fwd_train_test_data.output.desc,
                    bn_fwd_train_test_data.out_dev.get(),
                    &bn_fwd_train_test_data.scale.desc,
                    bn_fwd_train_test_data.scale_dev.get(),
                    bn_fwd_train_test_data.shift_dev.get(),
                    bn_fwd_train_test_data.averageFactor,
                    bn_fwd_train_test_data.runMean_dev.get(),
                    bn_fwd_train_test_data.runVariance_dev.get(),
                    bn_fwd_train_test_data.epsilon,
                    bn_fwd_train_test_data.saveMean_dev.get(),
                    bn_fwd_train_test_data.saveVariance_dev.get());
            }
            else if(api_type == BNApiType::testBNAPIV2)
            {
                res = miopenBatchNormalizationForwardTraining_V2(
                    &handle,
                    bn_mode,
                    &bn_fwd_train_test_data.alpha,
                    &bn_fwd_train_test_data.beta,
                    &bn_fwd_train_test_data.input.desc,
                    bn_fwd_train_test_data.in_dev.get(),
                    &bn_fwd_train_test_data.output.desc,
                    bn_fwd_train_test_data.out_dev.get(),
                    &bn_fwd_train_test_data.scale.desc,
                    &bn_fwd_train_test_data.shift.desc,
                    &bn_fwd_train_test_data.saveMean.desc,
                    &bn_fwd_train_test_data.saveVariance.desc,
                    bn_fwd_train_test_data.scale_dev.get(),
                    bn_fwd_train_test_data.shift_dev.get(),
                    bn_fwd_train_test_data.averageFactor,
                    bn_fwd_train_test_data.runMean_dev.get(),
                    bn_fwd_train_test_data.runVariance_dev.get(),
                    bn_fwd_train_test_data.epsilon,
                    bn_fwd_train_test_data.saveMean_dev.get(),
                    bn_fwd_train_test_data.saveVariance_dev.get());
            }
            else if(api_type == BNApiType::testBNAPIV3)
            {
                res = miopenBatchNormalizationForwardTraining_V3(
                    &handle,
                    bn_mode,
                    &bn_fwd_train_test_data.alpha,
                    &bn_fwd_train_test_data.beta,
                    &bn_fwd_train_test_data.input.desc,
                    bn_fwd_train_test_data.in_dev.get(),
                    &bn_fwd_train_test_data.output.desc,
                    bn_fwd_train_test_data.out_dev.get(),
                    &bn_fwd_train_test_data.scale.desc,
                    &bn_fwd_train_test_data.shift.desc,
                    &bn_fwd_train_test_data.saveMean.desc,
                    &bn_fwd_train_test_data.saveVariance.desc,
                    bn_fwd_train_test_data.scale_dev.get(),
                    bn_fwd_train_test_data.shift_dev.get(),
                    bn_fwd_train_test_data.averageFactor,
                    bn_fwd_train_test_data.prevRunMean_dev.get(),
                    bn_fwd_train_test_data.prevRunVariance_dev.get(),
                    bn_fwd_train_test_data.nextRunMean_dev.get(),
                    bn_fwd_train_test_data.nextRunVariance_dev.get(),
                    bn_fwd_train_test_data.epsilon,
                    bn_fwd_train_test_data.saveMean_dev.get(),
                    bn_fwd_train_test_data.saveVariance_dev.get());
            }
        }

        // bn_fwd_train_test_data.output.data = handle.Read<YDataType>(
        //     bn_fwd_train_test_data.out_dev, bn_fwd_train_test_data.output.data.size());

        // bn_fwd_train_test_data.saveMean.data = handle.Read<RunSaveDataType>(
        //     bn_fwd_train_test_data.saveMean_dev, bn_fwd_train_test_data.saveMean.data.size());
        // bn_fwd_train_test_data.saveVariance.data =
        //     handle.Read<RunSaveDataType>(bn_fwd_train_test_data.saveVariance_dev,
        //                                  bn_fwd_train_test_data.saveVariance_ref.data.size());
        // bn_fwd_train_test_data.runMean.data = handle.Read<RunSaveDataType>(
        //     bn_fwd_train_test_data.runMean_dev, bn_fwd_train_test_data.runMean_ref.data.size());
        // bn_fwd_train_test_data.runVariance.data =
        //     handle.Read<RunSaveDataType>(bn_fwd_train_test_data.runVariance_dev,
        //                                  bn_fwd_train_test_data.runVariance_ref.data.size());

        // // V3 API: Read back next buffers for verification
        // if(api_type == BNApiType::testBNAPIV3)
        // {
        //     bn_fwd_train_test_data.nextRunMean.data =
        //         handle.Read<RunSaveDataType>(bn_fwd_train_test_data.nextRunMean_dev,
        //                                      bn_fwd_train_test_data.runMean_ref.data.size());
        //     bn_fwd_train_test_data.nextRunVariance.data =
        //         handle.Read<RunSaveDataType>(bn_fwd_train_test_data.nextRunVariance_dev,
        //                                      bn_fwd_train_test_data.runVariance_ref.data.size());
        // }

        return res;
    }

    miopenStatus_t RunOptimizedCPU() override { return miopenStatusNotImplemented; }

    miopenStatus_t RunNaiveCPU() override
    {
        auto&& handle = get_handle();
        test::ComputeCPUBNFwdTrain(bn_fwd_train_test_data);
        activationHostInfer(bn_fwd_train_test_data.activ_mode,
                            0.0,
                            bn_fwd_train_test_data.activ_beta,
                            bn_fwd_train_test_data.activ_alpha,
                            bn_fwd_train_test_data.out_ref.data,
                            bn_fwd_train_test_data.out_ref.data);

        output_ref_dev = handle.Write(bn_fwd_train_test_data.out_ref.data);
        // // auto&& handle  = get_handle();
        // auto output_data =
        //     handle.Read<AccDataType>(output_ref_dev, bn_fwd_train_test_data.out_ref.data.size());

        // for(int i = 0; i < output_data.size(); i++)
        // {
        //     if(output_data[i] != bn_fwd_train_test_data.out_ref.data[i])
        //     {
        //         std::cout << "ERROR: " << output_data[i] << " "
        //                   << bn_fwd_train_test_data.out_ref.data[i] << std::endl;
        //     }
        // }
        saveMean_ref_dev     = handle.Write(bn_fwd_train_test_data.saveMean_ref.data);
        saveVariance_ref_dev = handle.Write(bn_fwd_train_test_data.saveVariance_ref.data);

        if(api_type == BNApiType::testBNAPIV3)
        {
            nextRunMean_ref_dev     = handle.Write(bn_fwd_train_test_data.runMean_ref.data);
            nextRunVariance_ref_dev = handle.Write(bn_fwd_train_test_data.runVariance_ref.data);
        }
        else
        {
            runMean_ref_dev     = handle.Write(bn_fwd_train_test_data.runMean_ref.data);
            runVariance_ref_dev = handle.Write(bn_fwd_train_test_data.runVariance_ref.data);
        }
        return miopenStatusSuccess;
    }

    std::pair<bool, std::unordered_map<std::string, TVerify>> Verify() override
    {
        // cpu
        // 4e-3 is tolerance used by CK kernel.
        // test::CompareTensor<YDataType>(
        //     bn_fwd_train_test_data.output, bn_fwd_train_test_data.out_ref, 4e-3);
        // test::CompareTensor<RunSaveDataType>(
        //     bn_fwd_train_test_data.saveMean, bn_fwd_train_test_data.saveMean_ref, 4e-3);
        //  test::CompareTensor<RunSaveDataType>(
        //     bn_fwd_train_test_data.saveVariance, bn_fwd_train_test_data.saveVariance_ref, 4e-3);
        // // For V3 API, compare next buffers; for V1/V2, compare runMean/runVariance
        // if(api_type == BNApiType::testBNAPIV3)
        // {
        //     test::CompareTensor<RunSaveDataType>(
        //         bn_fwd_train_test_data.nextRunMean, bn_fwd_train_test_data.runMean_ref, 4e-3);
        //
        //         test::CompareTensor<RunSaveDataType>(bn_fwd_train_test_data.nextRunVariance,
        //                                              bn_fwd_train_test_data.runVariance_ref,
        //                                              4e-3);
        // }
        // else
        // {
        //      test::CompareTensor<RunSaveDataType>(
        //         bn_fwd_train_test_data.runMean, bn_fwd_train_test_data.runMean_ref, 4e-3);
        //     test::CompareTensor<RunSaveDataType>(
        //         bn_fwd_train_test_data.runVariance, bn_fwd_train_test_data.runVariance_ref,
        //         4e-3);
        // }

        std::pair<bool, std::unordered_map<std::string, TVerify>> res = {false, {}};

        auto [res_out, error_out] = this->template VerifyOnGPU<YDataType, AccDataType>(
            bn_fwd_train_test_data.out_dev,
            output_ref_dev,
            bn_fwd_train_test_data.out_ref.data.size());
        EXPECT_FALSE(res_out.all_zeros_uut);
        EXPECT_FALSE(res_out.all_zeros_ref);
        EXPECT_TRUE(res_out.all_finite_and_non_nan_uut);
        EXPECT_TRUE(res_out.all_finite_and_non_nan_ref);
        EXPECT_TRUE(error_out < 4e-3);

        if(error_out < 4e-3)
        {
            res.first = true;
            res.second.insert({"output", error_out});
        }

        // auto [res_saveMean, error_saveMean] =
        //     this->VerifyOnGPU(bn_fwd_train_test_data.saveMean_dev,
        //                       saveMean_ref_dev,
        //                       bn_fwd_train_test_data.saveMean_ref.data.size());
        // EXPECT_FALSE(res_saveMean.all_zeros_uut);
        // EXPECT_FALSE(res_saveMean.all_zeros_ref);
        // EXPECT_TRUE(res_saveMean.all_finite_and_non_nan_uut);
        // EXPECT_TRUE(res_saveMean.all_finite_and_non_nan_ref);
        // EXPECT_TRUE(error_saveMean < 4e-3);

        // if(error_saveMean < 4e-3)
        // {
        //     res.first = true;
        //     res.second.insert({"saveMean", error_saveMean});
        // }

        // auto [res_saveVariance, error_saveVariance] =
        //     this->VerifyOnGPU(bn_fwd_train_test_data.saveVariance_dev,
        //                       saveVariance_ref_dev,
        //                       bn_fwd_train_test_data.saveVariance_ref.data.size());
        // EXPECT_FALSE(res_saveVariance.all_zeros_uut);
        // EXPECT_FALSE(res_saveVariance.all_zeros_ref);
        // EXPECT_TRUE(res_saveVariance.all_finite_and_non_nan_uut);
        // EXPECT_TRUE(res_saveVariance.all_finite_and_non_nan_ref);
        // EXPECT_TRUE(error_saveVariance < 4e-3);

        // if(error_saveVariance < 4e-3)
        // {
        //     res.first = true;
        //     res.second.insert({"saveVariance", error_saveVariance});
        // }

        // if(api_type == BNApiType::testBNAPIV3)
        // {
        //     auto [res_nextRunMean, error_nextRunMean] =
        //         this->VerifyOnGPU(bn_fwd_train_test_data.nextRunMean_dev,
        //                           nextRunMean_ref_dev,
        //                           bn_fwd_train_test_data.runMean_ref.data.size());
        //     EXPECT_FALSE(res_nextRunMean.all_zeros_uut);
        //     EXPECT_FALSE(res_nextRunMean.all_zeros_ref);
        //     EXPECT_TRUE(res_nextRunMean.all_finite_and_non_nan_uut);
        //     EXPECT_TRUE(res_nextRunMean.all_finite_and_non_nan_ref);
        //     EXPECT_TRUE(error_nextRunMean < 4e-3);

        //     if(error_nextRunMean < 4e-3)
        //     {
        //         res.first = true;
        //         res.second.insert({"nextRunMean", error_nextRunMean});
        //     }

        //     auto [res_nextRunVariance, error_nextRunVariance] =
        //         this->VerifyOnGPU(bn_fwd_train_test_data.nextRunVariance_dev,
        //                           nextRunVariance_ref_dev,
        //                           bn_fwd_train_test_data.runVariance_ref.data.size());
        //     EXPECT_FALSE(res_nextRunVariance.all_zeros_uut);
        //     EXPECT_FALSE(res_nextRunVariance.all_zeros_ref);
        //     EXPECT_TRUE(res_nextRunVariance.all_finite_and_non_nan_uut);
        //     EXPECT_TRUE(res_nextRunVariance.all_finite_and_non_nan_ref);
        //     EXPECT_TRUE(error_nextRunVariance < 4e-3);

        //     if(error_nextRunVariance < 4e-3)
        //     {
        //         res.first = true;
        //         res.second.insert({"nextRunVariance", error_nextRunVariance});
        //     }
        // }
        // else
        // {
        //     auto [res_RunMean, error_RunMean] =
        //         this->VerifyOnGPU(bn_fwd_train_test_data.runMean_dev,
        //                           runMean_ref_dev,
        //                           bn_fwd_train_test_data.runMean_ref.data.size());
        //     EXPECT_FALSE(res_RunMean.all_zeros_uut);
        //     EXPECT_FALSE(res_RunMean.all_zeros_ref);
        //     EXPECT_TRUE(res_RunMean.all_finite_and_non_nan_uut);
        //     EXPECT_TRUE(res_RunMean.all_finite_and_non_nan_ref);
        //     EXPECT_TRUE(error_RunMean < 4e-3);

        //     if(error_RunMean < 4e-3)
        //     {
        //         res.first = true;
        //         res.second.insert({"runMean", error_RunMean});
        //     }

        //     auto [res_runVariance, error_runVariance] =
        //         this->VerifyOnGPU(bn_fwd_train_test_data.runVariance_dev,
        //                           runVariance_ref_dev,
        //                           bn_fwd_train_test_data.runVariance_ref.data.size());
        //     EXPECT_FALSE(res_runVariance.all_zeros_uut);
        //     EXPECT_FALSE(res_runVariance.all_zeros_ref);
        //     EXPECT_TRUE(res_runVariance.all_finite_and_non_nan_uut);
        //     EXPECT_TRUE(res_runVariance.all_finite_and_non_nan_ref);
        //     EXPECT_TRUE(error_runVariance < 4e-3);

        //     if(error_runVariance < 4e-3)
        //     {
        //         res.first = true;
        //         res.second.insert({"runVariance", error_runVariance});
        //     }
        // }

        return res;
    }

    TestCase bn_config;
    bool test_skipped = false;
    BNFwdTrainTestData<XDataType,
                       YDataType,
                       ScaleDataType,
                       BiasDataType,
                       RunSaveDataType,
                       AccDataType,
                       TestCase>
        bn_fwd_train_test_data;

    miopen::Allocator::ManageDataPtr output_ref_dev;
    miopen::Allocator::ManageDataPtr saveMean_ref_dev;
    miopen::Allocator::ManageDataPtr saveVariance_ref_dev;
    miopen::Allocator::ManageDataPtr nextRunMean_ref_dev;
    miopen::Allocator::ManageDataPtr nextRunVariance_ref_dev;
    miopen::Allocator::ManageDataPtr runMean_ref_dev;
    miopen::Allocator::ManageDataPtr runVariance_ref_dev;

    miopenTensorLayout_t tensor_layout;
    miopenBatchNormMode_t bn_mode;
    BNApiType api_type;
    miopenActivationDescriptor_t activ_desc;

public:
    void Run() { this->RunAdaptiveTest(); }
};
