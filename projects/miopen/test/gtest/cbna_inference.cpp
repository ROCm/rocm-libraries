// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/fusion.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/stringutils.hpp>
#include <half/half.hpp>
#include <vector>
#include <limits>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "tensor_util.hpp"
#include "conv_test_base.hpp"
#include "../fusionHost.hpp"

namespace {

using float16 = half_float::half;

using CbnaTestParam = std::tuple<miopenActivationMode_t,
                                 ConvTestCaseBase,
                                 miopenBatchNormMode_t,
                                 miopenTensorLayout_t,
                                 double,
                                 double,
                                 double>;

template <typename T>
class GPU_CbnaInference : public testing::TestWithParam<CbnaTestParam>
{
protected:
    void SetUp() override
    {
        prng::reset_seed();
        if(!IsTestSupportedByDevice(Gpu::All))
        {
            GTEST_SKIP() << "Test not supported on this device";
        }

        const auto& [activ_mode, conv_config, bn_mode, layout, alpha, beta, gamma] = GetParam();
        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        auto in_dims = conv_config.GetInput();
        input = tensor<T>{layout, in_dims};
        input.generate(tensor_elem_gen_integer{max_value});

        auto wei_dims = conv_config.GetWeights();
        weights = tensor<T>{layout, wei_dims};
        weights.generate(tensor_elem_gen_integer{max_value});

        filter = conv_config.GetConv();
        
        output = tensor<T>{layout, filter.GetForwardOutputTensor(input.desc, weights.desc).GetLengths()};
        
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, output.desc, bn_mode);
        auto bn_lens = derivedBnDesc.GetLengths();
        bnscale = tensor<T>{bn_lens}; bnscale.generate(tensor_elem_gen_integer{max_value});
        bnbias  = tensor<T>{bn_lens}; bnbias.generate(tensor_elem_gen_integer{max_value});
        estMean = tensor<T>{bn_lens}; estMean.generate(tensor_elem_gen_integer{max_value});
        estVariance = tensor<T>{bn_lens}; estVariance.generate(tensor_elem_gen_integer{max_value});

        bias = tensor<T>{1, output.desc.GetLengths()[1], 1, 1};
        bias.generate(tensor_elem_gen_integer{max_value});

        auto&& handle = get_handle();
        fusePlanDesc = miopen::FusionPlanDescriptor(miopenVerticalFusion, input.desc);
        
        auto convOp = std::make_shared<miopen::ConvForwardOpDescriptor>(filter, weights.desc);
        EXPECT_EQ(fusePlanDesc.AddOp(convOp), miopenStatusSuccess);
        convOp->SetArgs(params, &alpha_val, &beta_val, handle.Write(weights.data).get());

        auto biasOp = std::make_shared<miopen::BiasFusionOpDescriptor>(bias.desc);
        EXPECT_EQ(fusePlanDesc.AddOp(biasOp), miopenStatusSuccess);
        biasOp->SetArgs(params, &alpha_val, &beta_val, handle.Write(bias.data).get());

        auto bnOp = std::make_shared<miopen::BatchNormInferenceFusionOpDescriptor>(bn_mode, derivedBnDesc);
        EXPECT_EQ(fusePlanDesc.AddOp(bnOp), miopenStatusSuccess);
        bnOp->SetArgs(params, &alpha_val, &beta_val, handle.Write(bnscale.data).get(), handle.Write(bnbias.data).get(), handle.Write(estMean.data).get(), handle.Write(estVariance.data).get(), epsilon);

        auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activ_mode);
        EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
        activOp->SetArgs(params, &alpha_val, &beta_val, alpha, beta, gamma);

        if(fusePlanDesc.Compile(handle) != miopenStatusSuccess)
        {
            GTEST_SKIP() << "Fusion plan compilation failed";
        }
    }

    void RunTest()
    {
        const auto& [activ_mode, conv_config, bn_mode, layout, alpha, beta, gamma] = GetParam();
        auto&& handle = get_handle();
        
        // GPU execution
        auto in_dev = handle.Write(input.data);
        auto out_dev = handle.Write(output.data);
        
        auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
            params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
        
        fusePlanDesc.Execute(handle, *plan_params);
        output.data = handle.Read<T>(out_dev, output.data.size());

        // CPU execution
        auto cpu_rout = tensor<T>{output.desc};
        auto cpu_bout = tensor<T>{output.desc};
        auto cpu_aout = tensor<T>{output.desc};

        convHostForward(input, cpu_rout, weights, true, bias, &filter);
        
        if(bn_mode == miopenBNPerActivation)
        {
            batchNormPerActivHostInference(cpu_rout, cpu_bout, bnscale, bnbias, epsilon, estMean, estVariance);
        }
        else
        {
            batchNormSpatialHostInference(cpu_rout, cpu_bout, bnscale, bnbias, epsilon, estMean, estVariance);
        }

        activationHostInfer(activ_mode, gamma, beta, alpha, cpu_bout.data, cpu_aout.data);

        // Comparison
        ASSERT_EQ(miopen::range_distance(cpu_aout), miopen::range_distance(output));
        const double tolerance = 80.0;
        const double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        const double rms_error = miopen::rms_range(cpu_aout, output);

        EXPECT_LE(rms_error, threshold) << "RMS error: " << rms_error << " exceeds threshold: " << threshold;
    }

    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> bias;
    tensor<T> bnscale;
    tensor<T> bnbias;
    tensor<T> estMean;
    tensor<T> estVariance;
    miopen::ConvolutionDescriptor filter;
    miopen::TensorDescriptor derivedBnDesc;
    miopen::FusionPlanDescriptor fusePlanDesc;
    miopen::OperatorArgs params;
    const float alpha_val = 1.0f;
    const float beta_val  = 0.0f;
    const double epsilon = 1.0e-5;
};

using GPU_CbnaInference_FP32 = GPU_CbnaInference<float>;
using GPU_CbnaInference_FP16 = GPU_CbnaInference<half_float::half>;

TEST_P(GPU_CbnaInference_FP32, FloatTest) { RunTest(); }
TEST_P(GPU_CbnaInference_FP16, HalfTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CbnaInference_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenBNSpatial),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.5),
                                          testing::Values(0.5),
                                          testing::Values(0.5)));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CbnaInference_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenBNSpatial),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.5),
                                          testing::Values(0.5),
                                          testing::Values(0.5)));

} // namespace
