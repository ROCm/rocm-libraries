// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cctype>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <half/half.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/miopen.h>

#include "../conv_common.hpp"
#include "../fusionHost.hpp"
#include "conv_test_base.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "tensor_util.hpp"

namespace {

using float16 = half_float::half;

struct CbnaParamNameGenerator
{
    template <typename ParamType>
    std::string operator()(const testing::TestParamInfo<ParamType>& info) const
    {
        std::string name = testing::PrintToString(info.param);
        std::transform(name.begin(), name.end(), name.begin(), [](const char c) {
            return std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
        });
        if(name.empty())
            name = "param";
        return "case_" + std::to_string(info.index) + "_" + name;
    }
};

using ptr_FusionPlanDesc = MIOPEN_MANAGE_PTR(miopenFusionPlanDescriptor_t, miopenDestroyFusionPlan);
using ptr_FusionPlanArgs = MIOPEN_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);

ptr_FusionPlanDesc GetManagedFusionPlanDesc(miopenTensorDescriptor_t inputDesc)
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
    return ptr_FusionPlanDesc{fusePlanDesc};
}

ptr_FusionPlanArgs GetManageFusionPlanArgs()
{
    miopenOperatorArgs_t fusionArgs;
    miopenCreateOperatorArgs(&fusionArgs);
    return ptr_FusionPlanArgs{fusionArgs};
}

template <class T>
struct verify_forward_cbna
{
    tensor<T> input;
    tensor<T> weights;
    miopenConvolutionDescriptor_t filter;
    tensor<T> bias{};
    miopenTensorDescriptor_t inputDesc{};
    miopenTensorDescriptor_t weightsDesc{};
    miopenTensorDescriptor_t outputDesc{};
    miopenTensorDescriptor_t biasDesc{};
    tensor<T> bnscale{};
    tensor<T> bnbias{};
    tensor<T> estMean{};
    tensor<T> estVariance{};
    miopenFusionPlanDescriptor_t fusionplan;
    miopenBatchNormMode_t bnmode;
    double epsilon;
    size_t workspace_size = 0;

    verify_forward_cbna(miopenFusionPlanDescriptor_t pfusionplan,
                        tensor<T>& pinput,
                        tensor<T>& pweights,
                        miopen::ConvolutionDescriptor& pfilter,
                        tensor<T>& pbias,
                        tensor<T>& pbnscale,
                        const tensor<T>& pbnbias,
                        const tensor<T>& pestMean,
                        const tensor<T>& pestVariance,
                        miopenBatchNormMode_t pbnmode,
                        size_t pworkspace_size)
    {
        input          = pinput;
        inputDesc      = &pinput.desc;
        weights        = pweights;
        weightsDesc    = &pweights.desc;
        bias           = pbias;
        biasDesc       = &pbias.desc;
        filter         = &pfilter;
        bnscale        = pbnscale;
        bnbias         = pbnbias;
        estMean        = pestMean;
        estVariance    = pestVariance;
        bnmode         = pbnmode;
        fusionplan     = pfusionplan;
        epsilon        = 1.0e-5;
        workspace_size = pworkspace_size;
    }

    tensor<T> cpu() const
    {
        auto rout = get_output_tensor(miopen::deref(filter), input, weights);
        auto bout = rout;
        auto aout = rout;
        std::fill(bout.begin(), bout.end(), 0.);
        std::fill(aout.begin(), aout.end(), 0.);

        convHostForward(input, rout, weights, 1, bias, filter);
        if(bnmode == miopenBNPerActivation)
        {
            batchNormPerActivHostInference(
                rout, bout, bnscale, bnbias, epsilon, estMean, estVariance);
        }
        else
        {
            batchNormSpatialHostInference(
                rout, bout, bnscale, bnbias, epsilon, estMean, estVariance);
        }
        activationHostInfer(miopenActivationRELU, 0.5, 0.5, 0.5, bout.data, aout.data);
        return aout;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rout     = get_output_tensor(miopen::deref(filter), input, weights);
        auto in_dev   = handle.Write(input.data);
        auto wei_dev  = handle.Write(weights.data);
        auto b_dev    = handle.Write(bias.data);
        auto out_dev  = handle.Write(rout.data);
        auto sc_dev   = handle.Write(bnscale.data);
        auto sh_dev   = handle.Write(bnbias.data);
        auto m_dev    = handle.Write(estMean.data);
        auto v_dev    = handle.Write(estVariance.data);
        auto workspace_dev =
            workspace_size > 0 ? handle.Write(std::vector<char>(workspace_size)) : nullptr;

        miopenFusionOpDescriptor_t convoOp = nullptr;
        miopenFusionOpDescriptor_t biasOp  = nullptr;
        miopenFusionOpDescriptor_t bnOp    = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;

        auto ptr_fusionargs = GetManageFusionPlanArgs();
        double alpha = 1.0, beta = 0.0;

        miopenFusionPlanGetOp(fusionplan, 0, &convoOp);
        miopenFusionPlanGetOp(fusionplan, 1, &biasOp);
        miopenFusionPlanGetOp(fusionplan, 2, &bnOp);
        miopenFusionPlanGetOp(fusionplan, 3, &activOp);

        miopenSetOpArgsConvForward(ptr_fusionargs.get(), convoOp, &alpha, &beta, wei_dev.get());
        miopenSetOpArgsBiasForward(ptr_fusionargs.get(), biasOp, &alpha, &beta, b_dev.get());
        miopenSetOpArgsBatchNormInference(ptr_fusionargs.get(),
                                          bnOp,
                                          &alpha,
                                          &beta,
                                          sc_dev.get(),
                                          sh_dev.get(),
                                          m_dev.get(),
                                          v_dev.get(),
                                          epsilon);
        miopenSetOpArgsActivForward(ptr_fusionargs.get(), activOp, &alpha, &beta, 0.5, 0.5, 0.5);

        miopenExecuteFusionPlan_v2(&handle,
                                   fusionplan,
                                   inputDesc,
                                   in_dev.get(),
                                   &rout.desc,
                                   out_dev.get(),
                                   ptr_fusionargs.get(),
                                   workspace_dev.get(),
                                   workspace_size);
        rout.data = handle.Read<T>(out_dev, rout.data.size());
        return rout;
    }
};

template <typename T, typename Verifier>
void VerifyAndValidate(const Verifier& verifier)
{
    auto cpu_result = verifier.cpu();
    auto gpu_result = verifier.gpu();

    EXPECT_EQ(miopen::range_distance(cpu_result), miopen::range_distance(gpu_result));

    using value_type       = T;
    const double tolerance = 80.0;
    const double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
    const double rms_error = miopen::rms_range(cpu_result, gpu_result);

    EXPECT_LE(rms_error, threshold)
        << "RMS error: " << rms_error << " exceeds threshold: " << threshold;
}

struct CbnaTestCase
{
    std::vector<int> input_dims;   // [N, C, H, W]
    std::vector<int> weights_dims; // [K, C, H, W]
    std::vector<int>
        pads_strides_dilations; // [pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w]
    bool bias_mode;
    std::string pad_mode;
    bool test_activ;
    int activ_mode;
    int batchnorm_mode;

    friend std::ostream& operator<<(std::ostream& os, const CbnaTestCase& tc)
    {
        return os << "input: [" << tc.input_dims[0] << "," << tc.input_dims[1] << ","
                  << tc.input_dims[2] << "," << tc.input_dims[3] << "] weights: ["
                  << tc.weights_dims[0] << "," << tc.weights_dims[1] << "," << tc.weights_dims[2]
                  << "," << tc.weights_dims[3] << "] bias:" << tc.bias_mode
                  << " pad_mode:" << tc.pad_mode << " activ:" << tc.test_activ;
    }
};

std::vector<CbnaTestCase> GetCbnaTestCases()
{
    std::vector<CbnaTestCase> result;
    auto input_shapes  = get_inputs<int>(0);
    auto weight_shapes = get_weights<int>(0);

    for(const auto& in : input_shapes)
    {
        for(auto stride : {1, 2})
        {
            int fpad_h = (stride == 2) ? 1 : 0;
            int fpad_w = (stride == 2) ? 1 : 0;

            for(const auto& wei : weight_shapes)
            {
                if(in[1] != wei[1])
                    continue;

                result.push_back({{1, in[1], in[2], in[3]},
                                  {wei[0], wei[1], wei[2], wei[3]},
                                  {fpad_h, fpad_w, stride, stride, 1, 1},
                                  true,
                                  "default",
                                  true,
                                  3,
                                  1});
            }
        }
    }
    return result;
}

template <typename T>
void RunCbnaInferenceTest(const CbnaTestCase& test_case)
{
    uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

    tensor<T> input{static_cast<size_t>(test_case.input_dims[0]),
                    static_cast<size_t>(test_case.input_dims[1]),
                    static_cast<size_t>(test_case.input_dims[2]),
                    static_cast<size_t>(test_case.input_dims[3])};
    input.generate(tensor_elem_gen_integer{max_value});

    tensor<T> weights{static_cast<size_t>(test_case.weights_dims[0]),
                      static_cast<size_t>(test_case.weights_dims[1]),
                      static_cast<size_t>(test_case.weights_dims[2]),
                      static_cast<size_t>(test_case.weights_dims[3])};
    weights.generate(tensor_elem_gen_integer{max_value});

    int input_c, input_h, input_w, wei_c, wei_k, wei_h, wei_w;
    std::tie(wei_k, wei_c, wei_h, wei_w)             = miopen::tien<4>(weights.desc.GetLengths());
    std::tie(std::ignore, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());
    EXPECT_EQ(input_c, wei_c) << "Aborting due to incorrect test config";
    if(input_c != wei_c)
    {
        return;
    }

    miopen::ConvolutionDescriptor filter;
    filter.mode         = miopenConvolution;
    filter.paddingMode  = miopenPaddingDefault;
    filter.pads[0]      = test_case.pads_strides_dilations[0];
    filter.pads[1]      = test_case.pads_strides_dilations[1];
    filter.strides[0]   = test_case.pads_strides_dilations[2];
    filter.strides[1]   = test_case.pads_strides_dilations[3];
    filter.dilations[0] = test_case.pads_strides_dilations[4];
    filter.dilations[1] = test_case.pads_strides_dilations[5];

    auto&& handle                      = get_handle();
    auto ptr_fusionplan                = GetManagedFusionPlanDesc(&input.desc);
    miopenFusionOpDescriptor_t convoOp = nullptr;
    miopenFusionOpDescriptor_t biasOp  = nullptr;
    miopenFusionOpDescriptor_t bnOp    = nullptr;
    miopenFusionOpDescriptor_t activOp = nullptr;

    miopenCreateOpConvForward(ptr_fusionplan.get(), &convoOp, &filter, &weights.desc);

    auto output = get_output_tensor(filter, input, weights);
    tensor<T> bias{1, output.desc.GetLengths()[1], 1, 1};
    bias.generate(tensor_elem_gen_integer{max_value});
    miopenCreateOpBiasForward(ptr_fusionplan.get(), &biasOp, &bias.desc);

    miopenBatchNormMode_t bnmode = miopenBNSpatial;
    auto derivedBnDesc           = miopen::TensorDescriptor{};
    miopen::DeriveBNTensorDescriptor(derivedBnDesc, output.desc, bnmode);
    tensor<T> bnscale{derivedBnDesc.GetLengths()};
    bnscale.generate(tensor_elem_gen_integer{max_value});
    tensor<T> bnbias{derivedBnDesc.GetLengths()};
    bnbias.generate(tensor_elem_gen_integer{max_value});
    tensor<T> estMean{derivedBnDesc.GetLengths()};
    estMean.generate(tensor_elem_gen_integer{max_value});
    tensor<T> estVariance{derivedBnDesc.GetLengths()};
    estVariance.generate(tensor_elem_gen_integer{max_value});

    miopenCreateOpBatchNormInference(ptr_fusionplan.get(), &bnOp, bnmode, &bnscale.desc);
    miopenCreateOpActivationForward(ptr_fusionplan.get(), &activOp, miopenActivationRELU);

    // A few basic dimension checks that we expect to be caught.
    const bool valid_test_case = wei_h > 2 * filter.pads[0] && wei_w > 2 * filter.pads[1] &&
                                 input_h >= (2 * filter.pads[0] + wei_h) &&
                                 input_w >= (2 * filter.pads[1] + wei_w);

    miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, ptr_fusionplan.get());

    size_t workspace_size = 0;
    if(miopenError == miopenStatusSuccess)
    {
        miopenFusionPlanGetWorkSpaceSize(
            &handle, ptr_fusionplan.get(), &workspace_size, miopenConvolutionFwdAlgoImplicitGEMM);
    }

    if(miopenError != miopenStatusSuccess)
    {
        std::stringstream ss;
        ss << "CBNA Inference plan not supported for: " << test_case;
        GTEST_SKIP() << ss.str();
    }
    else if(valid_test_case)
    {
        VerifyAndValidate<T>(verify_forward_cbna<T>{ptr_fusionplan.get(),
                                                    input,
                                                    weights,
                                                    filter,
                                                    bias,
                                                    bnscale,
                                                    bnbias,
                                                    estMean,
                                                    estVariance,
                                                    bnmode,
                                                    workspace_size});
    }
    else
    {
        GTEST_SKIP() << "Test case dimensions do not meet requirements";
    }
}

class GPU_CbnaInference_FP32 : public testing::TestWithParam<CbnaTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

} // namespace

TEST_P(GPU_CbnaInference_FP32, FloatTest_cbna_inference)
{
    RunCbnaInferenceTest<float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CbnaInference_FP32,
                         testing::ValuesIn(GetCbnaTestCases()),
                         CbnaParamNameGenerator{});
