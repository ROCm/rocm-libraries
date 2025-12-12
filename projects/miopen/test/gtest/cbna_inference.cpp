// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../conv_common.hpp"
#include "../fusionHost.hpp"
#include <miopen/stringutils.hpp>
#include <half/half.hpp>
#include <vector>
#include <limits>

namespace {

using ptr_FusionPlanDesc = MIOPEN_MANAGE_PTR(miopenFusionPlanDescriptor_t, miopenDestroyFusionPlan);
using ptr_FusionPlanArgs = MIOPEN_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);
using ptr_ActivationDesc = MIOPEN_MANAGE_PTR(miopenActivationDescriptor_t,
                                             miopenDestroyActivationDescriptor);
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

ptr_ActivationDesc GetManagedActivDesc()
{
    miopenActivationDescriptor_t activdesc;
    miopenCreateActivationDescriptor(&activdesc);
    return ptr_ActivationDesc{activdesc};
}

template <class T>
struct verify_forward_conv_bias_batchnorm_activ
{
    tensor<T> input;
    tensor<T> weights;
    miopenConvolutionDescriptor_t filter;
    tensor<T> bias{};
    miopenTensorDescriptor_t inputDesc{};
    miopenTensorDescriptor_t weightsDesc{};
    miopenTensorDescriptor_t outputDesc{};
    miopenTensorDescriptor_t biasDesc{};
    miopenActivationDescriptor_t activDesc{};
    miopenTensorDescriptor_t biasScaleTensor{};
    tensor<T> bnscale{};
    tensor<T> bnbias{};
    tensor<T> estMean{};
    tensor<T> estVariance{};
    miopenFusionPlanDescriptor_t fusionplan;

    miopenBatchNormMode_t bnmode;
    bool bias_mode = false;
    bool doactive  = false;
    double epsilon;

    // using conv_base<T>::search; //DLOWELL not needed right now
    verify_forward_conv_bias_batchnorm_activ(miopenFusionPlanDescriptor_t pfusionplan,
                                             tensor<T>& pinput,
                                             tensor<T>& pweights,
                                             miopen::ConvolutionDescriptor& pfilter,
                                             bool pbias_mode,
                                             tensor<T>& pbias,
                                             miopenActivationDescriptor_t pactivDesc,
                                             bool pdoactiv,
                                             tensor<T>& pbnscale,
                                             const tensor<T>& pbnbias,
                                             const tensor<T>& pestMean,
                                             const tensor<T>& pestVariance,
                                             miopenBatchNormMode_t pbnmode)
    {
        input           = pinput;
        inputDesc       = &pinput.desc;
        weights         = pweights;
        weightsDesc     = &pweights.desc;
        bias            = pbias;
        biasDesc        = &pbias.desc;
        filter          = &pfilter;
        activDesc       = pactivDesc;
        doactive        = pdoactiv;
        bias_mode       = pbias_mode;
        biasScaleTensor = &pbnscale.desc;
        bnscale         = pbnscale;
        bnbias          = pbnbias;
        estMean         = pestMean;
        estVariance     = pestVariance;
        bnmode          = pbnmode;
        fusionplan      = pfusionplan;
        epsilon         = 1.0e-5;
    }

    tensor<T> cpu() const
    {

        auto rout = get_output_tensor(miopen::deref(filter), input, weights);
        auto aout = rout;
        std::fill(aout.begin(), aout.end(), 0.);
        auto bout = rout;
        std::fill(bout.begin(), bout.end(), 0.);

        // If we are using convolutions as the base, we can calculate the
        convHostForward(input, rout, weights, bias_mode, bias, filter);
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
        if(doactive)
        {
            double activ_alpha, activ_beta, activ_gamma;
            miopenActivationMode_t activ_mode;
            miopenGetActivationDescriptor(
                activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
            activationHostInfer(
                activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
        }
        else
        {
            return bout;
        }
        return aout;
    }

    tensor<T> gpu() const
    {
        auto&& handle        = get_handle();
        auto rout            = get_output_tensor(miopen::deref(filter), input, weights);
        auto in_dev          = handle.Write(input.data);
        auto wei_dev         = handle.Write(weights.data);
        auto b_dev           = handle.Write(bias.data);
        auto out_dev         = handle.Write(rout.data);
        auto bnscale_dev     = handle.Write(bnscale.data);
        auto bnbias_dev      = handle.Write(bnbias.data);
        auto estMean_dev     = handle.Write(estMean.data);
        auto estVariance_dev = handle.Write(estVariance.data);

        miopenFusionOpDescriptor_t convoOp = nullptr;
        miopenFusionOpDescriptor_t biasOp  = nullptr;
        miopenFusionOpDescriptor_t bNormOp = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;
        auto ptr_fusionargs                = GetManageFusionPlanArgs();

        double alpha = 1., beta = 0.;
        auto opcounter             = 0;
        miopenStatus_t miopenError = miopenFusionPlanGetOp(fusionplan, opcounter++, &convoOp);
        EXPECT_EQ(miopenError, miopenStatusSuccess);
        miopenSetOpArgsConvForward(ptr_fusionargs.get(), convoOp, &alpha, &beta, wei_dev.get());

        if(bias_mode)
        {
            miopenError = miopenFusionPlanGetOp(fusionplan, opcounter++, &biasOp);
            EXPECT_EQ(miopenError, miopenStatusSuccess);
            miopenSetOpArgsBiasForward(ptr_fusionargs.get(), biasOp, &alpha, &beta, b_dev.get());
        }

        miopenError = miopenFusionPlanGetOp(fusionplan, opcounter++, &bNormOp);
        EXPECT_EQ(miopenError, miopenStatusSuccess);
        miopenSetOpArgsBatchNormInference(ptr_fusionargs.get(),
                                          bNormOp,
                                          &alpha,
                                          &beta,
                                          bnscale_dev.get(),
                                          bnbias_dev.get(),
                                          estMean_dev.get(),
                                          estVariance_dev.get(),
                                          epsilon);
        if(doactive)
        {
            miopenError = miopenFusionPlanGetOp(fusionplan, opcounter, &activOp);
            EXPECT_EQ(miopenError, miopenStatusSuccess);
            double activ_alpha, activ_beta, activ_gamma;
            miopenActivationMode_t activ_mode;
            miopenGetActivationDescriptor(
                activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
            miopenSetOpArgsActivForward(
                ptr_fusionargs.get(), activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
        }
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                inputDesc,
                                in_dev.get(),
                                &rout.desc,
                                out_dev.get(),
                                ptr_fusionargs.get());
        rout.data = handle.Read<T>(out_dev, rout.data.size());
        return rout;
    }

    void fail(float = 0) const
    {
        if(bias_mode)
        {
            if(doactive)
            {
                std::cerr << "Conv+Bias+BatchNorm+Activation Inference:" << std::endl;
            }
            else
            {
                std::cerr << "Conv+Bias+BatchNorm Inference:" << std::endl;
            }
        }
        else
        {
            if(doactive)
            {
                std::cerr << "Conv+BatchNorm+Activation Inference:" << std::endl;
            }
            else
            {
                std::cerr << "Conv+BatchNorm Inference:" << std::endl;
            }
        }
    }
};

struct CbnaTestCase
{
    std::vector<int> input_dims;        // [N, C, H, W]
    std::vector<int> weights_dims;      // [K, C, H, W]
    std::vector<int> pads_strides_dilations; // [pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w]
    bool bias_mode;
    std::string pad_mode;
    bool test_activ;
    int activ_mode; // 0-9 mapping to activation modes
    int batchnorm_mode; // 0 = per-activation, 1 = spatial
    double alpha;
    double beta;
    double gamma;

    friend std::ostream& operator<<(std::ostream& os, const CbnaTestCase& tc)
    {
        return os << "input: [" << tc.input_dims[0] << "," << tc.input_dims[1] << ","
                  << tc.input_dims[2] << "," << tc.input_dims[3] << "] weights: ["
                  << tc.weights_dims[0] << "," << tc.weights_dims[1] << "," << tc.weights_dims[2]
                  << "," << tc.weights_dims[3] << "] bias:" << tc.bias_mode
                  << " pad_mode:" << tc.pad_mode << " activ:" << tc.test_activ
                  << " bn_mode:" << tc.batchnorm_mode;
    }
};

template <typename T>
std::vector<CbnaTestCase> GetCbnaTestCases()
{
    return {
        // input_dims, weights_dims, pads_strides_dilations, bias_mode, pad_mode, test_activ, activ_mode, batchnorm_mode, alpha, beta, gamma
        {{16, 32, 8, 8}, {64, 32, 5, 5}, {0, 0, 1, 1, 1, 1}, true, "default", true, 3, 1, 0.5, 0.5, 0.5},
        {{16, 32, 8, 8}, {64, 32, 5, 5}, {1, 1, 2, 2, 1, 1}, true, "default", true, 3, 1, 0.5, 0.5, 0.5},
    };
}

template <typename T>
void RunCbnaInferenceTest(const CbnaTestCase& test_case)
{
    uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

    // Create input tensor
    tensor<T> input{static_cast<size_t>(test_case.input_dims[0]),
                    static_cast<size_t>(test_case.input_dims[1]),
                    static_cast<size_t>(test_case.input_dims[2]),
                    static_cast<size_t>(test_case.input_dims[3])};
    input.generate(tensor_elem_gen_integer{max_value});

    // Create weights tensor
    tensor<T> weights{static_cast<size_t>(test_case.weights_dims[0]),
                      static_cast<size_t>(test_case.weights_dims[1]),
                      static_cast<size_t>(test_case.weights_dims[2]),
                      static_cast<size_t>(test_case.weights_dims[3])};
    weights.generate(tensor_elem_gen_integer{max_value});

    int input_c, input_h, input_w, wei_c, wei_k, wei_h, wei_w;
    std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());
    std::tie(std::ignore, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());

    if(input_c != wei_c)
    {
        GTEST_SKIP() << "Input channels (" << input_c << ") != weight channels (" << wei_c << ")";
    }

    miopen::ConvolutionDescriptor filter;
    std::unordered_map<std::string, miopenConvolutionMode_t> cmode_lookup = {
        {"CONV", miopenConvolution}};
    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
        {"DEFAULT", miopenPaddingDefault}};

    filter.mode         = cmode_lookup["CONV"];
    filter.paddingMode  = pmode_lookup[miopen::ToUpper(test_case.pad_mode)];
    filter.pads[0]      = test_case.pads_strides_dilations[0];
    filter.pads[1]      = test_case.pads_strides_dilations[1];
    filter.strides[0]   = test_case.pads_strides_dilations[2];
    filter.strides[1]   = test_case.pads_strides_dilations[3];
    filter.dilations[0] = test_case.pads_strides_dilations[4];
    filter.dilations[1] = test_case.pads_strides_dilations[5];

    auto stride_h     = filter.strides[1];
    auto stride_w     = filter.strides[0];
    auto fpad_h       = filter.pads[1];
    auto fpad_w       = filter.pads[0];
    auto fpaddingMode = filter.paddingMode;

    if(fpaddingMode == miopenPaddingSame)
    {
        if(stride_h == 0 || stride_w == 0)
        {
            GTEST_SKIP() << "Invalid stride for SAME padding";
        }
        auto _pad_h = (input_h % stride_h == 0)
                          ? (std::max(static_cast<int>(wei_h - stride_h), 0))
                          : (std::max(static_cast<int>(wei_h - (input_h % stride_h)), 0));
        auto _pad_w = (input_w % stride_w == 0)
                          ? (std::max(static_cast<int>(wei_w - stride_w), 0))
                          : (std::max(static_cast<int>(wei_w - (input_w % stride_w)), 0));

        filter.pads[1] = _pad_h / 2;
        filter.pads[0] = _pad_w / 2;

        int out_h = std::ceil(static_cast<double>(input_h) / stride_h);
        int out_w = std::ceil(static_cast<double>(input_w) / stride_w);

        if(out_h <= 0 || out_w <= 0)
        {
            GTEST_SKIP() << "Invalid output dimensions for SAME padding";
        }
    }
    else if(fpaddingMode == miopenPaddingValid)
    {
        if(stride_h == 0 || stride_w == 0)
        {
            GTEST_SKIP() << "Invalid stride for VALID padding";
        }
        filter.pads[1] = 0;
        filter.pads[0] = 0;

        int out_h = std::ceil(static_cast<double>(input_h - wei_h + 1) / stride_h);
        int out_w = std::ceil(static_cast<double>(input_w - wei_w + 1) / stride_w);

        if(out_h <= 0 || out_w <= 0)
        {
            GTEST_SKIP() << "Invalid output dimensions for VALID padding";
        }
    }

    miopenBatchNormMode_t bnmode;
    if(test_case.batchnorm_mode == 1)
    {
        bnmode = miopenBNSpatial;
    }
    else
    {
        bnmode = miopenBNPerActivation;
    }

    auto output = get_output_tensor(filter, input, weights);
    auto derivedBnDesc = miopen::TensorDescriptor{};
    miopen::DeriveBNTensorDescriptor(derivedBnDesc, output.desc, bnmode);
    std::size_t ssn, ssc, ssh, ssw;
    std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());

    tensor<T> scale{ssn, ssc, ssh, ssw};
    scale.generate(tensor_elem_gen_integer{max_value});
    tensor<T> shift{ssn, ssc, ssh, ssw};
    shift.generate(tensor_elem_gen_integer{max_value});
    tensor<T> estMean{ssn, ssc, ssh, ssw};
    estMean.generate(tensor_elem_gen_integer{max_value});
    tensor<T> estVariance{ssn, ssc, ssh, ssw};
    estVariance.generate(tensor_elem_gen_integer{max_value});

    auto&& handle       = get_handle();
    auto ptr_fusionplan = GetManagedFusionPlanDesc(&input.desc);

    miopenFusionOpDescriptor_t convoOp = nullptr;
    miopenFusionOpDescriptor_t biasOp  = nullptr;
    miopenFusionOpDescriptor_t bNormOp = nullptr;
    miopenFusionOpDescriptor_t activOp = nullptr;

    miopenCreateOpConvForward(ptr_fusionplan.get(), &convoOp, &filter, &weights.desc);

    tensor<T> bias;
    if(test_case.bias_mode)
    {
        bias = tensor<T>{1, output.desc.GetLengths()[1], 1, 1};
        bias.generate(tensor_elem_gen_integer{max_value});
        miopenCreateOpBiasForward(ptr_fusionplan.get(), &biasOp, &bias.desc);
    }
    else
    {
        bias = tensor<T>{1, 1, 1, 1};
    }

    miopenCreateOpBatchNormInference(ptr_fusionplan.get(), &bNormOp, bnmode, &scale.desc);

    ptr_ActivationDesc ptr_activdesc = GetManagedActivDesc();
    miopenActivationMode_t activ_mode = miopenActivationRELU;
    switch(test_case.activ_mode)
    {
    case 0: activ_mode = miopenActivationPASTHRU; break;
    case 1: activ_mode = miopenActivationLOGISTIC; break;
    case 2: activ_mode = miopenActivationTANH; break;
    case 3: activ_mode = miopenActivationRELU; break;
    case 4: activ_mode = miopenActivationSOFTRELU; break;
    case 5: activ_mode = miopenActivationABS; break;
    case 6: activ_mode = miopenActivationPOWER; break;
    case 7: activ_mode = miopenActivationCLIPPEDRELU; break;
    case 8: activ_mode = miopenActivationLEAKYRELU; break;
    case 9: activ_mode = miopenActivationELU; break;
    }

    if(test_case.test_activ)
    {
        miopenSetActivationDescriptor(ptr_activdesc.get(),
                                      activ_mode,
                                      test_case.alpha,
                                      test_case.beta,
                                      test_case.gamma);
        miopenCreateOpActivationForward(ptr_fusionplan.get(), &activOp, activ_mode);
    }

    // Compile fusion plan
    miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, ptr_fusionplan.get());
    if(miopenError != miopenStatusSuccess)
    {
        GTEST_SKIP() << "Fusion plan compilation failed";
    }

    // Check if test case is valid
    if(input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1) &&
       wei_h > 2 * fpad_h && wei_w > 2 * fpad_w && input_h >= (2 * fpad_h + wei_h) &&
       input_w >= (2 * fpad_w + wei_w))
    {
        // Run verification
        verify_forward_conv_bias_batchnorm_activ<T> verifier{ptr_fusionplan.get(),
                                                               input,
                                                               weights,
                                                               filter,
                                                               test_case.bias_mode,
                                                               bias,
                                                               ptr_activdesc.get(),
                                                               test_case.test_activ,
                                                               scale,
                                                               shift,
                                                               estMean,
                                                               estVariance,
                                                               bnmode};

        auto cpu_result = verifier.cpu();
        auto gpu_result = verifier.gpu();

        // Compare results
        EXPECT_EQ(miopen::range_distance(cpu_result), miopen::range_distance(gpu_result));

        using value_type = T;
        const double tolerance = 80.0;
        const double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
        const double rms_error = miopen::rms_range(cpu_result, gpu_result);

        EXPECT_LE(rms_error, threshold)
            << "RMS error: " << rms_error << " exceeds threshold: " << threshold;

        if(rms_error > threshold)
        {
            const auto mxdiff = miopen::max_diff(cpu_result, gpu_result);
            std::cout << "Max diff: " << mxdiff << std::endl;
            const auto idx = miopen::mismatch_idx(cpu_result, gpu_result, miopen::float_equal);
            if(idx < miopen::range_distance(cpu_result))
            {
                std::cout << "Mismatch at " << idx << ": " << cpu_result[idx]
                          << " != " << gpu_result[idx] << std::endl;
            }
        }
    }
    else
    {
        GTEST_SKIP() << "Test case dimensions do not meet requirements";
    }
}

    {

} // namespace

class GPU_CbnaInference_FP32 : public testing::TestWithParam<CbnaTestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        if(!IsTestSupportedByDevice(Gpu::All))
        {
            GTEST_SKIP();
        }
    }
};

class GPU_CbnaInference_FP16 : public testing::TestWithParam<CbnaTestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        if(!IsTestSupportedByDevice(Gpu::All))
        {
            GTEST_SKIP();
        }
    }
};

TEST_P(GPU_CbnaInference_FP32, FloatTest_cbna_inference)
{
    RunCbnaInferenceTest<float>(GetParam());
}

TEST_P(GPU_CbnaInference_FP16, HalfTest_cbna_inference)
{
    RunCbnaInferenceTest<half_float::half>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_CbnaInference_FP32, testing::ValuesIn(GetCbnaTestCases<float>()));

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_CbnaInference_FP16, testing::ValuesIn(GetCbnaTestCases<half_float::half>()));
