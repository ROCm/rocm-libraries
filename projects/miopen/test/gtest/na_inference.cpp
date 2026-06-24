// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "../fusionHost.hpp"
#include "../random.hpp"
#include "compare_helper.hpp"
#include <common_utils/stringutils.hpp>

#define MIO_BN_USE_MIX_PREC 1
#if MIO_BN_USE_MIX_PREC == 1
#define PREC_TYPE float
#else
#define PREC_TYPE T
#endif

namespace {
using ptr_FusionPlanDesc = MIOPEN_MANAGE_PTR(miopenFusionPlanDescriptor_t, miopenDestroyFusionPlan);
using ptr_FusionPlanArgs = MIOPEN_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);
using ptr_ActivationDesc = MIOPEN_MANAGE_PTR(miopenActivationDescriptor_t,
                                             miopenDestroyActivationDescriptor);

constexpr int batch_factor = 0;

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

template <class T, class U>
struct verify_inference_batchnorm_activ
{
    tensor<T> input;
    miopenTensorDescriptor_t inputDesc{};
    miopenActivationDescriptor_t activDesc{};
    miopenTensorDescriptor_t biasScaleTensor{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    tensor<U> estMean{};
    tensor<U> estVariance{};
    miopenBatchNormMode_t bnmode;
    miopenFusionPlanDescriptor_t fusionplan;
    miopenFusionOpDescriptor_t bNormOp;
    miopenFusionOpDescriptor_t activOp;
    double epsilon;

    verify_inference_batchnorm_activ(miopenFusionPlanDescriptor_t pfusionplan,
                                     tensor<T>& pinput,
                                     miopenActivationDescriptor_t pactivDesc,
                                     tensor<U>& pbnscale,
                                     const tensor<U>& pbnbias,
                                     const tensor<U>& pestMean,
                                     const tensor<U>& pestVariance,
                                     miopenBatchNormMode_t pbnmode,
                                     miopenFusionOpDescriptor_t pbNormOp,
                                     miopenFusionOpDescriptor_t pactivOp)
    {
        input           = pinput;
        inputDesc       = &pinput.desc;
        activDesc       = pactivDesc;
        biasScaleTensor = &pbnscale.desc;
        bnscale         = pbnscale;
        bnbias          = pbnbias;
        estMean         = pestMean;
        estVariance     = pestVariance;
        bnmode          = pbnmode;
        fusionplan      = pfusionplan;
        bNormOp         = pbNormOp;
        activOp         = pactivOp;
        epsilon         = 1.0e-5;
    }

    tensor<T> cpu() const
    {
        auto bout = input;
        std::fill(bout.begin(), bout.end(), 0.);
        auto aout = input;
        std::fill(aout.begin(), aout.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        if(bnmode == miopenBNPerActivation)
        {
            batchNormPerActivHostInference(
                input, bout, bnscale, bnbias, epsilon, estMean, estVariance);
        }
        else
        {
            batchNormSpatialHostInference(
                input, bout, bnscale, bnbias, epsilon, estMean, estVariance);
        }
        activationHostInfer(activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
        return aout;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto baout    = input;
        std::fill(baout.begin(), baout.end(), 0.);
        auto in_dev          = handle.Write(input.data);
        auto out_dev         = handle.Write(baout.data);
        auto bnscale_dev     = handle.Write(bnscale.data);
        auto bnbias_dev      = handle.Write(bnbias.data);
        auto estMean_dev     = handle.Write(estMean.data);
        auto estVariance_dev = handle.Write(estVariance.data);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        double alpha = 1., beta = 0.;
        auto ptr_fusionargs = GetManageFusionPlanArgs();
        miopenSetOpArgsBatchNormInference(ptr_fusionargs.get(),
                                          bNormOp,
                                          &alpha,
                                          &beta,
                                          bnscale_dev.get(),
                                          bnbias_dev.get(),
                                          estMean_dev.get(),
                                          estVariance_dev.get(),
                                          epsilon);
        miopenSetOpArgsActivForward(
            ptr_fusionargs.get(), activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                inputDesc,
                                in_dev.get(),
                                inputDesc,
                                out_dev.get(),
                                ptr_fusionargs.get());
        baout.data = handle.Read<T>(out_dev, baout.data.size());
        return baout;
    }

    void fail() const { GTEST_FAIL() << "BatchNorm+Activation Inference:" << std::endl; }
};

static std::string transform_mode(std::string s)
{
    return miopen::RemovePrefix(miopen::ToUpper(s), "MIOPENACTIVATION");
}

using TestCase = std::tuple<std::vector<int>, double, double, double, std::string, int>;

auto GenCases(bool full = false)
{
    if(!full)
    {
        return ::testing::Combine(::testing::ValuesIn(std::set<std::vector<int>>{{16, 32, 8, 8}}),
                                  ::testing::ValuesIn({double{0.5}}),
                                  ::testing::ValuesIn({double{0.5}}),
                                  ::testing::ValuesIn({double{0.5}}),
                                  ::testing::ValuesIn({std::string{"MIOPENACTIVATIONRELU"}}),
                                  ::testing::ValuesIn({/*0, */ 1}));
    }
    return ::testing::Combine(
        ::testing::ValuesIn(get_inputs(batch_factor)),
        ::testing::ValuesIn({double{0.5}}),
        ::testing::ValuesIn({double{0.5}}),
        ::testing::ValuesIn({double{0.5}}),
        ::testing::ValuesIn({std::string{
            "MIOPENACTIVATIONRELU",
            /*, "std::string{MIOPENACTIVATIONLOGISTIC}", "std::string{MIOPENACTIVATIONABS}"*/}}),
        ::testing::ValuesIn({/*0, */ 1}));
}

auto GetSmokeCases()
{
    static auto cases = GenCases();
    return cases;
}

auto GetFullCases()
{
    static auto cases = GenCases(true);
    return cases;
}

template <class T>
struct na_fusion_inference_test : public ::testing::TestWithParam<TestCase>
{
    tensor<T> input;
    tensor<PREC_TYPE> scale;
    tensor<PREC_TYPE> shift;
    tensor<PREC_TYPE> estMean;
    tensor<PREC_TYPE> estVariance;
    ptr_ActivationDesc ptr_activdesc = nullptr;

    miopenActivationMode_t activ_mode = miopenActivationRELU;
    std::string amode;
    miopenBatchNormMode_t bnmode{};
    int batchnormMode = 1;

    const uint64_t max_value = miopen_type<T>{} == miopenHalf ? 3 : 17;
    double alpha = 0., beta = 0., gamma = 0.;

    void SetUp() override
    {
        std::vector<int> nchw{};
        std::tie(nchw, alpha, beta, gamma, amode, batchnormMode) = GetParam();
        input = tensor<T>{nchw[0], nchw[1], nchw[2], nchw[3]};
        input.generate(tensor_elem_gen_integer{max_value});
    }

    void Run()
    {
        amode = transform_mode(amode);

        // NOLINTBEGIN(*-braces-around-statements)
        if(amode == "PASSTHRU")
            activ_mode = miopenActivationPASTHRU;
        else if(amode == "LOGISTIC")
            activ_mode = miopenActivationLOGISTIC;
        else if(amode == "TANH")
            activ_mode = miopenActivationTANH;
        else if(amode == "RELU")
            activ_mode = miopenActivationRELU;
        else if(amode == "SOFTRELU")
            activ_mode = miopenActivationSOFTRELU;
        else if(amode == "ABS")
            activ_mode = miopenActivationABS;
        else if(amode == "POWER")
            activ_mode = miopenActivationPOWER;
        else if(amode == "CLIPPEDRELU")
            activ_mode = miopenActivationCLIPPEDRELU;
        else if(amode == "LEAKYRELU")
            activ_mode = miopenActivationLEAKYRELU;
        else if(amode == "ELU")
            activ_mode = miopenActivationELU;
        // NOLINTEND(*-braces-around-statements)

        int input_c, input_h, input_w;
        std::tie(std::ignore, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());
        ptr_activdesc                                    = GetManagedActivDesc();
        miopenSetActivationDescriptor(ptr_activdesc.get(), activ_mode, alpha, beta, gamma);

        if(batchnormMode == 1)
        {
            bnmode = miopenBNSpatial;
        }
        else if(batchnormMode == 0)
        {
            bnmode = miopenBNPerActivation;
        }

        std::size_t ssn, ssc, ssh, ssw;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bnmode);
        std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());

        scale = tensor<PREC_TYPE>{
            ssn, ssc, ssh, ssw}; //.generate(                tensor_elem_gen_integer{max_value});;
        shift = tensor<PREC_TYPE>{
            ssn, ssc, ssh, ssw}; //.generate(               tensor_elem_gen_integer{max_value});;
        estMean = tensor<PREC_TYPE>{
            ssn, ssc, ssh, ssw}; //.generate(                tensor_elem_gen_integer{max_value});;
        estVariance = tensor<PREC_TYPE>{
            ssn, ssc, ssh, ssw}; //.generate(                tensor_elem_gen_integer{max_value});;

        const double Data_scale = 0.01;
        for(std::size_t i = 0; i < scale.desc.GetElementSize(); i++)
        {
            scale[i]       = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            shift[i]       = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            estMean[i]     = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            estVariance[i] = static_cast<PREC_TYPE>(Data_scale * prng::gen_off_range(1, 100));
        }
        for(std::size_t i = 0; i < input.desc.GetElementSize(); i++)
        {
            input[i] = prng::gen_descreet_uniform_sign<T>(Data_scale, 100);
        }

        auto&& handle = get_handle();

        miopenFusionOpDescriptor_t bNormOp = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;

        auto ptr_fusionplan = GetManagedFusionPlanDesc(&input.desc);

        miopenCreateOpBatchNormInference(ptr_fusionplan.get(), &bNormOp, bnmode, &scale.desc);
        miopenCreateOpActivationForward(ptr_fusionplan.get(), &activOp, activ_mode);

        miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, ptr_fusionplan.get());
        if(miopenError != miopenStatusSuccess)
        {
            GTEST_SKIP() << "BatchNorm+Activation Inference plan not supported." << std::endl;
        }
        else
        {
            test_helpers::CompareResults(
                verify_inference_batchnorm_activ<T, PREC_TYPE>{ptr_fusionplan.get(),
                                                               input,
                                                               ptr_activdesc.get(),
                                                               scale,
                                                               shift,
                                                               estMean,
                                                               estVariance,
                                                               bnmode,
                                                               bNormOp,
                                                               activOp});
        }
    }
};

struct TestNameGenerator
{
    std::string operator()(const ::testing::TestParamInfo<TestCase>& param_info)
    {
        std::stringstream ss{};
        auto replace_dot = [](double value) // assuming there's only one
        {
            std::string str{std::to_string(value)};
            auto i = str.find('.');
            if(i != std::string::npos)
                str[i] = '_';
            return str;
        };

        auto print_nchw = [](std::vector<int> const& vec) {
            std::stringstream vec_ss{};
            for(auto el : vec)
            {
                vec_ss << std::to_string(el) << "_";
            }
            return vec_ss.str();
        };

        ss << "nchw_" << print_nchw(std::get<0>(param_info.param)) << "_alpha_"
           << replace_dot(std::get<1>(param_info.param)) << "_beta_"
           << replace_dot(std::get<2>(param_info.param)) << "_gamma_"
           << replace_dot(std::get<3>(param_info.param)) << "_amode_"
           << std::get<4>(param_info.param) << "_batchnormMode_" << std::get<5>(param_info.param);
        return ss.str();
    }
};

} // namespace

using GPU_na_fusion_inference_test_FP16 = na_fusion_inference_test<half_float::half>;
using GPU_na_fusion_inference_test_FP32 = na_fusion_inference_test<float>;

TEST_P(GPU_na_fusion_inference_test_FP16, TestFloat16) { Run(); }
TEST_P(GPU_na_fusion_inference_test_FP32, TestFloat32) { Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_na_fusion_inference_test_FP16,
                         GetSmokeCases(),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_na_fusion_inference_test_FP32,
                         GetSmokeCases(),
                         TestNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_na_fusion_inference_test_FP16,
                         GetFullCases(),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_na_fusion_inference_test_FP32,
                         GetFullCases(),
                         TestNameGenerator{});
