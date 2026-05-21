/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "get_handle.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>
#include <miopen/batchnorm/problem_description.hpp>

#include "bn_infer_tester.hpp"

#define PERF_ENABLE 0

#if PERF_ENABLE
#define NUM_WARMUP_RUNS_TEST 10
#define NUM_PERF_RUNS_TEST 100
#endif

void BatchNormInferenceGPU(const miopen::Handle& handle,
                           miopenBatchNormMode_t bn_mode,
                           miopenActivationMode_t activ_mode,
                           const float activ_alpha,
                           const float activ_beta,
                           const miopen::TensorDescriptor& xDesc,
                           ConstData_t x,
                           const miopen::TensorDescriptor& yDesc,
                           Data_t y,
                           const miopen::TensorDescriptor& /*bnScaleBiasMeanVarDesc*/,
                           ConstData_t bnScale,
                           ConstData_t bnBias,
                           ConstData_t estimatedMean,
                           ConstData_t estimatedVariance,
                           double epsilon,
                           PerfHelper& perf_helper)
{
    int n, c, h, w;
    std::tie(n, c, h, w) = miopen::tien<4>(xDesc.GetLengths());

    // Setup the kernel launch parameters
    if(xDesc.GetLayoutEnum() != yDesc.GetLayoutEnum())
    {
        throw std::runtime_error("Input and output tensor layout must be the same");
    }
    const bool isLayoutNHWC = (xDesc.GetLayoutEnum() == miopenTensorNHWC);
    unsigned int in_cstride = h * w;
    size_t max_localsize    = 256;
    size_t xlocalsize, xgridsize, ylocalsize, ygridsize, zlocalsize, zgridsize;
    size_t vectorsize = isLayoutNHWC ? (c % 4 == 0 ? 4 : (c % 2 == 0 ? 2 : 1))
                                     : (in_cstride % 4 == 0 ? 4 : (in_cstride % 2 == 0 ? 2 : 1));
    if(isLayoutNHWC)
    {
        xlocalsize = std::min(size_t{c / vectorsize}, max_localsize);
        xgridsize  = AlignUp(size_t{c / vectorsize}, xlocalsize);

        ylocalsize = max_localsize / xlocalsize;
        ygridsize  = AlignUp(size_t{in_cstride}, ylocalsize);
    }
    else
    {
        xlocalsize = 1;
        xgridsize  = AlignUp(size_t{c}, xlocalsize);

        ylocalsize = max_localsize;
        ygridsize  = AlignUp(size_t{in_cstride / vectorsize}, ylocalsize);
    }
    zlocalsize = 1;
    zgridsize  = 1;

    // HIP runtime does not support non-uniform blocks

    xgridsize = AlignUp(xgridsize, xlocalsize);
    ygridsize = AlignUp(ygridsize, ylocalsize);

    const std::vector<size_t> vgd{xgridsize, ygridsize, zgridsize};
    const std::vector<size_t> vld{xlocalsize, ylocalsize, zlocalsize};

    bool useFP32            = (xDesc.GetType() == miopenFloat);
    bool useFP16            = (xDesc.GetType() == miopenHalf);
    bool useBFP16           = (xDesc.GetType() == miopenBFloat16);
    const auto build_params = miopen::KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(useFP16)},
        {"MIOPEN_USE_FP32", static_cast<int>(useFP32)},
        {"MIOPEN_USE_BFP16", static_cast<int>(useBFP16)},
        {"MIOPEN_USE_BFPMIX", static_cast<int>(useBFP16)},
        {"MIO_BN_GRP0", xlocalsize},
        {"MIO_BN_GRP1", ylocalsize},
        {"MIO_BN_GRP2", zlocalsize},
        {"MIO_BN_GFX103X", (miopen::StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
        {"MIO_BN_GFX110X", (miopen::StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
        {"MIO_BN_GFX115X", (miopen::StartsWith(handle.GetDeviceName(), "gfx115") ? "1" : "0")},
        {"MIO_BN_GFX120X", (miopen::StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
        {"MIO_LAYOUT_NHWC", static_cast<int>(isLayoutNHWC)},
        {"MIO_BN_VECTORIZE", static_cast<int>(vectorsize > 1)},
        {"MIO_BN_VEC_SIZE", vectorsize},
        {"MIO_BN_N", static_cast<unsigned int>(n)},
        {"MIOPEN_NRN_OP_ID", static_cast<int>(activ_mode)}};

    std::string kernel_file = bn_mode == miopenBNSpatial ? "MIOpenBatchNormFwdInferSpatial.cpp"
                                                         : "MIOpenBatchNormFwdInferPerAct.cpp";
    std::string kernel_name = (bn_mode == miopenBNSpatial)
                                  ? "MIOpenBatchNormFwdInferSpatialEst"
                                  : "MIOpenBatchNormFwdInferPerActivationEst";

    std::string params = build_params.GenerateFor(miopen::kbp::HIP{});

    // Generate the network config
    std::ostringstream ss;
    ss << "fp16" << static_cast<int>(xDesc.GetType() == miopenHalf);
    ss << "fp32" << static_cast<int>(xDesc.GetType() == miopenFloat);
    ss << "mode" << bn_mode;
    ss << "N" << n;
    ss << "C" << c;
    ss << "H" << h;
    ss << "W" << w;
    ss << "activ" << activ_mode;
    std::string network_config = ss.str();

    // add the kernel to the handle
    [[maybe_unused]] auto kernelInvoke =
        handle.AddKernel(kernel_name, network_config, kernel_file, kernel_name, vld, vgd, params);

    if constexpr(PERF_ENABLE)
    {
        // run the perf test
        perf_helper.perfTest(handle,
                             kernel_name,
                             network_config,
                             x,
                             y,
                             estimatedMean,
                             estimatedVariance,
                             bnScale,
                             bnBias,
                             epsilon,
                             c,
                             h * w,
                             n,
                             isLayoutNHWC ? 1 : h * w,
                             isLayoutNHWC ? c : 1,
                             c * h * w,
                             activ_alpha,
                             activ_beta);
    }
    else
    {
        // execute the kernel
        kernelInvoke(x,
                     y,
                     estimatedMean,
                     estimatedVariance,
                     bnScale,
                     bnBias,
                     epsilon,
                     c,
                     h * w,
                     n,
                     isLayoutNHWC ? 1 : h * w,
                     isLayoutNHWC ? c : 1,
                     c * h * w,
                     activ_alpha,
                     activ_beta);
    }
}

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
struct BatchNormFwdInferTester : public BatchNormInferTester<XDataType,
                                                             YDataType,
                                                             ScaleDataType,
                                                             BiasDataType,
                                                             MeanVarDataType>
{
#if PERF_ENABLE
    BatchNormFwdInferTester()
    {
        this->perf_enable = static_cast<bool>(PERF_ENABLE);
        this->perf_helper.setWarmupRuns(NUM_WARMUP_RUNS_TEST);
        this->perf_helper.setPerfRuns(NUM_PERF_RUNS_TEST);
    }
#endif

    void RunTestGPU() override
    {
        auto&& handle    = get_handle();
        auto& output_ref = this->output.data;
        // Clear the output data
        std::fill(
            output_ref.begin(), output_ref.end(), std::numeric_limits<YDataType>::quiet_NaN());
        this->out_dev = handle.Write(output_ref);
        // Execute the implementation
        BatchNormInferenceGPU(handle,
                              this->bn_config.mode,
                              this->activ_mode,
                              this->activ_alpha,
                              this->activ_beta,
                              this->input.desc,
                              this->in_dev.get(),
                              this->output.desc,
                              this->out_dev.get(),
                              this->scale.desc,
                              this->scale_dev.get(),
                              this->shift_dev.get(),
                              this->estMean_dev.get(),
                              this->estVariance_dev.get(),
                              this->epsilon,
                              this->perf_helper);
        // Read the output
        output_ref = handle.Read<YDataType>(this->out_dev, this->output.data.size());
    }
};

namespace BatchNormFwdInfer {

struct GPU_bn_fwd_infer_spatial_FP32 : BatchNormFwdInferTester<float, float, float, float, float>
{
};

struct GPU_bn_fwd_infer_per_act_FP32 : BatchNormFwdInferTester<float, float, float, float, float>
{
};

struct GPU_bn_fwd_infer_spatial_FP16
    : BatchNormFwdInferTester<half_float::half, half_float::half, float, float, float>
{
};

struct GPU_bn_fwd_infer_per_act_FP16
    : BatchNormFwdInferTester<half_float::half, half_float::half, float, float, float>
{
};

struct GPU_bn_fwd_infer_spatial_BFP16
    : BatchNormFwdInferTester<bfloat16, bfloat16, float, float, float>
{
};

struct GPU_bn_fwd_infer_per_act_BFP16
    : BatchNormFwdInferTester<bfloat16, bfloat16, float, float, float>
{
};

std::vector<miopenActivationMode_t> ActivationConfigs(miopenBatchNormMode_t mode)
{
    std::vector<miopenActivationMode_t> activations = {miopenActivationPASTHRU};

    if(mode == miopenBNSpatial)
    {
        activations.push_back(miopenActivationRELU);
        activations.push_back(miopenActivationCLIPPEDRELU);
        activations.push_back(miopenActivationCLAMP);
    }

    return activations;
}

} // namespace BatchNormFwdInfer
using namespace BatchNormFwdInfer;

TEST_P(GPU_bn_fwd_infer_spatial_FP32, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_fwd_infer_per_act_FP32, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_fwd_infer_spatial_FP16, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_fwd_infer_per_act_FP16, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_fwd_infer_spatial_BFP16, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_fwd_infer_per_act_BFP16, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_fwd_infer_spatial_FP32,
    testing::Combine(testing::ValuesIn(ActivationConfigs(miopenBNSpatial)),
                     testing::ValuesIn(BNInferTestConfigs<float>(miopenBNSpatial)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_fwd_infer_per_act_FP32,
    testing::Combine(testing::ValuesIn(ActivationConfigs(miopenBNPerActivation)),
                     testing::ValuesIn(BNInferTestConfigs<float>(miopenBNPerActivation)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_fwd_infer_spatial_FP16,
    testing::Combine(testing::ValuesIn(ActivationConfigs(miopenBNSpatial)),
                     testing::ValuesIn(BNInferTestConfigs<half_float::half>(miopenBNSpatial)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_fwd_infer_per_act_FP16,
    testing::Combine(testing::ValuesIn(ActivationConfigs(miopenBNPerActivation)),
                     testing::ValuesIn(BNInferTestConfigs<half_float::half>(miopenBNPerActivation)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_fwd_infer_spatial_BFP16,
    testing::Combine(testing::ValuesIn(ActivationConfigs(miopenBNSpatial)),
                     testing::ValuesIn(BNInferTestConfigs<bfloat16>(miopenBNSpatial)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_fwd_infer_per_act_BFP16,
    testing::Combine(testing::ValuesIn(ActivationConfigs(miopenBNPerActivation)),
                     testing::ValuesIn(BNInferTestConfigs<bfloat16>(miopenBNPerActivation)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
