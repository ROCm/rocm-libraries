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
                           const float activ_gamma,
                           const miopen::TensorDescriptor& xDesc,
                           ConstData_t x,
                           const miopen::TensorDescriptor& /*yDesc*/,
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
    bool is_layout_NHWC = (xDesc.GetLayout_t() == miopenTensorNHWC);
    size_t read_len     = (bn_mode == miopenBNSpatial) ? (is_layout_NHWC ? c : h * w) : c * h * w;
    size_t read_unit    = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    // For vectorized r/rw of the input/output data
    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);
    size_t max_localsize  = 256;
    size_t xlocalsize     = std::min(size_t{read_len / read_unit}, max_localsize);
    size_t xgridsize      = AlignUp(size_t{read_len / read_unit}, xlocalsize);
    size_t ylocalsize     = 1;
    size_t ygridsize      = (bn_mode == miopenBNSpatial) ? size_t{is_layout_NHWC ? h * w : c} : 1;
    size_t zlocalsize     = 1;
    size_t zgridsize      = 1;

    const std::vector<size_t> vgd{xgridsize, ygridsize, zgridsize};
    const std::vector<size_t> vld{xlocalsize, ylocalsize, zlocalsize};

    const auto build_params = miopen::KernelBuildParameters{
        {"MIO_BN_CHW", static_cast<unsigned>(c * h * w)},
        {"MIO_BN_HW", static_cast<unsigned>(h * w)},
        {"MIO_BN_N", static_cast<unsigned>(n)},
        {"MIO_BN_C", static_cast<unsigned>(c)},
        {"MIO_BN_GRP0", xlocalsize},
        {"MIO_BN_GRP1", ylocalsize},
        {"MIO_BN_GRP2", zlocalsize},
        {"MIO_LAYOUT_NHWC", static_cast<int>(is_layout_NHWC)},
        {"MIOPEN_READ_UNIT", static_cast<int>(read_unit)},
        {"MIOPEN_SBN_BOUNDS", static_cast<unsigned int>(read_len / read_unit)},
        {"MIOPEN_READ_TYPE", READ_TYPE},
        {"MIOPEN_NRN_OP_ID", static_cast<int>(activ_mode)},
        {"MIOPEN_USE_BFPMIX", static_cast<int>(xDesc.GetType() == miopenBFloat16)},
        {"MIOPEN_USE_FPMIX", static_cast<int>(xDesc.GetType() == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(xDesc.GetType() == miopenFloat)}};

    std::string kernel_file = "MIOpenBatchNormActivInfer.cpp";
    std::string kernel_name = "MIOpenBatchNormActivInfer";

    std::string params = build_params.GenerateFor(miopen::kbp::HIP{});

    if(bn_mode == miopenBNSpatial)
    {
        kernel_name += "SpatialEst";
    }
    else
    {
        kernel_name += "PerActEst";
    }

    // Generate the network config
    std::ostringstream ss;
    ss << "bfp16" << static_cast<int>(xDesc.GetType() == miopenBFloat16);
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
        if(xDesc.GetType() == miopenFloat)
        {
            perf_helper.perfTest(handle,
                                 kernel_name,
                                 network_config,
                                 static_cast<float>(activ_alpha),
                                 static_cast<float>(activ_beta),
                                 static_cast<float>(activ_gamma),
                                 static_cast<double>(epsilon),
                                 x,
                                 y,
                                 bnBias,
                                 bnScale,
                                 estimatedMean,
                                 estimatedVariance);
        }
        else if(xDesc.GetType() == miopenHalf)
        {
            perf_helper.perfTest(handle,
                                 kernel_name,
                                 network_config,
                                 static_cast<_Float16>(activ_alpha),
                                 static_cast<_Float16>(activ_beta),
                                 static_cast<_Float16>(activ_gamma),
                                 static_cast<double>(epsilon),
                                 x,
                                 y,
                                 bnBias,
                                 bnScale,
                                 estimatedMean,
                                 estimatedVariance);
        }
        else if(xDesc.GetType() == miopenBFloat16)
        {
            perf_helper.perfTest(handle,
                                 kernel_name,
                                 network_config,
                                 static_cast<bfloat16>(activ_alpha),
                                 static_cast<bfloat16>(activ_beta),
                                 static_cast<bfloat16>(activ_gamma),
                                 static_cast<double>(epsilon),
                                 x,
                                 y,
                                 bnBias,
                                 bnScale,
                                 estimatedMean,
                                 estimatedVariance);
        }
    }
    else
    {
        // execute the kernel
        if(xDesc.GetType() == miopenFloat)
        {
            kernelInvoke(static_cast<float>(activ_alpha),
                         static_cast<float>(activ_beta),
                         static_cast<float>(activ_gamma),
                         static_cast<double>(epsilon),
                         x,
                         y,
                         bnBias,
                         bnScale,
                         estimatedMean,
                         estimatedVariance);
        }
        else if(xDesc.GetType() == miopenHalf)
        {
            kernelInvoke(static_cast<_Float16>(activ_alpha),
                         static_cast<_Float16>(activ_beta),
                         static_cast<_Float16>(activ_gamma),
                         static_cast<double>(epsilon),
                         x,
                         y,
                         bnBias,
                         bnScale,
                         estimatedMean,
                         estimatedVariance);
        }
        else if(xDesc.GetType() == miopenBFloat16)
        {
            kernelInvoke(static_cast<bfloat16>(activ_alpha),
                         static_cast<bfloat16>(activ_beta),
                         static_cast<bfloat16>(activ_gamma),
                         static_cast<double>(epsilon),
                         x,
                         y,
                         bnBias,
                         bnScale,
                         estimatedMean,
                         estimatedVariance);
        }
    }
}

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
struct BatchNormActivInferTester : public BatchNormInferTester<XDataType,
                                                               YDataType,
                                                               ScaleDataType,
                                                               BiasDataType,
                                                               MeanVarDataType>
{
#if PERF_ENABLE
    BatchNormActivInferTester()
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
                              this->activ_gamma,
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

namespace BatchNormActivInfer {

struct GPU_bn_activ_infer_spatial_FP32
    : BatchNormActivInferTester<float, float, float, float, float>
{
};

struct GPU_bn_activ_infer_per_act_FP32
    : BatchNormActivInferTester<float, float, float, float, float>
{
};

struct GPU_bn_activ_infer_spatial_FP16
    : BatchNormActivInferTester<half_float::half, half_float::half, float, float, float>
{
};

struct GPU_bn_activ_infer_per_act_FP16
    : BatchNormActivInferTester<half_float::half, half_float::half, float, float, float>
{
};

struct GPU_bn_activ_infer_spatial_BFP16
    : BatchNormActivInferTester<bfloat16, bfloat16, float, float, float>
{
};

struct GPU_bn_activ_infer_per_act_BFP16
    : BatchNormActivInferTester<bfloat16, bfloat16, float, float, float>
{
};

std::vector<miopenActivationMode_t> ActivationConfigs()
{
    return {miopenActivationPASTHRU,
            miopenActivationLOGISTIC,
            miopenActivationTANH,
            miopenActivationRELU,
            miopenActivationSOFTRELU,
            miopenActivationABS,
            miopenActivationPOWER,
            miopenActivationCLIPPEDRELU,
            miopenActivationLEAKYRELU,
            miopenActivationELU};
}

} // namespace BatchNormActivInfer
using namespace BatchNormActivInfer;

TEST_P(GPU_bn_activ_infer_spatial_FP32, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_activ_infer_per_act_FP32, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_activ_infer_spatial_FP16, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_activ_infer_per_act_FP16, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_activ_infer_spatial_BFP16, PortTest)
{
    // Run the CPU implementation
    RunTestCPU();
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_activ_infer_per_act_BFP16, PortTest)
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
    GPU_bn_activ_infer_spatial_FP32,
    testing::Combine(testing::ValuesIn(ActivationConfigs()),
                     testing::ValuesIn(BNInferTestConfigs<float>(miopenBNSpatial)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_per_act_FP32,
    testing::Combine(testing::ValuesIn(ActivationConfigs()),
                     testing::ValuesIn(BNInferTestConfigs<float>(miopenBNPerActivation)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_spatial_FP16,
    testing::Combine(testing::ValuesIn(ActivationConfigs()),
                     testing::ValuesIn(BNInferTestConfigs<half_float::half>(miopenBNSpatial)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_per_act_FP16,
    testing::Combine(testing::ValuesIn(ActivationConfigs()),
                     testing::ValuesIn(BNInferTestConfigs<half_float::half>(miopenBNPerActivation)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_spatial_BFP16,
    testing::Combine(testing::ValuesIn(ActivationConfigs()),
                     testing::ValuesIn(BNInferTestConfigs<bfloat16>(miopenBNSpatial)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_per_act_BFP16,
    testing::Combine(testing::ValuesIn(ActivationConfigs()),
                     testing::ValuesIn(BNInferTestConfigs<bfloat16>(miopenBNPerActivation)),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC})),
    TestNameGenerator());
