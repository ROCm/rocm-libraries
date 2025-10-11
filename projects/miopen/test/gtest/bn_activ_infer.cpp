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
#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>
#include <miopen/batchnorm/problem_description.hpp>

#include "na.hpp"
#include "perf_helper.hpp"

#define PERF_ENABLE 0

#if PERF_ENABLE
#define COMPARE_WITH_OPENCL 1
#define NUM_WARMUP_RUNS_TEST 10
#define NUM_PERF_RUNS_TEST 100
#endif

#ifndef warpSize
#define warpSize 32
#endif

void BatchNormInferenceGPU(const miopen::Handle& handle,
                           miopenBatchNormMode_t bn_mode,
                           miopenActivationMode_t activ_mode,
                           const float activ_alpha,
                           const float activ_beta,
                           const float activ_gamma,
                           const miopen::TensorDescriptor& xDesc,
                           ConstData_t x,
                           const miopen::TensorDescriptor& yDesc,
                           Data_t y,
                           const miopen::TensorDescriptor& bnScaleBiasMeanVarDesc,
                           ConstData_t bnScale,
                           ConstData_t bnBias,
                           ConstData_t estimatedMean,
                           ConstData_t estimatedVariance,
                           double epsilon,
                           PerfHelper<float>& perf_helper,
                           bool use_hip = true)
{
    int n, c, h, w;
    std::tie(n, c, h, w) = miopen::tien<4>(xDesc.GetLengths());

    size_t read_unit = 1;
    size_t read_len  = (bn_mode == miopenBNSpatial) ? h * w : c * h * w;
    // vectorized reads for spatial when not using fp16
    if(bn_mode == miopenBNSpatial && xDesc.GetType() != miopenHalf)
    {
        read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    }
    // For vectorized r/rw of the input/output data of FP_TYPE
    std::string READ_TYPE = (use_hip ? "FP_TYPE" : "_FLOAT");
    READ_TYPE             = (read_unit == 1) ? READ_TYPE : READ_TYPE + std::to_string(read_unit);
    // For vectorized r/rw of the other data of FP_TYPE_PREC
    std::string PREC_READ_TYPE = (use_hip ? "FP_TYPE_PREC" : "_FLOAT_PREC");
    PREC_READ_TYPE = (read_unit == 1) ? PREC_READ_TYPE : PREC_READ_TYPE + std::to_string(read_unit);
    // Setup the kernel launch parameters
    size_t xlocalsize = 256;
    size_t xgridsize  = read_len / read_unit;
    // HIP runtime does not support non-uniform blocks
    if(use_hip)
    {
        if(xgridsize < xlocalsize)
        {
            // round up the xlocalsize to the nearest wavefront size
            xlocalsize = AlignUp(xgridsize, warpSize);
            // Set xgridsize to the xlocalsize, to launch only one block
            xgridsize = xlocalsize;
        }
        else
        {
            xgridsize = AlignUp(xgridsize, xlocalsize);
        }
    }
    size_t ylocalsize = 1;
    size_t ygridsize  = (bn_mode == miopenBNSpatial) ? size_t(c) : 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    const std::vector<size_t> vgd{xgridsize, ygridsize, zgridsize};
    const std::vector<size_t> vld{xlocalsize, ylocalsize, zlocalsize};

    const auto build_params = miopen::KernelBuildParameters{
        {"MIO_BN_CHW", static_cast<unsigned>(c * h * w)},
        {"MIO_BN_HW", static_cast<unsigned>(h * w)},
        {"MIO_BN_N", static_cast<unsigned>(n)},
        {"MIO_BN_GRP0", xlocalsize},
        {"MIO_BN_GRP1", ylocalsize},
        {"MIO_BN_GRP2", zlocalsize},
        {"MIOPEN_READ_UNIT", static_cast<int>(read_unit)},
        {"MIOPEN_SBN_BOUNDS", static_cast<unsigned>(read_len / read_unit)},
        {"MIOPEN_READ_TYPE", READ_TYPE},
        {"MIOPEN_PREC_READ_TYPE", PREC_READ_TYPE},
        {"MIOPEN_NRN_OP_ID", static_cast<int>(activ_mode)},
        {"MIOPEN_USE_FP16", static_cast<int>(xDesc.GetType() == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(xDesc.GetType() == miopenFloat)}};

    std::string kernel_file =
        (use_hip ? "MIOpenBatchNormActivInfer.cpp" : "MIOpenBatchNormActivInferOCL.cl");
    std::string kernel_name = "MIOpenBatchNormActivInfer";

    std::string params = use_hip ? build_params.GenerateFor(miopen::kbp::OpenCL{})
                                 : build_params.GenerateFor(miopen::kbp::HIP{});

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
    ss << (use_hip ? "hip" : "ocl");
    ss << "fp16" << static_cast<int>(xDesc.GetType() == miopenHalf);
    ss << "fp32" << static_cast<int>(xDesc.GetType() == miopenFloat);
    ss << "mode" << bn_mode;
    ss << "N" << n;
    ss << "C" << c;
    ss << "H" << h;
    ss << "W" << w;
    ss << "activ" << activ_mode;
    std::string network_config = ss.str();

    if constexpr(PERF_ENABLE)
    {
#if COMPARE_WITH_OPENCL
        // disable the perf test for FP16 as OpenCL FP16 is broken
        if(xDesc.GetType() != miopenHalf)
        {
            // add the kernel to the handle
            handle.AddKernel(
                kernel_name, network_config, kernel_file, kernel_name, vld, vgd, params);
            // run the perf test
            perf_helper.perfTest(handle,
                                 kernel_name,
                                 network_config,
                                 use_hip,
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
#else
        // add the kernel to the handle
        handle.AddKernel(kernel_name, network_config, kernel_file, kernel_name, vld, vgd, params);
        // run the perf test
        perf_helper.perfTest(handle,
                             kernel_name,
                             network_config,
                             use_hip,
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
#endif
    }
    else
    {
        // add the kernel to the handle and execute it
        handle.AddKernel(kernel_name, network_config, kernel_file, kernel_name, vld, vgd, params)(
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
}

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
struct BatchNormActivInferTest
    : public ::testing::TestWithParam<std::tuple<miopenActivationMode_t, BNTestCase>>
{

    void SetUp() override
    {
        std::tie(activ_mode, bn_config) = GetParam();

        // Create tensors
        input              = tensor<XDataType>{bn_config.GetInput()};
        output             = tensor<YDataType>{bn_config.GetInput()};
        ref_out            = tensor<YDataType>{bn_config.GetInput()};
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bn_config.mode);
        scale       = tensor<ScaleDataType>{derivedBnDesc.GetLengths()};
        shift       = tensor<BiasDataType>{derivedBnDesc.GetLengths()};
        estMean     = tensor<MeanVarDataType>{derivedBnDesc.GetLengths()};
        estVariance = tensor<MeanVarDataType>{derivedBnDesc.GetLengths()};
        // Fill tensors
        auto gen_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<XDataType>(1e-2, 100);
        };
        input.generate(gen_value);

        auto gen_scale = [](auto...) {
            return prng::gen_descreet_uniform_sign<ScaleDataType>(1e-2, 100);
        };
        scale.generate(gen_scale);
        shift.generate(gen_scale);
        estMean.generate(gen_scale);

        auto gen_var = [](auto...) {
            return static_cast<MeanVarDataType>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };
        estVariance.generate(gen_var);
        // Write data to GPU
        auto&& handle   = get_handle();
        in_dev          = handle.Write(input.data);
        scale_dev       = handle.Write(scale.data);
        shift_dev       = handle.Write(shift.data);
        estMean_dev     = handle.Write(estMean.data);
        estVariance_dev = handle.Write(estVariance.data);

        // Enable profiling and set number of runs for perf tests
#if PERF_ENABLE
        handle.EnableProfiling(true); // enable profiling
        perf_helper.setWarmupRuns(NUM_WARMUP_RUNS_TEST);
        perf_helper.setPerfRuns(NUM_PERF_RUNS_TEST);
#endif
    }

    void RunTestGPU(bool hip_en = true)
    {
        auto&& handle    = get_handle();
        auto& output_ref = hip_en ? output.data : ref_out.data;
        // Clear the output data
        std::fill(
            output_ref.begin(), output_ref.end(), std::numeric_limits<YDataType>::quiet_NaN());
        out_dev = handle.Write(output_ref);
        // Execute the implementation
        BatchNormInferenceGPU(handle,
                              bn_config.mode,
                              activ_mode,
                              activ_alpha,
                              activ_beta,
                              activ_gamma,
                              input.desc,
                              in_dev.get(),
                              output.desc,
                              out_dev.get(),
                              scale.desc,
                              scale_dev.get(),
                              shift_dev.get(),
                              estMean_dev.get(),
                              estVariance_dev.get(),
                              epsilon,
                              perf_helper,
                              hip_en);
        // Read the output
        output_ref = handle.Read<YDataType>(out_dev, output.data.size());
    }

    void RunTestCPU()
    { // Run the CPU implementation
        if(bn_config.mode == miopenBNPerActivation)
        {
            batchNormPerActivHostInference(
                input, ref_out, scale, shift, epsilon, estMean, estVariance);
        }
        else
        {
            batchNormSpatialHostInference(
                input, ref_out, scale, shift, epsilon, estMean, estVariance);
        }
        activationHostInfer(
            activ_mode, activ_gamma, activ_beta, activ_alpha, ref_out.data, ref_out.data);
    }

    void Verify()
    { // Compare the outputs
      // NOTE: Some small tensors during perf tests produce zero outputs which will result in
      // non-fatal gtest failures. These can be safely ignored. In this situation both the referene
      // and the gpu outputs will be zero. Observed them in relu, power and clippedrelu.
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "CPU/GPU data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "GPU data is all zeros";
        EXPECT_FALSE(miopen::find_idx(output, miopen::not_finite) >= 0)
            << "Non finite number found in the GPU data";
        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU/GPU data";
        auto error         = miopen::rms_range(ref_out, output);
        auto mantissa_bits = std::is_same<YDataType, float>::value              ? 23
                             : std::is_same<YDataType, half_float::half>::value ? 10
                                                                                : 7;
        auto threshold =
            std::max(2.0 * std::sqrt(input.data.size()) / (1 << 23), 1.0 / (1 << mantissa_bits));
        EXPECT_LE(error, threshold);
    }

    void TearDown() override
    {
        if constexpr(PERF_ENABLE)
        {
            // get the kernel handle
            auto&& handle = get_handle();

            // get the input tensor size and store in a string with x in between
            std::vector<size_t> in_dims = bn_config.GetInput();
            std::string kernel_info     = std::to_string(in_dims[0]) + "x" +
                                      std::to_string(in_dims[1]) + "x" +
                                      std::to_string(in_dims[2]) + "x" + std::to_string(in_dims[3]);

            std::unordered_map<miopenActivationMode_t, std::string> activation_map = {
                {miopenActivationPASTHRU, "pasthru"},
                {miopenActivationLOGISTIC, "logistic"},
                {miopenActivationTANH, "tanh"},
                {miopenActivationRELU, "relu"},
                {miopenActivationSOFTRELU, "softrelu"},
                {miopenActivationABS, "abs"},
                {miopenActivationPOWER, "power"},
                {miopenActivationCLIPPEDRELU, "clippedrelu"},
                {miopenActivationLEAKYRELU, "leakyrelu"},
                {miopenActivationELU, "elu"}};

            auto it = activation_map.find(activ_mode);
            if(it != activation_map.end())
            {
                kernel_info += "_" + it->second;
            }

            perf_helper.writeStatsToCSV(
                "batch-norm-activ-infer-perf-" + handle.GetDeviceName() + ".csv",
                "_" + kernel_info + "_" + (input.desc.GetType() == miopenHalf ? "FP16" : "FP32"));
        }
    }

    BNTestCase bn_config;      // Holds the test configuration
    tensor<XDataType> input;   // Input tensor
    tensor<YDataType> output;  // Output tensor from GPU
    tensor<YDataType> ref_out; // Reference output tensor
    tensor<ScaleDataType> scale;
    tensor<BiasDataType> shift;
    tensor<MeanVarDataType> estMean;
    tensor<MeanVarDataType> estVariance;
    miopen::Allocator::ManageDataPtr in_dev;          // GPU input data
    miopen::Allocator::ManageDataPtr out_dev;         // GPU output data
    miopen::Allocator::ManageDataPtr scale_dev;       // GPU scale data
    miopen::Allocator::ManageDataPtr shift_dev;       // GPU shift data
    miopen::Allocator::ManageDataPtr estMean_dev;     // GPU estimated mean data
    miopen::Allocator::ManageDataPtr estVariance_dev; // GPU estimated variance data
    miopenActivationMode_t activ_mode;                // Activation mode
    const float activ_alpha = static_cast<float>(0.5f);
    const float activ_beta  = static_cast<float>(0.5f);
    const float activ_gamma = static_cast<float>(0.5f);
    double epsilon          = 1.0e-5;
    // GetKernelTime returns time in float
    PerfHelper<float> perf_helper;
};

namespace BatchNormActivInfer {

struct GPU_bn_activ_infer_spatial_FP32 : BatchNormActivInferTest<float, float, float, float, float>
{
};

struct GPU_bn_activ_infer_per_act_FP32 : BatchNormActivInferTest<float, float, float, float, float>
{
};

struct GPU_bn_activ_infer_spatial_FP16
    : BatchNormActivInferTest<half_float::half, half_float::half, float, float, float>
{
};

struct GPU_bn_activ_infer_per_act_FP16
    : BatchNormActivInferTest<half_float::half, half_float::half, float, float, float>
{
};

} // namespace BatchNormActivInfer
using namespace BatchNormActivInfer;

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

template <typename T>
std::vector<BNTestCase> BNActivInferTestConfigs(miopenBatchNormMode_t mode)
{
    // create an array of input tensor shapes to test
    int shapes_to_test[10][4] = {{64, 128, 56, 56},
                                 {64, 2048, 7, 7},
                                 {64, 256, 14, 14},
                                 {64, 256, 28, 28},
                                 {64, 256, 56, 56},
                                 {64, 512, 14, 14},
                                 {64, 512, 28, 28},
                                 {64, 512, 7, 7},
                                 {64, 64, 112, 112},
                                 {64, 64, 56, 56}};

    // return a vector of BNTestCase objects created using the above shapes
    std::vector<BNTestCase> test_cases;
    for(auto& shape : shapes_to_test)
    {
        test_cases.push_back(BNTestCase{shape[0],
                                        shape[1],
                                        shape[2],
                                        shape[3],
                                        mode,
                                        miopen::batchnorm::Direction::ForwardInference,
                                        0,
                                        0});
    }

    return test_cases;
}

TEST_P(GPU_bn_activ_infer_spatial_FP32, PortTest)
{
#if COMPARE_WITH_OPENCL
    // Run the OpenCL implementation
    RunTestGPU(false);
#else
    // Run the CPU implementation
    RunTestCPU();
#endif
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_activ_infer_per_act_FP32, PortTest)
{
#if COMPARE_WITH_OPENCL
    // Run the OpenCL implementation
    RunTestGPU(false);
#else
    // Run the CPU implementation
    RunTestCPU();
#endif
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_activ_infer_spatial_FP16, PortTest)
{
#if COMPARE_WITH_OPENCL
    // Run the OpenCL implementation
    RunTestGPU(false);
#else
    // Run the CPU implementation
    RunTestCPU();
#endif
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

TEST_P(GPU_bn_activ_infer_per_act_FP16, PortTest)
{
#if COMPARE_WITH_OPENCL
    // Run the OpenCL implementation
    RunTestGPU(false);
#else
    // Run the CPU implementation
    RunTestCPU();
#endif
    // Run the HIP implementation
    RunTestGPU();
    // Compare the outputs
    Verify();
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_spatial_FP32,
    testing::Combine(testing::ValuesIn(ActivationConfigs()),
                     testing::ValuesIn(BNActivInferTestConfigs<float>(miopenBNSpatial))));
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_per_act_FP32,
    testing::Combine(testing::ValuesIn(ActivationConfigs()),
                     testing::ValuesIn(BNActivInferTestConfigs<float>(miopenBNPerActivation))));
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_spatial_FP16,
    testing::Combine(
        testing::ValuesIn(ActivationConfigs()),
        testing::ValuesIn(BNActivInferTestConfigs<half_float::half>(miopenBNSpatial))));
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_bn_activ_infer_per_act_FP16,
    testing::Combine(
        testing::ValuesIn(ActivationConfigs()),
        testing::ValuesIn(BNActivInferTestConfigs<half_float::half>(miopenBNPerActivation))));
