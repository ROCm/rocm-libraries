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
#include "na.hpp"
#include "perf_helper.hpp"

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
struct BatchNormInferTester
    : public ::testing::TestWithParam<
          std::tuple<miopenActivationMode_t, BNTestCase, miopenTensorLayout_t>>
{
    void SetUp() override
    {
        std::tie(activ_mode, bn_config, tensor_layout) = GetParam();

        // Create tensors
        input              = tensor<XDataType>{tensor_layout, bn_config.GetInput()};
        output             = tensor<YDataType>{tensor_layout, bn_config.GetInput()};
        ref_out            = tensor<YDataType>{tensor_layout, bn_config.GetInput()};
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bn_config.mode);
        scale       = tensor<ScaleDataType>{tensor_layout, derivedBnDesc.GetLengths()};
        shift       = tensor<BiasDataType>{tensor_layout, derivedBnDesc.GetLengths()};
        estMean     = tensor<MeanVarDataType>{tensor_layout, derivedBnDesc.GetLengths()};
        estVariance = tensor<MeanVarDataType>{tensor_layout, derivedBnDesc.GetLengths()};
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
    }

    virtual void RunTestGPU() = 0;

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
        if(perf_enable)
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
                {miopenActivationELU, "elu"},
                {miopenActivationCLAMP, "clamp"}};

            auto it = activation_map.find(activ_mode);
            if(it != activation_map.end())
            {
                kernel_info += "_" + it->second;
            }

            perf_helper.writeStatsToCSV("batch-norm-infer-perf-" + handle.GetDeviceName() + ".csv",
                                        "_" + input.desc.GetLayout_str() + "_" + kernel_info + "_" +
                                            ((input.desc.GetType() == miopenHalf)       ? "FP16"
                                             : (input.desc.GetType() == miopenBFloat16) ? "BFP16"
                                                                                        : "FP32"));
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
    miopenTensorLayout_t tensor_layout;               // Tensor layout
    const float activ_alpha = 0.5f;
    const float activ_beta  = 0.5f;
    const float activ_gamma = 0.5f;
    double epsilon          = 1.0e-5;
    bool perf_enable        = false;
    PerfHelper perf_helper;
};

template <typename T>
std::vector<BNTestCase> BNInferTestConfigs(miopenBatchNormMode_t mode)
{
    // create an array of input tensor shapes to test
    int shapes_to_test[10][4] = {// from resnet50
                                 {64, 128, 56, 56},
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

struct TestNameGenerator
{
    std::string
    operator()(const testing::TestParamInfo<
               std::tuple<miopenActivationMode_t, BNTestCase, miopenTensorLayout_t>>& info) const
    {
        // activation mode
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
            {miopenActivationELU, "elu"},
            {miopenActivationCLAMP, "clamp"}};
        auto activation_it = activation_map.find(std::get<0>(info.param));
        std::string activation_mode_str =
            (activation_it != activation_map.end()) ? activation_it->second : "unknown";

        // bn configuration
        auto bn_config = std::get<1>(info.param);
        std::string bn_mode_str =
            (bn_config.mode == miopenBNSpatial) ? "BNSpatial" : "BNPerActivation";
        std::string tensor_dims = "N" + std::to_string(bn_config.N) + "_C" +
                                  std::to_string(bn_config.C) + "_H" + std::to_string(bn_config.H) +
                                  "_W" + std::to_string(bn_config.W);

        // tensor layout
        miopenTensorLayout_t tensor_layout = std::get<2>(info.param);
        std::string tensor_layout_str      = (tensor_layout == miopenTensorNCHW)   ? "NCHW"
                                             : (tensor_layout == miopenTensorNHWC) ? "NHWC"
                                                                                   : "UnknownLayout";

        std::ostringstream oss;
        oss << bn_mode_str + "_" + tensor_layout_str + "_" + tensor_dims + "_" +
                   activation_mode_str + "_test_id_" + std::to_string(info.index);
        return oss.str();
    }
};
