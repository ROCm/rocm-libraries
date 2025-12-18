/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "cpu_softmarginloss.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/softmarginloss.hpp>

#include "gtest_common.hpp"

namespace {

struct SoftMarginLossTestCaseNew
{
    // dim and stride of input tensor
    std::vector<size_t> dims;
    std::vector<size_t> strides;
    miopenLossReductionMode_t reduction_mode;

    friend std::ostream& operator<<(std::ostream& os, const SoftMarginLossTestCaseNew& tc)
    {
        os << "dims:";
        os << tc.dims[0];
        for(int i = 1; i < tc.dims.size(); i++)
            os << "x" << tc.dims[i];
        os << " strides:";
        os << tc.strides[0];
        for(int i = 1; i < tc.strides.size(); i++)
            os << "x" << tc.strides[i];
        os << " reduction_mode:" << tc.reduction_mode;
        return os;
    }
};

inline std::vector<SoftMarginLossTestCaseNew> SoftMarginLossTestConfigs(bool is_fwd,
                                                                        miopenDataType_t data_type)
{
    // clang-format off
    const std::vector<std::vector<size_t>> dim_config = { {256, 4, 8732}, {32, 80, 870}, {32, 80,
    870}, {4, 182403, 91}, {1534680}, {16, 1, 512, 512}, {2, 3, 160, 160}, {2, 3, 80, 80},
    {32756, 80},
          {64, 3, 80, 80}, {64, 3, 40, 40}, {22311, 80}, {64, 3, 20, 20}, {8, 4}, {56, 4}, {131,
          4}, {10000}, {200, 50}, {20, 50, 10}, {4, 25, 4, 25}, {12, 3, 4, 5, 6}, {10000}, {200,
          50}, {200, 50}, {20, 50, 10}, {20, 50, 10}, {4, 25, 4, 25}, {4, 25, 4, 25}, {12, 3, 4,
          5, 6}, {12, 3, 4, 5, 6} };
    const std::vector<std::vector<size_t>> stride_config =  { {34928, 8732, 1}, {69600, 1, 80},
    {69600, 870, 1}, {16598673, 91, 1}, {1}, {262144, 262144, 512, 1}, {6528000, 2176000, 13600,
    85}, {1632000, 544000, 6800, 85}, {85, 1},
          {1632000, 544000, 6800, 85}, {408000, 136000, 3400, 85}, {85, 1}, {102000, 34000, 1700,
          85}, {4, 1}, {4, 1}, {4, 1}, {1}, {50, 1}, {500, 10, 1}, {2500, 100, 25, 1}, {360, 120,
          30, 6, 1}, {3}, {1, 200}, {505, 1}, {1, 20, 1000}, {7575, 15, 1}, {1, 16, 4, 400},
          {5859, 217, 31, 1}, {360, 120, 6, 24, 1}, {5760, 960, 120, 12, 1} };
    std::vector<SoftMarginLossTestCaseNew> test_config;
    for (int i = 0; i < dim_config.size(); i++) test_config.push_back({dim_config[i], stride_config[i], MIOPEN_LOSS_REDUCTION_NONE}); 
    for (int i = 0; i < dim_config.size(); i++) {
        // Please note that with backward fp16 mean reduction, if input tensor is too big the result will be wrong because of fp16 underflow 
       size_t input_numel = 1; 
       for (auto x: dim_config[i]) input_numel *= x; 
       if (!is_fwd && data_type == miopenHalf && input_numel >= 320000) continue; 
       test_config.push_back({dim_config[i], stride_config[i], MIOPEN_LOSS_REDUCTION_MEAN});
    }
    for (int i = 0; i < dim_config.size(); i++) {
        // Please note that with forward fp16 sum reduction, if input tensor is too big the
        // result will be wrong because of fp16 overflow 
        size_t input_numel = 1; 
        for (auto x: dim_config[i]) input_numel *= x; 
        if (is_fwd && data_type == miopenHalf && input_numel >= 80000) continue; 
        test_config.push_back({dim_config[i], stride_config[i], MIOPEN_LOSS_REDUCTION_SUM});
    }
    return test_config;
    // clang-format on
}
} // namespace

template <typename T                   = float,
          miopenUnitUnderTest_t UUT    = miopenUnitNaiveGPU,
          miopenTestReference_t REF    = miopenTestReferenceOptimizedCPU,
          miopenAfterTestFailure_t ATF = miopenAfterTestFailureAnalyze>
struct SoftMarginLossForwardTestNew : public GTESTBase<UUT, REF, ATF>,
                                      public ::testing::TestWithParam<SoftMarginLossTestCaseNew>
{
protected:
    static void SetUpTestSuite()
    {
        if constexpr(!checkTestConfiguration(UUT, REF))
        {
            GTEST_SKIP() << "Test configuration is incorrect";
        }
    }

    void SetUp() override
    {
        auto&& handle         = get_handle();
        softmarginloss_config = GetParam();

        auto in_dims    = softmarginloss_config.dims;
        auto in_strides = softmarginloss_config.strides;
        reduction_mode  = softmarginloss_config.reduction_mode;

        // below commented code that I have seen in many files will not work correctly with
        // unpacked
        // tensor because tensor.generate() will call for_each() and this function only iterate
        // through desc.GetLengths().size(), not desc.GetElementSpace() Example input_tensor to
        // verify: input_tensor: dim(5, 3), stride(4, 1). Element space = 19, size = 15. Call
        // above
        // code will only generate value for 15 elements

        // input = tensor<T>{in_dims, in_strides}.generate(gen_in_value);

        // This is the right method to generate value for tensor
        input             = tensor<T>{in_dims, in_strides};
        auto gen_in_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-1), static_cast<T>(1));
        };
        std::generate(input.begin(), input.end(), gen_in_value);
        input_dev = handle.Write(input.data);

        auto gen_target_value = [](auto...) {
            return (prng::gen_A_to_B<int32_t>(0, 2) == 0) ? -1 : 1;
        };
        target = tensor<T>{in_dims, in_strides};
        std::generate(target.begin(), target.end(), gen_target_value);
        target_dev = handle.Write(target.data);

        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
        {
            output    = tensor<T>{in_dims, in_strides};
            oc_output = tensor<T>{in_dims, in_strides};
            nc_output = tensor<T>{in_dims, in_strides};
        }
        else
        {
            // Tensor with 1 element to store result after reduce
            output    = tensor<T>{std::vector<size_t>{1}};
            oc_output = tensor<T>{std::vector<size_t>{1}};
            nc_output = tensor<T>{std::vector<size_t>{1}};
        }
        std::fill(output.begin(), output.end(), 0);
        std::fill(oc_output.begin(), oc_output.end(), 0);
        std::fill(nc_output.begin(), nc_output.end(), 0);
        output_dev = handle.Write(output.data);

        ws_sizeInBytes = miopen::GetSoftMarginLossForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc, reduction_mode);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetMultiMarginLossForwardWorkspaceSize failed!";

        if(ws_sizeInBytes > 0)
        {
            workspace = tensor<float>{std::vector<size_t>{ws_sizeInBytes / sizeof(float)}};
            std::fill(workspace.begin(), workspace.end(), 0);
            workspace_dev = handle.Write(workspace.data);
        }
        else
        {
            workspace_dev = nullptr;
        }
    }

    void runOptimizedGPU() override { std::cout << "Optimized GPU not implemented" << std::endl; };
    void runNaiveGPU() override
    {
        auto&& handle = get_handle();
        miopen::SoftMarginLossForward(handle,
                                      workspace_dev.get(),
                                      ws_sizeInBytes,
                                      input.desc,
                                      input_dev.get(),
                                      target.desc,
                                      target_dev.get(),
                                      output.desc,
                                      output_dev.get(),
                                      reduction_mode);

        output.data = handle.Read<T>(output_dev, output.data.size());
    };
    void runOptimizedCPU() override
    {
        cpu_softmarginloss_forward_mt<T>(input, target, oc_output, reduction_mode);
    };
    void runNaiveCPU() override
    {
        cpu_softmarginloss_forward<T>(input, target, nc_output, reduction_mode);
    };

    void RunTest() { this->runTest(); }

    std::pair<bool, std::unordered_map<std::string, double>> verify() override
    {
        auto tolerance = std::numeric_limits<T>::epsilon() * 10;

        auto error = miopen::rms_range(*ref_output, *uut_output);
        EXPECT_EQ(miopen::range_distance(*ref_output), miopen::range_distance(*uut_output));
        EXPECT_LT(error, tolerance);

        if(error < tolerance)
        {
            return {true, {}};
        }
        else
        {
            return {false, {{"error", error}}};
        }
    }

    void setREFData() override
    {
        if(this->currentREF == miopenTestReferenceOptimizedCPU)
        {
            ref_output = &oc_output;
        }
        else
        {
            ref_output = &nc_output;
        }
    }

    void constexpr setUUTData() override
    {
        if(UUT == miopenUnitNaiveGPU)
        {
            uut_output = &output;
        }
        else
        {
            uut_output = &oc_output;
        }
    }

    SoftMarginLossTestCaseNew softmarginloss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<float> workspace;

    tensor<T> oc_output;
    tensor<T> nc_output;

    tensor<T>* ref_output;
    tensor<T>* uut_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    miopenLossReductionMode_t reduction_mode;
    size_t ws_sizeInBytes;
};

namespace softmarginloss {

struct GPU_SoftMarginLossForwardNew_FP32 : SoftMarginLossForwardTestNew<float>
{
};

struct GPU_SoftMarginLossForwardNew_FP16 : SoftMarginLossForwardTestNew<half_float::half>
{
};

struct GPU_SoftMarginLossForwardNew_BFP16 : SoftMarginLossForwardTestNew<bfloat16>
{
};

} // namespace softmarginloss

using namespace softmarginloss;

TEST_P(GPU_SoftMarginLossForwardNew_FP32, Test) { RunTest(); };

TEST_P(GPU_SoftMarginLossForwardNew_FP16, Test) { RunTest(); };

TEST_P(GPU_SoftMarginLossForwardNew_BFP16, Test) { RunTest(); };
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SoftMarginLossForwardNew_FP32,
                         testing::ValuesIn(SoftMarginLossTestConfigs(1, miopenFloat)));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SoftMarginLossForwardNew_FP16,
                         testing::ValuesIn(SoftMarginLossTestConfigs(1, miopenHalf)));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SoftMarginLossForwardNew_BFP16,
                         testing::ValuesIn(SoftMarginLossTestConfigs(1, miopenBFloat16)));
