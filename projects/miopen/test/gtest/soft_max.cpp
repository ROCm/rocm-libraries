// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "get_handle.hpp"
#include "miopen/miopen.h"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/softmax.hpp>

#define NEGATIVE_CUTOFF_VAL_FP32 (-1e20)
#define NEGATIVE_CUTOFF_VAL_FP16 (-1e4)

namespace {

template <typename T>
T logaddexp(T x, T y, T neg_inf)
{
    T a = std::max(x, y);
    T b = std::min(x, y);
    T c = b - a;

    return c <= neg_inf ? std::max(a, neg_inf) : std::max(T(a + log(T(1) + exp(b - a))), neg_inf);
}

struct TestCase
{
    std::vector<size_t> in_dim;
    std::vector<float> scale;
    miopenSoftmaxAlgorithm_t algo;
    miopenSoftmaxMode_t mode;
    miopenTensorLayout_t layout;
};

std::string PrintToString(const TestCase& test_case)
{
    std::stringstream ss;
    ss << "{in_dim = {";
    for(auto i = 0; i + 1 < test_case.in_dim.size(); ++i)
    {
        ss << test_case.in_dim[i] << ", ";
    }
    if(test_case.in_dim.size() > 0)
    {
        ss << test_case.in_dim[test_case.in_dim.size() - 1];
    }
    ss << "}, scale = {";
    for(auto i = 0; i + 1 < test_case.scale.size(); ++i)
    {
        ss << test_case.scale[i] << ", ";
    }
    if(test_case.scale.size() > 0)
    {
        ss << test_case.scale[test_case.scale.size() - 1];
    }
    ss << "}, algo = " << test_case.algo << ", mode = " << test_case.mode
       << ", layout = " << (test_case.layout == miopenTensorNCHW ? "NCHW" : "NHWC") << "}";
    return ss.str();
}

template <typename T>
void AddTestCasesForDifferentScales(std::vector<TestCase>& test_cases,
                                    const std::vector<size_t>& in_dim,
                                    int algo,
                                    int mode,
                                    miopenTensorLayout_t layout,
                                    const std::vector<std::vector<float>>& scales)
{
    // Result does not fit in data type
    if((miopen_type<T>{} == miopenHalf || miopen_type<T>{} == miopenBFloat16) &&
       in_dim[1] * in_dim[2] * in_dim[3] >= 2048 && mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
    {
        return;
    }

    for(const auto& scale : scales)
    {
        TestCase& test_case = test_cases.emplace_back();

        test_case.in_dim = in_dim;
        test_case.algo   = static_cast<miopenSoftmaxAlgorithm_t>(algo);
        test_case.mode   = static_cast<miopenSoftmaxMode_t>(mode);
        test_case.layout = layout;
        test_case.scale  = scale;
    }
}

template <typename T>
std::vector<TestCase> GenCases()
{
    int batch_factor = 0;

    std::set<std::vector<size_t>> in_dim_set = get_inputs<size_t>(batch_factor);

    std::vector<int> algos                    = {0, 1, 2};
    std::vector<int> modes                    = {0, 1};
    std::vector<std::vector<float>> scales    = {{1.0f, 0.0f}, {0.5f, 0.5f}};
    std::vector<miopenTensorLayout_t> layouts = {miopenTensorNCHW, miopenTensorNHWC};

    std::vector<TestCase> test_cases;

    for(const auto& in_dim : in_dim_set)
    {
        for(const int algo : algos)
        {
            for(const int mode : modes)
            {
                for(const miopenTensorLayout_t layout : layouts)
                    AddTestCasesForDifferentScales<T>(
                        test_cases, in_dim, algo, mode, layout, scales);
            }
        }
    }

    return test_cases;
}

template <typename T>
auto GetCases()
{
    static const auto cases = testing::ValuesIn(GenCases<T>());
    return cases;
}

} // namespace

template <typename T>
struct SoftmaxCommon : public testing::TestWithParam<TestCase>
{
    void SetUp() override { prng::reset_seed(); }

    void Run()
    {
        const TestCase& test_case = GetParam();

        uint64_t max_value = miopen_type<T>{} == miopenFloat        ? 17
                             : test_case.algo == MIOPEN_SOFTMAX_LOG ? 3
                                                                    : 5;

        input = tensor<T>{test_case.layout, test_case.in_dim}.generate(
            tensor_elem_gen_integer{max_value});
        size_t total_mem  = 2 * input.desc.GetNumBytes(); // estimate based on backward pass
        size_t device_mem = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;

            GTEST_SKIP();
        }

        output = tensor<T>{test_case.layout, test_case.in_dim}.generate(
            tensor_elem_gen_integer{max_value});

        std::vector<T> tensorCpuDataForward = GetForwardCpu();
        std::vector<T> tensorGpuDataForward = GetForwardGpu();

        // check forward results
        CompareResults(tensorGpuDataForward, tensorCpuDataForward, true);

        dout =
            tensor<T>{test_case.layout, test_case.in_dim}.generate([&](int n, int c, int h, int w) {
                T x = input(n, c, h, w);
                double y =
                    (877 * n + 547 * c + 701 * h + 1049 * w + static_cast<int>(769 * x)) % 2503;
                return ((x * y) / 1301.0);
            });
        dinput = tensor<T>{test_case.layout, test_case.in_dim}.generate(
            tensor_elem_gen_integer{max_value});

        std::vector<T> tensorCpuDataBackward = GetBackwardCpu();
        std::vector<T> tensorGpuDataBackward = GetBackwardGpu();

        // check backward results
        CompareResults(tensorGpuDataBackward, tensorCpuDataBackward, false);
    }

    std::vector<T> GetForwardCpu() const
    {
        const TestCase& test_case = GetParam();

        auto out = output;

        const auto [in_n, in_c, in_h, in_w] = miopen::tien<4>(input.desc.GetLengths());

        const auto [in_nstr, in_cstr, in_hstr, in_wstr] = miopen::tien<4>(input.desc.GetStrides());

        const auto [out_nstr, out_cstr, out_hstr, out_wstr] =
            miopen::tien<4>(out.desc.GetStrides());

        float alpha = test_case.scale[0];
        float beta  = test_case.scale[1];

        if(test_case.mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        {
            miopen::par_ford(in_n)([&](int o) {
                if(test_case.algo == MIOPEN_SOFTMAX_FAST)
                {
                    double sum = 0;
                    miopen::ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        sum +=
                            std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr]);
                    });
                    miopen::ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr] =
                            alpha *
                                (std::exp(
                                     input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr]) /
                                 sum) +
                            beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr];
                    });
                }
                else
                {
                    T max_c = std::numeric_limits<T>::lowest();
                    miopen::ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        max_c = std::max(
                            max_c, input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr]);
                    });

                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        double neg_inf = input.desc.GetType() == miopenFloat
                                             ? NEGATIVE_CUTOFF_VAL_FP32
                                             : NEGATIVE_CUTOFF_VAL_FP16;
                        double sum     = neg_inf;
                        miopen::ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            sum = logaddexp(
                                double(
                                    input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr] -
                                    max_c),
                                sum,
                                neg_inf);
                        });

                        miopen::ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr] =
                                alpha *
                                    (input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr] -
                                     max_c - sum) +
                                beta *
                                    out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr];
                        });
                    }
                    else
                    {
                        double sum = 0;
                        miopen::ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            sum += std::exp(
                                input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr] -
                                max_c);
                        });

                        miopen::ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr] =
                                alpha * (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr +
                                                        j * in_wstr] -
                                                  max_c) /
                                         sum) +
                                beta *
                                    out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr];
                        });
                    }
                }
            });
        }
        else
        {
            miopen::par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
                if(test_case.algo == MIOPEN_SOFTMAX_FAST)
                {
                    double sum = 0;
                    miopen::ford(in_c)([&](int w) {
                        sum +=
                            std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr]);
                    });
                    miopen::ford(in_c)([&](int w) {
                        out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr] =
                            alpha *
                                (std::exp(
                                     input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr]) /
                                 sum) +
                            beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr];
                    });
                }
                else
                {
                    T max_c = std::numeric_limits<T>::lowest();
                    miopen::ford(in_c)([&](int w) {
                        max_c = std::max(
                            max_c, input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr]);
                    });

                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        double neg_inf = input.desc.GetType() == miopenFloat
                                             ? NEGATIVE_CUTOFF_VAL_FP32
                                             : NEGATIVE_CUTOFF_VAL_FP16;
                        double sum     = neg_inf;
                        miopen::ford(in_c)([&](int w) {
                            sum = logaddexp(
                                double(
                                    input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr] -
                                    max_c),
                                sum,
                                neg_inf);
                        });

                        miopen::ford(in_c)([&](int w) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr] =
                                alpha *
                                    (input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr] -
                                     max_c - sum) +
                                beta *
                                    out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr];
                        });
                    }
                    else
                    {
                        double sum = 0;
                        miopen::ford(in_c)([&](int w) {
                            sum += std::exp(
                                input[o * in_nstr + w * in_cstr + i * in_hstr + j * in_wstr] -
                                max_c);
                        });

                        miopen::ford(in_c)([&](int w) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr] =
                                alpha * (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr +
                                                        j * in_wstr] -
                                                  max_c) /
                                         sum) +
                                beta *
                                    out[o * out_nstr + w * out_cstr + i * out_hstr + j * out_wstr];
                        });
                    }
                }
            });
        }
        return out.data;
    }

    std::vector<T> GetForwardGpu() const
    {
        const TestCase& test_case = GetParam();
        auto&& handle             = get_handle();

        auto in_dev  = handle.Write(input.data);
        auto out_dev = handle.Write(output.data);

        miopen::SoftmaxForward(handle,
                               &test_case.scale[0],
                               &test_case.scale[1],
                               input.desc,
                               in_dev.get(),
                               output.desc,
                               out_dev.get(),
                               test_case.algo,
                               test_case.mode);

        return handle.Read<T>(out_dev, output.data.size());
    }

    std::vector<T> GetBackwardCpu() const
    {
        const TestCase& test_case = GetParam();

        auto din = dinput;

        const auto [in_n, in_c, in_h, in_w] = miopen::tien<4>(din.desc.GetLengths());

        const auto [in_nstr, in_cstr, in_hstr, in_wstr] = miopen::tien<4>(din.desc.GetStrides());

        const auto [out_nstr, out_cstr, out_hstr, out_wstr] =
            miopen::tien<4>(dout.desc.GetStrides());

        float alpha = test_case.scale[0];
        float beta  = test_case.scale[1];

        if(test_case.mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        {
            miopen::par_ford(in_n)([&](int o) {
                double sum = 0;
                miopen::ford(in_c, in_h, in_w)([&](int c, int i, int j) {
                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        sum += dout[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr];
                    }
                    else
                    {
                        sum += output[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr] *
                               dout[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr];
                    }
                });

                miopen::ford(in_c, in_h, in_w)([&](int c, int i, int j) {
                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j * in_wstr] =
                            T(alpha *
                                  (dout[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr] -
                                   sum * std::exp(output[o * out_nstr + c * out_cstr +
                                                         i * out_hstr + j * out_wstr])) +
                              beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j * in_wstr]);
                    }
                    else
                    {
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j * in_wstr] =
                            alpha *
                                (output[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr] *
                                 (dout[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr] -
                                  sum)) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j * in_wstr];
                    }
                });
            });
        }
        else
        {
            miopen::par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
                double sum = 0;
                miopen::ford(in_c)([&](int c) {
                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        sum += dout[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr];
                    }
                    else
                    {
                        sum += output[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr] *
                               dout[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr];
                    }
                });

                miopen::ford(in_c)([&](int c) {
                    if(test_case.algo == MIOPEN_SOFTMAX_LOG)
                    {
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j * in_wstr] =
                            alpha *
                                (dout[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr] -
                                 sum * std::exp(output[o * out_nstr + c * out_cstr + i * out_hstr +
                                                       j * out_wstr])) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j * in_wstr];
                    }
                    else
                    {
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j * in_wstr] =
                            alpha *
                                (output[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr] *
                                 (dout[o * out_nstr + c * out_cstr + i * out_hstr + j * out_wstr] -
                                  sum)) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j * in_wstr];
                    }
                });
            });
        }
        return din.data;
    }

    std::vector<T> GetBackwardGpu() const
    {
        const TestCase& test_case = GetParam();

        auto&& handle = get_handle();
        // auto din      = dinput;

        auto din_dev  = handle.Write(dinput.data);
        auto dout_dev = handle.Write(dout.data);
        auto out_dev  = handle.Write(output.data);

        miopen::SoftmaxBackward(handle,
                                &test_case.scale[0],
                                output.desc,
                                out_dev.get(),
                                dout.desc,
                                dout_dev.get(),
                                &test_case.scale[1],
                                dinput.desc,
                                din_dev.get(),
                                test_case.algo,
                                test_case.mode);

        return handle.Read<T>(din_dev, dinput.data.size());
    }

    void CompareResults(const std::vector<T>& tensorGPUData,
                        const std::vector<T>& tensorCPUData,
                        bool isForward)
    {
        const TestCase& test_case = GetParam();

        // float tolerance taken from the original c test
        // cppcheck can't properly handle this trinary
        // cppcheck-suppress assignBoolToFloat
        double tolerance = std::is_same_v<T, bfloat16>           ? 10
                           : std::is_same_v<T, half_float::half> ? 80
                                                                 : 8000;

        double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        double error     = miopen::rms_range(tensorCPUData, tensorGPUData);

        ASSERT_LE(error, threshold)
            << "Tensor Dims: " << test_case.in_dim[0] << ", " << test_case.in_dim[1] << ", "
            << test_case.in_dim[2] << ", " << test_case.in_dim[3] << ", "
            << "Layout: " << (test_case.layout == miopenTensorNCHW ? "NCHW" : "NHWC") << ", "
            << "Alpha / Beta: " << test_case.scale[0] << ", " << test_case.scale[1]
            << ". Algo: " << test_case.algo << ". Mode: " << test_case.mode
            << ". Direction: " << (isForward ? "Forward" : "Backward") << std::endl;
    }

private:
    tensor<T> input;
    tensor<T> output;

    tensor<T> dinput;
    tensor<T> dout;
};

// Regression test: when beta=0 the kernel must not read the output buffer,
// even if it contains NaN. Pre-poisoning y (forward) and dx (backward) with NaN exposes
// the bug: without the guard, NaN * 0.0 == NaN propagates into the result.
struct GPU_Softmax_BetaZeroNaN_FP32
    : public testing::TestWithParam<std::tuple<miopenSoftmaxAlgorithm_t, miopenSoftmaxMode_t>>
{
    void RunForward()
    {
        auto&& handle                  = get_handle();
        auto [algo, mode]              = GetParam();
        const std::vector<size_t> dims = {2, 8, 4, 4};
        const float alpha = 1.0f, beta = 0.0f;

        auto input  = tensor<float>{miopenTensorNCHW, dims}.generate(tensor_elem_gen_integer{5});
        auto output = tensor<float>{miopenTensorNCHW, dims};
        std::fill(output.data.begin(), output.data.end(), std::numeric_limits<float>::quiet_NaN());

        auto in_dev  = handle.Write(input.data);
        auto out_dev = handle.Write(output.data);
        miopen::SoftmaxForward(handle,
                               &alpha,
                               &beta,
                               input.desc,
                               in_dev.get(),
                               output.desc,
                               out_dev.get(),
                               algo,
                               mode);

        auto result = handle.Read<float>(out_dev, output.data.size());
        for(std::size_t i = 0; i < result.size(); ++i)
            EXPECT_TRUE(std::isfinite(result[i])) << "NaN in forward output at index " << i;
    }

    void RunBackward()
    {
        auto&& handle                  = get_handle();
        auto [algo, mode]              = GetParam();
        const std::vector<size_t> dims = {2, 8, 4, 4};
        const float alpha = 1.0f, beta = 0.0f;

        auto output = tensor<float>{miopenTensorNCHW, dims}.generate(tensor_elem_gen_integer{5});
        auto dout   = tensor<float>{miopenTensorNCHW, dims}.generate(tensor_elem_gen_integer{5});
        auto dinput = tensor<float>{miopenTensorNCHW, dims};
        std::fill(dinput.data.begin(), dinput.data.end(), std::numeric_limits<float>::quiet_NaN());

        auto out_dev  = handle.Write(output.data);
        auto dout_dev = handle.Write(dout.data);
        auto din_dev  = handle.Write(dinput.data);
        miopen::SoftmaxBackward(handle,
                                &alpha,
                                output.desc,
                                out_dev.get(),
                                dout.desc,
                                dout_dev.get(),
                                &beta,
                                dinput.desc,
                                din_dev.get(),
                                algo,
                                mode);

        auto result = handle.Read<float>(din_dev, dinput.data.size());
        for(std::size_t i = 0; i < result.size(); ++i)
            EXPECT_TRUE(std::isfinite(result[i])) << "NaN in backward output at index " << i;
    }
};

using GPU_Softmax_FP32  = SoftmaxCommon<float>;
using GPU_Softmax_FP16  = SoftmaxCommon<half_float::half>;
using GPU_Softmax_BFP16 = SoftmaxCommon<bfloat16>;

TEST_P(GPU_Softmax_FP32, TestFloat) { this->Run(); }
TEST_P(GPU_Softmax_FP16, TestFloat16) { this->Run(); }
TEST_P(GPU_Softmax_BFP16, TestBFloat16) { this->Run(); }
TEST_P(GPU_Softmax_BetaZeroNaN_FP32, ForwardTest) { RunForward(); }
TEST_P(GPU_Softmax_BetaZeroNaN_FP32, BackwardTest) { RunBackward(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_FP32, GetCases<float>());
INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_FP16, GetCases<half_float::half>());
INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_BFP16, GetCases<bfloat16>());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Softmax_BetaZeroNaN_FP32,
                         testing::Combine(testing::Values(MIOPEN_SOFTMAX_FAST,
                                                          MIOPEN_SOFTMAX_ACCURATE,
                                                          MIOPEN_SOFTMAX_LOG),
                                          testing::Values(MIOPEN_SOFTMAX_MODE_INSTANCE,
                                                          MIOPEN_SOFTMAX_MODE_CHANNEL)));
