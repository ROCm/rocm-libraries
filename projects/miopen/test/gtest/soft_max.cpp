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

// --- Misaligned int4 vectorized load/store -------------------------
// The vectorized fast path casts &src[i + offset] to int4* (16 bytes). With
// stride==1 the per-block base offset is o * INNER_SIZE; this is 16-byte
// aligned only when INNER_SIZE is a multiple of load_factor (4 for fp32,
// 8 for fp16). For inner sizes that are NOT a multiple, the odd-o blocks
// could issue a final vector load/store that could be 16-byte-misaligned
// -> UB (memory fault or wrong result).
// Reuses SoftmaxCommon's full CPU-vs-GPU comparison.
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Softmax_FP32,
                         testing::Values(
                             // INSTANCE inner = C*H*W, none a multiple of 4:
                             TestCase{{2, 50, 1, 1},
                                      {1.0f, 0.0f},
                                      MIOPEN_SOFTMAX_ACCURATE,
                                      MIOPEN_SOFTMAX_MODE_INSTANCE,
                                      miopenTensorNCHW},
                             TestCase{{3, 10, 1, 1},
                                      {1.0f, 0.0f},
                                      MIOPEN_SOFTMAX_ACCURATE,
                                      MIOPEN_SOFTMAX_MODE_INSTANCE,
                                      miopenTensorNCHW},
                             TestCase{{2, 6, 1, 1},
                                      {0.5f, 0.5f},
                                      MIOPEN_SOFTMAX_ACCURATE,
                                      MIOPEN_SOFTMAX_MODE_INSTANCE,
                                      miopenTensorNCHW},
                             TestCase{{2, 50, 1, 1},
                                      {1.0f, 0.0f},
                                      MIOPEN_SOFTMAX_LOG,
                                      MIOPEN_SOFTMAX_MODE_INSTANCE,
                                      miopenTensorNCHW},
                             // NHWC-channel: inner = C, stride = 1, outer = N*H*W:
                             TestCase{{2, 50, 1, 1},
                                      {1.0f, 0.0f},
                                      MIOPEN_SOFTMAX_ACCURATE,
                                      MIOPEN_SOFTMAX_MODE_CHANNEL,
                                      miopenTensorNHWC}));

// fp16/bf16 elements are 2 bytes, so an ODD inner size makes the odd-o base
// offset 2-byte aligned -> the int4 (dwordx4) load is below the 4-byte hardware
// alignment CDNA requires. (An EVEN fp16 inner size is still >=4-byte aligned
// and does NOT fault on gfx942, same as fp32 -- so odd sizes are the real test.)
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Softmax_FP16,
                         testing::Values(TestCase{{2, 13, 1, 1},
                                                  {1.0f, 0.0f},
                                                  MIOPEN_SOFTMAX_ACCURATE,
                                                  MIOPEN_SOFTMAX_MODE_INSTANCE,
                                                  miopenTensorNCHW},
                                         TestCase{{2, 21, 1, 1},
                                                  {1.0f, 0.0f},
                                                  MIOPEN_SOFTMAX_ACCURATE,
                                                  MIOPEN_SOFTMAX_MODE_INSTANCE,
                                                  miopenTensorNCHW},
                                         TestCase{{3, 101, 1, 1},
                                                  {1.0f, 0.0f},
                                                  MIOPEN_SOFTMAX_ACCURATE,
                                                  MIOPEN_SOFTMAX_MODE_INSTANCE,
                                                  miopenTensorNCHW}));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Softmax_BFP16,
                         testing::Values(TestCase{{2, 13, 1, 1},
                                                  {1.0f, 0.0f},
                                                  MIOPEN_SOFTMAX_ACCURATE,
                                                  MIOPEN_SOFTMAX_MODE_INSTANCE,
                                                  miopenTensorNCHW},
                                         TestCase{{2, 21, 1, 1},
                                                  {1.0f, 0.0f},
                                                  MIOPEN_SOFTMAX_ACCURATE,
                                                  MIOPEN_SOFTMAX_MODE_INSTANCE,
                                                  miopenTensorNCHW}));

// --- Non-contiguous (non-packed) tensor ----------------------------
// GetStride/GetOuterSize/GetInnerSize derive geometry from lengths + layout
// enum only and ignore the descriptor's real strides. A prior implementation passed the
// real N/C/H/W strides into the kernel (with IS_*_CONTIGUOUS handling); the new
// code assumes a packed layout, so a padded (non-packed) tensor is addressed
// wrongly. Here the N stride is padded to 32 (packed would be 16).
struct GPU_Softmax_NonContiguous_FP32
    : public testing::TestWithParam<std::tuple<std::tuple<std::vector<size_t>, std::vector<size_t>>,
                                               miopenSoftmaxAlgorithm_t,
                                               miopenSoftmaxMode_t,
                                               int,
                                               int,
                                               int>>
{
    void RunForward()
    {
        auto&& handle                                                         = get_handle();
        auto [tensorDimsStrides, algo, mode, xdx_offset, y_offset, dy_offset] = GetParam();
        auto [dims, strides]                                                  = tensorDimsStrides;

        auto in_host   = tensor<float>{dims, strides}.generate(tensor_elem_gen_integer{5});
        size_t n_elems = dims[0] * strides[0];

        std::vector<float> xbuf(n_elems + xdx_offset, 1e30f);
        std::vector<float> ybuf(n_elems + y_offset, -42.0f);

        // Poison every element (incl. the 16-element inter-batch padding), then set
        // only the real elements. If the kernel assumes a packed layout it will read
        // the poisoned padding and/or write batch 1 to the wrong offset.
        auto off = [&](size_t n, size_t c, size_t h, size_t w) {
            return n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
        };
        for(size_t n = 0; n < dims[0]; ++n)
            for(size_t c = 0; c < dims[1]; ++c)
                for(size_t h = 0; h < dims[2]; ++h)
                    for(size_t w = 0; w < dims[3]; ++w)
                        xbuf[off(n, c, h, w) + xdx_offset] =
                            static_cast<float>((n * 13 + c * 7 + h * 3 + w + 1) % 5);

        const float alpha = 1.0f, beta = 0.0f;
        auto in_dev  = handle.Write(xbuf);
        auto out_dev = handle.Write(ybuf);

        miopen::SoftmaxForward(handle,
                               &alpha,
                               &beta,
                               in_host.desc,
                               in_dev.get(),
                               in_host.desc,
                               out_dev.get(),
                               algo,
                               mode,
                               xdx_offset,
                               y_offset);
        auto res = handle.Read<float>(out_dev, ybuf.size());

        // CPU reference over the real strides
        double max_err = 0.0;
        for(size_t n = 0; n < dims[0]; ++n)
        {
            if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
            {
                float mx = 0.0f;
                if(algo != MIOPEN_SOFTMAX_FAST)
                {
                    mx = std::numeric_limits<float>::lowest();
                    for(size_t c = 0; c < dims[1]; ++c)
                    {
                        for(size_t h = 0; h < dims[2]; ++h)
                        {
                            for(size_t w = 0; w < dims[3]; ++w)
                            {
                                mx = std::max(mx, xbuf[off(n, c, h, w) + xdx_offset]);
                            }
                        }
                    }
                }
                double sum = algo == MIOPEN_SOFTMAX_LOG ? NEGATIVE_CUTOFF_VAL_FP32 : 0.0;
                for(size_t c = 0; c < dims[1]; ++c)
                {
                    for(size_t h = 0; h < dims[2]; ++h)
                    {
                        for(size_t w = 0; w < dims[3]; ++w)
                        {
                            if(algo == MIOPEN_SOFTMAX_LOG)
                            {
                                sum = logaddexp<double>(sum,
                                                        xbuf[off(n, c, h, w) + xdx_offset] - mx,
                                                        NEGATIVE_CUTOFF_VAL_FP32);
                            }
                            else
                            {
                                sum += std::exp(xbuf[off(n, c, h, w) + xdx_offset] - mx);
                            }
                        }
                    }
                }
                for(size_t c = 0; c < dims[1]; ++c)
                {
                    for(size_t h = 0; h < dims[2]; ++h)
                    {
                        for(size_t w = 0; w < dims[3]; ++w)
                        {
                            double ref =
                                algo == MIOPEN_SOFTMAX_LOG
                                    ? xbuf[off(n, c, h, w) + xdx_offset] - mx - sum
                                    : std::exp(xbuf[off(n, c, h, w) + xdx_offset] - mx) / sum;
                            double err = std::abs(ref - res[off(n, c, h, w) + y_offset]);
                            if(ref != 0.0)
                            {
                                err = std::min(err, std::abs(err / ref));
                            }
                            max_err = std::max(max_err, err);
                        }
                    }
                }
            }
            else
            {
                for(size_t h = 0; h < dims[2]; ++h)
                {
                    for(size_t w = 0; w < dims[3]; ++w)
                    {
                        float mx = 0.0f;
                        if(algo != MIOPEN_SOFTMAX_FAST)
                        {
                            mx = std::numeric_limits<float>::lowest();
                            for(size_t c = 0; c < dims[1]; ++c)
                            {
                                mx = std::max(mx, xbuf[off(n, c, h, w) + xdx_offset]);
                            }
                        }
                        double sum = algo == MIOPEN_SOFTMAX_LOG ? NEGATIVE_CUTOFF_VAL_FP32 : 0.0;
                        for(size_t c = 0; c < dims[1]; ++c)
                        {
                            if(algo == MIOPEN_SOFTMAX_LOG)
                            {
                                sum = logaddexp<double>(sum,
                                                        xbuf[off(n, c, h, w) + xdx_offset] - mx,
                                                        NEGATIVE_CUTOFF_VAL_FP32);
                            }
                            else
                            {
                                sum += std::exp(xbuf[off(n, c, h, w) + xdx_offset] - mx);
                            }
                        }
                        for(size_t c = 0; c < dims[1]; ++c)
                        {
                            double ref =
                                algo == MIOPEN_SOFTMAX_LOG
                                    ? xbuf[off(n, c, h, w) + xdx_offset] - mx - sum
                                    : std::exp(xbuf[off(n, c, h, w) + xdx_offset] - mx) / sum;
                            double err = std::abs(ref - res[off(n, c, h, w) + y_offset]);
                            if(ref != 0.0)
                            {
                                err = std::min(err, std::abs(err / ref));
                            }
                            max_err = std::max(max_err, err);
                        }
                    }
                }
            }
        }
        EXPECT_LT(max_err, 1e-4) << "Non-packed tensor mis-addressed: GPU output does not "
                                    "match strided CPU reference (max abs err "
                                 << max_err << ").";
    }

    void RunBackward()
    {
        auto&& handle                                                         = get_handle();
        auto [tensorDimsStrides, algo, mode, xdx_offset, y_offset, dy_offset] = GetParam();
        auto [dims, strides]                                                  = tensorDimsStrides;

        auto in_host   = tensor<float>{dims, strides}.generate(tensor_elem_gen_integer{5});
        size_t n_elems = dims[0] * strides[0];

        std::vector<float> dybuf(n_elems + dy_offset, -42.0f);
        std::vector<float> ybuf(n_elems + y_offset, 42.0f);
        std::vector<float> dxbuf(n_elems + xdx_offset, 1e30f);

        // Poison every element (incl. the 16-element inter-batch padding), then set
        // only the real elements. If the kernel assumes a packed layout it will read
        // the poisoned padding and/or write batch 1 to the wrong offset.
        auto off = [&](size_t n, size_t c, size_t h, size_t w) {
            return n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
        };
        for(size_t n = 0; n < dims[0]; ++n)
        {
            for(size_t c = 0; c < dims[1]; ++c)
            {
                for(size_t h = 0; h < dims[2]; ++h)
                {
                    for(size_t w = 0; w < dims[3]; ++w)
                    {
                        dybuf[off(n, c, h, w) + dy_offset] =
                            static_cast<float>((n * 13 + c * 7 + h * 3 + w + 1) % 5);
                        ybuf[off(n, c, h, w) + y_offset] =
                            static_cast<float>((n * 17 + c * 11 + h * 5 + w + 3) % 7);
                    }
                }
            }
        }

        const float alpha = 1.0f, beta = 0.0f;
        auto dy_dev = handle.Write(dybuf);
        auto y_dev  = handle.Write(ybuf);
        auto dx_dev = handle.Write(dxbuf);

        miopen::SoftmaxBackward(handle,
                                &alpha,
                                in_host.desc,
                                y_dev.get(),
                                in_host.desc,
                                dy_dev.get(),
                                &beta,
                                in_host.desc,
                                dx_dev.get(),
                                algo,
                                mode,
                                y_offset,
                                dy_offset,
                                xdx_offset);
        auto res = handle.Read<float>(dx_dev, dxbuf.size());

        // CPU reference over the real strides
        double max_err = 0.0;
        for(size_t n = 0; n < dims[0]; ++n)
        {
            if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
            {
                double channel_dot = 0.0;
                for(size_t c = 0; c < dims[1]; ++c)
                {
                    for(size_t h = 0; h < dims[2]; ++h)
                    {
                        for(size_t w = 0; w < dims[3]; ++w)
                        {
                            float tmp = dybuf[off(n, c, h, w) + dy_offset];
                            if(algo != MIOPEN_SOFTMAX_LOG)
                            {
                                tmp *= ybuf[off(n, c, h, w) + y_offset];
                            }
                            channel_dot += tmp;
                        }
                    }
                }
                for(size_t c = 0; c < dims[1]; ++c)
                {
                    for(size_t h = 0; h < dims[2]; ++h)
                    {
                        for(size_t w = 0; w < dims[3]; ++w)
                        {
                            double ref = dybuf[off(n, c, h, w) + dy_offset];
                            if(algo == MIOPEN_SOFTMAX_LOG)
                            {
                                ref -= channel_dot * exp(ybuf[off(n, c, h, w) + y_offset]);
                            }
                            else
                            {
                                ref = (ref - channel_dot) * ybuf[off(n, c, h, w) + y_offset];
                            }
                            double err = std::abs(ref - res[off(n, c, h, w) + xdx_offset]);
                            if(ref != 0.0)
                            {
                                err = std::min(err, std::abs(err / ref));
                            }
                            max_err = std::max(max_err, err);
                        }
                    }
                }
            }
            else
            {
                for(size_t h = 0; h < dims[2]; ++h)
                {
                    for(size_t w = 0; w < dims[3]; ++w)
                    {
                        double channel_dot = 0.0;
                        for(size_t c = 0; c < dims[1]; ++c)
                        {
                            float tmp = dybuf[off(n, c, h, w) + dy_offset];
                            if(algo != MIOPEN_SOFTMAX_LOG)
                            {
                                tmp *= ybuf[off(n, c, h, w) + y_offset];
                            }
                            channel_dot += tmp;
                        }
                        for(size_t c = 0; c < dims[1]; ++c)
                        {
                            double ref = dybuf[off(n, c, h, w) + dy_offset];
                            if(algo == MIOPEN_SOFTMAX_LOG)
                            {
                                ref -= channel_dot * exp(ybuf[off(n, c, h, w) + y_offset]);
                            }
                            else
                            {
                                ref = (ref - channel_dot) * ybuf[off(n, c, h, w) + y_offset];
                            }
                            double err = std::abs(ref - res[off(n, c, h, w) + xdx_offset]);
                            if(ref != 0.0)
                            {
                                err = std::min(err, std::abs(err / ref));
                            }
                            max_err = std::max(max_err, err);
                        }
                    }
                }
            }
        }
        EXPECT_LT(max_err, 1e-4) << "Non-packed tensor mis-addressed: GPU output does not "
                                    "match strided CPU reference (max abs err "
                                 << max_err << ").";
    }
};

TEST_P(GPU_Softmax_NonContiguous_FP32, ForwardTest) { RunForward(); }
TEST_P(GPU_Softmax_NonContiguous_FP32, BackwardTest) { RunBackward(); }

namespace {

std::vector<std::tuple<std::vector<size_t>, std::vector<size_t>>> nonContiguousTestCases()
{
    return {{{2, 4, 2, 2}, {32, 4, 2, 1}},
            {{8, 2048, 5, 7}, {131072, 35, 7, 1}},
            {{2, 4, 2, 2}, {32, 1, 8, 4}},
            {{8, 2048, 5, 7}, {131072, 1, 14336, 2048}}};
}

} // namespace

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Softmax_NonContiguous_FP32,
                         testing::Combine(testing::ValuesIn(nonContiguousTestCases()),
                                          testing::Values(MIOPEN_SOFTMAX_FAST,
                                                          MIOPEN_SOFTMAX_ACCURATE,
                                                          MIOPEN_SOFTMAX_LOG),
                                          testing::Values(MIOPEN_SOFTMAX_MODE_INSTANCE,
                                                          MIOPEN_SOFTMAX_MODE_CHANNEL),
                                          testing::Values(0, 11),
                                          testing::Values(0, 13),
                                          testing::Values(0, 17)));

// --- Noncontiguous forward, beta != 0, CSR-Vector (NUM_BATCH==1) path -----
// Regression test for a bug where the beta-blend term read the RAW POINTER `y`
// instead of the indexed element `y[y_idx]` (CVT_FLOAT2ACCUM(y) instead of
// CVT_FLOAT2ACCUM(y[y_idx])) in MIOpenSoftmaxNoncontiguous.cpp. Since `y` is a
// `float* __restrict__`, casting the raw pointer to FLOAT_ACCUM does not
// type-check, so with this bug present the kernel FAILS TO JIT-COMPILE
// whenever beta != 0 (any test that hits this code path errors out rather
// than producing a wrong answer). This only affects the CSR-Vector branch,
// selected by SoftmaxNoncontiguous when vector_size (the softmax inner size)
// is >= LOCAL_SIZE (1024, see PerformanceConfigSoftmax::max_local_size), so
// the inner size below is intentionally >= 1024 (C*H*W == 1024).
struct GPU_Softmax_NonContiguousBeta_FP32
    : public testing::TestWithParam<std::tuple<std::tuple<std::vector<size_t>, std::vector<size_t>>,
                                               miopenSoftmaxAlgorithm_t,
                                               miopenSoftmaxMode_t>>
{
    void RunForward()
    {
        auto&& handle                        = get_handle();
        auto [tensorDimsStrides, algo, mode] = GetParam();
        auto [dims, strides]                 = tensorDimsStrides;

        auto in_host   = tensor<float>{dims, strides}.generate(tensor_elem_gen_integer{5});
        size_t n_elems = dims[0] * strides[0];

        const float alpha = 1.0f, beta = 0.5f;

        std::vector<float> xbuf(n_elems, 0.0f);
        // Pre-seed y with a known, non-trivial value so the beta*y_old blend term
        // is actually exercised (and must be read from the correct strided index).
        std::vector<float> ybuf(n_elems, 0.0f);

        auto off = [&](size_t n, size_t c, size_t h, size_t w) {
            return n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
        };
        for(size_t n = 0; n < dims[0]; ++n)
            for(size_t c = 0; c < dims[1]; ++c)
                for(size_t h = 0; h < dims[2]; ++h)
                    for(size_t w = 0; w < dims[3]; ++w)
                    {
                        xbuf[off(n, c, h, w)] =
                            static_cast<float>((n * 13 + c * 7 + h * 3 + w + 1) % 5);
                        ybuf[off(n, c, h, w)] =
                            static_cast<float>((n * 5 + c * 3 + h * 2 + w + 2) % 4);
                    }

        auto in_dev  = handle.Write(xbuf);
        auto out_dev = handle.Write(ybuf);

        miopen::SoftmaxForward(handle,
                               &alpha,
                               &beta,
                               in_host.desc,
                               in_dev.get(),
                               in_host.desc,
                               out_dev.get(),
                               algo,
                               mode);
        auto res = handle.Read<float>(out_dev, ybuf.size());

        // Strided CPU reference, INSTANCE mode only (inner = C*H*W), incl. the
        // beta * y_old blend so the fix's y_idx indexing is actually checked.
        double max_err = 0.0;
        for(size_t n = 0; n < dims[0]; ++n)
        {
            float mx = std::numeric_limits<float>::lowest();
            for(size_t c = 0; c < dims[1]; ++c)
                for(size_t h = 0; h < dims[2]; ++h)
                    for(size_t w = 0; w < dims[3]; ++w)
                        mx = std::max(mx, xbuf[off(n, c, h, w)]);
            double sum = 0.0;
            for(size_t c = 0; c < dims[1]; ++c)
                for(size_t h = 0; h < dims[2]; ++h)
                    for(size_t w = 0; w < dims[3]; ++w)
                        sum += std::exp(xbuf[off(n, c, h, w)] - mx);
            for(size_t c = 0; c < dims[1]; ++c)
            {
                for(size_t h = 0; h < dims[2]; ++h)
                {
                    for(size_t w = 0; w < dims[3]; ++w)
                    {
                        double ref = alpha * (std::exp(xbuf[off(n, c, h, w)] - mx) / sum) +
                                     beta * ybuf[off(n, c, h, w)];
                        double err = std::abs(ref - res[off(n, c, h, w)]);
                        max_err    = std::max(max_err, err);
                    }
                }
            }
        }
        EXPECT_LT(max_err, 1e-4) << "Noncontiguous fwd beta!=0 (CSR-Vector, NUM_BATCH==1) gave "
                                    "wrong result (max abs err "
                                 << max_err << ").";
    }
};

TEST_P(GPU_Softmax_NonContiguousBeta_FP32, ForwardTest) { RunForward(); }

namespace {

std::vector<std::tuple<std::vector<size_t>, std::vector<size_t>>> nonContiguousBetaTestCases()
{
    // inner = C*H*W == 1024 (>= LOCAL_SIZE => NUM_BATCH==1, CSR-Vector branch).
    // N stride padded to 1088 (packed would be 1024) so the tensor is non-packed.
    return {{{2, 1024, 1, 1}, {1088, 1, 1, 1}}};
}

} // namespace

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Softmax_NonContiguousBeta_FP32,
                         testing::Combine(testing::ValuesIn(nonContiguousBetaTestCases()),
                                          testing::Values(MIOPEN_SOFTMAX_ACCURATE),
                                          testing::Values(MIOPEN_SOFTMAX_MODE_INSTANCE)));

// --- Mixed packing (packed x, non-packed/strided y) --------------------
// Regression test for solver::softmax::Softmax::IsApplicable's forward gate,
// which read `if(!x.IsPacked() && !y.IsPacked()) return false;` -- i.e. it only
// rejected when BOTH operands were non-packed, so a MIXED case (one packed,
// one not) was wrongly accepted. The corresponding backward gate a few lines
// down already used `||` (reject if ANY operand is non-packed) -- the forward
// gate should match. The Softmax (packed) solver's kernel derives all output
// addressing from OUTER_SIZE/INNER_SIZE/STRIDE (no per-descriptor strides are
// passed to the kernel at all, unlike SoftmaxNoncontiguous which passes real
// N/C/H/W strides) -- so when accepted for a non-packed y, it writes assuming
// packed/contiguous addressing while the real y buffer has padding, producing
// wrong results at every row after the first.
//
// No forward softmax solver actually supports mixed packing, so the correct
// behavior is to fail cleanly (miopenStatusNotImplemented) rather than silently
// mis-address memory. miopen::SoftmaxForward runs
// solver::SolverContainer<AttnSoftmax, Softmax, SoftmaxNoncontiguous>
// ::ExecutePrimitive() in that FIXED order (see src/softmax.cpp) and takes the
// first applicable solver. For packed-x / non-packed-y:
//   * AttnSoftmax          -- rejects (requires both descriptors packed).
//   * Softmax              -- pre-fix `&&` wrongly ACCEPTS -> runs with packed
//                             addressing on the padded y -> silent wrong output
//                             (no throw). Post-fix `||` correctly rejects.
//   * SoftmaxNoncontiguous -- rejects: requires x and y to share identical
//                             strides (src/solver/softmax/softmax_noncontiguous
//                             .cpp), which mixed packing never does.
// So post-fix NO solver applies and SoftmaxForward throws. The test is RED
// pre-fix (Softmax runs, no throw) and GREEN post-fix (throws). The fix turns a
// silent wrong result into a clean unsupported-config error; it does not make
// mixed packing produce a correct result (no solver can, without extending
// SoftmaxNoncontiguous to independent x/y strides).
// Parameterized (with a single trivial value) so the test's full name is
// "Smoke/GPU_Softmax_MixedPacking_FP32.ForwardPackedXStridedY/0" -- the
// leading "Smoke/" prefix (from INSTANTIATE_TEST_SUITE_P) is required for
// TheRock CI's "*/GPU_Softmax*" filter (see test_categories.yaml) to select
// it; a bare TEST_F would not match that glob (no "Prefix/" component).
struct GPU_Softmax_MixedPacking_FP32 : public testing::TestWithParam<int>
{
};

TEST_P(GPU_Softmax_MixedPacking_FP32, ForwardPackedXStridedY)
{
    auto&& handle = get_handle();

    const std::vector<size_t> dims = {2, 32, 1, 1};
    // Packed would be {32, 1, 1, 1}; pad the N stride so y is non-packed.
    const std::vector<size_t> strided_strides = {40, 1, 1, 1};

    auto x_host = tensor<float>{miopenTensorNCHW, dims}.generate(tensor_elem_gen_integer{5});
    auto y_host = tensor<float>{dims, strided_strides};

    ASSERT_TRUE(x_host.desc.IsPacked()) << "test setup bug: x must be packed";
    ASSERT_FALSE(y_host.desc.IsPacked()) << "test setup bug: y must be non-packed";

    const size_t n_elems_y = dims[0] * strided_strides[0];

    std::vector<float> ybuf(n_elems_y, -42.0f);

    const float alpha = 1.0f, beta = 0.0f;
    auto x_dev = handle.Write(x_host.data);
    auto y_dev = handle.Write(ybuf);

    // No forward solver supports mixed packing -> SoftmaxForward must throw
    // (miopenStatusNotImplemented) rather than mis-address the padded y buffer.
    // Pre-fix, the `&&` gate let the packed Softmax solver run instead, silently
    // producing wrong output (no throw).
    EXPECT_ANY_THROW(miopen::SoftmaxForward(handle,
                                            &alpha,
                                            &beta,
                                            x_host.desc,
                                            x_dev.get(),
                                            y_host.desc,
                                            y_dev.get(),
                                            MIOPEN_SOFTMAX_ACCURATE,
                                            MIOPEN_SOFTMAX_MODE_INSTANCE));
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Softmax_MixedPacking_FP32, testing::Values(0));

// --- Extra coverage: non-zero x/y offsets --
// Findings from the tests below (both DISABLED -- see caveats):
//   * The Softmax solver DOES honor offsets correctly.
//   * But SoftmaxForward runs Find over {AttnSoftmax, Softmax, SoftmaxNoncontiguous}; AttnSoftmax
//     ignores the offsets and can
//     win the Find race even for ordinary softmax shapes -> SoftmaxForward with
//     a non-zero offset then writes to the un-offset location (wrong result).
struct GPU_Softmax_Offset_FP32 : public testing::Test
{
};

static double RunOffsetCase(int x_offset, int y_offset, const std::vector<size_t>& dims)
{
    auto&& handle = get_handle();

    auto in_host = tensor<float>{miopenTensorNCHW, dims}.generate(tensor_elem_gen_integer{5});
    const size_t n_elems = in_host.data.size();

    // Over-allocated device buffers; real data lives at [offset, offset+n_elems).
    std::vector<float> xbuf(n_elems + x_offset, 7.0f);
    std::vector<float> ybuf(n_elems + y_offset, 3.0f);
    std::copy(in_host.data.begin(), in_host.data.end(), xbuf.begin() + x_offset);

    auto x_dev = handle.Write(xbuf);
    auto y_dev = handle.Write(ybuf);

    const float alpha = 1.0f, beta = 0.0f;
    miopen::SoftmaxForward(handle,
                           &alpha,
                           &beta,
                           in_host.desc,
                           x_dev.get(),
                           in_host.desc,
                           y_dev.get(),
                           MIOPEN_SOFTMAX_ACCURATE,
                           MIOPEN_SOFTMAX_MODE_INSTANCE,
                           x_offset,
                           y_offset);
    auto res = handle.Read<float>(y_dev, ybuf.size());

    const auto [n, c, h, w] = miopen::tien<4>(dims);
    double max_err          = 0.0;
    for(size_t o = 0; o < n; ++o)
    {
        float mx = std::numeric_limits<float>::lowest();
        for(size_t i = 0; i < c * h * w; ++i)
            mx = std::max(mx, in_host.data[o * c * h * w + i]);
        double sum = 0.0;
        for(size_t i = 0; i < c * h * w; ++i)
            sum += std::exp(in_host.data[o * c * h * w + i] - mx);
        for(size_t i = 0; i < c * h * w; ++i)
        {
            double ref = std::exp(in_host.data[o * c * h * w + i] - mx) / sum;
            double got = res[y_offset + o * c * h * w + i];
            max_err    = std::max(max_err, std::abs(ref - got));
        }
    }
    return max_err;
}

// Non-zero offset via the public SoftmaxForward API. FLAKY by nature: passes
// when Find selects the Softmax solver (offsets honored), fails when Find
// selects AttnSoftmax (offsets dropped). DISABLED; run intentionally with
//   ./bin/test_soft_max --gtest_also_run_disabled_tests --gtest_filter='*Offset*'
TEST_F(GPU_Softmax_Offset_FP32, DISABLED_ForwardNonZeroOffset_FindRace)
{
    const std::vector<size_t> dims = {2, 13, 1, 1};

    double err0 = RunOffsetCase(0, 0, dims);
    EXPECT_LT(err0, 1e-4) << "zero-offset baseline wrong (harness bug?) err=" << err0;

    double err1 = RunOffsetCase(32, 48, dims);
    EXPECT_LT(err1, 1e-4) << "non-zero offset dropped (Find selected AttnSoftmax?); max abs err "
                          << err1;
}

// --- Divergent __syncthreads() deadlock ----------------------------
// CSR-Stream path (NUM_BATCH>1) with grid_size not a multiple of num_batch:
// whole wavefronts hit `if(o >= OUTER_SIZE) return;` and skip the block-wide
// __syncthreads() inside reduce_block(), while surviving lanes wait on it
// forever. {3,8,1,1} INSTANCE ACCURATE: inner=8 (aligned, so no fault masks the
// hang), outer=3, local=1024 -> num_batch=512; batches 3..511 (waves 1..15)
// early-return while wave 0 reaches the barrier -> hang.
//
// Formerly DISABLED_ under the premise "WILL WEDGE THE GPU". Empirically this
// does NOT hang: re-confirmed on gfx942/CDNA3 (wave64) and gfx1201/RDNA4
// (wave32) -- the same wavefront-width families TheRock CI runs (gfx94X,
// gfx125X) -- across all six candidate shapes below. Un-DISABLED and, since a
// bare TEST_F's name ("GPU_Softmax_Deadlock_FP32.*") has no "Prefix/" and so
// would NOT match test_categories.yaml's "*/GPU_Softmax*" glob (nor would
// TheRock ever pass --gtest_also_run_disabled_tests), converted to a
// Smoke-prefixed parameterized suite via INSTANTIATE_TEST_SUITE_P so its full
// name becomes "Smoke/GPU_Softmax_Deadlock_FP32.CsrStreamPartialBlock/<N>" and
// is selected by CI like the other Smoke/GPU_Softmax_* suites in this file.
struct SoftmaxDeadlockCand
{
    std::vector<size_t> dims;
    miopenSoftmaxMode_t mode;
};

struct GPU_Softmax_Deadlock_FP32 : public testing::TestWithParam<SoftmaxDeadlockCand>
{
};

TEST_P(GPU_Softmax_Deadlock_FP32, CsrStreamPartialBlock)
{
    auto&& handle = get_handle();
    const auto& cand = GetParam();

    auto input = tensor<float>{miopenTensorNCHW, cand.dims}.generate(tensor_elem_gen_integer{5});
    auto output = tensor<float>{miopenTensorNCHW, cand.dims};

    const float alpha = 1.0f, beta = 0.0f;
    auto in_dev  = handle.Write(input.data);
    auto out_dev = handle.Write(output.data);

    miopen::SoftmaxForward(handle,
                           &alpha,
                           &beta,
                           input.desc,
                           in_dev.get(),
                           output.desc,
                           out_dev.get(),
                           MIOPEN_SOFTMAX_ACCURATE,
                           cand.mode);
    handle.Finish(); // a hang would manifest here
    auto res = handle.Read<float>(out_dev, output.data.size());
    for(auto v : res)
        EXPECT_TRUE(std::isfinite(v));
}

// Each candidate is a CSR-Stream config (inner<local, BATCH_SIZE>1) whose
// grid is not a multiple of num_batch, so the last block mixes surviving
// lanes (which reach reduce_block's __syncthreads) with early-returning
// lanes / whole wavefronts.
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_Softmax_Deadlock_FP32,
    testing::Values(
        SoftmaxDeadlockCand{{3, 8, 1, 1}, MIOPEN_SOFTMAX_MODE_INSTANCE}, // 15/16 waves return
        SoftmaxDeadlockCand{{7, 8, 1, 1}, MIOPEN_SOFTMAX_MODE_INSTANCE},
        SoftmaxDeadlockCand{{70, 8, 1, 1}, MIOPEN_SOFTMAX_MODE_INSTANCE}, // ~2 waves survive
        SoftmaxDeadlockCand{{100, 8, 1, 1}, MIOPEN_SOFTMAX_MODE_INSTANCE}, // multiple survive
        SoftmaxDeadlockCand{{5, 16, 1, 1}, MIOPEN_SOFTMAX_MODE_INSTANCE},
        SoftmaxDeadlockCand{{2, 8, 3, 3}, MIOPEN_SOFTMAX_MODE_CHANNEL})); // stride=H*W=9 path
