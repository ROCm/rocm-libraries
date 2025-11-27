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

#include <concepts>
#include <ranges>

#include "gtest_common.hpp"
#include <tensor_util.hpp>
#include <miopen/lrn.hpp>

#include "network_data.hpp"

namespace {

template <typename T>
concept Printable = requires(std::ostream& os, T t)
{
    os << t;
};

template <typename T>
concept Container = requires(T t)
{
    // clang-format off
    {
        std::size(t)
    } -> std::same_as<std::size_t>;
    {
        std::begin(t)
    } -> std::same_as<typename T::iterator>;
    {
        std::end(t)
    } -> std::same_as<typename T::iterator>;
    // clang-format on
};

template <typename T>
requires Printable<T> && std::is_move_constructible_v<T>
struct NamedParameter
{
    NamedParameter(std::string parameterName, T parameterValue) noexcept
        : name(std::move(parameterName)), value(std::move(parameterValue))
    {
    }

    operator T() const { return value; }

    const T& operator()() const noexcept { return value; }

    friend std::ostream& operator<<(std::ostream& os, const NamedParameter<T>& param)
    {
        return os << param.name << ": " << param.value;
    }

    std::string name{};
    T value{};
};

template <typename T>
requires Container<T> && std::is_move_constructible_v<T>
struct NamedContainer
{
    NamedContainer(std::string containerName,
                   T containerValue,
                   std::string valueSeparator = " ") noexcept
        : name(std::move(containerName)),
          value(std::move(containerValue)),
          separator(std::move(valueSeparator))
    {
    }

    operator T() const { return value; }

    const T& operator()() const noexcept { return value; }

    friend std::ostream& operator<<(std::ostream& os, const NamedContainer<T>& param)
    {
        os << param.name << ": [";

        if(param.value.size() > 0)
        {
            os << *param.value.begin();

            for(auto it = param.value.begin() + 1; it != param.value.end(); ++it)
            {
                os << param.separator << *it;
            }
        }

        os << "]";

        return os;
    }

    std::string name{};
    T value{};
    std::string separator{};
};

template <typename... T>
static auto MakeNamedParameterValues(const std::string& name, T... values)
{
    return testing::Values(NamedParameter<T>{name, values}...);
}

using TestCase = std::tuple<NamedContainer<std::vector<int>>,
                            NamedParameter<unsigned int>,
                            NamedParameter<double>,
                            NamedParameter<double>,
                            NamedParameter<double>,
                            NamedParameter<miopenLRNMode_t>>;

template <class T>
struct verify_lrn_forward
{
    miopen::LRNDescriptor lrn;
    tensor<T> input;

    verify_lrn_forward(const miopen::LRNDescriptor& plrnDesc, const tensor<T>& pinput)
    {
        lrn   = plrnDesc;
        input = pinput;
    }

    tensor<T> cpu() const
    {
        auto output = tensor<T>{input.desc.GetLengths()};
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        const auto alpha       = lrn.GetAlpha();
        const auto beta        = lrn.GetBeta();
        const auto K           = lrn.GetK();
        const auto lrn_n       = lrn.GetN();
        const int radius_lower = static_cast<int>((lrn_n - 1) / 2);
        const int radius_upper = static_cast<int>(lrn_n / 2);
        const auto mode        = lrn.GetMode();

        if(mode == miopenLRNCrossChannel)
        {
            const auto alphaoverarea = alpha / lrn_n;

            miopen::par_ford(n_batch)([&](int b) {
                for(int c = 0; c < channels; ++c)
                {
                    const int start = c < radius_lower ? 0 : (c - radius_lower);
                    const int end =
                        (c + radius_upper + 1) > channels ? channels : (c + radius_upper + 1);

                    for(int h = 0; h < height; ++h)
                    {
                        for(int w = 0; w < width; ++w)
                        {
                            double scale = 0;
                            for(int k = start; k < end; ++k)
                            {
                                scale += std::pow(input(b, k, h, w), 2);
                            }

                            scale *= alphaoverarea;
                            scale += K;
                            scale = std::pow(scale, -beta);

                            output(b, c, h, w) = static_cast<T>(scale * input(b, c, h, w));
                        }
                    }
                }
            });
        }
        else
        {
            const double alphaoverarea = radius_upper == 0 ? 1 : alpha / (lrn_n * lrn_n);

            miopen::par_ford(n_batch)([&](int b) {
                for(int c = 0; c < channels; ++c)
                {
                    for(int h = 0; h < height; ++h)
                    {
                        const int top = (h - radius_lower) < 0 ? 0 : (h - radius_lower);
                        const int bottom =
                            (h + radius_upper + 1) > height ? height : (h + radius_upper + 1);

                        for(int w = 0; w < width; ++w)
                        {
                            const int left = (w - radius_lower) < 0 ? 0 : (w - radius_lower);
                            const int right =
                                (w + radius_upper + 1) > width ? width : (w + radius_upper + 1);
                            double scale = 0;

                            for(int i = left; i < right; ++i)
                            {
                                for(int j = top; j < bottom; ++j)
                                {
                                    scale += std::pow(input(b, c, j, i), 2);
                                }
                            }

                            scale *= alphaoverarea;
                            scale += K;
                            scale              = std::pow(scale, -beta);
                            output(b, c, h, w) = static_cast<T>(scale * input(b, c, h, w));
                        }
                    }
                }
            });
        }

        return output;
    }

    tensor<T> cpu_st() const
    {
        auto output = tensor<T>{input.desc.GetLengths()};
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        const auto alpha       = lrn.GetAlpha();
        const auto beta        = lrn.GetBeta();
        const auto K           = lrn.GetK();
        const auto lrn_n       = lrn.GetN();
        const int radius_lower = static_cast<int>((lrn_n - 1) / 2);
        const int radius_upper = static_cast<int>(lrn_n / 2);
        const auto mode        = lrn.GetMode();

        if(mode == miopenLRNCrossChannel)
        {
            const auto alphaoverarea = alpha / lrn_n;

            for(int b = 0; b < n_batch; ++b)
            {
                for(int c = 0; c < channels; ++c)
                {
                    const int start = c < radius_lower ? 0 : (c - radius_lower);
                    const int end =
                        (c + radius_upper + 1) > channels ? channels : (c + radius_upper + 1);

                    for(int h = 0; h < height; ++h)
                    {
                        for(int w = 0; w < width; ++w)
                        {
                            double scale = 0;
                            for(int k = start; k < end; ++k)
                            {
                                scale += std::pow(input(b, k, h, w), 2);
                            }

                            scale *= alphaoverarea;
                            scale += K;
                            scale = std::pow(scale, -beta);

                            output(b, c, h, w) = static_cast<T>(scale * input(b, c, h, w));
                        }
                    }
                }
            }
        }
        else
        {
            const double alphaoverarea = radius_upper == 0 ? 1 : alpha / (lrn_n * lrn_n);

            for(int b = 0; b < n_batch; ++b)
            {
                for(int c = 0; c < channels; ++c)
                {
                    for(int h = 0; h < height; ++h)
                    {
                        const int top = (h - radius_lower) < 0 ? 0 : (h - radius_lower);
                        const int bottom =
                            (h + radius_upper + 1) > height ? height : (h + radius_upper + 1);

                        for(int w = 0; w < width; ++w)
                        {
                            const int left = (w - radius_lower) < 0 ? 0 : (w - radius_lower);
                            const int right =
                                (w + radius_upper + 1) > width ? width : (w + radius_upper + 1);
                            double scale = 0;

                            for(int i = left; i < right; ++i)
                            {
                                for(int j = top; j < bottom; ++j)
                                {
                                    scale += std::pow(input(b, c, j, i), 2);
                                }
                            }

                            scale *= alphaoverarea;
                            scale += K;
                            scale              = std::pow(scale, -beta);
                            output(b, c, h, w) = static_cast<T>(scale * input(b, c, h, w));
                        }
                    }
                }
            }
        }

        return output;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto out      = tensor<T>{input.desc.GetLengths()};
        auto in_dev   = handle.Write(input.data);
        auto out_dev  = handle.Write(out.data);
        auto alpha    = lrn.GetAlpha();
        auto beta     = lrn.GetBeta();
        auto bDoBwd   = false;

        lrn.Forward(handle,
                    &alpha,
                    input.desc,
                    in_dev.get(),
                    &beta,
                    out.desc,
                    out_dev.get(),
                    bDoBwd,
                    nullptr);

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    void fail() const
    {
        std::cout << "verify_lrn_forward" << std::endl;
        std::cout << "Input Tensor" << " " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_lrn_bwd
{
    miopen::LRNDescriptor lrn;
    tensor<T> inputY;
    tensor<T> inputDY;
    tensor<T> inputX;
    tensor<T> scale;

    verify_lrn_bwd(const miopen::LRNDescriptor& plrn,
                   const tensor<T>& pout,
                   const tensor<T>& pdout,
                   const tensor<T>& pin,
                   const tensor<T>& pscale)
    {
        lrn     = plrn;
        inputY  = pout;
        inputDY = pdout;
        inputX  = pin;
        scale   = pscale;
    }

    tensor<T> cpu() const
    {
        auto routputDX = tensor<T>{inputX.desc.GetLengths()};
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(inputY.desc.GetLengths());

        const auto alpha       = lrn.GetAlpha();
        const auto beta        = lrn.GetBeta();
        const auto lrn_n       = lrn.GetN();
        const auto mode        = lrn.GetMode();
        const int radius_lower = static_cast<int>((lrn_n - 1) / 2);
        const int radius_upper = static_cast<int>(lrn_n / 2);

        if(mode == miopenLRNWithinChannel)
        {
            const auto adjust_area       = lrn_n * lrn_n;
            const auto cache_ratio_value = 2 * alpha * beta / adjust_area;

            miopen::par_ford(n_batch)([&](int b) {
                for(int c = 0; c < channels; ++c)
                {
                    for(int h = 0; h < height; ++h)
                    {
                        const int top = h < radius_upper ? 0 : (h - radius_upper);
                        const int bottom =
                            (h + radius_lower + 1) > height ? height : (h + radius_lower + 1);

                        for(int w = 0; w < width; ++w)
                        {
                            const int left = w < radius_upper ? 0 : (w - radius_upper);
                            const int right =
                                (w + radius_lower + 1) > width ? width : (w + radius_lower + 1);
                            double ydy = 0;

                            for(int i = left; i < right; i++)
                            {
                                for(int j = top; j < bottom; j++)
                                {
                                    ydy += (double(inputY(b, c, j, i) * inputDY(b, c, j, i)) /
                                            double(scale(b, c, j, i)));
                                }
                            }

                            routputDX(b, c, h, w) = static_cast<T>(
                                std::pow(static_cast<double>(scale(b, c, h, w)), -beta) *
                                    inputDY(b, c, h, w) -
                                cache_ratio_value * inputX(b, c, h, w) * ydy);
                        }
                    }
                }
            });
        }
        else
        {
            const auto cache_ratio_value = 2 * alpha * beta / lrn_n;

            miopen::par_ford(n_batch)([&](int b) {
                for(int c = 0; c < channels; ++c)
                {
                    const int start = c < radius_upper ? 0 : (c - radius_upper);
                    const int end =
                        (c + radius_lower + 1) > channels ? channels : (c + radius_lower + 1);

                    for(int h = 0; h < height; ++h)
                    {
                        for(int w = 0; w < width; ++w)
                        {
                            double ydy = 0;

                            for(auto k = start; k < end; ++k)
                            {
                                ydy += (double(inputY(b, k, h, w) * inputDY(b, k, h, w)) /
                                        double(scale(b, k, h, w)));
                            }

                            routputDX(b, c, h, w) = static_cast<T>(
                                std::pow(static_cast<double>(scale(b, c, h, w)), -beta) *
                                    inputDY(b, c, h, w) -
                                cache_ratio_value * inputX(b, c, h, w) * ydy);
                        }
                    }
                }
            });
        }

        return routputDX;
    }

    tensor<T> cpu_st() const
    {
        auto routputDX = tensor<T>{inputX.desc.GetLengths()};
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(inputY.desc.GetLengths());

        const auto alpha       = lrn.GetAlpha();
        const auto beta        = lrn.GetBeta();
        const auto lrn_n       = lrn.GetN();
        const auto mode        = lrn.GetMode();
        const int radius_lower = static_cast<int>((lrn_n - 1) / 2);
        const int radius_upper = static_cast<int>(lrn_n / 2);

        if(mode == miopenLRNWithinChannel)
        {
            const auto adjust_area       = lrn_n * lrn_n;
            const auto cache_ratio_value = 2 * alpha * beta / adjust_area;

            for(int b = 0; b < n_batch; ++b)
            {
                for(int c = 0; c < channels; ++c)
                {
                    for(int h = 0; h < height; ++h)
                    {
                        const int top = h < radius_upper ? 0 : (h - radius_upper);
                        const int bottom =
                            (h + radius_lower + 1) > height ? height : (h + radius_lower + 1);

                        for(int w = 0; w < width; ++w)
                        {
                            const int left = w < radius_upper ? 0 : (w - radius_upper);
                            const int right =
                                (w + radius_lower + 1) > width ? width : (w + radius_lower + 1);
                            double ydy = 0;

                            for(int i = left; i < right; i++)
                            {
                                for(int j = top; j < bottom; j++)
                                {
                                    ydy += (double(inputY(b, c, j, i) * inputDY(b, c, j, i)) /
                                            double(scale(b, c, j, i)));
                                }
                            }

                            routputDX(b, c, h, w) = static_cast<T>(
                                std::pow(static_cast<double>(scale(b, c, h, w)), -beta) *
                                    inputDY(b, c, h, w) -
                                cache_ratio_value * inputX(b, c, h, w) * ydy);
                        }
                    }
                }
            }
        }
        else
        {
            const auto cache_ratio_value = 2 * alpha * beta / lrn_n;

            for(int b = 0; b < n_batch; ++b)
            {
                for(int c = 0; c < channels; ++c)
                {
                    const int start = c < radius_upper ? 0 : (c - radius_upper);
                    const int end =
                        (c + radius_lower + 1) > channels ? channels : (c + radius_lower + 1);

                    for(int h = 0; h < height; ++h)
                    {
                        for(int w = 0; w < width; ++w)
                        {
                            double ydy = 0;

                            for(auto k = start; k < end; ++k)
                            {
                                ydy += (double(inputY(b, k, h, w) * inputDY(b, k, h, w)) /
                                        double(scale(b, k, h, w)));
                            }

                            routputDX(b, c, h, w) = static_cast<T>(
                                std::pow(static_cast<double>(scale(b, c, h, w)), -beta) *
                                    inputDY(b, c, h, w) -
                                cache_ratio_value * inputX(b, c, h, w) * ydy);
                        }
                    }
                }
            }
        }

        return routputDX;
    }

    tensor<T> gpu() const
    {
        auto&& handle     = get_handle();
        auto routputDX    = tensor<T>{inputX.desc.GetLengths()};
        auto inputY_dev   = handle.Write(inputY.data);
        auto inputDY_dev  = handle.Write(inputDY.data);
        auto inputX_dev   = handle.Write(inputX.data);
        auto outputDX_dev = handle.Create<T>(routputDX.data.size());
        auto scale_dev    = handle.Write(scale.data);

        auto alpha = lrn.GetAlpha(), beta = lrn.GetBeta();
        lrn.Backward(handle,
                     &alpha,
                     inputY.desc, // Y
                     inputY_dev.get(),
                     inputDY.desc, // DY
                     inputDY_dev.get(),
                     inputX.desc, // X
                     inputX_dev.get(),
                     &beta,
                     routputDX.desc, // DX
                     outputDX_dev.get(),
                     scale_dev.get());

        routputDX.data = handle.Read<T>(outputDX_dev, routputDX.data.size());
        return routputDX;
    }

    void fail() const
    {
        std::cout << "verify_lrn_bwd" << std::endl;
        std::cout << "Input Tensor Y" << " " << inputY.desc.ToString() << std::endl;
        std::cout << "Input Tensor DY" << " " << inputDY.desc.ToString() << std::endl;
        std::cout << "Input Tensor X" << " " << scale.desc.ToString() << std::endl;
    }
};

inline auto GenCases(bool limit = false)
{
    std::vector<NamedContainer<std::vector<int>>> input_dims;

    if(limit)
    {
        input_dims.emplace_back(
            NamedContainer<std::vector<int>>("input_dims", {16, 32, 8, 8}, "x"));
    }
    else
    {
        // taken from the original test
        const int batch_factor = 0;
        for(auto&& v : get_inputs(batch_factor))
        {
            input_dims.emplace_back(NamedContainer<std::vector<int>>("input_dims", v, "x"));
        }
    }

    return testing::Combine(testing::ValuesIn(input_dims),
                            MakeNamedParameterValues<unsigned int>("n", 1U, 4U, 5U),
                            testing::Values(NamedParameter<double>("alpha", 1.0)),
                            testing::Values(NamedParameter<double>("beta", 1.0)),
                            testing::Values(NamedParameter<double>("k", 1.0)),
                            MakeNamedParameterValues<miopenLRNMode_t>(
                                "mode", miopenLRNWithinChannel, miopenLRNCrossChannel));
}

inline auto GetCasesFull()
{
    static const auto cases = GenCases();
    return cases;
}

inline auto GetCasesSmoke()
{
    static const auto cases = GenCases(true);
    return cases;
}

} // namespace

std::ostream& operator<<(std::ostream& os, miopenLRNMode_t mode)
{
    switch(mode)
    {
    case miopenLRNWithinChannel: os << "WithinChannel"; break;

    case miopenLRNCrossChannel: os << "CrossChannel"; break;

    default: os << "Invalid"; break;
    }

    return os;
}

template <typename T>
class LrnCommon : public testing::TestWithParam<TestCase>
{
public:
    void SetUp() override
    {
        prng::reset_seed();
        std::tie(input_dims, n, alpha, beta, k, mode) = GetParam();
    }

    void Run()
    {
        input = tensor<T>{input_dims}.generate([](auto... is) {
            return tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17}() *
                   tensor_elem_gen_checkboard_sign{}(is...);
        });

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
        const size_t total_mem  = 5 * input.desc.GetNumBytes(); // estimate based on backward pass
        const size_t device_mem = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;

            FAIL() << "total_mem >= device_mem";
        }

        miopen::LRNDescriptor lrn{mode, n, {alpha, beta, k}};

        VerifyLrnForward(lrn, input);

        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        auto scale = tensor<T>{n_batch, channels, height, width}.generate(
            tensor_elem_gen_integer{max_value});
        auto dout = tensor<T>{n_batch, channels, height, width}.generate(
            tensor_elem_gen_integer{max_value});
        miopen::par_ford(n_batch, channels, height, width)(
            [&](int b, int c, int h, int w) { scale(b, c, h, w) += 1; });

        VerifyLrnBwd(lrn, cpu_results, dout, input, scale);
    }

    // we need cpu data for backward pass later, so return it from this function
    void VerifyLrnForward(const miopen::LRNDescriptor& plrnDesc, const tensor<T>& pinput)
    {
        verify_lrn_forward<T> verify_fwd{plrnDesc, pinput};
        CompareResults(verify_fwd, 1.5, true);
    }

    void VerifyLrnBwd(const miopen::LRNDescriptor& plrn,
                      const tensor<T>& pout,
                      const tensor<T>& pdout,
                      const tensor<T>& pin,
                      const tensor<T>& pscale)
    {
        verify_lrn_bwd<T> verify_bwd{plrn, pout, pdout, pin, pscale};
        CompareResults(verify_bwd, 6.0);
    }

    template <class TDirection>
    void CompareResults(const TDirection& direction, double tolerance, bool saveCpuResults = false)
    {
        {
            const tensor<T> cpu = direction.cpu();
            const tensor<T> gpu = direction.gpu();

            const double threshold = std::numeric_limits<T>::epsilon() * tolerance;
            const double error     = miopen::rms_range(cpu, gpu);

            if(saveCpuResults)
            {
                cpu_results = std::move(cpu);
            }

            if(error > threshold)
            {
                direction.fail();
            }

            ASSERT_LE(error, threshold) << "n: " << n << std::endl
                                        << "alpha: " << alpha << std::endl
                                        << "beta: " << beta << std::endl
                                        << "k: " << k << std::endl
                                        << "mode: " << mode << std::endl;
        }

        {
            const tensor<T> cpu = direction.cpu_st();
            const tensor<T> gpu = direction.gpu();

            const double threshold = std::numeric_limits<T>::epsilon() * tolerance;
            const double error     = miopen::rms_range(cpu, gpu);

            if(saveCpuResults)
            {
                cpu_results = std::move(cpu);
            }

            if(error > threshold)
            {
                direction.fail();
            }

            ASSERT_LE(error, threshold) << "n: " << n << std::endl
                                        << "alpha: " << alpha << std::endl
                                        << "beta: " << beta << std::endl
                                        << "k: " << k << std::endl
                                        << "mode: " << mode << std::endl;
        }
    }

private:
    tensor<T> input;

    std::vector<int> input_dims;
    unsigned int n       = 1;
    double alpha         = 1;
    double beta          = 1;
    double k             = 1;
    miopenLRNMode_t mode = miopenLRNWithinChannel;

    // cpu results of forward pass to be used for backward pass
    tensor<T> cpu_results;
};

using GPU_Lrn_FP32 = LrnCommon<float>;
using GPU_Lrn_FP16 = LrnCommon<half_float::half>;

template <typename TestSuiteT>
struct TestNameGenerator
{
    std::string operator()(const testing::TestParamInfo<typename TestSuiteT::ParamType>& info)
    {
        const auto& [input_dims, n, alpha, beta, k, mode] = info.param;
        std::stringstream ss;
        std::string str;

        ss << "input_dims_" << GetRangeAsString(input_dims(), "x") << "_n_" << n() << "_alpha_"
           << alpha() << "_beta_" << beta() << "_k_" << k() << "_mode_" << mode() << "_test_id_"
           << info.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }

private:
    std::string GetRangeAsString(const std::ranges::range auto& r, std::string_view separator)
    {
        std::string str;

        if(r.size() > 0)
        {
            std::stringstream ss;

            ss << *r.begin();

            for(auto it = r.begin() + 1; it != r.end(); ++it)
            {
                ss << separator << *it;
            }

            str = ss.str();

            // Name format only supports letters, numbers and underscores.
            std::transform(
                str.begin(), str.end(), str.begin(), [](char c) { return (c == '.') ? 'p' : c; });
        }

        return str;
    }
};

TEST_P(GPU_Lrn_FP32, TestFloat) { this->Run(); }
TEST_P(GPU_Lrn_FP16, TestFloat16) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Lrn_FP32, GetCasesSmoke(), TestNameGenerator<GPU_Lrn_FP32>{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_Lrn_FP32, GetCasesFull(), TestNameGenerator<GPU_Lrn_FP32>{});

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Lrn_FP16, GetCasesSmoke(), TestNameGenerator<GPU_Lrn_FP16>{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_Lrn_FP16, GetCasesFull(), TestNameGenerator<GPU_Lrn_FP16>{});
