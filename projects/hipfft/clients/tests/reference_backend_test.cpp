// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "../../shared/reference_fft.h"
#include <complex>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>

namespace
{
    size_t elements(const std::vector<size_t>& shape)
    {
        return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    }

    std::vector<size_t> coordinates(size_t i, const std::vector<size_t>& shape)
    {
        std::vector<size_t> result(shape.size());
        for(size_t d = shape.size(); d-- > 0;)
        {
            result[d] = i % shape[d];
            i /= shape[d];
        }
        return result;
    }

    std::vector<size_t> strides(const std::vector<size_t>& shape, size_t step)
    {
        std::vector<size_t> result(shape.size());
        for(size_t d = shape.size(); d-- > 0;)
        {
            result[d] = step;
            step *= shape[d];
        }
        return result;
    }

    size_t offset(size_t i, const std::vector<size_t>& shape, const std::vector<size_t>& stride)
    {
        const auto coord = coordinates(i, shape);
        return std::inner_product(coord.begin(), coord.end(), stride.begin(), size_t{0});
    }

    std::complex<double> dft(const std::vector<std::complex<double>>& input,
                             const std::vector<size_t>&               shape,
                             size_t                                   k,
                             int                                      sign)
    {
        const auto           frequency = coordinates(k, shape);
        std::complex<double> result{};
        for(size_t j = 0; j < input.size(); ++j)
        {
            const auto position = coordinates(j, shape);
            double     phase    = 0;
            for(size_t d = 0; d < shape.size(); ++d)
                phase += double(position[d] * frequency[d]) / shape[d];
            result += input[j] * std::polar(1.0, sign * 2 * std::acos(-1.0) * phase);
        }
        return result;
    }

    template <typename T>
    double tolerance(size_t n)
    {
        return n * (std::is_same_v<T, double> ? 2e-12 : std::is_same_v<T, float> ? 2e-5 : 3e-3);
    }

    template <typename T>
    void complex_cases()
    {
        for(const auto& shape : {std::vector<size_t>{7}, {8}, {2, 3}, {2, 3, 5}})
            for(int sign : {-1, 1})
            {
                const size_t         n     = elements(shape);
                const auto           is    = strides(shape, 2);
                const auto           os    = strides(shape, 3);
                const size_t         idist = n * 2 + 5, odist = n * 3 + 7;
                std::vector<hostbuf> input(1), output(1);
                input[0].alloc(2 * idist * 2 * sizeof(T));
                output[0].alloc(2 * odist * 2 * sizeof(reference_fft::real_t<T>));
                const auto type = sign == -1 ? fft_transform_type_complex_forward
                                             : fft_transform_type_complex_inverse;
                auto       plan = reference_fft::make_plan<T>(
                    shape, is, os, 2, idist, odist, type, input, output);
                // Reuse the plan with different input values and newly allocated buffers.
                for(int iteration = 0; iteration < 2; ++iteration)
                {
                    std::vector<hostbuf> new_input(1), new_output(1);
                    new_input[0].alloc(2 * idist * 2 * sizeof(T));
                    new_output[0].alloc(2 * odist * 2 * sizeof(reference_fft::real_t<T>));
                    auto* in = static_cast<T*>(new_input[0].data());
                    std::vector<std::vector<std::complex<double>>> values(2);
                    for(size_t b = 0; b < 2; ++b)
                        for(size_t j = 0; j < n; ++j)
                        {
                            const T      re  = T(int((j + b + iteration) % 7) - 3) / T(8);
                            const T      im  = T(int(j % 3) - 1) / T(4);
                            const size_t pos = 2 * (b * idist + offset(j, shape, is));
                            in[pos]          = re;
                            in[pos + 1]      = im;
                            values[b].emplace_back(double(re), double(im));
                        }
                    reference_fft::execute<T>(type, plan, new_input, new_output);
                    const auto* out = static_cast<const T*>(new_output[0].data());
                    for(size_t b = 0; b < 2; ++b)
                        for(size_t k = 0; k < n; ++k)
                        {
                            const size_t pos      = 2 * (b * odist + offset(k, shape, os));
                            const auto   expected = dft(values[b], shape, k, sign);
                            EXPECT_NEAR(double(out[pos]), expected.real(), tolerance<T>(n));
                            EXPECT_NEAR(double(out[pos + 1]), expected.imag(), tolerance<T>(n));
                        }
                }
            }
    }

    template <typename T>
    void real_cases()
    {
        for(const auto& shape : {std::vector<size_t>{7}, {8}, {3, 5}, {2, 8}, {2, 3, 4}})
        {
            auto hermitian   = shape;
            hermitian.back() = shape.back() / 2 + 1;
            const size_t n = elements(shape), h = elements(hermitian);
            const auto   real_stride = strides(shape, 2), complex_stride = strides(hermitian, 3);
            const size_t real_dist = n * 2 + 5, complex_dist = h * 3 + 7;
            std::vector<hostbuf> input(1), spectrum(1), restored(1);
            input[0].alloc(2 * real_dist * sizeof(T));
            spectrum[0].alloc(2 * complex_dist * 2 * sizeof(T));
            restored[0].alloc(2 * real_dist * sizeof(T));
            auto*                                          in = static_cast<T*>(input[0].data());
            std::vector<std::vector<std::complex<double>>> values(2);
            for(size_t b = 0; b < 2; ++b)
                for(size_t j = 0; j < n; ++j)
                {
                    const T value = T(int((j + b) % 7) - 3) / T(8);
                    in[b * real_dist + offset(j, shape, real_stride)] = value;
                    values[b].emplace_back(double(value), 0);
                }
            auto forward = reference_fft::make_plan<T>(shape,
                                                       real_stride,
                                                       complex_stride,
                                                       2,
                                                       real_dist,
                                                       complex_dist,
                                                       fft_transform_type_real_forward,
                                                       input,
                                                       spectrum);
            reference_fft::execute<T>(fft_transform_type_real_forward, forward, input, spectrum);
            auto* freq = static_cast<T*>(spectrum[0].data());
            for(size_t b = 0; b < 2; ++b)
                for(size_t k = 0; k < h; ++k)
                {
                    const auto coord      = coordinates(k, hermitian);
                    size_t     full_index = 0;
                    for(size_t d = 0; d < shape.size(); ++d)
                        full_index = full_index * shape[d] + coord[d];
                    const auto expected = dft(values[b], shape, full_index, -1);
                    const auto pos = 2 * (b * complex_dist + offset(k, hermitian, complex_stride));
                    EXPECT_NEAR(double(freq[pos]), expected.real(), tolerance<T>(n));
                    EXPECT_NEAR(double(freq[pos + 1]), expected.imag(), tolerance<T>(n));
                }
            auto inverse = reference_fft::make_plan<T>(shape,
                                                       complex_stride,
                                                       real_stride,
                                                       2,
                                                       complex_dist,
                                                       real_dist,
                                                       fft_transform_type_real_inverse,
                                                       spectrum,
                                                       restored);
            reference_fft::execute<T>(fft_transform_type_real_inverse, inverse, spectrum, restored);
            const auto* out = static_cast<const T*>(restored[0].data());
            for(size_t b = 0; b < 2; ++b)
                for(size_t j = 0; j < n; ++j)
                    EXPECT_NEAR(double(out[b * real_dist + offset(j, shape, real_stride)]),
                                n * values[b][j].real(),
                                tolerance<T>(n) * n);
        }
    }
}

static_assert(!std::is_same_v<reference_fft::plan<float>, reference_fft::plan<double>>);

TEST(reference_backend, complex_float)
{
    complex_cases<float>();
}
TEST(reference_backend, complex_double)
{
    complex_cases<double>();
}
TEST(reference_backend, complex_half)
{
    complex_cases<rocfft_fp16>();
}
TEST(reference_backend, real_float)
{
    real_cases<float>();
}
TEST(reference_backend, real_double)
{
    real_cases<double>();
}
TEST(reference_backend, unsupported_wisdom)
{
    if constexpr(!reference_fft::supports_wisdom)
    {
        EXPECT_THROW(reference_fft::import_wisdom("unused"), std::invalid_argument);
        EXPECT_THROW(reference_fft::export_wisdom("unused"), std::invalid_argument);
    }
}
