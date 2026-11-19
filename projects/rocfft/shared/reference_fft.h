// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "reference_fft_common.h"
#include "reference_fft_types.h"
#include <limits>

#ifdef ROCFFT_REFERENCE_BACKEND_FFTZ
#include "reference_fft_fftz.h"
#else
#include "reference_fft_fftw.h"
#endif

extern bool use_reference_wisdom;

namespace reference_fft
{
    inline constexpr const char* name            = reference_fft_backend::name;
    inline constexpr bool        supports_wisdom = reference_fft_backend::supports_wisdom;
    using reference_fft_backend::export_wisdom;
    using reference_fft_backend::import_wisdom;
    using reference_fft_backend::initialize;

    template <typename T>
    using real_t = std::conditional_t<std::is_same_v<T, double>, double, float>;

    template <typename T>
    using plan = reference_fft_backend::plan<real_t<T>>;

    template <typename T>
    plan<T> empty_plan()
    {
        return reference_fft_backend::empty_plan<real_t<T>>();
    }

    inline std::ptrdiff_t checked_index(size_t value)
    {
        if(value > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
            throw std::overflow_error("Reference FFT dimension/stride exceeds ptrdiff_t");
        return static_cast<std::ptrdiff_t>(value);
    }

    template <typename T>
    plan<T> make_plan(const std::vector<size_t>& length,
                      const std::vector<size_t>& input_stride,
                      const std::vector<size_t>& output_stride,
                      size_t                     batches,
                      size_t                     input_distance,
                      size_t                     output_distance,
                      fft_transform_type         type,
                      std::vector<hostbuf>&      input,
                      std::vector<hostbuf>&      output)
    {
        if(length.empty() || length.size() != input_stride.size()
           || length.size() != output_stride.size())
            throw std::invalid_argument("Invalid reference FFT dimensions");
        std::vector<dimension> dims;
        for(size_t i = 0; i < length.size(); ++i)
            dims.push_back({checked_index(length[i]),
                            checked_index(input_stride[i]),
                            checked_index(output_stride[i])});
        return reference_fft_backend::make_plan<real_t<T>>(
            type,
            dims,
            {checked_index(batches), checked_index(input_distance), checked_index(output_distance)},
            input.front().data(),
            output.empty() ? nullptr : output.front().data(),
            use_reference_wisdom);
    }

    template <typename T>
    void execute(fft_transform_type    type,
                 const plan<T>&        p,
                 std::vector<hostbuf>& input,
                 std::vector<hostbuf>& output)
    {
        if constexpr(std::is_same_v<T, rocfft_fp16>)
        {
            auto single_input = half_to_single_copy(input.front());
            reference_fft_backend::execute<float>(
                type, p, single_input.data(), output.front().data());
            single_to_half_inplace(output.front());
        }
        else
            reference_fft_backend::execute<T>(type, p, input.front().data(), output.front().data());
    }

    template <typename T>
    void cleanup()
    {
        reference_fft_backend::cleanup<real_t<T>>();
    }

    inline size_t workspace_bytes(const std::vector<size_t>& length, size_t io_bytes)
    {
        // Retain the existing conservative reference-workspace heuristic for both
        // providers until measured FFTZ peak-memory data supports a tighter estimate.
        const size_t factor
            = std::any_of(length.begin(),
                          length.end(),
                          [](size_t n) { return n > 2 && max_prime_factor(n) > 13; })
                  ? 5
                  : 2;
        if(io_bytes > std::numeric_limits<size_t>::max() / factor)
            throw std::overflow_error("Reference FFT workspace estimate overflow");
        return io_bytes * factor;
    }
}
