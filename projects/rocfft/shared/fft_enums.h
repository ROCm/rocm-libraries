// Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFT_ENUMS_H
#define FFT_ENUMS_H

#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>

// type-trait to identify fft-specific enums defined herein
template <typename T, std::enable_if_t<std::is_enum_v<T>, bool> = true>
struct is_fft_enum : std::false_type
{
};

template <typename T, std::enable_if_t<std::is_enum_v<T>, bool> = true>
constexpr bool is_fft_enum_v = is_fft_enum<T>::value;

// Return codes
enum fft_status
{
    fft_status_success,
    fft_status_failure,
    fft_status_invalid_arg_value,
    fft_status_invalid_dimensions,
    fft_status_invalid_array_type,
    fft_status_invalid_strides,
    fft_status_invalid_distance,
    fft_status_invalid_offset,
    fft_status_invalid_work_buffer,
};

template <>
struct is_fft_enum<fft_status, true> : std::true_type
{
};

// Transform types and corresponding helpers
enum fft_transform_type
{
    fft_transform_type_complex_forward,
    fft_transform_type_complex_inverse,
    fft_transform_type_real_forward,
    fft_transform_type_real_inverse,
};

inline void validate_or_throw(fft_transform_type dft_kind, const std::string& func_name)
{
    switch(dft_kind)
    {
    case fft_transform_type::fft_transform_type_complex_forward:
    case fft_transform_type::fft_transform_type_complex_inverse:
    case fft_transform_type::fft_transform_type_real_forward:
    case fft_transform_type::fft_transform_type_real_inverse:
        return;
    default:
        throw std::invalid_argument("invalid type of transform for " + func_name);
    }
}

inline constexpr bool is_real(const fft_transform_type& dft_type)
{
    return dft_type == fft_transform_type_real_forward
           || dft_type == fft_transform_type_real_inverse;
}
inline constexpr bool is_complex(const fft_transform_type& dft_type)
{
    return !is_real(dft_type);
}
inline constexpr bool is_fwd(const fft_transform_type& dft_type)
{
    return dft_type == fft_transform_type_real_forward
           || dft_type == fft_transform_type_complex_forward;
}
inline constexpr bool is_bwd(const fft_transform_type& dft_type)
{
    return !is_fwd(dft_type);
}

template <>
struct is_fft_enum<fft_transform_type, true> : std::true_type
{
};

// Floating-point precision and corresponding helpers

enum fft_precision
{
    fft_precision_half,
    fft_precision_single,
    fft_precision_double,
};

inline void validate_or_throw(fft_precision prec, const std::string& func_name)
{
    switch(prec)
    {
    case fft_precision::fft_precision_half:
    case fft_precision::fft_precision_single:
    case fft_precision::fft_precision_double:
        return;
    default:
        throw std::invalid_argument("invalid precision for " + func_name);
    }
}

template <>
struct is_fft_enum<fft_precision, true> : std::true_type
{
};

// input/output flag and corresponding helpers
enum fft_io
{
    fft_io_in,
    fft_io_out
};

inline void validate_or_throw(fft_io io, const std::string& func_name)
{
    switch(io)
    {
    case fft_io::fft_io_in:
    case fft_io::fft_io_out:
        return;
    default:
        throw std::invalid_argument("invalid io flag for " + func_name);
    }
}

template <>
struct is_fft_enum<fft_io, true> : std::true_type
{
};

// auto-allocation setting and corresponding helpers
enum fft_auto_allocation
{
    fft_auto_allocation_on,
    fft_auto_allocation_off,
    fft_auto_allocation_default
};

inline void validate_or_throw(fft_auto_allocation auto_alloc, const std::string& func_name)
{
    switch(auto_alloc)
    {
    case fft_auto_allocation::fft_auto_allocation_on:
    case fft_auto_allocation::fft_auto_allocation_off:
    case fft_auto_allocation::fft_auto_allocation_default:
        return;
    default:
        throw std::invalid_argument("invalid auto-allocation setting for " + func_name);
    }
}

template <>
struct is_fft_enum<fft_auto_allocation, true> : std::true_type
{
};

// input generator labels and corresponding helpers

// fft_input_generator: linearly spaced sequence in [-0.5,0.5]
// fft_input_random_generator: pseudo-random sequence in [-0.5,0.5]
enum fft_input_generator
{
    fft_input_random_generator_device,
    fft_input_random_generator_host,
    fft_input_generator_device,
    fft_input_generator_host,
};

inline void validate_or_throw(fft_input_generator input_gen, const std::string& func_name)
{
    switch(input_gen)
    {
    case fft_input_generator::fft_input_random_generator_device:
    case fft_input_generator::fft_input_random_generator_host:
    case fft_input_generator::fft_input_generator_device:
    case fft_input_generator::fft_input_generator_host:
        return;
    default:
        throw std::invalid_argument("invalid input generator for " + func_name);
    }
}

inline bool is_host_generator(const fft_input_generator& gen)
{
    return gen == fft_input_generator::fft_input_random_generator_host
           || gen == fft_input_generator::fft_input_generator_host;
}
inline bool is_device_generator(const fft_input_generator& gen)
{
    return gen == fft_input_generator::fft_input_random_generator_device
           || gen == fft_input_generator::fft_input_generator_device;
}
inline bool is_random_generator(const fft_input_generator& gen)
{
    return gen == fft_input_generator::fft_input_random_generator_host
           || gen == fft_input_generator::fft_input_random_generator_device;
}
inline bool is_deterministic_generator(const fft_input_generator& gen)
{
    return gen == fft_input_generator::fft_input_generator_host
           || gen == fft_input_generator::fft_input_generator_device;
}

template <>
struct is_fft_enum<fft_input_generator, true> : std::true_type
{
};

// Array types and corresponding helpers
enum fft_array_type
{
    fft_array_type_complex_interleaved,
    fft_array_type_complex_planar,
    fft_array_type_real,
    fft_array_type_hermitian_interleaved,
    fft_array_type_hermitian_planar,
    fft_array_type_unset,
};

inline void validate_or_throw(fft_array_type array_type, const std::string& func_name)
{
    switch(array_type)
    {

    case fft_array_type::fft_array_type_complex_interleaved:
    case fft_array_type::fft_array_type_complex_planar:
    case fft_array_type::fft_array_type_real:
    case fft_array_type::fft_array_type_hermitian_interleaved:
    case fft_array_type::fft_array_type_hermitian_planar:
    case fft_array_type::fft_array_type_unset:
        return;
    default:
        throw std::invalid_argument("invalid array type for " + func_name);
    }
}

template <>
struct is_fft_enum<fft_array_type, true> : std::true_type
{
};

// Result placement and corresponding helpers

enum fft_result_placement
{
    fft_placement_inplace,
    fft_placement_notinplace,
};

inline void validate_or_throw(fft_result_placement placement, const std::string& func_name)
{
    switch(placement)
    {
    case fft_result_placement::fft_placement_inplace:
    case fft_result_placement::fft_placement_notinplace:
        return;
    default:
        throw std::invalid_argument("invalid placement for " + func_name);
    }
}

template <>
struct is_fft_enum<fft_result_placement, true> : std::true_type
{
};

/**
 * @brief Generalized validator for any sequence of fft-specific enums
 * 
 * @tparam T fft-specific type of enum.
 * @tparam Args template pack of possible additional fft-specific types of enum.
 * @param func_name name of the calling function (reported in exception's message if validation fails).
 * @param val value of enum to be validated.
 * @param args values of possible additional fft-specific enums to be validated.
 */
template <typename T, typename... Args, std::enable_if_t<is_fft_enum_v<T>, bool> = true>
inline void validate_enums_or_throw(const std::string& func_name, T val, Args... args)
{
    validate_or_throw(val, func_name);
    if constexpr(sizeof...(args) > 0)
        validate_enums_or_throw(func_name, args...);
}

#endif // FFT_ENUMS_H
