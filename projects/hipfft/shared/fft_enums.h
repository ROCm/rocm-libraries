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
#include <string_view>
#include <type_traits>
#include <utility>

// type-trait to identify fft-specific enums defined herein
template <typename enumT, std::enable_if_t<std::is_enum_v<enumT>, bool> = true>
struct is_fft_enum : std::false_type
{
};

template <typename enumT, std::enable_if_t<std::is_enum_v<enumT>, bool> = true>
constexpr bool is_fft_enum_v = is_fft_enum<enumT>::value;

// Constexpr enum-to-string map for fft_enum types.
// Each specialization provides:
//   - type_name: a human-readable label for the enum type (used in error messages)
//   - entries[]: an array of (value, name) pairs covering all valid enum values
// These are the single source of truth for validation (validate_or_throw),
// serialization (fft_enum_to_string), and deserialization (fft_enum_from_string).
template <typename enumT, std::enable_if_t<is_fft_enum_v<enumT>, bool> = true>
struct fft_enum_map;

// Generic validate_or_throw: checks a runtime value against fft_enum_map<enumT>::entries.
template <typename enumT, std::enable_if_t<is_fft_enum_v<enumT>, bool> = true>
inline void validate_or_throw(enumT val, const std::string& func_name)
{
    for(const auto& [v, name] : fft_enum_map<enumT>::entries)
        if(v == val)
            return;
    throw std::invalid_argument(std::string("invalid ")
                                + std::string(fft_enum_map<enumT>::type_name) + " for "
                                + func_name);
}

/**
 * @brief Generalized validator for any sequence of fft-specific enums
 * 
 * @tparam T fft-specific type of enum.
 * @tparam Args template pack of possible additional fft-specific types of enum.
 * @param func_name name of the calling function (reported in exception's message if validation fails).
 * @param val value of enum to be validated.
 * @param args values of possible additional fft-specific enums to be validated.
 */
template <typename enumT, typename... Args, std::enable_if_t<is_fft_enum_v<enumT>, bool> = true>
inline void validate_enums_or_throw(const std::string& func_name, enumT val, Args... args)
{
    validate_or_throw(val, func_name);
    if constexpr(sizeof...(args) > 0)
        validate_enums_or_throw(func_name, args...);
}

// Generic runtime enum-to-string: looks up a runtime value in fft_enum_map<enumT>::entries.
template <typename enumT, std::enable_if_t<is_fft_enum_v<enumT>, bool> = true>
inline std::string fft_enum_to_string(enumT v)
{
    for(const auto& [val, name] : fft_enum_map<enumT>::entries)
        if(val == v)
            return std::string(name);
    throw std::invalid_argument("Unexpected value of " + std::string(fft_enum_map<enumT>::type_name)
                                + " given to fft_enum_to_string()");
}

// Generic runtime string-to-enum: longest prefix match against fft_enum_map<E>::entries.
// Optionally writes the number of characters consumed to *consumed.
template <typename enumT, std::enable_if_t<is_fft_enum_v<enumT>, bool> = true>
inline enumT fft_enum_from_string(std::string_view input, size_t* consumed = nullptr)
{
    enumT  best_val{};
    size_t best_len = 0;
    for(const auto& [val, entry_name] : fft_enum_map<enumT>::entries)
        if(entry_name.size() > best_len && input.substr(0, entry_name.size()) == entry_name)
        {
            best_val = val;
            best_len = entry_name.size();
        }
    if(best_len == 0)
        throw std::invalid_argument("No matching " + std::string(fft_enum_map<enumT>::type_name)
                                    + " at start of: " + std::string(input));
    if(consumed)
        *consumed = best_len;
    return best_val;
}

// Overload for sequential parsing: reads an enum value starting at input[pos]
// and advances pos past the consumed characters on success.
template <typename enumT, std::enable_if_t<is_fft_enum_v<enumT>, bool> = true>
inline enumT fft_enum_from_string(const char* input, size_t& pos)
{
    size_t consumed = 0;
    enumT  result   = fft_enum_from_string<enumT>(std::string_view(input + pos), &consumed);
    pos += consumed;
    return result;
}

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

template <>
struct fft_enum_map<fft_status>
{
    static constexpr std::string_view                        type_name = "status";
    static constexpr std::pair<fft_status, std::string_view> entries[] = {
        {fft_status_success, "success"},
        {fft_status_failure, "failure"},
        {fft_status_invalid_arg_value, "invalid_arg_value"},
        {fft_status_invalid_dimensions, "invalid_dimensions"},
        {fft_status_invalid_array_type, "invalid_array_type"},
        {fft_status_invalid_strides, "invalid_strides"},
        {fft_status_invalid_distance, "invalid_distance"},
        {fft_status_invalid_offset, "invalid_offset"},
        {fft_status_invalid_work_buffer, "invalid_work_buffer"},
    };
};

// Transform types and corresponding helpers
enum fft_transform_type
{
    fft_transform_type_complex_forward,
    fft_transform_type_complex_inverse,
    fft_transform_type_real_forward,
    fft_transform_type_real_inverse,
};

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

template <>
struct fft_enum_map<fft_transform_type>
{
    static constexpr std::string_view                                type_name = "transform type";
    static constexpr std::pair<fft_transform_type, std::string_view> entries[] = {
        {fft_transform_type_complex_forward, "complex_forward"},
        {fft_transform_type_complex_inverse, "complex_inverse"},
        {fft_transform_type_real_forward, "real_forward"},
        {fft_transform_type_real_inverse, "real_inverse"},
    };
};

// Floating-point precision and corresponding helpers

enum fft_precision
{
    fft_precision_half,
    fft_precision_single,
    fft_precision_double,
};

template <>
struct is_fft_enum<fft_precision, true> : std::true_type
{
};

template <>
struct fft_enum_map<fft_precision>
{
    static constexpr std::string_view                           type_name = "precision";
    static constexpr std::pair<fft_precision, std::string_view> entries[] = {
        {fft_precision_half, "half"},
        {fft_precision_single, "single"},
        {fft_precision_double, "double"},
    };
};

// input/output flag and corresponding helpers
enum fft_io
{
    fft_io_in,
    fft_io_out
};

template <>
struct is_fft_enum<fft_io, true> : std::true_type
{
};

template <>
struct fft_enum_map<fft_io>
{
    static constexpr std::string_view                    type_name = "io flag";
    static constexpr std::pair<fft_io, std::string_view> entries[] = {
        {fft_io_in, "input"},
        {fft_io_out, "output"},
    };
};

inline fft_io other(fft_io io)
{
    validate_or_throw(io, "other");
    return io == fft_io_in ? fft_io_out : fft_io_in;
}

// auto-allocation setting and corresponding helpers
enum fft_auto_allocation
{
    fft_auto_allocation_on,
    fft_auto_allocation_off,
    fft_auto_allocation_default
};

template <>
struct is_fft_enum<fft_auto_allocation, true> : std::true_type
{
};

template <>
struct fft_enum_map<fft_auto_allocation>
{
    static constexpr std::string_view type_name = "auto-allocation setting";
    static constexpr std::pair<fft_auto_allocation, std::string_view> entries[] = {
        {fft_auto_allocation_on, "on"},
        {fft_auto_allocation_off, "off"},
        {fft_auto_allocation_default, "default"},
    };
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

template <>
struct fft_enum_map<fft_input_generator>
{
    static constexpr std::string_view                                 type_name = "input generator";
    static constexpr std::pair<fft_input_generator, std::string_view> entries[] = {
        {fft_input_random_generator_device, "random_generator_device"},
        {fft_input_random_generator_host, "random_generator_host"},
        {fft_input_generator_device, "generator_device"},
        {fft_input_generator_host, "generator_host"},
    };
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

inline bool array_type_is_planar(fft_array_type array_type)
{
    return array_type == fft_array_type_complex_planar
           || array_type == fft_array_type_hermitian_planar;
}

inline bool array_type_is_interleaved(fft_array_type array_type)
{
    return array_type == fft_array_type_complex_interleaved
           || array_type == fft_array_type_hermitian_interleaved;
}

template <>
struct is_fft_enum<fft_array_type, true> : std::true_type
{
};

template <>
struct fft_enum_map<fft_array_type>
{
    static constexpr std::string_view                            type_name = "array type";
    static constexpr std::pair<fft_array_type, std::string_view> entries[] = {
        {fft_array_type_complex_interleaved, "complex_interleaved"},
        {fft_array_type_complex_planar, "complex_planar"},
        {fft_array_type_real, "real"},
        {fft_array_type_hermitian_interleaved, "hermitian_interleaved"},
        {fft_array_type_hermitian_planar, "hermitian_planar"},
        {fft_array_type_unset, "unset"},
    };
};

// Result placement and corresponding helpers

enum fft_result_placement
{
    fft_placement_inplace,
    fft_placement_notinplace,
};

template <>
struct is_fft_enum<fft_result_placement, true> : std::true_type
{
};

template <>
struct fft_enum_map<fft_result_placement>
{
    static constexpr std::string_view                                  type_name = "placement";
    static constexpr std::pair<fft_result_placement, std::string_view> entries[] = {
        {fft_placement_inplace, "ip"},
        {fft_placement_notinplace, "op"},
    };
};

// callback functions
enum fft_callback_type
{
    fft_callback_type_none, // don't run callbacks
    fft_callback_type_funcptr, // run callbacks specified via device function pointer
    fft_callback_type_jit, // run jit callbacks, where users provide a function as compiled SPIR-V
};

template <>
struct is_fft_enum<fft_callback_type, true> : std::true_type
{
};

template <>
struct fft_enum_map<fft_callback_type>
{
    static constexpr std::string_view                               type_name = "callback type";
    static constexpr std::pair<fft_callback_type, std::string_view> entries[] = {
        {fft_callback_type_none, "none"},
        {fft_callback_type_funcptr, "funcptr"},
    };
};

#endif // FFT_ENUMS_H
