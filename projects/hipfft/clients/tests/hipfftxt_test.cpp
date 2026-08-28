// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Tests for the hipfftXt multi-GPU API: descriptor allocation, H2D/D2H/D2D
// memcpy, multi-device transform execution, and numerical accuracy verification.

#include "hipfft/hipfft.h"
#include "hipfft/hipfftXt.h"

#include <algorithm>
#include <complex>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>

#include "../../shared/fft_enums.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/params_gen.h"
#include "../../shared/reference_fft_data.h"
#include "../../shared/rocfft_hip.h"
#include "../../shared/test_params.h"
#include "../hipfft_params.h"

#ifdef __HIP_PLATFORM_NVIDIA__
DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#endif
#include <hip/hip_runtime_api.h>
#ifdef __HIP_PLATFORM_NVIDIA__
DISABLE_WARNING_POP
#endif

// Minimum number of random probes per device in hipfftXt data distribution verification
extern size_t min_probes_per_dev_for_xt;

// Token string for manual hipfftXt test (set from CLI in gtest_main.cpp)
extern std::string hipfftxt_test_token;

template <>
struct is_fft_enum<hipfftXtSubFormat, true> : std::true_type
{
};

template <>
struct fft_enum_map<hipfftXtSubFormat>
{
    static constexpr std::string_view type_name = "multi-device descriptor subformat";
    static constexpr std::pair<hipfftXtSubFormat, std::string_view> entries[] = {
        {HIPFFT_XT_FORMAT_INPUT, "HIPFFT_XT_FORMAT_INPUT"},
        {HIPFFT_XT_FORMAT_OUTPUT, "HIPFFT_XT_FORMAT_OUTPUT"},
        {HIPFFT_XT_FORMAT_INPLACE, "HIPFFT_XT_FORMAT_INPLACE"},
        {HIPFFT_XT_FORMAT_INPLACE_SHUFFLED, "HIPFFT_XT_FORMAT_INPLACE_SHUFFLED"},
        {HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED, "HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED"},
        {HIPFFT_FORMAT_UNDEFINED, "HIPFFT_FORMAT_UNDEFINED"},
    };
};

// Deterministic, order-independent PRNG for a single test case (and possible I/O
// role being tested). Seeding from random_seed plus a hash of the case's canonical
// token makes each test's random sampling reproducible in isolation -- independent
// of test execution order or sharding, and exactly reproducible via --hipfftxt_test_token.
// The io argument may be irrelevant and omitted in some contexts; a default value
// of fft_io_in is used then.
static std::mt19937 make_test_prng(const std::string& token, fft_io io = fft_io_in)
{
    std::seed_seq seed{static_cast<size_t>(random_seed),
                       static_cast<size_t>(std::hash<std::string>{}(token)),
                       static_cast<size_t>(io)};
    return std::mt19937(seed);
}

#ifdef __HIP_PLATFORM_AMD__
static constexpr bool rocfft_backend = true;
#else
static constexpr bool rocfft_backend = false;
#endif

static fft_result_placement placement_from_subformat(hipfftXtSubFormat fmt)
{
    return fmt == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED || fmt == HIPFFT_XT_FORMAT_INPLACE
               ? fft_placement_inplace
               : fft_placement_notinplace;
}

static hipfftXtSubFormat natural_output_desc_format_for(hipfftXtSubFormat input_format,
                                                        size_t            batch_sz)
{
    if(batch_sz == 0)
        throw std::invalid_argument("natural_output_desc_format_for: batch_sz must be > 0");
    // Possible use case of HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED is unclear yet
    // (to be investigated with cuFFT backend)
    switch(input_format)
    {
    case HIPFFT_XT_FORMAT_INPUT:
        [[fallthrough]];
    case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
        return HIPFFT_XT_FORMAT_OUTPUT;
    case HIPFFT_XT_FORMAT_OUTPUT:
        // acceptable for the input descriptor format for complex plans or if the
        // input descriptor was allocated as output descriptor of the reciprocal
        // plan for instance...
        return HIPFFT_XT_FORMAT_INPUT;
    // in-place formats are modified through execution only for unbatched use cases
    // (slab decompositions are used for the unbatched cases motivating the flip,
    // batched cases are embarrassingly parallel and don't require any flip)
    case HIPFFT_XT_FORMAT_INPLACE:
        return batch_sz == 1 ? HIPFFT_XT_FORMAT_INPLACE_SHUFFLED : HIPFFT_XT_FORMAT_INPLACE;
    case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        return batch_sz == 1 ? HIPFFT_XT_FORMAT_INPLACE : HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
    case HIPFFT_FORMAT_UNDEFINED:
        return HIPFFT_FORMAT_UNDEFINED;
    default:
        throw std::runtime_error("Invalid value of input descriptor's format detected in "
                                 "natural_output_desc_format_for()");
    }
}

struct spmd_hipfft_params
{
    spmd_hipfft_params(fft_transform_type               _dft_type,
                       std::vector<int>                 _gpus,
                       size_t                           _batch,
                       std::vector<size_t>              _transform_lengths,
                       fft_precision                    _precision,
                       hipfftXtSubFormat                _input_desc_format,
                       std::optional<hipfftXtSubFormat> _output_desc_format = std::nullopt)
        : dft_type(_dft_type)
        , gpus(std::move(_gpus))
        , batch(_batch)
        , transform_lengths(std::move(_transform_lengths))
        , precision(_precision)
        , input_desc_format(_input_desc_format)
        , explicit_output_desc_format(_output_desc_format)
    {
        if(gpus.empty() || std::any_of(gpus.begin(), gpus.end(), [](int i) {
               return i < 0 || i >= rocfft_scoped_device::device_count();
           }))
        {
            throw std::invalid_argument(
                "spmd_hipfft_params: invalid gpus (must be non-empty and each < device_count)");
        }

        if(transform_lengths.empty() || transform_lengths.size() > 3)
            throw std::invalid_argument(
                "spmd_hipfft_params: transform_lengths must be non-empty and of rank <= 3");
        if(batch == 0
           || std::any_of(transform_lengths.begin(), transform_lengths.end(), [](const auto& l) {
                  return l == 0;
              }))
        {
            throw std::invalid_argument(
                "spmd_hipfft_params: batch and transform lengths must be non-zero");
        }
        validate_enums_or_throw("spmd_hipfft_params", dft_type, precision);
        if(precision != fft_precision_single && precision != fft_precision_double)
            throw std::invalid_argument(
                "spmd_hipfft_params: precision must be single or double for hipfftxt tests");

        validate_or_throw(input_desc_format, "spmd_hipfft_params");
        if(explicit_output_desc_format)
        {
            validate_or_throw(*explicit_output_desc_format, "spmd_hipfft_params");
            if(placement_from_subformat(*explicit_output_desc_format)
               != placement_from_subformat(input_desc_format))
            {
                throw std::invalid_argument("spmd_hipfft_params: explicit output_desc_format must "
                                            "be consistent with input_desc_format placement");
            }
        }
    }

    static spmd_hipfft_params make_reciprocal(const spmd_hipfft_params& params)
    {
        return spmd_hipfft_params(params.reciprocal_transform_type(),
                                  params.gpus,
                                  params.batch,
                                  params.transform_lengths,
                                  params.precision,
                                  params.output_desc_format(),
                                  params.input_desc_format);
    }

    size_t ngpus() const
    {
        return gpus.size();
    }

    const fft_transform_type  dft_type;
    const std::vector<int>    gpus;
    const size_t              batch;
    const std::vector<size_t> transform_lengths;
    const fft_precision       precision;
    const hipfftXtSubFormat   input_desc_format;

    fft_result_placement placement() const
    {
        return placement_from_subformat(input_desc_format);
    }

    hipfftXtSubFormat output_desc_format() const
    {
        if(explicit_output_desc_format.has_value())
            return explicit_output_desc_format.value();
        return natural_output_desc_format_for(input_desc_format, batch);
    }

    int hipfft_exec_dir() const
    {
        return is_fwd(dft_type) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
    }

    std::vector<size_t> logical_spans(fft_io io) const
    {
        validate_or_throw(io, "spmd_hipfft_params::logical_spans");
        auto ret = transform_lengths;
        if((dft_type == fft_transform_type_real_forward && io == fft_io_out)
           || (dft_type == fft_transform_type_real_inverse && io == fft_io_in))
            ret.back() = ret.back() / 2 + 1;
        return ret;
    }

    bool has_real_data_on(fft_io io) const
    {
        validate_or_throw(io, "spmd_hipfft_params::has_real_data_on");
        if(io == fft_io_in)
            return dft_type == fft_transform_type_real_forward;
        return dft_type == fft_transform_type_real_inverse;
    }

    hipfftType_t hipfft_transform_type() const
    {
        switch(dft_type)
        {
        case fft_transform_type_real_forward:
            return precision == fft_precision_single ? HIPFFT_R2C : HIPFFT_D2Z;
        case fft_transform_type_real_inverse:
            return precision == fft_precision_single ? HIPFFT_C2R : HIPFFT_Z2D;
        case fft_transform_type_complex_forward:
            [[fallthrough]];
        case fft_transform_type_complex_inverse:
            return precision == fft_precision_single ? HIPFFT_C2C : HIPFFT_Z2Z;
        default:
            throw std::logic_error("invalid dft_type");
        }
    }

    fft_transform_type reciprocal_transform_type() const
    {
        switch(dft_type)
        {
        case fft_transform_type_real_forward:
            return fft_transform_type_real_inverse;
        case fft_transform_type_real_inverse:
            return fft_transform_type_real_forward;
        case fft_transform_type_complex_forward:
            return fft_transform_type_complex_inverse;
        case fft_transform_type_complex_inverse:
            return fft_transform_type_complex_forward;
        default:
            throw std::logic_error(
                "spmd_hipfft_params::reciprocal_transform_type: invalid dft_type");
        }
    }

    fft_array_type array_type(fft_io io) const
    {
        validate_or_throw(io, "spmd_hipfft_params::data_type");
        if(is_complex(dft_type))
            return fft_array_type_complex_interleaved;
        return has_real_data_on(io) ? fft_array_type_real : fft_array_type_hermitian_interleaved;
    }

    std::vector<size_t> global_strides(fft_io io, fft_result_placement placement) const
    {
        validate_enums_or_throw("spmd_hipfft_params::global_strides", io, placement);
        return default_strides(dft_type, placement, io, transform_lengths);
    }
    size_t global_dist(fft_io io, fft_result_placement placement) const
    {
        validate_enums_or_throw("spmd_hipfft_params::global_dist", io, placement);
        return default_distance(dft_type, placement, io, transform_lengths, batch);
    }
    size_t global_byte_size(fft_io io, fft_result_placement placement) const
    {
        validate_enums_or_throw("spmd_hipfft_params::global_byte_size", io, placement);
        std::vector<fft_io> relevant_ios = {io};
        if(placement == fft_placement_inplace)
            relevant_ios.push_back(other(io));
        size_t ret = 0;
        for(const auto& relevant_io : relevant_ios)
        {
            ret = std::max(ret,
                           compute_ptrdiff(logical_spans(relevant_io),
                                           global_strides(relevant_io, placement),
                                           batch,
                                           global_dist(relevant_io, placement))
                               * var_size<size_t>(precision,
                                                  has_real_data_on(relevant_io)
                                                      ? fft_array_type_real
                                                      : fft_array_type_complex_interleaved));
        }
        return ret;
    }

    size_t global_buffer_index(size_t                     global_batch_idx,
                               const std::vector<size_t>& global_multi_idx,
                               fft_io                     io,
                               fft_result_placement       placement) const
    {
        validate_or_throw(io, "spmd_hipfft_params::global_buffer_index");
        validate_global_batch_idx(global_batch_idx);
        validate_global_multi_idx(global_multi_idx, io);
        const auto strides  = default_strides(dft_type, placement, io, transform_lengths);
        const auto distance = default_distance(dft_type, placement, io, transform_lengths, batch);
        return std::inner_product(global_multi_idx.begin(),
                                  global_multi_idx.end(),
                                  strides.begin(),
                                  global_batch_idx * distance);
    }

    std::pair<int, size_t> get_local_buffer_index(size_t                     global_batch_idx,
                                                  const std::vector<size_t>& global_multi_idx,
                                                  fft_io                     io,
                                                  hipfftXtSubFormat          desc_format) const
    {
        validate_enums_or_throw("spmd_hipfft_params::get_local_buffer_index", io, desc_format);
        validate_global_batch_idx(global_batch_idx);
        validate_global_multi_idx(global_multi_idx, io);
        const auto             placement = placement_from_subformat(desc_format);
        std::pair<int, size_t> ret; // (device index, local offset in corresponding device chunk)
        if(batch > 1)
        {
            // For batched transforms, the batch is split across the GPUs
            // local strides/distances match the global strides/distances
            ret.first = get_device_index(batch, ngpus(), global_batch_idx);
            const auto lower_global_batch
                = ret.first * (batch / ngpus())
                  + std::min(static_cast<size_t>(ret.first), batch % ngpus());
            const auto global_distance
                = default_distance(dft_type, placement, io, transform_lengths, batch);
            const auto global_strides = default_strides(dft_type, placement, io, transform_lengths);
            ret.second
                = std::inner_product(global_multi_idx.begin(),
                                     global_multi_idx.end(),
                                     global_strides.begin(),
                                     (global_batch_idx - lower_global_batch) * global_distance);
        }
        else
        {
            if(global_multi_idx.size() == 1)
            {
                throw std::runtime_error("No test-side support for 1D unbatched transforms yet.");
            }
            if(placement != fft_placement_inplace)
            {
                throw std::logic_error(
                    "No test-side support implemented (because none is expected) for multi-device "
                    "out-of-place unbatched transforms.");
            }
            const size_t split_dim           = desc_format == HIPFFT_XT_FORMAT_INPLACE ? 0 : 1;
            const auto   global_logical_span = logical_spans(io);
            ret.first                        = get_device_index(
                global_logical_span[split_dim], ngpus(), global_multi_idx[split_dim]);
            const auto split_dim_local_span
                = global_logical_span[split_dim] / ngpus()
                  + (static_cast<size_t>(ret.first) < global_logical_span[split_dim] % ngpus() ? 1
                                                                                               : 0);
            auto local_multi_idx = global_multi_idx;
            local_multi_idx[split_dim] -= (ret.first * (global_logical_span[split_dim] / ngpus())
                                           + std::min(static_cast<size_t>(ret.first),
                                                      global_logical_span[split_dim] % ngpus()));
            if(split_dim != global_multi_idx.size() - 1)
            {
                // local strides behave like default strides, only for the local data chunk's
                // (partial) lengths
                auto partial_lengths       = transform_lengths;
                partial_lengths[split_dim] = split_dim_local_span;
                const auto local_strides
                    = default_strides(dft_type, placement, io, partial_lengths);
                ret.second = std::inner_product(local_multi_idx.begin(),
                                                local_multi_idx.end(),
                                                local_strides.begin(),
                                                static_cast<size_t>(0));
            }
            else
            {
                // Split is on last dim: local chunk is packed row-major, no padding, regardless
                // of transform type, overall placement, I/O data, etc.
                auto local_logical_span       = global_logical_span;
                local_logical_span[split_dim] = split_dim_local_span;
                std::vector<size_t> local_strides(local_logical_span.size());
                local_strides.back() = 1;
                for(size_t dim = local_logical_span.size() - 1; dim-- > 0;)
                    local_strides[dim] = local_strides[dim + 1] * local_logical_span[dim + 1];
                ret.second = std::inner_product(local_multi_idx.begin(),
                                                local_multi_idx.end(),
                                                local_strides.begin(),
                                                static_cast<size_t>(0));
            }
        }
        return ret;
    }

    friend std::ostream& operator<<(std::ostream& stream, const spmd_hipfft_params& params)
    {
        stream << "dft type: " << fft_enum_to_string(params.dft_type) << ", transform lengths: (";
        for(auto it = params.transform_lengths.begin(); it != params.transform_lengths.end(); ++it)
            stream << (it != params.transform_lengths.begin() ? ", " : "") << *it;
        stream << "), precision: " << fft_enum_to_string(params.precision)
               << ", batch: " << params.batch << ", ngpus: " << params.ngpus()
               << ", input subformat: " << fft_enum_to_string(params.input_desc_format)
               << ", output subformat: " << fft_enum_to_string(params.output_desc_format());
        return stream;
    }

    // Produces a token string from this instance's fields:
    // {dft_type}_len_{L0}_{L1}_..._{precision}_batch_{batch}_ngpus_{ngpus}_input_fmt_{input_desc_format}_output_fmt_{output_desc_format}
    std::string str() const
    {
        std::ostringstream oss;
        oss << fft_enum_to_string(dft_type) << token_sep << lengths_tag << token_sep;
        for(auto len : transform_lengths)
            oss << len << token_sep;
        oss << fft_enum_to_string(precision) << token_sep << batch_tag << token_sep << batch
            << token_sep << ngpus_tag << token_sep << ngpus() << token_sep << input_fmt_tag
            << token_sep << fft_enum_to_string(input_desc_format) << token_sep << output_fmt_tag
            << token_sep << fft_enum_to_string(output_desc_format());
        return oss.str();
    }

    // Constructs a spmd_hipfft_params from a token string (as produced by str()).
    static spmd_hipfft_params make_from_token(std::string_view token)
    {
        const auto* token_raw = token.data();
        size_t      pos       = 0;

        // Helper: expect and consume a separator, optionally followed by a tag and another
        // separator. With no tag: consumes one token_sep. With a tag: consumes
        // token_sep + tag + token_sep.
        auto expect_sep = [&](std::string_view tag = {}) {
            if(token.substr(pos, token_sep.size()) != token_sep)
                throw std::invalid_argument(
                    "make_from_token: expected separator"
                    + (tag.empty() ? std::string{} : " before " + std::string(tag)));
            pos += token_sep.size();
            if(!tag.empty())
            {
                if(token.substr(pos, tag.size()) != tag)
                    throw std::invalid_argument("make_from_token: expected tag: "
                                                + std::string(tag));
                pos += tag.size();
                if(token.substr(pos, token_sep.size()) != token_sep)
                    throw std::invalid_argument("make_from_token: expected separator after "
                                                + std::string(tag));
                pos += token_sep.size();
            }
        };

        // Helper: parse a size_t at pos up to the next separator, advancing pos past
        // the digits to the separator position.
        auto parse_size = [&](std::string_view context) -> size_t {
            auto next_pos = token.find(token_sep, pos);
            if(next_pos == std::string_view::npos)
                next_pos = token.size();
            if(next_pos <= pos)
                throw std::invalid_argument("make_from_token: empty field for "
                                            + std::string(context));
            size_t ret = std::stoull(std::string(token.substr(pos, next_pos - pos)));
            pos        = next_pos;
            return ret;
        };

        // Parse dft_type
        const auto dft_type = fft_enum_from_string<fft_transform_type>(token_raw, pos);
        // Parse lengths: parse first value, then continue while lengths are successfully read
        expect_sep(lengths_tag);
        std::vector<size_t> transform_lengths;
        transform_lengths.push_back(parse_size("transform lengths"));
        while(token.substr(pos, token_sep.size()) == token_sep)
        {
            pos += token_sep.size();
            size_t len_to_add;
            try
            {
                len_to_add = parse_size("transform lengths");
            }
            catch(...)
            {
                break;
            }
            transform_lengths.push_back(len_to_add);
        }
        // Parse precision
        const auto precision = fft_enum_from_string<fft_precision>(token_raw, pos);
        // Parse batch
        expect_sep(batch_tag);
        const size_t batch = parse_size("batch");
        // Parse ngpus
        expect_sep(ngpus_tag);
        const size_t ngpus = parse_size("ngpus");

        std::vector<int> gpus(ngpus);
        std::iota(gpus.begin(), gpus.end(), 0);
        std::mt19937 rng(random_seed);
        std::shuffle(gpus.begin(), gpus.end(), rng);
        // Parse input descriptor format
        expect_sep(input_fmt_tag);
        const auto input_desc_format = fft_enum_from_string<hipfftXtSubFormat>(token_raw, pos);
        // Optionally parse output_format
        std::optional<hipfftXtSubFormat> explicit_output;
        if(pos < token.size() && token.substr(pos, token_sep.size()) == token_sep
           && token.substr(pos + token_sep.size(), output_fmt_tag.size()) == output_fmt_tag)
        {
            // skip _output_fmt_
            pos += token_sep.size() + output_fmt_tag.size() + token_sep.size();
            explicit_output = fft_enum_from_string<hipfftXtSubFormat>(token_raw, pos);
        }

        return spmd_hipfft_params(dft_type,
                                  std::move(gpus),
                                  batch,
                                  std::move(transform_lengths),
                                  precision,
                                  input_desc_format,
                                  explicit_output);
    }

private:
    // Token format constants shared across the hierarchy.
    static constexpr std::string_view token_sep      = "_";
    static constexpr std::string_view batch_tag      = "batch";
    static constexpr std::string_view lengths_tag    = "len";
    static constexpr std::string_view ngpus_tag      = "ngpus";
    static constexpr std::string_view input_fmt_tag  = "input_fmt";
    static constexpr std::string_view output_fmt_tag = "output_fmt";

    const std::optional<hipfftXtSubFormat> explicit_output_desc_format;

    void validate_global_batch_idx(size_t global_batch_idx) const
    {
        if(global_batch_idx >= batch)
            throw std::invalid_argument(
                "spmd_hipfft_params::validate_global_batch_idx: global_batch_idx out of range");
    }

    void validate_global_multi_idx(const std::vector<size_t>& global_multi_idx, fft_io io) const
    {
        validate_or_throw(io, "spmd_hipfft_params::validate_global_multi_idx");
        const auto global_logical_span = logical_spans(io);
        if(global_multi_idx.size() != global_logical_span.size())
            throw std::invalid_argument("spmd_hipfft_params::validate_global_multi_idx: "
                                        "global_multi_idx size mismatch");
        for(size_t dim = 0; dim < global_multi_idx.size(); ++dim)
        {
            if(global_multi_idx[dim] >= global_logical_span[dim])
                throw std::invalid_argument("spmd_hipfft_params::validate_global_multi_idx: "
                                            "global_multi_idx out of range for dimension "
                                            + std::to_string(dim));
        }
    }

    static int get_device_index(size_t global_span, size_t num_devices, size_t global_idx)
    {
        if(global_idx >= global_span)
            throw std::out_of_range(
                "spmd_hipfft_params::get_device_index: global_idx out of range");
        const auto min_span_per_dev = global_span / num_devices;
        const auto remainder        = global_span % num_devices;
        const auto split_global_idx = remainder * (min_span_per_dev + 1);
        if(global_idx < split_global_idx)
            return static_cast<int>(global_idx / (min_span_per_dev + 1));
        else
            return static_cast<int>(remainder + (global_idx - split_global_idx) / min_span_per_dev);
    }
};

static bool expects_successful_plan_creation(const spmd_hipfft_params& params)
{
    if(params.batch > 1 || params.transform_lengths.size() > 1)
        return true;
    if constexpr(rocfft_backend)
    {
        return false;
    }
    else
    {
        return is_complex(params.dft_type) && params.transform_lengths[0] % params.ngpus() == 0
               && (params.transform_lengths[0] & (params.transform_lengths[0] - 1)) == 0;
    }
}

// Creates and returns a hipFFT plan for the given params' GPU ids.
// If plan creation fails and expects_successful_plan_creation() is false, returns an
// empty (invalid) wrapper without throwing. Throws ROCFFT_FAIL on unexpected failures.
static hipfftHandle_wrapper_t make_plan(const spmd_hipfft_params& params)
{
    hipfftHandle_wrapper_t plan;
    auto                   hipfft_rt = plan.alloc_with_err();
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw ROCFFT_FAIL("make_plan: hipfftCreate failed (" + hipfftResult_string(hipfft_rt)
                          + ")");

    hipfft_rt = hipfftXtSetGPUs(
        plan, static_cast<int>(params.gpus.size()), const_cast<int*>(params.gpus.data()));
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw ROCFFT_FAIL("make_plan: hipfftXtSetGPUs failed (" + hipfftResult_string(hipfft_rt)
                          + ")");

    std::vector<size_t> workSize(params.ngpus(), std::numeric_limits<size_t>::max());
    if(params.batch > 1)
    {
        std::vector<int> lengths_int(params.transform_lengths.begin(),
                                     params.transform_lengths.end());
        hipfft_rt = hipfftMakePlanMany(plan,
                                       lengths_int.size(),
                                       lengths_int.data(),
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       0,
                                       0,
                                       params.hipfft_transform_type(),
                                       params.batch,
                                       workSize.data());
    }
    else
    {
        switch(params.transform_lengths.size())
        {
        case 1:
            hipfft_rt = hipfftMakePlan1d(plan,
                                         params.transform_lengths[0],
                                         params.hipfft_transform_type(),
                                         1,
                                         workSize.data());
            break;
        case 2:
            hipfft_rt = hipfftMakePlan2d(plan,
                                         params.transform_lengths[0],
                                         params.transform_lengths[1],
                                         params.hipfft_transform_type(),
                                         workSize.data());
            break;
        case 3:
            hipfft_rt = hipfftMakePlan3d(plan,
                                         params.transform_lengths[0],
                                         params.transform_lengths[1],
                                         params.transform_lengths[2],
                                         params.hipfft_transform_type(),
                                         workSize.data());
            break;
        default:
            throw ROCFFT_FAIL("make_plan: unsupported rank");
        }
    }
    if(hipfft_rt != HIPFFT_SUCCESS)
    {
        if(expects_successful_plan_creation(params))
            throw ROCFFT_FAIL("make_plan: plan creation unexpectedly failed ("
                              + hipfftResult_string(hipfft_rt) + ")");
        // Expected failure: return an empty wrapper
        return hipfftHandle_wrapper_t{};
    }
    if(!std::all_of(workSize.begin(), workSize.end(), [](size_t sz) {
           return sz < std::numeric_limits<size_t>::max();
       }))
    {
        throw ROCFFT_FAIL("make_plan: some workSize entry was not written by plan creation");
    }
    return plan;
}

static reference_fft_data_t make_reference_data(const spmd_hipfft_params& params)
{
    fft_params tmp;
    tmp.length    = params.transform_lengths;
    tmp.precision = params.precision;
    // Always set it out-of-place so that we can use it as is for reference CPU (requirement
    // for construction of reference_fft_data_t objects) but use the same data layout as the
    // test case's global strides/distances (regardless of test placement) so that the
    // reference results' input data can be funneled into the test's hipfftXtMemcpy as is.
    tmp.placement      = fft_placement_notinplace;
    tmp.transform_type = params.dft_type;
    tmp.nbatch         = params.batch;
    tmp.run_callbacks  = fft_callback_type_none;
    tmp.istride        = params.global_strides(fft_io_in, params.placement());
    tmp.ostride        = params.global_strides(fft_io_out, params.placement());
    tmp.idist          = params.global_dist(fft_io_in, params.placement());
    tmp.odist          = params.global_dist(fft_io_out, params.placement());
    tmp.validate(); // sets itype, otype, isize, osize, etc. from the above

    reference_fft_data_t ret(tmp);
    // reference_fft_data_t objects are unaware of our intention to feed the reference
    // results' input data as is into hipfftXtMemcpy, and the object construction may
    // have decided to re-use cached results that may have slightly different strides
    // in input data (e.g. if testing an in-place real fwd transform right after testing
    // the very same transform out-of-place and vice versa)
    // --> Verify that the reference results' input data's strides and distances
    // match what we expect in global params
    if(ret.get_params().istride != tmp.istride || ret.get_params().idist != tmp.idist)
    {
        if(verbose)
        {
            std::cout << "Reference results picked up from cache have input strides/distances that "
                         "do not match requirements: re-initializing reference results"
                      << std::endl;
        }
        ret = reference_fft_data_t(tmp);
        if(ret.get_params().istride != tmp.istride || ret.get_params().idist != tmp.idist)
            throw std::logic_error(
                "make_reference_data: re-initialized reference results still have incompatible "
                "input strides/distances for the test requirements");
    }
    return ret;
}

// Expected outcome of hipfftXtMalloc for a given plan configuration and subformat.
// ______________________________________________________________________________________________
// |                    |                     batch == 1                      |    batch > 1    |
// |                    |-----------------|-----------------|-----------------|-----------------|
// |                    |    rank == 1    |     rank == 2   |    rank == 3    |  1 <= rank <= 3 |
// |                    |-----------------|-----------------|-----------------|-----------------|
// | subformat          | R2C | C2R | C2C | R2C | C2R | C2C | R2C | C2R | C2C | R2C | C2R | C2C |
// |--------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
// | INPUT              |  /  |  /  |  Y? |  Y0 |  Y0 |  Y0 |  Y0 |  Y0 |  Y0 |  Y  |  Y  |  Y  |
// | OUTPUT             |  /  |  /  |  Y? |  Y0 |  Y0 |  Y0 |  Y0 |  Y0 |  Y0 |  Y  |  Y  |  Y  |
// | INPLACE            |  /  |  /  |  Y? |  Y  |  Y0 |  Y  |  Y  |  Y  |  Y  |  Y  |  Y  |  Y  |
// | INPLACE_SHUFFLED   |  /  |  /  |  Y? |  Y0 |  Y  |  Y  |  Y  |  Y  |  Y  |  -  |  -  |  -  |
// | 1D_INPUT_SHUFFLED  |  /  |  /  |  Y? |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
// | UNDEFINED          |  -  |  -  |  -  |  Y0 |  Y0 |  Y0 |  Y0 |  Y0 |  Y0 |  -  |  -  |  -  |
// ----------------------------------------------------------------------------------------------
// (Determined with 8 V100/CUDA 12.9 and 4 H100/CUDA 13.1; no difference observed.)
enum class xt_alloc_result
{
    accepted, // "Y"
    accepted_but_nullptrs, // "Y0" (cuFFT only; rejected on rocFFT backend)
    accepted_but_untestable, // "Y?"
    rejected, // "-"
    unreachable // "/"
};

static xt_alloc_result xt_alloc_expectation(const spmd_hipfft_params& params,
                                            const hipfftXtSubFormat&  subformat)
{
    validate_or_throw(subformat, "xt_alloc_expectation");
    if(!expects_successful_plan_creation(params))
        return xt_alloc_result::unreachable;

    const auto rank = params.transform_lengths.size();
    const bool r2c  = params.dft_type == fft_transform_type_real_forward;
    const bool c2r  = params.dft_type == fft_transform_type_real_inverse;
    const bool c2c  = is_complex(params.dft_type);

    switch(subformat)
    {
    case HIPFFT_XT_FORMAT_INPUT:
        [[fallthrough]];
    case HIPFFT_XT_FORMAT_OUTPUT:
        if(params.batch > 1)
            return xt_alloc_result::accepted;
        if(rank == 1)
            return c2c ? xt_alloc_result::accepted_but_untestable : xt_alloc_result::unreachable;
        return rocfft_backend ? xt_alloc_result::rejected : xt_alloc_result::accepted_but_nullptrs;

    case HIPFFT_XT_FORMAT_INPLACE:
        if(params.batch > 1)
            return xt_alloc_result::accepted;
        if(rank == 1)
            return c2c ? xt_alloc_result::accepted_but_untestable : xt_alloc_result::unreachable;
        if(rank == 2)
            return c2r ? (rocfft_backend ? xt_alloc_result::rejected
                                         : xt_alloc_result::accepted_but_nullptrs)
                       : xt_alloc_result::accepted;
        return xt_alloc_result::accepted;

    case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        if(params.batch > 1)
            return xt_alloc_result::rejected;
        if(rank == 1)
            return c2c ? xt_alloc_result::accepted_but_untestable : xt_alloc_result::unreachable;
        if(rank == 2)
            return r2c ? (rocfft_backend ? xt_alloc_result::rejected
                                         : xt_alloc_result::accepted_but_nullptrs)
                       : xt_alloc_result::accepted;
        return xt_alloc_result::accepted;

    case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
        if(params.batch == 1 && rank == 1)
            return c2c ? xt_alloc_result::accepted_but_untestable : xt_alloc_result::unreachable;
        return xt_alloc_result::rejected;

    case HIPFFT_FORMAT_UNDEFINED:
        if(params.batch == 1 && rank > 1)
            return rocfft_backend ? xt_alloc_result::rejected
                                  : xt_alloc_result::accepted_but_nullptrs;
        return xt_alloc_result::rejected;

    default:
        throw std::logic_error("Unexpected subformat in xt_alloc_expectation()");
    }
}

// Allocate and validate a multi-device Xt descriptor for the given plan and subformat.
// Returns a usable wrapper if the allocation is fully accepted, or an empty wrapper
// for all other outcomes. Throws ROCFFT_FAIL on unexpected results from hipfftXtMalloc.
static hipfftLibXtDesc_wrapper_t make_xt_desc(const spmd_hipfft_params&     params,
                                              const hipfftHandle_wrapper_t& plan,
                                              const hipfftXtSubFormat&      subformat)
{
    if(!plan)
        throw std::invalid_argument("make_xt_desc: plan must be valid (non-null)");
    validate_or_throw(subformat, "make_xt_desc");

    hipfftLibXtDesc_wrapper_t ret;
    auto                      hipfft_rt   = ret.alloc_with_err(plan, subformat);
    const auto                expectation = xt_alloc_expectation(params, subformat);
    switch(expectation)
    {
    case xt_alloc_result::rejected:
    {
        if(hipfft_rt == HIPFFT_SUCCESS)
        {
            std::ostringstream oss;
            oss << "hipfftXtMalloc unexpectedly succeeded for supposedly invalid descriptor "
                   "format "
                << fft_enum_to_string(subformat)
                << " (test-side revisions may be needed if testing with cuFFT backend)";
            throw ROCFFT_FAIL(oss.str());
        }
        if(verbose)
        {
            std::cout << "hipfftXtMalloc failed as anticipated for descriptor format "
                      << fft_enum_to_string(subformat) << std::endl;
        }
    }
    break;
    case xt_alloc_result::accepted:
        [[fallthrough]];
    case xt_alloc_result::accepted_but_nullptrs:
        [[fallthrough]];
    case xt_alloc_result::accepted_but_untestable:
    {
        if(hipfft_rt != HIPFFT_SUCCESS)
        {
            std::ostringstream oss;
            oss << "hipfftXtMalloc unexpectedly returned " << hipfftResult_string(hipfft_rt)
                << " for supposedly valid descriptor format " << fft_enum_to_string(subformat)
                << " (test-side revisions may be needed if testing with cuFFT backend)";
            throw ROCFFT_FAIL(oss.str());
        }
        if(verbose)
        {
            std::cout << "hipfftXtMalloc succeeded as expected for descriptor format "
                      << fft_enum_to_string(subformat) << std::endl;
        }
        if(subformat != HIPFFT_FORMAT_UNDEFINED || rocfft_backend)
        {
            if(static_cast<hipfftXtSubFormat>((*ret).subFormat) != subformat)
            {
                std::ostringstream oss;
                oss << "descriptor subFormat "
                    << fft_enum_to_string(static_cast<hipfftXtSubFormat>((*ret).subFormat))
                    << " does not match requested format " << fft_enum_to_string(subformat);
                throw ROCFFT_FAIL(oss.str());
            }
        }
        if((*ret).descriptor->nGPUs != static_cast<int>(params.ngpus()))
        {
            std::ostringstream oss;
            oss << "descriptor nGPUs (" << (*ret).descriptor->nGPUs
                << ") does not match requested ngpus (" << params.ngpus() << ")";
            throw ROCFFT_FAIL(oss.str());
        }
        for(size_t dev_idx = 0; dev_idx < params.ngpus(); ++dev_idx)
        {
            if((*ret).descriptor->GPUs[dev_idx] != params.gpus[dev_idx])
            {
                std::ostringstream oss;
                oss << "descriptor device[" << dev_idx << "] (" << (*ret).descriptor->GPUs[dev_idx]
                    << ") does not match requested GPU ID" << params.gpus[dev_idx];
                throw ROCFFT_FAIL(oss.str());
            }
            if(verbose > 2)
                std::cout << "buffer " << dev_idx << " size: " << (*ret).descriptor->size[dev_idx]
                          << " = " << byte_size_to_str((*ret).descriptor->size[dev_idx])
                          << std::endl;
            if((*ret).descriptor->size[dev_idx] > 0)
            {
                if((*ret).descriptor->data[dev_idx] == nullptr)
                {
                    std::ostringstream oss;
                    oss << "gpu buffer pointer is null for device index " << dev_idx
                        << " despite non-zero size " << (*ret).descriptor->size[dev_idx] << " = "
                        << byte_size_to_str((*ret).descriptor->size[dev_idx]);
                    throw ROCFFT_FAIL(oss.str());
                }
            }
            if(expectation == xt_alloc_result::accepted_but_nullptrs)
            {
                if((*ret).descriptor->data[dev_idx] != nullptr)
                {
                    std::ostringstream oss;
                    oss << "gpu buffer pointer is non-null for device index " << dev_idx
                        << " despite test-side expectation of null pointer";
                    throw ROCFFT_FAIL(oss.str());
                }
            }
        }
    }
    break;
    case xt_alloc_result::unreachable:
        throw std::logic_error("make_xt_desc: unreachable expectation");
    default:
        throw std::logic_error("make_xt_desc: unexpected xt_alloc_result value");
    }
    if(expectation != xt_alloc_result::accepted)
        ret.free();
    return ret;
}

// Parameterized test fixture for hipfftXt multi-GPU execution tests.
class hipfftXtGeneralizedUsage : public ::testing::TestWithParam<spmd_hipfft_params>
{
};

// Verify that the I/O data distributed across GPU buffers in `desc` matches the
// corresponding elements in `global_data`, a host buffer representing the full logical
// array (potentially with padding for in-place real transforms).
//
// Strategy: randomly sample valid (batch_idx, multi_idx) coordinates in the global data
// space, map each to
// - the corresponding element in `global_data` (determined via `global_buffer_index`);
// - the relevant element in one of the (*desc).descriptor->data buffers (determined via
//   get_local_buffer_index).
// The host- and device-residing elements are then compared to ensure an exact match.
// Sampling continues until every GPU's chunk has been probed at least `min_probes_per_dev`
// times. A bail-out at 10,000 * min_probes_per_dev * ngpus total iterations guards against
// infinite loops (e.g., if the test-side sampling/partitioning logic has an unidentified
// bug).
//
// Note: probabilistic (not exhaustive) coverage, reproducible under a fixed random seed.
static void verify_data_distribution(const hipfftLibXtDesc_wrapper_t& desc,
                                     const hostbuf&                   global_data,
                                     const spmd_hipfft_params&        params,
                                     const fft_io                     data_io_label,
                                     const size_t                     min_probes_per_dev)
{
    if(min_probes_per_dev == 0)
        throw std::invalid_argument("verify_data_distribution: min_probes_per_dev must be > 0");
    validate_or_throw(data_io_label, "verify_data_distribution");

    union possible_elem_t
    {
        possible_elem_t()
            : cd(0.0, 0.0)
        {
        }
        float                f;
        double               d;
        std::complex<float>  cf;
        std::complex<double> cd;
    };
    auto print = [&](const possible_elem_t& elem) -> std::string {
        std::ostringstream oss;
        if(params.has_real_data_on(data_io_label))
            oss << (params.precision == fft_precision_single ? elem.f : elem.d);
        else
        {
            if(params.precision == fft_precision_single)
                oss << "(" << elem.cf.real() << ", " << elem.cf.imag() << ")";
            else
                oss << "(" << elem.cd.real() << ", " << elem.cd.imag() << ")";
        }
        return oss.str();
    };
    const auto elem_sz = var_size<size_t>(params.precision,
                                          params.has_real_data_on(data_io_label)
                                              ? fft_array_type_real
                                              : fft_array_type_complex_interleaved);
    if(sizeof(possible_elem_t) < elem_sz)
        throw std::logic_error("size of possible_elem_t is smaller than elem_sz");

    const auto desc_subformat = static_cast<hipfftXtSubFormat>((*desc).subFormat);
    const auto placement      = placement_from_subformat(desc_subformat);

    // Verify that the expected data chunks have non-zero sizes and non-null pointers.
    // For unbatched transforms, all ngpus GPUs hold spatial (slab) data; for batched
    // transforms, min(batch, ngpus) GPUs hold data.
    const size_t expected_data_chunks
        = params.batch == 1 ? params.ngpus() : std::min(params.batch, params.ngpus());
    for(size_t i = 0; i < expected_data_chunks; ++i)
    {
        ASSERT_GT((*desc).descriptor->size[i], 0)
            << fft_enum_to_string(data_io_label) << " descriptor has zero size for chunk " << i;
        ASSERT_NE((*desc).descriptor->data[i], nullptr)
            << fft_enum_to_string(data_io_label) << " descriptor has null data pointer for chunk "
            << i << " despite non-zero size " << (*desc).descriptor->size[i];
    }
    // randomly pool multi-indices in the global data space until all device chunks have been
    // explored at least once
    std::uniform_int_distribution<size_t> batch_rng(0, params.batch - 1);
    const auto                            global_logical_span = params.logical_spans(data_io_label);
    std::vector<size_t>                   count_per_chunk(expected_data_chunks, 0);
    // Upper bound on total random probes before bailing out, as a multiple
    // of (min_probes_per_dev * ngpus).
    static constexpr size_t max_probe_multiplier = 10000;
    // Per-case, per-I/O-role PRNG: sampling is reproducible independent of test order.
    auto prng = make_test_prng(params.str(), data_io_label);
    while(std::any_of(count_per_chunk.begin(), count_per_chunk.end(), [&](const auto& count) {
        return count < min_probes_per_dev;
    }))
    {
        // sanity check to avoid infinite loop in case of a bug in the random sampling logic
        if(sum(count_per_chunk.begin(), count_per_chunk.end())
           > max_probe_multiplier * min_probes_per_dev * expected_data_chunks)
        {
            throw std::logic_error(
                "Possible test logic error in verify_data_distribution: some chunk of data was not "
                "explored as often as expected despite "
                + std::to_string(max_probe_multiplier)
                + " times the minimum number of probes per expected data chunk drawn from the "
                  "global data space.");
        }
        const auto          random_global_batch_idx = batch_rng(prng);
        std::vector<size_t> random_global_multi_idx;
        for(size_t dim = 0; dim < global_logical_span.size(); ++dim)
        {
            std::uniform_int_distribution<size_t> dim_rng(0, global_logical_span[dim] - 1);
            random_global_multi_idx.push_back(dim_rng(prng));
        }
        const auto global_buffer_index = params.global_buffer_index(
            random_global_batch_idx, random_global_multi_idx, data_io_label, placement);
        const auto [dev_idx, local_buffer_index] = params.get_local_buffer_index(
            random_global_batch_idx, random_global_multi_idx, data_io_label, desc_subformat);
        rocfft_scoped_device scoped_dev((*desc).descriptor->GPUs[dev_idx]);

        // Copy the single element from the device chunk back to host and compare it against the
        // corresponding element in the global (reference) host data.

        const auto device_byte_offset = local_buffer_index * elem_sz;
        ASSERT_LE(device_byte_offset + elem_sz, (*desc).descriptor->size[dev_idx])
            << "computed local offset lies outside of the device chunk on gpu " << dev_idx;
        possible_elem_t device_elem, host_elem;
        auto            hip_rt = hipMemcpy(&device_elem,
                                static_cast<const char*>((*desc).descriptor->data[dev_idx])
                                    + device_byte_offset,
                                elem_sz,
                                hipMemcpyDeviceToHost);
        if(hip_rt != hipSuccess)
            throw hip_runtime_error("hipMemcpy of element from device chunk failed on device "
                                        + std::to_string(dev_idx) + " (GPU id "
                                        + std::to_string((*desc).descriptor->GPUs[dev_idx]) + ")",
                                    hip_rt);

        const auto global_byte_offset = global_buffer_index * elem_sz;
        ASSERT_LE(global_byte_offset + elem_sz, global_data.size())
            << "computed global offset lies outside of the global data buffer";
        const auto* global_elem = static_cast<const char*>(global_data.data()) + global_byte_offset;
        std::memcpy(&host_elem, global_elem, elem_sz);
        ASSERT_EQ(std::memcmp(&device_elem, &host_elem, elem_sz), 0)
            << fft_enum_to_string(data_io_label) << " data mismatch on device index " << dev_idx
            << " (GPU id " << (*desc).descriptor->GPUs[dev_idx] << ") at local buffer index "
            << local_buffer_index << " expected to match global buffer index "
            << global_buffer_index << " corresponding to global batch index "
            << random_global_batch_idx << " and global multi-index (" <<
            [&] {
                std::ostringstream oss;
                for(size_t i = 0; i < random_global_multi_idx.size(); ++i)
                    oss << (i ? "," : "") << random_global_multi_idx[i];
                return oss.str();
            }()
            << "):\ndevice element value=" << print(device_elem)
            << " whereas host value=" << print(host_elem);

        count_per_chunk[dev_idx]++;
    }
}

static void multi_device_execution(const spmd_hipfft_params&     params,
                                   const hipfftHandle_wrapper_t& plan,
                                   hipfftLibXtDesc_wrapper_t&    input_desc,
                                   hipfftLibXtDesc_wrapper_t&    output_desc)
{
    if(verbose)
    {
        std::cout << "Executing "
                  << (input_desc.get_raw() == output_desc.get_raw() ? "in-place" : "out-of-place")
                  << " " << (is_fwd(params.dft_type) ? "forward" : "inverse")
                  << " transform, with input descriptor of subformat "
                  << fft_enum_to_string(static_cast<hipfftXtSubFormat>((*input_desc).subFormat));
        if(input_desc.get_raw() != output_desc.get_raw())
            std::cout << " and output descriptor of subformat "
                      << fft_enum_to_string(
                             static_cast<hipfftXtSubFormat>((*output_desc).subFormat));
        std::cout << "..." << std::endl;
    }
    auto hipfft_rt
        = hipfftXtExecDescriptor(plan, input_desc, output_desc, params.hipfft_exec_dir());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtExecDescriptor failed with code " << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";
    // In-place execution flips the descriptor's subFormat for unbatched cases.
    ASSERT_EQ(static_cast<hipfftXtSubFormat>((*output_desc).subFormat), params.output_desc_format())
        << "Post-execution subFormat ("
        << fft_enum_to_string(static_cast<hipfftXtSubFormat>((*output_desc).subFormat))
        << ") does not match the expected value ("
        << fft_enum_to_string(params.output_desc_format()) << ")";

    if(verbose)
        std::cout << "Transform execution completed successfully." << std::endl;
}

static void verify_device_side_content(const hipfftLibXtDesc_wrapper_t& desc,
                                       const spmd_hipfft_params&        params,
                                       const hipfftHandle_wrapper_t&    plan,
                                       fft_io                           desc_io,
                                       reference_fft_data_t&            ref_data,
                                       fft_io                           ref_io,
                                       double                           scaling = 1.0)
{
    validate_enums_or_throw("verify_device_side_content", desc_io, ref_io);
    std::vector<hostbuf> tmp_host_buf(1);
    tmp_host_buf[0].alloc(params.global_byte_size(desc_io, params.placement()));
    if(verbose)
        std::cout << "Copying device content to host..." << std::endl;
    auto hipfft_rt = hipfftXtMemcpy(plan, tmp_host_buf[0].data(), desc, HIPFFT_COPY_DEVICE_TO_HOST);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
        << "hipfftXtMemcpy D2H failed with code " << hipfftResult_string(hipfft_rt);
    if(verbose)
    {
        std::cout << "Finished device-to-host hipfftXtMemcpy.\n"
                  << "Verifying data distribution across descriptor's devices..." << std::endl;
    }

    verify_data_distribution(desc, tmp_host_buf[0], params, desc_io, min_probes_per_dev_for_xt);
    if(verbose)
        std::cout << "Verified data distribution across descriptor's devices." << std::endl;
    if(static_cast<hipfftXtSubFormat>((*desc).subFormat) == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
       && (is_complex(params.dft_type) || params.transform_lengths.size() > 2))
    {
        if(xt_alloc_expectation(params, HIPFFT_XT_FORMAT_INPLACE) != xt_alloc_result::accepted)
        {
            throw std::logic_error("verify_device_side_content: INPLACE descriptor is not expected "
                                   "to be allocatable (test-side logic error)");
        }
        if(verbose)
            std::cout << "Verifying D2D copy from INPLACE_SHUFFLED descriptor to INPLACE "
                         "descriptor..."
                      << std::endl;
        auto inplace_desc = make_xt_desc(params, plan, HIPFFT_XT_FORMAT_INPLACE);
        ASSERT_TRUE(inplace_desc) << "hipfftXtMalloc for INPLACE descriptor unexpectedly failed";
        hipfft_rt = hipfftXtMemcpy(plan, inplace_desc, desc, HIPFFT_COPY_DEVICE_TO_DEVICE);
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
            << "hipfftXtMemcpy D2D failed with code " << hipfftResult_string(hipfft_rt);
        verify_data_distribution(
            inplace_desc, tmp_host_buf[0], params, desc_io, min_probes_per_dev_for_xt);
        if(verbose)
            std::cout << "Verified D2D copy from INPLACE_SHUFFLED descriptor to INPLACE descriptor."
                      << std::endl;
    }
    if(verbose)
        std::cout << "Verifying accuracy of results..." << std::endl;

    const auto& ref_buffers = ref_io == fft_io_in ? ref_data.get_buffers<fft_io_in>()
                                                  : ref_data.get_buffers<fft_io_out>();
    const auto  ref_norm    = ref_io == fft_io_in ? ref_data.get_norm<fft_io_in>(params.batch).get()
                                                  : ref_data.get_norm<fft_io_out>(params.batch).get();
    const auto  total_length
        = product(params.transform_lengths.begin(), params.transform_lengths.end());
    const double linf_cutoff = type_epsilon(params.precision) * ref_norm.l_inf * log(total_length);

    const auto diff = distance(
        ref_buffers,
        tmp_host_buf,
        params.logical_spans(desc_io),
        params.batch /* may be smaller than ref_cpu_params' */,
        params.precision,
        ref_io == fft_io_in ? ref_data.get_params().itype : ref_data.get_params().otype,
        ref_io == fft_io_in ? ref_data.get_params().istride : ref_data.get_params().ostride,
        ref_io == fft_io_in ? ref_data.get_params().idist : ref_data.get_params().odist,
        params.array_type(desc_io),
        params.global_strides(desc_io, params.placement()),
        params.global_dist(desc_io, params.placement()),
        nullptr,
        linf_cutoff,
        {0},
        {0},
        scaling);
    if(verbose > 1)
        std::cout << "linf: " << diff.l_inf << " l2: " << diff.l_2 << " cutoff: " << linf_cutoff
                  << std::endl;

    switch(params.precision)
    {
    case fft_precision_single:
        max_linf_eps_single
            = std::max(max_linf_eps_single, diff.l_inf / ref_norm.l_inf / log(total_length));
        max_l2_eps_single
            = std::max(max_l2_eps_single, diff.l_2 / ref_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double
            = std::max(max_linf_eps_double, diff.l_inf / ref_norm.l_inf / log(total_length));
        max_l2_eps_double
            = std::max(max_l2_eps_double, diff.l_2 / ref_norm.l_2 * sqrt(log2(total_length)));
        break;
    default:
        throw std::logic_error("Unexpected precision in hipfftXtGeneralizedUsage test");
    }
    EXPECT_LE(diff.l_inf, linf_cutoff) << "l_inf tolerance failure. cutoff: " << linf_cutoff;
}

// Test that hipfftXt multi-GPU transforms correctly distribute data across GPUs and produce
// numerically accurate results.
//
// This test validates the full lifecycle of a multi-GPU FFT:
//   1. Plan creation (primary + reciprocal). For unsupported configurations, the test
//      returns early.
//   2. Descriptor allocation via hipfftXtMalloc with the parameterized sub-format.
//      For invalid or unimplemented subformats, the test verifies the expected error
//      code and returns.
//   3. Host-to-device data transfer via hipfftXtMemcpy (HIPFFT_COPY_HOST_TO_DEVICE).
//   4. Verification that input data is correctly distributed across GPU buffers.
//   5. Execution of the primary transform via hipfftXtExecDescriptor.
//   6. Verification of output: D2H copy, distribution check, and accuracy comparison
//      against a single-CPU FFTW reference (L-infinity tolerance).
//   7. Execution of the reciprocal transform (round trip).
//   8. Same as step 6 but for the reciprocal plan's output: verify that the
//      original input is recovered (after 1/N scaling).
//
// Steps 3-8 are only reached for fully-supported, multi-dimensional (or batched)
// configurations. They are skipped for unbatched 1D cases and for real transforms
// with mismatched descriptor subformats (HIPFFT_XT_FORMAT_OUTPUT as input format, etc.).
TEST_P(hipfftXtGeneralizedUsage, AllocH2DCopyExecD2HCopyVerifyRoundtrip)
try
{
    const auto& params            = GetParam();
    const auto  reciprocal_params = spmd_hipfft_params::make_reciprocal(params);

    if(expects_successful_plan_creation(params)
       != expects_successful_plan_creation(reciprocal_params))
    {
        throw std::logic_error("Plan creation expected to succeed for one of the primary and "
                               "reciprocal params but not the other. Test-side logic error.");
    }

    // Create FFTW reference for comparison if full support is expected for the test parameters
    std::optional<reference_fft_data_t> reference_results;
    // Reference results are required (for validation of full execution) if
    // - plan creation is expected to succeed;
    // - the I/O descriptor formats are expected to be accepted by hipfftXtMalloc;
    // - we have test infrastructure for the configuration (excluding unbatched 1D transforms);
    // - the usage is seemingly valid (e.g., ruling real transforms with
    //   HIPFFT_XT_FORMAT_OUTPUT/HIPFFT_XT_FORMAT_INPUT as input/output descriptor format).
    const auto unbatched_1d_case = (params.batch == 1 && params.transform_lengths.size() == 1);
    const auto invalid_io_format_real_case
        = (is_real(params.dft_type)
           && (params.input_desc_format == HIPFFT_XT_FORMAT_OUTPUT
               || params.output_desc_format() == HIPFFT_XT_FORMAT_INPUT));
    const auto ref_results_required
        = expects_successful_plan_creation(params)
          && (xt_alloc_expectation(params, params.input_desc_format) == xt_alloc_result::accepted)
          && (params.placement() == fft_placement_inplace
              || (xt_alloc_expectation(params, params.output_desc_format())
                  == xt_alloc_result::accepted))
          && !unbatched_1d_case && !invalid_io_format_real_case;

    if(ref_results_required)
    {
        if(!fftw_compare)
        {
            GTEST_SKIP() << "Test requires FFTW comparison for the given parameters "
                            "(fftw_compare == false in this run)";
        }
        reference_results = make_reference_data(params);
        if(reference_results->needs_computing())
        {
            if(reference_results->needs_input_initialization())
                reference_results->initialize_input(fft_input_generator_host);
            reference_results->launch_async_compute();
        }
    }

    // Create the xt plan:
    if(verbose)
        std::cout << "Creating primary plan..." << std::endl;
    const auto plan = make_plan(params);
    if(!plan)
    {
        // make_plan returns an invalid wrapper when plan creation fails expectedly
        ASSERT_FALSE(expects_successful_plan_creation(params))
            << "make_plan returned invalid handle but plan creation was expected to succeed";
        if(verbose)
            std::cout << "Plan creation failed as expected." << std::endl;
        return;
    }
    ASSERT_TRUE(expects_successful_plan_creation(params)) << "Plan creation unexpectedly succeeded";
    if(verbose)
        std::cout << "Primary plan created." << std::endl;

    hipfftHandle_wrapper_t reciprocal_plan;
    if(ref_results_required)
    {
        // full execution is expected, so create the reciprocal plan for round-trip verification
        reciprocal_plan = is_complex(params.dft_type) ? hipfftHandle_wrapper_t::make_nonowned(plan)
                                                      : make_plan(reciprocal_params);
        ASSERT_TRUE(reciprocal_plan) << "Reciprocal plan creation failed unexpectedly";
    }

    if(verbose)
        std::cout << "Allocating descriptor(s) from primary plan..." << std::endl;

    hipfftLibXtDesc_wrapper_t input_desc, output_desc;
    for(auto io : {fft_io_in, fft_io_out})
    {
        if(io == fft_io_out && params.placement() == fft_placement_inplace)
        {
            output_desc = hipfftLibXtDesc_wrapper_t::make_nonowned(input_desc.get_raw());
            continue;
        }
        auto&      io_desc = io == fft_io_in ? input_desc : output_desc;
        const auto io_desc_format
            = io == fft_io_in ? params.input_desc_format : params.output_desc_format();
        io_desc = make_xt_desc(params, plan, io_desc_format);
    }

    if(verbose)
        std::cout << "Allocation of descriptor(s) tested." << std::endl;

    // clean, usable descriptor(s) is(are) required to proceed with the rest of the test
    if(!input_desc || !output_desc)
    {
        // no need to proceed any further in such cases
        if(verbose)
        {
            std::cout << "Some descriptor(s) was(were) not fully accepted by hipfftXtMalloc for "
                         "format(s) "
                      << fft_enum_to_string(params.input_desc_format);
            if(params.placement() == fft_placement_notinplace)
                std::cout << " and/or " << fft_enum_to_string(params.output_desc_format());
            std::cout << ": Returning now." << std::endl;
        }
        return; // early exit from test for unsupported configuration
    }

    if(unbatched_1d_case)
    {
        GTEST_SKIP() << "Unbatched 1D transforms are not supported by the test infrastructure yet: "
                        "no verification of the execution steps of the test";
    }
    if(invalid_io_format_real_case)
    {
        GTEST_SKIP() << "Invalid test case: real transforms with HIPFFT_XT_FORMAT_OUTPUT (resp. "
                        "HIPFFT_XT_FORMAT_INPUT) set for the input (resp. output) descriptor are "
                        "not supported: such descriptors must be allocated from the reciprocal "
                        "(not primary) plan.";
    }

    // If this point is reached, reference results should have been created for the test parameters
    if(!ref_results_required || !reference_results)
    {
        throw std::logic_error("Test logic error: reference results should have been created for "
                               "this configuration, but they were not");
    }
    if(!reciprocal_plan)
    {
        throw std::logic_error(
            "Test logic error: reciprocal plan should have been created for this "
            "configuration, but it was not");
    }

    if(verbose)
        std::cout << "Starting host-to-device hipfftXtMemcpy (using primary plan)..." << std::endl;

    auto hipfft_rt = hipfftXtMemcpy(plan,
                                    input_desc.get_raw(),
                                    reference_results->get_buffers<fft_io_in>().front().data(),
                                    HIPFFT_COPY_HOST_TO_DEVICE);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMemcpy H2D failed with code " << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";

    if(verbose)
    {
        std::cout << "Finished host-to-device hipfftXtMemcpy (using primary plan)." << std::endl;
        std::cout << "Verifying input data distribution across GPUs..." << std::endl;
    }
    verify_data_distribution(input_desc,
                             reference_results->get_buffers<fft_io_in>().front(),
                             params,
                             fft_io_in,
                             min_probes_per_dev_for_xt);
    if(verbose)
    {
        std::cout << "Verified input data distribution across GPUs.\n"
                  << "Executing primary plan..." << std::endl;
    }
    multi_device_execution(params, plan, input_desc, output_desc);
    if(verbose)
    {
        std::cout << "Primary plan executed.\n"
                  << "Verifying device-side content for output descriptor of primary plan..."
                  << std::endl;
    }
    verify_device_side_content(
        output_desc, params, plan, fft_io_out, *reference_results, fft_io_out);
    if(verbose)
        std::cout << "Device-side content verified for output descriptor of primary plan."
                  << std::endl;

    // Now that the forward-direction execution has been fully verified, exercise a round trip:
    // execute the reciprocal transform swapping input for output and vice-verse: confirm the
    // original input data is recovered (scaled as expected).
    if(verbose)
        std::cout << "Executing reciprocal plan (roundtrip operation)..." << std::endl;
    multi_device_execution(reciprocal_params, reciprocal_plan, output_desc, input_desc);
    if(verbose)
    {
        std::cout << "Reciprocal plan executed.\n"
                  << "Verifying device-side content for output descriptor of reciprocal plan..."
                  << std::endl;
    }
    // output of reciprocal plan must match the original input data (of primary plan),
    // after 1/N scaling
    verify_device_side_content(
        input_desc,
        reciprocal_params,
        reciprocal_plan,
        fft_io_out,
        *reference_results,
        fft_io_in,
        1.0 / product(params.transform_lengths.begin(), params.transform_lengths.end()));
    if(verbose)
        std::cout << "Recovery of input results verified." << std::endl;
}
ROCFFT_CATCH_TEST_EXCEPTIONS

// Generates all possible spmd_hipfft_params structs for hipfftXt generalized-usage tests.
// A filter can be used to exclude parameters found/considered irrelevant for particular
// test instances. Probabilistic sampling is accounted for as well.
static std::vector<spmd_hipfft_params>
    generate_test_params_for_hipfftxt_tests(std::function<bool(const spmd_hipfft_params&)> filter
                                            = nullptr)
{
    std::vector<spmd_hipfft_params> ret;
    // TODO: make these test lengths randomly generated
    const std::vector<std::vector<size_t>> hipfftxt_test_lengths
        = {{32, 36, 38}, {32, 36}, {32 * 1024}};
    // Use a PRNG seeded from the test's random_seed to generate reproducible test parameters
    std::mt19937 rng(random_seed);
    for(const auto& dft_type : trans_type_range_full)
    {
        for(const auto& lengths : hipfftxt_test_lengths)
        {
            for(const auto& precision : {fft_precision_double, fft_precision_single})
            {
                for(size_t ngpus = 2; ngpus <= gpus_per_rank; ++ngpus)
                {
                    // Determine batch values to test for this ngpus:
                    // - MAX_HIP_DESCRIPTOR_GPUS guarantees all devices have work
                    // - 1 exercises the unbatched path
                    // - a random value in [2, ngpus) (when ngpus > 2) exercises the
                    //   case where batch > 1 but fewer batches than GPUs
                    std::vector<size_t> batch_values = {MAX_HIP_DESCRIPTOR_GPUS, 1};
                    if(ngpus > 2)
                    {
                        std::uniform_int_distribution<size_t> batch_dist(2, ngpus - 1);
                        batch_values.push_back(batch_dist(rng));
                    }

                    for(const auto& batch : batch_values)
                    {
                        std::vector<int> gpus(ngpus);
                        std::iota(gpus.begin(), gpus.end(), 0);
                        std::shuffle(gpus.begin(), gpus.end(), rng);
                        for(const auto& [input_fmt, _] : fft_enum_map<hipfftXtSubFormat>::entries)
                        {
                            spmd_hipfft_params to_add(
                                dft_type, gpus, batch, lengths, precision, input_fmt);
                            if(filter && !filter(to_add))
                                continue;
                            const double roll = hash_prob(random_seed, to_add.str());
                            // multi-device uses only interleaved complex data layout for now,
                            // no conditional check for that factor
                            const double run_prob
                                = test_prob * (is_real(to_add.dft_type) ? real_prob_factor : 1.0)
                                  * complex_interleaved_prob_factor;

                            if(roll > run_prob)
                            {
                                if(verbose > 4)
                                {
                                    std::cout << "Test skipped: (roll=" << roll << " > " << run_prob
                                              << ")" << std::endl;
                                }
                                continue;
                            }
                            ret.emplace_back(std::move(to_add));
                        }
                    }
                }
            }
        }
    }
    return ret;
}

// Manual hipfftXt test: when --hipfftxt_test_token is provided, build params from it and run.
static std::vector<spmd_hipfft_params> manual_test_params_for_hipfftxt_test()
{
    if(hipfftxt_test_token.empty())
        return {};
    return {spmd_hipfft_params::make_from_token(hipfftxt_test_token)};
}

INSTANTIATE_TEST_SUITE_P(
    hipfftXtSuite,
    hipfftXtGeneralizedUsage,
    ::testing::ValuesIn(
        generate_test_params_for_hipfftxt_tests([](const spmd_hipfft_params& params) {
            // Exclude cases exercising HIPFFT_XT_FORMAT_OUTPUT for input descriptor or
            // HIPFFT_XT_FORMAT_INPUT for output descriptor: for such usage, the I/O
            // descriptors must be allocated using the reciprocal (not primary) plan,
            // in general. The generalized usage test's roundtrip operations cover this
            // possible usage in a robust way, for any type of plan.
            if(params.input_desc_format == HIPFFT_XT_FORMAT_OUTPUT
               || params.output_desc_format() == HIPFFT_XT_FORMAT_INPUT)
                return false;
            // Exclude unbatched 1D transforms: more analyses are required to determine
            // exact expectations from source-of-truth implementations for such cases
            // (and full test-side infrastructure doesn't exist yet, anyways).
            if(params.batch == 1 && params.transform_lengths.size() == 1)
                return false;
            return true;
        })),
    [](const testing::TestParamInfo<hipfftXtGeneralizedUsage::ParamType>& info) {
        return info.param.str();
    });

INSTANTIATE_TEST_SUITE_P(
    manualHipfftXtTest,
    hipfftXtGeneralizedUsage,
    ::testing::ValuesIn(manual_test_params_for_hipfftxt_test()),
    [](const testing::TestParamInfo<hipfftXtGeneralizedUsage::ParamType>& info) {
        return info.param.str();
    });

// The list of test parameters dynamically generated in the instantiation above may be empty
// if only one device is available and/or if very low test probabilities are used. The following
// ensures such cases do not make gtest report an error due to uninstantiated hipfftXtGeneralizedUsage.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(hipfftXtGeneralizedUsage);
