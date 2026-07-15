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

#include "hipfft/hipfft.h"
#include "hipfft/hipfftXt.h"

#include <algorithm>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
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
static std::mt19937 make_test_prng(const std::string& token, fft_io io = fft_io::fft_io_in)
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

struct hipfftxt_test_params_t
{
    hipfftxt_test_params_t(fft_transform_type               _dft_type,
                           hipfftXtSubFormat                _input_desc_format,
                           size_t                           _ngpus,
                           size_t                           _batch,
                           std::vector<size_t>              _transform_lengths,
                           fft_precision                    _precision,
                           std::optional<hipfftXtSubFormat> _output_desc_format = std::nullopt)
        : dft_type(_dft_type)
        , input_desc_format(_input_desc_format)
        , ngpus(_ngpus)
        , batch(_batch)
        , transform_lengths(_transform_lengths)
        , precision(_precision)
        , explicit_output_desc_format(_output_desc_format)
    {
        if(ngpus <= 1)
            throw std::invalid_argument("hipfftxt_test_params_t: requires more than 1 GPU");
        if(transform_lengths.empty() || transform_lengths.size() > 3)
            throw std::invalid_argument(
                "hipfftxt_test_params_t: transform_lengths must be non-empty and of rank <= 3");
        if(batch == 0
           || std::any_of(transform_lengths.begin(), transform_lengths.end(), [](const auto& l) {
                  return l == 0;
              }))
        {
            throw std::invalid_argument(
                "hipfftxt_test_params_t: batch and transform lengths must be non-zero");
        }
        validate_enums_or_throw("hipfftxt_test_params_t", dft_type, input_desc_format, precision);
        if(explicit_output_desc_format)
        {
            validate_or_throw(*explicit_output_desc_format, "hipfftxt_test_params_t");
        }
        // precision can only be single or double for hipfftxt tests, so check that here.
        if(precision != fft_precision_single && precision != fft_precision_double)
            throw std::invalid_argument(
                "hipfftxt_test_params_t: precision must be single or double for hipfftxt tests");
    }

    const fft_transform_type  dft_type;
    const hipfftXtSubFormat   input_desc_format;
    const size_t              ngpus;
    const size_t              batch;
    const std::vector<size_t> transform_lengths;
    const fft_precision       precision;

    inline int hipfft_exec_dir() const
    {
        return is_fwd(dft_type) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
    }

    inline std::vector<size_t> logical_spans(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_params_t::logical_spans");
        auto ret = transform_lengths;
        if((dft_type == fft_transform_type_real_forward && io == fft_io_out)
           || (dft_type == fft_transform_type_real_inverse && io == fft_io_in))
            ret.back() = ret.back() / 2 + 1;
        return ret;
    }

    void validate_global_batch_idx(size_t global_batch_idx) const
    {
        if(global_batch_idx >= batch)
            throw std::invalid_argument(
                "hipfftxt_test_params_t::validate_global_batch_idx: global_batch_idx out of range");
    }

    void validate_global_multi_idx(const std::vector<size_t>& global_multi_idx, fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_params_t::validate_global_multi_idx");
        const auto global_logical_span = logical_spans(io);
        if(global_multi_idx.size() != global_logical_span.size())
            throw std::invalid_argument("hipfftxt_test_params_t::validate_global_multi_idx: "
                                        "global_multi_idx size mismatch");
        for(size_t dim = 0; dim < global_multi_idx.size(); ++dim)
        {
            if(global_multi_idx[dim] >= global_logical_span[dim])
                throw std::invalid_argument("hipfftxt_test_params_t::validate_global_multi_idx: "
                                            "global_multi_idx out of range for dimension "
                                            + std::to_string(dim));
        }
    }

    inline size_t global_buffer_index(size_t                     global_batch_idx,
                                      const std::vector<size_t>& global_multi_idx,
                                      fft_io                     io) const
    {
        validate_or_throw(io, "hipfftxt_test_params_t::global_buffer_index");
        validate_global_batch_idx(global_batch_idx);
        validate_global_multi_idx(global_multi_idx, io);
        const auto global_strides = default_strides(dft_type, placement(), io, transform_lengths);
        const auto global_distance
            = default_distance(dft_type, placement(), io, transform_lengths, batch);
        return std::inner_product(global_multi_idx.begin(),
                                  global_multi_idx.end(),
                                  global_strides.begin(),
                                  global_batch_idx * global_distance);
    }

    inline std::pair<int, size_t> get_local_buffer_index(
        size_t global_batch_idx, const std::vector<size_t>& global_multi_idx, fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_params_t::get_local_buffer_index");
        validate_global_batch_idx(global_batch_idx);
        validate_global_multi_idx(global_multi_idx, io);
        std::pair<int, size_t> ret; // (device index, local offset in corresponding device chunk)
        if(batch > 1)
        {
            // For batched transforms, the batch is split across the GPUs
            // local strides/distances match the global strides/distances
            ret.first = get_device_index(batch, ngpus, global_batch_idx);
            const auto lower_global_batch
                = ret.first * (batch / ngpus)
                  + std::min(static_cast<size_t>(ret.first), batch % ngpus);
            const auto global_distance
                = default_distance(dft_type, placement(), io, transform_lengths, batch);
            const auto global_strides
                = default_strides(dft_type, placement(), io, transform_lengths);
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
            const auto   desc_format = io == fft_io_in ? input_desc_format : output_desc_format();
            const size_t split_dim
                = desc_format == HIPFFT_XT_FORMAT_INPLACE || desc_format == HIPFFT_XT_FORMAT_INPUT
                      ? 0
                      : 1;
            const auto global_logical_span = logical_spans(io);
            ret.first                      = get_device_index(
                global_logical_span[split_dim], ngpus, global_multi_idx[split_dim]);
            const auto split_dim_local_span
                = global_logical_span[split_dim] / ngpus
                  + (static_cast<size_t>(ret.first) < global_logical_span[split_dim] % ngpus ? 1
                                                                                             : 0);
            auto local_multi_idx = global_multi_idx;
            local_multi_idx[split_dim] -= (ret.first * (global_logical_span[split_dim] / ngpus)
                                           + std::min(static_cast<size_t>(ret.first),
                                                      global_logical_span[split_dim] % ngpus));
            if(split_dim != global_multi_idx.size() - 1)
            {
                // local strides behaves like default strides, only for the local data chunk's
                // (partial) lengths
                auto partial_lengths       = transform_lengths;
                partial_lengths[split_dim] = split_dim_local_span;
                const auto local_strides
                    = default_strides(dft_type, placement(), io, partial_lengths);
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

    inline bool has_real_data_on(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_params_t::has_real_data_on");
        if(io == fft_io_in)
            return dft_type == fft_transform_type_real_forward;
        // io == fft_io_out
        return dft_type == fft_transform_type_real_inverse;
    }

    inline hipfftType_t hipfft_transform_type() const
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

    inline hipfftType_t reciprocal_hipfft_transform_type() const
    {
        switch(dft_type)
        {
        case fft_transform_type_real_forward:
            return precision == fft_precision_single ? HIPFFT_C2R : HIPFFT_Z2D;
        case fft_transform_type_real_inverse:
            return precision == fft_precision_single ? HIPFFT_R2C : HIPFFT_D2Z;
        case fft_transform_type_complex_forward:
            [[fallthrough]];
        case fft_transform_type_complex_inverse:
            return precision == fft_precision_single ? HIPFFT_C2C : HIPFFT_Z2Z;
        default:
            throw std::logic_error(
                "hipfftxt_test_params_t::reciprocal_hipfft_transform_type: invalid dft_type");
        }
    }

    inline fft_result_placement placement() const
    {
        return (input_desc_format == HIPFFT_XT_FORMAT_INPLACE
                || input_desc_format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
                   ? fft_placement_inplace
                   : fft_placement_notinplace;
    }

    inline std::vector<size_t> global_strides(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_params_t::global_strides");
        return default_strides(dft_type, placement(), io, transform_lengths);
    }
    inline size_t global_dist(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_params_t::global_dist");
        return default_distance(dft_type, placement(), io, transform_lengths, batch);
    }
    inline size_t global_byte_size(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_params_t::global_byte_size");
        std::vector<fft_io> relevant_ios = {io};
        if(placement() == fft_placement_inplace)
            relevant_ios.push_back(other(io));
        size_t ret = 0;
        for(const auto& relevant_io : relevant_ios)
        {
            ret = std::max(ret,
                           compute_ptrdiff(logical_spans(relevant_io),
                                           global_strides(relevant_io),
                                           batch,
                                           global_dist(relevant_io))
                               * var_size<size_t>(precision,
                                                  has_real_data_on(relevant_io)
                                                      ? fft_array_type_real
                                                      : fft_array_type_complex_interleaved));
        }
        return ret;
    }

    // produces a token string that reflects the test parameters' member values, e.g.,
    // single_real_inverse_input_fmt_HIPFFT_XT_FORMAT_INPLACE_output_fmt_HIPFFT_XT_FORMAT_INPLACE_batch_64_lengths_32_36_ngpus_5
    inline std::string str() const
    {
        std::ostringstream oss;
        oss << fft_enum_to_string(precision) << token_sep << fft_enum_to_string(dft_type)
            << token_sep << input_fmt_tag << token_sep << fft_enum_to_string(input_desc_format)
            << token_sep << output_fmt_tag << token_sep << fft_enum_to_string(output_desc_format())
            << token_sep << batch_tag << token_sep << batch << token_sep << lengths_tag
            << token_sep;
        for(auto len : transform_lengths)
            oss << len << token_sep;
        oss << ngpus_tag << token_sep << ngpus;

        return oss.str();
    }

    // Constructs a hipfftxt_test_params_t object from a token string (as produced by str()).
    static hipfftxt_test_params_t make_from_token(std::string_view token)
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
        // the digits to the separator position. Caller must ensure pos is at the start
        // of digit characters (not at a separator).
        auto parse_size = [&](std::string_view context) -> size_t {
            auto next_pos = token.find(token_sep, pos);
            // possibly a trailing value in the token, with no separator after it
            if(next_pos == std::string_view::npos)
                next_pos = token.size();
            if(next_pos <= pos)
                throw std::invalid_argument("make_from_token: empty field for "
                                            + std::string(context));
            size_t ret = std::stoull(std::string(token.substr(pos, next_pos - pos)));
            pos        = next_pos;
            return ret;
        };

        // Parse precision
        const auto precision = fft_enum_from_string<fft_precision>(token_raw, pos);
        expect_sep();
        // Parse dft_type
        const auto dft_type = fft_enum_from_string<fft_transform_type>(token_raw, pos);
        // Parse input_format
        expect_sep(input_fmt_tag);
        const auto input_desc_format = fft_enum_from_string<hipfftXtSubFormat>(token_raw, pos);
        // Optionally parse output_format
        std::optional<hipfftXtSubFormat> explicit_output;
        if(token.substr(pos, token_sep.size()) == token_sep
           && token.substr(pos + token_sep.size(), output_fmt_tag.size()) == output_fmt_tag)
        {
            expect_sep(output_fmt_tag);
            explicit_output = fft_enum_from_string<hipfftXtSubFormat>(token_raw, pos);
        }
        // Parse batch
        expect_sep(batch_tag);
        const size_t batch = parse_size("batch");
        // Parse lengths: parse first value, then continue while separator isn't followed by ngpus
        expect_sep(lengths_tag);
        std::vector<size_t> transform_lengths;
        transform_lengths.push_back(parse_size("transform lengths"));
        while(token.substr(pos, token_sep.size()) == token_sep
              && token.substr(pos + token_sep.size(), ngpus_tag.size()) != ngpus_tag)
        {
            pos += token_sep.size(); // skip separator before next length
            transform_lengths.push_back(parse_size("transform lengths"));
        }
        // Parse ngpus
        expect_sep(ngpus_tag);
        const size_t ngpus = parse_size("ngpus");
        // Construct and return the test parameters object
        return hipfftxt_test_params_t(dft_type,
                                      input_desc_format,
                                      ngpus,
                                      batch,
                                      transform_lengths,
                                      precision,
                                      explicit_output);
    }

    friend std::ostream& operator<<(std::ostream& stream, const hipfftxt_test_params_t& params)
    {
        stream << "precision: " << fft_enum_to_string(params.precision) << ", "
               << "dft type: " << fft_enum_to_string(params.dft_type) << ", "
               << "input subformat: " << fft_enum_to_string(params.input_desc_format) << ", "
               << "output subformat: " << fft_enum_to_string(params.output_desc_format()) << ", "
               << "ngpus: " << params.ngpus << ", "
               << "batch: " << params.batch << ", "
               << "transform lengths: (";
        for(auto it = params.transform_lengths.begin(); it != params.transform_lengths.end(); ++it)
            stream << (it != params.transform_lengths.begin() ? ", " : "") << *it;
        stream << ")";
        return stream;
    }

    fft_params get_global_data_params() const
    {
        fft_params tmp;
        tmp.length    = transform_lengths;
        tmp.precision = precision;
        // Always set it out-of-place so that we can use it as is for reference CPU (requirement
        // for construction of reference_fft_data_t objects) but use the same data layout as the
        // test case's global strides/distances (regardless of test placement) so that the
        // reference results' input data can be funneled into the test's hipfftXtMemcpy as is.
        tmp.placement      = fft_placement_notinplace;
        tmp.transform_type = dft_type;
        tmp.nbatch         = batch;
        tmp.run_callbacks  = fft_callback_type_none;
        tmp.istride        = global_strides(fft_io_in);
        tmp.ostride        = global_strides(fft_io_out);
        tmp.idist          = global_dist(fft_io_in);
        tmp.odist          = global_dist(fft_io_out);
        tmp.validate(); // sets itype, otype, isize, osize, etc. from the above
        return tmp;
    }

    static hipfftXtSubFormat natural_output_desc_format_for(const hipfftXtSubFormat input_format,
                                                            const size_t            batch_sz)
    {
        if(batch_sz == 0)
            throw std::invalid_argument(
                "hipfftxt_test_params_t::natural_output_desc_format_for: batch_sz must be > 0");
        // Possible use case of HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED is unclear yet
        // (to be investigated with cuFFT backend)
        switch(input_format)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            [[fallthrough]];
        case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
            return HIPFFT_XT_FORMAT_OUTPUT;
        case HIPFFT_XT_FORMAT_OUTPUT:
            // somewhat questionable to use HIPFFT_XT_FORMAT_OUTPUT as input, but seems tolerated
            // (and correctly handled) by some cases with cuFFT backend (some real transform use
            // cases were found to crash though)
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
                                     "hipfftxt_test_params_t::natural_output_desc_format_for()");
        }
    }

    inline hipfftXtSubFormat output_desc_format() const
    {
        if(explicit_output_desc_format.has_value())
            return explicit_output_desc_format.value();
        return natural_output_desc_format_for(input_desc_format, batch);
    }

    // Returns what's to be expected when attempting an allocation for a multi-device Xt descriptor
    // of a given subformat with a specific kind of plan (as determined by observations of the behavior
    // from the "source of truth" implementation to match, i.e., cufft).
    // The following table was determined using
    // - 8 V100 devices with CUDA 12.9;
    // - 4 H100 devices with CUDA 13.1.
    // No difference between either of the above was observed. Versions of CUDA following 13.1 may change
    // the overall behavior, so this table may need updates, e.g., with compilation-defined
    // guards/tweaks if multi-version support is desired.
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
    // Legend:
    enum class xt_alloc_expectation_t
    {
        accepted, // "Y"
        accepted_but_nullptrs, // "Y0"
        accepted_but_untestable, // "Y?"
        rejected, // "-"
        unreachable // "/"
    };
    inline xt_alloc_expectation_t xt_alloc_expectation(const hipfftXtSubFormat& subformat) const
    {
        validate_or_throw(subformat, "hipfftxt_test_params_t::xt_alloc_expectation");
        // A plan must be successfully created, first
        if(!expects_successful_plan_creation())
            return xt_alloc_expectation_t::unreachable; // "/"

        const auto rank = transform_lengths.size();
        const bool r2c  = dft_type == fft_transform_type_real_forward;
        const bool c2r  = dft_type == fft_transform_type_real_inverse;
        const bool c2c  = is_complex(dft_type);

        switch(subformat)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            [[fallthrough]];
        case HIPFFT_XT_FORMAT_OUTPUT:
            if(batch > 1)
                return xt_alloc_expectation_t::accepted; // "Y"
            // batch == 1
            if(rank == 1)
                // only C2C is reachable (and untestable) for unbatched 1D
                return c2c ? xt_alloc_expectation_t::accepted_but_untestable // "Y?"
                           : xt_alloc_expectation_t::unreachable; // "/"
            // rank == 2 or 3
            return xt_alloc_expectation_t::accepted_but_nullptrs; // "Y0"

        case HIPFFT_XT_FORMAT_INPLACE:
            if(batch > 1)
                return xt_alloc_expectation_t::accepted; // "Y"
            // batch == 1
            if(rank == 1)
                return c2c ? xt_alloc_expectation_t::accepted_but_untestable // "Y?"
                           : xt_alloc_expectation_t::unreachable; // "/"
            if(rank == 2)
                // C2R's data pointers are null; R2C and C2C are fully accepted
                return c2r ? xt_alloc_expectation_t::accepted_but_nullptrs // "Y0"
                           : xt_alloc_expectation_t::accepted; // "Y"
            // rank == 3
            return xt_alloc_expectation_t::accepted; // "Y"

        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
            if(batch > 1)
                return xt_alloc_expectation_t::rejected; // "-"
            // batch == 1
            if(rank == 1)
                return c2c ? xt_alloc_expectation_t::accepted_but_untestable // "Y?"
                           : xt_alloc_expectation_t::unreachable; // "/"
            if(rank == 2)
                // R2C's data pointers are null; C2R and C2C are fully accepted
                return r2c ? xt_alloc_expectation_t::accepted_but_nullptrs // "Y0"
                           : xt_alloc_expectation_t::accepted; // "Y"
            // rank == 3
            return xt_alloc_expectation_t::accepted; // "Y"

        case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
            // Only reachable for unbatched 1D transforms, where solely C2C is testable
            // (real cases being unreachable); every other configuration is rejected.
            if(batch == 1 && rank == 1)
                return c2c ? xt_alloc_expectation_t::accepted_but_untestable // "Y?"
                           : xt_alloc_expectation_t::unreachable; // "/"
            return xt_alloc_expectation_t::rejected; // "-"

        case HIPFFT_FORMAT_UNDEFINED:
            // Accepted (with null data pointers) only for unbatched multi-dimensional
            // transforms; rejected for unbatched 1D and for every batched transform.
            if(batch == 1 && rank > 1)
                return xt_alloc_expectation_t::accepted_but_nullptrs; // "Y0"
            return xt_alloc_expectation_t::rejected; // "-"

        default:
            throw std::logic_error("Unexpected subformat in "
                                   "hipfftxt_test_params_t::xt_alloc_expectation()");
        }
    }

    inline bool expects_successful_plan_creation() const
    {
        if(batch == 1 && transform_lengths.size() == 1)
        {
            if constexpr(rocfft_backend)
            {
                // no support with rocFFT backend, yet
                return false;
            }
            // cuFFT backend: only C2C power-of-two sizes that divide evenly across the GPUs
            return is_complex(dft_type) && transform_lengths[0] % ngpus == 0
                   && (transform_lengths[0] & (transform_lengths[0] - 1)) == 0;
        }
        // Unbatched multi-dimensional transforms always create successfully. Batched
        // transforms on the rocFFT backend additionally require at least one batch element
        // per GPU (ngpus <= batch) so every device has data; cuFFT imposes no such restriction.
        return !rocfft_backend || batch == 1 || ngpus <= batch;
    }

    bool requires_reference_results() const
    {
        if(!expects_successful_plan_creation())
            return false;
        if(xt_alloc_expectation(input_desc_format) != xt_alloc_expectation_t::accepted)
            return false;
        if(placement() == fft_placement_notinplace)
        {
            if(xt_alloc_expectation(output_desc_format()) != xt_alloc_expectation_t::accepted)
                return false;
            // Not enabling usage of HIPFFT_XT_FORMAT_OUTPUT as input descriptor's subformat at execution
            // for the primary plan (NOTE: execution of reciprocal plan during round-trip test exercises
            // that robustly for all transform types)
            if(input_desc_format == HIPFFT_XT_FORMAT_OUTPUT)
                return false;
            // This test infrastructure was developed, verified, and validated for natural output descriptor
            // format for the given input descriptor format and batch size. Any "non-natural" usage (e.g.,
            // execution from `HIPFFT_XT_FORMAT_INPUT` to `HIPFFT_XT_FORMAT_INPLACE`) has not been
            // investigated yet and the "expected behavior" is still unknown. If coverage for such use cases
            // is ever needed, that can be added to the test infrastructure later on (test generalizations
            // required, very likely)
            if(output_desc_format() != natural_output_desc_format_for(input_desc_format, batch))
                return false;
        }
        // No test infrastructure yet for unbatched 1D transforms (even if they are accepted by the backend).
        if(batch == 1 && transform_lengths.size() == 1)
            return false;
        return true;
    }

private:
    // Token format constants shared by str() and make_from_token() to guarantee
    // consistency.
    static constexpr std::string_view token_sep      = "_";
    static constexpr std::string_view input_fmt_tag  = "input_fmt";
    static constexpr std::string_view output_fmt_tag = "output_fmt";
    static constexpr std::string_view batch_tag      = "batch";
    static constexpr std::string_view lengths_tag    = "lengths";
    static constexpr std::string_view ngpus_tag      = "ngpus";

    const std::optional<hipfftXtSubFormat> explicit_output_desc_format;

    static int get_device_index(size_t global_span, size_t num_devices, size_t global_idx)
    {
        if(global_idx >= global_span)
            throw std::out_of_range(
                "hipfftxt_test_params_t::get_device_index: global_idx out of range");
        const auto min_span_per_dev = global_span / num_devices;
        const auto remainder        = global_span % num_devices;
        const auto split_global_idx = remainder * (min_span_per_dev + 1);
        if(global_idx < split_global_idx)
            return static_cast<int>(global_idx / (min_span_per_dev + 1));
        else
            return static_cast<int>(remainder + (global_idx - split_global_idx) / min_span_per_dev);
    }
};

// Parameters are real/complex, direction, format, dimension, and number of GPUs.
class hipfftXtGeneralizedUsage : public ::testing::TestWithParam<hipfftxt_test_params_t>
{
};

// Verify that the data distributed across GPU buffers in `desc` matches the corresponding
// elements in `global_data`, a host buffer representing the full logical array (potentially
// with padding for in-place real transforms).
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
                                     const hipfftxt_test_params_t&    params,
                                     const size_t                     min_probes_per_dev)
{
    const auto desc_subformat = static_cast<hipfftXtSubFormat>((*desc).subFormat);
    if(min_probes_per_dev == 0)
        throw std::invalid_argument("verify_data_distribution: min_probes_per_dev must be > 0");
    if(desc_subformat != params.input_desc_format && desc_subformat != params.output_desc_format())
    {
        throw std::invalid_argument("verify_data_distribution: descriptor format does not match "
                                    "any of input/output test parameters' format");
    }
    const auto desc_io_label
        = params.input_desc_format == desc_subformat ? fft_io::fft_io_in : fft_io::fft_io_out;

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
        if(params.has_real_data_on(desc_io_label))
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
                                          params.has_real_data_on(desc_io_label)
                                              ? fft_array_type_real
                                              : fft_array_type_complex_interleaved);
    if(sizeof(possible_elem_t) < elem_sz)
        throw std::logic_error("size of possible_elem_t is smaller than elem_sz");
    // randomly pool multi-indices in the global data space until all device chunks have been
    // explored at least once
    std::uniform_int_distribution<size_t> batch_rng(0, params.batch - 1);
    const auto                            global_logical_span = params.logical_spans(desc_io_label);
    std::vector<size_t>                   count_per_chunk(params.ngpus, 0);
    // Per-case, per-I/O-role PRNG: sampling is reproducible independent of test order.
    auto prng = make_test_prng(params.str(), desc_io_label);
    while(std::any_of(count_per_chunk.begin(), count_per_chunk.end(), [&](const auto& count) {
        return count < min_probes_per_dev;
    }))
    {
        // sanity check to avoid infinite loop in case of a bug in the random sampling logic
        if(sum(count_per_chunk.begin(), count_per_chunk.end())
           > 10000 * min_probes_per_dev * params.ngpus)
        {
            throw std::logic_error(
                "Possible test logic error in verify_data_distribution: some chunk of data was not "
                "explored as often as expected despite 10,000 times the minimum number of probes "
                "per device being drawn from the global data space.");
        }
        const auto          random_global_batch_idx = batch_rng(prng);
        std::vector<size_t> random_global_multi_idx;
        for(size_t dim = 0; dim < global_logical_span.size(); ++dim)
        {
            std::uniform_int_distribution<size_t> dim_rng(0, global_logical_span[dim] - 1);
            random_global_multi_idx.push_back(dim_rng(prng));
        }
        const auto global_buffer_index = params.global_buffer_index(
            random_global_batch_idx, random_global_multi_idx, desc_io_label);
        const auto [dev_idx, local_buffer_index] = params.get_local_buffer_index(
            random_global_batch_idx, random_global_multi_idx, desc_io_label);
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
            << fft_enum_to_string(desc_io_label) << " data mismatch on device index " << dev_idx
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

// Execute the reciprocal transform of the just-executed multi-GPU transform and verify that
// the original input is recovered (round trip usage).
//
// `output_desc` holds the result of the execution of the primary plan (its subFormat is
// `params.output_desc_format()`). The reciprocal transform is executed back into the primary
// plan's input descriptor `input_desc`, and the recovered data is compared against the
// original reference input, accounting for the 1/total_length scaling that results from the
// roundtrip operation.
static void run_and_verify_roundtrip(const hipfftxt_test_params_t& params,
                                     const hipfftHandle_wrapper_t& primary_plan,
                                     const std::vector<int>&       gpus,
                                     hipfftLibXtDesc_wrapper_t&    input_desc,
                                     hipfftLibXtDesc_wrapper_t&    output_desc,
                                     reference_fft_data_t&         reference_results)
{
    // Build (or reuse) the plan that performs the reciprocal transform.
    hipfftHandle_wrapper_t reciprocal_plan;
    hipfftResult           hipfft_rt = HIPFFT_SUCCESS;
    if(is_complex(params.dft_type))
    {
        // complex plans are expected to operate in both directions
        reciprocal_plan = hipfftHandle_wrapper_t::make_nonowned(primary_plan.get_raw());
    }
    else
    {
        if(verbose)
            std::cout << "Creating round-trip's reciprocal plan...\n";
        hipfft_rt = reciprocal_plan.alloc_with_err();
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "round-trip hipfftCreate failed";
        // hipfftXtSetGPUs takes a non-const int* but only reads the GPU ids.
        hipfft_rt = hipfftXtSetGPUs(
            reciprocal_plan, static_cast<int>(gpus.size()), const_cast<int*>(gpus.data()));
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "round-trip hipfftXtSetGPUs failed";
        std::vector<size_t> worksize(gpus.size(), std::numeric_limits<size_t>::max());

        if(params.batch > 1)
        {
            std::vector<int> lengths_int(params.transform_lengths.begin(),
                                         params.transform_lengths.end());
            hipfft_rt = hipfftMakePlanMany(reciprocal_plan,
                                           lengths_int.size(),
                                           lengths_int.data(),
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           0,
                                           0,
                                           params.reciprocal_hipfft_transform_type(),
                                           params.batch,
                                           worksize.data());
        }
        else
        {
            switch(params.transform_lengths.size())
            {
            case 2:
                hipfft_rt = hipfftMakePlan2d(reciprocal_plan,
                                             params.transform_lengths[0],
                                             params.transform_lengths[1],
                                             params.reciprocal_hipfft_transform_type(),
                                             worksize.data());
                break;
            case 3:
                hipfft_rt = hipfftMakePlan3d(reciprocal_plan,
                                             params.transform_lengths[0],
                                             params.transform_lengths[1],
                                             params.transform_lengths[2],
                                             params.reciprocal_hipfft_transform_type(),
                                             worksize.data());
                break;
            default:
                FAIL() << "round trip only supports 2D/3D unbatched or batched transforms";
            }
        }
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
            << "round-trip reciprocal plan creation failed with code " << hipfft_rt << " ("
            << hipfftResult_string(hipfft_rt) << ")";
        ASSERT_TRUE(std::all_of(worksize.begin(), worksize.end(), [](const auto& ws) {
            return ws < std::numeric_limits<size_t>::max();
        }));
        if(verbose)
            std::cout << "Round-trip's reciprocal plan created.\n";
    }

    if(verbose)
        std::cout << "Executing reciprocal transform...\n";
    hipfft_rt = hipfftXtExecDescriptor(reciprocal_plan,
                                       output_desc.get_raw(),
                                       input_desc.get_raw(),
                                       is_fwd(params.dft_type) ? HIPFFT_BACKWARD : HIPFFT_FORWARD);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
        << "round-trip hipfftXtExecDescriptor failed with code " << hipfft_rt << " ("
        << hipfftResult_string(hipfft_rt) << ")";
    if(params.placement() == fft_placement_inplace)
    {
        // In-place execution flips the descriptor's subFormat back to the original input format
        // (for unbatched cases; batched cases leave it unchanged, which already matches).
        ASSERT_EQ(static_cast<hipfftXtSubFormat>((*input_desc).subFormat), params.input_desc_format)
            << "in-place round-trip descriptor subFormat after reciprocal execution ("
            << fft_enum_to_string(static_cast<hipfftXtSubFormat>((*input_desc).subFormat))
            << ") does not match the original input format ("
            << fft_enum_to_string(params.input_desc_format) << ")";
    }

    if(verbose)
    {
        std::cout << "Round-trip reciprocal execution completed.\n";
        std::cout << "Copying round-trip results device-to-host...\n";
    }
    std::vector<hostbuf> mgpu_roundtrip(1);
    mgpu_roundtrip[0].alloc(params.global_byte_size(fft_io_in));
    auto d2h_rt = hipfftXtMemcpy(reciprocal_plan,
                                 mgpu_roundtrip[0].data(),
                                 input_desc.get_raw(),
                                 HIPFFT_COPY_DEVICE_TO_HOST);
    ASSERT_EQ(d2h_rt, HIPFFT_SUCCESS) << "round-trip hipfftXtMemcpy D2H failed with code " << d2h_rt
                                      << " (" << hipfftResult_string(d2h_rt) << ")";
    if(verbose)
    {
        std::cout << "Round-trip results copied.\n";
        std::cout << "Verifying recovery of original input data...\n";
    }

    // Compare the recovered input against the original reference input. A transform followed by
    // its reciprocal scales the data by the total transform length, so the round-trip result
    // is scaled by 1/total_length before comparison.
    const auto total_length
        = product(params.transform_lengths.begin(), params.transform_lengths.end());
    const auto   cpu_input_norm = reference_results.get_norm<fft_io_in>(params.batch).get();
    const double rt_linf_cutoff
        = type_epsilon(params.precision) * cpu_input_norm.l_inf * log(total_length);
    const auto rt_diff = distance(reference_results.get_buffers<fft_io_in>(),
                                  mgpu_roundtrip,
                                  params.logical_spans(fft_io_in),
                                  params.batch,
                                  params.precision,
                                  reference_results.get_params().itype,
                                  reference_results.get_params().istride,
                                  reference_results.get_params().idist,
                                  reference_results.get_params().itype,
                                  params.global_strides(fft_io_in),
                                  params.global_dist(fft_io_in),
                                  nullptr,
                                  rt_linf_cutoff,
                                  {0},
                                  {0},
                                  1.0 / total_length);
    if(verbose > 1)
        std::cout << "round-trip linf: " << rt_diff.l_inf << " l2: " << rt_diff.l_2
                  << " cutoff: " << rt_linf_cutoff << "\n";

    switch(params.precision)
    {
    case fft_precision_single:
        max_linf_eps_single = std::max(max_linf_eps_single,
                                       rt_diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_single   = std::max(max_l2_eps_single,
                                     rt_diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double = std::max(max_linf_eps_double,
                                       rt_diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_double   = std::max(max_l2_eps_double,
                                     rt_diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    default:
        throw std::logic_error("Unexpected precision in hipfftXtGeneralizedUsage round trip");
    }

    EXPECT_LE(rt_diff.l_inf, rt_linf_cutoff)
        << "round-trip l_inf tolerance failure. cutoff: " << rt_linf_cutoff;
    if(verbose)
        std::cout << "Recovery of input results verified.\n";
}

// Test that hipfftXt multi-GPU transforms correctly distribute data across GPUs and produce
// numerically accurate results.
//
// This test validates the full lifecycle of a multi-GPU FFT:
//   1. Plan creation (unbatched 1D, 2D, or 3D) with hipfftMakePlan{1d,2d,3d} or
//      hipfftMakePlanMany (batched). For configurations known to be unsupported
//      (e.g., multi-batch on ROCm, real 1D unbatched, non-power-of-2 1D unbatched),
//      the test asserts the expected failure code and returns early.
//   2. Descriptor allocation via hipfftXtMalloc with the parameterized sub-format.
//      For invalid or unimplemented subformats, the test verifies the expected error
//      code and continues/returns as appropriate.
//   3. Host-to-device data transfer via hipfftXtMemcpy (HIPFFT_COPY_HOST_TO_DEVICE).
//   4. Verification that input data is correctly distributed across GPU buffers
//      (probabilistic sampling via verify_data_distribution).
//   5. Execution of the transform via hipfftXtExecDescriptor.
//   6. Device-to-host transfer of results via hipfftXtMemcpy (HIPFFT_COPY_DEVICE_TO_HOST).
//   7. Verification that output data distribution across GPUs matches the expected
//      output format (probabilistic sampling via verify_data_distribution).
//   8. Accuracy comparison of the multi-GPU output against a single-CPU FFTW reference,
//      using an L-infinity norm tolerance scaled by machine epsilon, reference norm,
//      and log(N).
//   9. Round-trip verification (via run_and_verify_roundtrip): the reciprocal transform is
//      executed and the results of that operation are compared against the original
//      reference input (after 1/N scaling).
//
// Steps 3-9 are only reached for fully-supported, multi-dimensional (or batched)
// configurations with natural output descriptor formats. The test skips early for:
//   - non-natural explicit output descriptor formats (test infrastructure not yet extended)
//   - HIPFFT_XT_FORMAT_OUTPUT used as input descriptor format (crashes cuFFT for some cases)
//   - configurations where any device's data chunk is empty (semantics unclear)
TEST_P(hipfftXtGeneralizedUsage, AllocH2DCopyExecD2HCopyVerifyRoundtrip)
try
{
    const auto& params = GetParam();
    const auto  rank   = params.transform_lengths.size();

    // Create FFTW reference for comparison if full support is expected for the test parameters
    std::optional<reference_fft_data_t> reference_results;
    // No Test-side support for unbatched 1D transforms yet, so no need to create reference
    // results for those.
    const auto ref_results_required = params.requires_reference_results();
    if(ref_results_required)
    {
        if(!fftw_compare)
        {
            GTEST_SKIP() << "Test requires FFTW comparison for the given parameters "
                            "(fftw_compare == false in this run)";
        }

        const auto global_data_params = params.get_global_data_params();
        reference_results.emplace(global_data_params);
        // reference_fft_data_t objects are unaware of this test's intention to feed
        // the reference results' input data as is into hipfftXtMemcpy, and the object
        // construction may have decided to re-use cached results that may have slightly
        // different strides in input data (e.g. if testing an in-place real fwd transform
        // right after testing the very same transform out-of-place and vice versa)
        // --> Verify that the reference results' input data's strides and distances
        // match what we expect in global params
        if(reference_results->get_params().istride != global_data_params.istride
           || reference_results->get_params().idist != global_data_params.idist)
        {
            if(verbose)
            {
                std::cout << "Reference results' input data's strides/distances do not match "
                             "expected global params: clearing cache and re-initializing "
                             "reference results\n";
            }
            reference_results.reset();
            reference_fft_data_t::clear_cache();
            reference_results.emplace(global_data_params);
        }
        if(reference_results->needs_computing())
        {
            if(reference_results->needs_input_initialization())
                reference_results->initialize_input(fft_input_generator_host);
            reference_results->launch_async_compute();
        }
    }

    std::vector<int> gpus(params.ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    auto shuffle_prng = make_test_prng(params.str());
    std::shuffle(gpus.begin(), gpus.end(), shuffle_prng);
    std::vector<size_t> workSize(params.ngpus, std::numeric_limits<size_t>::max());

    // Create the xt plan and descriptor:
    hipfftHandle_wrapper_t plan;

    auto hipfft_rt = plan.alloc_with_err();
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";

    if(verbose)
        std::cout << "Creating plan...\n";
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
        switch(rank)
        {
        case 1:
            hipfft_rt = hipfftMakePlan1d(plan,
                                         params.transform_lengths[0],
                                         params.hipfft_transform_type(),
                                         1 /* unbatched case*/,
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
            FAIL() << "Test infrastructure only supports 2D and 3D transforms";
        }
    }
    if(params.expects_successful_plan_creation())
    {
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "Plan creation failed with return code "
                                             << hipfft_rt << "=" << hipfftResult_string(hipfft_rt);
    }
    else
    {
        ASSERT_NE(hipfft_rt, HIPFFT_SUCCESS) << "Plan creation unexpectedly succeeded";
        if(verbose)
            std::cout << "Plan creation failed as expected with return code " << hipfft_rt << "="
                      << hipfftResult_string(hipfft_rt) << "\n";
        return;
    }
    ASSERT_TRUE(std::all_of(workSize.begin(), workSize.end(), [](size_t sz) {
        return sz < std::numeric_limits<size_t>::max();
    })) << "some worksize wasn't set at plan creation time";
    if(verbose)
    {
        std::cout << "Plan created.\n";
        std::cout << "Allocating descriptor...\n";
    }

    hipfftLibXtDesc_wrapper_t input_desc, output_desc;
    bool                      io_descriptors_can_be_used = true;
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
        hipfft_rt              = io_desc.alloc_with_err(plan, io_desc_format);
        const auto expectation = params.xt_alloc_expectation(io_desc_format);
        io_descriptors_can_be_used
            &= (expectation == hipfftxt_test_params_t::xt_alloc_expectation_t::accepted);
        switch(expectation)
        {
        case hipfftxt_test_params_t::xt_alloc_expectation_t::rejected:
        {
            ASSERT_NE(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftXtMalloc unexpectedly succeeded on " << fft_enum_to_string(io)
                << " for supposedly invalid descriptor format "
                << fft_enum_to_string(io_desc_format)
                << " (test-side revisions may be needed if testing with cuFFT backend)";
            if(verbose)
            {
                std::cout << "hipfftXtMalloc failed as anticipated on " << fft_enum_to_string(io)
                          << " for descriptor format " << fft_enum_to_string(io_desc_format)
                          << std::endl;
            }
            continue;
        }
        case hipfftxt_test_params_t::xt_alloc_expectation_t::accepted:
            [[fallthrough]];
        case hipfftxt_test_params_t::xt_alloc_expectation_t::accepted_but_nullptrs:
            [[fallthrough]];
        case hipfftxt_test_params_t::xt_alloc_expectation_t::accepted_but_untestable:
        {
            ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftXtMalloc unexpectedly failed on " << fft_enum_to_string(io)
                << " for supposedly valid descriptor format " << fft_enum_to_string(io_desc_format)
                << " (test-side revisions may be needed if testing with cuFFT backend)";
            if(verbose)
            {
                std::cout << "hipfftXtMalloc succeeded as expected on " << fft_enum_to_string(io)
                          << " for descriptor format " << fft_enum_to_string(io_desc_format)
                          << std::endl;
            }
            // verify the content of the created descriptor
            if(io_desc_format != HIPFFT_FORMAT_UNDEFINED || rocfft_backend)
            {
                // created descriptor's subformat is expected to match the requested format,
                // except for cuFFT backend with HIPFFT_FORMAT_UNDEFINED
                ASSERT_EQ(static_cast<hipfftXtSubFormat>((*io_desc).subFormat), io_desc_format)
                    << fft_enum_to_string(io)
                    << " descriptor subFormat does not match requested format";
            }
            ASSERT_EQ((*io_desc).descriptor->nGPUs, static_cast<int>(params.ngpus))
                << fft_enum_to_string(io) << " descriptor nGPUs does not match requested ngpus";
            for(size_t dev_idx = 0; dev_idx < gpus.size(); ++dev_idx)
            {
                ASSERT_EQ((*io_desc).descriptor->GPUs[dev_idx], gpus[dev_idx])
                    << fft_enum_to_string(io) << " descriptor device[" << dev_idx << "] ("
                    << (*io_desc).descriptor->GPUs[dev_idx] << ") does not match requested GPU ID"
                    << gpus[dev_idx];
                if(verbose > 2)
                    std::cout << "buffer " << dev_idx
                              << " size: " << (*io_desc).descriptor->size[dev_idx] << " = "
                              << byte_size_to_str((*io_desc).descriptor->size[dev_idx]) << "\n";
                if((*io_desc).descriptor->size[dev_idx] > 0)
                {
                    ASSERT_NE((*io_desc).descriptor->data[dev_idx], nullptr)
                        << fft_enum_to_string(io) << " gpu buffer pointer is null for device index "
                        << dev_idx << " despite non-zero size "
                        << (*io_desc).descriptor->size[dev_idx] << " = "
                        << byte_size_to_str((*io_desc).descriptor->size[dev_idx]);
                }
                if(expectation
                   == hipfftxt_test_params_t::xt_alloc_expectation_t::accepted_but_nullptrs)
                {
                    ASSERT_EQ((*io_desc).descriptor->data[dev_idx], nullptr)
                        << fft_enum_to_string(io)
                        << " gpu buffer pointer is non-null for device index " << dev_idx
                        << " despite test-side expectation of null pointer";
                }
            }
        }
        break;
        case hipfftxt_test_params_t::xt_alloc_expectation_t::unreachable:
            throw std::logic_error("Supposedly-unreachable code was reached");
        default:
            throw std::logic_error("Unexpected xt_alloc_expectation_t value in "
                                   "hipfftxt_test_params_t::xt_alloc_expectation()");
        }
    }

    if(verbose)
        std::cout << "Descriptor allocation(s) tested.\n";

    // clean acceptance for all relevant descriptors is required to proceed with the rest of the test
    if(!io_descriptors_can_be_used)
    {
        // no need to proceed any further for such cases
        if(verbose)
        {
            std::cout << "No usable descriptor created by hipfftXtMalloc for format(s) "
                      << fft_enum_to_string(params.input_desc_format);
            if(params.placement() == fft_placement_notinplace)
                std::cout << " and/or " << fft_enum_to_string(params.output_desc_format());
            std::cout << ", as anticipated: plan execution cannot be tested." << std::endl;
        }
        return; // early exit from test for unsupported configuration
    }

    if(params.placement() == fft_placement_notinplace
       && params.output_desc_format()
              != hipfftxt_test_params_t::natural_output_desc_format_for(params.input_desc_format,
                                                                        params.batch))
    {
        GTEST_SKIP() << "Copy and execution steps of test implementation were not verified for "
                        "non-natural output descriptor format: expanding test implementation "
                        "required for testing full execution of this configuration";
    }
    if(params.input_desc_format == HIPFFT_XT_FORMAT_OUTPUT)
    {
        GTEST_SKIP()
            << "test skips copy and execution steps for HIPFFT_XT_FORMAT_OUTPUT set as input "
               "descriptor format for the primary plan: questionable usefulness of such a test, "
               "and/or dramatic failures to expect if the test were to be attempted";
    }
    if(params.batch == 1 && params.transform_lengths.size() == 1)
    {
        GTEST_SKIP() << "Unbatched 1D transforms are not supported by the test infrastructure "
                        "yet: no verification of the execution steps of the test";
    }
    // If this point is reached, reference results should have been created for the test parameters
    if(!ref_results_required || !reference_results)
    {
        throw std::logic_error("Test logic error: reference results should have been created for "
                               "this configuration, but they were not");
    }

    // TODO: handle case where some GPUs don't have data because there isn't enough to go
    // around (particularly for multi-batch cases). For multi-GPU transforms, if some
    // device's data chunk is empty, the expected behavior and/or reliability of this test
    // may need to be revised. Skip such cases for now, if ever attempted somehow
    if(std::any_of((*input_desc).descriptor->data,
                   (*input_desc).descriptor->data + (*input_desc).descriptor->nGPUs,
                   [](auto ptr) { return ptr == nullptr; })
       || std::any_of((*output_desc).descriptor->data,
                      (*output_desc).descriptor->data + (*output_desc).descriptor->nGPUs,
                      [](auto ptr) { return ptr == nullptr; }))
    {
        GTEST_SKIP() << "Some device's data chunk is empty for this multi-GPU transform, "
                        "expected full-execution behavior is unclear and/or test reliability "
                        "may be compromised, skipping test";
    }

    if(verbose)
        std::cout << "Starting host-to-device hipfftXtMemcpy...\n";

    hipfft_rt = hipfftXtMemcpy(plan,
                               input_desc.get_raw(),
                               reference_results->get_buffers<fft_io_in>().front().data(),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
        << "hipfftXtMemcpy H2D"
        << " failed with code " << hipfft_rt << " (" << hipfftResult_string(hipfft_rt) << ")";

    if(verbose)
    {
        std::cout << "Finished host-to-device hipfftXtMemcpy.\n";
        std::cout << "Verifying input data distribution across GPUs...\n";
    }
    verify_data_distribution(input_desc,
                             reference_results->get_buffers<fft_io_in>().front(),
                             params,
                             min_probes_per_dev_for_xt);

    if(verbose)
    {
        std::cout << "Verified input data distribution across GPUs.\n";
        std::cout << "Executing plan...\n";
    }
    // Execute the plan
    hipfft_rt = hipfftXtExecDescriptor(plan, input_desc, output_desc, params.hipfft_exec_dir());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtExecDescriptor failed with code " << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";
    if(params.placement() == fft_placement_inplace)
    {
        ASSERT_EQ(input_desc.get_raw(), output_desc.get_raw())
            << "in-place transform should have same input and output descriptors";
        // check that the descriptor's subformat was updated (resp. not updated) to
        // the expected output subformat after execution for unbatched (resp. batched) cases
        ASSERT_EQ((*input_desc).subFormat, params.output_desc_format())
            << "in-place transform's descriptor subFormat on output ("
            << fft_enum_to_string(static_cast<hipfftXtSubFormat>((*input_desc).subFormat))
            << ") is not as expected after execution ("
            << fft_enum_to_string(params.output_desc_format()) << ")";
    }

    if(verbose)
    {
        std::cout << "Plan executed.\n";
        std::cout << "Starting device-to-host hipfftXtMemcpy...\n";
    }
    std::vector<hostbuf> mgpu_output(1);
    mgpu_output[0].alloc(params.global_byte_size(fft_io_out));

    hipfft_rt = hipfftXtMemcpy(
        plan, mgpu_output[0].data(), output_desc.get_raw(), HIPFFT_COPY_DEVICE_TO_HOST);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
        << "hipfftXtMemcpy D2H"
        << " failed with code " << hipfft_rt << " (" << hipfftResult_string(hipfft_rt) << ")";
    if(verbose)
    {
        std::cout << "Finished device-to-host hipfftXtMemcpy.\n";
        std::cout << "Verifying output data distribution across GPUs...\n";
    }
    verify_data_distribution(output_desc, mgpu_output[0], params, min_probes_per_dev_for_xt);
    if(verbose)
    {
        std::cout << "Verified output data distribution across GPUs.\n";
        std::cout << "Verifying accuracy of results...\n";
    }

    // Compare multi-GPU output against FFTW reference
    const auto total_length
        = product(params.transform_lengths.begin(), params.transform_lengths.end());
    const auto   cpu_output_norm = reference_results->get_norm<fft_io_out>(params.batch).get();
    const double linf_cutoff
        = type_epsilon(params.precision) * cpu_output_norm.l_inf * log(total_length);

    const auto diff = distance(reference_results->get_buffers<fft_io_out>(),
                               mgpu_output,
                               params.logical_spans(fft_io_out),
                               params.batch /* may be smaller than ref_cpu_params' */,
                               params.precision,
                               reference_results->get_params().otype,
                               reference_results->get_params().ostride,
                               reference_results->get_params().odist,
                               reference_results->get_params().otype,
                               params.global_strides(fft_io_out),
                               params.global_dist(fft_io_out),
                               nullptr,
                               linf_cutoff,
                               {0},
                               {0});
    if(verbose > 1)
        std::cout << "linf: " << diff.l_inf << " l2: " << diff.l_2 << " cutoff: " << linf_cutoff
                  << "\n";

    switch(params.precision)
    {
    case fft_precision_single:
        max_linf_eps_single = std::max(
            max_linf_eps_single,
            diff.l_inf / cpu_output_norm.l_inf
                / log(product(params.transform_lengths.begin(), params.transform_lengths.end())));
        max_l2_eps_single = std::max(max_l2_eps_single,
                                     diff.l_2 / cpu_output_norm.l_2
                                         * sqrt(log2(product(params.transform_lengths.begin(),
                                                             params.transform_lengths.end()))));
        break;
    case fft_precision_double:
        max_linf_eps_double = std::max(
            max_linf_eps_double,
            diff.l_inf / cpu_output_norm.l_inf
                / log(product(params.transform_lengths.begin(), params.transform_lengths.end())));
        max_l2_eps_double = std::max(max_l2_eps_double,
                                     diff.l_2 / cpu_output_norm.l_2
                                         * sqrt(log2(product(params.transform_lengths.begin(),
                                                             params.transform_lengths.end()))));
        break;
    default:
        throw std::logic_error("Unexpected precision in hipfftXtGeneralizedUsage test");
    }

    EXPECT_LE(diff.l_inf, linf_cutoff) << "l_inf tolerance failure. cutoff: " << linf_cutoff;
    if(verbose)
        std::cout << "Accuracy verified.\n";

    // Now that the forward-direction execution has been fully verified, exercise a round trip:
    // execute its reciprocal transform and confirm the original input is recovered.
    if(verbose)
        std::cout << "Verifying round-trip usage...\n";
    run_and_verify_roundtrip(params, plan, gpus, input_desc, output_desc, *reference_results);
    if(verbose)
        std::cout << "Test completed.\n";
}
ROCFFT_CATCH_TEST_EXCEPTIONS

// Note: order test parameters so that caching of reference results is leveraged
static std::vector<hipfftxt_test_params_t> test_params_for_hipfftxt_execution_tests()
{
    std::vector<hipfftxt_test_params_t> params;
    // No test-side support for unbatched 1D transforms, for now: this is added for
    // completeness (verification of error code returned by hipFFT with rocfft backend).
    const std::vector<std::vector<size_t>> test_lengths = {{32, 36, 38}, {32, 36}, {32 * 1024}};
    for(const auto& dft_type : trans_type_range_full)
    {
        for(const auto& lengths : test_lengths)
        {
            for(const auto& precision : {fft_precision_double, fft_precision_single})
            {
                // Use MAX_HIP_DESCRIPTOR_GPUS for batch cases to guarantee all
                // devices have some work to do
                for(const auto& batch : {MAX_HIP_DESCRIPTOR_GPUS, 1})
                {
                    // Some test parameters have unacceptable descriptors' subformat and/or unimplemented
                    // support for it. The test consuming these parameters actually verifies that by checking
                    // the various error codes returned by hipFFT and choosing early skips when appropriate.
                    // Note: The possible usage of HIPFFT_XT_FORMAT_OUTPUT as an *input* descriptor's subformat
                    // is exercised in tests via the round-trip verifications (reached if/when the direct
                    // operation is actually supported).
                    for(const auto& input_subformat : {HIPFFT_XT_FORMAT_INPLACE,
                                                       HIPFFT_XT_FORMAT_INPLACE_SHUFFLED,
                                                       HIPFFT_XT_FORMAT_INPUT,
                                                       HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED,
                                                       HIPFFT_FORMAT_UNDEFINED})
                    {
                        for(int ngpus = 2; ngpus <= rocfft_scoped_device::device_count(); ++ngpus)
                        {
                            hipfftxt_test_params_t to_add(
                                dft_type, input_subformat, ngpus, batch, lengths, precision);
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
                                              << ")\n";
                                }
                                continue;
                            }
                            params.emplace_back(std::move(to_add));
                        }
                    }
                }
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    hipfftXtSuite,
    hipfftXtGeneralizedUsage,
    ::testing::ValuesIn(test_params_for_hipfftxt_execution_tests()),
    [](const testing::TestParamInfo<hipfftXtGeneralizedUsage::ParamType>& info) {
        return info.param.str();
    });

// The list of test parameters dynamically generated in the instantiation above may be empty
// if only one device is available and/or if very low test probabilities are used. The following
// ensures such cases do not make gtest report an error due to uninstantiated hipfftXtGeneralizedUsage.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(hipfftXtGeneralizedUsage);

// Manual hipfftXt test: when --hipfftxt_test_token is provided, build params from it and run.
static std::vector<hipfftxt_test_params_t> test_params_for_manual_hipfftxt_test()
{
    if(hipfftxt_test_token.empty())
        return {};
    return {hipfftxt_test_params_t::make_from_token(hipfftxt_test_token)};
}

INSTANTIATE_TEST_SUITE_P(
    manualHipfftXtTest,
    hipfftXtGeneralizedUsage,
    ::testing::ValuesIn(test_params_for_manual_hipfftxt_test()),
    [](const testing::TestParamInfo<hipfftXtGeneralizedUsage::ParamType>& info) {
        return info.param.str();
    });
