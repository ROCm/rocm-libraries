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

// hipfftXtMalloc takes (plan, &desc, format) but hip_object_wrapper_t expects TCreate(&obj, ...).
// This adapter reorders the arguments to match.
inline hipfftResult
    hipfftXtMalloc_adapted(hipLibXtDesc** desc, hipfftHandle plan, hipfftXtSubFormat fmt)
{
    return hipfftXtMalloc(plan, desc, fmt);
}
// RAII wrappers used herein
typedef hip_object_wrapper_t<hipfftHandle, hipfftCreate, hipfftDestroy, HIPFFT_SUCCESS>
    hipfftHandle_wrapper_t;
typedef hip_object_wrapper_t<hipLibXtDesc*, hipfftXtMalloc_adapted, hipfftXtFree, HIPFFT_SUCCESS>
    hipfftLibXtDesc_wrapper_t;

template <>
struct is_fft_enum<hipfftXtSubFormat, true> : std::true_type
{
};

inline void validate_or_throw(hipfftXtSubFormat subformat, const std::string& func_name)
{
    switch(subformat)
    {
    case HIPFFT_XT_FORMAT_INPLACE:
        [[fallthrough]];
    case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        [[fallthrough]];
    case HIPFFT_XT_FORMAT_INPUT:
        [[fallthrough]];
    case HIPFFT_XT_FORMAT_OUTPUT:
        [[fallthrough]];
    case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
        return;
    case HIPFFT_FORMAT_UNDEFINED:
        [[fallthrough]];
    default:
        throw std::invalid_argument("invalid/undefined subformat for " + func_name);
    }
}

static std::string format_name(const hipfftXtSubFormat format)
{
    switch(format)
    {
    case HIPFFT_XT_FORMAT_INPUT:
        return "HIPFFT_XT_FORMAT_INPUT";
    case HIPFFT_XT_FORMAT_OUTPUT:
        return "HIPFFT_XT_FORMAT_OUTPUT";
    case HIPFFT_XT_FORMAT_INPLACE:
        return "HIPFFT_XT_FORMAT_INPLACE";
    case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        return "HIPFFT_XT_FORMAT_INPLACE_SHUFFLED";
    case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
        return "HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED";
    case HIPFFT_FORMAT_UNDEFINED:
        return "HIPFFT_FORMAT_UNDEFINED";
    default:
        return "Unknown format";
    }
}

static std::string hipfft_direction_name(const int direction)
{
    switch(direction)
    {
    case HIPFFT_FORWARD:
        return "HIPFFT_FORWARD";
    case HIPFFT_BACKWARD:
        return "HIPFFT_BACKWARD";
    }
}

std::mt19937& get_prng()
{
    static std::mt19937 prng(random_seed);
    return prng;
}

#ifdef __HIP_PLATFORM_AMD__
static constexpr bool rocfft_backend = true;
#else
static constexpr bool rocfft_backend = false;
#endif

// Data holder struct for combining allowable direction / input format combinations.
struct directionformat_t
{
    int               direction;
    hipfftXtSubFormat informat;
};

// Real/complex hipfftxt multi-gpu transforms use HIPFFT_XT_FORMAT_INPLACE for the space format, and
// HIPFFT_XT_FORMAT_INPLACE_SHUFFLED for the frequency format.
static std::vector<directionformat_t> real_directionformat
    = {{HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE},
       {HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED}};

// Complex/complex hipfftxt multi-gpu transforms use either  HIPFFT_XT_FORMAT_INPLACE or
// HIPFFT_XT_FORMAT_INPLACE_SHUFFLED for both space and frequency formats.
// Out-of-place transforms may use HIPFFT_XT_FORMAT_INPUT/HIPFFT_XT_FORMAT_OUTPUT, but we do not
// currently support this functionality.
static std::vector<directionformat_t> complex_directionformat = {
    {HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE},
    {HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED},
    {HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE},
    {HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED},
    //{HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPUT},
    //{HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_OUTPUT},
    //{HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPUT},
    //{HIPFFT_FORWARD, HIPFFT_XT_FORMAT_OUTPUT},
};

// Combine the real/complex and complex/complex direction-format arrays, prepending a bool which is
// true for real/complex transforms.
static auto all_directionformat()
{
    std::vector<std::tuple<bool, directionformat_t>> combined;
    for(const auto& val : real_directionformat)
        combined.push_back(std::make_tuple(true, val));
    for(const auto& val : complex_directionformat)
        combined.push_back(std::make_tuple(false, val));
    return combined;
}

// Parameters are real/complex, direction, format, dimension, and number of GPUs.
class hipfftxtformats : public ::testing::TestWithParam<std::tuple<bool, int, hipfftXtSubFormat>>
{
};

// Test that we support exactly all of the data formats / FFT setups that we have implemented.
TEST_P(hipfftxtformats, supportlistsinglebatch)
{
    const size_t ngpus = rocfft_scoped_device::device_count();
    if(ngpus == 1)
        GTEST_SKIP() << "Need at least 2 gpus for this test";
    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);

    // Worst-case minimum size for cuda-backend is 1024.
    const int Nx = 1024;
    const int Ny = 1024;

    auto hipfft_rt = HIPFFT_SUCCESS;

    const bool realcomplex = std::get<0>(GetParam());
    const auto direction   = std::get<1>(GetParam());
    const auto format      = std::get<2>(GetParam());

    if(verbose > 1)
    {
        std::cout << (realcomplex ? "rc" : "cc") << " " << hipfft_direction_name(direction) << " "
                  << format_name(format) << "\n";
    }

#ifdef __HIP_PLATFORM_NVIDIA__
    if(realcomplex && format == HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED)
        GTEST_SKIP(); // Problematic unsupported case, so skip the test.
#endif

    auto good_rdfs = all_directionformat();
    bool goodcase  = false;
    for(const auto& val : good_rdfs)
    {
        if(realcomplex == std::get<0>(val) && std::get<1>(val).direction == direction
           && std::get<1>(val).informat == format)
        {
            goodcase = true;
            break;
        }
    }

    hipfftHandle plan;
    hipfft_rt = hipfftCreate(&plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    const auto ffttype
        = realcomplex ? (direction == HIPFFT_FORWARD ? HIPFFT_D2Z : HIPFFT_Z2D) : HIPFFT_Z2Z;
    std::vector<size_t> workSize(ngpus);
    if(verbose > 2)
        std::cout << "creating plan..." << std::flush;
    if(format == HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED)
    {
        const int batchsize = 1;
        hipfft_rt           = hipfftMakePlan1d(plan, Nx, ffttype, batchsize, workSize.data());
        if(rocfft_backend || realcomplex)
        {
            ASSERT_NE(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftMakePlan1d should have failed for real/complex multi-gpu";
        }
        else
        {
            ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftMakePlan1d failed with code " << hipfft_rt << " ("
                << hipfftResult_string(hipfft_rt) << ")";
        }
    }
    else
    {
        hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, ffttype, workSize.data());
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2d failed with code " << hipfft_rt
                                             << " (" << hipfftResult_string(hipfft_rt) << ")";
    }
    if(verbose > 2)
        std::cout << " done.\n";

    hipLibXtDesc* indesc = nullptr;
    hipfft_rt            = hipfftXtMalloc(plan, &indesc, format);

    if(goodcase)
    {
        EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed with code " << hipfft_rt
                                             << " (" << hipfftResult_string(hipfft_rt) << ")";
    }
    else
    {
        EXPECT_NE(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc passed but should have failed";
    }

    const bool inplace
        = (format == HIPFFT_XT_FORMAT_INPLACE) || (format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED);
    if(!inplace && format != HIPFFT_FORMAT_UNDEFINED)
    {
        hipfftXtSubFormat outformat;
        switch(format)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            outformat = HIPFFT_XT_FORMAT_OUTPUT;
            break;
        case HIPFFT_XT_FORMAT_OUTPUT:
            outformat = HIPFFT_XT_FORMAT_INPUT;
            break;
        case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
            outformat = HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED;
            break;
        case HIPFFT_FORMAT_UNDEFINED:
        case HIPFFT_XT_FORMAT_INPLACE:
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        default:
            throw std::runtime_error(
                "Test infrastructure error: input format is not actually an out-of-place format");
        }
        hipLibXtDesc* outdesc = nullptr;
        hipfft_rt             = hipfftXtMalloc(plan, &outdesc, outformat);
        if(goodcase)
        {
            EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed with code " << hipfft_rt
                                                 << " (" << hipfftResult_string(hipfft_rt) << ")";
        }
        else
        {
            EXPECT_NE(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc passed but should have failed";
        }

        hipfft_rt = hipfftXtFree(outdesc);
        EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    }

    hipfft_rt = hipfftXtFree(indesc);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftDestroy(plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}

const std::vector<int>               hipfft_directions = {HIPFFT_FORWARD, HIPFFT_BACKWARD};
const std::vector<hipfftXtSubFormat> hipfft_formats    = {HIPFFT_XT_FORMAT_INPUT,
                                                       HIPFFT_XT_FORMAT_OUTPUT,
                                                       HIPFFT_XT_FORMAT_INPLACE,
                                                       HIPFFT_XT_FORMAT_INPLACE_SHUFFLED,
                                                       HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED,
                                                       HIPFFT_FORMAT_UNDEFINED};

INSTANTIATE_TEST_SUITE_P(hipfftxttest,
                         hipfftxtformats,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::ValuesIn(hipfft_directions),
                                            ::testing::ValuesIn(hipfft_formats)),
                         [](const testing::TestParamInfo<hipfftxtformats::ParamType>& info) {
                             const auto  realcomplex = std::get<0>(info.param);
                             const auto  direction   = std::get<1>(info.param);
                             const auto  format      = std::get<2>(info.param);
                             std::string name        = realcomplex ? "rc" : "cc";
                             name += direction == HIPFFT_FORWARD ? "forward" : "backward";
                             name += format_name(format);
                             return name;
                         });

struct hipfftxt_test_param_t
{
    hipfftxt_test_param_t(fft_transform_type  _dft_type,
                          hipfftXtSubFormat   _input_desc_format,
                          size_t              _ngpus,
                          size_t              _batch,
                          std::vector<size_t> _transform_lengths,
                          fft_precision       _precision)
        : dft_type(_dft_type)
        , input_desc_format(_input_desc_format)
        , ngpus(_ngpus)
        , batch(_batch)
        , transform_lengths(_transform_lengths)
        , precision(_precision)
    {
        if(ngpus <= 1)
            throw std::invalid_argument("hipfftxt_test_param_t: requires more than 1 GPU");
        if(batch == 0
           || std::any_of(transform_lengths.begin(), transform_lengths.end(), [](const auto& l) {
                  return l == 0;
              }))
        {
            throw std::invalid_argument(
                "hipfftxt_test_param_t: batch and transform lengths must be non-zero");
        }
        validate_enums_or_throw("hipfftxt_test_param_t", dft_type, input_desc_format, precision);
        // precision can only be single or double for hipfftxt tests, so check that here.
        if(precision != fft_precision_single && precision != fft_precision_double)
            throw std::invalid_argument(
                "hipfftxt_test_param_t: precision must be single or double for hipfftxt tests");
    }
    const fft_transform_type  dft_type;
    const hipfftXtSubFormat   input_desc_format;
    const size_t              ngpus;
    const size_t              batch;
    const std::vector<size_t> transform_lengths;
    const fft_precision       precision;

    inline int exec_dir() const
    {
        return is_fwd(dft_type) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
    }

    inline std::vector<size_t> logical_spans(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_param_t::logical_spans");
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
                "hipfftxt_test_param_t::validate_global_batch_idx: global_batch_idx out of range");
    }

    void validate_global_multi_idx(const std::vector<size_t>& global_multi_idx, fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_param_t::validate_global_multi_idx");
        const auto global_logical_span = logical_spans(io);
        if(global_multi_idx.size() != global_logical_span.size())
            throw std::invalid_argument(
                "hipfftxt_test_param_t::validate_global_multi_idx: global_multi_idx size mismatch");
        for(size_t dim = 0; dim < global_multi_idx.size(); ++dim)
        {
            if(global_multi_idx[dim] >= global_logical_span[dim])
                throw std::invalid_argument("hipfftxt_test_param_t::validate_global_multi_idx: "
                                            "global_multi_idx out of range for dimension "
                                            + std::to_string(dim));
        }
    }

    inline size_t global_buffer_index(size_t                     global_batch_idx,
                                      const std::vector<size_t>& global_multi_idx,
                                      fft_io                     io) const
    {
        validate_or_throw(io, "hipfftxt_test_param_t::global_buffer_index");
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
        validate_or_throw(io, "hipfftxt_test_param_t::get_local_buffer_index");
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
            if(global_multi_idx.size() < 2 || global_multi_idx.size() > 3)
            {
                throw std::runtime_error(
                    "hipfftxt_test_param_t::get_local_buffer_index: test-side support for "
                    "unbatched "
                    "tranforms implemented only for 2D/3D transforms for now.");
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
        validate_or_throw(io, "hipfftxt_test_param_t::has_real_data_on");
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

    inline fft_result_placement placement() const
    {
        return (input_desc_format == HIPFFT_XT_FORMAT_INPLACE
                || input_desc_format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
                   ? fft_placement_inplace
                   : fft_placement_notinplace;
    }

    inline std::vector<size_t> global_strides(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_param_t::global_strides");
        return default_strides(dft_type, placement(), io, transform_lengths);
    }
    inline size_t global_dist(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_param_t::global_dist");
        return default_distance(dft_type, placement(), io, transform_lengths, batch);
    }
    inline size_t global_byte_size(fft_io io) const
    {
        validate_or_throw(io, "hipfftxt_test_param_t::global_byte_size");
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

    inline std::string str() const
    {
        std::ostringstream oss;
        oss << (precision == fft_precision_single ? "single" : "double") << "_"
            << transform_type_name(dft_type) << "_" << format_name(input_desc_format) << "_"
            << "batch_" << batch << "_"
            << "lengths_";
        for(auto len : transform_lengths)
            oss << len << "_";
        oss << "ngpus_" << ngpus;

        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& stream, const hipfftxt_test_param_t& params)
    {
        stream << "precision: " << (params.precision == fft_precision_single ? "single" : "double")
               << ", "
               << "dft type: " << transform_type_name(params.dft_type) << ", "
               << "input format: " << format_name(params.input_desc_format) << ", "
               << "ngpus: " << params.ngpus << ", "
               << "batch: " << params.batch << ", "
               << "transform lengths: (";
        for(auto it = params.transform_lengths.begin(); it != params.transform_lengths.end(); ++it)
            stream << (it != params.transform_lengths.begin() ? ", " : "") << *it;
        stream << ")";
        return stream;
    }

    fft_params make_params_for_reference_cpu() const
    {
        fft_params tmp;
        tmp.length    = transform_lengths;
        tmp.precision = precision;
        // always do it out-of-place for reference CPU (requirement for reference_fft_data_t construction)
        // but use the same data layout as the test case's global strides/distances (in-place or
        // out-of-place) for direct use in hipfftXtMemcpy of input data.
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

    inline hipfftXtSubFormat output_desc_format() const
    {
        switch(input_desc_format)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            return HIPFFT_XT_FORMAT_OUTPUT;
        case HIPFFT_XT_FORMAT_OUTPUT:
            return HIPFFT_XT_FORMAT_INPUT;
        case HIPFFT_XT_FORMAT_INPLACE:
            return HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
            return HIPFFT_XT_FORMAT_INPLACE;
        case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
            throw std::runtime_error(
                "Test-side support for HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED is not implemented yet");
        case HIPFFT_FORMAT_UNDEFINED:
            [[fallthrough]];
        default:
            throw std::runtime_error("Invalid value of input descriptor's format detected in "
                                     "hipfftxt_test_param_t::output_desc_format()");
        }
    }

    // Return true if the input_desc_format is a valid format for the transform of interest.
    // Notes:
    // - "validity" is defined by the existence of an implementation with cuFFT backend.
    // - the cases below have been verified but may not be exhaustive.
    inline bool has_valid_input_format() const
    {
        if(batch == 1)
        {
            // the following in-place configurations are known to be valid
            switch(dft_type)
            {
            case fft_transform_type_real_forward:
                return input_desc_format == HIPFFT_XT_FORMAT_INPLACE;
            case fft_transform_type_real_inverse:
                return input_desc_format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
            case fft_transform_type_complex_forward:
                [[fallthrough]];
            case fft_transform_type_complex_inverse:
                return input_desc_format == HIPFFT_XT_FORMAT_INPLACE
                       || input_desc_format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
            default:
                throw std::logic_error(
                    "Unexpected dft_type in hipfftxt_test_param_t::has_valid_input_format()");
            }
        }
        else
        {
            // Out-of-place cases (i.e., HIPFFT_XT_FORMAT_INPUT to HIPFFT_XT_FORMAT_OUTPUT) with
            // fewer devices than batch size are known to be valid. In-place usage may or may not
            // be valid (not clear yet, to be verified).
            return input_desc_format == HIPFFT_XT_FORMAT_INPUT && ngpus <= batch;
        }
    }

    inline bool expects_implementation() const
    {
        if(!has_valid_input_format())
            return false;
        // multi-batch multi-gpu transforms are not implemented yet for rocFFT backend
        if constexpr(rocfft_backend)
            return batch == 1;
        return true;
    }

private:
    static int get_device_index(size_t global_span, size_t num_devices, size_t global_idx)
    {
        if(global_idx >= global_span)
            throw std::out_of_range(
                "hipfftxt_test_param_t::get_device_index: global_idx out of range");
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
class hipfftxtexec : public ::testing::TestWithParam<hipfftxt_test_param_t>
{
};

static void verify_data_distribution(const hipfftLibXtDesc_wrapper_t& desc,
                                     const hostbuf&                   global_data,
                                     const hipfftxt_test_param_t&     params)
{
    const auto desc_subformat = static_cast<hipfftXtSubFormat>((*desc).subFormat);
    if(desc_subformat != params.input_desc_format && desc_subformat != params.output_desc_format())
    {
        throw std::invalid_argument("verify_data_distribution: descriptor format does not match "
                                    "any of input/output test parameters's format");
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
    std::set<int>                         explored_chunk_ids;
    while(explored_chunk_ids.size() < params.ngpus)
    {
        const auto          random_global_batch_idx = batch_rng(get_prng());
        std::vector<size_t> random_global_multi_idx;
        for(size_t dim = 0; dim < global_logical_span.size(); ++dim)
        {
            std::uniform_int_distribution<size_t> dim_rng(0, global_logical_span[dim] - 1);
            random_global_multi_idx.push_back(dim_rng(get_prng()));
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
            << str(desc_io_label) << " data mismatch on device index " << dev_idx << " (GPU id "
            << (*desc).descriptor->GPUs[dev_idx] << ") at local buffer index " << local_buffer_index
            << " expected to match global buffer index " << global_buffer_index
            << " corresponding to global batch index " << random_global_batch_idx
            << " and global multi-index (" <<
            [&] {
                std::ostringstream oss;
                for(size_t i = 0; i < random_global_multi_idx.size(); ++i)
                    oss << (i ? "," : "") << random_global_multi_idx[i];
                return oss.str();
            }()
            << "):\ndevice element value=" << print(device_elem)
            << " whereas host value=" << print(host_elem);

        explored_chunk_ids.insert(dev_idx);
    }
}

// Test that hipfftXt multi-GPU transforms correctly distribute data across GPUs and produce
// numerically accurate results.
//
// This test validates the full lifecycle of a multi-GPU FFT:
//   1. Plan creation (2D or 3D) with hipfftMakePlan{2d,3d} or hipfftMakePlanMany (batched).
//   2. Descriptor allocation via hipfftXtMalloc with the parameterized sub-format.
//   3. Host-to-device data transfer via hipfftXtMemcpy (HIPFFT_COPY_HOST_TO_DEVICE).
//   4. Verification that input data is correctly distributed across GPU buffers according to the
//      expected partitioning scheme.
//   5. Execution of the transform via hipfftXtExecDescriptor.
//   6. Device-to-host transfer of results via hipfftXtMemcpy (HIPFFT_COPY_DEVICE_TO_HOST).
//   7. Verification that output data distribution across GPUs matches the expected output format.
//   8. Accuracy comparison of the multi-GPU output against a single-CPU FFTW reference, using
//      an L-infinity norm tolerance scaled by machine epsilon, reference norm, and log(N).
//
// For unsupported configurations (e.g., multi-batch on ROCm backend), the test verifies that
// the appropriate error code (HIPFFT_NOT_IMPLEMENTED) is returned and exits early.
TEST_P(hipfftxtexec, data_distribution_and_execution)
{
    try
    {
        const auto& params = GetParam();
        const auto  rank   = params.transform_lengths.size();

        ASSERT_TRUE(rank == 2 || rank == 3) << "only 2D and 3D use cases supported in this test.";

        // Create FFTW reference for comparison
        reference_fft_data_t reference_results{params.make_params_for_reference_cpu()};
        if(params.expects_implementation() && reference_results.needs_computing())
        {
            if(reference_results.needs_input_initialization())
                reference_results.initialize_input(fft_input_generator_host);
            reference_results.launch_async_compute();
        }

        std::vector<int> gpus(params.ngpus);
        std::iota(gpus.begin(), gpus.end(), 0);
        std::shuffle(gpus.begin(), gpus.end(), get_prng());
        std::vector<size_t> workSize(params.ngpus, std::numeric_limits<size_t>::max());

        // Create the xt plan and descriptor:
        hipfftHandle_wrapper_t plan;

        auto hipfft_rt = plan.alloc_with_err();
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

        hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";

        if(verbose > 2)
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
            if constexpr(rocfft_backend)
            {
                ASSERT_EQ(hipfft_rt, HIPFFT_NOT_IMPLEMENTED)
                    << "multi-batch multi-gpu transforms should return not implemented";
                GTEST_SUCCEED();
                return;
            }
            else
            {
                ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
                    << "hipfftMakePlanMany failed with return code " << hipfft_rt << "="
                    << hipfftResult_string(hipfft_rt);
            }
        }
        else
        {
            switch(rank)
            {
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
            ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftMakePlan" << rank << "d failed with return code " << hipfft_rt << "="
                << hipfftResult_string(hipfft_rt);
        }
        ASSERT_TRUE(std::all_of(workSize.begin(), workSize.end(), [](size_t sz) {
            return sz < std::numeric_limits<size_t>::max();
        })) << "some worksize wasn't set at plan creation time";
        if(verbose > 2)
        {
            std::cout << "Plan created.\n";
            std::cout << "Allocating descriptor...\n";
        }

        hipfftLibXtDesc_wrapper_t input_desc, output_desc;
        for(auto io : {fft_io_in, fft_io_out})
        {
            if(io == fft_io_out && params.placement() == fft_placement_inplace)
            {
                output_desc = hipfftLibXtDesc_wrapper_t::make_nonowned(input_desc);
                continue;
            }
            auto&      io_desc = io == fft_io_in ? input_desc : output_desc;
            const auto io_desc_format
                = io == fft_io_in ? params.input_desc_format : params.output_desc_format();
            hipfft_rt = io_desc.alloc_with_err(plan, io_desc_format);
            if(!params.has_valid_input_format())
            {
                // The parameters' I/O descriptor format is/are invalid (validity defined by
                // what cuFFT supports) for the targeted transform, so hipfftXtMalloc is expected
                // to fail. If not, hipfftxt_test_param_t::has_valid_input_format may needs to be
                // revised.
                if(hipfft_rt == HIPFFT_SUCCESS)
                    throw std::logic_error(
                        "hipfftXtMalloc completed successfully for an invalid " + str(io)
                        + "descriptor format (test-side revisions may be needed)");
                continue;
            }
            // we may have unimplemented cases (e.g., multi-batch on ROCm backend), so check for that
            if(!params.expects_implementation())
            {
                if constexpr(rocfft_backend)
                {
                    ASSERT_EQ(hipfft_rt, HIPFFT_NOT_IMPLEMENTED)
                        << str(io)
                        << " hipfftXtMalloc did not return HIPFFT_NOT_IMPLEMENTED for a "
                           "supposedly-unimplemented use case (returned code "
                        << hipfft_rt << " = " << hipfftResult_string(hipfft_rt) << ")";
                }
                else
                {
                    throw std::logic_error(
                        "Test logic error: an implementation must be expected with cufft backend "
                        "for all test parameters with valid I/O descriptor formats.");
                }
                continue;
            }
            // valid case with an expected implementation
            ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
                << " hipfftXtMalloc failed for " << str(io) << " descriptor with code " << hipfft_rt
                << " (" << hipfftResult_string(hipfft_rt) << ")";
            hipfft_rt = io_desc.alloc_with_err(plan, io_desc_format);
            // verifty the content of the created descriptor
            ASSERT_EQ(static_cast<hipfftXtSubFormat>((*io_desc).subFormat), io_desc_format)
                << str(io) << " descriptor subFormat does not match requested format";
            ASSERT_EQ((*io_desc).descriptor->nGPUs, static_cast<int>(params.ngpus))
                << str(io) << " descriptor nGPUs does not match requested ngpus";
            for(size_t dev_idx = 0; dev_idx < gpus.size(); ++dev_idx)
            {
                ASSERT_EQ((*io_desc).descriptor->GPUs[dev_idx], gpus[dev_idx])
                    << str(io) << " descriptor device[" << dev_idx << "] ("
                    << (*io_desc).descriptor->GPUs[dev_idx] << ") does not match requested GPU ID"
                    << gpus[dev_idx];
                if(verbose > 3)
                    std::cout << "buffer " << dev_idx
                              << " size: " << (*io_desc).descriptor->size[dev_idx] << " = "
                              << byte_size_to_str((*io_desc).descriptor->size[dev_idx]) << "\n";
                // TODO: handle case where some GPUs don't have data because there isn't enough to go
                // around (particularly for multi-batch cases).
                ASSERT_NE((*io_desc).descriptor->size[dev_idx], size_t(0))
                    << str(io) << " gpu buffer size is zero for gpu " << dev_idx;
            }
        }
        if(!params.expects_implementation())
        {
            // no need to proceed any further for unimplemented cases (e.g., multi-batch on ROCm backend)
            GTEST_SUCCEED();
            return;
        }

        if(verbose > 2)
        {
            std::cout << "Descriptor allocated.\n";
            std::cout << "Starting host-to-device hipfftXtMemcpy...\n";
        }

        hipfft_rt = hipfftXtMemcpy(plan,
                                   input_desc.get_raw(),
                                   reference_results.get_buffers<fft_io_in>().front().data(),
                                   HIPFFT_COPY_HOST_TO_DEVICE);
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
            << "hipfftXtMemcpy H2D"
            << " failed with code " << hipfft_rt << " (" << hipfftResult_string(hipfft_rt) << ")";

        if(verbose > 2)
        {
            std::cout << "Finished host-to-device hipfftXtMemcpy.\n";
            std::cout << "Verifying input data distribution across GPUs...\n";
        }
        verify_data_distribution(
            input_desc, reference_results.get_buffers<fft_io_in>().front(), params);

        if(verbose > 2)
        {
            std::cout << "Verified input data distribution across GPUs.\n";
            std::cout << "Executing plan...\n";
        }
        // Execute the plan
        hipfft_rt = hipfftXtExecDescriptor(plan, input_desc, output_desc, params.exec_dir());
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
            << "hipfftXtExecDescriptor failed with code " << hipfft_rt << " ("
            << hipfftResult_string(hipfft_rt) << ")";

        if(verbose > 2)
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
        if(verbose > 2)
        {
            std::cout << "Finished device-to-host hipfftXtMemcpy.\n";
            std::cout << "Verifying output data distribution across GPUs...\n";
        }
        verify_data_distribution(output_desc, mgpu_output[0], params);
        if(verbose > 2)
        {
            std::cout << "Verified output data distribution across GPUs.\n";
            std::cout << "Verifying accuracy of results...\n";
        }

        // Compare multi-GPU output against FFTW reference
        const auto total_length
            = product(params.transform_lengths.begin(), params.transform_lengths.end());
        const auto   cpu_output_norm = reference_results.get_norm<fft_io_out>(params.batch).get();
        const double linf_cutoff
            = type_epsilon(params.precision) * cpu_output_norm.l_inf * log(total_length);

        const auto diff = distance(reference_results.get_buffers<fft_io_out>(),
                                   mgpu_output,
                                   params.logical_spans(fft_io_out),
                                   params.batch /* may be smaller than ref_cpu_params' */,
                                   params.precision,
                                   reference_results.get_params().otype,
                                   reference_results.get_params().ostride,
                                   reference_results.get_params().odist,
                                   reference_results.get_params().otype,
                                   params.global_strides(fft_io_out),
                                   params.global_dist(fft_io_out),
                                   nullptr,
                                   linf_cutoff,
                                   {0},
                                   {0});
        if(verbose > 1)
            std::cout << "linf: " << diff.l_inf << " l2: " << diff.l_2 << " cutoff: " << linf_cutoff
                      << "\n";
        EXPECT_LE(diff.l_inf, linf_cutoff) << "l_inf tolerance failure. cutoff: " << linf_cutoff;
        if(verbose > 2)
            std::cout << "Accuracy verified. Test completed.\n";
    }
    ROCFFT_CATCH_TEST_EXCEPTIONS
}

// Note: order test parameters so that caching of reference results is leveraged
static std::vector<hipfftxt_test_param_t> test_params_for_hipfftxt_execution_tests()
{
    std::vector<hipfftxt_test_param_t> params;
    // Test-side support only for 2D/3D transforms, for now
    const std::vector<std::vector<size_t>> test_lengths = {{32, 36}, {32, 36, 38}};
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
                    // some combos of test parameters are not supported/not implemented,
                    // the test actually verifies
                    for(const auto& input_subformat : {HIPFFT_XT_FORMAT_INPLACE,
                                                       HIPFFT_XT_FORMAT_INPLACE_SHUFFLED,
                                                       HIPFFT_XT_FORMAT_INPUT})
                    {
                        for(int ngpus = 2; ngpus <= rocfft_scoped_device::device_count(); ++ngpus)
                        {
                            hipfftxt_test_param_t to_add(
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

INSTANTIATE_TEST_SUITE_P(hipfftxttest,
                         hipfftxtexec,
                         ::testing::ValuesIn(test_params_for_hipfftxt_execution_tests()),
                         [](const testing::TestParamInfo<hipfftxtexec::ParamType>& info) {
                             return info.param.str();
                         });
