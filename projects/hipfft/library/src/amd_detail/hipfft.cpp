// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifdef HIPFFT_MPI_ENABLE
#include "hipfft/hipfftMp.h"
#endif
#include "rocfft/rocfft.h"
#include "rocfft_wrapper.h"
#include <algorithm>
#include <cstring> // std::memset
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "../../../shared/client_data_layout_helpers.h"
#include "../../../shared/gpubuf.h"
#include "../../../shared/ptrdiff.h"
#include "../../../shared/rocfft_enums_vs_fft_enums.h"
#include "../../../shared/rocfft_hip.h"

#ifndef NDEBUG
#include <iostream>
#define HIPFFT_DEBUG_LOG(DEBUG_MSG) std::cerr << "[hipFFT DEBUG LOG]: " << DEBUG_MSG << std::endl;
#else
#define HIPFFT_DEBUG_LOG(DEBUG_MSG)
#endif

// Helper macro to check for rocFFT errors and throw
// a HIPFFT_INTERNAL_ERROR if one occurs
#define ROCFFT_EXPECT_SUCCESS(ROCFFT_CALL)  \
    do                                      \
    {                                       \
        auto status = ROCFFT_CALL;          \
        if(status != rocfft_status_success) \
        {                                   \
            throw HIPFFT_INTERNAL_ERROR;    \
        }                                   \
    } while(0)

#define HIP_EXPECT_SUCCESS(HIP_CALL)     \
    do                                   \
    {                                    \
        auto status = HIP_CALL;          \
        if(status != hipSuccess)         \
        {                                \
            throw HIPFFT_INTERNAL_ERROR; \
        }                                \
    } while(0)

#define HIP_FFT_CHECK_AND_RETURN(ret) \
    {                                 \
        auto code = ret;              \
        if(code != HIPFFT_SUCCESS)    \
        {                             \
            return code;              \
        }                             \
    }

// get number of bytes per element of a given hipDataType
static size_t hipDataType_bytes(hipDataType t)
{
    switch(t)
    {
    case HIP_R_16F:
        // real half
        return 2;
    case HIP_C_16F:
    case HIP_R_32F:
        // complex half and real single
        return 4;
    case HIP_C_32F:
    case HIP_R_64F:
        // complex single and real double
        return 8;
    case HIP_C_64F:
        // complex double
        return 16;
    default:
        throw std::runtime_error("unsupported data type");
    }
}

struct hipfft_brick
{
    hipfft_brick(const std::vector<size_t>& lower,
                 const std::vector<size_t>& upper,
                 const std::vector<size_t>& strides,
                 int                        _device_id)
        : device_id(_device_id)
    {
        if(lower.empty() || lower.size() != upper.size() || lower.size() != strides.size())
        {
            // internal/programming error, not a user error, so throw an
            // internal error
            throw HIPFFT_INTERNAL_ERROR;
        }
        // current implementation assumes sorted (decreasing) strides and
        // unit stride for the fastest-moving dimension (last in row-major order)
        if(!std::is_sorted(
               strides.begin(), strides.end(), [](size_t a, size_t b) { return a >= b; })
           || strides.back() != 1)
        {
            throw HIPFFT_INTERNAL_ERROR;
        }
        axes.reserve(lower.size());
        for(size_t dim = 0; dim < lower.size(); ++dim)
            axes.push_back({lower[dim], upper[dim], strides[dim]});
    }

    const size_t data_byte_size(hipDataType data_type) const
    {
        // Not using compute_ptrdiff herein because real in-place cases
        // require the tailing padding elements
        size_t ret = 0;
        for(size_t dim = 0; dim < axes.size(); ++dim)
            ret = std::max(ret, axes[dim].stride * (axes[dim].upper - axes[dim].lower));
        ret *= hipDataType_bytes(data_type);
        return ret;
    }

    bool logically_contains(const hipfft_brick& other) const
    {
        if(axes.size() != other.axes.size())
            return false;
        return std::equal(axes.begin(),
                          axes.end(),
                          other.axes.begin(),
                          [](const hipfft_brick::axis_t& a, const hipfft_brick::axis_t& b) {
                              return a.lower <= b.lower && a.upper >= b.upper;
                          });
    }

    size_t offset_in(const hipfft_brick& other) const
    {
        if(!other.logically_contains(*this))
            throw HIPFFT_INTERNAL_ERROR;
        size_t offset = 0;
        return std::inner_product(
            axes.begin(),
            axes.end(),
            other.axes.begin(),
            offset,
            std::plus<size_t>(),
            [](const auto& a, const auto& b) { return (a.lower - b.lower) * b.stride; });
    }

    const int get_device_id() const
    {
        return device_id;
    }
    const size_t full_rank() const
    {
        return axes.size();
    }

    const std::vector<size_t> get_lower() const
    {
        std::vector<size_t> lower(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
            lower[dim] = axes[dim].lower;
        return lower;
    }
    const std::vector<size_t> get_upper() const
    {
        std::vector<size_t> upper(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
            upper[dim] = axes[dim].upper;
        return upper;
    }
    const std::vector<size_t> get_strides() const
    {
        std::vector<size_t> strides(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
            strides[dim] = axes[dim].stride;
        return strides;
    }
    const std::vector<size_t> get_spans() const
    {
        std::vector<size_t> spans(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
            spans[dim] = axes[dim].span();
        return spans;
    }

private:
    struct axis_t
    {
        size_t lower;
        size_t upper;
        size_t stride;
        bool   operator==(const axis_t& other) const
        {
            return lower == other.lower && upper == other.upper && stride == other.stride;
        }
        size_t span() const
        {
            return upper - lower;
        }
    };
    std::vector<axis_t> axes;
    int                 device_id;
    hipfft_brick() = default;
    friend struct hipfft_field;
};

struct hipfft_field
{
    hipfft_field(const std::vector<size_t>& transform_batch_and_lengths,
                 fft_transform_type         dft_type,
                 fft_result_placement       placement,
                 fft_io                     io,
                 size_t                     split_dim,
                 const std::vector<int>&    devices)
    {
        validate_enums_or_throw("hipfft_field::hipfft_field(...)", dft_type, placement, io);
        if(transform_batch_and_lengths.size() < 2
           || std::any_of(transform_batch_and_lengths.begin(),
                          transform_batch_and_lengths.end(),
                          [](const auto& l) { return l == 0; }))
        {
            throw std::invalid_argument("Invalid rank of transform or invalid batch/length value");
        }
        if(devices.empty())
            throw std::invalid_argument("devices must be non-empty");
        if(split_dim >= transform_batch_and_lengths.size())
            throw std::out_of_range(
                "split_dim is out of bounds for the given transform_batch_and_lengths");

        const size_t ngpus = devices.size();

        auto global_spans = transform_batch_and_lengths;
        if((dft_type == fft_transform_type_real_forward && io == fft_io_out)
           || (dft_type == fft_transform_type_real_inverse && io == fft_io_in))
        {
            global_spans.back() = (global_spans.back() / 2) + 1;
        }
        const auto global_inbuffer_strides
            = default_strides(dft_type, placement, io, transform_batch_and_lengths);

        global_field = hipfft_brick(std::vector<size_t>(global_spans.size(), 0),
                                    global_spans,
                                    global_inbuffer_strides,
                                    rocfft_scoped_device::current_device());

        for(size_t device_idx = 0; device_idx < devices.size(); ++device_idx)
        {
            std::vector<size_t> brick_lower(global_spans.size(), 0);
            std::vector<size_t> brick_upper(global_spans);
            brick_lower[split_dim] = device_idx * (global_spans[split_dim] / ngpus)
                                     + std::min(device_idx, global_spans[split_dim] % ngpus);
            brick_upper[split_dim] = (device_idx + 1) * (global_spans[split_dim] / ngpus)
                                     + std::min((device_idx + 1), global_spans[split_dim] % ngpus);
            std::vector<size_t> brick_strides(global_spans.size());
            for(size_t dim = brick_strides.size(); dim-- > 0;)
            {
                if(dim == brick_strides.size() - 1)
                    brick_strides[dim] = 1;
                else if(dim == brick_strides.size() - 2 && split_dim != global_spans.size() - 1
                        && placement == fft_placement_inplace
                        && ((dft_type == fft_transform_type_real_forward && io == fft_io_in)
                            || (dft_type == fft_transform_type_real_inverse && io == fft_io_out)))
                {
                    brick_strides[dim] = 2 * (global_spans.back() / 2 + 1);
                }
                else
                    brick_strides[dim]
                        = brick_strides[dim + 1] * (brick_upper[dim + 1] - brick_lower[dim + 1]);
            }
            bricks.emplace_back(std::move(brick_lower),
                                std::move(brick_upper),
                                std::move(brick_strides),
                                devices[device_idx]);
        }
    }

    static void add_field_to_descriptor(const hipfft_field&                field,
                                        rocfft_plan_description_wrapper_t& desc,
                                        fft_io                             field_label)
    {
        rocfft_field_wrapper_t field_wrapper;
        ROCFFT_EXPECT_SUCCESS(field_wrapper.alloc_with_err());
        for(const auto& brick : field.bricks)
        {
            rocfft_brick_wrapper_t brick_wrapper;

            auto brick_lower  = brick.get_lower();
            auto brick_upper  = brick.get_upper();
            auto brick_stride = brick.get_strides();
            // row-major order -> column-major order for rocFFT
            std::reverse(brick_lower.begin(), brick_lower.end());
            std::reverse(brick_upper.begin(), brick_upper.end());
            std::reverse(brick_stride.begin(), brick_stride.end());
            ROCFFT_EXPECT_SUCCESS(brick_wrapper.alloc_with_err(brick_lower.data(),
                                                               brick_upper.data(),
                                                               brick_stride.data(),
                                                               brick_lower.size(),
                                                               brick.get_device_id()));
            ROCFFT_EXPECT_SUCCESS(rocfft_field_add_brick(field_wrapper, brick_wrapper));
        }
        if(field_label == fft_io_in)
            ROCFFT_EXPECT_SUCCESS(rocfft_plan_description_add_infield(desc, field_wrapper));
        else
            ROCFFT_EXPECT_SUCCESS(rocfft_plan_description_add_outfield(desc, field_wrapper));
    }

    inline size_t brick_count() const
    {
        return bricks.size();
    }

    const hipfft_brick& get_brick(size_t brick_idx) const
    {
        if(brick_idx >= bricks.size())
            throw std::out_of_range("hipfft_field::brick: index out of range");
        return bricks[brick_idx];
    }

    std::pair<hipfft_brick, hipfft_brick>
        get_collapsed_brick_in_collapsed_field(size_t brick_idx) const
    {
        const auto&                           brick = get_brick(brick_idx);
        std::pair<hipfft_brick, hipfft_brick> ret{hipfft_brick{}, hipfft_brick{}};
        for(size_t global_dim = 0; global_dim < global_field.axes.size(); global_dim++)
        {
            // unit global spans are ignored
            if(global_field.axes[global_dim].span() == 1)
                continue;
            auto collapsed_brick_axis = brick.axes[global_dim];
            auto collapsed_field_axis = global_field.axes[global_dim];
            while(global_dim < global_field.axes.size() - 1
                  && brick.axes[global_dim + 1].lower == global_field.axes[global_dim + 1].lower
                  && brick.axes[global_dim + 1].upper == global_field.axes[global_dim + 1].upper
                  && brick.axes[global_dim].stride
                         == brick.axes[global_dim + 1].stride * brick.axes[global_dim + 1].span()
                  && global_field.axes[global_dim].stride
                         == global_field.axes[global_dim + 1].stride
                                * global_field.axes[global_dim + 1].span())
            {
                collapsed_brick_axis.stride = brick.axes[global_dim + 1].stride;
                collapsed_brick_axis.lower *= brick.axes[global_dim + 1].span();
                collapsed_brick_axis.upper *= brick.axes[global_dim + 1].span();
                collapsed_field_axis.stride = global_field.axes[global_dim + 1].stride;
                collapsed_field_axis.lower *= global_field.axes[global_dim + 1].span();
                collapsed_field_axis.upper *= global_field.axes[global_dim + 1].span();
                global_dim++;
            }
            ret.first.axes.push_back(collapsed_brick_axis);
            ret.second.axes.push_back(collapsed_field_axis);
        }
        ret.first.device_id  = brick.device_id;
        ret.second.device_id = global_field.device_id;
        return ret;
    }

private:
    std::vector<hipfft_brick> bricks;
    // for the xtMemcpy interface, we need to know the global field's
    // upper bounds and strides, so we can compute the offsets for each brick.
    hipfft_brick global_field;
};

struct hipfftIOType
{
private:
    hipDataType inputType  = HIP_C_32F;
    hipDataType outputType = HIP_C_32F;

    bool isinitialized = false;

public:
    auto get_inputType() const
    {
        return inputType;
    }
    auto get_outputType() const
    {
        return outputType;
    }

    hipfftIOType() = default;

    // initialize from data types specified by hipfftType enum
    hipfftResult_t init(hipfftType type)
    {
        switch(type)
        {
        case HIPFFT_R2C:
            inputType  = HIP_R_32F;
            outputType = HIP_C_32F;
            break;
        case HIPFFT_C2R:
            inputType  = HIP_C_32F;
            outputType = HIP_R_32F;
            break;
        case HIPFFT_C2C:
            inputType  = HIP_C_32F;
            outputType = HIP_C_32F;
            break;
        case HIPFFT_D2Z:
            inputType  = HIP_R_64F;
            outputType = HIP_C_64F;
            break;
        case HIPFFT_Z2D:
            inputType  = HIP_C_64F;
            outputType = HIP_R_64F;
            break;
        case HIPFFT_Z2Z:
            inputType  = HIP_C_64F;
            outputType = HIP_C_64F;
            break;
        default:
            return HIPFFT_NOT_IMPLEMENTED;
        }
        isinitialized = true;
        return HIPFFT_SUCCESS;
    }

    // initialize from separate input, output, exec types
    hipfftResult_t init(hipDataType input, hipDataType output, hipDataType exec)
    {
        // real input must have complex output + exec of same precision
        //
        // complex input could have complex or real output of same precision.
        // exec type must be complex, same precision
        switch(input)
        {
        case HIP_R_16F:
            if(output != HIP_C_16F || exec != HIP_C_16F)
                return HIPFFT_INVALID_VALUE;
            break;
        case HIP_R_32F:
            if(output != HIP_C_32F || exec != HIP_C_32F)
                return HIPFFT_INVALID_VALUE;
            break;
        case HIP_R_64F:
            if(output != HIP_C_64F || exec != HIP_C_64F)
                return HIPFFT_INVALID_VALUE;
            break;
        case HIP_C_16F:
            if((output != HIP_C_16F && output != HIP_R_16F) || exec != HIP_C_16F)
                return HIPFFT_INVALID_VALUE;
            break;
        case HIP_C_32F:
            if((output != HIP_C_32F && output != HIP_R_32F) || exec != HIP_C_32F)
                return HIPFFT_INVALID_VALUE;
            break;
        case HIP_C_64F:
            if((output != HIP_C_64F && output != HIP_R_64F) || exec != HIP_C_64F)
                return HIPFFT_INVALID_VALUE;
            break;
        default:
            return HIPFFT_NOT_IMPLEMENTED;
        }

        inputType     = input;
        outputType    = output;
        isinitialized = true;
        return HIPFFT_SUCCESS;
    }

    rocfft_precision precision() const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        switch(inputType)
        {
        case HIP_R_16F:
        case HIP_C_16F:
            return rocfft_precision_half;
        case HIP_C_32F:
        case HIP_R_32F:
            return rocfft_precision_single;
        case HIP_R_64F:
        case HIP_C_64F:
            return rocfft_precision_double;
        default:
            throw std::runtime_error("Required precision is invalid!");
        }
    }

    bool is_real_to_complex() const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        switch(inputType)
        {
        case HIP_R_16F:
        case HIP_R_32F:
        case HIP_R_64F:
            return true;
        case HIP_C_16F:
        case HIP_C_32F:
        case HIP_C_64F:
            return false;
        default:
            throw HIPFFT_NOT_IMPLEMENTED;
        }
    }

    bool is_complex_to_real() const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        switch(outputType)
        {
        case HIP_R_16F:
        case HIP_R_32F:
        case HIP_R_64F:
            return true;
        case HIP_C_16F:
        case HIP_C_32F:
        case HIP_C_64F:
            return false;
        default:
            throw HIPFFT_NOT_IMPLEMENTED;
        }
    }

    bool is_complex_to_complex() const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        return !is_complex_to_real() && !is_real_to_complex();
    }

    std::vector<rocfft_transform_type> transform_types() const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        std::vector<rocfft_transform_type> ret;
        if(is_real_to_complex())
            ret.push_back(rocfft_transform_type_real_forward);
        else if(is_complex_to_real())
            ret.push_back(rocfft_transform_type_real_inverse);
        // else, C2C which can be either direction
        else
        {
            ret.push_back(rocfft_transform_type_complex_forward);
            ret.push_back(rocfft_transform_type_complex_inverse);
        }
        return ret;
    }

    rocfft_array_type array_type(fft_io io) const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        validate_or_throw(io, "hipfftIOType::array_type");
        if(is_real_to_complex())
        {
            return io == fft_io::fft_io_in ? rocfft_array_type_real
                                           : rocfft_array_type_hermitian_interleaved;
        }
        else if(is_complex_to_real())
        {
            return io == fft_io::fft_io_in ? rocfft_array_type_hermitian_interleaved
                                           : rocfft_array_type_real;
        }
        else
        {
            return rocfft_array_type_complex_interleaved;
        }
    }

    // Get the data type based on a descriptor's sub-format value.
    auto data_type_for(const hipfftXtSubFormat subFormat)
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        switch(subFormat)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            [[fallthrough]];
        case HIPFFT_XT_FORMAT_INPLACE:
            return get_inputType();
        case HIPFFT_XT_FORMAT_OUTPUT:
            [[fallthrough]];
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
            return get_outputType();
        case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
            throw HIPFFT_NOT_IMPLEMENTED;
        default:
            throw HIPFFT_INVALID_VALUE;
        }
    }
};

struct hipfftHandle_t
{
    // Return true if the plans have been initialized - hipfftCreate
    // merely allocates a handle and a hipfftMakePlan* API initializes
    // them.
    bool initialized() const
    {
        return !exec_plans.empty();
    }

    hipfftIOType              io_type;
    std::vector<size_t>       transform_lengths;
    size_t                    batch;
    hipfft_ionembed_t<size_t> global_ionembed;
    double                    scale_factor  = 1.0;
    bool                      auto_allocate = true;

    // Plans (and their possible I/O fields) are keyed by transform type, input
    // descriptor's subformat, and output descriptor's subformat.
    // For single-device usage, the key's descriptors' subformat values are
    // unrelated to actual user-provided arguments (no descriptor is passed/expected),
    // but deduced at execution time as follows
    // - (..., HIPFFT_XT_FORMAT_INPLACE, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED) for in-place transforms;
    // - (..., HIPFFT_XT_FORMAT_INPUT, HIPFFT_XT_FORMAT_OUTPUT) for out-of-place transforms.
    // in order to query the map of execution plans.
    //
    // NOTE: for in-place multi-device transforms, the input and output descriptors are
    // (expected to be) identical at execution, but the descriptor's subformat value is
    // updated upon completion of the transform (from HIPFFT_XT_FORMAT_INPLACE to
    // HIPFFT_XT_FORMAT_INPLACE_SHUFFLED and vice versa).
    struct map_key_t
    {
        rocfft_transform_type transform_type;
        hipfftXtSubFormat     input_desc_format;
        hipfftXtSubFormat     output_desc_format;
        bool                  operator<(const map_key_t& other) const
        {
            return std::tie(transform_type, input_desc_format, output_desc_format) < std::tie(
                       other.transform_type, other.input_desc_format, other.output_desc_format);
        }
    };
    struct plan_and_fields_t
    {
        rocfft_plan_wrapper_t       rocfft_plan;
        std::optional<hipfft_field> input_field, output_field;
    };

    std::map<map_key_t, plan_and_fields_t> exec_plans;
    // the same execution info is used for all rocfft plans in `exec_plans`
    rocfft_execution_info_wrapper_t info;
    // in the order given by the user, for multi-device transforms
    std::vector<int> device_ids;
    struct hipfft_exec_info_params_t
    {
        static hipfft_exec_info_params_t create_for_current_device()
        {
            hipfft_exec_info_params_t ret;
            ret.work_buffer_byte_bsize = 0;
            HIP_EXPECT_SUCCESS(ret.stream.alloc_with_err());
            return ret;
        }

        size_t              work_buffer_byte_bsize;
        gpubuf              work_buffer; // may be owned or not
        hipStream_wrapper_t stream; // may be owned or not
    private:
        // forbid default construction possibly triggered by map's
        // operator[] (compile-time error if used)
        hipfft_exec_info_params_t() = default;
    };
    std::map<int, hipfft_exec_info_params_t> exec_data;

    void** load_callback_ptrs       = nullptr;
    void** load_callback_data       = nullptr;
    size_t load_callback_lds_bytes  = 0;
    void** store_callback_ptrs      = nullptr;
    void** store_callback_data      = nullptr;
    size_t store_callback_lds_bytes = 0;

    // Multi-processing communicator
    rocfft_comm_type comm_type   = rocfft_comm_none;
    void*            comm_handle = nullptr;

    rocfft_transform_type get_transform_type(int direction) const
    {
        if(!initialized())
            throw HIPFFT_INVALID_PLAN;
        if(direction != HIPFFT_FORWARD && direction != HIPFFT_BACKWARD)
            throw HIPFFT_INVALID_VALUE; // assume that comes from the user
        if(io_type.is_complex_to_complex())
            return direction == HIPFFT_FORWARD ? rocfft_transform_type_complex_forward
                                               : rocfft_transform_type_complex_inverse;
        if((io_type.is_real_to_complex() && direction == HIPFFT_BACKWARD)
           || (io_type.is_complex_to_real() && direction == HIPFFT_FORWARD))
            throw HIPFFT_INVALID_PLAN;
        return direction == HIPFFT_FORWARD ? rocfft_transform_type_real_forward
                                           : rocfft_transform_type_real_inverse;
    }

    const rocfft_plan_wrapper_t&
        get_single_device_rocfft_plan(int direction, const void* in, const void* out) const
    {
        hipfftHandle_t::map_key_t key{get_transform_type(direction),
                                      in == out ? HIPFFT_XT_FORMAT_INPLACE : HIPFFT_XT_FORMAT_INPUT,
                                      in == out ? HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
                                                : HIPFFT_XT_FORMAT_OUTPUT};

        const auto it = exec_plans.find(key);
        if(it == exec_plans.end())
            throw HIPFFT_INVALID_PLAN;
        return it->second.rocfft_plan;
    }

    const rocfft_plan_wrapper_t& get_rocfft_plan(int                 direction,
                                                 const hipLibXtDesc* in_desc,
                                                 const hipLibXtDesc* out_desc) const
    {
        const auto        key_insubFormat  = static_cast<hipfftXtSubFormat>(in_desc->subFormat);
        hipfftXtSubFormat key_outsubFormat = static_cast<hipfftXtSubFormat>(out_desc->subFormat);
        if(in_desc == out_desc)
        {
            switch(key_insubFormat)
            {
            case HIPFFT_XT_FORMAT_INPLACE:
                key_outsubFormat = HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
                break;
            case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
                key_outsubFormat = HIPFFT_XT_FORMAT_INPLACE;
                break;
            default:
                throw HIPFFT_INVALID_VALUE;
            }
        }
        hipfftHandle_t::map_key_t key{
            get_transform_type(direction), key_insubFormat, key_outsubFormat};

        const auto it = exec_plans.find(key);
        if(it == exec_plans.end())
            throw HIPFFT_INVALID_PLAN;
        return it->second.rocfft_plan;
    }
};

static inline hipfftResult handle_exception() noexcept
try
{
    throw;
}
catch(hipfftResult e)
{
    return e;
}
catch(const DEVICEBUF_MEM_USAGE& e)
{
    return HIPFFT_ALLOC_FAILED;
}
catch(const std::exception& e)
{
    HIPFFT_DEBUG_LOG(e.what());
    return HIPFFT_INTERNAL_ERROR;
}
catch(...)
{
    return HIPFFT_INTERNAL_ERROR;
}

hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch)
try
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan1d(*plan, nx, type, batch, nullptr);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type)
try
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan2d(*plan, nx, ny, type, nullptr);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type)
try
{

    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan3d(*plan, nx, ny, nz, type, nullptr);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftPlanMany(hipfftHandle* plan,
                            int           rank,
                            int*          n,
                            int*          inembed,
                            int           istride,
                            int           idist,
                            int*          onembed,
                            int           ostride,
                            int           odist,
                            hipfftType    type,
                            int           batch)
try
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlanMany(
        *plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, nullptr);
}
catch(...)
{
    return handle_exception();
}

// note: rm_lengths arg is in row-major order
static hipfftResult hipfftMakePlan_internal(hipfftHandle               plan,
                                            const std::vector<size_t>& rm_lengths,
                                            const hipfftIOType&        iotype,
                                            size_t                     number_of_transforms,
                                            hipfft_ionembed_t<size_t>* user_ionembed,
                                            size_t                     user_idist,
                                            size_t                     user_odist,
                                            size_t*                    workSize)
{
    if(!plan || plan->initialized())
    {
        // plan initialization can be done only once in the plan's lifetime
        return HIPFFT_INVALID_PLAN;
    }

    // magic static to handle rocfft setup/cleanup
    struct rocfft_initializer
    {
        rocfft_initializer()
        {
            rocfft_setup();
        }
        ~rocfft_initializer()
        {
            rocfft_cleanup();
        }
    };
    static rocfft_initializer                                                 init;
    static const std::vector<std::pair<hipfftXtSubFormat, hipfftXtSubFormat>> all_io_formats
        = {{HIPFFT_XT_FORMAT_INPLACE, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED},
           {HIPFFT_XT_FORMAT_INPLACE_SHUFFLED, HIPFFT_XT_FORMAT_INPLACE},
           {HIPFFT_XT_FORMAT_INPUT, HIPFFT_XT_FORMAT_OUTPUT}};

    plan->io_type = iotype;
    if(plan->device_ids.size() > 1)
    {
        // We currently do not support multi-batch multi-device transforms.
        if(number_of_transforms > 1)
            return HIPFFT_NOT_IMPLEMENTED;

        // We currently do not support 1D multi-device transforms.
        if(rm_lengths.size() == 1)
            return HIPFFT_NOT_IMPLEMENTED;
    }
    plan->batch             = number_of_transforms;
    plan->transform_lengths = rm_lengths;
    // copy the user's ionembed into the plan if there is one, use default otherwise
    plan->global_ionembed = !user_ionembed ? hipfft_ionembed_t<size_t>() : *user_ionembed;

    if(plan->device_ids.empty())
    {
        // not multi-device, so use the current device as the default
        plan->device_ids.push_back(rocfft_scoped_device::current_device());
    }

    const std::vector<size_t> cm_lengths_vec(plan->transform_lengths.rbegin(),
                                             plan->transform_lengths.rend());
    // NOTE: hipFFT ignores distance arguments if default layouts are used!
    const bool ignore_user_distances = !plan->global_ionembed.get_nembed(fft_io::fft_io_in)
                                       && !plan->global_ionembed.get_nembed(fft_io::fft_io_out);
    for(auto dft_type : iotype.transform_types())
    {
        for(const auto& io_formats : all_io_formats)
        {
            if(plan->device_ids.size() > 1)
            {
                if(io_formats == std::pair{HIPFFT_XT_FORMAT_INPUT, HIPFFT_XT_FORMAT_OUTPUT})
                {
                    // only in-place for any multi-device, for now
                    continue;
                }
                // R2C is only HIPFFT_XT_FORMAT_INPLACE-to-HIPFFT_XT_FORMAT_INPLACE_SHUFFLED.
                // C2R is only HIPFFT_XT_FORMAT_INPLACE_SHUFFLED-to-HIPFFT_XT_FORMAT_INPLACE.
                if((iotype.is_real_to_complex()
                    && io_formats
                           != std::pair{HIPFFT_XT_FORMAT_INPLACE,
                                        HIPFFT_XT_FORMAT_INPLACE_SHUFFLED})
                   || (iotype.is_complex_to_real()
                       && io_formats
                              != std::pair{HIPFFT_XT_FORMAT_INPLACE_SHUFFLED,
                                           HIPFFT_XT_FORMAT_INPLACE}))
                {
                    continue;
                }
            }
            else if(io_formats
                    == std::pair{HIPFFT_XT_FORMAT_INPLACE_SHUFFLED, HIPFFT_XT_FORMAT_INPLACE})
            {
                // redundant with
                // input_fmt == std::pair{HIPFFT_XT_FORMAT_INPLACE, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED}
                // for single-device cases
                continue;
            }
            const auto placement
                = io_formats == std::pair{HIPFFT_XT_FORMAT_INPUT, HIPFFT_XT_FORMAT_OUTPUT}
                      ? rocfft_placement_notinplace
                      : rocfft_placement_inplace;
            rocfft_plan_description_wrapper_t desc;

            ROCFFT_EXPECT_SUCCESS(desc.alloc_with_err());

            auto i_strides = plan->global_ionembed.as_generalized_strides(
                fft_io::fft_io_in,
                fft_transform_type_from_rocfft_transform_type(dft_type),
                fft_result_placement_from_rocfft_result_placement(placement),
                plan->transform_lengths);
            auto o_strides = plan->global_ionembed.as_generalized_strides(
                fft_io::fft_io_out,
                fft_transform_type_from_rocfft_transform_type(dft_type),
                fft_result_placement_from_rocfft_result_placement(placement),
                plan->transform_lengths);

            // rm -> cm:
            std::reverse(i_strides.begin(), i_strides.end());
            std::reverse(o_strides.begin(), o_strides.end());
            const auto inDist
                = !ignore_user_distances
                      ? user_idist
                      : default_distance(
                          fft_transform_type_from_rocfft_transform_type(dft_type),
                          fft_result_placement_from_rocfft_result_placement(placement),
                          fft_io::fft_io_in,
                          plan->transform_lengths,
                          number_of_transforms);
            const auto outDist
                = !ignore_user_distances
                      ? user_odist
                      : default_distance(
                          fft_transform_type_from_rocfft_transform_type(dft_type),
                          fft_result_placement_from_rocfft_result_placement(placement),
                          fft_io::fft_io_out,
                          plan->transform_lengths,
                          number_of_transforms);

            ROCFFT_EXPECT_SUCCESS(
                rocfft_plan_description_set_data_layout(desc,
                                                        iotype.array_type(fft_io::fft_io_in),
                                                        iotype.array_type(fft_io::fft_io_out),
                                                        nullptr,
                                                        nullptr,
                                                        i_strides.size(),
                                                        i_strides.data(),
                                                        inDist,
                                                        o_strides.size(),
                                                        o_strides.data(),
                                                        outDist));

            if(plan->scale_factor != 1.0)
                ROCFFT_EXPECT_SUCCESS(
                    rocfft_plan_description_set_scale_factor(desc, plan->scale_factor));

            if(plan->comm_type != rocfft_comm_none)
                ROCFFT_EXPECT_SUCCESS(
                    rocfft_plan_description_set_comm(desc, plan->comm_type, plan->comm_handle));

            std::optional<hipfft_field> input_field, output_field;
            // For multi-device, we need to create a field for each plan, so that the
            // bricks can be set up for the input and output fields.
            auto global_field_batch_lengths = rm_lengths;
            global_field_batch_lengths.insert(global_field_batch_lengths.begin(),
                                              number_of_transforms);
            for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
            {
                auto&      io_field = io == fft_io::fft_io_in ? input_field : output_field;
                const auto field_format
                    = io == fft_io::fft_io_in ? io_formats.first : io_formats.second;
                const auto split_dim = number_of_transforms > 1
                                           ? 0
                                           : (field_format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
                                                      || field_format == HIPFFT_XT_FORMAT_OUTPUT
                                                  ? 2
                                                  : 1);
                io_field.emplace(global_field_batch_lengths,
                                 fft_transform_type_from_rocfft_transform_type(dft_type),
                                 fft_result_placement_from_rocfft_result_placement(placement),
                                 io,
                                 split_dim,
                                 plan->device_ids);
                if(plan->device_ids.size() > 1)
                {
                    hipfft_field::add_field_to_descriptor(*io_field, desc, io);
                }
            }
            rocfft_plan_wrapper_t rocfft_plan;
            auto                  plan_creation_status = rocfft_plan.alloc_with_err(placement,
                                                                   dft_type,
                                                                   iotype.precision(),
                                                                   cm_lengths_vec.size(),
                                                                   cm_lengths_vec.data(),
                                                                   number_of_transforms,
                                                                   desc);
            if(plan_creation_status != rocfft_status_success)
            {
                // some plan creates might fail (legitimately) for explicit user-given strides,
                // (e.g., in-place real transforms have compliant strides only for one direction),
                continue;
            }
            // add successful plan to the map, keyed by transform type and input descriptor's subformat
            plan->exec_plans.emplace(
                hipfftHandle_t::map_key_t{dft_type, io_formats.first, io_formats.second},
                hipfftHandle_t::plan_and_fields_t{
                    std::move(rocfft_plan), std::move(input_field), std::move(output_field)});
        }
    }

    // If no plans got created or any map's plan is null, fail
    if(plan->exec_plans.empty()
       || std::any_of(plan->exec_plans.begin(), plan->exec_plans.end(), [](const auto& p) {
              return !p.second.rocfft_plan;
          }))
    {
        return HIPFFT_PARSE_ERROR;
    }

    // Initialize device-specific execution info parameters for each device in the plan:
    // - a stream is allocated for each device
    // - the required work buffer size is determined
    // - work buffers are allocated if auto_allocate is true
    for(size_t idx = 0; idx < plan->device_ids.size(); ++idx)
    {
        const auto           device_id = plan->device_ids[idx];
        rocfft_scoped_device scoped_dev(device_id);
        const auto& [tmp_iter, inserted] = plan->exec_data.emplace(
            device_id, hipfftHandle_t::hipfft_exec_info_params_t::create_for_current_device());
        if(!inserted)
            throw HIPFFT_INTERNAL_ERROR;
        std::for_each(plan->exec_plans.begin(), plan->exec_plans.end(), [&](const auto& p) {
            size_t tmp = 0;
            ROCFFT_EXPECT_SUCCESS(rocfft_plan_get_work_buffer_size(p.second.rocfft_plan, &tmp));
            tmp_iter->second.work_buffer_byte_bsize
                = std::max(tmp_iter->second.work_buffer_byte_bsize, tmp);
        });
        if(workSize != nullptr)
            workSize[idx] = tmp_iter->second.work_buffer_byte_bsize;
        if(plan->auto_allocate && tmp_iter->second.work_buffer_byte_bsize > 0)
        {
            if(tmp_iter->second.work_buffer.alloc(tmp_iter->second.work_buffer_byte_bsize)
               != hipSuccess)
                return HIPFFT_ALLOC_FAILED;
            ROCFFT_EXPECT_SUCCESS(
                rocfft_execution_info_set_work_buffer(plan->info,
                                                      tmp_iter->second.work_buffer.data(),
                                                      tmp_iter->second.work_buffer_byte_bsize));
        }
    }

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftCreate(hipfftHandle* plan)
try
{
    // NOTE: cufft backend uses int for handle type, so this wouldn't
    // work using cufft types.  This is the rocfft backend, but
    // cppcheck doesn't know that.  Compiler would complain anyway
    // about making integer from pointer without a cast.
    //
    // But just for good measure, we can at least assert that the
    // destination is wide enough to fit a pointer.
    //
    static_assert(sizeof(hipfftHandle) >= sizeof(void*),
                  "hipfftHandle type not wide enough for pointer");
    // cppcheck-suppress AssignmentAddressToInteger
    hipfftHandle h = new hipfftHandle_t;
    ROCFFT_EXPECT_SUCCESS(h->info.alloc_with_err());
    *plan = h;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExtPlanScaleFactor(hipfftHandle plan, double scalefactor)
try
{
    if(!std::isfinite(scalefactor))
        return HIPFFT_INVALID_VALUE;
    plan->scale_factor = scalefactor;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
try
{
    if(nx < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    std::vector<size_t>        lengths(1, nx);
    size_t                     number_of_transforms = batch;
    hipfft_ionembed_t<size_t>* user_ionembed        = nullptr;
    // ignored internally (default layout)
    size_t ignored_dist = 0;

    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(type));

    return hipfftMakePlan_internal(plan,
                                   lengths,
                                   iotype,
                                   number_of_transforms,
                                   user_ionembed,
                                   ignored_dist,
                                   ignored_dist,
                                   workSize);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
try
{
    if(nx < 0 || ny < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    std::vector<size_t>        lengths{static_cast<size_t>(nx), static_cast<size_t>(ny)};
    size_t                     number_of_transforms = 1;
    hipfft_ionembed_t<size_t>* user_ionembed        = nullptr;
    // ignored internally (default layout)
    size_t ignored_dist = 0;

    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(type));

    return hipfftMakePlan_internal(plan,
                                   lengths,
                                   iotype,
                                   number_of_transforms,
                                   user_ionembed,
                                   ignored_dist,
                                   ignored_dist,
                                   workSize);
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
try
{
    if(nx < 0 || ny < 0 || nz < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    std::vector<size_t> lengths{
        static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)};
    size_t                     number_of_transforms = 1;
    hipfft_ionembed_t<size_t>* user_ionembed        = nullptr;
    // ignored internally (default layout)
    size_t ignored_dist = 0;

    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(type));

    return hipfftMakePlan_internal(plan,
                                   lengths,
                                   iotype,
                                   number_of_transforms,
                                   user_ionembed,
                                   ignored_dist,
                                   ignored_dist,
                                   workSize);
}
catch(...)
{
    return handle_exception();
}

template <typename T>
static hipfftResult hipfftMakePlanMany_internal(hipfftHandle plan,
                                                int          rank,
                                                T*           n,
                                                T*           inembed,
                                                T            istride,
                                                T            idist,
                                                T*           onembed,
                                                T            ostride,
                                                T            odist,
                                                hipfftIOType type,
                                                T            batch,
                                                size_t*      workSize)
{
    if((inembed != nullptr && onembed == nullptr) || (inembed == nullptr && onembed != nullptr)
       || (rank < 0) || (istride < 0) || (idist < 0) || (ostride < 0) || (odist < 0)
       || (std::any_of(n, n + rank, [](T val) { return val < 0; })))
        return HIPFFT_INVALID_VALUE;

    for(auto ptr : {inembed, onembed})
    {
        if(ptr == nullptr)
            continue;
        if(std::any_of(ptr, ptr + rank, [](T val) { return val <= 0; }))
            return HIPFFT_INVALID_SIZE;
    }

    if(batch <= 0)
        return HIPFFT_INVALID_SIZE;

    std::vector<size_t>       lengths(n, n + rank);
    hipfft_ionembed_t<size_t> user_ionembed(rank, istride, inembed, ostride, onembed);
    size_t                    number_of_transforms = batch;
    const size_t              user_idist           = idist;
    const size_t              user_odist           = odist;

    hipfftResult ret = hipfftMakePlan_internal(plan,
                                               lengths,
                                               type,
                                               number_of_transforms,
                                               &user_ionembed,
                                               user_idist,
                                               user_odist,
                                               workSize);

    return ret;
}

hipfftResult hipfftMakePlanMany(hipfftHandle plan,
                                int          rank,
                                int*         n,
                                int*         inembed,
                                int          istride,
                                int          idist,
                                int*         onembed,
                                int          ostride,
                                int          odist,
                                hipfftType   type,
                                int          batch,
                                size_t*      workSize)
try
{
    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(type));

    return hipfftMakePlanMany_internal<int>(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, iotype, batch, workSize);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftMakePlanMany64(hipfftHandle   plan,
                                  int            rank,
                                  long long int* n,
                                  long long int* inembed,
                                  long long int  istride,
                                  long long int  idist,
                                  long long int* onembed,
                                  long long int  ostride,
                                  long long int  odist,
                                  hipfftType     type,
                                  long long int  batch,
                                  size_t*        workSize)
try
{
    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(type));

    return hipfftMakePlanMany_internal<long long int>(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, iotype, batch, workSize);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t* workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    hipfftHandle plan = nullptr;
    hipfftResult ret  = hipfftGetSize1d(plan, nx, type, batch, workSize);
    return ret;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    hipfftHandle plan = nullptr;
    hipfftResult ret  = hipfftGetSize2d(plan, nx, ny, type, workSize);
    return ret;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    hipfftHandle plan = nullptr;
    hipfftResult ret  = hipfftGetSize3d(plan, nx, ny, nz, type, workSize);
    return ret;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftEstimateMany(int        rank,
                                int*       n,
                                int*       inembed,
                                int        istride,
                                int        idist,
                                int*       onembed,
                                int        ostride,
                                int        odist,
                                hipfftType type,
                                int        batch,
                                size_t*    workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    hipfftHandle plan = nullptr;
    hipfftResult ret  = hipfftGetSizeMany(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
    return ret;
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    if(nx < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    p->auto_allocate = false;
    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlan1d(p, nx, type, batch, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    if(nx < 0 || ny < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    p->auto_allocate = false;
    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlan2d(p, nx, ny, type, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    if(nx < 0 || ny < 0 || nz < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    p->auto_allocate = false;
    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlan3d(p, nx, ny, nz, type, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetSizeMany(hipfftHandle plan,
                               int          rank,
                               int*         n,
                               int*         inembed,
                               int          istride,
                               int          idist,
                               int*         onembed,
                               int          ostride,
                               int          odist,
                               hipfftType   type,
                               int          batch,
                               size_t*      workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    hipfftHandle p = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    p->auto_allocate = false;
    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlanMany(
        p, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetSizeMany64(hipfftHandle   plan,
                                 int            rank,
                                 long long int* n,
                                 long long int* inembed,
                                 long long int  istride,
                                 long long int  idist,
                                 long long int* onembed,
                                 long long int  ostride,
                                 long long int  odist,
                                 hipfftType     type,
                                 long long int  batch,
                                 size_t*        workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    hipfftHandle p = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    p->auto_allocate = false;
    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlanMany64(
        p, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;
    for(size_t idx = 0; idx < plan->device_ids.size(); ++idx)
    {
        const auto it = plan->exec_data.find(plan->device_ids[idx]);
        if(it == plan->exec_data.end())
            return HIPFFT_INVALID_PLAN;
        workSize[idx] = it->second.work_buffer_byte_bsize;
    }
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
try
{
    if(!plan)
        return HIPFFT_INVALID_PLAN;
    plan->auto_allocate = bool(autoAllocate);
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea)
try
{
    if(!plan || !plan->initialized() || plan->device_ids.empty())
        return HIPFFT_INVALID_PLAN;
    if(plan->device_ids.size() > 1)
    {
        // wrong API for multi-device usage, hipfftXtSetWorkArea (yet to
        // be implemented) must be used for multi-device plans
        return HIPFFT_INVALID_PLAN;
    }

    auto it = plan->exec_data.find(plan->device_ids[0]);
    if(it == plan->exec_data.end())
        return HIPFFT_INTERNAL_ERROR;
    if(it->second.work_buffer_byte_bsize == 0)
        return HIPFFT_SUCCESS;
    if(!workArea)
        return HIPFFT_INVALID_VALUE;

    auto tmp               = gpubuf::make_nonowned(workArea, it->second.work_buffer_byte_bsize);
    it->second.work_buffer = std::move(tmp);
    if(workArea)
    {
        ROCFFT_EXPECT_SUCCESS(rocfft_execution_info_set_work_buffer(
            plan->info, it->second.work_buffer.data(), it->second.work_buffer_byte_bsize));
    }
    plan->auto_allocate = false;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

static hipfftResult hipfftExec(const rocfft_plan&           rplan,
                               const rocfft_execution_info& rinfo,
                               void*                        idata,
                               void*                        odata)
{
    if(!rplan)
        return HIPFFT_INVALID_PLAN;
    if(!idata || !odata)
        return HIPFFT_INVALID_VALUE;
    void*      in[1]  = {idata};
    void*      out[1] = {odata};
    const auto ret    = rocfft_execute(rplan, in, out, rinfo);
    return ret == rocfft_status_success ? HIPFFT_SUCCESS : HIPFFT_EXEC_FAILED;
}

template <rocfft_precision_e prec>
static inline bool is_ready_for_execution(const hipfftHandle_t* plan)
{
    return plan != nullptr && plan->initialized() && plan->io_type.precision() == prec;
}

hipfftResult
    hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
try
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftExec(
        plan->get_single_device_rocfft_plan(direction, idata, odata), plan->info, idata, odata);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
try
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftExec(plan->get_single_device_rocfft_plan(HIPFFT_FORWARD, idata, odata),
                      plan->info,
                      idata,
                      odata);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
try
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftExec(plan->get_single_device_rocfft_plan(HIPFFT_BACKWARD, idata, odata),
                      plan->info,
                      idata,
                      odata);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                           hipfftDoubleComplex* idata,
                           hipfftDoubleComplex* odata,
                           int                  direction)
try
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftExec(
        plan->get_single_device_rocfft_plan(direction, idata, odata), plan->info, idata, odata);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* idata, hipfftDoubleComplex* odata)
try
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftExec(plan->get_single_device_rocfft_plan(HIPFFT_FORWARD, idata, odata),
                      plan->info,
                      idata,
                      odata);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
try
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftExec(plan->get_single_device_rocfft_plan(HIPFFT_BACKWARD, idata, odata),
                      plan->info,
                      idata,
                      odata);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
try
{
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;
    auto dev_id = hipInvalidDeviceId;
    HIP_EXPECT_SUCCESS(hipStreamGetDevice(stream, &dev_id));
    if(dev_id == hipInvalidDeviceId)
        return HIPFFT_INTERNAL_ERROR;
    auto it = plan->exec_data.find(dev_id);
    if(it == plan->exec_data.end())
    {
        // given stream is on a device that is not part of the plan's device list
        return HIPFFT_INVALID_VALUE;
    }
    it->second.stream = hipStream_wrapper_t::make_nonowned(stream);
    ROCFFT_EXPECT_SUCCESS(rocfft_execution_info_set_stream(plan->info, it->second.stream));
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftDestroy(hipfftHandle plan)
try
{
    delete plan;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetVersion(int* version)
try
{
    if(!version)
        return HIPFFT_INVALID_VALUE;
    char v[256];
    ROCFFT_EXPECT_SUCCESS(rocfft_get_version_string(v, 256));

    // export major.minor.patch only, ignore tweak
    std::ostringstream       result;
    std::vector<std::string> sections;

    std::istringstream iss(v);
    std::string        tmp_str;
    while(std::getline(iss, tmp_str, '.'))
    {
        sections.push_back(tmp_str);
    }

    for(size_t i = 0; i < std::min<size_t>(sections.size(), 3); i++)
    {
        if(sections[i].size() == 1)
            result << "0" << sections[i];
        else
            result << sections[i];
    }

    *version = std::stoi(result.str());
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int* value)
try
{
    if(!value)
        return HIPFFT_INVALID_VALUE;
    int full;
    hipfftGetVersion(&full);

    int major = full / 10000;
    int minor = (full - major * 10000) / 100;
    int patch = (full - major * 10000 - minor * 100);

    if(type == HIPFFT_MAJOR_VERSION)
        *value = major;
    else if(type == HIPFFT_MINOR_VERSION)
        *value = minor;
    else if(type == HIPFFT_PATCH_LEVEL)
        *value = patch;
    else
        return HIPFFT_INVALID_VALUE;

    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetCallback(hipfftHandle         plan,
                                 void**               callbacks,
                                 hipfftXtCallbackType cbtype,
                                 void**               callbackData)
try
{
    if(!plan)
        return HIPFFT_INVALID_PLAN;

    // check that the input/output type matches what's being requested
    //
    // NOTE: cufft explicitly does not save shared memory bytes when
    // you set a new callback, so zero out our number when setting
    // pointers
    switch(cbtype)
    {
    case HIPFFT_CB_LD_COMPLEX:
        if(plan->io_type.precision() != rocfft_precision_single
           || plan->io_type.is_real_to_complex())
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
        if(plan->io_type.precision() != rocfft_precision_double
           || plan->io_type.is_real_to_complex())
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_REAL:
        if(plan->io_type.precision() != rocfft_precision_single
           || !plan->io_type.is_real_to_complex())
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_REAL_DOUBLE:
        if(plan->io_type.precision() != rocfft_precision_double
           || !plan->io_type.is_real_to_complex())
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_COMPLEX:
        if(plan->io_type.precision() != rocfft_precision_single
           || plan->io_type.is_complex_to_real())
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
        if(plan->io_type.precision() != rocfft_precision_double
           || plan->io_type.is_complex_to_real())
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_REAL:
        if(plan->io_type.precision() != rocfft_precision_single
           || !plan->io_type.is_complex_to_real())
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_REAL_DOUBLE:
        if(plan->io_type.precision() != rocfft_precision_double
           || !plan->io_type.is_complex_to_real())
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_UNDEFINED:
        return HIPFFT_INVALID_VALUE;
    }

    rocfft_status res;
    res = rocfft_execution_info_set_load_callback(plan->info,
                                                  plan->load_callback_ptrs,
                                                  plan->load_callback_data,
                                                  plan->load_callback_lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    res = rocfft_execution_info_set_store_callback(plan->info,
                                                   plan->store_callback_ptrs,
                                                   plan->store_callback_data,
                                                   plan->store_callback_lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtClearCallback(hipfftHandle plan, hipfftXtCallbackType cbtype)
try
{
    return hipfftXtSetCallback(plan, nullptr, cbtype, nullptr);
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftXtSetCallbackSharedSize(hipfftHandle plan, hipfftXtCallbackType cbtype, size_t sharedSize)
try
{
    if(!plan)
        return HIPFFT_INVALID_PLAN;

    switch(cbtype)
    {
    case HIPFFT_CB_LD_COMPLEX:
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
    case HIPFFT_CB_LD_REAL:
    case HIPFFT_CB_LD_REAL_DOUBLE:
        plan->load_callback_lds_bytes = sharedSize;
        break;
    case HIPFFT_CB_ST_COMPLEX:
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
    case HIPFFT_CB_ST_REAL:
    case HIPFFT_CB_ST_REAL_DOUBLE:
        plan->store_callback_lds_bytes = sharedSize;
        break;
    case HIPFFT_CB_UNDEFINED:
        return HIPFFT_INVALID_VALUE;
    }

    rocfft_status res;
    res = rocfft_execution_info_set_load_callback(plan->info,
                                                  plan->load_callback_ptrs,
                                                  plan->load_callback_data,
                                                  plan->load_callback_lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    res = rocfft_execution_info_set_store_callback(plan->info,
                                                   plan->store_callback_ptrs,
                                                   plan->store_callback_data,
                                                   plan->store_callback_lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtMakePlanMany(hipfftHandle   plan,
                                  int            rank,
                                  long long int* n,
                                  long long int* inembed,
                                  long long int  istride,
                                  long long int  idist,
                                  hipDataType    inputtype,
                                  long long int* onembed,
                                  long long int  ostride,
                                  long long int  odist,
                                  hipDataType    outputtype,
                                  long long int  batch,
                                  size_t*        workSize,
                                  hipDataType    executiontype)
try
{
    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(inputtype, outputtype, executiontype));
    return hipfftMakePlanMany_internal<long long int>(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, iotype, batch, workSize);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtGetSizeMany(hipfftHandle   plan,
                                 int            rank,
                                 long long int* n,
                                 long long int* inembed,
                                 long long int  istride,
                                 long long int  idist,
                                 hipDataType    inputtype,
                                 long long int* onembed,
                                 long long int  ostride,
                                 long long int  odist,
                                 hipDataType    outputtype,
                                 long long int  batch,
                                 size_t*        workSize,
                                 hipDataType    executiontype)
try
{
    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(inputtype, outputtype, executiontype));

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    p->auto_allocate = false;

    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlanMany_internal(
        p, rank, n, inembed, istride, idist, onembed, ostride, odist, iotype, batch, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExec(hipfftHandle plan, void* input, void* output, int direction)
try
{
    return hipfftExec(
        plan->get_single_device_rocfft_plan(direction, input, output), plan->info, input, output);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetGPUs(hipfftHandle plan, int count, int* gpus)
try
{
    if(count <= 0 || !gpus)
        return HIPFFT_INVALID_VALUE;
    if(!plan || plan->initialized())
        return HIPFFT_INVALID_PLAN;
    const auto dev_count = rocfft_scoped_device::device_count();
    if(dev_count <= 0)
        return HIPFFT_INTERNAL_ERROR;
    if(std::any_of(
           gpus, gpus + count, [=](int gpu_id) { return gpu_id < 0 || gpu_id >= dev_count; }))
        return HIPFFT_INVALID_VALUE;
    plan->device_ids.assign(gpus, gpus + count);

    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtMalloc(hipfftHandle plan, hipLibXtDesc** desc, hipfftXtSubFormat format)
try
{
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;
    if(!desc)
        return HIPFFT_INVALID_VALUE;

    if(format == HIPFFT_FORMAT_UNDEFINED)
        return HIPFFT_INVALID_VALUE;

    // 1D transforms are not currently implemented.
    if(format == HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED)
        return HIPFFT_NOT_IMPLEMENTED;

    // Only in-place multi-gpu transforms are currently implemented.
    if(format == HIPFFT_XT_FORMAT_INPUT || format == HIPFFT_XT_FORMAT_OUTPUT)
        return HIPFFT_NOT_IMPLEMENTED;

    // Real-to-complex is HIPFFT_XT_FORMAT_INPLACE-to-HIPFFT_XT_FORMAT_INPLACE_SHUFFLED.
    // Complex-to-real is HIPFFT_XT_FORMAT_INPLACE_SHUFFLED-to-HIPFFT_XT_FORMAT_INPLACE.
    if(plan->io_type.is_real_to_complex() && format != HIPFFT_XT_FORMAT_INPLACE)
        return HIPFFT_NOT_IMPLEMENTED;
    if(plan->io_type.is_complex_to_real() && format != HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
        return HIPFFT_NOT_IMPLEMENTED;

    std::unique_ptr<hipLibXtDesc, decltype(&hipfftXtFree)> lib_desc(new hipLibXtDesc, hipfftXtFree);
    std::memset(lib_desc.get(), 0, sizeof(hipLibXtDesc));

    lib_desc->version       = 0;
    lib_desc->library       = HIPLIB_FORMAT_HIPFFT;
    lib_desc->subFormat     = format;
    lib_desc->libDescriptor = nullptr;
    lib_desc->descriptor    = new hipXtDesc;
    std::memset(lib_desc->descriptor, 0, sizeof(hipXtDesc));
    auto xt_desc     = lib_desc->descriptor;
    xt_desc->version = 0;
    xt_desc->nGPUs   = static_cast<int>(plan->device_ids.size());
    std::copy(plan->device_ids.begin(), plan->device_ids.end(), xt_desc->GPUs);
    for(size_t dev_idx = 0; dev_idx < plan->device_ids.size(); ++dev_idx)
    {
        xt_desc->size[dev_idx] = 0;
        for(const auto& [key, plan_and_field] : plan->exec_plans)
        {
            if(key.input_desc_format != format && key.output_desc_format != format)
            {
                continue;
            }
            // must have input and output field with enough bricks
            if(!plan_and_field.input_field || !plan_and_field.output_field
               || dev_idx >= plan_and_field.input_field->brick_count()
               || dev_idx >= plan_and_field.output_field->brick_count())
                throw HIPFFT_INVALID_PLAN;
            if(format == HIPFFT_XT_FORMAT_INPUT || format == HIPFFT_XT_FORMAT_INPLACE
               || format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
            {
                xt_desc->size[dev_idx]
                    = std::max(xt_desc->size[dev_idx],
                               plan_and_field.input_field->get_brick(dev_idx).data_byte_size(
                                   plan->io_type.get_inputType()));
            }
            if(format == HIPFFT_XT_FORMAT_OUTPUT || format == HIPFFT_XT_FORMAT_INPLACE
               || format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
            {
                xt_desc->size[dev_idx]
                    = std::max(xt_desc->size[dev_idx],
                               plan_and_field.output_field->get_brick(dev_idx).data_byte_size(
                                   plan->io_type.get_outputType()));
            }
        }
        if(xt_desc->size[dev_idx] == 0)
        {
            // TODO: how should we handle the case where some devices don't have data?
            return HIPFFT_NOT_IMPLEMENTED;
        }
        rocfft_scoped_device dev(plan->device_ids[dev_idx]);
        if(hipMalloc(&(xt_desc->data[dev_idx]), xt_desc->size[dev_idx]) != hipSuccess)
            return HIPFFT_ALLOC_FAILED;
    }
    *desc = lib_desc.release();
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

// Collapse contiguous dimensions in the specified length + stride -
// user data might be split on any dimension so if we can simplify to
// just one split dimension and one contiguous dimension we can more
// easily map a XtMemcpy to a 2DMemcpy.
// Data is row-major.
static void collapse_contiguous_dims(std::vector<size_t>& brick_length,
                                     std::vector<size_t>& brick_stride,
                                     std::vector<size_t>& field_stride)
{
    // Easier error messages helper:
    auto paramstring = [&]() -> std::string {
        std::stringstream ss;
        ss << "brick_length:";
        for(auto val : brick_length)
            ss << " " << val;
        ss << "\nbrick_stride:";
        for(auto val : brick_stride)
            ss << " " << val;
        ss << "\nfield_stride:";
        for(auto val : field_stride)
            ss << " " << val;
        return ss.str();
    };

    if((brick_length.size() != brick_stride.size()) || (brick_length.size() != field_stride.size())
       || (brick_stride.size() != field_stride.size()))
    {
        throw std::runtime_error("Inconsistent dimensions for collapse_contiguous_dims.\n"
                                 + paramstring());
    }

    // Also remove all columns where length is 1:
    for(size_t idx = 0; idx < brick_length.size(); ++idx)
    {
        if(brick_length[idx] == 1)
        {
            brick_length.erase(brick_length.begin() + idx);
            brick_stride.erase(brick_stride.begin() + idx);
            field_stride.erase(field_stride.begin() + idx);
            --idx;
        }
    }

    // Collapse contiguous memory sections:
    for(size_t idx = brick_length.size(); idx-- > 1;)
    {
        if(brick_length[idx] * brick_stride[idx] == brick_stride[idx - 1]
           && brick_length[idx] * field_stride[idx] == field_stride[idx - 1])
        {
            brick_length[idx - 1] *= brick_length[idx];
            brick_length.erase(brick_length.begin() + idx);
            brick_stride.erase(brick_stride.begin() + idx - 1);
            field_stride.erase(field_stride.begin() + idx - 1);
        }
    }
}

hipfftResult hipfftXtMemcpy(hipfftHandle plan, void* dest, void* src, hipfftXtCopyType cptype)
try
{
    if(!plan || !plan->initialized() || plan->device_ids.size() < 1)
        return HIPFFT_INVALID_PLAN;
    if(!dest || !src || dest == src)
        return HIPFFT_INVALID_VALUE;

    // only H2D or D2H is currently implemented
    if(cptype == HIPFFT_COPY_DEVICE_TO_DEVICE)
        return HIPFFT_NOT_IMPLEMENTED;
    // any other value is invalid
    if(cptype != HIPFFT_COPY_HOST_TO_DEVICE && cptype != HIPFFT_COPY_DEVICE_TO_HOST)
        return HIPFFT_INVALID_VALUE;

    const bool h2d     = cptype == HIPFFT_COPY_HOST_TO_DEVICE;
    auto&      xt_desc = *static_cast<hipLibXtDesc*>(h2d ? dest : src);
    // validate user-given descriptor w.r.t. plan
    if(plan->device_ids.size() != static_cast<size_t>(xt_desc.descriptor->nGPUs))
        return HIPFFT_INVALID_VALUE;
    for(size_t dev_idx = 0; dev_idx < plan->device_ids.size(); ++dev_idx)
    {
        if(xt_desc.descriptor->GPUs[dev_idx] != plan->device_ids[dev_idx])
            return HIPFFT_INVALID_VALUE;
    }
    // given descriptor's format is considered input descriptor's format
    // for H2D and output descriptor's format for D2H
    const auto split_dim = plan->batch > 1
                               ? 0
                               : (xt_desc.subFormat == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
                                          || xt_desc.subFormat == HIPFFT_XT_FORMAT_OUTPUT
                                      ? 2
                                      : 1);
    auto       tmp       = plan->transform_lengths;
    tmp.insert(tmp.begin(), plan->batch);
    hipfft_field field(tmp,
                       plan->io_type.is_complex_to_complex() ? fft_transform_type_complex_forward
                                                             : fft_transform_type_real_forward,
                       xt_desc.subFormat == HIPFFT_XT_FORMAT_INPLACE
                               || xt_desc.subFormat == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
                           ? fft_placement_inplace
                           : fft_placement_notinplace,
                       xt_desc.subFormat == HIPFFT_XT_FORMAT_INPLACE
                               || xt_desc.subFormat == HIPFFT_XT_FORMAT_INPUT
                           ? fft_io_in
                           : fft_io_out,
                       split_dim,
                       plan->device_ids);
    const auto element_type = h2d ? plan->io_type.get_inputType() : plan->io_type.get_outputType();
    for(size_t brick_idx = 0; brick_idx < field.brick_count(); ++brick_idx)
    {
        const auto [collapsed_brick, collapsed_field]
            = field.get_collapsed_brick_in_collapsed_field(brick_idx);
        rocfft_scoped_device dev(collapsed_brick.get_device_id());
        const auto           dev_exec_data = plan->exec_data.find(collapsed_brick.get_device_id());
        if(dev_exec_data == plan->exec_data.end())
            return HIPFFT_INVALID_PLAN;
        void* host_ptr
            = static_cast<char*>(h2d ? src : dest)
              + collapsed_brick.offset_in(collapsed_field) * hipDataType_bytes(element_type);
        const auto data_sz = collapsed_brick.data_byte_size(element_type);
        if(collapsed_brick.full_rank() == 1)
        {
            HIP_EXPECT_SUCCESS(hipMemcpyAsync(h2d ? xt_desc.descriptor->data[brick_idx] : host_ptr,
                                              h2d ? host_ptr : xt_desc.descriptor->data[brick_idx],
                                              data_sz,
                                              h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost,
                                              dev_exec_data->second.stream));
        }
        else if(collapsed_brick.full_rank() == 2)
        {
            const auto brick_strides = collapsed_brick.get_strides();
            const auto brick_spans   = collapsed_brick.get_spans();
            const auto field_strides = collapsed_field.get_strides();
            HIP_EXPECT_SUCCESS(
                hipMemcpy2DAsync(h2d ? xt_desc.descriptor->data[brick_idx] : host_ptr,
                                 h2d ? brick_strides[0] * hipDataType_bytes(element_type)
                                     : field_strides[0] * hipDataType_bytes(element_type),
                                 h2d ? host_ptr : xt_desc.descriptor->data[brick_idx],
                                 h2d ? field_strides[0] * hipDataType_bytes(element_type)
                                     : brick_strides[0] * hipDataType_bytes(element_type),
                                 brick_spans[1] * hipDataType_bytes(element_type),
                                 brick_spans[0],
                                 h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost,
                                 dev_exec_data->second.stream));
        }
        else
        {
            return HIPFFT_INTERNAL_ERROR;
        }
    }

    for(auto dev_id : plan->device_ids)
    {
        rocfft_scoped_device dev(dev_id);
        const auto           it = plan->exec_data.find(dev_id);
        if(it == plan->exec_data.end())
            return HIPFFT_INVALID_PLAN;
        HIP_EXPECT_SUCCESS(hipStreamSynchronize(it->second.stream));
    }
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtFree(hipLibXtDesc* desc)
try
{
    if(desc && desc->descriptor)
    {
        for(size_t i = 0; i < static_cast<size_t>(desc->descriptor->nGPUs); ++i)
        {
            rocfft_scoped_device dev(desc->descriptor->GPUs[i]);
            (void)hipFree(desc->descriptor->data[i]);
        }
        delete desc->descriptor;
    }
    delete desc;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

static hipfftResult hipfftXtExecDescriptorBase(const hipfftHandle plan,
                                               int                direction,
                                               hipLibXtDesc*      input,
                                               hipLibXtDesc*      output)
{
    if(!input || !output)
        return HIPFFT_INVALID_VALUE;
    try
    {
        const auto ret = rocfft_execute(plan->get_rocfft_plan(direction, input, output),
                                        input->descriptor->data,
                                        output->descriptor->data,
                                        plan->info);
        if(ret == rocfft_status_success && input == output)
        {
            // If the execution was succesful, then we can change the subformat value if necessary.
            switch(input->subFormat)
            {
            case HIPFFT_XT_FORMAT_INPLACE:
                input->subFormat = HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
                break;
            case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
                input->subFormat = HIPFFT_XT_FORMAT_INPLACE;
                break;
            default:
                throw HIPFFT_INVALID_VALUE;
            }
        }
        if(ret != rocfft_status_success)
            return HIPFFT_EXEC_FAILED;
    }
    catch(...)
    {
        return handle_exception();
    }

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftXtExecDescriptorC2C(hipfftHandle  plan,
                                       hipLibXtDesc* input,
                                       hipLibXtDesc* output,
                                       int           direction)
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftXtExecDescriptorBase(plan, direction, input, output);
}

hipfftResult hipfftXtExecDescriptorR2C(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftXtExecDescriptorBase(plan, HIPFFT_FORWARD, input, output);
}

hipfftResult hipfftXtExecDescriptorC2R(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftXtExecDescriptorBase(plan, HIPFFT_BACKWARD, input, output);
}

hipfftResult hipfftXtExecDescriptorZ2Z(hipfftHandle  plan,
                                       hipLibXtDesc* input,
                                       hipLibXtDesc* output,
                                       int           direction)
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftXtExecDescriptorBase(plan, direction, input, output);
}

hipfftResult hipfftXtExecDescriptorD2Z(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftXtExecDescriptorBase(plan, HIPFFT_FORWARD, input, output);
}

hipfftResult hipfftXtExecDescriptorZ2D(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;
    return hipfftXtExecDescriptorBase(plan, HIPFFT_BACKWARD, input, output);
}

hipfftResult hipfftXtExecDescriptor(hipfftHandle  plan,
                                    hipLibXtDesc* input,
                                    hipLibXtDesc* output,
                                    int           direction)
{
    return hipfftXtExecDescriptorBase(plan, direction, input, output);
}

#ifdef HIPFFT_MPI_ENABLE
static rocfft_comm_type hipfftMpCommTypeToRocfftCommType(hipfftMpCommType_t hipfft_type)
{
    switch(hipfft_type)
    {
    case HIPFFT_COMM_MPI:
        return rocfft_comm_mpi;
    case HIPFFT_COMM_NONE:
        return rocfft_comm_none;
    }
    throw HIPFFT_INVALID_VALUE;
}

hipfftResult hipfftMpAttachComm(hipfftHandle plan, hipfftMpCommType comm_type, void* comm_handle)
try
{
    // comm must be known before plans are actually constructed
    if(!plan || plan->initialized())
        return HIPFFT_INVALID_PLAN;

    plan->comm_type   = hipfftMpCommTypeToRocfftCommType(comm_type);
    plan->comm_handle = comm_handle;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetDistribution(hipfftHandle         plan,
                                     int                  rank,
                                     const long long int* input_lower,
                                     const long long int* input_upper,
                                     const long long int* output_lower,
                                     const long long int* output_upper,
                                     const long long int* input_stride,
                                     const long long int* output_stride)
try
{
    // distribution must be set before plans are actually constructed
    if(!plan || plan->initialized())
        return HIPFFT_INVALID_PLAN;

    // one brick on this rank for each of input and output
    plan->spaceBricks.resize(1);
    plan->freqBricks.resize(1);

    auto setBrick = [=](hipfft_brick&        b,
                        const long long int* lower,
                        const long long int* upper,
                        const long long int* stride) {
        // init brick for FFT dimensions + batch dimension
        b.field_lower.resize(rank + 1);
        b.field_upper.resize(rank + 1);
        b.brick_stride.resize(rank + 1);

        // copy row-major coordinates and strides to column-major brick info
        std::reverse_iterator<const long long int*> lower_rbegin(lower + rank);
        std::reverse_iterator<const long long int*> lower_rend(lower);
        std::copy(lower_rbegin, lower_rend, b.field_lower.begin());
        std::reverse_iterator<const long long int*> upper_rbegin(upper + rank);
        std::reverse_iterator<const long long int*> upper_rend(upper);
        std::copy(upper_rbegin, upper_rend, b.field_upper.begin());
        std::reverse_iterator<const long long int*> stride_rbegin(stride + rank);
        std::reverse_iterator<const long long int*> stride_rend(stride);
        std::copy(stride_rbegin, stride_rend, b.brick_stride.begin());

        // hipFFT only supports batch-1 distributed FFTs, so set lower
        // + upper + stride for batch dimension
        b.field_lower.back()  = 0;
        b.field_upper.back()  = 1;
        b.brick_stride.back() = 0;

        (void)hipGetDevice(&b.device);
    };

    setBrick(plan->spaceBricks.front(), input_lower, input_upper, input_stride);
    setBrick(plan->freqBricks.front(), output_lower, output_upper, output_stride);
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetSubformatDefault(hipfftHandle      plan,
                                         hipfftXtSubFormat subformat_forward,
                                         hipfftXtSubFormat subformat_inverse)
try
{
    // formats must be set before plans are actually constructed
    if(!plan || plan->initialized())
        return HIPFFT_INVALID_PLAN;

    return HIPFFT_NOT_IMPLEMENTED;
}
catch(...)
{
    return handle_exception();
}

#endif
