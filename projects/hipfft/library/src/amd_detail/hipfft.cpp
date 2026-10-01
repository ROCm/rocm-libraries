// Copyright (C) 2016 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "../../../shared/client_data_layout_helpers.h"
#include "../../../shared/gpubuf.h"
#include "../../../shared/hipfft_object_wrapper.h"
#include "../../../shared/rocfft_enums_vs_fft_enums.h"
#include "../../../shared/rocfft_hip.h"

// Helper macro to check for errors: the status is thrown if not successful.
// handle_exception catches it and
// - returns it unchanged to the caller if it is a hipfftResult error code;
// - converts that to HIPFFT_INTERNAL_ERROR returned to user otherwise.
#define EXPECT_SUCCESS(CALL, SUCCESS_VALUE) \
    do                                      \
    {                                       \
        auto status = CALL;                 \
        if(status != SUCCESS_VALUE)         \
        {                                   \
            throw status;                   \
        }                                   \
    } while(0)

#define ROCFFT_EXPECT_SUCCESS(ROCFFT_CALL) EXPECT_SUCCESS(ROCFFT_CALL, rocfft_status_success)
#define HIP_EXPECT_SUCCESS(HIP_CALL) EXPECT_SUCCESS(HIP_CALL, hipSuccess)
#define HIPFFT_EXPECT_SUCCESS(HIPFFT_CALL) EXPECT_SUCCESS(HIPFFT_CALL, HIPFFT_SUCCESS)

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

struct device_context_t
{
    explicit device_context_t(int dev_id)
        : device_id(dev_id)
        , work_buffer_byte_bsize(0)
    {
        rocfft_scoped_device scoped_dev(dev_id);
        HIP_EXPECT_SUCCESS(stream.alloc_with_err());
    }

    static device_context_t make_nonowned_copy(const device_context_t& src)
    {
        device_context_t ret;
        ret.device_id              = src.device_id;
        ret.work_buffer_byte_bsize = src.work_buffer_byte_bsize;
        ret.work_buffer = gpubuf::make_nonowned(src.work_buffer.data(), src.work_buffer.size());
        ret.stream      = hipStream_wrapper_t::make_nonowned(src.stream.get_raw());
        return ret;
    }

    int                 device_id;
    size_t              work_buffer_byte_bsize;
    gpubuf              work_buffer; // may be owned or not
    hipStream_wrapper_t stream; // may be owned or not

private:
    device_context_t() = default;
};

inline rocfft_result_placement placement_from_format(const hipfftXtSubFormat& format)
{
    switch(format)
    {
    case HIPFFT_XT_FORMAT_INPLACE:
    case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        return rocfft_placement_inplace;
    case HIPFFT_XT_FORMAT_INPUT:
    case HIPFFT_XT_FORMAT_OUTPUT:
        return rocfft_placement_notinplace;
    case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
        // TODO: figure this out if ever implemented
        throw HIPFFT_NOT_IMPLEMENTED;
    case HIPFFT_FORMAT_UNDEFINED:
    default:
        throw std::invalid_argument("placement_from_format: invalid input format");
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
            throw std::invalid_argument(
                "hipfft_brick: lower, upper, and strides must be non-empty and of equal size");
        }
        axes.reserve(lower.size());
        for(size_t dim = 0; dim < lower.size(); ++dim)
            axes.push_back({lower[dim], upper[dim], strides[dim]});
    }

    // note: embedding_length is the number of elements in the data along that dimension,
    // which may be larger than dimension's span (e.g., in case of padding). For compact
    // layouts, embedding_length == span.
    size_t embedding_length(size_t dim) const
    {
        if(dim >= axes.size())
            throw std::out_of_range("hipfft_brick: embedding_length: dim out of range");
        if(dim == 0)
            return axes[0].span();

        if(!(axes[dim - 1].stride > axes[dim].stride && axes[dim - 1].stride % axes[dim].stride == 0
             && axes[dim - 1].stride / axes[dim].stride >= axes[dim].span()))
        {
            throw std::runtime_error(
                "hipfft_brick::embedding_length: invalid stride relationship between dimensions "
                + std::to_string(dim - 1) + " and " + std::to_string(dim));
        }
        // note: multiplicity of strides is guaranteed at construction
        return axes[dim - 1].stride / axes[dim].stride;
    }

    size_t data_byte_size(hipDataType data_type) const
    {
        // Not using compute_ptrdiff herein because real in-place cases
        // require the trailing padding elements
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
            throw std::logic_error(
                "hipfft_brick: this brick is not logically contained in the other brick");
        size_t offset = 0;
        return std::inner_product(
            axes.begin(),
            axes.end(),
            other.axes.begin(),
            offset,
            std::plus<size_t>(),
            [](const auto& a, const auto& b) { return (a.lower - b.lower) * b.stride; });
    }

    int get_device_id() const
    {
        return device_id;
    }
    size_t full_rank() const
    {
        return axes.size();
    }

    std::vector<size_t> get_lower() const
    {
        std::vector<size_t> lower(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
            lower[dim] = axes[dim].lower;
        return lower;
    }
    std::vector<size_t> get_upper() const
    {
        std::vector<size_t> upper(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
            upper[dim] = axes[dim].upper;
        return upper;
    }
    std::vector<size_t> get_strides() const
    {
        std::vector<size_t> strides(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
            strides[dim] = axes[dim].stride;
        return strides;
    }
    std::vector<size_t> get_spans() const
    {
        std::vector<size_t> spans(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
            spans[dim] = axes[dim].span();
        return spans;
    }

    bool is_empty() const
    {
        return std::any_of(axes.begin(), axes.end(), [](const axis_t& a) { return a.span() == 0; });
    }

    // Returns the sub-brick of *this whose logical range is common to other.
    // Retains this brick's strides and device_id; may be empty (use is_empty()).
    hipfft_brick get_overlap_with(const hipfft_brick& other) const
    {
        if(axes.size() != other.axes.size())
            throw std::invalid_argument("hipfft_brick::get_overlap_with: rank mismatch");
        std::vector<size_t> lower(axes.size()), upper(axes.size()), strides(axes.size());
        for(size_t dim = 0; dim < axes.size(); ++dim)
        {
            lower[dim]   = std::max(axes[dim].lower, other.axes[dim].lower);
            upper[dim]   = std::max(std::min(axes[dim].upper, other.axes[dim].upper), lower[dim]);
            strides[dim] = axes[dim].stride;
        }
        return hipfft_brick(lower, upper, strides, device_id);
    }

    // Strip unit-span dimensions shared by all bricks, then collapse adjacent
    // dimensions (dim, dim+1) where all bricks have identical lower/upper
    // bounds and embedding lengths at dim+1.
    template <typename... Args>
    static void collapse_matching_axes(Args&... args)
    {
        std::vector<std::reference_wrapper<hipfft_brick>> targets;
        auto                                              collect = [&targets](auto& arg) {
            using T = std::decay_t<decltype(arg)>;
            static_assert(std::is_same<T, hipfft_brick>::value
                              || std::is_same<T, std::vector<hipfft_brick>>::value,
                          "collapse_matching_axes: each argument must be hipfft_brick& "
                          "or std::vector<hipfft_brick>&");
            if constexpr(std::is_same<T, hipfft_brick>::value)
                targets.push_back(arg);
            else
                targets.insert(targets.end(), arg.begin(), arg.end());
        };
        (collect(args), ...);

        if(targets.size() < 2)
            throw std::invalid_argument(
                "collapse_matching_axes: need at least two bricks to collapse");

        if(std::any_of(targets.begin() + 1, targets.end(), [&](const hipfft_brick& brick) {
               return brick.full_rank() != targets[0].get().full_rank();
           }))
            throw std::invalid_argument("collapse_matching_axes: rank mismatch");

        for(size_t dim = 0; dim < targets[0].get().full_rank();)
        {
            if(std::all_of(targets.begin(), targets.end(), [dim](const hipfft_brick& brick) {
                   return brick.axes[dim].span() == 1;
               }))
            {
                std::for_each(targets.begin(), targets.end(), [dim](hipfft_brick& brick) {
                    brick.axes.erase(brick.axes.begin() + dim);
                });
                continue;
            }
            while(
                dim < targets[0].get().full_rank() - 1
                && std::all_of(
                    targets.begin() + 1, targets.end(), [dim, &targets](const hipfft_brick& brick) {
                        return brick.axes[dim + 1].lower == targets[0].get().axes[dim + 1].lower
                               && brick.axes[dim + 1].upper == targets[0].get().axes[dim + 1].upper
                               && brick.embedding_length(dim + 1)
                                      == targets[0].get().embedding_length(dim + 1);
                    }))
            {
                const auto multiplier = targets[0].get().embedding_length(dim + 1);
                for(auto& target : targets)
                {
                    target.get().axes[dim].stride = target.get().axes[dim + 1].stride;
                    target.get().axes[dim].lower *= multiplier;
                    target.get().axes[dim].upper *= multiplier;
                    target.get().axes.erase(target.get().axes.begin() + dim + 1);
                }
            }
            ++dim;
        }
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
    hipfft_field(fft_transform_type                   dft_type,
                 size_t                               batch_sz,
                 const std::vector<size_t>&           transform_lengths,
                 hipfftXtSubFormat                    format,
                 fft_io                               field_io_label,
                 const std::vector<device_context_t>& device_contexts)
    {
        validate_enums_or_throw("hipfft_field::hipfft_field(...)", dft_type, field_io_label);
        if(transform_lengths.empty() || batch_sz == 0
           || std::any_of(transform_lengths.begin(), transform_lengths.end(), [](const auto& l) {
                  return l == 0;
              }))
        {
            throw std::invalid_argument("Invalid rank of transform or invalid batch/length value");
        }
        const size_t ngpus = device_contexts.size();
        if(ngpus == 0)
            throw std::invalid_argument("device_contexts must be non-empty");
        if(format != HIPFFT_XT_FORMAT_INPUT && format != HIPFFT_XT_FORMAT_OUTPUT
           && format != HIPFFT_XT_FORMAT_INPLACE && format != HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
        {
            // no other formats are supported for now, so throw an invalid argument error
            throw std::invalid_argument("Invalid descriptor sub-format");
        }

        std::vector<size_t> transform_batch_and_lengths(1 + transform_lengths.size());
        transform_batch_and_lengths[0] = batch_sz;
        std::copy(transform_lengths.begin(),
                  transform_lengths.end(),
                  transform_batch_and_lengths.begin() + 1);

        const size_t split_dim
            = batch_sz > 1
                  ? 0
                  : (format == HIPFFT_XT_FORMAT_INPUT || format == HIPFFT_XT_FORMAT_INPLACE ? 1
                                                                                            : 2);

        if(split_dim >= transform_batch_and_lengths.size())
            throw std::out_of_range(
                "split_dim is out of bounds for the given transform_batch_and_lengths");
        // placement and io flag are relevant for real transforms.
        const auto placement
            = fft_result_placement_from_rocfft_result_placement(placement_from_format(format));

        const auto global_inbuffer_strides
            = default_strides(dft_type, placement, field_io_label, transform_batch_and_lengths);
        auto global_brick_spans = transform_batch_and_lengths;
        // Adjust logical spans for fields in "hermitian symmetric" domain
        if(is_real(dft_type) && (is_fwd(dft_type) == (field_io_label == fft_io_out)))
            global_brick_spans.back() = (global_brick_spans.back() / 2) + 1;

        global_brick            = hipfft_brick(std::vector<size_t>(global_brick_spans.size(), 0),
                                    global_brick_spans,
                                    global_inbuffer_strides,
                                    rocfft_scoped_device::current_device());
        const size_t num_bricks = batch_sz == 1 ? ngpus : std::min(batch_sz, ngpus);
        if(global_brick_spans[split_dim] < num_bricks)
        {
            // e.g. more devices than the length of the split dimension
            // This is not supported by hipFFT, so throw HIPFFT_NOT_SUPPORTED.
            // Note: cuFFT uses hard cutoffs of 32 for minimum lengths and
            // maximum 16 devices.
            throw HIPFFT_NOT_SUPPORTED;
        }
        for(size_t brick_idx = 0; brick_idx < num_bricks; ++brick_idx)
        {
            std::vector<size_t> brick_lower(global_brick_spans.size(), 0);
            std::vector<size_t> brick_upper(global_brick_spans);
            brick_lower[split_dim]
                = brick_idx * (global_brick_spans[split_dim] / num_bricks)
                  + std::min(brick_idx, global_brick_spans[split_dim] % num_bricks);
            brick_upper[split_dim]
                = (brick_idx + 1) * (global_brick_spans[split_dim] / num_bricks)
                  + std::min((brick_idx + 1), global_brick_spans[split_dim] % num_bricks);
            std::vector<size_t> brick_strides(global_brick_spans.size());
            for(size_t dim = brick_strides.size(); dim-- > 0;)
            {
                if(dim == brick_strides.size() - 1)
                    brick_strides[dim] = 1;
                else if(dim == brick_strides.size() - 2
                        && split_dim != global_brick_spans.size() - 1
                        && placement == fft_placement_inplace
                        && ((dft_type == fft_transform_type_real_forward
                             && field_io_label == fft_io_in)
                            || (dft_type == fft_transform_type_real_inverse
                                && field_io_label == fft_io_out)))
                {
                    brick_strides[dim] = 2 * (global_brick_spans.back() / 2 + 1);
                }
                else
                    brick_strides[dim]
                        = brick_strides[dim + 1] * (brick_upper[dim + 1] - brick_lower[dim + 1]);
            }
            bricks.emplace_back(std::move(brick_lower),
                                std::move(brick_upper),
                                std::move(brick_strides),
                                device_contexts[brick_idx].device_id);
        }
    }

#ifdef HIPFFT_MPI_ENABLE
    // make proc-local field content from a single brick (used for MPI)
    static hipfft_field make_proc_local_field(const hipfft_brick& brick)
    {
        hipfft_field ret;
        ret.bricks.emplace_back(brick);
        ret.global_brick.reset(); // unknown global field
        return ret;
    }
#endif

    void add_to(rocfft_plan_description_wrapper_t& desc, fft_io field_label)
    {
        rocfft_field_wrapper_t field_wrapper;
        ROCFFT_EXPECT_SUCCESS(field_wrapper.alloc_with_err());
        for(const auto& brick : bricks)
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

    size_t brick_count() const
    {
        return bricks.size();
    }

    const hipfft_brick& get_brick(size_t brick_idx) const
    {
        if(brick_idx >= bricks.size())
            throw std::out_of_range("hipfft_field::brick: index out of range");
        return bricks[brick_idx];
    }

    const hipfft_brick& get_global_brick() const
    {
        if(!global_brick)
            throw std::logic_error("hipfft_field::get_global_brick: global field is not known");
        return *global_brick;
    }

    static hipfft_field collapse(const hipfft_field& field_to_collapse)
    {
        if(!field_to_collapse.global_brick)
            throw std::logic_error("hipfft_field::collapse: global field is not known");
        // create a copy and collapse it, so that the original field is not modified
        auto ret = field_to_collapse;
        hipfft_brick::collapse_matching_axes(*ret.global_brick, ret.bricks);
        return ret;
    }

private:
    hipfft_field() = default;

    std::vector<hipfft_brick> bricks;
    // For library-defined decompositions, we need to know the global field's
    // upper bounds and strides, so we can compute the offsets for each brick.
    std::optional<hipfft_brick> global_brick;
};

struct hipfftIOType
{
private:
    hipDataType inputType  = HIP_C_32F;
    hipDataType outputType = HIP_C_32F;

    bool isinitialized = false;

public:
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
            throw std::runtime_error("hipfftIOType::precision: Unexpected input type");
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
            throw std::runtime_error("hipfftIOType::is_real_to_complex: Unexpected input type");
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
            throw std::runtime_error("hipfftIOType::is_complex_to_real: Unexpected output type");
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
            return io == fft_io_in ? rocfft_array_type_real
                                   : rocfft_array_type_hermitian_interleaved;
        }
        else if(is_complex_to_real())
        {
            return io == fft_io_in ? rocfft_array_type_hermitian_interleaved
                                   : rocfft_array_type_real;
        }
        else
        {
            return rocfft_array_type_complex_interleaved;
        }
    }

    hipDataType get_hip_data_type(fft_io io) const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");
        validate_or_throw(io, "hipfftIOType::get_hip_data_type");
        return io == fft_io_in ? inputType : outputType;
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

    // The key type for the exec_plans map is a variant of two types:
    // - a pair of (rocfft_transform_type, rocfft_result_placement) for single-device,
    //   multi-process, or batched single-process multi-device usage;
    // - a tuple of (rocfft_transform_type, hipfftXtSubFormat, hipfftXtSubFormat) for unbatched
    //   single-process multi-device usage (the I/O subformats define the slab decompositions);
    struct type_placement_key_t
    {
        rocfft_transform_type   dft_type;
        rocfft_result_placement placement;
        bool                    operator<(const type_placement_key_t& other) const
        {
            return std::tie(dft_type, placement) < std::tie(other.dft_type, other.placement);
        }
    };
    struct type_subformat_key_t
    {
        rocfft_transform_type dft_type;
        hipfftXtSubFormat     input_format;
        hipfftXtSubFormat     output_format;
        bool                  operator<(const type_subformat_key_t& other) const
        {
            return std::tie(dft_type, input_format, output_format)
                   < std::tie(other.dft_type, other.input_format, other.output_format);
        }
    };
    using map_key_t = std::variant<type_placement_key_t, type_subformat_key_t>;

    // Unbatched multi-device plans are keyed by I/O subformats. Extract that key,
    // or signal an internal logic error if that invariant is ever violated.
    static const type_subformat_key_t& subformat_variant_of(const map_key_t& key)
    {
        if(!std::holds_alternative<type_subformat_key_t>(key))
            throw HIPFFT_INTERNAL_ERROR;
        return std::get<type_subformat_key_t>(key);
    }
    // Other plans are keyed by placement. Extract that key, or signal an
    // internal logic error if that invariant is ever violated.
    static const type_placement_key_t& placement_variant_of(const map_key_t& key)
    {
        if(!std::holds_alternative<type_placement_key_t>(key))
            throw HIPFFT_INTERNAL_ERROR;
        return std::get<type_placement_key_t>(key);
    }

    // True if the field for I/O role `io` of the plan identified by `key` has a data decomposition
    // matching sub-format `fmt`. Placement-keyed plans (batched multi-device or single-device)
    // compare placement; sub-format-keyed plans (unbatched multi-device) compare the sub-format on
    // the side matching `io`.
    static bool key_matches_format(const map_key_t& key, hipfftXtSubFormat fmt, fft_io io)
    {
        if(std::holds_alternative<type_placement_key_t>(key))
            return placement_from_format(fmt) == placement_variant_of(key).placement;
        const auto& sk = subformat_variant_of(key);
        return fmt == (io == fft_io_in ? sk.input_format : sk.output_format);
    }

    // Return the library-defined field that stores sub-format `fmt` data for I/O role `io`.
    const hipfft_field& field_for_format(hipfftXtSubFormat fmt, fft_io io) const
    {
        const auto& fields = io == fft_io_in ? input_fields : output_fields;
        for(const auto& [key, _] : exec_plans)
        {
            if(key_matches_format(key, fmt, io))
                return fields.at(key);
        }
        throw HIPFFT_INVALID_PLAN; // no plan exposes this sub-format for this I/O role
    }

    std::map<map_key_t, rocfft_plan_wrapper_t> exec_plans;
    // Corresponding library-defined I/O fields
    std::map<map_key_t, hipfft_field> input_fields, output_fields;
#ifdef HIPFFT_MPI_ENABLE
    std::optional<hipfft_brick> mp_input_brick, mp_output_brick;
#endif

    // the same execution info is used for all rocfft plans in `exec_plans`
    rocfft_execution_info_wrapper_t info;
    std::vector<device_context_t>   device_contexts;

    struct callback_info_t
    {
        hipfftXtCallbackType type     = HIPFFT_CB_UNDEFINED;
        void**               funcptrs = nullptr;
        std::string          symbol;
        std::vector<char>    bitcode;
        void**               data      = nullptr;
        size_t               lds_bytes = 0;

        bool jit_enabled() const
        {
            return !symbol.empty() && !bitcode.empty();
        }

        // Throws HIPFFT_INVALID_VALUE if the callback type is not
        // valid for the precision and I/O
        static void validate_type(hipfftXtCallbackType cbtype, hipfftIOType io_type)
        {
            switch(cbtype)
            {
            case HIPFFT_CB_LD_COMPLEX:
                if(io_type.precision() != rocfft_precision_single || io_type.is_real_to_complex())
                    throw HIPFFT_INVALID_VALUE;
                break;
            case HIPFFT_CB_LD_COMPLEX_DOUBLE:
                if(io_type.precision() != rocfft_precision_double || io_type.is_real_to_complex())
                    throw HIPFFT_INVALID_VALUE;
                break;
            case HIPFFT_CB_LD_REAL:
                if(io_type.precision() != rocfft_precision_single || !io_type.is_real_to_complex())
                    throw HIPFFT_INVALID_VALUE;
                break;
            case HIPFFT_CB_LD_REAL_DOUBLE:
                if(io_type.precision() != rocfft_precision_double || !io_type.is_real_to_complex())
                    throw HIPFFT_INVALID_VALUE;
                break;
            case HIPFFT_CB_ST_COMPLEX:
                if(io_type.precision() != rocfft_precision_single || io_type.is_complex_to_real())
                    throw HIPFFT_INVALID_VALUE;
                break;
            case HIPFFT_CB_ST_COMPLEX_DOUBLE:
                if(io_type.precision() != rocfft_precision_double || io_type.is_complex_to_real())
                    throw HIPFFT_INVALID_VALUE;
                break;
            case HIPFFT_CB_ST_REAL:
                if(io_type.precision() != rocfft_precision_single || !io_type.is_complex_to_real())
                    throw HIPFFT_INVALID_VALUE;
                break;
            case HIPFFT_CB_ST_REAL_DOUBLE:
                if(io_type.precision() != rocfft_precision_double || !io_type.is_complex_to_real())
                    throw HIPFFT_INVALID_VALUE;
                break;
            case HIPFFT_CB_UNDEFINED:
                throw HIPFFT_INVALID_VALUE;
            }
        }
    };
    callback_info_t load_callback, store_callback;

    // Streams set before plan initialization, keyed by
    // device ID (not owned by the hipfft plan)
    std::map<int, hipStream_t> pending_streams;

    // Multi-processing communicator
    rocfft_comm_type comm_type   = rocfft_comm_none;
    void*            comm_handle = nullptr;

    bool is_valid_for(const hipLibXtDesc& lib_desc, fft_io desc_io_label) const
    {
        // Plan must be initialized for multi-device usage
        if(!initialized() || device_contexts.size() <= 1)
            return false;
        // Given descriptor comes straight from the user, so invalid
        // values thereof must be reported/escalated as such
        if(!lib_desc.descriptor)
            throw HIPFFT_INVALID_VALUE;
        const auto& field
            = field_for_format(static_cast<hipfftXtSubFormat>(lib_desc.subFormat), desc_io_label);
        const auto elem_type = io_type.get_hip_data_type(desc_io_label);
        if(static_cast<int>(field.brick_count()) > lib_desc.descriptor->nGPUs
           || field.brick_count() > device_contexts.size())
            return false;
        for(size_t brick_idx = 0; brick_idx < field.brick_count(); ++brick_idx)
        {
            const auto& brick = field.get_brick(brick_idx);
            if(brick.get_device_id() != lib_desc.descriptor->GPUs[brick_idx]
               || brick.get_device_id() != device_contexts[brick_idx].device_id)
                return false;
            if(lib_desc.descriptor->size[brick_idx] < brick.data_byte_size(elem_type))
                return false;
            if(brick.data_byte_size(elem_type) > 0 && !lib_desc.descriptor->data[brick_idx])
                return false;
        }
        return true;
    }

    template <typename TransformArgType>
    rocfft_transform_type get_transform_type_for(TransformArgType transform_arg) const
    {
        static_assert(std::is_same<TransformArgType, int>::value
                          || std::is_same<TransformArgType, rocfft_transform_type>::value,
                      "hipfftHandle_t::get_transform_type_for: TransformArgType must be either int "
                      "or rocfft_transform_type");
        if constexpr(std::is_same<TransformArgType, rocfft_transform_type>::value)
            return transform_arg;
        else
        {
            // transform_arg is an int, coming straight from the user: invalid values
            // must not be reported as internal errors
            if(transform_arg != HIPFFT_FORWARD && transform_arg != HIPFFT_BACKWARD)
                throw HIPFFT_INVALID_VALUE;
            if(io_type.is_real_to_complex())
            {
                if(transform_arg != HIPFFT_FORWARD)
                    throw HIPFFT_INVALID_PLAN;
                return rocfft_transform_type_real_forward;
            }
            else if(io_type.is_complex_to_real())
            {
                if(transform_arg != HIPFFT_BACKWARD)
                    throw HIPFFT_INVALID_PLAN;
                return rocfft_transform_type_real_inverse;
            }
            // C2C case
            return transform_arg == HIPFFT_FORWARD ? rocfft_transform_type_complex_forward
                                                   : rocfft_transform_type_complex_inverse;
        }
    }

    bool can_execute(rocfft_transform_type                  transform_type,
                     const std::optional<rocfft_precision>& execution_precision
                     = std::nullopt) const
    {
        if(!initialized())
            return false;
        if(execution_precision && io_type.precision() != *execution_precision)
            return false;
        // Validate that the requested transform type is compatible with the plan's io_type
        switch(transform_type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
        {
            if(!io_type.is_complex_to_complex())
                return false;
        }
        break;
        case rocfft_transform_type_real_inverse:
        {
            if(!io_type.is_complex_to_real())
                return false;
        }
        break;
        case rocfft_transform_type_real_forward:
        {
            if(!io_type.is_real_to_complex())
                return false;
        }
        break;
        default:
            // This would be an internal error, not a user error
            throw std::invalid_argument("hipfftHandle_t::can_execute: invalid transform_type");
        }

        return true;
    }

    // requires batch, transform_lengths, io_type, device_contexts to be set prior
    // in case of single-process multi-device usage.
    std::set<map_key_t> possible_exec_map_key() const
    {
        if(device_contexts.empty())
            throw std::invalid_argument(
                "hipfftHandle_t::possible_exec_map_key: device_contexts must be non-empty");
        std::set<map_key_t> ret;
        for(auto dft_type : io_type.transform_types())
        {
            if(device_contexts.size() > 1)
            {
#ifdef HIPFFT_MPI_ENABLE
                if(mp_input_brick || mp_output_brick)
                {
                    // hipfftHandle_t::possible_exec_map_key: multi-device usage with
                    // multi-processing bricks is not supported
                    throw HIPFFT_NOT_SUPPORTED;
                }
#endif
                if(batch > 1)
                {
                    // no more than one plan per transform type + placement is supported for
                    // multi-device batched transforms, so type_placement_key_t is the right
                    // key type to use in this case.
                    ret.insert(type_placement_key_t{dft_type, rocfft_placement_inplace});
                    ret.insert(type_placement_key_t{dft_type, rocfft_placement_notinplace});
                }
                else
                {
                    if(transform_lengths.size() == 1)
                        throw HIPFFT_NOT_IMPLEMENTED;
                    // INPLACE -> INPLACE_SHUFFLED for all 3D, and all 2D except 2D real inverse
                    // (splitting innermost dimension in real domain is not supported)
                    if(transform_lengths.size() > 2 || !io_type.is_complex_to_real())
                        ret.insert(type_subformat_key_t{
                            dft_type, HIPFFT_XT_FORMAT_INPLACE, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED});
                    // INPLACE_SHUFFLED -> INPLACE for all 3D, and all 2D except 2D real forward
                    // (splitting innermost dimension in real domain is not supported)
                    if(transform_lengths.size() > 2 || !io_type.is_real_to_complex())
                        ret.insert(type_subformat_key_t{
                            dft_type, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED, HIPFFT_XT_FORMAT_INPLACE});
                }
            }
            else
            {
                ret.insert(type_placement_key_t{dft_type, rocfft_placement_inplace});
                ret.insert(type_placement_key_t{dft_type, rocfft_placement_notinplace});
            }
        }
        return ret;
    }

    // Create a new handle copying all pre-initialization settings from src
    // (shallow copies), except for the auto-allocation flag
    // (auto_allocation is disabled in the returned plan).
    static hipfftHandle_wrapper_t make_size_querying_plan_from(const hipfftHandle_t* src)
    {
        hipfftHandle_wrapper_t ret;
        HIPFFT_EXPECT_SUCCESS(ret.alloc_with_err());
        (*ret).auto_allocate = false;
        if(!src)
            return ret;

        for(const auto& ctx : src->device_contexts)
        {
            (*ret).device_contexts.emplace_back(device_context_t::make_nonowned_copy(ctx));
        }
        (*ret).load_callback  = src->load_callback;
        (*ret).store_callback = src->store_callback;
        (*ret).scale_factor   = src->scale_factor;
        (*ret).comm_type      = src->comm_type;
        (*ret).comm_handle    = src->comm_handle;
#ifdef HIPFFT_MPI_ENABLE
        (*ret).mp_input_brick  = src->mp_input_brick;
        (*ret).mp_output_brick = src->mp_output_brick;
#endif
        return ret;
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
    HIPFFT_EXPECT_SUCCESS(hipfftCreate(&handle));
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
    HIPFFT_EXPECT_SUCCESS(hipfftCreate(&handle));
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
    HIPFFT_EXPECT_SUCCESS(hipfftCreate(&handle));
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
    HIPFFT_EXPECT_SUCCESS(hipfftCreate(&handle));
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

    // degenerate lengths and batch counts hang the planner
    if(std::any_of(rm_lengths.begin(), rm_lengths.end(), [](const auto& l) { return l == 0; })
       || number_of_transforms == 0)
    {
        return HIPFFT_INVALID_SIZE;
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
    static rocfft_initializer init;

    plan->io_type = iotype;
    if(plan->device_contexts.size() > 1)
    {
        // We currently do not support unbatched 1D multi-device transforms.
        if(rm_lengths.size() == 1 && number_of_transforms == 1)
            return HIPFFT_NOT_IMPLEMENTED;
    }
    plan->batch             = number_of_transforms;
    plan->transform_lengths = rm_lengths;
    // copy the user's ionembed into the plan if there is one, use default otherwise
    plan->global_ionembed = !user_ionembed ? hipfft_ionembed_t<size_t>() : *user_ionembed;

    if(plan->device_contexts.empty())
    {
        // not multi-device, so use the current device as the default
        plan->device_contexts.emplace_back(rocfft_scoped_device::current_device());
    }
#ifdef HIPFFT_MPI_ENABLE
    if(plan->mp_input_brick && plan->mp_output_brick)
    {
        // Multi-process usage requires one device per process.
        if(plan->device_contexts.size() > 1)
            return HIPFFT_NOT_SUPPORTED;
        // only for unbatched transforms
        if(plan->batch != 1)
            return HIPFFT_NOT_SUPPORTED;
    }
#endif

    const std::vector<size_t> cm_lengths_vec(plan->transform_lengths.rbegin(),
                                             plan->transform_lengths.rend());
    // NOTE: hipFFT ignores distance arguments if default layouts are used!
    const bool ignore_user_distances = !plan->global_ionembed.get_nembed(fft_io_in)
                                       && !plan->global_ionembed.get_nembed(fft_io_out);

    for(const auto& map_key : plan->possible_exec_map_key())
    {
        const auto dft_type  = std::visit([](const auto& key) { return key.dft_type; }, map_key);
        const auto placement = std::visit(
            [](const auto& key) {
                if constexpr(std::is_same<std::decay_t<decltype(key)>,
                                          hipfftHandle_t::type_placement_key_t>::value)
                    return key.placement;
                else
                    return placement_from_format(key.input_format);
            },
            map_key);

        rocfft_plan_description_wrapper_t desc;

        ROCFFT_EXPECT_SUCCESS(desc.alloc_with_err());

        auto i_strides = plan->global_ionembed.as_generalized_strides(
            fft_io_in,
            fft_transform_type_from_rocfft_transform_type(dft_type),
            fft_result_placement_from_rocfft_result_placement(placement),
            plan->transform_lengths);
        auto o_strides = plan->global_ionembed.as_generalized_strides(
            fft_io_out,
            fft_transform_type_from_rocfft_transform_type(dft_type),
            fft_result_placement_from_rocfft_result_placement(placement),
            plan->transform_lengths);

        // rm -> cm:
        std::reverse(i_strides.begin(), i_strides.end());
        std::reverse(o_strides.begin(), o_strides.end());
        const auto inDist
            = !ignore_user_distances
                  ? user_idist
                  : default_distance(fft_transform_type_from_rocfft_transform_type(dft_type),
                                     fft_result_placement_from_rocfft_result_placement(placement),
                                     fft_io_in,
                                     plan->transform_lengths,
                                     number_of_transforms);
        const auto outDist
            = !ignore_user_distances
                  ? user_odist
                  : default_distance(fft_transform_type_from_rocfft_transform_type(dft_type),
                                     fft_result_placement_from_rocfft_result_placement(placement),
                                     fft_io_out,
                                     plan->transform_lengths,
                                     number_of_transforms);

        ROCFFT_EXPECT_SUCCESS(rocfft_plan_description_set_data_layout(desc,
                                                                      iotype.array_type(fft_io_in),
                                                                      iotype.array_type(fft_io_out),
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

        // set JIT callbacks if specified

        // JIT callbacks cannot currently be combined with multi-GPU
        // transforms.
        const bool is_multi_gpu_transform = plan->device_contexts.size() > 1
#ifdef HIPFFT_MPI_ENABLE
                                            || plan->mp_input_brick || plan->mp_output_brick
#endif
            ;
        if((plan->load_callback.jit_enabled() || plan->store_callback.jit_enabled())
           && is_multi_gpu_transform)
            return HIPFFT_NOT_IMPLEMENTED;
        if(plan->load_callback.jit_enabled())
        {
            hipfftHandle_t::callback_info_t::validate_type(plan->load_callback.type, plan->io_type);
            ROCFFT_EXPECT_SUCCESS(
                rocfft_plan_description_set_load_callback(desc,
                                                          plan->load_callback.symbol.c_str(),
                                                          plan->load_callback.bitcode.data(),
                                                          plan->load_callback.bitcode.size(),
                                                          plan->load_callback.lds_bytes));
        }
        if(plan->store_callback.jit_enabled())
        {
            hipfftHandle_t::callback_info_t::validate_type(plan->store_callback.type,
                                                           plan->io_type);
            ROCFFT_EXPECT_SUCCESS(
                rocfft_plan_description_set_store_callback(desc,
                                                           plan->store_callback.symbol.c_str(),
                                                           plan->store_callback.bitcode.data(),
                                                           plan->store_callback.bitcode.size(),
                                                           plan->store_callback.lds_bytes));
        }

        if(plan->device_contexts.size() > 1)
        {
            // Determine the sub-formats of this rocfft plan's input and output fields. Unbatched
            // multi-device plans are keyed by explicit sub-formats; batched plans are keyed by
            // placement, so the canonical out-of-place (INPUT/OUTPUT) or in-place (INPLACE)
            // sub-formats are used (the batch split is sub-format independent).

            for(auto io : {fft_io_in, fft_io_out})
            {
                const hipfftXtSubFormat field_fmt
                    = std::holds_alternative<hipfftHandle_t::type_subformat_key_t>(map_key)
                          ? (io == fft_io_in
                                 ? hipfftHandle_t::subformat_variant_of(map_key).input_format
                                 : hipfftHandle_t::subformat_variant_of(map_key).output_format)
                          : (placement == rocfft_placement_inplace
                                 ? HIPFFT_XT_FORMAT_INPLACE
                                 : (io == fft_io_in ? HIPFFT_XT_FORMAT_INPUT
                                                    : HIPFFT_XT_FORMAT_OUTPUT));
                auto& plan_fields = io == fft_io_in ? plan->input_fields : plan->output_fields;
                auto  it          = plan_fields.find(map_key);
                if(it == plan_fields.end())
                {
                    it = plan_fields
                             .emplace(map_key,
                                      hipfft_field(
                                          fft_transform_type_from_rocfft_transform_type(dft_type),
                                          number_of_transforms,
                                          rm_lengths,
                                          field_fmt,
                                          io,
                                          plan->device_contexts))
                             .first;
                }
                it->second.add_to(desc, io);
            }
        }
#ifdef HIPFFT_MPI_ENABLE
        if(plan->mp_input_brick && plan->mp_output_brick)
        {
            hipfft_field::make_proc_local_field(*(plan->mp_input_brick))
                .add_to(desc, fft_io::fft_io_in);
            hipfft_field::make_proc_local_field(*(plan->mp_output_brick))
                .add_to(desc, fft_io::fft_io_out);
        }
#endif
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
        plan->exec_plans.emplace(map_key, std::move(rocfft_plan));
    }

    // If no plans got created or any map entry is null, fail
    if(plan->exec_plans.empty()
       || std::any_of(plan->exec_plans.begin(), plan->exec_plans.end(), [](const auto& p) {
              return !p.second;
          }))
    {
        return HIPFFT_PARSE_ERROR;
    }

    // Initialize device-specific execution info parameters for each device in the plan:
    // - a stream is allocated for each device
    // - the required work buffer size is determined
    // - work buffers are allocated if auto_allocate is true
    for(size_t idx = 0; idx < plan->device_contexts.size(); ++idx)
    {
        auto&                dev_info = plan->device_contexts[idx];
        rocfft_scoped_device scoped_dev(dev_info.device_id);
        std::for_each(plan->exec_plans.begin(), plan->exec_plans.end(), [&](const auto& p) {
            size_t tmp = 0;
            ROCFFT_EXPECT_SUCCESS(rocfft_plan_get_work_buffer_size(p.second, &tmp));
            dev_info.work_buffer_byte_bsize = std::max(dev_info.work_buffer_byte_bsize, tmp);
        });
        if(workSize != nullptr)
            workSize[idx] = dev_info.work_buffer_byte_bsize;
        if(plan->auto_allocate && dev_info.work_buffer_byte_bsize > 0)
        {
            if(dev_info.work_buffer.alloc(dev_info.work_buffer_byte_bsize) != hipSuccess)
                return HIPFFT_ALLOC_FAILED;
            ROCFFT_EXPECT_SUCCESS(rocfft_execution_info_set_work_buffer(
                plan->info, dev_info.work_buffer.data(), dev_info.work_buffer_byte_bsize));
        }
    }

    // if JIT callbacks are used, pass the cbdata to the execution info
    if(plan->load_callback.jit_enabled() && plan->load_callback.data)
    {
        ROCFFT_EXPECT_SUCCESS(rocfft_execution_info_set_load_callback_data(
            plan->info, plan->load_callback.data, plan->device_contexts.size()));
    }
    if(plan->store_callback.jit_enabled() && plan->store_callback.data)
    {
        ROCFFT_EXPECT_SUCCESS(rocfft_execution_info_set_store_callback_data(
            plan->info, plan->store_callback.data, plan->device_contexts.size()));
    }
    // Apply streams that were set before plan initialization, if any
    for(const auto& [dev_id, stream] : plan->pending_streams)
    {
        // Note: the plan is now initialized, so the following call is
        // actually applying the stream to the plan's execution info and
        // the device contexts (e.g. for hipfftXtMemcpy)
        HIPFFT_EXPECT_SUCCESS(hipfftSetStream(plan, stream));
    }
    plan->pending_streams.clear();

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
    if(!plan || plan->initialized())
        return HIPFFT_INVALID_PLAN;
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
    HIPFFT_EXPECT_SUCCESS(iotype.init(type));

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
    HIPFFT_EXPECT_SUCCESS(iotype.init(type));

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
    HIPFFT_EXPECT_SUCCESS(iotype.init(type));

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
    HIPFFT_EXPECT_SUCCESS(iotype.init(type));

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
    HIPFFT_EXPECT_SUCCESS(iotype.init(type));

    return hipfftMakePlanMany_internal<long long int>(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, iotype, batch, workSize);
}
catch(...)
{
    return handle_exception();
}

// no try-catch needed: hipfftGetSize* catches all exceptions internally
hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t* workSize)
{
    return hipfftGetSize1d(nullptr, nx, type, batch, workSize);
}

// no try-catch needed: hipfftGetSize* catches all exceptions internally
hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize)
{
    return hipfftGetSize2d(nullptr, nx, ny, type, workSize);
}

// no try-catch needed: hipfftGetSize* catches all exceptions internally
hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    return hipfftGetSize3d(nullptr, nx, ny, nz, type, workSize);
}

// no try-catch needed: hipfftGetSize* catches all exceptions internally
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
{
    return hipfftGetSizeMany(
        nullptr, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
}

hipfftResult
    hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
try
{
    if(!workSize)
        return HIPFFT_INVALID_VALUE;

    auto temp = hipfftHandle_t::make_size_querying_plan_from(plan);
    HIPFFT_EXPECT_SUCCESS(hipfftMakePlan1d(temp, nx, type, batch, workSize));
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

    auto temp = hipfftHandle_t::make_size_querying_plan_from(plan);
    HIPFFT_EXPECT_SUCCESS(hipfftMakePlan2d(temp, nx, ny, type, workSize));
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

    auto temp = hipfftHandle_t::make_size_querying_plan_from(plan);
    HIPFFT_EXPECT_SUCCESS(hipfftMakePlan3d(temp, nx, ny, nz, type, workSize));
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

    auto temp = hipfftHandle_t::make_size_querying_plan_from(plan);
    HIPFFT_EXPECT_SUCCESS(hipfftMakePlanMany(
        temp, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize));
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

    auto temp = hipfftHandle_t::make_size_querying_plan_from(plan);
    HIPFFT_EXPECT_SUCCESS(hipfftMakePlanMany64(
        temp, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize));
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

    for(size_t idx = 0; idx < plan->device_contexts.size(); ++idx)
    {
        workSize[idx] = plan->device_contexts[idx].work_buffer_byte_bsize;
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
    if(!plan || !plan->initialized() || plan->device_contexts.size() != 1)
        return HIPFFT_INVALID_PLAN;
    // work delegated to generalized version to avoid duplications
    return hipfftXtSetWorkArea(plan, &workArea);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetWorkArea(hipfftHandle plan, void** workArea)
try
{
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;
    if(!workArea)
        return HIPFFT_INVALID_VALUE;
    for(size_t idx = 0; idx < plan->device_contexts.size(); ++idx)
    {
        auto& dev_info = plan->device_contexts[idx];
        if(dev_info.work_buffer_byte_bsize == 0)
            continue;
        if(!workArea[idx])
            return HIPFFT_INVALID_VALUE;
        rocfft_scoped_device dev(dev_info.device_id);
        dev_info.work_buffer
            = gpubuf::make_nonowned(workArea[idx], dev_info.work_buffer_byte_bsize);
        ROCFFT_EXPECT_SUCCESS(rocfft_execution_info_set_work_buffer(
            plan->info, dev_info.work_buffer.data(), dev_info.work_buffer_byte_bsize));
    }
    plan->auto_allocate = false;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

// Execute an FFT on a single-device plan.
// TransformArgType may be either rocfft_transform_type or int (direction).
// Prefer passing rocfft_transform_type when the caller knows the exact transform
// kind (e.g., hipfftExecR2C passes rocfft_transform_type_real_forward): this
// enables stronger validation since the transform type is checked against the
// plan's io_type without ambiguity. The int overload exists for untyped APIs
// (hipfftXtExec) where direction is all the caller provides; in that case,
// get_transform_type_for derives the transform type from plan->io_type and
// additionally validates that the direction is compatible with the plan.
template <typename TransformArgType>
static hipfftResult hipfftExecBase(const hipfftHandle_t*                 plan,
                                   void*                                 input,
                                   void*                                 output,
                                   TransformArgType                      transform_arg,
                                   const std::optional<rocfft_precision> precision
                                   = std::nullopt) noexcept
try
{
    static_assert(std::is_same<TransformArgType, rocfft_transform_type>::value
                      || std::is_same<TransformArgType, int>::value,
                  "hipfftExecBase: transform_arg must be either rocfft_transform_type or int");
    if(!plan || !plan->initialized() || plan->device_contexts.size() > 1)
        return HIPFFT_INVALID_PLAN;
    const auto dft_type = plan->get_transform_type_for(transform_arg);
    if(!plan->can_execute(dft_type, precision))
        return HIPFFT_INVALID_PLAN;
    if(!input || !output)
        return HIPFFT_INVALID_VALUE;

    const auto it = plan->exec_plans.find(hipfftHandle_t::type_placement_key_t{
        dft_type, input == output ? rocfft_placement_inplace : rocfft_placement_notinplace});
    if(it == plan->exec_plans.end())
        throw HIPFFT_INVALID_PLAN;
    const auto& rplan  = it->second;
    void*       in[1]  = {input};
    void*       out[1] = {output};
    const auto  ret    = rocfft_execute(rplan, in, out, plan->info);
    return ret == rocfft_status_success ? HIPFFT_SUCCESS : HIPFFT_EXEC_FAILED;
}
catch(...)
{
    return handle_exception();
}

hipfftResult
    hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
{
    if(direction != HIPFFT_FORWARD && direction != HIPFFT_BACKWARD)
        return HIPFFT_INVALID_VALUE;
    return hipfftExecBase(plan,
                          idata,
                          odata,
                          direction == HIPFFT_FORWARD ? rocfft_transform_type_complex_forward
                                                      : rocfft_transform_type_complex_inverse,
                          rocfft_precision_single);
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
{
    return hipfftExecBase(
        plan, idata, odata, rocfft_transform_type_real_forward, rocfft_precision_single);
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
{
    return hipfftExecBase(
        plan, idata, odata, rocfft_transform_type_real_inverse, rocfft_precision_single);
}

hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                           hipfftDoubleComplex* idata,
                           hipfftDoubleComplex* odata,
                           int                  direction)
{
    if(direction != HIPFFT_FORWARD && direction != HIPFFT_BACKWARD)
        return HIPFFT_INVALID_VALUE;
    return hipfftExecBase(plan,
                          idata,
                          odata,
                          direction == HIPFFT_FORWARD ? rocfft_transform_type_complex_forward
                                                      : rocfft_transform_type_complex_inverse,
                          rocfft_precision_double);
}

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* idata, hipfftDoubleComplex* odata)
{
    return hipfftExecBase(
        plan, idata, odata, rocfft_transform_type_real_forward, rocfft_precision_double);
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
{
    return hipfftExecBase(
        plan, idata, odata, rocfft_transform_type_real_inverse, rocfft_precision_double);
}

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
try
{
    if(!plan)
        return HIPFFT_INVALID_PLAN;

    auto stream_dev_id = hipInvalidDeviceId;
    HIP_EXPECT_SUCCESS(hipStreamGetDevice(stream, &stream_dev_id));
    if(stream_dev_id == hipInvalidDeviceId)
        return HIPFFT_INTERNAL_ERROR;
    if(!plan->initialized())
    {
        plan->pending_streams[stream_dev_id] = stream;
        return HIPFFT_SUCCESS;
    }
    // plan is initialized: attach the stream to all relevant device contexts
    // Note: the given stream is irrelevant if attached to a device that is
    // _not_ used by the plan
    for(auto& dev_info : plan->device_contexts)
    {
        if(dev_info.device_id != stream_dev_id)
            continue;
        dev_info.stream = hipStream_wrapper_t::make_nonowned(stream);
    }
    ROCFFT_EXPECT_SUCCESS(rocfft_execution_info_set_stream(plan->info, stream));
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
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;

    hipfftHandle_t::callback_info_t::validate_type(cbtype, plan->io_type);

    // check that the input/output type matches what's being requested
    //
    // NOTE: cufft explicitly does not save shared memory bytes when
    // you set a new callback, so zero out our number when setting
    // pointers
    switch(cbtype)
    {
    case HIPFFT_CB_LD_COMPLEX:
        plan->load_callback.type      = cbtype;
        plan->load_callback.funcptrs  = callbacks;
        plan->load_callback.data      = callbackData;
        plan->load_callback.lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
        plan->load_callback.type      = cbtype;
        plan->load_callback.funcptrs  = callbacks;
        plan->load_callback.data      = callbackData;
        plan->load_callback.lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_REAL:
        plan->load_callback.type      = cbtype;
        plan->load_callback.funcptrs  = callbacks;
        plan->load_callback.data      = callbackData;
        plan->load_callback.lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_REAL_DOUBLE:
        plan->load_callback.type      = cbtype;
        plan->load_callback.funcptrs  = callbacks;
        plan->load_callback.data      = callbackData;
        plan->load_callback.lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_COMPLEX:
        plan->store_callback.type      = cbtype;
        plan->store_callback.funcptrs  = callbacks;
        plan->store_callback.data      = callbackData;
        plan->store_callback.lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
        plan->store_callback.type      = cbtype;
        plan->store_callback.funcptrs  = callbacks;
        plan->store_callback.data      = callbackData;
        plan->store_callback.lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_REAL:
        plan->store_callback.type      = cbtype;
        plan->store_callback.funcptrs  = callbacks;
        plan->store_callback.data      = callbackData;
        plan->store_callback.lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_REAL_DOUBLE:
        plan->store_callback.type      = cbtype;
        plan->store_callback.funcptrs  = callbacks;
        plan->store_callback.data      = callbackData;
        plan->store_callback.lds_bytes = 0;
        break;
    case HIPFFT_CB_UNDEFINED:
        return HIPFFT_INVALID_VALUE;
    }

    rocfft_status res;
    res = rocfft_execution_info_set_load_callback(plan->info,
                                                  plan->load_callback.funcptrs,
                                                  plan->load_callback.data,
                                                  plan->load_callback.lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    res = rocfft_execution_info_set_store_callback(plan->info,
                                                   plan->store_callback.funcptrs,
                                                   plan->store_callback.data,
                                                   plan->store_callback.lds_bytes);
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
        plan->load_callback.lds_bytes = sharedSize;
        break;
    case HIPFFT_CB_ST_COMPLEX:
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
    case HIPFFT_CB_ST_REAL:
    case HIPFFT_CB_ST_REAL_DOUBLE:
        plan->store_callback.lds_bytes = sharedSize;
        break;
    case HIPFFT_CB_UNDEFINED:
        return HIPFFT_INVALID_VALUE;
    }

    rocfft_status res;
    res = rocfft_execution_info_set_load_callback(plan->info,
                                                  plan->load_callback.funcptrs,
                                                  plan->load_callback.data,
                                                  plan->load_callback.lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    res = rocfft_execution_info_set_store_callback(plan->info,
                                                   plan->store_callback.funcptrs,
                                                   plan->store_callback.data,
                                                   plan->store_callback.lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtSetJITCallback(hipfftHandle         plan,
                                    const char*          symbol_name,
                                    const void*          bitcode_data,
                                    size_t               bitcode_len_bytes,
                                    hipfftXtCallbackType cbtype,
                                    void**               cbdata)
try
{
    if(!plan || plan->initialized())
        return HIPFFT_INVALID_PLAN;

    // Return an error if we're neither clearing callbacks nor setting
    // symbol + bitcode together.
    if(!(!symbol_name && !bitcode_data && !bitcode_len_bytes)
       && !(symbol_name && bitcode_data && bitcode_len_bytes))
        return HIPFFT_INVALID_VALUE;

    switch(cbtype)
    {
    case HIPFFT_CB_LD_COMPLEX:
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
    case HIPFFT_CB_LD_REAL:
    case HIPFFT_CB_LD_REAL_DOUBLE:
    {
        plan->load_callback.type = cbtype;
        if(symbol_name)
            plan->load_callback.symbol = symbol_name;
        else
            plan->load_callback.symbol.clear();
        plan->load_callback.bitcode.assign(static_cast<const char*>(bitcode_data),
                                           static_cast<const char*>(bitcode_data)
                                               + bitcode_len_bytes);
        plan->load_callback.data = cbdata;
        break;
    }
    case HIPFFT_CB_ST_COMPLEX:
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
    case HIPFFT_CB_ST_REAL:
    case HIPFFT_CB_ST_REAL_DOUBLE:
    {
        plan->store_callback.type = cbtype;
        if(symbol_name)
            plan->store_callback.symbol = symbol_name;
        else
            plan->store_callback.symbol.clear();
        plan->store_callback.bitcode.assign(static_cast<const char*>(bitcode_data),
                                            static_cast<const char*>(bitcode_data)
                                                + bitcode_len_bytes);
        plan->store_callback.data = cbdata;
        break;
    }
    case HIPFFT_CB_UNDEFINED:
        return HIPFFT_INVALID_VALUE;
    }
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
    HIPFFT_EXPECT_SUCCESS(iotype.init(inputtype, outputtype, executiontype));
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
    if(!workSize)
        return HIPFFT_INVALID_VALUE;
    hipfftIOType iotype;
    HIPFFT_EXPECT_SUCCESS(iotype.init(inputtype, outputtype, executiontype));
    auto temp = hipfftHandle_t::make_size_querying_plan_from(plan);
    HIPFFT_EXPECT_SUCCESS(hipfftMakePlanMany_internal(
        temp, rank, n, inembed, istride, idist, onembed, ostride, odist, iotype, batch, workSize));
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExec(hipfftHandle plan, void* input, void* output, int direction)
{
    return hipfftExecBase(plan, input, output, direction);
}

hipfftResult hipfftXtSetGPUs(hipfftHandle plan, int count, int* gpus)
try
{
    if(count <= 0 || count > MAX_HIP_DESCRIPTOR_GPUS || !gpus)
        return HIPFFT_INVALID_VALUE;
    if(!plan || plan->initialized())
        return HIPFFT_INVALID_PLAN;
    const auto dev_count = rocfft_scoped_device::device_count();
    if(dev_count <= 0)
        return HIPFFT_INTERNAL_ERROR;
    if(std::any_of(
           gpus, gpus + count, [=](int gpu_id) { return gpu_id < 0 || gpu_id >= dev_count; }))
        return HIPFFT_INVALID_VALUE;
    // TODO: verify that duplicated device IDs are robustly handled and remove the following check if so
    std::set<int> unique_gpus(gpus, gpus + count);
    if(static_cast<int>(unique_gpus.size()) != count)
        return HIPFFT_INVALID_VALUE;
    plan->device_contexts.clear();
    for(int i = 0; i < count; ++i)
        plan->device_contexts.emplace_back(gpus[i]);

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
    if(plan->device_contexts.size() == 1)
    {
#ifdef HIPFFT_MPI_ENABLE
        if(plan->comm_type != rocfft_comm_none)
            return HIPFFT_NOT_IMPLEMENTED;
#endif
        return HIPFFT_INVALID_PLAN;
    }
    if(!desc)
        return HIPFFT_INVALID_VALUE;

    // unbatched 1D transforms are not supported yet
    if(plan->transform_lengths.size() == 1 && plan->batch == 1)
        return HIPFFT_NOT_IMPLEMENTED;
    if(format == HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED)
        return HIPFFT_NOT_IMPLEMENTED;
    // No other value than the following can possibly be accepted for other cases
    if(format != HIPFFT_XT_FORMAT_INPLACE && format != HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
       && format != HIPFFT_XT_FORMAT_INPUT && format != HIPFFT_XT_FORMAT_OUTPUT)
    {
        return HIPFFT_INVALID_VALUE;
    }
    // batched cases accept everything except HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
    if(plan->batch > 1 && format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
        return HIPFFT_NOT_SUPPORTED;
    // unbatched cases accept only formats destined to in-place usage
    if(plan->batch == 1 && placement_from_format(format) != rocfft_placement_inplace)
        return HIPFFT_NOT_SUPPORTED;

    // The requested descriptor format must match (one of) the plan's expected
    // input format(s), unless explicitly requesting HIPFFT_XT_FORMAT_OUTPUT.
    const auto desc_io_label = format == HIPFFT_XT_FORMAT_OUTPUT ? fft_io_out : fft_io_in;

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
    xt_desc->nGPUs   = static_cast<decltype(xt_desc->nGPUs)>(plan->device_contexts.size());
    for(size_t brick_idx = 0; brick_idx < MAX_HIP_DESCRIPTOR_GPUS; ++brick_idx)
    {
        // do not allow possible misinterpretation of "0" as a valid device
        if(brick_idx >= plan->device_contexts.size())
        {
            xt_desc->GPUs[brick_idx] = hipInvalidDeviceId;
            continue;
        }
        xt_desc->GPUs[brick_idx] = plan->device_contexts[brick_idx].device_id;
        xt_desc->size[brick_idx] = 0;
        for(const auto& [key, _] : plan->exec_plans)
        {
            if(!hipfftHandle_t::key_matches_format(key, format, desc_io_label))
                continue; // requested format not compatible with this item
            for(auto io : {fft_io_in, fft_io_out})
            {
                // Descriptors for in-place transforms must fit both of the plan's
                // input and output fields' data
                if(placement_from_format(format) == rocfft_placement_notinplace
                   && desc_io_label != io)
                {
                    continue;
                }
                const auto& field
                    = io == fft_io_in ? plan->input_fields.at(key) : plan->output_fields.at(key);
                const auto elem_type = plan->io_type.get_hip_data_type(io);
                if(brick_idx >= field.brick_count())
                    continue;
                xt_desc->size[brick_idx] = std::max(
                    xt_desc->size[brick_idx], field.get_brick(brick_idx).data_byte_size(elem_type));
            }
        }
        if(xt_desc->size[brick_idx] > 0)
        {
            rocfft_scoped_device dev(plan->device_contexts[brick_idx].device_id);
            if(hipMalloc(&(xt_desc->data[brick_idx]), xt_desc->size[brick_idx]) != hipSuccess)
                return HIPFFT_ALLOC_FAILED;
        }
    }
    if(std::all_of(xt_desc->size, xt_desc->size + xt_desc->nGPUs, [](size_t sz) { return sz == 0; })
       && std::all_of(
           xt_desc->data, xt_desc->data + xt_desc->nGPUs, [](void* ptr) { return ptr == nullptr; }))
    {
        // cufft actually returns HIPFFT_SUCCESS for use cases reaching this
        // point, but that seems like a bug: The descriptor simply cannot be
        // used if no data chunk is allocated.
        return HIPFFT_NOT_SUPPORTED;
    }
    *desc = lib_desc.release();
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

struct p2p_enabler
{
    p2p_enabler(std::pair<int, int> src_and_peer_id)
        : src_id(src_and_peer_id.first)
        , peer_id(src_and_peer_id.second)
    {
        if(peer_id < 0 || src_id < 0 || peer_id >= rocfft_scoped_device::device_count()
           || src_id >= rocfft_scoped_device::device_count())
            throw std::out_of_range("p2p_enabler: a given device ID is out of range");
        if(peer_id == src_id)
            return;
        rocfft_scoped_device scoped_dev(src_id);
        int                  can_access = 0;
        HIP_EXPECT_SUCCESS(hipDeviceCanAccessPeer(&can_access, src_id, peer_id));
        if(can_access == 1)
        {
            const auto err = hipDeviceEnablePeerAccess(peer_id, 0);
            if(err == hipSuccess)
                enabled_by_us = true;
            else if(err != hipErrorPeerAccessAlreadyEnabled)
                HIP_EXPECT_SUCCESS(err);
        }
        else
        {
            // peer access not supported between the two devices
            throw HIPFFT_INTERNAL_ERROR;
        }
    }
    ~p2p_enabler()
    try
    {
        if(enabled_by_us && peer_id != src_id)
        {
            rocfft_scoped_device scoped_dev(src_id);
            (void)hipDeviceDisablePeerAccess(peer_id);
        }
    }
    catch(...)
    {
    }

    p2p_enabler(const p2p_enabler&) = delete;
    p2p_enabler& operator=(const p2p_enabler&) = delete;
    p2p_enabler& operator=(p2p_enabler&&) = delete;

private:
    bool      enabled_by_us = false;
    const int src_id        = hipInvalidDeviceId;
    const int peer_id       = hipInvalidDeviceId;
};

hipfftResult hipfftXtMemcpy(hipfftHandle plan, void* dest, void* src, hipfftXtCopyType cptype)
try
{
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;
    if(!dest || !src || dest == src)
        return HIPFFT_INVALID_VALUE;

    // Expected behavior is significantly different for multi-process plans.
    if(plan->comm_type == rocfft_comm_none)
    {
        // must be multi-device
        if(plan->device_contexts.size() < 2)
            return HIPFFT_INVALID_PLAN;
        // no implementation yet for unbatched 1D
        if(plan->batch == 1 && plan->transform_lengths.size() == 1)
            return HIPFFT_NOT_IMPLEMENTED;
        // P2P enablers for D2D copies between devices (enablers mustn't
        // be destructed before final synchronizations).
        std::map<std::pair<int, int>, p2p_enabler> p2p_enablers;
        switch(cptype)
        {
        case HIPFFT_COPY_HOST_TO_DEVICE:
        case HIPFFT_COPY_DEVICE_TO_HOST:
        {
            const auto h2d      = cptype == HIPFFT_COPY_HOST_TO_DEVICE;
            const auto lib_desc = static_cast<hipLibXtDesc*>(h2d ? dest : src);
            if(!lib_desc->descriptor)
                return HIPFFT_INVALID_VALUE;
            const auto desc_io_label = h2d ? fft_io_in : fft_io_out;
            if(!plan->is_valid_for(*lib_desc, desc_io_label))
                return HIPFFT_INVALID_PLAN;
            const auto  desc_subformat = static_cast<hipfftXtSubFormat>(lib_desc->subFormat);
            const auto& field          = plan->field_for_format(desc_subformat, desc_io_label);
            const auto& xt_desc        = *(lib_desc->descriptor);
            // Use field layout stored in the descriptor at allocation time
            const auto  collapsed_field   = hipfft_field::collapse(field);
            const auto& host_global_brick = collapsed_field.get_global_brick();
            const auto  elem_type         = plan->io_type.get_hip_data_type(desc_io_label);
            for(size_t brick_idx = 0; brick_idx < collapsed_field.brick_count(); ++brick_idx)
            {
                const auto& dev_info        = plan->device_contexts[brick_idx];
                const auto& collapsed_brick = collapsed_field.get_brick(brick_idx);
                auto        host_offset
                    = static_cast<char*>(h2d ? src : dest)
                      + collapsed_brick.offset_in(host_global_brick) * hipDataType_bytes(elem_type);
                // copy:
                rocfft_scoped_device dev(collapsed_brick.get_device_id());
                if(collapsed_brick.full_rank() == 1)
                {
                    const auto data_sz = collapsed_brick.data_byte_size(elem_type);
                    HIP_EXPECT_SUCCESS(
                        hipMemcpyAsync(h2d ? xt_desc.data[brick_idx] : host_offset,
                                       h2d ? host_offset : xt_desc.data[brick_idx],
                                       data_sz,
                                       h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost,
                                       dev_info.stream));
                }
                else if(collapsed_brick.full_rank() == 2)
                {
                    const auto brick_strides = collapsed_brick.get_strides();
                    const auto brick_spans   = collapsed_brick.get_spans();
                    const auto host_strides  = host_global_brick.get_strides();
                    HIP_EXPECT_SUCCESS(
                        hipMemcpy2DAsync(h2d ? xt_desc.data[brick_idx] : host_offset,
                                         h2d ? brick_strides[0] * hipDataType_bytes(elem_type)
                                             : host_strides[0] * hipDataType_bytes(elem_type),
                                         h2d ? host_offset : xt_desc.data[brick_idx],
                                         h2d ? host_strides[0] * hipDataType_bytes(elem_type)
                                             : brick_strides[0] * hipDataType_bytes(elem_type),
                                         brick_spans[1] * hipDataType_bytes(elem_type),
                                         brick_spans[0],
                                         h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost,
                                         dev_info.stream));
                }
                else
                {
                    return HIPFFT_INTERNAL_ERROR;
                }
            }
        }
        break;
        case HIPFFT_COPY_DEVICE_TO_DEVICE:
        {
            // Copy operation supposed to be done in plan's output domain
            // (according to cufft documentation). NOTE: INPLACE_SHUFFLE
            // to INPLACE copies result in identical operations in either
            // domain for plans supporting both formats, though.
            const auto  desc_io_label = fft_io_out;
            const auto& src_lib_desc  = *static_cast<hipLibXtDesc*>(src);
            const auto& dst_lib_desc  = *static_cast<hipLibXtDesc*>(dest);
            const auto  src_fmt       = static_cast<hipfftXtSubFormat>(src_lib_desc.subFormat);
            const auto  dst_fmt       = static_cast<hipfftXtSubFormat>(dst_lib_desc.subFormat);
            // Restricted to INPLACE_SHUFFLE -> INPLACE operations.
            if(src_fmt != HIPFFT_XT_FORMAT_INPLACE_SHUFFLED || dst_fmt != HIPFFT_XT_FORMAT_INPLACE)
                return HIPFFT_NOT_SUPPORTED;
            if(!plan->is_valid_for(src_lib_desc, desc_io_label)
               || !plan->is_valid_for(dst_lib_desc, desc_io_label))
                return HIPFFT_INVALID_PLAN;

            const auto& src_field = plan->field_for_format(src_fmt, desc_io_label);
            const auto& dst_field = plan->field_for_format(dst_fmt, desc_io_label);
            const auto  elem_type = plan->io_type.get_hip_data_type(desc_io_label);
            const auto  elem_sz   = hipDataType_bytes(elem_type);
            const auto& src_xt    = *src_lib_desc.descriptor;
            const auto& dst_xt    = *dst_lib_desc.descriptor;
            for(size_t si = 0; si < src_field.brick_count(); ++si)
            {
                auto&                cpy_stream = plan->device_contexts[si].stream;
                rocfft_scoped_device dev(plan->device_contexts[si].device_id);

                for(size_t di = 0; di < dst_field.brick_count(); ++di)
                {
                    // copy src and destination bricks before collapsing common chunks
                    hipfft_brick src_brick(src_field.get_brick(si));
                    hipfft_brick dst_brick(dst_field.get_brick(di));
                    auto         overlap = src_brick.get_overlap_with(dst_brick);
                    if(overlap.is_empty())
                        continue; // no overlap between these two bricks
                    hipfft_brick::collapse_matching_axes(overlap, src_brick, dst_brick);
                    if(overlap.full_rank() != 2)
                        throw std::logic_error(
                            "hipfftXtMemcpy: unexpected rank of overlap in D2D copy");
                    auto dev_pair
                        = std::make_pair(src_brick.get_device_id(), dst_brick.get_device_id());
                    p2p_enablers.try_emplace(dev_pair, dev_pair);
                    const auto spans       = overlap.get_spans();
                    const auto src_strides = src_brick.get_strides();
                    const auto dst_strides = dst_brick.get_strides();
                    if(src_strides.back() != 1 || dst_strides.back() != 1)
                        throw std::logic_error("hipfftXtMemcpy: unexpected elementary stride "
                                               "(non-unit) in src/dst of D2D copy");
                    for(size_t slab_idx = 0; slab_idx < spans[0]; ++slab_idx)
                    {
                        auto* src_ptr = static_cast<const char*>(src_xt.data[si])
                                        + (overlap.offset_in(src_brick) + src_strides[0] * slab_idx)
                                              * elem_sz;
                        auto* dst_ptr = static_cast<char*>(dst_xt.data[di])
                                        + (overlap.offset_in(dst_brick) + dst_strides[0] * slab_idx)
                                              * elem_sz;
                        if(src_brick.get_device_id() != dst_brick.get_device_id())
                        {
                            HIP_EXPECT_SUCCESS(hipMemcpyPeerAsync(dst_ptr,
                                                                  dst_brick.get_device_id(),
                                                                  src_ptr,
                                                                  src_brick.get_device_id(),
                                                                  spans[1] * elem_sz,
                                                                  cpy_stream));
                        }
                        else
                        {
                            HIP_EXPECT_SUCCESS(hipMemcpyAsync(dst_ptr,
                                                              src_ptr,
                                                              spans[1] * elem_sz,
                                                              hipMemcpyDeviceToDevice,
                                                              cpy_stream));
                        }
                    }
                }
            }
        }
        break;
        default:
            return HIPFFT_INVALID_VALUE;
        }
        for(const auto& dev_info : plan->device_contexts)
        {
            rocfft_scoped_device dev(dev_info.device_id);
            HIP_EXPECT_SUCCESS(hipStreamSynchronize(dev_info.stream));
        }
        return HIPFFT_SUCCESS;
    }
    else
    {
        // not implemented yet
        return HIPFFT_NOT_IMPLEMENTED;
    }
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtFree(hipLibXtDesc* desc)
try
{
    hipfftResult ret = HIPFFT_SUCCESS;
    if(desc && desc->descriptor)
    {
        for(size_t i = 0; i < static_cast<size_t>(desc->descriptor->nGPUs); ++i)
        {
            rocfft_scoped_device dev(desc->descriptor->GPUs[i]);
            const auto           tmp = hipFree(desc->descriptor->data[i]);
            if(tmp != hipSuccess)
                ret = HIPFFT_INTERNAL_ERROR;
        }
        delete desc->descriptor;
    }
    delete desc;
    return ret;
}
catch(...)
{
    return handle_exception();
}

// Execute an FFT on a multi-device plan using hipLibXtDesc descriptors.
// Same templating convention as hipfftExecBase: prefer rocfft_transform_type
// when the caller knows the exact transform kind (typed APIs like
// hipfftXtExecDescriptorR2C). The int overload exists for hipfftXtExecDescriptor
// where only a direction is available from the user.
template <typename TransformArgType>
static hipfftResult hipfftXtExecDescriptorBase(const hipfftHandle_t*                 plan,
                                               hipLibXtDesc*                         input,
                                               hipLibXtDesc*                         output,
                                               TransformArgType                      transform_arg,
                                               const std::optional<rocfft_precision> precision
                                               = std::nullopt) noexcept
try
{
    if(!plan || !plan->initialized() || plan->device_contexts.size() < 2)
        return HIPFFT_INVALID_PLAN;
    const auto dft_type = plan->get_transform_type_for(transform_arg);
    if(!plan->can_execute(dft_type, precision))
        return HIPFFT_INVALID_PLAN;
    if(!input || !output || !input->descriptor || (input != output && (!output->descriptor)))
        return HIPFFT_INVALID_VALUE;
    if(!plan->is_valid_for(*input, fft_io_in)
       || (input != output && !plan->is_valid_for(*output, fft_io_out)))
        return HIPFFT_INVALID_PLAN;

    const auto in_subfmt  = static_cast<hipfftXtSubFormat>(input->subFormat);
    const auto out_subfmt = static_cast<hipfftXtSubFormat>(output->subFormat);
    const auto placement = input == output ? rocfft_placement_inplace : rocfft_placement_notinplace;
    // formats' apparent placement must be reflected by argument descriptors
    if(placement_from_format(in_subfmt) != placement
       || placement_from_format(out_subfmt) != placement)
        return HIPFFT_INVALID_VALUE;

    hipfftHandle_t::map_key_t plan_key;
    if(plan->batch > 1)
    {
        // Batched multi-device plans are keyed by placement.
        plan_key = hipfftHandle_t::type_placement_key_t{dft_type, placement};
    }
    else
    {
        // Unbatched multi-device transforms are only in-place
        if(input != output)
            return HIPFFT_NOT_SUPPORTED;
        const auto output_subformat = in_subfmt == HIPFFT_XT_FORMAT_INPLACE
                                          ? HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
                                          : HIPFFT_XT_FORMAT_INPLACE;
        plan_key = hipfftHandle_t::type_subformat_key_t{dft_type, in_subfmt, output_subformat};
    }

    const auto it = plan->exec_plans.find(plan_key);
    if(it == plan->exec_plans.end())
        throw HIPFFT_NOT_SUPPORTED;

    const auto ret
        = rocfft_execute(it->second, input->descriptor->data, output->descriptor->data, plan->info);
    if(ret == rocfft_status_success && input == output && plan->batch == 1)
    {
        // If the execution was successful, then we can change the subformat value if necessary.
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
    return ret == rocfft_status_success ? HIPFFT_SUCCESS : HIPFFT_EXEC_FAILED;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorC2C(hipfftHandle  plan,
                                       hipLibXtDesc* input,
                                       hipLibXtDesc* output,
                                       int           direction)
{
    if(direction != HIPFFT_FORWARD && direction != HIPFFT_BACKWARD)
        return HIPFFT_INVALID_VALUE;
    return hipfftXtExecDescriptorBase(plan,
                                      input,
                                      output,
                                      direction == HIPFFT_FORWARD
                                          ? rocfft_transform_type_complex_forward
                                          : rocfft_transform_type_complex_inverse,
                                      rocfft_precision_single);
}

hipfftResult hipfftXtExecDescriptorR2C(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
{
    return hipfftXtExecDescriptorBase(
        plan, input, output, rocfft_transform_type_real_forward, rocfft_precision_single);
}

hipfftResult hipfftXtExecDescriptorC2R(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
{
    return hipfftXtExecDescriptorBase(
        plan, input, output, rocfft_transform_type_real_inverse, rocfft_precision_single);
}

hipfftResult hipfftXtExecDescriptorZ2Z(hipfftHandle  plan,
                                       hipLibXtDesc* input,
                                       hipLibXtDesc* output,
                                       int           direction)
{
    if(direction != HIPFFT_FORWARD && direction != HIPFFT_BACKWARD)
        return HIPFFT_INVALID_VALUE;
    return hipfftXtExecDescriptorBase(plan,
                                      input,
                                      output,
                                      direction == HIPFFT_FORWARD
                                          ? rocfft_transform_type_complex_forward
                                          : rocfft_transform_type_complex_inverse,
                                      rocfft_precision_double);
}

hipfftResult hipfftXtExecDescriptorD2Z(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
{
    return hipfftXtExecDescriptorBase(
        plan, input, output, rocfft_transform_type_real_forward, rocfft_precision_double);
}

hipfftResult hipfftXtExecDescriptorZ2D(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
{
    return hipfftXtExecDescriptorBase(
        plan, input, output, rocfft_transform_type_real_inverse, rocfft_precision_double);
}

hipfftResult hipfftXtExecDescriptor(hipfftHandle  plan,
                                    hipLibXtDesc* input,
                                    hipLibXtDesc* output,
                                    int           direction)
{
    return hipfftXtExecDescriptorBase(plan, input, output, direction);
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

    if(rank <= 0)
        return HIPFFT_INVALID_VALUE;

    for(auto ptr :
        {input_lower, input_upper, output_lower, output_upper, input_stride, output_stride})
    {
        if(!ptr || std::any_of(ptr, ptr + rank, [](long long int v) { return v < 0; }))
            return HIPFFT_INVALID_VALUE;
    }

    // No support for multiple devices per process
    if(plan->device_contexts.size() > 1)
        return HIPFFT_NOT_SUPPORTED;

    if(plan->device_contexts.empty())
        plan->device_contexts.emplace_back(rocfft_scoped_device::current_device());

    for(auto io : {fft_io_in, fft_io_out})
    {
        // unit batch is implicit for this API
        std::vector<size_t> brick_lower  = {0};
        std::vector<size_t> brick_upper  = {1};
        std::vector<size_t> brick_stride = {0};
        std::copy_n(
            io == fft_io_in ? input_lower : output_lower, rank, std::back_inserter(brick_lower));
        std::copy_n(
            io == fft_io_in ? input_upper : output_upper, rank, std::back_inserter(brick_upper));
        std::copy_n(
            io == fft_io_in ? input_stride : output_stride, rank, std::back_inserter(brick_stride));

        auto& mp_io_brick = io == fft_io_in ? plan->mp_input_brick : plan->mp_output_brick;
        mp_io_brick.emplace(std::move(brick_lower),
                            std::move(brick_upper),
                            std::move(brick_stride),
                            plan->device_contexts.front().device_id);
    }
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
