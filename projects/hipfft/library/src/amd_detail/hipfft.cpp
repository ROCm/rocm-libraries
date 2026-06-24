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
#include "../../../shared/client_data_layout_helpers.h"
#include "../../../shared/gpubuf.h"
#include "../../../shared/hipfft_brick.h"
#include "../../../shared/rocfft_enums_vs_fft_enums.h"
#include "hipfft/hipfftXt.h"
#include "rocfft/rocfft.h"
#include "rocfft_wrapper.h"
#include <algorithm>
#include <cstring> // std::memset
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef HIPFFT_MPI_ENABLE
#include "hipfft/hipfftMp.h"
#endif

#include "../../../shared/ptrdiff.h"
#include "../../../shared/rocfft_hip.h"

#define ROC_FFT_CHECK_INVALID_VALUE(ret)  \
    {                                     \
        auto code = ret;                  \
        if(code != rocfft_status_success) \
        {                                 \
            return HIPFFT_INVALID_VALUE;  \
        }                                 \
    }

#define HIP_FFT_CHECK_AND_RETURN(ret) \
    {                                 \
        auto code = ret;              \
        if(code != HIPFFT_SUCCESS)    \
        {                             \
            return code;              \
        }                             \
    }

// check plan creation - some might fail for specific placement, so
// maintain a count of how many got created, and clean up the plans
// if some failed.
template <typename... Params>
static void ROC_FFT_CHECK_PLAN_CREATE(rocfft_plan_wrapper_t& plan,
                                      unsigned int&          plans_created,
                                      Params&&... params)
{
    if(plan.alloc_with_err(std::forward<Params>(params)...) == rocfft_status_success)
    {
        ++plans_created;
    }
    else
    {
        plan.free();
    }
}

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

    static bool is_forward(rocfft_transform_type type)
    {
        switch(type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_real_forward:
            return true;
        case rocfft_transform_type_complex_inverse:
        case rocfft_transform_type_real_inverse:
            return false;
        default:
            throw HIPFFT_INVALID_VALUE;
        }
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

    hipDataType spaceType() const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        if(is_complex_to_complex())
        {
            if(inputType != outputType)
                throw std::runtime_error("input/output types differ");

            // Doesn't matter if we choose input of output type.
            return inputType;
        }
        else
        {
            if(is_real_to_complex())
                return inputType;
            else
                return outputType;
        }
    }

    hipDataType freqType() const
    {
        if(!isinitialized)
            throw std::runtime_error("hipfftIOType not initialized");

        if(is_complex_to_complex())
        {
            if(inputType != outputType)
                throw std::runtime_error("input/output types differ");

            // Doesn't matter if we choose input of output type.
            return inputType;
        }
        else
        {
            if(is_real_to_complex())
                return outputType;
            else
                return inputType;
        }
    }
};

struct hipfftHandle_t
{
    hipfftIOType type;

    // Logical transform lengths (row-major)
    std::vector<size_t>       lengths;
    size_t                    batch;
    hipfft_ionembed_t<size_t> ionembed;

    // Due to hipfftExec** compatibility to cuFFT, we have to reserve all 4 types
    // rocfft handle separately here.
    rocfft_plan_wrapper_t ip_forward;
    rocfft_plan_wrapper_t op_forward;
    rocfft_plan_wrapper_t ip_inverse;
    rocfft_plan_wrapper_t op_inverse;

    // Return true if the plans have been initialized - hipfftCreate
    // merely allocates a handle and a hipfftMakePlan* API initializes
    // them.
    bool initialized() const
    {
        return ip_forward || op_forward || ip_inverse || op_inverse;
    }

    rocfft_execution_info_wrapper_t info;
    gpubuf                          workBuffer;
    size_t                          workBufferSize = 0;
    bool                            autoAllocate   = true;

    void** load_callback_ptrs       = nullptr;
    void** load_callback_data       = nullptr;
    size_t load_callback_lds_bytes  = 0;
    void** store_callback_ptrs      = nullptr;
    void** store_callback_data      = nullptr;
    size_t store_callback_lds_bytes = 0;

    double scale_factor = 1.0;

    // Brick decomposition for multi-device transforms
    std::vector<hipfft_brick> spaceBricks;
    std::vector<hipfft_brick> freqBricks;
    // hipFFT will decompose the problem across multiple devices in a single process (i.e. via
    // hipfftXtSetGPUs)
    bool singleProcMultiDevice = false;

    // Multi-processing communicator
    rocfft_comm_type comm_type   = rocfft_comm_none;
    void*            comm_handle = nullptr;

    // Get the data type based on the sub-format value.
    auto brick_format_to_type(const int subFormat)
    {
        switch(subFormat)
        {
        case HIPFFT_XT_FORMAT_INPUT:
        case HIPFFT_XT_FORMAT_INPLACE:
            return type.get_inputType();
        case HIPFFT_XT_FORMAT_OUTPUT:
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
            return type.get_outputType();
        default:
            throw HIPFFT_INVALID_VALUE;
        }
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

// Given an array of bricks, set the brick format
static void hipfftxt_bricks(const std::vector<size_t>& batchlength,
                            std::vector<hipfft_brick>& bricks,
                            const bool                 isrealcomplex,
                            const hipfftXtSubFormat    subformat)
{
    // We assume that the brick vector has already been allocated, but the brick data is not yet
    // computed.
    if(bricks.size() == 0)
        throw std::runtime_error("Bricks vector needs to be allocated before passing");

    // Format is row-major.

    // batchlength includes the (single) batch dimension, so the batchlengths are {batch, X, Y, Z},
    // {batch, X, Y}, or {batch, X}.
    const size_t dim = batchlength.size();
    if(dim < 2)
        throw std::runtime_error("Need at least 1 length and batch dim");

    const size_t         nbatch = batchlength[0];
    fft_result_placement placement;
    fft_io               io;
    const bool           isherm = isrealcomplex && subformat == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
    const bool           isreal = isrealcomplex && subformat == HIPFFT_XT_FORMAT_INPLACE;

    // All complex data formats are treated as part of a complex/complex transform in order to allow
    // us to handle the split dimension being the Hermitian-symmetrized dimension.
    const fft_transform_type dft_type
        = isreal ? fft_transform_type_real_forward : fft_transform_type_complex_forward;

    // The subformat tells us which dimension is split.
    // Real in-place data needs extra padding.
    size_t splitdim = 0;
    if(nbatch == 1)
    {
        switch(subformat)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            splitdim  = 1; // X-axis is split
            placement = fft_placement_notinplace;
            io        = fft_io_in;
            break;
        case HIPFFT_XT_FORMAT_OUTPUT:
            splitdim  = 2; // Y-axis is split
            placement = fft_placement_notinplace;
            io        = fft_io_out;
            break;
        case HIPFFT_XT_FORMAT_INPLACE:
            splitdim  = 1; // X-axis is split
            placement = fft_placement_inplace;
            io        = fft_io_in;
            break;
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
            splitdim  = 2; // Y-axis is split
            placement = fft_placement_inplace;
            io        = fft_io_out;
            break;
        case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
            // TODO: implement 1D version.
            // TODO: what do we do with multi-gpu multi-batch 1D transforms?
            throw HIPFFT_NOT_IMPLEMENTED;
            break;
        case HIPFFT_FORMAT_UNDEFINED:
            break;
        default:
            throw std::runtime_error("Invalid subformat");
        }
    }
    else
    {
        // Multi-batch transforms are trivially divided.
        splitdim = 0;
        throw HIPFFT_NOT_IMPLEMENTED;
    }

    // Sanity check that split_dim isn't out-of-bounds:
    if(splitdim >= dim)
        throw HIPFFT_INTERNAL_ERROR;

    // We are going to put the Hermitian-symmetric length change here:
    auto batchlengthdata = batchlength;
    if(isherm)
    {
        // We have Hermitian-symmetric data
        const auto hindex       = batchlengthdata.size() - 1;
        const auto hlength      = batchlengthdata[hindex];
        batchlengthdata[hindex] = hlength / 2 + 1;
    }

    const auto nbricks = bricks.size();
    for(size_t ibrick = 0; ibrick < nbricks; ++ibrick)
    {
        auto& brick = bricks[ibrick];

        brick.field_lower.resize(dim);
        std::fill(brick.field_lower.begin(), brick.field_lower.end(), 0);

        const size_t splitlen      = batchlengthdata[splitdim];
        const size_t bricksplitlen = splitlen / nbricks + (ibrick < splitlen % nbricks ? 1 : 0);
        brick.field_upper          = batchlengthdata;
        if(ibrick > 0)
        {
            brick.field_lower[splitdim] = bricks[ibrick - 1].field_upper[splitdim];
        }
        brick.field_upper[splitdim] = brick.field_lower[splitdim] + bricksplitlen;

        brick.brick_stride = default_brick_strides(
            dft_type, placement, io, batchlength, brick.field_lower, brick.field_upper);
    }
}

// note: rm_lengths arg is in row-major order
static hipfftResult hipfftMakePlan_internal(hipfftHandle               plan,
                                            size_t                     dim,
                                            size_t*                    rm_lengths,
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
    static rocfft_initializer init;

    plan->type = iotype;
    if(plan->singleProcMultiDevice)
    {
        // We currently do not support multi-batch multi-device transforms.
        if(number_of_transforms > 1)
            return HIPFFT_NOT_IMPLEMENTED;

        // We currently do not support 1D multi-device transforms.
        if(dim == 1)
            return HIPFFT_NOT_IMPLEMENTED;
    }

    const bool                        isrealcomplex = !iotype.is_complex_to_complex();
    rocfft_plan_description_wrapper_t ip_forward_desc;
    rocfft_plan_description_wrapper_t op_forward_desc;
    rocfft_plan_description_wrapper_t ip_inverse_desc;
    rocfft_plan_description_wrapper_t op_inverse_desc;
    ip_forward_desc.alloc();
    op_forward_desc.alloc();
    ip_inverse_desc.alloc();
    op_inverse_desc.alloc();

    std::reference_wrapper<rocfft_plan_description_wrapper_t> fwd_descs[]
        = {ip_forward_desc, op_forward_desc};
    std::reference_wrapper<rocfft_plan_description_wrapper_t> inverse_descs[]
        = {ip_inverse_desc, op_inverse_desc};
    std::reference_wrapper<rocfft_plan_description_wrapper_t> all_descs[]
        = {ip_forward_desc, op_forward_desc, ip_inverse_desc, op_inverse_desc};

    plan->lengths.assign(rm_lengths, rm_lengths + dim);
    const std::vector<size_t> cm_lengths_vec(plan->lengths.rbegin(), plan->lengths.rend());

    plan->batch = number_of_transforms;

    // copy the user's ionembed into the plan if there is one, use default otherwise
    plan->ionembed = !user_ionembed ? hipfft_ionembed_t<size_t>() : *user_ionembed;
    // NOTE: hipFFT ignores distance arguments if default layouts are used!
    const bool ignore_user_distances = !plan->ionembed.get_nembed(fft_io::fft_io_in)
                                       && !plan->ionembed.get_nembed(fft_io::fft_io_out);

    for(auto dft_type : iotype.transform_types())
    {
        for(auto placement : {rocfft_placement_inplace, rocfft_placement_notinplace})
        {
            auto& plan_desc
                = placement == rocfft_placement_inplace
                      ? (iotype.is_forward(dft_type) ? ip_forward_desc : ip_inverse_desc)
                      : (iotype.is_forward(dft_type) ? op_forward_desc : op_inverse_desc);
            auto i_strides = plan->ionembed.as_generalized_strides(
                fft_io::fft_io_in,
                fft_transform_type_from_rocfft_transform_type(dft_type),
                fft_result_placement_from_rocfft_result_placement(placement),
                plan->lengths);
            auto o_strides = plan->ionembed.as_generalized_strides(
                fft_io::fft_io_out,
                fft_transform_type_from_rocfft_transform_type(dft_type),
                fft_result_placement_from_rocfft_result_placement(placement),
                plan->lengths);

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
                          plan->lengths,
                          number_of_transforms);
            const auto outDist
                = !ignore_user_distances
                      ? user_odist
                      : default_distance(
                          fft_transform_type_from_rocfft_transform_type(dft_type),
                          fft_result_placement_from_rocfft_result_placement(placement),
                          fft_io::fft_io_out,
                          plan->lengths,
                          number_of_transforms);

            auto ret
                = rocfft_plan_description_set_data_layout(plan_desc,
                                                          iotype.array_type(fft_io::fft_io_in),
                                                          iotype.array_type(fft_io::fft_io_out),
                                                          0,
                                                          0,
                                                          dim,
                                                          i_strides.data(),
                                                          inDist,
                                                          dim,
                                                          o_strides.data(),
                                                          outDist);
            if(ret != rocfft_status_success)
            {
                return HIPFFT_INVALID_VALUE;
            }
        }
    }

    if(plan->singleProcMultiDevice)
    {
        // Problem dimensions and strides are known, set up the bricks for single-proc multi-GPU

        std::vector<size_t> batchlength = {plan->batch};
        batchlength.insert(batchlength.end(), plan->lengths.begin(), plan->lengths.end());

        const hipfftXtSubFormat spacesubformat
            = isrealcomplex ? HIPFFT_XT_FORMAT_INPLACE : HIPFFT_XT_FORMAT_INPUT;
        hipfftxt_bricks(batchlength, plan->spaceBricks, isrealcomplex, spacesubformat);

        const hipfftXtSubFormat freqsubformat
            = isrealcomplex ? HIPFFT_XT_FORMAT_INPLACE_SHUFFLED : HIPFFT_XT_FORMAT_OUTPUT;
        hipfftxt_bricks(batchlength, plan->freqBricks, isrealcomplex, freqsubformat);
    }

    if(plan->singleProcMultiDevice)
    {
        // TODO: make sure we don't have a communicator.

        std::vector<size_t> batches      = {plan->batch};
        std::vector<size_t> batchlengths = batches;
        batchlengths.insert(batchlengths.end(), plan->lengths.begin(), plan->lengths.end());

        // Lambda for converting hipfft-bricks to rocfft-bricks and adding them to a rocfft
        // description:
        auto hipBricks2Fields
            = [](std::vector<hipfft_brick>& hipBricks, rocfft_field_wrapper_t& destField) {
                  for(const auto& brick : hipBricks)
                  {
                      // rm -> cm
                      auto cm_lower = brick.field_lower;
                      std::reverse(cm_lower.begin(), cm_lower.end());
                      auto cm_upper = brick.field_upper;
                      std::reverse(cm_upper.begin(), cm_upper.end());
                      auto cm_stride = brick.brick_stride;
                      std::reverse(cm_stride.begin(), cm_stride.end());

                      rocfft_brick_wrapper_t rbrick;
                      rbrick.alloc(cm_lower.data(),
                                   cm_upper.data(),
                                   cm_stride.data(),
                                   cm_lower.size(),
                                   brick.device);
                      if(rocfft_field_add_brick(destField, rbrick) != rocfft_status_success)
                          throw std::runtime_error("add brick failed");
                  }
              };

        rocfft_field_wrapper_t spaceField;
        spaceField.alloc();
        hipBricks2Fields(plan->spaceBricks, spaceField);

        rocfft_field_wrapper_t frequencyField;
        frequencyField.alloc();
        hipBricks2Fields(plan->freqBricks, frequencyField);

        for(auto& rocfft_desc : fwd_descs)
        {
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_description_add_infield(rocfft_desc.get(), spaceField));
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_description_add_outfield(rocfft_desc.get(), frequencyField));
        }
        for(auto& rocfft_desc : inverse_descs)
        {
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_description_add_infield(rocfft_desc.get(), frequencyField));
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_description_add_outfield(rocfft_desc.get(), spaceField));
        }
    }

    if(plan->scale_factor != 1.0)
    {
        for(auto& rocfft_desc : all_descs)
        {
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_description_set_scale_factor(rocfft_desc.get(), plan->scale_factor));
        }
    }

    if(plan->comm_type != rocfft_comm_none)
    {
        for(auto& rocfft_desc : all_descs)
        {
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_description_set_comm(
                rocfft_desc.get(), plan->comm_type, plan->comm_handle));
        }
    }

    // Count the number of plans that got created - it's possible to
    // have parameters that are valid for out-place but not for
    // in-place, so some of these rocfft_plan_creates could
    // legitimately fail.
    unsigned int plans_created = 0;
    for(auto t : iotype.transform_types())
    {
        for(const auto inplace : {true, false})
        {
            const bool forward   = iotype.is_forward(t);
            auto&      plan_ptr  = inplace ? (forward ? plan->ip_forward : plan->ip_inverse)
                                           : (forward ? plan->op_forward : plan->op_inverse);
            auto&      plan_desc = inplace ? (forward ? ip_forward_desc : ip_inverse_desc)
                                           : (forward ? op_forward_desc : op_inverse_desc);
            const auto placement = inplace ? rocfft_placement_inplace : rocfft_placement_notinplace;
            ROC_FFT_CHECK_PLAN_CREATE(plan_ptr,
                                      plans_created,
                                      placement,
                                      t,
                                      iotype.precision(),
                                      dim,
                                      cm_lengths_vec.data(),
                                      number_of_transforms,
                                      plan_desc);
        }
    }

    // If no plans got created, fail
    if(plans_created == 0)
        return HIPFFT_PARSE_ERROR;
    plan->type = iotype;

    size_t workBufferSize = 0;
    size_t tmpBufferSize  = 0;

    bool const has_forward = !iotype.is_complex_to_real();
    if(has_forward)
    {
        if(plan->ip_forward)
        {
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_get_work_buffer_size(plan->ip_forward, &tmpBufferSize));
            workBufferSize = std::max(workBufferSize, tmpBufferSize);
        }
        if(plan->op_forward)
        {
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_get_work_buffer_size(plan->op_forward, &tmpBufferSize));
            workBufferSize = std::max(workBufferSize, tmpBufferSize);
        }
    }

    bool const has_inverse = !iotype.is_real_to_complex();
    if(has_inverse)
    {
        if(plan->ip_inverse)
        {
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_get_work_buffer_size(plan->ip_inverse, &tmpBufferSize));
            workBufferSize = std::max(workBufferSize, tmpBufferSize);
        }
        if(plan->op_inverse)
        {
            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_get_work_buffer_size(plan->op_inverse, &tmpBufferSize));
            workBufferSize = std::max(workBufferSize, tmpBufferSize);
        }
    }

    if(workSize != nullptr)
        *workSize = workBufferSize;

    plan->workBufferSize = workBufferSize;

    if(workBufferSize > 0)
    {
        if(plan->autoAllocate)
        {
            if(plan->workBuffer.alloc(workBufferSize) != hipSuccess)
                return HIPFFT_ALLOC_FAILED;
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_set_work_buffer(
                plan->info, plan->workBuffer.data(), workBufferSize));
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
    h->info.alloc();
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

    size_t lengths[1];
    lengths[0]                                      = nx;
    size_t                     number_of_transforms = batch;
    hipfft_ionembed_t<size_t>* user_ionembed        = nullptr;
    // ignored internally (default layout)
    size_t ignored_dist = 0;

    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(type));

    return hipfftMakePlan_internal(plan,
                                   1,
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

    size_t                     lengths[2] = {static_cast<size_t>(nx), static_cast<size_t>(ny)};
    size_t                     number_of_transforms = 1;
    hipfft_ionembed_t<size_t>* user_ionembed        = nullptr;
    // ignored internally (default layout)
    size_t ignored_dist = 0;

    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(type));

    return hipfftMakePlan_internal(plan,
                                   2,
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

    size_t lengths[3];
    lengths[0]                                      = nx;
    lengths[1]                                      = ny;
    lengths[2]                                      = nz;
    size_t                     number_of_transforms = 1;
    hipfft_ionembed_t<size_t>* user_ionembed        = nullptr;
    // ignored internally (default layout)
    size_t ignored_dist = 0;

    hipfftIOType iotype;
    HIP_FFT_CHECK_AND_RETURN(iotype.init(type));

    return hipfftMakePlan_internal(plan,
                                   3,
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
                                               rank,
                                               lengths.data(),
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
    p->autoAllocate = false;
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
    p->autoAllocate = false;
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
    p->autoAllocate = false;
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
    p->autoAllocate = false;
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
    p->autoAllocate = false;
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
    *workSize = plan->workBufferSize;
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
    plan->autoAllocate = bool(autoAllocate);
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea)
try
{
    if(!plan)
        return HIPFFT_INVALID_PLAN;

    plan->workBuffer.free();
    if(workArea)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_execution_info_set_work_buffer(plan->info, workArea, plan->workBufferSize));
    }
    plan->autoAllocate = false;
    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

// Find the specific plan to execute - check placement and direction
static rocfft_plan get_exec_plan(const hipfftHandle plan, const bool inplace, const int direction)
{
    if(!plan || !plan->initialized())
    {
        throw HIPFFT_INVALID_PLAN;
    }

    switch(direction)
    {
    case HIPFFT_FORWARD:
        return inplace ? plan->ip_forward : plan->op_forward;
    case HIPFFT_BACKWARD:
        return inplace ? plan->ip_inverse : plan->op_inverse;
    default:
        throw HIPFFT_INVALID_VALUE;
    }
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

static hipfftResult hipfftExecForward(hipfftHandle plan, void* idata, void* odata)
{
    const bool inplace = idata == odata;
    const auto rplan   = get_exec_plan(plan, inplace, HIPFFT_FORWARD);
    return hipfftExec(rplan, plan->info, idata, odata);
}

static hipfftResult hipfftExecBackward(hipfftHandle plan, void* idata, void* odata)
{
    const bool inplace = idata == odata;
    const auto rplan   = get_exec_plan(plan, inplace, HIPFFT_BACKWARD);
    return hipfftExec(rplan, plan->info, idata, odata);
}

template <rocfft_precision_e prec>
static inline bool is_ready_for_execution(const hipfftHandle_t* plan)
{
    return plan != nullptr && plan->initialized() && plan->type.precision() == prec;
}

hipfftResult
    hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
try
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;
    switch(direction)
    {
    case HIPFFT_FORWARD:
        return hipfftExecForward(plan, idata, odata);
    case HIPFFT_BACKWARD:
        return hipfftExecBackward(plan, idata, odata);
    }
    return HIPFFT_INVALID_VALUE;
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
    return hipfftExecForward(plan, idata, odata);
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
    return hipfftExecBackward(plan, idata, odata);
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
    switch(direction)
    {
    case HIPFFT_FORWARD:
        return hipfftExecForward(plan, idata, odata);
    case HIPFFT_BACKWARD:
        return hipfftExecBackward(plan, idata, odata);
    }
    return HIPFFT_INVALID_VALUE;
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
    return hipfftExecForward(plan, idata, odata);
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
    return hipfftExecBackward(plan, idata, odata);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
try
{
    if(!plan)
        return HIPFFT_INVALID_PLAN;
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_set_stream(plan->info, stream));
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
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_get_version_string(v, 256));

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
        if(plan->type.precision() != rocfft_precision_single || plan->type.is_real_to_complex())
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
        if(plan->type.precision() != rocfft_precision_double || plan->type.is_real_to_complex())
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_REAL:
        if(plan->type.precision() != rocfft_precision_single || !plan->type.is_real_to_complex())
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_REAL_DOUBLE:
        if(plan->type.precision() != rocfft_precision_double || !plan->type.is_real_to_complex())
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_COMPLEX:
        if(plan->type.precision() != rocfft_precision_single || plan->type.is_complex_to_real())
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
        if(plan->type.precision() != rocfft_precision_double || plan->type.is_complex_to_real())
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_REAL:
        if(plan->type.precision() != rocfft_precision_single || !plan->type.is_complex_to_real())
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_REAL_DOUBLE:
        if(plan->type.precision() != rocfft_precision_double || !plan->type.is_complex_to_real())
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
    p->autoAllocate = false;

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
    bool        inplace  = input == output;
    rocfft_plan plan_ptr = get_exec_plan(plan, inplace, direction);
    return hipfftExec(plan_ptr, plan->info, input, output);
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
    int dev_count = 0;
    if(hipGetDeviceCount(&dev_count) != hipSuccess || dev_count <= 0)
        return HIPFFT_INTERNAL_ERROR;
    if(std::any_of(
           gpus, gpus + count, [=](int gpu_id) { return gpu_id < 0 || gpu_id >= dev_count; }))
        return HIPFFT_INVALID_VALUE;

    // we know how many bricks we will have, but we haven't been told
    // the problem dimensions yet so we don't know what the bricks
    // will look like.
    plan->spaceBricks.resize(static_cast<size_t>(count));
    plan->freqBricks.resize(static_cast<size_t>(count));

    // but at this point we know devices, so record what the user
    // gave us
    for(size_t i = 0; i < static_cast<size_t>(count); ++i)
    {
        plan->spaceBricks[i].device = gpus[i];
        plan->freqBricks[i].device  = gpus[i];
    }

    // FIXME: what if only 1 gpu is provided?
    plan->singleProcMultiDevice = true;

    return HIPFFT_SUCCESS;
}
catch(...)
{
    return handle_exception();
}

// get number of bytes used for elements of a given hipDataType
static size_t hipDataType_bits(hipDataType t)
{
    switch(t)
    {
    case HIP_R_16F:
        // real half
        return 16;
    case HIP_C_16F:
    case HIP_R_32F:
        // complex half and real single
        return 32;
    case HIP_C_32F:
    case HIP_R_64F:
        // complex single and real double
        return 64;
    case HIP_C_64F:
        // complex double
        return 128;
    default:
        throw std::runtime_error("unsupported data type");
    }
}

static size_t hipDataType_bytes(hipDataType t, size_t numElems)
{
    return hipDataType_bits(t) * numElems / 8;
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
    if(plan->type.is_real_to_complex() && format != HIPFFT_XT_FORMAT_INPLACE)
        return HIPFFT_NOT_IMPLEMENTED;
    if(plan->type.is_complex_to_real() && format != HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
        return HIPFFT_NOT_IMPLEMENTED;

    auto lib_desc = std::make_unique<hipLibXtDesc>();
    std::memset(lib_desc.get(), 0, sizeof(hipLibXtDesc));

    lib_desc->version       = 0;
    lib_desc->library       = HIPLIB_FORMAT_HIPFFT;
    lib_desc->subFormat     = format;
    lib_desc->libDescriptor = nullptr;

    auto xt_desc = std::make_unique<hipXtDesc>();
    std::memset(xt_desc.get(), 0, sizeof(hipXtDesc));
    xt_desc->version = 0;

    std::vector<hipfft_brick>* bricks = nullptr;

    std::vector<size_t> batches      = {plan->batch};
    std::vector<size_t> batchlengths = batches;
    batchlengths.insert(batchlengths.end(), plan->lengths.begin(), plan->lengths.end());
    const bool isspace = format == HIPFFT_XT_FORMAT_INPUT || format == HIPFFT_XT_FORMAT_INPLACE;

    const bool isinplace
        = format == HIPFFT_XT_FORMAT_INPLACE || format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;

    bricks = isspace ? &plan->spaceBricks : &plan->freqBricks;

    xt_desc->nGPUs = static_cast<int>(bricks->size());

    for(size_t idx = 0; idx < bricks->size(); ++idx)
    {
        auto& brick = (*bricks)[idx];

        rocfft_scoped_device dev(brick.device);

        xt_desc->GPUs[idx] = brick.device;

        if(isinplace)
        {
            // NB: we do not use compute_ptrdiff here because we need to be a bit greedy with
            // allocation: hipfftXtMemcpy will use the entire padded buffer, so we need the extra
            // one or two worth of real-values of space at the end of the buffer which
            // compute_ptrdiff would save us from allocating.
            auto spacebufsize
                = (plan->spaceBricks[idx].field_upper[0] - plan->spaceBricks[idx].field_lower[0])
                  * plan->spaceBricks[idx].brick_stride[0];
            const size_t space_bytes_per_element = hipDataType_bits(plan->type.spaceType()) / 8;
            auto         freqbufsize            = compute_ptrdiff(plan->freqBricks[idx].field_lower,
                                               plan->freqBricks[idx].field_upper,
                                               plan->freqBricks[idx].brick_stride);
            const size_t freq_bytes_per_element = hipDataType_bits(plan->type.freqType()) / 8;
            xt_desc->size[idx]                  = std::max(spacebufsize * space_bytes_per_element,
                                          freqbufsize * freq_bytes_per_element);
        }
        else
        {
            auto bufsize
                = compute_ptrdiff(brick.field_lower, brick.field_upper, brick.brick_stride);
            const size_t bits_per_element = hipDataType_bits(plan->brick_format_to_type(format));
            xt_desc->size[idx]            = bufsize * bits_per_element / 8;
        }

        if(xt_desc->size[idx] == 0)
        {
            // TODO: how should we handle the case where some devices don't have data?
            return HIPFFT_NOT_IMPLEMENTED;
        }
        if(hipMalloc(&(xt_desc->data[idx]), xt_desc->size[idx]) != hipSuccess)
            return HIPFFT_INTERNAL_ERROR;
    }

    lib_desc->descriptor = xt_desc.release();
    *desc                = lib_desc.release();
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
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;
    if(!dest || !src || dest == src)
        return HIPFFT_INVALID_VALUE;

    // Get pointer into buf, at the index pointed to by lower assuming lengths are strided by stride
    auto offset_buffer = [](void*                      buf,
                            hipDataType                dtype,
                            const std::vector<size_t>& lower,
                            const std::vector<size_t>& stride) {
        auto offset_elems = std::inner_product(
            lower.begin(),
            lower.end(),
            stride.begin(),
            static_cast<std::remove_reference_t<decltype(lower)>::value_type>(0));
        return static_cast<void*>(static_cast<char*>(buf) + hipDataType_bytes(dtype, offset_elems));
    };

    // This determines whether we use the space brick decomposition or the frequency brick
    // decomposition for the copy operation.
    auto brick_layout = [plan](int subFormat) -> const std::vector<hipfft_brick>& {
        switch(subFormat)
        {
        case HIPFFT_XT_FORMAT_INPUT:
        case HIPFFT_XT_FORMAT_INPLACE:
            return plan->spaceBricks;
        case HIPFFT_XT_FORMAT_OUTPUT:
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
            return plan->freqBricks;
        default:
            throw HIPFFT_INVALID_VALUE;
        }
    };

    // This determines whether we use the input brick decomposition or the output brick
    // decomposition for the copy operation.
    auto brick_format = [plan](int subFormat) -> hipDataType {
        switch(subFormat)
        {
        case HIPFFT_XT_FORMAT_INPUT:
        case HIPFFT_XT_FORMAT_INPLACE:
            return plan->type.spaceType();
        case HIPFFT_XT_FORMAT_OUTPUT:
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
            return plan->type.freqType();
        default:
            throw HIPFFT_INVALID_VALUE;
        }
    };

    switch(cptype)
    {
    case HIPFFT_COPY_HOST_TO_DEVICE:
        [[fallthrough]];
    case HIPFFT_COPY_DEVICE_TO_HOST:
    {
        const bool h2d = cptype == HIPFFT_COPY_HOST_TO_DEVICE;

        auto myDesc = static_cast<hipLibXtDesc*>(h2d ? dest : src);
        if(!myDesc->descriptor)
            return HIPFFT_INVALID_VALUE;

        const bool realdata = !plan->type.is_complex_to_complex()
                              && (myDesc->subFormat == HIPFFT_XT_FORMAT_INPLACE);
        const bool hermdata = !plan->type.is_complex_to_complex()
                              && (myDesc->subFormat == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED);
        const bool inplace = (myDesc->subFormat == HIPFFT_XT_FORMAT_INPLACE)
                             || (myDesc->subFormat == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED);

        std::vector<size_t> hostDataLengths = {plan->batch};
        hostDataLengths.insert(hostDataLengths.end(), plan->lengths.begin(), plan->lengths.end());
        const size_t lastdim = hostDataLengths.size() - 1;

        const auto dft_type = plan->type.is_complex_to_complex()
                                  ? fft_transform_type_complex_forward
                                  : fft_transform_type_real_forward;
        const auto io       = (myDesc->subFormat == HIPFFT_XT_FORMAT_INPUT
                         || myDesc->subFormat == HIPFFT_XT_FORMAT_INPLACE)
                                  ? fft_io_in
                                  : fft_io_out;
        auto       hostDataStride
            = default_strides(dft_type,
                              inplace ? fft_placement_inplace : fft_placement_notinplace,
                              io,
                              hostDataLengths);
        if(hermdata)
        {
            // Row-major, so fold on the last dim.
            hostDataLengths[lastdim] = hostDataLengths[lastdim] / 2 + 1;
        }
        if(realdata)
        {
            // We are going to expand the real data to include the padding so that we can
            // do contiguous memcpys.
            hostDataLengths[lastdim] = 2 * (hostDataLengths[lastdim] / 2 + 1);
        }

        for(size_t idx = 0; idx < static_cast<size_t>(myDesc->descriptor->nGPUs); ++idx)
        {
            rocfft_scoped_device dev(myDesc->descriptor->GPUs[idx]);

            // Space bricks or frequency bricks:
            const auto& brick = brick_layout(myDesc->subFormat)[idx];

            auto brick_length = brick.length();
            if(realdata)
            {
                // Make the brick real data contiguous as well.
                brick_length[lastdim] = 2 * (brick_length[lastdim] / 2 + 1);
            }
            auto brick_stride = brick.brick_stride;

            const auto host_offset = offset_buffer(h2d ? src : dest,
                                                   brick_format(myDesc->subFormat),
                                                   brick.field_lower,
                                                   hostDataStride);

            auto brick_length_collapsed   = brick_length;
            auto brick_stride_collapsed   = brick_stride;
            auto hostDataStride_collapsed = hostDataStride;
            collapse_contiguous_dims(
                brick_length_collapsed, brick_stride_collapsed, hostDataStride_collapsed);

            // Fastest dim is expected to be contiguous
            if(brick_stride_collapsed.back() != 1 || hostDataStride_collapsed.back() != 1)
            {
                throw std::runtime_error("fastest dim not contiguous after collapsing");
            }

            auto       destptr     = h2d ? myDesc->descriptor->data[idx] : host_offset;
            const auto srcptr      = h2d ? host_offset : myDesc->descriptor->data[idx];
            const auto cpdirection = h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
            switch(brick_length_collapsed.size())
            {
            case 1:
            {
                auto ret = hipMemcpy(destptr,
                                     srcptr,
                                     myDesc->descriptor->size[idx],
                                     h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost);
                if(ret != hipSuccess)
                {
                    return HIPFFT_INTERNAL_ERROR;
                }

                break;
            }
            case 2:
            {
                size_t valsize;
                switch(plan->type.precision())
                {
                case rocfft_precision_half:
                    valsize = sizeof(float) / 2;
                    break;
                case rocfft_precision_single:
                    valsize = sizeof(float);
                    break;
                case rocfft_precision_double:
                    valsize = sizeof(double);
                    break;
                default:
                    return HIPFFT_INTERNAL_ERROR;
                }
                if(!realdata)
                    valsize *= 2;

                size_t dpitch = h2d ? brick_stride_collapsed[0] : hostDataStride_collapsed[0];
                size_t spitch = h2d ? hostDataStride_collapsed[0] : brick_stride_collapsed[0];
                size_t width  = brick_length_collapsed[1];
                size_t height = brick_length_collapsed[0];

                auto ret = hipMemcpy2D(destptr,
                                       valsize * dpitch, //  dpitch (bytes between starts of rows)
                                       srcptr,
                                       valsize * spitch, //  spitch (bytes between starts of rows)
                                       valsize * width, // width  (bytes in a row)
                                       height, // height (how many rows)
                                       cpdirection);
                if(ret != hipSuccess)
                {
                    return HIPFFT_INTERNAL_ERROR;
                }
                break;
            }
            default:
                return HIPFFT_INTERNAL_ERROR;
            }
        }
        return HIPFFT_SUCCESS;
    }
    case HIPFFT_COPY_DEVICE_TO_DEVICE:
        return HIPFFT_NOT_IMPLEMENTED;
    case HIPFFT_COPY_UNDEFINED:
        return HIPFFT_NOT_IMPLEMENTED;
    default:
        throw HIPFFT_INVALID_VALUE;
    }
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

    const bool inplace = input == output;
    const auto rplan   = get_exec_plan(plan, inplace, direction);
    if(!rplan)
    {
        return HIPFFT_INVALID_PLAN;
    }

    auto ret = rocfft_status_success;
    try
    {
        ret = rocfft_execute(rplan, input->descriptor->data, output->descriptor->data, plan->info);
        if(ret == rocfft_status_success && inplace)
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
    }
    catch(...)
    {
        return handle_exception();
    }

    return ret == rocfft_status_success ? HIPFFT_SUCCESS : HIPFFT_EXEC_FAILED;
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
