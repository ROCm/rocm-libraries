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
#include "../../../shared/hipfft_brick.h"
#include "../../../shared/rocfft_enums_vs_fft_enums.h"
#include "hipfft/hipfftXt.h"
#include "rocfft/rocfft.h"
#include <algorithm>
#include <cstring> // std::memset
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
static void
    ROC_FFT_CHECK_PLAN_CREATE(rocfft_plan& plan, unsigned int& plans_created, Params&&... params)
{
    if(rocfft_plan_create(&plan, std::forward<Params>(params)...) == rocfft_status_success)
    {
        ++plans_created;
    }
    else
    {
        rocfft_plan_destroy(plan);
        plan = nullptr;
    }
}

struct hipfftIOType
{
    hipDataType inputType  = HIP_C_32F;
    hipDataType outputType = HIP_C_32F;

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

        inputType  = input;
        outputType = output;
        return HIPFFT_SUCCESS;
    }

    rocfft_precision precision() const
    {
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
};

struct hipfftHandle_t
{
    hipfftIOType type;

    // Due to hipfftExec** compatibility to cuFFT, we have to reserve all 4 types
    // rocfft handle separately here.
    rocfft_plan ip_forward = nullptr;
    rocfft_plan op_forward = nullptr;
    rocfft_plan ip_inverse = nullptr;
    rocfft_plan op_inverse = nullptr;

    // return true if the plans have been initialized - hipfftCreate
    // merely allocates a handle and a hipfftMakePlan* API initializes
    // them.
    bool initialized() const
    {
        return ip_forward || op_forward || ip_inverse || op_inverse;
    }

    rocfft_execution_info info                = nullptr;
    void*                 workBuffer          = nullptr;
    size_t                workBufferSize      = 0;
    bool                  autoAllocate        = true;
    bool                  workBufferNeedsFree = false;

    void** load_callback_ptrs       = nullptr;
    void** load_callback_data       = nullptr;
    size_t load_callback_lds_bytes  = 0;
    void** store_callback_ptrs      = nullptr;
    void** store_callback_data      = nullptr;
    size_t store_callback_lds_bytes = 0;

    std::vector<size_t>       inLength;
    std::vector<size_t>       outLength;
    hipfft_ionembed_t<size_t> ionembed;
    // FIXME: the following members are relevant to hipfftXtMemcpy but their
    // initialization in hipfftMakePlanMany_internal is not very well defined
    // (always set to the out-of-place values, which may be wrong in case of
    // inverse in-place real transforms with implicitly-set default layouts...)
    std::vector<size_t> inStrides;
    std::vector<size_t> outStrides;
    size_t              iDist = 0;
    size_t              oDist = 0;

    size_t batch;

    double scale_factor = 1.0;

    // brick decomposition for multi-device transforms
    std::vector<hipfft_brick> inBricks;
    std::vector<hipfft_brick> outBricks;
    // hipFFT will decompose the problem across multiple devices in a
    // single process (i.e. via hipfftXtSetGPUs)
    bool singleProcMultiDevice = false;

    // multi-processing communicator
    rocfft_comm_type comm_type   = rocfft_comm_none;
    void*            comm_handle = nullptr;
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
                                            size_t                     dim,
                                            size_t*                    rm_lengths,
                                            hipfftIOType               iotype,
                                            size_t                     number_of_transforms,
                                            hipfft_ionembed_t<size_t>* user_ionembed,
                                            size_t                     user_idist,
                                            size_t                     user_odist,
                                            size_t*                    workSize)
{
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

    if(!plan || plan->initialized())
    {
        // plan initialization can be done only once in the plan's lifetime
        return HIPFFT_INVALID_PLAN;
    }

    rocfft_plan_description ip_forward_desc = nullptr;
    rocfft_plan_description op_forward_desc = nullptr;
    rocfft_plan_description ip_inverse_desc = nullptr;
    rocfft_plan_description op_inverse_desc = nullptr;
    rocfft_plan_description_create(&ip_forward_desc);
    rocfft_plan_description_create(&op_forward_desc);
    rocfft_plan_description_create(&ip_inverse_desc);
    rocfft_plan_description_create(&op_inverse_desc);

    plan->inLength.resize(dim);
    plan->outLength.resize(dim);
    std::reverse_copy(rm_lengths, rm_lengths + dim, plan->inLength.begin());
    std::reverse_copy(rm_lengths, rm_lengths + dim, plan->outLength.begin());
    const std::vector<size_t> rm_lengths_vec(rm_lengths, rm_lengths + dim);
    const std::vector<size_t> cm_lengths_vec(rm_lengths_vec.rbegin(), rm_lengths_vec.rend());

    if(iotype.is_real_to_complex())
        plan->outLength.front() = plan->outLength.front() / 2 + 1;
    else if(iotype.is_complex_to_real())
        plan->inLength.front() = plan->inLength.front() / 2 + 1;
    plan->batch = number_of_transforms;

    // copy the user's ionembed into the plan if there is one, use default otherwise
    plan->ionembed = !user_ionembed ? hipfft_ionembed_t<size_t>() : *user_ionembed;
    // NOTE: hipFFT ignores distance arguments if default layouts are used!
    const bool ignore_user_distances = !plan->ionembed.get_nembed(fft_io::fft_io_in)
                                       && !plan->ionembed.get_nembed(fft_io::fft_io_out);
    std::vector<size_t> i_strides, o_strides;
    size_t              inDist = 0, outDist = 0;
    for(auto dft_type : iotype.transform_types())
    {
        for(auto placement : {rocfft_placement_inplace, rocfft_placement_notinplace})
        {
            auto& plan_desc
                = placement == rocfft_placement_inplace
                      ? (iotype.is_forward(dft_type) ? ip_forward_desc : ip_inverse_desc)
                      : (iotype.is_forward(dft_type) ? op_forward_desc : op_inverse_desc);
            i_strides = plan->ionembed.as_generalized_strides(
                fft_io::fft_io_in,
                fft_transform_type_from_rocfft_transform_type(dft_type),
                fft_result_placement_from_rocfft_result_placement(placement),
                rm_lengths_vec);
            o_strides = plan->ionembed.as_generalized_strides(
                fft_io::fft_io_out,
                fft_transform_type_from_rocfft_transform_type(dft_type),
                fft_result_placement_from_rocfft_result_placement(placement),
                rm_lengths_vec);
            // rm -> cm:
            std::reverse(i_strides.begin(), i_strides.end());
            std::reverse(o_strides.begin(), o_strides.end());
            inDist  = !ignore_user_distances
                          ? user_idist
                          : default_distance(
                             fft_transform_type_from_rocfft_transform_type(dft_type),
                             fft_result_placement_from_rocfft_result_placement(placement),
                             fft_io::fft_io_in,
                             rm_lengths_vec,
                             number_of_transforms);
            outDist = !ignore_user_distances
                          ? user_odist
                          : default_distance(
                              fft_transform_type_from_rocfft_transform_type(dft_type),
                              fft_result_placement_from_rocfft_result_placement(placement),
                              fft_io::fft_io_out,
                              rm_lengths_vec,
                              number_of_transforms);

            ROC_FFT_CHECK_INVALID_VALUE(
                rocfft_plan_description_set_data_layout(plan_desc,
                                                        iotype.array_type(fft_io::fft_io_in),
                                                        iotype.array_type(fft_io::fft_io_out),
                                                        0,
                                                        0,
                                                        dim,
                                                        i_strides.data(),
                                                        inDist,
                                                        dim,
                                                        o_strides.data(),
                                                        outDist));
        }
    }
    // FIXME: inconsistent initialization below for inverse real in-place
    // with implicitly default data layouts (e.g., nullptr for inembed and onembed)
    // [members used in hipfftXtMemcpy thereafter]
    plan->inStrides  = i_strides;
    plan->outStrides = o_strides;
    plan->iDist      = inDist;
    plan->oDist      = outDist;

    // problem dimensions and strides are known, set up the bricks
    // for single-proc multi-GPU
    if(plan->singleProcMultiDevice)
        set_io_bricks(
            plan->inLength, plan->outLength, plan->batch, plan->inBricks, plan->outBricks);

    // create fields for the bricks
    if(!plan->inBricks.empty())
    {
        rocfft_field inField = nullptr;
        if(rocfft_field_create(&inField) != rocfft_status_success)
            throw std::runtime_error("input field create failed");

        for(const auto& brick : plan->inBricks)
        {
            rocfft_brick rbrick = nullptr;
            if(rocfft_brick_create(&rbrick,
                                   brick.field_lower.data(),
                                   brick.field_upper.data(),
                                   brick.brick_stride.data(),
                                   brick.field_lower.size(),
                                   brick.device)
               != rocfft_status_success)
                throw std::runtime_error("create input brick failed");

            if(rocfft_field_add_brick(inField, rbrick) != rocfft_status_success)
                throw std::runtime_error("add input brick failed");
            rocfft_brick_destroy(rbrick);
        }

        // inBricks are used for out-of-place transforms
        for(auto rocfft_desc : {op_forward_desc, op_inverse_desc})
        {
            rocfft_plan_description_add_infield(rocfft_desc, inField);
        }

        (void)rocfft_field_destroy(inField);
    }
    if(!plan->outBricks.empty())
    {
        rocfft_field outField = nullptr;
        if(rocfft_field_create(&outField) != rocfft_status_success)
            throw std::runtime_error("output field create failed");

        for(const auto& brick : plan->outBricks)
        {
            rocfft_brick rbrick = nullptr;
            if(rocfft_brick_create(&rbrick,
                                   brick.field_lower.data(),
                                   brick.field_upper.data(),
                                   brick.brick_stride.data(),
                                   brick.field_lower.size(),
                                   brick.device)
               != rocfft_status_success)
                throw std::runtime_error("create output brick failed");

            if(rocfft_field_add_brick(outField, rbrick) != rocfft_status_success)
                throw std::runtime_error("add output brick failed");
            rocfft_brick_destroy(rbrick);
        }

        // outBricks are used for both sides of in-place transforms,
        // and output of out-of-place transforms
        for(auto rocfft_desc : {ip_forward_desc, ip_inverse_desc})
        {
            rocfft_plan_description_add_infield(rocfft_desc, outField);
            rocfft_plan_description_add_outfield(rocfft_desc, outField);
        }
        for(auto rocfft_desc : {op_forward_desc, op_inverse_desc})
        {
            rocfft_plan_description_add_outfield(rocfft_desc, outField);
        }

        (void)rocfft_field_destroy(outField);
    }

    if(plan->scale_factor != 1.0)
    {
        for(auto rocfft_desc : {ip_forward_desc, op_forward_desc, ip_inverse_desc, op_inverse_desc})
        {
            rocfft_plan_description_set_scale_factor(rocfft_desc, plan->scale_factor);
        }
    }

    // set comm handle on the plans
    if(plan->comm_type != rocfft_comm_none)
    {
        for(auto rocfft_desc : {ip_forward_desc, op_forward_desc, ip_inverse_desc, op_inverse_desc})
        {
            rocfft_plan_description_set_comm(rocfft_desc, plan->comm_type, plan->comm_handle);
        }
    }

    // count the number of plans that got created - it's possible to
    // have parameters that are valid for out-place but not for
    // in-place, so some of these rocfft_plan_creates could
    // legitimately fail.
    unsigned int plans_created = 0;
    for(auto t : iotype.transform_types())
    {
        // in-place
        auto& ip_plan_ptr  = iotype.is_forward(t) ? plan->ip_forward : plan->ip_inverse;
        auto& ip_plan_desc = iotype.is_forward(t) ? ip_forward_desc : ip_inverse_desc;
        ROC_FFT_CHECK_PLAN_CREATE(ip_plan_ptr,
                                  plans_created,
                                  rocfft_placement_inplace,
                                  t,
                                  iotype.precision(),
                                  dim,
                                  cm_lengths_vec.data(),
                                  number_of_transforms,
                                  ip_plan_desc);
        // out-of-place
        auto& op_plan_ptr  = iotype.is_forward(t) ? plan->op_forward : plan->op_inverse;
        auto& op_plan_desc = iotype.is_forward(t) ? op_forward_desc : op_inverse_desc;
        ROC_FFT_CHECK_PLAN_CREATE(op_plan_ptr,
                                  plans_created,
                                  rocfft_placement_notinplace,
                                  t,
                                  iotype.precision(),
                                  dim,
                                  cm_lengths_vec.data(),
                                  number_of_transforms,
                                  op_plan_desc);
    }

    // if no plans got created, fail
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
            if(plan->workBuffer && plan->workBufferNeedsFree)
            {
                if(hipFree(plan->workBuffer) != hipSuccess)
                    return HIPFFT_ALLOC_FAILED;
            }
            if(hipMalloc(&plan->workBuffer, workBufferSize) != hipSuccess)
                return HIPFFT_ALLOC_FAILED;
            plan->workBufferNeedsFree = true;
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_set_work_buffer(
                plan->info, plan->workBuffer, workBufferSize));
        }
    }

    rocfft_plan_description_destroy(ip_forward_desc);
    rocfft_plan_description_destroy(op_forward_desc);
    rocfft_plan_description_destroy(ip_inverse_desc);
    rocfft_plan_description_destroy(op_inverse_desc);

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
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_create(&h->info));
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

    size_t lengths[2];
    lengths[0]                                      = nx;
    lengths[1]                                      = ny;
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

    if(plan->workBuffer && plan->workBufferNeedsFree)
    {
        if(hipFree(plan->workBuffer) != hipSuccess)
            throw std::runtime_error("hipFree(plan->workBuffer) failed");
    }
    plan->workBufferNeedsFree = false;
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
        throw HIPFFT_INVALID_PLAN;

    // NOTE: direction is IGNORED by hipFFT in case of plans for real transforms
    if(plan->type.is_real_to_complex())
    {
        return inplace ? plan->ip_forward : plan->op_forward;
    }
    else if(plan->type.is_complex_to_real())
    {
        return inplace ? plan->ip_inverse : plan->op_inverse;
    }
    else
    {
        if(direction != HIPFFT_FORWARD && direction != HIPFFT_BACKWARD)
            throw HIPFFT_INVALID_VALUE;
        return inplace ? (direction == HIPFFT_FORWARD ? plan->ip_forward : plan->ip_inverse)
                       : (direction == HIPFFT_FORWARD ? plan->op_forward : plan->op_inverse);
    }

    return nullptr;
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
    if(plan != nullptr)
    {
        if(plan->ip_forward != nullptr)
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->ip_forward));
        if(plan->op_forward != nullptr)
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->op_forward));
        if(plan->ip_inverse != nullptr)
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->ip_inverse));
        if(plan->op_inverse != nullptr)
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->op_inverse));

        if(plan->workBufferNeedsFree)
        {
            if(hipFree(plan->workBuffer) != hipSuccess)
                throw std::runtime_error("hipFree(plan->workBuffer) failed");
        }

        ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_destroy(plan->info));

        delete plan;
    }

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
    plan->inBricks.resize(static_cast<size_t>(count));
    plan->outBricks.resize(static_cast<size_t>(count));

    // but at this point we know devices, so record what the user
    // gave us
    for(size_t i = 0; i < static_cast<size_t>(count); ++i)
    {
        plan->inBricks[i].device  = gpus[i];
        plan->outBricks[i].device = gpus[i];
    }

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

    auto lib_desc = std::make_unique<hipLibXtDesc>();
    std::memset(lib_desc.get(), 0, sizeof(hipLibXtDesc));

    lib_desc->version       = 0;
    lib_desc->library       = HIPLIB_FORMAT_HIPFFT;
    lib_desc->subFormat     = format;
    lib_desc->libDescriptor = nullptr;

    auto xt_desc = std::make_unique<hipXtDesc>();
    std::memset(xt_desc.get(), 0, sizeof(hipXtDesc));
    xt_desc->version = 0;

    std::vector<hipfft_brick>* bricks           = nullptr;
    size_t                     bits_per_element = 0;

    switch(format)
    {
    case HIPFFT_XT_FORMAT_INPUT:
        bricks           = &plan->inBricks;
        bits_per_element = hipDataType_bits(plan->type.inputType);
        break;
    case HIPFFT_XT_FORMAT_OUTPUT:
        bricks           = &plan->outBricks;
        bits_per_element = hipDataType_bits(plan->type.outputType);
        break;
    case HIPFFT_XT_FORMAT_INPLACE:
        bricks           = &plan->outBricks;
        bits_per_element = std::max(hipDataType_bits(plan->type.inputType),
                                    hipDataType_bits(plan->type.outputType));
        break;
    default:
        return HIPFFT_NOT_IMPLEMENTED;
    }

    xt_desc->nGPUs = static_cast<int>(bricks->size());

    for(size_t i = 0; i < bricks->size(); ++i)
    {
        auto& brick = (*bricks)[i];

        rocfft_scoped_device dev(brick.device);

        xt_desc->GPUs[i] = brick.device;
        xt_desc->size[i] = brick.min_size * bits_per_element / 8;
        if(hipMalloc(&(xt_desc->data[i]), xt_desc->size[i]) != hipSuccess)
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

// collapse contiguous dimensions in the specified length + stride -
// user data might be split on any dimension so if we can simplify to
// just one split dimension and one contiguous dimension we can more
// easily map a XtMemcpy to a 2DMemcpy.
static void collapse_contiguous_dims(std::vector<size_t>& brick_length,
                                     std::vector<size_t>& brick_stride,
                                     std::vector<size_t>& field_stride)
{
    // go backwards from slowest to fastest dims
    for(size_t i = brick_length.size() - 1; i != 0; --i)
    {
        if(brick_length[i - 1] * brick_stride[i - 1] == brick_stride[i]
           && brick_length[i - 1] * field_stride[i - 1] == field_stride[i])
        {
            brick_length[i - 1] *= brick_length[i];
            brick_length.erase(brick_length.begin() + i);
            brick_stride.erase(brick_stride.begin() + i);
            field_stride.erase(field_stride.begin() + i);
        }
    }
    // we should be splitting on exactly one dimension, so we
    // should end up with at most two dimensions after
    // collapsing
    if(brick_length.size() > 2 || brick_stride.size() > 2 || field_stride.size() > 2)
        throw std::runtime_error("should have at most 2 dims after collapsing");
    // fastest dim is expected to be contiguous
    if(brick_stride.front() != 1 || field_stride.front() != 1)
        throw std::runtime_error("fastest dim not contiguous after collapsing");
}

hipfftResult hipfftXtMemcpy(hipfftHandle plan, void* dest, void* src, hipfftXtCopyType cptype)
try
{
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;
    if(!dest || !src || dest == src)
        return HIPFFT_INVALID_VALUE;

    // get pointer into buf, at the index pointed to by lower
    // assuming lengths are strided by stride
    auto offset_buffer = [](void*                      buf,
                            hipDataType                dtype,
                            const std::vector<size_t>& lower,
                            const std::vector<size_t>& stride) {
        auto offset_elems = std::inner_product(lower.begin(), lower.end(), stride.begin(), 0);

        return static_cast<void*>(static_cast<char*>(buf) + hipDataType_bytes(dtype, offset_elems));
    };

    auto brick_layout = [plan](int subFormat) -> const std::vector<hipfft_brick>& {
        switch(subFormat)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            return plan->inBricks;
        case HIPFFT_XT_FORMAT_OUTPUT:
        case HIPFFT_XT_FORMAT_INPLACE:
            return plan->outBricks;
        default:
            throw HIPFFT_INVALID_VALUE;
        }
    };

    switch(cptype)
    {
    case HIPFFT_COPY_HOST_TO_DEVICE:
    {
        // dest is a hipLibXtDesc
        auto destDesc = static_cast<hipLibXtDesc*>(dest);
        if(!destDesc->descriptor)
            return HIPFFT_INVALID_VALUE;

        std::vector<size_t> srcStride = plan->inStrides;
        srcStride.push_back(plan->iDist);
        for(size_t i = 0; i < static_cast<size_t>(destDesc->descriptor->nGPUs); ++i)
        {
            rocfft_scoped_device dev(destDesc->descriptor->GPUs[i]);

            const auto& brick = brick_layout(destDesc->subFormat)[i];

            auto brick_length = brick.length();
            auto brick_stride = brick.brick_stride;
            auto field_stride = srcStride;
            collapse_contiguous_dims(brick_length, brick_stride, field_stride);

            // if we can do a 1D memcpy, just do that
            if(brick_length.size() == 1)
            {
                if(hipMemcpy(destDesc->descriptor->data[i],
                             offset_buffer(src, plan->type.inputType, brick.field_lower, srcStride),
                             destDesc->descriptor->size[i],
                             hipMemcpyHostToDevice)
                   != hipSuccess)
                    return HIPFFT_INTERNAL_ERROR;
            }
            else
            {
                if(hipMemcpy2D(
                       destDesc->descriptor->data[i],
                       hipDataType_bytes(plan->type.inputType, brick_stride[1]),
                       offset_buffer(src, plan->type.inputType, brick.field_lower, srcStride),
                       hipDataType_bytes(plan->type.inputType, field_stride[1]),
                       hipDataType_bytes(plan->type.inputType, brick_length[0]),
                       brick_length[1],
                       hipMemcpyHostToDevice)
                   != hipSuccess)
                    return HIPFFT_INTERNAL_ERROR;
            }
        }
        return HIPFFT_SUCCESS;
    }
    case HIPFFT_COPY_DEVICE_TO_HOST:
    {
        // src is a hipLibXtDesc
        auto srcDesc = static_cast<const hipLibXtDesc*>(src);
        if(!srcDesc->descriptor)
            return HIPFFT_INVALID_VALUE;

        std::vector<size_t> destStride = plan->outStrides;
        destStride.push_back(plan->oDist);
        for(size_t i = 0; i < static_cast<size_t>(srcDesc->descriptor->nGPUs); ++i)
        {
            rocfft_scoped_device dev(srcDesc->descriptor->GPUs[i]);

            const auto& brick = brick_layout(srcDesc->subFormat)[i];

            auto brick_length = brick.length();
            auto brick_stride = brick.brick_stride;
            auto field_stride = destStride;
            collapse_contiguous_dims(brick_length, brick_stride, field_stride);

            // if we can do a 1D memcpy, just do that
            if(brick_length.size() == 1)
            {
                if(hipMemcpy(
                       offset_buffer(dest, plan->type.outputType, brick.field_lower, destStride),
                       srcDesc->descriptor->data[i],
                       srcDesc->descriptor->size[i],
                       hipMemcpyDeviceToHost)
                   != hipSuccess)
                    return HIPFFT_INTERNAL_ERROR;
            }
            else
            {
                if(hipMemcpy2D(
                       offset_buffer(dest, plan->type.outputType, brick.field_lower, destStride),
                       hipDataType_bytes(plan->type.outputType, field_stride[1]),
                       srcDesc->descriptor->data[i],
                       hipDataType_bytes(plan->type.outputType, brick_stride[1]),
                       hipDataType_bytes(plan->type.outputType, brick_length[0]),
                       brick_length[1],
                       hipMemcpyDeviceToHost)
                   != hipSuccess)
                    return HIPFFT_INTERNAL_ERROR;
            }
        }
        return HIPFFT_SUCCESS;
    }
    case HIPFFT_COPY_DEVICE_TO_DEVICE:
    {
        // src and dest are both hipLibXtDescs
        auto srcDesc  = static_cast<const hipLibXtDesc*>(src);
        auto destDesc = static_cast<hipLibXtDesc*>(dest);
        if(!srcDesc->descriptor || !destDesc->descriptor
           || srcDesc->descriptor->nGPUs != destDesc->descriptor->nGPUs)
            return HIPFFT_INVALID_VALUE;

        for(size_t i = 0; i < static_cast<size_t>(srcDesc->descriptor->nGPUs); ++i)
        {
            rocfft_scoped_device dev(srcDesc->descriptor->GPUs[i]);
            if(hipMemcpy(destDesc->descriptor->data[i],
                         srcDesc->descriptor->data[i],
                         srcDesc->descriptor->size[i],
                         hipMemcpyDeviceToDevice)
               != hipSuccess)
                return HIPFFT_INTERNAL_ERROR;
        }
        return HIPFFT_SUCCESS;
    }
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

static hipfftResult hipfftXtExecDescriptorBase(const rocfft_plan&           rplan,
                                               const rocfft_execution_info& rinfo,
                                               hipLibXtDesc*                input,
                                               hipLibXtDesc*                output)
{
    if(!rplan)
        return HIPFFT_INVALID_PLAN;
    if(!input || !output)
        return HIPFFT_INVALID_VALUE;

    const auto ret
        = rocfft_execute(rplan, input->descriptor->data, output->descriptor->data, rinfo);
    return ret == rocfft_status_success ? HIPFFT_SUCCESS : HIPFFT_EXEC_FAILED;
}

hipfftResult hipfftXtExecDescriptorC2C(hipfftHandle  plan,
                                       hipLibXtDesc* input,
                                       hipLibXtDesc* output,
                                       int           direction)
try
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;

    const bool inplace = input == output;
    const auto rplan   = get_exec_plan(plan, inplace, direction);

    return hipfftXtExecDescriptorBase(rplan, plan->info, input, output);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorR2C(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
try
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;

    const bool inplace = input == output;
    const auto rplan   = get_exec_plan(plan, inplace, HIPFFT_FORWARD);

    return hipfftXtExecDescriptorBase(rplan, plan->info, input, output);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorC2R(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
try
{
    if(!is_ready_for_execution<rocfft_precision_single>(plan))
        return HIPFFT_INVALID_PLAN;

    const bool inplace = input == output;
    const auto rplan   = get_exec_plan(plan, inplace, HIPFFT_BACKWARD);

    return hipfftXtExecDescriptorBase(rplan, plan->info, input, output);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorZ2Z(hipfftHandle  plan,
                                       hipLibXtDesc* input,
                                       hipLibXtDesc* output,
                                       int           direction)
try
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;

    const bool inplace = input == output;
    const auto rplan   = get_exec_plan(plan, inplace, direction);

    return hipfftXtExecDescriptorBase(rplan, plan->info, input, output);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorD2Z(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
try
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;

    const bool inplace = input == output;
    const auto rplan   = get_exec_plan(plan, inplace, HIPFFT_FORWARD);

    return hipfftXtExecDescriptorBase(rplan, plan->info, input, output);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptorZ2D(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output)
try
{
    if(!is_ready_for_execution<rocfft_precision_double>(plan))
        return HIPFFT_INVALID_PLAN;

    const bool inplace = input == output;
    const auto rplan   = get_exec_plan(plan, inplace, HIPFFT_BACKWARD);

    return hipfftXtExecDescriptorBase(rplan, plan->info, input, output);
}
catch(...)
{
    return handle_exception();
}

hipfftResult hipfftXtExecDescriptor(hipfftHandle  plan,
                                    hipLibXtDesc* input,
                                    hipLibXtDesc* output,
                                    int           direction)
try
{
    // any precision
    if(!plan || !plan->initialized())
        return HIPFFT_INVALID_PLAN;
    const bool inplace = input == output;
    const auto rplan   = get_exec_plan(plan, inplace, direction);

    return hipfftXtExecDescriptorBase(rplan, plan->info, input, output);
}
catch(...)
{
    return handle_exception();
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
    plan->inBricks.resize(1);
    plan->outBricks.resize(1);

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

    setBrick(plan->inBricks.front(), input_lower, input_upper, input_stride);
    setBrick(plan->outBricks.front(), output_lower, output_upper, output_stride);
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
