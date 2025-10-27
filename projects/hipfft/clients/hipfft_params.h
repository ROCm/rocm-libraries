// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPFFT_PARAMS_H
#define HIPFFT_PARAMS_H

#include <atomic>
#include <map>
#include <numeric>
#include <optional>

#include "../shared/client_except.h"
#include "../shared/concurrency.h"
#include "../shared/fft_params.h"
#include "../shared/hipfft_brick.h"
#include "hipfft/hipfft.h"
#include "hipfft/hipfftXt.h"
#include <random>

#ifdef HIPFFT_MPI_ENABLE
#include "hipfft/hipfftMp.h"
#include <mpi.h>
#endif

template <typename T,
          typename... Args,
          std::enable_if_t<std::is_integral_v<T> && (std::is_same_v<T, Args> && ...), bool> = true>
static void set_with_random_nonnegative_values(const std::string& token, T& val, Args&... args)
{
    // using a hash of the token as random seed to avoid
    // dependencies on externally-defined variables
    std::hash<std::string>           hasher;
    std::ranlux24_base               gen(hasher(token));
    std::uniform_int_distribution<T> dis(static_cast<T>(0), std::numeric_limits<T>::max());
    val = dis(gen);
    ((args = dis(gen)), ...);
    return;
}

inline fft_status fft_status_from_hipfftparams(const hipfftResult_t val)
{
    switch(val)
    {
    case HIPFFT_SUCCESS:
        return fft_status_success;
    case HIPFFT_INVALID_PLAN:
    case HIPFFT_ALLOC_FAILED:
        return fft_status_failure;
    case HIPFFT_INVALID_TYPE:
    case HIPFFT_INVALID_VALUE:
    case HIPFFT_INVALID_SIZE:
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
    case HIPFFT_INVALID_DEVICE:
    case HIPFFT_NOT_IMPLEMENTED:
    case HIPFFT_NOT_SUPPORTED:
        return fft_status_invalid_arg_value;
    case HIPFFT_INTERNAL_ERROR:
    case HIPFFT_EXEC_FAILED:
    case HIPFFT_SETUP_FAILED:
    case HIPFFT_UNALIGNED_DATA:
    case HIPFFT_PARSE_ERROR:
        return fft_status_failure;
    case HIPFFT_NO_WORKSPACE:
        return fft_status_invalid_work_buffer;
    default:
        return fft_status_failure;
    }
}

inline std::string hipfftResult_string(const hipfftResult_t val)
{
    switch(val)
    {
    case HIPFFT_SUCCESS:
        return "HIPFFT_SUCCESS (0)";
    case HIPFFT_INVALID_PLAN:
        return "HIPFFT_INVALID_PLAN (1)";
    case HIPFFT_ALLOC_FAILED:
        return "HIPFFT_ALLOC_FAILED (2)";
    case HIPFFT_INVALID_TYPE:
        return "HIPFFT_INVALID_TYPE (3)";
    case HIPFFT_INVALID_VALUE:
        return "HIPFFT_INVALID_VALUE (4)";
    case HIPFFT_INTERNAL_ERROR:
        return "HIPFFT_INTERNAL_ERROR (5)";
    case HIPFFT_EXEC_FAILED:
        return "HIPFFT_EXEC_FAILED (6)";
    case HIPFFT_SETUP_FAILED:
        return "HIPFFT_SETUP_FAILED (7)";
    case HIPFFT_INVALID_SIZE:
        return "HIPFFT_INVALID_SIZE (8)";
    case HIPFFT_UNALIGNED_DATA:
        return "HIPFFT_UNALIGNED_DATA (9)";
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
        return "HIPFFT_INCOMPLETE_PARAMETER_LIST (10)";
    case HIPFFT_INVALID_DEVICE:
        return "HIPFFT_INVALID_DEVICE (11)";
    case HIPFFT_PARSE_ERROR:
        return "HIPFFT_PARSE_ERROR (12)";
    case HIPFFT_NO_WORKSPACE:
        return "HIPFFT_NO_WORKSPACE (13)";
    case HIPFFT_NOT_IMPLEMENTED:
        return "HIPFFT_NOT_IMPLEMENTED (14)";
    case HIPFFT_NOT_SUPPORTED:
        return "HIPFFT_NOT_SUPPORTED (16)";
    default:
        return "invalid hipfftResult";
    }
}

class hipfft_params : public fft_params
{
public:
    // plan handles are pointers for rocFFT backend, and ints for cuFFT
#ifdef __HIP_PLATFORM_AMD__
    static constexpr hipfftHandle INVALID_PLAN_HANDLE = nullptr;
#else
    static constexpr hipfftHandle INVALID_PLAN_HANDLE = -1;
#endif

    hipfftHandle plan = INVALID_PLAN_HANDLE;
    // keep track of token to check when attempting to create new plan
    std::string current_token;

    // hipFFT has two ways to specify transform type - the hipfftType
    // enum, and separate hipDataType enums for input/output.
    // hipfftType has no way to express an fp16 transform, so
    // hipfft_transform_type will not be set in that case.
    std::optional<hipfftType> hipfft_transform_type;
    hipDataType               inputType  = HIP_C_32F;
    hipDataType               outputType = HIP_C_32F;

    int direction;

    std::vector<int> int_length;
    std::vector<int> int_inembed;
    std::vector<int> int_onembed;

    std::vector<long long int> ll_length;
    std::vector<long long int> ll_inembed;
    std::vector<long long int> ll_onembed;

    template <typename T>
    struct many_api_layout_args
    {
        T *input_embed, *output_embed;
        T  input_stride, output_stride, input_distance, output_distance;
    };

    struct hipLibXtDesc_deleter
    {
        void operator()(hipLibXtDesc* d)
        {
            hipfftXtFree(d);
        }
    };
    // allocated memory on devices for multi-GPU transforms - inplace
    // just uses xt_output
    std::unique_ptr<hipLibXtDesc, hipLibXtDesc_deleter> xt_input;
    std::unique_ptr<hipLibXtDesc, hipLibXtDesc_deleter> xt_output;

    // rocFFT brick decomposition for Xt memory - multi-GPU tests will
    // confirm that rocFFT's decomposition matches cuFFT's
    std::vector<hipfft_brick> xt_inBricks;
    std::vector<hipfft_brick> xt_outBricks;

    // backend library can write N worksize values for N GPUs, so
    // allocate a vector for that if necessary
    std::vector<size_t> auto_allocated_worksizes;
    // if auto_allocate == fft_auto_allocation_off, the hipFFT plan(s)
    // will be provided with externally-managed work area(s):
    static std::vector<gpubuf> externally_managed_workareas;

    size_t auto_allocated_extra_vram_footprint() const
    {
        return std::accumulate(auto_allocated_worksizes.begin(),
                               auto_allocated_worksizes.end(),
                               static_cast<size_t>(0));
    }

    static size_t externally_managed_extra_vram_footprint()
    {
        return std::accumulate(externally_managed_workareas.begin(),
                               externally_managed_workareas.end(),
                               static_cast<size_t>(0),
                               [](size_t total, const gpubuf& buf) { return total + buf.size(); });
    }

    bool is_preventing_auto_allocation_at_generation() const
    {
        if(auto_allocate != fft_auto_allocation_off)
            return false;
        // Let hipFFT sometimes auto-allocate nonetheless so that tests cover its
        // ability to free resources (allocated at generation) when/if some
        // externally-managed workarea(s) are provided after plan generation
        // Note: this member function must return the same result even if called
        // more than once by a given instance, it must be stable for any instance
        return std::hash<std::string>()(token()) % 2 == 1;
    }

    hipfft_params() = default;

    hipfft_params(const fft_params& p)
        : fft_params(p)
    {
    }
    hipfft_params(hipfft_params&& p) = default;
    hipfft_params& operator=(hipfft_params&& other) = default;

    ~hipfft_params()
    {
        free();
    };

    void free()
    {
        if(plan != INVALID_PLAN_HANDLE)
        {
            hipfftDestroy(plan);
            plan = INVALID_PLAN_HANDLE;
        }
        xt_input.reset();
        xt_output.reset();
    }

    size_t vram_footprint() override
    {
        size_t val = fft_params::vram_footprint();
        // auto-allocated plans fail here if not enough VRAM, skip these tests
        try
        {
            if(create_plan() != fft_status_success)
            {
                throw std::runtime_error("Plan creation or struct setup failed");
            }
        }
        catch(fft_params::work_buffer_alloc_failure& e)
        {
            val += auto_allocated_extra_vram_footprint();
            val += externally_managed_extra_vram_footprint();
            std::stringstream msg;
            msg << "Plan work buffer size (" << val << " bytes raw data) too large for device";
            throw ROCFFT_SKIP{msg.str()};
        }
        val += auto_allocated_extra_vram_footprint();
        val += externally_managed_extra_vram_footprint();
        return val;
    }

    fft_status setup_structs()
    {
        // set direction
        switch(transform_type)
        {
        case fft_transform_type_complex_forward:
        case fft_transform_type_real_forward:
            direction = HIPFFT_FORWARD;
            break;
        case fft_transform_type_complex_inverse:
        case fft_transform_type_real_inverse:
            direction = HIPFFT_BACKWARD;
            break;
        }

        // set i/o types and transform type
        switch(transform_type)
        {
        case fft_transform_type_complex_forward:
        case fft_transform_type_complex_inverse:
        {
            switch(precision)
            {
            case fft_precision_half:
                inputType  = HIP_C_16F;
                outputType = HIP_C_16F;
                hipfft_transform_type.reset();
                break;
            case fft_precision_single:
                inputType             = HIP_C_32F;
                outputType            = HIP_C_32F;
                hipfft_transform_type = HIPFFT_C2C;
                break;
            case fft_precision_double:
                inputType             = HIP_C_64F;
                outputType            = HIP_C_64F;
                hipfft_transform_type = HIPFFT_Z2Z;
                break;
            }
            break;
        }
        case fft_transform_type_real_forward:
        {
            switch(precision)
            {
            case fft_precision_half:
                inputType  = HIP_R_16F;
                outputType = HIP_C_16F;
                hipfft_transform_type.reset();
                break;
            case fft_precision_single:
                inputType             = HIP_R_32F;
                outputType            = HIP_C_32F;
                hipfft_transform_type = HIPFFT_R2C;
                break;
            case fft_precision_double:
                inputType             = HIP_R_64F;
                outputType            = HIP_C_64F;
                hipfft_transform_type = HIPFFT_D2Z;
                break;
            }
            break;
        }
        case fft_transform_type_real_inverse:
        {
            switch(precision)
            {
            case fft_precision_half:
                inputType  = HIP_C_16F;
                outputType = HIP_R_16F;
                hipfft_transform_type.reset();
                break;
            case fft_precision_single:
                inputType             = HIP_C_32F;
                outputType            = HIP_R_32F;
                hipfft_transform_type = HIPFFT_C2R;
                break;
            case fft_precision_double:
                inputType             = HIP_C_64F;
                outputType            = HIP_R_64F;
                hipfft_transform_type = HIPFFT_Z2D;
                break;
            }
            break;
        }
        default:
            throw std::runtime_error("Invalid transform type");
        }

        int_length.resize(dim());
        int_inembed.resize(dim());
        int_onembed.resize(dim());

        ll_length.resize(dim());
        ll_inembed.resize(dim());
        ll_onembed.resize(dim());
        switch(dim())
        {
        case 3:
            ll_inembed[2] = istride[1] / istride[2];
            ll_onembed[2] = ostride[1] / ostride[2];
            [[fallthrough]];
        case 2:
            ll_inembed[1] = istride[0] / istride[1];
            ll_onembed[1] = ostride[0] / ostride[1];
            [[fallthrough]];
        case 1:
            ll_inembed[0] = istride[dim() - 1];
            ll_onembed[0] = ostride[dim() - 1];
            break;
        default:
            throw std::runtime_error("Invalid dimension");
        }

        for(size_t i = 0; i < dim(); ++i)
        {
            ll_length[i]   = length[i];
            int_length[i]  = length[i];
            int_inembed[i] = ll_inembed[i];
            int_onembed[i] = ll_onembed[i];
        }
        // reset auto_allocated_worksizes
        auto_allocated_worksizes.resize(get_num_used_gpus());
        std::for_each(auto_allocated_worksizes.begin(),
                      auto_allocated_worksizes.end(),
                      [](decltype(auto_allocated_worksizes)::value_type& val) { val = 0; });

        hipfftResult ret = HIPFFT_SUCCESS;
        return fft_status_from_hipfftparams(ret);
    }

    fft_status create_plan() override
    {
        // check if we need to make a new plan
        if(current_token == token())
        {
            return fft_status_success;
        }
        else
        {
            if(plan != INVALID_PLAN_HANDLE)
            {
                hipfftDestroy(plan);
                plan = INVALID_PLAN_HANDLE;
            }
        }

        auto fft_ret = setup_structs();
        if(fft_ret != fft_status_success)
        {
            return fft_ret;
        }

        hipfftResult ret{HIPFFT_INTERNAL_ERROR};
        switch(get_create_type())
        {
        case PLAN_Nd:
        {
            ret = create_plan_Nd();
            break;
        }
        case PLAN_MANY:
        {
            ret = create_plan_many();
            break;
        }
        case CREATE_MAKE_PLAN_Nd:
        {
            ret = create_make_plan_Nd();
            break;
        }
        case CREATE_MAKE_PLAN_MANY:
        {
            ret = create_make_plan_many();
            break;
        }
        case CREATE_MAKE_PLAN_MANY64:
        {
            ret = create_make_plan_many64();
            break;
        }
        case CREATE_XT_MAKE_PLAN_MANY:
        {
            ret = create_xt_make_plan_many();
            break;
        }
        default:
        {
            throw std::runtime_error("no valid plan creation type");
        }
        }

        if(ret == HIPFFT_SUCCESS && auto_allocate == fft_auto_allocation_off)
        {
            ret = set_externally_managed_work_areas();
        }

        // hipFFT can fail plan creation due to allocation failure -
        // tests are expecting a specific exception in that case,
        // because the test was unable to run.  Doesn't mean the test
        // case failed.
        if(ret == HIPFFT_ALLOC_FAILED)
        {
            if(!final_attempt_at_plan_creation && externally_managed_extra_vram_footprint() > 0)
            {
                final_attempt_at_plan_creation = true;
                // device allocation(s) in externally_managed_workareas might be
                // larger than needed or even unnecessary for the instance of interest.
                // Free them up and try again before concluding.
                externally_managed_workareas.clear();
                return create_plan();
            }
            else
            {
                throw fft_params::work_buffer_alloc_failure(
                    "plan create failed due to allocation failure",
                    externally_managed_extra_vram_footprint()
                        + auto_allocated_extra_vram_footprint());
            }
        }

        // store token to check if plan was already made
        current_token = token();
        return fft_status_from_hipfftparams(ret);
    }

    hipfftResult_t set_stream(hipStream_t stream)
    {
        if(plan == INVALID_PLAN_HANDLE)
            throw std::runtime_error("Plan must be created before setting a desired stream");
        return hipfftSetStream(plan, stream);
    }

    void validate_fields() const override
    {
        validate_brick_volume();

        // multi-process only works with batch-1 FFTs, as hipFFT has
        // no place in the API to communicate batch indexes for
        // bricks
        if(mp_lib != fft_mp_lib_none && nbatch > 1)
            throw std::runtime_error("multi-process FFTs require batch-1");

        // if user provided decomposition
        if(!ifields.empty() || !ofields.empty())
        {
            // then library-decomposed multi-GPU must not also be requested
            if(multiGPU > 1)
                throw std::runtime_error(
                    "cannot request both library-decomposed GPU and user decomposition");

            // count bricks per rank
            std::map<int, size_t> rank_ibrick_count;
            std::map<int, size_t> rank_obrick_count;
            for(const auto& b : ifields.front().bricks)
                rank_ibrick_count[b.rank]++;
            for(const auto& b : ofields.front().bricks)
                rank_obrick_count[b.rank]++;

            // make sure there's only one input/output brick per rank
            auto count_is_one
                = [](const std::pair<int, size_t>& entry) { return entry.second == 1; };
            if(!std::all_of(rank_ibrick_count.begin(), rank_ibrick_count.end(), count_is_one)
               || !std::all_of(rank_obrick_count.begin(), rank_obrick_count.end(), count_is_one))
                throw std::runtime_error("multiple bricks per rank are not supported");

            // also ensure that each input brick maps to an output on same rank
            if(rank_ibrick_count != rank_obrick_count)
                throw std::runtime_error("input and output bricks do not match up");
        }
    }

    fft_status set_callbacks(void*  load_cb_host,
                             void*  load_cb_data,
                             void*  store_cb_host,
                             void*  store_cb_data,
                             size_t load_cb_shared_mem_bytes  = 0,
                             size_t store_cb_shared_mem_bytes = 0) override
    {
        if(run_callbacks)
        {
            if(!hipfft_transform_type)
                throw std::runtime_error("callbacks require a valid hipfftType");

            hipfftResult ret{HIPFFT_EXEC_FAILED};
            switch(*hipfft_transform_type)
            {
            case HIPFFT_R2C:
                ret = hipfftXtSetCallback(plan, &load_cb_host, HIPFFT_CB_LD_REAL, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_COMPLEX, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_LD_REAL, load_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_ST_COMPLEX, store_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_D2Z:
                ret = hipfftXtSetCallback(
                    plan, &load_cb_host, HIPFFT_CB_LD_REAL_DOUBLE, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_COMPLEX_DOUBLE, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_LD_REAL_DOUBLE, load_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_ST_COMPLEX_DOUBLE, store_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_C2R:
                ret = hipfftXtSetCallback(plan, &load_cb_host, HIPFFT_CB_LD_COMPLEX, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(plan, &store_cb_host, HIPFFT_CB_ST_REAL, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_LD_COMPLEX, load_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_ST_REAL, store_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_Z2D:
                ret = hipfftXtSetCallback(
                    plan, &load_cb_host, HIPFFT_CB_LD_COMPLEX_DOUBLE, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_REAL_DOUBLE, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_LD_COMPLEX_DOUBLE, load_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_ST_REAL_DOUBLE, store_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_C2C:
                ret = hipfftXtSetCallback(plan, &load_cb_host, HIPFFT_CB_LD_COMPLEX, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_COMPLEX, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_LD_COMPLEX, load_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_ST_COMPLEX, store_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_Z2Z:
                ret = hipfftXtSetCallback(
                    plan, &load_cb_host, HIPFFT_CB_LD_COMPLEX_DOUBLE, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_COMPLEX_DOUBLE, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_LD_COMPLEX_DOUBLE, load_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallbackSharedSize(
                    plan, HIPFFT_CB_ST_COMPLEX_DOUBLE, store_cb_shared_mem_bytes);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            default:
                throw std::runtime_error("Invalid execution type");
            }
        }
        return fft_status_success;
    }

    virtual fft_status execute(void** in, void** out) override
    {
        return execute(in[0], out[0]);
    };

    fft_status execute(void* ibuffer, void* obuffer)
    {
        hipfftResult ret{HIPFFT_EXEC_FAILED};

        // if we're doing multi-GPU, we need to use ExecDescriptor
        // methods to execute.
        if(multiGPU > 1)
        {
            // rotate between generic ExecDescriptor and specific
            // ExecDescriptorX2Y functions by hashing token (for
            // stability across reruns of test cases)
            //
            // the specific functions are only for the main transform
            // types expressible through the hipfftType enum
            bool generic_ExecDescriptor
                = !hipfft_transform_type || std::hash<std::string>()(token()) % 2;

            if(generic_ExecDescriptor)
            {
                ret = hipfftXtExecDescriptor(plan,
                                             placement == fft_placement_inplace ? xt_output.get()
                                                                                : xt_input.get(),
                                             xt_output.get(),
                                             direction);
            }
            else
            {
                switch(*hipfft_transform_type)
                {
                case HIPFFT_R2C:
                    ret = hipfftXtExecDescriptorR2C(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get());
                    break;
                case HIPFFT_C2R:
                    ret = hipfftXtExecDescriptorC2R(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get());
                    break;
                case HIPFFT_C2C:
                    ret = hipfftXtExecDescriptorC2C(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get(),
                        direction);
                    break;
                case HIPFFT_D2Z:
                    ret = hipfftXtExecDescriptorD2Z(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get());
                    break;
                case HIPFFT_Z2D:
                    ret = hipfftXtExecDescriptorZ2D(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get());
                    break;
                case HIPFFT_Z2Z:
                    ret = hipfftXtExecDescriptorZ2Z(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get(),
                        direction);
                }
            }
            return fft_status_from_hipfftparams(ret);
        }

        // otherwise, we have two ways to execute in hipFFT -
        // hipfftExecFOO and hipfftXtExec

        // Transforms that aren't supported by the hipfftType enum
        // require using the Xt method, but otherwise we hash the
        // token to decide how to execute this FFT.  we want test
        // cases to rotate between different execution APIs, but we also
        // need the choice of API to be stable across reruns of the
        // same test cases.
        if(!hipfft_transform_type || std::hash<std::string>()(token()) % 2)
        {
            ret = hipfftXtExec(plan, ibuffer, obuffer, direction);
        }
        else
        {
            try
            {
                switch(*hipfft_transform_type)
                {
                case HIPFFT_R2C:
                    ret = hipfftExecR2C(
                        plan,
                        (hipfftReal*)ibuffer,
                        (hipfftComplex*)(placement == fft_placement_inplace ? ibuffer : obuffer));
                    break;
                case HIPFFT_D2Z:
                    ret = hipfftExecD2Z(plan,
                                        (hipfftDoubleReal*)ibuffer,
                                        (hipfftDoubleComplex*)(placement == fft_placement_inplace
                                                                   ? ibuffer
                                                                   : obuffer));
                    break;
                case HIPFFT_C2R:
                    ret = hipfftExecC2R(
                        plan,
                        (hipfftComplex*)ibuffer,
                        (hipfftReal*)(placement == fft_placement_inplace ? ibuffer : obuffer));
                    break;
                case HIPFFT_Z2D:
                    ret = hipfftExecZ2D(plan,
                                        (hipfftDoubleComplex*)ibuffer,
                                        (hipfftDoubleReal*)(placement == fft_placement_inplace
                                                                ? ibuffer
                                                                : obuffer));
                    break;
                case HIPFFT_C2C:
                    ret = hipfftExecC2C(
                        plan,
                        (hipfftComplex*)ibuffer,
                        (hipfftComplex*)(placement == fft_placement_inplace ? ibuffer : obuffer),
                        direction);
                    break;
                case HIPFFT_Z2Z:
                    ret = hipfftExecZ2Z(plan,
                                        (hipfftDoubleComplex*)ibuffer,
                                        (hipfftDoubleComplex*)(placement == fft_placement_inplace
                                                                   ? ibuffer
                                                                   : obuffer),
                                        direction);
                    break;
                default:
                    throw std::runtime_error("Invalid execution type");
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
            }
            catch(...)
            {
                std::cerr << "unknown exception in execute(void* ibuffer, void* obuffer)"
                          << std::endl;
            }
        }
        return fft_status_from_hipfftparams(ret);
    }

    bool is_contiguous() const
    {
        // compute contiguous stride, dist and check that the actual
        // strides/dists match
        std::vector<size_t> contiguous_istride
            = compute_stride(ilength(),
                             {},
                             placement == fft_placement_inplace
                                 && transform_type == fft_transform_type_real_forward);
        std::vector<size_t> contiguous_ostride
            = compute_stride(olength(),
                             {},
                             placement == fft_placement_inplace
                                 && transform_type == fft_transform_type_real_inverse);
        if(istride != contiguous_istride || ostride != contiguous_ostride)
            return false;
        return compute_idist() == idist && compute_odist() == odist;
    }

    // stride is row-major like everything else in fft_params.  brick
    // indexes/strides are col-major because those would normally be
    // passed to rocFFT directly
    static bool xt_desc_matches_brick(const hostbuf&                   field,
                                      const std::vector<size_t>&       stride,
                                      size_t                           dist,
                                      const hipXtDesc*                 desc,
                                      const std::vector<hipfft_brick>& bricks,
                                      size_t                           elem_size,
                                      const char*                      dir)
    {
        // construct field stride that includes batch distance too, since
        // brick coordinates include it
        auto field_stride_cm = stride;
        std::reverse(field_stride_cm.begin(), field_stride_cm.end());
        field_stride_cm.push_back(dist);

        std::atomic<bool> compare_err = false;
        std::atomic<bool> runtime_err = false;

        std::vector<hostbuf> brick_hosts;
        brick_hosts.resize(bricks.size());

#ifdef _OPENMP
#pragma omp parallel for num_threads(rocfft_concurrency())
#endif
        for(size_t i = 0; i < bricks.size(); ++i)
        {
            // copy the ith brick back to host memory
            rocfft_scoped_device device(desc->GPUs[i]);
            hostbuf&             brick_host = brick_hosts[i];
            brick_host.alloc(desc->size[i]);
            if(hipMemcpy(brick_host.data(), desc->data[i], brick_host.size(), hipMemcpyDeviceToHost)
               != hipSuccess)
            {
                runtime_err = true;
                continue;
            }

            // convert to row-major
            auto brick_length_rm = bricks[i].length();
            std::reverse(brick_length_rm.begin(), brick_length_rm.end());

            // start at brick origin
            auto brick_idx_rm = brick_length_rm;
            std::fill(brick_idx_rm.begin(), brick_idx_rm.end(), 0);

            do
            {
                auto brick_idx_cm = brick_idx_rm;
                std::reverse(brick_idx_cm.begin(), brick_idx_cm.end());

                auto field_offset = bricks[i].field_offset(brick_idx_cm, field_stride_cm);
                auto brick_offset = bricks[i].brick_offset(brick_idx_cm);

                if(memcmp(brick_host.data_offset(brick_offset * elem_size),
                          field.data_offset(field_offset * elem_size),
                          elem_size)
                   != 0)
                {
                    compare_err = true;
                    break;
                }
            } while(increment_rowmajor(brick_idx_rm, brick_length_rm));
        }

        if(runtime_err)
            throw std::runtime_error("failed to memcpy brick back to host");
        return !compare_err;
    }

    // call the hipFFT APIs to distribute data to multiple GPUs
    void multi_gpu_prepare(std::vector<hostbuf>& cpu_input,
                           std::vector<gpubuf>&  ibuffer,
                           std::vector<void*>&   pibuffer,
                           std::vector<void*>&   pobuffer) override
    {
        if(multiGPU <= 1)
            return;

        // input data is on the device - copy it back to the host so
        // hipfftXtMemcpy can deal with it
        hostbuf input_host;
        input_host.alloc(ibuffer.front().size());
        if(hipMemcpy(input_host.data(),
                     ibuffer.front().data(),
                     ibuffer.front().size(),
                     hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("copy back to host failed");

        // allocate data on the multiple GPUs
        if(placement == fft_placement_inplace)
        {
            hipLibXtDesc* xt_tmp = nullptr;
            if(hipfftXtMalloc(plan, &xt_tmp, HIPFFT_XT_FORMAT_INPLACE) != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMalloc failed");
            xt_output.reset(xt_tmp);
            xt_tmp = nullptr;

            if(hipfftXtMemcpy(plan, xt_output.get(), input_host.data(), HIPFFT_COPY_HOST_TO_DEVICE)
               != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMemcpy failed");

            pibuffer.clear();
            std::copy_n(xt_output->descriptor->data,
                        xt_output->descriptor->nGPUs,
                        std::back_inserter(pibuffer));
            pobuffer.clear();
        }
        else
        {
            hipLibXtDesc* xt_tmp = nullptr;
            if(hipfftXtMalloc(plan, &xt_tmp, HIPFFT_XT_FORMAT_INPUT) != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMalloc failed");
            xt_input.reset(xt_tmp);
            xt_tmp = nullptr;

            if(hipfftXtMemcpy(plan, xt_input.get(), input_host.data(), HIPFFT_COPY_HOST_TO_DEVICE)
               != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMemcpy failed");
            if(hipfftXtMalloc(plan, &xt_tmp, HIPFFT_XT_FORMAT_OUTPUT) != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMalloc failed");
            xt_output.reset(xt_tmp);
            xt_tmp = nullptr;

            pibuffer.clear();
            std::copy_n(xt_input->descriptor->data,
                        xt_input->descriptor->nGPUs,
                        std::back_inserter(pibuffer));
            pobuffer.clear();
            std::copy_n(xt_output->descriptor->data,
                        xt_output->descriptor->nGPUs,
                        std::back_inserter(pobuffer));
        }

        // create bricks for this transform so we can confirm data layout
        hipLibXtDesc* compare_desc
            = placement == fft_placement_inplace ? xt_output.get() : xt_input.get();
        xt_inBricks.resize(compare_desc->descriptor->nGPUs);
        xt_outBricks.resize(compare_desc->descriptor->nGPUs);
        set_io_bricks(ilength_cm(), olength_cm(), nbatch, xt_inBricks, xt_outBricks);

        // check cufftXtMemcpy versus hipfft's implementation
        if(!xt_desc_matches_brick(input_host,
                                  istride,
                                  idist,
                                  compare_desc->descriptor,
                                  xt_inBricks,
                                  var_size<size_t>(precision, itype),
                                  "input"))
            throw std::runtime_error("Xt input does not match");
    }

    // call the hipFFT APIs to gather the data back from the multiple GPUs
    virtual void multi_gpu_finalize(std::vector<hostbuf>& gpu_output,
                                    std::vector<gpubuf>&  obuffer,
                                    std::vector<void*>&   pobuffer) override
    {
        if(multiGPU <= 1)
            return;

        // allocate a host buffer for hipFFTXtMemcpy's sake
        hostbuf output_host;
        output_host.alloc(obuffer.front().size());

        if(hipfftXtMemcpy(plan, output_host.data(), xt_output.get(), HIPFFT_COPY_DEVICE_TO_HOST)
           != HIPFFT_SUCCESS)
            throw std::runtime_error("hipfftXtMemcpy failed");

        // check cufftXtMemcpy versus hipfft's implementation
        if(placement == fft_placement_notinplace)
        {
            if(!xt_desc_matches_brick(output_host,
                                      ostride,
                                      odist,
                                      xt_output->descriptor,
                                      xt_outBricks,
                                      var_size<size_t>(precision, otype),
                                      "output"))
                throw std::runtime_error("Xt output does not match");
        }

        // copy final result back to device for comparison
        if(hipMemcpy(obuffer.front().data(),
                     output_host.data(),
                     obuffer.front().size(),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("finalizing hipMemcpy failed");

        pobuffer.clear();
        pobuffer.push_back(obuffer.front().data());
    }

private:
    // hipFFT provides multiple ways to create FFT plans:
    // - hipfftPlan1d/2d/3d (combined allocate + init for specific dim)
    // - hipfftPlanMany (combined allocate + init with dim as param)
    // - hipfftCreate + hipfftMakePlan1d/2d/3d (separate alloc + init for specific dim)
    // - hipfftCreate + hipfftMakePlanMany (separate alloc + init with dim as param)
    // - hipfftCreate + hipfftMakePlanMany64 (separate alloc + init with dim as param, 64-bit)
    // - hipfftCreate + hipfftXtMakePlanMany (separate alloc + init with separate i/o/exec types)
    //
    // Rotate through the choices for better test coverage.
    enum PlanCreateAPI
    {
        PLAN_Nd,
        PLAN_MANY,
        CREATE_MAKE_PLAN_Nd,
        CREATE_MAKE_PLAN_MANY,
        CREATE_MAKE_PLAN_MANY64,
        CREATE_XT_MAKE_PLAN_MANY,
    };

    // check that worksize estimates can be successfully queried with or without a valid plan
    hipfftResult_t check_worksize_estimate()
    {
        hipfftResult_t ret{HIPFFT_INTERNAL_ERROR};
        if(!hipfft_transform_type)
        {
            throw std::runtime_error("Estimating worksize requires a valid type of transform");
        }
        std::vector<size_t> worksize_estimate(get_num_used_gpus(), absurd_init_worksize_estimate);
        switch(get_create_type())
        {
        case CREATE_MAKE_PLAN_Nd:
        {
            switch(dim())
            {
            case 1:
                if(plan == INVALID_PLAN_HANDLE)
                    ret = hipfftEstimate1d(
                        int_length[0], *hipfft_transform_type, nbatch, worksize_estimate.data());
                else
                    ret = hipfftGetSize1d(plan,
                                          int_length[0],
                                          *hipfft_transform_type,
                                          nbatch,
                                          worksize_estimate.data());
                break;
            case 2:
                if(plan == INVALID_PLAN_HANDLE)
                    ret = hipfftEstimate2d(int_length[0],
                                           int_length[1],
                                           *hipfft_transform_type,
                                           worksize_estimate.data());
                else
                    ret = hipfftGetSize2d(plan,
                                          int_length[0],
                                          int_length[1],
                                          *hipfft_transform_type,
                                          worksize_estimate.data());
                break;
            case 3:
                if(plan == INVALID_PLAN_HANDLE)
                    ret = hipfftEstimate3d(int_length[0],
                                           int_length[1],
                                           int_length[2],
                                           *hipfft_transform_type,
                                           worksize_estimate.data());
                else
                    ret = hipfftGetSize3d(plan,
                                          int_length[0],
                                          int_length[1],
                                          int_length[2],
                                          *hipfft_transform_type,
                                          worksize_estimate.data());
                break;
            default:
                throw std::runtime_error("invalid dim");
            }
            break;
        }
        case CREATE_MAKE_PLAN_MANY:
        {
            auto layout_args = make_valid_layout_args_for_plan_many<int>();
            if(plan == INVALID_PLAN_HANDLE)
                ret = hipfftEstimateMany(dim(),
                                         int_length.data(),
                                         layout_args.input_embed,
                                         layout_args.input_stride,
                                         layout_args.input_distance,
                                         layout_args.output_embed,
                                         layout_args.output_stride,
                                         layout_args.output_distance,
                                         *hipfft_transform_type,
                                         nbatch,
                                         worksize_estimate.data());
            else
                ret = hipfftGetSizeMany(plan,
                                        dim(),
                                        int_length.data(),
                                        layout_args.input_embed,
                                        layout_args.input_stride,
                                        layout_args.input_distance,
                                        layout_args.output_embed,
                                        layout_args.output_stride,
                                        layout_args.output_distance,
                                        *hipfft_transform_type,
                                        nbatch,
                                        worksize_estimate.data());
            break;
        }
        case CREATE_MAKE_PLAN_MANY64:
        {
            if(plan == INVALID_PLAN_HANDLE)
            {
                // no direct equivalent in estimate-fetching APIs
                std::for_each(worksize_estimate.begin(),
                              worksize_estimate.end(),
                              [](decltype(worksize_estimate)::value_type& val) { val = 0; });
                ret = HIPFFT_SUCCESS;
            }
            else
            {
                auto layout_args = make_valid_layout_args_for_plan_many<long long>();
                ret              = hipfftGetSizeMany64(plan,
                                          dim(),
                                          ll_length.data(),
                                          layout_args.input_embed,
                                          layout_args.input_stride,
                                          layout_args.input_distance,
                                          layout_args.output_embed,
                                          layout_args.output_stride,
                                          layout_args.output_distance,
                                          *hipfft_transform_type,
                                          nbatch,
                                          worksize_estimate.data());
            }
            break;
        }
        case CREATE_XT_MAKE_PLAN_MANY:
        {
            if(plan == INVALID_PLAN_HANDLE)
            {
                // no direct equivalent in estimate-fetching APIs
                std::for_each(worksize_estimate.begin(),
                              worksize_estimate.end(),
                              [](decltype(worksize_estimate)::value_type& val) { val = 0; });
                ret = HIPFFT_SUCCESS;
            }
            else
            {
                auto executionType = get_xt_api_execution_type();
                auto layout_args   = make_valid_layout_args_for_plan_many<long long>();
                ret                = hipfftXtGetSizeMany(plan,
                                          dim(),
                                          ll_length.data(),
                                          layout_args.input_embed,
                                          layout_args.input_stride,
                                          layout_args.input_distance,
                                          inputType,
                                          layout_args.output_embed,
                                          layout_args.output_stride,
                                          layout_args.output_distance,
                                          outputType,
                                          nbatch,
                                          worksize_estimate.data(),
                                          executionType);
            }
            break;
        }
        case PLAN_Nd:
        case PLAN_MANY:
        default:
        {
            // should be indirectly disabled via get_create_type()
            return HIPFFT_INTERNAL_ERROR;
        }
        }
        // check that the value(s) of worksize_estimate were actually set, assuming that
        // setting a worksize_estimate equal to absurd_init_worksize_estimate by hipFFT
        // cannot be considered "correct".
        // Note: worksize_estimate value(s) are *not* guaranteed to be greater than or equal
        // to the actual value(s) of the work area(s), queriable after plan generation via
        // hipfftGetSize.
        if(ret == HIPFFT_SUCCESS)
        {
            // the estimate can't have any knowledge about the number of GPUs being used if
            // the plan wasn't created first
            const size_t num_values_to_check
                = plan == INVALID_PLAN_HANDLE ? 1 : worksize_estimate.size();
            for(size_t idx = 0; ret == HIPFFT_SUCCESS && idx < num_values_to_check; idx++)
            {
                ret = worksize_estimate[idx] != absurd_init_worksize_estimate
                          ? HIPFFT_SUCCESS
                          : HIPFFT_INTERNAL_ERROR;
            }
        }
        return ret;
    }

    // provide a work area to a successfully generated plan
    hipfftResult_t set_externally_managed_work_areas()
    {
        std::vector<size_t> req_workarea_sizes(get_num_used_gpus(), absurd_init_worksize_estimate);
        hipfftResult_t      ret = hipfftGetSize(plan, req_workarea_sizes.data());
        if(ret != HIPFFT_SUCCESS)
        {
            return ret;
        }
        else if(std::any_of(req_workarea_sizes.begin(),
                            req_workarea_sizes.end(),
                            [](const decltype(req_workarea_sizes)::value_type& val) {
                                return val == absurd_init_worksize_estimate;
                            }))
        {
            return HIPFFT_INTERNAL_ERROR;
        }
        // req_workarea_sizes are known and validated
        // check if the current externally_managed_workareas can be used as is or not
        if(externally_managed_workareas.size() < get_num_used_gpus())
            externally_managed_workareas.resize(get_num_used_gpus());
        std::vector<void*> workareas(get_num_used_gpus(), nullptr);
        for(auto workarea_idx = 0; workarea_idx < get_num_used_gpus(); workarea_idx++)
        {
            const auto req_size = req_workarea_sizes[workarea_idx];
            auto&      buf      = externally_managed_workareas[workarea_idx];
            if(buf.size() < req_size)
            {
                // too small, free and reallocate to meet current needs
                buf.free();
                if(buf.alloc(req_size) != hipSuccess)
                {
                    return HIPFFT_ALLOC_FAILED;
                }
            }
            workareas[workarea_idx] = buf.data();
        }
        if(get_num_used_gpus() > 1)
        {
            // TODO: enable below once hipfftXtSetWorkArea is enabled
#if(0)
            ret = hipfftXtSetWorkArea(plan, workareas.data);
#else
            throw unimplemented_exception(
                "No implementation support for externally-managed work areas with multi-gpu usage");
#endif
        }
        else
        {
            ret = hipfftSetWorkArea(plan, workareas[0]);
        }
        if(ret == HIPFFT_SUCCESS)
        {
            // the above "SetWorkArea" frees auto_allocated worksizes (if any)
            auto_allocated_worksizes.clear();
        }
        return ret;
    }

    // return true if we need to use hipFFT APIs that separate plan
    // allocation and plan init
    bool need_separate_create_make() const
    {
        // scale factor and multi-GPU and disabled auto-allocation need API
        // calls between create + init
        if(scale_factor != 1.0 || multiGPU > 1 || mp_lib != fft_mp_lib_none
           || auto_allocate == fft_auto_allocation_off)
            return true;
        return false;
    }

    template <
        typename T,
        std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, long long int>, bool> = true>
    many_api_layout_args<T> make_valid_layout_args_for_plan_many()
    {
        many_api_layout_args<T> ret;
        if constexpr(std::is_same_v<T, int>)
        {
            ret.input_embed  = int_inembed.data();
            ret.output_embed = int_onembed.data();
        }
        else
        {
            ret.input_embed  = ll_inembed.data();
            ret.output_embed = ll_onembed.data();
        }
        ret.input_stride    = static_cast<T>(istride.back());
        ret.output_stride   = static_cast<T>(ostride.back());
        ret.input_distance  = static_cast<T>(idist);
        ret.output_distance = static_cast<T>(odist);
        if(is_using_default_layout())
        {
            // If using a default layout, users can
            // (A) either set explicitly inembed, onembed, strides, and distances (like above);
            // (B) or use nullptr as arguments for inembed and onembed. Strides and
            //     distances are supposed to be ignored in that case.
            // --> choose randomly between either valid usage when a default layout is
            //     used, so that all possible valid use case scenarios are considered.
            const std::string test_token = token();
            int               randomizer;
            set_with_random_nonnegative_values(test_token, randomizer);
            if(randomizer % 2 == 0)
            {
                ret.input_embed  = nullptr;
                ret.output_embed = nullptr;
                // FIXME: negative values are not truly ignored for now.
                set_with_random_nonnegative_values(test_token,
                                                   ret.input_stride,
                                                   ret.output_stride,
                                                   ret.input_distance,
                                                   ret.output_distance);
            }
        }
        return ret;
    }

    // Not all plan options work with all creation types.  Return a
    // suitable plan creation type for the current FFT parameters.
    int get_create_type()
    {
        bool contiguous = is_contiguous();
        bool batched    = nbatch > 1;

        std::vector<PlanCreateAPI> allowed_apis;

        // half-precision requires XtMakePlanMany
        if(precision == fft_precision_half)
        {
            allowed_apis.push_back(CREATE_XT_MAKE_PLAN_MANY);
        }
        else
        {
            // separate alloc + init "Many" APIs are always allowed
            allowed_apis.push_back(CREATE_MAKE_PLAN_MANY);
            allowed_apis.push_back(CREATE_MAKE_PLAN_MANY64);
            allowed_apis.push_back(CREATE_XT_MAKE_PLAN_MANY);

            if(!need_separate_create_make())
                allowed_apis.push_back(PLAN_MANY);

            // non-many APIs are only allowed if FFT is contiguous, and
            // only the 1D API allows for batched FFTs.
            if(contiguous && (!batched || dim() == 1))
            {
                if(!need_separate_create_make())
                    allowed_apis.push_back(PLAN_Nd);
                allowed_apis.push_back(CREATE_MAKE_PLAN_Nd);
            }
        }

        // hash the token to decide how to create this FFT.  we want
        // test cases to rotate between different create APIs, but we
        // also need the choice of API to be stable across reruns of
        // the same test cases.
        return allowed_apis[std::hash<std::string>()(token()) % allowed_apis.size()];
    }

    // call hipfftPlan* functions
    hipfftResult_t create_plan_Nd()
    {
        auto ret = HIPFFT_INVALID_PLAN;
        switch(dim())
        {
        case 1:
            ret = hipfftPlan1d(&plan, int_length[0], *hipfft_transform_type, nbatch);
            break;
        case 2:
            ret = hipfftPlan2d(&plan, int_length[0], int_length[1], *hipfft_transform_type);
            break;
        case 3:
            ret = hipfftPlan3d(
                &plan, int_length[0], int_length[1], int_length[2], *hipfft_transform_type);
            break;
        default:
            throw std::runtime_error("invalid dim");
        }
        return ret;
    }
    hipfftResult_t create_plan_many()
    {
        auto layout_args = make_valid_layout_args_for_plan_many<int>();
        auto ret         = hipfftPlanMany(&plan,
                                  dim(),
                                  int_length.data(),
                                  layout_args.input_embed,
                                  layout_args.input_stride,
                                  layout_args.input_distance,
                                  layout_args.output_embed,
                                  layout_args.output_stride,
                                  layout_args.output_distance,
                                  *hipfft_transform_type,
                                  nbatch);
        return ret;
    }

    // call hipfftCreate + hipfftMake* functions, inserting calls to
    // relevant pre-Make APIs (scale factor, XtSetGPUs)
    hipfftResult_t create_with_pre_make()
    {
        hipfftResult_t ret{HIPFFT_INVALID_PLAN};
        if(auto_allocate == fft_auto_allocation_off)
        {
            ret = check_worksize_estimate(); // read worksize estimate before plan creation
            if(ret != HIPFFT_SUCCESS)
                return ret;
        }
        ret = hipfftCreate(&plan);
        if(ret != HIPFFT_SUCCESS)
            return ret;
        if(scale_factor != 1.0)
        {
            ret = hipfftExtPlanScaleFactor(plan, scale_factor);
            if(ret != HIPFFT_SUCCESS)
                return ret;
        }
        if(multiGPU > 1)
        {
            int deviceCount = 0;
            if(hipGetDeviceCount(&deviceCount) != hipSuccess)
                throw std::runtime_error("hipGetDeviceCount failed");

            // ensure that users request less than or equal to the total number of devices
            if(static_cast<int>(multiGPU) > deviceCount)
                throw std::runtime_error("not enough devices for requested multi-gpu computation!");

            std::vector<int> GPUs(multiGPU);
            std::iota(GPUs.begin(), GPUs.end(), 0);
            ret = hipfftXtSetGPUs(plan, static_cast<int>(multiGPU), GPUs.data());
            if(ret != HIPFFT_SUCCESS)
                return ret;
        }
        if(mp_lib == fft_mp_lib_mpi)
        {
#ifdef HIPFFT_MPI_ENABLE
            ret = hipfftMpAttachComm(plan, HIPFFT_COMM_MPI, mp_comm);
            if(ret != HIPFFT_SUCCESS)
                return ret;

            int mpi_rank = 0;
            MPI_Comm_rank(*static_cast<MPI_Comm*>(mp_comm), &mpi_rank);

            const auto& in_bricks  = ifields.front().bricks;
            const auto& out_bricks = ofields.front().bricks;

            // find the input/output brick for this rank
            auto curr_rank_brick = [mpi_rank](const fft_brick& b) { return b.rank == mpi_rank; };
            auto in_brick  = std::find_if(in_bricks.begin(), in_bricks.end(), curr_rank_brick);
            auto out_brick = std::find_if(out_bricks.begin(), out_bricks.end(), curr_rank_brick);

            if(in_brick != in_bricks.end() && out_brick != out_bricks.end())
            {
                std::vector<long long int> input_lower;
                std::vector<long long int> input_upper;
                std::vector<long long int> output_lower;
                std::vector<long long int> output_upper;
                std::vector<long long int> input_stride;
                std::vector<long long int> output_stride;

                // convert brick info to long long int for hipFFT
                auto convert_intvec
                    = [](const std::vector<size_t>& in, std::vector<long long int>& out) {
                          // start with index 1 because hipFFT only wants to be
                          // told about FFT dimensions, not batch dimension
                          for(size_t i = 1; i < in.size(); ++i)
                              out.push_back(static_cast<long long int>(in[i]));
                      };
                convert_intvec(in_brick->lower, input_lower);
                convert_intvec(in_brick->upper, input_upper);
                convert_intvec(out_brick->lower, output_lower);
                convert_intvec(out_brick->upper, output_upper);
                convert_intvec(in_brick->stride, input_stride);
                convert_intvec(out_brick->stride, output_stride);

                ret = hipfftXtSetDistribution(plan,
                                              static_cast<int>(dim()),
                                              input_lower.data(),
                                              input_upper.data(),
                                              output_lower.data(),
                                              output_upper.data(),
                                              input_stride.data(),
                                              output_stride.data());
            }
            if(ret != HIPFFT_SUCCESS)
                return ret;
#else
            throw std::runtime_error("MPI is not enabled");
#endif
        }
        if(auto_allocate == fft_auto_allocation_off)
        {
            ret = check_worksize_estimate(); // read worksize estimate again after plan creation
            if(ret != HIPFFT_SUCCESS)
                return ret;
        }
        if(is_preventing_auto_allocation_at_generation())
        {
            ret = hipfftSetAutoAllocation(plan, 0);
        }

        return ret;
    }

    hipfftResult_t create_make_plan_Nd()
    {
        auto ret = create_with_pre_make();

        if(ret != HIPFFT_SUCCESS)
            return ret;
        // do not register plan's worksizes as "auto-allocated" if auto-allocation was explicitly prevented
        std::vector<size_t> tmp_worksize(get_num_used_gpus());
        size_t*             worksize_ptr = is_preventing_auto_allocation_at_generation()
                                               ? tmp_worksize.data()
                                               : auto_allocated_worksizes.data();
        switch(dim())
        {
        case 1:
            return hipfftMakePlan1d(
                plan, int_length[0], *hipfft_transform_type, nbatch, worksize_ptr);
        case 2:
            return hipfftMakePlan2d(
                plan, int_length[0], int_length[1], *hipfft_transform_type, worksize_ptr);
        case 3:
            return hipfftMakePlan3d(plan,
                                    int_length[0],
                                    int_length[1],
                                    int_length[2],
                                    *hipfft_transform_type,
                                    worksize_ptr);
        default:
            throw std::runtime_error("invalid dim");
        }
    }

    hipfftResult_t create_make_plan_many()
    {
        auto ret = create_with_pre_make();
        if(ret != HIPFFT_SUCCESS)
            return ret;
        // do not register plan's worksizes as "auto-allocated" if auto-allocation was explicitly prevented
        std::vector<size_t> tmp_worksize(get_num_used_gpus());
        size_t*             worksize_ptr = is_preventing_auto_allocation_at_generation()
                                               ? tmp_worksize.data()
                                               : auto_allocated_worksizes.data();
        auto                layout_args  = make_valid_layout_args_for_plan_many<int>();
        return hipfftMakePlanMany(plan,
                                  dim(),
                                  int_length.data(),
                                  layout_args.input_embed,
                                  layout_args.input_stride,
                                  layout_args.input_distance,
                                  layout_args.output_embed,
                                  layout_args.output_stride,
                                  layout_args.output_distance,
                                  *hipfft_transform_type,
                                  nbatch,
                                  worksize_ptr);
    }

    hipfftResult_t create_make_plan_many64()
    {
        auto ret = create_with_pre_make();
        if(ret != HIPFFT_SUCCESS)
            return ret;

        // do not register plan's worksizes as "auto-allocated" if auto-allocation was explicitly prevented
        std::vector<size_t> tmp_worksize(get_num_used_gpus());
        size_t*             worksize_ptr = is_preventing_auto_allocation_at_generation()
                                               ? tmp_worksize.data()
                                               : auto_allocated_worksizes.data();
        auto                layout_args  = make_valid_layout_args_for_plan_many<long long int>();
        return hipfftMakePlanMany64(plan,
                                    dim(),
                                    ll_length.data(),
                                    layout_args.input_embed,
                                    layout_args.input_stride,
                                    layout_args.input_distance,
                                    layout_args.output_embed,
                                    layout_args.output_stride,
                                    layout_args.output_distance,
                                    *hipfft_transform_type,
                                    nbatch,
                                    worksize_ptr);
    }

    hipDataType get_xt_api_execution_type() const
    {
        // execution type is always complex, matching the precision
        // of the transform
        // Initializing as double by default
        hipDataType ret = HIP_C_64F;
        switch(precision)
        {
        case fft_precision_half:
            ret = HIP_C_16F;
            break;
        case fft_precision_single:
            ret = HIP_C_32F;
            break;
        case fft_precision_double:
            ret = HIP_C_64F;
            break;
        default:
            throw std::runtime_error("Invalid precision");
        }
        return ret;
    }

    hipfftResult_t create_xt_make_plan_many()
    {
        auto ret = create_with_pre_make();
        if(ret != HIPFFT_SUCCESS)
            return ret;

        // do not register plan's worksizes as "auto-allocated" if auto-allocation was explicitly prevented
        std::vector<size_t> tmp_worksize(get_num_used_gpus());
        size_t*             worksize_ptr  = is_preventing_auto_allocation_at_generation()
                                                ? tmp_worksize.data()
                                                : auto_allocated_worksizes.data();
        auto                executionType = get_xt_api_execution_type();
        auto                layout_args   = make_valid_layout_args_for_plan_many<long long int>();
        return hipfftXtMakePlanMany(plan,
                                    dim(),
                                    ll_length.data(),
                                    layout_args.input_embed,
                                    layout_args.input_stride,
                                    layout_args.input_distance,
                                    inputType,
                                    layout_args.output_embed,
                                    layout_args.output_stride,
                                    layout_args.output_distance,
                                    outputType,
                                    nbatch,
                                    worksize_ptr,
                                    executionType);
    }
    static constexpr size_t absurd_init_worksize_estimate  = std::numeric_limits<size_t>::max();
    bool                    final_attempt_at_plan_creation = false;

    size_t get_num_used_gpus() const
    {
        return multiGPU > 1 ? multiGPU : 1;
    };
};

#endif
