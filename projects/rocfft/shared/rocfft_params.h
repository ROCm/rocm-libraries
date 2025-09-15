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

#ifndef ROCFFT_PARAMS_H
#define ROCFFT_PARAMS_H

#include "../shared/fft_params.h"
#include "../shared/gpubuf.h"
#include "../shared/precision_type.h"
#include "rocfft/rocfft.h"

#ifdef ROCFFT_MPI_ENABLE
#include <mpi.h>
#endif

#ifdef WIN32
#include <windows.h>
// psapi.h requires windows.h to be included first
#include <psapi.h>
#else
#include <dlfcn.h>
#include <link.h>
#endif

// Return the string of the rocfft_status code
static std::string rocfft_status_to_string(const rocfft_status ret)
{
    switch(ret)
    {
    case rocfft_status_success:
        return "rocfft_status_success";
    case rocfft_status_failure:
        return "rocfft_status_failure";
    case rocfft_status_invalid_arg_value:
        return "rocfft_status_invalid_arg_value";
    case rocfft_status_invalid_dimensions:
        return "rocfft_status_invalid_dimensions";
    case rocfft_status_invalid_array_type:
        return "rocfft_status_invalid_array_type";
    case rocfft_status_invalid_strides:
        return "rocfft_status_invalid_strides";
    case rocfft_status_invalid_distance:
        return "rocfft_status_invalid_distance";
    case rocfft_status_invalid_offset:
        return "rocfft_status_invalid_offset";
    case rocfft_status_invalid_work_buffer:
        return "rocfft_status_invalid_work_buffer";
    default:
        throw std::runtime_error("unknown rocfft_status");
    }
}

inline fft_status fft_status_from_rocfftparams(const rocfft_status val)
{
    switch(val)
    {
    case rocfft_status_success:
        return fft_status_success;
    case rocfft_status_failure:
        return fft_status_failure;
    case rocfft_status_invalid_arg_value:
        return fft_status_invalid_arg_value;
    case rocfft_status_invalid_dimensions:
        return fft_status_invalid_dimensions;
    case rocfft_status_invalid_array_type:
        return fft_status_invalid_array_type;
    case rocfft_status_invalid_strides:
        return fft_status_invalid_strides;
    case rocfft_status_invalid_distance:
        return fft_status_invalid_distance;
    case rocfft_status_invalid_offset:
        return fft_status_invalid_offset;
    case rocfft_status_invalid_work_buffer:
        return fft_status_invalid_work_buffer;
    default:
        throw std::runtime_error("Invalid status");
    }
}

inline rocfft_precision rocfft_precision_from_fftparams(const fft_precision val)
{
    switch(val)
    {
    case fft_precision_single:
        return rocfft_precision_single;
    case fft_precision_double:
        return rocfft_precision_double;
    case fft_precision_half:
        return rocfft_precision_half;
    default:
        throw std::runtime_error("Invalid precision");
    }
}

inline fft_precision fft_precision_from_rocfft_precision(const rocfft_precision val)
{
    switch(val)
    {
    case rocfft_precision_single:
        return fft_precision_single;
    case rocfft_precision_double:
        return fft_precision_double;
    case rocfft_precision_half:
        return fft_precision_half;
    default:
        throw std::runtime_error("Invalid precision");
    }
}

inline rocfft_array_type rocfft_array_type_from_fftparams(const fft_array_type val)
{
    switch(val)
    {
    case fft_array_type_complex_interleaved:
        return rocfft_array_type_complex_interleaved;
    case fft_array_type_complex_planar:
        return rocfft_array_type_complex_planar;
    case fft_array_type_real:
        return rocfft_array_type_real;
    case fft_array_type_hermitian_interleaved:
        return rocfft_array_type_hermitian_interleaved;
    case fft_array_type_hermitian_planar:
        return rocfft_array_type_hermitian_planar;
    case fft_array_type_unset:
        return rocfft_array_type_unset;
    }
    return rocfft_array_type_unset;
}

inline fft_array_type fft_array_type_from_rocfft_array_type(const rocfft_array_type val)
{
    switch(val)
    {
    case rocfft_array_type_complex_interleaved:
        return fft_array_type_complex_interleaved;
    case rocfft_array_type_complex_planar:
        return fft_array_type_complex_planar;
    case rocfft_array_type_real:
        return fft_array_type_real;
    case rocfft_array_type_hermitian_interleaved:
        return fft_array_type_hermitian_interleaved;
    case rocfft_array_type_hermitian_planar:
        return fft_array_type_hermitian_planar;
    case rocfft_array_type_unset:
        return fft_array_type_unset;
    }
    return fft_array_type_unset;
}

inline rocfft_transform_type rocfft_transform_type_from_fftparams(const fft_transform_type val)
{
    switch(val)
    {
    case fft_transform_type_complex_forward:
        return rocfft_transform_type_complex_forward;
    case fft_transform_type_complex_inverse:
        return rocfft_transform_type_complex_inverse;
    case fft_transform_type_real_forward:
        return rocfft_transform_type_real_forward;
    case fft_transform_type_real_inverse:
        return rocfft_transform_type_real_inverse;
    default:
        throw std::runtime_error("Invalid transform type");
    }
}

inline fft_transform_type
    fft_transform_type_from_rocfft_transform_type(const rocfft_transform_type val)
{
    switch(val)
    {
    case rocfft_transform_type_complex_forward:
        return fft_transform_type_complex_forward;
    case rocfft_transform_type_complex_inverse:
        return fft_transform_type_complex_inverse;
    case rocfft_transform_type_real_forward:
        return fft_transform_type_real_forward;
    case rocfft_transform_type_real_inverse:
        return fft_transform_type_real_inverse;
    default:
        throw std::runtime_error("Invalid transform type");
    }
}

inline rocfft_result_placement
    rocfft_result_placement_from_fftparams(const fft_result_placement val)
{
    switch(val)
    {
    case fft_placement_inplace:
        return rocfft_placement_inplace;
    case fft_placement_notinplace:
        return rocfft_placement_notinplace;
    default:
        throw std::runtime_error("Invalid result placement");
    }
}

inline fft_result_placement
    fft_result_placement_from_rocfft_result_placement(const rocfft_result_placement val)
{
    switch(val)
    {
    case rocfft_placement_inplace:
        return fft_placement_inplace;
    case rocfft_placement_notinplace:
        return fft_placement_notinplace;
    default:
        throw std::runtime_error("Invalid result placement");
    }
}

template <typename Funcs>
class rocfft_params_base : public fft_params
{
    Funcs rocfft;

public:
    rocfft_plan             plan = nullptr;
    rocfft_execution_info   info = nullptr;
    rocfft_plan_description desc = nullptr;
    gpubuf_t<void>          wbuffer;
    size_t                  workbuffersize = 0;

    explicit rocfft_params_base() = default;

    explicit rocfft_params_base(Funcs&& funcs)
        : rocfft(std::move(funcs)){};

    explicit rocfft_params_base(const fft_params& p, Funcs&& funcs)
        : fft_params(p)
        , rocfft(std::move(funcs))
    {
    }

    explicit rocfft_params_base(const fft_params& p)
        : fft_params(p)
    {
    }

    rocfft_params_base(const rocfft_params_base&) = delete;
    rocfft_params_base& operator=(const rocfft_params_base&) = delete;

    // move construct
    rocfft_params_base(rocfft_params_base&&) = default;
    rocfft_params_base& operator=(rocfft_params_base&&) = default;

    ~rocfft_params_base()
    {
        free();
    };

    void setup() override
    {
        rocfft.setup();
    }
    void cleanup() override
    {
        rocfft.cleanup();
    }

    void free()
    {
        if(plan != nullptr)
        {
            rocfft.plan_destroy(plan);
            plan = nullptr;
        }
        if(info != nullptr)
        {
            rocfft.execution_info_destroy(info);
            info = nullptr;
        }
        if(desc != nullptr)
        {
            rocfft.plan_description_destroy(desc);
            desc = nullptr;
        }
        wbuffer.free();
    }

    void validate_fields() const override
    {
        // rocFFT requires explicit bricks and cannot decide on
        // multi-GPU decomposition itself
        if(multiGPU > 1)
            throw std::runtime_error("library-decomposed multi-GPU is unsupported");

        validate_brick_volume();
    }

    rocfft_precision get_rocfft_precision()
    {
        return rocfft_precision_from_fftparams(precision);
    }

    size_t vram_footprint() override
    {
        size_t val = fft_params::vram_footprint();
        if(setup_structs() != fft_status_success)
        {
            throw std::runtime_error("Struct setup failed");
        }
        val += workbuffersize;

        return val;
    }

    // Convert the generic fft_field structure to a rocfft_field
    // structure that can be passed to rocFFT.  In particular, we need
    // to convert from row-major to column-major.
    rocfft_field fft_field_to_rocfft_field(const fft_field& f) const
    {
        rocfft_field rfield = nullptr;
        if(f.bricks.empty())
            return rfield;

        if(rocfft.field_create(&rfield) != rocfft_status_success)
            throw std::runtime_error("rocfft_field_create failed");
        for(const auto& b : f.bricks)
        {
            // if this is an MPI transform, only tell the current rank
            // about bricks for that rank
            if(mp_lib == fft_mp_lib_mpi)
            {
#ifdef ROCFFT_MPI_ENABLE
                int mpi_rank = 0;
                MPI_Comm_rank(*static_cast<MPI_Comm*>(mp_comm), &mpi_rank);

                if(mpi_rank != b.rank)
                    continue;
#else
                throw std::runtime_error("MPI is not enabled");
#endif
            }

            // rocFFT wants column-major bricks and fft_params stores
            // row-major
            std::vector<size_t> lower_cm;
            std::copy(b.lower.rbegin(), b.lower.rend(), std::back_inserter(lower_cm));
            std::vector<size_t> upper_cm;
            std::copy(b.upper.rbegin(), b.upper.rend(), std::back_inserter(upper_cm));
            std::vector<size_t> stride_cm;
            std::copy(b.stride.rbegin(), b.stride.rend(), std::back_inserter(stride_cm));

            rocfft_brick rbrick = nullptr;
            if(rocfft.brick_create(&rbrick,
                                   lower_cm.data(), // field_lower
                                   upper_cm.data(), // field_upper
                                   stride_cm.data(), // brick_stride
                                   lower_cm.size(), // dim
                                   b.device) // deviceID
               != rocfft_status_success)
                throw std::runtime_error("rocfft_brick_create failed");

            if(rocfft.field_add_brick(rfield, rbrick) != rocfft_status_success)
                throw std::runtime_error("rocfft_field_add_brick failed");

            rocfft.brick_destroy(rbrick);
        }
        return rfield;
    }

    fft_status setup_structs()
    {
        rocfft_status fft_status = rocfft_status_success;
        if(desc == nullptr)
        {
            rocfft.plan_description_create(&desc);
            if(fft_status != rocfft_status_success)
                return fft_status_from_rocfftparams(fft_status);

            fft_status
                = rocfft.plan_description_set_data_layout(desc,
                                                          rocfft_array_type_from_fftparams(itype),
                                                          rocfft_array_type_from_fftparams(otype),
                                                          ioffset.data(),
                                                          ooffset.data(),
                                                          istride_cm().size(),
                                                          istride_cm().data(),
                                                          idist,
                                                          ostride_cm().size(),
                                                          ostride_cm().data(),
                                                          odist);
            if(fft_status != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_plan_description_set_data_layout failed");
            }

            if(scale_factor != 1.0)
            {
                fft_status = rocfft.plan_description_set_scale_factor(desc, scale_factor);
                if(fft_status != rocfft_status_success)
                {
                    throw std::runtime_error("rocfft_plan_description_set_scale_factor failed");
                }
            }

            for(const auto& ifield : ifields)
            {
                rocfft_field infield = fft_field_to_rocfft_field(ifield);
                if(rocfft.plan_description_add_infield(desc, infield) != rocfft_status_success)
                    throw std::runtime_error("rocfft_description_add_infield failed");
                rocfft.field_destroy(infield);
            }

            for(const auto& ofield : ofields)
            {
                rocfft_field outfield = fft_field_to_rocfft_field(ofield);
                if(rocfft.plan_description_add_outfield(desc, outfield) != rocfft_status_success)
                    throw std::runtime_error("rocfft_description_add_outfield failed");
                rocfft.field_destroy(outfield);
            }

            if(mp_lib == fft_mp_lib_mpi)
            {
                if(rocfft.plan_description_set_comm(desc, rocfft_comm_mpi, mp_comm)
                   != rocfft_status_success)
                    throw std::runtime_error("rocfft_plan_description_set_comm failed");
            }
        }

        if(plan == nullptr)
        {
            fft_status = rocfft.plan_create(&plan,
                                            rocfft_result_placement_from_fftparams(placement),
                                            rocfft_transform_type_from_fftparams(transform_type),
                                            get_rocfft_precision(),
                                            length_cm().size(),
                                            length_cm().data(),
                                            nbatch,
                                            desc);
            if(fft_status != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_plan_create failed");
            }
        }

        if(info == nullptr)
        {
            fft_status = rocfft.execution_info_create(&info);
            if(fft_status != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_execution_info_create failed");
            }
        }

        fft_status = rocfft.plan_get_work_buffer_size(plan, &workbuffersize);
        if(fft_status != rocfft_status_success)
        {
            throw std::runtime_error("rocfft_plan_get_work_buffer_size failed");
        }

        return fft_status_from_rocfftparams(fft_status);
    }

    fft_status create_plan() override
    {
        fft_status ret = setup_structs();
        if(ret != fft_status_success)
        {
            return ret;
        }
        // default behavior is to feed rocfft with a work area if it needs one
        if(workbuffersize > 0 && auto_allocate != fft_auto_allocation_on)
        {
            hipError_t hip_status = hipSuccess;
            hip_status            = wbuffer.alloc(workbuffersize);
            if(hip_status != hipSuccess)
            {
                std::ostringstream oss;
                oss << "work buffer allocation failed (" << workbuffersize << " requested)";
                size_t mem_free  = 0;
                size_t mem_total = 0;
                hip_status       = hipMemGetInfo(&mem_free, &mem_total);
                if(hip_status == hipSuccess)
                {
                    oss << "free vram: " << mem_free << " total vram: " << mem_total;
                }
                else
                {
                    oss << "hipMemGetInfo also failed";
                }
                throw work_buffer_alloc_failure(oss.str(), workbuffersize);
            }

            auto rocret
                = rocfft.execution_info_set_work_buffer(info, wbuffer.data(), workbuffersize);
            if(rocret != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_execution_info_set_work_buffer failed");
            }
        }

        return ret;
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
            auto roc_status = rocfft.execution_info_set_load_callback(
                info, &load_cb_host, &load_cb_data, load_cb_shared_mem_bytes);
            if(roc_status != rocfft_status_success)
                return fft_status_from_rocfftparams(roc_status);

            roc_status = rocfft.execution_info_set_store_callback(
                info, &store_cb_host, &store_cb_data, store_cb_shared_mem_bytes);
            if(roc_status != rocfft_status_success)
                return fft_status_from_rocfftparams(roc_status);
        }
        return fft_status_success;
    }

    fft_status execute(void** in, void** out) override
    {
        auto ret = rocfft.execute(plan, in, out, info);
        return fft_status_from_rocfftparams(ret);
    }

    // scatter data to multiple GPUs and adjust I/O buffers to match
    virtual void multi_gpu_prepare(std::vector<hostbuf>& cpu_input,
                                   std::vector<gpubuf>&  ibuffer,
                                   std::vector<void*>&   pibuffer,
                                   std::vector<void*>&   pobuffer) override
    {
        auto alloc_fields = [&](const fft_params::fft_field& field,
                                fft_array_type               array_type,
                                std::vector<void*>&          pbuffer,
                                bool                         copy_input) {
            if(field.bricks.empty())
                return;

            // we have a field defined, clear the list of buffers as
            // we'll be allocating new ones for each brick
            pbuffer.clear();

            const size_t elem_size_bytes = var_size<size_t>(precision, array_type);

            for(const auto& b : field.bricks)
            {
                const size_t brick_size_elems = compute_ptrdiff(b.length(), b.stride, 0, 0);
                const size_t brick_size_bytes = brick_size_elems * elem_size_bytes;

                // set device for the alloc, but we want to return to the
                // default device as the source of a following memcpy
                {
                    rocfft_scoped_device dev(b.device);
                    multi_gpu_data.emplace_back();
                    if(multi_gpu_data.back().alloc(brick_size_bytes) != hipSuccess)
                        throw std::runtime_error("device allocation failure");
                    pbuffer.push_back(multi_gpu_data.back().data());
                }

                if(copy_input)
                {
                    // get this brick's starting offset in the field
                    const size_t brick_offset_elems = b.lower_field_offset(istride, idist);

                    // transpose input data to the brick's shape in
                    // host memory, then memcpy

                    // alloc a host-side brick that's the right shape
                    std::vector<hostbuf> host_brick(1);
                    host_brick.front().alloc(brick_size_bytes);

                    std::vector<size_t> istride_with_batch{idist};
                    std::copy(
                        istride.begin(), istride.end(), std::back_inserter(istride_with_batch));

                    copy_buffers(cpu_input,
                                 host_brick,
                                 b.length(),
                                 1,
                                 precision,
                                 array_type,
                                 istride_with_batch,
                                 0,
                                 array_type,
                                 b.stride,
                                 0,
                                 {brick_offset_elems},
                                 {0});

                    // memcpy the transposed brick to the device
                    if(hipMemcpy(pbuffer.back(),
                                 host_brick.front().data(),
                                 brick_size_bytes,
                                 hipMemcpyHostToDevice)
                       != hipSuccess)
                    {
                        throw std::runtime_error("hipMemcpy failure");
                    }
                }
            }

            // if we copied the input to all the other devices, and
            // this is an out-of-place transform, we no longer
            // need the original input
            if(copy_input && placement == fft_placement_notinplace)
                ibuffer.clear();
        };

        // assume one input, one output field for simple cases
        if(!ifields.empty())
            alloc_fields(ifields.front(), itype, pibuffer, true);
        if(!ofields.empty())
        {
            if(!ifields.empty() && placement == fft_placement_inplace)
                pobuffer = pibuffer;
            else
                alloc_fields(ofields.front(), otype, pobuffer, false);
        }
    }

    // when preparing for multi-GPU transform, we need to allocate data
    // on each GPU.  This vector remembers all of those allocations.
    std::vector<gpubuf> multi_gpu_data;

    // gather data after multi-GPU FFT for verification
    void multi_gpu_finalize(std::vector<hostbuf>& gpu_output,
                            std::vector<gpubuf>&  obuffer,
                            std::vector<void*>&   pobuffer) override
    {
        if(ofields.empty())
            return;

        const size_t elem_size_bytes = var_size<size_t>(precision, otype);

        for(size_t i = 0; i < pobuffer.size(); ++i)
        {
            const auto& b = ofields.front().bricks[i];

            const size_t brick_size_elems = compute_ptrdiff(b.length(), b.stride, 0, 0);
            const size_t brick_size_bytes = brick_size_elems * elem_size_bytes;

            // get this brick's starting offset in the field
            const size_t brick_offset_elems = b.lower_field_offset(ostride, odist);

            // switch device to where we're copying from
            rocfft_scoped_device dev(b.device);

            // copy the brick to host, then copy to output
            std::vector<hostbuf> host_brick(1);
            host_brick.front().alloc(brick_size_bytes);
            if(hipMemcpy(
                   host_brick.front().data(), pobuffer[i], brick_size_bytes, hipMemcpyDeviceToHost)
               != hipSuccess)
            {
                throw std::runtime_error("hipMemcpy failed");
            }

            std::vector<size_t> ostride_with_batch{odist};
            std::copy(ostride.begin(), ostride.end(), std::back_inserter(ostride_with_batch));

            copy_buffers(host_brick,
                         gpu_output,
                         b.length(),
                         1,
                         precision,
                         otype,
                         b.stride,
                         0,
                         otype,
                         ostride_with_batch,
                         0,
                         {0},
                         {brick_offset_elems});
        }
        // set pobuffer back to a single-device transform
        pobuffer.clear();
        pobuffer.push_back(obuffer.front().data());
    }

private:
    // return the dimension indexes that a set of bricks is splitting up
    static std::vector<size_t> get_split_dimensions(const fft_params::fft_field& f,
                                                    const std::vector<size_t>&   length_with_batch)
    {
        std::vector<size_t> splitDims;
        for(size_t dimIdx = 0; dimIdx < length_with_batch.size(); ++dimIdx)
        {
            // if bricks are all same length as this dim's actual length,
            // they're not splitting on this dimension.
            if(std::all_of(f.bricks.begin(), f.bricks.end(), [&](const fft_params::fft_brick& b) {
                   return b.length()[dimIdx] == length_with_batch[dimIdx];
               }))
                continue;

            splitDims.push_back(dimIdx);
        }
        if(splitDims.empty())
        {
            throw std::runtime_error("could not find a split dimension");
        }
        // We're only prepared to handle 2 splits.  If there's a single
        // split, we can manage each split with a single 2D memcpy. With 2
        // splits we can do one 2D memcpy per batch.
        if(splitDims.size() > 2)
        {
            throw std::runtime_error("too many split dimensions");
        }
        return splitDims;
    }
};

#define ROCFFT_API_WRAP(func)                            \
    template <typename... Arg>                           \
    rocfft_status func(Arg&&... arg) const               \
    {                                                    \
        return rocfft_##func(std::forward<Arg>(arg)...); \
    }

struct rocfft_funcs
{
    ROCFFT_API_WRAP(brick_create);
    ROCFFT_API_WRAP(brick_destroy);
    ROCFFT_API_WRAP(cleanup);
    ROCFFT_API_WRAP(execute);
    ROCFFT_API_WRAP(execution_info_create);
    ROCFFT_API_WRAP(execution_info_destroy);
    ROCFFT_API_WRAP(execution_info_set_load_callback);
    ROCFFT_API_WRAP(execution_info_set_store_callback);
    ROCFFT_API_WRAP(execution_info_set_work_buffer);
    ROCFFT_API_WRAP(field_add_brick);
    ROCFFT_API_WRAP(field_create);
    ROCFFT_API_WRAP(field_destroy);
    ROCFFT_API_WRAP(plan_create);
    ROCFFT_API_WRAP(plan_description_add_infield);
    ROCFFT_API_WRAP(plan_description_add_outfield);
    ROCFFT_API_WRAP(plan_description_create);
    ROCFFT_API_WRAP(plan_description_destroy);
    ROCFFT_API_WRAP(plan_description_set_comm);
    ROCFFT_API_WRAP(plan_description_set_data_layout);
    ROCFFT_API_WRAP(plan_description_set_scale_factor);
    ROCFFT_API_WRAP(plan_destroy);
    ROCFFT_API_WRAP(plan_get_work_buffer_size);
    ROCFFT_API_WRAP(setup);
};

#define ROCFFT_DYNA_API_WRAP(func) decltype(&rocfft_##func) func = nullptr

#define ROCFFT_DYNA_API_LOAD(func) \
    func = reinterpret_cast<decltype(&rocfft_##func)>(lib_symbol(lib, "rocfft_" #func))

struct dyna_rocfft_funcs
{
#ifdef WIN32
    typedef HMODULE ROCFFT_LIB;
#else
    typedef void* ROCFFT_LIB;
#endif

    // Load the rocfft library
    static ROCFFT_LIB lib_load(const std::string& path)
    {
#ifdef WIN32
        return LoadLibraryA(path.c_str());
#else
        return dlopen(path.c_str(), RTLD_LAZY);
#endif
    }

    // Return a string describing the error loading rocfft
    static const char* lib_load_error()
    {
#ifdef WIN32
        // just return the error number
        static std::string error_str;
        error_str = std::to_string(GetLastError());
        return error_str.c_str();
#else
        return dlerror();
#endif
    }

    // Get symbol from rocfft lib
    static void* lib_symbol(ROCFFT_LIB libhandle, const char* sym)
    {
#ifdef WIN32
        return reinterpret_cast<void*>(GetProcAddress(libhandle, sym));
#else
        return dlsym(libhandle, sym);
#endif
    }

    static void lib_close(ROCFFT_LIB libhandle)
    {
        if(!libhandle)
            return;
#ifdef WIN32
        FreeLibrary(libhandle);
#else
        dlclose(libhandle);
#endif
    }

    ROCFFT_LIB lib = nullptr;
    ROCFFT_DYNA_API_WRAP(brick_create);
    ROCFFT_DYNA_API_WRAP(brick_destroy);
    ROCFFT_DYNA_API_WRAP(cleanup);
    ROCFFT_DYNA_API_WRAP(execute);
    ROCFFT_DYNA_API_WRAP(execution_info_create);
    ROCFFT_DYNA_API_WRAP(execution_info_destroy);
    ROCFFT_DYNA_API_WRAP(execution_info_set_load_callback);
    ROCFFT_DYNA_API_WRAP(execution_info_set_store_callback);
    ROCFFT_DYNA_API_WRAP(execution_info_set_work_buffer);
    ROCFFT_DYNA_API_WRAP(field_add_brick);
    ROCFFT_DYNA_API_WRAP(field_create);
    ROCFFT_DYNA_API_WRAP(field_destroy);
    ROCFFT_DYNA_API_WRAP(plan_create);
    ROCFFT_DYNA_API_WRAP(plan_description_add_infield);
    ROCFFT_DYNA_API_WRAP(plan_description_add_outfield);
    ROCFFT_DYNA_API_WRAP(plan_description_create);
    ROCFFT_DYNA_API_WRAP(plan_description_destroy);
    ROCFFT_DYNA_API_WRAP(plan_description_set_comm);
    ROCFFT_DYNA_API_WRAP(plan_description_set_data_layout);
    ROCFFT_DYNA_API_WRAP(plan_description_set_scale_factor);
    ROCFFT_DYNA_API_WRAP(plan_destroy);
    ROCFFT_DYNA_API_WRAP(plan_get_work_buffer_size);
    ROCFFT_DYNA_API_WRAP(setup);

    dyna_rocfft_funcs(const std::string& path)
        : path(path)
    {
        lib = lib_load(path);
        if(!lib)
            throw std::runtime_error(lib_load_error());

        ROCFFT_DYNA_API_LOAD(brick_create);
        ROCFFT_DYNA_API_LOAD(brick_destroy);
        ROCFFT_DYNA_API_LOAD(cleanup);
        ROCFFT_DYNA_API_LOAD(execute);
        ROCFFT_DYNA_API_LOAD(execution_info_create);
        ROCFFT_DYNA_API_LOAD(execution_info_destroy);
        ROCFFT_DYNA_API_LOAD(execution_info_set_load_callback);
        ROCFFT_DYNA_API_LOAD(execution_info_set_store_callback);
        ROCFFT_DYNA_API_LOAD(execution_info_set_work_buffer);
        ROCFFT_DYNA_API_LOAD(field_add_brick);
        ROCFFT_DYNA_API_LOAD(field_create);
        ROCFFT_DYNA_API_LOAD(field_destroy);
        ROCFFT_DYNA_API_LOAD(plan_create);
        ROCFFT_DYNA_API_LOAD(plan_description_add_infield);
        ROCFFT_DYNA_API_LOAD(plan_description_add_outfield);
        ROCFFT_DYNA_API_LOAD(plan_description_create);
        ROCFFT_DYNA_API_LOAD(plan_description_destroy);
        ROCFFT_DYNA_API_LOAD(plan_description_set_comm);
        ROCFFT_DYNA_API_LOAD(plan_description_set_data_layout);
        ROCFFT_DYNA_API_LOAD(plan_description_set_scale_factor);
        ROCFFT_DYNA_API_LOAD(plan_destroy);
        ROCFFT_DYNA_API_LOAD(plan_get_work_buffer_size);
        ROCFFT_DYNA_API_LOAD(setup);
    }

    ~dyna_rocfft_funcs()
    {
        lib_close(lib);
        lib = nullptr;
    }

    // copy not allowed
    dyna_rocfft_funcs(const dyna_rocfft_funcs&) = delete;
    dyna_rocfft_funcs& operator=(const dyna_rocfft_funcs&) = delete;

    void swap(dyna_rocfft_funcs& other)
    {
        std::swap(this->lib, other.lib);
        std::swap(this->brick_create, other.brick_create);
        std::swap(this->brick_destroy, other.brick_destroy);
        std::swap(this->cleanup, other.cleanup);
        std::swap(this->execute, other.execute);
        std::swap(this->execution_info_create, other.execution_info_create);
        std::swap(this->execution_info_destroy, other.execution_info_destroy);
        std::swap(this->execution_info_set_load_callback, other.execution_info_set_load_callback);
        std::swap(this->execution_info_set_store_callback, other.execution_info_set_store_callback);
        std::swap(this->execution_info_set_work_buffer, other.execution_info_set_work_buffer);
        std::swap(this->field_add_brick, other.field_add_brick);
        std::swap(this->field_create, other.field_create);
        std::swap(this->field_destroy, other.field_destroy);
        std::swap(this->plan_create, other.plan_create);
        std::swap(this->plan_description_add_infield, other.plan_description_add_infield);
        std::swap(this->plan_description_add_outfield, other.plan_description_add_outfield);
        std::swap(this->plan_description_create, other.plan_description_create);
        std::swap(this->plan_description_destroy, other.plan_description_destroy);
        std::swap(this->plan_description_set_comm, other.plan_description_set_comm);
        std::swap(this->plan_description_set_data_layout, other.plan_description_set_data_layout);
        std::swap(this->plan_description_set_scale_factor, other.plan_description_set_scale_factor);
        std::swap(this->plan_destroy, other.plan_destroy);
        std::swap(this->plan_get_work_buffer_size, other.plan_get_work_buffer_size);
        std::swap(this->setup, other.setup);

        this->path.swap(other.path);
    }

    // move is allowed
    dyna_rocfft_funcs(dyna_rocfft_funcs&& other)
    {
        other.swap(*this);
    }
    dyna_rocfft_funcs& operator=(dyna_rocfft_funcs&& other)
    {
        other.swap(*this);
        return *this;
    }

    std::string path;

private:
    dyna_rocfft_funcs() = default;
};

typedef rocfft_params_base<rocfft_funcs>      rocfft_params;
typedef rocfft_params_base<dyna_rocfft_funcs> dyna_rocfft_params;
#endif
