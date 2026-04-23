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

#include "../shared/fft_enums.h"
#include "../shared/fft_params.h"
#include "../shared/gpubuf.h"
#include "../shared/precision_type.h"
#include "../shared/rocfft_hip.h"
#include "rocfft/rocfft.h"
#include "rocfft_enums_vs_fft_enums.h"

#ifdef ROCFFT_MPI_ENABLE
#include <mpi.h>
#endif

#ifdef _WIN32
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

template <typename Funcs>
class rocfft_params_base : public fft_params
{
    Funcs rocfft;

public:
    rocfft_plan             plan = nullptr;
    rocfft_execution_info   info = nullptr;
    rocfft_plan_description desc = nullptr;
    std::vector<gpubuf>     wbuffers;
    std::vector<size_t>     workbuffersizes;

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
        wbuffers.clear();
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

    std::vector<size_t> vram_footprint() override
    {
        auto footprint = io_vram_footprint();
        if(setup_structs() != fft_status_success)
        {
            throw std::runtime_error("Struct setup failed");
        }

        // add work buffer sizes returned by library
        for(size_t i = 0; i < footprint.size(); ++i)
        {
            footprint[i] += workbuffersizes[i];
        }
        return footprint;
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
        const auto proc_rank = get_process_rank();
        for(const auto& b : f.bricks)
        {
            // if this is an MPI transform, only tell the current rank
            // about bricks for that rank
            if(proc_rank != b.rank)
                continue;

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

            const bool test_default_strides_and_dist
                = is_using_default_layout() && std::hash<std::string>()(token()) % 2 == 1;

            fft_status = rocfft.plan_description_set_data_layout(
                desc,
                rocfft_array_type_from_fftparams(itype),
                rocfft_array_type_from_fftparams(otype),
                test_default_strides_and_dist ? nullptr : ioffset.data(),
                test_default_strides_and_dist ? nullptr : ooffset.data(),
                test_default_strides_and_dist ? 0 : istride_cm().size(),
                test_default_strides_and_dist ? nullptr : istride_cm().data(),
                test_default_strides_and_dist ? 0 : idist,
                test_default_strides_and_dist ? 0 : ostride_cm().size(),
                test_default_strides_and_dist ? nullptr : ostride_cm().data(),
                test_default_strides_and_dist ? 0 : odist);
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

        // Set work buffers for all HIP devices
        const int ndevices = rocfft_scoped_device::device_count();
        workbuffersizes.resize(ndevices);
        wbuffers.resize(ndevices);
        for(int device = 0; device < ndevices; ++device)
        {
            rocfft_scoped_device dev(device);
            fft_status = rocfft.plan_get_work_buffer_size(plan, workbuffersizes.data() + device);
            if(fft_status != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_plan_get_work_buffer_size failed");
            }
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
        bool need_workbuffers = std::any_of(
            workbuffersizes.begin(), workbuffersizes.end(), [](size_t s) { return s > 0; });
        if(need_workbuffers && auto_allocate != fft_auto_allocation_on)
        {
            const int ndevices = rocfft_scoped_device::device_count();
            for(int device = 0; device < ndevices; ++device)
            {
                if(workbuffersizes[device] == 0)
                    continue;

                rocfft_scoped_device dev(device);

                hipError_t hip_status = hipSuccess;
                hip_status            = wbuffers[device].alloc(workbuffersizes[device]);
                if(hip_status != hipSuccess)
                {
                    std::ostringstream oss;
                    oss << "work buffer allocation failed ("
                        << byte_size_to_str(workbuffersizes[device]) << " requested)";
                    oss << "\n" << device_memory_accountant::singleton().get_details(device);
                    throw work_buffer_alloc_failure(oss.str(), workbuffersizes[device]);
                }

                auto rocret = rocfft.execution_info_set_work_buffer(
                    info, wbuffers[device].data(), workbuffersizes[device]);
                if(rocret != rocfft_status_success)
                {
                    throw std::runtime_error("rocfft_execution_info_set_work_buffer failed");
                }
            }
        }

        return ret;
    }

    // Return the number of expected callback entries for supplied
    // fields.
    size_t expected_callback_count(const std::vector<fft_field>& fields)
    {
        // If fields are not specified, we consider the input or
        // output to have a single brick (and thus expect a single
        // callback entry)
        if(fields.empty())
            return 1;

        const int mpi_rank = get_process_rank();

        // count the number of bricks on this rank
        size_t expected_callbacks = 0;
        for(const auto& f : fields)
        {
            for(const auto& b : f.bricks)
            {
                if(b.rank == mpi_rank)
                    ++expected_callbacks;
            }
        }
        return expected_callbacks;
    }

    fft_status set_callbacks(std::vector<void*>* load_cb_func,
                             std::vector<void*>* load_cb_data,
                             std::vector<void*>* store_cb_func,
                             std::vector<void*>* store_cb_data,
                             size_t              load_cb_shared_mem_bytes  = 0,
                             size_t              store_cb_shared_mem_bytes = 0) override
    {
        if(run_callbacks)
        {
            auto expected_load_cb_count  = expected_callback_count(ifields);
            auto expected_store_cb_count = expected_callback_count(ofields);
            check_callback_vec(load_cb_func, expected_load_cb_count, true);
            check_callback_vec(load_cb_data, expected_load_cb_count, false);
            check_callback_vec(store_cb_func, expected_store_cb_count, true);
            check_callback_vec(store_cb_data, expected_store_cb_count, false);

            auto roc_status = rocfft.execution_info_set_load_callback(
                info,
                load_cb_func ? load_cb_func->data() : nullptr,
                load_cb_data ? load_cb_data->data() : nullptr,
                load_cb_shared_mem_bytes);
            if(roc_status != rocfft_status_success)
                return fft_status_from_rocfftparams(roc_status);

            roc_status = rocfft.execution_info_set_store_callback(
                info,
                store_cb_func ? store_cb_func->data() : nullptr,
                store_cb_data ? store_cb_data->data() : nullptr,
                store_cb_shared_mem_bytes);
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

    void multi_gpu_prepare(std::vector<hostbuf>& input_data_host,
                           std::vector<gpubuf>& /* input_data_gpu (unused) */,
                           std::vector<void*>& mgpu_ibuffers,
                           std::vector<void*>& mgpu_obuffers) override
    {
        if(ifields.empty() && ofields.empty())
        {
            // not a multi-device case
            return;
        }

        if(input_data_host.empty())
        {
            throw std::invalid_argument(
                "rocfft_params::multi_gpu_prepare: host-residing input buffer does not exist.");
        }
        const auto cpu_ref_params = make_params_for_reference_cpu();
        if(cpu_ref_params.itype == fft_array_type_complex_planar
           || cpu_ref_params.itype == fft_array_type_hermitian_planar)
            throw std::logic_error("rocfft_params::multi_gpu_prepare: planar input data considered "
                                   "by cpu reference calculation.");
        const auto req_min_size = cpu_ref_params.ibuffer_sizes()[0];
        if(input_data_host[0].size() < req_min_size)
        {
            std::ostringstream excpt_info;
            excpt_info << "rocfft_params::multi_gpu_prepare: given host-residing input buffer is "
                          "too small for scattering the multi-device transform inputs.\n"
                       << "Buffer size is " << input_data_host[0].size()
                       << ", required min size is " << req_min_size << ".";
            throw std::invalid_argument(excpt_info.str());
        }

        if(placement == fft_placement_inplace)
        {
            // validate test case configuration with respect to what we assume below (current
            // limitations with respect to what we test for "in-place multi-device")
            const std::runtime_error unmet_mgpu_inplace_requirement(
                "rocfft_params::multi_gpu_prepare requires in-place tests to have the same total "
                "number of fields and number of bricks per field on input and output. All bricks "
                "should be assigned to the same devices, in the same order on input and output.");
            if(ifields.size() != ofields.size())
                throw unmet_mgpu_inplace_requirement;
            for(size_t field_idx = 0; field_idx < ifields.size(); field_idx++)
            {
                const auto& ifield = ifields[field_idx];
                const auto& ofield = ofields[field_idx];
                if(ifield.bricks.size() != ofield.bricks.size())
                    throw unmet_mgpu_inplace_requirement;
                for(size_t b_idx = 0; b_idx < ifield.bricks.size(); b_idx++)
                {
                    const auto& ibrick = ifield.bricks[b_idx];
                    const auto& obrick = ofield.bricks[b_idx];
                    if(ibrick.rank != obrick.rank || ibrick.device != obrick.device)
                        throw unmet_mgpu_inplace_requirement;
                }
            }
        }

        // I/O raw pointer(s) are left untouched if there is no corresponding field,
        // i.e., no data decomposition
        if(!ifields.empty())
            mgpu_ibuffers.clear();
        if(!ofields.empty())
            mgpu_obuffers.clear();
        const auto process_rank = get_process_rank();
        for(size_t f_idx = 0; f_idx < std::max(ifields.size(), ofields.size()); f_idx++)
        {
            const auto* ifield = f_idx < ifields.size() ? &ifields[f_idx] : nullptr;
            const auto* ofield = f_idx < ofields.size() ? &ofields[f_idx] : nullptr;
            for(size_t b_idx = 0; b_idx < std::max(ifield ? ifield->bricks.size() : 0,
                                                   ofield ? ofield->bricks.size() : 0);
                b_idx++)
            {
                const auto* ibrick
                    = ifield && b_idx < ifield->bricks.size() ? &ifield->bricks[b_idx] : nullptr;
                const auto* obrick
                    = ofield && b_idx < ofield->bricks.size() ? &ofield->bricks[b_idx] : nullptr;
                for(const auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                {
                    if(placement == fft_placement_inplace && io == fft_io::fft_io_out)
                    {
                        // outputs and input are set together (when the input is set)
                        // for in-place operations
                        continue;
                    }
                    const auto* io_brick = io == fft_io::fft_io_in ? ibrick : obrick;
                    auto& io_buffer_vec  = io == fft_io::fft_io_in ? mgpu_ibuffers : mgpu_obuffers;
                    if(!io_brick || io_brick->rank != process_rank)
                        continue;
                    // calculate byte size for device allocation:
                    const auto array_type = io == fft_io::fft_io_in ? itype : otype;
                    size_t     alloc_byte_size
                        = var_size<size_t>(precision, array_type)
                          * compute_ptrdiff(io_brick->length(), io_brick->stride);
                    if(placement == fft_placement_inplace)
                    {
                        // The I/O buffers must be large enough for both input and output data.
                        // NOTE: see above argument validation checks for testing "in-place". As
                        // a consequence, if this point is reached, io_brick == ibrick,
                        // obrick != nullptr, ibrick->rank == obrick->rank, and
                        // ibrick->device == obrick->device.
                        alloc_byte_size
                            = std::max(alloc_byte_size,
                                       var_size<size_t>(precision, otype)
                                           * compute_ptrdiff(obrick->length(), obrick->stride));
                    }

                    // scope for device-specific
                    {
                        rocfft_scoped_device dev(io_brick->device);
                        multi_gpu_data.emplace_back();
                        if(multi_gpu_data.back().alloc(alloc_byte_size) != hipSuccess)
                            throw std::runtime_error(
                                "rocfft_params::multi_gpu_prepare: device allocation failed");
                        io_buffer_vec.push_back(multi_gpu_data.back().data());
                        if(placement == fft_placement_inplace)
                            mgpu_obuffers.push_back(multi_gpu_data.back().data());

                        if(io == fft_io::fft_io_in)
                        {
                            // copy cpu input data to device buffer(s)
                            const auto input_data_host_offset = io_brick->lower_field_offset(
                                cpu_ref_params.istride, cpu_ref_params.idist);

                            // transpose input data to the brick's shape in host memory, then
                            // memcpy (as is) into allocated device buffer
                            std::vector<hostbuf> host_tmp(1);
                            host_tmp.front().alloc(alloc_byte_size);

                            std::vector<size_t> cpu_istrides_with_idist(cpu_ref_params.istride);
                            cpu_istrides_with_idist.insert(cpu_istrides_with_idist.begin(),
                                                           cpu_ref_params.idist);

                            copy_buffers(input_data_host,
                                         host_tmp,
                                         io_brick->length(),
                                         /* "nbatch" = */ 1,
                                         cpu_ref_params.precision,
                                         cpu_ref_params.itype,
                                         cpu_istrides_with_idist,
                                         /* "idist" = */ 0,
                                         array_type,
                                         io_brick->stride,
                                         /* "odist" =  */ 0,
                                         {input_data_host_offset},
                                         /* "ooffset" = */ {0});

                            // memcpy the transposed brick to the device
                            if(hipMemcpy(io_buffer_vec.back(),
                                         host_tmp.front().data(),
                                         alloc_byte_size,
                                         hipMemcpyHostToDevice)
                               != hipSuccess)
                            {
                                throw std::runtime_error(
                                    "rocfft_params::multi_gpu_prepare: hipMemcpy failed");
                            }
                        }
                    }
                }
            }
        }
    }

    // when preparing for multi-GPU transform, we need to allocate data
    // on each GPU.  This vector remembers all of those allocations.
    std::vector<gpubuf> multi_gpu_data;

    void multi_gpu_finalize(std::vector<hostbuf>& gathered_results_host,
                            std::vector<gpubuf>& /* gathered_results_device (unused) */,
                            std::vector<void*>& mgpu_obuffers) override
    {
        if(ofields.empty())
        {
            // not multi-device, no data gathering to be done
            return;
        }
        if(gathered_results_host.empty())
        {
            throw std::invalid_argument(
                "rocfft_params::multi_gpu_finalize: given host-residing buffer does not exist.");
        }
        if(otype == fft_array_type_complex_planar || otype == fft_array_type_hermitian_planar)
            throw std::logic_error("rocfft_params::multi_gpu_finalize: planar output data "
                                   "considered by current object.");
        const auto req_min_size = obuffer_sizes()[0];
        if(gathered_results_host[0].size() < req_min_size)
        {
            throw std::invalid_argument(
                "rocfft_params::multi_gpu_finalize: given host-residing buffer does not exist or "
                "is too small for gathering the multi-device transform results.");
            std::ostringstream excpt_info;
            excpt_info << "rocfft_params::multi_gpu_finalize: given host-residing buffer is is too "
                          "small for gathering the multi-device transform results.\n"
                       << "Buffer size is " << gathered_results_host[0].size()
                       << ", required min size is " << req_min_size << ".";
            throw std::invalid_argument(excpt_info.str());
        }

        const auto          process_rank = get_process_rank();
        std::vector<size_t> ostrides_with_odist(ostride);
        ostrides_with_odist.insert(ostrides_with_odist.begin(), odist);

        size_t obuffer_idx = 0;
        for(const auto& ofield : ofields)
        {
            for(const auto& obrick : ofield.bricks)
            {
                if(obrick.rank != process_rank)
                    continue;
                if(obuffer_idx >= mgpu_obuffers.size())
                {
                    throw std::invalid_argument(
                        "rocfft_params::multi_gpu_finalize: not as many device output buffers as "
                        "expected when gathering the multi-device transform results.");
                }

                const size_t brick_byte_size = var_size<size_t>(precision, otype)
                                               * compute_ptrdiff(obrick.length(), obrick.stride);

                const auto offset_in_gathered_results_host
                    = obrick.lower_field_offset(ostride, odist);

                // switch device to where we're copying from
                rocfft_scoped_device dev(obrick.device);

                // copy the device results to host, then copy to gathered_results
                std::vector<hostbuf> host_tmp(1);
                host_tmp.front().alloc(brick_byte_size);
                if(hipMemcpy(host_tmp.front().data(),
                             mgpu_obuffers[obuffer_idx++],
                             brick_byte_size,
                             hipMemcpyDeviceToHost)
                   != hipSuccess)
                {
                    throw std::runtime_error("rocfft_params::multi_gpu_finalize: hipMemcpy failed");
                }

                copy_buffers(host_tmp,
                             gathered_results_host,
                             obrick.length(),
                             /* "nbatch" =  */ 1,
                             precision,
                             otype,
                             obrick.stride,
                             /* "idist" = */ 0,
                             otype,
                             ostrides_with_odist,
                             /* "odist" =  */ 0,
                             /* "ioffset" =  */ {0},
                             {offset_in_gathered_results_host});
            }
        }
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

    int get_process_rank() const
    {
        int process_rank = -1; // invalid initialization
        if(mp_lib == fft_mp_lib_mpi)
        {
#ifdef ROCFFT_MPI_ENABLE
            if(!mp_comm)
                throw std::runtime_error("Multi-process communicator is not defined");
            auto ret = MPI_Comm_rank(*static_cast<MPI_Comm*>(mp_comm), &process_rank);
            if(ret != MPI_SUCCESS || process_rank < 0)
                throw std::runtime_error("Rank of current process couldn't be set");
#else
            throw std::runtime_error("MPI is not enabled");
#endif
        }
        else
        {
            process_rank = 0;
        }
        return process_rank;
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
#ifdef _WIN32
    typedef HMODULE ROCFFT_LIB;
#else
    typedef void* ROCFFT_LIB;
#endif

    // Load the rocfft library
    static ROCFFT_LIB lib_load(const std::string& path)
    {
#ifdef _WIN32
        return LoadLibraryA(path.c_str());
#else
        return dlopen(path.c_str(), RTLD_LAZY);
#endif
    }

    // Return a string describing the error loading rocfft
    static const char* lib_load_error()
    {
#ifdef _WIN32
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
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(libhandle, sym));
#else
        return dlsym(libhandle, sym);
#endif
    }

    static void lib_close(ROCFFT_LIB libhandle)
    {
        if(!libhandle)
            return;
#ifdef _WIN32
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
