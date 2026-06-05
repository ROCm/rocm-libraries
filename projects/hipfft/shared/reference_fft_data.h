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

#ifndef REFERENCE_FFT_DATA
#define REFERENCE_FFT_DATA

#include <algorithm>
#include <future>
#include <string>
#include <variant>
#include <vector>

#include "client_except.h"
#include "enum_to_string.h"
#include "fft_params.h"
#include "fftw_transform.h"
#include "gpubuf.h"
#include "rocfft_against_fftw.h"
#include "sys_mem.h"
#include "test_callbacks.h"
#include "test_params.h"

template <typename Tfloat>
inline void execute_cpu_fft(const fft_params&            cpu_fft_params,
                            fftw_plan_wrapper_t<Tfloat>& cpu_plan,
                            std::vector<hostbuf>&        cpu_input,
                            std::vector<hostbuf>&        cpu_output)
{
    // If this is either C2R or callbacks are enabled, the
    // input will be modified.  So we need to modify the copy instead.
    std::vector<hostbuf>  cpu_input_copy(cpu_input.size());
    std::vector<hostbuf>* input_ptr = &cpu_input;
    if(cpu_fft_params.run_callbacks != fft_callback_type_none
       || cpu_fft_params.transform_type == fft_transform_type_real_inverse)
    {
        for(size_t i = 0; i < cpu_input.size(); ++i)
        {
            cpu_input_copy[i] = cpu_input[i].copy();
        }

        input_ptr = &cpu_input_copy;
    }

    // run FFTW (which may destroy CPU input)
    apply_load_callback(cpu_fft_params, *input_ptr);
    cpu_fft_params.apply_host_load_ops(*input_ptr);
    fftw_run<Tfloat>(cpu_fft_params.transform_type, cpu_plan, *input_ptr, cpu_output);
    // clean up
    // ask FFTW to fully clean up, since it tries to cache plan details
    cpu_plan.reset();
    if constexpr(std::is_same<Tfloat, double>::value)
        fftw_cleanup();
    else
        fftwf_cleanup();
    cpu_fft_params.apply_host_store_ops(cpu_output);
    apply_store_callback(cpu_fft_params, cpu_output);
}

struct reference_fft_data_t
{
    reference_fft_data_t(const fft_params& cpu_fft_params)
    {
        if(cpu_fft_params.placement != fft_placement_notinplace || !cpu_fft_params.valid())
            throw std::invalid_argument("Reference FFT results require valid parameters, "
                                        "configured for out-of-place calculations");
        // If current cached results are valid reference results for the given
        // parameters, just use them.
        if(cached_data.input_is_set.valid() && cached_data.output_is_set.valid()
           && cached_data.can_be_used_for(cpu_fft_params, false /* = strict_precision_check*/))
        {
            this->swap(cached_data);
            // former cached_data.params are now this->params
            if(cpu_fft_params.precision < params.precision)
                async_narrow_precision(cpu_fft_params.precision);
            if(verbose > 3)
            {
                std::cout << "CPU params:\n";
                std::cout << params.str("\n\t") << std::endl;
            }
            return;
        }
        // Clear cache as it is no longer useful: reference data will need to be computed.
        if(verbose > 2)
            std::cout << "Clearing cached reference FFT results" << std::endl;
        cached_data.clear();
        params = cpu_fft_params;
        // input buffer can have minimal size but output must be large enough for
        // precision used at compute time (requirement from fftw_run)
        cpu_input = allocate_host_buffer(params.precision, params.itype, params.isize);
        // Output buffer and fftw plan are needed iff `fftw_compare` is `true`.
        if(fftw_compare)
        {
            const auto compute_prec
                = params.precision == fft_precision_half ? fft_precision_single : params.precision;
            // Reserve some amount of system memory for the FFTW plan's possible workspace: Use a
            // conservative estimate for the size of the workspace that the fftw plan may need to
            // guard against OOM kills. The FFTW workspace is estimated as double (resp. five times)
            // the maximum size of I/O if all (resp. any) length prime factors do not exceed (resp.
            // does exceed) 13.
            const auto estimated_bytes_for_fftw_workspace
                = std::max(params.isize[0] * var_size<size_t>(compute_prec, params.itype),
                           params.osize[0] * var_size<size_t>(compute_prec, params.otype))
                  * (std::any_of(params.length.begin(),
                                 params.length.end(),
                                 [](size_t ell) { return ell > 2 && max_prime_factor(ell) > 13; })
                         ? 5
                         : 2);
            try
            {
                reservation_for_fftw_internal_alloc.set_desired_size(
                    estimated_bytes_for_fftw_workspace);
            }
            catch(const std::invalid_argument& e)
            {
                // Requested size is too large: reservation cannot be guaranteed
                std::stringstream info;
                info
                    << "Reservation of system memory for FFTW plan's possible workspace (estimated "
                       "size) cannot be guaranteed. Details:\n"
                    << e.what();
                throw ROCFFT_SKIP{info.str()};
            }
            if(verbose > 3)
            {
                std::cout << "Reserved " << byte_size_to_str(estimated_bytes_for_fftw_workspace)
                          << " of system memory temporarily for the fftw plan's possible internal "
                             "workspace (temporary reservation)."
                          << std::endl;
            }

            cpu_output = allocate_host_buffer(compute_prec, params.otype, params.osize);
            switch(compute_prec)
            {
            case fft_precision_double:
                cpu_plan = fftw_plan_via_rocfft<double>(params.length,
                                                        params.istride,
                                                        params.ostride,
                                                        params.nbatch,
                                                        params.idist,
                                                        params.odist,
                                                        params.transform_type,
                                                        cpu_input,
                                                        cpu_output);
                break;
            case fft_precision_single:
                cpu_plan = fftw_plan_via_rocfft<float>(params.length,
                                                       params.istride,
                                                       params.ostride,
                                                       params.nbatch,
                                                       params.idist,
                                                       params.odist,
                                                       params.transform_type,
                                                       cpu_input,
                                                       cpu_output);
                break;
            default:
                throw std::invalid_argument("Unexpected compute precision encountered when "
                                            "constructing reference FFT data");
                break;
            }
        }
        if(verbose > 3)
        {
            std::cout << "CPU params:\n";
            std::cout << params.str("\n\t") << std::endl;
        }
    }

    bool needs_input_initialization() const
    {
        return !input_is_set.valid();
    }

    void initialize_input(const fft_input_generator& gen)
    {
        if(!is_host_generator(gen))
            throw std::invalid_argument("Independent initialization of input data for reference "
                                        "FFT data requires a host-side input generator");
        // Avoid corruption of input data by concurrently-executing threads
        wait_if_needed_for<fft_io::fft_io_in>();
        params.igen  = gen; // <-- conditions the behavior of params.compute_input below
        input_is_set = std::async(std::launch::async, [&]() { params.compute_input(cpu_input); });
        // output data must be invalidated as it needs re-computing:
        wait_and_invalidate<fft_io::fft_io_out>();
    }

    void initialize_input_using(const std::vector<gpubuf>& src_buffers,
                                const fft_params&          test_params)
    {
        if(!can_be_used_for(test_params))
            throw std::invalid_argument(
                "Reference results cannot be used with the given test parameters");
        // Avoid corruption of input data by concurrently-executing threads
        wait_if_needed_for<fft_io::fft_io_in>();
        const auto ibuffer_sizes = test_params.ibuffer_sizes();
        if(params.itype == test_params.itype && params.istride == test_params.istride
           && params.idist == test_params.idist && params.isize == test_params.isize
           && params.ioffset == test_params.ioffset && src_buffers.size() == cpu_input.size())
        {
            // Direct copy of device input data into CPU input data
            for(unsigned int idx = 0; idx < src_buffers.size(); ++idx)
            {
                if(cpu_input[idx].size() < ibuffer_sizes[idx])
                    throw std::logic_error("cpu_input buffer " + std::to_string(idx)
                                           + " is too small ("
                                           + byte_size_to_str(cpu_input[idx].size())
                                           + ") for direct copy of device input data ("
                                           + byte_size_to_str(ibuffer_sizes[idx]) + ")");

                const auto hip_status = hipMemcpy(cpu_input.at(idx).data(),
                                                  src_buffers[idx].data(),
                                                  ibuffer_sizes[idx],
                                                  hipMemcpyDeviceToHost);
                if(hip_status != hipSuccess)
                {
                    throw hip_runtime_error("hipMemcpy failure", hip_status);
                }
            }
        }
        else
        {
            auto tmp_host_buffers = allocate_host_buffer(ibuffer_sizes);
            // Copy input to CPU
            for(unsigned int idx = 0; idx < src_buffers.size(); ++idx)
            {
                const auto hip_status = hipMemcpy(tmp_host_buffers.at(idx).data(),
                                                  src_buffers[idx].data(),
                                                  ibuffer_sizes[idx],
                                                  hipMemcpyDeviceToHost);
                if(hip_status != hipSuccess)
                {
                    throw hip_runtime_error("hipMemcpy failure", hip_status);
                }
            }

            copy_buffers(tmp_host_buffers,
                         cpu_input,
                         test_params.ilength(),
                         test_params.nbatch,
                         test_params.precision,
                         test_params.itype,
                         test_params.istride,
                         test_params.idist,
                         params.itype,
                         params.istride,
                         params.idist,
                         test_params.ioffset,
                         params.ioffset);
        }
        // input is readily available:
        std::promise<void> tmp_promise;
        tmp_promise.set_value();
        input_is_set = tmp_promise.get_future();
        // output data must be invalidated as it needs re-computing:
        wait_and_invalidate<fft_io::fft_io_out>();
    }

    void copy_input_data_in_device_buffers(const std::vector<gpubuf>& device_input_buffers,
                                           const fft_params&          test_params) const
    {
        if(!can_be_used_for(test_params))
            throw std::invalid_argument(
                "Reference results cannot be used with the given test parameters");

        if(!input_is_set.valid())
            throw std::logic_error("The input data of reference FFT results needs to be "
                                   "initialized before it may be copied");
        wait_if_needed_for<fft_io::fft_io_in>();
        const std::vector<hostbuf>* input_to_copy = &cpu_input;
        std::vector<hostbuf>        temp_host_buffers;
        const auto                  ibuffer_sizes = test_params.ibuffer_sizes();

        if(!(params.itype == test_params.itype && params.istride == test_params.istride
             && params.idist == test_params.idist && params.isize == test_params.isize
             && params.ioffset == test_params.ioffset && params.precision == test_params.precision))
        {
            temp_host_buffers = allocate_host_buffer(ibuffer_sizes);
            copy_buffers(cpu_input,
                         temp_host_buffers,
                         params.ilength(),
                         test_params.nbatch, // note: may be smaller than params.nbatch
                         params.precision,
                         params.itype,
                         params.istride,
                         params.idist,
                         test_params.itype,
                         test_params.istride,
                         test_params.idist,
                         params.ioffset,
                         test_params.ioffset);
            input_to_copy = &temp_host_buffers;
        }

        if(device_input_buffers.size() < input_to_copy->size())
            throw std::invalid_argument("Fewer device input buffers than expected ("
                                        + std::to_string(device_input_buffers.size()) + " < "
                                        + std::to_string(input_to_copy->size())
                                        + ") for copying reference input data");

        // Copy input data to GPU
        for(unsigned int idx = 0; idx < input_to_copy->size(); ++idx)
        {
            if(device_input_buffers[idx].size() < ibuffer_sizes[idx])
                throw std::invalid_argument("Device input buffer " + std::to_string(idx)
                                            + " is too small ("
                                            + byte_size_to_str(device_input_buffers[idx].size())
                                            + ") for copying reference input data("
                                            + byte_size_to_str(ibuffer_sizes[idx]) + " needed)");
            const auto hip_status = hipMemcpy(device_input_buffers[idx].data(),
                                              input_to_copy->at(idx).data(),
                                              ibuffer_sizes[idx],
                                              hipMemcpyHostToDevice);

            if(hip_status != hipSuccess)
            {
                throw hip_runtime_error("hipMemcpy failure", hip_status);
            }
        }
    }

    bool needs_computing() const
    {
        return fftw_compare && !output_is_set.valid();
    }

    ~reference_fft_data_t()
    {
        // Remember the results of the last FFT we computed with FFTW.  Tests
        // are ordered so that later cases can often reuse this result.
        if(input_is_set.valid() && (!fftw_compare || output_is_set.valid()))
        {
            cached_data.swap(*this);
        }
        if(verbose > 3 && reservation_for_fftw_internal_alloc.size() > 0)
        {
            std::cout << "Releasing temporary reservation of "
                      << byte_size_to_str(reservation_for_fftw_internal_alloc.size())
                      << " of system memory (estimated FFTW plan's possible internal workspace)."
                      << std::endl;
        }
    }
    static reference_fft_data_t make_default()
    {
        return reference_fft_data_t();
    }

    // Note: the FFTW cpu plan is destroyed when the thread completes
    void launch_async_compute()
    {
        if(!fftw_compare)
            throw std::logic_error(
                "Reference results should not be computed if `fftw_compare` is disabled");
        if(!input_is_set.valid())
            throw std::logic_error("Asynchronous computation of reference FFT results mustn't be "
                                   "launched before having set the reference input data");
        // Avoid corruption of output data by concurrently-executing threads
        wait_if_needed_for<fft_io::fft_io_out>();
        output_is_set = std::async(std::launch::async, [&]() {
            wait_if_needed_for<fft_io::fft_io_in>();
            switch(params.precision)
            {
            case(fft_precision_double):
                execute_cpu_fft<double>(
                    params, std::get<fftw_plan_wrapper_t<double>>(cpu_plan), cpu_input, cpu_output);
                break;
            case(fft_precision_single):
                execute_cpu_fft<float>(
                    params, std::get<fftw_plan_wrapper_t<float>>(cpu_plan), cpu_input, cpu_output);
                break;
            case(fft_precision_half):
                // cpu_input and cpu_output must be large enough for computing reference
                // results in single-precision floating-point arithmetics but the output's
                // precision is downgraded upon completion of the computation.
                execute_cpu_fft<rocfft_fp16>(
                    params, std::get<fftw_plan_wrapper_t<float>>(cpu_plan), cpu_input, cpu_output);
                break;
            default:
                throw std::logic_error("Unexpected precision for the CPU plan");
            }
            if(verbose > 3 && reservation_for_fftw_internal_alloc.size() > 0)
            {
                std::cout
                    << "Releasing temporary reservation of "
                    << byte_size_to_str(reservation_for_fftw_internal_alloc.size())
                    << " of system memory (estimated FFTW plan's possible internal workspace)."
                    << std::endl;
            }
            reservation_for_fftw_internal_alloc.release();
        });
    }

    template <fft_io io>
    void print_data() const
    {
        static_assert(io == fft_io::fft_io_in || io == fft_io::fft_io_out);
        if constexpr(io == fft_io::fft_io_out)
        {
            if(!fftw_compare)
                throw std::runtime_error(
                    "Reference output data cannot be printed if `fftw_compare` is disabled");
        }

        auto& flag = io == fft_io::fft_io_in ? input_is_set : output_is_set;
        if(!flag.valid())
            throw std::logic_error(
                "The desired reference data cannot be printed since it was not set");
        flag.wait();
        if constexpr(io == fft_io::fft_io_in)
        {
            std::cout << "CPU input:\n";
            params.print_ibuffer(cpu_input);
        }
        else
        {
            std::cout << "CPU output:\n";
            params.print_obuffer(cpu_output);
        }
    }

    template <fft_io io>
    std::shared_future<VectorNorms> get_norm(size_t relevant_batch_size)
    {
        static_assert(io == fft_io::fft_io_in || io == fft_io::fft_io_out);
        if constexpr(io == fft_io::fft_io_out)
        {
            if(!fftw_compare)
                throw std::runtime_error("Norms of reference output data cannot be computed if "
                                         "`fftw_compare` is disabled.");
        }

        if(relevant_batch_size > params.nbatch)
            throw std::invalid_argument("Invalid batch size in calculation of I/O norms.");
        auto& io_is_set = io == fft_io::fft_io_in ? input_is_set : output_is_set;
        if(!io_is_set.valid())
        {
            throw std::logic_error(
                "Desired norm cannot be computed since corresponding data was not set");
        }
        return std::async(std::launch::async, [&, relevant_batch_size]() {
            io_is_set.wait();
            return norm(io == fft_io::fft_io_in ? cpu_input : cpu_output,
                        io == fft_io::fft_io_in ? params.ilength() : params.olength(),
                        std::min(params.nbatch, relevant_batch_size),
                        params.precision,
                        io == fft_io::fft_io_in ? params.itype : params.otype,
                        io == fft_io::fft_io_in ? params.istride : params.ostride,
                        io == fft_io::fft_io_in ? params.idist : params.odist,
                        io == fft_io::fft_io_in ? params.ioffset : params.ooffset);
        });
    }

    template <fft_io io>
    const std::vector<hostbuf>& get_buffers()
    {
        static_assert(io == fft_io::fft_io_in || io == fft_io::fft_io_out);
        if constexpr(io == fft_io::fft_io_out)
        {
            if(!fftw_compare)
                throw std::runtime_error(
                    "Reference output data is not available if `fftw_compare` is disabled.");
        }
        auto& io_is_set = io == fft_io::fft_io_in ? input_is_set : output_is_set;
        if(!io_is_set.valid())
        {
            throw std::logic_error("Desired reference FFT results' input/output buffers cannot be "
                                   "queried since corresponding data was not set");
        }
        io_is_set.wait();
        return io == fft_io::fft_io_in ? cpu_input : cpu_output;
    }
    const fft_params& get_params() const
    {
        return params;
    }

    static void clear_cache()
    {
        cached_data.clear();
    }

private:
    // to validate test parameters as argument of public member functions
    bool can_use_direct_input_copy_with(const fft_params& other) const
    {
        return params.itype == other.itype && params.istride == other.istride
               && params.idist == other.idist && params.isize == other.isize
               && params.precision <= other.precision;
    }

    template <fft_io io>
    void wait_if_needed_for() const
    {
        static_assert(io == fft_io::fft_io_in || io == fft_io::fft_io_out);
        auto& io_is_set = io == fft_io::fft_io_in ? input_is_set : output_is_set;
        if(io_is_set.valid())
            io_is_set.wait();
    }
    template <fft_io io>
    void wait_and_invalidate()
    {
        wait_if_needed_for<io>();
        if constexpr(io == fft_io::fft_io_in)
            input_is_set = decltype(input_is_set){};
        else
            output_is_set = decltype(output_is_set){};
    }

    void clear()
    {
        params = fft_params{};
        wait_and_invalidate<fft_io::fft_io_in>();
        wait_and_invalidate<fft_io::fft_io_out>();
        cpu_input.clear();
        cpu_output.clear();
        reservation_for_fftw_internal_alloc.release();
        cpu_plan = fftw_trait<float>::make_wrapper(nullptr);
    }

    void swap(reference_fft_data_t& other)
    {
        if(this == &other)
            return;
        if(other.input_is_set.valid())
            other.input_is_set.wait();
        if(other.output_is_set.valid())
            other.output_is_set.wait();

        std::swap(input_is_set, other.input_is_set);
        std::swap(output_is_set, other.output_is_set);
        cpu_input.swap(other.cpu_input);
        cpu_output.swap(other.cpu_output);
        std::swap(params, other.params);
        std::swap(cpu_plan, other.cpu_plan);
        reservation_for_fftw_internal_alloc.swap(other.reservation_for_fftw_internal_alloc);
    }

    void async_narrow_precision(fft_precision narrower_prec)
    {
        if(!input_is_set.valid() || (fftw_compare && !output_is_set.valid()))
            throw std::logic_error("Precision of reference results cannot be narrowed if input or "
                                   "output data were not set prior.");
        // Avoid data corruption by concurrent threads
        input_is_set.wait();
        if(fftw_compare)
            output_is_set.wait();
        const auto invalid_ref_prec_excpt = std::logic_error(
            "Invalid precision encountered for reference results to be narrowed");
        switch(narrower_prec)
        {
        case fft_precision_single:
        {
            if(params.precision != fft_precision_double)
                throw invalid_ref_prec_excpt;
            input_is_set = std::async(std::launch::async, [&]() {
                narrow_precision_inplace<double, float>(cpu_input.front());
            });
            if(fftw_compare)
                output_is_set = std::async(std::launch::async, [&]() {
                    narrow_precision_inplace<double, float>(cpu_output.front());
                });
        }
        break;
        case fft_precision_half:
        {
            if(params.precision == fft_precision_double)
            {
                input_is_set = std::async(std::launch::async, [&]() {
                    narrow_precision_inplace<double, rocfft_fp16>(cpu_input.front());
                });
                if(fftw_compare)
                    output_is_set = std::async(std::launch::async, [&]() {
                        narrow_precision_inplace<double, rocfft_fp16>(cpu_output.front());
                    });
            }
            else if(params.precision == fft_precision_single)
            {
                input_is_set = std::async(std::launch::async, [&]() {
                    narrow_precision_inplace<float, rocfft_fp16>(cpu_input.front());
                });
                if(fftw_compare)
                    output_is_set = std::async(std::launch::async, [&]() {
                        narrow_precision_inplace<float, rocfft_fp16>(cpu_output.front());
                    });
            }
            else
                throw invalid_ref_prec_excpt;
        }
        break;
        default:
            throw std::invalid_argument(
                "Unexpected precision given to narrow reference FFT results");
        }
        params.precision = narrower_prec;
    }

    // Note: batch size uses >= since existing reference results can just be
    // cropped for smaller batch sizes, if needed. The check on precision  can be
    // strict or loose: Floating-point precision of reference results can never be
    // narrower than what is being tested, but some public member functions do
    // require *identical* precisions.
    bool can_be_used_for(const fft_params& test_params, bool strict_precision_check = true) const
    {
        auto ret = params.length == test_params.length
                   && params.transform_type == test_params.transform_type
                   && params.nbatch >= test_params.nbatch
                   && params.run_callbacks == test_params.run_callbacks
                   && params.scale_factor == test_params.scale_factor;
        if(strict_precision_check)
            ret &= params.precision == test_params.precision;
        else
            ret &= params.precision >= test_params.precision;
        return ret;
    }

    fft_params               params;
    std::shared_future<void> input_is_set;
    std::shared_future<void> output_is_set;
    // FFTW input/output
    std::vector<hostbuf> cpu_input, cpu_output;
    // CPU (FFTW) plans are either single or double precision
    // (no actual half precision plan on CPU)
    std::variant<fftw_plan_wrapper_t<float>, fftw_plan_wrapper_t<double>> cpu_plan
        = fftw_trait<float>::make_wrapper(nullptr);
    system_memory::nonowned_reservation_t reservation_for_fftw_internal_alloc;

    // Private default constructor. (only needed for definition of static cache)
    reference_fft_data_t() = default;

    static reference_fft_data_t cached_data;
};

#endif
