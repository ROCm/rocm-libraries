// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipfft/hipfftw.h"
#include "../../../shared/environment.h"
#include "rocfft/rocfft.h"
#include <algorithm>
#include <array>
#include <cstdint> // std::int64_t
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

// anonymous namespace for implementation details
namespace
{
    struct hipfftw_invalid_arg : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct hipfftw_unsupported : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct hipfftw_internal_logic_error : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct rocfft_failure : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct hipfftw_bad_alloc : public std::runtime_error
    {
        const size_t     attempted_size;
        const hipError_t hip_error;
        hipfftw_bad_alloc(const std::string& info,
                          size_t             alloc_size,
                          hipError_t         hip_status = hipSuccess)
            : std::runtime_error::runtime_error(info)
            , attempted_size(alloc_size)
            , hip_error(hip_status)

        {
        }
    };
    struct hipfftw_bad_gpu_alloc : public hipfftw_bad_alloc
    {
        using hipfftw_bad_alloc::hipfftw_bad_alloc;
    };
    struct hipfftw_runtime_error : public std::runtime_error
    {
        const hipError_t hip_error;
        hipfftw_runtime_error(const std::string& info, hipError_t hip_status)
            : std::runtime_error::runtime_error(info)
            , hip_error(hip_status)

        {
        }
    };

    enum class hipfftw_io_label
    {
        INPUT_DATA,
        OUTPUT_DATA
    };

    constexpr bool is_real(rocfft_transform_type dft_type)
    {
        return dft_type == rocfft_transform_type_real_forward
               || dft_type == rocfft_transform_type_real_inverse;
    }

    template <rocfft_precision prec>
    struct hipfftw_scalar_trait;
    template <>
    struct hipfftw_scalar_trait<rocfft_precision_single>
    {
        using complex_t = fftwf_complex;
        using real_t    = float;
    };
    template <>
    struct hipfftw_scalar_trait<rocfft_precision_double>
    {
        using complex_t = fftw_complex;
        using real_t    = double;
    };

    template <rocfft_precision prec>
    using hipfftw_complex_data_t = typename hipfftw_scalar_trait<prec>::complex_t;
    template <rocfft_precision prec>
    using hipfftw_real_data_t = typename hipfftw_scalar_trait<prec>::real_t;
    // template helper struct for data type consistency (compile-time checks)
    template <rocfft_transform_type dft_type, rocfft_precision prec, hipfftw_io_label io>
    using hipfftw_user_data_t
        = std::conditional_t<!is_real(dft_type)
                                 || (dft_type == rocfft_transform_type_real_forward
                                     ^ io == hipfftw_io_label::INPUT_DATA),
                             // user data is complex
                             hipfftw_complex_data_t<prec>,
                             // user data is real
                             hipfftw_real_data_t<prec>>;

    template <rocfft_transform_type dft_type, hipfftw_io_label io>
    constexpr rocfft_array_type hipfftw_get_array_type()
    {
        if constexpr(!is_real(dft_type))
            return rocfft_array_type_complex_interleaved;
        else if constexpr(dft_type == rocfft_transform_type_real_forward
                          ^ io == hipfftw_io_label::INPUT_DATA)
            return rocfft_array_type_hermitian_interleaved;
        else
            return rocfft_array_type_real;
    }

    enum class hipfftw_memcpy_kind : std::underlying_type_t<hipMemcpyKind>
    {
        H2D = static_cast<std::underlying_type_t<hipMemcpyKind>>(hipMemcpyHostToDevice),
        D2H = static_cast<std::underlying_type_t<hipMemcpyKind>>(hipMemcpyDeviceToHost),
        NONE
    };
    enum class hipfftw_memcpy_direction
    {
        TO,
        FROM
    };

    // constexpr used for readability
    constexpr bool hipfftw_owns_it = true;

    template <bool owning>
    struct hipfftw_data_ptr_bundle
    {
    private:
        void*                 ptr;
        hipPointerAttribute_t attributes;

        void free_owned_resources()
        {
            if constexpr(!owning)
                return;
            if(!ptr)
                return;
            switch(attributes.type)
            {
            case hipMemoryType::hipMemoryTypeManaged:
            case hipMemoryType::hipMemoryTypeDevice:
                (void)hipFree(ptr);
                break;
            case hipMemoryType::hipMemoryTypeHost:
                (void)hipHostFree(ptr);
                break;
            case hipMemoryType::hipMemoryTypeUnregistered:
#ifdef WIN32
                _aligned_free(ptr);
#else
                std::free(ptr);
#endif
                break;
            // hipMemoryTypeUnified & hipMemoryTypeArray not set by hipPointerGetAttributes on AMD platforms
            case hipMemoryType::hipMemoryTypeArray:
            case hipMemoryType::hipMemoryTypeUnified:
            default:
                throw hipfftw_internal_logic_error("unexpected type of allocation (supposedly "
                                                   "owned or created by hipfftw) to be freed.");
                break;
            }
            ptr = nullptr;
        }

    public:
        hipfftw_data_ptr_bundle(void* init_ptr = nullptr)
            : ptr(nullptr)
        {
            set(init_ptr);
        }
        void set(void* new_ptr)
        {
            if(ptr == new_ptr)
                return;
            free_owned_resources();
            ptr = new_ptr;
            // hipPointerGetAttributes sets default attributes
            // (with attributes.type == hipMemoryTypeUnregistered) if
            // new_ptr = nullptr or if new_ptr is not to be found in the map
            // that the run-time manages (error possibly reported to a log in
            // the latter case in case of debug runtime version)
            auto hip_status = hipPointerGetAttributes(&attributes, new_ptr);
            if(hip_status != hipSuccess)
            {
                throw hipfftw_runtime_error("pointer attributes could not be determined.",
                                            hip_status);
            }
        }
        ~hipfftw_data_ptr_bundle()
        {
            free_owned_resources();
        }

        template <hipfftw_memcpy_direction dir>
        hipfftw_memcpy_kind get_copy_kind(int plan_device_id) const
        {
            if(attributes.type != hipMemoryType::hipMemoryTypeUnregistered
               && attributes.device != plan_device_id)
            {
                throw hipfftw_invalid_arg(
                    "if using registered data allocation for I/O, hipfftw requires them to be "
                    "visible to the device used at plan creation.");
            }
            static_assert(dir == hipfftw_memcpy_direction::TO
                          || dir == hipfftw_memcpy_direction::FROM);
            switch(attributes.type)
            {
            case hipMemoryType::hipMemoryTypeManaged: // the runtime is supposed to manage it
                [[fallthrough]];
            case hipMemoryType::hipMemoryTypeDevice:
                return hipfftw_memcpy_kind::NONE;
            case hipMemoryType::hipMemoryTypeUnregistered:
            case hipMemoryType::hipMemoryTypeHost:
                // TODO: check if device is APU and return false (systematically?) in that case
                if constexpr(dir == hipfftw_memcpy_direction::TO)
                    return hipfftw_memcpy_kind::H2D;
                else
                    return hipfftw_memcpy_kind::D2H;
            // hipMemoryTypeUnified & hipMemoryTypeArray not set by hipPointerGetAttributes on AMD platforms
            case hipMemoryType::hipMemoryTypeArray:
            case hipMemoryType::hipMemoryTypeUnified:
            default:
                throw hipfftw_internal_logic_error(
                    "unexpected type of memory to deduce (possibly-required) copy kind(s) from.");
            }
            // unreachable
        }

        bool is_host_accessible() const
        {
            return attributes.type == hipMemoryType::hipMemoryTypeUnregistered
                   || attributes.type == hipMemoryType::hipMemoryTypeHost
                   || attributes.type == hipMemoryType::hipMemoryTypeManaged
                   || attributes.type == hipMemoryType::hipMemoryTypeUnified;
        }

        void* get_data_ptr() const
        {
            return ptr;
        }

        bool operator==(const hipfftw_data_ptr_bundle& other) const
        {
            return ptr == other.ptr;
        }
        bool operator!=(const hipfftw_data_ptr_bundle& other) const
        {
            return !(*this == other);
        }
        operator bool() const
        {
            return ptr;
        }

        // disable copies and move
        hipfftw_data_ptr_bundle(const hipfftw_data_ptr_bundle&) = delete;
        hipfftw_data_ptr_bundle& operator=(const hipfftw_data_ptr_bundle&) = delete;
        hipfftw_data_ptr_bundle(hipfftw_data_ptr_bundle&&)                 = delete;
        hipfftw_data_ptr_bundle& operator=(hipfftw_data_ptr_bundle&&) = delete;
    };

    int hipfftw_get_current_device_id()
    {
        auto ret        = hipInvalidDeviceId;
        auto hip_status = hipGetDevice(&ret);
        if(hip_status != hipSuccess)
            throw hipfftw_runtime_error("the current device ID could not be determined.",
                                        hip_status);
        return ret;
    }

    // helper routine for assigning a device allocation to an owning data_ptr_bundle
    // (the current device is used)
    void hipfftw_set_device_allocation(hipfftw_data_ptr_bundle<hipfftw_owns_it>& bundle,
                                       size_t                                    alloc_size,
                                       const std::string&                        buffer_qualifier)
    {
        void* temp = nullptr;
        if(alloc_size > 0)
        {
            const auto hip_status = hipMalloc(&temp, alloc_size);
            if(hip_status != hipSuccess || !temp)
            {
                std::ostringstream info;
                info << "device memory could not be allocated for the " << buffer_qualifier
                     << " buffer.";
                throw hipfftw_bad_gpu_alloc(info.str(), alloc_size, hip_status);
            }
        }
        bundle.set(temp);
    }

    void init_rocfft()
    {
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
        // magic static to handle rocfft setup/cleanup
        static rocfft_initializer init;
    }

    template <rocfft_precision prec,
              // single or double precision only
              std::enable_if_t<prec == rocfft_precision_single || prec == rocfft_precision_double,
                               bool> = true>
    struct hipfftw_plan_internal
    {
        hipfftw_plan_internal() = default;
        ~hipfftw_plan_internal()
        {
            if(internal_rocfft_info)
            {
                rocfft_execution_info_destroy(internal_rocfft_info);
                internal_rocfft_info = nullptr;
            }
            if(internal_rocfft_desc)
            {
                rocfft_plan_description_destroy(internal_rocfft_desc);
                internal_rocfft_desc = nullptr;
            }
            if(internal_rocfft_plan)
            {
                rocfft_plan_destroy(internal_rocfft_plan);
                internal_rocfft_plan = nullptr;
            }
        }

        // disallow copies and moves
        hipfftw_plan_internal(const hipfftw_plan_internal&) = delete;
        hipfftw_plan_internal& operator=(const hipfftw_plan_internal&) = delete;
        hipfftw_plan_internal(hipfftw_plan_internal&&)                 = delete;
        hipfftw_plan_internal& operator=(hipfftw_plan_internal&&) = delete;

        rocfft_plan             internal_rocfft_plan = nullptr;
        rocfft_plan_description internal_rocfft_desc = nullptr;
        rocfft_execution_info   internal_rocfft_info = nullptr;
        rocfft_result_placement plan_placement;
        rocfft_transform_type   plan_dft_type;
        // Sizes of the buffers so we know how much to copy
        size_t in_bytes         = 0;
        size_t out_bytes        = 0;
        size_t work_buffer_size = 0;
        // Once initialized, the plan is configured for the device id set when initializing it
        int device_id = hipInvalidDeviceId;
        // bundles for owned data pointers
        hipfftw_data_ptr_bundle<hipfftw_owns_it> work_buffer;
        // possibly allocated in new-array execute paths if not allocated at plan creation
        hipfftw_data_ptr_bundle<hipfftw_owns_it> in_device;
        hipfftw_data_ptr_bundle<hipfftw_owns_it> out_device;
        // bundles for non-owned (user's) data pointers used at plan creation
        hipfftw_data_ptr_bundle<!hipfftw_owns_it> plan_creation_input;
        hipfftw_data_ptr_bundle<!hipfftw_owns_it> plan_creation_output;

        void execute() const
        {
            internal_execute(plan_creation_input, plan_creation_output);
        }

        void new_array_execute(void* new_user_exec_in, void* new_user_exec_out)
        {
            const auto new_exec_placement = new_user_exec_in == new_user_exec_out
                                                ? rocfft_placement_inplace
                                                : rocfft_placement_notinplace;
            if(plan_placement != new_exec_placement)
            {
                throw hipfftw_invalid_arg("I/O data pointers used at execution must use the same "
                                          "placement as defined at plan creation.");
            }
            hipfftw_data_ptr_bundle<!hipfftw_owns_it> new_exec_in(new_user_exec_in);
            hipfftw_data_ptr_bundle<!hipfftw_owns_it> new_exec_out(new_user_exec_out);
            set_io_device_buffers_for_execution(new_exec_in, new_exec_out);
            internal_execute(new_exec_in, new_exec_out);
        }

        template <rocfft_transform_type dft_type, size_t rank, typename T, size_t batch_rank>
        void init(const std::array<T, rank>&                                          lengths_rm,
                  const std::array<T, rank>&                                          istrides_rm,
                  const std::array<T, rank>&                                          ostrides_rm,
                  hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::INPUT_DATA>*  user_in,
                  hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUTPUT_DATA>* user_out,
                  const std::array<T, batch_rank>&                                    batch,
                  const std::array<T, batch_rank>&                                    idist,
                  const std::array<T, batch_rank>&                                    odist,
                  unsigned                                                            flags)
        {
            // compile-time validations of template specialization values
            static_assert(1 <= rank && rank <= 3);
            static_assert(1 == batch_rank); // only supported case at the moment
            // assuming no overflow when converting values from T into size_t below
            static_assert(std::numeric_limits<T>::max() <= std::numeric_limits<size_t>::max());
            // Validation of input arguments:
            auto is_strictly_negative = [](const T& val) { return val < static_cast<T>(0); };
            auto is_strictly_positive = [](const T& val) { return val > static_cast<T>(0); };
            if(!std::all_of(lengths_rm.begin(), lengths_rm.end(), is_strictly_positive))
                throw hipfftw_invalid_arg("length(s) must be strictly positive.");
            // Negative strides are not supported
            for(const auto& strides : {istrides_rm, ostrides_rm})
                if(std::any_of(strides.begin(), strides.end(), is_strictly_negative))
                    throw hipfftw_unsupported("strides must be positive.");
            if(!std::all_of(batch.begin(), batch.end(), is_strictly_positive))
                throw hipfftw_invalid_arg("batch(es) must be strictly positive.");
            // Negative distances are not supported
            for(const auto& dist : {idist, odist})
                if(std::any_of(dist.begin(), dist.end(), is_strictly_negative))
                    throw hipfftw_unsupported("distance(s) must be positive.");
            // Valid flag values are defined as bitwise OR of zero or more (unsigned) power-of-2
            // compile-time constants, (enabling well-defined identification via bitwise manipulations).
            if(flags
               != (flags
                   & (FFTW_WISDOM_ONLY | FFTW_MEASURE | FFTW_DESTROY_INPUT | FFTW_UNALIGNED
                      | FFTW_CONSERVE_MEMORY | FFTW_EXHAUSTIVE | FFTW_PRESERVE_INPUT | FFTW_PATIENT
                      | FFTW_ESTIMATE)))
            {
                throw hipfftw_invalid_arg("flags are ill-defined.");
            }
            if((!user_in || !user_out) && !(flags & FFTW_ESTIMATE) && !(flags & FFTW_WISDOM_ONLY))
            {
                throw hipfftw_invalid_arg(
                    "input/output data pointer(s) cannot be nullptr(s) with the given flags.");
            }
            if(flags & FFTW_WISDOM_ONLY)
            {
                throw hipfftw_unsupported("FFTW_WISDOM_ONLY is not supported.");
            }
            if constexpr(dft_type == rocfft_transform_type_real_inverse && rank > 1)
            {
                if(flags & FFTW_PRESERVE_INPUT)
                {
                    throw hipfftw_unsupported(
                        "FFTW_PRESERVE_INPUT is not supported for multi-dimensional C2R DFTs.");
                }
            }
            plan_creation_input.set(user_in);
            plan_creation_output.set(user_out);
            plan_placement = plan_creation_input == plan_creation_output
                                 ? rocfft_placement_inplace
                                 : rocfft_placement_notinplace;
            plan_dft_type  = dft_type;
            if(plan_placement == rocfft_placement_inplace)
            {
                // Check that the memory location is identical on input an output for the first
                // element of every leading dimension's sub-array. In order words, using row-major
                // convention, check that for every integer arrays
                // {k[0], k[1], ..., k[rank - 2], 0} ":= k" and every
                // {m[0], m[1], .., m[batch_dim-1]} ":= m" (in applicable ranges), the byte offset
                // on input, i.e.,
                // sizeof(hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::INPUT_DATA>)*
                //      std::inner_product(m.begin(), m.end(), idist.begin(),
                //                         std::inner_product(k.begin(), k.end(), istrides_rm.begin(), 0))
                // must be equal to the byte offset on output, i.e.,
                // sizeof(hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUTPUT_DATA>)*
                //      std::inner_product(m.begin(), m.end(), odist.begin(),
                //                         std::inner_product(k.begin(), k.end(), ostrides_rm.begin(), 0)).
                // This requirement translates into the followng element-wise conditions on
                // idist, odist, istides_rm, and ostides_rm.
                for(auto batch_dim = 0; batch_dim < batch_rank; batch_dim++)
                {
                    // 0 <= m[batch_dim] < batch[batch_dim], so the corresponding distance is
                    // irrelevant if batch[batch_dim] == 1.
                    if(batch[batch_dim] == 1)
                        continue;
                    if(idist[batch_dim]
                           * sizeof(
                               hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::INPUT_DATA>)
                       != odist[batch_dim]
                              * sizeof(hipfftw_user_data_t<dft_type,
                                                           prec,
                                                           hipfftw_io_label::OUTPUT_DATA>))
                        throw hipfftw_invalid_arg("distances rejected for in-place configuration.");
                }
                for(auto dim = 0; dim < rank - 1 /* exclude leading dimension */; dim++)
                {
                    if(lengths_rm[dim] == 1)
                        continue;
                    if(istrides_rm[dim]
                           * sizeof(
                               hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::INPUT_DATA>)
                       != ostrides_rm[dim]
                              * sizeof(hipfftw_user_data_t<dft_type,
                                                           prec,
                                                           hipfftw_io_label::OUTPUT_DATA>))
                    {
                        throw hipfftw_invalid_arg("strides rejected for in-place configuration..");
                    }
                }
            }
            // TODO (required when user-defined strides/distances may be used): add validity check
            // on strides and distances for non-aliasing data

            // Generalized input are validated... Let's initialize the plan!
            init_rocfft(); // "magic" common to all template specializations
            // compute min i/o data sizes via the index of the last relevant i/o element
            // Note: strides and distances (resp. lengths and batch) are non-negative
            // (resp. strictly positive), as verified above
            size_t last_input_element_idx  = 0;
            size_t last_output_element_idx = 0;
            for(auto dim = 0; dim < rank; dim++)
            {
                const auto last_input_entry_for_dim
                    = dft_type == rocfft_transform_type_real_inverse && dim == rank - 1
                          ? lengths_rm[dim] / 2
                          : lengths_rm[dim] - 1;
                const auto last_output_entry_for_dim
                    = dft_type == rocfft_transform_type_real_forward && dim == rank - 1
                          ? lengths_rm[dim] / 2
                          : lengths_rm[dim] - 1;
                last_input_element_idx += last_input_entry_for_dim * istrides_rm[dim];
                last_output_element_idx += last_output_entry_for_dim * ostrides_rm[dim];
            }
            for(auto batch_dim = 0; batch_dim < batch_rank; batch_dim++)
            {
                last_input_element_idx += (batch[batch_dim] - 1) * idist[batch_dim];
                last_output_element_idx += (batch[batch_dim] - 1) * odist[batch_dim];
            }
            if(last_input_element_idx > std::numeric_limits<T>::max()
               || last_output_element_idx > std::numeric_limits<T>::max())
            {
                throw hipfftw_unsupported(
                    "data layouts involving element indices that exceed the maximum value "
                    "representable in the chosen integral type are not supported (consider using "
                    "guru64 plan creation functions).");
            }
            in_bytes = sizeof(hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::INPUT_DATA>)
                       * (last_input_element_idx + 1);
            out_bytes = sizeof(hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUTPUT_DATA>)
                        * (last_output_element_idx + 1);
            if(plan_placement == rocfft_placement_inplace)
            {
                in_bytes = out_bytes = std::max(in_bytes, out_bytes);
            }

            // Change row-major to col-major for all relevant inputs, converting from T to size_t
            auto reverse = [](const std::array<T, rank>& in_array) {
                auto ret = std::array<size_t, rank>();
                std::reverse_copy(in_array.begin(), in_array.end(), ret.begin());
                return ret;
            };
            const auto lengths_cm  = reverse(lengths_rm);
            const auto istrides_cm = reverse(istrides_rm);
            const auto ostrides_cm = reverse(ostrides_rm);

            // Create plan description
            if(rocfft_plan_description_create(&internal_rocfft_desc) != rocfft_status_success)
            {
                throw rocfft_failure(
                    "an error was received from rocfft when creating the plan description.");
            }
            if(rocfft_plan_description_set_data_layout(
                   internal_rocfft_desc,
                   hipfftw_get_array_type<dft_type, hipfftw_io_label::INPUT_DATA>(),
                   hipfftw_get_array_type<dft_type, hipfftw_io_label::OUTPUT_DATA>(),
                   nullptr /* in_offsets */,
                   nullptr /* out_offsets */,
                   rank /* in_strides_sizes */,
                   istrides_cm.data(),
                   idist[0],
                   rank /* out_strides_size */,
                   ostrides_cm.data(),
                   odist[0])
               != rocfft_status_success)
            {
                throw rocfft_failure(
                    "an error was received from rocfft when setting the data layout.");
            }

            if(rocfft_plan_create(&internal_rocfft_plan,
                                  plan_placement,
                                  dft_type,
                                  prec,
                                  rank,
                                  lengths_cm.data(),
                                  batch[0],
                                  internal_rocfft_desc)
               != rocfft_status_success)
            {
                throw rocfft_failure(
                    "an error was received from rocfft when creating the internal rocfft plan.");
            }

            if(rocfft_execution_info_create(&internal_rocfft_info) != rocfft_status_success)
            {
                throw rocfft_failure("an error was received from rocfft when creating the "
                                     "execution info structure.");
            }
            if(rocfft_plan_get_work_buffer_size(internal_rocfft_plan, &work_buffer_size)
               != rocfft_status_success)
            {
                throw rocfft_failure("an error was received from rocfft when fetching the size of "
                                     "the internal plan's work area.");
            }
            // set device id
            device_id = hipfftw_get_current_device_id();
            // create and set device work buffer
            if(work_buffer_size > 0)
            {
                hipfftw_set_device_allocation(work_buffer, work_buffer_size, "work");
                if(rocfft_execution_info_set_work_buffer(
                       internal_rocfft_info, work_buffer.get_data_ptr(), work_buffer_size)
                   != rocfft_status_success)
                    throw rocfft_failure(
                        "an error was received from rocfft when setting the plan's work buffer.");
            }
            // default execution uses the I/O data pointer used at creation
            set_io_device_buffers_for_execution(plan_creation_input, plan_creation_output);
            return;
        }

    private:
        // NOTE: new-array execute paths may need to allocate the I/O device buffers if the new
        // I/O require them (and if the I/O from plan creation did not).
        void set_io_device_buffers_for_execution(
            const hipfftw_data_ptr_bundle<!hipfftw_owns_it>& intended_execute_in,
            const hipfftw_data_ptr_bundle<!hipfftw_owns_it>& intended_execute_out)
        {
            // TODO: check if the current device is an APU and simply never use
            // I/O device buffers in that case
            if(in_device && (plan_placement == rocfft_placement_inplace || out_device))
            {
                // buffers are ready to go, nothing to do
                return;
            }
            // set device I/O buffer, if they're needed
            if(!in_device
               && intended_execute_in.get_copy_kind<hipfftw_memcpy_direction::TO>(device_id)
                      != hipfftw_memcpy_kind::NONE)
            {
                hipfftw_set_device_allocation(in_device, in_bytes, "input");
            }

            if(plan_placement != rocfft_placement_inplace && !out_device
               && intended_execute_out.get_copy_kind<hipfftw_memcpy_direction::FROM>(device_id)
                      != hipfftw_memcpy_kind::NONE)
            {
                hipfftw_set_device_allocation(out_device, out_bytes, "output");
            }
            return;
        }

        void internal_execute(const hipfftw_data_ptr_bundle<!hipfftw_owns_it>& exec_in,
                              const hipfftw_data_ptr_bundle<!hipfftw_owns_it>& exec_out) const
        {
            if(!internal_rocfft_plan)
                throw hipfftw_internal_logic_error("the rocfft plan (internal detail to hipfftw) "
                                                   "was uninitialized for execution (unexpected).");
            if(!exec_in || !exec_out)
                throw hipfftw_invalid_arg("nullptr(s) cannot be used for execution data pointers.");
            // in/out may or may not need to be copied to the device
            const auto input_copy_kind
                = exec_in.get_copy_kind<hipfftw_memcpy_direction::TO>(device_id);
            const auto output_copy_kind
                = exec_out.get_copy_kind<hipfftw_memcpy_direction::FROM>(device_id);

            void* exec_in_ptr = exec_in.get_data_ptr();
            if(input_copy_kind != hipfftw_memcpy_kind::NONE)
            {
                const auto hip_status = hipMemcpyAsync(in_device.get_data_ptr(),
                                                       exec_in_ptr,
                                                       in_bytes,
                                                       static_cast<hipMemcpyKind>(input_copy_kind));
                if(hip_status != hipSuccess)
                {
                    throw hipfftw_runtime_error(
                        "the input data could not be copied into the GPU input buffer.",
                        hip_status);
                }
                exec_in_ptr = in_device.get_data_ptr();
            }
            void* exec_out_ptr
                = plan_placement == rocfft_placement_inplace
                      ? exec_in_ptr
                      : (output_copy_kind != hipfftw_memcpy_kind::NONE ? out_device.get_data_ptr()
                                                                       : exec_out.get_data_ptr());
            rocfft_execute(internal_rocfft_plan, &exec_in_ptr, &exec_out_ptr, internal_rocfft_info);
            if(output_copy_kind != hipfftw_memcpy_kind::NONE)
            {
                const auto hip_status
                    = hipMemcpyAsync(exec_out.get_data_ptr(),
                                     exec_out_ptr,
                                     out_bytes,
                                     static_cast<hipMemcpyKind>(output_copy_kind));
                if(hip_status != hipSuccess)
                {
                    throw hipfftw_runtime_error(
                        "the output data could not be copied from the GPU output buffer.",
                        hip_status);
                }
            }
            if(exec_out.is_host_accessible())
            {
                // results must be accessible from the host upon completion
                hipEvent_t synchronizing_event = nullptr;
                auto       hip_status          = hipEventCreate(&synchronizing_event);
                if(hip_status != hipSuccess)
                    throw hipfftw_runtime_error("an event could not be created.", hip_status);
                hip_status = hipEventRecord(synchronizing_event);
                if(hip_status != hipSuccess)
                    throw hipfftw_runtime_error("an event could not be recorded.", hip_status);
                hip_status = hipEventSynchronize(synchronizing_event);
                if(hip_status != hipSuccess)
                    throw hipfftw_runtime_error("an event synchronization failed.", hip_status);
                hip_status = hipEventDestroy(synchronizing_event);
                if(hip_status != hipSuccess)
                    throw hipfftw_runtime_error("an event could not be destroyed.", hip_status);
            }
            return;
        }
    };

    template <size_t rank,
              size_t batch_rank,
              typename T,
              std::enable_if_t<std::is_integral_v<T> && (rank > 0 && batch_rank > 0), bool> = true>
    struct hipfftw_general_layout_data
    {
        std::array<T, rank>       lengths;
        std::array<T, rank>       istrides;
        std::array<T, rank>       ostrides;
        std::array<T, batch_rank> batches;
        std::array<T, batch_rank> idist;
        std::array<T, batch_rank> odist;
        // constexpr getters
        constexpr inline size_t get_rank() const
        {
            return rank;
        }
        constexpr inline size_t get_batch_rank() const
        {
            return batch_rank;
        }
    };

    template <size_t rank, rocfft_transform_type dft_type>
    hipfftw_general_layout_data<rank, 1, int> hipfftw_get_default_data_layout_info_rm(
        bool is_in_place, const std::array<int, rank>& user_lengths_rm, size_t batch_sz = 1)
    {
        auto overflow_guarded_mult = [](int& a, const int& b) {
            auto tmp = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b);
            if(tmp > std::numeric_limits<int>::max() || tmp < std::numeric_limits<int>::min())
                throw hipfftw_unsupported(
                    "length(s) that trigger integer overflow(s) for default stride(s) and/or "
                    "distance(s) are not supported via the default plan creation functions "
                    "(consider guru64 plan creation functions).");
            a = static_cast<int>(tmp);
        };

        hipfftw_general_layout_data<rank, 1, int> ret;
        ret.lengths = user_lengths_rm;
        int ival = 1, oval = 1;
        for(auto dim_idx = rank; dim_idx-- > 0;)
        {
            ret.istrides[dim_idx] = ival;
            ret.ostrides[dim_idx] = oval;
            if(is_real(dft_type) && dim_idx == rank - 1)
            {
                int& cmplx_stride_val
                    = dft_type == rocfft_transform_type_real_forward ? oval : ival;
                int& real_stride_val = dft_type == rocfft_transform_type_real_forward ? ival : oval;
                overflow_guarded_mult(cmplx_stride_val, ret.lengths[dim_idx] / 2 + 1);
                if(is_in_place)
                    overflow_guarded_mult(real_stride_val, 2 * (ret.lengths[dim_idx] / 2 + 1));
                else
                    overflow_guarded_mult(real_stride_val, ret.lengths[dim_idx]);
            }
            else
            {
                overflow_guarded_mult(ival, ret.lengths[dim_idx]);
                overflow_guarded_mult(oval, ret.lengths[dim_idx]);
            }
        }
        ret.batches[0] = batch_sz;
        ret.idist[0]   = ival;
        ret.odist[0]   = oval;
        return ret;
    }

    inline void hipfftw_validate_sign(int sign)
    {
        if(sign != FFTW_FORWARD && sign != FFTW_BACKWARD)
            throw hipfftw_invalid_arg("sign values must be FFTW_FORWARD or FFTW_BACKWARD.");
    }

    // read the environment variable env_var and convert it to a size_t value.
    // If the environment variable is not set or if the conversion fails, the default value
    // is returned.
    size_t hipfftw_fetch_env_var(const char* env_var, size_t default_val) noexcept
    {
        try
        {
            const std::string tmp = rocfft_getenv(env_var);
            if(tmp.empty())
                return default_val;
            const auto ret = std::stoull(tmp);
            return ret > std::numeric_limits<size_t>::max() ? std::numeric_limits<size_t>::max()
                                                            : static_cast<size_t>(ret);
        }
        catch(...)
        {
            return default_val;
        }
    }

    bool hipfftw_handler_is_verbose() noexcept
    {
        return hipfftw_fetch_env_var("HIPFFTW_LOG_EXCEPTIONS", 0) > 0;
    }

    inline void hipfftw_exception_handler(const char* user_facing_function) noexcept
    try
    {
        if(!hipfftw_handler_is_verbose())
            return;

        // log failure-specific information
        try
        {
            throw;
        }
        catch(const hipfftw_invalid_arg& e)
        {
            std::cerr << "Invalid argument reported by " << user_facing_function
                      << ". Details: " << e.what() << std::endl;
        }
        catch(const hipfftw_unsupported& e)
        {
            std::cerr << "Unsupported usage reported by " << user_facing_function
                      << ". Details: " << e.what() << std::endl;
        }
        catch(const hipfftw_internal_logic_error& e)
        {
            std::cerr << "A logic error internal to hipfftw was detected and reported by "
                      << user_facing_function << ". Details: " << e.what() << std::endl;
        }
        catch(const rocfft_failure& e)
        {
            std::cerr << "A rocfft failure was detected and reported by " << user_facing_function
                      << ". Details: " << e.what() << std::endl;
        }
        catch(const hipfftw_bad_gpu_alloc& e)
        {
            std::cerr << "A GPU allocation failure was detected and reported by "
                      << user_facing_function << ". Details: " << e.what();
            if(e.hip_error != hipSuccess)
                std::cerr << "\nThe hip error code was " << e.hip_error << ".";
            std::cerr << "\nThe attempted size was " << e.attempted_size << " bytes" << std::endl;
        }
        catch(const hipfftw_bad_alloc& e)
        {
            std::cerr << "An allocation failure was detected and reported by "
                      << user_facing_function << ". Details: " << e.what();
            if(e.hip_error != hipSuccess)
                std::cerr << "\nThe hip error code was " << e.hip_error << ".";
            std::cerr << "\nThe attempted size was " << e.attempted_size << " bytes" << std::endl;
        }
        catch(const hipfftw_runtime_error& e)
        {
            std::cerr << "A hip-specific runtime error was detected and reported by "
                      << user_facing_function << ". Details: " << e.what();
            if(e.hip_error != hipSuccess)
                std::cerr << "\nThe hip error code was " << e.hip_error << ".";
            std::cerr << std::endl;
        }
        catch(const std::runtime_error& e)
        {
            std::cerr << "A runtime error was detected and reported by " << user_facing_function
                      << ". Details: " << e.what() << std::endl;
        }
        catch(...)
        {
            std::cerr << "An unidentified exception was detected and reported by "
                      << user_facing_function << "." << std::endl;
        }
    }
    catch(...)
    {
        // one of the above catch blocks threw as it attempted to log failure-related information...
        // ignore (std::terminate invoked otherwise...)
    }

    template <hipMemoryType type>
    size_t hipfftw_alloc_host_limit() noexcept
    {
        static_assert(type == hipMemoryType::hipMemoryTypeHost
                      || type == hipMemoryType::hipMemoryTypeUnregistered);
        constexpr size_t no_limit = std::numeric_limits<size_t>::max();
        if constexpr(type == hipMemoryType::hipMemoryTypeHost)
            return hipfftw_fetch_env_var("HIPFFTW_BYTE_SIZE_LIMIT_PINNED_HOST_ALLOC", no_limit);
        else
            return hipfftw_fetch_env_var("HIPFFTW_BYTE_SIZE_LIMIT_PAGEABLE_HOST_ALLOC", no_limit);
    }

    // possible TODO for hipfftw_alloc_host_accessible: consider using hipMallocManaged,
    // first, if the device supports it [LINUX ONLY].
    // NOTE: a limit may be set for any attempted kind of allocation via a dedicated
    // environment variable. If the requested byte size exceeds that limit, a lesser-ranked
    // kind of allocation is attempted instead.
    // Current ranking:
    // 1. pinned host allocation
    // 2. pageable host allocation
    template <typename element_type = void, hipMemoryType type = hipMemoryType::hipMemoryTypeHost>
    element_type* hipfftw_alloc_host_accessible(size_t num_elements)
    {
        static_assert(type == hipMemoryType::hipMemoryTypeHost
                      || type == hipMemoryType::hipMemoryTypeUnregistered);
        // exception specific to this internal routine:
        struct hipfftw_flow_redirection : public std::runtime_error
        {
            using std::runtime_error::runtime_error;
        };

        void* ret = nullptr;
        try
        {
            const size_t byte_size
                = num_elements
                  * sizeof(
                      std::conditional_t<std::is_same_v<void, element_type>, char, element_type>);
            if(byte_size > 0)
            {
                if(byte_size > hipfftw_alloc_host_limit<type>())
                {
                    throw hipfftw_flow_redirection(
                        "the requested size exceeds the limit set via a dedicated environment "
                        "variable: a lesser-ranked allocation type may be attempted.");
                }
                if constexpr(type == hipMemoryType::hipMemoryTypeHost)
                {
                    auto hip_status = hipHostMalloc(&ret, byte_size);
                    if(hip_status != hipSuccess || !ret)
                    {
                        throw hipfftw_bad_alloc(
                            "allocation for pinned host memory failed.", byte_size, hip_status);
                    }
                }
                else
                {
                    constexpr size_t alignment = 64;
#ifdef WIN32
                    ret = _aligned_malloc(byte_size, alignment);
#else
                    ret = std::aligned_alloc(alignment, byte_size);
#endif
                    if(!ret)
                        throw hipfftw_bad_alloc("allocation for pageable host memory failed.",
                                                byte_size);
                }
            }
        }
        catch(const hipfftw_flow_redirection& e)
        {
            if(hipfftw_handler_is_verbose())
                std::cerr << "Redirecting execution flow: " << e.what() << std::endl;
            if constexpr(type == hipMemoryType::hipMemoryTypeHost)
            {
                // pinned host allocation was ruled out
                // --> attempt pageable allocation
                return hipfftw_alloc_host_accessible<element_type,
                                                     hipMemoryType::hipMemoryTypeUnregistered>(
                    num_elements);
            }
            else
            {
                // no other fallback
                ret = nullptr;
            }
        }
        catch(...)
        {
            throw;
        }
        return static_cast<element_type*>(ret);
    }

    void hipfftw_free(void* ptr)
    {
        // create an owning data pointer bundle on given pointed to leverage
        // the structure destructor's logic as it goes out of scope
        hipfftw_data_ptr_bundle<hipfftw_owns_it> bundle(ptr);
    }
} // end of implementation details

// definition of the header's precision-specific structures
struct fftwf_plan_s : public hipfftw_plan_internal<rocfft_precision_single>
{
};
struct fftw_plan_s : public hipfftw_plan_internal<rocfft_precision_double>
{
};

template <rocfft_precision prec>
struct hipfftw_plan;
template <>
struct hipfftw_plan<rocfft_precision_single>
{
    using type = fftwf_plan_s;
};
template <>
struct hipfftw_plan<rocfft_precision_double>
{
    using type = fftw_plan_s;
};
template <rocfft_precision prec>
using hipfftw_plan_t = typename hipfftw_plan<prec>::type;

template <rocfft_transform_type dft_type,
          rocfft_precision      prec,
          size_t                rank,
          typename T,
          size_t batch_rank = 1>
static hipfftw_plan_t<prec>* hipfftw_create_plan(
    const std::array<T, rank>&                                          lengths_rm,
    const std::array<T, rank>&                                          istrides_rm,
    const std::array<T, rank>&                                          ostrides_rm,
    hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::INPUT_DATA>*  user_in,
    hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUTPUT_DATA>* user_out,
    const std::array<T, batch_rank>&                                    batch,
    const std::array<T, batch_rank>&                                    idist,
    const std::array<T, batch_rank>&                                    odist,
    unsigned                                                            flags)
{
    auto ret = std::make_unique<hipfftw_plan_t<prec>>();
    ret->template init<dft_type, rank, T, batch_rank>(
        lengths_rm, istrides_rm, ostrides_rm, user_in, user_out, batch, idist, odist, flags);
    return ret.release();
}

template <rocfft_transform_type dft_type, rocfft_precision prec, size_t rank>
static hipfftw_plan_t<prec>* hipfftw_create_default_unbatched_plan(
    const std::array<int, rank>&                                        n,
    hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::INPUT_DATA>*  in,
    hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUTPUT_DATA>* out,
    unsigned                                                            flags)
{
    auto layout_data = hipfftw_get_default_data_layout_info_rm<rank, dft_type>(
        static_cast<void*>(in) == static_cast<void*>(out), n);
    return hipfftw_create_plan<dft_type, prec, rank, int>(layout_data.lengths,
                                                          layout_data.istrides,
                                                          layout_data.ostrides,
                                                          in,
                                                          out,
                                                          layout_data.batches,
                                                          layout_data.idist,
                                                          layout_data.odist,
                                                          flags);
}

template <rocfft_transform_type dft_type, rocfft_precision prec>
static hipfftw_plan_t<prec>* hipfftw_create_default_unbatched_plan(
    int                                                                 rank,
    const int*                                                          n,
    hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::INPUT_DATA>*  in,
    hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUTPUT_DATA>* out,
    unsigned                                                            flags)
{
    if(rank <= 0)
        throw hipfftw_invalid_arg("ranks must be strictly positive.");
    if(!n)
        throw hipfftw_invalid_arg("lengths argument must not be nullptr.");
    // rank == 1, 2, 3, or unsupported
    switch(rank)
    {
    case 1:
        return hipfftw_create_default_unbatched_plan<dft_type, prec, 1>(
            std::array<int, 1>({n[0]}), in, out, flags);
    case 2:
        return hipfftw_create_default_unbatched_plan<dft_type, prec, 2>(
            std::array<int, 2>({n[0], n[1]}), in, out, flags);
    case 3:
        return hipfftw_create_default_unbatched_plan<dft_type, prec, 3>(
            std::array<int, 3>({n[0], n[1], n[2]}), in, out, flags);
    default:
        throw hipfftw_unsupported("rank values larger than 3 are not supported.");
    }
    // unreachable
}

template <rocfft_precision prec, size_t rank>
static hipfftw_plan_t<prec>*
    hipfftw_create_default_unbatched_complex_plan(const std::array<int, rank>&  n,
                                                  int                           sign,
                                                  hipfftw_complex_data_t<prec>* in,
                                                  hipfftw_complex_data_t<prec>* out,
                                                  unsigned                      flags)
{
    hipfftw_validate_sign(sign);
    if(sign == FFTW_FORWARD)
        return hipfftw_create_default_unbatched_plan<rocfft_transform_type_complex_forward,
                                                     prec,
                                                     rank>(n, in, out, flags);
    else
        return hipfftw_create_default_unbatched_plan<rocfft_transform_type_complex_inverse,
                                                     prec,
                                                     rank>(n, in, out, flags);
}

template <rocfft_precision prec>
static hipfftw_plan_t<prec>*
    hipfftw_create_default_unbatched_complex_plan(int                           rank,
                                                  const int*                    n,
                                                  int                           sign,
                                                  hipfftw_complex_data_t<prec>* in,
                                                  hipfftw_complex_data_t<prec>* out,
                                                  unsigned                      flags)
{
    hipfftw_validate_sign(sign);
    if(sign == FFTW_FORWARD)
        return hipfftw_create_default_unbatched_plan<rocfft_transform_type_complex_forward, prec>(
            rank, n, in, out, flags);
    else
        return hipfftw_create_default_unbatched_plan<rocfft_transform_type_complex_inverse, prec>(
            rank, n, in, out, flags);
}

void* fftw_malloc(size_t n)
try
{
    return hipfftw_alloc_host_accessible(n);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

void* fftwf_malloc(size_t n)
try
{
    return hipfftw_alloc_host_accessible(n);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

double* fftw_alloc_real(size_t n)
try
{
    return hipfftw_alloc_host_accessible<double>(n);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_complex* fftw_alloc_complex(size_t n)
try
{
    return hipfftw_alloc_host_accessible<fftw_complex>(n);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

float* fftwf_alloc_real(size_t n)
try
{
    return hipfftw_alloc_host_accessible<float>(n);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_complex* fftwf_alloc_complex(size_t n)
try
{
    return hipfftw_alloc_host_accessible<fftwf_complex>(n);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

void fftw_free(void* p)
try
{
    hipfftw_free(p);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftwf_free(void* p)
try
{
    hipfftw_free(p);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftw_destroy_plan(fftw_plan plan)
{
    delete plan;
}

void fftwf_destroy_plan(fftwf_plan plan)
{
    delete plan;
}

void fftw_cleanup() {}

void fftwf_cleanup() {}

void fftw_execute(const fftw_plan plan)
try
{
    if(!plan)
        throw hipfftw_invalid_arg("plan argument cannot be nullptr.");
    plan->execute();
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftw_execute_dft(const fftw_plan plan, fftw_complex* in, fftw_complex* out)
try
{
    if(!plan)
        throw hipfftw_invalid_arg("plan argument cannot be nullptr.");
    if(is_real(plan->plan_dft_type))
        throw hipfftw_invalid_arg(
            "this function rejects plans created for real DFT(s), i.e., plans created by any of "
            "the fftw_plan_*_r2c or fftw_plan_*_c2r functions.");
    plan->new_array_execute(in, out);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftw_execute_dft_r2c(const fftw_plan plan, double* in, fftw_complex* out)
try
{
    if(!plan)
        throw hipfftw_invalid_arg("plan argument cannot be nullptr.");
    if(plan->plan_dft_type != rocfft_transform_type_real_forward)
        throw hipfftw_invalid_arg("this function requires a plan for real forward DFT(s), i.e., a "
                                  "plan created by any of the fftw_plan_*_r2c functions.");
    plan->new_array_execute(in, out);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex* in, double* out)
try
{
    if(!plan)
        throw hipfftw_invalid_arg("plan argument cannot be nullptr.");
    if(plan->plan_dft_type != rocfft_transform_type_real_inverse)
        throw hipfftw_invalid_arg("this function requires a plan for real backward DFT(s), i.e., a "
                                  "plan created by any of the fftw_plan_*_c2r functions.");
    plan->new_array_execute(in, out);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftwf_execute(const fftwf_plan plan)
try
{
    if(!plan)
        throw hipfftw_invalid_arg("plan argument cannot be nullptr.");
    plan->execute();
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftwf_execute_dft(const fftwf_plan plan, fftwf_complex* in, fftwf_complex* out)
try
{
    if(!plan)
        throw hipfftw_invalid_arg("plan argument cannot be nullptr.");
    if(is_real(plan->plan_dft_type))
        throw hipfftw_invalid_arg(
            "this function rejects plans created for real DFT(s), i.e., plans created by any of "
            "the fftwf_plan_*_r2c or fftwf_plan_*_c2r functions.");
    plan->new_array_execute(in, out);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftwf_execute_dft_r2c(const fftwf_plan plan, float* in, fftwf_complex* out)
try
{
    if(!plan)
        throw hipfftw_invalid_arg("plan argument cannot be nullptr.");
    if(plan->plan_dft_type != rocfft_transform_type_real_forward)
        throw hipfftw_invalid_arg("this function requires a plan for real forward DFT(s), i.e., a "
                                  "plan created by any of the fftwf_plan_*_r2c functions.");
    plan->new_array_execute(in, out);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

void fftwf_execute_dft_c2r(const fftwf_plan plan, fftwf_complex* in, float* out)
try
{
    if(!plan)
        throw hipfftw_invalid_arg("plan argument cannot be nullptr.");
    if(plan->plan_dft_type != rocfft_transform_type_real_inverse)
        throw hipfftw_invalid_arg("this function requires a plan for real backward DFT(s), i.e., a "
                                  "plan created by any of the fftwf_plan_*_c2r functions.");
    plan->new_array_execute(in, out);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return;
}

fftw_plan fftw_plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 1;
    constexpr auto prec = rocfft_precision_double;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 1;
    constexpr auto prec = rocfft_precision_single;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan
    fftw_plan_dft_2d(int n0, int n1, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 2;
    constexpr auto prec = rocfft_precision_double;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n0, n1}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_2d(
    int n0, int n1, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 2;
    constexpr auto prec = rocfft_precision_single;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n0, n1}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan fftw_plan_dft_3d(
    int n0, int n1, int n2, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 3;
    constexpr auto prec = rocfft_precision_double;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n0, n1, n2}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_3d(
    int n0, int n1, int n2, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 3;
    constexpr auto prec = rocfft_precision_single;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n0, n1, n2}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan fftw_plan_dft(
    int rank, const int* n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
try
{
    constexpr auto prec = rocfft_precision_double;
    return hipfftw_create_default_unbatched_complex_plan<prec>(rank, n, sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft(
    int rank, const int* n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
try
{
    constexpr auto prec = rocfft_precision_single;
    return hipfftw_create_default_unbatched_complex_plan<prec>(rank, n, sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 1;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_r2c_1d(int n, float* in, fftwf_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 1;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double* in, fftw_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 2;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_r2c_2d(int n0, int n1, float* in, fftwf_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 2;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan
    fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double* in, fftw_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 3;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1, n2}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan
    fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float* in, fftwf_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 3;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1, n2}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c(int rank, const int* n, double* in, fftw_complex* out, unsigned flags)
try
{
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec>(rank, n, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_r2c(int rank, const int* n, float* in, fftwf_complex* out, unsigned flags)
try
{
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec>(rank, n, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex* in, double* out, unsigned flags)
try
{
    constexpr int  rank     = 1;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_c2r_1d(int n, fftwf_complex* in, float* out, unsigned flags)
try
{
    constexpr int  rank     = 1;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex* in, double* out, unsigned flags)
try
{
    constexpr int  rank     = 2;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex* in, float* out, unsigned flags)
try
{
    constexpr int  rank     = 2;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan
    fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex* in, double* out, unsigned flags)
try
{
    constexpr int  rank     = 3;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1, n2}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan
    fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex* in, float* out, unsigned flags)
try
{
    constexpr int  rank     = 3;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1, n2}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r(int rank, const int* n, fftw_complex* in, double* out, unsigned flags)
try
{
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec>(rank, n, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

fftwf_plan fftwf_plan_dft_c2r(int rank, const int* n, fftwf_complex* in, float* out, unsigned flags)
try
{
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec>(rank, n, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler(__func__);
    return nullptr;
}

void   fftw_print_plan(const fftw_plan) {}
void   fftwf_print_plan(const fftwf_plan) {}
void   fftw_set_timelimit(double) {}
void   fftwf_set_timelimit(double) {}
double fftw_cost(const fftw_plan)
{
    return 0.0;
}
double fftwf_cost(const fftw_plan)
{
    return 0.0;
}
void fftw_flops(const fftw_plan, double*, double*, double*) {}
void fftwf_flops(const fftw_plan, double*, double*, double*) {}
