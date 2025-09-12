/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "definitions.hpp"
#include "rocblas.h"
#include "rocblas_ostream.hpp"
#include "utility.hpp"
#include <array>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <memory>
#include <tuple>
#include <type_traits>
#ifdef WIN32
#include <stdio.h>
#define STDOUT_FILENO _fileno(stdout)
#define STDERR_FILENO _fileno(stderr)
#else
#include <unistd.h>
#endif
#include <utility>

// forward declare hipblaslt handle
typedef void* hipblasLtHandle_t;

// forcing early cleanup
extern "C" ROCBLAS_EXPORT void rocblas_shutdown();

// Round up size to the nearest MIN_CHUNK_SIZE
constexpr size_t roundup_device_memory_size(size_t size)
{
    size_t MIN_CHUNK_SIZE = 64;
    return ((size + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
}

// Empty base class for device memory allocation
struct rocblas_device_malloc_base
{
};

// enum representing state of rocBLAS device memory ownership
enum class rocblas_device_memory_ownership
{
    rocblas_managed,
    user_owned,
};

enum class Processor : int
{
    // matching enum used in hipGcnArch
    // only including supported types
    gfx803  = 803,
    gfx900  = 900,
    gfx906  = 906,
    gfx908  = 908,
    gfx90a  = 910,
    gfx942  = 942,
    gfx950  = 950,
    gfx1010 = 1010,
    gfx1011 = 1011,
    gfx1012 = 1012,
    gfx1030 = 1030,
    gfx1031 = 1031,
    gfx1032 = 1032,
    gfx1034 = 1034,
    gfx1035 = 1035,
    gfx1100 = 1100,
    gfx1101 = 1101,
    gfx1102 = 1102,
    gfx1103 = 1103,
    gfx1150 = 1150,
    gfx1151 = 1151,
    gfx1200 = 1200,
    gfx1201 = 1201
};

// helper function in handle.cpp
static rocblas_status free_existing_device_memory(rocblas_handle);

// declare data packet methods for internal API
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_set_data_ptr(rocblas_handle handle, std::shared_ptr<void>& data_ptr);
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_get_data_ptr(rocblas_handle handle, std::shared_ptr<void>& data_ptr);

// cached device properties for handle device
ROCBLAS_INTERNAL_EXPORT_NOINLINE const hipDeviceProp_t*
                                       rocblas_internal_get_device_prop(rocblas_handle handle);

/*******************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle() and the returned handle mus
 * It should be destroyed at the end using rocblas_destroy_handle().
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
 ******************************************************************************/
struct _rocblas_handle
{
private:
    // Emulate C++17 std::conjunction
    template <class...>
    struct conjunction : std::true_type
    {
    };
    template <class T, class... Ts>
    struct conjunction<T, Ts...> : std::integral_constant<bool, T{} && conjunction<Ts...>{}>
    {
    };

    // Class for saving and restoring default device ID
    // clang-format off
    class [[nodiscard]] _rocblas_saved_device_id
    {
        int device_id;
        int old_device_id;

    public:
        // Constructor
        explicit _rocblas_saved_device_id(int device_id)
            : device_id(device_id)
            , old_device_id(-1)
        {
            hipGetDevice(&old_device_id);
            if(device_id != old_device_id)
                hipSetDevice(device_id);
        }

        // Old device ID is restored on destruction
        ~_rocblas_saved_device_id()
        {
            if(device_id != old_device_id)
                hipSetDevice(old_device_id);
        }

        // Move constructor
        _rocblas_saved_device_id(_rocblas_saved_device_id&& other)
            : device_id(other.device_id)
            , old_device_id(other.old_device_id)
        {
            other.device_id = other.old_device_id;
        }

        _rocblas_saved_device_id(const _rocblas_saved_device_id&) = delete;
        _rocblas_saved_device_id& operator=(const _rocblas_saved_device_id&) = delete;
        _rocblas_saved_device_id& operator=(_rocblas_saved_device_id&&) = delete;
    };
    // clang-format on

    // Class for temporarily modifying a state, restoring it on destruction
    // clang-format off
    template <typename STATE>
    class [[nodiscard]] _pushed_state
    {
        STATE* statep;
        STATE  old_state;

    public:
        // Constructor
        _pushed_state(STATE& state, STATE new_state)
            : statep(&state)
            , old_state(std::move(state))
        {
            state = std::move(new_state);
        }

        // Temporary object implicitly converts to old state
        operator const STATE&() const&
        {
            return old_state;
        }

        // Old state is restored on destruction
        ~_pushed_state()
        {
            if(statep)
                *statep = std::move(old_state);
        }

        // Move constructor
        _pushed_state(_pushed_state&& other)
            : statep(other.statep)
            , old_state(std::move(other.old_state))
        {
            other.statep = nullptr;
        }

        _pushed_state(const _pushed_state&) = delete;
        _pushed_state& operator=(const _pushed_state&) = delete;
        _pushed_state& operator=(_pushed_state&&) = delete;
    };
    // clang-format on

public:
    _rocblas_handle();
    ~_rocblas_handle();

    _rocblas_handle(const _rocblas_handle&) = delete;
    _rocblas_handle& operator=(const _rocblas_handle&) = delete;

    // Set the HIP default device ID to the handle's device ID, and restore on exit
    auto push_device_id()
    {
        return _rocblas_saved_device_id(device);
    }

    int getDevice()
    {
        return device;
    }

    int getArch()
    {
        return arch;
    }

    int getArchMajor()
    {
        return archMajor;
    }

    int getArchMajorMinor()
    {
        return archMajorMinor;
    }

    int getWarpSize()
    {
        return mWarpSize;
    }

    int getMaxSharedMemPerBlock()
    {
        // TODO review sharedMemPerBlockOptin or use cached device_properties
        int max_mem = -1;
        THROW_IF_HIP_ERROR(hipDeviceGetAttribute(
            &max_mem, hipDeviceAttribute_t(hipDeviceAttributeMaxSharedMemoryPerBlock), device));
        return max_mem;
    }

    bool isYZGridDim16bit()
    {
        return archMajor == 12;
    }

    int getBatchGridDim(int batch_count)
    {
        // c_YZ_grid_launch_limit <= min(MaxGridSize[2], MaxGridSize[3]) from hipDeviceProp.MaxGridSize[3]
        // in file /opt/rocm/include/hip/hip_runtime_api.h
        // This function returns a grid size that will not exceed c_YZ_grid_launch_limit
        // for now we are using a simple constant as there is only one variability (unsigned 16-bit/32-bit)
        if(isYZGridDim16bit())
            return std::min(batch_count, int(c_YZ_grid_launch_limit));
        else
            return batch_count;
    }

    bool isDefaultHipBLASLtArch()
    {
        int gfx_arch = getArch();
        if(gfx_arch == 1200 || gfx_arch == 1201 || gfx_arch == 950)
        {
            return true;
        }
        return false;
    }

    auto getHipblasLtHandle()
    {
        return hipblasLtHandle;
    }

    /*******************************************************************************
     * This function determines whether or not to try using the hipBLASLt backend
     * - If the enviornment variable is set, its value determines whether ot not to
     *   try the hipBLASLt backend.
     * - Otherwise try when the current architecture is defaulted to hipBLASLt support
     * - Always disable for any `batched` API when the current handle is in stream
     *   capture mode (as hipblaslt batched dispatch does synchronous memory copies)
     * - batched mode requires env variable opt in as has additional pointer copies
     ******************************************************************************/
    bool tryHipBLASLt(bool batched)
    {
        bool status = false;

#ifdef BUILD_WITH_HIPBLASLT
        if(hipblasltEnvVar < 0)
        {
            if(isDefaultHipBLASLtArch())
            {
                status = true;
            }
        }
        else
            status = hipblasltEnvVar == 1;
#endif

        if(status && batched)
        {
            static bool hipblasltEnvBatchedDisabled = [&] {
                auto* env_var = getenv("ROCBLAS_USE_HIPBLASLT_BATCHED");
                if(env_var)
                {
                    return strncmp(env_var, "0", 1) == 0;
                }
                return false;
            }();

            // only use for batched when explicitly enabled by env variable
            if(!hipblasltEnvBatchedDisabled)
                status = !is_stream_in_capture_mode();
            else
                status = false;
        }

        return status;
    }

    bool isHipBLASLtEnabled()
    {
        bool status = false;

#ifdef BUILD_WITH_HIPBLASLT
        if(hipblasltEnvVar < 0)
        {
            if(isDefaultHipBLASLtArch())
            {
                status = true;
            }
        }
        else
            status = hipblasltEnvVar == 1;
#endif

        return status;
    }

    /**
     * @brief Checks if the hipBLASLt feature is explicitly enabled by the user.
     *
     * This method determines whether the hipBLASLt feature is forced on by
     * checking the value of the `hipblasltEnvVar` environment variable.
     *
     * @return true if the `hipblasltEnvVar` is set to 1, indicating that the
     *         feature is explicitly enabled by the user; false otherwise.
     */
    bool isHipBLASLtForcedOn()
    {
        // Only true if the user has set the environment variable on
        return hipblasltEnvVar == 1;
    }

    inline int getDefaultDeviceMemorySize()
    {

        if(getArchMajor() == 9 && getArchMajorMinor() >= 94)
        {
            return DEFAULT_DEVICE_MEMORY_SIZE_EXTENDED;
        }
        else
        {
            return DEFAULT_DEVICE_MEMORY_SIZE;
        }
    }

    // hipEvent_t pointers (for internal use only)
    hipEvent_t startEvent = nullptr;
    hipEvent_t stopEvent  = nullptr;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // default logging_mode is no logging
    rocblas_layer_mode layer_mode = rocblas_layer_mode_none;

    // default atomics mode does not allows atomic operations
    rocblas_atomics_mode atomics_mode = rocblas_atomics_not_allowed;

    // Selects the benchmark library to be used for solution selection
    rocblas_performance_metric performance_metric = rocblas_default_performance_metric;

    // default check_numerics_mode is no numeric_check
    rocblas_check_numerics_mode check_numerics = rocblas_check_numerics_mode_no_check;

    // default math_mode is default_math
    rocblas_math_mode math_mode = rocblas_default_math;

    // logging streams
    std::unique_ptr<rocblas_internal_ostream> log_trace_os;
    std::unique_ptr<rocblas_internal_ostream> log_bench_os;
    std::unique_ptr<rocblas_internal_ostream> log_profile_os;
    void                                      init_logging();
    void                                      init_check_numerics();

    // data pointer for rocSOLVER
    std::shared_ptr<void> data_ptr;

    void get_data_ptr(std::shared_ptr<void>& data_ptr) const
    {
        data_ptr = this->data_ptr;
    }
    void set_data_ptr(std::shared_ptr<void>& data_ptr)
    {
        this->data_ptr = data_ptr;
    }

    // C interfaces for manipulating device memory
    friend rocblas_status(::rocblas_start_device_memory_size_query)(_rocblas_handle*);
    friend rocblas_status(::rocblas_stop_device_memory_size_query)(_rocblas_handle*, size_t*);
    friend rocblas_status(::rocblas_get_device_memory_size)(_rocblas_handle*, size_t*);
    friend rocblas_status(::free_existing_device_memory)(rocblas_handle);
    friend rocblas_status(::rocblas_set_workspace)(_rocblas_handle*, void*, size_t);
    friend bool(::rocblas_is_managing_device_memory)(_rocblas_handle*);
    friend rocblas_status(::rocblas_set_stream)(_rocblas_handle*, hipStream_t);

    // C interfaces that interact with the solution selection process
    friend rocblas_status(::rocblas_set_solution_fitness_query)(_rocblas_handle*, double*);
    friend rocblas_status(::rocblas_set_performance_metric)(_rocblas_handle*,
                                                            rocblas_performance_metric);
    friend rocblas_status(::rocblas_get_performance_metric)(_rocblas_handle*,
                                                            rocblas_performance_metric*);

    // Returns whether the current kernel call is a device memory size query
    bool is_device_memory_size_query() const
    {
        return device_memory_size_query;
    }

    size_t get_available_workspace()
    {
        return (device_memory_size - device_memory_in_use);
    }

    // Get the solution fitness query
    auto* get_solution_fitness_query() const
    {
        return solution_fitness_query;
    }

    void set_stream_order_memory_allocation(bool flag)
    {
        stream_order_alloc = flag;
    }

    // Sets the optimal size(s) of device memory for a kernel call
    // Maximum size is accumulated in device_memory_query_size
    // Returns rocblas_status_size_increased or rocblas_status_size_unchanged
    template <typename... Ss,
              std::enable_if_t<sizeof...(Ss) && conjunction<std::is_convertible<Ss, size_t>...>{},
                               int> = 0>
    rocblas_status set_optimal_device_memory_size(Ss... sizes)
    {
        if(!device_memory_size_query)
            return rocblas_status_size_query_mismatch;

#if __cplusplus >= 201703L
        // Compute the total size, rounding up each size to multiples of MIN_CHUNK_SIZE
        size_t total = (roundup_device_memory_size(sizes) + ...);
#else
        size_t total = 0;
        auto   dummy = {total += roundup_device_memory_size(sizes)...};
#endif

        return total > device_memory_query_size ? device_memory_query_size = total,
                                                  rocblas_status_size_increased
                                                : rocblas_status_size_unchanged;
    }

    // Temporarily change pointer mode, returning object which restores old mode when destroyed
    auto push_pointer_mode(rocblas_pointer_mode mode)
    {
        return _pushed_state<rocblas_pointer_mode>(pointer_mode, mode);
    }

    // Whether to use any_order scheduling in Tensile calls
    bool any_order = false;

    // Temporarily change any_order flag
    auto push_any_order(bool new_any_order)
    {
        return _pushed_state<bool>(any_order, new_any_order);
    }

    // Return the current stream
    hipStream_t get_stream() const
    {
        return stream;
    }

    bool is_stream_in_capture_mode()
    {
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        //default stream will not be in capture mode
        if(stream != 0)
        {
            hipError_t status = hipStreamIsCapturing(stream, &capture_status);
            if(status != hipSuccess)
            {
                //Avoid warning users about hipErrorContextIsDestroyed error on a destroyed stream,
                //as it may not be in capture mode, and such notifications would be unnecessary.
                if(status != hipErrorContextIsDestroyed)
                {
                    PRINT_IF_HIP_ERROR(status);
                }
                return false;
            }
        }

        if(capture_status == hipStreamCaptureStatusNone)
            return false;
        else
            return true; // returns true for both hipStreamCaptureStatusActive & hipStreamCaptureStatusInvalidated
    }

private:
    // device memory work buffer
    static constexpr size_t DEFAULT_DEVICE_MEMORY_SIZE          = 32 * 1024 * 1024;
    static constexpr size_t DEFAULT_DEVICE_MEMORY_SIZE_EXTENDED = 128 * 1024 * 1024;

    // Variables holding state of device memory allocation
    void*                           device_memory              = nullptr;
    size_t                          device_memory_size         = 0;
    size_t                          device_memory_in_use       = 0;
    bool                            device_memory_size_query   = false;
    bool                            alpha_beta_memcpy_complete = false;
    rocblas_device_memory_ownership device_memory_owner
        = rocblas_device_memory_ownership::rocblas_managed;
    size_t device_memory_query_size;

    bool stream_order_alloc = false;

    // Solution fitness query (used for internal testing)
    double* solution_fitness_query = nullptr;

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t stream = 0;

public:
    hipDeviceProp_t device_properties;

private:
    // Device ID is created at handle creation time and remains in effect for the life of the handle.
    const int device;

    // Arch ID is created at handle creation time and remains in effect for the life of the handle.
    const int arch;
    int       archMajor;
    int       archMajorMinor;

    const int mWarpSize;

    // hipBLASLt handle is created at handle creation time and remains in effect for the life of the handle.
    std::shared_ptr<hipblasLtHandle_t> hipblasLtHandle;
    int                                hipblasltEnvVar = -1;

    // used in constructor initialization list
    int       getActiveDevice();
    Processor getActiveArch();

    // Opaque smart allocator class to perform device memory allocations
    // clang-format off
    class [[nodiscard]] _device_malloc : public rocblas_device_malloc_base
    {
    protected:
        // Order is important (pointers member declared last):
        rocblas_handle handle;
        size_t         prev_device_memory_in_use;
        size_t         size;
        void*          dev_mem = nullptr;
        hipStream_t    stream_in_use;
        bool           success;

    private:
        std::vector<void*> pointers; // Important: must come last

        // Allocate one or more pointers to buffers of different sizes
        template <typename... Ss>
        decltype(pointers) allocate_pointers(Ss... sizes)
        {
            // This creates a list of partial sums which are the offsets of each of the allocated
            // arrays. The sizes are rounded up to the next multiple of MIN_CHUNK_SIZE.
            // size contains the total of all sizes at the end of the calculation of offsets.
            size = 0;
            size_t old;
            const size_t offsets[] = {(old = size, size += roundup_device_memory_size(sizes), old)...};
            char* addr = nullptr;

            if( handle->device_memory_owner == rocblas_device_memory_ownership::rocblas_managed)
            {
                if(!size)
                    return decltype(pointers)(sizeof...(sizes));

                hipError_t hipStatus = hipMallocAsync(&dev_mem, size, stream_in_use);

                if(hipStatus != hipSuccess)
                {
                    success = false;
                    rocblas_cerr << " rocBLAS internal error: hipMallocAsync() failed to allocate memory of size : " << size << std::endl;
                    return decltype(pointers)(sizeof...(sizes));
                }
                addr = static_cast<char*>(dev_mem);
            }
            else if (handle->device_memory_owner == rocblas_device_memory_ownership::user_owned)
            {
                success = size <= handle->device_memory_size - handle->device_memory_in_use;

                // If allocation failed, return an array of nullptr's
                // If total size is 0, return an array of nullptr's, but leave it marked as successful
                if(!success || !size)
                    return decltype(pointers)(sizeof...(sizes));

                // We allocate the total amount needed, taking it from the available device memory.
                addr = static_cast<char*>(handle->device_memory) + handle->device_memory_in_use;
                handle->device_memory_in_use += size;
            }

            // An array of pointers to all of the allocated arrays is formed.
            // If a size is 0, the corresponding pointer is nullptr
            size_t i = 0;
            // cppcheck-suppress arrayIndexOutOfBounds
            return {!sizes ? i++, nullptr : addr + offsets[i++]...};
        }

    public:
        // Constructor
        template <typename... Ss>
        explicit _device_malloc(rocblas_handle handle, Ss... sizes)
            : handle(handle)
            , prev_device_memory_in_use(handle->device_memory_in_use)
            , size(0)
            , stream_in_use(handle->stream)
            , success(true)
            , pointers(allocate_pointers(size_t(sizes)...))
        {

        }

        // Constructor for allocating count pointers of a certain total size
        explicit _device_malloc(rocblas_handle handle, std::nullptr_t, size_t count, size_t total)
            : handle(handle)
            , prev_device_memory_in_use(handle->device_memory_in_use)
            , size(roundup_device_memory_size(total))
            , stream_in_use(handle->stream)
            , success(true)
        {
            if( handle->device_memory_owner == rocblas_device_memory_ownership::rocblas_managed)
            {
                bool status = hipMallocAsync(&dev_mem, size, stream_in_use) == hipSuccess ;

                for(auto i= 0 ; i < count ; i++)
                    pointers.push_back(status ? dev_mem : nullptr);
            }
             else if (handle->device_memory_owner == rocblas_device_memory_ownership::user_owned)
            {

                success = size <= handle->device_memory_size - handle->device_memory_in_use;
                for(auto i= 0 ; i < count ; i++)
                {    pointers.push_back(success ? static_cast<char*>(handle->device_memory)
                                            + handle->device_memory_in_use : nullptr);
                }

                if(success)
                    handle->device_memory_in_use += size;
            }

        }

        // Move constructor
        // Warning: This should only be used to move temporary expressions,
        // such as the return values of functions and initialization with
        // rvalues. If std::move() is used to move a _device_malloc object
        // from a variable, then there must not be any alive allocations made
        // between the initialization of the variable and the object that it
        // moves to, or the LIFO ordering will be violated and flagged.
        _device_malloc(_device_malloc&& other) noexcept
            : handle(other.handle)
            , prev_device_memory_in_use(other.prev_device_memory_in_use)
            , size(other.size)
            , dev_mem(other.dev_mem)
            , stream_in_use(other.stream_in_use)
            , success(other.success)
            , pointers(std::move(other.pointers))
        {
            other.success = false;
        }

        // Move assignment is allowed as long as the object being assigned to
        // is 0-sized or an unsuccessful previous allocation.
        _device_malloc& operator=(_device_malloc&& other) & noexcept
        {
            this->~_device_malloc();
            return *new(this) _device_malloc(std::move(other));
        }

        // Copying and copy-assignment are deleted
        _device_malloc(const _device_malloc&) = delete;
        _device_malloc& operator=(const _device_malloc&) = delete;

        // The destructor marks the device memory as no longer in use
        ~_device_malloc()
        {
            // If success == false or size == 0, the destructor is a no-op
            if(success && size)
            {
                if( handle->device_memory_owner == rocblas_device_memory_ownership::rocblas_managed)
                {
                        if(dev_mem)
                        {
                            bool status = hipFreeAsync(dev_mem, stream_in_use) == hipSuccess ;
                            if(!status)
                            {
                                rocblas_cerr << " rocBLAS internal error: hipFreeAsync() Failed, "
                                "device memory could not be released to default memory pool" << std::endl;
                                rocblas_abort();
                            }
                            dev_mem = nullptr;
                        }

                }
                else if (handle->device_memory_owner == rocblas_device_memory_ownership::user_owned)
                {
                    // Subtract size from the handle's device_memory_in_use, making sure
                    // it matches the device_memory_in_use when this object was created.
                    if((handle->device_memory_in_use -= size) != prev_device_memory_in_use)
                    {
                        rocblas_cerr
                            << "rocBLAS internal error: device_malloc() RAII object not "
                            "destroyed in LIFO order.\n"
                            "Objects returned by device_malloc() must be 0-sized, "
                            "unsuccessfully allocated,\n"
                            "or destroyed in the reverse order that they are created.\n"
                            "device_malloc() objects cannot be assigned to unless they are 0-sized\n"
                            "or they were unsuccessfully allocated previously."
                            << std::endl;
                        rocblas_abort();
                    }
                }


                handle->gsu_workspace_size = 0;
                handle->gsu_workspace      = nullptr;
            }
        }

        // In the following functions, the trailing & prevents the functions from
        // applying to rvalue temporaries, to catch common mistakes such as:
        // void *p = (void*) handle->device_malloc(), which is a dangling pointer.

        // Conversion to bool to tell if the allocation succeeded
        explicit operator bool() &
        {
            return success;
        }

        // Return the ith pointer
        void*& operator[](size_t i) &
        {
            return pointers.at(i);
        }

        // Conversion to any pointer type (if pointers.size() == 1)
        template <typename T>
        explicit operator T*() &
        {
            // Index 1 - pointers.size() is used to make at() throw if size() != 1
            // but to otherwise return the first element.
            return static_cast<T*>(pointers.at(1 - pointers.size()));
        }
    };

    // clang-format on

    // Allocate workspace for GSU based on the needs.
    // clang-format off
    class [[nodiscard]] _gsu_malloc_by_size final : public _device_malloc
    {
    public:
        explicit _gsu_malloc_by_size(rocblas_handle handle, size_t requested_Workspace_Size)
        : _device_malloc(handle, requested_Workspace_Size)
        {
            handle->gsu_workspace_size = success ? size : 0;
            handle->gsu_workspace = static_cast<void*>(*this);
        }

        _gsu_malloc_by_size(rocblas_handle handle)
        : _device_malloc(handle, 0)
        {

        }

        ~_gsu_malloc_by_size()
        {
            if(success)
            {
                handle->gsu_workspace_size = 0;
                handle->gsu_workspace      = nullptr;
            }
        }

        // Move constructor allows initialization by rvalues and returns from functions
        _gsu_malloc_by_size(_gsu_malloc_by_size&&) = default;
    };
    // clang-format on

public:
    // Allocate one or more sizes
    template <typename... Ss,
              std::enable_if_t<sizeof...(Ss) && conjunction<std::is_convertible<Ss, size_t>...>{},
                               int> = 0>
    auto device_malloc(Ss... sizes)
    {
        return _device_malloc(this, size_t(sizes)...);
    }

    template <typename... Ss,
              std::enable_if_t<sizeof...(Ss) && conjunction<std::is_convertible<Ss, size_t>...>{},
                               int> = 0>
    auto device_malloc_with_GSU(Ss... sizes) //assume last size is gsu size
    {
        //have to assume it can be called to resize in following iteration (resize if > cur?)
        //next request enough alloc, assign pointer

        size_t i      = 0;
        size_t mine[] = {(i++, size_t(sizes))...};

        auto result = _device_malloc(this, size_t(sizes)...);

        this->gsu_workspace_size = result ? mine[i - 1] : 0;
        this->gsu_workspace      = result ? result[i - 1] : nullptr;

        return result;
    }

    // Allocate count pointers, reserving "size" total bytes
    auto device_malloc_count(size_t count, size_t size)
    {
        return _device_malloc(this, nullptr, count, size);
    }

    // Variables holding state of GSU device memory allocation
    size_t gsu_workspace_size = 0;
    void*  gsu_workspace      = nullptr;

    auto gsu_malloc_by_size(size_t requested_Workspace_Size)
    {
        if(this->gsu_workspace) // Added to accomodate quant, remove comment after testing
            return _gsu_malloc_by_size(this);

        return _gsu_malloc_by_size(this, requested_Workspace_Size);
    };
};

// For functions which don't use temporary device memory, and won't be likely
// to use them in the future, the RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle)
// macro can be used to return from a rocblas function with a requested size of 0.
#define RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(h) \
    do                                               \
    {                                                \
        if((h)->is_device_memory_size_query())       \
            return rocblas_status_size_unchanged;    \
    } while(0)

// Warn about potentially unsafe and synchronizing uses of hipMalloc and hipFree
#define hipMalloc(ptr, size)                                                                     \
    _Pragma(                                                                                     \
        "GCC warning \"Direct use of hipMalloc in rocBLAS is deprecated; see CONTRIBUTING.md\"") \
        hipMalloc(ptr, size)
#define hipFree(ptr)                                                                               \
    _Pragma("GCC warning \"Direct use of hipFree in rocBLAS is deprecated; see CONTRIBUTING.md\"") \
        hipFree(ptr)
