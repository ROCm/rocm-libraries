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
#include "handle.hpp"
#include <cstdarg>
#include <limits>

#ifdef WIN32
#include <windows.h>
#endif

#ifdef BUILD_WITH_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

#ifdef BUILD_WITH_TENSILE
#else
// see TensileHost.cpp for normal rocblas_initialize definition
// it isn't compiled if not BUILD_WITH_TENSILE so defining here
extern "C" void rocblas_initialize() {}
#endif

// forcing early cleanup
extern "C" void rocblas_shutdown()
{
    rocblas_internal_ostream::clear_workers();
}

/* read environment variable */
/* On windows, getenv take a copy of the environment at the beginning of the process */
/* This behavior is not suited for the purpose of the tests */
static const char* read_env(const char* env_var)
{
#ifdef WIN32
    const DWORD              nSize = _MAX_PATH;
    static thread_local char lpBuffer[nSize];
    lpBuffer[0] = 0; // terminate for reuse
    if(GetEnvironmentVariableA(env_var, lpBuffer, nSize) == 0)
        return nullptr;
    else
        return lpBuffer;
#else
    return getenv(env_var);
#endif
}

// This variable can be set in hipBLAS or other libraries to change the default
// device memory size
static thread_local size_t t_rocblas_device_malloc_default_memory_size;

extern "C" void rocblas_device_malloc_set_default_memory_size(size_t size)
{
    t_rocblas_device_malloc_default_memory_size = size;
}

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle()
    : device(getActiveDevice()) // getActiveDevice populates device_properties struct
    , arch(static_cast<int>(getActiveArch()))
    , mWarpSize(device_properties.warpSize)
{
    archMajor      = arch / 100; // this may need to switch to string handling in the future
    archMajorMinor = arch / 10;

    //ROCBLAS_DEFAULT_ATOMICS_MODE
    const char* atomics_mode_env = read_env("ROCBLAS_DEFAULT_ATOMICS_MODE");
    if(atomics_mode_env)
    {
        atomics_mode = strtoul(atomics_mode_env, nullptr, 0) ? rocblas_atomics_allowed
                                                             : rocblas_atomics_not_allowed;
    }
    // Device memory size
    const char* env = read_env("ROCBLAS_DEVICE_MEMORY_SIZE");
    if(env)
        device_memory_size = strtoul(env, nullptr, 0);

    // The following allocation & free of device memory using hipMallocAsync/hipFreeAsync will allocate memory from
    // the OS and release it to default memory pool. Further allocation of memory using hipMallocAsync
    // will be from the memory pool and it will be faster.
    if(env && device_memory_size)
    {
        THROW_IF_HIP_ERROR((hipMallocAsync)(&device_memory, device_memory_size, stream));

        THROW_IF_HIP_ERROR((hipFreeAsync)(device_memory, stream));
    }
    else //uses default memory size
    {
        THROW_IF_HIP_ERROR((hipMallocAsync)(&device_memory, getDefaultDeviceMemorySize(), stream));

        THROW_IF_HIP_ERROR((hipFreeAsync)(device_memory, stream));
    }

    device_memory = nullptr;

    // Initialize logging
    init_logging();

    // Initialize numerical checking
    init_check_numerics();

#ifdef BUILD_WITH_HIPBLASLT
    const char* hipblasltEnvVal = read_env("ROCBLAS_USE_HIPBLASLT");

    if(hipblasltEnvVal)
    {
        if(strncmp(hipblasltEnvVal, "1", 1) == 0)
        {
            hipblasltEnvVar = 1;
        }
        else
        {
            hipblasltEnvVar = 0;
        }
    }
    else
    {
        hipblasltEnvVar = -1;
    }

    if(isHipBLASLtEnabled())
    {
        hipblasLtHandle                = std::make_shared<hipblasLtHandle_t>();
        hipblasStatus_t hipblas_status = hipblasLtCreate(&(*hipblasLtHandle));
        if(HIPBLAS_STATUS_SUCCESS != hipblas_status)
        {
            rocblas_cerr << "rocBLAS internal error: Unable to initialize hipblaslt: "
                         << hipblas_status << std::endl;
            rocblas_abort();
        }
    }
#endif
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    if(device_memory_in_use)
    {
        rocblas_cerr
            << "rocBLAS internal error: Handle object destroyed while device memory still in use."
            << std::endl;
        rocblas_abort();
    }

    // Free device memory unless it's user-owned
    if(device_memory_owner != rocblas_device_memory_ownership::user_owned)
    {
        hipError_t hipStatus;

        hipStatus = (device_memory) ? (hipFreeAsync)(device_memory, stream) : hipSuccess;
        if(hipStatus != hipSuccess)
        {
            rocblas_cerr << "rocBLAS error during freeing of allocated memory in handle "
                            "destructor (stream order allocation): "
                         << rocblas_status_to_string(
                                rocblas_internal_convert_hip_to_rocblas_status(hipStatus))
                         << std::endl;
            rocblas_abort();
        };

        hipMemPool_t mem_pool;
        int          device;
        hipStatus = hipGetDevice(&device);
        if(hipStatus != hipSuccess)
        {
            rocblas_cerr << "rocBLAS error retreiving the device (deviceID: " << device << ")"
                         << std::endl;
            rocblas_abort();
        }
        hipStatus = hipDeviceGetDefaultMemPool(&mem_pool, device);
        if(hipStatus != hipSuccess)
        {
            rocblas_cerr << "rocBLAS error retreiving the device's memory pool (deviceID: "
                         << device << ")" << std::endl;
            rocblas_abort();
        }
        //Releases device memory back to OS
        hipStatus = hipMemPoolTrimTo(mem_pool, 0);
        if(hipStatus != hipSuccess)
        {
            rocblas_cerr << "rocBLAS error releasing the device's memory pool (deviceID: " << device
                         << ")" << std::endl;
            rocblas_abort();
        }
    }

#ifdef BUILD_WITH_HIPBLASLT
    if(hipblasLtHandle.unique())
    {
        hipblasStatus_t hipblas_status = hipblasLtDestroy(*hipblasLtHandle);
        if(HIPBLAS_STATUS_SUCCESS != hipblas_status)
        {
            rocblas_cerr << "rocBLAS internal error: Unable to destroy hipblaslt: "
                         << hipblas_status << std::endl;
            rocblas_abort();
        }
        hipblasLtHandle.reset();
    }
#endif
}

int _rocblas_handle::getActiveDevice()
{
    int deviceId;
    THROW_IF_HIP_ERROR(hipGetDevice(&deviceId));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&device_properties, deviceId));
    return deviceId;
}

Processor _rocblas_handle::getActiveArch()
{
    // strip out xnack/ecc from name
    std::string deviceFullString(device_properties.gcnArchName);
    std::string deviceString = deviceFullString.substr(0, deviceFullString.find(":"));

    if(deviceString.find("gfx803") != std::string::npos)
    {
        return Processor::gfx803;
    }
    else if(deviceString.find("gfx900") != std::string::npos)
    {
        return Processor::gfx900;
    }
    else if(deviceString.find("gfx906") != std::string::npos)
    {
        return Processor::gfx906;
    }
    else if(deviceString.find("gfx908") != std::string::npos)
    {
        return Processor::gfx908;
    }
    else if(deviceString.find("gfx90a") != std::string::npos)
    {
        return Processor::gfx90a;
    }
    else if(deviceString.find("gfx942") != std::string::npos)
    {
        return Processor::gfx942;
    }
    else if(deviceString.find("gfx950") != std::string::npos)
    {
        return Processor::gfx950;
    }
    else if(deviceString.find("gfx1010") != std::string::npos)
    {
        return Processor::gfx1010;
    }
    else if(deviceString.find("gfx1011") != std::string::npos)
    {
        return Processor::gfx1011;
    }
    else if(deviceString.find("gfx1012") != std::string::npos)
    {
        return Processor::gfx1012;
    }
    else if(deviceString.find("gfx1030") != std::string::npos)
    {
        return Processor::gfx1030;
    }
    else if(deviceString.find("gfx1100") != std::string::npos)
    {
        return Processor::gfx1100;
    }
    else if(deviceString.find("gfx1101") != std::string::npos)
    {
        return Processor::gfx1101;
    }
    else if(deviceString.find("gfx1102") != std::string::npos)
    {
        return Processor::gfx1102;
    }
    else if(deviceString.find("gfx1103") != std::string::npos)
    {
        return Processor::gfx1103;
    }
    else if(deviceString.find("gfx1150") != std::string::npos)
    {
        return Processor::gfx1150;
    }
    else if(deviceString.find("gfx1151") != std::string::npos)
    {
        return Processor::gfx1151;
    }
    else if(deviceString.find("gfx1200") != std::string::npos)
    {
        return Processor::gfx1200;
    }
    else if(deviceString.find("gfx1201") != std::string::npos)
    {
        return Processor::gfx1201;
    }
    return static_cast<Processor>(0);
}

/*******************************************************************************
 * Numeric_check initialization
 ******************************************************************************/
void _rocblas_handle::init_check_numerics()
{
    // set check_numerics from value of environment variable ROCBLAS_CHECK_NUMERICS
    const char* str_check_numerics_mode = read_env("ROCBLAS_CHECK_NUMERICS");
    if(str_check_numerics_mode)
    {
        check_numerics
            = static_cast<rocblas_check_numerics_mode>(strtol(str_check_numerics_mode, 0, 0));
    }
}

/*******************************************************************************
 * Set the external data packet pointer
 ******************************************************************************/
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_set_data_ptr(rocblas_handle handle, std::shared_ptr<void>& data_ptr)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    handle->set_data_ptr(data_ptr);
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Get the external data packet pointer
 ******************************************************************************/
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_get_data_ptr(rocblas_handle handle, std::shared_ptr<void>& data_ptr)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    handle->get_data_ptr(data_ptr);
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Get the handle cached const hipDeviceProp_t pointer
 ******************************************************************************/
ROCBLAS_INTERNAL_EXPORT_NOINLINE
const hipDeviceProp_t* rocblas_internal_get_device_prop(rocblas_handle handle)
try
{
    if(!handle)
        return nullptr;
    return &handle->device_properties;
}
catch(...)
{
    return nullptr;
}

/*******************************************************************************
 * start device memory size queries
 ******************************************************************************/
extern "C" rocblas_status rocblas_start_device_memory_size_query(rocblas_handle handle)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->device_memory_size_query)
        return rocblas_status_size_query_mismatch;
    handle->device_memory_size_query = true;
    handle->device_memory_query_size = 0;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * stop device memory size queries
 ******************************************************************************/
extern "C" rocblas_status rocblas_stop_device_memory_size_query(rocblas_handle handle, size_t* size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!handle->device_memory_size_query)
        return rocblas_status_size_query_mismatch;
    if(!size)
        return rocblas_status_invalid_pointer;
    *size                            = handle->device_memory_query_size;
    handle->device_memory_size_query = false;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Get the device memory size
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_device_memory_size(rocblas_handle handle, size_t* size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!size)
        return rocblas_status_invalid_pointer;
    *size = handle->device_memory_size;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Free any allocated memory unless owned by user, and reset the handle to being
 * rocBLAS-managed
 ******************************************************************************/
static rocblas_status free_existing_device_memory(rocblas_handle handle)
{
    // Cannot change memory allocation when a device_malloc object is alive and
    // using device memory. This should never happen unless this function is
    // called from inside library code which borrows allocated device memory.
    if(handle->device_memory_in_use)
        return rocblas_status_internal_error;

    // Free existing device memory in handle, unless owned by user
    if(handle->device_memory
       && handle->device_memory_owner != rocblas_device_memory_ownership::user_owned)
    {
        RETURN_IF_HIP_ERROR((hipFreeAsync)(handle->device_memory, handle->stream));
    }

    // Clear the memory size and address, and set the memory to be rocBLAS-managed
    handle->device_memory_size  = 0;
    handle->device_memory       = nullptr;
    handle->device_memory_owner = rocblas_device_memory_ownership::rocblas_managed;

    return rocblas_status_success;
}

/*******************************************************************************
 * Set the device memory size
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_device_memory_size(rocblas_handle handle, size_t size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Set the device memory workspace
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_workspace(rocblas_handle handle, void* addr, size_t size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // Temporarily change the thread's default device ID to the handle's device ID
    auto saved_device_id = handle->push_device_id();

    // Free any allocated memory unless owned by user, and set device memory to
    // the default of being rocBLAS-managed
    rocblas_status status = free_existing_device_memory(handle);
    if(status != rocblas_status_success)
        return status;

    // For nonzero size and non-nullptr address, mark device memory as user-owned,
    // with a specific size and address; otherwise leave it as rocBLAS-managed
    if(size && addr)
    {
        handle->device_memory_owner = rocblas_device_memory_ownership::user_owned;
        handle->device_memory_size  = size;
        handle->device_memory       = addr;
    }

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Returns whether device memory is rocblas-managed
 ******************************************************************************/
extern "C" bool rocblas_is_managing_device_memory(rocblas_handle handle)
{
    return handle
           && handle->device_memory_owner == rocblas_device_memory_ownership::rocblas_managed;
}

/*******************************************************************************
 * Returns whether device memory is user-managed
 ******************************************************************************/
extern "C" bool rocblas_is_user_managing_device_memory(rocblas_handle handle)
{
    return false;
}

/* \brief
   \details
   Returns true if the handle is in device memory size query mode.
   @param[in]
   handle           rocblas handle
 ******************************************************************************/
extern "C" bool rocblas_is_device_memory_size_query(rocblas_handle handle)
{
    return handle && handle->is_device_memory_size_query();
}

// Helper function to round up sizes and compute total size
static inline size_t va_total_device_memory_size(size_t count, va_list ap)
{
    size_t total = 0;
    while(count--)
        total += roundup_device_memory_size(va_arg(ap, size_t));
    return total;
}

/* \brief
   \details
   Sets the optimal device memory size during a query
   Returns rocblas_status_size_increased if the maximum size was increased,
   rocblas_status_size_unchanged if the maximum size was unchanged, or
   rocblas_status_size_query_mismatch if the handle is not in query mode.
   @param[in]
   handle           rocblas handle
   count            number of sizes
   ...              sizes needed for optimal execution of the current kernel
 ******************************************************************************/
extern "C" rocblas_status
    rocblas_set_optimal_device_memory_size_impl(rocblas_handle handle, size_t count, ...)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    va_list ap;
    va_start(ap, count);
    size_t total = va_total_device_memory_size(count, ap);
    va_end(ap);
    return handle->set_optimal_device_memory_size(total);
}

/*! \brief
    \details
    Borrows size bytes from the device memory allocated in handle.
    Returns rocblas_status_invalid_handle if handle is nullptr; rocblas_status_invalid_pointer if res is nullptr; otherwise rocblas_status_success
    @param[in]
    handle          rocblas handle
    count           number of sizes
    ...             sizes to allocate
    @param[out]
    res             pointer to pointer to struct rocblas_device_malloc_base
 ******************************************************************************/
extern "C" rocblas_status rocblas_device_malloc_alloc(rocblas_handle               handle,
                                                      rocblas_device_malloc_base** res,
                                                      size_t                       count,
                                                      ...)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!res)
        return rocblas_status_invalid_pointer;
    if(!count)
        return rocblas_status_invalid_size;

    *res = nullptr; // in case of exception

    // Compute the total of the rounded up sizes
    va_list ap;
    va_start(ap, count);
    size_t total = va_total_device_memory_size(count, ap);
    va_end(ap);

    // Borrow allocated memory from the handle
    auto mem = handle->device_malloc_count(count, total);

    // If unsuccessful
    if(!mem)
        return rocblas_status_memory_error;

    // Get the base of the allocated pointers
    char* addr = static_cast<char*>(mem[0]);

    // Compute each pointer based on offsets
    va_start(ap, count);
    for(size_t i = 0; i < count; ++i)
    {
        size_t size = roundup_device_memory_size(va_arg(ap, size_t));
        mem[i]      = size ? addr : nullptr;
        addr += size;
    }
    va_end(ap);

    // Move it to the heap
    *res = new auto(std::move(mem));

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*! \brief
    \details
    Tells whether an allocation succeeded
    @param[in]
    ptr             pointer to struct rocblas_device_malloc_base
 ******************************************************************************/
extern "C" bool rocblas_device_malloc_success(rocblas_device_malloc_base* ptr)
{
    using _device_malloc = decltype(rocblas_handle {} -> device_malloc(0));
    return ptr && *static_cast<_device_malloc*>(ptr);
}

/*! \brief
    \details
    Converts rocblas_device_malloc() to a pointer if it only has one pointer.
    Retuns rocblas_status_invalid_pointer if ptr or res is nullptr, there is more than one pointer, or the underyling object is not from rocblas_device_malloc(); rocblas_status_success otherwise
    @param[in]
    ptr             pointer to struct rocblas_device_malloc_base
    @param[out]
    res             pointer to pointer to void
*/
extern "C" rocblas_status rocblas_device_malloc_ptr(rocblas_device_malloc_base* ptr, void** res)
try
{
    using _device_malloc = decltype(rocblas_handle {} -> device_malloc(0));
    if(!ptr || !res)
        return rocblas_status_invalid_pointer;
    *res = static_cast<void*>(*static_cast<_device_malloc*>(ptr));
    return rocblas_status_success;
}
catch(...)
{
    return rocblas_status_invalid_pointer;
}

/*! \brief
    \details
    Gets a pointer to device memory allocated by rocblas_device_malloc().
    Returns rocblas_status_invalid_pointer if ptr or res is nullptr or the underyling object is not from rocblas_device_malloc(); rocblas_status_success otherwise
    @param[in]
    ptr             pointer to struct rocblas_device_malloc_base
    index           index of the pointer to get
    @param[out]
    res             pointer to pointer to void
*/
extern "C" rocblas_status
    rocblas_device_malloc_get(rocblas_device_malloc_base* ptr, size_t index, void** res)
try
{
    using _device_malloc = decltype(rocblas_handle {} -> device_malloc(0));
    if(!ptr || !res)
        return rocblas_status_invalid_pointer;
    *res = (*static_cast<_device_malloc*>(ptr))[index];
    return rocblas_status_success;
}
catch(...)
{
    return rocblas_status_invalid_pointer;
}

/*! \brief
    \details
    Frees memory borrowed from the device memory allocated in handle.
    @param[in]
    ptr             pointer to struct rocblas_device_malloc_base
*/
extern "C" rocblas_status rocblas_device_malloc_free(rocblas_device_malloc_base* ptr)
{
    using _device_malloc = decltype(rocblas_handle {} -> device_malloc(0));
    delete static_cast<_device_malloc*>(ptr);
    return rocblas_status_success;
}

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Return a stream opened for logging.
 *                  If the environment variable with name environment_variable_name
 *                  is set, then it indicates the name of the file to be opened.
 *                  If the environment variable with name environment_variable_name
 *                  is not set, and the environment variable ROCBLAS_LOG_PATH is set,
 *                  then ROCBLAS_LOG_PATH indicates the name of the file to open.
 *                  Otherwise open the stream to stderr.
 *
 *  @param[in]
 *  environment_variable_name   const char*
 *                              Name of environment variable that contains
 *                              the full logfile path.
 */

static auto open_log_stream(const char* environment_variable_name)
{
    const char* logfile;
    logfile = read_env(environment_variable_name);
    if(!logfile)
        logfile = read_env("ROCBLAS_LOG_PATH");
    return logfile ? std::make_unique<rocblas_internal_ostream>(logfile)
                   : std::make_unique<rocblas_internal_ostream>(STDERR_FILENO);
}

/*******************************************************************************
 * Logging initialization
 ******************************************************************************/
void _rocblas_handle::init_logging()
{
    // set layer_mode from value of environment variable ROCBLAS_LAYER
    const char* str_layer_mode = read_env("ROCBLAS_LAYER");
    if(str_layer_mode)
    {
        layer_mode = static_cast<rocblas_layer_mode>(strtol(str_layer_mode, 0, 0));

        // open log_trace file
        if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_internal))
            log_trace_os = open_log_stream("ROCBLAS_LOG_TRACE_PATH");

        // open log_bench file
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench_os = open_log_stream("ROCBLAS_LOG_BENCH_PATH");

        // open log_profile file
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile_os = open_log_stream("ROCBLAS_LOG_PROFILE_PATH");
    }
}

/*******************************************************************************
 * Solution fitness query, for internal testing only
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_solution_fitness_query(rocblas_handle handle, double* fitness)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    handle->solution_fitness_query = fitness;
    if(fitness)
        *fitness = std::numeric_limits<double>::lowest();
    return rocblas_status_success;
}

/*******************************************************************************
 * Choose performance metric used to select solution
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_performance_metric(rocblas_handle             handle,
                                                         rocblas_performance_metric metric)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    handle->performance_metric = metric;
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_get_performance_metric(rocblas_handle              handle,
                                                         rocblas_performance_metric* metric)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(metric)
    {
        *metric = handle->performance_metric;
        return rocblas_status_success;
    }
    else
        return rocblas_status_invalid_pointer;
}
