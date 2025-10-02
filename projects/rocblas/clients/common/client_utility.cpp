/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef WIN32
#include <windows.h>
//
#include <random>
#endif
#include "../../library/src/include/handle.hpp"
#include "client_utility.hpp"
#include "d_vector.hpp"
#include "gtest_helpers.hpp"
#include <chrono>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <stdlib.h>
#include <thread>

#ifdef BUILD_WITH_HIPBLASLT
#include <hipblaslt/hipblaslt-ext.hpp>
#endif

#ifdef WIN32
#define strcasecmp(A, B) _stricmp(A, B)
#define setenv(A, B, C) _putenv_s(A, B)
#define unsetenv(A) _putenv_s(A, "")

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// Not WIN32
#else
#include <fcntl.h>
#include <unistd.h>
#endif

void rocblas_client_init()
{
#ifndef WIN32
    auto* ld_library_path_override = getenv("LD_LIBRARY_PATH");
    if(ld_library_path_override)
    {
        rocblas_cout << "rocBLAS warning: LD_LIBRARY_PATH override may use incompatible rocblas "
                     << std::endl;
    }
#endif

    // Warn users if using older reference library
    print_reference_lib_warning();
}

void rocblas_client_shutdown() {}

void get_version_string(std::string& str)
{
    size_t size;
    rocblas_get_version_string_size(&size);
    str.resize(size - 1, '\0');
    rocblas_get_version_string((char*)str.data(), size);
}

void print_rocblas_version_string()
{
    std::string rocblas_version;
    get_version_string(rocblas_version);

    rocblas_cout << "rocBLAS version: " << rocblas_version << std::endl;

    print_rocblas_client_commit_hashes();
}

#define TOSTR2(s) #s
#define TOSTR(s) TOSTR2(s)

void print_reference_lib_warning()
{
#ifdef ROCBLAS_REFERENCE_LIB
    rocblas_cout << "rocBLAS info: Using reference library '" << TOSTR(ROCBLAS_REFERENCE_LIB) << "'"
                 << std::endl;
#endif
    // prints a warning to cout if the recommended reference library isn't used
#ifdef ROCBLAS_REFERENCE_LIB_WARN
    rocblas_cout << "rocBLAS warning: Reference library may not support 64-bit input arguments. "
                    "If running a test suite, please use "
                 << "--gtest_filter=-*stress* to avoid 64-bit test failures.\n";
#endif
}

// Print rocBLAS and Tensile commit hashes
void print_rocblas_client_commit_hashes()
{
    static const char* rocblas_tensile_commit_hash[] = {ROCBLAS_TENSILE_COMMIT_ID};

    rocblas_cout << "rocBLAS-commit-hash: " << rocblas_tensile_commit_hash[0] << std::endl;
#if BUILD_WITH_TENSILE
    rocblas_cout << "Tensile-commit-hash: " << rocblas_tensile_commit_hash[1] << std::endl;
#else
    rocblas_cout << "Tensile-commit-hash: N/A, as rocBLAS was built without Tensile" << std::endl;
#endif
#ifdef BUILD_WITH_HIPBLASLT
    rocblas_cout << "hipBLASLt version: " << TOSTR(HIPBLASLT_VERSION_MAJOR) << "."
                 << TOSTR(HIPBLASLT_VERSION_MINOR) << "." << TOSTR(HIPBLASLT_VERSION_PATCH)
                 << " commit-hash: " << TOSTR(HIPBLASLT_VERSION_TWEAK) << std::endl;
#else
    rocblas_cout << "hipBLASLt: N/A, as rocBLAS was built without hipBLASLt" << std::endl;
#endif

    size_t size;
    rocblas_get_commit_hash_string_size(&size);

    std::string hash(size - 1, '\0');
    rocblas_get_commit_hash_string((char*)hash.data(), size);

    if(strcmp(rocblas_tensile_commit_hash[0], hash.data()))
    {
        rocblas_cout
            << "rocBLAS warning: client rocblas commit differs from library rocblas commit: "
            << hash << std::endl;
    }
}

#undef TOSTR
#undef TOSTR2

bool rocblas_file_exists(const char* path)
{
#ifdef WIN32
    fs::path file_path(path);
    return fs::exists(file_path);
#else
    return (access(path, F_OK) != -1);
#endif
}

/* ============================================================================================ */
// Return path of this executable
std::string rocblas_exepath()
{
#ifdef WIN32
    // for now not building with wide chars
    // wchar_t wpath[MAX_PATH + 1] = {0};
    // GetModuleFileNameW(NULL, wpath, MAX_PATH + 1);
    // std::vector<wchar_t> result(MAX_PATH + 1);

    std::vector<TCHAR> result(MAX_PATH + 1);
    // Ensure result is large enough to accommodate the path
    DWORD length = 0;
    for(;;)
    {
        length = GetModuleFileNameA(nullptr, result.data(), result.size());
        if(length < result.size() - 1)
        {
            result.resize(length + 1);
            break;
        }
        result.resize(result.size() * 2);
    }

    // std::wstring          wspath(result.data());
    // fs::path exepath(wspath.begin(), wspath.end());

    fs::path exepath(result.begin(), result.end());
    exepath = exepath.remove_filename();
    // Add trailing "/" to exepath if required
    exepath += exepath.empty() ? "" : "/";
    return exepath.string();
#else
    std::string pathstr;
    char*       path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = strrchr(path, '/');
        if(p)
        {
            p[1]    = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
#endif
}

/* ============================================================================================ */
// Temp directory rooted random path
std::string rocblas_tempname()
{
#ifdef WIN32
    // Generate "/tmp/rocblas-XXXXXX" like file name
    const std::string alphanum     = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv";
    int               stringlength = alphanum.length() - 1;
    std::string       uniquestr    = "rocblas-";

    for(auto n : {0, 1, 2, 3, 4, 5})
        uniquestr += alphanum.at(rand() % stringlength);

    fs::path tmpname = fs::temp_directory_path() / uniquestr;

    return tmpname.string();
#else
    char tmp[] = "/tmp/rocblas-XXXXXX";
    int  fd    = mkostemp(tmp, O_CLOEXEC);
    if(fd == -1)
    {
        dprintf(STDERR_FILENO, "Cannot open temporary file: %m\n");
        exit(EXIT_FAILURE);
    }

    return std::string(tmp);
#endif
}

/* ============================================================================================ */
/*  memory allocation requirements :*/

/*! \brief Compute strided batched matrix allocation size allowing for strides smaller than full matrix */
size_t
    strided_batched_matrix_size(int rows, int cols, int lda, rocblas_stride stride, int batch_count)
{
    size_t size = size_t(lda) * cols;
    if(batch_count > 1)
    {
        // for cases where batch_count strides may not exceed full matrix size use full matrix size
        // e.g. row walking a larger matrix we just use full matrix size
        size_t size_strides = (batch_count - 1) * stride;
        size += size < size_strides + (cols - 1) * size_t(lda) + rows ? size_strides : 0;
    }
    return size;
}

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us_sync_device(void)
{
    hipDeviceSynchronize();

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    static bool oldStreamSync = [&] {
        // set old mode if env variable is set
        const char* str_stream_sync
            = getenv("ROCBLAS_BENCH_STREAM_SYNC"); // windows getenv is fine for static init
        if(str_stream_sync)
        {
            return (strncmp(str_stream_sync, "1", 1) == 0 ? true : false);
        }
        return false;
    }();

    if(oldStreamSync)
    {
        THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));

        auto now = std::chrono::steady_clock::now();
        // now.time_since_epoch() is the duration since epoch
        // which is converted to microseconds
        auto duration
            = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
        return (static_cast<double>(duration));
    }
    else
    {
        // avoid any stream synchronization overhead on timing
        static thread_local hipEvent_t start{nullptr};
        static thread_local hipEvent_t stop{nullptr};
        if(!start)
        {
            THROW_IF_HIP_ERROR(hipEventCreate(&start));
            THROW_IF_HIP_ERROR(hipEventCreate(&stop));
            THROW_IF_HIP_ERROR(hipEventRecord(start, stream));

            return 0.0;
        }
        else
        {
            THROW_IF_HIP_ERROR(hipEventRecord(stop, stream));
            THROW_IF_HIP_ERROR(hipEventSynchronize(stop));

            float milliseconds = 0;
            THROW_IF_HIP_ERROR(hipEventElapsedTime(&milliseconds, start, stop));
            THROW_IF_HIP_ERROR(hipEventDestroy(stop));
            THROW_IF_HIP_ERROR(hipEventDestroy(start));
            start = nullptr; // reset start event for next use
            stop  = nullptr;

            return milliseconds * 1000.0; // convert to microseconds
        }
    }
};

/*! \brief  CPU Timer(in microsecond): no GPU synchronization */
double get_time_us_no_sync(void)
{
    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
rocblas_int query_device_property()
{
    int            device_count;
    rocblas_status status = (rocblas_status)hipGetDeviceCount(&device_count);
    if(status != rocblas_status_success)
    {
        rocblas_cerr << "Query device error: cannot get device count" << std::endl;
        return -1;
    }
    else
    {
        rocblas_cout << "Query device success: there are " << device_count << " devices"
                     << std::endl;
    }

    for(rocblas_int i = 0;; i++)
    {
        rocblas_cout
            << "-------------------------------------------------------------------------------"
            << std::endl;

        if(i >= device_count)
            break;

        hipDeviceProp_t props;
        rocblas_status  status = (rocblas_status)hipGetDeviceProperties(&props, i);
        if(status != rocblas_status_success)
        {
            rocblas_cerr << "Query device error: cannot get device ID " << i << "'s property"
                         << std::endl;
        }
        else
        {
            char buf[320];
            snprintf(
                buf,
                sizeof(buf),
                "Device ID %d : %s %s\n"
                "with %3.1f GB memory, max. SCLK %d MHz, max. MCLK %d MHz, memoryBusWidth %d "
                "Bytes, compute capability "
                "%d.%d\n"
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                i,
                props.name,
                props.gcnArchName,
                props.totalGlobalMem / 1e9,
                (int)(props.clockRate / 1000),
                (int)(props.memoryClockRate / 1000),
                (int)(props.memoryBusWidth / 8),
                props.major,
                props.minor,
                props.maxGridSize[0],
                props.sharedMemPerBlock / 1e3,
                props.maxThreadsPerBlock,
                props.warpSize);
            rocblas_cout << buf;
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(rocblas_int device_id)
{
    rocblas_status status = (rocblas_status)hipSetDevice(device_id);
    if(status != rocblas_status_success)
    {
        rocblas_cerr << "Set device error: cannot set device ID " << device_id
                     << ", there may not be such device ID" << std::endl;
    }
}

/*********************************************
 * callback function
 *********************************************/
thread_local std::unique_ptr<std::function<void(rocblas_handle)>> t_set_stream_callback;

/*****************
 * local handles *
 *****************/

rocblas_local_handle::rocblas_local_handle()
{
    auto status = rocblas_create_handle(&m_handle);
    if(status != rocblas_status_success)
        throw std::runtime_error(rocblas_status_to_string(status));

#ifdef GOOGLE_TEST
    if(t_set_stream_callback)
    {
        (*t_set_stream_callback)(m_handle);
        t_set_stream_callback.reset();
    }
#endif
}

rocblas_local_handle::rocblas_local_handle(const Arguments& arg)
{
    if(arg.use_hipblaslt >= 0)
    {
        auto hipblaslt_env = getenv("ROCBLAS_USE_HIPBLASLT");
        if(hipblaslt_env)
            m_hipblaslt_saved_status = std::string(hipblaslt_env);
        m_hipblaslt_env_set = true;
        setenv("ROCBLAS_USE_HIPBLASLT", std::to_string(arg.use_hipblaslt).c_str(), true);
    }

    auto status = rocblas_create_handle(&m_handle);
    if(status != rocblas_status_success)
        throw std::runtime_error(rocblas_status_to_string(status));

#ifdef GOOGLE_TEST
    if(t_set_stream_callback)
    {
        (*t_set_stream_callback)(m_handle);
        t_set_stream_callback.reset();
    }
#endif

    // Set the atomics mode
    status = rocblas_set_atomics_mode(m_handle, arg.atomics_mode);

    // The check_numerics mode conditional defeat with "rocblas_check_numerics_mode_no_check"
    // Defeat check numerics when initializing any data with NaN with due to alpha or beta having NaN flags,
    // or if the arg.initalization is rocblas_initialization::denorm or rocblas_initialization::denorm2
    // The denorm initialization are sometimes used when testing the gemm_ex and gemm_strided_batched_ex functions
    if(rocblas_isnan(arg.alpha) || rocblas_isnan(arg.beta)
       || arg.initialization == rocblas_initialization::denorm
       || arg.initialization == rocblas_initialization::denorm2)
    {
        m_handle->check_numerics = rocblas_check_numerics_mode_no_check;
    }
    else if(m_handle->check_numerics != rocblas_check_numerics_mode_no_check)
    {
        // bypass denorm reporting for f16 gemms, to be analyzed further
        if(arg.a_type == rocblas_datatype_f16_r && m_handle->getArchMajor() == 11
           && strstr(arg.function, "gemm"))
        {
            m_handle->check_numerics = static_cast<rocblas_check_numerics_mode>(
                m_handle->check_numerics | rocblas_check_numerics_mode_only_nan_inf);
        }
    }

    if(status == rocblas_status_success)
    {
        // If the test specifies user allocated workspace, allocate and use it
        if(arg.user_allocated_workspace)
        {
            if((hipMalloc)(&m_memory, arg.user_allocated_workspace) != hipSuccess)
                throw std::bad_alloc();
            status = rocblas_set_workspace(m_handle, m_memory, arg.user_allocated_workspace);
        }
    }

    // memory guard control, with multi-threading should not change values across threads
    d_vector_set_pad_length(arg.pad);

    if(status != rocblas_status_success)
        throw std::runtime_error(rocblas_status_to_string(status));

    status = rocblas_set_math_mode(m_handle, rocblas_math_mode(arg.math_mode));
    if(status != rocblas_status_success)
        throw std::runtime_error(rocblas_status_to_string(status));
}

rocblas_local_handle::~rocblas_local_handle()
{
    if(m_graph_stream)
        rocblas_stream_end_capture();

    if(m_memory)
        (hipFree)(m_memory);

    if(m_hipblaslt_env_set)
    {
        setenv("ROCBLAS_USE_HIPBLASLT", m_hipblaslt_saved_status.c_str(), true);
    }

    rocblas_destroy_handle(m_handle);
}

void rocblas_local_handle::rocblas_stream_begin_capture()
{
    if(m_graph_stream)
        throw std::runtime_error("recursive rocblas_stream_begin_capture");

    CHECK_ROCBLAS_ERROR(rocblas_get_stream(m_handle, &m_old_stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(m_old_stream));

    CHECK_HIP_ERROR(hipStreamCreate(&m_graph_stream));
    CHECK_ROCBLAS_ERROR(rocblas_set_stream(m_handle, m_graph_stream));

    // BEGIN GRAPH CAPTURE
    CHECK_HIP_ERROR(hipStreamBeginCapture(m_graph_stream, hipStreamCaptureModeGlobal));
}

void rocblas_local_handle::rocblas_stream_end_capture()
{
    hipGraph_t     graph;
    hipGraphExec_t instance;

    // END GRAPH CAPTURE
    CHECK_HIP_ERROR(hipStreamEndCapture(m_graph_stream, &graph));
    CHECK_HIP_ERROR(hipGraphInstantiate(&instance, graph, NULL, NULL, 0));

    CHECK_HIP_ERROR(hipGraphDestroy(graph));
    CHECK_HIP_ERROR(hipGraphLaunch(instance, m_graph_stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(m_graph_stream));
    CHECK_HIP_ERROR(hipGraphExecDestroy(instance));

    CHECK_ROCBLAS_ERROR(rocblas_set_stream(m_handle, m_old_stream));
    CHECK_HIP_ERROR(hipStreamDestroy(m_graph_stream));
    m_graph_stream = nullptr;
}

void rocblas_parallel_initialize_thread(int id, size_t& memory_used)
{
    size_t before_init, after_init, total_memory;
    CHECK_HIP_ERROR(hipSetDevice(id));
    CHECK_HIP_ERROR(hipMemGetInfo(&before_init, &total_memory));
    rocblas_initialize();
    CHECK_HIP_ERROR(hipMemGetInfo(&after_init, &total_memory));
    memory_used = before_init - after_init;
}

/*!
 * Initialize rocBLAS for the requested number of  HIP devices
 * and report the time taken to complete the initialization.
 * This is to avoid costly startup time at the first call on
 * that device. Internal use for benchmark & testing.
 * Initializes devices indexed from 0 to parallel_devices-1.
 * If parallel_devices is 1, hipSetDevice should be called
 * before calling this function.
 */
void rocblas_parallel_initialize(int parallel_devices)
{
    auto                thread = std::make_unique<std::thread[]>(parallel_devices);
    std::vector<size_t> init_memory(parallel_devices);

    // Store the start timepoint of rocblas initialize
    auto start_time = std::chrono::steady_clock::now();

    if(parallel_devices == 1)
    {
        size_t before_init, after_init, total_memory;
        CHECK_HIP_ERROR(hipMemGetInfo(&before_init, &total_memory));
        rocblas_initialize();
        CHECK_HIP_ERROR(hipMemGetInfo(&after_init, &total_memory));
        init_memory[0] = before_init - after_init;
    }
    else
    {

        for(int id = 0; id < parallel_devices; ++id)
            thread[id]
                = std::thread(rocblas_parallel_initialize_thread, id, std::ref(init_memory[id]));
        for(int id = 0; id < parallel_devices; ++id)
            thread[id].join();
    }

    // Store the end timepoint of rocblas initialize
    auto end_time = std::chrono::steady_clock::now();

    // Compute the time taken to load the Tensile kernels (in milliseconds).
    auto init_time_in_ms
        = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    rocblas_cout << "\nrocBLAS info: Time taken to complete rocBLAS library initialization is "
                 << init_time_in_ms << " milliseconds." << std::endl;

    // Calculate average initialization time per GPU
    auto avg_init_time_in_ms = init_time_in_ms / parallel_devices;
    if(parallel_devices > 1)
    {
        rocblas_cout
            << "\nrocBLAS info: Average time taken to complete rocBLAS library initialization "
               "per device is "
            << avg_init_time_in_ms << " milliseconds." << std::endl;
    }

    // If average initialization time exceeds the max duration, display the following info message.
    constexpr static int max_duration = 5000;
    if(avg_init_time_in_ms > max_duration)
        rocblas_cerr << "\nrocBLAS info: average time to initialize each device exceeded the max "
                        "duration of "
                     << max_duration << " milliseconds. Check CPU's load metrics." << std::endl;

    constexpr static float max_memory = 1.0;
    auto                   max_library_size
        = *std::max_element(std::begin(init_memory), std::end(init_memory)) * 1.0e-9;

    rocblas_cout << "\nrocBLAS info: maximum library size per device is " << max_library_size
                 << " GB." << std::endl;
    if(max_library_size > max_memory)
        rocblas_cerr << "\nrocBLAS info: max kernel library size " << max_library_size
                     << " GB exceeds the max recommended memory " << max_memory
                     << " GB. Check library logic file sizes." << std::endl;
}

size_t calculate_flush_batch_count(size_t arg_flush_batch_count,
                                   size_t arg_flush_memory_size,
                                   size_t cached_size)
{
    size_t default_arg_flush_batch_count = 1;
    size_t default_arg_flush_memory_size = 0;
    size_t flush_batch_count             = default_arg_flush_batch_count;

    if(arg_flush_batch_count != default_arg_flush_batch_count
       && arg_flush_memory_size != default_arg_flush_memory_size)
    {
        rocblas_cout << "rocBLAS WARNING: cannot set both flush_batch_count and flush_memory_size"
                     << std::endl;
        rocblas_cout << "rocBLAS WARNING: using flush_batch_count = " << arg_flush_batch_count
                     << std::endl;
        flush_batch_count = arg_flush_batch_count;
    }
    else if(arg_flush_batch_count != default_arg_flush_batch_count)
    {
        flush_batch_count = arg_flush_batch_count;
        rocblas_cout << "flush_memory_size = ";
        print_memory_size(flush_batch_count * cached_size);
        rocblas_cout << std::endl;
    }
    else if(arg_flush_memory_size != default_arg_flush_memory_size)
    {
        flush_batch_count = 1 + (arg_flush_memory_size - 1) / cached_size;
        rocblas_cout << "flush_batch_count = " << flush_batch_count << std::endl;
    }
    return flush_batch_count;
}

//Function to limit the number of devices to be used in a mult-gpu setup
hipError_t limit_device_count(int& device_count, int max_limit)
{
    hipError_t hipStatus = hipGetDeviceCount(&device_count);
    if(hipStatus == hipSuccess)
        device_count = std::min(device_count, max_limit);
    return hipStatus;
}
