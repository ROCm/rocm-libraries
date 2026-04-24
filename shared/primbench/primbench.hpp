// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#ifdef _WIN32
    // Disables the min() and max() macros from windows.h
    #define NOMINMAX

    #include <io.h>
    #include <windows.h>
#else
    #include <unistd.h>
#endif

#ifdef __HIP__
    #include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
    #include <cuda/std/chrono>
#else
    #error "Unsupported GPU platform"
#endif

#ifdef PRIMBENCH_NO_MONITORING
    #define PRIMBENCH_HAS_MONITORING 0
#else
    #define PRIMBENCH_HAS_MONITORING 1
#endif

#if PRIMBENCH_HAS_MONITORING
    #ifdef __HIP__
        #include <amd_smi/amdsmi.h>
    #elif defined(__CUDACC__)
        #include <nvml.h>
    #endif
#endif

#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <variant>

/// Default GPU cache size used for clearing caches.
///
/// This conservative size is currently used to evict cached data before
/// kernel launches. In the future, introducing HSA as a dependency may
/// allow querying the actual largest GPU cache at runtime.
#ifndef PRIMBENCH_GPU_CACHE_SIZE
    #define PRIMBENCH_GPU_CACHE_SIZE 256 * primbench::MiB
#endif

/// Registers a custom type name, used by `primbench::name<T>()`.
#define PRIMBENCH_REGISTER_TYPE(TYPE, NAME)    \
    namespace primbench::detail                \
    {                                          \
    template<>                                 \
    struct type_name<TYPE>                     \
    {                                          \
        static inline const char* name = NAME; \
    };                                         \
    }

#ifdef __HIP__
    /// Exits the program with an error message if the given HIP API call returns a failure status.
    #define PRIMBENCH_CHECK(condition)                                             \
        do                                                                         \
        {                                                                          \
            hipError_t status = condition;                                         \
            if(status != hipSuccess)                                               \
            {                                                                      \
                std::cerr << __FILE__ << ":" << __LINE__                           \
                          << ": HIP error: " << hipGetErrorString(status) << "\n"; \
                exit(status);                                                      \
            }                                                                      \
        }                                                                          \
        while(0)
#else
    /// Exits the program with an error message if the given CUDA API call returns a failure status.
    #define PRIMBENCH_CHECK(condition)                                               \
        do                                                                           \
        {                                                                            \
            cudaError_t status = condition;                                          \
            if(status != cudaSuccess)                                                \
            {                                                                        \
                std::cerr << __FILE__ << ":" << __LINE__                             \
                          << ": CUDA error: " << cudaGetErrorString(status) << "\n"; \
                exit(status);                                                        \
            }                                                                        \
        }                                                                            \
        while(0)
#endif

#if defined(__HIP__) && PRIMBENCH_HAS_MONITORING
    /// Exits the program with an error message if the given AMD SMI API call returns a failure status.
    #define PRIMBENCH_AMDSMI_CHECK(condition)                                                     \
        do                                                                                        \
        {                                                                                         \
            amdsmi_status_t status = condition;                                                   \
            if(status != AMDSMI_STATUS_SUCCESS)                                                   \
            {                                                                                     \
                const char* errstr = "(unknown)";                                                 \
                amdsmi_status_code_to_string(status, &errstr);                                    \
                std::cerr << __FILE__ << ":" << __LINE__ << ": AMDSMI error: " << errstr << "\n"; \
                exit(status);                                                                     \
            }                                                                                     \
        }                                                                                         \
        while(0)
#elif defined(__CUDACC__) && PRIMBENCH_HAS_MONITORING
    /// Exits the program with an error message if the given NVML API call returns a failure status.
    #define PRIMBENCH_NVML_CHECK(condition)                                       \
        do                                                                        \
        {                                                                         \
            nvmlReturn_t status = condition;                                      \
            if(status != NVML_SUCCESS)                                            \
            {                                                                     \
                std::cerr << __FILE__ << ":" << __LINE__                          \
                          << ": NVML error: " << nvmlErrorString(status) << "\n"; \
                exit(status);                                                     \
            }                                                                     \
        }                                                                         \
        while(0)
#endif

namespace primbench
{

inline constexpr size_t KiB = 1024;
inline constexpr size_t MiB = 1024 * KiB;
inline constexpr size_t GiB = 1024 * MiB;

// Backend-agnostic stream type and default stream constant.
#ifdef __HIP__
using stream_t                    = hipStream_t;
constexpr stream_t default_stream = hipStreamDefault;
#elif defined(__CUDACC__)
using stream_t                    = cudaStream_t;
constexpr stream_t default_stream = cudaStreamDefault;
#endif

// Forward declarations for the detail namespace.
extern const size_t MiB;
template<typename... Args>
void log(Args&&... args);

/// Settings that benchmarks and users can pass.
struct settings
{
    size_t      size                 = 128 * primbench::MiB; ///< Input array size.
    bool        hot                  = false; ///< Hot means not clearing GPU cache between batches.
    uint32_t    seed                 = 42; ///< The seed to use for input array generation.
    std::string json_out             = "results.json"; ///< Output JSON file path.
    std::string csv_out              = ""; ///< Output CSV file path.
    std::string filter               = ""; ///< Regex filter of specialization names to benchmark.
    bool        dry                  = false; ///< Flag to perform a dry run.
    double      min_gpu_ms_per_batch = 10.0; ///< Minimum GPU batch duration.
    double      min_secs             = 1.0; ///< Minimum benchmark duration.
    double      noise_timeout_secs   = 10.0; ///< Max duration before noisy benchmark times out.
    size_t      batch_window_size    = 10; ///< Noise window size for early stopping.
    double      noise_tolerance_percent = 1.0; ///< Noise tolerance for early stopping.
    uint16_t    min_gpu_temp            = 50; ///< Minimum GPU temperature.
    uint16_t    max_gpu_temp            = 60; ///< Maximum GPU temperature.
    double      max_warming_secs        = 60.0; ///< Max GPU warmup time.
    double      max_cooling_secs        = 60.0; ///< Max GPU cooldown time.
    bool        output_batches          = false; ///< Flag to output batch details.
    uint32_t    spaces_per_indent       = 4; ///< JSON indentation spaces.
    double stream_blocking_timeout_secs = 10.0; ///< Max duration before stream blocking times out.

    using custom_arg_value = std::variant<std::string, bool, double, int, unsigned int, size_t>;
    std::map<std::string, custom_arg_value>
        custom_args; ///< Custom user-registered arguments with types.

}; // struct settings

namespace detail
{

/// Fallback for primbench::name().
template<class T>
struct type_name
{
    static_assert(sizeof(T) == 0, "Call PRIMBENCH_REGISTER_TYPE() for this type");

    static inline const char* name = "";
};

#ifdef _WIN32
/// Required to print horizontal_bar (u8"─").
inline const auto init_console_utf8 = []
{
    SetConsoleOutputCP(CP_UTF8);
    return 0;
}();
#endif

/// Returns whether the environment variable is set.
inline bool get_env(std::string_view key)
{
#ifdef _WIN32
    char*  value = nullptr;
    size_t len   = 0;

    if(_dupenv_s(&value, &len, key.data()) != 0 || value == nullptr)
        return false;

    free(value);
    return true;
#else
    return std::getenv(key.data());
#endif
}

/// Caches whether ANSI color output is enabled.
/// The standard is described at https://bixense.com/clicolors/
inline bool use_color()
{
    static const bool result = []
    {
        if(get_env("NO_COLOR"))
            return false;
        if(get_env("CLICOLOR_FORCE"))
            return true;

#ifdef _WIN32
        return bool(_isatty(_fileno(stdout)));
#else
        return bool(isatty(fileno(stdout)));
#endif
    }();
    return result;
}

/// Clears the current line if colors are enabled.
inline std::ostream& clearline(std::ostream& os)
{
    if(use_color())
        os << "\r\033[K";
    return os;
}

/// Resets the output text formatting to default if colors are enabled.
inline std::ostream& reset(std::ostream& os)
{
    if(use_color())
        os << "\033[0m";
    return os;
}

/// Sets the output text color to gray if colors are enabled.
inline std::ostream& gray(std::ostream& os)
{
    if(use_color())
        os << "\033[90m";
    return os;
}

/// Sets the output text color to green if colors are enabled.
inline std::ostream& green(std::ostream& os)
{
    if(use_color())
        os << "\033[92m";
    return os;
}

/// Sets the output text color to red if colors are enabled.
inline std::ostream& red(std::ostream& os)
{
    if(use_color())
        os << "\033[91m";
    return os;
}

/// Sets the output text color to yellow if colors are enabled.
inline std::ostream& yellow(std::ostream& os)
{
    if(use_color())
        os << "\033[93m";
    return os;
}

/// Sets the output text color to blue if colors are enabled.
inline std::ostream& blue(std::ostream& os)
{
    if(use_color())
        os << "\033[94m";
    return os;
}

/// Backend-agnostic wrappers for HIP and CUDA.
/// The functions in this namespace are sorted alphabetically.
namespace gpu_backend
{

#ifdef __HIP__
using error_t                           = hipError_t;
using event_t                           = hipEvent_t;
constexpr unsigned host_register_mapped = hipHostRegisterMapped;
#elif defined(__CUDACC__)
using error_t                           = cudaError_t;
using event_t                           = cudaEvent_t;
constexpr unsigned host_register_mapped = cudaHostRegisterMapped;
#endif

#ifdef __HIP__
    #define PRIMBENCH_PREFIX_BACKEND(fn) hip##fn
#elif defined(__CUDACC__)
    #define PRIMBENCH_PREFIX_BACKEND(fn) cuda##fn
#endif

inline error_t event_create(event_t* event)
{
    return PRIMBENCH_PREFIX_BACKEND(EventCreate)(event);
}

inline error_t event_destroy(event_t event)
{
    return PRIMBENCH_PREFIX_BACKEND(EventDestroy)(event);
}

inline error_t event_elapsed_time(float* ms, event_t start, event_t stop)
{
    return PRIMBENCH_PREFIX_BACKEND(EventElapsedTime)(ms, start, stop);
}

inline error_t event_record(event_t event, stream_t stream)
{
    return PRIMBENCH_PREFIX_BACKEND(EventRecord)(event, stream);
}

inline error_t get_last_error()
{
    return PRIMBENCH_PREFIX_BACKEND(GetLastError)();
}

inline error_t gpu_free(void* device)
{
    return PRIMBENCH_PREFIX_BACKEND(Free)(device);
}

inline error_t gpu_malloc(void** device, size_t size)
{
    return PRIMBENCH_PREFIX_BACKEND(Malloc)(device, size);
}

inline error_t host_get_device_pointer(void** device, void* host, unsigned flags)
{
    return PRIMBENCH_PREFIX_BACKEND(HostGetDevicePointer)(device, host, flags);
}

inline error_t host_register(void* ptr, size_t size, unsigned flags)
{
    return PRIMBENCH_PREFIX_BACKEND(HostRegister)(ptr, size, flags);
}

inline error_t host_unregister(void* ptr)
{
    return PRIMBENCH_PREFIX_BACKEND(HostUnregister)(ptr);
}

inline error_t memset_async(void* device, int value, size_t count, stream_t stream)
{
    return PRIMBENCH_PREFIX_BACKEND(MemsetAsync)(device, value, count, stream);
}

template<typename F>
inline error_t occupancy_max_potential_block_size(int*   min_grid_size,
                                                  int*   block_size,
                                                  F      kernel,
                                                  size_t dynamic_shared_memory_per_block,
                                                  int    block_size_limit)
{
    return PRIMBENCH_PREFIX_BACKEND(OccupancyMaxPotentialBlockSize)(min_grid_size,
                                                                    block_size,
                                                                    kernel,
                                                                    dynamic_shared_memory_per_block,
                                                                    block_size_limit);
}

inline error_t stream_create(stream_t* stream)
{
    return PRIMBENCH_PREFIX_BACKEND(StreamCreate)(stream);
}

inline error_t stream_destroy(stream_t stream)
{
    return PRIMBENCH_PREFIX_BACKEND(StreamDestroy)(stream);
}

inline error_t stream_synchronize(stream_t stream)
{
    return PRIMBENCH_PREFIX_BACKEND(StreamSynchronize)(stream);
}

#undef PRIMBENCH_PREFIX_BACKEND

} // namespace gpu_backend

/// Bring GPU backend wrappers into this namespace so we can call
/// functions like gpu_malloc() without the gpu_backend:: prefix.
using namespace gpu_backend;

#ifdef __HIP__
/// Serializes information about the used GPU.
std::string serialize_gpu_info()
{
    std::ostringstream ss;
    ss << "{";

    int hip_device;
    PRIMBENCH_CHECK(hipGetDevice(&hip_device));

    hipDeviceProp_t dev_prop;
    PRIMBENCH_CHECK(hipGetDeviceProperties(&dev_prop, hip_device));
    ss << "\"name\":\"" << dev_prop.name << "\"";

    std::string arch = dev_prop.gcnArchName;
    size_t      pos  = arch.find(':');
    if(pos != std::string::npos)
    {
        arch = arch.substr(0, pos);
    }
    ss << ",\"arch\":\"" << arch << "\"";

    char pci_bus_id_arr[32];
    PRIMBENCH_CHECK(hipDeviceGetPCIBusId(pci_bus_id_arr, sizeof(pci_bus_id_arr), hip_device));
    ss << ",\"pci_bus_id\":\"" << pci_bus_id_arr << "\"";

    ss << "}";
    return ss.str();
}
#elif defined(__CUDACC__)
/// Serializes information about the used GPU.
std::string serialize_gpu_info()
{
    std::ostringstream ss;
    ss << "{";

    int cuda_device;
    PRIMBENCH_CHECK(cudaGetDevice(&cuda_device));

    cudaDeviceProp dev_prop;
    PRIMBENCH_CHECK(cudaGetDeviceProperties(&dev_prop, cuda_device));
    ss << "\"name\":\"" << dev_prop.name << "\"";

    ss << ",\"arch\":\"" << dev_prop.major << "." << dev_prop.minor << "\"";

    char pci_bus_id_arr[32];
    PRIMBENCH_CHECK(cudaDeviceGetPCIBusId(pci_bus_id_arr, sizeof(pci_bus_id_arr), cuda_device));
    ss << ",\"pci_bus_id\":\"" << pci_bus_id_arr << "\"";

    ss << "}";
    return ss.str();
}
#endif

#if defined(__HIP__) && PRIMBENCH_HAS_MONITORING

/// Wrapper for AMD SMI (System Management Interface) GPU monitoring.
class monitor
{
public:
    // Delete all copy/move constructors and assignment operators.
    monitor(const monitor&)            = delete;
    monitor& operator=(const monitor&) = delete;
    monitor(monitor&&)                 = delete;
    monitor& operator=(monitor&&)      = delete;

    /// Singleton accessor.
    static monitor& instance()
    {
        static monitor instance;
        return instance;
    }

    /// Saves the current temperature.
    void save_start_temp()
    {
        m_saved_start_temp = get_temp();
    }

    /// Gets the saved starting temperature.
    /// Specializations output this to JSON as `gpu_temp_celcius.start`.
    uint16_t get_saved_start_temp() const
    {
        return m_saved_start_temp;
    }

    /// Returns the current temperature.
    uint16_t get_temp() const
    {
        int64_t t;
        PRIMBENCH_AMDSMI_CHECK(amdsmi_get_temp_metric(m_amdsmi_device,
                                                      get_temperature_type(),
                                                      AMDSMI_TEMP_CURRENT,
                                                      &t));
        return t;
    }

    /// Serializes information about the monitoring library.
    std::string serialize_monitoring() const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"name\":\"amdsmi\"";

        amdsmi_version_t amdsmi_version;
        PRIMBENCH_AMDSMI_CHECK(amdsmi_get_lib_version(&amdsmi_version));
        ss << ",\"version\":\"" << amdsmi_version.build << "\"";

        ss << "}";
        return ss.str();
    }

    /// Returns either "edge" or "hotspot".
    std::string get_used_temperature_type_name() const
    {
        return get_temperature_type_name(get_temperature_type());
    }

private:
    monitor()
    {
        PRIMBENCH_AMDSMI_CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS));

        // This can't be run in a member initializer list,
        // since the above amdsmi_init() call must be run first.
        m_amdsmi_device = get_amdsmi_device();
    }

    ~monitor()
    {
        PRIMBENCH_AMDSMI_CHECK(amdsmi_shut_down());
    }

    /// Determines the first available GPU temperature sensor type.
    ///
    /// Attempts to read temperature from multiple sensor types in priority order.
    /// Returns the first type that successfully provides a temperature reading.
    ///
    /// The result is cached in a static variable.
    amdsmi_temperature_type_t get_temperature_type() const
    {
        static const amdsmi_temperature_type_t temperature_type = [&]
        {
            const amdsmi_temperature_type_t types[] = {
                AMDSMI_TEMPERATURE_TYPE_EDGE,
                AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
            };

            int64_t t;
            for(auto type : types)
            {
                if(amdsmi_get_temp_metric(m_amdsmi_device, type, AMDSMI_TEMP_CURRENT, &t)
                   == AMDSMI_STATUS_SUCCESS)
                {
                    return type;
                }
            }

            std::cerr << "\nError: Failed to get any of these GPU temperatures: ";
            bool first = true;
            for(auto type : types)
            {
                if(!first)
                    std::cerr << ", ";
                first = false;
                std::cerr << get_temperature_type_name(type);
            }
            std::cerr << "\n";
            exit(EXIT_FAILURE);
        }();

        return temperature_type;
    }

    /// Converts a temperature type enum to a string.
    static const char* get_temperature_type_name(amdsmi_temperature_type_t type)
    {
        switch(type)
        {
            case AMDSMI_TEMPERATURE_TYPE_EDGE: return "edge";
            case AMDSMI_TEMPERATURE_TYPE_HOTSPOT: return "hotspot";
            default:
                std::cerr << "\nError: Failed to match temperature type " << type
                          << " to a string\n";
                exit(EXIT_FAILURE);
        }
    }

    /// Finds the AMD SMI processor handle matching the current HIP device.
    amdsmi_processor_handle get_amdsmi_device() const
    {
        int hip_device;
        PRIMBENCH_CHECK(hipGetDevice(&hip_device));

        hipDeviceProp_t hip_props;
        PRIMBENCH_CHECK(hipGetDeviceProperties(&hip_props, hip_device));

        // Build the AMD SMI BDF struct from HIP device properties.
        amdsmi_bdf_t addr{
            .function_number = 0, // HIP doesn't expose PCI function ID.
            .device_number   = static_cast<uint8_t>(hip_props.pciDeviceID),
            .bus_number      = static_cast<uint8_t>(hip_props.pciBusID),
            .domain_number   = static_cast<uint16_t>(hip_props.pciDomainID),
        };

        amdsmi_processor_handle amdsmi_device;
        PRIMBENCH_AMDSMI_CHECK(amdsmi_get_processor_handle_from_bdf(addr, &amdsmi_device));

        return amdsmi_device;
    }

    /// Saved starting temperature for a specialization in state.run().
    uint16_t m_saved_start_temp;

    amdsmi_processor_handle m_amdsmi_device; ///< AMD SMI device.
}; // class monitor

#elif defined(__CUDACC__) && PRIMBENCH_HAS_MONITORING

/// Wrapper for NVML (NVIDIA Management Library) GPU monitoring.
class monitor
{
public:
    // Delete all copy/move constructors and assignment operators.
    monitor(const monitor&)            = delete;
    monitor& operator=(const monitor&) = delete;
    monitor(monitor&&)                 = delete;
    monitor& operator=(monitor&&)      = delete;

    /// Singleton accessor.
    static monitor& instance()
    {
        static monitor instance;
        return instance;
    }

    /// Saves the current temperature.
    void save_start_temp()
    {
        m_saved_start_temp = get_temp();
    }

    /// Gets the saved starting temperature.
    /// Specializations output this to JSON as `gpu_temp_celcius.start`.
    uint16_t get_saved_start_temp() const
    {
        return m_saved_start_temp;
    }

    /// Returns the current temperature.
    uint16_t get_temp() const
    {
        unsigned int t;
        PRIMBENCH_NVML_CHECK(nvmlDeviceGetTemperature(m_nvml_device, NVML_TEMPERATURE_GPU, &t));
        return t;
    }

    /// Serializes information about the monitoring library.
    std::string serialize_monitoring() const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"name\":\"nvml\"";

        char nvml_ver[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE];
        PRIMBENCH_NVML_CHECK(nvmlSystemGetNVMLVersion(nvml_ver, sizeof(nvml_ver)));
        ss << ",\"version\":\"" << nvml_ver << "\"";

        ss << "}";
        return ss.str();
    }

    /// NVML only exposes a single temperature sensor.
    std::string get_used_temperature_type_name() const
    {
        return "gpu";
    }

private:
    monitor()
    {
        PRIMBENCH_NVML_CHECK(nvmlInit());

        PRIMBENCH_NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &m_nvml_device));
    }

    ~monitor()
    {
        PRIMBENCH_NVML_CHECK(nvmlShutdown());
    }

    /// Saved starting temperature for a specialization in state.run().
    uint16_t m_saved_start_temp;

    nvmlDevice_t m_nvml_device; ///< NVML device.
}; // class monitor

#endif

/// Namespace for flag definitions and utilities.
namespace flags
{

/// Enum representing different flags.
enum class Flags : uint32_t
{
    none = 0x0, ///< No flags set.
    sync = 0x1, ///< Synchronization flag.
};

/// Wrapper for Flags with utility operations.
struct FlagTag
{
    Flags value{Flags::none}; ///< Underlying flag value.

    /// Construct from a specific flag.
    constexpr FlagTag(Flags v) : value(v) {}

    /// Bitwise OR operator for combining flags.
    friend constexpr FlagTag operator|(FlagTag a, FlagTag b)
    {
        return FlagTag(
            static_cast<Flags>(static_cast<uint32_t>(a.value) | static_cast<uint32_t>(b.value)));
    }

    /// Check if a flag is set.
    constexpr bool has(FlagTag f) const
    {
        return (static_cast<uint32_t>(value) & static_cast<uint32_t>(f.value)) != 0;
    }
};

} // namespace flags

/// Logger for saving benchmark results in JSON format.
///
/// Because the logger writes partial JSON data incrementally, a JSON library is not used.
class logger
{
public:
    // Delete all copy/move constructors and assignment operators.
    logger(const logger&)            = delete;
    logger& operator=(const logger&) = delete;
    logger(logger&&)                 = delete;
    logger& operator=(logger&&)      = delete;

    /// Singleton accessor.
    static logger& instance()
    {
        static logger instance;
        return instance;
    }

    /// Saves the program start time.
    void save_program_start_time()
    {
        m_program_start_time = std::chrono::steady_clock::now();
    }

    /// Returns the program start time.
    std::chrono::time_point<std::chrono::steady_clock> get_program_start_time() const
    {
        return m_program_start_time;
    }

    /// Initializes the logger, and opens the output JSON and CSV files.
    void init(std::string_view algorithm,
              size_t           specialization_count,
              const settings&  settings,
              flags::FlagTag   flags)
    {
        m_output_batches    = settings.output_batches;
        m_spaces_per_indent = settings.spaces_per_indent;
        m_outputting_csv    = !settings.csv_out.empty();

        m_json_out.open(std::string(settings.json_out), std::ios::out | std::ios::trunc);
        if(!m_json_out)
        {
            std::cerr << "Error: Failed to open " << settings.json_out << " for writing\n";
            exit(EXIT_FAILURE);
        }

        if(m_outputting_csv)
        {
            m_csv_out.open(std::string(settings.csv_out), std::ios::out | std::ios::trunc);
            if(!m_csv_out)
            {
                std::cerr << "Error: Failed to open " << settings.csv_out << " for writing\n";
                exit(EXIT_FAILURE);
            }
        }

        m_json_out << indent(
            serialize_json_prologue(algorithm, specialization_count, settings, flags),
            0);
        m_json_out << "[";
        if(m_spaces_per_indent > 0)
            m_json_out << "\n";
        m_json_out.flush();

        if(m_outputting_csv)
        {
            // Output the CSV header.
            m_csv_out << "index,name,bytes_per_second,gib_per_second,items_per_second"
                         ",noise_timeout,noise_percent\n";
            m_csv_out.flush();
        }

        m_first_specialization = true;
    }

    /// Stores a batch of benchmark results.
    void save(double batch_ms, const std::vector<float>& iterations_ms)
    {
        struct batch batch
        {};

        batch.batch_ms      = batch_ms;
        batch.iterations_ms = iterations_ms;

        m_batches.push_back(batch);
    }

    /// Outputs JSON and CSV specialization information.
    void output_specialization(size_t           index,
                               std::string_view name,
                               std::string_view serialized_meta,
                               size_t           kernels_per_batch,
                               double           ms_per_batch,
                               double           bytes_per_sec,
                               double           items_per_sec,
                               size_t           bytes_per_item,
                               size_t           items,
                               double           noise_percent,
                               double           elapsed_host_secs,
                               double           elapsed_gpu_secs,
                               bool             noise_timeout)
    {
        output_json_specialization(index,
                                   name,
                                   serialized_meta,
                                   kernels_per_batch,
                                   ms_per_batch,
                                   bytes_per_sec,
                                   items_per_sec,
                                   bytes_per_item,
                                   items,
                                   noise_percent,
                                   elapsed_host_secs,
                                   elapsed_gpu_secs,
                                   noise_timeout);

        if(m_outputting_csv)
        {
            output_csv_specialization(index,
                                      name,
                                      bytes_per_sec,
                                      items_per_sec,
                                      noise_timeout,
                                      noise_percent);
        }

        m_total_elapsed_gpu_secs += elapsed_gpu_secs;

        if(noise_timeout)
            m_noise_timeouts++;

        m_batches.clear();
    }

    /// Outputs a summary object at the end of the JSON file.
    void output_summary()
    {
        // Close the JSON file's outer array.
        if(m_spaces_per_indent > 0)
        {
            m_json_out << "\n";
            m_json_out << std::string(m_spaces_per_indent, ' ');
        }
        m_json_out << "],";

        if(m_spaces_per_indent > 0)
            m_json_out << "\n";
        m_json_out << indent(serialize_summary(), 1);

        // Close the JSON file's outer object.
        if(m_spaces_per_indent > 0)
            m_json_out << "\n";
        m_json_out << "}";
        if(m_spaces_per_indent > 0)
            m_json_out << "\n";

        m_json_out.close();

        if(m_outputting_csv)
        {
            m_csv_out.close();
        }
    }

private:
    logger() = default;

    struct batch
    {
        double             batch_ms; ///< Total time for the batch.
        std::vector<float> iterations_ms; ///< Time per iteration.
    };

    /// Serializes the start of the JSON file.
    ///
    /// It adds the `context` object, and adds the `specializations` key.
    std::string serialize_json_prologue(std::string_view algorithm,
                                        size_t           specialization_count,
                                        const settings&  settings,
                                        flags::FlagTag   flags)
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"context\":" << serialize_context(algorithm, specialization_count, settings, flags);

        ss << ",";
        if(m_spaces_per_indent > 0)
            ss << "\n";

        ss << "\"specializations\":";

        return ss.str();
    }

    /// Serializes the benchmark context into JSON.
    std::string serialize_context(std::string_view algorithm,
                                  size_t           specialization_count,
                                  const settings&  settings,
                                  flags::FlagTag   flags) const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"results_version\":\"4.0.0\"";
        ss << ",\"general\":" << serialize_general(algorithm, specialization_count);
        ss << ",\"settings\":" << serialize_settings(settings);

        std::string custom_cli = serialize_custom_settings(settings);
        if(!custom_cli.empty())
        {
            ss << ",\"custom_settings\":" << custom_cli;
        }

        ss << ",\"flags\":" << serialize_flags(flags);

        ss << "}";
        return ss.str();
    }

    /// Serializes general benchmark context info into JSON.
    std::string serialize_general(std::string_view algorithm, size_t specialization_count) const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"algorithm\":\"" << algorithm << "\"";
        ss << ",\"specialization_count\":" << specialization_count;

#ifdef NDEBUG
        const char build_type[] = "release";
#else
        const char build_type[] = "debug";
#endif
        ss << ",\"library_build_type\":\"" << build_type << "\"";

        ss << ",\"gpu\":" << serialize_gpu_info();

        ss << ",\"backend\":" << serialize_backend();

#if PRIMBENCH_HAS_MONITORING
        ss << ",\"monitoring\":" << monitor::instance().serialize_monitoring();

        ss << ",\"temperature_type\":\"" << monitor::instance().get_used_temperature_type_name()
           << "\"";
#endif

        ss << ",\"host_name\":\"" << get_host_name() << "\"";

        ss << ",\"date\":\"" << date() << "\"";

#ifdef BRANCH_NAME
        ss << ",\"branch_name\":\"" << BRANCH_NAME << "\"";
#endif
#ifdef COMMIT_HASH
        ss << ",\"commit_hash\":\"" << COMMIT_HASH << "\"";
#endif

        ss << "}";
        return ss.str();
    }

#ifdef __HIP__
    /// Serializes information about the HIP backend.
    std::string serialize_backend() const
    {
        std::ostringstream ss;
        ss << "{";

        // Backend name.
        ss << "\"name\":\"hip\"";

        // HIP version.
        ss << ",\"hip_version\":\"" << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "."
           << HIP_VERSION_PATCH << "-" << HIP_VERSION_GITHASH << "\"";

        // HIP runtime version (integer, e.g., 60443482 -> 6.4.43482).
        int runtime_ver;
        PRIMBENCH_CHECK(hipRuntimeGetVersion(&runtime_ver));
        int major = runtime_ver / 10000000;
        int minor = (runtime_ver / 100000) % 100;
        int patch = runtime_ver % 100000;
        ss << ",\"runtime_version\":\"" << major << "." << minor << "." << patch << "\"";

        // HIP driver version (integer, same format).
        int driver_ver;
        PRIMBENCH_CHECK(hipDriverGetVersion(&driver_ver));
        major = driver_ver / 10000000;
        minor = (driver_ver / 100000) % 100;
        patch = driver_ver % 100000;
        ss << ",\"driver_version\":\"" << major << "." << minor << "." << patch << "\"";

        // Compiler.
        ss << ",\"compiler\":{";
        ss << "\"name\":\"clang\"";
        ss << ",\"version\":\"" << __clang_version__ << "\"";
        ss << "}";

        ss << "}";
        return ss.str();
    }
#elif defined(__CUDACC__)
    /// Serializes information about the CUDA backend.
    std::string serialize_backend() const
    {
        std::ostringstream ss;
        ss << "{";

        // Backend name.
        ss << "\"name\":\"cuda\"";

        // CUDA runtime version.
        int major = CUDART_VERSION / 1000;
        int minor = (CUDART_VERSION % 1000) / 10;
        ss << ",\"runtime_version\":\"" << major << "." << minor << "\"";

        // CUDA driver version (integer, e.g., 9020 -> 9.2).
        int driver_ver;
        PRIMBENCH_CHECK(cudaDriverGetVersion(&driver_ver));
        major = driver_ver / 1000;
        minor = (driver_ver % 1000) / 10;
        ss << ",\"driver_version\":\"" << major << "." << minor << "\"";

        // Compiler.
        ss << ",\"compiler\":{";
        ss << "\"name\":\"nvcc\"";
        ss << ",\"version\":\"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "."
           << __CUDACC_VER_BUILD__ << "\"";
        ss << "}";

        ss << "}";
        return ss.str();
    }
#endif

    static std::string get_host_name()
    {
#ifdef _WIN32
        char  buffer[MAX_COMPUTERNAME_LENGTH + 1] = {0};
        DWORD size                                = sizeof(buffer);
        if(!GetComputerNameA(buffer, &size))
        {
            std::cerr << "Error: GetComputerNameA failed with Win32 Error Code: " << GetLastError()
                      << "\n";
            exit(EXIT_FAILURE);
        }
#else
        char buffer[HOST_NAME_MAX + 1] = {0};
        // POSIX allows truncation without a null terminator.
        if(gethostname(buffer, HOST_NAME_MAX) != 0)
        {
            std::cerr << "Error: gethostname failed: " << std::strerror(errno) << "\n";
            exit(EXIT_FAILURE);
        }
#endif
        return std::string(buffer);
    }

    /// Returns the local date and time as an RFC3339 string: `yyyy-mm-ddTHH:MM:SS±HH:MM`.
    static std::string date()
    {
        using namespace std::chrono;

        // Get current time as system_clock::time_point.
        auto        now   = system_clock::now();
        std::time_t now_c = system_clock::to_time_t(now);

        // Convert to local time.
        std::tm local_tm{};
#ifdef _WIN32
        localtime_s(&local_tm, &now_c);
#else
        localtime_r(&now_c, &local_tm);
#endif

        // Format date and time.
        std::ostringstream oss;
        oss << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S");

        // Compute timezone offset.
        std::tm utc_tm{};
#ifdef _WIN32
        gmtime_s(&utc_tm, &now_c);
#else
        gmtime_r(&now_c, &utc_tm);
#endif

        // Offset in seconds.
        int offset_sec
            = static_cast<int>(std::difftime(std::mktime(&local_tm), std::mktime(&utc_tm)));
        char sign          = offset_sec >= 0 ? '+' : '-';
        offset_sec         = std::abs(offset_sec);
        int offset_hours   = offset_sec / 3600;
        int offset_minutes = (offset_sec % 3600) / 60;

        // Append timezone.
        oss << sign << std::setw(2) << std::setfill('0') << offset_hours << ':' << std::setw(2)
            << std::setfill('0') << offset_minutes;

        return oss.str();
    }

    /// Serializes CLI settings into JSON.
    std::string serialize_settings(const settings& settings) const
    {
        std::ostringstream ss;
        ss << "{";

        const auto& s = settings;

        ss << "\"size\":" << s.size;
        ss << ",\"hot\":" << std::boolalpha << s.hot;
        ss << ",\"seed\":" << s.seed;
        ss << ",\"json_out\":\"" << s.json_out << "\"";
        ss << ",\"csv_out\":\"" << s.csv_out << "\"";
        ss << ",\"filter\":\"" << json_escape(s.filter) << "\"";
        ss << ",\"dry\":" << s.dry;
        ss << ",\"min_gpu_ms_per_batch\":" << s.min_gpu_ms_per_batch;
        ss << ",\"min_secs\":" << s.min_secs;
        ss << ",\"noise_timeout_secs\":" << s.noise_timeout_secs;
        ss << ",\"batch_window_size\":" << s.batch_window_size;
        ss << ",\"noise_tolerance_percent\":" << s.noise_tolerance_percent;
        ss << ",\"min_gpu_temp\":" << s.min_gpu_temp;
        ss << ",\"max_gpu_temp\":" << s.max_gpu_temp;
        ss << ",\"max_warming_secs\":" << s.max_warming_secs;
        ss << ",\"max_cooling_secs\":" << s.max_cooling_secs;
        ss << ",\"output_batches\":" << s.output_batches;
        ss << ",\"spaces_per_indent\":" << s.spaces_per_indent;
        ss << ",\"stream_blocking_timeout_secs\":" << s.stream_blocking_timeout_secs;

        ss << "}";
        return ss.str();
    }

    /// Escapes a JSON string, so it transforms "\" to "\\".
    std::string json_escape(std::string_view s) const
    {
        std::ostringstream o;
        for(unsigned char c : s)
        {
            switch(c)
            {
                case '\"': o << "\\\""; break;
                case '\\': o << "\\\\"; break;
                case '\b': o << "\\b"; break;
                case '\f': o << "\\f"; break;
                case '\n': o << "\\n"; break;
                case '\r': o << "\\r"; break;
                case '\t': o << "\\t"; break;
                default:
                    if(c < 0x20)
                    {
                        o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                    }
                    else
                    {
                        o << c;
                    }
            }
        }
        return o.str();
    }

    /// Serializes custom CLI settings into JSON.
    std::string serialize_custom_settings(const settings& settings) const
    {
        if(settings.custom_args.empty())
        {
            return "";
        }

        std::ostringstream ss;
        ss << "{";
        ss << std::boolalpha;

        bool first = true;
        for(const auto& [key, value] : settings.custom_args)
        {
            if(!first)
                ss << ",";
            first = false;

            ss << "\"" << key << "\":";

            std::visit(
                [&](auto const& val)
                {
                    using V = std::decay_t<decltype(val)>;
                    if constexpr(std::is_same_v<V, std::string>)
                    {
                        ss << "\"" << val << "\"";
                    }
                    else
                    {
                        ss << val;
                    }
                },
                value);
        }

        ss << "}";
        return ss.str();
    }

    /// Formats a JSON string with indentation.
    std::string indent(std::string_view str, size_t starting_indent_level)
    {
        if(m_spaces_per_indent == 0)
            return std::string(str);

        std::ostringstream out;
        size_t             indent_level = starting_indent_level;
        bool               in_string    = false;

        if(indent_level > 0)
            out << std::string(indent_level * m_spaces_per_indent, ' ');

        for(size_t i = 0; i < str.size(); ++i)
        {
            char c = str[i];

            if(c == '\"')
            {
                // Detect escaped quotes.
                bool   escaped = false;
                size_t j       = i;
                while(j > 0 && str[--j] == '\\')
                    escaped = !escaped;
                if(!escaped)
                    in_string = !in_string;
                out << c;
            }
            else if(!in_string)
            {
                switch(c)
                {
                    case '{':
                    case '[':
                        out << c << '\n';
                        ++indent_level;
                        out << std::string(indent_level * m_spaces_per_indent, ' ');
                        break;
                    case '}':
                    case ']':
                        out << '\n';
                        if(indent_level > 0)
                            --indent_level;
                        out << std::string(indent_level * m_spaces_per_indent, ' ') << c;
                        break;
                    case ',':
                        out << c << '\n' << std::string(indent_level * m_spaces_per_indent, ' ');
                        break;
                    case ':': out << c << ' '; break;
                    default:
                        if(!isspace(static_cast<unsigned char>(c)))
                            out << c;
                        break;
                }
            }
            else
            {
                out << c;
            }
        }

        return out.str();
    }

    /// Serializes benchmark flags into JSON.
    std::string serialize_flags(flags::FlagTag flags) const
    {
        std::ostringstream ss;
        ss << "{";
        ss << "\"sync\":" << std::boolalpha << flags.has(flags::Flags::sync);
        ss << "}";
        return ss.str();
    }

    /// Outputs specialization information to JSON.
    void output_json_specialization(size_t           index,
                                    std::string_view name,
                                    std::string_view serialized_meta,
                                    size_t           kernels_per_batch,
                                    double           ms_per_batch,
                                    double           bytes_per_sec,
                                    double           items_per_sec,
                                    size_t           bytes_per_item,
                                    size_t           items,
                                    double           noise_percent,
                                    double           elapsed_host_secs,
                                    double           elapsed_gpu_secs,
                                    bool             noise_timeout)
    {
        // Specializations need a comma between them.
        if(!m_first_specialization)
        {
            m_json_out << ",";
            if(m_spaces_per_indent > 0)
                m_json_out << "\n";
        }
        else
            m_first_specialization = false;

        m_json_out << indent(serialize_specialization(index,
                                                      name,
                                                      serialized_meta,
                                                      kernels_per_batch,
                                                      ms_per_batch,
                                                      bytes_per_sec,
                                                      items_per_sec,
                                                      bytes_per_item,
                                                      items,
                                                      noise_percent,
                                                      elapsed_host_secs,
                                                      elapsed_gpu_secs,
                                                      noise_timeout),
                             2);

        m_json_out.flush();
    }

    /// Outputs specialization information to CSV.
    void output_csv_specialization(size_t           index,
                                   std::string_view name,
                                   double           bytes_per_sec,
                                   double           items_per_sec,
                                   bool             noise_timeout,
                                   double           noise_percent)
    {
        m_csv_out << index << ",\"" << name << "\"" << "," << bytes_per_sec << ","
                  << bytes_per_sec / (1024.0 * 1024.0 * 1024.0) << "," << items_per_sec << ","
                  << noise_timeout << "," << noise_percent << "\n"
                  << std::flush;
    }

    /// Serializes a specialization into JSON format.
    std::string serialize_specialization(size_t           index,
                                         std::string_view name,
                                         std::string_view serialized_meta,
                                         size_t           kernels_per_batch,
                                         double           ms_per_batch,
                                         double           bytes_per_sec,
                                         double           items_per_sec,
                                         size_t           bytes_per_item,
                                         size_t           items,
                                         double           noise_percent,
                                         double           elapsed_host_secs,
                                         double           elapsed_gpu_secs,
                                         bool             noise_timeout) const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"index\":" << index;
        ss << ",\"name\":\"" << name << "\"";

        ss << ",\"bytes_per_second\":" << bytes_per_sec;
        ss << ",\"items_per_second\":" << items_per_sec;

        ss << ",\"bytes_per_item\":" << bytes_per_item;
        ss << ",\"items\":" << items;

        ss << ",\"noise_timeout\":" << std::boolalpha << noise_timeout;
        ss << ",\"noise_percent\":" << noise_percent;

        ss << ",\"meta\":" << serialized_meta;

        ss << ",\"elapsed_secs\":{";
        ss << "\"host\":" << elapsed_host_secs;
        ss << ",\"gpu\":" << elapsed_gpu_secs;
        ss << "}";

#if PRIMBENCH_HAS_MONITORING
        ss << ",\"gpu_temp_celsius\":{";
        ss << "\"start\":" << monitor::instance().get_saved_start_temp();
        ss << ",\"end\":" << monitor::instance().get_temp();
        ss << "}";
#endif

        ss << ",\"calls\":{";
        ss << "\"kernel_calls_per_batch\":" << kernels_per_batch;
        ss << ",\"ms_per_batch\":" << ms_per_batch;
        ss << ",\"batches\":" << m_batches.size();
        ss << ",\"kernel_calls\":" << kernels_per_batch * m_batches.size();
        ss << "}";

        if(m_output_batches)
        {
            ss << ",\"batches\":[";
            for(size_t i = 0; i < m_batches.size(); ++i)
            {
                if(i > 0)
                    ss << ",";
                const auto& batch = m_batches[i];
                ss << "{";
                ss << "\"batch_ms\":" << batch.batch_ms;
                ss << ",\"iterations_ms\":" << serialize_iterations_ms(batch.iterations_ms);
                ss << "}";
            }
            ss << "]";
        }

        ss << "}";
        return ss.str();
    }

    /// Serializes iteration times into JSON array.
    std::string serialize_iterations_ms(const std::vector<float>& iterations_ms) const
    {
        std::ostringstream ss;
        ss << "[";
        for(size_t i = 0; i < iterations_ms.size(); ++i)
        {
            if(i > 0)
                ss << ",";
            ss << iterations_ms[i];
        }
        ss << "]";
        return ss.str();
    }

    /// Serializes summary info into JSON format.
    std::string serialize_summary() const
    {
        std::ostringstream ss;
        ss << "\"summary\":{";

        ss << "\"noise_timeouts\":" << m_noise_timeouts;

        auto now          = std::chrono::steady_clock::now();
        auto elapsed_host = now - m_program_start_time;

        double elapsed_host_secs = std::chrono::duration<double>(elapsed_host).count();

        ss << ",\"elapsed_secs\":{";
        ss << "\"host\":" << elapsed_host_secs;
        ss << ",\"gpu\":" << m_total_elapsed_gpu_secs;
        ss << "}";

        ss << "}";
        return ss.str();
    }

    std::chrono::time_point<std::chrono::steady_clock> m_program_start_time;

    std::ofstream      m_json_out; ///< JSON output file stream.
    std::ofstream      m_csv_out; ///< CSV output file stream.
    bool               m_first_specialization; ///< True if first specialization output.
    std::vector<batch> m_batches; ///< Stored batch results.
    bool               m_output_batches; ///< Whether to output each batch.
    uint32_t           m_spaces_per_indent; ///< JSON indentation spaces.
    double             m_total_elapsed_gpu_secs = 0; ///< Number of elapsed GPU seconds.
    uint32_t           m_noise_timeouts         = 0; ///< Number of noise timeouts.
    bool               m_outputting_csv; ///< Whether a CSV file is output.

}; // class logger

/// Provides functions for progress display and formatting during GPU benchmarking.
namespace progress
{
// Column widths and formatting constants.
static constexpr int         noise_column_width         = 5;
static constexpr int         gpu_temp_column_width      = 6;
static constexpr int         bytes_per_sec_column_width = 9;
static constexpr const char* horizontal_bar             = u8"─";

/// Prints the table header for dry-run mode output.
inline void print_dry_header(std::string_view algo_name,
                             size_t           specialization_column_width,
                             size_t           index_column_width,
                             size_t           specialization_count)
{
    std::string status_header       = "Status of " + std::string(algo_name);
    size_t      status_column_width = status_header.size();

    std::cout << std::setw(status_column_width) << std::left << status_header << "  "
              << std::setw(specialization_column_width) << std::left << "Specialization";

    if(specialization_count > 1)
    {
        std::cout << "  " << std::setw(index_column_width) << std::right
                  << ("Index/" + std::to_string(specialization_count));
    }
    std::cout << "\n";

    size_t underline_width = status_column_width + 2 + specialization_column_width;
    if(specialization_count > 1)
    {
        underline_width += 2 + index_column_width;
    }

    for(size_t i = 0; i < underline_width; ++i)
        std::cout << horizontal_bar;
    std::cout << "\n";
}

/// Prints the table header for algorithm progress output.
inline void print_header(std::string_view          algo_name,
                         size_t                    specialization_column_width,
                         size_t                    index_column_width,
                         size_t                    specialization_count,
                         std::chrono::seconds::rep noise_timeout_secs)
{
    std::string status_header = "Status of " + std::string(algo_name);
    std::string noisy_status  = "Noisy timed out after " + std::to_string(noise_timeout_secs) + "s";

    size_t status_column_width = std::max(status_header.size(), noisy_status.size());

    std::cout << std::setw(status_column_width) << std::left << status_header << "  "
              << std::setw(noise_column_width) << std::left << "Noise";

#if PRIMBENCH_HAS_MONITORING
    std::cout << "  " << std::setw(gpu_temp_column_width) << std::left << "GPU °C";
#endif

    std::cout << "  " << std::setw(bytes_per_sec_column_width) << std::left << "Bytes/sec" << "  "
              << std::setw(specialization_column_width) << std::left << "Specialization";

    if(specialization_count > 1)
    {
        std::cout << "  " << std::setw(index_column_width) << std::right
                  << ("Index/" + std::to_string(specialization_count));
    }
    std::cout << "\n";

    size_t underline_width = status_column_width;
    underline_width += 2 + noise_column_width;
#if PRIMBENCH_HAS_MONITORING
    underline_width += 2 + gpu_temp_column_width;
#endif
    underline_width += 2 + bytes_per_sec_column_width;
    underline_width += 2 + specialization_column_width;
    if(specialization_count > 1)
    {
        underline_width += 2 + index_column_width;
    }

    for(size_t i = 0; i < underline_width; ++i)
        std::cout << horizontal_bar;
    std::cout << "\n";
}

/// Displays GPU warming progress.
inline void print_warming(uint16_t gpu_temp, uint16_t min_gpu_temp)
{
    std::cout << clearline << "Warming GPU from " << blue << gpu_temp << "°C" << reset << " to "
              << green << min_gpu_temp << "°C" << reset << std::flush;
}

/// Displays GPU cooling progress.
inline void print_cooling(uint16_t gpu_temp, uint16_t max_gpu_temp)
{
    std::cout << clearline << "Cooling GPU from " << red << gpu_temp << "°C" << reset << " to "
              << green << max_gpu_temp << "°C" << reset << std::flush;
}

/// Prints a single line of output for dry-run mode, containing the status, specialization and index.
inline void print_dry_progress(std::string_view specialization,
                               std::string_view algo_name,
                               size_t           specialization_index,
                               size_t           specialization_column_width,
                               size_t           index_column_width,
                               bool             print_index)
{
    std::ostringstream line;

    std::string status_header       = "Status of " + std::string(algo_name);
    size_t      status_column_width = status_header.size();

    line << clearline << green << std::setw(status_column_width) << std::left << "Success" << reset;

    line << "  " << std::setw(specialization_column_width) << std::left << specialization;

    if(print_index)
    {
        line << "  " << std::setw(index_column_width) << std::right << specialization_index;
    }

    std::cout << line.str() << "\n" << std::flush;
}

/// Returns an estimated time in the format "9h 59m 59s".
inline std::string format_eta(double remaining_secs)
{
    assert(remaining_secs > 0.0);

    if(remaining_secs > 35999.0)
        remaining_secs = 35999.0; // 9h 59m 59s

    int h = static_cast<int>(remaining_secs) / 3600;
    int m = (static_cast<int>(remaining_secs) % 3600) / 60;
    int s = static_cast<int>(remaining_secs) % 60;

    std::ostringstream oss;
    if(h > 0)
        oss << h << "h " << m << "m " << s << "s";
    else if(m > 0)
        oss << std::setw(3) << "" << m << "m " << s << "s";
    else
        oss << std::setw(7) << "" << s << "s";

    return oss.str();
}

/// Prints real-time progress updates for algorithm execution.
inline void print_progress(uint64_t         iteration,
                           double           noise_percent,
                           double           bytes_per_sec,
                           std::string_view status_msg,
                           std::string_view specialization,
                           std::string_view algo_name,
                           uint64_t         batch_window_size,
                           size_t           specialization_index,
                           size_t           specialization_column_width,
                           size_t           index_column_width,
                           bool             print_index,
                           double           elapsed_host_secs,
                           double           noise_timeout_secs,
                           double           noise_tolerance_percent,
                           double           estimated_remaining_secs)
{
    std::string status_header = "Status of " + std::string(algo_name);

    std::chrono::seconds::rep secs         = noise_timeout_secs; // Casts to an integer.
    std::string               noisy_status = "Noisy timed out after " + std::to_string(secs) + "s";

    size_t status_column_width = std::max(status_header.size(), noisy_status.size());

    std::string batch_str = status_msg.empty() ? "Batch " + std::to_string(iteration) + "/"
                                                     + std::to_string(batch_window_size)
                                               : std::string(status_msg);

    size_t bar_width = 0;
    if(batch_str.size() + 1 < status_column_width)
    {
        bar_width = status_column_width - batch_str.size() - 1;
    }

    std::ostringstream line;
    line << clearline << batch_str;

    // Progress bar (only shown during iteration).
    if(status_msg.empty())
    {
        line << " ";

        uint64_t capped_iteration = std::min(iteration, batch_window_size);
        size_t   filled           = (bar_width * capped_iteration) / batch_window_size;

        // Compute fraction of the yellow noise timeout overlay.
        size_t yellow_chars = 0;
        if(filled >= bar_width)
        {
            double frac  = std::min(1.0, elapsed_host_secs / noise_timeout_secs);
            yellow_chars = bar_width * frac;
        }

        // Build the progress bar in one pass.
        for(size_t j = 0; j < bar_width; ++j)
        {
            if(j < yellow_chars)
                line << yellow << horizontal_bar;
            else if(j < filled)
                line << green << horizontal_bar;
            else
                line << gray << horizontal_bar;
        }
        line << reset;
    }

    // Alignment for subsequent columns.
    size_t used = batch_str.size() + status_msg.empty() + (status_msg.empty() ? bar_width : 0);
    if(used < status_column_width)
        line << std::string(status_column_width - used, ' ');

    // Noise %.
    std::ostringstream percent_stream;

    if(noise_percent >= 100.0)
    {
        percent_stream << static_cast<int>(noise_percent);
    }
    else
    {
        percent_stream << std::fixed << std::setprecision(1) << noise_percent;
    }

    auto& noise_color = (noise_percent > noise_tolerance_percent) ? red : reset;

    line << "  " << noise_color << std::setw(noise_column_width - sizeof('%')) << std::right
         << percent_stream.str() << "%" << reset;

#if PRIMBENCH_HAS_MONITORING
    // GPU temperature.
    line << "  " << std::setw(gpu_temp_column_width) << std::right
         << monitor::instance().get_temp();
#endif

    // Bytes/sec.
    line << "  " << std::setw(bytes_per_sec_column_width) << std::right << std::scientific
         << std::setprecision(2) << bytes_per_sec;

    // Specialization.
    line << "  " << std::setw(specialization_column_width) << std::left << specialization;

    // Index.
    if(print_index)
    {
        if(status_msg.empty() && estimated_remaining_secs > 0.0)
        {
            line << "  " << std::setw(index_column_width) << std::right
                 << format_eta(estimated_remaining_secs);
        }
        else
        {
            line << "  " << std::setw(index_column_width) << std::right << specialization_index;
        }
    }

    // Colorized status messages.
    if(status_msg.find("Success") != std::string::npos)
        std::cout << green;
    else if(status_msg.find("Noisy timed out") != std::string::npos)
        std::cout << red;

    std::cout << line.str() << std::flush;
}

} // namespace progress

#ifdef __HIP__

/// Kernel that blocks the GPU stream until unblocked or timeout occurs.
__global__
void block_stream_kernel(volatile int32_t* is_blocked,
                         volatile int32_t* timeout_flag,
                         double            timeout_seconds,
                         long long int     wall_clock_rate)
{
    const long long int start_time = wall_clock64();
    const long long int timeout_cycles
        = static_cast<long long int>(timeout_seconds * wall_clock_rate * 1000.0);

    while(*is_blocked == 1)
    {
        if(wall_clock64() - start_time > timeout_cycles)
        {
            *timeout_flag = 1; // Signal timeout to host.
            break; // Exit loop.
        }
    }
}

#elif defined(__CUDACC__)

/// Kernel that blocks the GPU stream until unblocked or timeout occurs.
__global__
void block_stream_kernel(volatile int32_t* is_blocked,
                         volatile int32_t* timeout_flag,
                         double            timeout_seconds)
{
    using clock                                    = cuda::std::chrono::high_resolution_clock;
    auto                                start_time = clock::now();
    cuda::std::chrono::duration<double> timeout_duration(timeout_seconds);

    while(*is_blocked == 1)
    {
        auto elapsed = cuda::std::chrono::duration<double>(clock::now() - start_time);
        if(elapsed >= timeout_duration)
        {
            *timeout_flag = 1; // Signal timeout to host.
            break; // Exit loop.
        }
    }
}

#endif

/// Manages synchronization between host and GPU streams
/// by blocking execution, until explicitly unblocked or a timeout occurs.
class stream_blocker
{
public:
    stream_blocker() = delete;

    /// Constructs a stream_blocker for a given stream.
    stream_blocker(stream_t stream, double stream_blocking_timeout_secs)
        : m_stream(stream), m_stream_blocking_timeout_secs(stream_blocking_timeout_secs)
    {
        // Register host memory for blocking flags.
        PRIMBENCH_CHECK(host_register(&m_host_flag, sizeof(m_host_flag), host_register_mapped));
        PRIMBENCH_CHECK(
            host_register(&m_host_timeout_flag, sizeof(m_host_timeout_flag), host_register_mapped));

        // Temporary non-volatile pointers.
        int32_t* temp_device_flag         = nullptr;
        int32_t* temp_device_timeout_flag = nullptr;

        // Get device pointers to mapped host memory.
        PRIMBENCH_CHECK(
            host_get_device_pointer(reinterpret_cast<void**>(&temp_device_flag), &m_host_flag, 0));
        PRIMBENCH_CHECK(host_get_device_pointer(reinterpret_cast<void**>(&temp_device_timeout_flag),
                                                &m_host_timeout_flag,
                                                0));

        // Assign temporary pointers to volatile members.
        m_device_flag         = temp_device_flag;
        m_device_timeout_flag = temp_device_timeout_flag;

#ifdef __HIP__
        // Query wall clock rate once (constant per device).
        int device_id;
        PRIMBENCH_CHECK(hipGetDevice(&device_id));
        int wall_clk_rate_k_hz = 0;
        PRIMBENCH_CHECK(
            hipDeviceGetAttribute(&wall_clk_rate_k_hz, hipDeviceAttributeWallClockRate, device_id));
        m_wall_clock_rate = wall_clk_rate_k_hz;
#endif
    }

    /// Destructor that unregisters host memory.
    ~stream_blocker()
    {
        PRIMBENCH_CHECK(host_unregister(&m_host_flag));
        PRIMBENCH_CHECK(host_unregister(&m_host_timeout_flag));
    }

    /// Launches a blocking kernel on the stream until unblocked or timed out.
    void block()
    {
        volatile int32_t& flag = m_host_flag;
        flag                   = 1;

        volatile int32_t& timeout_flag = m_host_timeout_flag;
        timeout_flag                   = 0;

#ifdef __HIP__
        block_stream_kernel<<<dim3(1), dim3(1), 0, m_stream>>>(m_device_flag,
                                                               m_device_timeout_flag,
                                                               m_stream_blocking_timeout_secs,
                                                               m_wall_clock_rate);
#elif defined(__CUDACC__)
        block_stream_kernel<<<dim3(1), dim3(1), 0, m_stream>>>(m_device_flag,
                                                               m_device_timeout_flag,
                                                               m_stream_blocking_timeout_secs);
#endif
    }

    /// Unblocks the stream by resetting the blocking flag.
    void unblock()
    {
        volatile int32_t& flag = m_host_flag;
        flag                   = 0;
    }

    /// Checks if the GPU timed out during blocking.
    ///
    /// This should be called after stream synchronization.
    void check_timeout()
    {
        if(m_host_timeout_flag)
        {
            std::cerr
                << "\nError: Stream blocking timed out after " << m_stream_blocking_timeout_secs
                << " seconds.\nThe stream is blocked while queueing kernel calls.\nIf your kernel "
                   "is synchronous, pass primbench::flags::sync to disable stream blocking.\nYou "
                   "can increase the timeout by passing --stream-blocking-timeout-secs <secs>.\n";
            exit(EXIT_FAILURE);
        }
    }

private:
    stream_t      m_stream;
    double        m_stream_blocking_timeout_secs;
    long long int m_wall_clock_rate;

    int32_t m_host_flag         = 0;
    int32_t m_host_timeout_flag = 0;

    volatile int32_t* m_device_flag         = nullptr;
    volatile int32_t* m_device_timeout_flag = nullptr;
}; // class stream_blocker

/// Simple JSON-like container.
///
/// Stores key-value pairs where values can be nested JSON objects,
/// strings, integers, or doubles. Provides basic serialization to JSON and to a human-readable name.
struct json
{
    using key_type = std::string;
    using value_type
        = std::variant<std::shared_ptr<json>, std::string, bool, double, int, unsigned int, size_t>;

    // Using std::map instead of std::unordered_map,
    // as we want a deterministic order.
    using map_type = std::map<key_type, value_type>;

    /// Adds a key-value pair to the JSON object.
    template<typename T>
    json& add(std::string_view key, T value)
    {
        if constexpr(std::is_same_v<T, json>)
        {
            // Copy nested JSON.
            m_map[std::string(key)] = std::make_shared<json>(value);
        }
        else if constexpr(std::is_convertible_v<T, std::string_view>)
        {
            m_map[std::string(key)] = std::string(std::string_view(value));
        }
        else
        {
            m_map[std::string(key)] = value;
        }
        return *this;
    }

    /// Serializes the JSON object to a JSON string.
    std::string serialize() const
    {
        std::ostringstream ss;
        ss << "{";
        ss << std::boolalpha;

        bool is_first = true;
        for(auto const& [key, value] : m_map)
        {
            if(!is_first)
                ss << ",";
            else
                is_first = false;

            ss << "\"" << key << "\":";

            std::visit(
                [&](auto const& val)
                {
                    using V = std::decay_t<decltype(val)>;
                    if constexpr(std::is_same_v<V, std::shared_ptr<json>>)
                    {
                        ss << val->serialize();
                    }
                    else if constexpr(std::is_same_v<V, std::string>)
                    {
                        ss << "\"" << val << "\"";
                    }
                    else
                    {
                        ss << val;
                    }
                },
                value);
        }

        ss << "}";
        return ss.str();
    }

    /// Serializes the JSON object into a human-readable name.
    std::string serialize_name() const
    {
        static const std::vector<std::string_view> blacklist         = {"lvl", "algo"};
        static const std::vector<std::string_view> stripped_prefixes = {"rocprim::", "common::"};

        auto strip_prefixes = [&](std::string& s)
        {
            for(auto p : stripped_prefixes)
            {
                while(true)
                {
                    size_t pos = s.find(p);
                    if(pos == std::string::npos)
                        break;
                    s.erase(pos, p.size());
                }
            }
        };

        // Recursive formatter.
        std::function<std::string(const json&, bool)> fmt
            = [&](const json& obj, bool toplevel) -> std::string
        {
            std::string out;
            if(!toplevel)
                out += "{ ";

            bool first = true;
            for(auto const& [k_sv, val] : obj.m_map)
            {
                std::string key(k_sv);

                // Skip blacklisted keys.
                if(std::find(blacklist.begin(), blacklist.end(), key) != blacklist.end())
                    continue;

                std::string vstr;
                std::visit(
                    [&](auto const& v)
                    {
                        using V = std::decay_t<decltype(v)>;

                        if constexpr(std::is_same_v<V, std::shared_ptr<json>>)
                        {
                            vstr = fmt(*v, false);
                        }
                        else if constexpr(std::is_same_v<V, std::string>)
                        {
                            vstr = v;
                            strip_prefixes(vstr);
                        }
                        else
                        {
                            // Can't use std::to_string(v),
                            // as that doesn't trim trailing zeros.
                            std::ostringstream ss;
                            ss << std::boolalpha;
                            ss << v;
                            vstr = ss.str();
                        }
                    },
                    val);

                // Skip default configs.
                if(key == "cfg" && vstr == "default")
                    continue;

                if(!first)
                    out += ", ";
                else
                    first = false;

                out += key + ": " + vstr;
            }

            if(!toplevel)
                out += " }";
            return out;
        };

        return fmt(*this, true);
    }

    /// Retrieves a value by key.
    template<typename T>
    const T& get(std::string_view key) const
    {
        const auto& val = m_map.at(std::string(key));
        if constexpr(std::is_same_v<T, json>)
        {
            return *std::get<std::shared_ptr<json>>(val);
        }
        else
        {
            return std::get<T>(val);
        }
    }

private:
    map_type m_map = {};
}; // struct json

struct device_storage
{
    device_storage(size_t size) : m_size(size), m_device_ptr(nullptr)
    {
        if(size > 0)
            PRIMBENCH_CHECK(gpu_malloc(&m_device_ptr, size));
    }

    ~device_storage()
    {
        if(m_device_ptr != nullptr)
            PRIMBENCH_CHECK(gpu_free(m_device_ptr));
    }

    size_t get_size() const
    {
        return m_size;
    }

    template<typename T = void>
    T* get_ptr() const
    {
        return static_cast<T*>(m_device_ptr);
    };

    // This type is not copy assignable.
    device_storage& operator=(const device_storage&) = delete;
    // This type is not copy constructible.
    device_storage(const device_storage&) = delete;

private:
    const size_t m_size;
    void*        m_device_ptr;
}; // struct device_storage

/// Cache thrashing utility to ensure the cache contains irrelevant data.
struct cache_thrasher
{
public:
    /// Initializes the cache thrasher by allocating the required memory on device.
    ///
    /// Currently, the actual largest GPU cache size cannot be queried via HIP, so this
    /// conservative size is used instead. Future support via HSA could make this runtime-
    /// queryable.
    cache_thrasher(size_t cache_size = PRIMBENCH_GPU_CACHE_SIZE) : m_device_storage(cache_size) {}

    /// Clears the cache by thrashing memory on device.
    ///
    /// Zeros a buffer of size `m_cache_size` to evict cached data. Should be called before
    /// each kernel launch.
    void clear_cache(stream_t stream)
    {
        PRIMBENCH_CHECK(
            memset_async(m_device_storage.get_ptr(), 0, m_device_storage.get_size(), stream));
    }

    // This type is not copy assignable.
    cache_thrasher& operator=(const cache_thrasher&) = delete;
    // This type is not copy constructible.
    cache_thrasher(const cache_thrasher&) = delete;

private:
    const device_storage m_device_storage;
}; // struct cache_thrasher

#if PRIMBENCH_HAS_MONITORING

/// Warms the GPU, using complex enough dummy kernel code
/// that the compiler can't optimize it away.
__global__
void warmup_kernel(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float x = 0.5f + idx * 0.0001f;

        for(int i = 0; i < 10000; ++i)
        {
            x += sinf(x) * cosf(x);
            x *= 1.0000001f;
            x = sqrtf(x + 1.0f);
            x = logf(x + 1.0f);
        }

        data[idx] = x;
    }
}

class gpu_warmer
{
public:
    // Delete all copy/move constructors and assignment operators.
    gpu_warmer(const gpu_warmer&)            = delete;
    gpu_warmer& operator=(const gpu_warmer&) = delete;
    gpu_warmer(gpu_warmer&&)                 = delete;
    gpu_warmer& operator=(gpu_warmer&&)      = delete;

    /// Singleton accessor.
    static gpu_warmer& instance()
    {
        static gpu_warmer instance;
        return instance;
    }

    gpu_warmer()
    {
        int grid_size;
        PRIMBENCH_CHECK(occupancy_max_potential_block_size(
            &grid_size, // Minimum grid size for full occupancy.
            &m_threads_per_block, // Block size for full occupancy.
            warmup_kernel,
            0, // Dynamic shared memory.
            0 // Block size limit (0 = no limit).
            ));

        m_num_items = grid_size * m_threads_per_block;

        m_device_storage = std::make_unique<device_storage>(m_num_items * sizeof(float));
    }

    void warm_up(stream_t stream, uint16_t min_gpu_temp, double max_warming_secs) const
    {
        auto start = std::chrono::steady_clock::now();

        while(true)
        {
            uint16_t gpu_temp = monitor::instance().get_temp();
            if(gpu_temp >= min_gpu_temp)
                break;

            if(use_color())
                progress::print_warming(gpu_temp, min_gpu_temp);

            dim3 blocks(m_num_items / m_threads_per_block);
            dim3 threads(m_threads_per_block);

            warmup_kernel<<<blocks, threads, 0, stream>>>(m_device_storage->get_ptr<float>(),
                                                          m_num_items);

            PRIMBENCH_CHECK(stream_synchronize(stream));

            auto duration = std::chrono::steady_clock::now() - start;
            if(duration >= std::chrono::duration<double>(max_warming_secs))
            {
                std::cerr << "\nError: Failed to warm up after " << max_warming_secs
                          << " seconds\n";
                exit(EXIT_FAILURE);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    int m_threads_per_block;
    int m_num_items;

    std::unique_ptr<device_storage> m_device_storage;
}; // class gpu_warmer

#endif // PRIMBENCH_HAS_MONITORING

/// Manages benchmark execution, GPU warm-up/cool-down, timing, and logging.
class state
{
public:
    state(std::string_view algo,
          json             meta,
          size_t           specialization_index,
          size_t           specialization_count,
          stream_t         stream,
          logger&          logger,
          stream_blocker&  stream_blocker,
          const settings&  settings,
          flags::FlagTag   flags,
          size_t           specialization_column_width,
          size_t           index_column_width,
          bool             print_index,
          cache_thrasher&  cache)
        : stream(stream)
        , size(settings.size)
        , seed(settings.seed)
        , m_algo(algo)
        , m_meta(std::move(meta))
        , m_specialization_index(specialization_index)
        , m_specialization_count(specialization_count)
        , m_logger(logger)
        , m_stream_blocker(stream_blocker)
        , m_settings(settings)
        , m_flags(flags)
        , m_specialization_column_width(specialization_column_width)
        , m_index_column_width(index_column_width)
        , m_print_index(print_index)
        , m_cache(cache)
    {}

    /// Sets the total number of items processed per iteration.
    ///
    /// This must be called exactly once before calling \ref run() or any
    /// memory tracking methods such as \ref add_reads() or \ref add_writes().
    void set_items(size_t items)
    {
        if(m_has_set_items)
        {
            std::cerr << "Error: Can't call set_items() twice\n";
            exit(EXIT_FAILURE);
        }
        m_has_set_items = true;

        m_items = items;
    }

    /// Adds an estimate of global memory reads performed by the benchmark.
    ///
    /// Must be called after \ref set_items() and before any call to
    /// \ref add_writes(). Multiple calls accumulate total read bytes.
    ///
    /// The total number of bytes read (from all calls to this function)
    /// is **summed together with the total bytes written** (added via
    /// \ref add_writes()) to compute the reported memory throughput.
    template<typename T>
    void add_reads(size_t items)
    {
        if(!m_has_set_items)
        {
            std::cerr << "Error: Can't call add_reads() before calling set_items()\n";
            exit(EXIT_FAILURE);
        }
        if(m_has_set_writes)
        {
            std::cerr << "Error: Can't call add_reads() after calling add_writes()\n";
            exit(EXIT_FAILURE);
        }

        size_t bytes = items * sizeof(T);
        m_read_write_bytes += bytes;
    }

    /// Adds an estimate of global memory writes performed by the benchmark.
    ///
    /// Must be called after \ref set_items(). Multiple calls accumulate total
    /// written bytes.
    ///
    /// The total number of bytes written (from all calls to this function)
    /// is **summed together with the total bytes read** (added via
    /// \ref add_reads()) to compute the reported memory throughput.
    template<typename T>
    void add_writes(size_t items)
    {
        if(!m_has_set_items)
        {
            std::cerr << "Error: Can't call add_writes() before calling set_items()\n";
            exit(EXIT_FAILURE);
        }
        m_has_set_writes = true;

        size_t bytes = items * sizeof(T);
        m_read_write_bytes += bytes;
    }

    /// Sets a callback to run before each iteration, which should be used
    /// to reset the input data of in-place algorithms.
    void run_before_every_iteration(std::function<void()> lambda)
    {
        m_run_before_every_iteration_lambda = lambda;
    }

    /// Executes the benchmark loop for the provided kernel.
    ///
    /// Handles warm-up, timing, CV-based stopping, and logging.
    ///
    /// The benchmark manages all required stream synchronization internally to
    /// ensure accurate timing and prevent command queue buildup. Users should not
    /// perform any manual synchronization before or during the benchmark run.
    void run(std::function<void()> kernel)
    {
        if(!m_has_set_items)
        {
            std::cerr << "Error: Can't call run() before calling set_items()\n";
            exit(EXIT_FAILURE);
        }

        std::string name            = m_meta.serialize_name();
        std::string serialized_meta = m_meta.serialize();

        size_t bytes_per_item = m_read_write_bytes / m_items;

#if PRIMBENCH_HAS_MONITORING
        warm_up();
        cool_down();
#endif

        init_kernels_per_batch(kernel);

        // Reserve space for start and stop events for each iteration.
        std::vector<event_t> events(m_kernels_per_batch * 2);
        for(auto& event : events)
            PRIMBENCH_CHECK(event_create(&event));

        uint64_t           iterations = 0;
        std::vector<float> iterations_ms(m_kernels_per_batch);

        double elapsed_gpu_secs = 0.0;

        double estimated_remaining_secs = get_estimated_remaining_secs();

#if PRIMBENCH_HAS_MONITORING
        monitor::instance().save_start_temp();
#endif

        auto start = std::chrono::steady_clock::now();

        while(true)
        {
            if(run_iteration(kernel,
                             events,
                             iterations,
                             iterations_ms,
                             start,
                             elapsed_gpu_secs,
                             name,
                             serialized_meta,
                             bytes_per_item,
                             estimated_remaining_secs))
            {
                break;
            }
        }

        for(const auto& event : events)
            PRIMBENCH_CHECK(event_destroy(event));
    }

    // Public fields accessed directly by benchmarks.
    const stream_t stream; ///< Stream used by benchmarks for kernel launches.
    const size_t   size; ///< Input size processed per iteration.
    const uint32_t seed; ///< Random seed used for reproducible benchmark inputs.

private:
    /// Performs a single iteration of the benchmark loop.
    ///
    /// @return `true` if the benchmark should stop (stable noise or timeout).
    bool run_iteration(std::function<void()>&                       kernel,
                       std::vector<event_t>&                        events,
                       uint64_t&                                    iterations,
                       std::vector<float>&                          iterations_ms,
                       const std::chrono::steady_clock::time_point& start,
                       double&                                      elapsed_gpu_secs,
                       const std::string&                           name,
                       const std::string&                           serialized_meta,
                       size_t                                       bytes_per_item,
                       double                                       estimated_remaining_secs)
    {
        const auto& s = m_settings;

        iterations++;

        run_batch(events, kernel);

        fill_iterations_ms(iterations_ms, events);

        double batch_gpu_ms = std::accumulate(iterations_ms.begin(), iterations_ms.end(), 0.0);
        m_times.emplace_back(batch_gpu_ms);

        m_logger.save(batch_gpu_ms, iterations_ms);

        // Gather batch_window_size number of iteration times.
        auto window_start = m_times.end() - std::min(iterations, s.batch_window_size);
        std::vector<double> recent_times(window_start, m_times.end());

        // Compute noise (CV) from recent times.
        double recent_mean   = get_mean(recent_times);
        double recent_stddev = get_stddev(recent_times);
        double noise_percent = get_cv(recent_times, recent_stddev, recent_mean) * 100.0;

        double batch_gpu_secs = batch_gpu_ms / 1000;

        elapsed_gpu_secs += batch_gpu_secs;

        double bytes_per_batch = m_read_write_bytes * m_kernels_per_batch;
        double bytes_per_sec   = bytes_per_batch / batch_gpu_secs;

        double items_per_batch = m_items * m_kernels_per_batch;
        double items_per_sec   = items_per_batch / batch_gpu_secs;

        auto now          = std::chrono::steady_clock::now();
        auto elapsed_host = now - start;

        // Stop early if the noise stabilized.
        bool stop_early = elapsed_host >= std::chrono::duration<double>(s.min_secs)
                          && iterations >= s.batch_window_size
                          && noise_percent < s.noise_tolerance_percent;

        // Stop early if the benchmark has been noisy for too long.
        bool noise_timeout = elapsed_host > std::chrono::duration<double>(s.noise_timeout_secs)
                             && iterations >= s.batch_window_size
                             && noise_percent >= s.noise_tolerance_percent;

        double elapsed_host_secs = std::chrono::duration<double>(elapsed_host).count();

        std::string               status;
        std::chrono::seconds::rep secs = elapsed_host_secs; // Casts to an integer.
        if(stop_early)
            status = "Success after " + std::to_string(secs) + "s";
        else if(noise_timeout)
            status = "Noisy timed out after " + std::to_string(secs) + "s";

        // Only print progress if it's a terminal (real-time updates)
        // or if the specialization has finished (single final update).
        if(use_color() || stop_early || noise_timeout)
        {
            progress::print_progress(iterations,
                                     noise_percent,
                                     bytes_per_sec,
                                     status,
                                     name,
                                     m_algo,
                                     s.batch_window_size,
                                     m_specialization_index,
                                     m_specialization_column_width,
                                     m_index_column_width,
                                     m_print_index,
                                     elapsed_host_secs,
                                     s.noise_timeout_secs,
                                     s.noise_tolerance_percent,
                                     estimated_remaining_secs);
        }

        if(stop_early || noise_timeout)
        {
            std::cout << "\n";

            m_logger.output_specialization(m_specialization_index,
                                           name,
                                           serialized_meta,
                                           m_kernels_per_batch,
                                           m_ms_per_batch,
                                           bytes_per_sec,
                                           items_per_sec,
                                           bytes_per_item,
                                           m_items,
                                           noise_percent,
                                           elapsed_host_secs,
                                           elapsed_gpu_secs,
                                           noise_timeout);

            return true;
        }

        return false;
    }

#if PRIMBENCH_HAS_MONITORING
    /// Warms up the GPU until minimum temperature is reached.
    void warm_up() const
    {
        const auto& s = m_settings;

        gpu_warmer::instance().warm_up(stream, s.min_gpu_temp, s.max_warming_secs);
    }

    /// Waits for GPU to cool down below maximum temperature.
    void cool_down() const
    {
        auto start = std::chrono::steady_clock::now();

        const auto& s = m_settings;

        while(true)
        {
            uint16_t gpu_temp = monitor::instance().get_temp();
            if(gpu_temp <= s.max_gpu_temp)
                break;

            if(use_color())
                progress::print_cooling(gpu_temp, s.max_gpu_temp);

            auto duration = std::chrono::steady_clock::now() - start;
            if(duration >= std::chrono::duration<double>(s.max_cooling_secs))
            {
                std::cerr << "\nError: Failed to cool down after " << s.max_cooling_secs
                          << " seconds\n";
                exit(EXIT_FAILURE);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
#endif

    /// Determines number of kernels per batch based on minimum duration.
    void init_kernels_per_batch(std::function<void()> kernel)
    {
        std::vector<event_t> events(2);
        std::vector<float>   iterations_ms;
        m_kernels_per_batch = 1;

        // Without this, the very first timed batch can be very slow.
        log("Running warmup");
        for(auto& event : events)
            PRIMBENCH_CHECK(event_create(&event));
        run_batch(events, kernel);
        for(const auto& event : events)
            PRIMBENCH_CHECK(event_destroy(event));

        while(true)
        {
            log("Timing batch size ", m_kernels_per_batch);

            // Reserve space for start and stop events for each iteration.
            events.resize(m_kernels_per_batch * 2);

            iterations_ms.resize(m_kernels_per_batch);

            for(auto& event : events)
                PRIMBENCH_CHECK(event_create(&event));

            run_batch(events, kernel);

            fill_iterations_ms(iterations_ms, events);

            m_ms_per_batch = std::accumulate(iterations_ms.begin(), iterations_ms.end(), 0.0);
            std::chrono::duration<double> batch_ms(m_ms_per_batch);

            for(const auto& event : events)
                PRIMBENCH_CHECK(event_destroy(event));

            if(batch_ms > std::chrono::duration<double>(m_settings.min_gpu_ms_per_batch))
                break;

            // Doubling is the simplest form of exponential growth.
            m_kernels_per_batch *= 2;
        }
    }

    /// Executes a batch of kernel iterations with event timing.
    void run_batch(const std::vector<event_t>& events, std::function<void()> kernel)
    {
        for(size_t i = 0; i < m_kernels_per_batch; i++)
        {
            if(m_run_before_every_iteration_lambda)
                m_run_before_every_iteration_lambda();

            if(!m_settings.hot)
                clear_gpu_cache(stream);

            // We block the stream to ensure the start event is recorded
            // immediately before the kernel launch.
            // Without this, event_record() might be queued much earlier, so the "start" event
            // could capture a timestamp well before the kernel actually begins executing.
            // block_stream() guarantees there is no time gap between recording
            // the start event and queuing the kernel on the GPU.
            if(!m_flags.has(flags::Flags::sync))
                m_stream_blocker.block();

            // Even events record the start time.
            PRIMBENCH_CHECK(event_record(events[i * 2], stream));

            // In order for the event timing to be accurate, this kernel lambda
            // shouldn't do more than just calling the __global__ kernel function.
            kernel();

            // Odd events record the stop time.
            PRIMBENCH_CHECK(event_record(events[i * 2 + 1], stream));

            // Allows the GPU to start running its queued events.
            if(!m_flags.has(flags::Flags::sync))
                m_stream_blocker.unblock();

            // Catches kernel launch errors.
            // We deliberately don't do this right after the kernel() call,
            // since that'd keep the GPU blocked for slightly longer.
            // The kernel lambda is still responsible for catching
            // host-side algorithm errors, using say PRIMBENCH_CHECK().
            PRIMBENCH_CHECK(get_last_error());

            // Periodically sync to avoid overflowing the stream's command queue.
            // Without this, too many pending events can exhaust driver resources and cause a hang.
            // 64 was chosen empirically. It'll sync on i=63, i=127, etc.
            constexpr size_t n = 64;
            if(i % n == n - 1)
            {
                PRIMBENCH_CHECK(stream_synchronize(stream));
                // Catch runtime/device execution errors.
                PRIMBENCH_CHECK(get_last_error());

                // Check for blocking kernel timeout after sync.
                if(!m_flags.has(flags::Flags::sync))
                    m_stream_blocker.check_timeout();
            }
        }

        // Final stream synchronization.
        PRIMBENCH_CHECK(stream_synchronize(stream));
        // Catch any runtime/device errors from the last kernels.
        PRIMBENCH_CHECK(get_last_error());

        // Final blocking kernel timeout check.
        if(!m_flags.has(flags::Flags::sync))
            m_stream_blocker.check_timeout();
    }

    /// Fills iteration times (ms) using HIP event timing.
    void fill_iterations_ms(std::vector<float>&         iterations_ms,
                            const std::vector<event_t>& events) const
    {
        for(size_t i = 0; i < m_kernels_per_batch; i++)
        {
            float iteration_ms;

            // Gets the number of milliseconds between the start and stop event.
            PRIMBENCH_CHECK(event_elapsed_time(&iteration_ms, events[i * 2], events[i * 2 + 1]));

            iterations_ms[i] = iteration_ms;
        }
    }

    /// Clears GPU caches.
    void clear_gpu_cache(stream_t stream) const
    {
        m_cache.clear_cache(stream);
    }

    /// Returns the estimated time remaining for the entire program.
    double get_estimated_remaining_secs() const
    {
        // We can only estimate once at least one specialization has finished.
        if(m_specialization_index == 0)
            return 0.0;

        auto now   = std::chrono::steady_clock::now();
        auto start = m_logger.get_program_start_time();

        auto elapsed_total = std::chrono::duration<double>(now - start).count();

        // Average time taken per completed specialization.
        double avg_time_per_spec = elapsed_total / m_specialization_index;

        // Number of specializations yet to be completed (including the current one).
        size_t remaining_specs = m_specialization_count - m_specialization_index;

        return avg_time_per_spec * remaining_specs;
    }

    /// Computes mean of time samples.
    double get_mean(const std::vector<double>& times) const
    {
        return std::reduce(times.begin(), times.end()) / times.size();
    }

    /// Computes standard deviation of time samples.
    double get_stddev(const std::vector<double>& times) const
    {
        const size_t n = times.size();
        if(n <= 1)
            return 0.0;

        double mean   = std::accumulate(times.begin(), times.end(), 0.0) / n;
        double sum_sq = 0.0;
        for(double x : times)
            sum_sq += (x - mean) * (x - mean);

        return std::sqrt(sum_sq / (n - 1));
    }

    /// Computes coefficient of variation (CV).
    double get_cv(const std::vector<double>& times, double stddev, double mean) const
    {
        return times.size() >= 2 ? stddev / mean : 0.0;
    }

    std::string m_algo;
    const json  m_meta;

    size_t m_specialization_index;
    size_t m_specialization_count;

    logger&         m_logger;
    stream_blocker& m_stream_blocker;

    const settings& m_settings;

    flags::FlagTag m_flags;

    size_t m_specialization_column_width;

    size_t m_index_column_width;
    bool   m_print_index;

    cache_thrasher& m_cache;

    std::function<void()> m_run_before_every_iteration_lambda = nullptr;
    std::vector<double>   m_times;
    size_t                m_kernels_per_batch = 0;
    double                m_ms_per_batch      = 0.0;

    bool   m_has_set_items    = false;
    bool   m_has_set_writes   = false;
    size_t m_items            = 0;
    size_t m_read_write_bytes = 0;
}; // class state

/// Simple command-line argument parser.
class cli
{
public:
    cli(int argc, char** argv) : _appname(argv[0])
    {
        // Always register --help.
        register_description("help", "Display this help and exit.");

        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if(arg.rfind("--", 0) != 0)
                continue;

            std::string key   = arg.substr(2);
            std::string value = "";

            // Consume all consecutive arguments that don't start with `--`.
            while(i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
            {
                if(!value.empty())
                    value += " ";
                value += argv[++i];
            }

            _parsed[key] = value;
        }
    }

    /// Gets argument value, registering it with default and description if needed.
    template<typename T>
    T get(std::string_view name, const T& default_val, std::string_view description)
    {
        std::string key{name};

        // Register if not already registered.
        if(_registered.find(key) == _registered.end())
        {
            register_description(key, description);
            std::ostringstream oss;

            // For bools, explicitly output "true" or "false" instead of "0" or "1".
            if constexpr(std::is_same_v<T, bool>)
            {
                oss << std::boolalpha << default_val;
            }
            // For vectors, join elements with spaces.
            else if constexpr(is_vector<T>::value)
            {
                for(size_t i = 0; i < default_val.size(); ++i)
                {
                    if(i > 0)
                        oss << " ";
                    oss << default_val[i];
                }
            }
            else
            {
                oss << default_val;
            }

            _defaults[key] = oss.str();
            _registered.insert(key);

            // For bools, ensure no value was provided on command line.
            if constexpr(std::is_same_v<T, bool>)
            {
                auto it = _parsed.find(key);
                if(it != _parsed.end() && !it->second.empty())
                {
                    std::cerr << "Error: Boolean flag --" << key << " does not take a value.\n";
                    exit(EXIT_FAILURE);
                }
            }
        }

        auto it         = _parsed.find(key);
        auto default_it = _defaults.find(key);

        // If not provided on command line, use default.
        if(it == _parsed.end())
        {
            T out{};
            if(default_it != _defaults.end() && !default_it->second.empty())
            {
                std::istringstream ss(default_it->second);

                // For bools, use boolalpha to parse "true"/"false".
                if constexpr(std::is_same_v<T, bool>)
                {
                    ss >> std::boolalpha >> out;
                    if(ss.fail())
                    {
                        std::cerr << "Error: Failed to parse default for --" << key
                                  << ": invalid value \"" << default_it->second << "\"\n";
                        exit(EXIT_FAILURE);
                    }
                }
                else if constexpr(is_vector<T>::value)
                {
                    using Elem = typename T::value_type;
                    std::string token;
                    while(ss >> token)
                    {
                        std::istringstream es(token);
                        Elem               e{};
                        es >> e;
                        if(!es)
                        {
                            std::cerr << "Error: Failed to parse default vector for --" << key
                                      << ": invalid value \"" << token << "\"\n";
                            exit(EXIT_FAILURE);
                        }
                        out.push_back(e);
                    }
                }
                else
                {
                    ss >> out;
                    if(!ss || !ss.eof())
                    {
                        std::cerr << "Error: Failed to parse default for --" << key
                                  << ": invalid value \"" << default_it->second << "\"\n";
                        exit(EXIT_FAILURE);
                    }
                }
            }
            return out;
        }

        // Parse provided value.
        if constexpr(std::is_same_v<T, bool>)
        {
            return it->second.empty() ? true : false;
        }
        else if constexpr(is_vector<T>::value)
        {
            using Elem = typename T::value_type;
            T                  out{};
            std::istringstream ss(it->second);
            std::string        token;
            while(ss >> token)
            {
                std::istringstream es(token);
                Elem               e{};
                if constexpr(std::is_same_v<Elem, std::string>)
                {
                    e = token;
                }
                else
                {
                    es >> e;
                    if(!es)
                    {
                        std::cerr << "Error: Failed to parse --" << key << ": invalid value \""
                                  << token << "\"\n";
                        exit(EXIT_FAILURE);
                    }
                }
                out.push_back(e);
            }
            return out;
        }
        else
        {
            T                  out{};
            std::istringstream ss(it->second);

            if constexpr(std::is_same_v<T, std::string>)
            {
                std::getline(ss, out);
            }
            else
            {
                ss >> out;
            }

            if(!ss)
            {
                std::cerr << "Error: Failed to parse --" << key << ": invalid value \""
                          << it->second << "\"\n";
                exit(EXIT_FAILURE);
            }

            // When "--size" is specified, the number can optionally be suffixed with KiB/MiB/GiB.
            if(key == "size")
            {
                if constexpr(std::is_same_v<T, size_t>)
                {
                    std::string suffix;
                    ss >> suffix;

                    if(!suffix.empty())
                    {
                        if(suffix == "KiB")
                            out *= KiB;
                        else if(suffix == "MiB")
                            out *= MiB;
                        else if(suffix == "GiB")
                            out *= GiB;
                        else
                        {
                            std::cerr << "Error: Failed to parse --" << key << ": unknown suffix \""
                                      << suffix << "\"\n";
                            std::exit(EXIT_FAILURE);
                        }
                    }
                }
            }

            return out;
        }
    }

    /// Returns all registered arguments with their parsed values.
    std::map<std::string, settings::custom_arg_value> get_all_custom_options() const
    {
        std::map<std::string, settings::custom_arg_value> custom_args;

        // Skip built-in arguments that are already in settings.
        static const std::unordered_set<std::string> builtin_args
            = {"help",
               "size",
               "hot",
               "seed",
               "json-out",
               "csv-out",
               "filter",
               "dry",
               "min-gpu-ms-per-batch",
               "min-secs",
               "noise-timeout-secs",
               "batch-window-size",
               "noise-tolerance-percent",
               "min-gpu-temp",
               "max-gpu-temp",
               "max-warming-secs",
               "max-cooling-secs",
               "output-batches",
               "spaces-per-indent",
               "stream-blocking-timeout-secs"};

        auto parse_value = [](const std::string& value) -> settings::custom_arg_value
        {
            if(value.empty())
            {
                return true; // Boolean flag.
            }

            // Check for boolean strings.
            if(value == "true")
            {
                return true;
            }
            if(value == "false")
            {
                return false;
            }

            // Try int.
            try
            {
                size_t idx     = 0;
                int    int_val = std::stoi(value, &idx);
                if(idx == value.size())
                {
                    return int_val;
                }
            }
            catch(...)
            {}

            // Try double.
            try
            {
                size_t idx        = 0;
                double double_val = std::stod(value, &idx);
                if(idx == value.size())
                {
                    return double_val;
                }
            }
            catch(...)
            {}

            // Fall back to string.
            return value;
        };

        // Process all registered arguments, checking both defaults and parsed values.
        std::unordered_set<std::string> processed;

        // First, add all parsed custom arguments.
        for(const auto& [key, value] : _parsed)
        {
            if(builtin_args.find(key) == builtin_args.end())
            {
                custom_args[key] = parse_value(value);
                processed.insert(key);
            }
        }

        // Then, add defaults for custom arguments that weren't parsed.
        for(const auto& [key, default_val] : _defaults)
        {
            if(builtin_args.find(key) == builtin_args.end()
               && processed.find(key) == processed.end())
            {
                custom_args[key] = parse_value(default_val);
            }
        }

        return custom_args;
    }

    /// Prints help if requested and validates all arguments are registered.
    void finalize() const
    {
        possibly_print_help();
        validate_arguments();
    }

private:
    /// Prints help message and exits if --help was requested.
    void possibly_print_help() const
    {
        if(_parsed.find("help") == _parsed.end())
            return;

        std::cout << "Usage: " << _appname << " [options]\n\n";
        for(const auto& [key, desc] : _descriptions)
        {
            std::cout << "  --" << key;

            // Print default if available.
            auto it_def = _defaults.find(key);
            if(it_def != _defaults.end() && !it_def->second.empty())
            {
                // Boolean flags are implicitly false by default.
                if(it_def->second != "false")
                {
                    std::cout << " (default: " << it_def->second << ")";
                }
            }
            std::cout << "\n";

            if(!desc.empty())
            {
                std::cout << "      " << desc << "\n";
            }
        }
        exit(EXIT_SUCCESS);
    }

    /// Validates that all parsed arguments were registered; exits if not.
    void validate_arguments() const
    {
        for(const auto& [key, value] : _parsed)
        {
            if(_registered.find(key) == _registered.end())
            {
                std::cerr << "Error: Unrecognized argument --" << key << ".\n";
                std::cerr
                    << "To register this argument, call .get() with a type and description:\n";
                std::cerr << "  executor.get<std::string>(\"" << key
                          << "\", default_value, \"Description\");\n";
                exit(EXIT_FAILURE);
            }
        }
    }

    std::string _appname; ///< Stores argv[0].
    std::unordered_map<std::string, std::string>
        _parsed; ///< Stores pairs of passed flag_name+flag_value.

    std::vector<std::pair<std::string, std::string>> _descriptions; ///< Preserves insertion order.

    std::unordered_set<std::string>
        _description_keys_set; ///< Prevents duplicate descriptions being printed.

    std::unordered_map<std::string, std::string>
        _defaults; ///< Stores string representation of default values.

    std::unordered_set<std::string> _registered; ///< Tracks which arguments were registered.

    /// Registers a description for an argument; auto-adds trailing period.
    void register_description(const std::string& key, std::string_view description)
    {
        std::string desc{description};

        // All descriptions should be sentences ending with a period.
        if(!desc.empty() && desc.back() != '.')
        {
            desc.push_back('.');
        }

        _descriptions.emplace_back(key, desc);
        _description_keys_set.insert(key);
    }

    // Helper to detect vectors.
    template<typename T>
    struct is_vector : std::false_type
    {};

    template<typename T, typename A>
    struct is_vector<std::vector<T, A>> : std::true_type
    {};
}; // class cli

} // namespace detail

using json  = detail::json;
using state = detail::state;

/// Used to retrieve the name of a type that was registered with PRIMBENCH_REGISTER_TYPE().
template<class T>
std::string name()
{
    return detail::type_name<T>::name;
}

/// This function is primarily used in benchmarks to display progress or setup messages
/// (for example, "Generating matrix of size 32x64"). It accepts any number of arguments
/// of varying types, concatenates them, and prints them as a gray line.
///
/// This logging is especially helpful for diagnosing slow setup steps.
///
/// Examples:
/// ```cpp
/// primbench::log("Loading dataset...");
/// // Output: Loading dataset...
///
/// primbench::log("Generating matrix of size ", 32, "x", 64);
/// // Output: Generating matrix of size 32x64
/// ```
template<typename... Args>
void log(Args&&... args)
{
    if(!detail::use_color())
        return;

    std::cout << detail::clearline << detail::gray;
    (std::cout << ... << args);
    std::cout << detail::reset << std::flush;
}

/// Namespace for flag definitions and utilities.
namespace flags
{

/// FlagTag representing no flags.
inline constexpr detail::flags::FlagTag none{detail::flags::Flags::none};

/// FlagTag representing the sync flag.
inline constexpr detail::flags::FlagTag sync{detail::flags::Flags::sync};

} // namespace flags

/// Interface for all benchmark specializations.
///
/// A benchmark implementation describes:
///   - the algorithm (`meta()["algo"]`),
///   - a JSON-formatted specialization identifier (other keys in `meta()`),
///   - and the code that performs the timed measurement (`run()`).
///
/// The executor uses this interface to:
///   - validate and sort benchmarks,
///   - construct per-benchmark state objects,
///   - run kernels and collect performance data,
///   - and emit structured JSON results.
struct benchmark_interface
{
    /// Returns a JSON object describing the benchmark.
    ///
    /// The returned JSON must include:
    ///   - "algo": canonical algorithm name,
    ///   - other keys describing the specialization.
    ///
    /// All benchmarks queued for one executor run must have the same "algo".
    virtual json meta() const = 0;

    /// Executes the benchmark using the provided state.
    ///
    /// Implementations allocate input/output data, perform any required setup,
    /// and launch the algorithm under test.
    virtual void run(state& state) = 0;

    /// Virtual destructor for polymorphic cleanup.
    virtual ~benchmark_interface() = default;
};

/// Generates and manages `SeedCount` seeds, given a single input seed.
template<size_t SeedCount>
class seeds
{
public:
    seeds() = delete;

    seeds(uint32_t seed)
    {
        std::seed_seq seq{seed};
        seq.generate(m_seeds.begin(), m_seeds.end());
    }

    uint32_t operator[](size_t index) const
    {
        return m_seeds[index];
    }

private:
    std::array<uint32_t, SeedCount> m_seeds;
}; // class seeds

/// Executes a suite of GPU benchmarks with configurable parameters.
///
/// The executor class handles command-line parsing, benchmark queueing,
/// execution, and logging of results in JSON format. Supports tuning
/// GPU and benchmark parameters, including batch sizes, durations, and
/// temperature limits.
class executor
{
public:
    /// Constructs the executor, and runs setup code.
    executor(int                    argc,
             char*                  argv[],
             primbench::settings    settings = {},
             detail::flags::FlagTag flags    = flags::none,
             stream_t               stream   = default_stream)
        : m_settings(settings)
        , m_flags(flags)
        , m_stream(stream)
        , m_own_stream(stream == default_stream)
        , m_cli(argc, argv)
    {
        get_logger().save_program_start_time();

        parse();

        // If user did not provide a stream, create a fast private one.
        // We can't use default_stream, as it synchronizes with the host.
        if(m_own_stream)
            PRIMBENCH_CHECK(detail::stream_create(&m_stream));

        m_stream_blocker
            = std::make_unique<detail::stream_blocker>(m_stream,
                                                       m_settings.stream_blocking_timeout_secs);
    }

    ~executor()
    {
        if(m_own_stream && m_stream != nullptr)
            PRIMBENCH_CHECK(detail::stream_destroy(m_stream));
    }

    /// \brief Queue a benchmark for execution.
    ///
    /// \tparam Benchmark Type of benchmark to queue.
    /// \tparam Args Argument types for benchmark constructor.
    /// \param args Arguments to forward to the benchmark constructor.
    ///
    /// \return true, which allows the function to be called in global scope.
    template<typename Benchmark, typename... Args>
    static bool queue(Args&&... args)
    {
        specializations.push_back(std::make_unique<Benchmark>(std::forward<Args>(args)...));
        return true;
    }

    /// \brief Queue benchmarks using an autotune bulk creation function.
    ///
    /// \tparam BulkCreateFunction Callable that populates specializations.
    /// \param fn Function that creates benchmarks.
    ///
    /// \return true, which allows the function to be called in global scope.
    template<typename BulkCreateFunction>
    static bool queue_autotune(BulkCreateFunction&& fn)
    {
        std::forward<BulkCreateFunction>(fn)(specializations);
        return true;
    }

    /// Prepares and runs all queued benchmark specializations.
    void run()
    {
        static bool run_called = false;
        if(run_called)
        {
            std::cerr << "Error: executor's run() can't be called more than once per algorithm\n";
            exit(EXIT_FAILURE);
        }
        run_called = true;

        m_cli.finalize();

        // Capture custom arguments after all have been registered.
        m_settings.custom_args = m_cli.get_all_custom_options();

        filter_specializations();
        ensure_specializations_exist();
        sort_specializations();

        std::string algorithm = get_common_algorithm();

        get_logger().init(algorithm, specializations.size(), m_settings, m_flags);

        m_specialization_column_width = compute_max_specialization_width(algorithm);

        constexpr size_t min_width = sizeof("9h 59m 59s") - 1;

        m_index_column_width = std::max(min_width,
                                        std::string("Index/").size()
                                            + std::to_string(specializations.size()).size());

        m_print_index = specializations.size() > 1;

        print_header(algorithm);

        run_all_specializations();

        get_logger().output_summary();
    }

    /// Parses a command-line argument.
    template<typename T>
    T get(std::string_view name, const T& default_val, std::string_view description)
    {
        return m_cli.get<T>(name, default_val, description);
    }

private:
    /// Parse optional arguments.
    void parse()
    {
        auto& cli = m_cli;
        auto& s   = m_settings;

        s.size = cli.get<size_t>("size",
                                 s.size,
                                 "Input size. Benchmarks decide what this represents, but it is "
                                 "commonly the number of bytes or items."
                                 " Supports the suffixes KiB/MiB/GiB, e.g. `--size 256KiB`.");
        if(s.size == 0)
        {
            std::cerr << "Error: --size must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.hot
            = cli.get<bool>("hot", s.hot, "Skip clearing the GPU cache between batch iterations.");

        s.seed = cli.get<uint32_t>("seed", s.seed, "Seed used for input generation.");

        s.json_out = cli.get<std::string>("json-out",
                                          s.json_out,
                                          "JSON path to write benchmark results to.");

        s.csv_out
            = cli.get<std::string>("csv-out", s.csv_out, "CSV path to write benchmark results to.");

        s.filter = cli.get<std::string>("filter",
                                        s.filter,
                                        "Regex filter of specialization names to benchmark.");

        s.dry = cli.get<bool>("dry",
                              s.dry,
                              "Perform a dry run. JSON and CSV files are still output, but "
                              "benchmark setup and execution are skipped.");

        s.min_gpu_ms_per_batch
            = cli.get<double>("min-gpu-ms-per-batch",
                              s.min_gpu_ms_per_batch,
                              "Minimum duration of a batch in milliseconds (GPU time).");
        if(s.min_gpu_ms_per_batch <= 0.0)
        {
            std::cerr << "Error: --min-gpu-ms-per-batch must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.min_secs = cli.get<double>("min-secs",
                                     s.min_secs,
                                     "Minimum total benchmark duration in seconds (wall time).");
        if(s.min_secs <= 0.0)
        {
            std::cerr << "Error: --min-secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.noise_timeout_secs
            = cli.get<double>("noise-timeout-secs",
                              s.noise_timeout_secs,
                              "Maximum total benchmark duration in seconds before timing out a "
                              "noisy run (wall time).");
        if(s.noise_timeout_secs <= 0.0)
        {
            std::cerr << "Error: --noise-timeout-secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }
        if(s.min_secs > s.noise_timeout_secs)
        {
            std::cerr << "Error: --min-secs must be equal to or less than --noise-timeout-secs\n";
            exit(EXIT_FAILURE);
        }

        s.batch_window_size
            = cli.get<size_t>("batch-window-size",
                              s.batch_window_size,
                              "Number of batch times used in the noise (coefficient of variation) "
                              "window to decide early benchmark stopping.");
        if(s.batch_window_size == 0)
        {
            std::cerr << "Error: --batch-window-size must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.noise_tolerance_percent
            = cli.get<double>("noise-tolerance-percent",
                              s.noise_tolerance_percent,
                              "Noise tolerance of batch times in percent, used to determine "
                              "whether a benchmark can be stopped early.");
        if(s.noise_tolerance_percent <= 0.0)
        {
            std::cerr << "Error: --noise-tolerance-percent must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.min_gpu_temp = cli.get<uint16_t>(
            "min-gpu-temp",
            s.min_gpu_temp,
            "Minimum GPU temperature in °C. Too low slows benchmarks; too high increases noise.");
        s.max_gpu_temp = cli.get<uint16_t>(
            "max-gpu-temp",
            s.max_gpu_temp,
            "Maximum GPU temperature in °C. Too low slows benchmarks; too high increases noise.");
        if(s.min_gpu_temp > s.max_gpu_temp)
        {
            std::cerr << "Error: --min-gpu-temp must be equal to or less than --max-gpu-temp\n";
            exit(EXIT_FAILURE);
        }

        s.max_warming_secs
            = cli.get<double>("max-warming-secs",
                              s.max_warming_secs,
                              "Maximum seconds allowed for GPU warming before an error is thrown.");
        if(s.max_warming_secs <= 0.0)
        {
            std::cerr << "Error: --max-warming-secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.max_cooling_secs
            = cli.get<double>("max-cooling-secs",
                              s.max_cooling_secs,
                              "Maximum seconds allowed for GPU cooling before an error is thrown.");
        if(s.max_cooling_secs <= 0.0)
        {
            std::cerr << "Error: --max-cooling-secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.output_batches = cli.get<bool>(
            "output-batches",
            s.output_batches,
            "Output a `batches` array for each specialization, containing per-batch details.");

        s.spaces_per_indent = cli.get<uint32_t>(
            "spaces-per-indent",
            s.spaces_per_indent,
            "Number of spaces per indentation level in JSON output. Set to 0 for no indentation.");
        if(s.spaces_per_indent > 8)
        {
            std::cerr << "Error: --spaces-per-indent must be less than or equal to 8\n";
            exit(EXIT_FAILURE);
        }

        s.stream_blocking_timeout_secs = cli.get<double>(
            "stream-blocking-timeout-secs",
            s.stream_blocking_timeout_secs,
            "Maximum stream blocking duration in seconds before timing out. Stream is blocked "
            "while queueing kernel calls. Use `primbench::flags::sync` if kernel is synchronous.");
        if(s.stream_blocking_timeout_secs <= 0.0)
        {
            std::cerr << "Error: --stream-blocking-timeout-secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }
    }

    /// Only keep filtered specializations, based on their name.
    void filter_specializations()
    {
        std::regex pattern(m_settings.filter);
        specializations.erase(std::remove_if(specializations.begin(),
                                             specializations.end(),
                                             [&pattern](const auto& spec) {
                                                 return !std::regex_search(
                                                     spec.get()->meta().serialize_name(),
                                                     pattern);
                                             }),
                              specializations.end());
    }

    /// Ensures that at least one specialization is queued.
    void ensure_specializations_exist()
    {
        if(specializations.size() == 0)
        {
            std::cerr << "Error: At least one benchmark must be queued\n";
            if(!m_settings.filter.empty())
            {
                std::cerr << "Hint: The currently used --filter '" << m_settings.filter
                          << "' is likely incorrect\n";
            }
            exit(EXIT_FAILURE);
        }
    }

    /// Sorts all registered specializations alphabetically by name.
    ///
    /// When specializations are added in global scope across multiple
    /// translation units, the order of initialization is unspecified.
    /// Calling this function ensures a consistent alphabetical order
    /// regardless of initialization order.
    void sort_specializations()
    {
        std::sort(specializations.begin(),
                  specializations.end(),
                  [](const auto& l, const auto& r)
                  { return l->meta().serialize_name() < r->meta().serialize_name(); });
    }

    /// Returns the algorithm name shared by all specializations.
    ///
    /// Validates that every specialization in `specializations` has
    /// the same `"algo"` value in its meta data. Exits with an error
    /// message if the `"algo"` key is missing or if algo values differ.
    std::string get_common_algorithm()
    {
        std::string algorithm;
        try
        {
            algorithm = specializations.front()->meta().get<std::string>("algo");

            for(const auto& bp : specializations)
            {
                auto bp_algo = bp->meta().get<std::string>("algo");
                if(bp_algo != algorithm)
                {
                    std::cerr
                        << "Error: All specializations must have identical `algo` names, but '"
                        << bp_algo << "' and '" << algorithm << "' are different\n";
                    exit(EXIT_FAILURE);
                }
            }
        }
        catch(const std::out_of_range& e)
        {
            std::cerr
                << "Error: All benchmarks must return an \"algo\" key from their meta() method\n";
            exit(EXIT_FAILURE);
        }

        return algorithm;
    }

    /// Computes the maximum column width of specialization names
    /// and ensures all specialization names are unique.
    size_t compute_max_specialization_width(std::string_view algorithm)
    {
        size_t                          max_width = sizeof("Specialization") - 1;
        std::unordered_set<std::string> seen_names;

        for(const auto& bp : specializations)
        {
            std::string name = bp->meta().serialize_name();
            size_t      len  = name.size();
            if(len > max_width)
                max_width = len;

            if(!seen_names.insert(name).second)
            {
                std::cerr << "Error: Algorithm '" << algorithm
                          << "' has multiple specializations with the name '" << name << "'\n";
                exit(EXIT_FAILURE);
            }
        }

        return max_width;
    }

    /// Prints a (dry) header.
    void print_header(std::string_view algorithm)
    {
        if(m_settings.dry)
        {
            detail::progress::print_dry_header(algorithm,
                                               m_specialization_column_width,
                                               m_index_column_width,
                                               specializations.size());
        }
        else
        {
            detail::progress::print_header(algorithm,
                                           m_specialization_column_width,
                                           m_index_column_width,
                                           specializations.size(),
                                           m_settings.noise_timeout_secs);
        }
    }

    /// Runs all benchmark specializations.
    void run_all_specializations()
    {
        size_t specialization_index = 0;
        for(auto& b_unique_ptr : specializations)
        {
            auto b    = b_unique_ptr.get();
            auto meta = b->meta();
            auto algo = meta.get<std::string>("algo");

            if(m_settings.dry)
            {
                output_dry_specialization(algo, meta, specialization_index);
            }
            else
            {
                auto state = new_state(algo, meta, specialization_index);
                b->run(state);
            }

            specialization_index++;
        }
    }

    /// Create a benchmark state object for execution.
    state new_state(std::string_view algo, json meta, size_t specialization_index)
    {
        return state(algo,
                     std::move(meta),
                     specialization_index,
                     specializations.size(),
                     m_stream,
                     get_logger(),
                     *m_stream_blocker,
                     m_settings,
                     m_flags,
                     m_specialization_column_width,
                     m_index_column_width,
                     m_print_index,
                     m_cache);
    }

    /// Outputs a single dry specialization.
    void output_dry_specialization(std::string_view algo,
                                   const json&      meta,
                                   size_t           specialization_index)
    {
        std::string name            = meta.serialize_name();
        std::string serialized_meta = meta.serialize();

        // A dry run doesn't run anything on the GPU, so output placeholder values.
        size_t kernels_per_batch = 0;
        double ms_per_batch      = 0.0;
        double bytes_per_sec     = 0.0;
        double items_per_sec     = 0.0;
        double bytes_per_item    = 0.0;
        size_t items             = 0;
        double noise_percent     = 0.0;
        double elapsed_host_secs = 0.0;
        double elapsed_gpu_secs  = 0.0;
        bool   noise_timeout     = false;

        detail::progress::print_dry_progress(name,
                                             algo,
                                             specialization_index,
                                             m_specialization_column_width,
                                             m_index_column_width,
                                             m_print_index);

        get_logger().output_specialization(specialization_index,
                                           name,
                                           serialized_meta,
                                           kernels_per_batch,
                                           ms_per_batch,
                                           bytes_per_sec,
                                           items_per_sec,
                                           bytes_per_item,
                                           items,
                                           noise_percent,
                                           elapsed_host_secs,
                                           elapsed_gpu_secs,
                                           noise_timeout);
    }

    /// Returns the logger singleton.
    detail::logger& get_logger()
    {
        return detail::logger::instance();
    }

    using specializations_t = std::vector<std::unique_ptr<benchmark_interface>>;

    /// This vector is static, as benchmarks can be registered in global scope.
    inline static specializations_t specializations;

    settings m_settings; ///< CLI user settings.

    detail::flags::FlagTag m_flags; ///< Executor flags.

    stream_t m_stream; ///< Stream used for execution.
    bool     m_own_stream; ///< Whether primbench should create its own stream.

    detail::cli m_cli; ///< Command-line argument parser.

    std::unique_ptr<detail::stream_blocker>
        m_stream_blocker; ///< Stream blocker to serialize output.

    size_t m_specialization_column_width; ///< Column width for specialization names.

    size_t m_index_column_width; ///< Column width for specialization index.
    bool   m_print_index; ///< Whether to print the index column.

    detail::cache_thrasher m_cache; ///< Cache clearing utility.
}; // class executor

} // namespace primbench
