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

#include <amd_smi/amdsmi.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <variant>

/**
 * \brief Default GPU cache size used for clearing caches.
 *
 * This conservative size is currently used to evict cached data before
 * kernel launches. In the future, introducing HSA as a dependency may
 * allow querying the actual largest GPU cache at runtime.
 */
#ifndef PRIMBENCH_GPU_CACHE_SIZE
    #define PRIMBENCH_GPU_CACHE_SIZE 256 * primbench::MiB
#endif

/**
 * \brief Registers a custom type name, used by `primbench::name<T>()`.
 */
#define PRIMBENCH_REGISTER_TYPE(TYPE, NAME)    \
    namespace primbench::detail                \
    {                                          \
    template<>                                 \
    struct type_name<TYPE>                     \
    {                                          \
        static inline const char* name = NAME; \
    };                                         \
    }

/**
 * \brief Exits the program with an error message if the given HIP API call returns a failure
 * status.
 */
#define PRIMBENCH_HIP_CHECK(status)                                            \
    {                                                                          \
        if(status != hipSuccess)                                               \
        {                                                                      \
            std::cerr << __FILE__ << ":" << __LINE__                           \
                      << ": HIP error: " << hipGetErrorString(status) << "\n"; \
            exit(status);                                                      \
        }                                                                      \
    }

/**
 * \brief Exits the program with an error message if the given AMD SMI API call returns a
 * failure status.
 */
#define PRIMBENCH_AMDSMI_CHECK(status)                                                        \
    {                                                                                         \
        if(status != AMDSMI_STATUS_SUCCESS)                                                   \
        {                                                                                     \
            const char* errstr = "(unknown)";                                                 \
            amdsmi_status_code_to_string(status, &errstr);                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << ": AMDSMI error: " << errstr << "\n"; \
            std::exit(status);                                                                \
        }                                                                                     \
    }

namespace primbench
{

// Forward declarations for the detail namespace.
extern const size_t MiB;
template<typename... Args>
void log(Args&&... args);

/**
 * \brief Settings that the user can change by passing arguments via their CLI.
 */
struct settings
{
    size_t      size     = 128 * primbench::MiB; /**< Input array size */
    bool        hot      = false; /**< Hot means not clearing GPU cache between batches */
    uint32_t    seed     = 42; /**< The seed to use for input array generation */
    std::string json_out = "results.json"; /**< Output JSON file path */
    std::string csv_out  = ""; /**< Output CSV file path */
    std::string filter   = ""; /**< Regex filter of specialization names to benchmark */
    bool        dry      = false; /**< Flag to perform a dry run */
    double      min_gpu_ms_per_batch = 10.0; /**< Minimum GPU batch duration */
    double      min_secs             = 1.0; /**< Minimum benchmark duration */
    double      noise_timeout_secs   = 10.0; /**< Max duration before noisy benchmark times out */
    size_t      batch_window_size    = 10; /**< Noise window size for early stopping */
    double      noise_tolerance_percent = 1.0; /**< Noise tolerance for early stopping */
    uint16_t    min_gpu_temp            = 50; /**< Minimum GPU temperature */
    uint16_t    max_gpu_temp            = 60; /**< Maximum GPU temperature */
    double      max_warming_secs        = 60.0; /**< Max GPU warmup time */
    double      max_cooling_secs        = 60.0; /**< Max GPU cooldown time */
    bool        output_hip_device_properties_context
        = false; /**< Flag to output HIP device properties context */
    bool     output_amdsmi_context = false; /**< Flag to output AMD SMI context */
    bool     output_batches        = false; /**< Flag to output batch details */
    uint32_t spaces_per_indent     = 4; /**< JSON indentation spaces */
    double   stream_blocking_timeout_secs
        = 10.0; /**< Max duration before stream blocking times out */

    using custom_arg_value = std::variant<std::string, bool, double, int, unsigned int, size_t>;
    std::map<std::string, custom_arg_value>
        custom_args; /**< Custom user-registered arguments with types */

}; // struct settings

namespace detail
{

/**
 * \brief Fallback for primbench::name().
 */
template<class T>
struct type_name
{
    static_assert(sizeof(T) == 0, "Call PRIMBENCH_REGISTER_TYPE() for this type");

    static inline const char* name = "";
};

/**
 * \brief Caches whether ANSI color output is enabled.
 *
 * The standard is described at https://bixense.com/clicolors/
 */
inline bool use_color()
{
    static const bool result = []
    {
        if(std::getenv("NO_COLOR"))
            return false;
        if(std::getenv("CLICOLOR_FORCE"))
            return true;
        return isatty(fileno(stdout)) == 1;
    }();
    return result;
}

/**
 * \brief Clears the current line if colors are enabled, otherwise prints a newline.
 */
inline std::ostream& clearline(std::ostream& os)
{
    if(use_color())
        os << "\r\033[K";
    else
        os << "\n";
    return os;
}

/**
 * \brief Resets the output text formatting to default if colors are enabled.
 */
inline std::ostream& reset(std::ostream& os)
{
    if(use_color())
        os << "\033[0m";
    return os;
}

/**
 * \brief Sets the output text color to gray if colors are enabled.
 */
inline std::ostream& gray(std::ostream& os)
{
    if(use_color())
        os << "\033[90m";
    return os;
}

/**
 * \brief Sets the output text color to green if colors are enabled.
 */
inline std::ostream& green(std::ostream& os)
{
    if(use_color())
        os << "\033[92m";
    return os;
}

/**
 * \brief Sets the output text color to red if colors are enabled.
 */
inline std::ostream& red(std::ostream& os)
{
    if(use_color())
        os << "\033[91m";
    return os;
}

/**
 * \brief Sets the output text color to yellow if colors are enabled.
 */
inline std::ostream& yellow(std::ostream& os)
{
    if(use_color())
        os << "\033[93m";
    return os;
}

/**
 * \brief Sets the output text color to blue if colors are enabled.
 */
inline std::ostream& blue(std::ostream& os)
{
    if(use_color())
        os << "\033[94m";
    return os;
}

/**
 * \brief Wrapper for AMD SMI (System Management Interface) GPU monitoring.
 *
 * Initializes AMD GPU metrics, clocks, and memory usage on construction,
 * and shuts down AMD SMI on destruction. Provides methods to:
 * - Retrieve current GPU statistics (`get_stats`)
 * - Serialize GPU statistics and device context to JSON (`serialize_stats`, `serialize_context`)
 * - Read GPU temperature (`get_temp`)
 *
 * Contains internal types for holding GPU stats and context information.
 */
class amdsmi
{
public:
    // Delete all copy/move constructors and assignment operators
    amdsmi(const amdsmi&)            = delete;
    amdsmi& operator=(const amdsmi&) = delete;
    amdsmi(amdsmi&&)                 = delete;
    amdsmi& operator=(amdsmi&&)      = delete;

    // Singleton accessor
    static amdsmi& instance()
    {
        static amdsmi instance;
        return instance;
    }

    /**
     * \brief Represents GPU statistics including metrics, clocks, and VRAM usage.
     */
    struct stats
    {
        amdsmi_gpu_metrics_t metrics;

        // Clocks (current frequencies in MHz)
        std::unordered_map<std::string, std::optional<uint32_t>> clocks;

        std::optional<uint64_t> vram_used_bytes;
    };

    /**
     * \brief Converts a stats object to a JSON string.
     * \param stats The stats object to serialize.
     * \return JSON string representing the stats.
     */
    std::string serialize_stats(const stats& stats) const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"vram_used_bytes\":" << serialize_optional(stats.vram_used_bytes);

        if(has_clock(stats.clocks))
        {
            ss << ",\"clocks_mhz\":{";
            bool first = true;
            for(const auto& kv : stats.clocks)
            {
                if(!first)
                    ss << ",";
                ss << "\"" << kv.first << "\":" << serialize_optional(kv.second);
                first = false;
            }
            ss << "}";
        }

        ss << ",\"metrics\":" << serialize_metrics(stats.metrics);

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Converts the GPU context to a JSON string.
     * \return JSON string representing the context and current GPU state.
     */
    std::string serialize_context() const
    {
        const auto& ctx = m_context;

        std::ostringstream ss;
        ss << "{";

        ss << "\"identity\":{";

        ss << "\"product_name\":" << serialize_optional(ctx.product_name);

        ss << ",\"version\":" << serialize_optional(ctx.amdsmi_version);

        ss << ",\"metrics_version\":{";
        ss << "\"format\":" << std::to_string(ctx.amdsmi_metrics_version.format_revision);
        ss << ",\"content\":" << std::to_string(ctx.amdsmi_metrics_version.content_revision);
        ss << "}";

        ss << "}"; // End of "identity" object.

        ss << ",\"power_cap\":{";
        ss << "\"current_microwatts\":" << serialize_optional(ctx.power_cap);
        ss << ",\"default_microwatts\":" << serialize_optional(ctx.power_cap_default);
        ss << ",\"dpm_mhz\":" << serialize_optional(ctx.power_cap_dpm);
        ss << "}";

        ss << ",\"vram\":{";
        ss << "\"vendor\":" << serialize_optional(ctx.vram_vendor);
        ss << ",\"total_bytes\":" << serialize_optional(ctx.vram_total_bytes);
        ss << "}";

        if(has_clock(ctx.clocks))
        {
            ss << ",\"clocks\":{";
            bool first = true;
            for(const auto& kv : ctx.clocks)
            {
                if(!first)
                    ss << ",";
                ss << "\"" << kv.first << "\":";
                if(kv.second)
                    ss << "{" << "\"min_mhz\":" << kv.second->first
                       << ",\"max_mhz\":" << kv.second->second << "}";
                else
                    ss << "null";
                first = false;
            }
            ss << "}";
        }

        ss << ",\"stats\":" << serialize_stats(ctx.stats);

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Retrieve current GPU statistics.
     * \return stats object containing current metrics, clocks, and memory usage.
     */
    stats get_stats() const
    {
        stats stats{};

        // Copy all GPU metrics
        amdsmi_gpu_metrics_t metrics{};
        if(amdsmi_get_gpu_metrics_info(m_target, &metrics) == AMDSMI_STATUS_SUCCESS)
            stats.metrics = metrics;

        // Clocks
        for(auto clk : clk_types)
        {
            amdsmi_clk_info_t clk_info{};
            if(amdsmi_get_clock_info(m_target, clk, &clk_info) == AMDSMI_STATUS_SUCCESS)
                stats.clocks[clk_type_to_string(clk)] = clk_info.clk;
        }

        // Memory usage
        uint64_t vram_used;
        if(amdsmi_get_gpu_memory_usage(m_target, AMDSMI_MEM_TYPE_VRAM, &vram_used)
           == AMDSMI_STATUS_SUCCESS)
            stats.vram_used_bytes = vram_used;

        return stats;
    }

    /**
     * \brief Converts a temperature type enum to a string.
     * \param type Temperature type enum value.
     * \return String representation of the temperature type.
     * \note Exits with failure if the temperature type is not recognized.
     */
    static const char* get_temp_type_name(amdsmi_temperature_type_t type)
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

    /**
     * \brief Determines the first available GPU temperature sensor type.
     *
     * Attempts to read temperature from multiple sensor types in priority order.
     * Returns the first type that successfully provides a temperature reading.
     *
     * \return The first successfully queried temperature sensor type.
     * \note Exits with failure if no temperature sensors are accessible.
     *       The result is cached as a static variable.
     */
    amdsmi_temperature_type_t get_temp_type() const
    {
        static const amdsmi_temperature_type_t temp_type = [&]
        {
            const amdsmi_temperature_type_t types[] = {
                AMDSMI_TEMPERATURE_TYPE_EDGE,
                AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
            };

            int64_t t;
            for(auto type : types)
            {
                if(amdsmi_get_temp_metric(m_target, type, AMDSMI_TEMP_CURRENT, &t)
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
                std::cerr << get_temp_type_name(type);
            }
            std::cerr << "\n";
            exit(EXIT_FAILURE);
        }();

        return temp_type;
    }

    /**
     * \brief Reads the GPU temperature.
     * \return Temperature in °C.
     */
    uint16_t get_temp() const
    {
        int64_t t = 0;
        PRIMBENCH_AMDSMI_CHECK(
            amdsmi_get_temp_metric(m_target, get_temp_type(), AMDSMI_TEMP_CURRENT, &t));
        return t;
    }

private:
    amdsmi()
    {
        PRIMBENCH_AMDSMI_CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS));

        // These can't be turned into a member initializer list,
        // because the above amdsmi_init() call must happen first.
        m_target  = get_target();
        m_context = get_context(m_target);
    }

    ~amdsmi()
    {
        PRIMBENCH_AMDSMI_CHECK(amdsmi_shut_down());
    }

    /**
     * \brief Holds detailed information about the GPU and AMD SMI context.
     */
    struct context
    {
        std::optional<std::string> product_name;

        std::string amdsmi_version;

        struct
        {
            uint8_t format_revision;
            uint8_t content_revision;
        } amdsmi_metrics_version;

        // Clocks (min/max in MHz)
        std::unordered_map<std::string, std::optional<std::pair<uint32_t, uint32_t>>> clocks;

        std::optional<uint64_t> power_cap;
        std::optional<uint64_t> power_cap_default;
        std::optional<uint64_t> power_cap_dpm;

        std::optional<std::string> vram_vendor;
        std::optional<uint64_t>    vram_total_bytes;

        stats stats;
    };

    /**
     * \brief Clock types queried for the GPU.
     * \note Used internally to query all supported GPU clock domains (sys, mem, soc, etc.).
     */
    const std::vector<amdsmi_clk_type_t> clk_types = {
        AMDSMI_CLK_TYPE_SYS,
        AMDSMI_CLK_TYPE_DF,
        AMDSMI_CLK_TYPE_DCEF,
        AMDSMI_CLK_TYPE_SOC,
        AMDSMI_CLK_TYPE_MEM,
        AMDSMI_CLK_TYPE_PCIE,
        AMDSMI_CLK_TYPE_VCLK0,
        AMDSMI_CLK_TYPE_VCLK1,
        AMDSMI_CLK_TYPE_DCLK0,
        AMDSMI_CLK_TYPE_DCLK1,
    };

    /**
     * \brief Finds the AMD SMI processor handle matching the current HIP device.
     * \return AMD SMI processor handle of the target GPU.
     */
    amdsmi_processor_handle get_target() const
    {
        int hip_dev;
        PRIMBENCH_HIP_CHECK(hipGetDevice(&hip_dev));

        hipDeviceProp_t hip_props;
        PRIMBENCH_HIP_CHECK(hipGetDeviceProperties(&hip_props, hip_dev));

        // Build the AMD SMI BDF struct from HIP device properties
        amdsmi_bdf_t addr{
            .function_number = 0, // HIP doesn't expose PCI function ID
            .device_number   = static_cast<uint8_t>(hip_props.pciDeviceID),
            .bus_number      = static_cast<uint8_t>(hip_props.pciBusID),
            .domain_number   = static_cast<uint16_t>(hip_props.pciDomainID),
        };

        amdsmi_processor_handle target;
        PRIMBENCH_AMDSMI_CHECK(amdsmi_get_processor_handle_from_bdf(addr, &target));

        return target;
    }

    /**
     * \brief Builds a context object with GPU details and metrics.
     * \param target AMD SMI processor handle of the target GPU.
     * \return Context object with detailed GPU information.
     */
    context get_context(amdsmi_processor_handle target) const
    {
        context ctx{};

        amdsmi_board_info_t board_info;
        if(amdsmi_get_gpu_board_info(target, &board_info) == AMDSMI_STATUS_SUCCESS)
            ctx.product_name = board_info.product_name;

        amdsmi_version_t amdsmi_version;
        if(amdsmi_get_lib_version(&amdsmi_version) == AMDSMI_STATUS_SUCCESS)
            ctx.amdsmi_version = amdsmi_version.build;

        amdsmi_gpu_metrics_t metrics{};
        if(amdsmi_get_gpu_metrics_info(target, &metrics) == AMDSMI_STATUS_SUCCESS)
        {
            ctx.amdsmi_metrics_version.format_revision  = metrics.common_header.format_revision;
            ctx.amdsmi_metrics_version.content_revision = metrics.common_header.content_revision;
        }

        for(auto clk : clk_types)
        {
            amdsmi_clk_info_t clk_info{};
            if(amdsmi_get_clock_info(target, clk, &clk_info) == AMDSMI_STATUS_SUCCESS)
                ctx.clocks[clk_type_to_string(clk)]
                    = std::make_pair(clk_info.min_clk, clk_info.max_clk);
        }

        amdsmi_power_cap_info_t pcap;
        if(amdsmi_get_power_cap_info(target, 0, &pcap) == AMDSMI_STATUS_SUCCESS)
        {
            ctx.power_cap         = pcap.power_cap;
            ctx.power_cap_default = pcap.default_power_cap;
            ctx.power_cap_dpm     = pcap.dpm_cap;
        }

        char vram_vendor_buf[128];
        if(amdsmi_get_gpu_vram_vendor(target, vram_vendor_buf, sizeof(vram_vendor_buf))
           == AMDSMI_STATUS_SUCCESS)
            ctx.vram_vendor = vram_vendor_buf;

        uint64_t vram_total;
        if(amdsmi_get_gpu_memory_total(target, AMDSMI_MEM_TYPE_VRAM, &vram_total)
           == AMDSMI_STATUS_SUCCESS)
            ctx.vram_total_bytes = vram_total;

        ctx.stats = get_stats();

        return ctx;
    }

    /**
     * \brief Converts a clock type enum to a string.
     * \param clk Clock type.
     * \return Corresponding string representation of the clock type.
     */
    std::string clk_type_to_string(amdsmi_clk_type_t clk) const
    {
        switch(clk)
        {
            case AMDSMI_CLK_TYPE_SYS: return "sys";
            case AMDSMI_CLK_TYPE_DF: return "df";
            case AMDSMI_CLK_TYPE_DCEF: return "dcef";
            case AMDSMI_CLK_TYPE_SOC: return "soc";
            case AMDSMI_CLK_TYPE_MEM: return "mem";
            case AMDSMI_CLK_TYPE_PCIE: return "pcie";
            case AMDSMI_CLK_TYPE_VCLK0: return "vclk0";
            case AMDSMI_CLK_TYPE_VCLK1: return "vclk1";
            case AMDSMI_CLK_TYPE_DCLK0: return "dclk0";
            case AMDSMI_CLK_TYPE_DCLK1: return "dclk1";
        }
        std::cerr << "Error: Failed to match clock type " << clk << " to a string\n";
        exit(EXIT_FAILURE);
    }

    /**
     * \brief Serializes an optional string to JSON format.
     * \param opt Optional string to serialize.
     * \return JSON string or "null" if empty.
     */
    std::string serialize_optional(const std::optional<std::string>& opt) const
    {
        return opt ? ("\"" + *opt + "\"") : "null";
    }

    /**
     * \brief Serializes an optional numeric value to JSON format.
     * \tparam T Numeric type.
     * \param opt Optional value to serialize.
     * \return JSON numeric string or "null" if empty.
     */
    template<typename T>
    std::string serialize_optional(const std::optional<T>& opt) const
    {
        return opt ? std::to_string(*opt) : "null";
    }

    /**
     * \brief Checks if any clock frequency is available in the provided clock map.
     * \tparam T The value type of the optional (either uint32_t or std::pair<uint32_t, uint32_t>).
     * \param clocks Map of clock names to optional frequency values.
     * \return true if at least one clock has a value, false if all are std::nullopt.
     */
    template<typename T>
    static bool has_clock(const std::unordered_map<std::string, std::optional<T>>& clocks)
    {
        return std::any_of(clocks.begin(),
                           clocks.end(),
                           [](const auto& kv) { return kv.second.has_value(); });
    }

    /**
     * \brief Serializes the raw GPU metrics to a JSON string.
     * \param metrics GPU metrics to serialize.
     * \return JSON string representing all available metrics.
     * \note Serialization includes fields conditionally depending on the metrics content revision.
     */
    std::string serialize_metrics(const amdsmi_gpu_metrics_t& metrics) const
    {
        std::ostringstream ss;
        ss << "{";

        bool first     = true;
        auto add_comma = [&]()
        {
            if(!first)
                ss << ",";
            first = false;
        };

        auto add_field = [&](const char* name, const auto& value)
        {
            add_comma();
            ss << "\"" << name << "\":" << value;
        };

        auto add_array = [&](const char* name, auto&& arr)
        {
            add_comma();
            ss << "\"" << name << "\":[";
            for(size_t i = 0; i < (sizeof(arr) / sizeof(*arr)); ++i)
            {
                if(i > 0)
                    ss << ",";
                ss << arr[i];
            }
            ss << "]";
        };

        struct revision_block
        {
            int                   min_content_revision;
            std::function<void()> serialize;
        };

        std::vector<revision_block> blocks = {
            {0,
             [&]
             {
             add_field("average_socket_power_watts", metrics.average_socket_power);
             add_field("energy_accumulator", metrics.energy_accumulator);
             add_field("system_clock_counter_ns", metrics.system_clock_counter);
             add_field("throttle_status", metrics.throttle_status);
             add_field("current_fan_speed_rpm", metrics.current_fan_speed);

             ss << ",\"average_activity_percent\":{";
             first = true;
             add_field("gfx", metrics.average_gfx_activity);
             add_field("umc", metrics.average_umc_activity);
             add_field("mm", metrics.average_mm_activity);
             ss << "}";

             ss << ",\"temp_celsius\":{";
             first = true;
             add_field("edge", metrics.temperature_edge);
             add_field("hotspot", metrics.temperature_hotspot);
             add_field("mem", metrics.temperature_mem);
             add_field("vrgfx", metrics.temperature_vrgfx);
             add_field("vrsoc", metrics.temperature_vrsoc);
             add_field("vrmem", metrics.temperature_vrmem);
             ss << "}";

             ss << ",\"average_frequency_mhz\":{";
             first = true;
             add_field("gfxclk", metrics.average_gfxclk_frequency);
             add_field("socclk", metrics.average_socclk_frequency);
             add_field("uclk", metrics.average_uclk_frequency);
             add_field("vclk0", metrics.average_vclk0_frequency);
             add_field("dclk0", metrics.average_dclk0_frequency);
             add_field("vclk1", metrics.average_vclk1_frequency);
             add_field("dclk1", metrics.average_dclk1_frequency);
             ss << "}";

             ss << ",\"current_frequency_mhz\":{";
             first = true;
             add_field("gfxclk", metrics.current_gfxclk);
             add_field("socclk", metrics.current_socclk);
             add_field("uclk", metrics.current_uclk);
             add_field("vclk0", metrics.current_vclk0);
             add_field("dclk0", metrics.current_dclk0);
             add_field("vclk1", metrics.current_vclk1);
             add_field("dclk1", metrics.current_dclk1);
             ss << "}";

             ss << ",\"pcie_link\":{";
             first = true;
             add_field("pcie_link_width", metrics.pcie_link_width);
             add_field("pcie_link_speed", metrics.pcie_link_speed);
             ss << "}";
             }                                                                      },
            {1,
             [&]
             {
             add_field("gfx_activity_acc", metrics.gfx_activity_acc);
             add_field("mem_activity_acc", metrics.mem_activity_acc);
             add_array("hbm_temp_celsius", metrics.temperature_hbm);
             }                                                                      },
            {2, [&] { add_field("firmware_timestamp", metrics.firmware_timestamp); }},
            {3,
             [&]
             {
             add_field("indep_throttle_status", metrics.indep_throttle_status);

             ss << ",\"voltage_mv\":{";
             first = true;
             add_field("soc", metrics.voltage_soc);
             add_field("gfx", metrics.voltage_gfx);
             add_field("mem", metrics.voltage_mem);
             ss << "}";
             }                                                                      },
            {4,
             [&]
             {
             add_field("current_socket_power", metrics.current_socket_power);
             add_field("gfxclk_lock_status", metrics.gfxclk_lock_status);

             ss << ",\"xgmi_link\":{";
             first = true;
             add_field("width", metrics.xgmi_link_width);
             add_field("speed", metrics.xgmi_link_speed);
             ss << "}";

             ss << ",\"pcie_bandwidth\":{";
             first = true;
             add_field("acc", metrics.pcie_bandwidth_acc);
             add_field("inst", metrics.pcie_bandwidth_inst);
             ss << "}";

             ss << ",\"pcie_count_acc\":{";
             first = true;
             add_field("l0_to_recov", metrics.pcie_l0_to_recov_count_acc);
             add_field("replay", metrics.pcie_replay_count_acc);
             add_field("replay_rover", metrics.pcie_replay_rover_count_acc);
             ss << "}";

             ss << ",\"xgmi_data_acc\":{";
             first = true;
             add_array("read", metrics.xgmi_read_data_acc);
             add_array("write", metrics.xgmi_write_data_acc);
             ss << "}";

             ss << ",\"current\":{";
             first = true;
             add_array("gfxclks", metrics.current_gfxclks);
             add_array("socclks", metrics.current_socclks);
             add_array("vclk0s", metrics.current_vclk0s);
             add_array("dclk0s", metrics.current_dclk0s);
             ss << "}";

             add_array("vcn_activity", metrics.vcn_activity);
             }                                                                      },
            {5,
             [&]
             {
             add_array("jpeg_activity_percent", metrics.jpeg_activity);

             ss << ",\"pcie_nak_count_acc\":{";
             first = true;
             add_field("sent", metrics.pcie_nak_sent_count_acc);
             add_field("rcvd", metrics.pcie_nak_rcvd_count_acc);
             ss << "}";
             }                                                                      },
            {6,
             [&]
             {
             add_field("accumulation_counter", metrics.accumulation_counter);
             add_field("num_partition", metrics.num_partition);
             add_field("pcie_lc_perf_other_end_recovery",
             metrics.pcie_lc_perf_other_end_recovery);

             ss << ",\"residency_acc\":{";
             first = true;
             add_field("prochot", metrics.prochot_residency_acc);
             add_field("ppt", metrics.ppt_residency_acc);
             add_field("socket_thm", metrics.socket_thm_residency_acc);
             add_field("vr_thm", metrics.vr_thm_residency_acc);
             add_field("hbm_thm", metrics.hbm_thm_residency_acc);
             ss << "}";

             // xcp_stats is too annoying and unimportant to serialize.
             }                                                                      },
            {7,
             [&]
             {
             add_field("vram_max_bandwidth", metrics.vram_max_bandwidth);
             add_array("xgmi_link_status", metrics.xgmi_link_status);
             }                                                                      },
        };

        int current_revision = m_context.amdsmi_metrics_version.content_revision;
        for(auto& block : blocks)
        {
            if(current_revision < block.min_content_revision)
                break;
            block.serialize();
        }

        ss << "}";
        return ss.str();
    }

    amdsmi_processor_handle m_target; /**< AMD SMI handle of the GPU. */
    context                 m_context; /**< Cached GPU context and stats. */

}; // class amdsmi

/**
 * \brief Namespace for flag definitions and utilities.
 */
namespace flags
{

/**
 * \brief Enum representing different flags.
 */
enum class Flags : uint32_t
{
    none = 0x0, /**< \brief No flags set */
    sync = 0x1, /**< \brief Synchronization flag */
};

/**
 * \brief Wrapper for Flags with utility operations.
 */
struct FlagTag
{
    Flags value{Flags::none}; /**< \brief Underlying flag value */

    /** \brief Construct from a specific flag */
    constexpr FlagTag(Flags v) : value(v) {}

    /** \brief Bitwise OR operator for combining flags */
    friend constexpr FlagTag operator|(FlagTag a, FlagTag b)
    {
        return FlagTag(
            static_cast<Flags>(static_cast<uint32_t>(a.value) | static_cast<uint32_t>(b.value)));
    }

    /** \brief Check if a flag is set */
    constexpr bool has(FlagTag f) const
    {
        return (static_cast<uint32_t>(value) & static_cast<uint32_t>(f.value)) != 0;
    }
};

} // namespace flags

/**
 * \brief Logger for saving benchmark results in JSON format.
 *
 * Handles initialization of output, storing batch data, and writing
 * specialization and device information in a structured JSON file.
 *
 * Because the logger writes partial JSON data incrementally
 * and at high volume, a JSON library is not used.
 */
class logger
{
public:
    // Delete all copy/move constructors and assignment operators
    logger(const logger&)            = delete;
    logger& operator=(const logger&) = delete;
    logger(logger&&)                 = delete;
    logger& operator=(logger&&)      = delete;

    // Singleton accessor
    static logger& instance()
    {
        static logger instance;
        return instance;
    }

    /**
     * \brief Saves the program start time.
     */
    void save_program_start_time()
    {
        m_program_start_time = std::chrono::steady_clock::now();
    }

    /**
     * \brief Initializes the logger, and opens the output JSON and CSV files.
     */
    void init(std::string_view algorithm,
              size_t           specialization_count,
              const settings&  settings,
              flags::FlagTag   flags,
              const amdsmi&    amdsmi)
    {
        m_output_batches    = settings.output_batches;
        m_spaces_per_indent = settings.spaces_per_indent;
        m_outputting_csv    = !settings.csv_out.empty();

        m_json_out.open(std::string(settings.json_out), std::ios::out | std::ios::trunc);
        if(!m_json_out)
        {
            std::cerr << "Error: Failed to open " << settings.json_out << " for writing\n";
            std::exit(EXIT_FAILURE);
        }

        if(m_outputting_csv)
        {
            m_csv_out.open(std::string(settings.csv_out), std::ios::out | std::ios::trunc);
            if(!m_csv_out)
            {
                std::cerr << "Error: Failed to open " << settings.csv_out << " for writing\n";
                std::exit(EXIT_FAILURE);
            }
        }

        m_json_out << indent(
            serialize_json_prologue(algorithm, specialization_count, settings, flags, amdsmi),
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

    /**
     * \brief Stores a batch of benchmark results.
     * \param batch_ms Total time for the batch.
     * \param iterations_ms Times for individual iterations.
     * \param amdsmi_stats AMD SMI stats after the batch.
     */
    void save(double                    batch_ms,
              const std::vector<float>& iterations_ms,
              const amdsmi::stats&      amdsmi_stats)
    {
        struct batch batch
        {};

        batch.batch_ms      = batch_ms;
        batch.iterations_ms = iterations_ms;
        batch.amdsmi_stats  = amdsmi_stats;

        m_batches.push_back(batch);
    }

    /**
     * \brief Outputs JSON and CSV specialization information.
     */
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
                               uint16_t         start_temp,
                               uint16_t         end_temp,
                               double           elapsed_host_secs,
                               double           elapsed_gpu_secs,
                               bool             noise_timeout,
                               const amdsmi&    amdsmi)
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
                                   start_temp,
                                   end_temp,
                                   elapsed_host_secs,
                                   elapsed_gpu_secs,
                                   noise_timeout,
                                   amdsmi);

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

    /**
     * \brief Outputs a summary object.
     */
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
        double             batch_ms; ///< Total time for the batch
        std::vector<float> iterations_ms; ///< Time per iteration
        amdsmi::stats      amdsmi_stats; ///< AMD SMI stats after batch
    };

    /**
     * \brief Serializes the start of the JSON file,
     * adding the `context` object, and starting the `specializations` array.
     */
    std::string serialize_json_prologue(std::string_view algorithm,
                                        size_t           specialization_count,
                                        const settings&  settings,
                                        flags::FlagTag   flags,
                                        const amdsmi&    amdsmi)
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"context\":"
           << serialize_context(algorithm, specialization_count, settings, flags, amdsmi);

        ss << ",";
        if(m_spaces_per_indent > 0)
            ss << "\n";

        ss << "\"specializations\":";

        return ss.str();
    }

    /**
     * \brief Serializes the benchmark context into JSON.
     */
    std::string serialize_context(std::string_view algorithm,
                                  size_t           specialization_count,
                                  const settings&  settings,
                                  flags::FlagTag   flags,
                                  const amdsmi&    amdsmi) const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"results_version\":\"2.0.0\"";
        ss << ",\"general\":" << serialize_general(algorithm, specialization_count, amdsmi);
        ss << ",\"settings\":" << serialize_settings(settings);

        std::string custom_cli = serialize_custom_settings(settings);
        if(!custom_cli.empty())
        {
            ss << ",\"custom_settings\":" << custom_cli;
        }

        ss << ",\"flags\":" << serialize_flags(flags);
        if(settings.output_hip_device_properties_context)
            ss << ",\"hip_device_properties\":" << serialize_hip_device_properties();
        if(settings.output_amdsmi_context)
            ss << ",\"amdsmi\":" << amdsmi.serialize_context();

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Serializes general benchmark context info into JSON.
     *
     * If the macros BRANCH_NAME and/or COMMIT_HASH are defined at compile time,
     * they are output as the JSON keys "branch_name" and "commit_hash" respectively.
     */
    std::string serialize_general(std::string_view algorithm,
                                  size_t           specialization_count,
                                  const amdsmi&    amdsmi) const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"algorithm\":\"" << algorithm << "\"";
        ss << ",\"specialization_count\":" << specialization_count;

        int device_id;
        PRIMBENCH_HIP_CHECK(hipGetDevice(&device_id));

        hipDeviceProp_t dev_prop;
        PRIMBENCH_HIP_CHECK(hipGetDeviceProperties(&dev_prop, device_id));

        ss << ",\"gpu_name\":\"" << dev_prop.name << "\"";
        ss << ",\"gpu_arch\":\"" << get_arch_name(dev_prop.gcnArchName) << "\"";

        char pci_bus_id_str[32];
        PRIMBENCH_HIP_CHECK(
            hipDeviceGetPCIBusId(pci_bus_id_str, sizeof(pci_bus_id_str), device_id));
        ss << ",\"gpu_pci_bus_id\":\"" << pci_bus_id_str << "\"";

#if defined(NDEBUG)
        const char build_type[] = "release";
#else
        const char build_type[] = "debug";
#endif
        ss << ",\"library_build_type\":\"" << build_type << "\"";

        ss << ",\"temp_type\":\"" << amdsmi.get_temp_type_name(amdsmi.get_temp_type()) << "\"";

        char host_name[HOST_NAME_MAX + 1]; // +1 for null terminator
        if(gethostname(host_name, sizeof(host_name)) != 0)
        {
            std::cerr << "Error: Failed to get host name\n";
            exit(EXIT_FAILURE);
        }
        host_name[sizeof(host_name) - 1] = '\0'; // Ensure null termination
        ss << ",\"host_name\":\"" << host_name << "\"";

        ss << ",\"date\":\"" << date() << "\"";

#ifdef BRANCH_NAME
        ss << ",\"branch_name\":\"" << BRANCH_NAME << "\"";
#endif
#ifdef COMMIT_HASH
        ss << ",\"commit_hash\":\"" << COMMIT_HASH << "\"";
#endif

        ss << ",\"hip_version\":\"" << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "."
           << HIP_VERSION_PATCH << "-" << HIP_VERSION_GITHASH << "\"";
        ss << ",\"clang_version\":\"" << __clang_version__ << "\"";

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Extracts the GPU architecture name (gfx123) from a full gcnArchName string.
     */
    std::string_view get_arch_name(std::string_view gcn_arch_name) const
    {
        // Find the position of the first ':' (if any)
        size_t pos = gcn_arch_name.find(':');
        if(pos == std::string_view::npos)
            return gcn_arch_name; // no colon, return the whole string
        return gcn_arch_name.substr(0, pos); // return only the part before ':'
    }

    /**
     * \brief Returns the local date and time as an RFC3339 string (yyyy-mm-ddTHH:MM:SS±HH:MM).
     */
    std::string date() const
    {
        using namespace std::chrono;

        // Get current time as system_clock::time_point
        auto        now   = system_clock::now();
        std::time_t now_c = system_clock::to_time_t(now);

        // Convert to local time
        std::tm local_tm{};
#if defined(_WIN32)
        localtime_s(&local_tm, &now_c);
#else
        localtime_r(&now_c, &local_tm);
#endif

        // Format date and time
        std::ostringstream oss;
        oss << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S");

        // Compute timezone offset
        std::tm utc_tm{};
#if defined(_WIN32)
        gmtime_s(&utc_tm, &now_c);
#else
        gmtime_r(&now_c, &utc_tm);
#endif

        // Offset in seconds
        int offset_sec
            = static_cast<int>(std::difftime(std::mktime(&local_tm), std::mktime(&utc_tm)));
        char sign          = offset_sec >= 0 ? '+' : '-';
        offset_sec         = std::abs(offset_sec);
        int offset_hours   = offset_sec / 3600;
        int offset_minutes = (offset_sec % 3600) / 60;

        // Append timezone
        oss << sign << std::setw(2) << std::setfill('0') << offset_hours << ':' << std::setw(2)
            << std::setfill('0') << offset_minutes;

        return oss.str();
    }

    /**
     * \brief Serializes CLI settings into JSON.
     */
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
        ss << ",\"filter\":\"" << s.filter << "\"";
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
        ss << ",\"output_hip_device_properties_context\":"
           << s.output_hip_device_properties_context;
        ss << ",\"output_amdsmi_context\":" << s.output_amdsmi_context;
        ss << ",\"output_batches\":" << s.output_batches;
        ss << ",\"spaces_per_indent\":" << s.spaces_per_indent;
        ss << ",\"stream_blocking_timeout_secs\":" << s.stream_blocking_timeout_secs;

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Serializes custom CLI settings into JSON.
     */
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

    /**
     * \brief Formats a JSON string with indentation.
     */
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
                // Detect escaped quotes
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

    /**
     * \brief Serializes benchmark flags into JSON.
     */
    std::string serialize_flags(flags::FlagTag flags) const
    {
        std::ostringstream ss;
        ss << "{";
        ss << "\"sync\":" << std::boolalpha << flags.has(flags::Flags::sync);
        ss << "}";
        return ss.str();
    }

    /**
     * \brief Returns a JSON string describing the current HIP device's properties.
     */
    std::string serialize_hip_device_properties() const
    {
        std::ostringstream ss;
        ss << "{";
        ss << std::boolalpha;

        bool first     = true;
        auto add_comma = [&]()
        {
            if(!first)
                ss << ",";
            first = false;
        };

        auto add_field = [&](std::string_view name, const auto& value)
        {
            add_comma();
            ss << "\"" << name << "\":" << value;
        };

        auto add_bool = [&](std::string_view name, const bool& value) { add_field(name, value); };

        auto add_string = [&](std::string_view name, const std::string& string)
        {
            add_comma();
            ss << "\"" << name << "\":\"" << string << "\"";
        };

        auto add_dim2 = [&](std::string_view name, auto&& arr)
        {
            add_comma();
            ss << "\"" << name << "\":{\"x\":" << arr[0] << ",\"y\":" << arr[1] << "}";
        };

        auto add_dim3 = [&](std::string_view name, auto&& arr)
        {
            add_comma();
            ss << "\"" << name << "\":{\"x\":" << arr[0] << ",\"y\":" << arr[1]
               << ",\"z\":" << arr[2] << "}";
        };

        int device_id;
        PRIMBENCH_HIP_CHECK(hipGetDevice(&device_id));

        hipDeviceProp_t dev_prop;
        PRIMBENCH_HIP_CHECK(hipGetDeviceProperties(&dev_prop, device_id));

        ss << "\"identity\":{";
        add_string("name", dev_prop.name);
        add_string("gcn_arch_name", dev_prop.gcnArchName);
        add_field("asic_revision", dev_prop.asicRevision);
        ss << "}";

        ss << ",\"clocks\":{";
        first = true;
        add_field("core_hz", dev_prop.clockRate);
        add_field("instruction_hz", dev_prop.clockInstructionRate);
        add_field("memory_hz", dev_prop.memoryClockRate);
        ss << "}";

        ss << ",\"compute\":{";

        first = true;
        add_field("mode", dev_prop.computeMode);

        ss << ",\"capability\":{";
        first = true;
        add_field("major", dev_prop.major);
        add_field("minor", dev_prop.minor);
        ss << "}";

        ss << ",\"execution\":{";
        first = true;
        add_field("max_threads_per_block", dev_prop.maxThreadsPerBlock);
        add_field("max_threads_per_multiprocessor", dev_prop.maxThreadsPerMultiProcessor);
        add_field("multi_processor_count", dev_prop.multiProcessorCount);
        add_field("regs_per_block", dev_prop.regsPerBlock);
        add_field("warp_size", dev_prop.warpSize);
        ss << "}";

        ss << ",\"limits\":{";
        first = true;
        add_dim3("grid_size", dev_prop.maxGridSize);
        add_dim3("threads_dim", dev_prop.maxThreadsDim);
        ss << "}";

        ss << "}"; // End of "compute" object.

        ss << ",\"memory\":{";

        ss << "\"global\":{";
        first = true;
        add_field("total", dev_prop.totalGlobalMem);
        add_field("pitch", dev_prop.memPitch);
        add_field("bus_width", dev_prop.memoryBusWidth);
        add_field("l2_cache_size", dev_prop.l2CacheSize);
        ss << "}";

        ss << ",\"shared\":{";
        first = true;
        add_field("per_block", dev_prop.sharedMemPerBlock);
        add_field("per_multi_processor", dev_prop.maxSharedMemoryPerMultiProcessor);
        ss << "}";

        ss << ",\"const\":{";
        first = true;
        add_field("total", dev_prop.totalConstMem);
        ss << "}";

        ss << "}"; // End of "memory" object.

        ss << ",\"pci\":{";
        first = true;
        add_field("bus_id", dev_prop.pciBusID);
        add_field("device_id", dev_prop.pciDeviceID);
        add_field("domain_id", dev_prop.pciDomainID);
        ss << "}";

        ss << ",\"texture\":{";
        first = true;
        add_field("alignment", dev_prop.textureAlignment);
        add_field("pitch_alignment", dev_prop.texturePitchAlignment);
        add_field("max_1d", dev_prop.maxTexture1D);
        add_field("max_1d_linear", dev_prop.maxTexture1DLinear);
        add_dim2("max_2d", dev_prop.maxTexture2D);
        add_dim3("max_3d", dev_prop.maxTexture3D);
        ss << "}";

        ss << ",\"capabilities\":{";

        ss << "\"architecture\":{";

        const auto arch = dev_prop.arch;

        ss << "\"atomics\":{";
        first = true;
        add_bool("float_atomic_add", arch.hasFloatAtomicAdd);
        add_bool("global_float_atomic_exch", arch.hasGlobalFloatAtomicExch);
        add_bool("global_int32_atomics", arch.hasGlobalInt32Atomics);
        add_bool("global_int64_atomics", arch.hasGlobalInt64Atomics);
        add_bool("shared_float_atomic_exch", arch.hasSharedFloatAtomicExch);
        add_bool("shared_int32_atomics", arch.hasSharedInt32Atomics);
        add_bool("shared_int64_atomics", arch.hasSharedInt64Atomics);
        ss << "}";

        ss << ",\"warp\":{";
        first = true;
        add_bool("warp_ballot", arch.hasWarpBallot);
        add_bool("warp_shuffle", arch.hasWarpShuffle);
        add_bool("warp_vote", arch.hasWarpVote);
        ss << "}";

        ss << ",\"misc\":{";
        first = true;
        add_bool("3d_grid", arch.has3dGrid);
        add_bool("doubles", arch.hasDoubles);
        add_bool("dynamic_parallelism", arch.hasDynamicParallelism);
        add_bool("funnel_shift", arch.hasFunnelShift);
        add_bool("surface_funcs", arch.hasSurfaceFuncs);
        add_bool("sync_threads_ext", arch.hasSyncThreadsExt);
        add_bool("thread_fence_system", arch.hasThreadFenceSystem);
        ss << "}";

        ss << "}"; // End of "architecture" object.

        ss << ",\"cooperative_multi_device\":{";

        ss << "\"launch\":" << static_cast<bool>(dev_prop.cooperativeMultiDeviceLaunch);

        ss << ",\"unmatched\":{";
        first = true;
        add_bool("block_dim", dev_prop.cooperativeMultiDeviceUnmatchedBlockDim);
        add_bool("func", dev_prop.cooperativeMultiDeviceUnmatchedFunc);
        add_bool("grid_dim", dev_prop.cooperativeMultiDeviceUnmatchedGridDim);
        add_bool("shared_mem", dev_prop.cooperativeMultiDeviceUnmatchedSharedMem);
        ss << "}";

        ss << "}"; // End of "cooperative_multi_device" object.

        ss << ",\"device\":{";

        ss << "\"compute\":{";
        first = true;
        add_bool("concurrent_kernels", dev_prop.concurrentKernels);
        add_bool("cooperative_launch", dev_prop.cooperativeLaunch);
        add_bool("is_large_bar", dev_prop.isLargeBar);
        add_bool("kernel_exec_timeout_enabled", dev_prop.kernelExecTimeoutEnabled);
        ss << "}";

        ss << ",\"memory\":{";
        first = true;
        add_bool("can_map_host_memory", dev_prop.canMapHostMemory);
        add_bool("concurrent_managed_access", dev_prop.concurrentManagedAccess);
        add_bool("direct_managed_mem_access_from_host", dev_prop.directManagedMemAccessFromHost);
        add_bool("managed_memory", dev_prop.managedMemory);
        add_bool("pageable_memory_access", dev_prop.pageableMemoryAccess);
        add_bool("pageable_memory_access_uses_host_page_tables",
                 dev_prop.pageableMemoryAccessUsesHostPageTables);
        ss << "}";

        ss << ",\"misc\":{";
        first = true;
        add_bool("ecc_enabled", dev_prop.ECCEnabled);
        add_bool("integrated", dev_prop.integrated);
        add_bool("is_multi_gpu_board", dev_prop.isMultiGpuBoard);
        add_bool("tcc_driver", dev_prop.tccDriver);
        ss << "}";

        ss << "}"; // End of "device" object.

        ss << "}"; // End of "capabilities" object.

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Outputs specialization information to JSON.
     */
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
                                    uint16_t         start_temp,
                                    uint16_t         end_temp,
                                    double           elapsed_host_secs,
                                    double           elapsed_gpu_secs,
                                    bool             noise_timeout,
                                    const amdsmi&    amdsmi)
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
                                                      start_temp,
                                                      end_temp,
                                                      elapsed_host_secs,
                                                      elapsed_gpu_secs,
                                                      noise_timeout,
                                                      amdsmi),
                             2);

        m_json_out.flush();
    }

    /**
     * \brief Outputs specialization information to CSV.
     */
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

    /**
     * \brief Serializes a specialization into JSON format.
     */
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
                                         uint16_t         start_temp,
                                         uint16_t         end_temp,
                                         double           elapsed_host_secs,
                                         double           elapsed_gpu_secs,
                                         bool             noise_timeout,
                                         const amdsmi&    amdsmi) const
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

        ss << ",\"gpu_temp_celsius\":{";
        ss << "\"start\":" << start_temp;
        ss << ",\"end\":" << end_temp;
        ss << "}";

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
                ss << ",\"amdsmi_stats_after_iterations\":"
                   << amdsmi.serialize_stats(batch.amdsmi_stats);
                ss << "}";
            }
            ss << "]";
        }

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Serializes iteration times into JSON array.
     */
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

    /**
     * \brief Serializes summary info into JSON format.
     */
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

    std::ofstream      m_json_out; ///< JSON output file stream
    std::ofstream      m_csv_out; ///< CSV output file stream
    bool               m_first_specialization; ///< True if first specialization output
    std::vector<batch> m_batches; ///< Stored batch results
    bool               m_output_batches; ///< Whether to output each batch
    uint32_t           m_spaces_per_indent; ///< JSON indentation spaces
    double             m_total_elapsed_gpu_secs = 0; ///< Number of elapsed GPU seconds
    uint32_t           m_noise_timeouts         = 0; ///< Number of noise timeouts
    bool               m_outputting_csv; ///< Whether a CSV file is output

}; // class logger

/**
 * \brief Provides functions for progress display and formatting during GPU benchmarking.
 */
namespace progress
{
/// \brief Column widths and formatting constants.
static constexpr int         noise_col_width         = 5;
static constexpr int         gpu_temp_col_width      = 6;
static constexpr int         bytes_per_sec_col_width = 9;
static constexpr const char* horizontal_bar          = u8"─";

/**
 * \brief Prints the table header for dry-run mode output.
 *
 * \param algo_name Algorithm name to display.
 * \param spec_col_width Width of the specialization column.
 * \param family_col_width Width of the family column.
 * \param specialization_count Total number of specializations.
 */
inline void print_dry_header(std::string_view algo_name,
                             size_t           spec_col_width,
                             size_t           family_col_width,
                             size_t           specialization_count)
{
    std::string status_header    = "Status of " + std::string(algo_name);
    size_t      status_col_width = status_header.size();

    std::cout << std::setw(status_col_width) << std::left << status_header << "  "
              << std::setw(spec_col_width) << std::left << "Specialization" << "  " << "Index/"
              << specialization_count << "\n";

    size_t underline_width = status_col_width + 2 + spec_col_width + 2 + family_col_width;

    for(size_t i = 0; i < underline_width; ++i)
        std::cout << horizontal_bar;
    std::cout << "\n";
}

/**
 * \brief Prints the table header for algorithm progress output.
 *
 * \param algo_name Algorithm name to display.
 * \param spec_col_width Width of the specialization column.
 * \param family_col_width Width of the family column.
 * \param specialization_count Total number of specializations.
 * \param noise_timeout_secs Duration before noisy timeout.
 */
inline void print_header(std::string_view          algo_name,
                         size_t                    spec_col_width,
                         size_t                    family_col_width,
                         size_t                    specialization_count,
                         std::chrono::seconds::rep noise_timeout_secs)
{
    std::string status_header = "Status of " + std::string(algo_name);
    std::string noisy_status  = "Noisy timed out after " + std::to_string(noise_timeout_secs) + "s";

    size_t status_col_width = std::max(status_header.size(), noisy_status.size());

    std::cout << std::setw(status_col_width) << std::left << status_header << "  "
              << std::setw(noise_col_width) << std::left << "Noise" << "  "
              << std::setw(gpu_temp_col_width) << std::left << "GPU °C" << "  "
              << std::setw(bytes_per_sec_col_width) << std::left << "Bytes/sec" << "  "
              << std::setw(spec_col_width) << std::left << "Specialization" << "  " << "Index/"
              << specialization_count << "\n";

    size_t underline_width = status_col_width + 2 + noise_col_width + 2 + gpu_temp_col_width + 2
                             + bytes_per_sec_col_width + 2 + spec_col_width + 2 + family_col_width;

    for(size_t i = 0; i < underline_width; ++i)
        std::cout << horizontal_bar;
    std::cout << "\n";
}

/**
 * \brief Displays GPU warming progress.
 *
 * \param gpu_temp Current GPU temperature.
 * \param min_gpu_temp Minimum temperature target.
 */
inline void print_warming(uint16_t gpu_temp, uint16_t min_gpu_temp)
{
    std::cout << clearline << "Warming GPU from " << blue << gpu_temp << "°C" << reset << " to "
              << green << min_gpu_temp << "°C" << reset << std::flush;
}

/**
 * \brief Displays GPU cooling progress.
 *
 * \param gpu_temp Current GPU temperature.
 * \param max_gpu_temp Maximum temperature target.
 */
inline void print_cooling(uint16_t gpu_temp, uint16_t max_gpu_temp)
{
    std::cout << clearline << "Cooling GPU from " << red << gpu_temp << "°C" << reset << " to "
              << green << max_gpu_temp << "°C" << reset << std::flush;
}

/**
 * \brief Prints a single line of output for dry-run mode.
 *
 * Displays only status, specialization and family index.
 * Used when simulating algorithm execution without actually running benchmarks.
 */
inline void print_dry_progress(std::string_view specialization,
                               std::string_view algo_name,
                               size_t           family_index,
                               size_t           spec_col_width,
                               size_t           family_col_width)
{
    std::ostringstream line;

    std::string status_header    = "Status of " + std::string(algo_name);
    size_t      status_col_width = status_header.size();

    line << clearline << green << std::setw(status_col_width) << std::left << "Success" << reset;

    line << "  " << std::setw(spec_col_width) << std::left << specialization;
    line << "  " << std::setw(family_col_width) << std::right << family_index;

    std::cout << line.str() << "\n" << std::flush;
}

/**
 * \brief Prints real-time progress updates for algorithm execution.
 *
 * Displays bytes per second, temperature, and specialization data.
 * Highlights noisy or timed-out iterations with color-coded output.
 */
inline void print_progress(uint64_t         iteration,
                           double           noise_percent,
                           double           bytes_per_sec,
                           std::string_view status_msg,
                           std::string_view specialization,
                           std::string_view algo_name,
                           uint64_t         batch_window_size,
                           size_t           family_index,
                           size_t           spec_col_width,
                           size_t           family_col_width,
                           double           elapsed_host_secs,
                           double           noise_timeout_secs,
                           double           noise_tolerance_percent,
                           uint16_t         gpu_temp)
{
    std::string status_header = "Status of " + std::string(algo_name);

    std::chrono::seconds::rep secs         = noise_timeout_secs; // Casts to an integer.
    std::string               noisy_status = "Noisy timed out after " + std::to_string(secs) + "s";

    size_t status_col_width = std::max(status_header.size(), noisy_status.size());

    std::string batch_str = status_msg.empty() ? "Batch " + std::to_string(iteration) + "/"
                                                     + std::to_string(batch_window_size)
                                               : std::string(status_msg);

    size_t bar_width = 0;
    if(batch_str.size() + 1 < status_col_width)
    {
        bar_width = status_col_width - batch_str.size() - 1;
    }

    std::ostringstream line;
    line << clearline << batch_str;

    // Progress bar (only shown during iteration)
    if(status_msg.empty())
    {
        line << " ";

        uint64_t capped_iteration = std::min(iteration, batch_window_size);
        size_t   filled           = (bar_width * capped_iteration) / batch_window_size;

        // Compute fraction of the yellow noise timeout overlay
        size_t yellow_chars = 0;
        if(filled >= bar_width)
        {
            double frac  = std::min(1.0, elapsed_host_secs / noise_timeout_secs);
            yellow_chars = bar_width * frac;
        }

        // Build the progress bar in one pass
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

    // Alignment for subsequent columns
    size_t used = batch_str.size() + status_msg.empty() + (status_msg.empty() ? bar_width : 0);
    if(used < status_col_width)
        line << std::string(status_col_width - used, ' ');

    // Noise %
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

    line << "  " << noise_color << std::setw(noise_col_width - sizeof('%')) << std::right
         << percent_stream.str() << "%" << reset;

    // GPU temperature
    line << "  " << std::setw(gpu_temp_col_width) << std::right << gpu_temp;

    // Bytes/sec
    line << "  " << std::setw(bytes_per_sec_col_width) << std::right << std::scientific
         << std::setprecision(2) << bytes_per_sec;

    // Specialization and index
    line << "  " << std::setw(spec_col_width) << std::left << specialization;
    line << "  " << std::setw(family_col_width) << std::right << family_index;

    // Colorized status messages
    if(status_msg.find("Success") != std::string::npos)
        std::cout << green;
    else if(status_msg.find("Noisy timed out") != std::string::npos)
        std::cout << red;

    std::cout << line.str() << std::flush;
}
} // namespace progress

/**
 * \brief Manages synchronization between host and GPU streams by blocking execution
 *        until explicitly unblocked or a timeout occurs.
 */
class stream_blocker
{
public:
    stream_blocker() = delete;

    /**
     * \brief Constructs a stream_blocker for a given HIP stream.
     */
    stream_blocker(hipStream_t stream, double stream_blocking_timeout_secs)
        : m_stream(stream), m_stream_blocking_timeout_secs(stream_blocking_timeout_secs)
    {
        // Register host memory for blocking flags
        PRIMBENCH_HIP_CHECK(
            hipHostRegister(&m_host_flag, sizeof(m_host_flag), hipHostRegisterMapped));
        PRIMBENCH_HIP_CHECK(hipHostRegister(&m_host_timeout_flag,
                                            sizeof(m_host_timeout_flag),
                                            hipHostRegisterMapped));

        // Get device pointers to mapped host memory using temporary non-volatile pointers
        int32_t* temp_device_flag         = nullptr;
        int32_t* temp_device_timeout_flag = nullptr;

        PRIMBENCH_HIP_CHECK(
            hipHostGetDevicePointer(reinterpret_cast<void**>(&temp_device_flag), &m_host_flag, 0));
        PRIMBENCH_HIP_CHECK(
            hipHostGetDevicePointer(reinterpret_cast<void**>(&temp_device_timeout_flag),
                                    &m_host_timeout_flag,
                                    0));

        // Assign to volatile members
        m_device_flag         = temp_device_flag;
        m_device_timeout_flag = temp_device_timeout_flag;

        int device_id;
        PRIMBENCH_HIP_CHECK(hipGetDevice(&device_id));

        // Query wall clock rate once (constant per device)
        int wall_clk_rate_k_hz = 0;
        PRIMBENCH_HIP_CHECK(
            hipDeviceGetAttribute(&wall_clk_rate_k_hz, hipDeviceAttributeWallClockRate, device_id));

        m_wall_clock_rate = static_cast<long long int>(wall_clk_rate_k_hz);
    }

    /**
     * \brief Destructor that unregisters host memory.
     */
    ~stream_blocker()
    {
        PRIMBENCH_HIP_CHECK(hipHostUnregister(&m_host_flag));
        PRIMBENCH_HIP_CHECK(hipHostUnregister(&m_host_timeout_flag));
    }

    /**
     * \brief Launches a blocking kernel on the stream until unblocked or timed out.
     */
    void block()
    {
        volatile int32_t& flag = m_host_flag;
        flag                   = 1;

        volatile int32_t& timeout_flag = m_host_timeout_flag;
        timeout_flag                   = 0;

        block_stream_kernel<<<dim3(1), dim3(1), 0, m_stream>>>(m_device_flag,
                                                               m_device_timeout_flag,
                                                               m_wall_clock_rate,
                                                               m_stream_blocking_timeout_secs);
    }

    /**
     * \brief Unblocks the stream by resetting the blocking flag.
     */
    void unblock()
    {
        volatile int32_t& flag = m_host_flag;
        flag                   = 0;
    }

    /**
     * \brief Checks if the GPU timed out during blocking.
     * \note Should be called after stream synchronization.
     */
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
    hipStream_t   m_stream;
    double        m_stream_blocking_timeout_secs;
    long long int m_wall_clock_rate;

    int32_t m_host_flag         = 0;
    int32_t m_host_timeout_flag = 0;

    volatile int32_t* m_device_flag         = nullptr;
    volatile int32_t* m_device_timeout_flag = nullptr;

    /**
     * \brief Kernel that blocks the GPU stream until unblocked or timeout occurs.
     * \param is_blocked Pointer to blocking flag.
     * \param timeout_flag Pointer to timeout flag.
     * \param wall_clock_rate Wall clock rate in kHz.
     * \param timeout_seconds Timeout duration in seconds.
     */
    static __global__
    void block_stream_kernel(volatile int32_t* is_blocked,
                             volatile int32_t* timeout_flag,
                             long long int     wall_clock_rate,
                             double            timeout_seconds)
    {
        const long long int start_time = wall_clock64();
        const long long int timeout_cycles
            = static_cast<long long int>(timeout_seconds * wall_clock_rate * 1000.0);

        while(*is_blocked == 1)
        {
            if(wall_clock64() - start_time > timeout_cycles)
            {
                *timeout_flag = 1; // Signal timeout to host
                break; // Exit loop
            }
        }
    }
}; // class stream_blocker

/**
 * \brief Simple JSON-like container.
 *
 * Stores key-value pairs where values can be nested JSON objects,
 * strings, integers, or doubles. Provides basic serialization to JSON and to a human-readable name.
 */
struct json
{
    using key_type = std::string;
    using value_type
        = std::variant<std::shared_ptr<json>, std::string, bool, double, int, unsigned int, size_t>;

    // Using std::map instead of std::unordered_map,
    // as we want a deterministic order.
    using map_type = std::map<key_type, value_type>;

    /**
     * \brief Adds a key-value pair to the JSON object.
     */
    template<typename T>
    json& add(std::string_view key, T value)
    {
        if constexpr(std::is_same_v<T, json>)
        {
            m_map[std::string(key)] = std::make_shared<json>(value); // Copy nested JSON.
        }
        else
        {
            m_map[std::string(key)] = value;
        }
        return *this;
    }

    /**
     * \brief Serializes the JSON object to a JSON string.
     */
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

    /**
     * \brief Serializes the JSON object into a human-readable name.
     */
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
                            // std::to_string(v) wouldn't trim trailing zeros.
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

    /**
     * \brief Retrieves a value by key.
     */
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
        {
            PRIMBENCH_HIP_CHECK(hipMalloc(&m_device_ptr, size))
        }
    }

    ~device_storage()
    {
        if(m_device_ptr != nullptr)
        {
            PRIMBENCH_HIP_CHECK(hipFree(m_device_ptr));
        }
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

/**
 * \brief Cache thrashing utility to ensure the cache contains irrelevant data.
 */
struct cache_thrasher
{
public:
    /**
     * \brief Initializes the cache thrasher by allocating the required memory on device.
     *
     * Currently, the actual largest GPU cache size cannot be queried via HIP, so this
     * conservative size is used instead. Future support via HSA could make this runtime-
     * queryable.
     */
    cache_thrasher(size_t cache_size = PRIMBENCH_GPU_CACHE_SIZE) : m_device_storage(cache_size) {}

    /**
     * \brief Clears the cache by thrashing memory on device.
     *
     * Zeros a buffer of size `m_cache_size` to evict cached data. Should be called before
     * each kernel launch.
     */
    void clear_cache(hipStream_t stream = hipStreamDefault)
    {
        PRIMBENCH_HIP_CHECK(
            hipMemsetAsync(m_device_storage.get_ptr(), 0, m_device_storage.get_size(), stream));
    }

    // This type is not copy assignable.
    cache_thrasher& operator=(const cache_thrasher&) = delete;
    // This type is not copy constructible.
    cache_thrasher(const cache_thrasher&) = delete;

private:
    const device_storage m_device_storage;
}; // struct cache_thrasher

struct gpu_warmer
{
    static constexpr int threads_per_block = 256;
    static constexpr int num_items         = 1 << 20; // 1 million items.

    gpu_warmer(settings& settings, amdsmi& amdsmi)
        : m_settings(settings), m_amdsmi(amdsmi), m_device_storage(num_items * sizeof(float))
    {}

    static __global__
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

    void warm_up(hipStream_t stream = hipStreamDefault) const
    {
        auto ceil_div = [](int a, int b) -> int { return (a + b - 1) / b; };

        auto start = std::chrono::steady_clock::now();

        const auto& s = m_settings;

        while(true)
        {
            uint16_t gpu_temp = m_amdsmi.get_temp();
            if(gpu_temp >= s.min_gpu_temp)
                break;

            progress::print_warming(gpu_temp, s.min_gpu_temp);

            dim3 threads(threads_per_block);
            dim3 blocks(ceil_div(num_items, threads.x));

            warmup_kernel<<<blocks, threads, 0, stream>>>(m_device_storage.get_ptr<float>(),
                                                          num_items);

            PRIMBENCH_HIP_CHECK(hipStreamSynchronize(stream));

            auto duration = std::chrono::steady_clock::now() - start;
            if(duration >= std::chrono::duration<double>(s.max_warming_secs))
            {
                std::cerr << "\nError: Failed to warm up after " << s.max_warming_secs
                          << " seconds\n";
                exit(EXIT_FAILURE);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // This type is not copy assignable.
    gpu_warmer& operator=(const gpu_warmer&) = delete;
    // This type is not copy constructible.
    gpu_warmer(const gpu_warmer&) = delete;

private:
    const settings&      m_settings;
    const amdsmi&        m_amdsmi;
    const device_storage m_device_storage;
}; // struct gpu_warmer

/**
 * \brief Manages benchmark execution, GPU warm-up/cool-down, timing, and logging.
 *
 * The `state` class coordinates benchmark runs by controlling GPU temperature,
 * iteration timing, throughput calculation, and result logging.
 */
class state
{
public:
    /**
     * \brief Constructs a benchmark state.
     */
    state(std::string_view algo,
          json             meta,
          size_t           family_index,
          hipStream_t      stream,
          logger&          logger,
          amdsmi&          amdsmi,
          stream_blocker&  stream_blocker,
          const settings&  settings,
          flags::FlagTag   flags,
          size_t           spec_col_width,
          size_t           family_col_width,
          cache_thrasher&  cache,
          gpu_warmer&      warmer)
        : stream(stream)
        , size(settings.size)
        , seed(settings.seed)
        , m_algo(algo)
        , m_meta(std::move(meta))
        , m_family_index(family_index)
        , m_logger(logger)
        , m_amdsmi(amdsmi)
        , m_stream_blocker(stream_blocker)
        , m_settings(settings)
        , m_flags(flags)
        , m_spec_col_width(spec_col_width)
        , m_family_col_width(family_col_width)
        , m_cache(cache)
        , m_warmer(warmer)
    {}

    /**
     * \brief Sets the total number of items processed per iteration.
     *
     * This must be called exactly once before calling \ref run() or any
     * memory tracking methods such as \ref add_reads() or \ref add_writes().
     *
     * \param items The number of items processed per iteration.
     */
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

    /**
     * \brief Adds an estimate of global memory reads performed by the benchmark.
     *
     * Must be called after \ref set_items() and before any call to
     * \ref add_writes(). Multiple calls accumulate total read bytes.
     *
     * The total number of bytes read (from all calls to this function)
     * is **summed together with the total bytes written** (added via
     * \ref add_writes()) to compute the reported memory throughput.
     *
     * \tparam T The data type of the items being read.
     * \param items The number of items read.
     */
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

    /**
     * \brief Adds an estimate of global memory writes performed by the benchmark.
     *
     * Must be called after \ref set_items(). Multiple calls accumulate total
     * written bytes.
     *
     * The total number of bytes written (from all calls to this function)
     * is **summed together with the total bytes read** (added via
     * \ref add_reads()) to compute the reported memory throughput.
     *
     * \tparam T The data type of the items being written.
     * \param items The number of items written.
     */
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

    /**
     * \brief Sets a callback to run before each iteration.
     *
     * Useful for resetting input data in in-place algorithms.
     */
    void run_before_every_iteration(std::function<void()> lambda)
    {
        m_run_before_every_iteration_lambda = lambda;
    }

    /**
     * \brief Executes the benchmark loop for the provided kernel.
     *
     * Handles warm-up, timing, CV-based stopping, and logging.
     *
     * The benchmark manages all required stream synchronization internally to
     * ensure accurate timing and prevent command queue buildup. Users should not
     * perform any manual synchronization before or during the benchmark run.
     */
    void run(std::function<void()> kernel)
    {
        if(!m_has_set_items)
        {
            std::cerr << "Error: Can't call run() before calling set_items()\n";
            exit(EXIT_FAILURE);
        }

        const auto& s = m_settings;

        std::string name            = m_meta.serialize_name();
        std::string serialized_meta = m_meta.serialize();

        size_t bytes_per_item = m_read_write_bytes / m_items;

        warm_up();
        cool_down();

        init_kernels_per_batch(kernel);

        // Reserve space for start and stop events for each iteration.
        std::vector<hipEvent_t> events(m_kernels_per_batch * 2);
        for(auto& event : events)
            PRIMBENCH_HIP_CHECK(hipEventCreate(&event));

        uint64_t           iterations = 0;
        std::vector<float> iterations_ms(m_kernels_per_batch);

        uint16_t start_temp = m_amdsmi.get_temp();

        double elapsed_gpu_secs = 0.0;

        auto start = std::chrono::steady_clock::now();

        while(true)
        {
            iterations++;

            run_batch(events, kernel);

            fill_iterations_ms(iterations_ms, events);

            double batch_gpu_ms = std::accumulate(iterations_ms.begin(), iterations_ms.end(), 0.0);
            m_times.emplace_back(batch_gpu_ms);

            amdsmi::stats amdsmi_stats = m_amdsmi.get_stats();

            m_logger.save(batch_gpu_ms, iterations_ms, amdsmi_stats);

            // Compute noise (CV) for recent window
            auto window_start = m_times.end() - std::min(iterations, s.batch_window_size);
            std::vector<double> recent_times(window_start, m_times.end());
            double              recent_mean   = get_mean(recent_times);
            double              recent_stddev = get_stddev(recent_times);
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

            uint16_t gpu_temp = m_amdsmi.get_temp();

            progress::print_progress(iterations,
                                     noise_percent,
                                     bytes_per_sec,
                                     status,
                                     name,
                                     m_algo,
                                     s.batch_window_size,
                                     m_family_index,
                                     m_spec_col_width,
                                     m_family_col_width,
                                     elapsed_host_secs,
                                     s.noise_timeout_secs,
                                     s.noise_tolerance_percent,
                                     gpu_temp);

            if(stop_early || noise_timeout)
            {
                std::cout << "\n";

                m_logger.output_specialization(m_family_index,
                                               name,
                                               serialized_meta,
                                               m_kernels_per_batch,
                                               m_ms_per_batch,
                                               bytes_per_sec,
                                               items_per_sec,
                                               bytes_per_item,
                                               m_items,
                                               noise_percent,
                                               start_temp,
                                               gpu_temp,
                                               elapsed_host_secs,
                                               elapsed_gpu_secs,
                                               noise_timeout,
                                               m_amdsmi);

                break;
            }
        }

        for(const auto& event : events)
            PRIMBENCH_HIP_CHECK(hipEventDestroy(event));
    }

    /**
     * \brief Public fields accessed directly by benchmarks.
     */
    const hipStream_t stream; ///< HIP stream used by benchmarks for kernel launches.
    const size_t      size; ///< Input size processed per iteration.
    const uint32_t    seed; ///< Random seed used for reproducible benchmark inputs.

private:
    /**
     * \brief Warms up the GPU until minimum temperature is reached.
     */
    void warm_up() const
    {
        m_warmer.warm_up(stream);
    }

    /**
     * \brief Waits for GPU to cool down below maximum temperature.
     */
    void cool_down() const
    {
        auto start = std::chrono::steady_clock::now();

        const auto& s = m_settings;

        while(true)
        {
            uint16_t gpu_temp = m_amdsmi.get_temp();
            if(gpu_temp <= s.max_gpu_temp)
                break;

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

    /**
     * \brief Determines number of kernels per batch based on minimum duration.
     */
    void init_kernels_per_batch(std::function<void()> kernel)
    {
        std::vector<hipEvent_t> events(2);
        std::vector<float>      iterations_ms;
        m_kernels_per_batch = 1;

        // Without this, the very first timed batch is very slow.
        log("Running warmup");
        for(auto& event : events)
            PRIMBENCH_HIP_CHECK(hipEventCreate(&event));
        run_batch(events, kernel);
        for(const auto& event : events)
            PRIMBENCH_HIP_CHECK(hipEventDestroy(event));

        while(true)
        {
            log("Timing batch size ", m_kernels_per_batch);

            // Reserve space for start and stop events for each iteration.
            events.resize(m_kernels_per_batch * 2);

            iterations_ms.resize(m_kernels_per_batch);

            for(auto& event : events)
                PRIMBENCH_HIP_CHECK(hipEventCreate(&event));

            run_batch(events, kernel);

            fill_iterations_ms(iterations_ms, events);

            m_ms_per_batch = std::accumulate(iterations_ms.begin(), iterations_ms.end(), 0.0);
            std::chrono::duration<double> batch_ms(m_ms_per_batch);

            for(const auto& event : events)
                PRIMBENCH_HIP_CHECK(hipEventDestroy(event));

            if(batch_ms > std::chrono::duration<double>(m_settings.min_gpu_ms_per_batch))
                break;

            // Doubling is the simplest form of exponential growth.
            m_kernels_per_batch *= 2;
        }
    }

    /**
     * \brief Executes a batch of kernel iterations with event timing.
     */
    void run_batch(const std::vector<hipEvent_t>& events, std::function<void()> kernel)
    {
        for(size_t i = 0; i < m_kernels_per_batch; i++)
        {
            if(m_run_before_every_iteration_lambda)
                m_run_before_every_iteration_lambda();

            if(!m_settings.hot)
                clear_gpu_cache(stream);

            // We block the stream to ensure the start event is recorded immediately before the
            // kernel launch. Without this, hipEventRecord() might be queued much earlier, so the
            // "start" event could capture a timestamp well before the kernel actually begins
            // executing. block_stream() guarantees there is no time gap between recording the start
            // event and queuing the kernel on the GPU.
            if(!m_flags.has(flags::Flags::sync))
                m_stream_blocker.block();

            // Even events record the start time.
            PRIMBENCH_HIP_CHECK(hipEventRecord(events[i * 2], stream));

            // In order for the event timing to be accurate, this kernel lambda
            // shouldn't do more than just calling the __global__ kernel function.
            kernel();

            // Odd events record the stop time.
            PRIMBENCH_HIP_CHECK(hipEventRecord(events[i * 2 + 1], stream));

            // Allows the GPU to start running its queued events.
            if(!m_flags.has(flags::Flags::sync))
                m_stream_blocker.unblock();

            // Catches kernel launch errors.
            // We deliberately don't do this right after the kernel() call,
            // since that'd keep the GPU blocked for slightly longer.
            // The kernel lambda is still responsible for catching
            // host-side algorithm errors using PRIMBENCH_HIP_CHECK().
            PRIMBENCH_HIP_CHECK(hipGetLastError());

            // Periodically sync to avoid overflowing the stream's command queue.
            // Without this, too many pending events can exhaust driver resources and cause a hang.
            // 64 was chosen empirically. It'll sync on i=63, i=127, etc.
            constexpr size_t n = 64;
            if(i % n == n - 1)
            {
                PRIMBENCH_HIP_CHECK(hipStreamSynchronize(stream));
                // Catch runtime/device execution errors.
                PRIMBENCH_HIP_CHECK(hipGetLastError());

                // Check for blocking kernel timeout after sync.
                if(!m_flags.has(flags::Flags::sync))
                    m_stream_blocker.check_timeout();
            }
        }

        // Final stream synchronization.
        PRIMBENCH_HIP_CHECK(hipStreamSynchronize(stream));
        // Catch any runtime/device errors from the last kernels.
        PRIMBENCH_HIP_CHECK(hipGetLastError());

        // Final blocking kernel timeout check.
        if(!m_flags.has(flags::Flags::sync))
            m_stream_blocker.check_timeout();
    }

    /**
     * \brief Fills iteration times (ms) using HIP event timing.
     */
    void fill_iterations_ms(std::vector<float>&            iterations_ms,
                            const std::vector<hipEvent_t>& events) const
    {
        for(size_t i = 0; i < m_kernels_per_batch; i++)
        {
            float iteration_ms;

            // Gets the number of milliseconds between the start and stop event.
            PRIMBENCH_HIP_CHECK(
                hipEventElapsedTime(&iteration_ms, events[i * 2], events[i * 2 + 1]));

            iterations_ms[i] = iteration_ms;
        }
    }

    /**
     * \brief Clears GPU caches.
     */
    void clear_gpu_cache(hipStream_t stream) const
    {
        m_cache.clear_cache(stream);
    }

    /**
     * \brief Computes mean of time samples.
     */
    double get_mean(const std::vector<double>& times) const
    {
        return std::reduce(times.begin(), times.end()) / times.size();
    }

    /**
     * \brief Computes standard deviation of time samples.
     */
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

    /**
     * \brief Computes coefficient of variation (CV).
     */
    double get_cv(const std::vector<double>& times, double stddev, double mean) const
    {
        return times.size() >= 2 ? stddev / mean : 0.0;
    }

    std::string m_algo;
    const json  m_meta;
    size_t      m_family_index;

    logger&         m_logger;
    amdsmi&         m_amdsmi;
    stream_blocker& m_stream_blocker;

    const settings& m_settings;

    flags::FlagTag m_flags;

    size_t m_spec_col_width;
    size_t m_family_col_width;

    cache_thrasher& m_cache;
    gpu_warmer&     m_warmer;

    std::function<void()> m_run_before_every_iteration_lambda = nullptr;
    std::vector<double>   m_times;
    size_t                m_kernels_per_batch = 0;
    double                m_ms_per_batch      = 0.0;

    bool   m_has_set_items    = false;
    bool   m_has_set_writes   = false;
    size_t m_items            = 0;
    size_t m_read_write_bytes = 0;
}; // class state

/**
 * \brief Simple command-line argument parser.
 */
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
            // Only consume next argument if it's a value (doesn't start with --)
            if(i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
            {
                value = argv[++i];
            }
            _parsed[key] = value;
        }
    }

    /// \brief Gets argument value, registering it with default and description if needed.
    template<typename T>
    T get(std::string_view name, const T& default_val, std::string_view description)
    {
        std::string key{name};

        // Register if not already registered
        if(_registered.find(key) == _registered.end())
        {
            register_description(key, description);
            std::ostringstream oss;

            // For bools, explicitly output "true" or "false" instead of "0" or "1"
            if constexpr(std::is_same_v<T, bool>)
            {
                oss << std::boolalpha;
            }
            oss << default_val;

            _defaults[key] = oss.str();
            _registered.insert(key);

            // For bools, ensure no value was provided on command line.
            if constexpr(std::is_same_v<T, bool>)
            {
                auto it = _parsed.find(key);
                if(it != _parsed.end() && !it->second.empty())
                {
                    std::cerr << "Error: Boolean flag --" << key << " does not take a value.\n";
                    std::exit(EXIT_FAILURE);
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

                // For bools, use boolalpha to parse "true"/"false"
                if constexpr(std::is_same_v<T, bool>)
                {
                    ss >> std::boolalpha >> out;
                    if(ss.fail())
                    {
                        std::cerr << "Error: Failed to parse default for --" << key
                                  << ": invalid value \"" << default_it->second << "\"\n";
                        std::exit(EXIT_FAILURE);
                    }
                }
                else
                {
                    ss >> out;
                    if(!ss || !ss.eof())
                    {
                        std::cerr << "Error: Failed to parse default for --" << key
                                  << ": invalid value \"" << default_it->second << "\"\n";
                        std::exit(EXIT_FAILURE);
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
                std::exit(EXIT_FAILURE);
            }
            return out;
        }
    }

    /// \brief Returns all registered arguments with their parsed values.
    std::map<std::string, settings::custom_arg_value> get_all_custom_options() const
    {
        std::map<std::string, settings::custom_arg_value> custom_args;

        // Skip built-in arguments that are already in settings
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
               "output-hip-device-properties-context",
               "output-amdsmi-context",
               "output-batches",
               "spaces-per-indent",
               "stream-blocking-timeout-secs"};

        auto parse_value = [](const std::string& value) -> settings::custom_arg_value
        {
            if(value.empty())
            {
                return true; // Boolean flag
            }

            // Check for boolean strings
            if(value == "true")
            {
                return true;
            }
            if(value == "false")
            {
                return false;
            }

            // Try int
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

            // Try double
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

            // Fall back to string
            return value;
        };

        // Process all registered arguments, checking both defaults and parsed values
        std::unordered_set<std::string> processed;

        // First, add all parsed custom arguments
        for(const auto& [key, value] : _parsed)
        {
            if(builtin_args.find(key) == builtin_args.end())
            {
                custom_args[key] = parse_value(value);
                processed.insert(key);
            }
        }

        // Then, add defaults for custom arguments that weren't parsed
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

    /// \brief Prints help if requested and validates all arguments are registered.
    void finalize() const
    {
        possibly_print_help();
        validate_arguments();
    }

private:
    /// \brief Prints help message and exits if --help was requested.
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
                std::cout << " (default: " << it_def->second << ")";
            }
            std::cout << "\n";

            if(!desc.empty())
            {
                std::cout << "      " << desc << "\n";
            }
        }
        std::exit(EXIT_SUCCESS);
    }

    /// \brief Validates that all parsed arguments were registered; exits if not.
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
                std::exit(EXIT_FAILURE);
            }
        }
    }

    std::string                                  _appname;
    std::unordered_map<std::string, std::string> _parsed;

    // Preserves insertion order.
    std::vector<std::pair<std::string, std::string>> _descriptions;

    // Prevents duplicate descriptions being printed.
    std::unordered_set<std::string> _description_keys_set;

    // Store string representation of default values.
    std::unordered_map<std::string, std::string> _defaults;

    // Track which arguments were registered with set().
    std::unordered_set<std::string> _registered;

    /// \brief Registers a description for an argument; auto-adds trailing period.
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
}; // class cli

} // namespace detail

inline constexpr size_t KiB = 1024;
inline constexpr size_t MiB = 1024 * KiB;
inline constexpr size_t GiB = 1024 * MiB;

using json  = detail::json;
using state = detail::state;

/**
 * \brief Used to retrieve the name of a type
 * that was registered with PRIMBENCH_REGISTER_TYPE().
 */
template<class T>
std::string name()
{
    return detail::type_name<T>::name;
}

/**
 * \brief Logs a gray line of text to stdout, overwriting the previous line.
 *
 * This function is primarily used in benchmarks to display progress or setup messages
 * (for example, "Generating matrix of size 32x64"). It accepts any number of arguments
 * of varying types, concatenates them, and prints them in gray text to the console.
 *
 * This logging is especially helpful for diagnosing **slow setup steps** and
 * **slow computers**.
 *
 * ### Examples
 * ```cpp
 * primbench::log("Loading dataset...");
 * // Output: Loading dataset...
 *
 * primbench::log("Generating matrix of size ", 32, "x", 64);
 * // Output: Generating matrix of size 32x64
 * ```
 */
template<typename... Args>
void log(Args&&... args)
{
    std::cout << detail::clearline << detail::gray;
    (std::cout << ... << args);
    std::cout << detail::reset << std::flush;
}

/**
 * \brief Namespace for flag definitions and utilities.
 */
namespace flags
{

/** \brief FlagTag representing no flags */
inline constexpr detail::flags::FlagTag none{detail::flags::Flags::none};

/** \brief FlagTag representing the sync flag */
inline constexpr detail::flags::FlagTag sync{detail::flags::Flags::sync};

} // namespace flags

/**
 * \brief Base interface for all benchmark specializations.
 *
 * A benchmark implementation describes:
 *   - the algorithm (`meta()["algo"]`),
 *   - a JSON-formatted specialization identifier (other keys in `meta()`),
 *   - and the code that performs the timed measurement (`run()`).
 *
 * The executor uses this interface to:
 *   - validate and sort benchmarks,
 *   - construct per-benchmark state objects,
 *   - run kernels and collect performance data,
 *   - and emit structured JSON results.
 */
struct benchmark_interface
{
    /**
     * \brief Returns a JSON object describing the benchmark.
     *
     * The returned JSON must include:
     *   - "algo": canonical algorithm name,
     *   - other keys describing the specialization.
     *
     * All benchmarks queued for one executor run must have the same "algo".
     */
    virtual json meta() const = 0;

    /**
     * \brief Executes the benchmark using the provided state.
     *
     * Implementations allocate input/output data, perform any required setup,
     * and launch the algorithm under test. Timing, iteration control, and
     * result reporting are handled through the supplied `state` object.
     */
    virtual void run(state& state) = 0;

    /// Virtual destructor for polymorphic cleanup.
    virtual ~benchmark_interface() = default;
};

/**
 * \brief Generates and manages `SeedCount` seeds, from a single seed.
 */
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

/**
 * \brief Executes a suite of GPU benchmarks with configurable parameters.
 *
 * The executor class handles command-line parsing, benchmark queueing,
 * execution, and logging of results in JSON format. Supports tuning
 * GPU and benchmark parameters, including batch sizes, durations, and
 * temperature limits.
 */
class executor
{
public:
    /**
     * \brief Constructs the executor and initializes parsing and logging.
     * \param argc Argument count from main().
     * \param argv Argument values from main().
     * \param settings Optional benchmark-specific settings.
     * \param flags Optional flags controlling executor behavior.
     * \param stream Optional HIP stream to run the benchmarks on.
     */
    executor(int                    argc,
             char*                  argv[],
             primbench::settings    settings = {},
             detail::flags::FlagTag flags    = flags::none,
             hipStream_t            stream   = hipStreamDefault)
        : m_settings(settings)
        , m_flags(flags)
        , m_stream(stream)
        , m_own_stream(stream == hipStreamDefault)
        , m_cli(argc, argv)
    {
        get_logger().save_program_start_time();

        parse();

        // If user did not provide a stream, create a fast private one.
        // We can't use hipStreamDefault, as it synchronizes with the host.
        if(m_own_stream)
            PRIMBENCH_HIP_CHECK(hipStreamCreate(&m_stream));

        m_stream_blocker
            = std::make_unique<detail::stream_blocker>(m_stream,
                                                       m_settings.stream_blocking_timeout_secs);
    }

    ~executor()
    {
        if(m_own_stream && m_stream != nullptr)
            PRIMBENCH_HIP_CHECK(hipStreamDestroy(m_stream));
    }

    /**
     * \brief Queue a benchmark for execution.
     * \tparam Benchmark Type of benchmark to queue.
     * \tparam Args Argument types for benchmark constructor.
     * \param args Arguments to forward to the benchmark constructor.
     * \return true to allow usage in global static initialization.
     */
    template<typename Benchmark, typename... Args>
    static bool queue(Args&&... args)
    {
        static_specializations.push_back(std::make_unique<Benchmark>(std::forward<Args>(args)...));
        return true;
    }

    /**
     * \brief Queue benchmarks using an autotune bulk creation function.
     * \tparam BulkCreateFunction Callable that populates static_specializations.
     * \param fn Function that creates benchmarks.
     * \return true to allow usage in global static initialization.
     */
    template<typename BulkCreateFunction>
    static bool queue_autotune(BulkCreateFunction&& fn)
    {
        std::forward<BulkCreateFunction>(fn)(static_specializations);
        return true;
    }

    /**
     * \brief Run all queued benchmarks and print progress/results.
     *
     * Performs the following:
     * - Ensures run() is called only once per algorithm executor.
     * - Sorts benchmarks to achieve a consistent order.
     * - Validates that all benchmarks have the same algorithm name (`algo`).
     * - Verifies that at least one benchmark is queued.
     * - Ensures that all human-readable specialization names (`name`) are unique.
     * - Computes output column widths and prints the benchmark header.
     * - Executes each benchmark and prints progress/results.
     * - Outputs a summary after all benchmarks are complete.
     *
     * \throws Exits the program with EXIT_FAILURE on validation errors.
     */
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

        // Only keep filtered specializations, based on their name.
        std::regex pattern(m_settings.filter);
        static_specializations.erase(std::remove_if(static_specializations.begin(),
                                                    static_specializations.end(),
                                                    [&pattern](const auto& spec) {
                                                        return !std::regex_search(
                                                            spec.get()->meta().serialize_name(),
                                                            pattern);
                                                    }),
                                     static_specializations.end());

        // Sort to get a consistent order.
        std::sort(static_specializations.begin(),
                  static_specializations.end(),
                  [](const auto& l, const auto& r)
                  { return l->meta().serialize_name() < r->meta().serialize_name(); });

        size_t specialization_count = static_specializations.size();

        if(specialization_count == 0)
        {
            std::cerr << "Error: At least one benchmark must be queued\n";
            if(!m_settings.filter.empty())
            {
                std::cerr << "Hint: The currently used --filter '" << m_settings.filter
                          << "' is likely incorrect\n";
            }
            exit(EXIT_FAILURE);
        }

        std::string algorithm;
        try
        {
            algorithm = static_specializations.front()->meta().get<std::string>("algo");

            // Validate that benchmarks have identical algo names.
            for(const auto& bp : static_specializations)
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

        get_logger().init(algorithm, specialization_count, m_settings, m_flags, get_amdsmi());

        // Determine max specialization width, and validate that every name is unique.
        m_spec_col_width = 0;
        std::unordered_set<std::string> seen_names;

        for(const auto& bp : static_specializations)
        {
            std::string name = bp->meta().serialize_name();
            size_t      len  = name.size();
            if(len > m_spec_col_width)
                m_spec_col_width = len;

            if(!seen_names.insert(name).second)
            {
                std::cerr << "Error: Algorithm '" << algorithm
                          << "' has multiple specializations with the name '" << name << "'\n";
                exit(EXIT_FAILURE);
            }
        }

        m_family_col_width
            = std::string("Index/").size() + std::to_string(specialization_count).size();

        if(m_settings.dry)
        {
            detail::progress::print_dry_header(algorithm,
                                               m_spec_col_width,
                                               m_family_col_width,
                                               specialization_count);
        }
        else
        {
            detail::progress::print_header(algorithm,
                                           m_spec_col_width,
                                           m_family_col_width,
                                           specialization_count,
                                           m_settings.noise_timeout_secs);
        }

        // Run all benchmarks.
        size_t family_index = 0;
        for(auto& b_unique_ptr : static_specializations)
        {
            auto b    = b_unique_ptr.get();
            auto meta = b->meta();
            auto algo = meta.get<std::string>("algo");

            if(m_settings.dry)
            {
                output_dry_specialization(algo, meta, family_index);
            }
            else
            {
                auto state = new_state(algo, meta, family_index);
                b->run(state);
            }

            family_index++;
        }

        get_logger().output_summary();
    }

    /**
     * \brief Parses a command-line argument.
     */
    template<typename T>
    T get(std::string_view name, const T& default_val, std::string_view description)
    {
        return m_cli.get<T>(name, default_val, description);
    }

private:
    /**
     * \brief Parse optional arguments.
     */
    void parse()
    {
        auto& cli = m_cli;
        auto& s   = m_settings;

        s.size = cli.get<size_t>("size",
                                 s.size,
                                 "Input size. Benchmarks decide what this represents, but it is "
                                 "commonly the number of bytes or items.");
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

        s.output_hip_device_properties_context
            = cli.get<bool>("output-hip-device-properties-context",
                            s.output_hip_device_properties_context,
                            "Output a `hip_device_properties` object in the context object, "
                            "containing details about the GPU.");

        s.output_amdsmi_context = cli.get<bool>(
            "output-amdsmi-context",
            s.output_amdsmi_context,
            "Output an `amdsmi` object in the context object, containing details about the GPU.");

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

    /**
     * \brief Create a benchmark state object for execution.
     * \param algo Algorithm name.
     * \param meta Specialization metadata.
     * \param family_index Index of benchmark in family.
     * \return Configured state object for the benchmark.
     */
    state new_state(std::string_view algo, json meta, size_t family_index)
    {
        return state(algo,
                     std::move(meta),
                     family_index,
                     m_stream,
                     get_logger(),
                     get_amdsmi(),
                     *m_stream_blocker,
                     m_settings,
                     m_flags,
                     m_spec_col_width,
                     m_family_col_width,
                     m_cache,
                     m_warmer);
    }

    /**
     * \brief Outputs a single dry specialization.
     */
    void output_dry_specialization(std::string_view algo, const json& meta, size_t family_index)
    {
        std::string name            = meta.serialize_name();
        std::string serialized_meta = meta.serialize();

        // A dry run doesn't run anything on the GPU, so output placeholder values.
        size_t   kernels_per_batch = 0;
        double   ms_per_batch      = 0.0;
        double   bytes_per_sec     = 0.0;
        double   items_per_sec     = 0.0;
        double   bytes_per_item    = 0.0;
        size_t   items             = 0;
        double   noise_percent     = 0.0;
        uint16_t start_temp        = 0;
        uint16_t end_temp          = 0;
        double   elapsed_host_secs = 0.0;
        double   elapsed_gpu_secs  = 0.0;
        bool     noise_timeout     = false;

        detail::progress::print_dry_progress(name,
                                             algo,
                                             family_index,
                                             m_spec_col_width,
                                             m_family_col_width);

        get_logger().output_specialization(family_index,
                                           name,
                                           serialized_meta,
                                           kernels_per_batch,
                                           ms_per_batch,
                                           bytes_per_sec,
                                           items_per_sec,
                                           bytes_per_item,
                                           items,
                                           noise_percent,
                                           start_temp,
                                           end_temp,
                                           elapsed_host_secs,
                                           elapsed_gpu_secs,
                                           noise_timeout,
                                           get_amdsmi());
    }

    /**
     * \brief Returns the logger singleton.
     */
    detail::logger& get_logger()
    {
        return detail::logger::instance();
    }

    /**
     * \brief Returns the amdsmi singleton.
     */
    detail::amdsmi& get_amdsmi()
    {
        return detail::amdsmi::instance();
    }

    /**
     * This vector is static, allowing queue_autotune()
     * to register all specializations of autotuned benchmarks in one place.
     */
    inline static std::vector<std::unique_ptr<benchmark_interface>> static_specializations;

    settings m_settings; /**< CLI user settings */

    detail::flags::FlagTag m_flags; /**< Executor flags */

    hipStream_t m_stream; /**< HIP stream used for execution */
    bool        m_own_stream; /** Whether primbench should create its own stream */

    detail::cli m_cli; /**< Command-line argument parser */

    std::unique_ptr<detail::stream_blocker>
        m_stream_blocker; /**< Stream blocker to serialize output */

    size_t m_spec_col_width; /**< Column width for specialization names */
    size_t m_family_col_width; /**< Column width for family index */

    detail::cache_thrasher m_cache = detail::cache_thrasher(); /**< Cache clearing utility */
    detail::gpu_warmer     m_warmer
        = detail::gpu_warmer(m_settings, get_amdsmi()); /**< GPU warm-up utility */
}; // class executor

} // namespace primbench
