// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP
#define GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <miopen/env.hpp>
#include <miopen/utility/modified_z.hpp>

// Declare the performance logs environment variable
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_PERFORMANCE_LOGS)

namespace miopen {

/// Kernel execution phase enumeration
/// Determines the context in which kernels are executed for logging purposes
enum class KernelPhase
{
    Unknown,      // Default/unset phase
    Execution,    // Normal kernel execution
    Validation,   // Validation phase
    SolverTuning, // Solver tuning/search phase
    Tuning
};

/// Convert KernelPhase enum to string for JSON output
inline const char* KernelPhaseToString(KernelPhase phase)
{
    switch(phase)
    {
    case KernelPhase::Unknown: return "unknown";
    case KernelPhase::Execution: return "execution";
    case KernelPhase::Validation: return "validation";
    case KernelPhase::SolverTuning: return "solver_tuning";
    case KernelPhase::Tuning: return "tuning";
    default: return "unknown";
    }
}

/// Thread-local storage for current kernel execution phase
/// Default phase is Unknown
inline KernelPhase& GetKernelPhase()
{
    thread_local KernelPhase current_phase = KernelPhase::Unknown;
    return current_phase;
}

/// Set the current kernel phase
inline void SetKernelPhase(KernelPhase phase) { GetKernelPhase() = phase; }

/// Get the last printed solution name
inline std::string& GetLastPrintedSolutionName()
{
    thread_local std::string last_solution;
    return last_solution;
}

/// Get the last printed solver ID
inline uint64_t& GetLastPrintedSolverId()
{
    thread_local uint64_t last_solver_id = 0;
    return last_solver_id;
}

/// Check if a kernel name indicates it's a transpose or transformation kernel
inline bool IsTransposeOrTransformKernel(const std::string& kernel_name)
{
    // Check for common transpose/transformation patterns
    return kernel_name.find("transpose") != std::string::npos ||
           kernel_name.find("Transpose") != std::string::npos ||
           kernel_name.find("transform") != std::string::npos ||
           kernel_name.find("Transform") != std::string::npos ||
           kernel_name.find("SubTensor") != std::string::npos ||
           kernel_name.find("reorder") != std::string::npos ||
           kernel_name.find("Reorder") != std::string::npos ||
           // Im2Col and Col2Im transformations (2D and 3D)
           kernel_name.find("Im2d2Col") != std::string::npos ||
           kernel_name.find("Im3d2Col") != std::string::npos ||
           kernel_name.find("Col2Im2d") != std::string::npos ||
           kernel_name.find("Col2Im3d") != std::string::npos ||
           kernel_name.find("Im2Col") != std::string::npos ||
           kernel_name.find("Col2Im") != std::string::npos ||
           // NCHW to CNHW and vice versa transformations
           kernel_name.find("NCHW2CNHW") != std::string::npos ||
           kernel_name.find("CNHW2NCHW") != std::string::npos ||
           kernel_name.find("NCHW2Vec") != std::string::npos ||
           kernel_name.find("NCHWVec") != std::string::npos ||
           // Packed matrix transpose
           kernel_name.find("MN2NM") != std::string::npos;
}

/// Structure to hold kernel execution data for JSON output (single execution)
struct KernelExecutionData
{
    std::string kernel_name;
    float time_ms;
    bool is_transformation;
};

/// Structure to hold performance config data with its kernels
struct PerformanceConfigData
{
    std::string config_name;       // Kernel name (if available) or solution name
    std::string config_descriptor; // Performance config parameters string
    std::vector<KernelExecutionData> kernels;
    std::vector<float> invoker_times_ms; // Direct timing from invoker (if available)

    void Clear()
    {
        config_name.clear();
        config_descriptor.clear();
        kernels.clear();
        invoker_times_ms.clear();
    }
};

/// Structure to hold grouped kernel data by kernel_name (multiple executions)
struct GroupedKernelData
{
    std::string kernel_name;
    std::vector<float> time_executions_ms;
    bool is_transformation;
};

/// Structure to hold solution data for JSON output
struct SolutionExecutionData
{
    SolutionExecutionData() = default;

    std::string solution_name;
    uint64_t solver_id = 0;
    std::string phase;
    size_t workspace_bytes = 0;                             // Workspace size for this solution
    std::vector<PerformanceConfigData> performance_configs; // Array of performance configs

    // Current config being accumulated (temporary)
    PerformanceConfigData current_config;

    void Clear()
    {
        solution_name.clear();
        solver_id = 0;
        phase.clear();
        workspace_bytes = 0;
        performance_configs.clear();
        current_config.Clear();
    }

    void FinalizeCurrentConfig()
    {
        // Finalize if we have a config name and either kernels OR invoker times
        if(!current_config.config_name.empty())
        {
            performance_configs.push_back(current_config);
            current_config.Clear();
        }
    }
};

/// Escape string for JSON
inline std::string JsonEscape(const std::string& str)
{
    std::ostringstream oss;
    for(char c : str)
    {
        switch(c)
        {
        case '"': oss << "\\\""; break;
        case '\\': oss << "\\\\"; break;
        case '\b': oss << "\\b"; break;
        case '\f': oss << "\\f"; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default:
            if(c < 32 || c > 126)
            {
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c);
            }
            else
            {
                oss << c;
            }
        }
    }
    return oss.str();
}

/// Thread-local JSON accumulator
inline SolutionExecutionData& GetJsonAccumulator()
{
    thread_local SolutionExecutionData accumulator;
    return accumulator;
}

/// Check if this kernel should be logged
/// - When log_level <= 2: Only log execution phase kernels (not tuning)
/// - When log_level > 2: Log all kernels (all phases)
inline bool IsLoggingKernel()
{
    const KernelPhase current_phase = GetKernelPhase();
    const auto log_level            = env::value(MIOPEN_PERFORMANCE_LOGS);
    if(log_level > 2)
    {
        // Log all kernels (all phases) when log_level > 2
        return true;
    }
    else if(log_level > 0)
    {
        // Only log execution phase kernels when 0 < log_level <= 2
        return (current_phase == KernelPhase::Execution);
    }
    return false;
}

/// Check if Performance Logs are enabled mode is enabled
inline bool IsPerformanceLoggingEnabled()
{
    const auto log_level = env::value(MIOPEN_PERFORMANCE_LOGS);
    return log_level > 0;
}

/// Add a new performance config with its name and descriptor
/// This creates a new performance config entry. Call AddInvokerTimes() afterwards
/// to add timing samples to this config.
inline void AddPerformanceConfig(const std::string& config_name,
                                 const std::string& config_descriptor = "")
{
    const bool logging_enabled = IsLoggingKernel();
    if(logging_enabled)
    {
        auto& data = GetJsonAccumulator();

        // Finalize the previous config if it exists
        data.FinalizeCurrentConfig();

        // Start a new config
        data.current_config.config_name       = config_name;
        data.current_config.config_descriptor = config_descriptor;
    }
}

/// Add invoker timing samples to the current performance config
/// This should be called after AddPerformanceConfig() and after kernel execution completes
/// Invoker times are exepected to be in order of execution with all invokations
inline void AddInvokerTimes(const std::vector<float>& invoker_times_ms)
{
    const bool logging_enabled = IsLoggingKernel();
    if(!logging_enabled)
        return;

    auto& data = GetJsonAccumulator();

    if(data.current_config.config_name.empty())
    {
        std::cerr << "WARNING: AddInvokerTimes called but no performance config exists"
                  << std::endl;
        return;
    }

    // Add times to the current performance config
    data.current_config.invoker_times_ms = invoker_times_ms;
}

/// Output accumulated JSON data
inline void FlushJsonAccumulator()
{
    auto& data = GetJsonAccumulator();
    data.FinalizeCurrentConfig();

    if(data.performance_configs.empty())
        return;

    // Always write to logs if performance logging is enabled (log_level > 0)
    const auto log_level = env::value(MIOPEN_PERFORMANCE_LOGS);
    if(log_level == 0)
    {
        data.Clear();
        return;
    }

    // Determine if we should show individual kernels
    // Show kernels for log_level 2 and 4, hide for 1 and 3
    const bool show_kernels = (log_level == 2 || log_level == 4);

    // Output JSON with solution-level metrics
    std::cerr << R"({"solution":")" << JsonEscape(data.solution_name) << "\","
              << "\"solver_id\":" << data.solver_id << ","
              << "\"workspace_bytes\":" << data.workspace_bytes << "," << R"("phase":")"
              << data.phase << "\"," << "\"performance_configs\":[";

    // Output each performance config
    bool first_config = true;
    for(const auto& config : data.performance_configs)
    {
        if(!first_config)
            std::cerr << ",";
        first_config = false;

        // Determine config_name: prefer first kernel name, fallback to config.config_name
        std::string display_name = config.config_name;
        if(!config.kernels.empty())
        {
            // Use first non-transformation kernel name if available
            for(const auto& k : config.kernels)
            {
                if(!k.is_transformation && !k.kernel_name.empty())
                {
                    display_name = k.kernel_name;
                    break;
                }
            }
        }

        // If all kernels are transformations or there are no kernels, use the solution name
        if(display_name == config.config_name)
        {
            display_name = data.solution_name;
        }

        std::cerr << R"({"config_name":")" << JsonEscape(display_name) << "\",";

        // Output config_descriptor if present
        if(!config.config_descriptor.empty())
        {
            std::cerr << R"("config_descriptor":")" << JsonEscape(config.config_descriptor)
                      << "\",";
        }

        // Group kernels by kernel_name for this config
        std::map<std::string, GroupedKernelData> grouped_kernels;

        // Variables for config-level aggregation
        std::vector<float> all_exec_times;
        size_t min_exec_number    = std::numeric_limits<size_t>::max();
        int total_transformations = 0;

        // Check if we have invoker timings - if so, use those instead of kernel accumulation
        if(!config.invoker_times_ms.empty())
        {
            all_exec_times  = config.invoker_times_ms;
            min_exec_number = 1; // Invoker timings start from execution 1
        }

        for(const auto& k : config.kernels)
        {
            auto& grouped = grouped_kernels[k.kernel_name];
            // Initialize on first occurrence
            if(grouped.time_executions_ms.empty())
            {
                grouped.kernel_name       = k.kernel_name;
                grouped.is_transformation = k.is_transformation;
            }
            // Append timing data
            grouped.time_executions_ms.push_back(k.time_ms);
        }

        // Count transformations based on grouped kernels (not individual executions)
        for(const auto& entry : grouped_kernels)
        {
            if(entry.second.is_transformation)
            {
                total_transformations++;
            }
        }

        // Calculate config-level statistics
        float config_time_ms  = 0.0f;
        float config_time_std = 0.0f;
        float config_time_min = 0.0f;
        float config_time_max = 0.0f;

        if(!all_exec_times.empty())
        {
            if(all_exec_times.size() <= 1)
            {
                config_time_ms  = all_exec_times[0];
                config_time_std = 0.0f;
                config_time_min = config_time_ms;
                config_time_max = config_time_ms;
            }
            else
            {
                // Drop first two executions (warmup), then removeHighOutliersAndGetMean
                // (same idea as GenericSearch). If too few samples, keep all.
                static constexpr size_t kWarmupExecutions = 2;
                std::vector<float> samples_copy;
                if(all_exec_times.size() > kWarmupExecutions)
                {
                    samples_copy.assign(all_exec_times.begin() + kWarmupExecutions,
                                        all_exec_times.end());
                }
                else
                {
                    samples_copy = all_exec_times;
                }
                config_time_ms = removeHighOutliersAndGetMean(samples_copy, 2.0f);

                // samples_copy is now sorted by removeHighOutliersAndGetMean
                config_time_min = samples_copy.front();
                config_time_max = samples_copy.back();

                // Standard deviation over the same (post-warmup) samples vs. reported mean
                const size_t std_begin =
                    (all_exec_times.size() > kWarmupExecutions) ? kWarmupExecutions : 0;
                float sq_sum = 0.0f;
                int count    = 0;
                for(size_t i = std_begin; i < all_exec_times.size(); ++i)
                {
                    float diff = all_exec_times[i] - config_time_ms;
                    sq_sum += diff * diff;
                    count++;
                }
                config_time_std = (count > 1) ? std::sqrt(sq_sum / count) : 0.0f;
            }
        }

        // Output config-level aggregated stats
        std::cerr << "\"exec_number\":" << min_exec_number << ",";

        std::cerr << "\"time_executions_ms\":[";
        for(size_t i = 0; i < all_exec_times.size(); ++i)
        {
            if(i > 0)
                std::cerr << ",";
            std::cerr << all_exec_times[i];
        }
        std::cerr << "],";

        std::cerr << "\"time_ms\":" << config_time_ms << ","
                  << "\"time_std_ms\":" << config_time_std << ","
                  << "\"time_min_ms\":" << config_time_min << ","
                  << "\"time_max_ms\":" << config_time_max << ","
                  << "\"number_of_transformations\":" << total_transformations;

        // Only output kernels array if show_kernels is true (log_level 2 or 4)
        if(show_kernels)
        {
            std::cerr << ",\"kernels\":[";
            bool first_kernel = true;
            for(const auto& entry : grouped_kernels)
            {
                const auto& g = entry.second;
                if(!first_kernel)
                    std::cerr << ",";
                first_kernel = false;

                std::cerr << R"({"kernel_name":")" << JsonEscape(g.kernel_name) << "\","
                          << "\"time_executions_ms\":[";

                for(size_t i = 0; i < g.time_executions_ms.size(); ++i)
                {
                    if(i > 0)
                        std::cerr << ",";
                    std::cerr << g.time_executions_ms[i];
                }

                std::cerr << "],";
                std::cerr << "\"is_transformation\":" << (g.is_transformation ? "true" : "false")
                          << "}";
            }

            std::cerr << "]"; // Close kernels array
        }
        else
        {
            std::cerr << ",\"kernels\":null";
        }

        std::cerr << "}"; // Close config object
    }

    std::cerr << "]}" << std::endl; // Close performance_configs array and root object

    // Only clear the performance configs, preserve solution-level metadata
    // This ensures solution_name, solver_id, and phase remain available until overwritten
    data.performance_configs.clear();
}

/// Manually flush any pending JSON data (call at program end or context switch)
inline void FinalizeJsonLogging() { FlushJsonAccumulator(); }

/// Add kernel to JSON accumulator
/// All kernels are stored individually when logging is enabled
inline void
AddKernelToJsonAccumulator(const std::string& kernel_name, float time_ms, bool is_transform)
{
    const bool logging_enabled = IsLoggingKernel();
    if(!logging_enabled)
        return;

    // Always store kernels individually
    auto& data = GetJsonAccumulator();
    data.current_config.kernels.push_back({kernel_name, time_ms, is_transform});
}

/// Log solution name if appropriate for the current log level
/// Only prints if the solution name has changed since the last call
inline void
LogSolutionName(const std::string& solution_name, uint64_t solver_id, size_t workspace_bytes = 0)
{
    const bool logging_enabled = IsLoggingKernel();
    if(logging_enabled && !solution_name.empty())
    {
        auto& last_solution  = GetLastPrintedSolutionName();
        auto& last_solver_id = GetLastPrintedSolverId();

        // Check if solution name or solver_id has changed
        if(solution_name != last_solution || solver_id != last_solver_id)
        {
            // Flush previous solution's JSON data
            FlushJsonAccumulator();

            // Set up new solution in accumulator
            const KernelPhase current_phase = GetKernelPhase();
            auto& data                      = GetJsonAccumulator();
            data.solution_name              = solution_name;
            data.solver_id                  = solver_id;
            data.workspace_bytes            = workspace_bytes;
            data.phase                      = KernelPhaseToString(current_phase);
            last_solution                   = solution_name;
            last_solver_id                  = solver_id;
        }
    }
}

/// RAII helper to set kernel phase for a scope
/// Automatically restores previous phase when scope exits
class ScopedKernelPhase
{
public:
    explicit ScopedKernelPhase(KernelPhase phase) : prev_phase(GetKernelPhase())
    {
        SetKernelPhase(phase);
    }

    ~ScopedKernelPhase() { SetKernelPhase(prev_phase); }

    ScopedKernelPhase(const ScopedKernelPhase&)            = delete;
    ScopedKernelPhase& operator=(const ScopedKernelPhase&) = delete;

private:
    KernelPhase prev_phase;
};

} // namespace miopen

#endif // GUARD_MIOPEN_KERNEL_TUNING_MODE_HPP
