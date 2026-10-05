// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file grouped_conv_registry.hpp
 * @brief Grouped Convolution kernel registry and dispatcher
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <locale>

#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_invocation.hpp"

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// GroupedConvKernelKey - Unique identifier for a grouped convolution kernel
// =============================================================================

struct GroupedConvKernelKey
{
    // Signature fields
    std::string dtype_in;
    std::string dtype_wei;
    std::string dtype_out;
    std::string layout;   // e.g., "nhwgc"
    int ndim_spatial = 2; // 2 or 3
    GroupedConvOp op = GroupedConvOp::Forward;

    // Tile configuration
    int tile_m = 1;
    int tile_n = 128;
    int tile_k = 128;

    // Wave/warp configuration
    int wave_m = 2;
    int wave_n = 2;
    int wave_k = 1;
    int warp_m = 32;
    int warp_n = 32;
    int warp_k = 16;

    // Pipeline
    std::string pipeline  = "compv3";
    std::string scheduler = "intrawave";
    std::string epilogue  = "cshuffle";

    // ConvConfigBase parity fields
    int vector_size_a       = 4;
    int vector_size_b       = 8;
    int vector_size_c       = 8;
    int block_per_cu        = 1;
    int num_wave_groups     = 1;
    int num_groups_to_merge = 1;

    // Convolution specialization (e.g., "default", "filter1x1_stride1_pad0")
    std::string specialization = "default";

    // Large tensor (split image) support
    bool large_tensor = false;

    // Stream-K configuration
    bool streamk_enabled          = false;
    std::string streamk_reduction = "none"; // "none", "tree", "linear"
    bool streamk_persistent       = false;

    // GPU architecture (for filter_by_arch)
    std::string arch = "gfx942";

    // Generated instance name. Unique per kernel by construction, so it is what
    // makes kernel_id() an exact identity; the descriptive fields above are
    // parsed from it and are reported, not identifying.
    std::string name;

    bool operator==(const GroupedConvKernelKey& other) const
    {
        return dtype_in == other.dtype_in && dtype_wei == other.dtype_wei &&
               dtype_out == other.dtype_out && layout == other.layout &&
               ndim_spatial == other.ndim_spatial && op == other.op && arch == other.arch &&
               name == other.name;
    }

    std::string to_string() const
    {
        std::string op_str;
        switch(op)
        {
        case GroupedConvOp::Forward: op_str = "fwd"; break;
        case GroupedConvOp::BackwardData: op_str = "bwd_data"; break;
        case GroupedConvOp::BackwardWeight: op_str = "bwd_weight"; break;
        }
        return "grouped_conv_" + op_str + "_" + dtype_in + "_" + std::to_string(ndim_spatial) +
               "d_" + std::to_string(tile_m) + "x" + std::to_string(tile_n) + "x" +
               std::to_string(tile_k) + "_" + std::to_string(wave_m) + "x" +
               std::to_string(wave_n) + "x" + std::to_string(wave_k) + "_" +
               std::to_string(warp_m) + "x" + std::to_string(warp_n) + "x" +
               std::to_string(warp_k) + "_" + pipeline +
               (specialization != "default" ? "_" + specialization : "");
    }

    std::string kernel_id() const
    {
        std::string result = "cktile_gconv_v1";
        const auto append = [&result](const auto& value) {
            std::ostringstream text;
            text.imbue(std::locale::classic());
            text << value;
            const std::string encoded = text.str();
            result += "|" + std::to_string(encoded.size()) + ":" + encoded;
        };
        append(dtype_in); append(dtype_wei); append(dtype_out); append(layout);
        append(ndim_spatial); append(static_cast<int>(op)); append(arch); append(name);
        return result;
    }
};

struct GroupedConvKernelKeyHash
{
    std::size_t operator()(const GroupedConvKernelKey& key) const
    {
        // Intentionally trade hashing cost for exact operator== parity; cache IDs if the catalog grows.
        return std::hash<std::string>{}(key.kernel_id());
    }
};

// =============================================================================
// GroupedConvKernelInstance - Runtime representation of a kernel
// =============================================================================

// Forward declaration for shared_ptr type alias
class GroupedConvKernelInstance;
using GroupedConvKernelInstancePtr = std::shared_ptr<GroupedConvKernelInstance>;

class GroupedConvKernelInstance
{
    public:
    using RunFn         = std::function<float(const GroupedConvProblem&, void*)>;
    using IsSupportedFn = std::function<bool(const GroupedConvProblem&)>;

    GroupedConvKernelInstance(const GroupedConvKernelKey& key,
                              RunFn run_fn,
                              IsSupportedFn is_supported_fn      = nullptr,
                              const std::string& instance_string = "")
        : key_(key),
          run_fn_(std::move(run_fn)),
          is_supported_fn_(std::move(is_supported_fn)),
          instance_string_(instance_string),
          kernel_id_(key.kernel_id())
    {
    }

    const GroupedConvKernelKey& key() const { return key_; }
    const std::string& kernel_id() const { return kernel_id_; }
    bool executable() const
    {
        return static_cast<bool>(run_fn_) && static_cast<bool>(is_supported_fn_);
    }

    /// Return the kernel name.
    /// @param use_instance_string  When true, return the CK Tile
    ///        GetInstanceString() representation (e.g.
    ///        "GroupedConvolutionBackwardWeightKernel<2,Default,...>")
    ///        if available; otherwise fall back to the dispatcher name.
    const std::string& name(bool use_instance_string = false) const
    {
        if(use_instance_string && !instance_string_.empty())
            return instance_string_;
        return key_.name;
    }

    // Check whether this kernel supports the given problem.
    bool is_supported(const GroupedConvProblem& problem) const
    {
        return problem.is_valid() && is_supported_fn_ && is_supported_fn_(problem);
    }

    float run(const GroupedConvProblem& problem, void* stream = nullptr) const
    {
        return run_fn_(problem, stream);
    }

    bool matches(const GroupedConvProblem& problem) const
    {
        return problem.dtype_in == key_.dtype_in && problem.dtype_wei == key_.dtype_wei &&
               problem.dtype_out == key_.dtype_out && problem.layout == key_.layout &&
               problem.ndim_spatial == key_.ndim_spatial && problem.op == key_.op &&
               problem.arch == key_.arch;
    }

    private:
    GroupedConvKernelKey key_;
    RunFn run_fn_;
    IsSupportedFn is_supported_fn_;
    std::string instance_string_;
    std::string kernel_id_;
};

// =============================================================================
// GroupedConvRegistry - Stores and manages grouped convolution kernels
// =============================================================================

class GroupedConvRegistry : public BaseRegistry<GroupedConvRegistry,
                                                GroupedConvKernelKey,
                                                GroupedConvKernelInstance,
                                                GroupedConvKernelKeyHash>
{
    using Base = BaseRegistry<GroupedConvRegistry,
                              GroupedConvKernelKey,
                              GroupedConvKernelInstance,
                              GroupedConvKernelKeyHash>;

    public:
    GroupedConvRegistry() = default;

    /// Singleton instance for global kernel registration
    static GroupedConvRegistry& instance()
    {
        static GroupedConvRegistry registry;
        return registry;
    }

    /// Register an executable kernel. Metadata-only instances are rejected.
    bool register_kernel(const GroupedConvKernelKey& key,
                         GroupedConvKernelInstancePtr instance,
                         Priority priority = Priority::Normal)
    {
        if(!instance || !instance->executable() || !(instance->key() == key) || key.name.empty())
            return false;
        std::lock_guard<std::mutex> lock(mutex());
        auto it = entries().find(key);
        if(it != entries().end() && it->second.priority >= priority)
            return false;
        entries_mut()[key] = typename Base::Entry{std::move(instance), priority};
        return true;
    }

    /// Find the best kernel for a problem
    const GroupedConvKernelInstance* find(const GroupedConvProblem& problem) const
    {
        return find_best_supported(problem);
    }

    const GroupedConvKernelInstance* find_best_supported(const GroupedConvProblem& problem) const
    {
        std::lock_guard<std::mutex> lock(mutex());
        const GroupedConvKernelInstance* best = nullptr;
        Priority best_priority                = Priority::Low;

        for(const auto& [key, entry] : entries())
        {
            if(is_executable_candidate(*entry.instance, problem))
            {
                if(!best || entry.priority > best_priority ||
                   (entry.priority == best_priority &&
                    entry.instance->kernel_id() < best->kernel_id()))
                {
                    best          = entry.instance.get();
                    best_priority = entry.priority;
                }
            }
        }

        return best;
    }

    std::vector<const GroupedConvKernelInstance*>
    find_all_supported(const GroupedConvProblem& problem) const
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::vector<const GroupedConvKernelInstance*> result;
        for(const auto& item : entries())
            if(is_executable_candidate(*item.second.instance, problem))
                result.push_back(item.second.instance.get());
        std::sort(result.begin(), result.end(), [](const auto* lhs, const auto* rhs) {
            return lhs->kernel_id() < rhs->kernel_id();
        });
        return result;
    }

    const GroupedConvKernelInstance*
    find_by_id(const GroupedConvProblem& problem, const std::string& kernel_id) const
    {
        std::lock_guard<std::mutex> lock(mutex());
        for(const auto& item : entries())
        {
            const auto& instance = *item.second.instance;
            if(instance.kernel_id() == kernel_id && is_executable_candidate(instance, problem))
                return &instance;
        }
        return nullptr;
    }

    /// Get all registered kernels
    std::vector<const GroupedConvKernelInstance*> all_kernels() const
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::vector<const GroupedConvKernelInstance*> result;
        for(const auto& [key, entry] : entries())
        {
            result.push_back(entry.instance.get());
        }
        return result;
    }

    /// Export registry to JSON string
    std::string export_json(bool include_statistics = false) const
    {
        // Note: get_name() acquires the mutex internally, so we must NOT hold
        // the registry mutex here (std::mutex is not recursive).
        std::string reg_name = get_name();

        std::lock_guard<std::mutex> lock(mutex());
        std::ostringstream json;

        json << "{\n";
        json << "  \"metadata\": {\n";
        json << "    \"registry_name\": \"" << json_escape(reg_name) << "\",\n";
        json << "    \"total_kernels\": " << entries().size() << "\n";
        json << "  }";

        if(include_statistics && !entries().empty())
        {
            std::map<std::string, int> by_datatype;
            std::map<std::string, int> by_pipeline;
            std::map<std::string, int> by_arch;

            for(const auto& [key, entry] : entries())
            {
                std::string dtype_key = key.dtype_in + "_" + key.dtype_wei + "_" + key.dtype_out;
                by_datatype[dtype_key]++;
                by_pipeline[key.pipeline]++;
                by_arch[key.arch]++;
            }

            json << ",\n  \"statistics\": {\n";
            json << "    \"by_datatype\": {";
            bool first = true;
            for(const auto& [dtype, count] : by_datatype)
            {
                if(!first)
                    json << ",";
                json << "\"" << json_escape(dtype) << "\":" << count;
                first = false;
            }
            json << "},\n";
            json << "    \"by_pipeline\": {";
            first = true;
            for(const auto& [pipeline, count] : by_pipeline)
            {
                if(!first)
                    json << ",";
                json << "\"" << json_escape(pipeline) << "\":" << count;
                first = false;
            }
            json << "},\n";
            json << "    \"by_arch\": {";
            first = true;
            for(const auto& [arch, count] : by_arch)
            {
                if(!first)
                    json << ",";
                json << "\"" << json_escape(arch) << "\":" << count;
                first = false;
            }
            json << "}\n  }";
        }

        json << ",\n  \"kernels\": [\n";
        bool first = true;
        for(const auto& [key, entry] : entries())
        {
            if(!first)
                json << ",\n";
            json << "    " << export_kernel_json(*entry.instance);
            first = false;
        }
        json << "\n  ]\n";
        json << "}\n";

        return json.str();
    }

    /// Export registry to JSON file
    void export_json_to_file(const std::string& filename, bool include_statistics = false) const
    {
        std::string json_str = export_json(include_statistics);
        std::ofstream file(filename);
        if(!file.is_open())
        {
            throw std::runtime_error("Failed to open file for export: " + filename);
        }
        file << json_str;
    }

    /// Get kernels matching a predicate
    std::vector<const GroupedConvKernelInstance*>
    filter(std::function<bool(const GroupedConvKernelInstance&)> predicate) const
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::vector<const GroupedConvKernelInstance*> result;
        for(const auto& [key, entry] : entries())
        {
            if(predicate(*entry.instance))
            {
                result.push_back(entry.instance.get());
            }
        }
        return result;
    }

    /// Remove kernels not matching the arch
    std::size_t filter_by_arch(const std::string& gpu_arch)
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::vector<GroupedConvKernelKey> to_remove;
        for(const auto& [key, entry] : entries())
        {
            if(key.arch != gpu_arch)
            {
                to_remove.push_back(key);
            }
        }
        for(const auto& key : to_remove)
        {
            entries_mut().erase(key);
        }
        return to_remove.size();
    }

    public:
    static bool is_executable_candidate(const GroupedConvKernelInstance& instance,
                                        const GroupedConvProblem& problem)
    {
        return problem.is_valid() && instance.executable() && instance.matches(problem) &&
               instance.is_supported(problem);
    }

    private:
    static std::string json_escape(const std::string& str)
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
                if(c < 0x20)
                {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                }
                else
                {
                    oss << c;
                }
            }
        }
        return oss.str();
    }

    static std::string export_kernel_json(const GroupedConvKernelInstance& kernel)
    {
        std::ostringstream json;
        const auto& key = kernel.key();

        std::string op_str;
        switch(key.op)
        {
        case GroupedConvOp::Forward: op_str = "fwd"; break;
        case GroupedConvOp::BackwardData: op_str = "bwd_data"; break;
        case GroupedConvOp::BackwardWeight: op_str = "bwd_weight"; break;
        }

        json << "{\n";
        json << "      \"name\": \"" << json_escape(kernel.name()) << "\",\n";
        json << "      \"signature\": {\n";
        json << "        \"dtype_in\": \"" << json_escape(key.dtype_in) << "\",\n";
        json << "        \"dtype_wei\": \"" << json_escape(key.dtype_wei) << "\",\n";
        json << "        \"dtype_out\": \"" << json_escape(key.dtype_out) << "\",\n";
        json << "        \"layout\": \"" << json_escape(key.layout) << "\",\n";
        json << "        \"ndim_spatial\": " << key.ndim_spatial << ",\n";
        json << "        \"op\": \"" << op_str << "\"\n";
        json << "      },\n";
        json << "      \"algorithm\": {\n";
        json << "        \"tile_m\": " << key.tile_m << ",\n";
        json << "        \"tile_n\": " << key.tile_n << ",\n";
        json << "        \"tile_k\": " << key.tile_k << ",\n";
        json << "        \"wave\": \"" << key.wave_m << "x" << key.wave_n << "x" << key.wave_k
             << "\",\n";
        json << "        \"warp\": \"" << key.warp_m << "x" << key.warp_n << "x" << key.warp_k
             << "\",\n";
        json << "        \"pipeline\": \"" << json_escape(key.pipeline) << "\",\n";
        json << "        \"scheduler\": \"" << json_escape(key.scheduler) << "\",\n";
        json << "        \"epilogue\": \"" << json_escape(key.epilogue) << "\",\n";
        json << "        \"vector_sizes\": [" << key.vector_size_a << "," << key.vector_size_b
             << "," << key.vector_size_c << "],\n";
        json << "        \"block_per_cu\": " << key.block_per_cu << ",\n";
        json << "        \"num_wave_groups\": " << key.num_wave_groups << ",\n";
        json << "        \"num_groups_to_merge\": " << key.num_groups_to_merge << "\n";
        json << "      },\n";
        json << "      \"arch\": \"" << json_escape(key.arch) << "\"\n";
        json << "    }";

        return json.str();
    }
};

// =============================================================================
// GroupedConvDispatcher - Selects and runs the best kernel for a problem
// =============================================================================

class GroupedConvDispatcher
{
    public:
    enum class SelectionStrategy
    {
        PriorityBased,
        Heuristic
    };

    using HeuristicFunction = std::function<std::vector<std::string>(const GroupedConvProblem&)>;

    explicit GroupedConvDispatcher(GroupedConvRegistry* registry)
        : registry_(registry), strategy_(SelectionStrategy::PriorityBased)
    {
    }

    void set_strategy(SelectionStrategy s) { strategy_ = s; }
    void set_heuristic(HeuristicFunction fn) { heuristic_ = std::move(fn); }

    /// Select the best kernel for a problem (does not run it)
    const GroupedConvKernelInstance* select_kernel(const GroupedConvProblem& problem) const
    {
        if(strategy_ == SelectionStrategy::Heuristic)
            return select_heuristic(problem);
        return registry_->find(problem);
    }

    /// Run convolution with automatic kernel selection (legacy - no buffers)
    float run(const GroupedConvProblem& problem, void* stream = nullptr)
    {
        const auto* kernel = select_kernel(problem);
        if(!kernel)
        {
            throw NoKernelFound("No suitable grouped convolution kernel found for problem: " +
                                problem.to_string());
        }
        return kernel->run(problem, stream);
    }

    /// Teaching/compat one-shot. Not the production path: launches on device 0
    /// with warmup=0/repeat=1. Prefer the overload that takes device_ordinal.
    float run(const void* input_ptr,
              const void* weight_ptr,
              void* output_ptr,
              const GroupedConvProblem& problem,
              void* stream = nullptr)
    {
        return run(input_ptr, weight_ptr, output_ptr, problem, 0, stream);
    }

    /// Run a convolution exactly once on an explicitly identified device.
    /// device_ordinal < 0 is rejected by every generated backend run path
    /// (forward, backward-data, backward-weight); do not pass -1 to mean
    /// "current device".
    float run(const void* input_ptr,
              const void* weight_ptr,
              void* output_ptr,
              const GroupedConvProblem& problem,
              int device_ordinal,
              void* stream                         = nullptr,
              ConvDeviceArchProvider arch_provider = {})
    {
        const auto* kernel = select_kernel(problem);
        if(!kernel)
        {
            throw NoKernelFound("No suitable grouped convolution kernel found for problem: " +
                                problem.to_string());
        }
        ScopedConvInvocationContext invocation({input_ptr,
                                                weight_ptr,
                                                output_ptr,
                                                device_ordinal,
                                                problem.split_k,
                                                std::move(arch_provider)});
        return kernel->run(problem, stream);
    }

    /// Alias kept for backward compatibility
    const GroupedConvKernelInstance* select(const GroupedConvProblem& problem) const
    {
        return select_kernel(problem);
    }

    private:
    const GroupedConvKernelInstance* select_heuristic(const GroupedConvProblem& problem) const
    {
        if(!heuristic_)
            return registry_->find(problem);

        auto ranked_names = heuristic_(problem);
        auto all          = registry_->all_kernels();
        for(const auto& name : ranked_names)
        {
            for(const auto* kernel : all)
            {
                if(kernel->name().find(name) != std::string::npos &&
                   GroupedConvRegistry::is_executable_candidate(*kernel, problem))
                {
                    return kernel;
                }
            }
        }
        return registry_->find(problem);
    }

    GroupedConvRegistry* registry_;
    SelectionStrategy strategy_;
    HeuristicFunction heuristic_;
};

} // namespace dispatcher
} // namespace ck_tile
