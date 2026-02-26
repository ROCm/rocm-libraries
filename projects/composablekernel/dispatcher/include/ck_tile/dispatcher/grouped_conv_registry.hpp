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

#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_kernel_decl.hpp"

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// GroupedConvKernelKey - Unique identifier for a grouped convolution kernel
// =============================================================================

struct GroupedConvKernelKey
{
    std::string dtype_in;
    std::string dtype_wei;
    std::string dtype_out;
    std::string layout; // e.g., "nhwgc_gkyxc_nhwgk"
    int ndim_spatial;   // 1, 2, or 3
    GroupedConvOp op;

    // Tile configuration
    int tile_m;
    int tile_n;
    int tile_k;

    // Pipeline
    std::string pipeline;
    std::string scheduler;

    // GPU architecture (for filter_by_arch)
    std::string arch = "gfx942";

    bool operator==(const GroupedConvKernelKey& other) const
    {
        return dtype_in == other.dtype_in && dtype_wei == other.dtype_wei &&
               dtype_out == other.dtype_out && layout == other.layout &&
               ndim_spatial == other.ndim_spatial && op == other.op && tile_m == other.tile_m &&
               tile_n == other.tile_n && tile_k == other.tile_k && pipeline == other.pipeline &&
               scheduler == other.scheduler && arch == other.arch;
    }

    std::string to_string() const
    {
        std::string op_str;
        switch(op)
        {
        case GroupedConvOp::Forward: op_str = "fwd"; break;
        case GroupedConvOp::BackwardData: op_str = "bwdd"; break;
        case GroupedConvOp::BackwardWeight: op_str = "bwdw"; break;
        }
        return "grouped_conv_" + op_str + "_" + dtype_in + "_" + std::to_string(ndim_spatial) +
               "d_" + std::to_string(tile_m) + "x" + std::to_string(tile_n) + "x" +
               std::to_string(tile_k);
    }
};

struct GroupedConvKernelKeyHash
{
    std::size_t operator()(const GroupedConvKernelKey& key) const
    {
        std::size_t h = std::hash<std::string>{}(key.dtype_in);
        h ^= std::hash<std::string>{}(key.layout) << 1;
        h ^= std::hash<int>{}(key.ndim_spatial) << 2;
        h ^= std::hash<int>{}(static_cast<int>(key.op)) << 3;
        h ^= std::hash<int>{}(key.tile_m) << 4;
        h ^= std::hash<int>{}(key.tile_n) << 5;
        h ^= std::hash<int>{}(key.tile_k) << 6;
        h ^= std::hash<std::string>{}(key.arch) << 7;
        return h;
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
    using RunFn = std::function<float(const GroupedConvProblem&, void*)>;

    GroupedConvKernelInstance(const GroupedConvKernelKey& key,
                              const std::string& name,
                              RunFn run_fn)
        : key_(key), name_(name), run_fn_(std::move(run_fn))
    {
    }

    const GroupedConvKernelKey& key() const { return key_; }
    const std::string& name() const { return name_; }

    float run(const GroupedConvProblem& problem, void* stream = nullptr) const
    {
        return run_fn_(problem, stream);
    }

    bool matches(const GroupedConvProblem& problem) const
    {
        // Check if this kernel can handle the problem
        return problem.op == key_.op;
    }

    private:
    GroupedConvKernelKey key_;
    std::string name_;
    RunFn run_fn_;
};

// =============================================================================
// GroupedConvRegistry - Stores and manages grouped convolution kernels
// =============================================================================

class GroupedConvRegistry
{
    public:
    enum class Priority
    {
        Low    = 0,
        Normal = 1,
        High   = 2
    };

    GroupedConvRegistry() = default;

    /// Singleton instance for global kernel registration
    static GroupedConvRegistry& instance()
    {
        static GroupedConvRegistry registry;
        return registry;
    }

    void set_name(const std::string& name) { name_ = name; }
    const std::string& name() const { return name_; }

    /// Register a kernel instance
    bool register_kernel(std::shared_ptr<GroupedConvKernelInstance> kernel,
                        Priority priority = Priority::Normal)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto& key  = kernel->key();
        kernels_[key]    = kernel;
        priorities_[key] = priority;
        return true;
    }

    /// Register kernels from a GroupedConvKernelSet
    bool register_set(const GroupedConvKernelSet& kernel_set, Priority priority = Priority::Normal)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for(const auto& decl : kernel_set.declarations())
        {
            // Create kernel instance from declaration
            GroupedConvKernelKey key;
            key.dtype_in     = decl.signature.dtype_in_;
            key.dtype_wei    = decl.signature.dtype_wei_;
            key.dtype_out    = decl.signature.dtype_out_;
            key.layout       = decl.signature.layout_;
            key.ndim_spatial = decl.signature.num_dims_;
            key.op           = (decl.signature.conv_op_ == "forward")
                                   ? GroupedConvOp::Forward
                                   : (decl.signature.conv_op_ == "bwd_data")
                                         ? GroupedConvOp::BackwardData
                                         : GroupedConvOp::BackwardWeight;
            key.tile_m       = decl.algorithm.tile_m_;
            key.tile_n       = decl.algorithm.tile_n_;
            key.tile_k       = decl.algorithm.tile_k_;
            key.pipeline     = decl.algorithm.pipeline_;
            key.scheduler    = decl.algorithm.scheduler_;
            key.arch         = decl.arch;

            auto instance = std::make_shared<GroupedConvKernelInstance>(
                key,
                decl.name(),
                [](const GroupedConvProblem&, void*) -> float { return 0.0f; } // Placeholder
            );
            kernels_[key]    = instance;
            priorities_[key] = priority;
        }
        return true;
    }

    /// Find the best kernel for a problem
    const GroupedConvKernelInstance* find(const GroupedConvProblem& problem) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const GroupedConvKernelInstance* best = nullptr;
        Priority best_priority               = Priority::Low;

        for(const auto& [key, kernel] : kernels_)
        {
            if(kernel->matches(problem))
            {
                auto it           = priorities_.find(key);
                Priority priority = (it != priorities_.end()) ? it->second : Priority::Normal;
                if(!best || priority > best_priority)
                {
                    best          = kernel.get();
                    best_priority = priority;
                }
            }
        }

        return best;
    }

    /// Get all registered kernels
    std::vector<const GroupedConvKernelInstance*> all_kernels() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<const GroupedConvKernelInstance*> result;
        for(const auto& [key, kernel] : kernels_)
        {
            result.push_back(kernel.get());
        }
        return result;
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return kernels_.size();
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return kernels_.empty();
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        kernels_.clear();
        priorities_.clear();
    }

    /// Export registry to JSON string
    std::string export_json(bool include_statistics = false) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream json;

        json << "{\n";
        json << "  \"metadata\": {\n";
        json << "    \"registry_name\": \"" << json_escape(name_) << "\",\n";
        json << "    \"total_kernels\": " << kernels_.size() << "\n";
        json << "  }";

        if(include_statistics && !kernels_.empty())
        {
            std::map<std::string, int> by_datatype;
            std::map<std::string, int> by_pipeline;
            std::map<std::string, int> by_arch;

            for(const auto& [key, kernel] : kernels_)
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
        for(const auto& [key, kernel] : kernels_)
        {
            if(!first)
                json << ",\n";
            json << "    " << export_kernel_json(*kernel);
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
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<const GroupedConvKernelInstance*> result;
        for(const auto& [key, kernel] : kernels_)
        {
            if(predicate(*kernel))
            {
                result.push_back(kernel.get());
            }
        }
        return result;
    }

    /// Remove kernels not matching the arch
    std::size_t filter_by_arch(const std::string& gpu_arch)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<GroupedConvKernelKey> to_remove;
        for(const auto& [key, kernel] : kernels_)
        {
            if(key.arch != gpu_arch)
            {
                to_remove.push_back(key);
            }
        }
        for(const auto& key : to_remove)
        {
            kernels_.erase(key);
            priorities_.erase(key);
        }
        return to_remove.size();
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
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
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
        case GroupedConvOp::BackwardData: op_str = "bwdd"; break;
        case GroupedConvOp::BackwardWeight: op_str = "bwdw"; break;
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
        json << "        \"pipeline\": \"" << json_escape(key.pipeline) << "\",\n";
        json << "        \"scheduler\": \"" << json_escape(key.scheduler) << "\"\n";
        json << "      },\n";
        json << "      \"arch\": \"" << json_escape(key.arch) << "\"\n";
        json << "    }";

        return json.str();
    }

    std::string name_ = "default";
    mutable std::mutex mutex_;
    std::unordered_map<GroupedConvKernelKey,
                       std::shared_ptr<GroupedConvKernelInstance>,
                       GroupedConvKernelKeyHash>
        kernels_;
    std::unordered_map<GroupedConvKernelKey, Priority, GroupedConvKernelKeyHash> priorities_;
};

// =============================================================================
// GroupedConvDispatcher - Selects and runs the best kernel for a problem
// =============================================================================

class GroupedConvDispatcher
{
    public:
    explicit GroupedConvDispatcher(GroupedConvRegistry* registry) : registry_(registry) {}

    /// Run convolution with automatic kernel selection
    float run(const GroupedConvProblem& problem, void* stream = nullptr)
    {
        const auto* kernel = registry_->find(problem);
        if(!kernel)
        {
            throw std::runtime_error("No suitable grouped convolution kernel found for problem: " +
                                     problem.to_string());
        }
        return kernel->run(problem, stream);
    }

    /// Get the kernel that would be selected for a problem
    const GroupedConvKernelInstance* select(const GroupedConvProblem& problem) const
    {
        return registry_->find(problem);
    }

    private:
    GroupedConvRegistry* registry_;
};

} // namespace dispatcher
} // namespace ck_tile
