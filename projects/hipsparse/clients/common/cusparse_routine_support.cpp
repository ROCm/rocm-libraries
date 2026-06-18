/*! \file */
/* ************************************************************************
* Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */

#include "cusparse_routine_support.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>

#include <yaml-cpp/yaml.h>

// ── Loading helpers ───────────────────────────────────────────────────────────

// Parses a single algorithm entry node { default_value, description, supported_values }.
static AlgorithmEntry parse_algorithm_entry(const YAML::Node& node)
{
    AlgorithmEntry e;
    if(!node || !node.IsMap())
        return e;
    if(node["default_value"])
        e.default_value = node["default_value"].as<int>();
    if(node["description"])
        e.description = node["description"].as<std::string>();
    if(node["supported_values"] && node["supported_values"].IsSequence())
    {
        for(const auto& v : node["supported_values"])
            e.supported_values.push_back(v.as<int>());
    }
    return e;
}

// ── Loading ───────────────────────────────────────────────────────────────────

bool CusparseRoutineSupport::load(const std::string& filepath)
{
    try
    {
        YAML::Node root = YAML::LoadFile(filepath);

        // ── routines ─────────────────────────────────────────────────────
        const YAML::Node& routines_node = root["routines"];
        if(!routines_node || !routines_node.IsMap())
        {
            std::cerr << "Warning: cusparse_support.yaml: missing or invalid 'routines' map in '"
                      << filepath << "'\n";
            return false;
        }

        for(const auto& entry : routines_node)
        {
            const std::string     name = entry.first.as<std::string>();
            CudaVersionConstraint constraint;
            const YAML::Node&     val = entry.second;

            if(val && val.IsMap())
            {
                if(val["min_cuda_version"])
                    constraint.min_version = val["min_cuda_version"].as<int>();
                if(val["max_cuda_version"])
                    constraint.max_version = val["max_cuda_version"].as<int>();
            }

            routines_[name] = constraint;
        }

        // ── algorithm_support ────────────────────────────────────────────
        const YAML::Node& alg_node = root["algorithm_support"];
        if(alg_node && alg_node.IsMap())
        {
            for(const auto& alg_entry : alg_node)
            {
                const std::string  alg_type = alg_entry.first.as<std::string>();
                const YAML::Node&  alg_val  = alg_entry.second;
                AlgorithmSupport   support;

                if(alg_val["rocm"])
                    support.rocm_entry = parse_algorithm_entry(alg_val["rocm"]);

                if(alg_val["cuda_version_ranges"] && alg_val["cuda_version_ranges"].IsSequence())
                {
                    for(const auto& range_node : alg_val["cuda_version_ranges"])
                    {
                        AlgorithmCudaRange range;
                        if(range_node["min_cuda_version"])
                            range.min_cuda_version = range_node["min_cuda_version"].as<int>();
                        if(range_node["max_cuda_version"])
                            range.max_cuda_version = range_node["max_cuda_version"].as<int>();
                        range.entry = parse_algorithm_entry(range_node);
                        support.cuda_ranges.push_back(std::move(range));
                    }
                }

                algorithm_support_[alg_type] = std::move(support);
            }
        }

        loaded_ = true;
        return true;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Warning: failed to load cusparse support file '" << filepath
                  << "': " << ex.what() << '\n';
        return false;
    }
}

// ── Querying ──────────────────────────────────────────────────────────────────

bool CusparseRoutineSupport::is_supported(const std::string& routine, int cuda_version) const
{
    if(cuda_version < 0)
        return true; // ROCm/HIP backend: all routines are always supported

    auto it = routines_.find(routine);
    if(it == routines_.end())
        return true; // absent from CUDA table → assume supported

    return it->second.is_supported(cuda_version);
}

// ── Human-readable helpers ────────────────────────────────────────────────────

// static
std::string CusparseRoutineSupport::format_cuda_version(int v)
{
    int major = v / 1000;
    int minor = (v % 1000) / 10;
    int patch = v % 10;
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

std::string CusparseRoutineSupport::get_support_range_string(const std::string& routine) const
{
    auto it = routines_.find(routine);
    if(it == routines_.end())
        return "all CUDA versions";

    const auto& c = it->second;
    if(!c.min_version.has_value() && !c.max_version.has_value())
        return "all CUDA versions";

    if(c.min_version.has_value() && c.max_version.has_value())
    {
        // The max bound is exclusive; show the last supported version as max-1.
        return "CUDA >= " + format_cuda_version(*c.min_version) + " and CUDA < "
               + format_cuda_version(*c.max_version);
    }
    if(c.min_version.has_value())
        return "CUDA >= " + format_cuda_version(*c.min_version);

    return "CUDA < " + format_cuda_version(*c.max_version);
}

std::string CusparseRoutineSupport::get_support_warning(const std::string& routine,
                                                         int                cuda_version) const
{
    std::ostringstream oss;
    oss << "Warning: routine '" << routine << "' is not supported for CUDA version "
        << format_cuda_version(cuda_version) << " (CUDART_VERSION=" << cuda_version << ").\n"
        << "  Supported range: " << get_support_range_string(routine) << '\n';
    return oss.str();
}

// ── Algorithm-support queries ─────────────────────────────────────────────────

AlgorithmEntry CusparseRoutineSupport::get_algorithm_entry(const std::string& alg_type,
                                                            int                cuda_version) const
{
    auto it = algorithm_support_.find(alg_type);
    if(it == algorithm_support_.end())
        return AlgorithmEntry{}; // unknown operation → "not supported" default

    const AlgorithmSupport& support = it->second;

    if(cuda_version < 0)
        return support.rocm_entry;

    // Return the first matching CUDA range (ranges are listed top-to-bottom
    // in the YAML in order of ascending version, so the first match is the
    // tightest applicable range).
    for(const auto& range : support.cuda_ranges)
    {
        if(range.matches(cuda_version))
            return range.entry;
    }

    return AlgorithmEntry{}; // no matching range → "not supported" default
}

int CusparseRoutineSupport::get_algorithm_default(const std::string& alg_type,
                                                   int                cuda_version) const
{
    return get_algorithm_entry(alg_type, cuda_version).default_value;
}

std::string CusparseRoutineSupport::get_algorithm_description(const std::string& alg_type,
                                                               int                cuda_version) const
{
    return get_algorithm_entry(alg_type, cuda_version).description;
}

std::vector<int>
    CusparseRoutineSupport::get_algorithm_supported_values(const std::string& alg_type,
                                                           int                cuda_version) const
{
    return get_algorithm_entry(alg_type, cuda_version).supported_values;
}

// ── File discovery ────────────────────────────────────────────────────────────

// static
std::string CusparseRoutineSupport::find_support_file()
{
    // 1. Explicit environment variable override
    if(const char* env = std::getenv("HIPSPARSE_CUSPARSE_SUPPORT_FILE"))
    {
        if(env[0] != '\0')
            return env;
    }

#ifdef __linux__
    std::error_code ec;
    auto            exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if(!ec)
    {
        auto exe_dir = exe.parent_path();

        // 2. Alongside the running executable
        {
            auto candidate = exe_dir / "cusparse_support.yaml";
            if(std::filesystem::exists(candidate))
                return candidate.string();
        }

        // 3. <exe_dir>/../staging/  (common build-tree layout)
        {
            std::error_code ec2;
            auto            candidate
                = std::filesystem::weakly_canonical(exe_dir / "../staging/cusparse_support.yaml",
                                                    ec2);
            if(!ec2 && std::filesystem::exists(candidate))
                return candidate.string();
        }
    }
#endif

    // 4. CMake-configured install path (injected at build time via -D)
#ifdef HIPSPARSE_CUSPARSE_SUPPORT_FILE_PATH
    {
        const std::string configured{HIPSPARSE_CUSPARSE_SUPPORT_FILE_PATH};
        if(std::filesystem::exists(configured))
            return configured;
    }
#endif

    return {};
}

// ── Singleton ─────────────────────────────────────────────────────────────────

// static
const CusparseRoutineSupport& CusparseRoutineSupport::instance()
{
    // C++11 guarantees thread-safe one-time initialisation of function-local statics.
    static CusparseRoutineSupport inst = []() {
        CusparseRoutineSupport s;
        const std::string      path = find_support_file();
        if(path.empty())
        {
            std::cerr << "Warning: cusparse_support.yaml not found.\n"
                      << "  All routines will be treated as supported.\n"
                      << "  Set HIPSPARSE_CUSPARSE_SUPPORT_FILE to specify its location.\n";
        }
        else
        {
            s.load(path);
        }
        return s;
    }();

    return inst;
}
