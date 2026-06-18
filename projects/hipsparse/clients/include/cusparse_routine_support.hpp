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
#pragma once

#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

// Represents the CUDA version support constraints for a single routine.
//
// CUDART_VERSION integer encoding: major * 1000 + minor * 10 + patch
//   CUDA 11.3.1  ->  11031
//   CUDA 12.0.0  ->  12000
struct CudaVersionConstraint
{
    // Minimum CUDART_VERSION required (inclusive).
    // Routine is supported when CUDART_VERSION >= min_version.
    std::optional<int> min_version;

    // Exclusive upper CUDART_VERSION bound.
    // Routine is supported when CUDART_VERSION < max_version.
    std::optional<int> max_version;

    bool is_supported(int cuda_version) const
    {
        if(min_version.has_value() && cuda_version < *min_version)
            return false;
        if(max_version.has_value() && cuda_version >= *max_version)
            return false;
        return true;
    }
};

// Holds the algorithm support data for a single operation on a single
// CUDA version range (or the ROCm backend).
struct AlgorithmEntry
{
    // Integer value of the default algorithm enum.  -1 means not supported.
    int default_value = -1;

    // Human-readable description string (used in --help / option descriptions).
    std::string description{"No algorithm supported in selected cusparse version"};

    // All valid algorithm integer values for this version range.
    std::vector<int> supported_values;
};

// Loads and queries a YAML file that describes which cuSPARSE routines are
// supported for different CUDA version ranges, and what algorithm choices are
// available for each generic sparse operation.
//
// Typical usage – routine support:
//   bool ok = CusparseRoutineSupport::instance().is_supported("axpyi", CUDART_VERSION);
//
// Typical usage – algorithm support:
//   auto entry = CusparseRoutineSupport::instance().get_algorithm_entry("spmm", CUDART_VERSION);
//   hipsparseSpMMAlg_t alg = static_cast<hipsparseSpMMAlg_t>(entry.default_value);
//
// The YAML file is located at startup using the following search order:
//   1. HIPSPARSE_CUSPARSE_SUPPORT_FILE  environment variable (full path)
//   2. cusparse_support.yaml  alongside the running executable
//   3. ../staging/cusparse_support.yaml  relative to the executable directory
//   4. HIPSPARSE_CUSPARSE_SUPPORT_FILE_PATH  (CMake-configured install path)
//
// If the file cannot be found a warning is printed and all routines are treated
// as supported (safe default that avoids silently skipping tests).
class CusparseRoutineSupport
{
public:
    // ── Routine-support queries ───────────────────────────────────────────

    // Load support data from a YAML file.  Returns true on success.
    bool load(const std::string& filepath);

    // Returns true if the named routine is supported for the given CUDART_VERSION.
    // Routines absent from the file default to supported.
    bool is_supported(const std::string& routine, int cuda_version) const;

    // Returns a short human-readable description of the supported version range,
    // e.g. "CUDA < 12.0.0"  or  "CUDA >= 11.3.0".
    std::string get_support_range_string(const std::string& routine) const;

    // Returns a multi-line warning message describing why the routine is not
    // supported for cuda_version and what the supported range is.
    std::string get_support_warning(const std::string& routine, int cuda_version) const;

    // ── Algorithm-support queries ─────────────────────────────────────────

    // Returns the full AlgorithmEntry (default value, description, supported
    // values) for the named operation and CUDA version.
    // Pass cuda_version < 0 to request the ROCm / non-CUDA entry.
    // Returns a "not supported" default entry when the operation is not found.
    AlgorithmEntry get_algorithm_entry(const std::string& alg_type, int cuda_version) const;

    // Convenience wrappers around get_algorithm_entry().
    int get_algorithm_default(const std::string& alg_type, int cuda_version) const;
    std::string get_algorithm_description(const std::string& alg_type, int cuda_version) const;
    std::vector<int> get_algorithm_supported_values(const std::string& alg_type,
                                                     int               cuda_version) const;

    // ── Singleton ────────────────────────────────────────────────────────

    // Returns the process-wide singleton.  The YAML file is located and loaded
    // on the first call.
    static const CusparseRoutineSupport& instance();

    bool is_loaded() const
    {
        return loaded_;
    }

private:
    // ── Routine-support internals ─────────────────────────────────────────
    std::unordered_map<std::string, CudaVersionConstraint> routines_;

    // ── Algorithm-support internals ───────────────────────────────────────
    struct AlgorithmCudaRange
    {
        std::optional<int> min_cuda_version;
        std::optional<int> max_cuda_version;
        AlgorithmEntry     entry;

        bool matches(int cuda_version) const
        {
            if(min_cuda_version.has_value() && cuda_version < *min_cuda_version)
                return false;
            if(max_cuda_version.has_value() && cuda_version >= *max_cuda_version)
                return false;
            return true;
        }
    };

    struct AlgorithmSupport
    {
        AlgorithmEntry               rocm_entry;
        std::vector<AlgorithmCudaRange> cuda_ranges;
    };

    std::unordered_map<std::string, AlgorithmSupport> algorithm_support_;

    // ── Shared internals ──────────────────────────────────────────────────
    bool loaded_ = false;

    static std::string find_support_file();
    static std::string format_cuda_version(int v);
};
