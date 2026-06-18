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
#include <fstream>
#include <iostream>
#include <sstream>

// ── Minimal YAML parser ───────────────────────────────────────────────────────
//
// Parses the specific subset of YAML used by cusparse_support.yaml without any
// third-party dependency.  Supported constructs:
//
//   top-level keys, indented block mappings, block sequences (- item),
//   inline mappings { k: v, k: v }, inline integer sequences [1, 2, 3],
//   double-quoted strings, integer scalars, inline # comments.
//
// Indentation is always 2 spaces per nesting level (as written in the file).

namespace
{

static std::string trim_str(const std::string& s)
{
    const size_t b = s.find_first_not_of(" \t\r\n");
    if(b == std::string::npos)
        return {};
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static int leading_spaces(const std::string& line)
{
    int n = 0;
    for(char c : line)
    {
        if(c == ' ')
            ++n;
        else
            break;
    }
    return n;
}

// Split "key: everything-after-first-colon" on the FIRST colon only.
static bool split_kv(const std::string& s, std::string& key, std::string& val)
{
    const size_t pos = s.find(':');
    if(pos == std::string::npos)
        return false;
    key = trim_str(s.substr(0, pos));
    val = trim_str(s.substr(pos + 1));
    return true;
}

// Strip an inline # comment, respecting double-quoted strings.
static std::string strip_comment(const std::string& line)
{
    bool in_quote = false;
    for(size_t i = 0; i < line.size(); ++i)
    {
        if(line[i] == '"')
            in_quote = !in_quote;
        if(!in_quote && line[i] == '#' && (i == 0 || std::isspace((unsigned char)line[i - 1])))
            return line.substr(0, i);
    }
    return line;
}

// Parse "{ k: v, k: v }" into a flat string->string map.
// Only used for the simple integer-valued inline maps in the routines section.
static std::unordered_map<std::string, std::string> parse_inline_map(const std::string& s)
{
    std::unordered_map<std::string, std::string> m;
    const size_t a = s.find('{'), b = s.rfind('}');
    if(a == std::string::npos || b == std::string::npos || b <= a)
        return m;
    std::istringstream ss(s.substr(a + 1, b - a - 1));
    std::string token;
    while(std::getline(ss, token, ','))
    {
        std::string k, v;
        if(split_kv(trim_str(token), k, v))
            m[k] = v;
    }
    return m;
}

// Parse "[1, 2, 3]" into a vector<int>.
static std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> result;
    const size_t a = s.find('['), b = s.rfind(']');
    if(a == std::string::npos || b == std::string::npos || b <= a)
        return result;
    std::istringstream ss(s.substr(a + 1, b - a - 1));
    std::string tok;
    while(std::getline(ss, tok, ','))
    {
        tok = trim_str(tok);
        if(!tok.empty())
            try
            {
                result.push_back(std::stoi(tok));
            }
            catch(...)
            {
            }
    }
    return result;
}

// Remove surrounding double-quotes.
static std::string unquote(const std::string& s)
{
    if(s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

// Apply a key/value pair to an AlgorithmEntry.
static void apply_entry_field(const std::string& key,
                               const std::string& val,
                               AlgorithmEntry&    entry)
{
    if(key == "default_value")
        try
        {
            entry.default_value = std::stoi(val);
        }
        catch(...)
        {
        }
    else if(key == "description")
        entry.description = unquote(val);
    else if(key == "supported_values")
        entry.supported_values = parse_int_list(val);
}

} // namespace

// ── Loading ───────────────────────────────────────────────────────────────────

bool CusparseRoutineSupport::load(const std::string& filepath)
{
    std::ifstream file(filepath);
    if(!file.is_open())
    {
        std::cerr << "Warning: cannot open cusparse support file '" << filepath << "'\n";
        return false;
    }

    // Parser state machine.
    // The file has a fixed two-space-per-level indentation; we key off indent
    // depth to know which section/sub-section we are currently inside.
    enum class Ctx
    {
        Top,        // between top-level sections
        Routines,   // inside routines:
        AlgSupport, // inside algorithm_support:
        AlgType,    // inside a named algorithm entry
        Rocm,       // inside rocm: block
        CudaRanges, // inside cuda_version_ranges: list
        CudaRange   // inside one - item of that list
    };

    Ctx        ctx          = Ctx::Top;
    std::string alg_type;
    AlgorithmCudaRange pending_range;
    bool               has_pending = false;

    // Flush the accumulated range item into the current algorithm's list.
    auto flush_pending = [&]() {
        if(has_pending)
        {
            algorithm_support_[alg_type].cuda_ranges.push_back(std::move(pending_range));
            pending_range  = {};
            has_pending    = false;
        }
    };

    try
    {
        std::string raw;
        while(std::getline(file, raw))
        {
            if(!raw.empty() && raw.back() == '\r')
                raw.pop_back();

            std::string line = strip_comment(raw);
            if(trim_str(line).empty())
                continue;

            const int   ind = leading_spaces(line);
            std::string t   = trim_str(line);
            std::string key, val;

            // ── Top-level section header (indent 0) ──────────────────────
            if(ind == 0)
            {
                flush_pending();
                if(t == "routines:")
                    ctx = Ctx::Routines;
                else if(t == "algorithm_support:")
                    ctx = Ctx::AlgSupport;
                else
                    ctx = Ctx::Top;
                continue;
            }

            // ── routines: entries (indent 2) ─────────────────────────────
            if(ctx == Ctx::Routines && ind == 2)
            {
                if(!split_kv(t, key, val))
                    continue;
                CudaVersionConstraint c;
                if(!val.empty() && val != "{}")
                {
                    auto m = parse_inline_map(val);
                    if(m.count("min_cuda_version"))
                        try { c.min_version = std::stoi(m["min_cuda_version"]); } catch(...) {}
                    if(m.count("max_cuda_version"))
                        try { c.max_version = std::stoi(m["max_cuda_version"]); } catch(...) {}
                }
                routines_[key] = c;
                continue;
            }

            // ── algorithm_support: sub-structure ─────────────────────────
            if(ctx == Ctx::AlgSupport || ctx == Ctx::AlgType || ctx == Ctx::Rocm
               || ctx == Ctx::CudaRanges || ctx == Ctx::CudaRange)
            {
                if(ind == 2) // algorithm type name, e.g. "  spmm:"
                {
                    flush_pending();
                    if(!split_kv(t, key, val))
                        continue;
                    alg_type                    = key;
                    algorithm_support_[alg_type] = {};
                    ctx                         = Ctx::AlgType;
                    continue;
                }

                if(ind == 4) // "    rocm:" or "    cuda_version_ranges:"
                {
                    flush_pending();
                    if(t == "rocm:")
                        ctx = Ctx::Rocm;
                    else if(t == "cuda_version_ranges:")
                        ctx = Ctx::CudaRanges;
                    continue;
                }

                if(ind == 6)
                {
                    if(ctx == Ctx::Rocm) // fields of the rocm: block
                    {
                        if(split_kv(t, key, val))
                            apply_entry_field(key, val, algorithm_support_[alg_type].rocm_entry);
                    }
                    else if(ctx == Ctx::CudaRanges || ctx == Ctx::CudaRange)
                    {
                        // Sequence item: "      - key: val"
                        if(t.size() >= 2 && t[0] == '-' && t[1] == ' ')
                        {
                            flush_pending();
                            pending_range = {};
                            has_pending   = true;
                            ctx           = Ctx::CudaRange;

                            // Parse the field that appears on the same line as "-"
                            const std::string rest = trim_str(t.substr(2));
                            if(!rest.empty() && split_kv(rest, key, val))
                            {
                                if(key == "min_cuda_version")
                                    try { pending_range.min_cuda_version = std::stoi(val); } catch(...) {}
                                else if(key == "max_cuda_version")
                                    try { pending_range.max_cuda_version = std::stoi(val); } catch(...) {}
                                else
                                    apply_entry_field(key, val, pending_range.entry);
                            }
                        }
                    }
                    continue;
                }

                if(ind == 8 && ctx == Ctx::CudaRange) // continuation fields of a range item
                {
                    if(split_kv(t, key, val))
                    {
                        if(key == "min_cuda_version")
                            try { pending_range.min_cuda_version = std::stoi(val); } catch(...) {}
                        else if(key == "max_cuda_version")
                            try { pending_range.max_cuda_version = std::stoi(val); } catch(...) {}
                        else
                            apply_entry_field(key, val, pending_range.entry);
                    }
                    continue;
                }
            }
        }

        flush_pending();
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Warning: error parsing '" << filepath << "': " << ex.what() << '\n';
        return false;
    }

    loaded_ = (!routines_.empty() || !algorithm_support_.empty());
    return loaded_;
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
