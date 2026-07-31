/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "gemm_reject.hpp"
#include "utility.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    // Name of the environment variable that points to the rejection YAML file.
    constexpr const char* kRejectFileEnv = "HIPBLASLT_REJECT_GEMM_FILE";

    // A single rejection rule. Only fields that identify a GEMM (size + data
    // type) are stored; any unset field acts as a wildcard.
    struct RejectEntry
    {
        std::optional<int64_t>     m, n, k, batch_count;
        std::optional<char>        transA, transB;
        std::optional<std::string> a_type, b_type, c_type, d_type;
        std::optional<int>         compute_type; // rocblaslt_compute_type as int

        // Whether this entry carries the minimum information (M, N and K) that we
        // require before it may reject anything. This avoids an accidental
        // catch-all entry silently rejecting every GEMM.
        bool hasSize() const
        {
            return m.has_value() && n.has_value() && k.has_value();
        }
    };

    std::string trim(const std::string& s)
    {
        size_t b = 0, e = s.size();
        while(b < e && std::isspace(static_cast<unsigned char>(s[b])))
            ++b;
        while(e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
            --e;
        return s.substr(b, e - b);
    }

    std::string stripQuotes(std::string s)
    {
        if(s.size() >= 2 && ((s.front() == '"' && s.back() == '"')
                             || (s.front() == '\'' && s.back() == '\'')))
            return s.substr(1, s.size() - 2);
        return s;
    }

    std::string toLower(std::string s)
    {
        for(auto& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    // Map a compute-type token as written in the file (e.g. "c_f32_r", "f32_r")
    // to the corresponding rocblaslt_compute_type value. Returns std::nullopt for
    // tokens we do not understand, in which case the constraint is treated as a
    // wildcard.
    std::optional<int> computeTypeFromToken(std::string t)
    {
        t = toLower(trim(t));
        if(t.rfind("c_", 0) == 0)
            t = t.substr(2);
        if(t == "f16_r")
            return rocblaslt_compute_f16;
        if(t == "f32_r")
            return rocblaslt_compute_f32;
        if(t == "xf32_r")
            return rocblaslt_compute_f32_fast_xf32;
        if(t == "f64_r")
            return rocblaslt_compute_f64;
        if(t == "i32_r")
            return rocblaslt_compute_i32;
        return std::nullopt;
    }

    char normalizeOp(const std::string& v)
    {
        for(char c : v)
        {
            char u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            if(u == 'N' || u == 'T' || u == 'C')
                return u;
        }
        return 'N';
    }

    // Split a flow-map body ("key: value, key: value, ...") on top-level commas,
    // honoring quotes so that values such as "gfx950, gfx1250" stay intact.
    std::vector<std::string> splitTopLevel(const std::string& body)
    {
        std::vector<std::string> out;
        std::string              cur;
        char                     quote = 0;
        for(char c : body)
        {
            if(quote)
            {
                if(c == quote)
                    quote = 0;
                cur.push_back(c);
            }
            else if(c == '"' || c == '\'')
            {
                quote = c;
                cur.push_back(c);
            }
            else if(c == ',')
            {
                out.push_back(cur);
                cur.clear();
            }
            else
            {
                cur.push_back(c);
            }
        }
        if(!trim(cur).empty())
            out.push_back(cur);
        return out;
    }

    void assignField(RejectEntry& e, const std::string& rawKey, const std::string& rawVal)
    {
        const std::string key = toLower(trim(rawKey));
        const std::string val = stripQuotes(trim(rawVal));
        if(val.empty())
            return;

        auto asInt = [&](std::optional<int64_t>& dst) {
            try
            {
                dst = static_cast<int64_t>(std::stoll(val));
            }
            catch(...)
            {
            }
        };

        if(key == "m")
            asInt(e.m);
        else if(key == "n")
            asInt(e.n);
        else if(key == "k")
            asInt(e.k);
        else if(key == "batch_count")
            asInt(e.batch_count);
        else if(key == "transa")
            e.transA = normalizeOp(val);
        else if(key == "transb")
            e.transB = normalizeOp(val);
        else if(key == "a_type")
            e.a_type = toLower(val);
        else if(key == "b_type")
            e.b_type = toLower(val);
        else if(key == "c_type")
            e.c_type = toLower(val);
        else if(key == "d_type")
            e.d_type = toLower(val);
        else if(key == "compute_type")
            e.compute_type = computeTypeFromToken(val);
        // Every other key (lda/ldb/strides/alpha/beta/scale/bias/iter/rotating/
        // solution_index/kernel/...) is intentionally ignored.
    }

    std::optional<RejectEntry> parseFlowMap(const std::string& body)
    {
        RejectEntry e;
        for(const auto& pair : splitTopLevel(body))
        {
            const auto colon = pair.find(':');
            if(colon == std::string::npos)
                continue;
            assignField(e, pair.substr(0, colon), pair.substr(colon + 1));
        }
        return e.hasSize() ? std::optional<RejectEntry>(e) : std::nullopt;
    }

    // Parse the whole file. Each flow map is delimited by '{' ... '}'. This
    // handles both single-line entries ("- { ... }") and flow maps that span
    // multiple lines. Braces are not expected to nest in these files.
    std::vector<RejectEntry> parseFile(const std::string& path)
    {
        std::vector<RejectEntry> entries;
        std::ifstream            in(path);
        if(!in)
        {
            fprintf(stderr,
                    "hipBLASLt: %s is set to '%s' but the file could not be opened; "
                    "no GEMMs will be rejected.\n",
                    kRejectFileEnv,
                    path.c_str());
            return entries;
        }

        // Read the file line by line, dropping full-line comments (lines whose
        // first non-whitespace character is '#') so a commented-out entry is not
        // parsed or matched. Remaining lines are concatenated so that flow maps
        // may still span multiple lines.
        std::string content;
        std::string line;
        while(std::getline(in, line))
        {
            if(trim(line).rfind('#', 0) == 0)
                continue;
            content += line;
            content += '\n';
        }

        size_t pos = 0;
        while((pos = content.find('{', pos)) != std::string::npos)
        {
            const size_t end = content.find('}', pos);
            if(end == std::string::npos)
                break;
            if(auto e = parseFlowMap(content.substr(pos + 1, end - pos - 1)))
                entries.push_back(*e);
            pos = end + 1;
        }
        return entries;
    }

    const std::vector<RejectEntry>& rejectEntries()
    {
        static std::vector<RejectEntry> entries = [] {
            std::vector<RejectEntry> e;
            const char*              path = std::getenv(kRejectFileEnv);
            if(path && *path)
            {
                e = parseFile(path);
                fprintf(stderr,
                        "hipBLASLt: loaded %zu GEMM rejection rule(s) from %s='%s'.\n",
                        e.size(),
                        kRejectFileEnv,
                        path);
            }
            return e;
        }();
        return entries;
    }

    bool matches(const RejectEntry& e, const RocblasltContractionProblem& prob)
    {
        if(e.m && *e.m != static_cast<int64_t>(prob.m))
            return false;
        if(e.n && *e.n != static_cast<int64_t>(prob.n))
            return false;
        if(e.k && *e.k != static_cast<int64_t>(prob.k))
            return false;
        if(e.batch_count && *e.batch_count != static_cast<int64_t>(prob.batch_count))
            return false;

        auto opChar = [](hipblasOperation_t op) -> char {
            return op == HIPBLAS_OP_N ? 'N' : (op == HIPBLAS_OP_C ? 'C' : 'T');
        };
        if(e.transA && *e.transA != opChar(prob.trans_a))
            return false;
        if(e.transB && *e.transB != opChar(prob.trans_b))
            return false;

        if(e.a_type && *e.a_type != hipDataType_to_bench_string(prob.a_type))
            return false;
        if(e.b_type && *e.b_type != hipDataType_to_bench_string(prob.b_type))
            return false;
        if(e.c_type && *e.c_type != hipDataType_to_bench_string(prob.c_type))
            return false;
        if(e.d_type && *e.d_type != hipDataType_to_bench_string(prob.d_type))
            return false;

        if(e.compute_type && *e.compute_type != static_cast<int>(prob.compute_type))
            return false;

        return true;
    }
}

bool rocblaslt_gemm_is_rejected(const RocblasltContractionProblem& prob)
{
    const auto& entries = rejectEntries();
    if(entries.empty())
        return false;

    for(size_t i = 0; i < entries.size(); ++i)
    {
        if(matches(entries[i], prob))
        {
            fprintf(stderr,
                    "hipBLASLt: GEMM REJECTED size=[M=%zu N=%zu K=%zu batch=%zu] "
                    "transA=%c transB=%c types=[a=%s b=%s c=%s d=%s] "
                    "(matched rule #%zu from %s)\n",
                    prob.m,
                    prob.n,
                    prob.k,
                    prob.batch_count,
                    prob.trans_a == HIPBLAS_OP_N ? 'N' : (prob.trans_a == HIPBLAS_OP_C ? 'C' : 'T'),
                    prob.trans_b == HIPBLAS_OP_N ? 'N' : (prob.trans_b == HIPBLAS_OP_C ? 'C' : 'T'),
                    hipDataType_to_bench_string(prob.a_type),
                    hipDataType_to_bench_string(prob.b_type),
                    hipDataType_to_bench_string(prob.c_type),
                    hipDataType_to_bench_string(prob.d_type),
                    i,
                    kRejectFileEnv);
            return true;
        }
    }
    return false;
}
