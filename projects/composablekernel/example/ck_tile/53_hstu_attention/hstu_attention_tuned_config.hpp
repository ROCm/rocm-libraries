// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Tuned-config override table for the no-softmax jagged forward path.
//
// A small CSV maps a deployment config (dtype, causal, B, H, N, D) to a
// specific suggested kernel tile signature. At dispatch time the table is
// consulted BEFORE the legacy heuristic (get_hstu_attention_fwd_mtile /
// shall_use_splitkv): on an exact match whose tile signature corresponds to a
// kernel that is actually compiled into the binary, that instance is launched;
// on any miss (no CSV, no matching row, or an uncompiled signature) the call
// falls through to the unchanged legacy heuristic. A missing/empty CSV
// therefore reproduces today's behavior byte-for-byte.
//
// CSV columns (one suggested kernel per config):
//   dtype,causal,B,H,Nmin,D,mtile,kn0,kn0sub,kn1,kk1,splitkv
// Match rule: exact on (dtype, causal, B, H, D) and a THRESHOLD on N, i.e. the
// row fires when the runtime max_seqlen_q >= Nmin. A threshold (not an exact N)
// is required because for jagged/sparse inputs the kernel's max_seqlen_q is
// data-dependent (e.g. ~15.5k for a nominal-16384 sparse deployment, since the
// example binary derives it from max(uih)+max_target and callers do not pass
// -max_seqlen). When several rows match the same (dtype,causal,B,H,D), the one
// with the largest Nmin wins (most specific). The tile columns use the same
// "0 = use base dim" convention as the codegen
// (dispatcher/codegen/hstu/instance_gen.py). `#` lines and blank lines are ignored.
//
// Path resolution (first hit wins):
//   1. $HSTU_TUNED_CONFIG
//   2. <dir-of-executable>/hstu_tuned.csv
//   3. none -> pure legacy heuristic

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <limits.h>
#include <unistd.h>
#endif

namespace ck_tile {
namespace hstu_tuned {

// Runtime-selectable tile signature decoded from a CSV row. The tile fields use
// the "0 = base dim" override convention (matches the JIT instance names, e.g.
// the unified winner `..._mtile64_splitkv0_std_n032...` is {mtile=64, kn0=32,
// rest 0, splitkv=false}).
struct TileSig
{
    int mtile   = 0;
    int kn0     = 0;
    int kn0sub  = 0;
    int kn1     = 0;
    int kk1     = 0;
    bool splitkv = false;
};

struct Key
{
    std::string dtype; // "bf16" | "fp16"
    int causal = 0;
    int b      = 0;
    int h      = 0;
    int n_min  = 0; // minimum runtime max_seqlen_q for this row to fire
    int d      = 0;
};

class TunedConfig
{
    public:
    TunedConfig() { load(); }

    // Exact on (dtype, causal, B, H, D); threshold on N (max_seqlen_q >= n_min).
    // Among matching rows the largest n_min wins (most specific). Returns nullptr
    // when no config was loaded or no row matches; the caller then uses the
    // legacy heuristic.
    const TileSig* lookup(const char* dtype, int causal, int b, int h, int max_seqlen_q, int d) const
    {
        if(rows_.empty() || dtype == nullptr)
            return nullptr;
        const TileSig* best = nullptr;
        int best_n_min      = -1;
        for(const auto& r : rows_)
        {
            const Key& k = r.first;
            if(k.causal == causal && k.b == b && k.h == h && k.d == d && k.dtype == dtype &&
               max_seqlen_q >= k.n_min && k.n_min > best_n_min)
            {
                best       = &r.second;
                best_n_min = k.n_min;
            }
        }
        return best;
    }

    bool loaded() const { return !rows_.empty(); }

    private:
    std::vector<std::pair<Key, TileSig>> rows_;

    static std::string trim(const std::string& s)
    {
        const char* ws = " \t\r\n";
        const auto a   = s.find_first_not_of(ws);
        if(a == std::string::npos)
            return "";
        const auto b = s.find_last_not_of(ws);
        return s.substr(a, b - a + 1);
    }

    static std::string resolve_path()
    {
        if(const char* env = std::getenv("HSTU_TUNED_CONFIG"))
        {
            if(env[0] != '\0')
                return std::string(env);
        }
#if defined(__linux__)
        char buf[PATH_MAX];
        const ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if(len > 0)
        {
            buf[len]              = '\0';
            std::string exe       = buf;
            const auto slash      = exe.find_last_of('/');
            const std::string dir = (slash == std::string::npos) ? "." : exe.substr(0, slash);
            return dir + "/hstu_tuned.csv";
        }
#endif
        return "";
    }

    void load()
    {
        const std::string path = resolve_path();
        if(path.empty())
            return;
        std::ifstream f(path);
        if(!f.is_open())
            return;

        std::string line;
        while(std::getline(f, line))
        {
            const std::string s = trim(line);
            if(s.empty() || s[0] == '#')
                continue;

            std::vector<std::string> cols;
            std::stringstream ss(s);
            std::string tok;
            while(std::getline(ss, tok, ','))
                cols.push_back(trim(tok));
            if(cols.size() < 12)
                continue;

            // dtype guard: skip non-numeric header rows / unsupported dtypes.
            if(cols[0] != "bf16" && cols[0] != "fp16")
                continue;

            auto as_int = [](const std::string& v, bool& ok) {
                ok = !v.empty();
                char* end = nullptr;
                const long r = std::strtol(v.c_str(), &end, 10);
                if(end == v.c_str() || *end != '\0')
                    ok = false;
                return static_cast<int>(r);
            };

            bool ok = true, t = true;
            Key key;
            key.dtype  = cols[0];
            key.causal = as_int(cols[1], t); ok &= t;
            key.b      = as_int(cols[2], t); ok &= t;
            key.h      = as_int(cols[3], t); ok &= t;
            key.n_min  = as_int(cols[4], t); ok &= t;
            key.d      = as_int(cols[5], t); ok &= t;

            TileSig sig;
            sig.mtile  = as_int(cols[6], t); ok &= t;
            sig.kn0    = as_int(cols[7], t); ok &= t;
            sig.kn0sub = as_int(cols[8], t); ok &= t;
            sig.kn1    = as_int(cols[9], t); ok &= t;
            sig.kk1    = as_int(cols[10], t); ok &= t;
            sig.splitkv = as_int(cols[11], t) != 0; ok &= t;

            if(!ok)
                continue;
            rows_.emplace_back(std::move(key), sig);
        }
    }
};

// Process-wide singleton; the CSV is parsed once on first use. C++11 guarantees
// thread-safe initialization of the function-local static.
inline const TunedConfig& config()
{
    static const TunedConfig cfg;
    return cfg;
}

} // namespace hstu_tuned
} // namespace ck_tile
