// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Stage-2 Dispatcher: maps a Problem to the best registered candidate's
// cache_key. This is the runtime (C++) twin of ck_dsl.dispatch; the shared
// identity is cache_key. It is a thin selector over an ArtifactStore (the
// per-arch candidate set), NOT a monolith that hides compile/launch.
//
// v1 selection = FirstFit with a divisibility + tile-size preference, mirroring
// CandidateRegistry.select(priority). The intent is that the offline Python
// dispatcher and this C++ dispatcher pick the same cache_key for a problem; a
// CI parity test (planned) pins that.
#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "ck_dsl_runtime/artifact_store.hpp"

namespace ck_dsl {

struct Problem {
    std::string op;      // "gemm", "conv", "attention", ...
    std::string dtype;   // "fp16", "bf16", "fp8", ...
    std::string layout;  // "RCR", ...
    std::string arch;    // "gfx950" (bundles are per-arch; informational here)
    long M = 0, N = 0, K = 0;
    // Attention-specific dims (set by the attention parser; used by the FMHA
    // ML feature extractor).
    long batch = 0, nhead_q = 0, nhead_k = 0, seqlen_q = 0, seqlen_k = 0, hdim_q = 0, hdim_v = 0;
    int mask_type = 0;  // 0=none,1=causal/top_left,...
    bool use_sinks = false;
    // Conv-specific dims (set by the conv parser; used by the conv ML feature
    // extractor). Zero when op != "conv".
    long conv_N = 0, conv_C = 0, conv_K = 0, conv_G = 1;
    long Hi = 0, Wi = 0, Y = 0, X = 0;
    int stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0, dilation_h = 1, dilation_w = 1;
};

// Selection strategy, mirroring the CK Tile dispatcher: FirstFit (priority /
// largest-tile heuristic) or Heuristic (a pluggable ranker, e.g. a trained ML
// model). When a heuristic is installed the strategy switches to Heuristic.
class Dispatcher {
   public:
    explicit Dispatcher(const ArtifactStore& store) : store_(store) {}

    struct Choice {
        std::string cache_key;
        int block_m = 0, block_n = 0, block_k = 0;
        bool valid() const {
            return !cache_key.empty();
        }
    };

    enum class Strategy { FirstFit, Heuristic };

    // A heuristic reorders the supported candidates (best first). Installing one
    // switches selection to Strategy::Heuristic. Mirrors Dispatcher::set_heuristic.
    using HeuristicFn = std::function<std::vector<Choice>(const Problem&, std::vector<Choice>)>;
    void set_heuristic(HeuristicFn h) {
        heuristic_ = std::move(h);
        strategy_ = Strategy::Heuristic;
    }
    void set_strategy(Strategy s) {
        strategy_ = s;
    }
    Strategy strategy() const {
        return strategy_;
    }

    // Return candidates that support the problem, best first (FirstFit order).
    std::vector<Choice> rank(const Problem& p) const {
        std::vector<Choice> out;
        for (const auto& kv : store_.entries()) {
            const Manifest& m = kv.second.manifest;
            if (!kind_matches(m.kind, p.op)) continue;
            if (!supports_shape(m, p)) continue;
            out.push_back({kv.first, m.block_m, m.block_n, m.block_k});
        }
        // Prefer the largest CTA tile (block_m*block_n) that fits — a simple,
        // deterministic FirstFit-style heuristic. Ties broken by cache_key for
        // stability.
        std::sort(out.begin(), out.end(), [](const Choice& a, const Choice& b) {
            long ta = (long)a.block_m * a.block_n, tb = (long)b.block_m * b.block_n;
            if (ta != tb) return ta > tb;
            return a.cache_key < b.cache_key;
        });
        return out;
    }

    Choice select(const Problem& p) const {
        auto r = rank(p);  // supported candidates in FirstFit order
        if (r.empty()) return Choice{};
        if (strategy_ == Strategy::Heuristic && heuristic_) r = heuristic_(p, std::move(r));
        return r.empty() ? Choice{} : r.front();
    }

   private:
    Strategy strategy_ = Strategy::FirstFit;
    HeuristicFn heuristic_;
    static bool kind_matches(const std::string& kind, const std::string& op) {
        // "gemm_fp16" -> op "gemm"; "conv_fp16" -> "conv"; "attention_unified" -> "attention"
        return kind.rfind(op, 0) == 0;
    }
    static bool supports_shape(const Manifest& m, const Problem& p) {
        if (p.op == "gemm") {
            if (m.block_m <= 0 || m.block_n <= 0 || m.block_k <= 0) return false;
            // No-padding kernels require exact divisibility.
            return (p.M % m.block_m == 0) && (p.N % m.block_n == 0) && (p.K % m.block_k == 0);
        }
        if (p.op == "conv") {
            // Baked conv kernels have fixed problem dims encoded in manifest["conv"]
            // as [N, Hi, Wi, C, K, R, S, sH, sW, pH, pW, dH, dW]. If the array is
            // absent the kernel is shape-generic (C-JIT path); accept all shapes.
            if (!m.raw.has("conv")) return true;
            const auto& arr = m.raw.at("conv").as_array();
            if (arr.size() < 13) return true;
            return p.conv_N == arr[0].as_int() && p.Hi == arr[1].as_int() &&
                   p.Wi == arr[2].as_int() && p.conv_C == arr[3].as_int() &&
                   p.conv_K == arr[4].as_int() && p.Y == arr[5].as_int() &&
                   p.X == arr[6].as_int() && p.stride_h == arr[7].as_int() &&
                   p.stride_w == arr[8].as_int() && p.pad_h == arr[9].as_int() &&
                   p.pad_w == arr[10].as_int() && p.dilation_h == arr[11].as_int() &&
                   p.dilation_w == arr[12].as_int() &&
                   p.conv_G == m.raw.get_int("groups", 1);
        }
        return true;  // other ops: shape support refined per-engine
    }

    const ArtifactStore& store_;
};

}  // namespace ck_dsl
