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
    // Attention path-selection inputs (mirror UnifiedAttentionProblem).
    long total_q = 0, num_seqs = 0;  // total_q = batch*seqlen_q; num_seqs = batch
    long block_kv = 16;              // paged-KV block_size modulus {16,32,64}
    long sliding_window = 0;
    long num_sms = 120;
    // Conv-specific dims (set by the conv parser; used by the conv ML feature
    // extractor). Zero when op != "conv".
    long conv_N = 0, conv_C = 0, conv_K = 0, conv_G = 1;
    long Hi = 0, Wi = 0, Y = 0, X = 0;
    int stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0, dilation_h = 1, dilation_w = 1;
    // Norm-specific dims (set by the norm parser). rows == M, cols == per-row N.
    long rows = 0, cols = 0;
    std::string kind;  // norm: "rmsnorm"|"layernorm"; gemm/conv: unused
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
        int priority = 1000000;  // Python registry priority (lower == preferred)
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
            if (!dtype_matches(m.kind, p.dtype)) continue;
            if (!arch_family_matches(p.arch, m)) continue;
            if (!supports_shape(m, p)) continue;
            int prio =
                m.raw.has("priority") ? static_cast<int>(m.raw.get_int("priority")) : 1000000;
            out.push_back({kv.first, m.block_m, m.block_n, m.block_k, prio});
        }
        // Norm has no MMA tile: rank by block_size (block_m) first, then vec
        // (block_n), mirroring the Python norm registry priority (largest
        // block_size, then widest vec). This must be a strict total order, so we
        // fall through to cache_key for stability.
        if (p.op == "norm") {
            std::sort(out.begin(), out.end(), [](const Choice& a, const Choice& b) {
                if (a.block_m != b.block_m) return a.block_m > b.block_m;  // block_size
                if (a.block_n != b.block_n) return a.block_n > b.block_n;  // vec
                return a.cache_key < b.cache_key;
            });
            return out;
        }
        // Conv (and any kind that carries an explicit "priority") ranks by the
        // Python registry priority (lower value == higher priority), then by the
        // largest CTA tile, then cache_key. This mirrors CandidateRegistry.select,
        // which sorts candidates by (priority, name) and keeps the first
        // supported one. Conv's cshuffle/mem tiles can have equal block_m*block_n,
        // so priority is the load-bearing tiebreak the generic GEMM ordering lacks.
        if (p.op == "conv") {
            std::sort(out.begin(), out.end(), [](const Choice& a, const Choice& b) {
                if (a.priority != b.priority) return a.priority < b.priority;
                long ta = (long)a.block_m * a.block_n, tb = (long)b.block_m * b.block_n;
                if (ta != tb) return ta > tb;
                return a.cache_key < b.cache_key;
            });
            return out;
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
    // dtype gate, mirroring the Python per-dtype family split
    // (dispatch_gemm_fp16 vs dispatch_gemm_bf16). The manifest "kind" encodes the
    // dtype as a suffix ("gemm_fp16", "gemm_bf16"); a request dtype must match it
    // so a bf16 problem never picks an fp16 kernel (and vice versa). Empty request
    // dtype is treated as "match anything" for callers that don't set it.
    static bool dtype_matches(const std::string& kind, const std::string& dtype) {
        if (dtype.empty()) return true;
        // Only enforce when the kind carries a dtype token we recognize.
        for (const char* dt : {"fp16", "bf16", "fp8", "bf8", "fp32"}) {
            bool kind_has = kind.find(dt) != std::string::npos;
            if (kind_has) return dtype == dt;
        }
        return true;  // dtype-agnostic kind (e.g. "attention_unified")
    }
    // Arch-family gate, mirroring Python's arch_family_supported. Bundles are
    // per-arch so this is normally a no-op, but it keeps the C++ predicate
    // consistent with Python: a kernel built for a CDNA arch must not be selected
    // for an RDNA problem (and vice versa). The manifest records its build arch in
    // the retained raw JSON ("arch") when available; absent that we accept.
    static bool arch_family_matches(const std::string& prob_arch, const Manifest& m) {
        if (prob_arch.empty()) return true;
        std::string kern_arch;
        if (m.raw.has("arch")) kern_arch = m.raw.get_str("arch");
        if (kern_arch.empty()) return true;  // manifest didn't record its arch
        return arch_family(prob_arch) == arch_family(kern_arch);
    }
    // gfx9xx -> "cdna", gfx10xx/gfx11xx/gfx12xx -> "rdna". Mirrors the
    // ArchTarget.family split used by the Python dispatcher.
    static std::string arch_family(const std::string& gfx) {
        // expects "gfxNNNN"; classify on the first numeric digit after "gfx".
        auto pos = gfx.find("gfx");
        if (pos == std::string::npos || gfx.size() < pos + 4) return "";
        char major = gfx[pos + 3];
        if (major == '9') return "cdna";
        if (major == '1') return "rdna";  // gfx10xx/11xx/12xx
        return "";
    }
    static bool supports_shape(const Manifest& m, const Problem& p) {
        if (p.op == "gemm") {
            if (m.block_m <= 0 || m.block_n <= 0 || m.block_k <= 0) return false;
            // No-padding kernels require exact divisibility.
            return (p.M % m.block_m == 0) && (p.N % m.block_n == 0) && (p.K % m.block_k == 0);
        }
        if (p.op == "conv") {
            // Shape-generic implicit-GEMM kernels (the dispatch-parity bundle):
            // the kernel is tiled but not baked to one problem. Mirror the Python
            // conv family's no-padding divisibility on the DERIVED implicit-GEMM
            // dims: M = N*Ho*Wo, N_gemm = K, K_gemm = Y*X*C. block_m/n/k must
            // divide them exactly. (Ho/Wo are derived from the conv geometry.)
            if (m.block_m > 0 && m.block_n > 0 && m.block_k > 0 && !m.raw.has("conv")) {
                long Ho = (p.Hi + 2 * p.pad_h - p.dilation_h * (p.Y - 1) - 1) / p.stride_h + 1;
                long Wo = (p.Wi + 2 * p.pad_w - p.dilation_w * (p.X - 1) - 1) / p.stride_w + 1;
                if (Ho <= 0 || Wo <= 0) return false;
                long Mg = p.conv_N * Ho * Wo;
                long Ng = p.conv_K;
                long Kg = (long)p.Y * p.X * p.conv_C;
                return (Mg % m.block_m == 0) && (Ng % m.block_n == 0) && (Kg % m.block_k == 0);
            }
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
                   p.dilation_w == arr[12].as_int() && p.conv_G == m.raw.get_int("groups", 1);
        }
        if (p.op == "norm") {
            // Mirror ck_dsl.instances.common.{rmsnorm2d,layernorm2d}.is_valid_spec:
            // kind match, n_per_block % (block_size*vec)==0, elems_per_thread cap
            // (single-pass path only), and LDS capacity. block_m carries
            // block_size; "vec" lives in raw JSON; "norm_kind" disambiguates
            // rmsnorm vs layernorm.
            if (!p.kind.empty() && m.raw.has("norm_kind") && m.raw.get_str("norm_kind") != p.kind)
                return false;
            long block_size = m.block_m;
            long vec = m.raw.has("vec") ? m.raw.get_int("vec") : m.block_n;
            if (block_size <= 0 || vec <= 0) return false;
            long chunk = block_size * vec;
            if (p.cols % chunk != 0) return false;
            // elems_per_thread = cols / block_size; cap mirrors
            // REGISTER_TILE_MAX_ELEMS_PER_THREAD only on the single-pass path.
            // The two-pass path (elems > two_pass_threshold) is uncapped. Both
            // thresholds are carried in the manifest so the C++ side stays in
            // lockstep with the Python validator without hardcoding constants.
            long elems = p.cols / block_size;
            long cap =
                m.raw.has("max_elems_per_thread") ? m.raw.get_int("max_elems_per_thread") : 0;
            long two_pass =
                m.raw.has("two_pass_threshold") ? m.raw.get_int("two_pass_threshold") : 0;
            if (cap > 0 && two_pass > 0 && elems <= two_pass && elems > cap) return false;
            return true;
        }
        if (p.op == "moe") {
            // Element-path gate mirroring the Python MoE family: f16/bf16 ->
            // the f16 mega-kernel manifest, fp8 -> the fp8 block-scale manifest.
            // The manifest records its path in "moe_path" ("f16"|"fp8"). Arch
            // coverage (gfx950-only) is handled by the bundle being synthesized
            // only for supported arches, so an unsupported arch has an empty
            // moe bundle and selects nothing.
            std::string mpath = m.raw.has("moe_path") ? m.raw.get_str("moe_path") : "";
            if (mpath.empty()) return true;
            bool is_fp8 =
                (p.dtype == "fp8" || p.dtype == "fp8e4m3" || p.dtype == "f8" || p.dtype == "e4m3");
            return is_fp8 ? (mpath == "fp8") : (mpath == "f16");
        }
        if (p.op == "attention") {
            // Mirror UnifiedAttentionProblem.select_path (pure) +
            // supports_native_unified_attention (pure). The manifest carries the
            // kernel "path" ("2d"|"3d") and head_size/block_size; we accept the
            // manifest whose path matches where this problem routes, and whose
            // (head_size, block_size, dtype) are in the native-backend coverage.
            // dtype coverage:
            if (p.dtype != "fp16" && p.dtype != "bf16") return false;
            long hd = p.hdim_q;
            if (hd != 64 && hd != 128 && hd != 256) return false;
            // block_size here is the paged-KV block_size (modulus). The harness
            // passes it via Problem::block_kv (defaulted to 16). Coverage {16,32,64}.
            long bs = p.block_kv;
            if (bs != 16 && bs != 32 && bs != 64) return false;
            if (p.nhead_k <= 0 || p.nhead_q % p.nhead_k != 0) return false;
            if (p.hdim_q != p.hdim_v) return false;
            // manifest path gate
            std::string mpath = m.raw.has("path") ? m.raw.get_str("path") : "";
            long mhd = m.block_m;  // head_size stored in block_m
            long mbs = m.block_n;  // block_size stored in block_n
            if (mhd != 0 && mhd != hd) return false;
            if (mbs != 0 && mbs != bs) return false;
            // select_path (pure):
            long nqpkv = p.nhead_q / p.nhead_k;
            long block_m_sel = (nqpkv <= 16) ? 16 : next_pow2(nqpkv);
            long block_q = block_m_sel / nqpkv;
            if (block_q <= 0) block_q = 1;
            long total_q = p.total_q > 0 ? p.total_q : (long)p.seqlen_q * 1;
            long num_seqs = p.num_seqs > 0 ? p.num_seqs : 1;
            long total_num_q_blocks_ub = total_q / block_q + num_seqs;
            long num_2d = total_num_q_blocks_ub * p.nhead_k;
            long target = (long)p.num_sms * 4;
            bool is_2d = (p.sliding_window > 0) || (p.seqlen_k <= 512) || (num_2d > target);
            std::string want = is_2d ? "2d" : "3d";
            return mpath.empty() || mpath == want;
        }
        return true;  // other ops: shape support refined per-engine
    }
    static long next_pow2(long x) {
        if (x <= 1) return 1;
        long p = 1;
        long v = x - 1;
        int bits = 0;
        while (v > 0) {
            v >>= 1;
            ++bits;
        }
        (void)p;
        return 1L << bits;
    }

    const ArtifactStore& store_;
};

}  // namespace ck_dsl
