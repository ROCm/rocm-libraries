// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// c_engine.hpp -- a GENERALIZED, header-only C++ bridge that turns the pure-C
// ck_dsl engine (libckc_core.a) into a runtime JIT source for the provider,
// uniform across op families.
//
// One shape (CEngineResult) and one pattern per op: a small POD problem in,
// a (.ll text + synthesized ck_dsl::Manifest + grid[3] + block) out. The
// caller never sees any op-specific C struct; the only op-specific surface is
// the POD problem it fills.
//
//   auto r = CEngine::build_gemm(p);                  // POD -> CEngineResult
//   Kernel k = Kernel::from_llvm_ir(r.llvm_ir, r.manifest, Compiler::isa_for(arch));
//   k.ensure_compiled();
//   k.launch(ptr_args, scalar_args, r.grid, r.block, stream);
//
// Each build_* calls the matching ckc_* build+lower entry (extern "C" from
//   ckc/ir.h + the per-op instance header), gets the .ll string + kernel_name,
// and SYNTHESIZES a ck_dsl::Manifest (schema "ck.dsl.example.manifest/v1",
// kernel_name, args_signature with the exact recon ABI, block_m/n/k,
// grid_order) plus the computed grid[3] + block.
//
// Build/link (header-only, links the C engine archive):
//   hipcc -std=c++17
//     -I <repo>/projects/composablekernel/python/ck_dsl_c/include   // ckc/*
//     -I <provider>/runtime/include                                 // ck_dsl_runtime/*
//     -I /opt/rocm/include
//     ... -L<libdir> -lckc_core -lamd_comgr
//
// A standalone self-test main (guarded by CKC_CENGINE_TEST) builds all three
// ops' .ll and prints sizes; it does NOT touch the GPU.

#pragma once

#include <array>
#include <stdexcept>
#include <string>

#include "ck_dsl_runtime/manifest.hpp"

// The pure-C ck_dsl engine. These are C99 TUs compiled into libckc_core.a; the
// declarations must be visible with C linkage.
extern "C" {
#include "ckc/instance_attention_unified.h"
#include "ckc/instance_conv_implicit_gemm.h"
#include "ckc/instance_gemm_universal.h"
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
}

namespace ck_dsl {

// --------------------------------------------------------------------------
// Uniform result of any C-engine build. Same shape for every op family.
//   llvm_ir  : malloc'd-then-copied AMDGPU LLVM IR text (NUL-free std::string)
//   manifest : synthesized ck.dsl.example.manifest/v1 (kernel_name + ABI + grid)
//   grid     : launch grid {x, y, z} already computed from the problem dims
//   block    : threads per block (== manifest.threads_per_block)
// --------------------------------------------------------------------------
struct CEngineResult {
    std::string llvm_ir;
    Manifest manifest;
    std::array<unsigned, 3> grid{1, 1, 1};
    unsigned block = 0;
};

// --------------------------------------------------------------------------
// CEngine: one static build_* per op family. The POD problem types are nested
// so callers never include the ckc/ headers themselves.
// --------------------------------------------------------------------------
struct CEngine {
    // ---- POD problems (the ONLY op-specific surface a caller touches) ----

    struct GemmProblem {
        int M = 0, N = 0, K = 0;
        // Tile geometry (recon defaults: 128x128x32, 2x2x1 warps, 32x32x16 atom).
        int tile_m = 128, tile_n = 128, tile_k = 32;
        int warp_m = 2, warp_n = 2, warp_k = 1;
        int warp_tile_m = 32, warp_tile_n = 32, warp_tile_k = 16;
        const char* pipeline = "compv4";      // mem|compv3|compv4|wsp3
        const char* scheduler = "intrawave";  // intrawave|interwave
        const char* epilogue = "cshuffle";    // default|cshuffle
        const char* dtype_a = "fp16";
        const char* dtype_b = "fp16";
        const char* dtype_c = "fp16";
        const char* dtype_acc = "fp32";
        const char* layout = "RCR";
        const char* arch = "gfx950";
    };

    struct ConvProblem {
        int N = 0, Hi = 0, Wi = 0, C = 0;
        int K = 0, R = 0, S = 0;
        int sH = 1, sW = 1, pH = 0, pW = 0, dH = 1, dW = 1;
        // implicit-GEMM tile (recon default 64x64x64, 2x2 warps, 32x32x16 atom).
        int tile_m = 64, tile_n = 64, tile_k = 64;
        int warp_m = 2, warp_n = 2;
        int warp_tile_m = 32, warp_tile_n = 32, warp_tile_k = 16;
        const char* pipeline = "mem";      // mem|compv3|compv4
        const char* epilogue = "default";  // default|cshuffle
        const char* arch = "gfx950";
    };

    struct SdpaProblem {
        int total_q = 0;   // B * S_q
        int num_seqs = 0;  // B
        int num_query_heads = 0;
        int num_kv_heads = 0;
        int head_size = 0;  // hdim_q (64|128|256)
        int block_size = 16;
        int max_seqlen_q = 0;
        int max_seqlen_k = 0;
        const char* dtype = "fp16";  // fp16|bf16 (scalar path)
        int sliding_window = 0;
        double softcap = 0.0;
        const char* arch = "gfx950";

        // ---- Heuristic-overlaid attention knobs (defaulted; existing callers
        // are unaffected). These mirror the FMHA manifest's attention_config
        // (block_q / tile_size / num_warps). On the scalar 2D reference path the
        // ONE knob that changes the emitted kernel is the KV block size; the
        // heuristic carries it as block_q (== block_size in attention_config).
        // apply_attn_knobs() folds block_q into block_size so the JIT'd kernel
        // identity (name + paged-KV block geometry) tracks the chosen candidate.
        // tile_size / num_warps are retained for parity with the manifest knob
        // space (they feed the tiled kernel, not the scalar reference, so they
        // are not consumed by the scalar lower below but ARE carried/applied so
        // the overlay is a faithful mirror of apply_gemm_knobs).
        int block_q = 0;    // 0 == not provided by manifest
        int tile_size = 0;  // 0 == not provided
        int num_warps = 0;  // 0 == not provided
    };

    // ---- builders --------------------------------------------------------

    static CEngineResult build_gemm(const GemmProblem& p);
    static CEngineResult build_conv(const ConvProblem& p);
    static CEngineResult build_sdpa(const SdpaProblem& p);

   private:
    // Common manifest header for every op.
    static Manifest base_manifest(const char* kind, const std::string& kernel_name,
                                  int threads_per_block) {
        Manifest m;
        m.schema = "ck.dsl.example.manifest/v1";
        m.kind = kind;
        m.kernel_name = kernel_name;
        m.hsaco = "";
        m.cache_key = kernel_name;
        m.threads_per_block = threads_per_block;
        return m;
    }

    static ArgSpec arg(const char* name, const char* type, int size_bytes) {
        ArgSpec a;
        a.name = name;
        a.type = type;
        a.size_bytes = size_bytes;
        return a;
    }

    static unsigned ceil_div_u(long x, int d) {
        return d > 0 ? static_cast<unsigned>((x + d - 1) / d) : 1u;
    }

    [[noreturn]] static void fail(const char* op, const char* what) {
        throw std::runtime_error(std::string("CEngine::build_") + op + ": " + what);
    }
};

// ==========================================================================
// GEMM
//   ABI (ordered): A,B,C : ptr<f16,global>:8 ; M,N,K : i32:4
//   grid_order "NM" -> {ceil(N/block_n), ceil(M/block_m), 1}
//   block = warp_m*warp_n*warp_k*wave_size (derived by finalize)
// ==========================================================================
inline CEngineResult CEngine::build_gemm(const GemmProblem& p) {
    ckc_gemm_universal_spec_t spec = ckc_gemm_universal_spec_default();
    spec.name = "gemm";
    spec.tile.tile_m = p.tile_m;
    spec.tile.tile_n = p.tile_n;
    spec.tile.tile_k = p.tile_k;
    spec.tile.warp_m = p.warp_m;
    spec.tile.warp_n = p.warp_n;
    spec.tile.warp_k = p.warp_k;
    spec.tile.warp_tile_m = p.warp_tile_m;
    spec.tile.warp_tile_n = p.warp_tile_n;
    spec.tile.warp_tile_k = p.warp_tile_k;
    spec.trait.pipeline = p.pipeline;
    spec.trait.scheduler = p.scheduler;
    spec.trait.epilogue = p.epilogue;
    spec.data.dtype_a = p.dtype_a;
    spec.data.dtype_b = p.dtype_b;
    spec.data.dtype_c = p.dtype_c;
    spec.data.dtype_acc = p.dtype_acc;
    spec.data.layout = p.layout;
    spec.block_size = 0;  // derived
    ckc_gemm_universal_spec_finalize(&spec);

    char reason[CKC_ERR_MSG_CAP] = {0};
    if (!ckc_gemm_universal_is_valid_spec(&spec, p.arch, reason, sizeof reason))
        fail("gemm", reason[0] ? reason : "is_valid_spec rejected");

    char kname[256] = {0};
    if (ckc_gemm_universal_kernel_name(&spec, kname, sizeof kname) != CKC_OK)
        fail("gemm", "kernel_name overflow");

    char* ll = nullptr;
    char err[CKC_ERR_MSG_CAP] = {0};
    if (ckc_gemm_universal_lower_to_llvm(&spec, p.arch, CKC_LLVM_FLAVOR_AUTO, &ll, err,
                                         sizeof err) != CKC_OK ||
        !ll)
        fail("gemm", err[0] ? err : "lower_to_llvm failed");

    CEngineResult r;
    r.llvm_ir.assign(ll);
    free(ll);

    r.manifest = base_manifest("gemm_fp16", kname, spec.block_size);
    r.manifest.block_m = spec.tile.tile_m;
    r.manifest.block_n = spec.tile.tile_n;
    r.manifest.block_k = spec.tile.tile_k;
    r.manifest.grid_order = "NM";
    r.manifest.sig_has_bytes = false;
    r.manifest.args_signature = {
        arg("A", "ptr<f16, global>", 8),
        arg("B", "ptr<f16, global>", 8),
        arg("C", "ptr<f16, global>", 8),
        arg("M", "i32", 4),
        arg("N", "i32", 4),
        arg("K", "i32", 4),
    };

    r.block = static_cast<unsigned>(spec.block_size);
    // grid_order "NM": x <- N tiles, y <- M tiles.
    r.grid = {ceil_div_u(p.N, spec.tile.tile_n), ceil_div_u(p.M, spec.tile.tile_m), 1};
    return r;
}

// ==========================================================================
// CONV (implicit-GEMM, NHWC x KRSC -> NHWK)
//   ABI (ordered): A,B,D : ptr<f16,global>:8 ; A_bytes,B_bytes,D_bytes : i32:4
//   sig_has_bytes = true (runtime byte-size bounds checks)
//   GEMM rows M = N*Ho*Wo ; cols N_gemm = K. grid_order "NM" ->
//     {ceil(N_gemm/block_n), ceil(M/block_m), 1}
//   block = warp_m*warp_n*wave_size
// ==========================================================================
inline CEngineResult CEngine::build_conv(const ConvProblem& p) {
    ckc_implicit_gemm_conv_spec_t spec = ckc_implicit_gemm_conv_spec_default();
    spec.problem = ckc_conv_problem_make(p.N, p.Hi, p.Wi, p.C, p.K, p.R, p.S, p.sH, p.sW, p.pH,
                                         p.pW, p.dH, p.dW);
    spec.name = "conv_igemm";
    spec.tile_m = p.tile_m;
    spec.tile_n = p.tile_n;
    spec.tile_k = p.tile_k;
    spec.warp_m = p.warp_m;
    spec.warp_n = p.warp_n;
    spec.warp_tile_m = p.warp_tile_m;
    spec.warp_tile_n = p.warp_tile_n;
    spec.warp_tile_k = p.warp_tile_k;
    spec.pipeline = p.pipeline;
    spec.epilogue = p.epilogue;

    char reason[512] = {0};
    if (!ckc_implicit_gemm_conv_is_valid_spec(&spec, p.arch, reason, sizeof reason))
        fail("conv", reason[0] ? reason : "is_valid_spec rejected");

    char kname[256] = {0};
    if (ckc_implicit_gemm_conv_spec_kernel_name(&spec, kname, sizeof kname) != CKC_OK)
        fail("conv", "kernel_name overflow");

    char* ll = nullptr;
    char err[512] = {0};
    if (ckc_conv_implicit_gemm_lower_to_llvm(&spec, p.arch, CKC_LLVM_FLAVOR_AUTO, &ll, err,
                                             sizeof err) != CKC_OK ||
        !ll)
        fail("conv", err[0] ? err : "lower_to_llvm failed");

    CEngineResult r;
    r.llvm_ir.assign(ll);
    free(ll);

    const int block_size = ckc_implicit_gemm_conv_spec_block_size(&spec);
    r.manifest = base_manifest("conv_fp16", kname, block_size);
    r.manifest.block_m = spec.tile_m;
    r.manifest.block_n = spec.tile_n;
    r.manifest.block_k = spec.tile_k;
    r.manifest.grid_order = "NM";
    r.manifest.sig_has_bytes = true;
    r.manifest.args_signature = {
        arg("A", "ptr<f16, global>", 8), arg("B", "ptr<f16, global>", 8),
        arg("D", "ptr<f16, global>", 8), arg("A_bytes", "i32", 4),
        arg("B_bytes", "i32", 4),        arg("D_bytes", "i32", 4),
    };

    r.block = static_cast<unsigned>(block_size);
    const int M = ckc_conv_problem_m(&spec.problem);            // N*Ho*Wo
    const int N_gemm = ckc_conv_problem_n_gemm(&spec.problem);  // K
    // grid_order "NM": x <- N_gemm tiles, y <- M tiles.
    r.grid = {ceil_div_u(N_gemm, spec.tile_n), ceil_div_u(M, spec.tile_m), 1};
    return r;
}

// ==========================================================================
// SDPA (unified-attention 2D scalar forward)
//   ABI (18 args): out,q,k_cache,v_cache,sink : ptr<f16>:8 ;
//     block_tables,seq_lens : ptr<i32>:8 ; alibi_slopes,qq_bias : ptr<f32>:8 ;
//     query_start_len : ptr<i32>:8 ;
//     scale,k_scale,v_scale,out_scale,softcap : f32:4 ;
//     num_seqs,block_table_stride,qq_bias_stride_0 : i32:4
//   ckc_unified_attention_2d_scalar_grid -> {total_q, num_query_heads, head_size};
//   launch grid composed as {num_kv_heads, ceil(total_q/block_q)+num_seqs, 1}
//   matching the CK unified-attention block-id space (block_q == block_size).
// ==========================================================================
inline CEngineResult CEngine::build_sdpa(const SdpaProblem& p) {
    ckc_unified_attention_problem_t prob = ckc_unified_attention_problem_default();
    prob.total_q = p.total_q;
    prob.num_seqs = p.num_seqs;
    prob.num_query_heads = p.num_query_heads;
    prob.num_kv_heads = p.num_kv_heads;
    prob.head_size = p.head_size;
    // The KV block size is the scalar path's one tile knob. Prefer the
    // heuristic-overlaid block_q (== attention_config block_q/block_size) when
    // present; else fall back to block_size. This is what makes the kernel
    // identity (name + paged-KV block geometry) track the chosen candidate.
    prob.block_size = p.block_q > 0 ? p.block_q : p.block_size;
    prob.max_seqlen_q = p.max_seqlen_q;
    prob.max_seqlen_k = p.max_seqlen_k;
    prob.dtype = p.dtype;
    prob.sliding_window = p.sliding_window;
    prob.softcap = p.softcap;

    const char* reason = nullptr;
    if (!ckc_unified_attention_supports_scalar(&prob, &reason))
        fail("sdpa", reason ? reason : "unsupported by scalar 2D backend");

    char kname[256] = {0};
    if (ckc_unified_attention_2d_scalar_kernel_name(&prob, nullptr, kname, sizeof kname) != CKC_OK)
        fail("sdpa", "kernel_name overflow");

    char* ll = nullptr;
    char err[CKC_ERR_MSG_CAP] = {0};
    if (ckc_unified_attention_2d_scalar_lower_to_llvm(&prob, nullptr, CKC_LLVM_FLAVOR_AUTO, &ll,
                                                      err, sizeof err) != CKC_OK ||
        !ll)
        fail("sdpa", err[0] ? err : "lower_to_llvm failed");

    CEngineResult r;
    r.llvm_ir.assign(ll);
    free(ll);

    // Scalar kernel: one workgroup of 64 threads.
    const int threads = 64;
    r.manifest = base_manifest("attention_unified", kname, threads);
    r.manifest.grid_order = "MN";  // explicit grid below; grid_order unused
    r.manifest.sig_has_bytes = false;
    r.manifest.args_signature = {
        arg("output_ptr", "ptr<f16, global>", 8),
        arg("query_ptr", "ptr<f16, global>", 8),
        arg("key_cache_ptr", "ptr<f16, global>", 8),
        arg("value_cache_ptr", "ptr<f16, global>", 8),
        arg("sink_ptr", "ptr<f16, global>", 8),
        arg("block_tables_ptr", "ptr<i32, global>", 8),
        arg("seq_lens_ptr", "ptr<i32, global>", 8),
        arg("alibi_slopes_ptr", "ptr<f32, global>", 8),
        arg("qq_bias_ptr", "ptr<f32, global>", 8),
        arg("query_start_len_ptr", "ptr<i32, global>", 8),
        arg("scale", "f32", 4),
        arg("k_scale", "f32", 4),
        arg("v_scale", "f32", 4),
        arg("out_scale", "f32", 4),
        arg("softcap", "f32", 4),
        arg("num_seqs", "i32", 4),
        arg("block_table_stride", "i32", 4),
        arg("qq_bias_stride_0", "i32", 4),
    };

    r.block = static_cast<unsigned>(threads);

    int g[3] = {0, 0, 0};
    ckc_unified_attention_2d_scalar_grid(&prob, g);  // {total_q, num_query_heads, head_size}
    // build_sdpa generates the unified 2D *scalar* attention kernel, whose
    // block-id space is (q_tok, q_head, dim) = (total_q, num_query_heads,
    // head_size). That IS the launch grid -- NOT the tiled CK paged-KV
    // {num_kv_heads, ceil(total_q/block_q)+num_seqs, 1} block space (which
    // belongs to a different, shipped tiled kernel). Launch with the scalar grid.
    r.grid = {static_cast<unsigned>(g[0]), static_cast<unsigned>(g[1]),
              static_cast<unsigned>(g[2])};
    r.manifest.grid_explicit = std::array<int, 3>{g[0], g[1], g[2]};
    return r;
}

}  // namespace ck_dsl

// ==========================================================================
// Standalone self-test: build all 3 ops' .ll and print sizes. CPU-only.
//   hipcc -std=c++17 -DCKC_CENGINE_TEST \
//     -I .../ck_dsl_c/include -I .../runtime/include -I /opt/rocm/include \
//     this.cpp -L<libdir> -lckc_core -lamd_comgr
// ==========================================================================
#ifdef CKC_CENGINE_TEST
#include <cstdio>

int main() {
    using namespace ck_dsl;
    int failures = 0;

    try {
        CEngine::GemmProblem g;
        g.M = 256;
        g.N = 256;
        g.K = 256;
        g.pipeline = "compv3";
        g.epilogue = "default";  // demo-proven combo
        auto r = CEngine::build_gemm(g);
        std::printf(
            "[gemm] kernel=%s  ll=%zu bytes  block=%u  grid={%u,%u,%u}  "
            "block_m/n/k=%d/%d/%d  args=%zu\n",
            r.manifest.kernel_name.c_str(), r.llvm_ir.size(), r.block, r.grid[0], r.grid[1],
            r.grid[2], r.manifest.block_m, r.manifest.block_n, r.manifest.block_k,
            r.manifest.args_signature.size());
    } catch (const std::exception& e) {
        std::printf("[gemm] FAILED: %s\n", e.what());
        ++failures;
    }

    try {
        CEngine::ConvProblem c;
        c.N = 8;
        c.Hi = 56;
        c.Wi = 56;
        c.C = 64;
        c.K = 64;
        c.R = 3;
        c.S = 3;
        c.sH = c.sW = 1;
        c.pH = c.pW = 1;
        c.dH = c.dW = 1;
        auto r = CEngine::build_conv(c);
        std::printf(
            "[conv] kernel=%s  ll=%zu bytes  block=%u  grid={%u,%u,%u}  "
            "block_m/n/k=%d/%d/%d  args=%zu sig_has_bytes=%d\n",
            r.manifest.kernel_name.c_str(), r.llvm_ir.size(), r.block, r.grid[0], r.grid[1],
            r.grid[2], r.manifest.block_m, r.manifest.block_n, r.manifest.block_k,
            r.manifest.args_signature.size(), r.manifest.sig_has_bytes ? 1 : 0);
    } catch (const std::exception& e) {
        std::printf("[conv] FAILED: %s\n", e.what());
        ++failures;
    }

    try {
        CEngine::SdpaProblem s;
        s.total_q = 8;
        s.num_seqs = 1;
        s.num_query_heads = 8;
        s.num_kv_heads = 8;
        s.head_size = 64;
        s.block_size = 16;
        s.max_seqlen_q = 8;
        s.max_seqlen_k = 128;
        s.dtype = "fp16";
        auto r = CEngine::build_sdpa(s);
        std::printf("[sdpa] kernel=%s  ll=%zu bytes  block=%u  grid={%u,%u,%u}  args=%zu\n",
                    r.manifest.kernel_name.c_str(), r.llvm_ir.size(), r.block, r.grid[0], r.grid[1],
                    r.grid[2], r.manifest.args_signature.size());
    } catch (const std::exception& e) {
        std::printf("[sdpa] FAILED: %s\n", e.what());
        ++failures;
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL OK" : "FAILURES", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
#endif  // CKC_CENGINE_TEST
