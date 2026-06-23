// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <memory>

#include "CkDslHandle.hpp"
#include "ck_dsl_runtime/kernel.hpp"
#include "engines/CkDslAttnParamParser.hpp"

namespace ck_dsl_plugin {

// SDPA forward plan. Holds an AOT-compiled ck_dsl unified-attention kernel and,
// at plan-build time, synthesizes the paged-KV metadata (block_tables /
// seq_lens / query_start_len) that the kernel requires from a *dense* hipDNN
// SDPA graph.
//
// Dense->paged mapping: for BSHD layout, dense K[B,S,Hkv,D] memory is
// byte-identical to a paged cache KC[(B*S)/block_size, block_size, Hkv, D] with
// per-sequence contiguous block_tables, so K/V buffers are passed directly as
// the cache pointers and only the (small, int32) metadata is synthesized.
// (BHSD is head-major and would need a transpose into paged layout -- a
// documented follow-on.)
class CkDslAttnPlan : public hipdnn_plugin_sdk::IPlan<CkDslHandle> {
   public:
    // Single 2D kernel plan (prefill / short-KV decode).
    CkDslAttnPlan(CkDslAttnParamParser::ParsedAttnParams params,
                  std::unique_ptr<ck_dsl::Kernel> kernel);
    // Two-launch split-KV 3D decode plan: segment kernel -> reduce kernel, with
    // a per-(q,head,segment) (m, l, acc) workspace. The explicit launch grids
    // are taken from each kernel's manifest grid_explicit (set by the C engine).
    CkDslAttnPlan(CkDslAttnParamParser::ParsedAttnParams params,
                  std::unique_ptr<ck_dsl::Kernel> segment_kernel,
                  std::unique_ptr<ck_dsl::Kernel> reduce_kernel, int num_segments);
    ~CkDslAttnPlan() override;

    size_t getWorkspaceSize(const CkDslHandle& handle) const override {
        return workspace_bytes_;
    }
    void execute(const CkDslHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers, void* workspace) const override;

    const ck_dsl::Kernel& kernel() const {
        return *kernel_;
    }

   private:
    static void* findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs, uint32_t count);
    std::array<unsigned, 3> grid() const;
    // Per-kernel explicit grid from the manifest grid_explicit.
    static std::array<unsigned, 3> explicit_grid(const ck_dsl::Kernel& k);
    void synthesize_paged_kv_meta();  // shared ctor body (metadata upload)
    void execute_2d(const CkDslHandle& handle, void* q, void* kc, void* vc, void* o) const;
    void execute_3d(const CkDslHandle& handle, void* q, void* kc, void* vc, void* o,
                    void* workspace) const;

    CkDslAttnParamParser::ParsedAttnParams params_;
    std::unique_ptr<ck_dsl::Kernel> kernel_;         // 2D, or 3D segment kernel
    std::unique_ptr<ck_dsl::Kernel> reduce_kernel_;  // 3D reduce kernel (else null)
    bool is_3d_ = false;
    int num_segments_ = 0;
    size_t workspace_bytes_ = 0;
    // Sub-offsets into the single workspace blob (segm_output | segm_max | segm_expsum).
    size_t ws_off_output_ = 0, ws_off_max_ = 0, ws_off_expsum_ = 0;
    // hipGraph capture of the two-launch decode pipeline (lazily captured in
    // execute_3d on the first launch with a given workspace pointer).
    mutable hipGraph_t graph_ = nullptr;
    mutable hipGraphExec_t graph_exec_ = nullptr;
    mutable void* graph_workspace_ = nullptr;  // workspace ptr the graph was captured against
    mutable void* graph_output_ = nullptr;     // output ptr the graph was captured against
    // Synthesized paged-KV metadata (device), owned by the plan.
    void* d_block_tables_ = nullptr;
    void* d_seq_lens_ = nullptr;
    void* d_query_start_len_ = nullptr;
    int block_size_ = 16;
    int block_q_ = 16;
    int max_blocks_ = 0;
    int num_seqs_ = 1;
};

}  // namespace ck_dsl_plugin
