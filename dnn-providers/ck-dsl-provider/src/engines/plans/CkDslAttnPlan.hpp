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
    CkDslAttnPlan(CkDslAttnParamParser::ParsedAttnParams params,
                  std::unique_ptr<ck_dsl::Kernel> kernel);
    ~CkDslAttnPlan() override;

    size_t getWorkspaceSize(const CkDslHandle& handle) const override {
        return 0;
    }
    void execute(const CkDslHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers, void* workspace) const override;

    const ck_dsl::Kernel& kernel() const {
        return *kernel_;
    }

   private:
    static void* findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs, uint32_t count);
    std::array<unsigned, 3> grid() const;

    CkDslAttnParamParser::ParsedAttnParams params_;
    std::unique_ptr<ck_dsl::Kernel> kernel_;
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
