// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslAttnPlan.hpp"

#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <unordered_map>
#include <vector>

#include "ck_dsl_runtime/timing.hpp"

namespace ck_dsl_plugin {

CkDslAttnPlan::CkDslAttnPlan(CkDslAttnParamParser::ParsedAttnParams params,
                             std::unique_ptr<ck_dsl::Kernel> kernel)
    : params_(std::move(params)), kernel_(std::move(kernel)) {
    if (params_.is_bhsd)
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "CkDslAttn: dense BHSD needs a KV transpose into paged layout (follow-on); "
            "BSHD is supported");

    const auto& cfg = kernel_->manifest().raw.at("attention_config");
    block_size_ = (int)cfg.get_int("block_size", 16);
    block_q_ = (int)cfg.get_int("block_q", 16);
    // A manifest that carries block_size/block_q <= 0 would divide by zero below
    // (max_blocks_) and in grid(); reject it cleanly instead of crashing.
    if (block_size_ <= 0 || block_q_ <= 0)
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "CkDslAttn: manifest block_size/block_q must be positive");
    num_seqs_ = (int)params_.batch;
    const int B = num_seqs_, Sk = (int)params_.seqlen_k, Sq = (int)params_.seqlen_q;
    max_blocks_ = (Sk + block_size_ - 1) / block_size_;

    // Synthesize paged-KV metadata for contiguous BSHD KV (no reformat).
    std::vector<int32_t> bt((size_t)B * max_blocks_), sl(B), cu(B + 1, 0);
    for (int b = 0; b < B; ++b) {
        sl[b] = Sk;
        cu[b + 1] = cu[b] + Sq;
        for (int j = 0; j < max_blocks_; ++j) bt[(size_t)b * max_blocks_ + j] = b * max_blocks_ + j;
    }
    // RAII device allocation: if any later upload fails the earlier ones are
    // freed automatically (no leak on a partially-constructed plan). Members are
    // committed only after all three succeed.
    auto deleter = [](void* p) {
        if (p) hipFree(p);
    };
    using DevPtr = std::unique_ptr<void, decltype(deleter)>;
    auto up = [&](const std::vector<int32_t>& v) {
        void* p = nullptr;
        ck_dsl::hip_check(hipMalloc(&p, v.size() * 4), "attn meta malloc");
        DevPtr owned(p, deleter);
        ck_dsl::hip_check(hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice),
                          "attn meta h2d");
        return owned;
    };
    DevPtr block_tables = up(bt);
    DevPtr seq_lens = up(sl);
    DevPtr query_start_len = up(cu);
    d_block_tables_ = block_tables.release();
    d_seq_lens_ = seq_lens.release();
    d_query_start_len_ = query_start_len.release();
}

CkDslAttnPlan::~CkDslAttnPlan() {
    if (d_block_tables_) hipFree(d_block_tables_);
    if (d_seq_lens_) hipFree(d_seq_lens_);
    if (d_query_start_len_) hipFree(d_query_start_len_);
}

void* CkDslAttnPlan::findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs,
                                uint32_t count) {
    for (uint32_t i = 0; i < count; ++i)
        if (bufs[i].uid == uid) return bufs[i].ptr;
    return nullptr;
}

std::array<unsigned, 3> CkDslAttnPlan::grid() const {
    // The C-JIT (CEngine) backend generates the unified 2D *scalar* attention
    // kernel and carries its native (q_tok, q_head, dim) block-id space via the
    // manifest's grid_explicit -- launch with that. The shipped tiled paged-KV
    // kernel has no grid_explicit and uses the (nkvh, q-tiles, 1) block space.
    if (kernel_ && kernel_->manifest().grid_explicit) {
        const auto& g = *kernel_->manifest().grid_explicit;
        return {(unsigned)g[0], (unsigned)g[1], (unsigned)g[2]};
    }
    long total_q = (long)params_.batch * params_.seqlen_q;
    const int bq = block_q_ > 0 ? block_q_ : 1;  // ctor rejects <=0; defensive guard
    unsigned gy = (unsigned)(total_q / bq + num_seqs_);
    return {(unsigned)params_.nhead_k, gy, 1};
}

void CkDslAttnPlan::execute(const CkDslHandle& handle,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers, void* /*workspace*/) const {
    void* q = findBuffer(params_.q_uid, deviceBuffers, numDeviceBuffers);
    void* kc = findBuffer(params_.k_uid, deviceBuffers, numDeviceBuffers);
    void* vc = findBuffer(params_.v_uid, deviceBuffers, numDeviceBuffers);
    void* o = findBuffer(params_.o_uid, deviceBuffers, numDeviceBuffers);
    if (!q || !kc || !vc || !o)
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "CkDslAttn: missing Q/K/V/O");

    std::unordered_map<std::string, void*> ptrs = {{"output_ptr", o},
                                                   {"query_ptr", q},
                                                   {"key_cache_ptr", kc},
                                                   {"value_cache_ptr", vc},
                                                   {"sink_ptr", nullptr},
                                                   {"block_tables_ptr", d_block_tables_},
                                                   {"seq_lens_ptr", d_seq_lens_},
                                                   {"alibi_slopes_ptr", nullptr},
                                                   {"qq_bias_ptr", nullptr},
                                                   {"query_start_len_ptr", d_query_start_len_}};
    auto f32 = [](float x) {
        uint32_t b;
        std::memcpy(&b, &x, 4);
        return (uint64_t)b;
    };
    std::unordered_map<std::string, uint64_t> scalars = {
        {"scale", f32(params_.scale)},
        {"k_scale", f32(1.f)},
        {"v_scale", f32(1.f)},
        {"out_scale", f32(1.f)},
        {"softcap", f32(0.f)},
        {"num_seqs", (uint64_t)num_seqs_},
        {"block_table_stride", (uint64_t)max_blocks_},
        {"qq_bias_stride_0", 0}};

    unsigned block = (unsigned)kernel_->manifest().threads_per_block;
    // Launch, timed under CK_DSL_TIME=1 (launchUs, stream-synced).
    ck_dsl::ScopedTimer t("attn", ck_dsl::ScopedTimer::Unit::Us);
    kernel_->launch(ptrs, scalars, grid(), block, handle.getStream());
    if (ck_dsl::timing_enabled())
        ck_dsl::hip_check(hipStreamSynchronize(handle.getStream()), "attn sync");
}

}  // namespace ck_dsl_plugin
