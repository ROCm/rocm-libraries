// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslAttnPlan.hpp"

#include <cstdlib>
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
    synthesize_paged_kv_meta();
}

// Split-KV 3D decode: segment kernel + reduce kernel + per-segment workspace.
CkDslAttnPlan::CkDslAttnPlan(CkDslAttnParamParser::ParsedAttnParams params,
                             std::unique_ptr<ck_dsl::Kernel> segment_kernel,
                             std::unique_ptr<ck_dsl::Kernel> reduce_kernel, int num_segments)
    : params_(std::move(params)),
      kernel_(std::move(segment_kernel)),
      reduce_kernel_(std::move(reduce_kernel)),
      is_3d_(true),
      num_segments_(num_segments) {
    if (params_.is_bhsd)
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "CkDslAttn: dense BHSD needs a KV transpose into paged layout (follow-on); "
            "BSHD is supported");
    if (num_segments_ <= 0)
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "CkDslAttn: num_segments must be positive");
    // The segment kernel's KV-block tile size (paged-KV geometry). The C engine
    // injects attention_config into the segment kernel manifest exactly like the
    // 2D path; fall back to 16 if absent.
    const auto& cfg = kernel_->manifest().raw.has("attention_config")
                          ? kernel_->manifest().raw.at("attention_config")
                          : kernel_->manifest().raw;
    block_size_ = (int)cfg.get_int("block_size", 16);
    block_q_ = (int)cfg.get_int("block_q", 16);
    if (block_size_ <= 0) block_size_ = 16;
    if (block_q_ <= 0) block_q_ = 16;

    // Workspace: three f32 tensors (CK Tile split-KV lse_acc/o_acc analogue),
    // laid out contiguously in one blob:
    //   segm_output [total_q, num_query_heads, num_segments, head_size] f32
    //   segm_max    [total_q, num_query_heads, num_segments]            f32
    //   segm_expsum [total_q, num_query_heads, num_segments]            f32
    const long total_q = (long)params_.batch * params_.seqlen_q;
    const long nqh = params_.nhead_q;
    const long nseg = num_segments_;
    const long hd = params_.hdim_q;
    const size_t n_out = (size_t)total_q * nqh * nseg * hd;
    const size_t n_ml = (size_t)total_q * nqh * nseg;
    auto align64 = [](size_t b) { return (b + 63) & ~size_t(63); };
    ws_off_output_ = 0;
    ws_off_max_ = align64(n_out * sizeof(float));
    ws_off_expsum_ = align64(ws_off_max_ + n_ml * sizeof(float));
    workspace_bytes_ = align64(ws_off_expsum_ + n_ml * sizeof(float));

    synthesize_paged_kv_meta();
}

void CkDslAttnPlan::synthesize_paged_kv_meta() {
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
    if (graph_exec_) hipGraphExecDestroy(graph_exec_);
    if (graph_) hipGraphDestroy(graph_);
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

std::array<unsigned, 3> CkDslAttnPlan::explicit_grid(const ck_dsl::Kernel& k) {
    if (k.manifest().grid_explicit) {
        const auto& g = *k.manifest().grid_explicit;
        return {(unsigned)g[0], (unsigned)g[1], (unsigned)g[2]};
    }
    return {1, 1, 1};
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
                            uint32_t numDeviceBuffers, void* workspace) const {
    void* q = findBuffer(params_.q_uid, deviceBuffers, numDeviceBuffers);
    void* kc = findBuffer(params_.k_uid, deviceBuffers, numDeviceBuffers);
    void* vc = findBuffer(params_.v_uid, deviceBuffers, numDeviceBuffers);
    void* o = findBuffer(params_.o_uid, deviceBuffers, numDeviceBuffers);
    if (!q || !kc || !vc || !o)
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "CkDslAttn: missing Q/K/V/O");
    if (is_3d_)
        execute_3d(handle, q, kc, vc, o, workspace);
    else
        execute_2d(handle, q, kc, vc, o);
}

void CkDslAttnPlan::execute_2d(const CkDslHandle& handle, void* q, void* kc, void* vc,
                               void* o) const {
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

void CkDslAttnPlan::execute_3d(const CkDslHandle& handle, void* q, void* kc, void* vc, void* o,
                               void* workspace) const {
    if (!workspace)
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "CkDslAttn: split-KV 3D decode requires a workspace (got null)");

    char* ws = static_cast<char*>(workspace);
    void* segm_output = ws + ws_off_output_;
    void* segm_max = ws + ws_off_max_;
    void* segm_expsum = ws + ws_off_expsum_;

    auto f32 = [](float x) {
        uint32_t b;
        std::memcpy(&b, &x, 4);
        return (uint64_t)b;
    };

    // Segment kernel: 19-arg ABI (segm_* workspace + Q/K/V/meta, 7 scalars).
    std::unordered_map<std::string, void*> seg_ptrs = {{"segm_output_ptr", segm_output},
                                                       {"segm_max_ptr", segm_max},
                                                       {"segm_expsum_ptr", segm_expsum},
                                                       {"query_ptr", q},
                                                       {"key_cache_ptr", kc},
                                                       {"value_cache_ptr", vc},
                                                       {"sink_ptr", nullptr},
                                                       {"block_tables_ptr", d_block_tables_},
                                                       {"seq_lens_ptr", d_seq_lens_},
                                                       {"alibi_slopes_ptr", nullptr},
                                                       {"qq_bias_ptr", nullptr},
                                                       {"query_start_len_ptr", d_query_start_len_}};
    std::unordered_map<std::string, uint64_t> seg_scalars = {
        {"scale", f32(params_.scale)},
        {"k_scale", f32(1.f)},
        {"v_scale", f32(1.f)},
        {"softcap", f32(0.f)},
        {"num_seqs", (uint64_t)num_seqs_},
        {"block_table_stride", (uint64_t)max_blocks_},
        {"qq_bias_stride_0", 0}};

    // Reduce kernel: 5-arg ABI (output + segm_* workspace + seq_lens).
    std::unordered_map<std::string, void*> red_ptrs = {{"output_ptr", o},
                                                       {"segm_output_ptr", segm_output},
                                                       {"segm_max_ptr", segm_max},
                                                       {"segm_expsum_ptr", segm_expsum},
                                                       {"seq_lens_ptr", d_seq_lens_}};
    std::unordered_map<std::string, uint64_t> red_scalars;

    const auto seg_grid = explicit_grid(*kernel_);
    const auto red_grid = explicit_grid(*reduce_kernel_);
    const unsigned seg_block = (unsigned)kernel_->manifest().threads_per_block;
    const unsigned red_block = (unsigned)reduce_kernel_->manifest().threads_per_block;
    const hipStream_t stream = handle.getStream();

    // Ensure both kernels are loaded before any capture.
    kernel_->ensure_compiled();
    reduce_kernel_->ensure_compiled();

    // hipGraph capture/replay of the two launches -- the dominant decode lever
    // (launch-bound; replay collapses two host launches into one graph launch).
    // The graph is captured once per (workspace, output) pointer pair; if the
    // caller rotates the workspace/output we re-capture. Graph capture is gated
    // OFF by CK_DSL_ATTN_NO_GRAPH=1 (falls back to two stream launches).
    static const bool no_graph = [] {
        const char* v = std::getenv("CK_DSL_ATTN_NO_GRAPH");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
    }();

    ck_dsl::ScopedTimer t("attn3d", ck_dsl::ScopedTimer::Unit::Us);

    if (!no_graph && stream != nullptr) {
        if (!graph_exec_ || graph_workspace_ != workspace || graph_output_ != o) {
            if (graph_exec_) {
                hipGraphExecDestroy(graph_exec_);
                graph_exec_ = nullptr;
            }
            if (graph_) {
                hipGraphDestroy(graph_);
                graph_ = nullptr;
            }
            ck_dsl::hip_check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
                              "attn3d beginCapture");
            kernel_->launch(seg_ptrs, seg_scalars, seg_grid, seg_block, stream);
            reduce_kernel_->launch(red_ptrs, red_scalars, red_grid, red_block, stream);
            ck_dsl::hip_check(hipStreamEndCapture(stream, &graph_), "attn3d endCapture");
            ck_dsl::hip_check(hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
                              "attn3d graphInstantiate");
            graph_workspace_ = workspace;
            graph_output_ = o;
        }
        ck_dsl::hip_check(hipGraphLaunch(graph_exec_, stream), "attn3d graphLaunch");
    } else {
        // Two-launch fallback (no graph): segment then reduce on the stream.
        kernel_->launch(seg_ptrs, seg_scalars, seg_grid, seg_block, stream);
        reduce_kernel_->launch(red_ptrs, red_scalars, red_grid, red_block, stream);
    }

    if (ck_dsl::timing_enabled()) ck_dsl::hip_check(hipStreamSynchronize(stream), "attn3d sync");
}

}  // namespace ck_dsl_plugin
