// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Unified test for Sparge pipeline: blockmap generation + sparse attention (Jenga/VSA).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-copy-with-user-provided-dtor"
#include "ck_tile/host.hpp"
#pragma clang diagnostic pop
#include "ck_tile/core.hpp"
#include "ck_tile/host/reference/reference_blocked_attention.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

#include "01_fmha/mask.hpp" // mask_info::decode, mask_enum
#include "fmha_fwd_trek.hpp"
#include "sparge_blockmap_trek.hpp"
#include "sparge_tool.hpp"

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
ck_tile::HostTensor<T> make_qkv_tensor(ck_tile::index_t batch,
                                       ck_tile::index_t nhead,
                                       ck_tile::index_t seqlen,
                                       ck_tile::index_t hdim,
                                       bool i_perm)
{
    if(i_perm)
        return ck_tile::HostTensor<T>({batch, nhead, seqlen, hdim});
    return ck_tile::HostTensor<T>({batch, seqlen, nhead, hdim});
}

template <typename T>
ck_tile::HostTensor<T> to_bhsd(const ck_tile::HostTensor<T>& tensor, bool is_bhsd)
{
    auto lens               = tensor.get_lengths();
    ck_tile::index_t batch  = lens[0];
    ck_tile::index_t seqlen = is_bhsd ? lens[2] : lens[1];
    ck_tile::index_t nhead  = is_bhsd ? lens[1] : lens[2];
    ck_tile::index_t hdim   = lens[3];

    ck_tile::HostTensor<T> out({batch, nhead, seqlen, hdim});
    for(ck_tile::index_t b = 0; b < batch; ++b)
        for(ck_tile::index_t h = 0; h < nhead; ++h)
            for(ck_tile::index_t s = 0; s < seqlen; ++s)
                for(ck_tile::index_t d = 0; d < hdim; ++d)
                    out(b, h, s, d) = is_bhsd ? tensor(b, h, s, d) : tensor(b, s, h, d);
    return out;
}

template <typename T>
auto get_error_tolerance()
{
    // Matches dense FMHA fp16/bf16 bounds; validated on (b=1,h=2,d=128,
    // s in {512, 2048, 4096, 8192}) with maxdiff = 0.00 across both dtypes.
    double rtol = 1e-2;
    double atol = 4e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <typename T>
float to_float_for_compare(T value)
{
    return static_cast<float>(value);
}

template <>
float to_float_for_compare<ck_tile::bf16_t>(ck_tile::bf16_t value)
{
    return ck_tile::type_convert<float>(value);
}

// ============================================================================
// Arg parser
// ============================================================================
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "0:no validation, 1:cpu validation")
        .insert("pipeline", "jenga", "attention pipeline: jenga / vsa")
        .insert("b", "1", "batch size")
        .insert("h", "4", "num of head for q")
        .insert("h_k", "-1", "num of head for k/v, -1 means equal to h")
        .insert("s", "4096", "seqlen_q")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("topk", "0.3", "topk ratio for blockmap (fraction of K-blocks to keep)")
        .insert("cdfthreshd", "-1", "CDF threshold for blockmap (overrides topk if >= 0)")
        .insert("simthreshd1", "0.6", "similarity threshold for blockmap")
        .insert("prec", "fp16", "data type: fp16/bf16")
        .insert("iperm", "1", "permute input, 1: b*h*s*d, 0: b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("seed", "42", "random seed")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "benchmark iterations")
        .insert("kname", "0", "print kernel name")
        .insert("dump_o",
                "",
                "if non-empty, dump raw output buffer bytes to this path (for bit-identical "
                "baseline comparison)")
        .insert("pv_threshold",
                "1e30",
                "SpargeAttn PV-skip per-Q-tile threshold; default +1e30 disables skip")
        .insert("pv_threshold_per_head",
                "",
                "comma-separated per-head pv_threshold list "
                "(length must == h). Empty = scalar mode using -pv_threshold.")
        .insert("pv_skip_compile",
                "1",
                "1=use kEnablePVSkip=true template instance; 0=use kEnablePVSkip=false instance "
                "(PV-skip AST removed at compile time, equivalent to VSA baseline). "
                "Deprecated by -pv_mode; kept for back-compat scripts.")
        .insert("pv_mode",
                "warp",
                "PV-skip mode select. one of {none, warp, block}. "
                "none = no skip (kNone binary; matches VSA baseline). "
                "warp = per-wavefront butterfly vote (default). "
                "block = per-block AND vote via 1 LDS slot + block_sync_lds. "
                "Overrides -pv_skip_compile when set explicitly.")
        .insert("mask",
                "0",
                "0: no mask, 1: top-left(same as 't'), 2:bottom-right(same as 'b')\n"
                "'t', top-left causal mask, 'b', bottom-r causal mask\n"
                "'t:l,r', top-left sliding window attn(swa) with FA style left right size\n"
                "'b:l,r', bottom-r sliding window attn(swa) with FA style left right size\n"
                "'xt:window_size', xformer style masking from top-left, "
                "window_size negative is causal, positive is swa\n"
                "'xb:window_size', xformer style masking from bottom-r, "
                "window_size negative is causal, positive is swa\n"
                "'g:y,x', generic attention mask coordinate with y/x size "
                "(only debug purpose for now)")
        .insert("attention_sink",
                "0",
                "SpargeAttn: force block-map column 0 ON (kb=0 always selected). "
                "0=off, 1=on. Block-map level only; independent of -mask sink prefix.")
        .insert("cross_ref",
                "0",
                "1=independent dense+causal cross-reference of attention output. "
                "Reuses the 01_fmha-style GEMM+masking+softmax+GEMM chain (different "
                "code path / formula than the sparge per-block causal pipeline). "
                "Intended with -mask=t -topk=1.0 (sparge degenerates to dense+causal).")
        .insert("qscale",
                "n",
                "Quant-plumbing scaffold (sage-style Q/K/V static quant). "
                "'n' = no descale buffers, ptr=nullptr (default; bit-identical to "
                "pre-scaffold; -pipeline=vsa goes through codegen NO_SCALE path). "
                "'bs' = block-scale mode, alloc unit-scale (= 1.0f) buffers and "
                "pass pointers through Kargs. For -pipeline=vsa + fp16/hdim=128/bm0=64 "
                "this routes to the hand-written BLOCKSCALE oneshot "
                "(kDoFp8StaticQuant=true) — pipeline body still (void)-casts the "
                "descale ptrs, so results stay bit-identical to NO_SCALE.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// ============================================================================
// Main test
// ============================================================================
template <typename T>
bool run_test(const ck_tile::ArgParser& arg_parser)
{
    int do_validation         = arg_parser.get_int("v");
    std::string pipeline      = arg_parser.get_str("pipeline");
    ck_tile::index_t batch    = arg_parser.get_int("b");
    ck_tile::index_t nhead    = arg_parser.get_int("h");
    ck_tile::index_t nhead_k  = arg_parser.get_int("h_k");
    ck_tile::index_t seqlen_q = arg_parser.get_int("s");
    ck_tile::index_t seqlen_k = arg_parser.get_int("s_k");
    ck_tile::index_t hdim_q   = arg_parser.get_int("d");
    ck_tile::index_t hdim_v   = arg_parser.get_int("d_v");
    float topk                = arg_parser.get_float("topk");
    float cdfthreshd          = arg_parser.get_float("cdfthreshd");
    float simthreshd1         = arg_parser.get_float("simthreshd1");
    bool i_perm               = arg_parser.get_bool("iperm");
    bool o_perm               = arg_parser.get_bool("operm");
    uint32_t seed             = arg_parser.get_uint32("seed");
    int warmup                = arg_parser.get_int("warmup");
    int repeat                = arg_parser.get_int("repeat");
    int kname                 = arg_parser.get_int("kname");
    std::string dump_o_path   = arg_parser.get_str("dump_o");
    float pv_threshold        = arg_parser.get_float("pv_threshold");
    int pv_skip_compile       = arg_parser.get_int("pv_skip_compile");
    std::string pv_per_head_s = arg_parser.get_str("pv_threshold_per_head");
    std::string pv_mode_str   = arg_parser.get_str("pv_mode");
    std::string mask_str      = arg_parser.get_str("mask");
    bool attention_sink       = arg_parser.get_bool("attention_sink");
    int cross_ref             = arg_parser.get_int("cross_ref");
    std::string qscale_str    = arg_parser.get_str("qscale");

    // P1 quant plumbing scaffold dispatch:
    //   'n'  -> qscale_bs = false, descale ptrs = nullptr (default, bit-identical to pre-scaffold)
    //   'bs' -> qscale_bs = true,  alloc unit-scale (= 1.0f) buffers and pass pointers
    // Anything else is a CLI error.
    bool qscale_bs = false;
    if(qscale_str == "n")
        qscale_bs = false;
    else if(qscale_str == "bs")
        qscale_bs = true;
    else
    {
        std::cerr << "Unknown -qscale value: '" << qscale_str << "' (expected one of: n, bs)"
                  << std::endl;
        return false;
    }

    // --pv_mode -> dispatched int.
    //   none -> 0 (kNone), warp -> 1 (kPerWave), block -> 2 (kPerBlock).
    // Back-compat: if -pv_skip_compile=0 was passed but -pv_mode stayed default
    // ("warp"), legacy intent wins (mode=0). CLI can't tell "was this passed
    // explicitly", so we mirror the rule: bool 0 => kNone, bool 1 => kPerWave.
    int pv_mode_compile;
    if(pv_mode_str == "none")
        pv_mode_compile = 0;
    else if(pv_mode_str == "warp")
        pv_mode_compile = 1;
    else if(pv_mode_str == "block")
        pv_mode_compile = 2;
    else
    {
        std::cerr << "Unknown -pv_mode value: '" << pv_mode_str
                  << "' (expected one of: none, warp, block)" << std::endl;
        return false;
    }
    // Legacy bool wins iff user explicitly disabled and pv_mode stayed warp.
    if(pv_skip_compile == 0 && pv_mode_str == "warp")
        pv_mode_compile = 0;

    if(nhead_k < 0)
        nhead_k = nhead;
    if(seqlen_k < 0)
        seqlen_k = seqlen_q;
    if(hdim_v < 0)
        hdim_v = hdim_q;

    mask_info mask = mask_info::decode(mask_str, seqlen_q, seqlen_k);
    if(mask.type != mask_enum::no_mask && mask.type != mask_enum::mask_top_left)
        std::fprintf(stderr,
                     "[test_sparge] WARN: -mask='%s' (type=%d) - block-map only "
                     "filters mask_top_left; selection will not prune past-diagonal "
                     "blocks. attention kernel still applies the mask.\n",
                     mask_str.c_str(),
                     static_cast<int>(mask.type));

    // If cdfthreshd >= 0, use CDF mode; otherwise use topk mode
    if(cdfthreshd >= 0.0f)
        topk = -1.0f;

    constexpr ck_tile::index_t BLKQ = 64;
    constexpr ck_tile::index_t BLKK = 128;

    if(hdim_q != 128 || hdim_v != 128)
    {
        std::cout << "\n>>> TEST SKIPPED <<<\n"
                  << "Kernel instances are generated for hdim=128 only.\n";
        return true;
    }

    ck_tile::index_t num_q_blocks = (seqlen_q + BLKQ - 1) / BLKQ;
    ck_tile::index_t num_k_blocks = (seqlen_k + BLKK - 1) / BLKK;

    std::string prec_str = std::is_same_v<T, ck_tile::half_t> ? "fp16" : "bf16";
    std::cout << "[" << pipeline << "|" << prec_str << "] b=" << batch << " h=" << nhead
              << " s=" << seqlen_q << " d=" << hdim_q << " topk=" << topk << " sim1=" << simthreshd1
              << std::flush;

    // ---- allocate host tensors ----
    auto q_host      = make_qkv_tensor<T>(batch, nhead, seqlen_q, hdim_q, i_perm);
    auto k_host      = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_q, i_perm);
    auto v_host      = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_v, i_perm);
    auto output_host = o_perm ? ck_tile::HostTensor<T>({batch, nhead, seqlen_q, hdim_v})
                              : ck_tile::HostTensor<T>({batch, seqlen_q, nhead, hdim_v});

    ck_tile::HostTensor<uint8_t> block_map_host({batch, nhead, num_q_blocks, num_k_blocks});
    ck_tile::HostTensor<int32_t> lut_host({batch, nhead, num_q_blocks, num_k_blocks});
    ck_tile::HostTensor<int32_t> valid_block_num_host({batch, nhead, num_q_blocks});

    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_host);

    // ---- device tensors ----
    ck_tile::DeviceMem q_dev(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_dev(output_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem block_map_dev(block_map_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lut_dev(lut_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem valid_bn_dev(valid_block_num_host.get_element_space_size_in_bytes());

    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());
    o_dev.SetZero();
    block_map_dev.SetZero();
    lut_dev.SetZero();
    valid_bn_dev.SetZero();

    // ---- strides (BHSD when i_perm=true) ----
    auto q_strides = q_host.get_strides();
    auto k_strides = k_host.get_strides();
    auto v_strides = v_host.get_strides();
    auto o_strides = output_host.get_strides();

    float scale_s = 1.0f / std::sqrt(static_cast<float>(hdim_q));

    // ---- build blockmap args ----
    sparge_blockmap_traits bmap_traits;
    bmap_traits.data_type = std::is_same_v<T, ck_tile::half_t> ? "fp16" : "bf16";
    bmap_traits.hdim_q    = hdim_q;

    sparge_blockmap_args bmap_args;
    bmap_args.q_ptr               = q_dev.GetDeviceBuffer();
    bmap_args.k_ptr               = k_dev.GetDeviceBuffer();
    bmap_args.batch               = batch;
    bmap_args.seqlen_q            = seqlen_q;
    bmap_args.seqlen_k            = seqlen_k;
    bmap_args.hdim_q              = hdim_q;
    bmap_args.nhead_q             = nhead;
    bmap_args.nhead_k             = nhead_k;
    bmap_args.stride_q            = q_strides[i_perm ? 2 : 1];
    bmap_args.stride_k            = k_strides[i_perm ? 2 : 1];
    bmap_args.nhead_stride_q      = q_strides[i_perm ? 1 : 2];
    bmap_args.nhead_stride_k      = k_strides[i_perm ? 1 : 2];
    bmap_args.batch_stride_q      = q_strides[0];
    bmap_args.batch_stride_k      = k_strides[0];
    bmap_args.simthreshd1         = simthreshd1;
    bmap_args.cdfthreshd          = (topk < 0.0f) ? cdfthreshd : -1.0f;
    bmap_args.topk                = topk;
    bmap_args.scale               = scale_s;
    bmap_args.block_map_ptr       = block_map_dev.GetDeviceBuffer();
    bmap_args.lut_ptr             = (pipeline == "vsa") ? lut_dev.GetDeviceBuffer() : nullptr;
    bmap_args.valid_block_num_ptr = (pipeline == "vsa") ? valid_bn_dev.GetDeviceBuffer() : nullptr;
    bmap_args.mask_type           = mask.type;
    bmap_args.attention_sink      = attention_sink;

    // K-stats workspace: caller-owned, sized via host helper, allocated once outside any timing.
    const size_t ws_bytes = sparge_blockmap_get_workspace_size(bmap_traits, bmap_args);
    ck_tile::DeviceMem kstats_ws_dev(ws_bytes);
    bmap_args.workspace_ptr = kstats_ws_dev.GetDeviceBuffer();

    // Per-head superparam buffers, all sized [nhead_q] to match SpargeAttn upstream contract.
    // K-side kernel reads only the first nhead_k entries via [hk].
    // Filled with scalar broadcast; per-head index correctness verified by separate unit test.
    ck_tile::DeviceMem topk_per_head_dev(static_cast<size_t>(nhead) * sizeof(float));
    ck_tile::DeviceMem sim1_per_head_dev(static_cast<size_t>(nhead) * sizeof(float));
    ck_tile::DeviceMem cdf_per_head_dev(static_cast<size_t>(nhead) * sizeof(float));
    {
        std::vector<float> topk_h(nhead, topk);
        std::vector<float> sim1_h(nhead, simthreshd1);
        std::vector<float> cdf_h(nhead, cdfthreshd);
        topk_per_head_dev.ToDevice(topk_h.data());
        sim1_per_head_dev.ToDevice(sim1_h.data());
        cdf_per_head_dev.ToDevice(cdf_h.data());
        bmap_args.topk_per_head_ptr =
            static_cast<const float*>(topk_per_head_dev.GetDeviceBuffer());
        bmap_args.simthreshd1_per_head_ptr =
            static_cast<const float*>(sim1_per_head_dev.GetDeviceBuffer());
        bmap_args.cdfthreshd_per_head_ptr =
            static_cast<const float*>(cdf_per_head_dev.GetDeviceBuffer());
    }

    // Optional per-head pv_threshold buffer. CLI comma list (length must match
    // nhead); empty list -> scalar broadcast (single launch via host).
    ck_tile::DeviceMem pv_per_head_dev(static_cast<size_t>(nhead) * sizeof(float));
    std::vector<float> pv_per_head_host;
    bool use_pv_per_head = false;
    if(!pv_per_head_s.empty())
    {
        std::stringstream ss(pv_per_head_s);
        std::string item;
        while(std::getline(ss, item, ','))
        {
            if(!item.empty())
                pv_per_head_host.push_back(std::stof(item));
        }
        if(static_cast<ck_tile::index_t>(pv_per_head_host.size()) != nhead)
        {
            std::cerr << "\n[pv_threshold_per_head] length " << pv_per_head_host.size()
                      << " != h=" << nhead << std::endl;
            return false;
        }
        pv_per_head_dev.ToDevice(pv_per_head_host.data());
        use_pv_per_head = true;
    }

    // ---- build attention args ----
    ck_tile::stream_config stream_cfg;
    stream_cfg.stream_id_   = nullptr;
    stream_cfg.time_kernel_ = true;
    stream_cfg.log_level_   = kname;
    stream_cfg.cold_niters_ = warmup;
    stream_cfg.nrepeat_     = repeat;

    // Quant-plumbing scaffold: optional unit-scale (= 1.0f) descale buffers.
    // Per-block granularity matches the pipeline's traits: kBlockScaleSizeQ=kM0 (=BLKQ),
    // kBlockScaleSizeK=kN0 (=BLKK). Buffer shape is [batch, nhead, ceil(seqlen/blk)] floats.
    // -qscale=bs + -pipeline=vsa + fp16/hdim=128/bm0=64 routes to the
    // hand-written BLOCKSCALE oneshot (kDoFp8StaticQuant=true); the pipeline
    // body still (void)-casts the descale ptrs at this milestone, so the
    // unit-scale buffers exercise host plumbing + Kargs forwarding only.
    ck_tile::DeviceMem q_descale_dev;
    ck_tile::DeviceMem k_descale_dev;
    ck_tile::DeviceMem v_descale_dev;
    const void* q_descale_ptr = nullptr;
    const void* k_descale_ptr = nullptr;
    const void* v_descale_ptr = nullptr;
    // Hoisted to function scope so the matched-shape int8 BLOCKSCALE branch
    // below can refill them from the real fp16 Q/K and re-upload to device.
    std::vector<float> q_descale_host;
    std::vector<float> k_descale_host;
    std::vector<float> v_descale_host;
    if(qscale_bs)
    {
        const ck_tile::index_t num_q_blocks_qs = (seqlen_q + BLKQ - 1) / BLKQ;
        const ck_tile::index_t num_k_blocks_qs = (seqlen_k + BLKK - 1) / BLKK;
        const size_t q_elts =
            static_cast<size_t>(batch) * nhead * static_cast<size_t>(num_q_blocks_qs);
        const size_t k_elts =
            static_cast<size_t>(batch) * nhead_k * static_cast<size_t>(num_k_blocks_qs);
        // V uses K-side blocking (V is sequence-major like K).
        const size_t v_elts = k_elts;

        // BLOCKSCALE host fill: when -qscale=bs + the matched-shape int8
        // BLOCKSCALE oneshot is active (fp16/hdim=128/bm0=64), the per-block
        // descale tensors are populated from the real fp16 Q/K below right
        // before quantization (so int8 Q/K + per-block scales agree). For
        // any other shape we fall through to the NO_SCALE codegen path,
        // which doesn't consume the descale buffers — unit-scale is fine.
        q_descale_host.assign(q_elts, 1.0f);
        k_descale_host.assign(k_elts, 1.0f);
        v_descale_host.assign(v_elts, 1.0f);

        // DeviceMem has no copy/move assignment; use Realloc to populate the
        // default-constructed instances safely (avoid use-after-free from
        // implicit assignment of a temporary DeviceMem).
        q_descale_dev.Realloc(q_elts * sizeof(float));
        k_descale_dev.Realloc(k_elts * sizeof(float));
        v_descale_dev.Realloc(v_elts * sizeof(float));
        q_descale_dev.ToDevice(q_descale_host.data());
        k_descale_dev.ToDevice(k_descale_host.data());
        v_descale_dev.ToDevice(v_descale_host.data());

        q_descale_ptr = q_descale_dev.GetDeviceBuffer();
        k_descale_ptr = k_descale_dev.GetDeviceBuffer();
        v_descale_ptr = v_descale_dev.GetDeviceBuffer();
    }

    float avg_ms = -1.0f;

    if(pipeline == "jenga")
    {
        fmha_jenga_fwd_traits attn_traits;
        attn_traits.hdim_q        = hdim_q;
        attn_traits.hdim_v        = hdim_v;
        attn_traits.data_type     = std::is_same_v<T, ck_tile::half_t> ? "fp16" : "bf16";
        attn_traits.is_v_rowmajor = true;
        attn_traits.mask_type     = mask.type;
        attn_traits.bm0           = BLKQ;

        fmha_jenga_fwd_args attn_args;
        attn_args.q_ptr                     = q_dev.GetDeviceBuffer();
        attn_args.k_ptr                     = k_dev.GetDeviceBuffer();
        attn_args.v_ptr                     = v_dev.GetDeviceBuffer();
        attn_args.block_relation_onehot_ptr = block_map_dev.GetDeviceBuffer();
        attn_args.o_ptr                     = o_dev.GetDeviceBuffer();
        attn_args.seqlen_q                  = seqlen_q;
        attn_args.seqlen_k                  = seqlen_k;
        attn_args.batch                     = batch;
        attn_args.max_seqlen_q              = seqlen_q;
        attn_args.hdim_q                    = hdim_q;
        attn_args.hdim_v                    = hdim_v;
        attn_args.nhead_q                   = nhead;
        attn_args.nhead_k                   = nhead_k;
        attn_args.scale_s                   = scale_s;
        attn_args.stride_q                  = q_strides[i_perm ? 2 : 1];
        attn_args.stride_k                  = k_strides[i_perm ? 2 : 1];
        attn_args.stride_v                  = v_strides[i_perm ? 2 : 1];
        attn_args.stride_o                  = o_strides[o_perm ? 2 : 1];
        attn_args.nhead_stride_q            = q_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_k            = k_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_v            = v_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_o            = o_strides[o_perm ? 1 : 2];
        attn_args.batch_stride_q            = q_strides[0];
        attn_args.batch_stride_k            = k_strides[0];
        attn_args.batch_stride_v            = v_strides[0];
        attn_args.batch_stride_o            = o_strides[0];
        attn_args.window_size_left          = mask.left;
        attn_args.window_size_right         = mask.right;
        attn_args.mask_type                 = static_cast<ck_tile::index_t>(mask.type);
        // P1 plumbing scaffold: descale buffers (nullptr unless -qscale=bs).
        attn_args.q_descale_ptr = q_descale_ptr;
        attn_args.k_descale_ptr = k_descale_ptr;
        attn_args.v_descale_ptr = v_descale_ptr;

        avg_ms = sparge_jenga_fwd(bmap_traits, bmap_args, attn_traits, attn_args, stream_cfg);
    }
    else if(pipeline == "vsa")
    {
        // -pipeline=vsa dispatches the sparge family with SpargeAttn §4.4
        // PV-skip; pv_threshold = +1e30 disables skip (matches plain vsa).
        fmha_sparge_fwd_traits attn_traits;
        attn_traits.hdim_q        = hdim_q;
        attn_traits.hdim_v        = hdim_v;
        attn_traits.data_type     = std::is_same_v<T, ck_tile::half_t> ? "fp16" : "bf16";
        attn_traits.is_v_rowmajor = true;
        attn_traits.mask_type     = mask.type;
        attn_traits.bm0           = BLKQ;

        fmha_sparge_fwd_args attn_args;
        attn_args.q_ptr               = q_dev.GetDeviceBuffer();
        attn_args.k_ptr               = k_dev.GetDeviceBuffer();
        attn_args.v_ptr               = v_dev.GetDeviceBuffer();
        attn_args.lut_ptr             = lut_dev.GetDeviceBuffer();
        attn_args.valid_block_num_ptr = valid_bn_dev.GetDeviceBuffer();
        attn_args.o_ptr               = o_dev.GetDeviceBuffer();
        attn_args.seqlen_q            = seqlen_q;
        attn_args.seqlen_k            = seqlen_k;
        attn_args.batch               = batch;
        attn_args.max_seqlen_q        = seqlen_q;
        attn_args.hdim_q              = hdim_q;
        attn_args.hdim_v              = hdim_v;
        attn_args.nhead_q             = nhead;
        attn_args.nhead_k             = nhead_k;
        attn_args.scale_s             = scale_s;
        attn_args.hp.pv_threshold     = pv_threshold;
        attn_args.hp.pv_skip_compile  = (pv_skip_compile != 0);
        attn_args.hp.pv_mode_compile  = pv_mode_compile; // 0=none,1=warp,2=block
        // CLI per-head list -> device buffer to the combined wrapper; host code
        // there partitions heads into 2 buckets and issues per-bucket launches.
        attn_args.hp.pv_threshold_per_head_ptr =
            use_pv_per_head ? static_cast<const float*>(pv_per_head_dev.GetDeviceBuffer())
                            : nullptr;
        attn_args.stride_q          = q_strides[i_perm ? 2 : 1];
        attn_args.stride_k          = k_strides[i_perm ? 2 : 1];
        attn_args.stride_v          = v_strides[i_perm ? 2 : 1];
        attn_args.stride_o          = o_strides[o_perm ? 2 : 1];
        attn_args.nhead_stride_q    = q_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_k    = k_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_v    = v_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_o    = o_strides[o_perm ? 1 : 2];
        attn_args.batch_stride_q    = q_strides[0];
        attn_args.batch_stride_k    = k_strides[0];
        attn_args.batch_stride_v    = v_strides[0];
        attn_args.batch_stride_o    = o_strides[0];
        attn_args.window_size_left  = mask.left;
        attn_args.window_size_right = mask.right;
        attn_args.mask_type         = static_cast<ck_tile::index_t>(mask.type);
        // P1 plumbing scaffold: descale buffers (nullptr unless -qscale=bs).
        attn_args.q_descale_ptr = q_descale_ptr;
        attn_args.k_descale_ptr = k_descale_ptr;
        attn_args.v_descale_ptr = v_descale_ptr;

        if(qscale_bs)
        {
            // BLOCKSCALE (kDoFp8StaticQuant=true) hand-written variant. Only
            // fp16 / hdim=128 / bm0=64 (BLKQ) is wired today; if any other
            // shape is requested the dispatch falls through to the codegen
            // NO_SCALE path so we don't silently mis-route.
            const bool fp16_d128_bm64 =
                std::is_same_v<T, ck_tile::half_t> && hdim_q == 128 && hdim_v == 128;
            if(!fp16_d128_bm64)
            {
                std::cerr << "[test_sparge] -qscale=bs BLOCKSCALE oneshot only supports "
                             "fp16/hdim=128/bm0=64; falling back to NO_SCALE codegen path."
                          << std::endl;
                avg_ms = sparge_sparge_fwd_combined(
                    bmap_traits, bmap_args, attn_traits, attn_args, stream_cfg);
            }
            else
            {
                // S3c2: real host-side per-block absmax/127 quantization.
                // For each (batch, head, q_blk) chunk of BLKQ rows compute
                //   q_descale[blk] = absmax(fp16_Q[blk]) / 127
                //   int8_Q[blk]    = clip(round(fp16_Q[blk] / q_descale[blk]), -127, 127)
                // Same for K with BLKK. The fp16 q_dev/k_dev stay live for
                // kstats / blockmap / reference paths; int8 buffers are
                // separate so the int8 kernel reads packed int8 bytes while
                // the reference path keeps fp16. The kernel uses host-passed
                // strides (in elements) via reinterpret_cast<QDataType*>, so
                // an int8 buffer of size batch*nhead*seqlen*hdim bytes packs
                // naturally with the BHSD layout.
                const size_t q_int8_bytes = static_cast<size_t>(batch) * nhead * seqlen_q * hdim_q;
                const size_t k_int8_bytes =
                    static_cast<size_t>(batch) * nhead_k * seqlen_k * hdim_q;
                std::vector<int8_t> q_int8_host(q_int8_bytes);
                std::vector<int8_t> k_int8_host(k_int8_bytes);

                const ck_tile::index_t nqb = (seqlen_q + BLKQ - 1) / BLKQ;
                const ck_tile::index_t nkb = (seqlen_k + BLKK - 1) / BLKK;

                auto quantize_per_block = [&](const auto& src_host,
                                              std::vector<int8_t>& dst,
                                              std::vector<float>& desc,
                                              ck_tile::index_t nh,
                                              ck_tile::index_t slen,
                                              ck_tile::index_t blk,
                                              ck_tile::index_t nb) {
                    for(ck_tile::index_t b = 0; b < batch; ++b)
                    {
                        for(ck_tile::index_t h = 0; h < nh; ++h)
                        {
                            for(ck_tile::index_t kb = 0; kb < nb; ++kb)
                            {
                                const ck_tile::index_t s_beg = kb * blk;
                                const ck_tile::index_t s_end =
                                    std::min<ck_tile::index_t>(s_beg + blk, slen);
                                // absmax over this block
                                float absmax = 0.f;
                                for(ck_tile::index_t s = s_beg; s < s_end; ++s)
                                {
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                    {
                                        const float fv = ck_tile::type_convert<float>(
                                            i_perm ? src_host(b, h, s, d) : src_host(b, s, h, d));
                                        const float a = fv < 0.f ? -fv : fv;
                                        if(a > absmax)
                                            absmax = a;
                                    }
                                }
                                // Avoid div-by-zero for an all-zero block; use
                                // descale=1 (encodes int8 = 0 round-trip safely).
                                const float desc_val = absmax > 0.f ? absmax / 127.0f : 1.0f;
                                desc[(static_cast<size_t>(b) * nh + h) * nb + kb] = desc_val;
                                const float inv_desc                              = 1.0f / desc_val;
                                for(ck_tile::index_t s = s_beg; s < s_end; ++s)
                                {
                                    for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                                    {
                                        const float fv = ck_tile::type_convert<float>(
                                            i_perm ? src_host(b, h, s, d) : src_host(b, s, h, d));
                                        float q = fv * inv_desc;
                                        // round-half-away-from-zero, clip to [-127,127]
                                        q = q >= 0.f ? std::floor(q + 0.5f) : std::ceil(q - 0.5f);
                                        if(q > 127.f)
                                            q = 127.f;
                                        if(q < -127.f)
                                            q = -127.f;
                                        // Pack to int8 using the same flat-element index the
                                        // kernel reads (BHSD when i_perm=1, BSHD when 0).
                                        size_t idx;
                                        if(i_perm)
                                            idx = ((static_cast<size_t>(b) * nh + h) * slen + s) *
                                                      hdim_q +
                                                  d;
                                        else
                                            idx = ((static_cast<size_t>(b) * slen + s) * nh + h) *
                                                      hdim_q +
                                                  d;
                                        dst[idx] = static_cast<int8_t>(static_cast<int>(q));
                                    }
                                }
                            }
                        }
                    }
                };

                quantize_per_block(q_host, q_int8_host, q_descale_host, nhead, seqlen_q, BLKQ, nqb);
                quantize_per_block(
                    k_host, k_int8_host, k_descale_host, nhead_k, seqlen_k, BLKK, nkb);

                // Re-upload descale buffers populated from real data
                // (overrides the unit-scale fill above).
                q_descale_dev.ToDevice(q_descale_host.data());
                k_descale_dev.ToDevice(k_descale_host.data());

                ck_tile::DeviceMem q_int8_dev(q_int8_bytes);
                ck_tile::DeviceMem k_int8_dev(k_int8_bytes);
                q_int8_dev.ToDevice(q_int8_host.data());
                k_int8_dev.ToDevice(k_int8_host.data());

                fmha_sparge_fwd_args int8_attn_args = attn_args;
                int8_attn_args.q_ptr                = q_int8_dev.GetDeviceBuffer();
                int8_attn_args.k_ptr                = k_int8_dev.GetDeviceBuffer();

                if(stream_cfg.log_level_ > 0)
                    std::cout << ", sparge_kstats_" << bmap_traits.data_type << "_d"
                              << bmap_traits.hdim_q << ", sparge_blockmap_" << bmap_traits.data_type
                              << "_d" << bmap_traits.hdim_q << std::flush;
                avg_ms = ck_tile::launch_kernel(
                    stream_cfg,
                    [=](const ck_tile::stream_config& s_) {
                        sparge_kstats_fwd_oneshot(bmap_traits, bmap_args, s_);
                    },
                    [=](const ck_tile::stream_config& s_) {
                        sparge_blockmap_only_fwd_oneshot(bmap_traits, bmap_args, s_);
                    },
                    [=](const ck_tile::stream_config& s_) {
                        fmha_sparge_int8_fwd_oneshot_fp16_d128_bm64(s_, int8_attn_args);
                    });
            }
        }
        else
        {
            avg_ms = sparge_sparge_fwd_combined(
                bmap_traits, bmap_args, attn_traits, attn_args, stream_cfg);
        }
    }
    else
    {
        std::cerr << "Unknown pipeline: " << pipeline << " (use jenga or vsa)\n";
        return false;
    }

    // ---- TFLOPS calculation (dense FMHA formula, so sparsity gains show as higher TFLOPS) ----
    std::size_t flop = static_cast<std::size_t>(batch) * nhead *
                       (static_cast<std::size_t>(2) * seqlen_q * seqlen_k * hdim_q +
                        static_cast<std::size_t>(2) * seqlen_q * seqlen_k * hdim_v);
    float tflops = (avg_ms > 0.f) ? static_cast<float>(flop) / 1.E9f / avg_ms : 0.f;

    if(avg_ms > 0.f)
    {
        std::cout << std::fixed << ", " << std::setprecision(3) << avg_ms << " ms, "
                  << std::setprecision(2) << tflops << " TFlops" << std::flush;
    }

    // ---- copy results back ----
    o_dev.FromDevice(output_host.data());
    block_map_dev.FromDevice(block_map_host.data());

#ifdef P2_SCALE_DUMP
    // Task #180 host-side dump of k_scale_lut + pooled_k[0,0,0,:] for cross-check.
    // Compiled only when -DP2_SCALE_DUMP=1 passed; reverted after measurement.
    {
        const auto layout_dump = sparge_blockmap_compute_workspace_layout(bmap_traits, bmap_args);
        const int kN0_dump     = (hdim_q == 128) ? 128 : ((hdim_q == 64) ? 256 : 0);
        const int N_k_dump = (kN0_dump > 0) ? ck_tile::integer_divide_ceil(seqlen_k, kN0_dump) : 0;
        const size_t total_blocks_dump = static_cast<size_t>(batch) * nhead_k * N_k_dump;
        std::vector<float> k_scale_h(total_blocks_dump, 0.f);
        std::vector<uint16_t> pooled_first_d(hdim_q, 0);
        auto* ws_base = static_cast<char*>(kstats_ws_dev.GetDeviceBuffer());
        (void)hipMemcpy(k_scale_h.data(),
                        ws_base + layout_dump.k_scale_offset,
                        total_blocks_dump * sizeof(float),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(pooled_first_d.data(),
                        ws_base + layout_dump.pooled_k_offset,
                        hdim_q * sizeof(uint16_t),
                        hipMemcpyDeviceToHost);

        float smin   = std::numeric_limits<float>::infinity();
        float smax   = -std::numeric_limits<float>::infinity();
        double sum   = 0.0;
        size_t zeros = 0, infnan = 0;
        for(size_t i = 0; i < total_blocks_dump; ++i)
        {
            float v = k_scale_h[i];
            if(std::isnan(v) || std::isinf(v))
            {
                ++infnan;
                continue;
            }
            if(v == 0.f)
                ++zeros;
            if(v < smin)
                smin = v;
            if(v > smax)
                smax = v;
            sum += v;
        }
        float mean = (total_blocks_dump > 0) ? static_cast<float>(sum / total_blocks_dump) : 0.f;

        std::cout << "\n[P2_SCALE_DUMP] prec=" << bmap_traits.data_type << " N_k=" << N_k_dump
                  << " total=" << total_blocks_dump << " min=" << smin << " max=" << smax
                  << " mean=" << mean << " zeros=" << zeros << " inf_nan=" << infnan << "\n";
        std::cout << "[P2_SCALE_DUMP] first8:";
        for(size_t i = 0; i < std::min<size_t>(8, total_blocks_dump); ++i)
            std::cout << " " << k_scale_h[i];
        std::cout << "\n";

        // Host cross-check on block [0,0,0]: absmax(pooled_k_mean[0,0,0,:])/127
        float pmax = 0.f;
        for(int d = 0; d < hdim_q; ++d)
        {
            float fv;
            if(bmap_traits.data_type == "fp16")
            {
                ck_tile::half_t h;
                std::memcpy(&h, &pooled_first_d[d], sizeof(uint16_t));
                fv = ck_tile::type_convert<float>(h);
            }
            else
            {
                ck_tile::bf16_t b;
                std::memcpy(&b, &pooled_first_d[d], sizeof(uint16_t));
                fv = ck_tile::type_convert<float>(b);
            }
            float a = fv < 0.f ? -fv : fv;
            if(a > pmax)
                pmax = a;
        }
        float host_val   = pmax / 127.0f;
        float kernel_val = k_scale_h[0];
        std::cout << "[P2_SCALE_DUMP] xcheck block[0,0,0] kernel=" << kernel_val
                  << " host(pooled_k_mean/127)=" << host_val
                  << " abs_diff=" << std::abs(kernel_val - host_val) << "\n";
    }
#endif

    // ---- optional raw output dump (for bit-identical baseline comparison) ----
    if(!dump_o_path.empty())
    {
        std::ofstream ofs(dump_o_path, std::ios::binary | std::ios::trunc);
        if(!ofs)
        {
            std::cerr << "\n  [dump_o] failed to open " << dump_o_path << std::endl;
        }
        else
        {
            ofs.write(reinterpret_cast<const char*>(output_host.data()),
                      static_cast<std::streamsize>(output_host.get_element_space_size_in_bytes()));
            std::cout << "\n  [dump_o] wrote " << output_host.get_element_space_size_in_bytes()
                      << " bytes to " << dump_o_path;
        }
    }

    // ---- count active blocks ----
    ck_tile::index_t total_blocks  = batch * nhead * num_q_blocks * num_k_blocks;
    ck_tile::index_t active_blocks = 0;
    for(size_t i = 0; i < block_map_host.mData.size(); ++i)
        if(block_map_host.mData[i])
            active_blocks++;
    float actual_sparsity =
        1.0f - static_cast<float>(active_blocks) / static_cast<float>(total_blocks);
    std::cout << ", sparsity=" << std::setprecision(2) << actual_sparsity << "(" << active_blocks
              << "/" << total_blocks << ")" << std::flush;

    // ---- validation ----
    bool pass = true;
    if(do_validation)
    {
        auto q_ref = to_bhsd(q_host, i_perm);
        auto k_ref = to_bhsd(k_host, i_perm);
        auto v_ref = to_bhsd(v_host, i_perm);

        // CPU reference lacks causal mask + attention_sink; skip block_map
        // cross-check + VSA LUT self-consistency when either is in effect. The
        // attention-output check below still runs (consumes GPU bmap).
        const bool skip_cpu_bm_check = (mask.type != mask_enum::no_mask) || attention_sink;

        bool bm_pass  = true;
        bool lut_pass = true;
        if(!skip_cpu_bm_check)
        {

            sparge::SpargeParams sp;
            sp.BLKQ        = BLKQ;
            sp.BLKK        = BLKK;
            sp.simthreshd1 = simthreshd1;
            sp.cdfthreshd  = cdfthreshd;
            sp.topk        = topk;
            sp.i_perm      = i_perm;

            auto block_map_cpu = sparge::build_block_map_meansim<T>(q_host, k_host, sp);

            size_t bm_total          = block_map_host.mData.size();
            size_t bm_mismatch       = 0;
            size_t shown             = 0;
            constexpr size_t MAXSHOW = 10;
            std::cout << "\n  [block_map cross-check] total=" << bm_total;
            for(size_t i = 0; i < bm_total; ++i)
            {
                uint8_t g = block_map_host.mData[i];
                uint8_t c = block_map_cpu.mData[i];
                if(g != c)
                {
                    if(shown < MAXSHOW)
                    {
                        size_t k_idx = i % num_k_blocks;
                        size_t q_idx = (i / num_k_blocks) % num_q_blocks;
                        size_t h_idx = (i / (num_k_blocks * num_q_blocks)) % nhead;
                        size_t b_idx = i / (num_k_blocks * num_q_blocks * nhead);
                        std::cout << "\n    miss[" << shown << "] (b=" << b_idx << ",h=" << h_idx
                                  << ",qb=" << q_idx << ",kb=" << k_idx << ") gpu=" << int(g)
                                  << " cpu=" << int(c);
                        ++shown;
                    }
                    ++bm_mismatch;
                }
            }
            bm_pass        = (bm_mismatch == 0);
            float bm_ratio = bm_total ? 100.0f * float(bm_mismatch) / float(bm_total) : 0.0f;
            std::cout << "\n  [block_map cross-check] mismatch=" << bm_mismatch << "/" << bm_total
                      << " (" << std::setprecision(4) << bm_ratio << "%) "
                      << (bm_pass ? "PASS" : "FAIL");

            auto cpu_lut     = sparge::block_map_to_vsa_lut_delta<uint8_t>(block_map_cpu);
            size_t lut_fails = 0;
            for(ck_tile::index_t b = 0; b < batch && lut_fails < MAXSHOW; ++b)
            {
                for(ck_tile::index_t h = 0; h < nhead && lut_fails < MAXSHOW; ++h)
                {
                    for(ck_tile::index_t qb = 0; qb < num_q_blocks && lut_fails < MAXSHOW; ++qb)
                    {
                        int32_t valid        = cpu_lut.valid_block_num(b, h, qb);
                        int32_t active_count = 0;
                        for(ck_tile::index_t kb = 0; kb < num_k_blocks; ++kb)
                            if(block_map_cpu(b, h, qb, kb))
                                ++active_count;
                        int32_t recon_kb = 0;
                        bool delta_ok    = true;
                        for(int32_t i = 0; i < valid; ++i)
                        {
                            int32_t d = cpu_lut.lut(b, h, qb, i);
                            if(d < 0)
                            {
                                delta_ok = false;
                                break;
                            }
                            recon_kb += d;
                            if(recon_kb >= num_k_blocks)
                            {
                                delta_ok = false;
                                break;
                            }
                            if(!block_map_cpu(b, h, qb, recon_kb))
                            {
                                delta_ok = false;
                                break;
                            }
                        }
                        if(valid != active_count || !delta_ok)
                        {
                            lut_pass = false;
                            if(lut_fails < MAXSHOW)
                                std::cout << "\n    lut_fail (b=" << b << ",h=" << h << ",qb=" << qb
                                          << ") valid=" << valid << " active=" << active_count
                                          << " delta_ok=" << delta_ok;
                            ++lut_fails;
                        }
                    }
                }
            }
            std::cout << "\n  [VSA LUT self-consistency] " << (lut_pass ? "PASS" : "FAIL");
        } // end if(!skip_cpu_bm_check)
        else
        {
            std::cout << "\n  [block_map cross-check] SKIPPED (mask/sink active; CPU ref lacks)";
            std::cout << "\n  [VSA LUT self-consistency] SKIPPED";
        }

        ck_tile::HostTensor<T> output_ref({batch, nhead, seqlen_q, hdim_v});
        ck_tile::reference_blocked_attention<T, uint8_t>(
            q_ref, k_ref, v_ref, block_map_host, output_ref, BLKQ, BLKK, scale_s);

        auto [rtol, atol] = get_error_tolerance<T>();

        float max_diff    = 0.0f;
        size_t num_errors = 0;

        auto output_host_bhsd = to_bhsd(output_host, o_perm);
        for(size_t i = 0; i < output_host_bhsd.mData.size(); ++i)
        {
            float gpu_val  = to_float_for_compare(output_host_bhsd.mData[i]);
            float ref_val  = to_float_for_compare(output_ref.mData[i]);
            float diff     = std::abs(gpu_val - ref_val);
            float rel_diff = (std::abs(ref_val) > 1e-6f) ? diff / std::abs(ref_val) : diff;

            max_diff = std::max(max_diff, diff);

            if(diff > atol && rel_diff > rtol)
                num_errors++;
        }

        pass = (num_errors == 0) && bm_pass && lut_pass;
        std::cout << "\n  [attention output] " << ((num_errors == 0) ? "PASS" : "FAIL")
                  << "(err=" << num_errors << "/" << output_host_bhsd.mData.size()
                  << " maxdiff=" << max_diff << ")";

        // ----------------------------------------------------------------
        // Independent dense+causal cross-reference.
        //
        // The cells above compare GPU sparge output to
        // `reference_blocked_attention(..., block_map_host, ...)`, which
        // *consumes the GPU-selected block_map*. That only proves GPU vs.
        // its own block-map; it does NOT independently prove the causal
        // formula is correct. A bug shared between the GPU pipeline and a
        // CPU mirror would PASS silently.
        //
        // Independent path: build the dense reference via the same chain
        // 01_fmha uses (reference_batched_gemm + reference_batched_masking
        // with GenericAttentionMask + reference_batched_softmax +
        // reference_batched_gemm). The mask is authored by ops/fmha/block
        // (different code path, different author) so a shared bug is
        // implausible. Sparge with -topk=1.0 selects all causal-valid
        // blocks and so degenerates to dense+causal; output must match.
        // ----------------------------------------------------------------
        if(cross_ref)
        {
            using AccT  = float;
            using CompT = float;

            // Flatten [B, H, S, D] -> [B*H, ...] to feed the 3D
            // reference_batched_* helpers. V is laid out [B*H, D_v, S_k]
            // because reference_batched_gemm wants B in [b, n, k] order
            // and the second GEMM has n=D_v, k=S_k (matches the 01_fmha
            // runner's v_host_ref shape, fmha_fwd_runner.hpp:1903).
            ck_tile::HostTensor<T> q_flat({batch * nhead, seqlen_q, hdim_q});
            ck_tile::HostTensor<T> k_flat({batch * nhead, seqlen_k, hdim_q});
            ck_tile::HostTensor<T> v_flat({batch * nhead, hdim_v, seqlen_k});
            // GQA: q has nhead heads; k/v have nhead_k. Replicate k/v across
            // q's heads via nhead_ratio.
            const ck_tile::index_t nhead_ratio = nhead / nhead_k;
            for(ck_tile::index_t b = 0; b < batch; ++b)
                for(ck_tile::index_t h = 0; h < nhead; ++h)
                {
                    const ck_tile::index_t hk = h / nhead_ratio;
                    for(ck_tile::index_t s = 0; s < seqlen_q; ++s)
                        for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                            q_flat(b * nhead + h, s, d) = q_ref(b, h, s, d);
                    for(ck_tile::index_t s = 0; s < seqlen_k; ++s)
                    {
                        for(ck_tile::index_t d = 0; d < hdim_q; ++d)
                            k_flat(b * nhead + h, s, d) = k_ref(b, hk, s, d);
                        for(ck_tile::index_t d = 0; d < hdim_v; ++d)
                            v_flat(b * nhead + h, d, s) = v_ref(b, hk, s, d);
                    }
                }

            // S = Q * K^T * scale
            ck_tile::HostTensor<AccT> s_flat({batch * nhead, seqlen_q, seqlen_k});
            ck_tile::reference_batched_gemm<T, T, AccT, AccT>(q_flat,
                                                              k_flat,
                                                              s_flat,
                                                              ck_tile::identity{},
                                                              ck_tile::identity{},
                                                              ck_tile::scales(scale_s));

            // Apply mask using the 01_fmha GenericAttentionMask path. This
            // mask object's IsOutOfSinkBound is authored in
            // ops/fmha/block/block_masking.hpp - unrelated to the sparge
            // per-block kb*BLKK >= (qb+1)*BLKQ formula.
            using NoMask     = ck_tile::GenericAttentionMask<false>;
            using CausalMask = ck_tile::GenericAttentionMask<true, false>;
            if(mask.type == mask_enum::no_mask)
            {
                ck_tile::reference_batched_masking<AccT>(s_flat, NoMask{seqlen_q, seqlen_k});
            }
            else if(mask.type == mask_enum::mask_top_left)
            {
                ck_tile::reference_batched_masking<AccT>(
                    s_flat,
                    ck_tile::make_generic_attention_mask_from_lr_window<CausalMask>(
                        -1, 0, seqlen_q, seqlen_k, /*is_top_left=*/true));
            }
            else
            {
                std::cout
                    << "\n  [cross-ref] SKIPPED (unsupported -mask for cross-ref; use 0 or t)";
                std::cout << std::endl;
                return pass;
            }

            // softmax
            ck_tile::HostTensor<CompT> p_flat({batch * nhead, seqlen_q, seqlen_k});
            ck_tile::reference_batched_softmax<AccT, CompT, CompT>(s_flat, p_flat);

            // out = P * V (treat P as the new A)
            // reference_batched_gemm expects b_b_n_k, so V is already [B*H,S_k,D_v].
            ck_tile::HostTensor<T> out_cross({batch * nhead, seqlen_q, hdim_v});
            ck_tile::reference_batched_gemm<CompT, T, AccT, T>(p_flat, v_flat, out_cross);

            // Compare GPU output [B,H,S,D_v] (bhsd) vs out_cross [B*H,S,D_v].
            float cross_max_diff  = 0.0f;
            size_t cross_errors   = 0;
            size_t cross_elements = 0;
            // Reuse rtol/atol from the earlier cell (same ck_tile::tuple type).
            for(ck_tile::index_t b = 0; b < batch; ++b)
                for(ck_tile::index_t h = 0; h < nhead; ++h)
                    for(ck_tile::index_t s = 0; s < seqlen_q; ++s)
                        for(ck_tile::index_t d = 0; d < hdim_v; ++d)
                        {
                            float gpu_v = to_float_for_compare(output_host_bhsd(b, h, s, d));
                            float ref_v = to_float_for_compare(out_cross(b * nhead + h, s, d));
                            float diff  = std::abs(gpu_v - ref_v);
                            float rdiff = (std::abs(ref_v) > 1e-6f) ? diff / std::abs(ref_v) : diff;
                            cross_max_diff = std::max(cross_max_diff, diff);
                            if(diff > atol && rdiff > rtol)
                                ++cross_errors;
                            ++cross_elements;
                        }
            const bool cross_pass = (cross_errors == 0);
            std::cout << "\n  [cross-ref indep dense"
                      << (mask.type == mask_enum::mask_top_left ? "+causal] " : "] ")
                      << (cross_pass ? "PASS" : "FAIL") << " (err=" << cross_errors << "/"
                      << cross_elements << " maxdiff=" << cross_max_diff << ")";
            if(!cross_pass)
            {
                // Print first few mismatches for debugging.
                int shown = 0;
                for(ck_tile::index_t b = 0; b < batch && shown < 5; ++b)
                    for(ck_tile::index_t h = 0; h < nhead && shown < 5; ++h)
                        for(ck_tile::index_t s = 0; s < seqlen_q && shown < 5; ++s)
                            for(ck_tile::index_t d = 0; d < hdim_v && shown < 5; ++d)
                            {
                                float gpu_v = to_float_for_compare(output_host_bhsd(b, h, s, d));
                                float ref_v = to_float_for_compare(out_cross(b * nhead + h, s, d));
                                float diff  = std::abs(gpu_v - ref_v);
                                if(diff > atol)
                                {
                                    std::cout << "\n    diff[" << shown << "] (b=" << b
                                              << ",h=" << h << ",sq=" << s << ",d=" << d
                                              << ") gpu=" << std::setprecision(6) << gpu_v
                                              << " ref=" << ref_v << " diff=" << diff;
                                    ++shown;
                                }
                            }
            }
            pass = pass && cross_pass;
        }
    }

    std::cout << std::endl;
    return pass;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cerr << "Failed to parse arguments\n";
        return -1;
    }

    std::string prec = arg_parser.get_str("prec");

    bool test_result = false;
    if(prec == "fp16")
    {
        test_result = run_test<ck_tile::half_t>(arg_parser);
    }
    else if(prec == "bf16")
    {
        test_result = run_test<ck_tile::bf16_t>(arg_parser);
    }
    else
    {
        std::cerr << "Unsupported precision: " << prec << "\n";
        return -1;
    }

    return test_result ? 0 : -1;
}
