// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "sparse_attn_fwd.hpp"
#include "sparse_attn_fwd_runner.hpp"

#include <string>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser
        .insert("api",
                "jenga",
                "sparse attention API:\n"
                "  jenga:        block-sparse attention (one-hot mask; Jenga, arXiv 2505.16864)\n"
                "  vsa:          block sparse attention (LUT format)\n"
                "  sparge: SpargeAttention (preprocess + mask prediction + attention)\n"
                "  sparge_sage: quantized SpargeAttention (INT8 QK, FP8 V; -qscale)")
        .insert("qscale",
                "perwarp",
                "sparge_sage quantization scale mode: perwarp|perblock|perthread|pertensor")
        .insert("qkdtype",
                "int8",
                "sparge_sage Q/K quant dtype: int8 (i8fp8bf16) | fp8 (fp8bf16). V is always fp8.")
        .insert("v", "1", "0:no validation, 1:validation")
        .insert("b", "1", "batch size")
        .insert("h", "4", "num of head, for q")
        .insert("h_k", "-1", "num of head, for k/v, -1 means equal to h")
        .insert("s", "4096", "seqlen_q")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("scale_s",
                "0",
                "softmax scale factor; 0 => 1/sqrt(d) (default).\n"
                "Override for fixed-scale eval / RoPE-aware models.")
        .insert("logits_soft_cap",
                "0",
                "Gemma-style logits soft cap; 0 => disabled.\n"
                "Pre-softmax: s = cap * tanh(s * scale / cap).")
        .insert("sparsity",
                "0.02",
                "target sparsity ratio [0,1). 0=dense, higher=more sparse.\n"
                "  jenga / vsa: random-mask activation probability (skip ratio).\n"
                "  sparge / sparge_sage: passed to the algorithm selected by -sparge_mode (see "
                "below).\n"
                "  default 0.02 -> cdf threshold 0.98 (matches official SpargeAttn meansim).")
        .insert("sparge_mode",
                "cdf",
                "sparge / sparge_sage: block-selection algorithm.\n"
                "  cdf (default): CDF threshold; greedily add blocks until cumulative softmax\n"
                "                 probability >= 1-sparsity (1-0.02 = 0.98, official default).\n"
                "  topk:          pick max(1, round((1-sparsity) * num_k_blocks)) blocks per\n"
                "                 Q-block. Realised sparsity matches -sparsity exactly.")
        .insert("simthreshold",
                "0.6",
                "cosine similarity threshold (sparge & sparge_sage). Official SpargeAttn meansim "
                "default 0.6.")
        .insert("mask",
                "0",
                "0: no mask, 1: top-left(same as 't'), 2:bottom-right(same as 'b')\n"
                "'t', top-left causal mask, 'b', bottom-r causal mask\n"
                "'t:l,r', top-left sliding window attn(swa) with FA style left right size\n"
                "'b:l,r', bottom-r sliding window attn(swa) with FA style left right size\n"
                "(supported by all: `jenga` / `vsa` / `sparge` / `sparge_sage`)")
        .insert(
            "sink", "0", "1: attention sink (always include first K block, sparge & sparge_sage)")
        .insert("pvthreshd",
                "50",
                "P*V runtime block-skip threshold (log2 units) for sparge & sparge_sage\n"
                "(0 = disabled; >0 enables Stage 2). Official SpargeAttn default 50.")
        .insert("perhead_test",
                "0",
                "sparge/sparge_sage: synthesize a per-head hyperparam pattern for smoke test "
                "(requires -h >= 2).")
        .insert(
            "sparsity_per_head",
            "",
            "sparge/sparge_sage: per-Q-head sparsity (nhead_q comma-separated floats). Overrides "
            "-sparsity / -perhead_test; routed to topk/cdf by -sparge_mode (as 1 - sparsity[h]).")
        .insert("sim_per_head",
                "",
                "sparge/sparge_sage: per-Q-head simthreshold (nhead_q floats). Overrides "
                "-simthreshold / -perhead_test (requires -simthreshold > 0).")
        .insert("pvthreshd_per_head",
                "",
                "sparge/sparge_sage: per-Q-head pvthreshd (nhead_q floats). Overrides "
                "-pvthreshd / -perhead_test.")
        .insert("bias",
                "n",
                "n or 0, no bias\n"
                "e(lementwise) or 1, elementwise bias 1*1*sq*sk. e:1, 1*h*sq*sk. e:2, b*h*sq*sk\n"
                "a(libi) or 2, alibi 1*h slope (needs causal mask). a:1, b*h\n"
                "(jenga / vsa / sparge / sparge_sage; batch + group)")
        .insert("smooth_k",
                "1",
                "center K by per-channel mean (selection pool/sim + sage K-quant). 0 disables. "
                "Official SpargeAttn default on.")
        .insert("print_sparsity",
                "0",
                "sparge: 1 = read back actual sparsity; needed for accurate TFlops/GB/s")
        .insert("block_size", "128", "block size for sparse attention (BLKQ=BLKK)")
        .insert("vlayout", "r", "r for row-major(seqlen*hdim), c for col-major(hdim*seqlen)")
        .insert("mode", "0", "kernel mode. 0:batch, 1:group (jenga + vsa + sparge + sparge_sage)")
        .insert("json", "0", "1 to also emit a JSON-Lines summary (prefix 'JSON '). 0 to skip.")
        .insert("jsonfile",
                "",
                "if non-empty, append the JSON summary to this file (one line per call) "
                "instead of stdout.")
        .insert("prec", "fp16", "data type: fp16/bf16")
        .insert("iperm", "1", "permute input, 1: b*h*s*d, 0: b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("seed", "42", "random seed")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "benchmark iterations")
        .insert("timer", "gpu", "timer type: gpu or cpu")
        .insert("kname", "0", "print kernel name");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataTypeConfig>
auto run(const ck_tile::ArgParser& arg_parser)
{
    std::string api                = arg_parser.get_str("api");
    int do_validation              = arg_parser.get_int("v");
    ck_tile::index_t batch         = arg_parser.get_int("b");
    ck_tile::index_t nhead         = arg_parser.get_int("h");
    ck_tile::index_t nhead_k       = arg_parser.get_int("h_k");
    ck_tile::index_t seqlen_q      = arg_parser.get_int("s");
    ck_tile::index_t seqlen_k      = arg_parser.get_int("s_k");
    ck_tile::index_t hdim_q        = arg_parser.get_int("d");
    ck_tile::index_t hdim_v        = arg_parser.get_int("d_v");
    float sparsity                 = arg_parser.get_float("sparsity");
    std::string sparge_mode        = arg_parser.get_str("sparge_mode");
    float simthreshold             = arg_parser.get_float("simthreshold");
    std::string mask_str           = arg_parser.get_str("mask");
    bool attention_sink            = arg_parser.get_bool("sink");
    float pvthreshd                = arg_parser.get_float("pvthreshd");
    bool perhead_test              = arg_parser.get_bool("perhead_test");
    std::string sparsity_per_head  = arg_parser.get_str("sparsity_per_head");
    std::string sim_per_head       = arg_parser.get_str("sim_per_head");
    std::string pvthreshd_per_head = arg_parser.get_str("pvthreshd_per_head");
    bool smooth_k                  = arg_parser.get_bool("smooth_k");
    std::string bias_str           = arg_parser.get_str("bias");
    bool print_sparsity            = arg_parser.get_bool("print_sparsity");
    ck_tile::index_t block_size    = arg_parser.get_int("block_size");
    float scale_s                  = arg_parser.get_float("scale_s");
    float logits_soft_cap          = arg_parser.get_float("logits_soft_cap");
    bool i_perm                    = arg_parser.get_bool("iperm");
    bool o_perm                    = arg_parser.get_bool("operm");
    bool is_v_rowmajor             = (arg_parser.get_str("vlayout") == "r");
    uint32_t seed                  = arg_parser.get_uint32("seed");
    sparse_attn_mode mode          = static_cast<sparse_attn_mode>(arg_parser.get_uint32("mode"));
    bool json_out                  = arg_parser.get_bool("json");
    std::string json_file          = arg_parser.get_str("jsonfile");
    std::string qscale             = arg_parser.get_str("qscale");
    std::string qkdtype            = arg_parser.get_str("qkdtype");

    if(nhead_k < 0)
        nhead_k = nhead;
    if(seqlen_k < 0)
        seqlen_k = seqlen_q;
    if(hdim_v < 0)
        hdim_v = hdim_q;

    if(print_sparsity && api != "sparge")
        std::cerr << "[warn] -print_sparsity is only meaningful for -api=sparge; ignored.\n";

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /* log_level = */ (arg_parser.get_bool("kname") ? 1 : 0),
                                         arg_parser.get_int("warmup"),
                                         arg_parser.get_int("repeat"),
                                         arg_parser.get_str("timer") == std::string("gpu")};

    return sparse_attn_fwd_run<DataTypeConfig>(api,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               seqlen_q,
                                               seqlen_k,
                                               hdim_q,
                                               hdim_v,
                                               sparsity,
                                               simthreshold,
                                               mask_str,
                                               attention_sink,
                                               block_size,
                                               i_perm,
                                               o_perm,
                                               is_v_rowmajor,
                                               seed,
                                               do_validation,
                                               pvthreshd,
                                               sparge_mode,
                                               perhead_test,
                                               sparsity_per_head,
                                               sim_per_head,
                                               pvthreshd_per_head,
                                               smooth_k,
                                               print_sparsity,
                                               stream_config,
                                               mode,
                                               json_out,
                                               json_file,
                                               bias_str,
                                               scale_s,
                                               logits_soft_cap,
                                               qscale,
                                               qkdtype);
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, arg_parser] = create_args(argc, argv);
        if(!result)
            return -1;

        const std::string data_type = arg_parser.get_str("prec");
        sparse_attn_result res;
        if(data_type == "fp16")
        {
            res = run<FmhaSparseFwdFp16>(arg_parser);
        }
        else if(data_type == "bf16")
        {
            res = run<FmhaSparseFwdBf16>(arg_parser);
        }
        else
        {
            std::cerr << "Unsupported precision: " << data_type << std::endl;
            return -1;
        }
        // POSIX exit codes: 0=success, 1=failure, 77=skipped.
        switch(res)
        {
        case sparse_attn_result::success: return 0;
        case sparse_attn_result::skipped: return 77;
        case sparse_attn_result::failure: return 1;
        }
        return 1;
    }
    catch(const std::invalid_argument& e)
    {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        return -1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }
}
