// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// gfx1250 fp8 decode attention over the SHUFFLE 5D paged KV cache (see
// ck_tile/ops/fmha/kernel/fmha_fwd_decode_paged5d_kernel.hpp), with a CPU reference.

#include <cstdio>
#include <iostream>

#include "fmha_decode_paged5d_runner.hpp"

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "1", "batch size")
        .insert("h", "16", "number of query heads")
        .insert("h_k", "1", "number of kv heads (h / h_k <= 16)")
        .insert("s_k", "8192", "context length; comma separated for one per batch")
        .insert("window", "2048", "keys attended, the query's own included; 0 = no window")
        .insert("splits", "0", "workgroups per (batch, kv head); 0 = heuristic")
        .insert("pages_per_split", "0", "pages per split; 0 = heuristic")
        .insert("layout", "auto", "auto, coop or wave")
        .insert("v", "1", "validate against the CPU reference")
        .insert("seed", "1", "random seed")
        .insert("warmup", "10", "warmup launches")
        .insert("repeat", "100", "timed launches");
    if(!arg_parser.parse(argc, argv))
        return -1;

    fmha_decode_paged5d::Case c;
    c.batch   = arg_parser.get_int("b");
    c.nhead   = arg_parser.get_int("h");
    c.nhead_k = arg_parser.get_int("h_k");
    c.seqlen_k.clear();
    for(auto x : arg_parser.get_int_vec("s_k"))
        c.seqlen_k.push_back(x);
    c.window            = arg_parser.get_int("window");
    c.splits            = arg_parser.get_int("splits");
    c.pages_per_split   = arg_parser.get_int("pages_per_split");
    c.layout            = arg_parser.get_str("layout");
    c.seed              = arg_parser.get_int("seed");
    const bool validate = arg_parser.get_bool("v");

    const auto r = fmha_decode_paged5d::run(
        c, validate, arg_parser.get_int("warmup"), arg_parser.get_int("repeat"));
    if(!r.error.empty())
    {
        std::cerr << r.error << std::endl;
        return -1;
    }
    if(validate)
        std::printf(
            "max abs err %.4g (max |ref| %.4g), cos %.6f, ", r.max_err, r.max_ref, r.cosine);
    std::printf("b:%d h:%d/%d s_k:%d window:%d layout:%s groups:%d splits:%d pages/split:%d, "
                "stream %.2f us, graph %.2f us%s\n",
                c.batch,
                c.nhead,
                c.nhead_k,
                c.seqlen_k.front(),
                c.window,
                r.config.wave_split ? "wave" : "coop",
                r.config.num_groups,
                r.config.num_splits,
                r.config.pages_per_split,
                r.us_stream,
                r.us_graph,
                validate ? (r.valid ? ", valid:y" : ", valid:n") : "");
    return !validate || r.valid ? 0 : 1;
}
