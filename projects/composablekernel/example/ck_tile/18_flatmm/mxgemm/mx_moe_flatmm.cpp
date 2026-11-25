// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// tile_example_mx_moe_flatmm: one MX (a4w4 or a8w4) MoE GEMM on gfx1250, checked against a CPU
// reference. See mx_moe_flatmm.hpp for the problem layout.

#include "mx_moe_flatmm.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("experts", "8", "number of experts")
        .insert("NumTokens", "512", "number of tokens")
        .insert("TopK", "2", "experts per token")
        .insert("N",
                "1024",
                "N of the GEMM, a multiple of 256 (gemm1_gate_up: 2 x intermediate; gemm2: hidden)")
        .insert("K", "1024", "K of the GEMM (gemm1_gate_up: hidden; gemm2: intermediate)")
        .insert(
            "gemm_kind",
            "gemm1_gate_up",
            "gemm1_gate_up | gemm2 | gemm2_per_slot | layer (gate_up with MXFP4 output, then down)")
        .insert("combine", "per_slot", "layer only: per_slot | atomic")
        .insert("fused",
                "0",
                "layer only: > 0 runs the layer in one launch with this many intermediate slices")
        .insert(
            "persistent", "0", "layer only: 1 runs the layer as one persistent task-queue launch")
        .insert("hidden", "1024", "layer only: hidden size")
        .insert("inter", "512", "layer only: intermediate size")
        .insert("mx_prec", "fp4xfp4", "A x B data types: fp4xfp4 (a4w4) or fp8xfp4 (a8w4)")
        .insert("init",
                "0",
                "0: random, 1: constant (all 1), 2: row r of A = v(r), rest 1 (row-mapping check)")
        .insert("validate", "1", "0: no validation, 1: validate against a CPU reference")
        .insert("warmup", "0", "warm-up launches before timing (0 with repeat 0: no timing)")
        .insert("repeat", "0", "timed launches")
        .insert("seed", "11939", "seed of the routing and data");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

int main(int argc, char* argv[])
{
    auto [ok, arg_parser] = create_args(argc, argv);
    if(!ok)
        return EXIT_FAILURE;
    try
    {
        const std::string prec = arg_parser.get_str("mx_prec");
        const std::string kind = arg_parser.get_str("gemm_kind");
        if(kind == "layer")
        {
            if(prec != "fp4xfp4")
                throw std::runtime_error("layer: only mx_prec=fp4xfp4");
            MxMoeLayerCase l;
            l.experts      = arg_parser.get_int("experts");
            l.num_tokens   = arg_parser.get_int("NumTokens");
            l.topk         = arg_parser.get_int("TopK");
            l.hidden       = arg_parser.get_int("hidden");
            l.inter        = arg_parser.get_int("inter");
            l.per_slot     = arg_parser.get_str("combine") != "atomic";
            l.fused_slices = arg_parser.get_int("fused");
            l.persistent   = arg_parser.get_int("persistent") != 0;
            l.init         = arg_parser.get_int("init");
            l.seed         = arg_parser.get_uint32("seed");
            l.validate     = arg_parser.get_int("validate") != 0;
            l.warmup       = arg_parser.get_int("warmup");
            l.repeat       = arg_parser.get_int("repeat");
            const auto r   = run_mx_moe_layer(l);
            if(l.repeat > 0)
                std::cout << "time gate_up(+quant) " << r.ms_gate_up << " ms, down " << r.ms_down
                          << " ms" << std::endl;
            return r.pass ? 0 : 1;
        }
        MxMoeCase c;
        c.experts    = arg_parser.get_int("experts");
        c.num_tokens = arg_parser.get_int("NumTokens");
        c.topk       = arg_parser.get_int("TopK");
        c.N          = arg_parser.get_int("N");
        c.K          = arg_parser.get_int("K");
        c.init       = arg_parser.get_int("init");
        c.seed       = arg_parser.get_uint32("seed");
        c.validate   = arg_parser.get_int("validate") != 0;
        c.warmup     = arg_parser.get_int("warmup");
        c.repeat     = arg_parser.get_int("repeat");
        auto run     = [&](auto a_type) -> int {
            using A = decltype(a_type);
            using K = ck_tile::MoeFlatmmKind;
            if(kind == "gemm1_gate_up")
                return run_mx_moe_case<K::kFFN_gemm1_gate_up, A>(c).pass ? 0 : 1;
            if(kind == "gemm2")
                return run_mx_moe_case<K::kFFN_gemm2, A>(c).pass ? 0 : 1;
            if(kind == "gemm2_per_slot")
                return run_mx_moe_case<K::kFFN_gemm2_per_slot, A>(c).pass ? 0 : 1;
            throw std::runtime_error("gemm_kind must be gemm1_gate_up, gemm2 or gemm2_per_slot");
        };
        if(prec == "fp4xfp4")
            return run(ck_tile::pk_fp4_t{});
        if(prec == "fp8xfp4")
            return run(ck_tile::fp8_t{});
        throw std::runtime_error("mx_prec must be fp4xfp4 or fp8xfp4");
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
