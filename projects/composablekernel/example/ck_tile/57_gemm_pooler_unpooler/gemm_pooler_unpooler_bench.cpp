// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Correctness check and timed benchmark of the two linear layers of a pooler / unpooler pair,
// built on the run-ahead TDM pipeline (GemmPipelineAgBgCrCompTDMRunAhead) for gfx1250.
//
// bf16 A and B, fp32 accumulate, bf16 C row-major:
//   -op pooler     E = A * B + bias, M x N x K = 2048 x 4096 x 8192. bias is an N-vector broadcast
//                  over M (a row-major D tensor with row stride 0), fused in TdmMultiDEpilogue.
//   -op unpooler   C = A * B, M x N x K = 2048 x 8192 x 4096, TdmEpilogue.
// -m, -n, -k and -bias override the op's defaults.
//
// The shapes are those of the hipBLASLt (column-major) calls D = A * B (+ bias over D's rows)
// with m x n x k = 4096 x 2048 x 8192 and 8192 x 2048 x 4096, transposed into row-major terms:
// this M is hipBLASLt's n and this N is its m. The hipBLASLt NN, NT and TN calls are the rrr,
// crr and rcr layouts here.
//
// -tile picks the block tile, the wave grid and the LDS ring <NumSlots, RunAhead>:
//   256x256x64_s3r2 | 256x256x128_s2r1      warps 2x2 (hipBLASLt MT256x256)
//   176x256x64_s4r3 | 176x256x128_s2r1      warps 1x4 (hipBLASLt MT256x176)
//   128x256x128_s3r2                        warps 2x2
//   192x256x64_s4r3 | 192x256x128_s2r1      warps 2x2 (the pooler's C is 176 of these)
// -epilogue tdmstaged (TdmEpilogueStagedForPipeline / TdmMultiDEpilogueStagedForPipeline, one M
// iteration per wave row at a time) | tdm (the whole C tile at once).
// -bias_load row (one N-vector per wave, TdmMultiDEpilogue's DsRowBroadcastMask) | tile (the bias
// as a full stride-0 D tile, one scalar load per element).
// -streamk linear | tree runs the plain GEMM on StreamKKernel (256x256 tiles, -epilogue=tdm), with
// -persistent 0 | 1 for its data-parallel section. Each timed launch includes zeroing the
// workspace's ready flags.
// -split_n N0 (plain GEMM) times two back-to-back kernels per launch: -tile on C's columns
// [0, N0) and 192x256x128_s2r1 on [N0, N). For the unpooler, -split_n 6144 with 256x256 tiles is
// 192 + 88 tiles, two full rounds and one of 88 on 96 WGPs, where 256x256 over all of C is 256
// tiles, 2.67 rounds.
//
// Modes (-mode=):
//   verify-int   integers in [-4, 4]: C must be the exact result rounded once to bf16.
//   verify-rand  normal(0, 1) in bf16, |got - ref| <= |ref| * 2^-7 + sum|a*b| * 2^-16.
//   bench        -rounds samples of -repeat launches after -warmup, rotating through copies of
//                A, B and C that total at least -rotating_mb. Prints the median time per launch.
// Exit code: 0 pass, 1 wrong result, 2 rejected or bad arguments, 3 error.

#include "gemm_pooler_unpooler_bench.hpp"

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("op", "unpooler", "pooler (GEMM + bias) | unpooler (GEMM)")
        .insert("m", "-1", "M (-1: the op's default)")
        .insert("n", "-1", "N (-1: the op's default)")
        .insert("k", "-1", "K (-1: the op's default)")
        .insert("bias", "-1", "fuse the N-vector bias: 1 | 0 (-1: the op's default)")
        .insert("layout", "rcr", "A and B layouts, C row-major: rcr | rrr | crr")
        .insert("tile", "256x256x64_s3r2", "block tile, warps and ring (see the file header)")
        .insert("epilogue",
                "tdmstaged",
                "tdm (whole C tile at once) | tdmstaged (one M slab at a time)")
        .insert("bias_load", "row", "row (one N-vector per wave) | tile (full stride-0 D tile)")
        .insert("streamk",
                "none",
                "none | linear | tree: Stream-K (unpooler, 256x256 tiles, -epilogue=tdm)")
        .insert("split_n",
                "0",
                "> 0: plain GEMM as two kernels, -tile on C's first split_n columns and "
                "192x256x128_s2r1 on the rest")
        .insert("persistent", "0", "Stream-K: persistent data-parallel section (1) or not (0)")
        .insert("mode", "verify-int", "verify-int | verify-rand | bench")
        .insert("seed", "12345", "RNG seed for the inputs")
        .insert("check_per_tile", "16", "verify: random points per 128x128 output tile")
        .insert("warmup", "20", "bench: untimed launches per sample")
        .insert("repeat", "100", "bench: timed launches per sample")
        .insert("rounds", "5", "bench: samples")
        .insert("rotating_mb", "512", "bench: rotate through at least this many MB of A/B/C");
    if(!arg.parse(argc, argv))
        return 2;

    const std::string op = arg.get_str("op");
    if(op != "pooler" && op != "unpooler")
    {
        std::printf("unknown -op=%s (pooler, unpooler)\n", op.c_str());
        return 2;
    }
    const bool pooler = op == "pooler";
    // -1 keeps the op's default.
    auto pick = [&](const char* key, int dflt) {
        const int v = arg.get_int(key);
        return v < 0 ? dflt : v;
    };
    const pooler_bench::Shape shape{pick("m", 2048),
                                    pick("n", pooler ? 4096 : 8192),
                                    pick("k", pooler ? 8192 : 4096),
                                    pick("bias", pooler ? 1 : 0) != 0,
                                    arg.get_int("split_n")};
    if(shape.M <= 0 || shape.N <= 0 || shape.K <= 0)
    {
        std::printf("M, N and K must be positive\n");
        return 2;
    }

    if(shape.split_n != 0 &&
       (shape.bias || arg.get_str("streamk") != "none" || shape.split_n < 0 ||
        shape.split_n >= shape.N || shape.split_n % 64 != 0))
    {
        std::printf("-split_n needs -bias=0, -streamk=none and 0 < split_n < N, a multiple of 64\n");
        return 2;
    }
    const std::string layout = arg.get_str("layout");
    const std::string epi    = arg.get_str("epilogue");
    const std::string name   = "ck_ra_" + op + "_bf16_" + layout + "_" + arg.get_str("tile") + "_" +
                             epi + (shape.bias ? "_bias_" + arg.get_str("bias_load") : "") +
                             (shape.split_n > 0 ? "_split" + std::to_string(shape.split_n) : "");
    if(arg.get_str("bias_load") != "row" && arg.get_str("bias_load") != "tile")
    {
        std::printf("unknown -bias_load=%s (row, tile)\n", arg.get_str("bias_load").c_str());
        return 2;
    }
    if(epi != "tdm" && epi != "tdmstaged")
    {
        std::printf("unknown -epilogue=%s (tdm, tdmstaged)\n", epi.c_str());
        return 2;
    }
    const bool staged = epi == "tdmstaged";
    try
    {
        if(layout == "rcr")
            return staged ? pooler_bench::dispatch_rcr_tdmstaged(arg, shape, name)
                          : pooler_bench::dispatch_rcr_tdm(arg, shape, name);
        if(layout == "rrr")
            return staged ? pooler_bench::dispatch_rrr_tdmstaged(arg, shape, name)
                          : pooler_bench::dispatch_rrr_tdm(arg, shape, name);
        if(layout == "crr")
            return staged ? pooler_bench::dispatch_crr_tdmstaged(arg, shape, name)
                          : pooler_bench::dispatch_crr_tdm(arg, shape, name);
        std::printf("unknown -layout=%s (rcr, rrr, crr)\n", layout.c_str());
        return 2;
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
