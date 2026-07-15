// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace {
// Runs one shape across all reference-supported activations (gelu/silu/gelu_tanh).
void run_case(
    ck::index_t N, ck::index_t K, ck::index_t tokens, ck::index_t experts, ck::index_t topk)
{
    moe_gemm_test::run_moe_gemm1<ADataType, BDataType, EDataType, AccDataType>(
        N, K, tokens, experts, topk);
}
} // namespace

TEST(TestMoeGemm1Xdl, SmallShapes)
{
    run_case(/*N=*/128, /*K=*/256, /*tokens=*/64, /*experts=*/8, /*topk=*/2);
    run_case(/*N=*/256, /*K=*/512, /*tokens=*/256, /*experts=*/8, /*topk=*/2);
}

TEST(TestMoeGemm1Xdl, Tokens)
{
    for(int tokens : {128, 512, 1024})
        run_case(/*N=*/256, /*K=*/512, tokens, /*experts=*/8, /*topk=*/2);
}

TEST(TestMoeGemm1Xdl, ExpertsAndTopk)
{
    for(int experts : {8, 16, 32})
        for(int topk : {1, 2})
            run_case(/*N=*/256, /*K=*/512, /*tokens=*/512, experts, topk);
}

TEST(TestMoeGemm1Xdl, LargerShape)
{
    run_case(/*N=*/512, /*K=*/1024, /*tokens=*/1024, /*experts=*/8, /*topk=*/2);
}
