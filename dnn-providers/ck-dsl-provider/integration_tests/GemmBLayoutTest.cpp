// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only unit test for the GEMM B-operand layout detector
// (CkDslParamParser::detectBLayout / isSupportedBLayout). No GPU, no hipDNN
// frontend, no flatbuffers -- it links only the parser's pure functions, so it
// runs on any host and gates the layout policy:
//
//   RCR  (B physically [N,K], K axis contiguous, strides {1,K})  -> supported
//   NN   (row-major [K,N],     N axis contiguous, strides {N,1})  -> rejected
//   junk / degenerate                                             -> Unknown
//
// usage: ck_dsl_gemm_blayout_test   (exit 0 = all pass)
#include <cstdio>

#include "engines/CkDslParamParser.hpp"

using ck_dsl_plugin::CkDslParamParser::BLayout;
using ck_dsl_plugin::CkDslParamParser::bLayoutName;
using ck_dsl_plugin::CkDslParamParser::detectBLayout;
using ck_dsl_plugin::CkDslParamParser::isSupportedBLayout;

namespace {
int g_fail = 0;

void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_fail;
}

// detectBLayout(K, N, stride_outer (K axis), stride_inner (N axis))
void run() {
    const long K = 256, N = 512;

    // RCR: B stored [N,K]; K axis contiguous => stride_outer=1, stride_inner=K.
    {
        auto l = detectBLayout(K, N, /*outer=*/1, /*inner=*/K);
        check(l == BLayout::RCR_NK, "RCR strides {1,K} -> RCR_NK");
        check(isSupportedBLayout(l), "RCR_NK is supported");
    }
    // Row-major [K,N]; N axis contiguous => stride_outer=N, stride_inner=1.
    {
        auto l = detectBLayout(K, N, /*outer=*/N, /*inner=*/1);
        check(l == BLayout::RowMajor_KN, "row-major strides {N,1} -> RowMajor_KN");
        check(!isSupportedBLayout(l), "RowMajor_KN is rejected (not supported)");
    }
    // Garbage / non-canonical strides -> Unknown (rejected).
    {
        auto l = detectBLayout(K, N, /*outer=*/7, /*inner=*/3);
        check(l == BLayout::Unknown, "non-canonical strides -> Unknown");
        check(!isSupportedBLayout(l), "Unknown is rejected");
    }
    // Zero / negative strides -> Unknown.
    {
        check(detectBLayout(K, N, 0, K) == BLayout::Unknown, "zero outer stride -> Unknown");
        check(detectBLayout(K, N, 1, 0) == BLayout::Unknown, "zero inner stride -> Unknown");
        check(detectBLayout(K, N, -1, K) == BLayout::Unknown, "negative stride -> Unknown");
    }
    // Degenerate K==1 / N==1: both candidate strides collapse to 1 -> ambiguous.
    {
        check(detectBLayout(/*K=*/1, N, 1, 1) == BLayout::Unknown, "K==1 ambiguous -> Unknown");
        check(detectBLayout(K, /*N=*/1, 1, 1) == BLayout::Unknown, "N==1 ambiguous -> Unknown");
    }
    // A non-square shape exercises that the test isn't keying off K==N coincidence.
    {
        const long K2 = 384, N2 = 128;
        check(detectBLayout(K2, N2, 1, K2) == BLayout::RCR_NK, "non-square RCR {1,K}");
        check(detectBLayout(K2, N2, N2, 1) == BLayout::RowMajor_KN, "non-square row-major {N,1}");
    }
    // Name mapping sanity.
    {
        check(std::string(bLayoutName(BLayout::RCR_NK)) == "RCR", "bLayoutName(RCR_NK)==RCR");
        check(std::string(bLayoutName(BLayout::RowMajor_KN)) == "RRR",
              "bLayoutName(RowMajor_KN)==RRR");
        check(std::string(bLayoutName(BLayout::Unknown)) == "unknown",
              "bLayoutName(Unknown)==unknown");
    }
}
}  // namespace

int main() {
    std::printf("=== ck-dsl-provider GEMM B-layout detector unit test ===\n");
    run();
    std::printf(g_fail == 0 ? "ALL PASS\n" : "FAILURES (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
