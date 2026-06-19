// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only fuzz/robustness test for the three ck-dsl-provider param parsers
// (CkDsl{,Attn,Conv}ParamParser). It builds randomized + edge-case hipDNN graph
// flatbuffers and feeds them through the parsers, asserting that:
//
//   * a well-formed graph parses to the expected logical dims, and
//   * a malformed / degenerate / unsupported graph either THROWS a clean
//     std::exception OR returns a value -- but NEVER triggers an out-of-bounds
//     read, a crash, or undefined behaviour.
//
// This is the contract the engine's isApplicable() relies on: parse errors must
// surface as exceptions it can catch and decline, not as memory unsafety. No
// GPU, no comgr, no hipDNN frontend -- only the flatbuffers data SDK + the
// parser pure functions. (Compile with -fsanitize=address to make any OOB fatal.)
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cstdint>
#include <cstdio>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <random>
#include <vector>

#include "engines/CkDslAttnParamParser.hpp"
#include "engines/CkDslConvParamParser.hpp"
#include "engines/CkDslParamParser.hpp"

namespace gemmp = ck_dsl_plugin::CkDslParamParser;
namespace attnp = ck_dsl_plugin::CkDslAttnParamParser;
namespace convp = ck_dsl_plugin::CkDslConvParamParser;
using namespace hipdnn_flatbuffers_sdk::data_objects;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace {
int g_fail = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("  [FAIL] %s\n", what);
        ++g_fail;
    }
}

// Run a parser callable and assert it does not escape via a non-std exception or
// a crash. Either a clean parse or a std::exception is acceptable.
template <typename Fn>
bool parse_is_safe(Fn&& f) {
    try {
        f();
        return true;  // parsed without throwing
    } catch (const std::exception&) {
        return true;  // declined cleanly -- the engine catches this
    } catch (...) {
        return false;  // non-std exception escaped -- a bug
    }
}

// ---- builders ------------------------------------------------------------

// Build an SDPA graph. `qDims`/`qStrides` etc. may be empty (omitted vectors) or
// any rank to exercise the rank/stride guards.
std::vector<uint8_t> buildSdpa(const std::vector<int64_t>& qDims,
                               const std::vector<int64_t>& qStrides,
                               const std::vector<int64_t>& kvDims,
                               const std::vector<int64_t>& kvStrides, DataType dt, bool causal) {
    flatbuffers::FlatBufferBuilder b;
    auto mk = [&](int64_t uid, const char* nm, const std::vector<int64_t>& d,
                  const std::vector<int64_t>& s) {
        const std::vector<int64_t>* dp = d.empty() ? nullptr : &d;
        const std::vector<int64_t>* sp = s.empty() ? nullptr : &s;
        return CreateTensorAttributesDirect(b, uid, nm, dt, sp, dp);
    };
    auto q = mk(1, "q", qDims, qStrides);
    auto k = mk(2, "k", kvDims, kvStrides);
    auto v = mk(3, "v", kvDims, kvStrides);
    auto o = mk(4, "o", qDims, qStrides);
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors{q, k, v, o};
    SdpaAttributesBuilder sb(b);
    sb.add_q_tensor_uid(1);
    sb.add_k_tensor_uid(2);
    sb.add_v_tensor_uid(3);
    sb.add_o_tensor_uid(4);
    if (causal) sb.add_causal_mask(true);
    auto sdpa = sb.Finish();
    auto node =
        CreateNodeDirect(b, "sdpa", DataType::FLOAT, NodeAttributes::SdpaAttributes, sdpa.Union());
    std::vector<flatbuffers::Offset<Node>> nodes{node};
    auto g = CreateGraphDirect(b, "g", DataType::FLOAT, dt, dt, &tensors, &nodes);
    b.Finish(g);
    return std::vector<uint8_t>(b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize());
}

std::vector<uint8_t> buildGemm(const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims,
                               const std::vector<int64_t>& bStrides,
                               const std::vector<int64_t>& cDims, DataType dt) {
    flatbuffers::FlatBufferBuilder b;
    auto mk = [&](int64_t uid, const char* nm, const std::vector<int64_t>& d,
                  const std::vector<int64_t>& s) {
        const std::vector<int64_t>* dp = d.empty() ? nullptr : &d;
        const std::vector<int64_t>* sp = s.empty() ? nullptr : &s;
        return CreateTensorAttributesDirect(b, uid, nm, dt, sp, dp);
    };
    auto a = mk(1, "a", aDims, {});
    auto bt = mk(2, "b", bDims, bStrides);
    auto c = mk(3, "c", cDims, {});
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors{a, bt, c};
    MatmulAttributesBuilder mb(b);
    mb.add_a_tensor_uid(1);
    mb.add_b_tensor_uid(2);
    mb.add_c_tensor_uid(3);
    auto mm = mb.Finish();
    auto node =
        CreateNodeDirect(b, "mm", DataType::FLOAT, NodeAttributes::MatmulAttributes, mm.Union());
    std::vector<flatbuffers::Offset<Node>> nodes{node};
    auto g = CreateGraphDirect(b, "g", DataType::FLOAT, dt, dt, &tensors, &nodes);
    b.Finish(g);
    return std::vector<uint8_t>(b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize());
}

std::vector<uint8_t> buildConv(const std::vector<int64_t>& xDims, const std::vector<int64_t>& wDims,
                               const std::vector<int64_t>& stride, const std::vector<int64_t>& pad,
                               const std::vector<int64_t>& dil, DataType dt) {
    flatbuffers::FlatBufferBuilder b;
    auto mk = [&](int64_t uid, const char* nm, const std::vector<int64_t>& d) {
        const std::vector<int64_t>* dp = d.empty() ? nullptr : &d;
        return CreateTensorAttributesDirect(b, uid, nm, dt, nullptr, dp);
    };
    auto x = mk(1, "x", xDims);
    auto w = mk(2, "w", wDims);
    auto y = mk(3, "y", {});
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors{x, w, y};
    auto sp = stride.empty() ? 0 : b.CreateVector(stride);
    auto pp = pad.empty() ? 0 : b.CreateVector(pad);
    auto dp = dil.empty() ? 0 : b.CreateVector(dil);
    ConvolutionFwdAttributesBuilder cb(b);
    cb.add_x_tensor_uid(1);
    cb.add_w_tensor_uid(2);
    cb.add_y_tensor_uid(3);
    if (sp.o) cb.add_stride(sp);
    if (pp.o) cb.add_pre_padding(pp);
    if (dp.o) cb.add_dilation(dp);
    auto conv = cb.Finish();
    auto node = CreateNodeDirect(b, "conv", DataType::FLOAT,
                                 NodeAttributes::ConvolutionFwdAttributes, conv.Union());
    std::vector<flatbuffers::Offset<Node>> nodes{node};
    auto g = CreateGraphDirect(b, "g", DataType::FLOAT, dt, dt, &tensors, &nodes);
    b.Finish(g);
    return std::vector<uint8_t>(b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize());
}

// ---- happy-path sanity (so the fuzzer's negative cases mean something) ----

void test_happy_paths() {
    {  // SDPA B=4 GQA BSHD
        auto buf = buildSdpa({4, 16, 64, 128}, {16 * 64 * 128, 128, 16 * 128, 1}, {4, 2, 64, 128},
                             {2 * 64 * 128, 128, 2 * 128, 1}, DataType::HALF, true);
        fbu::GraphWrapper w(buf.data(), buf.size());
        check(w.isValid() && attnp::isSdpaGraph(w), "sdpa happy: valid + recognized");
        auto p = attnp::parseSdpaGraph(w);
        check(p.batch == 4 && p.nhead_q == 16 && p.nhead_k == 2 && p.seqlen_q == 64 &&
                  p.hdim_q == 128,
              "sdpa happy: dims [B,H,S,D]");
        check(!p.is_bhsd, "sdpa happy: BSHD physical not flagged BHSD");
        auto prob = attnp::buildProblem(p, "gfx950");
        check(prob.total_q == 4 * 64 && prob.num_seqs == 4, "sdpa B>1: total_q/num_seqs populated");
    }
    {  // GEMM RCR
        const int64_t M = 256, K = 128, N = 512;
        auto buf = buildGemm({M, K}, {K, N}, {1, K}, {M, N}, DataType::HALF);
        fbu::GraphWrapper w(buf.data(), buf.size());
        check(w.isValid() && gemmp::isGemmGraph(w), "gemm happy: valid + recognized");
        auto p = gemmp::parseGemmGraph(w);
        check(p.M == M && p.K == K && p.N == N, "gemm happy: M/K/N");
        check(p.b_layout == gemmp::BLayout::RCR_NK, "gemm happy: RCR detected");
    }
    {  // Conv 3x3 grouped
        auto buf =
            buildConv({8, 64, 56, 56}, {128, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, DataType::HALF);
        fbu::GraphWrapper w(buf.data(), buf.size());
        check(w.isValid() && convp::isConvGraph(w), "conv happy: valid + recognized");
        auto p = convp::parseConvGraph(w);
        check(p.N == 8 && p.C == 64 && p.K == 128 && p.G == 2 && p.R == 3 && p.S == 3,
              "conv happy: N/C/K/G/R/S (G derived from C/(C/G))");
    }
}

// ---- negative / edge cases (must not crash) ------------------------------

void test_edge_cases() {
    auto safe_sdpa = [](const std::vector<uint8_t>& buf) {
        return parse_is_safe([&] {
            fbu::GraphWrapper w(buf.data(), buf.size());
            if (attnp::isSdpaGraph(w)) (void)attnp::parseSdpaGraph(w);
        });
    };
    auto safe_gemm = [](const std::vector<uint8_t>& buf) {
        return parse_is_safe([&] {
            fbu::GraphWrapper w(buf.data(), buf.size());
            if (gemmp::isGemmGraph(w)) (void)gemmp::parseGemmGraph(w);
        });
    };
    auto safe_conv = [](const std::vector<uint8_t>& buf) {
        return parse_is_safe([&] {
            fbu::GraphWrapper w(buf.data(), buf.size());
            if (convp::isConvGraph(w)) (void)convp::parseConvGraph(w);
        });
    };

    // SDPA edge cases.
    check(safe_sdpa(buildSdpa({}, {}, {}, {}, DataType::HALF, false)), "sdpa: omitted dims");
    check(safe_sdpa(buildSdpa({2, 8}, {}, {2, 2}, {}, DataType::HALF, false)),
          "sdpa: rank-2 (< 4) dims");
    check(safe_sdpa(buildSdpa({2, 8, 64, 128}, {1, 2}, {2, 2, 64, 128}, {}, DataType::HALF, false)),
          "sdpa: short stride vector");
    check(safe_sdpa(
              buildSdpa({2, 8, 64, 128}, {0, 0, 0, 0}, {2, 2, 64, 128}, {}, DataType::HALF, false)),
          "sdpa: zero strides");
    check(safe_sdpa(buildSdpa({2, 8, 64, 128}, {}, {2, 2, 64, 128}, {}, DataType::FLOAT, false)),
          "sdpa: unsupported dtype (FLOAT)");

    // GEMM edge cases.
    check(safe_gemm(buildGemm({}, {}, {}, {}, DataType::HALF)), "gemm: omitted dims");
    check(safe_gemm(buildGemm({256}, {128, 512}, {1, 128}, {256, 512}, DataType::HALF)),
          "gemm: rank-1 A");
    check(safe_gemm(buildGemm({256, 128}, {999, 512}, {1, 999}, {256, 512}, DataType::HALF)),
          "gemm: B-K mismatches A-K");
    check(safe_gemm(buildGemm({256, 128}, {128, 512}, {0, 0}, {256, 512}, DataType::HALF)),
          "gemm: zero strides -> Unknown layout");
    check(safe_gemm(buildGemm({256, 128}, {128, 512}, {}, {256, 512}, DataType::HALF)),
          "gemm: no strides -> RCR fallback");
    check(safe_gemm(buildGemm({256, 128}, {128, 512}, {1, 128}, {256, 512}, DataType::BFLOAT16)),
          "gemm: bf16 dtype");

    // Conv edge cases.
    check(safe_conv(buildConv({}, {}, {}, {}, {}, DataType::HALF)), "conv: omitted dims");
    check(safe_conv(
              buildConv({8, 64, 56, 56}, {128, 0, 3, 3}, {1, 1}, {1, 1}, {1, 1}, DataType::HALF)),
          "conv: zero cpg (bad weight dim[1])");
    check(safe_conv(
              buildConv({8, 64, 56, 56}, {128, 7, 3, 3}, {1, 1}, {1, 1}, {1, 1}, DataType::HALF)),
          "conv: cpg not a divisor of C");
    check(safe_conv(
              buildConv({8, 64, 56, 56}, {130, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, DataType::HALF)),
          "conv: K not divisible by derived G");
    check(safe_conv(buildConv({8, 64, 56, 56}, {128, 64, 3, 3}, {}, {}, {}, DataType::HALF)),
          "conv: omitted stride/pad/dilation -> defaults");
}

// ---- randomized stress ---------------------------------------------------

void test_random_stress() {
    std::mt19937 rng(0xC0FFEE);
    auto rd = [&](int64_t lo, int64_t hi) {
        return std::uniform_int_distribution<int64_t>(lo, hi)(rng);
    };
    int safe = 0;
    const int iters = 4000;
    for (int i = 0; i < iters; ++i) {
        // random rank 0..5, random extents (incl 0 and large), random strides,
        // random dtype. Many of these are nonsense graphs by design.
        int kind = (int)rd(0, 2);
        DataType dt = (DataType)(int)rd(0, 30);  // mostly unsupported -> must decline
        auto vec = [&](int n) {
            std::vector<int64_t> v;
            for (int j = 0; j < n; ++j) v.push_back(rd(0, 4096));
            return v;
        };
        bool ok = true;
        if (kind == 0) {
            int rqd = (int)rd(0, 5), rqs = (int)rd(0, 5), rkd = (int)rd(0, 5);
            auto buf = buildSdpa(vec(rqd), vec(rqs), vec(rkd), vec((int)rd(0, 5)), dt, rd(0, 1));
            ok = parse_is_safe([&] {
                fbu::GraphWrapper w(buf.data(), buf.size());
                if (attnp::isSdpaGraph(w)) {
                    auto p = attnp::parseSdpaGraph(w);
                    (void)attnp::buildProblem(p, "gfx950");
                }
            });
        } else if (kind == 1) {
            auto buf = buildGemm(vec((int)rd(0, 4)), vec((int)rd(0, 4)), vec((int)rd(0, 4)),
                                 vec((int)rd(0, 4)), dt);
            ok = parse_is_safe([&] {
                fbu::GraphWrapper w(buf.data(), buf.size());
                if (gemmp::isGemmGraph(w)) {
                    auto p = gemmp::parseGemmGraph(w);
                    (void)gemmp::buildProblem(p, "gfx950");
                }
            });
        } else {
            auto buf = buildConv(vec((int)rd(0, 5)), vec((int)rd(0, 5)), vec((int)rd(0, 3)),
                                 vec((int)rd(0, 3)), vec((int)rd(0, 3)), dt);
            ok = parse_is_safe([&] {
                fbu::GraphWrapper w(buf.data(), buf.size());
                if (convp::isConvGraph(w)) {
                    auto p = convp::parseConvGraph(w);
                    (void)convp::buildProblem(p, "gfx950");
                }
            });
        }
        if (ok)
            ++safe;
        else {
            std::printf("  [FAIL] random iter %d (kind=%d) escaped non-std exception\n", i, kind);
            ++g_fail;
        }
    }
    std::printf("  random stress: %d/%d iterations parsed safely (no crash/OOB)\n", safe, iters);
}

}  // namespace

int main() {
    std::printf("=== ck-dsl-provider param parser fuzz/robustness test ===\n");
    test_happy_paths();
    test_edge_cases();
    test_random_stress();
    std::printf(g_fail == 0 ? "ALL PASS\n" : "FAILURES (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
