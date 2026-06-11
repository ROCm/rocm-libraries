// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end: hipDNN Graph API (SDPA) -> ck-dsl-provider attention engine ->
// ck_dsl unified-attention kernel on GPU -> numerical verify vs a C++ causal
// paged-attention reference. B=1 BSHD (dense K/V are byte-compatible with the
// kernel's paged cache + contiguous block_tables, which the plan synthesizes).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <hipdnn_frontend.hpp>
#include <random>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using fp16 = __half;

#define HIP_CHECK(e)                                                               \
    do {                                                                           \
        hipError_t _e = (e);                                                       \
        if (_e != hipSuccess) {                                                    \
            std::fprintf(stderr, "HIP %s @%d\n", hipGetErrorString(_e), __LINE__); \
            return 2;                                                              \
        }                                                                          \
    } while (0)
#define FE_CHECK(e)                                                                  \
    do {                                                                             \
        auto _e = (e);                                                               \
        if (_e.get_code() != hipdnn_frontend::ErrorCode::OK) {                       \
            std::fprintf(stderr, "FE %s @%d\n", _e.get_message().c_str(), __LINE__); \
            return 3;                                                                \
        }                                                                            \
    } while (0)

int main(int argc, char** argv) {
    const int B = 1, S = (argc > 1 ? atoi(argv[1]) : 64), NQH = 16, NKVH = 2, HD = 128;
    const int gqa = NQH / NKVH;
    const float scale = 1.0f / std::sqrt((float)HD);

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("=== ck-dsl-provider SDPA end-to-end (hipDNN graph) ===\n");
    std::printf("device=%s  B=%d S=%d NQH=%d NKVH=%d HD=%d\n", prop.gcnArchName, B, S, NQH, NKVH,
                HD);

    if (const char* pdir = std::getenv("HIPDNN_PLUGIN_PATH")) {
        const char* paths[] = {pdir};
        hipdnnSetEnginePluginPaths_ext(1, paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    }
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    Graph graph;
    graph.set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_name("ckdsl_sdpa");
    graph.set_preferred_engine_id_ext(std::string("CK_DSL_ATTENTION_ENGINE"));

    // BSHD: dims [B,S,H,D], contiguous.
    auto Q = Graph::tensor(TensorAttributes()
                               .set_dim({B, S, NQH, HD})
                               .set_stride({S * NQH * HD, NQH * HD, HD, 1})
                               .set_uid(1));
    auto K = Graph::tensor(TensorAttributes()
                               .set_dim({B, S, NKVH, HD})
                               .set_stride({S * NKVH * HD, NKVH * HD, HD, 1})
                               .set_uid(2));
    auto V = Graph::tensor(TensorAttributes()
                               .set_dim({B, S, NKVH, HD})
                               .set_stride({S * NKVH * HD, NKVH * HD, HD, 1})
                               .set_uid(3));

    SdpaAttributes attrs;
    attrs.attn_scale_value = scale;
    attrs.causal_mask = true;
    auto [O, stats] = graph.sdpa(Q, K, V, std::move(attrs));
    O->set_output(true)
        .set_dim({B, S, NQH, HD})
        .set_stride({S * NQH * HD, NQH * HD, HD, 1})
        .set_uid(4);

    FE_CHECK(graph.build(handle));
    std::printf("graph.build() ok -- attention engine selected + AOT-compiled a ck_dsl kernel\n");
    int64_t ws = 0;
    FE_CHECK(graph.get_workspace_size(ws));

    // Host data (BSHD).
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<fp16> Qh((size_t)S * NQH * HD), Kh((size_t)S * NKVH * HD),
        Vh((size_t)S * NKVH * HD), Oh((size_t)S * NQH * HD, fp16(0));
    for (auto& x : Qh) x = (fp16)(nd(rng) * 0.5f);
    for (auto& x : Kh) x = (fp16)(nd(rng) * 0.5f);
    for (auto& x : Vh) x = (fp16)(nd(rng) * 0.5f);

    void *dQ, *dK, *dV, *dO, *dWs = nullptr;
    HIP_CHECK(hipMalloc(&dQ, Qh.size() * 2));
    HIP_CHECK(hipMalloc(&dK, Kh.size() * 2));
    HIP_CHECK(hipMalloc(&dV, Vh.size() * 2));
    HIP_CHECK(hipMalloc(&dO, Oh.size() * 2));
    if (ws > 0) HIP_CHECK(hipMalloc(&dWs, ws));
    HIP_CHECK(hipMemcpy(dQ, Qh.data(), Qh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, Kh.data(), Kh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, Vh.data(), Vh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dO, 0, Oh.size() * 2));

    std::unordered_map<std::shared_ptr<TensorAttributes>, void*> vp = {
        {Q, dQ}, {K, dK}, {V, dV}, {O, dO}};
    FE_CHECK(graph.execute(handle, vp, dWs));
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("graph.execute() ok -- ck_dsl attention kernel launched\n");
    HIP_CHECK(hipMemcpy(Oh.data(), dO, Oh.size() * 2, hipMemcpyDeviceToHost));

    // C++ causal reference (BSHD, single seq, GQA).
    auto qv = [&](int s, int h, int d) { return (float)Qh[((size_t)s * NQH + h) * HD + d]; };
    auto kv = [&](int s, int hk, int d) { return (float)Kh[((size_t)s * NKVH + hk) * HD + d]; };
    auto vv = [&](int s, int hk, int d) { return (float)Vh[((size_t)s * NKVH + hk) * HD + d]; };
    double worst = 0;
    int bad = 0;
    std::vector<float> sc(S);
    for (int qi = 0; qi < S; ++qi)
        for (int h = 0; h < NQH; ++h) {
            int hk = h / gqa, lim = qi + 1;
            float mx = -1e30f;
            for (int ki = 0; ki < lim; ++ki) {
                float s = 0;
                for (int d = 0; d < HD; ++d) s += qv(qi, h, d) * kv(ki, hk, d);
                s *= scale;
                sc[ki] = s;
                mx = std::max(mx, s);
            }
            float den = 0;
            for (int ki = 0; ki < lim; ++ki) {
                sc[ki] = std::exp(sc[ki] - mx);
                den += sc[ki];
            }
            for (int d = 0; d < HD; ++d) {
                float acc = 0;
                for (int ki = 0; ki < lim; ++ki) acc += sc[ki] * vv(ki, hk, d);
                acc /= den;
                float diff = std::fabs((float)Oh[((size_t)qi * NQH + h) * HD + d] - acc);
                worst = std::max(worst, (double)diff);
                if (diff > 0.05f) ++bad;
            }
        }
    std::printf("verify: max_abs_diff=%g bad=%d/%zu  %s\n", worst, bad, Oh.size(),
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
