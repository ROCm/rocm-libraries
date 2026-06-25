// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end DECODE (seqlen_q == 1, long KV): hipDNN Graph API (SDPA) ->
// ck-dsl-provider attention engine -> ck_dsl split-KV 3D segment+reduce kernels
// on GPU -> numerical verify vs a C++ paged-attention reference. GQA kv=2 (the
// known-correct decode path). The provider routes this shape to the optimal 3D
// split-KV pipeline (two launches + per-segment workspace, graph-captured),
// not the single suboptimal 2D kernel.
//
// Usage: EndToEndSdpaDecodeDemo [Sk] [iters]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    const int B = 1, SQ = 1, SK = (argc > 1 ? atoi(argv[1]) : 2048);
    const int NQH = 16, NKVH = 2, HD = 128;  // GQA kv=2 (known-correct decode path)
    const int iters = (argc > 2 ? atoi(argv[2]) : 200);
    const int gqa = NQH / NKVH;
    const float scale = 1.0f / std::sqrt((float)HD);

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("=== ck-dsl-provider SDPA DECODE end-to-end (split-KV 3D) ===\n");
    std::printf("device=%s  B=%d Sq=%d Sk=%d NQH=%d NKVH=%d HD=%d  iters=%d\n", prop.gcnArchName, B,
                SQ, SK, NQH, NKVH, HD, iters);

    if (const char* pdir = std::getenv("HIPDNN_PLUGIN_PATH")) {
        const char* paths[] = {pdir};
        hipdnnSetEnginePluginPaths_ext(1, paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    }
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    Graph graph;
    graph.set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_name("ckdsl_sdpa_decode");
    graph.set_preferred_engine_id_ext(std::string("CK_DSL_ATTENTION_ENGINE"));

    // Logical BHSD dims with physical BSHD strides (S stored outside H), so the
    // plugin selects the BSHD-native paged path. Q has S=1 (decode), K/V have
    // S=SK (the KV history).
    auto Q = Graph::tensor(TensorAttributes()
                               .set_dim({B, NQH, SQ, HD})
                               .set_stride({SQ * NQH * HD, HD, NQH * HD, 1})
                               .set_uid(1));
    auto K = Graph::tensor(TensorAttributes()
                               .set_dim({B, NKVH, SK, HD})
                               .set_stride({SK * NKVH * HD, HD, NKVH * HD, 1})
                               .set_uid(2));
    auto V = Graph::tensor(TensorAttributes()
                               .set_dim({B, NKVH, SK, HD})
                               .set_stride({SK * NKVH * HD, HD, NKVH * HD, 1})
                               .set_uid(3));

    SdpaAttributes attrs;
    attrs.attn_scale_value = scale;
    attrs.causal_mask = false;  // decode: the single query attends to all KV
    auto [O, stats] = graph.sdpa(Q, K, V, std::move(attrs));
    O->set_output(true)
        .set_dim({B, NQH, SQ, HD})
        .set_stride({SQ * NQH * HD, HD, NQH * HD, 1})
        .set_uid(4);

    FE_CHECK(graph.build(handle));
    std::printf("graph.build() ok -- attention engine selected + AOT-compiled ck_dsl kernel(s)\n");
    int64_t ws = 0;
    FE_CHECK(graph.get_workspace_size(ws));
    std::printf("workspace size = %lld bytes (%s)\n", (long long)ws,
                ws > 0 ? "split-KV 3D segment workspace -> 3D path active" : "0 -> 2D path");

    // Host data (BSHD).
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<fp16> Qh((size_t)SQ * NQH * HD), Kh((size_t)SK * NKVH * HD),
        Vh((size_t)SK * NKVH * HD), Oh((size_t)SQ * NQH * HD, fp16(0.f));
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
    std::printf("graph.execute() ok -- ck_dsl decode kernel(s) launched\n");
    HIP_CHECK(hipMemcpy(Oh.data(), dO, Oh.size() * 2, hipMemcpyDeviceToHost));

    // C++ decode reference (BSHD, single query attends to all SK KV; GQA).
    auto qv = [&](int h, int d) { return (float)Qh[(size_t)h * HD + d]; };
    auto kv = [&](int s, int hk, int d) { return (float)Kh[((size_t)s * NKVH + hk) * HD + d]; };
    auto vv = [&](int s, int hk, int d) { return (float)Vh[((size_t)s * NKVH + hk) * HD + d]; };
    double worst = 0;
    int bad = 0;
    std::vector<float> sc(SK);
    for (int h = 0; h < NQH; ++h) {
        int hk = h / gqa;
        float mx = -1e30f;
        for (int ki = 0; ki < SK; ++ki) {
            float s = 0;
            for (int d = 0; d < HD; ++d) s += qv(h, d) * kv(ki, hk, d);
            s *= scale;
            sc[ki] = s;
            mx = std::max(mx, s);
        }
        float den = 0;
        for (int ki = 0; ki < SK; ++ki) {
            sc[ki] = std::exp(sc[ki] - mx);
            den += sc[ki];
        }
        for (int d = 0; d < HD; ++d) {
            float acc = 0;
            for (int ki = 0; ki < SK; ++ki) acc += sc[ki] * vv(ki, hk, d);
            acc /= den;
            float diff = std::fabs((float)Oh[(size_t)h * HD + d] - acc);
            worst = std::max(worst, (double)diff);
            if (diff > 0.05f) ++bad;
        }
    }
    std::printf("verify: max_abs_diff=%g bad=%d/%zu  %s\n", worst, bad, Oh.size(),
                bad == 0 ? "PASS" : "FAIL");

    // Decode latency through the provider (per graph.execute()). Warm up, then
    // time `iters` executions wall-clock (each execute() issues the graph
    // replay / two launches on the default stream and the loop syncs once).
    for (int i = 0; i < 20; ++i) FE_CHECK(graph.execute(handle, vp, dWs));
    HIP_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) FE_CHECK(graph.execute(handle, vp, dWs));
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::printf("decode latency: %.2f us / execute  (avg of %d)\n", us, iters);

    return bad == 0 ? 0 : 1;
}
