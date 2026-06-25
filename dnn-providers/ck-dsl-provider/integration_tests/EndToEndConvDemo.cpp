// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end: hipDNN Graph API (ConvolutionFwd) -> ck-dsl-provider conv engine
// -> ck_dsl implicit-GEMM conv kernel on GPU -> numerical verify vs a C++ NHWC
// x KRSC -> NHWK reference. Shape matches the baked kernel (conv geometry is
// compile-time in the implicit-GEMM descriptor).
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

int main(int, char**) {
    const int N = 8, Hi = 56, Wi = 56, C = 64, K = 64, R = 3, Sf = 3;
    const int sH = 1, sW = 1, pH = 1, pW = 1, dH = 1, dW = 1;
    const int Ho = (Hi + 2 * pH - dH * (R - 1) - 1) / sH + 1;
    const int Wo = (Wi + 2 * pW - dW * (Sf - 1) - 1) / sW + 1;

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("=== ck-dsl-provider Conv end-to-end (hipDNN graph) ===\n");
    std::printf("device=%s  N%d H%d W%d C%d K%d R%d S%d -> Ho%d Wo%d\n", prop.gcnArchName, N, Hi,
                Wi, C, K, R, Sf, Ho, Wo);

    if (const char* pdir = std::getenv("HIPDNN_PLUGIN_PATH")) {
        const char* paths[] = {pdir};
        hipdnnSetEnginePluginPaths_ext(1, paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    }
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    Graph graph;
    graph.set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_name("ckdsl_conv");
    graph.set_preferred_engine_id_ext(std::string("CK_DSL_CONV_ENGINE"));

    // cuDNN-style logical dims (N,C,H,W)/(K,C,R,S)/(N,K,Ho,Wo) with NHWC/KRSC/
    // NHWK *physical* strides (channels-last) -- the ck_dsl kernel reads NHWC/KRSC.
    auto X = Graph::tensor(TensorAttributes()
                               .set_dim({N, C, Hi, Wi})
                               .set_stride({Hi * Wi * C, 1, Wi * C, C})
                               .set_uid(1));
    auto W = Graph::tensor(TensorAttributes()
                               .set_dim({K, C, R, Sf})
                               .set_stride({R * Sf * C, 1, Sf * C, C})
                               .set_uid(2));
    auto Y = graph.conv_fprop(
        X, W,
        ConvFpropAttributes().set_padding({pH, pW}).set_stride({sH, sW}).set_dilation({dH, dW}));
    Y->set_output(true).set_dim({N, K, Ho, Wo}).set_stride({Ho * Wo * K, 1, Wo * K, K}).set_uid(3);

    FE_CHECK(graph.build(handle));
    std::printf("graph.build() ok -- conv engine selected + AOT-compiled a ck_dsl kernel\n");
    int64_t ws = 0;
    FE_CHECK(graph.get_workspace_size(ws));

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const float sc = 0.02f;  // keep fp16 stable over R*S*C accumulations
    std::vector<fp16> Xh((size_t)N * Hi * Wi * C), Wh((size_t)K * R * Sf * C),
        Yh((size_t)N * Ho * Wo * K, fp16(0.f));
    for (auto& x : Xh) x = (fp16)(d(rng) * sc);
    for (auto& x : Wh) x = (fp16)(d(rng) * sc);

    void *dX, *dWdev, *dY, *dWs = nullptr;
    HIP_CHECK(hipMalloc(&dX, Xh.size() * 2));
    HIP_CHECK(hipMalloc(&dWdev, Wh.size() * 2));
    HIP_CHECK(hipMalloc(&dY, Yh.size() * 2));
    if (ws > 0) HIP_CHECK(hipMalloc(&dWs, ws));
    HIP_CHECK(hipMemcpy(dX, Xh.data(), Xh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dWdev, Wh.data(), Wh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dY, 0, Yh.size() * 2));

    std::unordered_map<std::shared_ptr<TensorAttributes>, void*> vp = {
        {X, dX}, {W, dWdev}, {Y, dY}};
    FE_CHECK(graph.execute(handle, vp, dWs));
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("graph.execute() ok -- ck_dsl conv kernel launched\n");
    HIP_CHECK(hipMemcpy(Yh.data(), dY, Yh.size() * 2, hipMemcpyDeviceToHost));

    auto xi = [&](int n, int h, int w, int c) {
        return (float)Xh[(((size_t)n * Hi + h) * Wi + w) * C + c];
    };
    auto wi = [&](int k, int r, int s, int c) {
        return (float)Wh[(((size_t)k * R + r) * Sf + s) * C + c];
    };
    double worst = 0;
    int bad = 0;
    for (int n = 0; n < N; ++n)
        for (int ho = 0; ho < Ho; ++ho)
            for (int wo = 0; wo < Wo; ++wo)
                for (int k = 0; k < K; ++k) {
                    float acc = 0;
                    for (int r = 0; r < R; ++r) {
                        int h = ho * sH - pH + r * dH;
                        if (h < 0 || h >= Hi) continue;
                        for (int s = 0; s < Sf; ++s) {
                            int w = wo * sW - pW + s * dW;
                            if (w < 0 || w >= Wi) continue;
                            for (int c = 0; c < C; ++c) acc += xi(n, h, w, c) * wi(k, r, s, c);
                        }
                    }
                    float got = (float)Yh[(((size_t)n * Ho + ho) * Wo + wo) * K + k];
                    float diff = std::fabs(got - acc);
                    worst = std::max(worst, (double)diff);
                    if (diff > 1e-2f) ++bad;
                }
    std::printf("verify: max_abs_diff=%g bad=%d/%zu  %s\n", worst, bad, Yh.size(),
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
