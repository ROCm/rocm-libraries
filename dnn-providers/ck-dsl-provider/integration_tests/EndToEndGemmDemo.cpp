// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end: hipDNN Graph API (Matmul) -> ck-dsl-provider plugin -> ck_dsl
// kernel on GPU -> numerical verify. Proves the whole Python-free stack through
// the real frontend: graph.build() (plugin discovered, isApplicable, buildPlan,
// AOT comgr-compile/load) + graph.execute() (kernarg pack + launch).
//
// Layout note: the shipped ck_dsl GEMM is RCR (C[m,n]=sum_k A[m,k]*B[n,k], with
// B stored [N,K]). hipDNN Matmul validates K-consistency on A[M,K] @ B[K,N], so
// we DECLARE B with logical dims [K,N] but with RCR strides {1,K} (the K axis is
// contiguous == B physically [N,K]), and UPLOAD the device buffer in [N,K]
// order. The provider's B-layout detector reads these strides and selects the
// RCR kernel; a genuine row-major {N,1} B is now rejected cleanly instead of
// silently miscomputed.
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

#define HIP_CHECK(e)                                                                       \
    do {                                                                                   \
        hipError_t _e = (e);                                                               \
        if (_e != hipSuccess) {                                                            \
            std::fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(_e), __LINE__); \
            return 2;                                                                      \
        }                                                                                  \
    } while (0)

#define FE_CHECK(e)                                                                       \
    do {                                                                                  \
        auto _e = (e);                                                                    \
        if (_e.get_code() != hipdnn_frontend::ErrorCode::OK) {                            \
            std::fprintf(stderr, "hipDNN FE error: %s at %d\n", _e.get_message().c_str(), \
                         __LINE__);                                                       \
            return 3;                                                                     \
        }                                                                                 \
    } while (0)

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 256;
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("=== ck-dsl-provider GEMM end-to-end (hipDNN graph) ===\n");
    std::printf("device=%s  M=%d N=%d K=%d\n", prop.gcnArchName, M, N, K);

    if (const char* pdir = std::getenv("HIPDNN_PLUGIN_PATH")) {
        const char* paths[] = {pdir};
        hipdnnSetEnginePluginPaths_ext(1, paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
        std::printf("plugin path: %s\n", pdir);
    }
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    Graph graph;
    graph.set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_name("ckdsl_gemm");
    graph.set_preferred_engine_id_ext(std::string("CK_DSL_GEMM_ENGINE"));

    auto A = Graph::tensor(TensorAttributes().set_dim({M, K}).set_stride({K, 1}).set_uid(1));
    // RCR strides: logical dims [K,N] but the K axis is contiguous ({1,K}),
    // i.e. B is physically stored [N,K] -- the shipped kernel's native ABI.
    auto B = Graph::tensor(TensorAttributes().set_dim({K, N}).set_stride({1, K}).set_uid(2));
    auto C = graph.matmul(A, B, MatmulAttributes());
    C->set_output(true).set_dim({M, N}).set_stride({N, 1}).set_uid(3);

    FE_CHECK(graph.build(handle));
    std::printf("graph.build() ok -- plugin selected a ck_dsl kernel\n");

    int64_t ws = 0;
    FE_CHECK(graph.get_workspace_size(ws));

    // Host data: A[M,K] row-major, B_std[K,N] row-major (the math operands).
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> d(-4.f, 4.f);
    std::vector<fp16> A_h((size_t)M * K), Bstd((size_t)K * N), Bbuf((size_t)N * K),
        C_h((size_t)M * N, fp16(0.f));
    for (auto& x : A_h) x = (fp16)std::round(d(rng));
    for (auto& x : Bstd) x = (fp16)std::round(d(rng));
    // RCR device buffer: Bbuf[n*K+k] = Bstd[k*N+n]  (so kernel's B[N,K] read == Bstd^T)
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) Bbuf[(size_t)n * K + k] = Bstd[(size_t)k * N + n];

    void *dA, *dB, *dC, *dWs = nullptr;
    HIP_CHECK(hipMalloc(&dA, A_h.size() * 2));
    HIP_CHECK(hipMalloc(&dB, Bbuf.size() * 2));
    HIP_CHECK(hipMalloc(&dC, C_h.size() * 2));
    if (ws > 0) HIP_CHECK(hipMalloc(&dWs, ws));
    HIP_CHECK(hipMemcpy(dA, A_h.data(), A_h.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, Bbuf.data(), Bbuf.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, C_h.size() * 2));

    std::unordered_map<std::shared_ptr<TensorAttributes>, void*> vp = {{A, dA}, {B, dB}, {C, dC}};
    FE_CHECK(graph.execute(handle, vp, dWs));
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("graph.execute() ok -- ck_dsl kernel launched\n");

    HIP_CHECK(hipMemcpy(C_h.data(), dC, C_h.size() * 2, hipMemcpyDeviceToHost));

    double worst = 0;
    int bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (float)A_h[(size_t)m * K + k] * (float)Bstd[(size_t)k * N + n];
            float diff = std::fabs((float)C_h[(size_t)m * N + n] - acc);
            worst = std::max(worst, (double)diff);
            if (diff > 4.0f) ++bad;
        }
    std::printf("verify: max_abs_diff=%g bad=%d/%d  %s\n", worst, bad, M * N,
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
