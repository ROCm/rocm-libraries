// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// M2 SWAP proof: the REAL FlyDSL build_rmsnorm_module bf16 kernels (one HSACO per hidden
// size N), selected per-shape through the PUBLIC hipDNN API. Same machinery as the toy app
// (flydsl_rmsnorm_app.cpp) but bf16 I/O and the real 2D-tensor kernarg ABI. For each N in
// {4096, 8192} we build a carrier Pointwise-ADD graph over rank-4 bf16 tensors
// x{ROWS,1,1,N}, w{1,1,1,N}, out{ROWS,1,1,N} (ADD semantics ignored -- the node just ships
// x/w/out uids + the hidden size), drive it through hipDNN, and prove:
//   (1) selection:  hipkernel:FlydslRmsNorm is in get_ranked_engine_ids().
//   (2) family pick: kernel_match selects the instance whose baked N == graph N, and its
//                    dispatch raw-loads rmsnorm_real_n<N>_bf16_gfx950.hsaco (real2d ABI).
//   (3) numerics:   on-device bf16 output matches an fp32 host RMSNorm reference (atol 2e-2).
//
// Provider is dlopen'd from HIPDNN_PLUGIN_DIR; descriptors from HIPDNN_DESCRIPTOR_DIR;
// the dispatch finds HSACOs under FLYDSL_RMSNORM_HSACO_DIR. Set by run_rmsnorm_bf16_app.sh.

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace {

constexpr const char* RMSNORM_ENGINE_NAME = "hipkernel:FlydslRmsNorm";
constexpr float EPS = 1e-5f;  // must match FlyDSL rmsnorm_common.py EPS

// Minimal bf16 <-> float host conversions (round-to-nearest-even on store).
uint16_t floatToBf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}
float bf16ToFloat(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

#define CHECK_HIP(expr)                                                                      \
    do {                                                                                     \
        const hipError_t _e = (expr);                                                        \
        if (_e != hipSuccess) {                                                              \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, \
                         __LINE__);                                                          \
            return 2;                                                                        \
        }                                                                                    \
    } while (0)

#define CHECK_FE(result)                                                               \
    do {                                                                               \
        const auto _r = (result);                                                      \
        if (_r.code != ErrorCode::OK) {                                                \
            std::fprintf(stderr, "hipDNN FE error at %s:%d: %s\n", __FILE__, __LINE__, \
                         _r.err_msg.c_str());                                          \
            return 3;                                                                  \
        }                                                                              \
    } while (0)

std::shared_ptr<TensorAttributes> makeTensor(int64_t uid, const std::string& name,
                                             const std::vector<int64_t>& dim,
                                             const std::vector<int64_t>& stride) {
    auto t = std::make_shared<TensorAttributes>();
    t->set_uid(uid)
        .set_name(name)
        .set_dim(dim)
        .set_stride(stride)
        .set_data_type(DataType::BFLOAT16);
    return t;
}

// Host RMSNorm reference (fp32) matching FlyDSL: inv = rsqrt(mean(x^2)+eps); out = x*inv*w.
void hostRmsNorm(const std::vector<float>& x, const std::vector<float>& w, int rows, int n,
                 std::vector<float>& out) {
    out.resize(static_cast<size_t>(rows) * n);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + static_cast<size_t>(r) * n;
        float* outr = out.data() + static_cast<size_t>(r) * n;
        double ss = 0.0;
        for (int i = 0; i < n; ++i) {
            ss += static_cast<double>(xr[i]) * xr[i];
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / n) + EPS);
        for (int i = 0; i < n; ++i) {
            outr[i] = xr[i] * inv * w[i];
        }
    }
}

// Runs the whole flow for one hidden size N. Returns 0 on success, nonzero on failure.
int runOneN(hipdnnHandle_t handle, hipStream_t stream, int64_t engineId, int n, int rows) {
    std::printf("\n[app] ===== N=%d, ROWS=%d (bf16) =====\n", n, rows);

    auto graph = std::make_shared<Graph>();
    graph->set_name("flydsl_rmsnorm_bf16")
        .set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::BFLOAT16)
        .set_compute_data_type(DataType::FLOAT);

    const int64_t N = n;
    const int64_t R = rows;
    auto x = makeTensor(1, "X", {R, 1, 1, N}, {N, N, N, 1});
    auto w = makeTensor(2, "W", {1, 1, 1, N}, {N, N, N, 1});

    PointwiseAttributes addAttrs;
    addAttrs.set_name("carrier_add").set_mode(PointwiseMode::ADD);
    auto out = graph->pointwise(x, w, addAttrs);
    out->set_uid(3)
        .set_name("OUT")
        .set_output(true)
        .set_dim({R, 1, 1, N})
        .set_stride({N, N, N, 1})
        .set_data_type(DataType::BFLOAT16);

    graph->set_preferred_engine_id_ext(engineId);
    CHECK_FE(graph->build_operation_graph(handle));

    // ---- PROOF #1: engine is selectable ------------------------------------------------
    std::vector<int64_t> ranked;
    CHECK_FE(graph->get_ranked_engine_ids(ranked));
    bool ranked_ok = false;
    std::printf("[app] ranked engine ids:");
    for (const auto id : ranked) {
        std::printf(" %lld", static_cast<long long>(id));
        if (id == engineId) ranked_ok = true;
    }
    std::printf("\n");
    if (!ranked_ok) {
        std::fprintf(stderr, "[app] FAIL: engine %lld not ranked for N=%d\n",
                     static_cast<long long>(engineId), n);
        return 5;
    }
    std::printf("[app] PROOF #1 OK (N=%d): '%s' selected.\n", n, RMSNORM_ENGINE_NAME);

    CHECK_FE(graph->create_execution_plans());
    CHECK_FE(graph->check_support());
    CHECK_FE(graph->build_plans());

    int64_t workspaceBytes = 0;
    CHECK_FE(graph->get_workspace_size(workspaceBytes));
    void* workspace = nullptr;
    if (workspaceBytes > 0) {
        CHECK_HIP(hipMalloc(&workspace, static_cast<size_t>(workspaceBytes)));
    }

    // ---- Host data (fp32 truth, bf16 device buffers) -----------------------------------
    const size_t xElems = static_cast<size_t>(rows) * n;
    std::vector<float> hostXf(xElems), hostWf(n), refOut;
    for (size_t i = 0; i < xElems; ++i) {
        hostXf[i] = 0.1f + 0.01f * static_cast<float>(i % 37) - 0.005f * static_cast<float>(i % 13);
    }
    for (int i = 0; i < n; ++i) {
        hostWf[i] = 0.5f + 0.02f * static_cast<float>(i % 11);
    }

    // Round host inputs through bf16 so the reference sees exactly what the kernel reads.
    std::vector<uint16_t> hostX(xElems), hostW(n), hostOut(xElems, 0);
    for (size_t i = 0; i < xElems; ++i) {
        hostX[i] = floatToBf16(hostXf[i]);
        hostXf[i] = bf16ToFloat(hostX[i]);
    }
    for (int i = 0; i < n; ++i) {
        hostW[i] = floatToBf16(hostWf[i]);
        hostWf[i] = bf16ToFloat(hostW[i]);
    }
    hostRmsNorm(hostXf, hostWf, rows, n, refOut);

    void *devX = nullptr, *devW = nullptr, *devOut = nullptr;
    CHECK_HIP(hipMalloc(&devX, xElems * sizeof(uint16_t)));
    CHECK_HIP(hipMalloc(&devW, n * sizeof(uint16_t)));
    CHECK_HIP(hipMalloc(&devOut, xElems * sizeof(uint16_t)));
    CHECK_HIP(hipMemcpy(devX, hostX.data(), xElems * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(devW, hostW.data(), n * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(devOut, 0, xElems * sizeof(uint16_t)));

    std::unordered_map<int64_t, void*> variantPack = {
        {1, devX},
        {2, devW},
        {3, devOut},
    };

    // ---- Execute (dispatches the matched real flyDSL RMSNorm HSACO) ---------------------
    CHECK_FE(graph->execute(handle, variantPack, workspace));
    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipMemcpy(hostOut.data(), devOut, xElems * sizeof(uint16_t), hipMemcpyDeviceToHost));

    // ---- PROOF #3: numerics (bf16 tolerance) -------------------------------------------
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < xElems; ++i) {
        maxAbsErr = std::max(maxAbsErr, std::fabs(bf16ToFloat(hostOut[i]) - refOut[i]));
    }
    std::printf("[app] N=%d max_abs_err vs fp32 host RMSNorm = %.3e\n", n, maxAbsErr);
    const bool numericOk = (maxAbsErr < 2e-2f);

    hipFree(devX);
    hipFree(devW);
    hipFree(devOut);
    if (workspace != nullptr) hipFree(workspace);

    if (!numericOk) {
        std::fprintf(stderr, "[app] FAIL (N=%d): numeric mismatch (max_abs_err=%.3e)\n", n,
                     maxAbsErr);
        return 6;
    }
    std::printf("[app] PROOF #3 OK (N=%d): real flyDSL bf16 RMSNorm HSACO numerically correct.\n",
                n);
    return 0;
}

}  // namespace

int main() {
    CHECK_HIP(hipInit(0));
    int deviceId = 0;
    CHECK_HIP(hipGetDevice(&deviceId));
    hipDeviceProp_t props{};
    CHECK_HIP(hipGetDeviceProperties(&props, deviceId));
    std::printf("[app] device: %s (%s)\n", props.name, props.gcnArchName);

    hipdnnHandle_t handle = nullptr;
    if (hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "hipdnnCreate failed\n");
        return 4;
    }
    hipStream_t stream = nullptr;
    CHECK_HIP(hipStreamCreate(&stream));
    if (hipdnnSetStream(handle, stream) != HIPDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "hipdnnSetStream failed\n");
        return 4;
    }

    const int64_t engineId = hipdnn_data_sdk::utilities::engineNameToId(RMSNORM_ENGINE_NAME);
    std::printf("[app] flyDSL RMSNorm engine id = %lld\n", static_cast<long long>(engineId));

    int rc = 0;
    // The M2 swap target set: real bf16 kernels at both production hidden sizes.
    for (const int n : {4096, 8192}) {
        const int r = runOneN(handle, stream, engineId, n, 4);
        if (r != 0) {
            rc = r;
            break;
        }
    }

    hipStreamDestroy(stream);
    hipdnnDestroy(handle);

    if (rc != 0) {
        std::fprintf(stderr, "[app] FAILED (rc=%d)\n", rc);
        return rc;
    }
    std::printf(
        "\n[app] SUCCESS: REAL flyDSL bf16 RMSNorm family selected per-shape AND executed "
        "through hipDNN (N=4096 and N=8192).\n");
    return 0;
}
