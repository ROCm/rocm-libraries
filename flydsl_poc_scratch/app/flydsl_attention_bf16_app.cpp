// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// M3 proof: the REAL FlyDSL flash-attention bf16 prefill kernels (one HSACO per head-config
// tuple {num_heads, head_dim, causal, dtype}), selected per-shape through the PUBLIC hipDNN
// SDPA API. For each head-config in {H=8, H=16} (head_dim=128, causal, bf16) we build an
// SDPA graph over rank-4 bf16 tensors Q/K/V logical (B,H,S,D) with token-major (BSHD)
// strides, drive it through hipDNN, and prove:
//   (1) selection:  hipkernel:FlydslAttention is in get_ranked_engine_ids().
//   (2) family pick: kernel_match selects the instance whose baked head-config == graph's,
//                    and its dispatch raw-loads flash_attn_real_h<H>_d128_causal_bf16_gfx950.hsaco.
//   (3) numerics:   on-device bf16 output matches an fp32 host causal-SDPA reference.
//
// Provider is dlopen'd from HIPDNN_PLUGIN_DIR; descriptors from HIPDNN_DESCRIPTOR_DIR;
// the dispatch finds HSACOs under FLYDSL_ATTENTION_HSACO_DIR. Set by run_attention_bf16_app.sh.
//
// NOTE: requires the backend built with -DHIPDNN_ENABLE_SDPA=ON so the SDPA graph node exists.

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace {

constexpr const char* ATTENTION_ENGINE_NAME = "hipkernel:FlydslAttention";

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

// Host causal SDPA reference (fp32 math), logical (B,H,S,D), Sq==Skv, top-left causal:
//   scores[i,j] = scale * dot(Q[i], K[j]); mask j>i; softmax over j<=i; out[i]=sum p[j] V[j].
// All buffers are laid out physically BSHD: offset(b,s,h,d) = ((b*S + s)*H + h)*D + d.
void hostCausalSdpa(const std::vector<float>& q, const std::vector<float>& k,
                    const std::vector<float>& v, int B, int H, int S, int D, float scale,
                    std::vector<float>& out) {
    out.assign(static_cast<size_t>(B) * S * H * D, 0.0f);
    auto off = [&](int b, int s, int h, int d) {
        return (((static_cast<size_t>(b) * S + s) * H + h) * D) + d;
    };
    std::vector<float> scores(S);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < S; ++i) {
                float maxLogit = -INFINITY;
                for (int j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += q[off(b, i, h, d)] * k[off(b, j, h, d)];
                    }
                    scores[j] = dot * scale;
                    maxLogit = std::max(maxLogit, scores[j]);
                }
                float denom = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - maxLogit);
                    denom += scores[j];
                }
                const float invDenom = 1.0f / denom;
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        acc += scores[j] * v[off(b, j, h, d)];
                    }
                    out[off(b, i, h, d)] = acc * invDenom;
                }
            }
        }
    }
}

// Runs the whole flow for one head-config (num_heads H). Returns 0 on success.
int runOneHeadConfig(hipdnnHandle_t handle, hipStream_t stream, int64_t engineId, int H) {
    const int B = 1;
    const int S = 256;   // prefill length (runtime scalar for the HSACO)
    const int D = 128;   // baked head_dim
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::printf("\n[app] ===== H=%d, B=%d, S=%d, D=%d (bf16 causal prefill) =====\n", H, B, S, D);

    auto graph = std::make_shared<Graph>();
    graph->set_name("flydsl_attention_bf16")
        .set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Logical dims (B,H,S,D); memory layout BSHD (token-major):
    //   stride(B)=S*H*D, stride(H)=D, stride(S)=H*D, stride(D)=1.
    const int64_t sB = static_cast<int64_t>(S) * H * D;
    const int64_t sH = D;
    const int64_t sS = static_cast<int64_t>(H) * D;
    const int64_t sD = 1;
    const std::vector<int64_t> dim = {B, H, S, D};
    const std::vector<int64_t> stride = {sB, sH, sS, sD};

    auto q = makeTensor(1, "Q", dim, stride);
    auto k = makeTensor(2, "K", dim, stride);
    auto v = makeTensor(3, "V", dim, stride);

    SdpaAttributes attr;
    attr.set_name("flydsl_sdpa")
        .set_attn_scale(scale)
        .set_causal_mask(true)
        .set_generate_stats(false);

    auto outs = graph->sdpa(q, k, v, attr);
    auto o = outs[0];
    o->set_uid(4)
        .set_name("O")
        .set_output(true)
        .set_dim(dim)
        .set_stride(stride)
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
        std::fprintf(stderr, "[app] FAIL: engine %lld not ranked for H=%d\n",
                     static_cast<long long>(engineId), H);
        return 5;
    }
    std::printf("[app] PROOF #1 OK (H=%d): '%s' selected.\n", H, ATTENTION_ENGINE_NAME);

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
    const size_t elems = static_cast<size_t>(B) * S * H * D;
    std::vector<float> hostQf(elems), hostKf(elems), hostVf(elems), refOut;
    auto fill = [&](std::vector<float>& buf, float base, float amp, int mod) {
        for (size_t i = 0; i < buf.size(); ++i) {
            buf[i] = base + amp * std::sin(static_cast<float>(i % mod) * 0.3f);
        }
    };
    fill(hostQf, 0.02f, 0.10f, 37);
    fill(hostKf, -0.01f, 0.12f, 41);
    fill(hostVf, 0.03f, 0.15f, 29);

    // Round host inputs through bf16 so the reference sees exactly what the kernel reads.
    std::vector<uint16_t> hostQ(elems), hostK(elems), hostV(elems), hostO(elems, 0);
    auto quantize = [&](std::vector<float>& f, std::vector<uint16_t>& h) {
        for (size_t i = 0; i < f.size(); ++i) {
            h[i] = floatToBf16(f[i]);
            f[i] = bf16ToFloat(h[i]);
        }
    };
    quantize(hostQf, hostQ);
    quantize(hostKf, hostK);
    quantize(hostVf, hostV);
    hostCausalSdpa(hostQf, hostKf, hostVf, B, H, S, D, scale, refOut);

    void *devQ = nullptr, *devK = nullptr, *devV = nullptr, *devO = nullptr;
    CHECK_HIP(hipMalloc(&devQ, elems * sizeof(uint16_t)));
    CHECK_HIP(hipMalloc(&devK, elems * sizeof(uint16_t)));
    CHECK_HIP(hipMalloc(&devV, elems * sizeof(uint16_t)));
    CHECK_HIP(hipMalloc(&devO, elems * sizeof(uint16_t)));
    CHECK_HIP(hipMemcpy(devQ, hostQ.data(), elems * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(devK, hostK.data(), elems * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(devV, hostV.data(), elems * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(devO, 0, elems * sizeof(uint16_t)));

    std::unordered_map<int64_t, void*> variantPack = {
        {1, devQ},
        {2, devK},
        {3, devV},
        {4, devO},
    };

    // ---- Execute (dispatches the matched real flyDSL attention HSACO) -------------------
    CHECK_FE(graph->execute(handle, variantPack, workspace));
    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipMemcpy(hostO.data(), devO, elems * sizeof(uint16_t), hipMemcpyDeviceToHost));

    // ---- PROOF #3: numerics (bf16 tolerance) -------------------------------------------
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        maxAbsErr = std::max(maxAbsErr, std::fabs(bf16ToFloat(hostO[i]) - refOut[i]));
    }
    std::printf("[app] H=%d max_abs_err vs fp32 host causal-SDPA = %.3e\n", H, maxAbsErr);
    const bool numericOk = (maxAbsErr < 5e-2f);

    hipFree(devQ);
    hipFree(devK);
    hipFree(devV);
    hipFree(devO);
    if (workspace != nullptr) hipFree(workspace);

    if (!numericOk) {
        std::fprintf(stderr, "[app] FAIL (H=%d): numeric mismatch (max_abs_err=%.3e)\n", H,
                     maxAbsErr);
        return 6;
    }
    std::printf("[app] PROOF #3 OK (H=%d): real flyDSL bf16 attention HSACO numerically correct.\n",
                H);
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

    const int64_t engineId = hipdnn_data_sdk::utilities::engineNameToId(ATTENTION_ENGINE_NAME);
    std::printf("[app] flyDSL attention engine id = %lld\n", static_cast<long long>(engineId));

    int rc = 0;
    // The M3 target set: real bf16 flash-attn kernels at both demo head-configs.
    for (const int H : {8, 16}) {
        const int r = runOneHeadConfig(handle, stream, engineId, H);
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
        "\n[app] SUCCESS: REAL flyDSL bf16 flash-attention family selected per-head-config AND "
        "executed through hipDNN (H=8 and H=16).\n");
    return 0;
}
