// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// M4 autotune/bench: for the NON-CAUSAL bf16 head_dim=128 prefill SDPA bake-off shape(s) on
// gfx950, enumerate all applicable engines (no pin), then for EACH engine build its plan, verify
// numerics once vs an fp32 host reference, warm up, and TIME it (hipEvent, median of N reps).
// Prints a per-shape ranked winner table (fastest first) — the "autotune picks a winner per
// shape" substance of M4. Engines currently in the ring: FlydslAttention (raw-loaded flyDSL
// HSACO) + ASM_SDPA_ENGINE (aiter fwd_hd128_bf16.co); any additional engine (e.g. rocKE
// Gfx950AttentionDense) that enumerates will be timed and ranked automatically with no code change.
//
// Usage: flydsl_attention_autotune_app [reps]   (reps default 50)

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>

#include <algorithm>
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
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace {

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

void hostSdpaNonCausal(const std::vector<float>& q, const std::vector<float>& k,
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
                for (int j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += q[off(b, i, h, d)] * k[off(b, j, h, d)];
                    scores[j] = dot * scale;
                    maxLogit = std::max(maxLogit, scores[j]);
                }
                float denom = 0.0f;
                for (int j = 0; j < S; ++j) {
                    scores[j] = std::exp(scores[j] - maxLogit);
                    denom += scores[j];
                }
                const float invDenom = 1.0f / denom;
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < S; ++j) acc += scores[j] * v[off(b, j, h, d)];
                    out[off(b, i, h, d)] = acc * invDenom;
                }
            }
        }
    }
}

struct Shape {
    int B, H, S, D;
};

int buildGraph(hipdnnHandle_t handle, const Shape& sh, float scale, int64_t pinnedEngine,
               std::shared_ptr<Graph>& graphOut) {
    auto graph = std::make_shared<Graph>();
    graph->set_name("bakeoff_autotune")
        .set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    const int64_t sB = static_cast<int64_t>(sh.S) * sh.H * sh.D;
    const int64_t sH = sh.D;
    const int64_t sS = static_cast<int64_t>(sh.H) * sh.D;
    const std::vector<int64_t> dim = {sh.B, sh.H, sh.S, sh.D};
    const std::vector<int64_t> stride = {sB, sH, sS, 1};

    auto q = makeTensor(1, "Q", dim, stride);
    auto k = makeTensor(2, "K", dim, stride);
    auto v = makeTensor(3, "V", dim, stride);

    SdpaAttributes attr;
    attr.set_name("bakeoff_sdpa").set_attn_scale(scale).set_causal_mask(false).set_generate_stats(
        false);

    auto outs = graph->sdpa(q, k, v, attr);
    outs[0]
        ->set_uid(4)
        .set_name("O")
        .set_output(true)
        .set_dim(dim)
        .set_stride(stride)
        .set_data_type(DataType::BFLOAT16);

    if (pinnedEngine != 0) graph->set_preferred_engine_id_ext(pinnedEngine);
    CHECK_FE(graph->build_operation_graph(handle));
    graphOut = graph;
    return 0;
}

struct BenchResult {
    int64_t id;
    std::string name;
    bool ran = false;
    bool numOk = false;
    float maxAbsErr = 0.0f;
    float medianMs = 0.0f;
};

}  // namespace

int main(int argc, char** argv) {
    const int reps = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 50;

    CHECK_HIP(hipInit(0));
    int deviceId = 0;
    CHECK_HIP(hipGetDevice(&deviceId));
    hipDeviceProp_t props{};
    CHECK_HIP(hipGetDeviceProperties(&props, deviceId));
    std::printf("[app] device: %s (%s), reps=%d\n", props.name, props.gcnArchName, reps);

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

    // A few prefill shapes: same head-config (H=8 -> HSACO family present), varying seq_len/batch
    // (free runtime dims for one flyDSL HSACO) plus an H=16 config.
    const std::vector<Shape> shapes = {
        {1, 8, 256, 128}, {1, 8, 512, 128}, {2, 8, 512, 128}, {1, 16, 256, 128},
    };

    int totalFailures = 0;
    for (const Shape& sh : shapes) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(sh.D));
        const size_t elems = static_cast<size_t>(sh.B) * sh.S * sh.H * sh.D;

        std::vector<float> hostQf(elems), hostKf(elems), hostVf(elems), refOut;
        auto fill = [&](std::vector<float>& buf, float base, float amp, int mod) {
            for (size_t i = 0; i < buf.size(); ++i)
                buf[i] = base + amp * std::sin(static_cast<float>(i % mod) * 0.3f);
        };
        fill(hostQf, 0.02f, 0.10f, 37);
        fill(hostKf, -0.01f, 0.12f, 41);
        fill(hostVf, 0.03f, 0.15f, 29);
        std::vector<uint16_t> hostQ(elems), hostK(elems), hostV(elems);
        auto quantize = [&](std::vector<float>& f, std::vector<uint16_t>& h) {
            for (size_t i = 0; i < f.size(); ++i) {
                h[i] = floatToBf16(f[i]);
                f[i] = bf16ToFloat(h[i]);
            }
        };
        quantize(hostQf, hostQ);
        quantize(hostKf, hostK);
        quantize(hostVf, hostV);
        hostSdpaNonCausal(hostQf, hostKf, hostVf, sh.B, sh.H, sh.S, sh.D, scale, refOut);

        void *devQ = nullptr, *devK = nullptr, *devV = nullptr, *devO = nullptr;
        CHECK_HIP(hipMalloc(&devQ, elems * sizeof(uint16_t)));
        CHECK_HIP(hipMalloc(&devK, elems * sizeof(uint16_t)));
        CHECK_HIP(hipMalloc(&devV, elems * sizeof(uint16_t)));
        CHECK_HIP(hipMalloc(&devO, elems * sizeof(uint16_t)));
        CHECK_HIP(hipMemcpy(devQ, hostQ.data(), elems * sizeof(uint16_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(devK, hostK.data(), elems * sizeof(uint16_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(devV, hostV.data(), elems * sizeof(uint16_t), hipMemcpyHostToDevice));

        std::shared_ptr<Graph> probe;
        if (int r = buildGraph(handle, sh, scale, 0, probe)) return r;
        std::vector<int64_t> ranked;
        CHECK_FE(probe->get_ranked_engine_ids(ranked));

        std::printf("\n================ shape B=%d H=%d S=%d D=%d : %zu engine(s) =============\n",
                    sh.B, sh.H, sh.S, sh.D, ranked.size());

        std::vector<BenchResult> results;
        for (const int64_t id : ranked) {
            BenchResult br;
            br.id = id;
            br.name = hipdnn_data_sdk::utilities::engineNameOrHex(id);

            std::shared_ptr<Graph> g;
            if (int r = buildGraph(handle, sh, scale, id, g)) return r;
            std::vector<int64_t> gr;
            CHECK_FE(g->get_ranked_engine_ids(gr));
            if (std::find(gr.begin(), gr.end(), id) == gr.end()) {
                std::printf("  [%s] pin failed; skipping\n", br.name.c_str());
                results.push_back(br);
                continue;
            }
            CHECK_FE(g->create_execution_plans());
            CHECK_FE(g->check_support());
            CHECK_FE(g->build_plans());
            int64_t wsBytes = 0;
            CHECK_FE(g->get_workspace_size(wsBytes));
            void* ws = nullptr;
            if (wsBytes > 0) CHECK_HIP(hipMalloc(&ws, static_cast<size_t>(wsBytes)));

            std::unordered_map<int64_t, void*> vp = {{1, devQ}, {2, devK}, {3, devV}, {4, devO}};

            // Correctness (one shot).
            CHECK_HIP(hipMemset(devO, 0, elems * sizeof(uint16_t)));
            CHECK_FE(g->execute(handle, vp, ws));
            CHECK_HIP(hipStreamSynchronize(stream));
            std::vector<uint16_t> hostO(elems, 0);
            CHECK_HIP(
                hipMemcpy(hostO.data(), devO, elems * sizeof(uint16_t), hipMemcpyDeviceToHost));
            br.maxAbsErr = 0.0f;
            for (size_t i = 0; i < elems; ++i)
                br.maxAbsErr = std::max(br.maxAbsErr, std::fabs(bf16ToFloat(hostO[i]) - refOut[i]));
            br.numOk = (br.maxAbsErr < 5e-2f) && std::isfinite(br.maxAbsErr);
            br.ran = true;

            // Warmup.
            for (int w = 0; w < 5; ++w) {
                CHECK_FE(g->execute(handle, vp, ws));
            }
            CHECK_HIP(hipStreamSynchronize(stream));

            // Timed reps (per-iter event timing -> median).
            std::vector<float> samples(reps, 0.0f);
            hipEvent_t start, stop;
            CHECK_HIP(hipEventCreate(&start));
            CHECK_HIP(hipEventCreate(&stop));
            for (int it = 0; it < reps; ++it) {
                CHECK_HIP(hipEventRecord(start, stream));
                CHECK_FE(g->execute(handle, vp, ws));
                CHECK_HIP(hipEventRecord(stop, stream));
                CHECK_HIP(hipEventSynchronize(stop));
                float ms = 0.0f;
                CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
                samples[it] = ms;
            }
            CHECK_HIP(hipEventDestroy(start));
            CHECK_HIP(hipEventDestroy(stop));
            std::sort(samples.begin(), samples.end());
            br.medianMs = samples[samples.size() / 2];

            if (ws != nullptr) CHECK_HIP(hipFree(ws));
            std::printf("  [%-38s] err=%.2e %-4s median=%.4f ms\n", br.name.c_str(), br.maxAbsErr,
                        br.numOk ? "OK" : "BAD", br.medianMs);
            if (!br.numOk) ++totalFailures;
            results.push_back(br);
        }

        // Rank by median latency among engines that ran + were numerically OK.
        std::vector<BenchResult> ok;
        for (const auto& r : results)
            if (r.ran && r.numOk) ok.push_back(r);
        std::sort(ok.begin(), ok.end(),
                  [](const BenchResult& a, const BenchResult& b) { return a.medianMs < b.medianMs; });
        std::printf("  ---- WINNER RANKING (fastest first) ----\n");
        for (size_t i = 0; i < ok.size(); ++i) {
            const char* tag = (i == 0) ? "  <== WINNER" : "";
            std::printf("   %zu. %-38s %.4f ms%s\n", i + 1, ok[i].name.c_str(), ok[i].medianMs, tag);
        }

        CHECK_HIP(hipFree(devQ));
        CHECK_HIP(hipFree(devK));
        CHECK_HIP(hipFree(devV));
        CHECK_HIP(hipFree(devO));
    }

    hipStreamDestroy(stream);
    hipdnnDestroy(handle);

    if (totalFailures != 0) {
        std::fprintf(stderr, "\n[app] %d engine/shape combo(s) failed numerics.\n", totalFailures);
        return 6;
    }
    std::printf("\n[app] SUCCESS: bake-off autotuned across %zu shapes; ranked winner per shape.\n",
                shapes.size());
    return 0;
}
