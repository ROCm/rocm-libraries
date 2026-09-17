// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// M4 de-risk probe: build ONE bf16 causal-prefill SDPA graph and let hipDNN enumerate ALL
// applicable engines (NO set_preferred_engine_id_ext pin). Prints the full ranked engine
// list by NAME so we can see which SDPA engines "sit in the ring" on gfx950 for this shape:
//   - hipkernel:FlydslAttention  (our M3 flyDSL raw-load pack)
//   - the in-tree ASM_SDPA engine (if its tuned config covers D=128 causal bf16)
//   - rocKE                       (only if built with -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON)
//
// This is the "confirm all engines enumerate" half of M4. Autotuned winner-selection +
// per-engine execution/numerics come next. No pinning, no numeric check here.

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace {

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

int probeOneShape(hipdnnHandle_t handle, int H, bool causal) {
    const int B = 1;
    const int S = 256;
    const int D = 128;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::printf("\n[app] ===== bake-off probe: H=%d, B=%d, S=%d, D=%d (bf16 %s prefill) =====\n",
                H, B, S, D, causal ? "causal" : "non-causal");

    auto graph = std::make_shared<Graph>();
    graph->set_name("flydsl_attention_bakeoff")
        .set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    const int64_t sB = static_cast<int64_t>(S) * H * D;
    const int64_t sH = D;
    const int64_t sS = static_cast<int64_t>(H) * D;
    const std::vector<int64_t> dim = {B, H, S, D};
    const std::vector<int64_t> stride = {sB, sH, sS, 1};

    auto q = makeTensor(1, "Q", dim, stride);
    auto k = makeTensor(2, "K", dim, stride);
    auto v = makeTensor(3, "V", dim, stride);

    SdpaAttributes attr;
    attr.set_name("bakeoff_sdpa")
        .set_attn_scale(scale)
        .set_causal_mask(causal)
        .set_generate_stats(false);

    auto outs = graph->sdpa(q, k, v, attr);
    outs[0]
        ->set_uid(4)
        .set_name("O")
        .set_output(true)
        .set_dim(dim)
        .set_stride(stride)
        .set_data_type(DataType::BFLOAT16);

    // NO set_preferred_engine_id_ext -> let every applicable engine enumerate.
    CHECK_FE(graph->build_operation_graph(handle));

    std::vector<int64_t> ranked;
    CHECK_FE(graph->get_ranked_engine_ids(ranked));
    std::printf("[app] %zu engine(s) enumerated for this SDPA shape:\n", ranked.size());
    for (const auto id : ranked) {
        const std::string name = hipdnn_data_sdk::utilities::engineNameOrHex(id);
        std::printf("        %-40s (id=%lld)\n", name.c_str(), static_cast<long long>(id));
    }
    if (ranked.empty()) {
        std::fprintf(stderr, "[app] FAIL: no engines enumerated for H=%d\n", H);
        return 5;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1] = causal flag (default 1). Pass 0 to probe the non-causal shape that the
    // gfx950 asm_sdpa catalog covers (fwd_hd128_bf16.co, mask=0).
    const bool causal = (argc > 1) ? (std::atoi(argv[1]) != 0) : true;

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

    int rc = 0;
    for (const int H : {8, 16}) {
        const int r = probeOneShape(handle, H, causal);
        if (r != 0) {
            rc = r;
            break;
        }
    }

    hipStreamDestroy(stream);
    hipdnnDestroy(handle);

    if (rc != 0) {
        std::fprintf(stderr, "[app] probe FAILED (rc=%d)\n", rc);
        return rc;
    }
    std::printf("\n[app] bake-off probe done: see the per-shape engine lists above.\n");
    return 0;
}
