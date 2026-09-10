// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Standalone proof: a plain C++ app builds a hipDNN pointwise-ADD graph, drives it
// through the PUBLIC hipDNN API, and shows hipDNN's kernel-ingestor engine SELECT and
// RUN the flyDSL kernel (compiled from flyDSL -> HSACO at build time, raw-loaded at
// dispatch). Two independent proofs:
//   (1) selection: get_ranked_engine_ids() lists "hipkernel:Flydsl".
//   (2) execution: 3 + 4 == 7 computed on-device by the flyDSL vadd_0 kernel.
//
// Nothing here links the provider; it is dlopen'd from HIPDNN_PLUGIN_DIR (or the path
// passed to hipdnnSetEnginePluginPaths_ext). Descriptors come from HIPDNN_DESCRIPTOR_DIR.

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>  // C API: hipdnnCreate/SetStream/Destroy

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace {

constexpr const char* FLYDSL_ENGINE_NAME = "hipkernel:Flydsl";

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

std::shared_ptr<TensorAttributes> scalarTensor(int64_t uid, const std::string& name) {
    auto t = std::make_shared<TensorAttributes>();
    t->set_uid(uid)
        .set_name(name)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(DataType::FLOAT);
    return t;
}

}  // namespace

int main() {
    // ---- Device + provider plugin load -------------------------------------------------
    CHECK_HIP(hipInit(0));
    int deviceId = 0;
    CHECK_HIP(hipGetDevice(&deviceId));

    hipDeviceProp_t props{};
    CHECK_HIP(hipGetDeviceProperties(&props, deviceId));
    std::printf("[app] device: %s (%s)\n", props.name, props.gcnArchName);

    // The provider .so is discovered via HIPDNN_PLUGIN_DIR (set by the run script). No
    // link dependency on the provider -- it is dlopen'd by the backend at handle creation.

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

    const int64_t flydslEngineId = hipdnn_data_sdk::utilities::engineNameToId(FLYDSL_ENGINE_NAME);
    std::printf("[app] flyDSL engine id = %lld\n", static_cast<long long>(flydslEngineId));

    // ---- Build a single-node pointwise ADD graph (rank 4, single element) --------------
    auto graph = std::make_shared<Graph>();
    graph->set_name("flydsl_add")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto a = scalarTensor(1, "A");
    auto b = scalarTensor(2, "B");

    PointwiseAttributes addAttrs;
    addAttrs.set_name("add").set_mode(PointwiseMode::ADD);
    auto c = graph->pointwise(a, b, addAttrs);
    c->set_uid(3).set_name("C").set_output(true).set_data_type(DataType::FLOAT);

    // Pin the flyDSL engine so selection is unambiguous, then build the op graph.
    graph->set_preferred_engine_id_ext(flydslEngineId);
    CHECK_FE(graph->build_operation_graph(handle));

    // ---- PROOF #1: flyDSL engine is in the ranked (selectable) set ----------------------
    std::vector<int64_t> ranked;
    CHECK_FE(graph->get_ranked_engine_ids(ranked));
    bool flydslRanked = false;
    std::printf("[app] ranked engine ids:");
    for (const auto id : ranked) {
        std::printf(" %lld", static_cast<long long>(id));
        if (id == flydslEngineId) {
            flydslRanked = true;
        }
    }
    std::printf("\n");
    if (!flydslRanked) {
        std::fprintf(stderr, "[app] FAIL: flyDSL engine %lld not in ranked set -- not selected.\n",
                     static_cast<long long>(flydslEngineId));
        return 5;
    }
    std::printf("[app] PROOF #1 OK: hipDNN selected engine '%s'.\n", FLYDSL_ENGINE_NAME);

    // ---- Compile the plan --------------------------------------------------------------
    CHECK_FE(graph->create_execution_plans());
    CHECK_FE(graph->check_support());
    CHECK_FE(graph->build_plans());

    int64_t workspaceBytes = 0;
    CHECK_FE(graph->get_workspace_size(workspaceBytes));
    void* workspace = nullptr;
    if (workspaceBytes > 0) {
        CHECK_HIP(hipMalloc(&workspace, static_cast<size_t>(workspaceBytes)));
    }

    // ---- Device buffers: A=3, B=4, C=? -------------------------------------------------
    const float hostA = 3.0f;
    const float hostB = 4.0f;
    float hostC = -1.0f;
    float *devA = nullptr, *devB = nullptr, *devC = nullptr;
    CHECK_HIP(hipMalloc(&devA, sizeof(float)));
    CHECK_HIP(hipMalloc(&devB, sizeof(float)));
    CHECK_HIP(hipMalloc(&devC, sizeof(float)));
    CHECK_HIP(hipMemcpy(devA, &hostA, sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(devB, &hostB, sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(devC, 0, sizeof(float)));

    std::unordered_map<int64_t, void*> variantPack = {
        {1, devA},
        {2, devB},
        {3, devC},
    };

    // ---- Execute through hipDNN (dispatches the flyDSL kernel) --------------------------
    CHECK_FE(graph->execute(handle, variantPack, workspace));
    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipMemcpy(&hostC, devC, sizeof(float), hipMemcpyDeviceToHost));

    // ---- PROOF #2: numeric result from the flyDSL kernel --------------------------------
    std::printf("[app] executed: %.1f + %.1f = %.1f (flyDSL kernel result)\n", hostA, hostB, hostC);
    const bool numericOk = (hostC == hostA + hostB);
    if (!numericOk) {
        std::fprintf(stderr, "[app] FAIL: expected %.1f, got %.1f\n", hostA + hostB, hostC);
    } else {
        std::printf("[app] PROOF #2 OK: flyDSL kernel computed the correct sum on device.\n");
    }

    // ---- Cleanup -----------------------------------------------------------------------
    hipFree(devA);
    hipFree(devB);
    hipFree(devC);
    if (workspace != nullptr) {
        hipFree(workspace);
    }
    hipStreamDestroy(stream);
    hipdnnDestroy(handle);

    if (!numericOk) {
        return 6;
    }
    std::printf("[app] SUCCESS: flyDSL kernel selected AND executed through hipDNN.\n");
    return 0;
}
