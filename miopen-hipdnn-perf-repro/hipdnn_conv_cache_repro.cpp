// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Standalone reproducer for hipDNN's cross-process non-caching of the conv
// plan build. This program uses ONLY the hipDNN backend C API (hipdnnBackend*) —
// it does not touch MIOpen's public API.
//
// It builds a conv forward execution plan the standard backend way: tensor
// descriptors -> conv-fwd op -> op graph -> EngineHeur(FALLBACK) -> EngineConfig
// -> ExecutionPlan -> finalize, then a VariantPack execute.
//
//   N=16 C=16 H=16 W=16  K=16 R=3 S=3  pad=1 stride=1 dil=1  (NCHW, fp32)
//
// Each process builds the plan exactly once and times it. Run the SAME binary
// twice as two separate processes (see run_repro.sh). If the second process's
// "conv plan build" is still multi-second, the non-caching is inherent to the
// hipDNN backend plan-build path itself. Compare with native MIOpen via
// run_miopen_driver.sh.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_backend.h>

namespace {

constexpr int64_t kUidX = 1;
constexpr int64_t kUidW = 2;
constexpr int64_t kUidY = 3;

long long ns_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

#define CHECK_HIPDNN(expr)                                                                \
    do                                                                                    \
    {                                                                                     \
        const hipdnnStatus_t _s = (expr);                                                 \
        if(_s != HIPDNN_STATUS_SUCCESS)                                                   \
        {                                                                                 \
            std::fprintf(stderr, "[repro] %s failed: status=%d (%s:%d)\n", #expr,         \
                         static_cast<int>(_s), __FILE__, __LINE__);                       \
            return false;                                                                 \
        }                                                                                 \
    } while(0)

#define CHECK_HIP(expr)                                                                   \
    do                                                                                    \
    {                                                                                     \
        const hipError_t _e = (expr);                                                     \
        if(_e != hipSuccess)                                                              \
        {                                                                                 \
            std::fprintf(stderr, "[repro] %s failed: %s (%s:%d)\n", #expr,               \
                         hipGetErrorString(_e), __FILE__, __LINE__);                      \
            return false;                                                                 \
        }                                                                                 \
    } while(0)

hipdnnBackendDescriptor_t make_tensor_desc(int64_t uid,
                                           const std::vector<int64_t>& dims,
                                           const std::vector<int64_t>& strides)
{
    hipdnnBackendDescriptor_t d = nullptr;
    if(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, &d) != HIPDNN_STATUS_SUCCESS)
        return nullptr;
    int64_t uidValue          = uid;
    bool isVirtual            = false;
    hipdnnDataType_t dtype    = HIPDNN_DATA_FLOAT;
    bool ok = hipdnnBackendSetAttribute(d, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uidValue) == HIPDNN_STATUS_SUCCESS;
    ok = ok && hipdnnBackendSetAttribute(d, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtype) == HIPDNN_STATUS_SUCCESS;
    ok = ok && hipdnnBackendSetAttribute(d, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, static_cast<int64_t>(dims.size()), dims.data()) == HIPDNN_STATUS_SUCCESS;
    ok = ok && hipdnnBackendSetAttribute(d, HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, static_cast<int64_t>(strides.size()), strides.data()) == HIPDNN_STATUS_SUCCESS;
    ok = ok && hipdnnBackendSetAttribute(d, HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, &isVirtual) == HIPDNN_STATUS_SUCCESS;
    ok = ok && hipdnnBackendFinalize(d) == HIPDNN_STATUS_SUCCESS;
    if(!ok)
    {
        hipdnnBackendDestroyDescriptor(d);
        return nullptr;
    }
    return d;
}

bool run_once(hipdnnHandle_t h)
{
    // NCHW contiguous dims/strides.
    const std::vector<int64_t> xDims{16, 16, 16, 16};
    const std::vector<int64_t> xStr{16 * 16 * 16, 16 * 16, 16, 1};
    const std::vector<int64_t> wDims{16, 16, 3, 3};
    const std::vector<int64_t> wStr{16 * 3 * 3, 3 * 3, 3, 1};
    const std::vector<int64_t> yDims{16, 16, 16, 16};
    const std::vector<int64_t> yStr{16 * 16 * 16, 16 * 16, 16, 1};
    const std::vector<int64_t> pads{1, 1};
    const std::vector<int64_t> fstrides{1, 1};
    const std::vector<int64_t> dils{1, 1};

    std::vector<hipdnnBackendDescriptor_t> retained;
    auto keep = [&](hipdnnBackendDescriptor_t d) {
        if(d != nullptr)
            retained.push_back(d);
    };
    auto cleanup = [&]() {
        for(auto d : retained)
            hipdnnBackendDestroyDescriptor(d);
    };

    const auto tBuild0 = std::chrono::steady_clock::now();

    hipdnnBackendDescriptor_t xDesc = make_tensor_desc(kUidX, xDims, xStr);
    hipdnnBackendDescriptor_t wDesc = make_tensor_desc(kUidW, wDims, wStr);
    hipdnnBackendDescriptor_t yDesc = make_tensor_desc(kUidY, yDims, yStr);
    keep(xDesc);
    keep(wDesc);
    keep(yDesc);
    if(xDesc == nullptr || wDesc == nullptr || yDesc == nullptr)
    {
        std::fprintf(stderr, "[repro] tensor descriptor build failed\n");
        cleanup();
        return false;
    }

    hipdnnBackendDescriptor_t convOp = nullptr;
    if(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &convOp) != HIPDNN_STATUS_SUCCESS)
    {
        cleanup();
        return false;
    }
    keep(convOp);
    hipdnnDataType_t compType    = HIPDNN_DATA_FLOAT;
    hipdnnConvolutionMode_t mode = HIPDNN_CROSS_CORRELATION;
    {
        bool ok = hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &mode) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, static_cast<int64_t>(dils.size()), dils.data()) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, static_cast<int64_t>(fstrides.size()), fstrides.data()) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, static_cast<int64_t>(pads.size()), pads.data()) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(convOp, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, static_cast<int64_t>(pads.size()), pads.data()) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendFinalize(convOp) == HIPDNN_STATUS_SUCCESS;
        if(!ok)
        {
            std::fprintf(stderr, "[repro] conv op attrs failed\n");
            cleanup();
            return false;
        }
    }

    hipdnnBackendDescriptor_t opGraph = nullptr;
    if(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph) != HIPDNN_STATUS_SUCCESS)
    {
        cleanup();
        return false;
    }
    keep(opGraph);
    {
        bool ok = hipdnnBackendSetAttribute(opGraph, HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convOp) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(opGraph, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &h) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendFinalize(opGraph) == HIPDNN_STATUS_SUCCESS;
        if(!ok)
        {
            std::fprintf(stderr, "[repro] op graph attrs failed\n");
            cleanup();
            return false;
        }
    }

    hipdnnBackendDescriptor_t heur = nullptr;
    if(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur) != HIPDNN_STATUS_SUCCESS)
    {
        cleanup();
        return false;
    }
    keep(heur);
    hipdnnBackendHeurMode_t heurMode = HIPDNN_HEUR_MODE_FALLBACK;
    {
        bool ok = hipdnnBackendSetAttribute(heur, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(heur, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &heurMode) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendFinalize(heur) == HIPDNN_STATUS_SUCCESS;
        if(!ok)
        {
            std::fprintf(stderr, "[repro] heur attrs failed\n");
            cleanup();
            return false;
        }
    }

    int64_t avail = 0;
    if(hipdnnBackendGetAttribute(heur, HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &avail, nullptr) != HIPDNN_STATUS_SUCCESS || avail == 0)
    {
        std::fprintf(stderr, "[repro] no engine configs available\n");
        cleanup();
        return false;
    }

    hipdnnBackendDescriptor_t engCfg = nullptr;
    if(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engCfg) != HIPDNN_STATUS_SUCCESS)
    {
        cleanup();
        return false;
    }
    keep(engCfg);
    int64_t got                       = 0;
    hipdnnBackendDescriptor_t shallow[1] = {engCfg};
    if(hipdnnBackendGetAttribute(heur, HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &got, shallow) != HIPDNN_STATUS_SUCCESS || got == 0)
    {
        std::fprintf(stderr, "[repro] get engine config failed\n");
        cleanup();
        return false;
    }
    if(hipdnnBackendFinalize(engCfg) != HIPDNN_STATUS_SUCCESS)
    {
        cleanup();
        return false;
    }

    hipdnnBackendDescriptor_t plan = nullptr;
    if(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan) != HIPDNN_STATUS_SUCCESS)
    {
        cleanup();
        return false;
    }
    keep(plan);
    if(hipdnnBackendSetAttribute(plan, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engCfg) != HIPDNN_STATUS_SUCCESS)
    {
        cleanup();
        return false;
    }
    if(hipdnnBackendFinalize(plan) != HIPDNN_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "[repro] finalize plan failed\n");
        cleanup();
        return false;
    }

    const long long buildNs = ns_since(tBuild0);
    std::printf("[repro] conv plan build: %lld ns  (%.3f s)\n", buildNs,
                static_cast<double>(buildNs) / 1e9);

    // Workspace + device buffers.
    int64_t wsCount = 0;
    int64_t wsSize  = 0;
    if(hipdnnBackendGetAttribute(plan, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &wsCount, &wsSize) != HIPDNN_STATUS_SUCCESS)
        wsSize = 0;
    void* ws = nullptr;
    if(wsSize > 0 && hipMalloc(&ws, static_cast<size_t>(wsSize)) != hipSuccess)
    {
        std::fprintf(stderr, "[repro] workspace alloc failed\n");
        cleanup();
        return false;
    }

    const size_t xBytes = 16ull * 16 * 16 * 16 * sizeof(float);
    const size_t wBytes = 16ull * 16 * 3 * 3 * sizeof(float);
    const size_t yBytes = 16ull * 16 * 16 * 16 * sizeof(float);
    void* xMem          = nullptr;
    void* wMem          = nullptr;
    void* yMem          = nullptr;
    if(hipMalloc(&xMem, xBytes) != hipSuccess || hipMalloc(&wMem, wBytes) != hipSuccess ||
       hipMalloc(&yMem, yBytes) != hipSuccess)
    {
        std::fprintf(stderr, "[repro] device alloc failed\n");
        cleanup();
        return false;
    }
    (void)hipMemset(xMem, 0, xBytes);
    (void)hipMemset(wMem, 0, wBytes);

    auto execute = [&](const char* tag) -> bool {
        hipdnnBackendDescriptor_t vp = nullptr;
        if(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vp) != HIPDNN_STATUS_SUCCESS)
            return false;
        int64_t uids[3] = {kUidX, kUidW, kUidY};
        void* ptrs[3]   = {xMem, wMem, yMem};
        bool ok = hipdnnBackendSetAttribute(vp, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, 3, uids) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(vp, HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS, HIPDNN_TYPE_VOID_PTR, 3, ptrs) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendSetAttribute(vp, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &ws) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendFinalize(vp) == HIPDNN_STATUS_SUCCESS;
        if(!ok)
        {
            hipdnnBackendDestroyDescriptor(vp);
            return false;
        }
        (void)hipDeviceSynchronize();
        const auto te0 = std::chrono::steady_clock::now();
        const hipdnnStatus_t es = hipdnnBackendExecute(h, plan, vp);
        (void)hipDeviceSynchronize();
        const long long execNs = ns_since(te0);
        hipdnnBackendDestroyDescriptor(vp);
        if(es != HIPDNN_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "[repro] execute failed: status=%d\n", static_cast<int>(es));
            return false;
        }
        std::printf("[repro] conv execute (%s): %lld ns\n", tag, execNs);
        return true;
    };

    bool ran = execute("cold");
    for(int i = 0; ran && i < 5; ++i)
        ran = execute("warm");

    if(ws != nullptr)
        (void)hipFree(ws);
    (void)hipFree(xMem);
    (void)hipFree(wMem);
    (void)hipFree(yMem);
    cleanup();
    return ran;
}

} // namespace

int main()
{
    hipdnnHandle_t h = nullptr;
    if(hipdnnCreate(&h) != HIPDNN_STATUS_SUCCESS || h == nullptr)
    {
        std::fprintf(stderr, "[repro] hipdnnCreate failed (is HIPDNN_PLUGIN_DIR set to the engines dir?)\n");
        return 2;
    }

    hipStream_t stream = nullptr;
    if(hipStreamCreate(&stream) == hipSuccess)
        (void)hipdnnSetStream(h, stream);

    const bool ok = run_once(h);

    if(stream != nullptr)
        (void)hipStreamDestroy(stream);
    (void)hipdnnDestroy(h);

    if(!ok)
    {
        std::fprintf(stderr, "[repro] FAILED\n");
        return 1;
    }
    std::printf("[repro] OK\n");
    return 0;
}
