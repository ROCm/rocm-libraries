#include "hipdnn_backend.h"
#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include <cstdlib>
#include <iostream>

#define CHECK_HIPDNN(expr)                                                                          \
    do                                                                                              \
    {                                                                                               \
        auto status = (expr);                                                                       \
        if(status != HIPDNN_STATUS_SUCCESS)                                                         \
        {                                                                                           \
            std::cerr << #expr << " failed with status " << static_cast<int>(status) << "\n";       \
            return 1;                                                                               \
        }                                                                                           \
    } while(false)

#define CHECK_HIP(expr)                                                                             \
    do                                                                                              \
    {                                                                                               \
        auto status = (expr);                                                                       \
        if(status != hipSuccess)                                                                    \
        {                                                                                           \
            std::cerr << #expr << " failed with status " << static_cast<int>(status) << "\n";       \
            return 1;                                                                               \
        }                                                                                           \
    } while(false)

int main()
{
    const char* enginePath
        = "/cluster/apps/ubuntu-24/rocm/rocm-7.12.0.60610/lib/hipdnn_plugins/engines/"
          "libmiopen_plugin.so";
    const char* heuristicPath
        = "/home/AMD/ysoliman/rocm-libraries-heuristic-prototype/projects/hipdnn/build/lib/"
          "hipdnn_plugins/heuristics/libconv_heuristic.so";
    const char* policyOrder = "ConvHeuristic::RegimeClassifier,SelectionHeuristic::StaticOrdering";

    setenv("HIPDNN_HEUR_POLICY_ORDER", policyOrder, 1);
    CHECK_HIPDNN(hipdnnSetEnginePluginPaths_ext(1, &enginePath, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
    CHECK_HIPDNN(
        hipdnnSetHeuristicPluginPaths_ext(1, &heuristicPath, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    hipdnnHandle_t handle = nullptr;
    hipStream_t stream = nullptr;
    hipdnnBackendDescriptor_t graph = nullptr;
    hipdnnBackendDescriptor_t heuristic = nullptr;

    CHECK_HIPDNN(hipdnnCreate(&handle));
    CHECK_HIP(hipStreamCreate(&stream));
    CHECK_HIPDNN(hipdnnSetStream(handle, stream));

    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        {32, 64, 56, 56},
        {200704, 3136, 56, 1},
        {64, 64, 3, 3},
        {576, 9, 3, 1},
        {32, 64, 56, 56},
        {200704, 3136, 56, 1},
        {1, 1},
        {1, 1},
        {1, 1});
    auto buffer = builder.Release();
    CHECK_HIPDNN(hipdnnBackendCreateAndDeserializeGraph_ext(&graph, buffer.data(), buffer.size()));
    CHECK_HIPDNN(hipdnnBackendSetAttribute(
        graph, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle));
    CHECK_HIPDNN(hipdnnBackendFinalize(graph));

    CHECK_HIPDNN(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristic));
    CHECK_HIPDNN(hipdnnBackendSetAttribute(heuristic,
                                           HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           static_cast<const void*>(&graph)));
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
    CHECK_HIPDNN(hipdnnBackendSetAttribute(
        heuristic, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));

    CHECK_HIPDNN(hipdnnBackendFinalize(heuristic));
    std::cout << "verify_conv_miopen: finalize success\n";

    CHECK_HIPDNN(hipdnnBackendDestroyDescriptor(heuristic));
    CHECK_HIPDNN(hipdnnBackendDestroyDescriptor(graph));
    CHECK_HIPDNN(hipdnnDestroy(handle));
    CHECK_HIP(hipStreamDestroy(stream));
    return 0;
}
