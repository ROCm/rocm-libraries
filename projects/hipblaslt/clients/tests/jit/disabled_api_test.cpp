// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <cstring>
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>
#include <hipblaslt/hipblaslt-jit.hpp>
#include <iostream>

int main()
{
    namespace tl = hipblaslt_ext::experimental::jit::tensilelite;
    tl::Diagnostics                  diagnostics{"stale"};
    hipblasLtMatmulHeuristicResult_t result;
    std::memset(&result, 0xa5, sizeof(result));
    const auto unsupported = HIPBLAS_STATUS_NOT_SUPPORTED;
    if(tl::getGemmAlgo(nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       {},
                       0,
                       result,
                       diagnostics)
           != unsupported
       || result.state != unsupported || result.workspaceSize != 0)
        return 1;
    hipblasLtMatmulAlgo_t empty{};
    if(std::memcmp(&result.algo, &empty, sizeof(empty)) != 0 || diagnostics.message.empty())
        return 1;
    namespace jit = hipblaslt_ext::experimental::jit;
    jit::Backend     backend;
    jit::Request     request;
    jit::Solution    solution;
    jit::Diagnostics genericDiagnostics{"stale", "stale"};
    std::memset(&result, 0xa5, sizeof(result));
    if(tl::createBackend({}, backend, genericDiagnostics) != unsupported
       || jit::makeGemmRequest(nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               request,
                               genericDiagnostics)
              != unsupported
       || jit::getJitAlgo(0, request, backend, 0, solution, genericDiagnostics) != unsupported
       || jit::getGemmAlgo(solution, result, genericDiagnostics) != unsupported
       || result.state != unsupported || result.workspaceSize != 0
       || std::memcmp(&result.algo, &empty, sizeof(empty)) != 0
       || genericDiagnostics.message.empty())
        return 1;
    std::cout << "PASS disabled direct and generic APIs and cleared results\n";
    return 0;
}
