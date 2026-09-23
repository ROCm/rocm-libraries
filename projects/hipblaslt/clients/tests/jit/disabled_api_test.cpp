// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <cstring>
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>
#include <hipblaslt/hipblaslt-jit.hpp>
#include <iostream>

int main()
{
    namespace jit = hipblaslt_ext::experimental::jit;
    jit::Backend                     backend;
    jit::Request                     request;
    jit::Solution                    solution;
    jit::Diagnostics                 diagnostics{"stale", "stale"};
    hipblasLtMatmulHeuristicResult_t result;
    std::memset(&result, 0xa5, sizeof(result));
    const auto unsupported = HIPBLAS_STATUS_NOT_SUPPORTED;
    if(jit::tensilelite::createBackend({}, backend, diagnostics) != unsupported
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
                               diagnostics)
              != unsupported
       || jit::getJitAlgo(0, request, backend, 0, solution, diagnostics) != unsupported
       || jit::getGemmAlgo(solution, result, diagnostics) != unsupported
       || result.state != unsupported || result.workspaceSize != 0)
        return 1;
    hipblasLtMatmulAlgo_t empty{};
    if(std::memcmp(&result.algo, &empty, sizeof(empty)) != 0)
        return 1;
    std::cout << "PASS disabled public headers, factories, selection and cleared result\n";
    return 0;
}
