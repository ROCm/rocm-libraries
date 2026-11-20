// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Asks hipBLASLt whether a pinned MX FP6 solution accepts a problem, without building any data,
// so problems hipblaslt-test cannot build (a batch above 1, padded lda or ldb) still reach the
// library's predicate check. refusals.py builds and drives it.
//
// Reads one problem per line on stdin:
//   label solution_index a_type tiled_b m n k batch lda ldb ldd alpha beta
// with a_type f6_r or bf6_r and tiled_b 0 or 1 (B in HIPBLASLT_ORDER_COL256_R128_EXT), and
// prints one line per problem:
//   RESULT label accepted|refused stage status solution_name
// where stage is the call that decided: setProblem, getAlgosFromIndex or isAlgoSupported.

#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    const char* statusName(hipblasStatus_t status)
    {
        switch(status)
        {
        case HIPBLAS_STATUS_SUCCESS:
            return "SUCCESS";
        case HIPBLAS_STATUS_INVALID_VALUE:
            return "INVALID_VALUE";
        case HIPBLAS_STATUS_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        default:
            return "OTHER";
        }
    }

    struct Problem
    {
        std::string label;
        int         index = -1;
        std::string type;
        int         tiledB = 1;
        int64_t     m = 0, n = 0, k = 0, batch = 1, lda = 0, ldb = 0, ldd = 0;
        float       alpha = 1.0f, beta = 0.0f;
    };

    void query(hipblasLtHandle_t handle, void* storage, const Problem& p)
    {
        const hipDataType type = p.type == "bf6_r" ? HIP_R_6F_E3M2 : HIP_R_6F_E2M3;
        const auto        report
            = [&](const char* verdict, const char* stage, hipblasStatus_t status, const std::string& name) {
                  std::cout << "RESULT " << p.label << " " << verdict << " " << stage << " "
                            << statusName(status) << " " << (name.empty() ? "-" : name)
                            << std::endl;
              };

        hipblaslt_ext::GemmProblemType problemType(
            HIPBLAS_OP_T, HIPBLAS_OP_N, type, type, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
        problemType.setOrderA(HIPBLASLT_ORDER_COL256_R128_EXT);
        if(p.tiledB)
            problemType.setOrderB(HIPBLASLT_ORDER_COL256_R128_EXT);
        hipblaslt_ext::GemmEpilogue epilogue;
        epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT);
        epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT);
        hipblaslt_ext::GemmInputs inputs;
        inputs.setA(storage);
        inputs.setB(storage);
        inputs.setC(storage);
        inputs.setD(storage);
        inputs.setAlpha(&p.alpha);
        inputs.setBeta(&p.beta);
        inputs.setScaleA(storage);
        inputs.setScaleB(storage);

        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N, type, type, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
        hipblasStatus_t status = gemm.setProblem(p.m,
                                                 p.n,
                                                 p.k,
                                                 p.batch,
                                                 p.lda,
                                                 p.ldb,
                                                 p.ldd,
                                                 p.ldd,
                                                 p.lda * p.m,
                                                 p.ldb * p.n,
                                                 p.ldd * p.n,
                                                 p.ldd * p.n,
                                                 epilogue,
                                                 inputs,
                                                 problemType);
        if(status != HIPBLAS_STATUS_SUCCESS)
            return report("refused", "setProblem", status, "");

        std::vector<int>                              indices{p.index};
        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        status = hipblaslt_ext::getAlgosFromIndex(handle, indices, algos);
        if(status != HIPBLAS_STATUS_SUCCESS || algos.empty())
            return report("refused", "getAlgosFromIndex", status, "");
        const std::string name = hipblaslt_ext::getSolutionNameFromAlgo(handle, algos[0].algo);

        size_t workspace = 0;
        status           = gemm.isAlgoSupported(algos[0].algo, workspace);
        report(status == HIPBLAS_STATUS_SUCCESS ? "accepted" : "refused",
               "isAlgoSupported",
               status,
               name);
    }
}

int main()
{
    void*             storage = nullptr;
    hipblasLtHandle_t handle  = nullptr;
    if(hipMalloc(&storage, 256) != hipSuccess || hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
    {
        std::cerr << "error: no GPU or no hipBLASLt handle" << std::endl;
        return 3;
    }
    std::string line;
    while(std::getline(std::cin, line))
    {
        std::istringstream in(line);
        Problem            p;
        if(!(in >> p.label >> p.index >> p.type >> p.tiledB >> p.m >> p.n >> p.k >> p.batch >> p.lda
             >> p.ldb >> p.ldd >> p.alpha >> p.beta))
        {
            if(!line.empty())
                std::cerr << "error: cannot parse: " << line << std::endl;
            continue;
        }
        query(handle, storage, p);
    }
    static_cast<void>(hipblasLtDestroy(handle));
    static_cast<void>(hipFree(storage));
    return 0;
}
