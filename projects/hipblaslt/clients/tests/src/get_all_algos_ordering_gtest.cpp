// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for the solution-enumeration order of getAllSolutions().
//
// getAllSolutions() (library/src/amd_detail/rocblaslt/src/tensile_host.cpp)
// collects candidates into a std::set<std::shared_ptr<ContractionSolution>>.
// That set orders by raw pointer value, i.e. by heap address, which ASLR
// randomizes per process, so the enumeration order varied from run to run for
// identical input. Callers consume the list positionally -- the fallback in
// rocblaslt_matmul_algo_get_heuristic() takes the first entries that pass
// isSolutionSupported() -- so a different kernel could win on each run.
//
// The fix orders the enumeration by the library-assigned solution index, which
// is deserialized from the library data and is stable across runs. This test
// pins that invariant: because getAllSolutions() also drops repeated indices,
// the indices it reports must come back STRICTLY INCREASING.
//
// On the buggy build the returned order is the heap-address order, which is
// unrelated to the index order, so the check below fails.
//
// This asserts ordering only. It deliberately does not assert WHICH kernel wins
// for a given problem: that is a property of the library content, not of this
// code path.

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <string>
#include <vector>

namespace
{
    inline bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    struct GemmTypeCombo
    {
        const char*          name;
        hipblasOperation_t   opA;
        hipblasOperation_t   opB;
        hipDataType          abType;
        hipblasComputeType_t computeType;
    };

    // A fixture rather than bare TEST() only so the checker can reach
    // RecordProperty, which gtest exposes to Test subclasses alone.
    class GetAllAlgosOrdering_pre_checkin : public ::testing::Test
    {
    protected:
        void checkOrdering(const GemmTypeCombo& combo)
        {
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available for hipblasLt handle";

            hipblasLtHandle_t handle = nullptr;
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

            std::vector<hipblasLtMatmulHeuristicResult_t> algos;
            const auto status = hipblaslt_ext::getAllAlgos(handle,
                                                           hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                           combo.opA,
                                                           combo.opB,
                                                           combo.abType,
                                                           combo.abType,
                                                           HIP_R_32F,
                                                           HIP_R_32F,
                                                           combo.computeType,
                                                           algos);

            RecordProperty("combo", combo.name);
            RecordProperty("enumerated", static_cast<int>(algos.size()));

            // An arch that has no kernels of this type at all is not a failure
            // of the ordering contract; there is simply nothing to order.
            if(status != HIPBLAS_STATUS_SUCCESS || algos.size() < 2)
            {
                static_cast<void>(hipblasLtDestroy(handle));
                GTEST_SKIP() << "getAllAlgos returned " << algos.size()
                             << " solutions for " << combo.name
                             << " (status " << status << "); nothing to order";
            }

            int         previous   = hipblaslt_ext::getIndexFromAlgo(algos[0].algo);
            int         inversions = 0;
            std::string firstInversion;
            for(size_t i = 1; i < algos.size(); ++i)
            {
                const int current = hipblaslt_ext::getIndexFromAlgo(algos[i].algo);
                if(current <= previous)
                {
                    if(inversions == 0)
                        firstInversion = "at position " + std::to_string(i) + ": index "
                                         + std::to_string(current) + " follows index "
                                         + std::to_string(previous);
                    ++inversions;
                }
                previous = current;
            }

            static_cast<void>(hipblasLtDestroy(handle));

            RecordProperty("inversions", inversions);
            EXPECT_EQ(inversions, 0)
                << "getAllAlgos() must report solutions in strictly increasing solution-index "
                   "order for "
                << combo.name << ", but " << inversions << " of " << (algos.size() - 1)
                << " adjacent pairs are out of order (first inversion " << firstInversion
                << "). An order that is not index order is the heap-address order of the "
                   "underlying std::set<std::shared_ptr<>>, which ASLR varies per run.";
        }
    };

    TEST_F(GetAllAlgosOrdering_pre_checkin, Fp32NN)
    {
        checkOrdering({"f32_r NN compute_32f",
                       HIPBLAS_OP_N,
                       HIPBLAS_OP_N,
                       HIP_R_32F,
                       HIPBLAS_COMPUTE_32F});
    }

    TEST_F(GetAllAlgosOrdering_pre_checkin, Fp16NN)
    {
        checkOrdering({"f16_r NN compute_32f",
                       HIPBLAS_OP_N,
                       HIPBLAS_OP_N,
                       HIP_R_16F,
                       HIPBLAS_COMPUTE_32F});
    }

    TEST_F(GetAllAlgosOrdering_pre_checkin, Bf16TN)
    {
        checkOrdering({"bf16_r TN compute_32f",
                       HIPBLAS_OP_T,
                       HIPBLAS_OP_N,
                       HIP_R_16BF,
                       HIPBLAS_COMPUTE_32F});
    }
}
