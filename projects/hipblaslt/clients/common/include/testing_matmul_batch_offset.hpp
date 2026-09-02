/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "hipblaslt_test.hpp"
#include "host_vector.hpp"
#include "utility.hpp"
#include <hipblaslt/host_numerics/Types.hpp>
#include <roc/host_numerics/validation.hpp>
#include <limits>
#include <stdexcept>

namespace
{
    struct OffsetMatrixPlan
    {
        size_t matrixElements;
        size_t padding;
        size_t allocationElements;
        int64_t offset;

        ptrdiff_t logicalStart() const
        {
            return offset < 0 ? 0 : static_cast<ptrdiff_t>(offset);
        }
    };

    inline OffsetMatrixPlan offsetMatrixPlan(size_t matrixElements, int64_t offset)
    {
        const uint64_t magnitude = offset < 0 ? uint64_t(-(offset + 1)) + 1 : uint64_t(offset);
        if(magnitude > uint64_t(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("Batch offset exceeds ptrdiff_t.");

        const size_t offsetElements = static_cast<size_t>(magnitude);
        const size_t padding        = offset < 0 ? offsetElements : 0;
        const size_t trailing       = offset > 0 ? offsetElements : 0;
        if(matrixElements > size_t(std::numeric_limits<ptrdiff_t>::max()) - padding
           || matrixElements + padding
                  > size_t(std::numeric_limits<ptrdiff_t>::max()) - trailing)
            throw std::overflow_error("Batch-offset allocation size exceeds ptrdiff_t.");

        return {
            matrixElements,
            padding,
            padding + matrixElements + trailing,
            offset,
        };
    }
}

template <typename Ti, typename To, typename Tc>
void testing_matmul_batch_offset_impl(const Arguments& arg)
{
    if(arg.batch_mode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
        GTEST_SKIP() << "Batch offset requires pointer-array batching";

    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    const hipblasOperation_t transA = char_to_hipblas_operation(arg.transA);
    const hipblasOperation_t transB = char_to_hipblas_operation(arg.transB);
    const int64_t M = arg.M[0], N = arg.N[0], K = arg.K[0];
    const int64_t lda = arg.lda[0], ldb = arg.ldb[0], ldc = arg.ldc[0], ldd = arg.ldd[0];
    const int32_t batchCount = arg.batch_count;
    const int64_t aRows = transA == HIPBLAS_OP_N ? M : K;
    const int64_t aColumns = transA == HIPBLAS_OP_N ? K : M;
    const int64_t bRows = transB == HIPBLAS_OP_N ? K : N;
    const int64_t bColumns = transB == HIPBLAS_OP_N ? N : K;

    const auto aPlan = offsetMatrixPlan(size_t(lda) * aColumns, arg.batch_offset_a);
    const auto bPlan = offsetMatrixPlan(size_t(ldb) * bColumns, arg.batch_offset_b);
    const auto cPlan = offsetMatrixPlan(size_t(ldc) * N, arg.batch_offset_c);
    const auto dPlan = offsetMatrixPlan(size_t(ldd) * N, arg.batch_offset_d);

    host_vector<Ti> hostA(aPlan.allocationElements * batchCount);
    host_vector<Ti> hostB(bPlan.allocationElements * batchCount);
    host_vector<To> hostC(cPlan.allocationElements * batchCount);
    host_vector<To> observedD(dPlan.allocationElements * batchCount);
    host_vector<To> expectedD(dPlan.matrixElements * batchCount);

    auto generate = [&](auto&                   storage,
                        size_t                  rows,
                        size_t                  columns,
                        int64_t                 leadingDimension,
                        const OffsetMatrixPlan& plan,
                        const GenerationRecipe& recipe) {
        auto tensor = copyTensorFromEncodedStorage(
            storage.data(),
            storage.size(),
            Layout(Shape{rows, columns, size_t(batchCount)},
                   {1, leadingDimension, ptrdiff_t(plan.allocationElements)},
                   plan.logicalStart()));
        roc::host_numerics::generate(tensor, recipe);
        copyTensorEncodedBackingStorageToBuffer(storage.data(), storage.size(), tensor);
    };

    generate(hostA,
             aRows,
             aColumns,
             lda,
             aPlan,
             GenerationRecipe::realOnly(
                 GenerationRecipe::affineIndexRemainder(
                     {.dimensionCoefficients = {1, 1, 1}, .positiveDivisor = 7})
                     .withAffineValueMapping({.offset = 1})));
    generate(hostB,
             bRows,
             bColumns,
             ldb,
             bPlan,
             GenerationRecipe::realOnly(
                 GenerationRecipe::affineIndexRemainder(
                     {.dimensionCoefficients = {1, -1, 1}, .positiveDivisor = 5})
                     .withAffineValueMapping({.offset = 1})));
    generate(hostC,
             M,
             N,
             ldc,
             cPlan,
             GenerationRecipe::realOnly(GenerationRecipe::affineIndexRemainder(
                 {.dimensionCoefficients = {1, 1, 0}, .positiveDivisor = 3})));

    device_vector<Ti> deviceA(aPlan.allocationElements * batchCount);
    device_vector<Ti> deviceB(bPlan.allocationElements * batchCount);
    device_vector<To> deviceC(cPlan.allocationElements * batchCount);
    device_vector<To> deviceD(dPlan.allocationElements * batchCount);
    CHECK_HIP_ERROR(hipMemcpy(deviceA,
                              hostA.data(),
                              sizeof(Ti) * hostA.size(),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(deviceB,
                              hostB.data(),
                              sizeof(Ti) * hostB.size(),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(deviceC,
                              hostC.data(),
                              sizeof(To) * hostC.size(),
                              hipMemcpyHostToDevice));

    host_vector<uint64_t> pointersA(batchCount);
    host_vector<uint64_t> pointersB(batchCount);
    host_vector<uint64_t> pointersC(batchCount);
    host_vector<uint64_t> pointersD(batchCount);
    for(int32_t batch = 0; batch < batchCount; ++batch)
    {
        pointersA[batch] = reinterpret_cast<uint64_t>(
            static_cast<Ti*>(deviceA) + batch * aPlan.allocationElements + aPlan.padding);
        pointersB[batch] = reinterpret_cast<uint64_t>(
            static_cast<Ti*>(deviceB) + batch * bPlan.allocationElements + bPlan.padding);
        pointersC[batch] = reinterpret_cast<uint64_t>(
            static_cast<To*>(deviceC) + batch * cPlan.allocationElements + cPlan.padding);
        pointersD[batch] = reinterpret_cast<uint64_t>(
            static_cast<To*>(deviceD) + batch * dPlan.allocationElements + dPlan.padding);
    }
    HipDeviceBuffer devicePointersA(HIP_R_64U, batchCount);
    HipDeviceBuffer devicePointersB(HIP_R_64U, batchCount);
    HipDeviceBuffer devicePointersC(HIP_R_64U, batchCount);
    HipDeviceBuffer devicePointersD(HIP_R_64U, batchCount);
    CHECK_HIP_ERROR(hipMemcpy(devicePointersA.buf(),
                              pointersA.data(),
                              sizeof(uint64_t) * batchCount,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(devicePointersB.buf(),
                              pointersB.data(),
                              sizeof(uint64_t) * batchCount,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(devicePointersC.buf(),
                              pointersC.data(),
                              sizeof(uint64_t) * batchCount,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(devicePointersD.buf(),
                              pointersD.data(),
                              sizeof(uint64_t) * batchCount,
                              hipMemcpyHostToDevice));

    hipblaslt_local_handle handle;
    hipblaslt_local_matmul_descr matmul(transA, transB, arg.compute_type, arg.scale_type);
    hipblaslt_local_matrix_layout matA(aRows, aColumns, lda, arg.a_type);
    hipblaslt_local_matrix_layout matB(bRows, bColumns, ldb, arg.b_type);
    hipblaslt_local_matrix_layout matC(M, N, ldc, arg.c_type);
    hipblaslt_local_matrix_layout matD(M, N, ldd, arg.d_type);
    for(auto* layout : {static_cast<hipblasLtMatrixLayout_t>(matA),
                        static_cast<hipblasLtMatrixLayout_t>(matB),
                        static_cast<hipblasLtMatrixLayout_t>(matC),
                        static_cast<hipblasLtMatrixLayout_t>(matD)})
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
        const int32_t mode = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode)));
    }
    for(const auto [layout, offset] :
        {std::pair<hipblasLtMatrixLayout_t, int64_t>{matA, aPlan.offset},
         {matB, bPlan.offset},
         {matC, cPlan.offset},
         {matD, dPlan.offset}})
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            layout, HIPBLASLT_MATRIX_LAYOUT_OFFSET, &offset, sizeof(offset)));
    }

    hipblaslt_local_preference preference;
    const size_t workspaceLimit = 128 * 1024 * 1024;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceLimit,
        sizeof(workspaceLimit)));
    const int32_t requested
        = arg.requested_solution_num < 0 ? HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM
                                        : std::max(1, arg.requested_solution_num);
    std::vector<hipblasLtMatmulHeuristicResult_t> algorithms(requested);
    int32_t algorithmCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                           matmul,
                                                           matA,
                                                           matB,
                                                           matC,
                                                           matD,
                                                           preference,
                                                           requested,
                                                           algorithms.data(),
                                                           &algorithmCount));
    if(algorithmCount == 0)
        GTEST_SKIP() << "No algorithm found for this configuration";
    algorithms.resize(algorithmCount);
    size_t workspaceBytes = 0;
    for(const auto& algorithm : algorithms)
        workspaceBytes = std::max(workspaceBytes, algorithm.workspaceSize);
    device_vector<unsigned char> workspace(workspaceBytes);

    const Tc alpha = arg.get_alpha<Tc>();
    const Tc beta  = arg.get_beta<Tc>();
    for(int32_t batch = 0; batch < batchCount; ++batch)
    {
        const ptrdiff_t aRowStride = transA == HIPBLAS_OP_N ? 1 : lda;
        const ptrdiff_t aColumnStride = transA == HIPBLAS_OP_N ? lda : 1;
        const ptrdiff_t bRowStride = transB == HIPBLAS_OP_N ? 1 : ldb;
        const ptrdiff_t bColumnStride = transB == HIPBLAS_OP_N ? ldb : 1;
        auto result = copyTensorFromEncodedStorage(expectedD.data() + batch * dPlan.matrixElements,
                                                   dPlan.matrixElements,
                                                   Layout(Shape{size_t(M), size_t(N)}, {1, ldd}));
        GemmOperand operandA(copyTensorFromEncodedStorage(
            hostA.data() + batch * aPlan.allocationElements + aPlan.logicalStart(),
            aPlan.matrixElements,
            Layout(Shape{size_t(M), size_t(K)}, {aRowStride, aColumnStride})));
        GemmOperand operandB(copyTensorFromEncodedStorage(
            hostB.data() + batch * bPlan.allocationElements + bPlan.logicalStart(),
            bPlan.matrixElements,
            Layout(Shape{size_t(K), size_t(N)}, {bRowStride, bColumnStride})));
        Tensor      c = copyTensorFromEncodedStorage(hostC.data() + batch * cPlan.allocationElements
                                                         + cPlan.logicalStart(),
                                                     cPlan.matrixElements,
                                                     Layout(Shape{size_t(M), size_t(N)}, {1, ldc}));
        GemmOptions options(scalarType<Tc>());
        options.epilogue.alpha = static_cast<double>(alpha);
        options.epilogue.beta  = static_cast<double>(beta);
        result.copyLogicalElementsFrom(referenceGemm(
            std::move(operandA), std::move(operandB), std::move(c), result.type(), options));
        copyTensorEncodedBackingStorageToBuffer(
            expectedD.data() + batch * dPlan.matrixElements, dPlan.matrixElements, result);
    }

    const double tolerance = std::numeric_limits<Tc>::epsilon() * 100 * K;
    int passed = 0, failed = 0;
    for(int32_t algorithm = 0; algorithm < algorithmCount; ++algorithm)
    {
        CHECK_HIP_ERROR(hipMemcpy(deviceD,
                                  observedD.data(),
                                  sizeof(To) * observedD.size(),
                                  hipMemcpyHostToDevice));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                              matmul,
                                              &alpha,
                                              devicePointersA.buf(),
                                              matA,
                                              devicePointersB.buf(),
                                              matB,
                                              &beta,
                                              devicePointersC.buf(),
                                              matC,
                                              devicePointersD.buf(),
                                              matD,
                                              &algorithms[algorithm].algo,
                                              workspace,
                                              algorithms[algorithm].workspaceSize,
                                              nullptr));
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        CHECK_HIP_ERROR(hipMemcpy(observedD.data(),
                                  deviceD,
                                  sizeof(To) * observedD.size(),
                                  hipMemcpyDeviceToHost));

        bool matched = true;
        double maximumDifference = 0.0;
        for(int32_t batch = 0; batch < batchCount; ++batch)
        {
            const To* observed = observedD.data() + batch * dPlan.allocationElements
                                 + dPlan.logicalStart();
            const To* expected = expectedD.data() + batch * dPlan.matrixElements;
            const Layout layout(Shape{size_t(M), size_t(N)}, {1, ldd});
            ComparisonOptions options{
                .absoluteTolerance = std::nextafter(tolerance, 0.0),
                .maxReportedMismatches = 0,
            };
            options.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
            const auto comparison
                = compare(copyTensorFromEncodedStorage(observed, dPlan.matrixElements, layout),
                          copyTensorFromEncodedStorage(expected, dPlan.matrixElements, layout),
                          options);
            matched &= comparison.passed();
            maximumDifference
                = std::max(maximumDifference, comparison.maxAbsoluteDifference);
        }
        if(arg.unit_check && !matched)
        {
            ++failed;
            EXPECT_LT(maximumDifference, tolerance)
                << "Solution " << algorithm << "/" << algorithmCount << " failed";
        }
        else if(arg.unit_check)
        {
            ++passed;
        }
    }
    if(algorithmCount > 1 && arg.unit_check)
        hipblaslt_cout << "Tested " << algorithmCount << " solutions: " << passed << " passed, "
                       << failed << " failed" << std::endl;
}

inline void testing_matmul_batch_offset(const Arguments& arg)
{
    if(arg.a_type == HIP_R_32F && arg.b_type == HIP_R_32F && arg.c_type == HIP_R_32F
       && arg.d_type == HIP_R_32F)
        testing_matmul_batch_offset_impl<float, float, float>(arg);
    else if(arg.a_type == HIP_R_16F && arg.b_type == HIP_R_16F && arg.c_type == HIP_R_16F
            && arg.d_type == HIP_R_16F)
        testing_matmul_batch_offset_impl<hipblasLtHalf, hipblasLtHalf, float>(arg);
    else if(arg.a_type == HIP_R_16BF && arg.b_type == HIP_R_16BF && arg.c_type == HIP_R_16BF
            && arg.d_type == HIP_R_16BF)
        testing_matmul_batch_offset_impl<hip_bfloat16, hip_bfloat16, float>(arg);
    else
        GTEST_SKIP() << "Unsupported type combination for batch-offset test";
}
