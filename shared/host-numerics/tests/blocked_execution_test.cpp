// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_blocked_backend_test_support.hpp"

namespace blocked_backend_test {
void testFullSelection() {
    constexpr size_t rows = 35;
    constexpr size_t reductionElements = 17;
    constexpr size_t columns = 34;

    const std::vector<float> a = makeValues(rows, reductionElements, 11);
    const std::vector<float> b = makeValues(reductionElements, columns, 12);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);

    GemmTestCase blockedProblem =
        makeProblem(a, b, blockedOutput, rows, reductionElements, columns);
    const GemmTestRunInfo run =
        runAndCheck(blockedProblem, blockedOutput, "Full blocked selection result mismatch.");
    require(
        run.outputElementsWritten == rows * columns && run.outputElementsCovered == rows * columns,
        "Full blocked selection did not preserve complete-output accounting.");
}

void testAutomaticSelectionUsesBlockedBackend() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 32;
    constexpr size_t reductionElements = 8;
    constexpr size_t columns = 32;
    const std::vector<float> a = makeValues(rows, reductionElements, 18);
    const std::vector<float> b = makeValues(reductionElements, columns, 19);
    Tensor output = makeOutput(rows, columns, untouchedValue);
    GemmTestCase problem = makeProblem(a, b, output, rows, reductionElements, columns);

    const GemmSupportInfo fullSupport = queryGemmSupport(problem, GemmBackend::Blocked);
    require(fullSupport.supported, "Blocked backend rejected dense work.");
    const GemmTestRunInfo full = referenceGemm(problem);
    require(full.backendUsed == GemmBackend::Blocked && !full.fallbackReason,
            "Automatic GEMM did not select Blocked for dense reusable work.");
    const float firstOutput = output.loadAs<float>({0, 0});

    fillTensor(output, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({0});
    const GemmSupportInfo sparseSupport = queryGemmSupport(problem, GemmBackend::Blocked);
    require(sparseSupport.supported, "Blocked backend rejected sparse work.");
    const GemmTestRunInfo sparse = referenceGemm(problem);
    require(sparse.backendUsed == GemmBackend::Blocked && !sparse.fallbackReason,
            "Automatic GEMM did not use Blocked for sparse work.");
    require(sparse.outputElementsWritten == 1 && sparse.outputElementsCovered == 1,
            "Sparse blocked GEMM reported the wrong write or coverage count.");
    require(output.loadAs<float>({0, 0}) == firstOutput,
            "Automatic backend selection changed the selected numerical result.");
    requireOnlySelectedOutputsStored(output, problem.outputSelection);
}

void testParallelFullSelection() {
    constexpr size_t rows = 128;
    constexpr size_t reductionElements = 128;
    constexpr size_t columns = 128;

    const std::vector<float> a = makeValues(rows, reductionElements, 15);
    const std::vector<float> b = makeValues(reductionElements, columns, 16);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);

    GemmTestCase blockedProblem =
        makeProblem(a, b, blockedOutput, rows, reductionElements, columns);

    const GemmTestRunInfo run =
        runAndCheck(blockedProblem, blockedOutput, "Parallel blocked GEMM result mismatch.");
    require(
        run.outputElementsWritten == rows * columns && run.outputElementsCovered == rows * columns,
        "Parallel blocked GEMM reported the wrong output counts.");
}

}  // namespace blocked_backend_test
