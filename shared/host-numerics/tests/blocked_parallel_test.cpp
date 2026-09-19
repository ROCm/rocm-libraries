// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_blocked_backend_test_support.hpp"

namespace blocked_backend_test {
void testParallelSparseSelection() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 64;
    constexpr size_t reductionElements = 4096;
    constexpr size_t columns = 64;
    constexpr float sentinel = -99.0f;

    std::vector<float> a(rows * reductionElements);
    for (size_t row = 0; row < rows; ++row)
        std::fill_n(a.begin() + row * reductionElements, reductionElements,
                    static_cast<float>(row + 1));
    const std::vector<float> b(reductionElements * columns, 1.0f);
    Tensor output = makeOutput(rows, columns, sentinel);
    GemmTestCase problem = makeProblem(a, b, output, rows, reductionElements, columns);
    problem.outputSelection =
        OutputSelection::primeStride(output.elementCount(), output.elementCount(), 128);

    const GemmTestRunInfo run = referenceGemm(problem, GemmBackend::Blocked);

    const std::vector<size_t> selected = problem.outputSelection.indices(output.elementCount());
    require(run.outputElementsWritten == selected.size() &&
                run.outputElementsCovered == selected.size(),
            "Sparse blocked GEMM did not report its directly computed outputs.");
    for (const size_t linearIndex : selected) {
        const size_t row = linearIndex / columns;
        const size_t column = linearIndex % columns;
        require(output.loadAs<float>({row, column}) ==
                    static_cast<float>((row + 1) * reductionElements),
                "Parallel blocked GEMM produced an incorrect selected output.");
    }
    require(output.loadAs<float>({0, 1}) == sentinel,
            "Parallel blocked GEMM changed an unselected output.");
}

}  // namespace blocked_backend_test
