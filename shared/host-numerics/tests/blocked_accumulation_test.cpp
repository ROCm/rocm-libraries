// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_blocked_backend_test_support.hpp"

namespace blocked_backend_test {
void testReducedPrecisionAccumulators() {
    using namespace roc::host_numerics;

    constexpr size_t reductions = 64;
    const std::vector<float> values(reductions, 0.1f);
    for (const ScalarType type : {ScalarType::Float16, ScalarType::BFloat16}) {
        Tensor a = Tensor::copyValuesWithConversion(type, Shape{1, reductions},
                                                    std::span<const float>(values));
        Tensor b = Tensor::copyValuesWithConversion(type, Shape{reductions, 1},
                                                    std::span<const float>(values));
        Tensor d(type, Shape{1, 1});
        GemmTestCase problem(a, b, d, type);

        float expected = 0.0f;
        float fullPrecision = 0.0f;
        for (size_t reduction = 0; reduction < reductions; ++reduction) {
            const float product = roundThrough(
                type, a.loadAs<float>({0, reduction}) * b.loadAs<float>({reduction, 0}));
            expected = roundThrough(type, expected + product);
            fullPrecision += a.loadAs<float>({0, reduction}) * b.loadAs<float>({reduction, 0});
        }

        const GemmTestRunInfo run = referenceGemm(problem, GemmBackend::Blocked);
        require(run.backendUsed == GemmBackend::Blocked && d.loadAs<float>({0, 0}) == expected,
                "Blocked GEMM did not preserve reduced-precision accumulation.");

        problem.accumulationRounding = AccumulationRounding::FullPrecision;
        referenceGemm(problem, GemmBackend::Blocked);
        require(d.loadAs<float>({0, 0}) == roundThrough(type, fullPrecision) &&
                    d.loadAs<float>({0, 0}) != expected,
                "Blocked GEMM did not honor full-precision accumulation.");
    }
}

void testSelectedBlockAccumulatorFamilies() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 32;
    constexpr size_t columns = 32;
    constexpr size_t reductions = 64;
    const OutputSelection selection = OutputSelection::explicitIndices({0});

    const std::vector<float> reducedA(rows * reductions, 0.1f);
    const std::vector<float> reducedB(reductions * columns, 0.1f);
    for (const ScalarType accumulatorType : {ScalarType::Float16, ScalarType::BFloat16}) {
        Tensor output = makeOutput(rows, columns, untouchedValue);
        GemmTestCase problem = makeProblem(reducedA, reducedB, output, rows, reductions, columns);
        problem.accumulatorType = accumulatorType;
        problem.outputSelection = selection;

        float expected = 0.0f;
        for (size_t reduction = 0; reduction < reductions; ++reduction) {
            const float product = roundThrough(accumulatorType, 0.1f * 0.1f);
            expected = roundThrough(accumulatorType, expected + product);
        }
        const GemmTestRunInfo run = referenceGemm(problem, GemmBackend::Blocked);
        require(run.outputElementsWritten == 1 && run.outputElementsCovered == 1 &&
                    output.loadAs<float>({0, 0}) == expected &&
                    output.loadAs<float>({0, 1}) == untouchedValue,
                "Selected blocked GEMM mishandled reduced-precision accumulation.");
    }

    const std::vector<int32_t> integerA(rows * 2, std::numeric_limits<int32_t>::max());
    const std::vector<int32_t> integerB(2 * columns, 2);
    const std::vector<int32_t> integerInitial(rows * columns, -99);
    Tensor integerOutput = Tensor::copyNativeValues<int32_t>(Shape{rows, columns}, integerInitial);
    GemmTestCase integerProblem(Tensor::copyNativeValues<int32_t>(Shape{rows, 2}, integerA),
                                Tensor::copyNativeValues<int32_t>(Shape{2, columns}, integerB),
                                integerOutput, ScalarType::Int32);
    integerProblem.outputSelection = selection;
    const GemmTestRunInfo integerRun = referenceGemm(integerProblem, GemmBackend::Blocked);
    const int32_t wrappedProduct =
        std::bit_cast<int32_t>(static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) * 2U);
    const int32_t wrappedSum = std::bit_cast<int32_t>(static_cast<uint32_t>(wrappedProduct) * 2U);
    require(integerRun.outputElementsCovered == 1 &&
                integerOutput.loadAs<int32_t>({0, 0}) == wrappedSum &&
                integerOutput.loadAs<int32_t>({0, 1}) == -99,
            "Selected blocked GEMM mishandled Int32 wrapping accumulation.");

    using Complex = std::complex<float>;
    const std::vector<Complex> complexA(rows * 2, Complex(1.0f, 2.0f));
    const std::vector<Complex> complexB(2 * columns, Complex(3.0f, 4.0f));
    const std::vector<Complex> complexInitial(rows * columns, Complex(-99.0f, -99.0f));
    Tensor complexOutput = Tensor::copyNativeValues<Complex>(Shape{rows, columns}, complexInitial);
    GemmTestCase complexProblem(Tensor::copyNativeValues<Complex>(Shape{rows, 2}, complexA),
                                Tensor::copyNativeValues<Complex>(Shape{2, columns}, complexB),
                                complexOutput, ScalarType::ComplexFloat32);
    complexProblem.outputSelection = selection;
    const GemmTestRunInfo complexRun = referenceGemm(complexProblem, GemmBackend::Blocked);
    require(complexRun.outputElementsCovered == 1 &&
                complexOutput.loadAs<Complex>({0, 0}) == Complex(-10.0f, 20.0f) &&
                complexOutput.loadAs<Complex>({0, 1}) == Complex(-99.0f, -99.0f),
            "Selected blocked GEMM mishandled complex accumulation.");
}

void requireOnlySelectedOutputsStored(const roc::host_numerics::Tensor& output,
                                      const OutputSelection& selection) {
    std::vector<size_t> selected = selection.indices(output.elementCount());
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    for (size_t index = 0; index < output.elementCount(); ++index) {
        if (!std::binary_search(selected.begin(), selected.end(), index))
            require(output.loadAs<float>({index / output.shape()[1], index % output.shape()[1]}) ==
                        untouchedValue,
                    "Blocked backend modified an unselected output element.");
    }
}

}  // namespace blocked_backend_test
