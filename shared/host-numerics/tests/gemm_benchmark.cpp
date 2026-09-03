// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <roc/host_numerics/comparison.hpp>
#include <roc/host_numerics/gemm.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gemm_test_adapter.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using roc::host_numerics::GemmBackend;
using roc::host_numerics::GemmTestCase;
using roc::host_numerics::GemmTestRunInfo;
using roc::host_numerics::Layout;
using roc::host_numerics::OutputSelection;
using roc::host_numerics::ScalarType;
using roc::host_numerics::Shape;
using roc::host_numerics::Tensor;

struct Options {
    GemmBackend backend = GemmBackend::Blocked;
    std::string profile = "f32";
    size_t rows = 512;
    size_t columns = 512;
    size_t reductions = 512;
    size_t selectedOutputs = 0;
    size_t warmups = 1;
    size_t iterations = 5;
};

struct NumericProfile {
    ScalarType inputType;
    ScalarType outputType;
    ScalarType accumulatorType;
    bool blockScaled = false;
    bool affineLayouts = false;
    bool commonEpilogue = false;
};

NumericProfile numericProfile(std::string_view name) {
    if (name == "f32") return {ScalarType::Float32, ScalarType::Float32, ScalarType::Float32};
    if (name == "f64") return {ScalarType::Float64, ScalarType::Float64, ScalarType::Float64};
    if (name == "f16") return {ScalarType::Float16, ScalarType::Float16, ScalarType::Float32};
    if (name == "bf16") return {ScalarType::BFloat16, ScalarType::BFloat16, ScalarType::Float32};
    if (name == "f8") return {ScalarType::Float8E4M3, ScalarType::Float32, ScalarType::Float32};
    if (name == "bf8") return {ScalarType::Float8E5M2, ScalarType::Float32, ScalarType::Float32};
    if (name == "f32-affine-epilogue")
        return {ScalarType::Float32, ScalarType::Float32, ScalarType::Float32, false, true, true};
    if (name == "f4mx")
        return {ScalarType::Float4E2M1, ScalarType::Float32, ScalarType::Float32, true};
    if (name == "f6mx")
        return {ScalarType::Float6E2M3, ScalarType::Float32, ScalarType::Float32, true};
    if (name == "bf6mx")
        return {ScalarType::Float6E3M2, ScalarType::Float32, ScalarType::Float32, true};
    throw std::invalid_argument(
        "Profile must be f32, f64, f16, bf16, f8, bf8, f32-affine-epilogue, f4mx, f6mx, "
        "or bf6mx.");
}

size_t parseSize(const char* text, const char* name) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0')
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
    return static_cast<size_t>(value);
}

Options parseOptions(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: host-numerics-gemm-benchmark "
                     "<automatic|blocked> <M> <N> <K> <selected-or-0-for-all> "
                     "<warmups> <iterations> [profile]\n";
        std::exit(0);
    }
    if (argc != 8 && argc != 9) throw std::invalid_argument("Use --help for benchmark arguments.");

    Options options;
    const std::string_view backend(argv[1]);
    if (backend == "automatic")
        options.backend = GemmBackend::Automatic;
    else if (backend == "blocked")
        options.backend = GemmBackend::Blocked;
    else
        throw std::invalid_argument("Backend must be automatic or blocked.");

    options.rows = parseSize(argv[2], "M");
    options.columns = parseSize(argv[3], "N");
    options.reductions = parseSize(argv[4], "K");
    options.selectedOutputs = parseSize(argv[5], "selected output count");
    options.warmups = parseSize(argv[6], "warmup count");
    options.iterations = parseSize(argv[7], "iteration count");
    if (argc == 9) options.profile = argv[8];
    if (options.iterations == 0) throw std::invalid_argument("Iterations must be nonzero.");
    return options;
}

Tensor makeMatrix(ScalarType type, const Layout& layout, size_t stream, bool zero = false) {
    Tensor tensor(type, layout);
    for (size_t row = 0; row < layout.shape()[0]; ++row) {
        for (size_t column = 0; column < layout.shape()[1]; ++column) {
            const size_t index = row * layout.shape()[1] + column;
            const int encoded = static_cast<int>((index * 17 + stream * 13) % 15) - 7;
            tensor.storeFrom({row, column}, zero ? 0.0f : static_cast<float>(encoded) * 0.0625f);
        }
    }
    return tensor;
}

Tensor makeBlockScales(size_t freeElements, size_t reductionBlocks, size_t stream) {
    Tensor scales(ScalarType::E8M0, Shape{freeElements, reductionBlocks});
    for (size_t free = 0; free < freeElements; ++free) {
        for (size_t block = 0; block < reductionBlocks; ++block) {
            const int exponent = static_cast<int>((free + block + stream) % 3) - 1;
            scales.storeFrom({free, block}, std::ldexp(1.0f, exponent));
        }
    }
    return scales;
}

template <typename Accumulator>
void runScalarOracle(const GemmTestCase& problem, const Tensor& destination) {
    using namespace roc::host_numerics;

    if (problem.computeTypeA || problem.computeTypeB || !problem.preQuantizationScalesA.empty() ||
        !problem.preQuantizationScalesB.empty() || problem.mathMode != MathMode::Default ||
        problem.conjugateA || problem.conjugateB ||
        problem.outputConversion != OutputConversion::Default)
        throw std::invalid_argument("Benchmark scalar oracle does not support this profile.");
    if (problem.activation != Activation::None && problem.activation != Activation::Relu)
        throw std::invalid_argument("Benchmark scalar oracle does not support this activation.");

    const size_t rows = problem.a.shape()[0];
    const size_t reductions = problem.a.shape()[1];
    const size_t columns = problem.b.shape()[1];
    const Shape outputShape{rows, columns};
    const auto scaleValue = [&](const std::optional<Tensor>& scale, size_t row, size_t column) {
        return scale ? scale->broadcastTo(outputShape).loadAs<Accumulator>({row, column})
                     : Accumulator(1);
    };

    for (const size_t logicalIndex :
         problem.outputSelection.indices(problem.d.shape().elementCount())) {
        const auto coordinates =
            problem.d.shape().coordinates(logicalIndex, problem.outputSelection.indexOrder());
        const size_t row = coordinates[0];
        const size_t column = coordinates[1];
        Accumulator accumulation = Accumulator(0);

        for (size_t blockBase = 0; blockBase < reductions;) {
            const size_t remainingA = problem.blockScaleA
                                          ? problem.blockSizeA - blockBase % problem.blockSizeA
                                          : reductions - blockBase;
            const size_t remainingB = problem.blockScaleB
                                          ? problem.blockSizeB - blockBase % problem.blockSizeB
                                          : reductions - blockBase;
            const size_t blockEnd =
                blockBase + std::min({reductions - blockBase, remainingA, remainingB});
            Accumulator partial = Accumulator(0);
            for (size_t reduction = blockBase; reduction < blockEnd; ++reduction)
                partial += problem.a.loadAs<Accumulator>({row, reduction}) *
                           problem.b.loadAs<Accumulator>({reduction, column});
            if (problem.blockScaleA)
                partial *=
                    problem.blockScaleA->loadAs<Accumulator>({row, blockBase / problem.blockSizeA});
            if (problem.blockScaleB)
                partial *= problem.blockScaleB->loadAs<Accumulator>(
                    {column, blockBase / problem.blockSizeB});
            accumulation += partial;
            blockBase = blockEnd;
        }

        Accumulator effectiveAlpha = problem.alpha.as<Accumulator>();
        effectiveAlpha *= scaleValue(problem.scaleA, row, column);
        effectiveAlpha *= scaleValue(problem.scaleB, row, column);
        effectiveAlpha *= scaleValue(problem.scaleAlpha, row, column);
        Accumulator result = effectiveAlpha * accumulation;
        if (problem.beta.as<Accumulator>() != Accumulator(0))
            result += problem.beta.as<Accumulator>() * problem.scaleC.as<Accumulator>() *
                      problem.c.loadAs<Accumulator>({row, column});
        if (problem.bias)
            result += problem.bias->broadcastTo(outputShape).loadAs<Accumulator>({row, column});
        if (problem.activation == Activation::Relu) result = std::max(Accumulator(0), result);
        result *= problem.outputScale.as<Accumulator>();
        destination.storeFrom({row, column}, result);
    }
}

void runScalarOracle(const GemmTestCase& problem, const Tensor& destination) {
    using roc::host_numerics::ScalarType;

    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            runScalarOracle<float>(problem, destination);
            return;
        case ScalarType::Float64:
            runScalarOracle<double>(problem, destination);
            return;
        default:
            throw std::invalid_argument(
                "Benchmark scalar oracle requires F32 or F64 accumulation.");
    }
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

const char* backendName(GemmBackend backend) {
    switch (backend) {
        case GemmBackend::Automatic:
            return "automatic";
        case GemmBackend::Blocked:
            return "blocked";
        default:
            return "unexpected";
    }
}
}  // namespace

int main(int argc, char** argv) {
    using namespace roc::host_numerics;

    try {
        const Options options = parseOptions(argc, argv);
        const NumericProfile profile = numericProfile(options.profile);
        const size_t outputElements = options.rows * options.columns;
        const OutputSelection selection =
            options.selectedOutputs == 0 || options.selectedOutputs >= outputElements
                ? OutputSelection::all()
                : OutputSelection::primeStride(outputElements, outputElements,
                                               options.selectedOutputs);

        const ptrdiff_t aLeadingDimension =
            static_cast<ptrdiff_t>(options.rows + (profile.affineLayouts ? 3 : 0));
        const ptrdiff_t bLeadingDimension =
            static_cast<ptrdiff_t>(options.reductions + (profile.affineLayouts ? 5 : 0));
        const ptrdiff_t outputLeadingDimension =
            static_cast<ptrdiff_t>(options.rows + (profile.affineLayouts ? 7 : 0));
        const ptrdiff_t inputOffset = profile.affineLayouts ? 2 : 0;
        const ptrdiff_t outputOffset = profile.affineLayouts ? 3 : 0;
        const Layout aLayout(Shape{options.rows, options.reductions}, {1, aLeadingDimension},
                             inputOffset);
        const Layout bLayout(Shape{options.reductions, options.columns}, {1, bLeadingDimension},
                             inputOffset);
        const Layout outputLayout(Shape{options.rows, options.columns}, {1, outputLeadingDimension},
                                  outputOffset);
        const Tensor a = makeMatrix(profile.inputType, aLayout, 1);
        const Tensor b = makeMatrix(profile.inputType, bLayout, 2);
        const Tensor c = makeMatrix(profile.outputType, outputLayout, 5, !profile.commonEpilogue);
        const Tensor output(profile.outputType, outputLayout);
        GemmOptions requestOptions(profile.accumulatorType);
        requestOptions.outputSelection = selection;
        if (profile.blockScaled) {
            constexpr size_t blockSize = 32;
            if (options.reductions % blockSize != 0)
                throw std::invalid_argument("MX profiles require K divisible by 32.");
            const size_t reductionBlocks = options.reductions / blockSize;
            requestOptions.blockScaleA = makeBlockScales(options.rows, reductionBlocks, 3);
            requestOptions.blockSizeA = blockSize;
            requestOptions.blockScaleB = makeBlockScales(options.columns, reductionBlocks, 4);
            requestOptions.blockSizeB = blockSize;
        }
        if (profile.commonEpilogue) {
            requestOptions.alpha = 0.75f;
            requestOptions.beta = -0.25f;
            requestOptions.bias =
                makeMatrix(ScalarType::Float32,
                           Layout::contiguousLastDimensionFastest(Shape{1, options.columns}), 6);
            requestOptions.scaleA =
                makeMatrix(ScalarType::Float32,
                           Layout::contiguousLastDimensionFastest(Shape{options.rows, 1}), 7);
            requestOptions.scaleB =
                makeMatrix(ScalarType::Float32,
                           Layout::contiguousLastDimensionFastest(Shape{1, options.columns}), 8);
            requestOptions.activation = Activation::Relu;
            requestOptions.outputScale = 0.5f;
        }
        GemmTestCase request(a, b, c, output, requestOptions);

        const GemmBackend backend = options.backend;
        GemmTestRunInfo runInfo;
        for (size_t iteration = 0; iteration < options.warmups; ++iteration)
            runInfo = referenceGemm(request, backend);

        std::vector<double> milliseconds;
        milliseconds.reserve(options.iterations);
        for (size_t iteration = 0; iteration < options.iterations; ++iteration) {
            const auto start = Clock::now();
            runInfo = referenceGemm(request, backend);
            const auto end = Clock::now();
            milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        const OutputSelection validationSelection =
            selection.selectsAll() && outputElements > 128
                ? OutputSelection::primeStride(outputElements, outputElements, 128)
                : selection;
        const Tensor expected(profile.outputType, outputLayout);
        GemmOptions expectedOptions = requestOptions;
        expectedOptions.outputSelection = validationSelection;
        GemmTestCase expectedRequest(a, b, c, expected, expectedOptions);
        runScalarOracle(expectedRequest, expected);

        const std::vector<size_t> validationIndices = validationSelection.indices(outputElements);
        const double tolerance = defaultSymmetricRelativeTolerance(profile.outputType);
        double checksum = 0.0;
        for (const size_t index : validationIndices) {
            const double observed =
                output.loadAs<double>({index / options.columns, index % options.columns});
            const double reference =
                expected.loadAs<double>({index / options.columns, index % options.columns});
            const double difference = std::abs(observed - reference);
            if (observed != reference &&
                !(difference < tolerance * (std::abs(observed) + std::abs(reference) + 1.0)))
                throw std::runtime_error("Benchmark result differs from the scalar oracle.");
            checksum += observed;
        }

        const double minimumMs = *std::min_element(milliseconds.begin(), milliseconds.end());
        const double medianMs = median(milliseconds);
        const double meanMs = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) /
                              static_cast<double>(milliseconds.size());
        const double operations = 2.0 * static_cast<double>(runInfo.outputElementsCovered) *
                                  static_cast<double>(options.reductions);

        std::cout << std::setprecision(10) << "backend=" << backendName(runInfo.backendUsed)
                  << ",profile=" << options.profile << ",m=" << options.rows
                  << ",n=" << options.columns << ",k=" << options.reductions
                  << ",selected=" << runInfo.outputElementsWritten
                  << ",covered=" << runInfo.outputElementsCovered << ",warmups=" << options.warmups
                  << ",iterations=" << options.iterations << ",min_ms=" << minimumMs
                  << ",median_ms=" << medianMs << ",mean_ms=" << meanMs
                  << ",effective_gflops=" << operations / (medianMs * 1.0e6)
                  << ",checksum=" << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host-numerics-gemm-benchmark: " << error.what() << '\n';
        return 1;
    }
}
