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
#include <roc/host_validation/backends/blocked.hpp>
#include <roc/host_validation/comparison.hpp>
#include <roc/host_validation/gemm.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using roc::host_validation::BlockedGemmBackend;
using roc::host_validation::GemmBackend;
using roc::host_validation::GemmExecution;
using roc::host_validation::GemmOperand;
using roc::host_validation::GemmRequest;
using roc::host_validation::GemmRunInfo;
using roc::host_validation::Layout;
using roc::host_validation::OutputSelection;
using roc::host_validation::ScalarType;
using roc::host_validation::Shape;
using roc::host_validation::Tensor;

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
};

NumericProfile numericProfile(std::string_view name) {
    if (name == "f32") return {ScalarType::Float32, ScalarType::Float32, ScalarType::Float32};
    if (name == "f64") return {ScalarType::Float64, ScalarType::Float64, ScalarType::Float64};
    if (name == "f16") return {ScalarType::Float16, ScalarType::Float16, ScalarType::Float32};
    if (name == "bf16") return {ScalarType::BFloat16, ScalarType::BFloat16, ScalarType::Float32};
    if (name == "f8") return {ScalarType::Float8E4M3, ScalarType::Float32, ScalarType::Float32};
    if (name == "bf8") return {ScalarType::Float8E5M2, ScalarType::Float32, ScalarType::Float32};
    if (name == "f4mx")
        return {ScalarType::Float4E2M1, ScalarType::Float32, ScalarType::Float32, true};
    if (name == "f6mx")
        return {ScalarType::Float6E2M3, ScalarType::Float32, ScalarType::Float32, true};
    if (name == "bf6mx")
        return {ScalarType::Float6E3M2, ScalarType::Float32, ScalarType::Float32, true};
    throw std::invalid_argument(
        "Profile must be f32, f64, f16, bf16, f8, bf8, f4mx, f6mx, "
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
        std::cout << "Usage: host-validation-gemm-benchmark "
                     "<pointwise|blocked> <M> <N> <K> <selected-or-0-for-all> "
                     "<warmups> <iterations> [profile]\n";
        std::exit(0);
    }
    if (argc != 8 && argc != 9) throw std::invalid_argument("Use --help for benchmark arguments.");

    Options options;
    const std::string_view backend(argv[1]);
    if (backend == "pointwise")
        options.backend = GemmBackend::Pointwise;
    else if (backend == "blocked")
        options.backend = GemmBackend::Blocked;
    else
        throw std::invalid_argument("Backend must be pointwise or blocked.");

    options.rows = parseSize(argv[2], "M");
    options.columns = parseSize(argv[3], "N");
    options.reductions = parseSize(argv[4], "K");
    options.selectedOutputs = parseSize(argv[5], "selected output count");
    options.warmups = parseSize(argv[6], "warmup count");
    options.iterations = parseSize(argv[7], "iteration count");
    if (argc == 9) options.profile = argv[8];
    if (options.rows == 0 || options.columns == 0 || options.reductions == 0)
        throw std::invalid_argument("M, N, and K must be nonzero.");
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

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

const char* backendName(GemmBackend backend) {
    switch (backend) {
        case GemmBackend::Pointwise:
            return "pointwise";
        case GemmBackend::Blocked:
            return "blocked";
        default:
            return "unexpected";
    }
}
}  // namespace

int main(int argc, char** argv) {
    using namespace roc::host_validation;

    try {
        const Options options = parseOptions(argc, argv);
        const NumericProfile profile = numericProfile(options.profile);
        const size_t outputElements = options.rows * options.columns;
        const OutputSelection selection =
            options.selectedOutputs == 0 || options.selectedOutputs >= outputElements
                ? OutputSelection::all()
                : OutputSelection::primeStride(outputElements, outputElements,
                                               options.selectedOutputs);

        const Layout aLayout(Shape{options.rows, options.reductions},
                             {1, static_cast<ptrdiff_t>(options.rows)});
        const Layout bLayout(Shape{options.reductions, options.columns},
                             {1, static_cast<ptrdiff_t>(options.reductions)});
        const Layout outputLayout(Shape{options.rows, options.columns},
                                  {1, static_cast<ptrdiff_t>(options.rows)});
        const Tensor a = makeMatrix(profile.inputType, aLayout, 1);
        const Tensor b = makeMatrix(profile.inputType, bLayout, 2);
        const Tensor c = makeMatrix(profile.outputType, outputLayout, 0, true);
        const Tensor output(profile.outputType, outputLayout);
        GemmRequest request(GemmOperand(a), GemmOperand(b), c, output, profile.accumulatorType);
        request.outputSelection = selection;
        if (profile.blockScaled) {
            constexpr size_t blockSize = 32;
            if (options.reductions % blockSize != 0)
                throw std::invalid_argument("MX profiles require K divisible by 32.");
            const size_t reductionBlocks = options.reductions / blockSize;
            request.a.blockScale =
                BlockScaleBinding(makeBlockScales(options.rows, reductionBlocks, 3), blockSize);
            request.b.blockScale =
                BlockScaleBinding(makeBlockScales(options.columns, reductionBlocks, 4), blockSize);
        }

        BlockedGemmBackend blockedBackend;
        const GemmExecution execution{
            .backend = options.backend,
            .requireRequestedBackend = true,
        };
        const auto* implementation =
            options.backend == GemmBackend::Blocked ? &blockedBackend : nullptr;

        GemmRunInfo runInfo;
        for (size_t iteration = 0; iteration < options.warmups; ++iteration)
            runInfo = referenceGemm(request, execution, implementation);

        std::vector<double> milliseconds;
        milliseconds.reserve(options.iterations);
        for (size_t iteration = 0; iteration < options.iterations; ++iteration) {
            const auto start = Clock::now();
            runInfo = referenceGemm(request, execution, implementation);
            const auto end = Clock::now();
            milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        const OutputSelection validationSelection =
            selection.selectsAll() && outputElements > 128
                ? OutputSelection::primeStride(outputElements, outputElements, 128)
                : selection;
        const Tensor expected(profile.outputType, outputLayout);
        GemmRequest expectedRequest(GemmOperand(a), GemmOperand(b), c, expected,
                                    profile.accumulatorType);
        expectedRequest.outputSelection = validationSelection;
        expectedRequest.a.blockScale = request.a.blockScale;
        expectedRequest.b.blockScale = request.b.blockScale;
        referenceGemm(expectedRequest, {
                                           .backend = GemmBackend::Pointwise,
                                           .requireRequestedBackend = true,
                                       });

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
                throw std::runtime_error("Benchmark backend differs from pointwise reference.");
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
        std::cerr << "host-validation-gemm-benchmark: " << error.what() << '\n';
        return 1;
    }
}
