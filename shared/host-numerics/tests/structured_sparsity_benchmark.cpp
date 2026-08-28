// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <roc/host_numerics/validation.hpp>
#include <span>
#include <stdexcept>
#include <vector>

#if HOST_NUMERICS_BENCHMARK_OPENMP
#include <omp.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;

template <typename Function>
double milliseconds(Function&& function, size_t iterations) {
    const auto begin = Clock::now();
    for (size_t iteration = 0; iteration < iterations; ++iteration) function();
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() /
           static_cast<double>(iterations);
}

void legacyTwoPass(std::vector<float>& pruned, std::vector<float>& compressed,
                   std::vector<uint8_t>& metadata, size_t rows, size_t columns) {
    std::fill(compressed.begin(), compressed.end(), 0);
    std::fill(metadata.begin(), metadata.end(), 0);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(rows); ++row) {
        for (size_t column = 0; column < columns; column += 4) {
            pruned[static_cast<size_t>(row) + (column + 2) * rows] = 0;
            pruned[static_cast<size_t>(row) + (column + 3) * rows] = 0;
        }
    }

    const size_t metadataColumns = columns / 8;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(rows); ++row) {
        for (size_t group = 0; group < columns / 4; ++group) {
            const size_t source = static_cast<size_t>(row) + group * 4 * rows;
            const size_t destination = static_cast<size_t>(row) + group * 2 * rows;
            compressed[destination] = pruned[source];
            compressed[destination + rows] = pruned[source + rows];
            const size_t metadataIndex = static_cast<size_t>(row) * metadataColumns + group / 2;
            metadata[metadataIndex] = static_cast<uint8_t>(
                metadata[metadataIndex] | static_cast<uint8_t>(0x4U << ((group % 2) * 4)));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace roc::host_numerics;
    const size_t rows = argc > 1 ? static_cast<size_t>(std::stoull(argv[1])) : 4096;
    const size_t columns = argc > 2 ? static_cast<size_t>(std::stoull(argv[2])) : 4096;
    const size_t iterations = argc > 3 ? static_cast<size_t>(std::stoull(argv[3])) : 5;
    if (columns % 8 != 0) throw std::invalid_argument("Benchmark columns must be divisible by 8.");

    std::vector<float> input(rows * columns);
    for (size_t column = 0; column < columns; ++column)
        for (size_t row = 0; row < rows; ++row)
            input[row + column * rows] = static_cast<float>((row + column * rows) % 97 + 1);

    std::vector<float> pruned = input;
    std::vector<float> compressed(rows * columns / 2);
    std::vector<uint8_t> metadata(rows * columns / 8);
    const Layout inputLayout(Shape{rows, columns}, {1, static_cast<ptrdiff_t>(rows)});
    const Layout compressedLayout(Shape{rows, columns / 2}, {1, static_cast<ptrdiff_t>(rows)});
    const Layout metadataLayout(Shape{rows, columns / 8}, {static_cast<ptrdiff_t>(columns / 8), 1});
    StructuredSparsityPattern pattern;
    pattern.axis = 1;
    pattern.fixedPositions = {0, 1};
    Tensor prunedTensor =
        Tensor::copyNativeStorage<float>(inputLayout, std::span<const float>(pruned));
    Tensor compressedTensor =
        Tensor::copyNativeStorage<float>(compressedLayout, std::span<const float>(compressed));
    Tensor metadataTensor =
        Tensor::copyNativeStorage<uint8_t>(metadataLayout, std::span<const uint8_t>(metadata));
    StructuredSparsityRequest request(prunedTensor, prunedTensor, compressedTensor, std::nullopt,
                                      metadataTensor, pattern);

    const double componentMilliseconds =
        milliseconds([&] { applyStructuredSparsity(request); }, iterations);
    std::memcpy(pruned.data(), prunedTensor.rawEncodedBackingStorage().data(),
                prunedTensor.rawEncodedBackingStorage().size());
    std::memcpy(compressed.data(), compressedTensor.rawEncodedBackingStorage().data(),
                compressedTensor.rawEncodedBackingStorage().size());
    std::memcpy(metadata.data(), metadataTensor.rawEncodedBackingStorage().data(),
                metadataTensor.rawEncodedBackingStorage().size());

    std::vector<float> legacyPruned = input;
    std::vector<float> legacyCompressed(rows * columns / 2);
    std::vector<uint8_t> legacyMetadata(rows * columns / 8);
    const double legacyMilliseconds = milliseconds(
        [&] { legacyTwoPass(legacyPruned, legacyCompressed, legacyMetadata, rows, columns); },
        iterations);

    if (pruned != legacyPruned || compressed != legacyCompressed || metadata != legacyMetadata)
        throw std::runtime_error("Structured sparsity benchmark result mismatch.");

    const double bytes = static_cast<double>(input.size() * sizeof(float));
    const double componentBandwidth = bytes / (componentMilliseconds * 1.0e6);
    const double legacyBandwidth = bytes / (legacyMilliseconds * 1.0e6);
#if HOST_NUMERICS_BENCHMARK_OPENMP
    const int openmpThreads = omp_get_max_threads();
#else
    const int openmpThreads = 1;
#endif
    std::cout << "shape=" << rows << 'x' << columns << '\n'
              << "openmp_threads=" << openmpThreads << '\n'
              << "component_ms=" << componentMilliseconds << " input_GBps=" << componentBandwidth
              << '\n'
              << "legacy_openmp_two_pass_ms=" << legacyMilliseconds
              << " input_GBps=" << legacyBandwidth << '\n'
              << "ratio_component_over_legacy=" << componentMilliseconds / legacyMilliseconds
              << '\n';
}
