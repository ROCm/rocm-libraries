// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <roc/host_numerics/generation.hpp>
#include <roc/host_numerics/tensor.hpp>

namespace {
using Clock = std::chrono::steady_clock;

template <typename Function>
double bestSeconds(Function&& function) {
    double best = std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < 4; ++iteration) {
        const auto start = Clock::now();
        function();
        const auto end = Clock::now();
        best = std::min(best, std::chrono::duration<double>(end - start).count());
    }
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    using namespace roc::host_numerics;

    const size_t rows = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 576;
    const size_t columns = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 28000;
    const Shape shape{rows, columns};
    const Layout destinationLayout = Layout::contiguousFirstDimensionFastest(shape);

    for (const ScalarType sourceType :
         {ScalarType::Float16, ScalarType::BFloat16, ScalarType::Float8E4M3}) {
        for (const IndexOrder sourceOrder :
             {IndexOrder::FirstDimensionFastest, IndexOrder::LastDimensionFastest}) {
            const Layout sourceLayout = sourceOrder == IndexOrder::FirstDimensionFastest
                                            ? Layout::contiguousFirstDimensionFastest(shape)
                                            : Layout::contiguousLastDimensionFastest(shape);
            Tensor source = Tensor::allocateUninitialized(sourceType, sourceLayout);
            generate(source,
                     GenerationRecipe::realOnly(
                         GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}), {.seed = 1}));

            Tensor converted = source.copyConvertedTo(ScalarType::Float32, destinationLayout);
            const double seconds = bestSeconds([&] {
                converted = source.copyConvertedTo(ScalarType::Float32, destinationLayout);
            });
            if (converted.shape() != shape) return 1;

            const double bytes = static_cast<double>(source.rawEncodedBackingStorage().size() +
                                                     converted.rawEncodedBackingStorage().size());
            std::cout << "source_type=" << scalarTypeName(sourceType) << " source_order="
                      << (sourceOrder == IndexOrder::FirstDimensionFastest ? "first" : "last")
                      << " rows=" << rows << " columns=" << columns << " seconds=" << seconds
                      << " GBps=" << bytes / seconds / 1e9 << '\n';
        }
    }
}
