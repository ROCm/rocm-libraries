// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <roc/host_numerics/generation.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using namespace roc::host_numerics;

struct Options {
    ScalarType type;
    std::string recipe;
    size_t rows;
    size_t columns;
    size_t batches;
    size_t warmups;
    size_t iterations;
};

size_t parseSize(const char* text, const char* name) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0')
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
    return static_cast<size_t>(value);
}

ScalarType parseType(std::string_view name) {
    for (size_t index = 0; index < scalarTypeCount; ++index) {
        const ScalarType type = static_cast<ScalarType>(index);
        if (name == scalarTypeName(type)) return type;
    }
    throw std::invalid_argument("Unknown scalar type.");
}

Options parseOptions(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: host-numerics-generation-benchmark "
                     "<scalar-type> "
                     "<zero|constant|choice|uniform-integer|alternating-integer|"
                     "absolute-uniform-integer|uniform-real|uniform-encoded|normal|sine|serial> "
                     "<rows> <columns> <batches> <warmups> <iterations>\n";
        std::exit(0);
    }
    if (argc != 8) throw std::invalid_argument("Use --help for benchmark arguments.");

    return {
        .type = parseType(argv[1]),
        .recipe = argv[2],
        .rows = parseSize(argv[3], "rows"),
        .columns = parseSize(argv[4], "columns"),
        .batches = parseSize(argv[5], "batches"),
        .warmups = parseSize(argv[6], "warmup count"),
        .iterations = parseSize(argv[7], "iteration count"),
    };
}

GenerationRecipe generationRecipe(std::string_view name) {
    using Component = GenerationRecipe::Component;
    Component component = [&] {
        if (name == "zero") return GenerationRecipe::zero();
        if (name == "constant") return GenerationRecipe::constant({.value = 0.25});
        if (name == "choice")
            return GenerationRecipe::choice({.values = {-2.0, -0.5, 0.0, 1.0, 3.0}});
        if (name == "uniform-integer")
            return GenerationRecipe::uniformInteger({.lower = -2, .upper = 2});
        if (name == "alternating-integer")
            return GenerationRecipe::uniformInteger({.lower = -2, .upper = 2})
                .withAlternatingSign({.dimensions = {0, 1}, .negativeWhenOdd = false});
        if (name == "absolute-uniform-integer")
            return GenerationRecipe::absoluteUniformInteger({.lower = -2, .upper = 2});
        if (name == "uniform-real")
            return GenerationRecipe::uniformReal({.lower = -0.5, .upper = 0.5});
        if (name == "uniform-encoded") return GenerationRecipe::uniformFiniteEncodedValue();
        if (name == "normal") return GenerationRecipe::normal({});
        if (name == "sine") return GenerationRecipe::sine();
        if (name == "serial") return GenerationRecipe::serialIndex();
        throw std::invalid_argument(
            "Recipe must be zero, constant, choice, uniform-integer, alternating-integer, "
            "absolute-uniform-integer, uniform-real, uniform-encoded, normal, sine, or serial.");
    }();
    return GenerationRecipe::realOnly(std::move(component), {.seed = 17});
}

double median(std::vector<double> values) {
    std::ranges::sort(values);
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5 : values[middle];
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.iterations == 0)
            throw std::invalid_argument("Iteration count must be nonzero.");

        const Shape shape{options.rows, options.columns, options.batches};
        const Layout layout = Layout::contiguousFirstDimensionFastest(shape);
        Tensor output = scalarTypeInfo(options.type).isPacked()
                            ? Tensor(options.type, layout)
                            : Tensor::allocateUninitialized(options.type, layout);
        const GenerationRecipe recipe = generationRecipe(options.recipe);

        for (size_t iteration = 0; iteration < options.warmups; ++iteration)
            generate(output, recipe);

        std::vector<double> milliseconds;
        milliseconds.reserve(options.iterations);
        for (size_t iteration = 0; iteration < options.iterations; ++iteration) {
            const auto start = Clock::now();
            generate(output, recipe);
            milliseconds.push_back(
                std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        }

        const size_t bytes = output.rawEncodedBackingStorage().size();
        const double medianMs = median(milliseconds);
        const double meanMs = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) /
                              static_cast<double>(milliseconds.size());
        double checksum = 0.0;
        if (output.elementCount() != 0) {
            for (const size_t index :
                 {size_t{0}, output.elementCount() / 2, output.elementCount() - 1})
                checksum += output.loadAs<double>(
                    shape.coordinates(index, IndexOrder::FirstDimensionFastest));
        }
        std::cout << std::setprecision(10) << "type=" << scalarTypeName(options.type)
                  << ",recipe=" << options.recipe << ",rows=" << options.rows
                  << ",columns=" << options.columns << ",batches=" << options.batches
                  << ",bytes=" << bytes << ",warmups=" << options.warmups
                  << ",iterations=" << options.iterations
                  << ",min_ms=" << *std::ranges::min_element(milliseconds)
                  << ",median_ms=" << medianMs << ",mean_ms=" << meanMs
                  << ",gigabytes_per_second=" << static_cast<double>(bytes) / (medianMs * 1.0e6)
                  << ",checksum=" << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host-numerics-generation-benchmark: " << error.what() << '\n';
        return 1;
    }
}
