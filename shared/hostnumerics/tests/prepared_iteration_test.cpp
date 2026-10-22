// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/hostnumerics/generation.hpp>
#include <roc/hostnumerics/reduction.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>
#include <vector>

#include "detail/generation_values.hpp"
#include "detail/layout_iteration.hpp"
#include "detail/runtime_tensor_io.hpp"

using namespace roc::hostnumerics;

TEST(PreparedIteration, ScalarCodecsAgreeWithCheckedAccessForEveryType) {
    for (size_t value = 0; value < static_cast<size_t>(ScalarType::Count); ++value) {
        const auto type = static_cast<ScalarType>(value);
        if (!isConcreteScalarType(type)) continue;
        const Layout layout(Shape{2, 3}, {-7, 2}, 7);
        Tensor checked(type, layout), prepared(type, layout);
        const detail::RuntimeTensorWriter<double> writer(prepared);
        const detail::RuntimeTensorReader<double> reader(prepared);
        for (size_t row = 0; row < 2; ++row)
            for (size_t column = 0; column < 3; ++column) {
                const double number = static_cast<double>(row + column);
                checked.storeFrom({row, column}, number);
                writer.store({row, column}, number);
                EXPECT_EQ(reader({row, column}), checked.loadAs<double>({row, column}));
            }
        EXPECT_TRUE(std::equal(checked.rawEncodedBackingStorage().begin(),
                               checked.rawEncodedBackingStorage().end(),
                               prepared.rawEncodedBackingStorage().begin()));
    }
}

TEST(PreparedIteration, ZeroStrideHugeCoordinatesDoNotOverflowSignedOffsets) {
    Tensor tensor(ScalarType::Float32, Layout(Shape{SIZE_MAX}, {0}));
    tensor.storeFrom({0}, 7.0f);
    const detail::RuntimeTensorReader<float> reader(tensor);
    EXPECT_EQ(reader({SIZE_MAX - 1}), 7.0f);
    EXPECT_EQ(detail::preparedElementOffset(tensor.layout(), std::array<size_t, 1>{SIZE_MAX - 1}),
              0);
}

TEST(PreparedIteration, PermutationsAndScalarHaveCorrectLoopBounds) {
    const Shape shape{2, 3, 4};
    std::vector<size_t> order{0, 1, 2};
    do {
        const detail::IndexIteration plan(shape, order);
        std::array<size_t, 3> point{};
        for (size_t index = 0; index < plan.count(); ++index) {
            EXPECT_EQ(shape.coordinates(index, order),
                      (std::vector<size_t>(point.begin(), point.end())));
            EXPECT_EQ(plan.linearIndex(point), index);
            EXPECT_EQ(plan.advance(point), index + 1 < plan.count());
        }
    } while (std::next_permutation(order.begin(), order.end()));
    const detail::IndexIteration scalar(Shape{});
    EXPECT_EQ(scalar.count(), 1);
    EXPECT_EQ(scalar.linearIndex({}), 0);
    EXPECT_FALSE(scalar.advance({}));
    EXPECT_EQ(detail::IndexIteration(Shape{3, 0, 4}).count(), 0);
    EXPECT_THROW(detail::IndexIteration(shape, std::vector<size_t>{0, 0, 2}), std::runtime_error);
}

TEST(PreparedIteration, CheckedPublicAccessAndNumericalChecksRemain) {
    Tensor tensor(ScalarType::Float32, Shape{2});
    tensor.storeFrom({0}, std::numeric_limits<float>::infinity());
    EXPECT_THROW(tensor.loadAs<float>({2}), std::runtime_error);
    EXPECT_THROW(tensor.storeFrom({2}, 1.0f), std::runtime_error);
    const detail::RuntimeTensorReader<int64_t> integerReader(tensor);
    EXPECT_THROW(integerReader({0}), std::runtime_error);
    const auto affine = GenerationRecipe::realOnly(GenerationRecipe::affineIndexRemainder(
        {.dimensionCoefficients = {std::numeric_limits<int64_t>::max()},
         .offset = 1,
         .positiveDivisor = 7}));
    EXPECT_THROW(generate(tensor, affine), std::runtime_error);
    EXPECT_TRUE(std::isinf(tensor.loadAs<float>({0})));
}

TEST(PreparedIteration, HugeZeroStrideRunAndInvalidEmptyOrders) {
    const Layout layout(Shape{SIZE_MAX}, {0});
    size_t calls = 0;
    detail::forEachLayoutOffsetPairRange(layout, layout, {}, SIZE_MAX - 2, SIZE_MAX,
                                         [&](size_t, ptrdiff_t first, ptrdiff_t second) {
                                             EXPECT_EQ(first, 0);
                                             EXPECT_EQ(second, 0);
                                             ++calls;
                                         });
    EXPECT_EQ(calls, 2);
    EXPECT_THROW(detail::forEachLayoutOffsetPairRange(layout, layout, std::vector<size_t>{1}, 0, 0,
                                                      [](size_t, ptrdiff_t, ptrdiff_t) {}),
                 std::runtime_error);
}

TEST(PreparedIteration, AffinePrefixOverflowFailsBeforeAnyOutputWrite) {
    Tensor tensor(ScalarType::Float32, Shape{2, 2});
    generate(tensor, GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 7.0})));
    const auto recipe = GenerationRecipe::realOnly(GenerationRecipe::affineIndexRemainder({
        .dimensionCoefficients = {1, -1},
        .offset = std::numeric_limits<int64_t>::max(),
        .positiveDivisor = 7,
    }));
    EXPECT_THROW(generate(tensor, recipe), std::runtime_error);
    EXPECT_EQ(tensor.loadAs<float>({0, 0}), 7.0f);
}

TEST(PreparedIteration, ElementwiseSelectionUsesPermutationAndRejectsInvalidEmptyOrders) {
    const Shape shape{2, 3, 4};
    Tensor input(ScalarType::Float32, shape), output(ScalarType::Float32, shape);
    generate(input, GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    const Tensor one = Tensor::scalar(ScalarType::Float32, 1.0f);
    const std::vector<size_t> order{1, 0, 2};
    for (const auto& selection :
         {OutputSelection::all(order), OutputSelection::strided(1, 3, 5, order),
          OutputSelection::explicitIndices({1, 7, 19}, order)}) {
        generate(output, GenerationRecipe::realOnly(GenerationRecipe::constant({.value = -7})));
        addInto(input, one, output, ScalarType::Float32, selection);
        const auto selected = selection.indices(shape.elementCount());
        for (size_t index = 0; index < shape.elementCount(); ++index) {
            const auto coordinates = shape.coordinates(index, order);
            const bool included =
                std::find(selected.begin(), selected.end(), index) != selected.end();
            EXPECT_EQ(output.loadAs<float>(coordinates),
                      included ? input.loadAs<float>(coordinates) + 1.0f : -7.0f);
        }
    }
    for (const Shape& emptyOrScalar : {Shape{2, 0, 4}, Shape{}}) {
        Tensor tensor(ScalarType::Float32, emptyOrScalar);
        const std::array<size_t, 3> invalidOrder{0, 0, 2};
        for (const auto& selection : {OutputSelection::all(invalidOrder),
                                      OutputSelection::explicitIndices({}, invalidOrder)}) {
            EXPECT_THROW(addInto(tensor, one, tensor, ScalarType::Float32, selection),
                         std::runtime_error);
            EXPECT_THROW(
                activateInto(tensor, tensor, ReluActivation{}, ScalarType::Float32, selection),
                std::runtime_error);
        }
    }
}

TEST(PreparedIteration, RecipesAndSelectionsOwnOrdersPassedAsSpans) {
    static_assert(std::is_same_v<decltype(std::declval<GenerationRecipe>().indexOrder()),
                                 std::span<const size_t>>);
    static_assert(std::is_same_v<decltype(std::declval<OutputSelection>().indexOrder()),
                                 std::span<const size_t>>);
    const auto recipe = GenerationRecipe::realOnly(GenerationRecipe::serialIndex());
    std::array<size_t, 3> order{2, 1, 0};
    const auto ordered = recipe.withIndexOrder(order);
    const auto selection = OutputSelection::explicitIndices({1, 7}, order);
    order = {0, 1, 2};
    const std::array<size_t, 3> expected{2, 1, 0};
    EXPECT_TRUE(std::ranges::equal(ordered.indexOrder(), expected));
    EXPECT_TRUE(std::ranges::equal(selection.indexOrder(), expected));
    const auto scopedSelection = [] {
        const std::array<size_t, 3> local{2, 1, 0};
        return OutputSelection::strided(0, 2, 5, local);
    }();
    EXPECT_TRUE(std::ranges::equal(scopedSelection.indexOrder(), expected));
    const auto tensor = generate(ScalarType::Float32, Shape{2, 3, 4}, ordered);
    EXPECT_EQ(tensor.loadAs<float>({1, 0, 2}), 13.0f);
    const Tensor selected = tensor.copySelectedElements(std::array<size_t, 2>{1, 7}, expected);
    EXPECT_EQ(selected.loadAs<float>({0}), 1.0f);
    EXPECT_EQ(selected.loadAs<float>({1}), 7.0f);
}

TEST(PreparedIteration, EncodedExponentIntervalSupportsFullIntRange) {
    const RandomEncodedExponentGenerationParameters parameters{
        .lowerUnbiasedExponent = std::numeric_limits<int>::min(),
        .upperUnbiasedExponent = std::numeric_limits<int>::max(),
    };
    const auto plan = detail::prepareEncodedExponent(parameters, ScalarType::Float32);
    EXPECT_EQ(plan.lowerExponent, std::numeric_limits<int>::min());
    EXPECT_EQ(plan.exponentCount, uint64_t{1} << 32);
    const auto recipe =
        GenerationRecipe::realOnly(GenerationRecipe::randomEncodedExponent(parameters));
    EXPECT_NO_THROW(generate(Tensor(ScalarType::Float32, Shape{64}), recipe));
}

TEST(PreparedIteration, EncodedExponentDecodersMatchTensorAccessForEveryFloatingType) {
    for (size_t value = 0; value < scalarTypeCount; ++value) {
        const auto type = static_cast<ScalarType>(value);
        const auto& info = scalarTypeInfo(type);
        if (info.category != ScalarCategory::FloatingPoint) continue;
        SCOPED_TRACE(info.name);
        const auto plan = detail::prepareEncodedExponent({}, type);
        for (uint64_t raw :
             {uint64_t{0}, uint64_t{1}, uint64_t{0x17}, std::numeric_limits<uint64_t>::max()}) {
            const auto storage =
                std::as_bytes(std::span(&raw, 1)).first((info.storageBits + 7) / 8);
            const auto tensor =
                Tensor::copyEncodedBackingStorage(type, Layout(Shape{}, {}), storage);
            const double expected = tensor.loadAs<double>({});
            const double decoded = plan.decode(raw);
            if (std::isnan(expected))
                EXPECT_TRUE(std::isnan(decoded));
            else
                EXPECT_EQ(decoded, expected);
        }
    }
}
