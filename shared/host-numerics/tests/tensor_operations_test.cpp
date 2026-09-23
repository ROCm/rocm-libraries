// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testTensorOperations() {
    using namespace roc::host_numerics;

    const Shape shape{2, 2, 2};
    Tensor x(ScalarType::Float16, Layout(shape, {1, 3, 6}));
    Tensor y(ScalarType::BFloat16, Layout(shape, {3, 1, 6}));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 8}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 2; ++column) {
                const std::array<size_t, 3> indices{row, column, batch};
                x.storeFrom(indices, static_cast<float>(1 + row + 2 * column + batch));
                y.storeFrom(indices, static_cast<float>(2 - 2 * row + column + batch));
            }
        }
    }

    const Tensor scaledX = multiply(x, Tensor(2.0f), ScalarType::Float32, ScalarType::Float32);
    const Tensor scaledY = multiply(y, Tensor(-0.5f), ScalarType::Float32, ScalarType::Float32);
    addInto(scaledX, scaledY, output, ScalarType::Float32);

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 2; ++column) {
                const std::array<size_t, 3> indices{row, column, batch};
                const float expected =
                    2.0f * x.loadAs<float>(indices) - 0.5f * y.loadAs<float>(indices);
                require(output.loadAs<float>(indices) == expected,
                        "Tensor arithmetic value mismatch.");
            }
        }
    }

    Tensor yOnlyOutput(ScalarType::Float32, shape);
    multiplyInto(y, Tensor(3.0f), yOnlyOutput, ScalarType::Float32);
    require(yOnlyOutput.loadAs<float>({1, 0, 1}) == 3.0f * y.loadAs<float>({1, 0, 1}),
            "Tensor-scalar multiplication mismatch.");

    Tensor coefficientTensor(ScalarType::Float32, Shape{});
    coefficientTensor.storeFrom({}, 3.0f);
    coefficientTensor.storeFrom({}, 7.0f);
    const Tensor tensorCoefficientOutput =
        multiply(x, coefficientTensor, ScalarType::Float32, ScalarType::Float32);
    require(tensorCoefficientOutput.loadAs<float>({1, 0, 1}) == 7.0f * x.loadAs<float>({1, 0, 1}),
            "Rank-zero Tensor coefficient did not retain tensor aliasing semantics.");

    const std::array<float, 2> columnValues{1.0f, 2.0f};
    const std::array<float, 3> rowValues{10.0f, 20.0f, 30.0f};
    const Tensor column = Tensor::copyNativeValues<float>(Shape{2, 1}, columnValues);
    const Tensor row = Tensor::copyNativeValues<float>(Shape{1, 3}, rowValues);
    const Tensor broadcastOutput =
        add(multiply(column, Tensor(2.0f), ScalarType::Float32, ScalarType::Float32),
            multiply(row, Tensor(-1.0f), ScalarType::Float32, ScalarType::Float32),
            ScalarType::Float32, ScalarType::Float32);
    require(broadcastOutput.shape() == Shape{2, 3} &&
                broadcastOutput.loadAs<float>({0, 0}) == -8.0f &&
                broadcastOutput.loadAs<float>({0, 2}) == -28.0f &&
                broadcastOutput.loadAs<float>({1, 0}) == -6.0f &&
                broadcastOutput.loadAs<float>({1, 2}) == -26.0f,
            "Tensor arithmetic did not apply NumPy-style broadcasting.");

    const Tensor emptyBroadcast = add(Tensor(ScalarType::Float32, Shape{0, 3}),
                                      Tensor::copyNativeValues<float>(Shape{1, 3}, rowValues));
    require(emptyBroadcast.shape() == Shape{0, 3} && emptyBroadcast.elementCount() == 0,
            "Tensor addition did not preserve a broadcast zero extent.");

    bool rejectedIncompatibleBroadcast = false;
    try {
        (void)add(Tensor(ScalarType::Float32, Shape{2}), Tensor(ScalarType::Float32, Shape{3}));
    } catch (const std::invalid_argument&) {
        rejectedIncompatibleBroadcast = true;
    }
    require(rejectedIncompatibleBroadcast,
            "Tensor addition accepted incompatible broadcast shapes.");

    const std::array<std::complex<float>, 1> complexXValues{std::complex<float>(1, 2)};
    const std::array<std::complex<float>, 1> complexYValues{std::complex<float>(3, -1)};
    Tensor complexX = Tensor::copyNativeValues<std::complex<float>>(Shape{1}, complexXValues);
    Tensor complexY = Tensor::copyNativeValues<std::complex<float>>(Shape{1}, complexYValues);
    Tensor complexOutput(ScalarType::ComplexFloat32, Shape{1});
    const Tensor scaledComplexX = multiply(complexX, Tensor(std::complex<double>(0.5, 1.0)),
                                           ScalarType::ComplexFloat32, ScalarType::ComplexFloat32);
    const Tensor scaledComplexY =
        multiply(complexY, Tensor(-2.0), ScalarType::ComplexFloat32, ScalarType::ComplexFloat32);
    addInto(scaledComplexX, scaledComplexY, complexOutput, ScalarType::ComplexFloat32);
    require(complexOutput.loadAs<std::complex<float>>({0}) == std::complex<float>(-7.5f, 4.0f),
            "Complex tensor arithmetic mismatch.");

    const Tensor owned = add(scaledX, scaledY);
    require(owned.layout() == Layout::contiguousLastDimensionFastest(shape) &&
                owned.type() == ScalarType::Float32 &&
                owned.loadAs<float>({1, 0, 1}) ==
                    2.0f * x.loadAs<float>({1, 0, 1}) - 0.5f * y.loadAs<float>({1, 0, 1}),
            "Owning tensor addition result contract mismatch.");

    const Tensor operatorResult =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f}) * 2.0f + 1.0f;
    require(operatorResult.loadAs<float>({0}) == 3.0f && operatorResult.loadAs<float>({1}) == 5.0f,
            "Tensor arithmetic operators do not match the named operations.");

    const Tensor subtractionResult = 10.0f - operatorResult - 1.0f;
    require(subtractionResult.loadAs<float>({0}) == 6.0f &&
                subtractionResult.loadAs<float>({1}) == 4.0f,
            "Tensor subtraction operators do not match NumPy-style arithmetic.");
    const Tensor negated = -operatorResult;
    require(negated.loadAs<float>({0}) == -3.0f && negated.loadAs<float>({1}) == -5.0f,
            "Tensor unary negation mismatch.");

    const std::array<float, 4> fp4Values{0.5f, 1.0f, -1.5f, 3.0f};
    const std::array<float, 4> fp6Values{0.5f, 2.0f, -0.5f, 1.0f};
    const Tensor fp4 = Tensor::copyValuesWithConversion(ScalarType::Float4E2M1, Shape{4},
                                                        std::span<const float>(fp4Values));
    const Tensor fp6 = Tensor::copyValuesWithConversion(ScalarType::Float6E2M3, Shape{4},
                                                        std::span<const float>(fp6Values));
    const Tensor packedProduct = multiply(fp4, fp6, ScalarType::Float32, ScalarType::Float32);
    const std::array<float, 4> packedExpected{0.25f, 2.0f, 0.75f, 3.0f};
    for (size_t index = 0; index < packedExpected.size(); ++index)
        require(packedProduct.loadAs<float>({index}) == packedExpected[index],
                "Packed input arithmetic mismatch.");

    requireInvalidArgument([&] { (void)(fp4 * fp4); },
                           "Implicit packed arithmetic did not require a compute type.");

    const std::array<float, 4> fp4OutputValues{0.5f, 1.5f, -3.0f, 6.0f};
    const Tensor fp4Output =
        multiply(Tensor::copyNativeValues<float>(Shape{4}, std::span<const float>(fp4OutputValues)),
                 Tensor(1.0f), ScalarType::Float4E2M1, ScalarType::Float32);
    for (size_t index = 0; index < fp4OutputValues.size(); ++index)
        require(fp4Output.loadAs<float>({index}) == fp4OutputValues[index],
                "Packed output arithmetic mismatch.");

    const std::array<int32_t, 2> intValues{4, -5};
    const Tensor saturatedInt4 =
        multiply(Tensor::copyNativeValues<int32_t>(Shape{2}, std::span<const int32_t>(intValues)),
                 Tensor(int32_t{2}), ScalarType::Int4, ScalarType::Int32);
    require(saturatedInt4.loadAs<int32_t>({0}) == 7 && saturatedInt4.loadAs<int32_t>({1}) == -8,
            "Packed Int4 output did not apply its implicit saturation policy.");

    const std::array<float, 3> initialPackedValues{0.5f, 0.5f, 0.5f};
    Tensor selectedPackedOutput = Tensor::copyValuesWithConversion(
        ScalarType::Float4E2M1, Shape{3}, std::span<const float>(initialPackedValues));
    const std::byte packedPaddingBefore = selectedPackedOutput.rawEncodedBackingStorage().back();
    const std::array<float, 3> selectedInputValues{2.0f, 2.0f, 2.0f};
    multiplyInto(
        Tensor::copyNativeValues<float>(Shape{3}, std::span<const float>(selectedInputValues)),
        Tensor(1.0f), selectedPackedOutput, ScalarType::Float32,
        OutputSelection::explicitIndices({1}));
    require(selectedPackedOutput.loadAs<float>({0}) == 0.5f &&
                selectedPackedOutput.loadAs<float>({1}) == 2.0f &&
                selectedPackedOutput.loadAs<float>({2}) == 0.5f,
            "Selected packed arithmetic modified an unselected value.");
    require((std::to_integer<uint8_t>(selectedPackedOutput.rawEncodedBackingStorage().back()) &
             0xf0U) == (std::to_integer<uint8_t>(packedPaddingBefore) & 0xf0U),
            "Selected packed arithmetic modified padding bits.");

    const std::array<float, 2> subtractionInputValues{5.0f, 8.0f};
    Tensor subtractionOutput(ScalarType::Float32, Shape{2});
    subtractInto(
        Tensor::copyNativeValues<float>(Shape{2}, std::span<const float>(subtractionInputValues)),
        Tensor(3.0f), subtractionOutput, ScalarType::Float32);
    require(subtractionOutput.loadAs<float>({0}) == 2.0f &&
                subtractionOutput.loadAs<float>({1}) == 5.0f,
            "Tensor subtraction-into mismatch.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)multiply(x, Tensor(std::complex<double>(1.0, 1.0)), ScalarType::Float32,
                       ScalarType::Float32);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Tensor multiplication accepted an invalid coefficient.");

    const Tensor activationInput =
        Tensor::copyNativeValues<float>(Shape{4}, std::array<float, 4>{-2.0f, -0.5f, 0.0f, 2.0f});
    const Tensor reluOutput = relu(activationInput);
    const Tensor clippedOutput = clip(activationInput, -1.0, 1.0);
    require(reluOutput.shape() == Shape{4} && reluOutput.loadAs<float>({0}) == 0.0f &&
                reluOutput.loadAs<float>({3}) == 2.0f,
            "Named ReLU operation mismatch.");
    require(clippedOutput.loadAs<float>({0}) == -1.0f && clippedOutput.loadAs<float>({3}) == 1.0f,
            "Named clip operation mismatch.");
    require(compare(activationInput.relu(), reluOutput).passed() &&
                compare(activationInput.clip(-1.0, 1.0), clippedOutput).passed(),
            "Tensor activation methods differ from the named operations.");
}

}  // namespace host_numerics_test
