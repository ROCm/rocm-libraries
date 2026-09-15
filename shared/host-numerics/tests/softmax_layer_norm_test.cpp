// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testReferenceSoftmax() {
    using namespace roc::host_numerics;

    const Shape shape{2, 3, 2};
    Tensor input(ScalarType::Float16, Layout(shape, {1, 3, 10}));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 12}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                input.storeFrom({row, column, batch}, static_cast<float>(static_cast<int>(column) +
                                                                         2 * static_cast<int>(row) -
                                                                         static_cast<int>(batch)));
            }
        }
    }

    referenceSoftmaxInto(input, output, 1, ScalarType::Float32);

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            float sum = 0;
            for (size_t column = 0; column < 3; ++column)
                sum += output.loadAs<float>({row, column, batch});
            require(std::abs(sum - 1.0f) < 1e-6f, "Reference softmax slice does not sum to one.");
        }
    }
    require(output.loadAs<float>({0, 0, 0}) < output.loadAs<float>({0, 1, 0}) &&
                output.loadAs<float>({0, 1, 0}) < output.loadAs<float>({0, 2, 0}),
            "Reference softmax ordering mismatch.");

    const Tensor owned = referenceSoftmax(input, 1, ScalarType::Float32, ScalarType::Float32);
    require(owned.layout() == Layout::contiguousLastDimensionFastest(shape) &&
                owned.type() == ScalarType::Float32,
            "Owning reference softmax result contract mismatch.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceSoftmax(input, shape.rank(), ScalarType::Float32, ScalarType::Float32);
    } catch (const std::out_of_range&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference softmax accepted an invalid problem.");
}

void testReferenceLayerNorm() {
    using namespace roc::host_numerics;

    const Shape shape{2, 3, 2};
    Tensor input(ScalarType::Float32, Layout(shape, {1, 3, 10}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 3; ++column)
                input.storeFrom({row, column, batch},
                                static_cast<float>(1 + column + 3 * row + 6 * batch));
        }
    }
    const std::array<float, 3> gammaValues{1.0f, 2.0f, 0.5f};
    const std::array<float, 3> betaValues{0.25f, -0.5f, 1.0f};
    const Tensor gamma = Tensor::copyValuesWithConversion(ScalarType::Float16, Shape{3},
                                                          std::span<const float>(gammaValues));
    const Tensor beta = Tensor::copyValuesWithConversion(ScalarType::BFloat16, Shape{3},
                                                         std::span<const float>(betaValues));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 12}));
    Tensor mean(ScalarType::Float32, Layout(Shape{2, 2}, {3, 1}));
    Tensor inverseVariance(ScalarType::Float32, Layout(Shape{2, 2}, {1, 3}));

    LayerNormOptions options;
    options.axis = 1;
    options.gamma = gamma;
    options.beta = beta;
    referenceLayerNormInto(
        input, {.output = output, .mean = mean, .inverseVariance = inverseVariance}, options);
    require(mean.loadAs<float>({0, 0}) == 2.0f, "Reference LayerNorm mean mismatch.");
    require(std::abs(inverseVariance.loadAs<float>({0, 0}) -
                     1.0f / std::sqrt(2.0f / 3.0f + 1e-5f)) < 1e-6f,
            "Reference LayerNorm inverse variance mismatch.");
    require(output.loadAs<float>({0, 1, 0}) == -0.5f,
            "Reference LayerNorm affine output mismatch.");

    const LayerNormOutputs owned = referenceLayerNorm(input,
                                                      {.output = ScalarType::Float32,
                                                       .mean = ScalarType::Float32,
                                                       .inverseVariance = ScalarType::Float32},
                                                      options);
    require(owned.output.layout() == Layout::contiguousLastDimensionFastest(shape) && owned.mean &&
                owned.mean->layout() == Layout::contiguousLastDimensionFastest(Shape{2, 2}) &&
                owned.inverseVariance &&
                owned.inverseVariance->layout() ==
                    Layout::contiguousLastDimensionFastest(Shape{2, 2}) &&
                owned.output.loadAs<float>({0, 1, 0}) == -0.5f,
            "Owning reference LayerNorm result contract mismatch.");

    LayerNormOptions invalidOptions = options;
    invalidOptions.epsilon = std::numeric_limits<double>::quiet_NaN();
    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceLayerNorm(input, {}, invalidOptions);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference LayerNorm accepted invalid epsilon.");

    const LayerNormOutputs outputOnly = referenceLayerNorm(input, {}, options);
    require(!outputOnly.mean && !outputOnly.inverseVariance,
            "Owning reference LayerNorm created unrequested statistics.");

    const std::array<float, 3> rankOneValues{1.0f, 2.0f, 3.0f};
    const LayerNormOutputs rankOneResult = referenceLayerNorm(
        Tensor::copyNativeValues<float>(Shape{3}, std::span<const float>(rankOneValues)),
        {.output = ScalarType::Float32, .mean = ScalarType::Float32});
    require(rankOneResult.mean && rankOneResult.mean->shape() == Shape{} &&
                !rankOneResult.inverseVariance,
            "Owning reference LayerNorm did not preserve a requested rank-zero statistic.");
}

}  // namespace host_numerics_test
