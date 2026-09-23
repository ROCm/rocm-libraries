// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testReferenceReduction() {
    using namespace roc::host_numerics;

    std::array<float, 30> storage;
    storage.fill(-1);
    Tensor input = Tensor::copyNativeStorage<float>(Layout(Shape{2, 3, 4}, {15, 5, 1}),
                                                    std::span<float>(storage));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 4; ++column)
                input.storeFrom({batch, row, column}, 100 * batch + 10 * row + column);
        }
    }

    Tensor output(ScalarType::Float32, Layout(Shape{3}, {2}));
    referenceSumInto(input, output, {0, 2}, ScalarType::Float32);
    require(compare(output,
                    Tensor::copyNativeValues<float>(Shape{3}, std::array<float, 3>{412, 492, 572}))
                .passed(),
            "Reference reduction result mismatch.");

    const std::array<int32_t, 2> wrappingInputValues{std::numeric_limits<int32_t>::max(), 1};
    const Tensor wrappingSum =
        referenceSum(Tensor::copyNativeStorage<int32_t>(
                         Layout::contiguousLastDimensionFastest(Shape{2}), wrappingInputValues),
                     {0}, ScalarType::Int32, ScalarType::Int32);
    require(wrappingSum.loadAs<int32_t>({}) == std::numeric_limits<int32_t>::min(),
            "Int32 reference reduction did not use defined wrapping arithmetic.");

    Tensor maximumAbsolute(ScalarType::Float32, Shape{});
    referenceMaximumAbsoluteInto(input, maximumAbsolute, ScalarType::Float32);
    require(maximumAbsolute.loadAs<float>({}) == 123.0f,
            "Reference maximum-absolute result mismatch.");

    const Tensor owned = referenceSum(input, {0, 2}, ScalarType::Float32, ScalarType::Float32);
    require(owned.layout() == Layout::contiguousLastDimensionFastest(Shape{3}) &&
                owned.type() == ScalarType::Float32 &&
                compare(owned, Tensor::copyNativeValues<float>(Shape{3},
                                                               std::array<float, 3>{412, 492, 572}))
                    .passed(),
            "Owning reference sum result contract mismatch.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceSum(input, {0, 0}, ScalarType::Float32, ScalarType::Float32);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference sum accepted invalid axes.");

    const Tensor ownedMaximum =
        referenceMaximumAbsolute(input, ScalarType::Float32, ScalarType::Float32);
    require(ownedMaximum.shape() == Shape{} &&
                ownedMaximum.layout() == Layout::contiguousLastDimensionFastest(Shape{}) &&
                ownedMaximum.loadAs<float>({}) == 123.0f,
            "Owning reference maximum-absolute result contract mismatch.");
}

}  // namespace host_numerics_test
