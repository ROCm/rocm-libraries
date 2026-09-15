// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

namespace tensor_core_test {
void testTensorStorageOwnership() {
    using namespace roc::host_numerics;
    Tensor tensor(ScalarType::Float32, Shape{2, 3});
    tensor.storeFrom({1, 2}, 7.0f);
    Tensor broadcast(ScalarType::Float32, Layout(Shape{2, 2}, {0, 0}));
    broadcast.storeFrom({0, 0}, 5.0f);
    Tensor broadcastCopy = broadcast.copyConvertedTo(ScalarType::Float32);
    broadcastCopy.storeFrom({0, 0}, 7.0f);
    require(broadcast.loadAs<float>({1, 1}) == 5.0f && broadcastCopy.loadAs<float>({1, 1}) == 7.0f,
            "Same-layout Tensor conversion rejected or aliased a broadcast layout.");

    Tensor copied = tensor;
    copied.storeFrom({1, 2}, 11.0f);
    require(tensor.loadAs<float>({1, 2}) == 11.0f,
            "Copying a tensor did not preserve shared-storage semantics.");
    Tensor cloned = tensor.deepCopy();
    cloned.storeFrom({1, 2}, 13.0f);
    require(tensor.loadAs<float>({1, 2}) == 11.0f && cloned.loadAs<float>({1, 2}) == 13.0f,
            "Tensor clone did not deep-copy storage.");

    auto externalOwner = std::make_shared<std::vector<std::byte>>(sizeof(float));
    std::weak_ptr<std::vector<std::byte>> externalLifetime = externalOwner;
    Tensor externallyShared = Tensor::shareExternalMutableBackingStorage(
        ScalarType::Float32, Layout::contiguousLastDimensionFastest(Shape{1}), externalOwner,
        std::span<std::byte>(*externalOwner));
    externalOwner.reset();
    externallyShared.storeFrom({0}, 17.0f);
    require(!externalLifetime.expired() && externallyShared.loadAs<float>({0}) == 17.0f,
            "Externally shared Tensor did not retain its lifetime anchor.");
    requireInvalidArgument(
        [&] {
            (void)Tensor::shareExternalMutableBackingStorage(
                ScalarType::Float32, Layout::contiguousLastDimensionFastest(Shape{1}), {},
                externallyShared.rawEncodedBackingStorage());
        },
        "Externally shared Tensor accepted nonempty storage without a lifetime anchor.");

    std::vector<std::byte> paddedBackingStorage(2 * sizeof(float), std::byte{0x5a});
    Tensor paddedBacking = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::Float32, Layout::contiguousLastDimensionFastest(Shape{1}),
        std::move(paddedBackingStorage));
    Tensor paddedBackingCopy = paddedBacking.deepCopy();
    require(paddedBacking.rawEncodedBackingStorage().size() == 2 * sizeof(float) &&
                paddedBackingCopy.rawEncodedBackingStorage().size() == 2 * sizeof(float) &&
                std::ranges::equal(paddedBacking.rawEncodedBackingStorage(),
                                   paddedBackingCopy.rawEncodedBackingStorage()),
            "Tensor deep copy did not preserve complete backing storage.");

    const std::array<int, 4> overlappingValues{1, 2, 3, 4};
    Tensor overlappingCopy = Tensor::copyValuesWithConversion(
        ScalarType::Int32, Shape{4}, std::span<const int>(overlappingValues));
    Tensor overlappingSource = overlappingCopy.shareStorageWithLayout(Layout(Shape{3}, {1}, 0));
    Tensor overlappingDestination =
        overlappingCopy.shareStorageWithLayout(Layout(Shape{3}, {1}, 1));
    overlappingDestination.copyLogicalElementsFrom(overlappingSource);
    require(overlappingCopy.loadAs<int>({0}) == 1 && overlappingCopy.loadAs<int>({1}) == 1 &&
                overlappingCopy.loadAs<int>({2}) == 2 && overlappingCopy.loadAs<int>({3}) == 3,
            "Overlapping Tensor copy did not preserve source values.");

    Tensor collidingCopySource = Tensor::copyValuesWithConversion(
        ScalarType::Int32, Shape{2}, std::span<const int>(overlappingValues).first(2));
    Tensor collidingCopyDestination(ScalarType::Int32, Layout(Shape{2}, {0}));
    requireInvalidArgument(
        [&] { collidingCopyDestination.copyLogicalElementsFrom(collidingCopySource); },
        "Tensor copy accepted a destination with colliding logical element offsets.");
}
}  // namespace tensor_core_test
