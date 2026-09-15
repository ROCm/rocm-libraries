// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

namespace tensor_core_test {
void testTensorEncodedCopies() {
    using namespace roc::host_numerics;
    Tensor packedCopySource(ScalarType::Int4, Shape{4});
    packedCopySource.storeFrom({0}, -8);
    packedCopySource.storeFrom({1}, -3);
    packedCopySource.storeFrom({2}, 2);
    packedCopySource.storeFrom({3}, 7);
    const std::array<size_t, 2> selectedCopyIndices{1, 3};
    std::array<std::byte, 2> packedCopyBytes{
        static_cast<std::byte>(0x11),
        static_cast<std::byte>(0x11),
    };
    packedCopySource.copySelectedElementsToEncodedStorage(packedCopyBytes, selectedCopyIndices,
                                                          IndexOrder::LastDimensionFastest);
    const Tensor selectedPackedCopy = Tensor::copyEncodedBackingStorage(
        ScalarType::Int4, Layout::contiguousLastDimensionFastest(Shape{4}),
        std::span<std::byte>(packedCopyBytes.data(), packedCopyBytes.size()));
    require(selectedPackedCopy.loadAs<int32_t>({0}) == 1 &&
                selectedPackedCopy.loadAs<int32_t>({1}) == -3 &&
                selectedPackedCopy.loadAs<int32_t>({2}) == 1 &&
                selectedPackedCopy.loadAs<int32_t>({3}) == 7,
            "Selected Tensor copy-to changed unselected packed elements.");

    std::array<std::byte, 2> fullPackedCopy{};
    packedCopySource.copyLogicalElementsToEncodedStorage(fullPackedCopy);
    require(std::equal(fullPackedCopy.begin(), fullPackedCopy.end(),
                       packedCopySource.rawEncodedBackingStorage().begin()),
            "Full Tensor copy-to changed packed storage.");

    Tensor packedBoundarySource(ScalarType::Int4, Shape{2});
    packedBoundarySource.storeFrom({0}, 1);
    packedBoundarySource.storeFrom({1}, 7);
    const Tensor highNibble = packedBoundarySource.shareStorageWithLayout(Layout(Shape{1}, {1}, 1));
    std::array<std::byte, 1> packedBoundaryDestination{std::byte{0x02}};
    highNibble.copyLogicalElementsToEncodedStorage(packedBoundaryDestination);
    require(packedBoundaryDestination[0] == std::byte{0x72},
            "Packed Tensor copy-to changed bits outside the addressed layout.");

    const Layout offsetLayout(Shape{2}, {1}, 2);
    Tensor offsetCopySource(ScalarType::Int32, offsetLayout);
    std::fill(offsetCopySource.rawEncodedBackingStorage().begin(),
              offsetCopySource.rawEncodedBackingStorage().end(), static_cast<std::byte>(0x7f));
    offsetCopySource.storeFrom({0}, 3);
    offsetCopySource.storeFrom({1}, 5);
    std::array<std::byte, 4 * sizeof(int32_t)> offsetCopyDestination{};
    offsetCopySource.copyLogicalElementsToEncodedStorage(offsetCopyDestination);
    require(std::all_of(offsetCopyDestination.begin(),
                        offsetCopyDestination.begin() + 2 * sizeof(int32_t),
                        [](std::byte value) { return value == std::byte{0}; }),
            "Tensor copy-to changed storage before the layout's first element.");
    const Tensor offsetCopyResult = Tensor::copyEncodedBackingStorage(
        ScalarType::Int32, offsetLayout,
        std::span<std::byte>(offsetCopyDestination.data(), offsetCopyDestination.size()));
    require(
        offsetCopyResult.loadAs<int32_t>({0}) == 3 && offsetCopyResult.loadAs<int32_t>({1}) == 5,
        "Tensor copy-to missed an offset layout value.");

    const Layout gappedLayout(Shape{2, 2}, {1, 3}, 1);
    Tensor gappedCopySource(ScalarType::Int32, gappedLayout);
    gappedCopySource.storeFrom({0, 0}, 2);
    gappedCopySource.storeFrom({0, 1}, 3);
    gappedCopySource.storeFrom({1, 0}, 5);
    gappedCopySource.storeFrom({1, 1}, 7);
    std::array<int32_t, 6> gappedDestination{-1, -1, -1, -1, -1, -1};
    gappedCopySource.copyLogicalElementsToEncodedStorage(
        std::as_writable_bytes(std::span(gappedDestination)));
    require(gappedDestination == std::array<int32_t, 6>{-1, 2, 5, -1, 3, 7},
            "Logical Tensor copy-to changed layout gaps.");

    std::array<std::byte, 1> undersizedCopyDestination{};
    requireInvalidArgument(
        [&] { packedCopySource.copyLogicalElementsToEncodedStorage(undersizedCopyDestination); },
        "Logical Tensor copy-to accepted undersized storage.");
    requireInvalidArgument(
        [&] {
            packedCopySource.copySelectedElementsToEncodedStorage(
                undersizedCopyDestination, selectedCopyIndices, IndexOrder::LastDimensionFastest);
        },
        "Selected Tensor copy-to accepted undersized storage.");

    const std::array<int32_t, 4> overlappingCopyOutValues{1, 2, 3, 4};
    Tensor overlappingCopyOut =
        Tensor::copyNativeStorage(std::span<const int32_t>(overlappingCopyOutValues));
    const Tensor overlappingCopyOutSource =
        overlappingCopyOut.shareStorageWithLayout(Layout(Shape{3}, {1}));
    overlappingCopyOutSource.copyLogicalElementsToEncodedStorage(
        overlappingCopyOut.rawEncodedBackingStorage().subspan(sizeof(int32_t)));
    require(overlappingCopyOut.loadAs<int32_t>({0}) == 1 &&
                overlappingCopyOut.loadAs<int32_t>({1}) == 1 &&
                overlappingCopyOut.loadAs<int32_t>({2}) == 2 &&
                overlappingCopyOut.loadAs<int32_t>({3}) == 3,
            "Overlapping logical Tensor copy-out did not preserve source values.");

    overlappingCopyOut =
        Tensor::copyNativeStorage(std::span<const int32_t>(overlappingCopyOutValues));
    const Tensor overlappingSelectedCopyOutSource =
        overlappingCopyOut.shareStorageWithLayout(Layout(Shape{3}, {1}));
    const std::array<size_t, 2> overlappingSelectedIndices{0, 1};
    overlappingSelectedCopyOutSource.copySelectedElementsToEncodedStorage(
        overlappingCopyOut.rawEncodedBackingStorage().subspan(sizeof(int32_t)),
        overlappingSelectedIndices, IndexOrder::LastDimensionFastest);
    require(overlappingCopyOut.loadAs<int32_t>({0}) == 1 &&
                overlappingCopyOut.loadAs<int32_t>({1}) == 1 &&
                overlappingCopyOut.loadAs<int32_t>({2}) == 2 &&
                overlappingCopyOut.loadAs<int32_t>({3}) == 4,
            "Overlapping selected Tensor copy-out did not preserve source values.");

    std::vector<std::byte> overlappingPackedBytes{std::byte{0x21}, std::byte{0x43},
                                                  std::byte{0x65}};
    Tensor overlappingPackedCopyOut = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::Int4, Layout::contiguousLastDimensionFastest(Shape{4}),
        std::move(overlappingPackedBytes));
    overlappingPackedCopyOut.copyLogicalElementsToEncodedStorage(
        overlappingPackedCopyOut.rawEncodedBackingStorage().subspan(1));
    require(overlappingPackedCopyOut.rawEncodedBackingStorage()[0] == std::byte{0x21} &&
                overlappingPackedCopyOut.rawEncodedBackingStorage()[1] == std::byte{0x21} &&
                overlappingPackedCopyOut.rawEncodedBackingStorage()[2] == std::byte{0x43},
            "Overlapping packed Tensor copy-out did not preserve source encodings.");

    Tensor orderedCopySource(ScalarType::Int32, Shape{2, 2});
    orderedCopySource.storeFrom({1, 0}, 7);
    std::array<int32_t, 4> orderedCopyDestination{};
    const std::array<size_t, 1> orderedCopyIndices{1};
    orderedCopySource.copySelectedElementsToEncodedStorage(
        std::as_writable_bytes(std::span(orderedCopyDestination)), orderedCopyIndices,
        IndexOrder::FirstDimensionFastest);
    require(orderedCopyDestination == std::array<int32_t, 4>{0, 0, 7, 0},
            "Selected Tensor copy-out ignored the requested logical index order.");
}
}  // namespace tensor_core_test
