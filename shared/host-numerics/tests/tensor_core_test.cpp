// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <roc/host_numerics/tensor.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(float observed, float expected, float tolerance, const char* message) {
    if (std::abs(observed - expected) > tolerance) throw std::runtime_error(message);
}

template <typename Function>
void requireOverflow(Function function, const char* message) {
    bool overflowed = false;
    try {
        function();
    } catch (const std::overflow_error&) {
        overflowed = true;
    }
    require(overflowed, message);
}

template <typename Function>
void requireInvalidArgument(Function function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}
}  // namespace

int main() {
    using namespace roc::host_numerics;

    static_assert(std::is_same_v<decltype(std::declval<const Tensor&>().rawEncodedBackingStorage()),
                                 std::span<std::byte>>);

    const Shape shape{2, 3};
    require(shape.rank() == 2, "Shape rank mismatch.");
    require(shape.extent(0) == 2 && shape.extent(1) == 3, "Shape extent mismatch.");
    require(shape.elementCount() == 6, "Shape element count mismatch.");
    const Shape dimensionalShape{2, 3, 4};
    require(
        dimensionalShape.elementCount(1, 3) == 12 && dimensionalShape.elementCountExcluding(1) == 8,
        "Shape dimension product helper mismatch.");
    require(Shape{2, 0, 4}.elementCountExcluding(1) == 8,
            "Shape excluded-dimension product incorrectly retained a zero extent.");
    const std::array<size_t, 3> logicalCoordinates{1, 0, 2};
    require(
        dimensionalShape.linearIndex(logicalCoordinates, IndexOrder::FirstDimensionFastest) == 13 &&
            dimensionalShape.linearIndex(logicalCoordinates, IndexOrder::LastDimensionFastest) ==
                14,
        "Shape linear-index conversion mismatch.");
    require(dimensionalShape.coordinates(17, IndexOrder::FirstDimensionFastest) ==
                    std::vector<size_t>({1, 2, 2}) &&
                dimensionalShape.coordinates(17, IndexOrder::LastDimensionFastest) ==
                    std::vector<size_t>({1, 1, 1}),
            "Shape coordinate conversion mismatch.");
    const Shape overflowingLinearShape{std::numeric_limits<size_t>::max(), 2};
    const std::array<size_t, 2> overflowingLinearCoordinates{std::numeric_limits<size_t>::max() - 1,
                                                             1};
    requireOverflow(
        [&] {
            (void)overflowingLinearShape.linearIndex(overflowingLinearCoordinates,
                                                     IndexOrder::LastDimensionFastest);
        },
        "Shape linear index accepted overflowing coordinate arithmetic.");
    const Layout rowMajor = Layout::contiguousLastDimensionFastest(shape);
    const Layout columnMajor = Layout::contiguousFirstDimensionFastest(shape);
    require(rowMajor == Layout::contiguousLastDimensionFastest(shape),
            "Contiguous compatibility API changed dimension order.");
    require(rowMajor.shape().rank() == 2 && rowMajor.shape().dimensions().size() == 2 &&
                rowMajor.shape().extent(0) == 2 && rowMajor.shape().elementCount() == 6 &&
                rowMajor.shape().elementCount(0, 1) == 2 &&
                rowMajor.shape().elementCountExcluding(0) == 3 && rowMajor.stride(0) == 3 &&
                rowMajor.stride(1) == 1,
            "Last-dimension-fastest layout helper mismatch.");
    require(columnMajor.stride(0) == 1 && columnMajor.stride(1) == 2,
            "First-dimension-fastest layout helper mismatch.");

    Tensor tensor(ScalarType::Float32, shape);
    tensor.storeFrom({1, 2}, 7.0f);
    require(tensor.loadAs<float>({1, 2}) == 7.0f, "Owning tensor view mismatch.");
    require(tensor.layout().strides()[0] == 3 && tensor.layout().strides()[1] == 1,
            "Contiguous tensor strides mismatch.");
    require(tensor.rawEncodedBackingStorage().size() == 6 * sizeof(float),
            "Float32 storage size mismatch.");

    const std::array<float, 3> nativeValues{2.0f, 4.0f, 6.0f};
    const Tensor nativeTensor =
        Tensor::copyNativeValues<float>(Shape{3}, std::span<const float>(nativeValues));
    require(nativeTensor.type() == ScalarType::Float32 && nativeTensor.loadAs<float>({2}) == 6.0f,
            "Native tensor factory mismatch.");

    Tensor scalarTensor(ScalarType::Float32, Layout(Shape{}, {}, 2));
    scalarTensor.storeFrom({}, -3.5f);
    const Scalar scalarSnapshot = scalarTensor.item();
    scalarTensor.storeFrom({}, 9.0f);
    require(scalarSnapshot.type() == ScalarType::Float32 && scalarSnapshot.as<float>() == -3.5f,
            "Tensor item did not preserve independent scalar value semantics.");

    Tensor packedScalarTensor(ScalarType::Float6E3M2, Layout(Shape{}, {}, 1));
    packedScalarTensor.storeFrom({}, 1.5f);
    const Scalar packedScalar = packedScalarTensor;
    require(packedScalar.type() == ScalarType::Float6E3M2 && packedScalar.as<float>() == 1.5f,
            "Packed rank-zero Tensor did not convert to Scalar.");
    requireInvalidArgument([&] { (void)nativeTensor.item(); },
                           "Tensor item accepted a non-scalar shape.");

    Tensor uninitialized = Tensor::allocateUninitialized(ScalarType::Float32, Shape{2});
    uninitialized.storeFrom({0}, 3.0f);
    uninitialized.storeFrom({1}, 7.0f);
    require(uninitialized.loadAs<float>({0}) == 3.0f && uninitialized.loadAs<float>({1}) == 7.0f,
            "Uninitialized Tensor storage did not retain written values.");
    requireInvalidArgument([&] { (void)Tensor::allocateUninitialized(ScalarType::Int4, Shape{2}); },
                           "Uninitialized Tensor allocation accepted a packed scalar type.");

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

    Tensor reshapeSource(ScalarType::Int32, Shape{2, 3});
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            reshapeSource.storeFrom({row, column}, static_cast<int32_t>(row * 3 + column));
    Tensor reshaped = reshapeSource.reshapeSharingStorage(Shape{3, 2});
    require(reshaped.shape() == Shape{3, 2} && reshaped.loadAs<int32_t>({1, 1}) == 3,
            "Tensor reshape changed logical linear order.");
    reshaped.storeFrom({2, 1}, 19);
    require(reshapeSource.loadAs<int32_t>({1, 2}) == 19,
            "Tensor reshape did not return a shallow alias.");
    requireInvalidArgument([&] { (void)reshapeSource.reshapeSharingStorage(Shape{7}); },
                           "Tensor reshape accepted a different element count.");

    Tensor packedPaddingSource(ScalarType::Int4, Shape{3});
    packedPaddingSource.storeFrom({0}, -8);
    packedPaddingSource.storeFrom({1}, -3);
    packedPaddingSource.storeFrom({2}, 7);
    const Tensor packedPadded = packedPaddingSource.copyWithZeroPadding(Shape{5});
    require(packedPadded.shape() == Shape{5} &&
                packedPadded.rawEncodedBackingStorage().size() == 3 &&
                packedPadded.loadAs<int32_t>({0}) == -8 &&
                packedPadded.loadAs<int32_t>({1}) == -3 && packedPadded.loadAs<int32_t>({2}) == 7 &&
                packedPadded.loadAs<int32_t>({3}) == 0 && packedPadded.loadAs<int32_t>({4}) == 0,
            "Tensor padding did not preserve packed values and zero-fill new elements.");
    requireInvalidArgument([&] { (void)packedPaddingSource.copyWithZeroPadding(Shape{2}); },
                           "Tensor padding accepted a shrinking shape.");
    requireInvalidArgument([&] { (void)packedPaddingSource.copyWithZeroPadding(Shape{3, 1}); },
                           "Tensor padding accepted a different rank.");

    Tensor permutationSource(ScalarType::Float6E3M2, Shape{2, 3});
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            permutationSource.storeFrom({row, column}, static_cast<int32_t>(3 * row + column));
    const std::array<size_t, 2> transpose{1, 0};
    const Tensor permutationResult = permutationSource.copyWithPermutedDimensions(transpose);
    require(permutationResult.shape() == Shape{3, 2} &&
                permutationResult.rawEncodedBackingStorage().size() == 5,
            "Tensor permutation produced the wrong packed output geometry.");
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            require(permutationResult.loadAs<float>({column, row}) ==
                        permutationSource.loadAs<float>({row, column}),
                    "Tensor permutation changed a packed value.");
    const std::array<size_t, 2> duplicatePermutation{0, 0};
    const std::array<size_t, 2> outOfRangePermutation{0, 2};
    const std::array<size_t, 1> wrongRankPermutation{0};
    requireInvalidArgument(
        [&] { (void)permutationSource.copyWithPermutedDimensions(duplicatePermutation); },
        "Tensor permutation accepted a duplicate dimension.");
    requireInvalidArgument(
        [&] { (void)permutationSource.copyWithPermutedDimensions(outOfRangePermutation); },
        "Tensor permutation accepted an out-of-range dimension.");
    requireInvalidArgument(
        [&] { (void)permutationSource.copyWithPermutedDimensions(wrongRankPermutation); },
        "Tensor permutation accepted the wrong rank.");

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

    Tensor paddedTensor(ScalarType::Int32, Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1));
    paddedTensor.storeFrom({0, 0}, 4);
    paddedTensor.storeFrom({1, 1}, 9);
    require(paddedTensor.loadAs<int32_t>({0, 0}) == 4 && paddedTensor.loadAs<int32_t>({1, 1}) == 9,
            "Strided tensor layout mismatch.");
    requireInvalidArgument([&] { (void)paddedTensor.reshapeSharingStorage(Shape{4}); },
                           "Tensor reshape accepted a noncontiguous layout.");

    Tensor paddedAlias = paddedTensor;
    paddedAlias.storeFrom({0, 1}, 12);
    require(paddedTensor.loadAs<int32_t>({0, 1}) == 12, "Tensor aliases did not share storage.");
    const Tensor constPaddedAlias = paddedTensor;
    constPaddedAlias.storeFrom({1, 0}, 15);
    require(paddedTensor.loadAs<int32_t>({1, 0}) == 15,
            "Const tensor handle did not retain shallow mutability.");

    const std::array<int32_t, 3> reversedStorage{1, 2, 3};
    const Tensor reversed = Tensor::copyEncodedBackingStorage(
        ScalarType::Int32, Layout(Shape{3}, std::vector<ptrdiff_t>{-1}, 2),
        std::as_bytes(std::span<const int32_t>(reversedStorage)));
    require(reversed.loadAs<int32_t>({0}) == 3 && reversed.loadAs<int32_t>({2}) == 1,
            "Negative-stride tensor layout mismatch.");
    const Tensor reversedFloat = reversed.copyConvertedTo(ScalarType::Float32);
    require(reversedFloat.layout() == reversed.layout() &&
                reversedFloat.loadAs<float>({0}) == 3.0f &&
                reversedFloat.loadAs<float>({2}) == 1.0f,
            "Tensor conversion did not preserve the logical layout.");

    const ptrdiff_t maximumOffset = std::numeric_limits<ptrdiff_t>::max();
    const ptrdiff_t minimumOffset = std::numeric_limits<ptrdiff_t>::min();
    const std::array<size_t, 1> edgeIndex{1};
    require(Layout(Shape{2}, {maximumOffset}, minimumOffset).elementOffset(edgeIndex) == -1,
            "Layout rejected an exactly representable positive offset contribution.");
    require(Layout(Shape{2}, {minimumOffset}, maximumOffset).elementOffset(edgeIndex) == -1,
            "Layout rejected an exactly representable negative offset contribution.");

    const std::array<size_t, 1> overflowIndex{2};
    const Layout positiveMultiplyOverflow(Shape{3}, {maximumOffset});
    requireOverflow([&] { (void)positiveMultiplyOverflow.elementOffset(overflowIndex); },
                    "Layout element offset accepted positive multiplication overflow.");
    const Layout negativeMultiplyOverflow(Shape{3}, {minimumOffset});
    requireOverflow([&] { (void)negativeMultiplyOverflow.elementOffset(overflowIndex); },
                    "Layout element offset accepted negative multiplication overflow.");
    const Layout positiveAddOverflow(Shape{2}, {1}, maximumOffset);
    requireOverflow([&] { (void)positiveAddOverflow.elementOffset(edgeIndex); },
                    "Layout element offset accepted positive addition overflow.");
    const Layout negativeAddOverflow(Shape{2}, {-1}, minimumOffset);
    requireOverflow([&] { (void)negativeAddOverflow.elementOffset(edgeIndex); },
                    "Layout element offset accepted negative addition overflow.");

    requireOverflow(
        [&] { (void)storageBytesForLayout(ScalarType::Float32, positiveMultiplyOverflow); },
        "Tensor layout bounds accepted multiplication overflow.");
    requireOverflow([&] { (void)storageBytesForLayout(ScalarType::Float32, positiveAddOverflow); },
                    "Tensor layout bounds accepted addition overflow.");

    if constexpr (std::numeric_limits<size_t>::digits >= 64 &&
                  std::numeric_limits<ptrdiff_t>::digits >= 63) {
        constexpr uint64_t storageBits = 4;
        constexpr uint64_t maximumElements = std::numeric_limits<uint64_t>::max() / storageBits;
        constexpr uint64_t totalBits = maximumElements * storageBits;
        constexpr uint64_t expectedBytes =
            totalBits / 8 + static_cast<uint64_t>(totalBits % 8 != 0);
        const Layout maximumPackedLayout(Shape{static_cast<size_t>(maximumElements)}, {1});
        require(storageBytesForLayout(ScalarType::Int4, maximumPackedLayout) ==
                    static_cast<size_t>(expectedBytes),
                "Packed tensor storage byte rounding overflowed at its valid boundary.");

        const Layout overflowingPackedLayout(Shape{static_cast<size_t>(maximumElements + 1)}, {1});
        requireOverflow(
            [&] { (void)storageBytesForLayout(ScalarType::Int4, overflowingPackedLayout); },
            "Packed tensor storage accepted a bit-count multiplication overflow.");
    }

    const std::array<float, 2> conversionSource{1.1f, -2.25f};
    const Tensor convertedBFloat16 = Tensor::copyNativeValues<float>(Shape{2}, conversionSource)
                                         .copyConvertedTo(ScalarType::BFloat16);
    const Tensor convertedBack = convertedBFloat16.copyConvertedTo(ScalarType::Float32);
    requireNear(convertedBack.loadAs<float>({0}), 1.1015625f, 0.0f,
                "Tensor conversion did not apply BFloat16 rounding.");
    requireNear(convertedBack.loadAs<float>({1}), -2.25f, 0.0f,
                "Tensor conversion changed an exactly representable value.");

    const std::array<float, 6> matrixConversionSource{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const Tensor matrixSource =
        Tensor::copyNativeValues<float>(Shape{2, 3}, matrixConversionSource);
    const Layout columnMajorLayout(Shape{2, 3}, {1, 2});
    const Tensor columnMajorFloat16 =
        matrixSource.copyConvertedTo(ScalarType::Float16, columnMajorLayout);
    require(columnMajorFloat16.layout() == columnMajorLayout,
            "Tensor conversion did not preserve the requested destination layout.");
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            requireNear(columnMajorFloat16.loadAs<float>({row, column}),
                        matrixSource.loadAs<float>({row, column}), 0.0f,
                        "Tensor conversion changed a logical value in a new layout.");
    requireInvalidArgument(
        [&] {
            (void)matrixSource.copyConvertedTo(ScalarType::Float16, Layout(Shape{2, 3}, {0, 1}));
        },
        "Tensor conversion accepted overlapping destination elements.");

    Tensor rawBFloat16(ScalarType::BFloat16, Shape{2, 2});
    const std::array<std::byte, 8> sourceEncodings{
        std::byte{0xa1}, std::byte{0x7f}, std::byte{0xa2}, std::byte{0xff},
        std::byte{0x00}, std::byte{0x80}, std::byte{0x80}, std::byte{0x3f},
    };
    std::copy(sourceEncodings.begin(), sourceEncodings.end(),
              rawBFloat16.rawEncodedBackingStorage().begin());
    const Tensor relaidBFloat16 =
        rawBFloat16.copyConvertedTo(ScalarType::BFloat16, Layout(Shape{2, 2}, {1, 2}));
    const std::array<std::byte, 8> expectedRelaidEncodings{
        std::byte{0xa1}, std::byte{0x7f}, std::byte{0x00}, std::byte{0x80},
        std::byte{0xa2}, std::byte{0xff}, std::byte{0x80}, std::byte{0x3f},
    };
    require(std::equal(expectedRelaidEncodings.begin(), expectedRelaidEncodings.end(),
                       relaidBFloat16.rawEncodedBackingStorage().begin()),
            "Same-type Tensor relayout did not preserve exact logical encodings.");

    Tensor int4(ScalarType::Int4, Shape{5});
    auto int4View = int4;
    int4View.storeFrom({0}, -9);
    int4View.storeFrom({1}, -3);
    int4View.storeFrom({2}, 0);
    int4View.storeFrom({3}, 7);
    int4View.storeFrom({4}, 9);
    require(int4.rawEncodedBackingStorage().size() == 3, "Int4 packed storage size mismatch.");
    require(int4.loadAs<int32_t>({0}) == -8 && int4.loadAs<int32_t>({1}) == -3 &&
                int4.loadAs<int32_t>({3}) == 7 && int4.loadAs<int32_t>({4}) == 7,
            "Int4 packed codec mismatch.");

    // The second and fourth Float6 elements cross byte boundaries.
    Tensor float6(ScalarType::Float6E3M2, Shape{4});
    float6.storeFrom({0}, -6.0f);
    float6.storeFrom({1}, -1.0f);
    float6.storeFrom({2}, 1.0f);
    float6.storeFrom({3}, 6.0f);
    require(float6.rawEncodedBackingStorage().size() == 3, "Float6 packed storage size mismatch.");
    require(float6.loadAs<float>({0}) == -6.0f && float6.loadAs<float>({1}) == -1.0f &&
                float6.loadAs<float>({2}) == 1.0f && float6.loadAs<float>({3}) == 6.0f,
            "Float6 cross-byte codec mismatch.");

    Tensor fp4(ScalarType::Float4E2M1, Shape{4});
    fp4.storeFrom({0}, -6.0f);
    fp4.storeFrom({1}, -0.5f);
    fp4.storeFrom({2}, 1.5f);
    fp4.storeFrom({3}, 6.0f);
    const Tensor fp4Copy = fp4.copyConvertedTo(ScalarType::Float4E2M1);
    require(fp4Copy.rawEncodedBackingStorage().size() == fp4.rawEncodedBackingStorage().size() &&
                std::equal(fp4Copy.rawEncodedBackingStorage().begin(),
                           fp4Copy.rawEncodedBackingStorage().end(),
                           fp4.rawEncodedBackingStorage().begin()),
            "Same-type tensor conversion changed raw storage.");
    const Tensor fp4Float = fp4.copyConvertedTo(ScalarType::Float32);
    requireNear(fp4Float.loadAs<float>({0}), -6.0f, 0.0f,
                "Packed tensor conversion minimum mismatch.");
    requireNear(fp4Float.loadAs<float>({2}), 1.5f, 0.0f,
                "Packed tensor conversion normal mismatch.");
    requireNear(fp4.loadAs<float>({0}), -6.0f, 0.0f, "FP4 minimum mismatch.");
    requireNear(fp4.loadAs<float>({1}), -0.5f, 0.0f, "FP4 subnormal mismatch.");
    requireNear(fp4.loadAs<float>({2}), 1.5f, 0.0f, "FP4 normal mismatch.");
    requireNear(fp4.loadAs<float>({3}), 6.0f, 0.0f, "FP4 maximum mismatch.");

    Tensor fp6(ScalarType::Float6E2M3, Shape{4});
    fp6.storeFrom({0}, -7.5f);
    fp6.storeFrom({1}, -0.125f);
    fp6.storeFrom({2}, 0.875f);
    fp6.storeFrom({3}, 7.5f);
    requireNear(fp6.loadAs<float>({0}), -7.5f, 0.0f, "FP6 minimum mismatch.");
    requireNear(fp6.loadAs<float>({3}), 7.5f, 0.0f, "FP6 maximum mismatch.");

    Tensor float16(ScalarType::Float16, Shape{2});
    float16.storeFrom({0}, 1.5f);
    float16.storeFrom({1}, -0.25f);
    requireNear(float16.loadAs<float>({0}), 1.5f, 0.0f, "Float16 codec mismatch.");
    requireNear(float16.loadAs<float>({1}), -0.25f, 0.0f, "Float16 codec mismatch.");

    Tensor bfloat16(ScalarType::BFloat16, Shape{1});
    bfloat16.storeFrom({0}, 1.25f);
    requireNear(bfloat16.loadAs<float>({0}), 1.25f, 0.01f, "BFloat16 codec mismatch.");

    Tensor complex(ScalarType::ComplexFloat32, Shape{1});
    complex.storeFrom({0}, std::complex<float>(2.0f, -3.0f));
    require(complex.loadAs<std::complex<float>>({0}) == std::complex<float>(2.0f, -3.0f),
            "Complex codec mismatch.");

    Tensor float8(ScalarType::Float8E4M3, Shape{1});
    float8.storeFrom({0}, 1.25f);
    requireNear(float8.loadAs<float>({0}), 1.25f, 0.0f, "Float8 codec mismatch.");

    return 0;
}
