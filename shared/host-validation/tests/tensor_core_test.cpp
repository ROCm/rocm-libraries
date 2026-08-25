// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/host_validation/tensor.hpp>
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
    using namespace roc::host_validation;

    static_assert(
        std::is_same_v<decltype(std::declval<const Tensor&>().storage()), std::span<std::byte>>);

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
    const Layout rowMajor = Layout::contiguousLastDimensionFastest(shape);
    const Layout columnMajor = Layout::contiguousFirstDimensionFastest(shape);
    require(rowMajor == Layout::contiguous(shape),
            "Contiguous compatibility API changed dimension order.");
    require(rowMajor.rank() == 2 && rowMajor.dimensions().size() == 2 && rowMajor.extent(0) == 2 &&
                rowMajor.elementCount() == 6 && rowMajor.elementCount(0, 1) == 2 &&
                rowMajor.elementCountExcluding(0) == 3 && rowMajor.stride(0) == 3 &&
                rowMajor.stride(1) == 1,
            "Last-dimension-fastest layout helper mismatch.");
    require(columnMajor.stride(0) == 1 && columnMajor.stride(1) == 2,
            "First-dimension-fastest layout helper mismatch.");

    Tensor tensor(ScalarType::Float32, shape);
    tensor.storeFrom({1, 2}, 7.0f);
    require(tensor.loadAs<float>({1, 2}) == 7.0f, "Owning tensor view mismatch.");
    require(tensor.layout().strides()[0] == 3 && tensor.layout().strides()[1] == 1,
            "Contiguous tensor strides mismatch.");
    require(tensor.storage().size() == 6 * sizeof(float), "Float32 storage size mismatch.");

    const std::array<float, 3> nativeValues{2.0f, 4.0f, 6.0f};
    const Tensor nativeTensor =
        Tensor::fromNativeValues<float>(Shape{3}, std::span<const float>(nativeValues));
    require(nativeTensor.type() == ScalarType::Float32 && nativeTensor.loadAs<float>({2}) == 6.0f,
            "Native tensor factory mismatch.");

    size_t allocatorCalls = 0;
    TensorStorageAllocator allocator = [&allocatorCalls](size_t bytes) {
        ++allocatorCalls;
        return TensorStorage::allocate(bytes + 16);
    };
    Tensor allocated(ScalarType::Float32, Shape{2}, allocator);
    require(allocatorCalls == 1 && allocated.storage().size() == 2 * sizeof(float) + 16,
            "Tensor storage allocator contract mismatch.");
    allocated.storeFrom({1}, 5.0f);
    Tensor allocatedClone = allocated.clone(allocator);
    allocatedClone.storeFrom({1}, 9.0f);
    require(allocatorCalls == 2 && allocated.loadAs<float>({1}) == 5.0f &&
                allocatedClone.loadAs<float>({1}) == 9.0f,
            "Allocator-backed Tensor clone mismatch.");

    Tensor uninitialized(ScalarType::Float32, Shape{2}, TensorStorage::allocateUninitialized);
    uninitialized.storeFrom({0}, 3.0f);
    uninitialized.storeFrom({1}, 7.0f);
    require(uninitialized.loadAs<float>({0}) == 3.0f && uninitialized.loadAs<float>({1}) == 7.0f,
            "Uninitialized Tensor storage did not retain written values.");

    Tensor copied = tensor;
    copied.storeFrom({1, 2}, 11.0f);
    require(tensor.loadAs<float>({1, 2}) == 11.0f,
            "Copying a tensor did not preserve shared-storage semantics.");
    Tensor cloned = tensor.clone();
    cloned.storeFrom({1, 2}, 13.0f);
    require(tensor.loadAs<float>({1, 2}) == 11.0f && cloned.loadAs<float>({1, 2}) == 13.0f,
            "Tensor clone did not deep-copy storage.");

    Tensor reshapeSource(ScalarType::Int32, Shape{2, 3});
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            reshapeSource.storeFrom({row, column}, static_cast<int32_t>(row * 3 + column));
    Tensor reshaped = reshapeSource.reshape(Shape{3, 2});
    require(reshaped.shape() == Shape{3, 2} && reshaped.loadAs<int32_t>({1, 1}) == 3,
            "Tensor reshape changed logical linear order.");
    reshaped.storeFrom({2, 1}, 19);
    require(reshapeSource.loadAs<int32_t>({1, 2}) == 19,
            "Tensor reshape did not return a shallow alias.");
    requireInvalidArgument([&] { (void)reshapeSource.reshape(Shape{7}); },
                           "Tensor reshape accepted a different element count.");

    Tensor packedPaddingSource(ScalarType::Int4, Shape{3});
    packedPaddingSource.storeFrom({0}, -8);
    packedPaddingSource.storeFrom({1}, -3);
    packedPaddingSource.storeFrom({2}, 7);
    const Tensor packedPadded = packedPaddingSource.pad(Shape{5});
    require(packedPadded.shape() == Shape{5} && packedPadded.storage().size() == 3 &&
                packedPadded.loadAs<int32_t>({0}) == -8 &&
                packedPadded.loadAs<int32_t>({1}) == -3 && packedPadded.loadAs<int32_t>({2}) == 7 &&
                packedPadded.loadAs<int32_t>({3}) == 0 && packedPadded.loadAs<int32_t>({4}) == 0,
            "Tensor padding did not preserve packed values and zero-fill new elements.");
    requireInvalidArgument([&] { (void)packedPaddingSource.pad(Shape{2}); },
                           "Tensor padding accepted a shrinking shape.");
    requireInvalidArgument([&] { (void)packedPaddingSource.pad(Shape{3, 1}); },
                           "Tensor padding accepted a different rank.");

    Tensor permutationSource(ScalarType::Int12, Shape{2, 3});
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            permutationSource.storeFrom({row, column}, static_cast<int32_t>(100 * row + column));
    const std::array<size_t, 2> transpose{1, 0};
    const Tensor permutationResult = permutationSource.permute(transpose);
    require(permutationResult.shape() == Shape{3, 2} && permutationResult.storage().size() == 9,
            "Tensor permutation produced the wrong packed output geometry.");
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            require(permutationResult.loadAs<int32_t>({column, row}) ==
                        permutationSource.loadAs<int32_t>({row, column}),
                    "Tensor permutation changed a packed value.");
    const std::array<size_t, 2> duplicatePermutation{0, 0};
    const std::array<size_t, 2> outOfRangePermutation{0, 2};
    const std::array<size_t, 1> wrongRankPermutation{0};
    requireInvalidArgument([&] { (void)permutationSource.permute(duplicatePermutation); },
                           "Tensor permutation accepted a duplicate dimension.");
    requireInvalidArgument([&] { (void)permutationSource.permute(outOfRangePermutation); },
                           "Tensor permutation accepted an out-of-range dimension.");
    requireInvalidArgument([&] { (void)permutationSource.permute(wrongRankPermutation); },
                           "Tensor permutation accepted the wrong rank.");

    Tensor packedCopySource(ScalarType::Int4, Shape{4});
    packedCopySource.storeFrom({0}, -8);
    packedCopySource.storeFrom({1}, -3);
    packedCopySource.storeFrom({2}, 2);
    packedCopySource.storeFrom({3}, 7);
    Tensor packedCopyDestination(ScalarType::Int4, Shape{4});
    for (size_t index = 0; index < packedCopyDestination.size(); ++index)
        packedCopyDestination.storeFrom({index}, 1);
    const std::array<size_t, 2> selectedCopyIndices{1, 3};
    packedCopyDestination.copyFrom(packedCopySource, selectedCopyIndices);
    require(packedCopyDestination.loadAs<int32_t>({0}) == 1 &&
                packedCopyDestination.loadAs<int32_t>({1}) == -3 &&
                packedCopyDestination.loadAs<int32_t>({2}) == 1 &&
                packedCopyDestination.loadAs<int32_t>({3}) == 7,
            "Selected Tensor copy changed unselected packed elements.");

    std::array<std::byte, 2> packedCopyBytes{
        static_cast<std::byte>(0x11),
        static_cast<std::byte>(0x11),
    };
    packedCopySource.copyTo(packedCopyBytes, selectedCopyIndices);
    const Tensor selectedPackedCopy(
        ScalarType::Int4, Layout::contiguous(Shape{4}),
        std::span<std::byte>(packedCopyBytes.data(), packedCopyBytes.size()));
    require(selectedPackedCopy.loadAs<int32_t>({0}) == 1 &&
                selectedPackedCopy.loadAs<int32_t>({1}) == -3 &&
                selectedPackedCopy.loadAs<int32_t>({2}) == 1 &&
                selectedPackedCopy.loadAs<int32_t>({3}) == 7,
            "Selected Tensor copy-to changed unselected packed elements.");

    std::array<std::byte, 2> fullPackedCopy{};
    packedCopySource.copyTo(fullPackedCopy);
    require(std::equal(fullPackedCopy.begin(), fullPackedCopy.end(),
                       packedCopySource.storage().begin()),
            "Full Tensor copy-to changed packed storage.");

    const Layout offsetLayout(Shape{2}, {1}, 2);
    Tensor offsetCopySource(ScalarType::Int32, offsetLayout);
    std::fill(offsetCopySource.storage().begin(), offsetCopySource.storage().end(),
              static_cast<std::byte>(0x7f));
    offsetCopySource.storeFrom({0}, 3);
    offsetCopySource.storeFrom({1}, 5);
    std::array<std::byte, 4 * sizeof(int32_t)> offsetCopyDestination{};
    offsetCopySource.copyTo(offsetCopyDestination);
    require(std::all_of(offsetCopyDestination.begin(),
                        offsetCopyDestination.begin() + 2 * sizeof(int32_t),
                        [](std::byte value) { return value == std::byte{0}; }),
            "Tensor copy-to changed storage before the layout's first element.");
    const Tensor offsetCopyResult(
        ScalarType::Int32, offsetLayout,
        std::span<std::byte>(offsetCopyDestination.data(), offsetCopyDestination.size()));
    require(
        offsetCopyResult.loadAs<int32_t>({0}) == 3 && offsetCopyResult.loadAs<int32_t>({1}) == 5,
        "Tensor copy-to missed an offset layout value.");

    std::array<std::byte, 1> undersizedCopyDestination{};
    requireInvalidArgument([&] { packedCopySource.copyTo(undersizedCopyDestination); },
                           "Full Tensor copy-to accepted undersized storage.");
    requireInvalidArgument(
        [&] { packedCopySource.copyTo(undersizedCopyDestination, selectedCopyIndices); },
        "Selected Tensor copy-to accepted undersized storage.");

    Tensor paddedTensor(ScalarType::Int32, Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1));
    paddedTensor.storeFrom({0, 0}, 4);
    paddedTensor.storeFrom({1, 1}, 9);
    require(paddedTensor.loadAs<int32_t>({0, 0}) == 4 && paddedTensor.loadAs<int32_t>({1, 1}) == 9,
            "Strided tensor layout mismatch.");
    requireInvalidArgument([&] { (void)paddedTensor.reshape(Shape{4}); },
                           "Tensor reshape accepted a noncontiguous layout.");

    Tensor paddedAlias = paddedTensor;
    paddedAlias.storeFrom({0, 1}, 12);
    require(paddedTensor.loadAs<int32_t>({0, 1}) == 12, "Tensor aliases did not share storage.");
    const Tensor constPaddedAlias = paddedTensor;
    constPaddedAlias.storeFrom({1, 0}, 15);
    require(paddedTensor.loadAs<int32_t>({1, 0}) == 15,
            "Const tensor handle did not retain shallow mutability.");

    const std::array<int32_t, 3> reversedStorage{1, 2, 3};
    const Tensor reversed(ScalarType::Int32, Layout(Shape{3}, std::vector<ptrdiff_t>{-1}, 2),
                          std::as_bytes(std::span<const int32_t>(reversedStorage)));
    require(reversed.loadAs<int32_t>({0}) == 3 && reversed.loadAs<int32_t>({2}) == 1,
            "Negative-stride tensor layout mismatch.");
    const Tensor reversedFloat = reversed.to(ScalarType::Float32);
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
    const Tensor convertedBFloat16 =
        Tensor::fromNativeValues<float>(Shape{2}, conversionSource).to(ScalarType::BFloat16);
    const Tensor convertedBack = convertedBFloat16.to(ScalarType::Float32);
    requireNear(convertedBack.loadAs<float>({0}), 1.1015625f, 0.0f,
                "Tensor conversion did not apply BFloat16 rounding.");
    requireNear(convertedBack.loadAs<float>({1}), -2.25f, 0.0f,
                "Tensor conversion changed an exactly representable value.");

    Tensor int4(ScalarType::Int4, Shape{5});
    auto int4View = int4;
    int4View.storeFrom({0}, -9);
    int4View.storeFrom({1}, -3);
    int4View.storeFrom({2}, 0);
    int4View.storeFrom({3}, 7);
    int4View.storeFrom({4}, 9);
    require(int4.storage().size() == 3, "Int4 packed storage size mismatch.");
    require(int4.loadAs<int32_t>({0}) == -8 && int4.loadAs<int32_t>({1}) == -3 &&
                int4.loadAs<int32_t>({3}) == 7 && int4.loadAs<int32_t>({4}) == 7,
            "Int4 packed codec mismatch.");

    // Int12 is intentional generality coverage for cross-byte packed scalar storage.
    Tensor int12(ScalarType::Int12, Shape{2});
    int12.storeFrom({0}, -2048);
    int12.storeFrom({1}, 2047);
    require(scalarTypeInfo(ScalarType::Int12).isPacked() && int12.storage().size() == 3,
            "Int12 packed storage size mismatch.");
    require(int12.loadAs<int32_t>({0}) == -2048 && int12.loadAs<int32_t>({1}) == 2047,
            "Int12 cross-byte codec mismatch.");

    Tensor fp4(ScalarType::Float4E2M1, Shape{4});
    fp4.storeFrom({0}, -6.0f);
    fp4.storeFrom({1}, -0.5f);
    fp4.storeFrom({2}, 1.5f);
    fp4.storeFrom({3}, 6.0f);
    const Tensor fp4Copy = fp4.to(ScalarType::Float4E2M1);
    require(
        fp4Copy.storage().size() == fp4.storage().size() &&
            std::equal(fp4Copy.storage().begin(), fp4Copy.storage().end(), fp4.storage().begin()),
        "Same-type tensor conversion changed raw storage.");
    const Tensor fp4Float = fp4.to(ScalarType::Float32);
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
