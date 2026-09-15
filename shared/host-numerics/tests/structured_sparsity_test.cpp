// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testStructuredSparsity() {
    using namespace roc::host_numerics;

    std::array<float, 20> inputStorage;
    inputStorage.fill(-99);
    Tensor input =
        Tensor::copyEncodedBackingStorage(ScalarType::Float32, Layout(Shape{2, 8}, {10, 1}),
                                          std::as_writable_bytes(std::span<float>(inputStorage)));
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 8; ++column)
            input.storeFrom({row, column}, static_cast<float>(1 + row * 8 + column));

    std::array<float, 20> prunedStorage;
    prunedStorage.fill(-7);
    Tensor pruned =
        Tensor::copyEncodedBackingStorage(ScalarType::Float32, Layout(Shape{2, 8}, {10, 1}),
                                          std::as_writable_bytes(std::span<float>(prunedStorage)));
    std::array<float, 8> compressedStorage{};
    Tensor compressed = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 4}), std::span<float>(compressedStorage));
    std::array<uint8_t, 8> indexStorage{};
    Tensor retainedIndices = Tensor::copyNativeStorage<uint8_t>(
        Layout::contiguousLastDimensionFastest(Shape{2, 4}), std::span<uint8_t>(indexStorage));

    StructuredSparsityPattern pattern;
    pattern.axis = 1;
    pattern.fixedPositions = {1, 3};
    applyStructuredSparsityInto(
        input, {.pruned = pruned, .compressed = compressed, .retainedIndices = retainedIndices},
        pattern);

    for (size_t row = 0; row < 2; ++row) {
        for (size_t group = 0; group < 2; ++group) {
            const size_t inputBase = group * 4;
            const size_t compressedBase = group * 2;
            require(pruned.loadAs<float>({row, inputBase}) == 0 &&
                        pruned.loadAs<float>({row, inputBase + 1}) ==
                            input.loadAs<float>({row, inputBase + 1}) &&
                        pruned.loadAs<float>({row, inputBase + 2}) == 0 &&
                        pruned.loadAs<float>({row, inputBase + 3}) ==
                            input.loadAs<float>({row, inputBase + 3}),
                    "Structured sparsity pruned output mismatch.");
            require(compressed.loadAs<float>({row, compressedBase}) ==
                            input.loadAs<float>({row, inputBase + 1}) &&
                        compressed.loadAs<float>({row, compressedBase + 1}) ==
                            input.loadAs<float>({row, inputBase + 3}),
                    "Structured sparsity compressed output mismatch.");
            require(retainedIndices.loadAs<uint8_t>({row, compressedBase}) == 1 &&
                        retainedIndices.loadAs<uint8_t>({row, compressedBase + 1}) == 3,
                    "Structured sparsity retained-index output mismatch.");
        }
    }

    Tensor metadata(ScalarType::UInt8, Shape{2, 1});
    encodeTwoOfFourMetadataInto(retainedIndices, metadata, 1);
    require(metadata.loadAs<uint8_t>({0, 0}) == 0xdd && metadata.loadAs<uint8_t>({1, 0}) == 0xdd,
            "Two-of-four metadata encoding mismatch.");

    Tensor fusedPruned(ScalarType::Float32, Shape{2, 8});
    Tensor fusedCompressed(ScalarType::Float32, Shape{2, 4});
    Tensor fusedMetadata(ScalarType::UInt8, Shape{2, 1});
    const StructuredSparseTensor fusedOutputs{
        .pruned = fusedPruned,
        .compressed = fusedCompressed,
        .twoOfFourMetadata = fusedMetadata,
    };
    applyStructuredSparsityInto(input, fusedOutputs, pattern, {.firstSlice = 0, .sliceCount = 1});
    applyStructuredSparsityInto(input, fusedOutputs, pattern, {.firstSlice = 1, .sliceCount = 1});
    require(compare(fusedMetadata, metadata).passed(),
            "Fused structured sparsity metadata mismatch.");

    Tensor inPlace =
        Tensor::copyNativeValues<float>(Shape{8}, std::array<float, 8>{1, 2, 3, 4, 5, 6, 7, 8});
    Tensor inPlaceCompressed(ScalarType::Float32, Shape{4});
    Tensor inPlaceIndices(ScalarType::UInt8, Shape{4});
    pattern.axis = 0;
    pattern.fixedPositions = {0, 2};
    applyStructuredSparsityInto(
        inPlace,
        {.pruned = inPlace, .compressed = inPlaceCompressed, .retainedIndices = inPlaceIndices},
        pattern);
    require(inPlace.loadAs<float>({0}) == 1 && inPlace.loadAs<float>({1}) == 0 &&
                inPlace.loadAs<float>({2}) == 3 && inPlace.loadAs<float>({3}) == 0,
            "In-place structured sparsity mismatch.");

    const std::array<float, 12> ownedValues{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    StructuredSparsityPattern ownedPattern;
    ownedPattern.axis = 0;
    ownedPattern.fixedPositions = {0, 2};
    const Tensor ownedInput =
        Tensor::copyNativeValues<float>(Shape{12}, std::span<const float>(ownedValues));
    const StructuredSparseTensor owned = applyStructuredSparsity(
        ownedInput, ownedPattern, {.retainedIndices = true, .twoOfFourMetadata = true});
    require(
        owned.pruned.layout() == Layout::contiguousLastDimensionFastest(Shape{12}) &&
            owned.compressed.layout() == Layout::contiguousLastDimensionFastest(Shape{6}) &&
            owned.retainedIndices &&
            owned.retainedIndices->layout() == Layout::contiguousLastDimensionFastest(Shape{6}) &&
            owned.twoOfFourMetadata &&
            owned.twoOfFourMetadata->layout() == Layout::contiguousLastDimensionFastest(Shape{2}),
        "Owning structured-sparsity result contract mismatch.");
    require(owned.twoOfFourMetadata->loadAs<uint8_t>({0}) == 0x88 &&
                owned.twoOfFourMetadata->loadAs<uint8_t>({1}) == 0x08,
            "Owning structured-sparsity metadata retained unexpected bits.");

    StructuredSparsityPattern invalidMetadataPattern = ownedPattern;
    invalidMetadataPattern.retainedElements = 1;
    invalidMetadataPattern.fixedPositions = {0};
    bool rejectedBeforeAllocation = false;
    try {
        (void)applyStructuredSparsity(ownedInput, invalidMetadataPattern,
                                      {.retainedIndices = false, .twoOfFourMetadata = true});
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation,
            "Owning structured sparsity accepted an invalid metadata policy.");

    const Tensor ownedMetadata = encodeTwoOfFourMetadata(*owned.retainedIndices, 0);
    require(ownedMetadata.shape() == Shape{2} && ownedMetadata.loadAs<uint8_t>({0}) == 0x88 &&
                ownedMetadata.loadAs<uint8_t>({1}) == 0x08,
            "Owning two-of-four metadata result contract mismatch.");

    for (const auto& [shape, axis] :
         std::array<std::pair<Shape, size_t>, 2>{{{Shape{0, 8}, 0}, {Shape{2, 0}, 1}}}) {
        StructuredSparsityPattern emptyPattern;
        emptyPattern.axis = axis;
        const StructuredSparseTensor empty =
            applyStructuredSparsity(Tensor(ScalarType::Float32, shape), emptyPattern,
                                    {.retainedIndices = true, .twoOfFourMetadata = true});
        require(empty.pruned.shape() == shape && empty.pruned.elementCount() == 0 &&
                    empty.compressed.elementCount() == 0 && empty.retainedIndices &&
                    empty.retainedIndices->elementCount() == 0 && empty.twoOfFourMetadata &&
                    empty.twoOfFourMetadata->elementCount() == 0,
                "Structured sparsity did not preserve a zero extent as empty work.");
    }

    const std::array<uint8_t, 2> invalidRetainedValues{2, 1};
    const Tensor invalidRetained = Tensor::copyNativeValues<uint8_t>(
        Shape{2}, std::span<const uint8_t>(invalidRetainedValues));
    rejectedBeforeAllocation = false;
    try {
        (void)encodeTwoOfFourMetadata(invalidRetained, 0);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation,
            "Owning two-of-four metadata accepted invalid retained positions.");

    StructuredSparsityPattern widePattern;
    widePattern.axis = 0;
    widePattern.groupSize = 257;
    widePattern.retainedElements = 1;
    widePattern.fixedPositions = {0};
    const StructuredSparseTensor wide =
        applyStructuredSparsity(Tensor(ScalarType::Float32, Shape{257}), widePattern);
    require(wide.compressed.shape() == Shape{1} && !wide.retainedIndices,
            "Structured sparsity imposed the UInt8 index limit without an index output.");

    Tensor overlappingPrunedA(ScalarType::Float32, Layout(Shape{2, 8}, {1, 1}));
    Tensor overlappingPrunedB(ScalarType::Float32, Layout(Shape{2, 8}, {1, 1}));
    Tensor independentCompressedA(ScalarType::Float32, Shape{2, 4});
    Tensor independentCompressedB(ScalarType::Float32, Shape{2, 4});
    StructuredSparsityPattern overlappingPattern;
    overlappingPattern.axis = 1;
    overlappingPattern.fixedPositions = {1, 3};
    applyStructuredSparsityInto(
        input, {.pruned = overlappingPrunedA, .compressed = independentCompressedA},
        overlappingPattern);
    applyStructuredSparsityInto(
        input, {.pruned = overlappingPrunedB, .compressed = independentCompressedB},
        overlappingPattern);
    require(std::ranges::equal(overlappingPrunedA.rawEncodedBackingStorage(),
                               overlappingPrunedB.rawEncodedBackingStorage()) &&
                compare(independentCompressedA, independentCompressedB).passed(),
            "Structured sparsity was nondeterministic for overlapping nonzero strides.");
}

}  // namespace host_numerics_test
