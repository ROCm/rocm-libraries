// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testReferenceOperationAliasing() {
    using namespace roc::host_numerics;

    Tensor arithmeticInPlace =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f});
    const Tensor arithmeticY =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{3.0f, 4.0f});
    multiplyInto(arithmeticInPlace, Tensor(2.0f), arithmeticInPlace, ScalarType::Float32);
    addInto(arithmeticInPlace, arithmeticY, arithmeticInPlace, ScalarType::Float32);
    require(arithmeticInPlace.loadAs<float>({0}) == 5.0f &&
                arithmeticInPlace.loadAs<float>({1}) == 8.0f,
            "Tensor arithmetic rejected or corrupted an exact in-place mapping.");

    Tensor arithmeticOverlap =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{11.0f, 22.0f});
    Tensor arithmeticReversedOutput =
        arithmeticOverlap.shareStorageWithLayout(Layout(Shape{2}, {-1}, 1));
    const std::vector<std::byte> arithmeticOverlapBefore =
        copyRawEncodedBackingStorage(arithmeticOverlap);
    requireInvalidArgument(
        [&] {
            multiplyInto(arithmeticOverlap, Tensor(1.0f), arithmeticReversedOutput,
                         ScalarType::Float32);
        },
        "Tensor multiplication accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        arithmeticOverlap, arithmeticOverlapBefore,
        "Tensor multiplication modified destination storage before rejecting overlap.");

    const Tensor distinctLinearCombinationInput =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f});
    Tensor selfCollidingLinearCombinationOutput(ScalarType::Float32, Layout(Shape{2}, {0}));
    selfCollidingLinearCombinationOutput.storeFrom({0}, 17.0f);
    const std::vector<std::byte> selfCollidingLinearCombinationOutputBefore =
        copyRawEncodedBackingStorage(selfCollidingLinearCombinationOutput);
    requireInvalidArgument(
        [&] {
            multiplyInto(distinctLinearCombinationInput, Tensor(1.0f),
                         selfCollidingLinearCombinationOutput, ScalarType::Float32);
        },
        "Tensor multiplication accepted a self-colliding destination.");
    requireRawEncodedBackingStorageEquals(
        selfCollidingLinearCombinationOutput, selfCollidingLinearCombinationOutputBefore,
        "Tensor multiplication modified a self-colliding destination before rejecting it.");

    Tensor softmaxInPlace =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f});
    referenceSoftmaxInto(softmaxInPlace, softmaxInPlace, 0);
    require(std::abs(softmaxInPlace.loadAs<float>({0}) + softmaxInPlace.loadAs<float>({1}) - 1.0f) <
                    1e-6f &&
                softmaxInPlace.loadAs<float>({0}) < softmaxInPlace.loadAs<float>({1}),
            "Reference softmax rejected or corrupted an exact in-place mapping.");

    Tensor softmaxOverlap =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f});
    Tensor softmaxReversedOutput = softmaxOverlap.shareStorageWithLayout(Layout(Shape{2}, {-1}, 1));
    const std::vector<std::byte> softmaxOverlapBefore =
        copyRawEncodedBackingStorage(softmaxOverlap);
    requireInvalidArgument([&] { referenceSoftmaxInto(softmaxOverlap, softmaxReversedOutput, 0); },
                           "Reference softmax accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        softmaxOverlap, softmaxOverlapBefore,
        "Reference softmax modified destination storage before rejecting overlap.");

    Tensor layerNormInPlace =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{1.0f, 3.0f});
    LayerNormOptions layerNormOptions;
    layerNormOptions.axis = 1;
    referenceLayerNormInto(layerNormInPlace, {.output = layerNormInPlace}, layerNormOptions);
    require(layerNormInPlace.loadAs<float>({0, 0}) < 0.0f &&
                layerNormInPlace.loadAs<float>({0, 1}) > 0.0f &&
                std::abs(layerNormInPlace.loadAs<float>({0, 0}) +
                         layerNormInPlace.loadAs<float>({0, 1})) < 1e-6f,
            "Reference LayerNorm rejected or corrupted an exact in-place mapping.");

    Tensor layerNormOverlap =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{1.0f, 3.0f});
    Tensor layerNormReversedOutput =
        layerNormOverlap.shareStorageWithLayout(Layout(Shape{1, 2}, {2, -1}, 1));
    const std::vector<std::byte> layerNormOverlapBefore =
        copyRawEncodedBackingStorage(layerNormOverlap);
    requireInvalidArgument(
        [&] {
            referenceLayerNormInto(layerNormOverlap, {.output = layerNormReversedOutput},
                                   layerNormOptions);
        },
        "Reference LayerNorm accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        layerNormOverlap, layerNormOverlapBefore,
        "Reference LayerNorm modified destination storage before rejecting overlap.");

    Tensor reductionInPlace =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{5.0f, 7.0f});
    referenceSumInto(reductionInPlace, reductionInPlace, {}, ScalarType::Float32);
    require(
        reductionInPlace.loadAs<float>({0}) == 5.0f && reductionInPlace.loadAs<float>({1}) == 7.0f,
        "Reference reduction rejected or corrupted an exact element mapping.");

    Tensor reductionOverlap =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{5.0f, 7.0f});
    Tensor reductionReversedOutput =
        reductionOverlap.shareStorageWithLayout(Layout(Shape{2}, {-1}, 1));
    const std::vector<std::byte> reductionOverlapBefore =
        copyRawEncodedBackingStorage(reductionOverlap);
    requireInvalidArgument(
        [&] {
            referenceSumInto(reductionOverlap, reductionReversedOutput, {}, ScalarType::Float32);
        },
        "Reference reduction accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        reductionOverlap, reductionOverlapBefore,
        "Reference reduction modified destination storage before rejecting overlap.");

    Tensor epilogueInPlace =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{1.0f, -2.0f});
    EpilogueOptions epilogueInPlaceOptions;
    epilogueInPlaceOptions.outputScale = 2.0;
    referenceEpilogueInto(epilogueInPlace, {.output = epilogueInPlace}, epilogueInPlaceOptions);
    require(epilogueInPlace.loadAs<float>({0, 0}) == 2.0f &&
                epilogueInPlace.loadAs<float>({0, 1}) == -4.0f,
            "Reference epilogue rejected or corrupted an exact in-place mapping.");

    Tensor epilogueOverlap =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{1.0f, -2.0f});
    Tensor epilogueReversedOutput =
        epilogueOverlap.shareStorageWithLayout(Layout(Shape{1, 2}, {2, -1}, 1));
    const std::vector<std::byte> epilogueOverlapBefore =
        copyRawEncodedBackingStorage(epilogueOverlap);
    requireInvalidArgument(
        [&] { referenceEpilogueInto(epilogueOverlap, {.output = epilogueReversedOutput}); },
        "Reference epilogue accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        epilogueOverlap, epilogueOverlapBefore,
        "Reference epilogue modified destination storage before rejecting overlap.");

    const Tensor epilogueInput =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{3.0f, 4.0f});
    Tensor overlappingEpilogueOutputs =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{19.0f, 23.0f});
    const std::vector<std::byte> overlappingEpilogueOutputsBefore =
        copyRawEncodedBackingStorage(overlappingEpilogueOutputs);
    requireInvalidArgument(
        [&] {
            referenceEpilogueInto(epilogueInput, {.output = overlappingEpilogueOutputs,
                                                  .rawOutput = overlappingEpilogueOutputs});
        },
        "Reference epilogue accepted overlapping result tensors.");
    requireRawEncodedBackingStorageEquals(
        overlappingEpilogueOutputs, overlappingEpilogueOutputsBefore,
        "Reference epilogue modified result storage before rejecting overlapping outputs.");
}

}  // namespace host_numerics_test
