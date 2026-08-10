// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <roc/host_validation/validation.hpp>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void testRuntimeReferenceGemm() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    std::array<float, 4> d{};
    const std::array<float, 2> bias{1, -10000};
    const std::array<float, 2> scaleA{2, 3};
    const std::array<float, 2> scaleB{5, 7};

    GemmProblem problem(
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a))),
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b))),
        TensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<float>(d)),
        ScalarType::Float32);
    problem.epilogue.beta = 1.0;
    problem.epilogue.bias = VectorBinding{
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(bias)),
        MatrixAxis::Row,
    };
    problem.epilogue.scaleA =
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(scaleA));
    problem.epilogue.scaleB =
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(scaleB));
    problem.epilogue.activation = Activation::Relu;

    GemmInvocation invocation(std::move(problem));
    require(static_cast<bool>(queryGemmSupport(invocation)),
            "Runtime reference GEMM invocation support mismatch.");
    const GemmResult run = referenceGemm(invocation);
    require(run.backendUsed == GemmBackend::Canonical && run.outputElementsComputed == 4,
            "Runtime reference GEMM run information mismatch.");

    const std::array<float, 4> expected{
        58 * 2 * 5 + 1 + 1,
        0,
        64 * 2 * 7 + 1 + 1,
        0,
    };
    require(compare(std::span<const float>(d), std::span<const float>(expected)).passed(),
            "Runtime reference GEMM result mismatch.");

    invocation.execution.backend = GemmBackend::Tiled;
    require(!queryGemmSupport(invocation),
            "Runtime reference GEMM invocation unexpectedly supports a missing backend.");
    const GemmResult fallback = referenceGemm(invocation);
    require(fallback.backendUsed == GemmBackend::Canonical && fallback.fallbackReason.has_value(),
            "Runtime reference GEMM backend fallback mismatch.");
}

void testRuntimeMixedAndBlockScaledGemm() {
    using namespace roc::host_validation;

    const std::array<float, 2> aValues{1.25f, 2.5f};
    const std::array<float, 2> bValues{2.0f, 3.0f};
    const std::array<float, 1> cValues{1.0f};
    Tensor a =
        Tensor::fromValues(ScalarType::Float8E4M3, Shape{1, 2}, std::span<const float>(aValues));
    Tensor b =
        Tensor::fromValues(ScalarType::Float8E5M2, Shape{2, 1}, std::span<const float>(bValues));
    Tensor c =
        Tensor::fromValues(ScalarType::BFloat16, Shape{1, 1}, std::span<const float>(cValues));
    Tensor d(ScalarType::Float16, Shape{1, 1});

    GemmOperand operandA(a.view());
    operandA.computeType = ScalarType::Float4E2M1;
    GemmProblem mixed(std::move(operandA), GemmOperand(b.view()), c.view(), d.mutableView(),
                      ScalarType::Float32);
    mixed.epilogue.beta = 1.0;
    referenceGemm(mixed);
    require(d.view().loadAs<float>({0, 0}) == 9.0f, "Runtime mixed-type GEMM result mismatch.");

    const std::array<float, 4> ones{1, 1, 1, 1};
    const std::array<float, 1> zero{0};
    const std::array<float, 2> scaleAValues{2, 4};
    const std::array<float, 2> scaleBValues{8, 16};
    Tensor blockA =
        Tensor::fromValues(ScalarType::Float32, Shape{1, 4}, std::span<const float>(ones));
    Tensor blockB =
        Tensor::fromValues(ScalarType::Float32, Shape{4, 1}, std::span<const float>(ones));
    Tensor blockC =
        Tensor::fromValues(ScalarType::Float32, Shape{1, 1}, std::span<const float>(zero));
    Tensor blockD(ScalarType::Float32, Shape{1, 1});
    Tensor scalesA =
        Tensor::fromValues(ScalarType::E8M0, Shape{1, 2}, std::span<const float>(scaleAValues));
    Tensor scalesB =
        Tensor::fromValues(ScalarType::E8M0, Shape{1, 2}, std::span<const float>(scaleBValues));

    GemmOperand blockOperandA(blockA.view());
    GemmOperand blockOperandB(blockB.view());
    blockOperandA.blockScale = BlockScaleBinding{scalesA.view(), 2};
    blockOperandB.blockScale = BlockScaleBinding{scalesB.view(), 2};
    GemmProblem blockScaled(std::move(blockOperandA), std::move(blockOperandB), blockC.view(),
                            blockD.mutableView(), ScalarType::Float32);
    referenceGemm(blockScaled);
    require(blockD.view().loadAs<float>({0, 0}) == 2 * 2 * 8 + 2 * 4 * 16,
            "Runtime block-scaled GEMM result mismatch.");
}

void testRuntimeComplexAndExplicitAxisGemm() {
    using namespace roc::host_validation;

    const std::array<std::complex<float>, 1> complexA{std::complex<float>(1.0f, 2.0f)};
    const std::array<std::complex<float>, 1> complexB{std::complex<float>(3.0f, 4.0f)};
    const std::array<std::complex<float>, 1> complexC{};
    std::array<std::complex<float>, 1> complexD{};

    GemmOperand complexOperandA(TensorView::fromNative<std::complex<float>>(
        Layout::contiguous(Shape{1, 1}), std::span<const std::complex<float>>(complexA)));
    complexOperandA.conjugate = true;
    GemmProblem complexProblem(
        std::move(complexOperandA),
        GemmOperand(TensorView::fromNative<std::complex<float>>(
            Layout::contiguous(Shape{1, 1}), std::span<const std::complex<float>>(complexB))),
        TensorView::fromNative<std::complex<float>>(Layout::contiguous(Shape{1, 1}),
                                                    std::span<const std::complex<float>>(complexC)),
        MutableTensorView::fromNative<std::complex<float>>(
            Layout::contiguous(Shape{1, 1}), std::span<std::complex<float>>(complexD)),
        ScalarType::ComplexFloat32);
    referenceGemm(complexProblem);
    require(complexD[0] == std::complex<float>(11.0f, -2.0f),
            "Runtime complex GEMM result mismatch.");

    const std::array<float, 1> realA{1};
    const std::array<float, 2> realB{0, 0};
    const std::array<float, 2> realC{0, 0};
    const std::array<float, 2> columnBias{2, 3};
    std::array<float, 2> realD{};
    GemmProblem axisProblem(GemmOperand(TensorView::fromNative<float>(
                                Layout::contiguous(Shape{1, 1}), std::span<const float>(realA))),
                            GemmOperand(TensorView::fromNative<float>(
                                Layout::contiguous(Shape{1, 2}), std::span<const float>(realB))),
                            TensorView::fromNative<float>(Layout::contiguous(Shape{1, 2}),
                                                          std::span<const float>(realC)),
                            MutableTensorView::fromNative<float>(Layout::contiguous(Shape{1, 2}),
                                                                 std::span<float>(realD)),
                            ScalarType::Float32);
    axisProblem.epilogue.bias = VectorBinding{
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}),
                                      std::span<const float>(columnBias)),
        MatrixAxis::Column,
    };
    referenceGemm(axisProblem);
    require(realD == columnBias, "Runtime GEMM explicit column-axis bias mismatch.");
}

void testOutputSelection() {
    using namespace roc::host_validation;

    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{5, 6, 7, 8};
    const std::array<float, 4> c{};
    std::array<float, 4> d{-99, -99, -99, -99};

    GemmProblem problem(
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                  std::span<const float>(a))),
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                  std::span<const float>(b))),
        TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<float>(d)),
        ScalarType::Float32);
    problem.outputSelection = OutputSelection::explicitIndices({0, 3});
    const GemmRunInfo run = referenceGemm(problem);
    require(run.outputElementsComputed == 2,
            "Selected-output GEMM reported the wrong element count.");
    require(d[0] == 19 && d[1] == -99 && d[2] == -99 && d[3] == 50,
            "Selected-output GEMM modified the wrong elements.");

    const auto prime = OutputSelection::primeStride(10, 10, 3).indices(10);
    require(prime == std::vector<size_t>({0, 3, 6, 9}), "Prime-stride output selection mismatch.");
    require(OutputSelection::primeStride(10, 10, 0).selectsAll(),
            "Zero requested elements did not preserve all-output behavior.");
}

void testReferenceEpilogue() {
    using namespace roc::host_validation;

    const std::array<float, 4> input{-2, 1, 3, -4};
    const std::array<float, 2> bias{1, 2};
    std::array<uint16_t, 4> output{};
    std::array<float, 4> rawOutput{};
    std::array<uint16_t, 4> auxiliary{};
    std::array<float, 1> amax{};

    EpilogueProblem problem(TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                          std::span<const float>(input)),
                            MutableTensorView(ScalarType::Float16, Layout::contiguous(Shape{2, 2}),
                                              std::as_writable_bytes(std::span<uint16_t>(output))),
                            ScalarType::Float32);
    problem.rawOutput = MutableTensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                             std::span<float>(rawOutput));
    problem.auxiliaryOutput =
        MutableTensorView(ScalarType::BFloat16, Layout::contiguous(Shape{2, 2}),
                          std::as_writable_bytes(std::span<uint16_t>(auxiliary)));
    problem.amax =
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{1}), std::span<float>(amax));
    problem.bias = VectorBinding{
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(bias)),
        MatrixAxis::Row,
    };
    problem.outputScale = 2.0;
    problem.auxiliaryScale = 3.0;
    problem.activation = Activation::Relu;
    referenceEpilogue(problem);

    const TensorView outputView(ScalarType::Float16, Layout::contiguous(Shape{2, 2}),
                                std::as_bytes(std::span<const uint16_t>(output)));
    const TensorView auxiliaryView(ScalarType::BFloat16, Layout::contiguous(Shape{2, 2}),
                                   std::as_bytes(std::span<const uint16_t>(auxiliary)));
    require(outputView.loadAs<float>({0, 0}) == 0 && outputView.loadAs<float>({0, 1}) == 4 &&
                outputView.loadAs<float>({1, 0}) == 10 && outputView.loadAs<float>({1, 1}) == 0,
            "Reference epilogue output mismatch.");
    require(rawOutput == std::array<float, 4>{0, 4, 10, 0},
            "Reference epilogue raw output mismatch.");
    require(auxiliaryView.loadAs<float>({0, 0}) == -3 && auxiliaryView.loadAs<float>({0, 1}) == 6 &&
                auxiliaryView.loadAs<float>({1, 0}) == 15 &&
                auxiliaryView.loadAs<float>({1, 1}) == -6,
            "Reference epilogue auxiliary output mismatch.");
    require(amax[0] == 5, "Reference epilogue AMax mismatch.");

    const std::array<float, 4> gradientInput{10, 20, 30, 40};
    const std::array<float, 4> activationInput{-1, 1, 2, -2};
    std::array<float, 4> gradientOutput{};
    EpilogueProblem gradient(TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                           std::span<const float>(gradientInput)),
                             MutableTensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                                  std::span<float>(gradientOutput)),
                             ScalarType::Float32);
    gradient.auxiliaryInput = TensorView::fromNative<float>(
        Layout::contiguous(Shape{2, 2}), std::span<const float>(activationInput));
    gradient.activation = Activation::Relu;
    gradient.activationApplication = ActivationApplication::Gradient;
    referenceEpilogue(gradient);
    require(gradientOutput == std::array<float, 4>{0, 20, 30, 0},
            "Reference gradient epilogue mismatch.");

    const std::array<float, 4> gate{0.5f, 2.0f, -1.0f, 0.25f};
    std::array<float, 4> gatedOutput{};
    EpilogueProblem gated(TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                        std::span<const float>(input)),
                          MutableTensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                               std::span<float>(gatedOutput)),
                          ScalarType::Float32);
    gated.gateResidual = TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                       std::span<const float>(gate));
    gated.outputScale = 2.0;
    referenceEpilogue(gated);
    require(gatedOutput == std::array<float, 4>{-1.5f, 6.0f, -7.0f, -1.75f},
            "Reference gate-residual epilogue mismatch.");
}

void testReferenceReduction() {
    using namespace roc::host_validation;

    std::array<float, 30> storage;
    storage.fill(-1);
    MutableTensorView input = MutableTensorView::fromNative<float>(
        Layout(Shape{2, 3, 4}, {15, 5, 1}), std::span<float>(storage));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 4; ++column)
                input.storeFrom({batch, row, column}, 100 * batch + 10 * row + column);
        }
    }

    std::array<float, 3> output{};
    ReductionProblem problem(input.asConst(),
                             MutableTensorView::fromNative<float>(Layout::contiguous(Shape{3}),
                                                                  std::span<float>(output)),
                             ScalarType::Float32, {0, 2});
    const ReductionRunInfo run = referenceSum(problem);
    require(run.outputElementsComputed == 3 && run.inputElementsRead == 24,
            "Reference reduction run information mismatch.");
    require(output == std::array<float, 3>{412, 492, 572}, "Reference reduction result mismatch.");

    std::array<float, 1> maximumAbsolute{};
    const ReductionRunInfo maximumRun = referenceMaximumAbsolute(
        input.asConst(),
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{}),
                                             std::span<float>(maximumAbsolute)),
        ScalarType::Float32);
    require(maximumRun.outputElementsComputed == 1 && maximumRun.inputElementsRead == 24,
            "Reference maximum-absolute run information mismatch.");
    require(maximumAbsolute[0] == 123.0f, "Reference maximum-absolute result mismatch.");
}

void testStructuredSparsity() {
    using namespace roc::host_validation;

    std::array<float, 20> inputStorage;
    inputStorage.fill(-99);
    MutableTensorView input(ScalarType::Float32, Layout(Shape{2, 8}, {10, 1}),
                            std::as_writable_bytes(std::span<float>(inputStorage)));
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 8; ++column)
            input.storeFrom({row, column}, static_cast<float>(1 + row * 8 + column));

    std::array<float, 20> prunedStorage;
    prunedStorage.fill(-7);
    MutableTensorView pruned(ScalarType::Float32, Layout(Shape{2, 8}, {10, 1}),
                             std::as_writable_bytes(std::span<float>(prunedStorage)));
    std::array<float, 8> compressedStorage{};
    MutableTensorView compressed = MutableTensorView::fromNative<float>(
        Layout::contiguous(Shape{2, 4}), std::span<float>(compressedStorage));
    std::array<uint8_t, 8> indexStorage{};
    MutableTensorView retainedIndices = MutableTensorView::fromNative<uint8_t>(
        Layout::contiguous(Shape{2, 4}), std::span<uint8_t>(indexStorage));

    StructuredSparsityPattern pattern;
    pattern.axis = 1;
    pattern.fixedPositions = {1, 3};
    const StructuredSparsityRunInfo run = applyStructuredSparsity(
        StructuredSparsityProblem(input.asConst(), pruned, compressed, retainedIndices, pattern));
    require(run.groupsProcessed == 4 && run.prunedElementsWritten == 16 &&
                run.compressedElementsWritten == 8,
            "Structured sparsity run information mismatch.");

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

    std::array<uint8_t, 2> metadataStorage{};
    MutableTensorView metadata = MutableTensorView::fromNative<uint8_t>(
        Layout::contiguous(Shape{2, 1}), std::span<uint8_t>(metadataStorage));
    const TwoOfFourMetadataRunInfo metadataRun =
        encodeTwoOfFourMetadata(TwoOfFourMetadataProblem(retainedIndices.asConst(), metadata, 1));
    require(metadataRun.sparsityGroupsEncoded == 4 && metadataRun.metadataBytesWritten == 2,
            "Two-of-four metadata run information mismatch.");
    require(metadataStorage[0] == 0xdd && metadataStorage[1] == 0xdd,
            "Two-of-four metadata encoding mismatch.");

    std::array<float, 16> fusedPrunedStorage{};
    std::array<float, 8> fusedCompressedStorage{};
    std::array<uint8_t, 2> fusedMetadataStorage{};
    StructuredSparsityProblem fusedProblem(
        input.asConst(),
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{2, 8}),
                                             std::span<float>(fusedPrunedStorage)),
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{2, 4}),
                                             std::span<float>(fusedCompressedStorage)),
        pattern);
    fusedProblem.twoOfFourMetadata = MutableTensorView::fromNative<uint8_t>(
        Layout::contiguous(Shape{2, 1}), std::span<uint8_t>(fusedMetadataStorage));
    const StructuredSparsityRunInfo firstFusedRun =
        applyStructuredSparsity(fusedProblem, {.firstSlice = 0, .sliceCount = 1});
    const StructuredSparsityRunInfo secondFusedRun =
        applyStructuredSparsity(fusedProblem, {.firstSlice = 1, .sliceCount = 1});
    require(firstFusedRun.groupsProcessed == 2 && secondFusedRun.groupsProcessed == 2 &&
                firstFusedRun.retainedIndicesWritten == 0 &&
                secondFusedRun.retainedIndicesWritten == 0 &&
                firstFusedRun.metadataBytesWritten == 1 &&
                secondFusedRun.metadataBytesWritten == 1 && fusedMetadataStorage == metadataStorage,
            "Fused structured sparsity metadata mismatch.");

    Tensor inPlace =
        Tensor::fromNativeValues<float>(Shape{8}, std::array<float, 8>{1, 2, 3, 4, 5, 6, 7, 8});
    Tensor inPlaceCompressed(ScalarType::Float32, Shape{4});
    Tensor inPlaceIndices(ScalarType::UInt8, Shape{4});
    pattern.axis = 0;
    pattern.fixedPositions = {0, 2};
    applyStructuredSparsity(StructuredSparsityProblem(inPlace.view(), inPlace.mutableView(),
                                                      inPlaceCompressed.mutableView(),
                                                      inPlaceIndices.mutableView(), pattern));
    require(inPlace.view().loadAs<float>({0}) == 1 && inPlace.view().loadAs<float>({1}) == 0 &&
                inPlace.view().loadAs<float>({2}) == 3 && inPlace.view().loadAs<float>({3}) == 0,
            "In-place structured sparsity mismatch.");
}

void testIndexedGeneration() {
    using namespace roc::host_validation;

    Tensor serial(ScalarType::Float32, Shape{2, 3});
    GenerationOptions serialOptions;
    serialOptions.real.pattern = GenerationPattern::SerialIndex;
    const GenerationRunInfo serialRun = generate(serial.mutableView(), serialOptions);
    require(serialRun.elementsGenerated == 6, "Indexed generation count mismatch.");
    require(serial.view().loadAs<float>({0, 0}) == 0 && serial.view().loadAs<float>({1, 0}) == 1 &&
                serial.view().loadAs<float>({0, 1}) == 2 &&
                serial.view().loadAs<float>({1, 2}) == 5,
            "First-dimension-fast serial generation mismatch.");

    Tensor complex(ScalarType::ComplexFloat32, Shape{2, 2});
    GenerationOptions trigonometric;
    trigonometric.real.pattern = GenerationPattern::Sine;
    trigonometric.imaginary.pattern = GenerationPattern::Cosine;
    generate(complex.mutableView(), trigonometric);
    const std::complex<float> value = complex.view().loadAs<std::complex<float>>({1, 0});
    require(std::abs(value.real() - std::sin(1.0f)) < 1e-6f &&
                std::abs(value.imag() - std::cos(1.0f)) < 1e-6f,
            "Complex trigonometric generation mismatch.");

    Tensor candidates(ScalarType::Float32, Shape{8});
    GenerationOptions candidateOptions;
    candidateOptions.seed = 37;
    candidateOptions.real.pattern = GenerationPattern::CandidateSet;
    candidateOptions.real.stream = 5;
    candidateOptions.real.candidates = {-6.0, -1.5, 0.0, 4.0};
    generate(candidates.mutableView(), candidateOptions);
    for (size_t index = 0; index < candidates.size(); ++index) {
        const double expected =
            candidateOptions.real.candidates[counterRandom(candidateOptions.seed,
                                                           candidateOptions.real.stream, index) %
                                             candidateOptions.real.candidates.size()];
        require(candidates.view().loadAs<float>({index}) == expected,
                "Candidate-set generation mismatch.");
    }

    Tensor point(ScalarType::Float32, Shape{2, 3, 2});
    GenerationOptions pointOptions;
    pointOptions.real.pattern = GenerationPattern::Constant;
    pointOptions.real.parameter0 = 9.0;
    const GenerationRunInfo pointRun = generateAt(point.mutableView(), 3, pointOptions);
    require(pointRun.elementsGenerated == 1 && point.view().loadAs<float>({1, 1, 0}) == 9.0f &&
                point.view().loadAs<float>({0, 1, 1}) == 0.0f,
            "First-dimension-fast point generation mismatch.");

    pointOptions.indexOrder = LogicalIndexOrder::LastDimensionFastest;
    pointOptions.real.parameter0 = 7.0;
    generateAt(point.mutableView(), 3, pointOptions);
    require(point.view().loadAs<float>({0, 1, 1}) == 7.0f,
            "Last-dimension-fast point generation mismatch.");

    Tensor affine(ScalarType::Float32, Shape{2, 3, 2});
    GenerationOptions affineOptions;
    affineOptions.real.pattern = GenerationPattern::AffineIndexRemainder;
    affineOptions.real.dimensionCoefficients = {1, -1, 2};
    affineOptions.real.affineOffset = -2;
    affineOptions.real.remainderDivisor = 5;
    affineOptions.real.valueOffset = 1.0;
    generate(affine.mutableView(), affineOptions);
    require(affine.view().loadAs<float>({0, 0, 0}) == -1.0f &&
                affine.view().loadAs<float>({1, 2, 1}) == 0.0f,
            "Affine-index remainder generation mismatch.");
}

void testReferenceAxpby() {
    using namespace roc::host_validation;

    const Shape shape{2, 2, 2};
    Tensor x(ScalarType::Float16, Layout(shape, {1, 3, 6}));
    Tensor y(ScalarType::BFloat16, Layout(shape, {3, 1, 6}));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 8}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 2; ++column) {
                const std::array<size_t, 3> indices{row, column, batch};
                x.mutableView().storeFrom(indices,
                                          static_cast<float>(1 + row + 2 * column + batch));
                y.mutableView().storeFrom(indices,
                                          static_cast<float>(2 - 2 * row + column + batch));
            }
        }
    }

    AxpbyProblem problem(x.view(), y.view(), output.mutableView(), ScalarType::Float32);
    problem.alpha = 2.0;
    problem.beta = -0.5;
    const AxpbyRunInfo run = referenceAxpby(problem);
    require(run.elementsComputed == shape.elementCount(),
            "Reference AXPBY element count mismatch.");

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 2; ++column) {
                const std::array<size_t, 3> indices{row, column, batch};
                const float expected =
                    2.0f * x.view().loadAs<float>(indices) - 0.5f * y.view().loadAs<float>(indices);
                require(output.view().loadAs<float>(indices) == expected,
                        "Reference AXPBY value mismatch.");
            }
        }
    }

    Tensor yOnlyOutput(ScalarType::Float32, shape);
    AxpbyProblem yOnly(std::nullopt, y.view(), yOnlyOutput.mutableView(), ScalarType::Float32);
    yOnly.beta = 3.0;
    referenceAxpby(yOnly);
    require(yOnlyOutput.view().loadAs<float>({1, 0, 1}) == 3.0f * y.view().loadAs<float>({1, 0, 1}),
            "Reference AXPBY optional-input mismatch.");

    const std::array<std::complex<float>, 1> complexXValues{std::complex<float>(1, 2)};
    const std::array<std::complex<float>, 1> complexYValues{std::complex<float>(3, -1)};
    Tensor complexX = Tensor::fromNativeValues<std::complex<float>>(Shape{1}, complexXValues);
    Tensor complexY = Tensor::fromNativeValues<std::complex<float>>(Shape{1}, complexYValues);
    Tensor complexOutput(ScalarType::ComplexFloat32, Shape{1});
    AxpbyProblem complexProblem(complexX.view(), complexY.view(), complexOutput.mutableView(),
                                ScalarType::ComplexFloat32);
    complexProblem.alpha = std::complex<double>(0.5, 1.0);
    complexProblem.beta = -2.0;
    referenceAxpby(complexProblem);
    require(
        complexOutput.view().loadAs<std::complex<float>>({0}) == std::complex<float>(-7.5f, 4.0f),
        "Complex reference AXPBY mismatch.");
}

void testReferenceSoftmax() {
    using namespace roc::host_validation;

    const Shape shape{2, 3, 2};
    Tensor input(ScalarType::Float16, Layout(shape, {1, 3, 10}));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 12}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                input.mutableView().storeFrom(
                    {row, column, batch},
                    static_cast<float>(static_cast<int>(column) + 2 * static_cast<int>(row) -
                                       static_cast<int>(batch)));
            }
        }
    }

    const SoftmaxRunInfo run = referenceSoftmax(
        SoftmaxProblem(input.view(), output.mutableView(), 1, ScalarType::Float32));
    require(run.slicesComputed == 4 && run.elementsComputed == shape.elementCount(),
            "Reference softmax run information mismatch.");

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            float sum = 0;
            for (size_t column = 0; column < 3; ++column)
                sum += output.view().loadAs<float>({row, column, batch});
            require(std::abs(sum - 1.0f) < 1e-6f, "Reference softmax slice does not sum to one.");
        }
    }
    require(output.view().loadAs<float>({0, 0, 0}) < output.view().loadAs<float>({0, 1, 0}) &&
                output.view().loadAs<float>({0, 1, 0}) < output.view().loadAs<float>({0, 2, 0}),
            "Reference softmax ordering mismatch.");
}

void testReferenceLayerNorm() {
    using namespace roc::host_validation;

    const Shape shape{2, 3, 2};
    Tensor input(ScalarType::Float32, Layout(shape, {1, 3, 10}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 3; ++column)
                input.mutableView().storeFrom({row, column, batch},
                                              static_cast<float>(1 + column + 3 * row + 6 * batch));
        }
    }
    const std::array<float, 3> gammaValues{1.0f, 2.0f, 0.5f};
    const std::array<float, 3> betaValues{0.25f, -0.5f, 1.0f};
    const Tensor gamma =
        Tensor::fromValues(ScalarType::Float16, Shape{3}, std::span<const float>(gammaValues));
    const Tensor beta =
        Tensor::fromValues(ScalarType::BFloat16, Shape{3}, std::span<const float>(betaValues));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 12}));
    Tensor mean(ScalarType::Float32, Shape{2, 2});
    Tensor inverseVariance(ScalarType::Float32, Shape{2, 2});

    LayerNormProblem problem(input.view(), output.mutableView(), 1, ScalarType::Float32);
    problem.mean = mean.mutableView();
    problem.inverseVariance = inverseVariance.mutableView();
    problem.gamma = gamma.view();
    problem.beta = beta.view();
    const LayerNormRunInfo run = referenceLayerNorm(problem);
    require(run.slicesComputed == 4 && run.elementsComputed == shape.elementCount(),
            "Reference LayerNorm run information mismatch.");
    require(mean.view().loadAs<float>({0, 0}) == 2.0f, "Reference LayerNorm mean mismatch.");
    require(std::abs(inverseVariance.view().loadAs<float>({0, 0}) -
                     1.0f / std::sqrt(2.0f / 3.0f + 1e-5f)) < 1e-6f,
            "Reference LayerNorm inverse variance mismatch.");
    require(output.view().loadAs<float>({0, 1, 0}) == -0.5f,
            "Reference LayerNorm affine output mismatch.");
}

void testActivations() {
    using namespace roc::host_validation;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{1};
    const std::array<float, 1> c{0};
    std::array<float, 1> d{};

    GemmProblem problem(
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                                  std::span<const float>(a))),
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                                  std::span<const float>(b))),
        TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<float>(d)),
        ScalarType::Float32);

    problem.epilogue.activation = Activation::Gelu;
    referenceGemm(problem);
    require(std::abs(d[0] - 1.9545977f) < 1e-6f, "GELU result mismatch.");

    problem.epilogue.activation = Activation::Silu;
    problem.epilogue.activationParameter0 = 1;
    referenceGemm(problem);
    require(std::abs(d[0] - 1.7615942f) < 1e-6f, "SiLU result mismatch.");

    problem.epilogue.activation = Activation::Clamp;
    problem.epilogue.activationParameter0 = -1;
    problem.epilogue.activationParameter1 = 1;
    referenceGemm(problem);
    require(d[0] == 1, "Clamp result mismatch.");
}

void testStridedAndOffsetViews() {
    using namespace roc::host_validation;

    // Logical A and B are the same matrices as testReferenceGemm, but both
    // are stored transposed with padded leading dimensions. C and D use
    // different padding, and D begins at an adjusted base pointer.
    const std::array<float, 8> a{1, 2, 3, -1, 4, 5, 6, -1};
    const std::array<float, 9> b{7, 8, -1, 9, 10, -1, 11, 12, -1};
    const std::array<float, 8> c{1, 1, -1, -1, 1, 1, -1, -1};
    std::array<float, 12> d;
    d.fill(-99);

    const Layout outputLayout(Shape{2, 2}, {1, 5}, 1);
    GemmProblem problem(
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{2, 3}, {4, 1}), std::span<const float>(a))),
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{3, 2}, {3, 1}), std::span<const float>(b))),
        TensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 4}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(outputLayout, std::span<float>(d)),
        ScalarType::Float32);
    problem.epilogue.alpha = 2.0;
    problem.epilogue.beta = 3.0;

    referenceGemm(problem);

    std::array<float, 12> expected;
    expected.fill(-99);
    expected[1] = 2 * 58 + 3;
    expected[2] = 2 * 139 + 3;
    expected[6] = 2 * 64 + 3;
    expected[7] = 2 * 154 + 3;
    const auto comparison =
        compare(TensorView::fromNative<float>(outputLayout, std::span<const float>(d)),
                TensorView::fromNative<float>(outputLayout, std::span<const float>(expected)));
    require(comparison.passed(), "Strided GEMM matrix comparison failed.");
    require(d[0] == -99 && d[3] == -99 && d[11] == -99, "Strided GEMM modified padding.");
}

void testGenerationAndComparison() {
    using namespace roc::host_validation;

    require(counterRandom(7, 3, 11) == counterRandom(7, 3, 11),
            "Counter-based generation is not deterministic.");
    require(counterRandom(7, 3, 11) != counterRandom(7, 3, 12),
            "Counter-based generation does not vary by logical index.");
    const int indexedValue = indexedUniformInteger(7, 3, 11, -4, 5);
    require(indexedValue >= -4 && indexedValue <= 5,
            "Counter-based integer generation exceeded its bounds.");

    GenerationOptions binaryGeneration;
    binaryGeneration.seed = 42;
    binaryGeneration.real.pattern = GenerationPattern::CandidateSet;
    binaryGeneration.real.candidates = {-1.0, 1.0};
    Tensor a(ScalarType::Float32, Shape{32});
    Tensor b(ScalarType::Float32, Shape{32});
    generate(a.mutableView(), binaryGeneration);
    generate(b.mutableView(), binaryGeneration);
    require(compare(b.view(), a.view()).passed(),
            "Random generation is not repeatable for equal seeds.");

    b.mutableView().storeFrom({7}, b.view().loadAs<float>({7}) + 1.0f);
    const auto result =
        compare(b.view(), a.view(),
                {.absoluteTolerance = 0.0, .relativeTolerance = 0.0, .maxReportedMismatches = 4});
    require(result.mismatches == 1, "Comparison did not count one mismatch.");
    require(result.reportedMismatches.size() == 1, "Comparison did not report one mismatch.");
    require(result.reportedMismatches[0].index == 7,
            "Comparison reported the wrong mismatch index.");

    const std::array<double, 2> nonFiniteA{
        std::numeric_limits<double>::infinity(),
        1.0,
    };
    const std::array<double, 2> nonFiniteB{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    const auto nonFiniteResult =
        compare(std::span<const double>(nonFiniteA), std::span<const double>(nonFiniteB),
                {.relativeTolerance = 1.0});
    require(nonFiniteResult.mismatches == 1,
            "Comparison did not distinguish finite and infinite values.");

    std::array<int, 8> generated;
    generated.fill(-1);
    generate(MatrixView<int>(generated.data() + 1, 2, 2, 1, 3),
             [](size_t row, size_t column) { return 10 * column + row; });
    require(generated[1] == 0 && generated[2] == 1 && generated[4] == 10 && generated[5] == 11,
            "Matrix generation produced incorrect logical values.");
    require(generated[0] == -1 && generated[3] == -1 && generated[7] == -1,
            "Matrix generation modified padding.");

    Tensor runtimeExpected(ScalarType::Float32, Shape{2, 3});
    GenerationOptions runtimeGeneration;
    runtimeGeneration.seed = 7;
    runtimeGeneration.real.pattern = GenerationPattern::UniformInteger;
    runtimeGeneration.real.parameter0 = -2;
    runtimeGeneration.real.parameter1 = 2;
    generate(runtimeExpected.mutableView(), runtimeGeneration);
    Tensor runtimeObserved = runtimeExpected;
    runtimeObserved.mutableView().storeFrom({1, 2},
                                            runtimeExpected.view().loadAs<float>({1, 2}) + 1.0f);
    const auto runtimeComparison = compare(runtimeObserved.view(), runtimeExpected.view(),
                                           {.absoluteTolerance = 0.0, .maxReportedMismatches = 2});
    require(runtimeComparison.compared == 6 && runtimeComparison.mismatches == 1 &&
                runtimeComparison.reportedMismatches[0].index == 5,
            "Runtime tensor generation/comparison mismatch.");
}

void testComparisonProgram() {
    using namespace roc::host_validation;

    const std::array<float, 8> expectedStorage{1.0f, 2.0f, -99.0f, 3.0f, 4.0f, -99.0f, 5.0f, 6.0f};
    auto observedStorage = expectedStorage;
    observedStorage[4] += 0.5f;
    observedStorage[6] += 1.0f;
    const Layout layout(Shape{2, 3}, {1, 3});

    ComparisonOptions selected;
    selected.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
    selected.selection.stride = 2;
    const auto selectedResult = compare(std::span<const float>(observedStorage), layout,
                                        std::span<const float>(expectedStorage), layout, selected);
    require(selectedResult.compared == 3 && selectedResult.mismatches == 1,
            "Selected comparison visited the wrong logical elements.");
    require(selectedResult.reportedMismatches[0].index == 4 &&
                selectedResult.reportedMismatches[0].coordinates == std::vector<size_t>({0, 2}) &&
                selectedResult.reportedMismatches[0].observedOffset == 6,
            "Selected comparison reported the wrong logical location.");

    const std::array<double, 3> expected{3.0, 4.0, 0.0};
    const std::array<double, 3> observed{0.0, 4.0, 3.0};
    ComparisonOptions metrics;
    metrics.pointwise = false;
    metrics.relativeFrobeniusTolerance = 0.9;
    metrics.computeUlp = true;
    metrics.ulpType = ScalarType::Float64;
    const auto metricResult =
        compare(std::span<const double>(observed), std::span<const double>(expected), metrics);
    require(std::abs(metricResult.frobeniusExpected - 5.0) < 1e-12 &&
                std::abs(metricResult.frobeniusObserved - 5.0) < 1e-12 &&
                std::abs(metricResult.frobeniusDifference - std::sqrt(18.0)) < 1e-12 &&
                std::abs(metricResult.relativeFrobeniusError - std::sqrt(18.0) / 5.0) < 1e-12 &&
                metricResult.frobeniusPassed,
            "Comparison Frobenius evidence is incorrect.");

    const double oneUlp = std::ldexp(1.0, -52);
    const std::array<double, 1> ulpObserved{1.0 + oneUlp};
    const std::array<double, 1> ulpExpected{1.0};
    ComparisonOptions ulp;
    ulp.computeUlp = true;
    ulp.ulpType = ScalarType::Float64;
    ulp.maximumUlpTolerance = 1.0;
    const auto ulpResult =
        compare(std::span<const double>(ulpObserved), std::span<const double>(ulpExpected), ulp);
    require(ulpResult.maximumUlp == 1.0 && ulpResult.averageUlp == 1.0 && ulpResult.ulpPassed,
            "Comparison ULP evidence is incorrect.");
    require(encodedUlpDistance(0.0, static_cast<double>(std::numeric_limits<float>::denorm_min()),
                               ScalarType::Float32) == 1.0,
            "Encoded ULP distance mishandled the F32 zero/subnormal boundary.");
    require(encodedUlpDistance(1.0, 1.5, ScalarType::Float4E2M1) == 1.0,
            "Encoded ULP distance mishandled a packed scalar sign width.");

    const std::array<std::complex<double>, 2> complexExpected{
        std::complex<double>(std::numeric_limits<double>::infinity(), 2.0),
        std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 4.0)};
    const auto complexObserved = complexExpected;
    ComparisonOptions nonFinite;
    nonFinite.equalNaNs = true;
    const auto nonFiniteResult =
        compare(std::span<const std::complex<double>>(complexObserved),
                std::span<const std::complex<double>>(complexExpected), nonFinite);
    require(nonFiniteResult.passed() && nonFiniteResult.matchedInfinities == 1 &&
                nonFiniteResult.matchedNaNs == 1,
            "Complex non-finite comparison policy is incorrect.");

    const std::array<double, 4> absoluteCandidates{1e-6, 1e-5, 1e-4, 1e-3};
    const std::array<double, 1> relativeCandidates{0.0};
    const std::array<double, 1> closeObserved{1.00009};
    const std::array<double, 1> closeExpected{1.0};
    const auto tolerance = findAllCloseTolerance(
        std::span<const double>(closeObserved), Layout::contiguous(Shape{1}),
        std::span<const double>(closeExpected), Layout::contiguous(Shape{1}),
        std::span<const double>(absoluteCandidates), std::span<const double>(relativeCandidates));
    require(tolerance && tolerance->absolute == 1e-4 && tolerance->relative == 0.0,
            "Allclose tolerance search selected the wrong candidate.");

    std::array<float, 5> guarded{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    TensorView guardedView =
        TensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 3}), std::span<const float>(guarded));
    require(checkUnusedTensorStorage(guardedView, guarded.size()).passed(),
            "Unwritten tensor padding sentinel was rejected.");
    guarded[2] = 0.0f;
    const auto sentinel = checkUnusedTensorStorage(guardedView, guarded.size());
    require(
        !sentinel.passed() && sentinel.mismatches == 1 && sentinel.reportedMismatches[0].index == 2,
        "Written tensor padding was not detected.");
}
}  // namespace

int main() {
    testRuntimeReferenceGemm();
    testRuntimeMixedAndBlockScaledGemm();
    testRuntimeComplexAndExplicitAxisGemm();
    testOutputSelection();
    testReferenceEpilogue();
    testReferenceReduction();
    testStructuredSparsity();
    testIndexedGeneration();
    testReferenceAxpby();
    testReferenceSoftmax();
    testReferenceLayerNorm();
    testActivations();
    testStridedAndOffsetViews();
    testGenerationAndComparison();
    testComparisonProgram();
    return 0;
}
