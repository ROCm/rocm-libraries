// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testReferenceEpilogue() {
    using namespace roc::host_numerics;

    const std::array<float, 4> input{-2, 1, 3, -4};
    const std::array<float, 2> bias{1, 2};
    Tensor output(ScalarType::Float16, Shape{2, 2});
    Tensor rawOutput(ScalarType::Float32, Shape{2, 2});
    Tensor auxiliary(ScalarType::BFloat16, Shape{2, 2});
    Tensor amax(ScalarType::Float32, Shape{1});

    const Tensor inputTensor = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(input));
    EpilogueOptions options(ScalarType::Float32);
    require(options.outputScale.type() == ScalarType::Float32 &&
                options.auxiliaryScale.type() == ScalarType::Float32 &&
                std::holds_alternative<IdentityActivation>(options.activation),
            "Reference epilogue defaults do not use the requested compute type.");
    options.bias = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 1}), std::span<const float>(bias));
    options.outputScale = 2.0;
    options.auxiliaryScale = 3.0;
    options.activation = ReluActivation{};
    referenceEpilogueInto(
        inputTensor,
        {.output = output, .rawOutput = rawOutput, .auxiliaryOutput = auxiliary, .amax = amax},
        options);

    require(output.loadAs<float>({0, 0}) == 0 && output.loadAs<float>({0, 1}) == 4 &&
                output.loadAs<float>({1, 0}) == 10 && output.loadAs<float>({1, 1}) == 0,
            "Reference epilogue output mismatch.");
    require(compare(rawOutput,
                    Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{0, 4, 10, 0}))
                .passed(),
            "Reference epilogue raw output mismatch.");
    require(auxiliary.loadAs<float>({0, 0}) == -3 && auxiliary.loadAs<float>({0, 1}) == 6 &&
                auxiliary.loadAs<float>({1, 0}) == 15 && auxiliary.loadAs<float>({1, 1}) == -6,
            "Reference epilogue auxiliary output mismatch.");
    require(amax.loadAs<float>({0}) == 5, "Reference epilogue AMax mismatch.");

    Tensor affineOutput(ScalarType::Float32, Shape{2, 2});
    EpilogueOptions affineOptions(ScalarType::Float32);
    affineOptions.inputScale = Tensor::scalar(ScalarType::Float32, 2.0f);
    affineOptions.addend =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{10, 20, 30, 40});
    affineOptions.addendScale = Tensor::scalar(ScalarType::Float32, 0.5f);
    referenceEpilogueInto(inputTensor, {.output = affineOutput}, affineOptions);
    require(compare(affineOutput, Tensor::copyNativeValues<float>(
                                      Shape{2, 2}, std::array<float, 4>{1, 12, 21, 12}))
                .passed(),
            "Reference epilogue fused input/addend scaling mismatch.");

    constexpr size_t parallelRows = 128;
    constexpr size_t parallelColumns = 256;
    const std::vector<float> parallelInputValues(parallelRows * parallelColumns, 0.25f);
    const std::vector<float> parallelAddendValues(parallelRows * parallelColumns, 0.5f);
    const std::vector<float> parallelBiasValues(parallelColumns, 0.125f);
    const Tensor parallelInput =
        Tensor::copyNativeValues<float>(Shape{parallelRows, parallelColumns}, parallelInputValues);
    Tensor parallelOutput(ScalarType::Float16, Shape{parallelRows, parallelColumns});
    EpilogueOptions parallelOptions(ScalarType::Float32);
    parallelOptions.inputScale = Tensor::scalar(ScalarType::Float32, 1.5f);
    parallelOptions.inputScaleFactors.push_back(Tensor::scalar(ScalarType::Float32, 0.5f));
    parallelOptions.inputScaleFactors.push_back(
        Tensor::copyNativeValues<float>(Shape{1, parallelColumns}, parallelBiasValues));
    parallelOptions.addend =
        Tensor::copyNativeValues<float>(Shape{parallelRows, parallelColumns}, parallelAddendValues);
    parallelOptions.addendScale = Tensor::scalar(ScalarType::Float32, 0.75f);
    parallelOptions.bias =
        Tensor::copyNativeValues<float>(Shape{1, parallelColumns}, parallelBiasValues);
    parallelOptions.activation = ClampActivation{-2.0, 2.0};
    parallelOptions.outputScale = Tensor::scalar(ScalarType::Float32, 2.0f);
    referenceEpilogueInto(parallelInput, {.output = parallelOutput}, parallelOptions);
    require(parallelOutput.loadAs<float>({0, 0}) == 1.046875f &&
                parallelOutput.loadAs<float>({parallelRows - 1, parallelColumns - 1}) == 1.046875f,
            "Parallel reference epilogue result mismatch.");
    Tensor parallelAmax(ScalarType::Float32, Shape{1});
    referenceEpilogueInto(parallelInput, {.output = parallelOutput, .amax = parallelAmax},
                          parallelOptions);
    require(parallelAmax.loadAs<float>({0}) == 0.5234375f,
            "Parallel reference epilogue AMax mismatch.");

    Tensor aliasedAddend =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{10, 20, 30, 40});
    affineOptions.addend = aliasedAddend;
    referenceEpilogueInto(inputTensor, {.output = aliasedAddend}, affineOptions);
    require(compare(aliasedAddend, Tensor::copyNativeValues<float>(
                                       Shape{2, 2}, std::array<float, 4>{1, 12, 21, 12}))
                .passed(),
            "Reference epilogue rejected an identically mapped addend/output alias.");

    const std::array<float, 4> gradientInput{10, 20, 30, 40};
    const std::array<float, 4> activationInput{-1, 1, 2, -2};
    Tensor gradientOutput(ScalarType::Float32, Shape{2, 2});
    EpilogueOptions gradientOptions(ScalarType::Float32);
    gradientOptions.auxiliaryInput =
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(activationInput));
    gradientOptions.activation = ReluActivation{};
    gradientOptions.activationApplication = ActivationApplication::Gradient;
    referenceEpilogueInto(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(gradientInput)),
        {.output = gradientOutput}, gradientOptions);
    require(compare(gradientOutput, Tensor::copyNativeValues<float>(
                                        Shape{2, 2}, std::array<float, 4>{0, 20, 30, 0}))
                .passed(),
            "Reference gradient epilogue mismatch.");

    const std::array<float, 4> gate{0.5f, 2.0f, -1.0f, 0.25f};
    Tensor gatedOutput(ScalarType::Float32, Shape{2, 2});
    EpilogueOptions gatedOptions(ScalarType::Float32);
    gatedOptions.gateResidual = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(gate));
    gatedOptions.outputScale = 2.0;
    referenceEpilogueInto(inputTensor, {.output = gatedOutput}, gatedOptions);
    require(compare(gatedOutput, Tensor::copyNativeValues<float>(
                                     Shape{2, 2}, std::array<float, 4>{-1.5f, 6.0f, -7.0f, -1.75f}))
                .passed(),
            "Reference gate-residual epilogue mismatch.");

    const std::array<float, 4> int8Input{-200.0f, -128.5f, 126.5f, 300.0f};
    Tensor int8Output(ScalarType::Int8, Shape{2, 2});
    EpilogueOptions int8Options(ScalarType::Float32);
    int8Options.outputConversion = OutputConversion::SaturatingInt8;
    referenceEpilogueInto(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(int8Input)),
        {.output = int8Output}, int8Options);
    require(compare(int8Output, Tensor::copyNativeValues<int8_t>(
                                    Shape{2, 2}, std::array<int8_t, 4>{-128, -128, 126, 127}))
                .passed(),
            "Reference epilogue Int8 saturation mismatch.");

    constexpr double highPrecisionInput = 1.0000000001;
    const std::array<double, 1> highPrecisionValues{highPrecisionInput};
    Tensor highPrecisionOutput(ScalarType::Float64, Shape{1, 1});
    EpilogueOptions highPrecisionOptions(ScalarType::Float64);
    highPrecisionOptions.activation = SigmoidActivation{};
    referenceEpilogueInto(
        Tensor::copyNativeStorage<double>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                          highPrecisionValues),
        {.output = highPrecisionOutput}, highPrecisionOptions);
    const double expectedHighPrecision = 1.0 / (1.0 + std::exp(-highPrecisionInput));
    require(std::abs(highPrecisionOutput.loadAs<double>({0, 0}) - expectedHighPrecision) < 1e-15,
            "Float64 reference activation used reduced-precision intermediates.");

    for (const bool useTanh : {true, false}) {
        Tensor gradientResult(ScalarType::Float64, Shape{1, 1});
        EpilogueOptions highPrecisionGradient(ScalarType::Float64);
        highPrecisionGradient.auxiliaryInput = Tensor::copyNativeValues<double>(
            Shape{1, 1}, std::array<double, 1>{highPrecisionInput});
        highPrecisionGradient.activation =
            useTanh ? ActivationFunction(TanhActivation{}) : ActivationFunction(SwishActivation{});
        highPrecisionGradient.activationApplication = ActivationApplication::Gradient;
        referenceEpilogueInto(
            Tensor::copyNativeValues<double>(Shape{1, 1}, std::array<double, 1>{1.0}),
            {.output = gradientResult}, highPrecisionGradient);

        const double sigmoid = 1.0 / (1.0 + std::exp(-highPrecisionInput));
        const double hyperbolicTangent = std::tanh(highPrecisionInput);
        const double expectedGradient =
            useTanh ? 1.0 - hyperbolicTangent * hyperbolicTangent
                    : sigmoid + highPrecisionInput * sigmoid * (1.0 - sigmoid);
        require(std::abs(gradientResult.loadAs<double>({0, 0}) - expectedGradient) < 1e-15,
                "Float64 reference activation gradient used reduced-precision intermediates.");
    }

    const Tensor ownedInput = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(input));
    EpilogueOptions ownedOptions = options;
    ownedOptions.outputSelection = OutputSelection::explicitIndices({1, 2});
    const EpilogueOutputs owned = referenceEpilogue(ownedInput,
                                                    {.output = ScalarType::Float16,
                                                     .rawOutput = ScalarType::Float32,
                                                     .auxiliaryOutput = ScalarType::BFloat16,
                                                     .amax = ScalarType::Float32},
                                                    ownedOptions);
    require(owned.output.layout() == Layout::contiguousLastDimensionFastest(Shape{2, 2}) &&
                owned.rawOutput && owned.auxiliaryOutput && owned.amax,
            "Owning reference epilogue result contract mismatch.");
    require(owned.output.loadAs<float>({0, 0}) == 0 && owned.output.loadAs<float>({0, 1}) == 4 &&
                owned.output.loadAs<float>({1, 0}) == 10 &&
                owned.output.loadAs<float>({1, 1}) == 0 &&
                owned.rawOutput->loadAs<float>({0, 0}) == 0 &&
                owned.rawOutput->loadAs<float>({1, 1}) == 0 &&
                owned.auxiliaryOutput->loadAs<float>({0, 0}) == 0 &&
                owned.auxiliaryOutput->loadAs<float>({1, 1}) == 0 &&
                owned.amax->loadAs<float>({0}) == 5,
            "Owning reference epilogue did not zero unselected values.");

    EpilogueOptions emptyOptions = ownedOptions;
    emptyOptions.outputSelection = OutputSelection::strided(4, 1);
    const EpilogueOutputs empty = referenceEpilogue(ownedInput,
                                                    {.output = ScalarType::Float16,
                                                     .rawOutput = ScalarType::Float32,
                                                     .auxiliaryOutput = ScalarType::BFloat16,
                                                     .amax = ScalarType::Float32},
                                                    emptyOptions);
    require(empty.output.loadAs<float>({0, 0}) == 0 && empty.rawOutput &&
                empty.rawOutput->loadAs<float>({0, 0}) == 0 && empty.auxiliaryOutput &&
                empty.auxiliaryOutput->loadAs<float>({0, 0}) == 0 && empty.amax &&
                empty.amax->loadAs<float>({0}) == 0,
            "Owning reference epilogue empty selection was not zero initialized.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceEpilogue(ownedInput, {.output = ScalarType::E8M0});
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference epilogue accepted an invalid output type.");

    Tensor preservedOutput =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});
    Tensor preservedRaw = preservedOutput.deepCopy();
    Tensor preservedAuxiliary = preservedOutput.deepCopy();
    Tensor accumulatedAmax = Tensor::copyNativeValues<float>(Shape{1}, std::array<float, 1>{100});
    EpilogueOptions preservedOptions;
    preservedOptions.outputSelection = OutputSelection::explicitIndices({0, 3});
    preservedOptions.accumulateAmax = true;
    referenceEpilogueInto(ownedInput,
                          {.output = preservedOutput,
                           .rawOutput = preservedRaw,
                           .auxiliaryOutput = preservedAuxiliary,
                           .amax = accumulatedAmax},
                          preservedOptions);
    require(preservedOutput.loadAs<float>({0, 1}) == -99 &&
                preservedOutput.loadAs<float>({1, 0}) == -99 &&
                preservedRaw.loadAs<float>({0, 1}) == -99 &&
                preservedAuxiliary.loadAs<float>({1, 0}) == -99 &&
                accumulatedAmax.loadAs<float>({0}) == 100,
            "Explicit reference epilogue did not preserve unselected or accumulated state.");
}

}  // namespace host_numerics_test
