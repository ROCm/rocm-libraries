// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// This executable is both a tutorial and a CTest contract test. Keep the
// numbered examples in a readable order and prefer small, inspectable values.

#include <algorithm>
#include <array>
#include <complex>
#include <iostream>
#include <roc/host_numerics/comparison.hpp>
#include <roc/host_numerics/gemm.hpp>
#include <roc/host_numerics/generation.hpp>
#include <roc/host_numerics/layer_norm.hpp>
#include <roc/host_numerics/mx.hpp>
#include <roc/host_numerics/reduction.hpp>
#include <roc/host_numerics/softmax.hpp>
#include <roc/host_numerics/structured_sparsity.hpp>
#include <roc/host_numerics/tensor_operations.hpp>
#include <span>
#include <stdexcept>
#include <string>

namespace {
using namespace roc::host_numerics;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Step 1: create tensors from ordinary values. ScalarType records the encoded
// storage format, while Shape and Layout describe logical indexing.
void tensorBasics() {
    const Tensor values =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f});
    check(values.type() == ScalarType::Float32 && values.shape() == Shape{2, 2},
          "Float32 tensor metadata mismatch.");

    const Tensor scalar = Tensor::scalar(ScalarType::Float16, 1.5f);
    check(scalar.shape() == Shape{} && scalar.item<float>() == 1.5f,
          "Rank-zero tensor conversion mismatch.");

    const std::array<float, 4> fp4Values{0.0f, 0.5f, 1.0f, 6.0f};
    const Tensor fp4 = Tensor::copyValuesWithConversion(ScalarType::Float4E2M1, Shape{4},
                                                        std::span<const float>(fp4Values));
    check(fp4.rawEncodedBackingStorage().size() == 2 && fp4.loadAs<float>({3}) == 6.0f,
          "Packed FP4 tensor mismatch.");
}

// Step 2: generate reproducible real, integer, and complex tensors. A caller
// supplies one seed per logical operand; replaying a seed reproduces its bytes.
void deterministicGeneration() {
    const GenerationRecipe realRecipe = GenerationRecipe::realOnly(
        GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}), {.seed = 17});
    const Tensor real = generate(ScalarType::Float32, Shape{8}, realRecipe);
    const Tensor replay = generate(ScalarType::Float32, Shape{8}, realRecipe);
    check(std::equal(real.rawEncodedBackingStorage().begin(), real.rawEncodedBackingStorage().end(),
                     replay.rawEncodedBackingStorage().begin(),
                     replay.rawEncodedBackingStorage().end()),
          "Equal generation seeds produced different storage.");

    const Tensor integers =
        generate(ScalarType::Int32, Shape{8},
                 GenerationRecipe::realOnly(
                     GenerationRecipe::uniformInteger({.lower = 1, .upper = 4}), {.seed = 27}));
    bool anyNonzero = false;
    for (size_t index = 0; index < integers.elementCount(); ++index)
        anyNonzero |= integers.loadAs<int32_t>({index}) != 0;
    check(anyNonzero, "Integer generation unexpectedly produced only zeros.");

    const Tensor complex =
        generate(ScalarType::ComplexFloat32, Shape{8},
                 GenerationRecipe::cartesian(
                     GenerationRecipe::uniformInteger({.lower = 1, .upper = 4}),
                     GenerationRecipe::uniformInteger({.lower = -4, .upper = -1}), {.seed = 37}));
    check(complex.loadAs<std::complex<float>>({0}).imag() != 0.0f,
          "Complex generation omitted its imaginary component.");
}

// Step 3: compose broadcast elementwise operations. Operators are shorthand
// for the same-type defaults; named calls expose compute/output policy.
void tensorArithmetic() {
    const Tensor column =
        Tensor::copyNativeValues<float>(Shape{2, 1}, std::array<float, 2>{1.0f, 2.0f});
    const Tensor row =
        Tensor::copyNativeValues<float>(Shape{1, 3}, std::array<float, 3>{10.0f, 20.0f, 30.0f});
    const Tensor result = 2.0f * column + row;
    check(result.shape() == Shape{2, 3} && result.loadAs<float>({0, 0}) == 12.0f &&
              result.loadAs<float>({1, 2}) == 34.0f,
          "Broadcast tensor arithmetic mismatch.");
    check(relu(clip(result, -20.0, 20.0)).loadAs<float>({1, 2}) == 20.0f,
          "Named activation mismatch.");
}

// Step 4: matrix multiplication keeps storage and arithmetic types explicit.
void matrixMultiplication() {
    const Tensor a =
        Tensor::copyNativeValues<float>(Shape{2, 3}, std::array<float, 6>{1, 2, 3, 4, 5, 6});
    const Tensor b =
        Tensor::copyNativeValues<float>(Shape{3, 2}, std::array<float, 6>{7, 8, 9, 10, 11, 12});
    const Tensor product = matmul(a, b, ScalarType::Float32);
    check(compare(product, Tensor::copyNativeValues<float>(Shape{2, 2},
                                                           std::array<float, 4>{58, 64, 139, 154}))
              .passed(),
          "Float32 matmul mismatch.");

    MatmulOptions integerOptions(ScalarType::Int32);
    const Tensor integerProduct =
        matmul(Tensor::copyNativeValues<int8_t>(Shape{1, 2}, std::array<int8_t, 2>{2, 3}),
               Tensor::copyNativeValues<int8_t>(Shape{2, 1}, std::array<int8_t, 2>{4, 5}),
               ScalarType::Int32, integerOptions);
    check(integerProduct.loadAs<int32_t>({0, 0}) == 23, "Int8-to-Int32 matmul mismatch.");

    MatmulOptions complexOptions(ScalarType::ComplexFloat32);
    complexOptions.conjugateA = true;
    const Tensor complexProduct =
        matmul(Tensor::copyNativeValues<std::complex<float>>(
                   Shape{1, 2}, std::array{std::complex<float>{1, 1}, std::complex<float>{2, -1}}),
               Tensor::copyNativeValues<std::complex<float>>(
                   Shape{2, 1}, std::array{std::complex<float>{3, 0}, std::complex<float>{1, 2}}),
               ScalarType::ComplexFloat32, complexOptions);
    check(complexProduct.loadAs<std::complex<float>>({0, 0}) == std::complex<float>(3, 2),
          "Conjugated complex matmul mismatch.");

    const Tensor emptyProduct =
        matmul(Tensor(ScalarType::Float32, Shape{2, 0}), Tensor(ScalarType::Float32, Shape{0, 3}),
               ScalarType::Float32);
    check(emptyProduct.shape() == Shape{2, 3} && emptyProduct.loadAs<float>({1, 2}) == 0.0f,
          "Zero-reduction matmul is not the additive identity.");
}

// Step 5: MX generation returns packed data, natural block scales, a scale map,
// and decoded reference values together.
void blockScaledMxGeneration() {
    const GenerationRecipe recipe = GenerationRecipe::realOnly(
        GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}), {.seed = 47});
    const MxDataGeneration data =
        MxDataGeneration::preserveRange(recipe, {.lower = -1.0, .upper = 1.0});
    MxGenerationOptions options;
    options.dataType = ScalarType::Float4E2M1;
    options.scaleType = ScalarType::E8M0;
    options.leadingDimension = 32;
    options.blockAxis = 0;
    options.blockSize = 32;
    const MxTensor mx = generateMx(Shape{32, 2}, data, options);
    check(mx.data.type() == ScalarType::Float4E2M1 && mx.scales.shape() == Shape{2, 1} &&
              mx.scaleIndices.shape() == mx.data.shape() && mx.reference.shape() == mx.data.shape(),
          "MXFP4 generation result mismatch.");
}

// Step 6: higher-level references compose the same Tensor model.
void higherLevelOperations() {
    const Tensor input =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor sums = referenceSum(input, {1}, ScalarType::Float32, ScalarType::Float32);
    check(sums.shape() == Shape{2} && sums.loadAs<float>({0}) == 3.0f &&
              sums.loadAs<float>({1}) == 7.0f,
          "Reduction mismatch.");

    const Tensor probabilities =
        referenceSoftmax(input, 1, ScalarType::Float32, ScalarType::Float32);
    check(std::abs(probabilities.loadAs<float>({0, 0}) + probabilities.loadAs<float>({0, 1}) -
                   1.0f) < 1e-6f,
          "Softmax row does not sum to one.");

    const LayerNormOutputs normalized =
        referenceLayerNorm(input, {.output = ScalarType::Float32, .mean = ScalarType::Float32});
    check(normalized.mean && normalized.mean->shape() == Shape{2},
          "LayerNorm statistics mismatch.");

    StructuredSparsityPattern pattern;
    pattern.axis = 1;
    pattern.fixedPositions = {0, 2};
    const StructuredSparseTensor sparse = applyStructuredSparsity(
        Tensor::copyNativeValues<float>(Shape{1, 4}, std::array<float, 4>{1, 2, 3, 4}), pattern,
        {.retainedIndices = true, .twoOfFourMetadata = true});
    check(sparse.pruned.loadAs<float>({0, 1}) == 0.0f && sparse.retainedIndices &&
              sparse.twoOfFourMetadata,
          "Structured sparsity outputs mismatch.");
}

// Step 7: comparison returns structured evidence and a reusable diagnostic.
void comparisonDiagnostics() {
    const Tensor expected =
        Tensor::copyNativeValues<float>(Shape{3}, std::array<float, 3>{1, 2, 3});
    const Tensor observed =
        Tensor::copyNativeValues<float>(Shape{3}, std::array<float, 3>{1, 9, 3});
    const ComparisonReport report = compare(observed, expected);
    const std::string message = formatComparisonReport(report);
    check(!report.passed() &&
              message.starts_with("comparison failed: 1 of 3 compared elements mismatched") &&
              message.find("index 1 [1]") != std::string::npos,
          "Comparison failure diagnostic mismatch.");
}
}  // namespace

int main() {
    try {
        tensorBasics();
        deterministicGeneration();
        tensorArithmetic();
        matrixMultiplication();
        blockScaledMxGeneration();
        higherLevelOperations();
        comparisonDiagnostics();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host-numerics tutorial failed: " << error.what() << '\n';
        return 1;
    }
}
