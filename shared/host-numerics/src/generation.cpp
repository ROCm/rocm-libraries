// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/data_generation.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics {
uint64_t deriveDeterministicSeed(uint64_t baseSeed, uint64_t streamIdentifier,
                                 uint64_t sequenceIndex) {
    return detail::counterRandom(baseSeed, streamIdentifier, sequenceIndex);
}

namespace {
void validateIntegerInterval(const UniformIntegerGenerationParameters& parameters) {
    if (parameters.lower > parameters.upper)
        throw std::invalid_argument("Generation lower bound exceeds upper bound.");
}

#ifdef _OPENMP
void incrementLastDimensionFast(std::vector<size_t>& indices, const Shape& shape) {
    for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
        const size_t index = dimension - 1;
        if (++indices[index] < shape[index]) return;
        indices[index] = 0;
    }
}
#endif

void generateSerial(Tensor destination, const GenerationRecipe& recipe) {
    detail::forEachIndex(destination.shape(), [&](std::span<const size_t> indices, size_t) {
        const size_t logicalIndex = destination.shape().linearIndex(indices, recipe.indexOrder());
        detail::generateElement(destination, recipe, indices, logicalIndex);
    });
}

void generateParallel(Tensor destination, const GenerationRecipe& recipe, int threadCount) {
#ifdef _OPENMP
    std::exception_ptr error;
    const size_t elementCount = destination.shape().elementCount();
#pragma omp parallel num_threads(threadCount)
    {
        try {
            const size_t threadIndex = static_cast<size_t>(omp_get_thread_num());
            const size_t actualThreadCount = static_cast<size_t>(omp_get_num_threads());
            const size_t baseCount = elementCount / actualThreadCount;
            const size_t remainder = elementCount % actualThreadCount;
            const size_t first = threadIndex * baseCount + std::min(threadIndex, remainder);
            const size_t count = baseCount + static_cast<size_t>(threadIndex < remainder);
            const size_t end = first + count;
            if (first != end) {
                std::vector<size_t> indices =
                    destination.shape().coordinates(first, IndexOrder::LastDimensionFastest);
                for (size_t traversalIndex = first; traversalIndex < end; ++traversalIndex) {
                    const size_t logicalIndex =
                        destination.shape().linearIndex(indices, recipe.indexOrder());
                    detail::generateElement(destination, recipe, indices, logicalIndex);
                    incrementLastDimensionFast(indices, destination.shape());
                }
            }
        } catch (...) {
#pragma omp critical(roc_host_numerics_generation_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
#else
    (void)threadCount;
    generateSerial(destination, recipe);
#endif
}
}  // namespace

GenerationRecipe::Component::Component(Pattern pattern) : pattern_(std::move(pattern)) {}

bool GenerationRecipe::Component::isRaw() const {
    return std::visit(
        [](const auto& pattern) {
            using Pattern = std::remove_cvref_t<decltype(pattern)>;
            return std::is_same_v<Pattern, RawConstantPattern> ||
                   std::is_same_v<Pattern, UniformRawIntegerPattern> ||
                   std::is_same_v<Pattern, UniformFiniteEncodedValuePattern> ||
                   std::is_same_v<Pattern, RandomRawBitsPattern> ||
                   std::is_same_v<Pattern, RawSerialDimensionPattern>;
        },
        pattern_);
}

GenerationRecipe::Component GenerationRecipe::Component::withAbsoluteTransform() const {
    if (isRaw())
        throw std::invalid_argument(
            "Numerical generation modifiers do not apply to raw storage recipes.");
    Component result = *this;
    result.unaryTransform_ = UnaryTransform::Absolute;
    return result;
}

GenerationRecipe::Component GenerationRecipe::Component::withSineTransform() const {
    if (isRaw())
        throw std::invalid_argument(
            "Numerical generation modifiers do not apply to raw storage recipes.");
    Component result = *this;
    result.unaryTransform_ = UnaryTransform::Sine;
    return result;
}

GenerationRecipe::Component GenerationRecipe::Component::withCosineTransform() const {
    if (isRaw())
        throw std::invalid_argument(
            "Numerical generation modifiers do not apply to raw storage recipes.");
    Component result = *this;
    result.unaryTransform_ = UnaryTransform::Cosine;
    return result;
}

GenerationRecipe::Component GenerationRecipe::Component::withAffineValueMapping(
    GenerationAffineValueParameters parameters) const {
    if (isRaw())
        throw std::invalid_argument(
            "Numerical generation modifiers do not apply to raw storage recipes.");
    Component result = *this;
    result.affineValue_ = parameters;
    return result;
}

GenerationRecipe::Component GenerationRecipe::Component::withAlternatingSign(
    AlternatingSignGenerationParameters parameters) const {
    if (isRaw())
        throw std::invalid_argument(
            "Numerical generation modifiers do not apply to raw storage recipes.");
    if (parameters.dimensions.empty())
        throw std::invalid_argument("Alternating-sign generation requires at least one dimension.");
    Component result = *this;
    result.alternatingSign_ = std::move(parameters);
    return result;
}

GenerationRecipe::Component GenerationRecipe::Component::withZeroOutsideMainDiagonal() const {
    if (isRaw())
        throw std::invalid_argument("Diagonal masking does not apply to raw storage recipes.");
    Component result = *this;
    result.zeroOutsideMainDiagonal_ = true;
    return result;
}

GenerationRecipe::Component GenerationRecipe::zero() {
    return Component(Component::ZeroPattern{});
}

GenerationRecipe::Component GenerationRecipe::constant(ConstantGenerationParameters parameters) {
    return Component(Component::ConstantPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::candidateSet(
    CandidateSetGenerationParameters parameters) {
    if (parameters.values.empty())
        throw std::invalid_argument("Candidate-set generation requires at least one value.");
    return Component(Component::CandidateSetPattern{std::move(parameters)});
}

GenerationRecipe::Component GenerationRecipe::uniformInteger(
    UniformIntegerGenerationParameters parameters) {
    validateIntegerInterval(parameters);
    return Component(Component::UniformIntegerPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::absoluteUniformInteger(
    UniformIntegerGenerationParameters parameters) {
    validateIntegerInterval(parameters);
    return Component(Component::AbsoluteUniformIntegerPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::uniformReal(
    UniformRealGenerationParameters parameters) {
    if (!(parameters.lower <= parameters.upper))
        throw std::invalid_argument("Uniform-real bounds must be ordered and must not be NaN.");
    return Component(Component::UniformRealPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::normal(NormalGenerationParameters parameters) {
    if (!std::isfinite(parameters.mean) || !std::isfinite(parameters.standardDeviation) ||
        parameters.standardDeviation < 0.0)
        throw std::invalid_argument(
            "Normal generation requires a finite mean and nonnegative finite standard deviation.");
    return Component(Component::NormalPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::sine() {
    return Component(Component::SinePattern{});
}

GenerationRecipe::Component GenerationRecipe::cosine() {
    return Component(Component::CosinePattern{});
}

GenerationRecipe::Component GenerationRecipe::absoluteSine() {
    return Component(Component::AbsoluteSinePattern{});
}

GenerationRecipe::Component GenerationRecipe::absoluteCosine() {
    return Component(Component::AbsoluteCosinePattern{});
}

GenerationRecipe::Component GenerationRecipe::serialIndex() {
    return Component(Component::SerialIndexPattern{});
}

GenerationRecipe::Component GenerationRecipe::serialDimension(
    DimensionGenerationParameters parameters) {
    return Component(Component::SerialDimensionPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::affineIndexRemainder(
    AffineIndexRemainderGenerationParameters parameters) {
    if (parameters.positiveDivisor <= 0)
        throw std::invalid_argument("Affine-index remainder divisor must be positive.");
    return Component(Component::AffineIndexRemainderPattern{std::move(parameters)});
}

GenerationRecipe::Component GenerationRecipe::identity() {
    return Component(Component::IdentityPattern{});
}

GenerationRecipe::Component GenerationRecipe::checkerboardUniformInteger(
    UniformIntegerGenerationParameters parameters) {
    validateIntegerInterval(parameters);
    return Component(Component::CheckerboardUniformIntegerPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::typeMaximum() {
    return Component(Component::TypeMaximumPattern{});
}

GenerationRecipe::Component GenerationRecipe::typeLowest() {
    return Component(Component::TypeLowestPattern{});
}

GenerationRecipe::Component GenerationRecipe::typeDenormalMinimum() {
    return Component(Component::TypeDenormalMinimumPattern{});
}

GenerationRecipe::Component GenerationRecipe::typeDenormalMaximum() {
    return Component(Component::TypeDenormalMaximumPattern{});
}

GenerationRecipe::Component GenerationRecipe::typeNaN() {
    return Component(Component::TypeNaNPattern{});
}

GenerationRecipe::Component GenerationRecipe::typeInfinity() {
    return Component(Component::TypeInfinityPattern{});
}

GenerationRecipe::Component GenerationRecipe::typeNegativeInfinity() {
    return Component(Component::TypeNegativeInfinityPattern{});
}

GenerationRecipe::Component GenerationRecipe::typeNegativeZero() {
    return Component(Component::TypeNegativeZeroPattern{});
}

GenerationRecipe::Component GenerationRecipe::uniformTypeRange() {
    return Component(Component::UniformTypeRangePattern{});
}

GenerationRecipe::Component GenerationRecipe::randomEncodedExponent(
    RandomEncodedExponentGenerationParameters parameters) {
    if (parameters.lowerUnbiasedExponent > parameters.upperUnbiasedExponent)
        throw std::invalid_argument("Random encoded-exponent lower bound exceeds upper bound.");
    if (parameters.sourceType.has_value()) {
        if (!isConcreteScalarType(*parameters.sourceType) ||
            scalarTypeInfo(*parameters.sourceType).exponentBits == 0)
            throw std::invalid_argument(
                "Random encoded-exponent source type must have an exponent field.");
    }
    return Component(Component::RandomEncodedExponentPattern{std::move(parameters)});
}

GenerationRecipe::Component GenerationRecipe::rawConstant(
    RawConstantGenerationParameters parameters) {
    return Component(Component::RawConstantPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::uniformRawInteger(
    UniformIntegerGenerationParameters parameters) {
    validateIntegerInterval(parameters);
    return Component(Component::UniformRawIntegerPattern{parameters});
}

GenerationRecipe::Component GenerationRecipe::uniformFiniteEncodedValue() {
    return Component(Component::UniformFiniteEncodedValuePattern{});
}

GenerationRecipe::Component GenerationRecipe::randomRawBits() {
    return Component(Component::RandomRawBitsPattern{});
}

GenerationRecipe::Component GenerationRecipe::rawSerialDimension(
    DimensionGenerationParameters parameters) {
    return Component(Component::RawSerialDimensionPattern{parameters});
}

GenerationRecipe::GenerationRecipe(GenerationRecipeSettings settings, ComplexPolicy complexPolicy)
    : settings_(settings), complexPolicy_(std::move(complexPolicy)) {}

GenerationRecipe GenerationRecipe::realOnly(Component component,
                                            GenerationRecipeSettings settings) {
    return GenerationRecipe(
        settings, RealOnlyPolicy{
                      .real = {
                          .component = std::move(component),
                          .randomDomain = detail::generation_random_domain_version_1::realComponent,
                      }});
}

GenerationRecipe GenerationRecipe::replicated(Component component,
                                              GenerationRecipeSettings settings) {
    if (component.isRaw())
        throw std::invalid_argument(
            "Replicated complex generation requires a numerical component.");
    return GenerationRecipe(
        settings, ReplicatedPolicy{
                      .value = {
                          .component = std::move(component),
                          .randomDomain = detail::generation_random_domain_version_1::realComponent,
                      }});
}

GenerationRecipe GenerationRecipe::cartesian(Component real, Component imaginary,
                                             GenerationRecipeSettings settings) {
    if (real.isRaw() || imaginary.isRaw())
        throw std::invalid_argument("Cartesian complex generation requires numerical components.");
    return GenerationRecipe(
        settings,
        CartesianPolicy{
            .real =
                {
                    .component = std::move(real),
                    .randomDomain = detail::generation_random_domain_version_1::realComponent,
                },
            .imaginary =
                {
                    .component = std::move(imaginary),
                    .randomDomain = detail::generation_random_domain_version_1::imaginaryComponent,
                },
        });
}

uint64_t GenerationRecipe::seed() const noexcept {
    return settings_.seed;
}

IndexOrder GenerationRecipe::indexOrder() const noexcept {
    return settings_.indexOrder;
}

uint64_t GenerationRecipe::randomDomain() const noexcept {
    return settings_.randomDomain;
}

GenerationRecipe GenerationRecipe::withSeed(uint64_t seed) const {
    GenerationRecipe result = *this;
    result.settings_.seed = seed;
    return result;
}

GenerationRecipe GenerationRecipe::withIndexOrder(IndexOrder order) const {
    GenerationRecipe result = *this;
    result.settings_.indexOrder = order;
    return result;
}

GenerationRecipe GenerationRecipe::withRandomDomain(uint64_t domain) const {
    GenerationRecipe result = *this;
    result.settings_.randomDomain = domain;
    return result;
}

void generate(Tensor destination, const GenerationRecipe& recipe) {
    const size_t elementCount = destination.shape().elementCount();
    const int threadCount = detail::hasProvablyIndependentElements(destination)
                                ? detail::operationThreadCount(elementCount)
                                : 1;
    if (threadCount == 1)
        generateSerial(destination, recipe);
    else
        generateParallel(destination, recipe, threadCount);
}

Tensor generate(ScalarType type, Layout layout, const GenerationRecipe& recipe) {
    Tensor result(type, std::move(layout));
    generate(result, recipe);
    return result;
}

Tensor generate(ScalarType type, Shape shape, const GenerationRecipe& recipe) {
    return generate(type, Layout::contiguousLastDimensionFastest(shape), recipe);
}

void generateAt(Tensor destination, size_t logicalIndex, const GenerationRecipe& recipe) {
    const std::vector<size_t> indices =
        destination.shape().coordinates(logicalIndex, recipe.indexOrder());
    detail::generateElement(destination, recipe, indices, logicalIndex);
}
}  // namespace roc::host_numerics
