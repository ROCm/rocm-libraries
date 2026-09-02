// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <roc/host_numerics/mx.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/data_generation.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics {
namespace {
constexpr uint64_t boundedScaleRandomDomain = 0xa24baed4963ee407ULL;
constexpr uint64_t unboundedScaleRandomDomain = 0x94d049bb133111ebULL;
}  // namespace

MxDataGeneration::MxDataGeneration(GenerationRecipe recipe, MxDataQuantization quantization,
                                   std::optional<MxRepresentedValueRange> representedValueRange)
    : recipe_(std::move(recipe)),
      quantization_(quantization),
      representedValueRange_(representedValueRange) {}

MxDataGeneration MxDataGeneration::quantize(GenerationRecipe recipe) {
    return MxDataGeneration(std::move(recipe), MxDataQuantization::Nearest, std::nullopt);
}

MxDataGeneration MxDataGeneration::preserveRange(GenerationRecipe recipe,
                                                 MxRepresentedValueRange representedValueRange) {
    if (!std::isfinite(representedValueRange.lower) ||
        !std::isfinite(representedValueRange.upper) ||
        !(representedValueRange.lower < representedValueRange.upper))
        throw std::invalid_argument(
            "Range-preserving MX generation requires finite lower < upper.");
    return MxDataGeneration(std::move(recipe), MxDataQuantization::PreserveRange,
                            representedValueRange);
}

MxDataGeneration MxDataGeneration::preserveGeneratedEncoding(GenerationRecipe recipe) {
    return MxDataGeneration(std::move(recipe), MxDataQuantization::PreserveGeneratedEncoding,
                            std::nullopt);
}

const GenerationRecipe& MxDataGeneration::recipe() const noexcept {
    return recipe_;
}

MxDataQuantization MxDataGeneration::quantization() const noexcept {
    return quantization_;
}

const std::optional<MxRepresentedValueRange>& MxDataGeneration::representedValueRange()
    const noexcept {
    return representedValueRange_;
}

MxDataGeneration MxDataGeneration::withSeed(uint64_t seed) const {
    return MxDataGeneration(recipe_.withSeed(seed), quantization_, representedValueRange_);
}

namespace {
struct MxGenerationInvocation : MxGenerationOptions {
    MxGenerationInvocation(Shape inputShape, MxDataGeneration dataGeneration,
                           const MxGenerationOptions& options)
        : MxGenerationOptions(options),
          shape(std::move(inputShape)),
          data(std::move(dataGeneration)) {}

    Shape shape;
    MxDataGeneration data;
};

size_t checkedMultiply(size_t first, size_t second, const char* message) {
    if (first != 0 && second > std::numeric_limits<size_t>::max() / first)
        throw std::overflow_error(message);
    return first * second;
}

size_t ceilDivide(size_t value, size_t divisor) {
    return value / divisor + static_cast<size_t>(value % divisor != 0);
}

struct ScaleBlocking {
    size_t blockAxis;
    size_t blockSize;
    size_t blockedExtent;
    size_t freeExtent;
    size_t blockCount;
    size_t scaleCount;

    explicit ScaleBlocking(const MxGenerationInvocation& problem)
        : blockAxis(problem.blockAxis),
          blockSize(problem.blockSize),
          blockedExtent(problem.shape[problem.blockAxis]),
          freeExtent(problem.shape[1 - problem.blockAxis]),
          blockCount(ceilDivide(blockedExtent, blockSize)),
          scaleCount(checkedMultiply(blockCount, freeExtent, "MX scale count overflow.")) {}

    std::pair<size_t, size_t> blockAndFreeCoordinate(size_t scaleIndexValue) const {
        if (blockAxis == 0) return {scaleIndexValue % blockCount, scaleIndexValue / blockCount};
        return {scaleIndexValue / freeExtent, scaleIndexValue % freeExtent};
    }

    std::pair<size_t, size_t> dataCoordinates(size_t block, size_t freeCoordinate,
                                              size_t offsetInBlock) const {
        if (blockAxis == 0) return {block * blockSize + offsetInBlock, freeCoordinate};
        return {freeCoordinate, block * blockSize + offsetInBlock};
    }

    size_t blockElementCount(size_t block) const {
        return std::min(blockSize, blockedExtent - block * blockSize);
    }

    Shape naturalScaleShape() const {
        return blockAxis == 0 ? Shape{freeExtent, blockCount} : Shape{blockCount, freeExtent};
    }
};

bool isSupportedDataType(ScalarType type) {
    switch (type) {
        case ScalarType::Float4E2M1:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
            return true;
        default:
            return false;
    }
}

bool isSupportedScaleType(ScalarType type) {
    return type == ScalarType::E8M0 || type == ScalarType::E8M0Zero || type == ScalarType::E5M3 ||
           type == ScalarType::E4M3;
}

bool isSupportedTypePair(ScalarType dataType, ScalarType scaleType) {
    if (dataType == ScalarType::Float4E2M1)
        return scaleType == ScalarType::E8M0 || scaleType == ScalarType::E8M0Zero ||
               scaleType == ScalarType::E5M3 || scaleType == ScalarType::E4M3;
    return isSupportedDataType(dataType) &&
           (scaleType == ScalarType::E8M0 || scaleType == ScalarType::E8M0Zero);
}

uint8_t dataRawForValue(ScalarType type, double value) {
    return static_cast<uint8_t>(detail::encodeBinaryFloat(type, static_cast<float>(value)));
}

double dataValueForRaw(ScalarType type, uint8_t raw) {
    return detail::decodeBinaryFloat(type, raw);
}

uint8_t scaleRawForValue(ScalarType type, double value) {
    if (type == ScalarType::E8M0) return detail::encodeE8M0(static_cast<float>(value));
    if (type == ScalarType::E8M0Zero) return detail::encodeE8M0Zero(static_cast<float>(value));
    return static_cast<uint8_t>(detail::encodeBinaryFloat(type, static_cast<float>(value)));
}

double scaleValueForRaw(ScalarType type, uint8_t raw) {
    if (type == ScalarType::E8M0) return detail::decodeE8M0(raw);
    if (type == ScalarType::E8M0Zero) return detail::decodeE8M0Zero(raw);
    return detail::decodeBinaryFloat(type, raw);
}

uint8_t maximumScaleRaw(ScalarType type) {
    if (type == ScalarType::E8M0 || type == ScalarType::E8M0Zero) return 0xfeU;
    return static_cast<uint8_t>(detail::binaryFloatFormat(type).maximumPositiveFiniteRaw);
}

struct ScaleCandidate {
    uint8_t raw;
    double value;
};

std::vector<ScaleCandidate> finiteNonzeroScaleCandidates(ScalarType type) {
    std::vector<ScaleCandidate> candidates;
    for (uint32_t raw = 0; raw <= maximumScaleRaw(type); ++raw) {
        const double value = scaleValueForRaw(type, static_cast<uint8_t>(raw));
        if (std::isfinite(value) && value > 0.0)
            candidates.push_back({static_cast<uint8_t>(raw), value});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const ScaleCandidate& first, const ScaleCandidate& second) {
                  return first.value < second.value;
              });
    if (candidates.empty())
        throw std::invalid_argument("MX scale type has no finite nonzero values.");
    return candidates;
}

uint8_t scaleAtLeast(double requested, const std::vector<ScaleCandidate>& candidates,
                     bool allowLargerCandidate, uint64_t seed, size_t scaleIndex) {
    auto candidate = std::lower_bound(
        candidates.begin(), candidates.end(), requested,
        [](const ScaleCandidate& available, double value) { return available.value < value; });
    if (candidate == candidates.end()) return candidates.back().raw;
    if (allowLargerCandidate && candidate + 1 != candidates.end() &&
        (detail::counterRandom(seed, boundedScaleRandomDomain, scaleIndex) & 1U) != 0)
        ++candidate;
    return candidate->raw;
}

std::optional<uint8_t> explicitScaleRaw(const MxGenerationInvocation& problem) {
    switch (problem.scale) {
        case MxScaleGenerationMode::Derived:
        case MxScaleGenerationMode::RandomFinite:
            return std::nullopt;
        case MxScaleGenerationMode::Minimum:
            return scaleRawForValue(problem.scaleType, 0.0);
        case MxScaleGenerationMode::One:
            return scaleRawForValue(problem.scaleType, 1.0);
        case MxScaleGenerationMode::Two:
            return scaleRawForValue(problem.scaleType, 2.0);
        case MxScaleGenerationMode::Maximum:
            return maximumScaleRaw(problem.scaleType);
        case MxScaleGenerationMode::NaN:
            return scaleRawForValue(problem.scaleType, std::numeric_limits<double>::quiet_NaN());
    }
    throw std::invalid_argument("Invalid MX scale generation mode.");
}

double generatedValue(const MxGenerationInvocation& problem, size_t row, size_t column,
                      size_t logicalIndex) {
    const std::array<size_t, 2> indices{row, column};
    return detail::GenerationRecipeAccess::generatedNumericalValue(
        problem.data.recipe(), indices, problem.shape, logicalIndex, problem.dataType);
}

std::vector<double> decodedDataValues(ScalarType dataType) {
    const uint16_t bits = scalarTypeInfo(dataType).storageBits;
    const uint32_t count = 1U << bits;
    std::vector<double> values(count);
    for (uint32_t raw = 0; raw < count; ++raw)
        values[raw] = dataValueForRaw(dataType, static_cast<uint8_t>(raw));
    return values;
}

uint8_t constrainDataRawToInterval(ScalarType dataType, uint8_t raw, double scale, double minimum,
                                   double maximum, const std::vector<double>& dataValues) {
    const detail::BinaryFloatFormat format = detail::binaryFloatFormat(dataType);
    const uint8_t signMask = static_cast<uint8_t>(1U << (format.totalBits - 1U));
    uint8_t sign = raw & signMask;
    uint8_t magnitude = raw & static_cast<uint8_t>(signMask - 1U);

    for (uint32_t attempt = 0; attempt <= format.maximumPositiveFiniteRaw; ++attempt) {
        const double represented = dataValues[sign | magnitude] * scale;
        if (represented >= minimum && represented <= maximum) return sign | magnitude;

        const bool negative = sign != 0;
        const bool increaseMagnitude = represented < minimum ? !negative : negative;
        if (increaseMagnitude) {
            if (magnitude == format.maximumPositiveFiniteRaw) break;
            ++magnitude;
        } else {
            if (magnitude == 0) break;
            --magnitude;
        }
    }
    throw std::invalid_argument(
        "MX bounded interval contains no representable value at the selected block scale.");
}

std::vector<std::byte> packRawValues(std::span<const uint8_t> rawValues, uint16_t bitsPerValue,
                                     int threadCount) {
    (void)threadCount;
    const size_t totalBits = checkedMultiply(rawValues.size(), static_cast<size_t>(bitsPerValue),
                                             "MX packed storage size overflow.");
    std::vector<std::byte> storage(ceilDivide(totalBits, size_t{8}), std::byte{0});
    if (bitsPerValue == 8) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threadCount)
#endif
        for (size_t index = 0; index < rawValues.size(); ++index)
            storage[index] = static_cast<std::byte>(rawValues[index]);
    } else if (bitsPerValue == 4) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threadCount)
#endif
        for (size_t byteIndex = 0; byteIndex < storage.size(); ++byteIndex) {
            const size_t firstIndex = byteIndex * 2;
            const uint8_t first = rawValues[firstIndex] & 0x0fU;
            const uint8_t second =
                firstIndex + 1 < rawValues.size() ? rawValues[firstIndex + 1] & 0x0fU : 0;
            storage[byteIndex] = static_cast<std::byte>(first | (second << 4));
        }
    } else if (bitsPerValue == 6) {
        const size_t groups = (rawValues.size() + 3U) / 4U;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threadCount)
#endif
        for (size_t group = 0; group < groups; ++group) {
            const size_t firstIndex = group * 4U;
            uint32_t word = 0;
            for (size_t offset = 0; offset < 4U && firstIndex + offset < rawValues.size(); ++offset)
                word |= static_cast<uint32_t>(rawValues[firstIndex + offset] & 0x3fU)
                        << (offset * 6U);
            const size_t byteOffset = group * 3U;
            for (size_t byte = 0; byte < 3U && byteOffset + byte < storage.size(); ++byte)
                storage[byteOffset + byte] = static_cast<std::byte>((word >> (byte * 8U)) & 0xffU);
        }
    } else {
        for (size_t index = 0; index < rawValues.size(); ++index)
            detail::writePackedBits(storage, index * static_cast<size_t>(bitsPerValue),
                                    bitsPerValue, rawValues[index]);
    }
    return storage;
}

void generateUnbounded(const MxGenerationInvocation& problem, const ScaleBlocking& blocking,
                       std::vector<uint8_t>& dataRawValues, std::vector<uint8_t>& scaleRawValues,
                       std::vector<uint32_t>& scaleIndexValues, std::vector<float>& referenceValues,
                       int threadCount) {
    (void)threadCount;
    const std::vector<double> dataValues = decodedDataValues(problem.dataType);
    const std::optional<uint8_t> fixedScale = explicitScaleRaw(problem);
    const std::vector<ScaleCandidate> scaleCandidates =
        fixedScale ? std::vector<ScaleCandidate>{}
                   : finiteNonzeroScaleCandidates(problem.scaleType);
    const size_t rows = problem.shape[0];
    const size_t leadingDimension =
        problem.leadingDimension == 0 ? rows : static_cast<size_t>(problem.leadingDimension);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threadCount)
#endif
    for (size_t scaleIndex = 0; scaleIndex < blocking.scaleCount; ++scaleIndex) {
        const auto [block, freeCoordinate] = blocking.blockAndFreeCoordinate(scaleIndex);
        const uint8_t scaleRaw =
            fixedScale
                ? *fixedScale
                : scaleCandidates[detail::counterRandom(problem.data.recipe().seed(),
                                                        unboundedScaleRandomDomain, scaleIndex) %
                                  scaleCandidates.size()]
                      .raw;
        scaleRawValues[scaleIndex] = scaleRaw;
        const double scaleValue = scaleValueForRaw(problem.scaleType, scaleRaw);

        for (size_t offset = 0; offset < blocking.blockElementCount(block); ++offset) {
            const auto [row, column] = blocking.dataCoordinates(block, freeCoordinate, offset);
            const std::array<size_t, 2> indices{row, column};
            const size_t recipeIndex =
                problem.shape.linearIndex(indices, problem.data.recipe().indexOrder());
            const size_t logicalIndex = row + column * rows;
            const size_t physicalIndex = row + column * leadingDimension;
            const uint16_t dataBits = scalarTypeInfo(problem.dataType).storageBits;
            const uint8_t dataMask = static_cast<uint8_t>((uint16_t{1} << dataBits) - 1U);
            const uint8_t dataRaw =
                static_cast<uint8_t>(detail::GenerationRecipeAccess::generatedRawValue(
                    problem.data.recipe(), indices, problem.shape, recipeIndex, problem.dataType)) &
                dataMask;
            dataRawValues[physicalIndex] = dataRaw;
            scaleIndexValues[logicalIndex] = static_cast<uint32_t>(scaleIndex);
            referenceValues[logicalIndex] = static_cast<float>(dataValues[dataRaw] * scaleValue);
        }
    }
}

void generateQuantized(const MxGenerationInvocation& problem, const ScaleBlocking& blocking,
                       std::vector<uint8_t>& dataRawValues, std::vector<uint8_t>& scaleRawValues,
                       std::vector<uint32_t>& scaleIndexValues, std::vector<float>& referenceValues,
                       int threadCount) {
    (void)threadCount;
    const size_t rows = problem.shape[0];
    const size_t leadingDimension =
        problem.leadingDimension == 0 ? rows : static_cast<size_t>(problem.leadingDimension);

    const std::optional<uint8_t> fixedScale = explicitScaleRaw(problem);
    const std::vector<ScaleCandidate> scaleCandidates =
        fixedScale ? std::vector<ScaleCandidate>{}
                   : finiteNonzeroScaleCandidates(problem.scaleType);
    const std::vector<double> dataValues = decodedDataValues(problem.dataType);
    const double maximumDataValue = detail::typeMaximum(problem.dataType);

    auto generateRange = [&](size_t firstScaleIndex, size_t endScaleIndex) {
        std::vector<double> blockValues(
            fixedScale ? 0 : std::min(blocking.blockSize, blocking.blockedExtent));
        for (size_t scaleIndex = firstScaleIndex; scaleIndex < endScaleIndex; ++scaleIndex) {
            const auto [block, freeCoordinate] = blocking.blockAndFreeCoordinate(scaleIndex);
            const size_t blockElementCount = blocking.blockElementCount(block);
            uint8_t scaleRaw = fixedScale.value_or(0);

            if (!fixedScale && problem.scale == MxScaleGenerationMode::Derived) {
                bool hasNaN = false;
                double maximumMagnitude = 0.0;
                for (size_t offset = 0; offset < blockElementCount; ++offset) {
                    const auto [row, column] =
                        blocking.dataCoordinates(block, freeCoordinate, offset);
                    const std::array<size_t, 2> indices{row, column};
                    const size_t recipeIndex =
                        problem.shape.linearIndex(indices, problem.data.recipe().indexOrder());
                    const double value = generatedValue(problem, row, column, recipeIndex);
                    blockValues[offset] = value;
                    hasNaN = hasNaN || std::isnan(value);
                    if (std::isfinite(value))
                        maximumMagnitude = std::max(maximumMagnitude, std::abs(value));
                }
                if (hasNaN) {
                    scaleRaw = scaleRawForValue(problem.scaleType,
                                                std::numeric_limits<double>::quiet_NaN());
                } else if (maximumMagnitude == 0.0) {
                    scaleRaw = scaleRawForValue(problem.scaleType, 1.0);
                } else {
                    const bool allowLargerCandidate =
                        problem.data.quantization() == MxDataQuantization::PreserveRange;
                    scaleRaw = scaleAtLeast(maximumMagnitude / maximumDataValue, scaleCandidates,
                                            allowLargerCandidate, problem.data.recipe().seed(),
                                            scaleIndex);
                }
            } else if (!fixedScale) {
                scaleRaw =
                    scaleCandidates[detail::counterRandom(problem.data.recipe().seed(),
                                                          unboundedScaleRandomDomain, scaleIndex) %
                                    scaleCandidates.size()]
                        .raw;
            }

            scaleRawValues[scaleIndex] = scaleRaw;
            const double scaleValue = scaleValueForRaw(problem.scaleType, scaleRaw);
            for (size_t offset = 0; offset < blockElementCount; ++offset) {
                const auto [row, column] = blocking.dataCoordinates(block, freeCoordinate, offset);
                const size_t logicalIndex = row + column * rows;
                const size_t physicalIndex = row + column * leadingDimension;
                const std::array<size_t, 2> indices{row, column};
                const size_t recipeIndex =
                    problem.shape.linearIndex(indices, problem.data.recipe().indexOrder());
                const double sourceValue =
                    fixedScale || problem.scale == MxScaleGenerationMode::RandomFinite
                        ? generatedValue(problem, row, column, recipeIndex)
                        : blockValues[offset];
                const double scaledValue =
                    sourceValue == 0.0 ? sourceValue : sourceValue / scaleValue;
                uint8_t dataRaw = dataRawForValue(problem.dataType, scaledValue);
                if (problem.data.quantization() == MxDataQuantization::PreserveRange)
                    dataRaw = constrainDataRawToInterval(
                        problem.dataType, dataRaw, scaleValue,
                        problem.data.representedValueRange()->lower,
                        problem.data.representedValueRange()->upper, dataValues);
                dataRawValues[physicalIndex] = dataRaw;
                scaleIndexValues[logicalIndex] = static_cast<uint32_t>(scaleIndex);
                referenceValues[logicalIndex] =
                    static_cast<float>(dataValues[dataRaw] * scaleValue);
            }
        }
    };

    if (threadCount == 1) {
        generateRange(0, blocking.scaleCount);
        return;
    }

#ifdef _OPENMP
    std::exception_ptr error;
#pragma omp parallel num_threads(threadCount)
    {
        try {
            const size_t threadIndex = static_cast<size_t>(omp_get_thread_num());
            const size_t actualThreadCount = static_cast<size_t>(omp_get_num_threads());
            const size_t baseCount = blocking.scaleCount / actualThreadCount;
            const size_t remainder = blocking.scaleCount % actualThreadCount;
            const size_t first = threadIndex * baseCount + std::min(threadIndex, remainder);
            const size_t count = baseCount + static_cast<size_t>(threadIndex < remainder);
            generateRange(first, first + count);
        } catch (...) {
#pragma omp critical(roc_host_numerics_mx_generation_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
#else
    generateRange(0, blocking.scaleCount);
#endif
}

void validateInvocation(const MxGenerationInvocation& problem) {
    if (problem.shape.rank() != 2)
        throw std::invalid_argument("MX generation requires a rank-two tensor.");
    if (problem.shape[0] == 0 || problem.shape[1] == 0)
        throw std::invalid_argument("MX generation dimensions must be nonzero.");
    if (problem.blockAxis > 1) throw std::out_of_range("MX block axis exceeds the tensor rank.");
    if (problem.blockSize == 0) throw std::invalid_argument("MX block size must be nonzero.");
    if (problem.shape[0] > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
        throw std::overflow_error("MX first tensor dimension exceeds ptrdiff_t.");
    const ptrdiff_t leadingDimension = problem.leadingDimension == 0
                                           ? static_cast<ptrdiff_t>(problem.shape[0])
                                           : problem.leadingDimension;
    if (leadingDimension < static_cast<ptrdiff_t>(problem.shape[0]))
        throw std::invalid_argument(
            "MX leading dimension is smaller than the first tensor dimension.");
    if (static_cast<uintmax_t>(leadingDimension) >
        static_cast<uintmax_t>(std::numeric_limits<size_t>::max()))
        throw std::overflow_error("MX leading dimension exceeds size_t.");
    if (!isSupportedDataType(problem.dataType))
        throw std::invalid_argument("Unsupported MX data scalar type.");
    if (!isSupportedScaleType(problem.scaleType))
        throw std::invalid_argument("Unsupported MX scale scalar type.");
    if (!isSupportedTypePair(problem.dataType, problem.scaleType))
        throw std::invalid_argument("Unsupported MX data/scale scalar type combination.");
}
}  // namespace

MxTensor generateMx(Shape shape, MxDataGeneration generation, const MxGenerationOptions& options) {
    const MxGenerationInvocation problem(std::move(shape), std::move(generation), options);
    validateInvocation(problem);
    const size_t rows = problem.shape[0];
    const size_t columns = problem.shape[1];
    const size_t leadingDimension =
        problem.leadingDimension == 0 ? rows : static_cast<size_t>(problem.leadingDimension);
    const size_t physicalElementCount =
        checkedMultiply(leadingDimension, columns, "MX physical element count overflow.");
    const size_t logicalElementCount =
        checkedMultiply(rows, columns, "MX logical element count overflow.");
    if (leadingDimension > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
        throw std::overflow_error("MX leading dimension exceeds ptrdiff_t.");

    const ScaleBlocking blocking(problem);
    if (blocking.scaleCount > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("MX scale count exceeds UInt32 scale-index storage.");
    std::vector<uint8_t> dataRawValues(physicalElementCount, 0);
    std::vector<uint8_t> scaleRawValues(blocking.scaleCount, 0);
    std::vector<uint32_t> scaleIndexValues(logicalElementCount, 0);
    std::vector<float> referenceValues(logicalElementCount, 0.0f);
    const int threadCount =
        detail::operationThreadCount(std::max(logicalElementCount, physicalElementCount));

    if (problem.data.quantization() == MxDataQuantization::PreserveGeneratedEncoding)
        generateUnbounded(problem, blocking, dataRawValues, scaleRawValues, scaleIndexValues,
                          referenceValues, threadCount);
    else
        generateQuantized(problem, blocking, dataRawValues, scaleRawValues, scaleIndexValues,
                          referenceValues, threadCount);

    std::vector<std::byte> dataStorage =
        packRawValues(dataRawValues, scalarTypeInfo(problem.dataType).storageBits, threadCount);
    Tensor data = Tensor::takeOwnershipOfEncodedBackingStorage(
        problem.dataType, Layout(problem.shape, {1, static_cast<ptrdiff_t>(leadingDimension)}),
        std::move(dataStorage));
    std::vector<std::byte> scaleStorage(scaleRawValues.size());
    std::memcpy(scaleStorage.data(), scaleRawValues.data(), scaleRawValues.size());
    Tensor scales = Tensor::takeOwnershipOfEncodedBackingStorage(
        problem.scaleType, Layout::contiguousLastDimensionFastest(blocking.naturalScaleShape()),
        std::move(scaleStorage));
    Tensor scaleIndices =
        Tensor::copyNativeStorage(Layout(problem.shape, {1, static_cast<ptrdiff_t>(rows)}),
                                  std::span<const uint32_t>(scaleIndexValues));
    Tensor reference =
        Tensor::copyNativeStorage(Layout(problem.shape, {1, static_cast<ptrdiff_t>(rows)}),
                                  std::span<const float>(referenceValues));
    return {std::move(data), std::move(scales), std::move(scaleIndices), std::move(reference)};
}
}  // namespace roc::host_numerics
