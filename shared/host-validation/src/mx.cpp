// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <roc/host_validation/mx.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/data_generation.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace roc::host_validation {
namespace {
constexpr uint64_t dataRandomDomain = 0x3f84d5b5b5470917ULL;
constexpr uint64_t normalRandomDomain = 0x9216d5d98979fb1bULL;
constexpr uint64_t boundedScaleRandomDomain = 0xa24baed4963ee407ULL;
constexpr uint64_t unboundedDataRandomDomain = 0xd1b54a32d192ed03ULL;
constexpr uint64_t unboundedScaleRandomDomain = 0x94d049bb133111ebULL;
constexpr double twoPi = 6.28318530717958647692528676655900576;
constexpr int defaultMaximumThreadCount = 8;
constexpr size_t minimumElementsPerThread = 4096;

size_t checkedMultiply(size_t first, size_t second, const char* message) {
    if (first != 0 && second > std::numeric_limits<size_t>::max() / first)
        throw std::overflow_error(message);
    return first * second;
}

size_t ceilDivide(size_t value, size_t divisor) {
    return value / divisor + static_cast<size_t>(value % divisor != 0);
}

int operationThreadCount(size_t elementCount) {
#ifdef _OPENMP
    if (omp_in_parallel()) return 1;
    const int runtimeMaximum = std::max(1, omp_get_max_threads());
    const char* configuredThreadCount = std::getenv("OMP_NUM_THREADS");
    const int maximum = configuredThreadCount != nullptr && configuredThreadCount[0] != '\0'
                            ? runtimeMaximum
                            : std::min(runtimeMaximum, defaultMaximumThreadCount);
    const size_t usefulThreadCount =
        std::max(size_t{1}, ceilDivide(elementCount, minimumElementsPerThread));
    return static_cast<int>(std::min(usefulThreadCount, static_cast<size_t>(maximum)));
#else
    (void)elementCount;
    return 1;
#endif
}

struct ScaleBlocking {
    size_t blockAxis;
    size_t blockSize;
    size_t blockedExtent;
    size_t freeExtent;
    size_t blockCount;
    size_t scaleCount;

    explicit ScaleBlocking(const MxGenerationProblem& problem)
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
    return type == ScalarType::E8M0 || type == ScalarType::E5M3 || type == ScalarType::E4M3;
}

bool isSupportedTypePair(ScalarType dataType, ScalarType scaleType) {
    if (dataType == ScalarType::Float4E2M1)
        return scaleType == ScalarType::E8M0 || scaleType == ScalarType::E5M3 ||
               scaleType == ScalarType::E4M3;
    return isSupportedDataType(dataType) && scaleType == ScalarType::E8M0;
}

uint8_t dataRawForValue(ScalarType type, double value) {
    return static_cast<uint8_t>(detail::encodeBinaryFloat(type, static_cast<float>(value)));
}

double dataValueForRaw(ScalarType type, uint8_t raw) {
    return detail::decodeBinaryFloat(type, raw);
}

uint8_t scaleRawForValue(ScalarType type, double value) {
    if (type == ScalarType::E8M0) return detail::encodeE8M0(static_cast<float>(value));
    return static_cast<uint8_t>(detail::encodeBinaryFloat(type, static_cast<float>(value)));
}

double scaleValueForRaw(ScalarType type, uint8_t raw) {
    if (type == ScalarType::E8M0) return detail::decodeE8M0(raw);
    return detail::decodeBinaryFloat(type, raw);
}

uint8_t maximumScaleRaw(ScalarType type) {
    if (type == ScalarType::E8M0) return 0xfeU;
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
        (counterRandom(seed, boundedScaleRandomDomain, scaleIndex) & 1U) != 0)
        ++candidate;
    return candidate->raw;
}

bool recipesEqual(const MxGenerationRecipe& first, const MxGenerationRecipe& second) {
    return first.mode == second.mode && first.parameter0 == second.parameter0 &&
           first.parameter1 == second.parameter1;
}

std::optional<uint8_t> constantScaleRaw(ScalarType scaleType, const MxGenerationRecipe& recipe) {
    switch (recipe.mode) {
        case MxGenerationMode::Zeros:
            return scaleRawForValue(scaleType, 0.0);
        case MxGenerationMode::Ones:
        case MxGenerationMode::NegativeOnes:
        case MxGenerationMode::DenormalMinimum:
        case MxGenerationMode::DenormalMaximum:
        case MxGenerationMode::Infinity:
            return scaleRawForValue(scaleType, 1.0);
        case MxGenerationMode::Twos:
            return scaleRawForValue(scaleType, 2.0);
        case MxGenerationMode::Maximum:
            return maximumScaleRaw(scaleType);
        case MxGenerationMode::NaN:
            return scaleRawForValue(scaleType, std::numeric_limits<double>::quiet_NaN());
        default:
            return std::nullopt;
    }
}

std::optional<uint8_t> selectedConstantScale(const MxGenerationProblem& problem) {
    if (problem.scale && !recipesEqual(problem.data, *problem.scale)) {
        if (const auto explicitScale = constantScaleRaw(problem.scaleType, *problem.scale))
            return explicitScale;
        throw std::invalid_argument(
            "Independent MX scale generation currently supports only constant recipes.");
    }

    switch (problem.data.mode) {
        case MxGenerationMode::Zeros:
            return scaleRawForValue(problem.scaleType, 0.0);
        case MxGenerationMode::NaN:
            return scaleRawForValue(problem.scaleType, std::numeric_limits<double>::quiet_NaN());
        case MxGenerationMode::Bounded:
        case MxGenerationMode::BoundedAlternatingSign:
        case MxGenerationMode::Unbounded:
        case MxGenerationMode::Trigonometric:
        case MxGenerationMode::Normal:
            return std::nullopt;
        default:
            return scaleRawForValue(problem.scaleType, 1.0);
    }
}

double generatedValue(const MxGenerationProblem& problem, size_t row, size_t column,
                      size_t logicalIndex) {
    const MxGenerationRecipe& recipe = problem.data;
    switch (recipe.mode) {
        case MxGenerationMode::Bounded: {
            const double unit =
                detail::indexedUniformUnit(problem.seed, dataRandomDomain, logicalIndex);
            return recipe.parameter0 + unit * (recipe.parameter1 - recipe.parameter0);
        }
        case MxGenerationMode::BoundedAlternatingSign: {
            const double maximumMagnitude =
                std::max(std::abs(recipe.parameter0), std::abs(recipe.parameter1));
            const double magnitude =
                maximumMagnitude *
                detail::indexedUniformUnit(problem.seed, dataRandomDomain, logicalIndex);
            return (logicalIndex & 1U) == 0 ? magnitude : -magnitude;
        }
        case MxGenerationMode::Unbounded:
            throw std::logic_error("Unbounded MX generation uses raw encodings.");
        case MxGenerationMode::Identity:
            return row == column ? 1.0 : 0.0;
        case MxGenerationMode::Ones:
            return 1.0;
        case MxGenerationMode::Zeros:
            return 0.0;
        case MxGenerationMode::Sequential:
            return static_cast<double>(((row % 256U) * (problem.shape[1] % 256U) + column % 256U) %
                                       256U);
        case MxGenerationMode::RowIndex:
            return static_cast<double>(row % 256U);
        case MxGenerationMode::ColumnIndex:
            return static_cast<double>(column % 256U);
        case MxGenerationMode::Checkerboard:
            return ((row + column) & 1U) == 0 ? 1.0 : 0.0;
        case MxGenerationMode::ScaledDiagonal:
            return row == column ? static_cast<double>(row + 1U) : 0.0;
        case MxGenerationMode::Twos:
            return 2.0;
        case MxGenerationMode::NegativeOnes:
            return -1.0;
        case MxGenerationMode::Maximum:
            return detail::typeMaximum(problem.dataType);
        case MxGenerationMode::DenormalMinimum:
            return detail::typeDenormalMinimum(problem.dataType);
        case MxGenerationMode::DenormalMaximum:
            return detail::typeDenormalMaximum(problem.dataType);
        case MxGenerationMode::NaN:
            return std::numeric_limits<double>::quiet_NaN();
        case MxGenerationMode::Infinity:
            return std::numeric_limits<double>::infinity();
        case MxGenerationMode::Trigonometric: {
            const double angle =
                twoPi * detail::indexedUniformUnit(problem.seed, dataRandomDomain, logicalIndex);
            return std::cos(angle);
        }
        case MxGenerationMode::Normal: {
            const uint64_t firstIndex = static_cast<uint64_t>(logicalIndex) * 2U;
            const double first =
                detail::indexedUniformUnit(problem.seed, normalRandomDomain, firstIndex);
            const double second =
                detail::indexedUniformUnit(problem.seed, normalRandomDomain, firstIndex + 1U);
            const double standardNormal =
                std::sqrt(-2.0 * std::log(first)) * std::cos(twoPi * second);
            return recipe.parameter0 + recipe.parameter1 * standardNormal;
        }
        case MxGenerationMode::UniformInteger:
            return indexedUniformInteger(problem.seed, dataRandomDomain, logicalIndex,
                                         static_cast<int>(recipe.parameter0),
                                         static_cast<int>(recipe.parameter1));
    }
    throw std::invalid_argument("Unsupported MX generation mode.");
}

std::vector<uint8_t> finiteDataRawCandidates(ScalarType dataType) {
    const uint16_t bits = scalarTypeInfo(dataType).storageBits;
    const uint32_t count = 1U << bits;
    std::vector<uint8_t> candidates;
    candidates.reserve(count);
    for (uint32_t raw = 0; raw < count; ++raw) {
        const uint8_t encoded = static_cast<uint8_t>(raw);
        if (std::isfinite(dataValueForRaw(dataType, encoded))) candidates.push_back(encoded);
    }
    if (candidates.empty()) throw std::invalid_argument("MX data type has no finite encodings.");
    return candidates;
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

template <typename Native>
Tensor nativeTensor(ScalarType type, Layout layout, const std::vector<Native>& values) {
    std::vector<std::byte> storage(
        checkedMultiply(values.size(), sizeof(Native), "MX native tensor storage overflow."));
    std::memcpy(storage.data(), values.data(), storage.size());
    return Tensor::fromStorage(type, std::move(layout), std::move(storage));
}

void generateUnbounded(const MxGenerationProblem& problem, const ScaleBlocking& blocking,
                       std::vector<uint8_t>& dataRawValues, std::vector<uint8_t>& scaleRawValues,
                       std::vector<uint32_t>& scaleIndexValues, std::vector<float>& referenceValues,
                       int threadCount) {
    (void)threadCount;
    const std::vector<uint8_t> dataCandidates = finiteDataRawCandidates(problem.dataType);
    const std::vector<double> dataValues = decodedDataValues(problem.dataType);
    const std::optional<uint8_t> fixedScale = selectedConstantScale(problem);
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
            fixedScale ? *fixedScale
                       : scaleCandidates[counterRandom(problem.seed, unboundedScaleRandomDomain,
                                                       scaleIndex) %
                                         scaleCandidates.size()]
                             .raw;
        scaleRawValues[scaleIndex] = scaleRaw;
        const double scaleValue = scaleValueForRaw(problem.scaleType, scaleRaw);

        for (size_t offset = 0; offset < blocking.blockElementCount(block); ++offset) {
            const auto [row, column] = blocking.dataCoordinates(block, freeCoordinate, offset);
            const size_t logicalIndex = row + column * rows;
            const size_t physicalIndex = row + column * leadingDimension;
            const uint8_t dataRaw =
                dataCandidates[counterRandom(problem.seed, unboundedDataRandomDomain,
                                             logicalIndex) %
                               dataCandidates.size()];
            dataRawValues[physicalIndex] = dataRaw;
            scaleIndexValues[logicalIndex] = static_cast<uint32_t>(scaleIndex);
            referenceValues[logicalIndex] = static_cast<float>(dataValues[dataRaw] * scaleValue);
        }
    }
}

void generateQuantized(const MxGenerationProblem& problem, const ScaleBlocking& blocking,
                       std::vector<uint8_t>& dataRawValues, std::vector<uint8_t>& scaleRawValues,
                       std::vector<uint32_t>& scaleIndexValues, std::vector<float>& referenceValues,
                       int threadCount) {
    (void)threadCount;
    const size_t rows = problem.shape[0];
    const size_t leadingDimension =
        problem.leadingDimension == 0 ? rows : static_cast<size_t>(problem.leadingDimension);

    const std::optional<uint8_t> fixedScale = selectedConstantScale(problem);
    const std::vector<ScaleCandidate> scaleCandidates =
        fixedScale ? std::vector<ScaleCandidate>{}
                   : finiteNonzeroScaleCandidates(problem.scaleType);
    const std::vector<double> dataValues = decodedDataValues(problem.dataType);
    const double maximumDataValue = detail::typeMaximum(problem.dataType);

#ifdef _OPENMP
#pragma omp parallel num_threads(threadCount)
#endif
    {
        std::vector<double> blockValues(
            fixedScale ? 0 : std::min(blocking.blockSize, blocking.blockedExtent));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t scaleIndex = 0; scaleIndex < blocking.scaleCount; ++scaleIndex) {
            const auto [block, freeCoordinate] = blocking.blockAndFreeCoordinate(scaleIndex);
            const size_t blockElementCount = blocking.blockElementCount(block);
            uint8_t scaleRaw = fixedScale.value_or(0);

            if (!fixedScale) {
                bool hasNaN = false;
                double maximumMagnitude = 0.0;
                for (size_t offset = 0; offset < blockElementCount; ++offset) {
                    const auto [row, column] =
                        blocking.dataCoordinates(block, freeCoordinate, offset);
                    const size_t logicalIndex = row + column * rows;
                    const double value = generatedValue(problem, row, column, logicalIndex);
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
                        problem.data.mode == MxGenerationMode::Bounded ||
                        problem.data.mode == MxGenerationMode::BoundedAlternatingSign;
                    scaleRaw = scaleAtLeast(maximumMagnitude / maximumDataValue, scaleCandidates,
                                            allowLargerCandidate, problem.seed, scaleIndex);
                }
            }

            scaleRawValues[scaleIndex] = scaleRaw;
            const double scaleValue = scaleValueForRaw(problem.scaleType, scaleRaw);
            for (size_t offset = 0; offset < blockElementCount; ++offset) {
                const auto [row, column] = blocking.dataCoordinates(block, freeCoordinate, offset);
                const size_t logicalIndex = row + column * rows;
                const size_t physicalIndex = row + column * leadingDimension;
                const double sourceValue = fixedScale
                                               ? generatedValue(problem, row, column, logicalIndex)
                                               : blockValues[offset];
                const double scaledValue =
                    sourceValue == 0.0 ? sourceValue : sourceValue / scaleValue;
                uint8_t dataRaw = dataRawForValue(problem.dataType, scaledValue);
                if (problem.data.mode == MxGenerationMode::Bounded) {
                    dataRaw = constrainDataRawToInterval(problem.dataType, dataRaw, scaleValue,
                                                         problem.data.parameter0,
                                                         problem.data.parameter1, dataValues);
                } else if (problem.data.mode == MxGenerationMode::BoundedAlternatingSign) {
                    const double maximumMagnitude = std::max(std::abs(problem.data.parameter0),
                                                             std::abs(problem.data.parameter1));
                    dataRaw =
                        constrainDataRawToInterval(problem.dataType, dataRaw, scaleValue,
                                                   -maximumMagnitude, maximumMagnitude, dataValues);
                }
                dataRawValues[physicalIndex] = dataRaw;
                scaleIndexValues[logicalIndex] = static_cast<uint32_t>(scaleIndex);
                referenceValues[logicalIndex] =
                    static_cast<float>(dataValues[dataRaw] * scaleValue);
            }
        }
    }
}

void validateRecipe(const MxGenerationProblem& problem) {
    const MxGenerationRecipe& recipe = problem.data;
    if (recipe.mode == MxGenerationMode::Bounded) {
        if (!std::isfinite(recipe.parameter0) || !std::isfinite(recipe.parameter1) ||
            !(recipe.parameter0 < recipe.parameter1))
            throw std::invalid_argument("MX bounded generation requires finite minimum < maximum.");
    }
    if (recipe.mode == MxGenerationMode::BoundedAlternatingSign &&
        (!std::isfinite(recipe.parameter0) || !std::isfinite(recipe.parameter1)))
        throw std::invalid_argument("MX alternating bounded generation requires finite bounds.");
    if (recipe.mode == MxGenerationMode::Normal) {
        if (!std::isfinite(recipe.parameter0) || !std::isfinite(recipe.parameter1) ||
            recipe.parameter1 < 0.0)
            throw std::invalid_argument(
                "MX normal generation requires a finite mean and nonnegative finite standard "
                "deviation.");
    }
    if (recipe.mode == MxGenerationMode::UniformInteger) {
        if (!std::isfinite(recipe.parameter0) || !std::isfinite(recipe.parameter1) ||
            recipe.parameter0 < std::numeric_limits<int>::min() ||
            recipe.parameter0 > std::numeric_limits<int>::max() ||
            recipe.parameter1 < std::numeric_limits<int>::min() ||
            recipe.parameter1 > std::numeric_limits<int>::max() ||
            static_cast<int>(recipe.parameter0) > static_cast<int>(recipe.parameter1))
            throw std::invalid_argument("MX integer generation bounds are invalid.");
    }
    if (recipe.mode == MxGenerationMode::Infinity &&
        !scalarTypeInfo(problem.dataType).supportsInfinity)
        throw std::invalid_argument("MX data type has no infinity encoding.");
}

void validateProblem(const MxGenerationProblem& problem) {
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
    if (problem.scale && !recipesEqual(problem.data, *problem.scale) &&
        !constantScaleRaw(problem.scaleType, *problem.scale))
        throw std::invalid_argument(
            "Independent MX scale generation currently supports only constant recipes.");
    validateRecipe(problem);
}
}  // namespace

MxGenerationResult generateMx(const MxGenerationProblem& problem) {
    validateProblem(problem);
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
        operationThreadCount(std::max(logicalElementCount, physicalElementCount));

    if (problem.data.mode == MxGenerationMode::Unbounded)
        generateUnbounded(problem, blocking, dataRawValues, scaleRawValues, scaleIndexValues,
                          referenceValues, threadCount);
    else
        generateQuantized(problem, blocking, dataRawValues, scaleRawValues, scaleIndexValues,
                          referenceValues, threadCount);

    Tensor data = Tensor::fromStorage(
        problem.dataType, Layout(problem.shape, {1, static_cast<ptrdiff_t>(leadingDimension)}),
        packRawValues(dataRawValues, scalarTypeInfo(problem.dataType).storageBits, threadCount));
    std::vector<std::byte> scaleStorage(scaleRawValues.size());
    std::memcpy(scaleStorage.data(), scaleRawValues.data(), scaleRawValues.size());
    Tensor scales = Tensor::fromStorage(
        problem.scaleType, Layout::contiguous(Shape{blocking.scaleCount}), std::move(scaleStorage));
    Tensor scaleIndices =
        nativeTensor(ScalarType::UInt32, Layout(problem.shape, {1, static_cast<ptrdiff_t>(rows)}),
                     scaleIndexValues);
    Tensor reference =
        nativeTensor(ScalarType::Float32, Layout(problem.shape, {1, static_cast<ptrdiff_t>(rows)}),
                     referenceValues);
    return {std::move(data), std::move(scales), std::move(scaleIndices), std::move(reference)};
}
}  // namespace roc::host_validation
