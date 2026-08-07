// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/ocp_e2m1_mxfp4.hpp>
#include <mxDataGenerator/ocp_e2m3_mxfp6.hpp>
#include <mxDataGenerator/ocp_e3m2_mxfp6.hpp>
#include <mxDataGenerator/ocp_e4m3_mxfp8.hpp>
#include <mxDataGenerator/ocp_e5m2_mxfp8.hpp>
#include <roc/host_validation/mx.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace roc::host_validation {
namespace {
DGen::DataInitMode dataInitMode(const MxGenerationRecipe& recipe) {
    switch (recipe.mode) {
        case MxGenerationMode::Bounded:
            return DGen::Bounded{};
        case MxGenerationMode::BoundedAlternatingSign:
            return DGen::BoundedAlternatingSign{};
        case MxGenerationMode::Unbounded:
            return DGen::Unbounded{};
        case MxGenerationMode::Identity:
            return DGen::Identity{};
        case MxGenerationMode::Ones:
            return DGen::Ones{};
        case MxGenerationMode::Zeros:
            return DGen::Zeros{};
        case MxGenerationMode::Sequential:
            return DGen::Sequential{};
        case MxGenerationMode::RowIndex:
            return DGen::RowIndex{};
        case MxGenerationMode::ColumnIndex:
            return DGen::ColIndex{};
        case MxGenerationMode::Checkerboard:
            return DGen::Checkerboard{};
        case MxGenerationMode::ScaledDiagonal:
            return DGen::ScaledDiagonal{};
        case MxGenerationMode::Twos:
            return DGen::Twos{};
        case MxGenerationMode::NegativeOnes:
            return DGen::NegOnes{};
        case MxGenerationMode::Maximum:
            return DGen::MaxVals{};
        case MxGenerationMode::DenormalMinimum:
            return DGen::DenormMins{};
        case MxGenerationMode::DenormalMaximum:
            return DGen::DenormMaxs{};
        case MxGenerationMode::NaN:
            return DGen::NaNs{};
        case MxGenerationMode::Infinity:
            return DGen::Infs{};
        case MxGenerationMode::Trigonometric:
            return DGen::TrigonometricFromFloat{};
        case MxGenerationMode::Normal:
            return DGen::NormalFromFloat{recipe.parameter0, recipe.parameter1};
        case MxGenerationMode::UniformInteger:
            return DGen::RandInt{static_cast<int>(recipe.parameter0),
                                 static_cast<int>(recipe.parameter1)};
    }
    throw std::invalid_argument("Unsupported MX generation mode.");
}

DGen::DataGeneratorOptions generatorOptions(const MxGenerationProblem& problem) {
    DGen::DataGeneratorOptions options;
    options.blockScaling = static_cast<DGen::index_t>(problem.blockSize);
    options.initMode = dataInitMode(problem.data);
    options.min = problem.data.parameter0;
    options.max = problem.data.parameter1;
    options.forceDenorm = false;
    if (problem.scale) options.scaleInitMode = dataInitMode(*problem.scale);
    return options;
}

std::vector<std::byte> byteStorage(const std::vector<uint8_t>& bytes) {
    std::vector<std::byte> storage(bytes.size());
    std::memcpy(storage.data(), bytes.data(), bytes.size());
    return storage;
}

template <typename DataType>
inline constexpr bool packedMxData = !std::is_same_v<DataType, DGen::ocp_e4m3_mxfp8> &&
                                     !std::is_same_v<DataType, DGen::ocp_e5m2_mxfp8>;

template <typename DataType>
std::vector<uint8_t> unpackData(const std::vector<uint8_t>& packed, size_t elements) {
    if constexpr (!packedMxData<DataType>)
        return {packed.begin(), packed.begin() + elements};
    else {
        std::vector<uint8_t> unpacked(elements);
        constexpr size_t bits = std::is_same_v<DataType, DGen::ocp_e2m1_mxfp4> ||
                                        std::is_same_v<DataType, DGen::ocp_e2m1_mxfp4_e4m3> ||
                                        std::is_same_v<DataType, DGen::ocp_e2m1_mxfp4_e5m3>
                                    ? 4
                                    : 6;
        for (size_t index = 0; index < elements; ++index) {
            const size_t bitOffset = index * bits;
            const size_t byteIndex = bitOffset / 8;
            const size_t bitIndex = bitOffset % 8;
            uint16_t word = packed[byteIndex];
            if (byteIndex + 1 < packed.size())
                word |= static_cast<uint16_t>(packed[byteIndex + 1]) << 8;
            unpacked[index] = static_cast<uint8_t>((word >> bitIndex) & ((1U << bits) - 1U));
        }
        return unpacked;
    }
}

template <typename DataType>
std::vector<uint8_t> packData(const std::vector<uint8_t>& unpacked) {
    if constexpr (!packedMxData<DataType>)
        return unpacked;
    else {
        constexpr size_t bits = std::is_same_v<DataType, DGen::ocp_e2m1_mxfp4> ||
                                        std::is_same_v<DataType, DGen::ocp_e2m1_mxfp4_e4m3> ||
                                        std::is_same_v<DataType, DGen::ocp_e2m1_mxfp4_e5m3>
                                    ? 4
                                    : 6;
        std::vector<uint8_t> packed((unpacked.size() * bits + 7) / 8, 0);
        for (size_t index = 0; index < unpacked.size(); ++index) {
            const size_t bitOffset = index * bits;
            const size_t byteIndex = bitOffset / 8;
            const size_t bitIndex = bitOffset % 8;
            uint16_t word = packed[byteIndex];
            if (byteIndex + 1 < packed.size())
                word |= static_cast<uint16_t>(packed[byteIndex + 1]) << 8;
            const uint16_t mask = static_cast<uint16_t>(((1U << bits) - 1U) << bitIndex);
            word = static_cast<uint16_t>(
                (word & ~mask) | ((static_cast<uint16_t>(unpacked[index]) << bitIndex) & mask));
            packed[byteIndex] = static_cast<uint8_t>(word);
            if (byteIndex + 1 < packed.size())
                packed[byteIndex + 1] = static_cast<uint8_t>(word >> 8);
        }
        return packed;
    }
}

template <typename Native>
Tensor nativeTensor(ScalarType type, Layout layout, const std::vector<Native>& values) {
    std::vector<std::byte> storage(values.size() * sizeof(Native));
    std::memcpy(storage.data(), values.data(), storage.size());
    return Tensor::fromStorage(type, std::move(layout), std::move(storage));
}

template <typename DataType>
MxGenerationResult generateTyped(const MxGenerationProblem& problem) {
    const size_t rows = problem.shape[0];
    const size_t columns = problem.shape[1];
    const size_t leadingDimension =
        problem.leadingDimension == 0 ? rows : static_cast<size_t>(problem.leadingDimension);
    const bool blockAxisIsContiguous = problem.blockAxis == 0;

    DGen::DataGenerator<DataType> generator;
    generator.setSeed(problem.seed);
    generator.generate({static_cast<DGen::index_t>(rows), static_cast<DGen::index_t>(columns)},
                       {1, static_cast<DGen::index_t>(leadingDimension)},
                       generatorOptions(problem));

    const std::vector<uint8_t> generatedData = generator.getDataBytes();
    const std::vector<uint8_t> generatedScales = generator.getScaleBytes();
    const size_t physicalElements = leadingDimension * columns;
    const uint16_t scaleBits = scalarTypeInfo(problem.scaleType).storageBits;
    if (scaleBits == 0 || (generatedScales.size() * 8) % scaleBits != 0)
        throw std::invalid_argument(
            "MX scale storage does not contain a whole number of scalar values.");
    const size_t scaleElements = generatedScales.size() * 8 / scaleBits;
    const std::vector<uint8_t> sourceData = unpackData<DataType>(generatedData, physicalElements);
    std::vector<uint8_t> outputData =
        blockAxisIsContiguous ? sourceData : std::vector<uint8_t>(physicalElements, 0);
    std::vector<uint32_t> scaleIndexValues(rows * columns);
    std::vector<float> referenceValues(rows * columns);
    const size_t freeDimension = rows;
    const size_t tailStartFree = freeDimension % problem.blockSize == 0
                                     ? freeDimension
                                     : freeDimension / problem.blockSize * problem.blockSize;
    std::atomic<bool> invalidMapping{false};

#pragma omp parallel for
    for (size_t logicalIndex = 0; logicalIndex < rows * columns; ++logicalIndex) {
        const size_t row = logicalIndex % rows;
        const size_t column = logicalIndex / rows;
        const size_t destinationIndex = row + column * leadingDimension;
        size_t sourceIndex = destinationIndex;
        size_t scaleIndex = generator.scaleIndexForData(sourceIndex);
        if (!blockAxisIsContiguous) {
            const size_t blockIndex = column / problem.blockSize;
            const size_t offsetInBlock = column - blockIndex * problem.blockSize;
            const size_t desiredScaleIndex = blockIndex * freeDimension + row;
            const size_t desiredSourceIndex = desiredScaleIndex * problem.blockSize + offsetInBlock;
            if (row < tailStartFree && desiredScaleIndex < scaleElements &&
                desiredSourceIndex < physicalElements) {
                scaleIndex = desiredScaleIndex;
                sourceIndex = desiredSourceIndex;
            } else {
                sourceIndex = row + column * freeDimension;
                scaleIndex = generator.scaleIndexForData(sourceIndex);
            }
        }
        if (sourceIndex >= physicalElements || scaleIndex >= scaleElements) {
            invalidMapping.store(true, std::memory_order_relaxed);
            continue;
        }

        outputData[destinationIndex] = sourceData[sourceIndex];
        scaleIndexValues[logicalIndex] = static_cast<uint32_t>(scaleIndex);
        referenceValues[logicalIndex] = DGen::toFloat<DataType>(
            generatedScales.data(), sourceData.data(), static_cast<DGen::index_t>(scaleIndex),
            static_cast<DGen::index_t>(sourceIndex));
    }
    if (invalidMapping.load(std::memory_order_relaxed))
        throw std::out_of_range("MX generator produced an out-of-range data/scale mapping.");

    const std::vector<uint8_t> outputBytes =
        blockAxisIsContiguous ? generatedData : packData<DataType>(outputData);
    Tensor data = Tensor::fromStorage(
        problem.dataType, Layout(problem.shape, {1, static_cast<ptrdiff_t>(leadingDimension)}),
        byteStorage(outputBytes));
    Tensor scales = Tensor::fromStorage(problem.scaleType, Layout::contiguous(Shape{scaleElements}),
                                        byteStorage(generatedScales));
    Tensor scaleIndices =
        nativeTensor(ScalarType::UInt32, Layout(problem.shape, {1, static_cast<ptrdiff_t>(rows)}),
                     scaleIndexValues);
    Tensor reference =
        nativeTensor(ScalarType::Float32, Layout(problem.shape, {1, static_cast<ptrdiff_t>(rows)}),
                     referenceValues);
    return {std::move(data), std::move(scales), std::move(scaleIndices), std::move(reference)};
}

void validateProblem(const MxGenerationProblem& problem) {
    if (problem.shape.rank() != 2)
        throw std::invalid_argument("MX generation requires a rank-two tensor.");
    if (problem.shape[0] == 0 || problem.shape[1] == 0)
        throw std::invalid_argument("MX generation dimensions must be nonzero.");
    if (problem.blockAxis > 1) throw std::out_of_range("MX block axis exceeds the tensor rank.");
    if (problem.blockSize == 0) throw std::invalid_argument("MX block size must be nonzero.");
    const ptrdiff_t leadingDimension = problem.leadingDimension == 0
                                           ? static_cast<ptrdiff_t>(problem.shape[0])
                                           : problem.leadingDimension;
    if (leadingDimension < static_cast<ptrdiff_t>(problem.shape[0]))
        throw std::invalid_argument(
            "MX leading dimension is smaller than the first tensor dimension.");
    if (scalarTypeInfo(problem.scaleType).storageBits != 8)
        throw std::invalid_argument("MX generation currently requires an eight-bit scale format.");
}
}  // namespace

MxGenerationResult generateMx(const MxGenerationProblem& problem) {
    validateProblem(problem);
    switch (problem.dataType) {
        case ScalarType::Float8E5M2:
            if (problem.scaleType != ScalarType::E8M0) break;
            return generateTyped<DGen::ocp_e5m2_mxfp8>(problem);
        case ScalarType::Float8E4M3:
            if (problem.scaleType != ScalarType::E8M0) break;
            return generateTyped<DGen::ocp_e4m3_mxfp8>(problem);
        case ScalarType::Float6E2M3:
            if (problem.scaleType != ScalarType::E8M0) break;
            return generateTyped<DGen::ocp_e2m3_mxfp6>(problem);
        case ScalarType::Float6E3M2:
            if (problem.scaleType != ScalarType::E8M0) break;
            return generateTyped<DGen::ocp_e3m2_mxfp6>(problem);
        case ScalarType::Float4E2M1:
            if (problem.scaleType == ScalarType::Float8E4M3)
                return generateTyped<DGen::ocp_e2m1_mxfp4_e4m3>(problem);
            if (problem.scaleType == ScalarType::E5M3)
                return generateTyped<DGen::ocp_e2m1_mxfp4_e5m3>(problem);
            if (problem.scaleType == ScalarType::E8M0)
                return generateTyped<DGen::ocp_e2m1_mxfp4>(problem);
            break;
        default:
            break;
    }
    throw std::invalid_argument("Unsupported MX data/scale scalar type combination.");
}
}  // namespace roc::host_validation
