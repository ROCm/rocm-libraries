// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from TensileLite initialization descriptors to
// product-independent host-validation generation recipes.

#include "DataInitialization.hpp"

#include <roc/host_validation/adapters/tensilelite/HostValidationBridge.hpp>
#include <roc/host_validation/validation.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace TensileLite::Client
{
    std::uint64_t stableDataInitializationStream(std::string_view semanticName)
    {
        std::uint64_t hash = 1469598103934665603ULL;
        for(const unsigned char character : semanticName)
        {
            hash ^= character;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    namespace
    {
        using roc::host_validation::GenerationOptions;
        using roc::host_validation::GenerationPattern;
        using roc::host_validation::Layout;
        using roc::host_validation::ScalarType;
        using roc::host_validation::Shape;
        using roc::host_validation::StructuredSparsityPattern;
        using roc::host_validation::StructuredSparsityProblem;
        using roc::host_validation::StructuredSparsitySelection;
        using roc::host_validation::StructuredSparsitySliceRange;
        using roc::host_validation::Tensor;

        std::optional<ScalarType> generationScalarType(rocisa::DataType type)
        {
            switch(type)
            {
            case rocisa::DataType::Float:
            case rocisa::DataType::Double:
            case rocisa::DataType::ComplexFloat:
            case rocisa::DataType::ComplexDouble:
            case rocisa::DataType::Half:
            case rocisa::DataType::BFloat16:
            case rocisa::DataType::Int8:
            case rocisa::DataType::Int32:
            case rocisa::DataType::Int64:
            case rocisa::DataType::XFloat32:
            case rocisa::DataType::Float8:
            case rocisa::DataType::BFloat8:
            case rocisa::DataType::Float8_fnuz:
            case rocisa::DataType::BFloat8_fnuz:
            case rocisa::DataType::Float6:
            case rocisa::DataType::BFloat6:
            case rocisa::DataType::Float4:
            case rocisa::DataType::E5M3:
                return toHostValidationScalarType(type);
            case rocisa::DataType::E8:
                // Generation operates on the shared raw OCP E8M0 encoding.
                // Tensile's later numerical interpretation of raw zero remains
                // a product compatibility concern.
                return ScalarType::E8M0;
            default:
                return std::nullopt;
            }
        }

        std::optional<GenerationOptions> generationOptions(rocisa::DataType      dataType,
                                                           InitMode              mode,
                                                           bool                  problemDependent,
                                                           std::uint64_t         seed,
                                                           std::optional<double> freeValue
                                                           = std::nullopt)
        {
            GenerationOptions options;
            options.seed = seed;
            switch(mode)
            {
            case InitMode::Zero:
                if(dataType == rocisa::DataType::E8)
                    options.real.pattern = GenerationPattern::RawConstant;
                return options;
            case InitMode::One:
                options.real.pattern    = GenerationPattern::Constant;
                options.real.parameter0 = 1;
                return options;
            case InitMode::Two:
                options.real.pattern    = GenerationPattern::Constant;
                options.real.parameter0 = 2;
                return options;
            case InitMode::NegOne:
                options.real.pattern    = GenerationPattern::Constant;
                options.real.parameter0 = -1;
                return options;
            case InitMode::Max:
                if(dataType == rocisa::DataType::BFloat16)
                {
                    // Tensile's BF16 conversion rounds Float32 max to +Inf.
                    options.real.pattern = GenerationPattern::TypeInfinity;
                }
                else if(dataType == rocisa::DataType::BFloat6)
                {
                    // Preserve the existing Tensile compatibility value even
                    // though the E3M2 format itself has a larger finite range.
                    options.real.pattern    = GenerationPattern::Constant;
                    options.real.parameter0 = 7.5;
                }
                else
                {
                    options.real.pattern = GenerationPattern::TypeMaximum;
                }
                options.imaginary = options.real;
                return options;
            case InitMode::DenormMin:
                if(dataType == rocisa::DataType::BFloat6)
                {
                    options.real.pattern    = GenerationPattern::Constant;
                    options.real.parameter0 = 0.125;
                }
                else
                {
                    options.real.pattern = GenerationPattern::TypeDenormalMinimum;
                }
                options.imaginary    = options.real;
                return options;
            case InitMode::DenormMax:
                if(dataType == rocisa::DataType::BFloat6)
                {
                    options.real.pattern    = GenerationPattern::Constant;
                    options.real.parameter0 = 0.875;
                }
                else
                {
                    options.real.pattern = GenerationPattern::TypeDenormalMaximum;
                }
                options.imaginary    = options.real;
                return options;
            case InitMode::NaN:
                options.real.pattern = GenerationPattern::TypeNaN;
                options.imaginary    = options.real;
                return options;
            case InitMode::Inf:
                if(dataType == rocisa::DataType::Float8
                   || dataType == rocisa::DataType::Float8_fnuz
                   || dataType == rocisa::DataType::BFloat8_fnuz)
                    options.real.pattern = GenerationPattern::TypeNaN;
                else
                    options.real.pattern = GenerationPattern::TypeInfinity;
                options.imaginary = options.real;
                return options;
            case InitMode::BadInput:
                if(dataType == rocisa::DataType::Int8
                   || dataType == rocisa::DataType::Int32)
                    options.real.pattern = GenerationPattern::TypeMaximum;
                else
                    options.real.pattern = GenerationPattern::TypeNaN;
                options.imaginary = options.real;
                return options;
            case InitMode::BadOutput:
                if(dataType == rocisa::DataType::Int8
                   || dataType == rocisa::DataType::Int32)
                    options.real.pattern = GenerationPattern::TypeLowest;
                else if(dataType == rocisa::DataType::Float8
                        || dataType == rocisa::DataType::Float8_fnuz
                        || dataType == rocisa::DataType::BFloat8_fnuz
                        || dataType == rocisa::DataType::E8
                        || dataType == rocisa::DataType::E5M3)
                    options.real.pattern = GenerationPattern::TypeNaN;
                else
                    options.real.pattern = GenerationPattern::TypeInfinity;
                options.imaginary = options.real;
                return options;
            case InitMode::Random:
                options.real.pattern = GenerationPattern::UniformInteger;
                if(dataType == rocisa::DataType::E8)
                {
                    options.real.pattern    = GenerationPattern::RandomEncodedExponent;
                    options.real.parameter0 = -3;
                    options.real.parameter1 = 3;
                }
                else if(dataType == rocisa::DataType::E5M3)
                {
                    options.real.pattern    = GenerationPattern::AbsoluteUniformInteger;
                    options.real.parameter0 = -3;
                    options.real.parameter1 = 3;
                }
                else if(dataType == rocisa::DataType::Float4)
                {
                    options.real.pattern    = GenerationPattern::UniformRawInteger;
                    options.real.parameter0 = 0;
                    options.real.parameter1 = 14;
                }
                else if(dataType == rocisa::DataType::Float)
                {
                    options.real.parameter0 = -100;
                    options.real.parameter1 = 100;
                }
                else if(dataType == rocisa::DataType::Double)
                {
                    options.real.parameter0 = -1000;
                    options.real.parameter1 = 1000;
                }
                else
                {
                    options.real.parameter0 = -3;
                    options.real.parameter1 = 3;
                }
                options.imaginary        = options.real;
                options.imaginary.stream = 1;
                return options;
            case InitMode::RandomNarrow:
                if(dataType == rocisa::DataType::E8
                   || dataType == rocisa::DataType::E5M3
                   || dataType == rocisa::DataType::Float4)
                    return generationOptions(
                        dataType, InitMode::Random, problemDependent, seed, freeValue);
                if(dataType == rocisa::DataType::Int8
                   || dataType == rocisa::DataType::Int32
                   || dataType == rocisa::DataType::Int64)
                {
                    options.real.pattern    = GenerationPattern::UniformInteger;
                    options.real.parameter0 = -3;
                    options.real.parameter1 = 3;
                }
                else
                {
                    options.real.pattern = GenerationPattern::RandomEncodedExponent;
                    options.real.parameter0 =
                        dataType == rocisa::DataType::Double ? -189 : -100;
                    options.real.parameter1 = 0;
                    if(dataType == rocisa::DataType::Float6
                       || dataType == rocisa::DataType::BFloat6)
                        options.real.sourceType = ScalarType::Float32;
                }
                options.imaginary        = options.real;
                options.imaginary.stream = 1;
                return options;
            case InitMode::RandomNegPosLimited:
                if(dataType == rocisa::DataType::E8)
                {
                    options.real.pattern    = GenerationPattern::UniformRawInteger;
                    options.real.parameter0 = -128;
                    options.real.parameter1 = 128;
                }
                else if(dataType == rocisa::DataType::E5M3)
                {
                    options.real.pattern    = GenerationPattern::AbsoluteUniformInteger;
                    options.real.parameter0 = -128;
                    options.real.parameter1 = 128;
                }
                else if(dataType == rocisa::DataType::Int8
                        || dataType == rocisa::DataType::Int32
                        || dataType == rocisa::DataType::Int64)
                {
                    options.real.pattern    = GenerationPattern::UniformInteger;
                    options.real.parameter0 = -128;
                    options.real.parameter1 = 128;
                }
                else
                {
                    options.real.pattern    = GenerationPattern::UniformReal;
                    options.real.parameter0 = -1;
                    options.real.parameter1 = 1;
                }
                options.imaginary        = options.real;
                options.imaginary.stream = 1;
                return options;
            case InitMode::UniformLowPrecision:
                options.real.pattern = GenerationPattern::UniformReal;
                if(dataType == rocisa::DataType::Float4)
                {
                    options.real.parameter0 = -6.0;
                    options.real.parameter1 = 6.0;
                }
                else if(dataType == rocisa::DataType::Float6
                        || dataType == rocisa::DataType::BFloat6)
                {
                    options.real.parameter0 = -7.5;
                    options.real.parameter1 = 7.5;
                }
                else
                {
                    return std::nullopt;
                }
                options.imaginary        = options.real;
                options.imaginary.stream = 1;
                return options;
            case InitMode::Free:
                if(!freeValue)
                    return std::nullopt;
                options.real.pattern    = GenerationPattern::Constant;
                options.real.parameter0 = *freeValue;
                return options;
            case InitMode::SerialIdx:
                options.real.pattern = GenerationPattern::SerialIndex;
                options.imaginary    = options.real;
                return options;
            case InitMode::SerialDim0:
                if(!problemDependent)
                    return std::nullopt;
                options.real.pattern   = dataType == rocisa::DataType::Half
                                             ? GenerationPattern::RawSerialDimension
                                             : GenerationPattern::SerialDimension;
                options.real.dimension = 0;
                options.imaginary      = options.real;
                return options;
            case InitMode::SerialDim1:
                if(!problemDependent)
                    return std::nullopt;
                options.real.pattern   = dataType == rocisa::DataType::Half
                                             ? GenerationPattern::RawSerialDimension
                                             : GenerationPattern::SerialDimension;
                options.real.dimension = 1;
                options.imaginary      = options.real;
                return options;
            case InitMode::Identity:
                if(!problemDependent)
                    return std::nullopt;
                options.real.pattern = GenerationPattern::Identity;
                options.imaginary    = options.real;
                return options;
            case InitMode::TrigSin:
            case InitMode::TrigIndSin:
                options.real.pattern = GenerationPattern::Sine;
                options.imaginary    = options.real;
                return options;
            case InitMode::TrigCos:
            case InitMode::TrigIndCos:
                options.real.pattern = GenerationPattern::Cosine;
                options.imaginary    = options.real;
                return options;
            case InitMode::TrigAbsSin:
            case InitMode::TrigIndAbsSin:
                options.real.pattern = GenerationPattern::AbsoluteSine;
                options.imaginary    = options.real;
                return options;
            case InitMode::TrigAbsCos:
            case InitMode::TrigIndAbsCos:
                options.real.pattern = GenerationPattern::AbsoluteCosine;
                options.imaginary    = options.real;
                return options;
            default:
                return std::nullopt;
            }
        }

        Layout tensorLayout(const TensorDescriptor& descriptor)
        {
            std::vector<ptrdiff_t> strides;
            strides.reserve(descriptor.strides().size());
            for(const size_t stride : descriptor.strides())
                strides.push_back(static_cast<ptrdiff_t>(stride));
            return Layout(Shape(descriptor.sizes()), std::move(strides));
        }

        Layout logicalSparseMetadataLayout(const TensorDescriptor& dense,
                                           const TensorDescriptor& metadata,
                                           size_t                  sparseAxis,
                                           size_t                  metadataAxis)
        {
            if(dense.dimensions() != metadata.dimensions())
                throw std::invalid_argument(
                    "TensileLite sparse data and metadata ranks differ.");
            if(sparseAxis >= dense.dimensions())
                throw std::out_of_range(
                    "TensileLite sparse axis exceeds the data tensor rank.");
            if(metadataAxis >= metadata.dimensions())
                throw std::out_of_range(
                    "TensileLite metadata axis exceeds the metadata tensor rank.");
            if(dense.sizes()[sparseAxis] == 0)
                throw std::invalid_argument(
                    "TensileLite sparse axis extent must be nonzero.");
            if(dense.sizes()[sparseAxis] % 4 != 0)
                throw std::invalid_argument(
                    "TensileLite sparse axis extent must be divisible by four.");

            std::vector<size_t> logicalDimensions = dense.sizes();
            logicalDimensions[sparseAxis]
                = (dense.sizes()[sparseAxis] / 4 + 1) / 2;
            std::vector<ptrdiff_t> logicalStrides(dense.dimensions());
            logicalStrides[sparseAxis]
                = static_cast<ptrdiff_t>(metadata.strides()[metadataAxis]);
            if(metadata.sizes()[metadataAxis]
               != logicalDimensions[sparseAxis])
                throw std::invalid_argument(
                    "TensileLite sparse metadata axis extent mismatch.");

            size_t metadataDimension = 0;
            for(size_t denseDimension = 0;
                denseDimension < dense.dimensions();
                ++denseDimension)
            {
                if(denseDimension == sparseAxis)
                    continue;
                while(metadataDimension == metadataAxis)
                    ++metadataDimension;
                if(metadataDimension >= metadata.dimensions()
                   || metadata.sizes()[metadataDimension]
                          != logicalDimensions[denseDimension])
                    throw std::invalid_argument(
                        "TensileLite sparse metadata non-axis extent mismatch.");
                logicalStrides[denseDimension] = static_cast<ptrdiff_t>(
                    metadata.strides()[metadataDimension]);
                ++metadataDimension;
            }
            return Layout(Shape(std::move(logicalDimensions)),
                          std::move(logicalStrides));
        }

        StructuredSparsityPattern sparsePattern(PruneSparseMode mode,
                                                size_t          sparseAxis)
        {
            StructuredSparsityPattern pattern;
            pattern.axis = sparseAxis;
            switch(mode)
            {
            case PruneSparseMode::PruneRandom:
                pattern.selection = StructuredSparsitySelection::Random;
                pattern.seed      = 0x54454e53494c454cULL;
                pattern.stream    = 1;
                break;
            case PruneSparseMode::PruneXX00:
                pattern.fixedPositions = {0, 1};
                break;
            case PruneSparseMode::PruneX0X0:
                pattern.fixedPositions = {0, 2};
                break;
            case PruneSparseMode::Prune0XX0:
                pattern.fixedPositions = {1, 2};
                break;
            case PruneSparseMode::PruneX00X:
                pattern.fixedPositions = {0, 3};
                break;
            case PruneSparseMode::Prune0X0X:
                pattern.fixedPositions = {1, 3};
                break;
            case PruneSparseMode::Prune00XX:
                pattern.fixedPositions = {2, 3};
                break;
            default:
                throw std::invalid_argument(
                    "Unsupported TensileLite sparse pruning mode.");
            }
            return pattern;
        }

        bool generate(rocisa::DataType      dataType,
                      InitMode              mode,
                      Layout                layout,
                      std::span<std::byte>  storage,
                      bool                  problemDependent,
                      DataInitializationKey key,
                      std::optional<double> freeValue = std::nullopt)
        {
            const std::optional<ScalarType>        type = generationScalarType(dataType);
            const std::optional<GenerationOptions> options
                = generationOptions(dataType, mode, problemDependent, key.seed, freeValue);
            if(!type || !options)
                return false;

            GenerationOptions adjusted = *options;
            adjusted.real.stream += 2 * key.semanticStream;
            adjusted.imaginary.stream += 2 * key.semanticStream;
            Tensor generated(*type, std::move(layout), storage);
            roc::host_validation::generate(generated, adjusted);
            std::ranges::copy(generated.storage(), storage.begin());
            return true;
        }
    } // namespace

    bool tryHostValidationInitialize(rocisa::DataType      dataType,
                                     InitMode              mode,
                                     void*                 array,
                                     size_t                elements,
                                     DataInitializationKey key)
    {
        const std::optional<ScalarType> type = generationScalarType(dataType);
        if(!type)
            return false;
        const size_t bytes
            = (elements * roc::host_validation::scalarTypeInfo(*type).storageBits + 7) / 8;
        if(array == nullptr && bytes != 0)
            throw std::invalid_argument("Null TensileLite host initialization buffer.");
        return generate(dataType,
                        mode,
                        Layout::contiguous(Shape{elements}),
                        {static_cast<std::byte*>(array), bytes},
                        false,
                        key);
    }

    bool tryHostValidationInitialize(rocisa::DataType      dataType,
                                     InitMode              mode,
                                     void*                 array,
                                     size_t                elements,
                                     DataInitializationKey key,
                                     double                freeValue)
    {
        const std::optional<ScalarType> type = generationScalarType(dataType);
        if(!type)
            return false;
        const size_t bytes
            = (elements * roc::host_validation::scalarTypeInfo(*type).storageBits + 7) / 8;
        if(array == nullptr && bytes != 0)
            throw std::invalid_argument("Null TensileLite host initialization buffer.");
        return generate(dataType,
                        mode,
                        Layout::contiguous(Shape{elements}),
                        {static_cast<std::byte*>(array), bytes},
                        false,
                        key,
                        freeValue);
    }

    bool tryHostValidationInitialize(rocisa::DataType        dataType,
                                     InitMode                mode,
                                     void*                   array,
                                     TensorDescriptor const& descriptor,
                                     DataInitializationKey   key)
    {
        const std::optional<ScalarType> type = generationScalarType(dataType);
        if(!type)
            return false;

        std::vector<ptrdiff_t> strides;
        strides.reserve(descriptor.strides().size());
        for(const size_t stride : descriptor.strides())
            strides.push_back(static_cast<ptrdiff_t>(stride));

        Layout layout(Shape(descriptor.sizes()), std::move(strides));
        const size_t bytes = roc::host_validation::storageBytesForLayout(*type, layout);
        if(array == nullptr && bytes != 0)
            throw std::invalid_argument("Null TensileLite tensor initialization buffer.");
        return generate(dataType,
                        mode,
                        std::move(layout),
                        {static_cast<std::byte*>(array), bytes},
                        true,
                        key);
    }

    double hostValidationDoubleValue(InitMode mode, DataInitializationKey key, double freeValue)
    {
        double value = 0;
        if(!tryHostValidationInitialize(rocisa::DataType::Double, mode, &value, 1, key, freeValue))
            throw std::invalid_argument(
                "TensileLite constant initialization mode is unsupported.");
        return value;
    }

    double hostValidationUniformDouble(double lower, double upper, DataInitializationKey key)
    {
        if(lower > upper)
            throw std::invalid_argument(
                "TensileLite uniform lower bound exceeds upper bound.");

        GenerationOptions options;
        options.seed            = key.seed;
        options.real.pattern    = GenerationPattern::UniformReal;
        options.real.parameter0 = lower;
        options.real.parameter1 = upper;
        options.real.stream     = 2 * key.semanticStream;

        Tensor generated(ScalarType::Float64, Shape{1});
        roc::host_validation::generate(generated, options);
        return generated.loadAs<double>({0});
    }

    void initCPUSparseInput(PruneSparseMode         mode,
                            void*                   dstPruned,
                            void*                   dstCompressed,
                            void*                   dstMeta,
                            TensorDescriptor const& tensor,
                            TensorDescriptor const& tensorC,
                            TensorDescriptor const& tensorMeta,
                            size_t                  dim,
                            bool                    metadataLayout)
    {
        const ScalarType scalarType = toHostValidationScalarType(tensor.dataType());
        if(tensorC.dataType() != tensor.dataType())
            throw std::invalid_argument(
                "TensileLite sparse data and compressed tensor types differ.");
        if(tensorMeta.dataType() != rocisa::DataType::Int8)
            throw std::invalid_argument(
                "TensileLite sparse metadata must use byte storage.");
        if(dstPruned == nullptr && tensor.totalAllocatedBytes() != 0)
            throw std::invalid_argument(
                "Null TensileLite sparse input buffer.");
        if(dstCompressed == nullptr && tensorC.totalAllocatedBytes() != 0)
            throw std::invalid_argument(
                "Null TensileLite compressed sparse buffer.");
        if(dstMeta == nullptr && tensorMeta.totalAllocatedBytes() != 0)
            throw std::invalid_argument(
                "Null TensileLite sparse metadata buffer.");

        if(tensorC.totalAllocatedElements() > tensorC.totalLogicalElements())
            std::memset(dstCompressed, 0, tensorC.totalAllocatedBytes());
        if(tensorMeta.totalAllocatedElements() > tensorMeta.totalLogicalElements())
            std::memset(dstMeta, 0, tensorMeta.totalAllocatedBytes());

        std::span<std::byte> prunedStorage(
            static_cast<std::byte*>(dstPruned),
            tensor.totalAllocatedBytes());
        std::span<std::byte> compressedStorage(
            static_cast<std::byte*>(dstCompressed),
            tensorC.totalAllocatedBytes());
        std::span<std::byte> metadataStorage(
            static_cast<std::byte*>(dstMeta),
            tensorMeta.totalAllocatedBytes());
        const Layout denseLayout      = tensorLayout(tensor);
        const Layout compressedLayout = tensorLayout(tensorC);
        const Layout metadataTensorLayout = logicalSparseMetadataLayout(
            tensor,
            tensorMeta,
            dim,
            static_cast<size_t>(metadataLayout));
        Tensor prunedTensor(scalarType, denseLayout, prunedStorage);
        Tensor compressedTensor(scalarType, compressedLayout, compressedStorage);
        Tensor metadataTensor(ScalarType::UInt8, metadataTensorLayout, metadataStorage);
        StructuredSparsityProblem problem(
            prunedTensor, prunedTensor, compressedTensor, sparsePattern(mode, dim));
        problem.twoOfFourMetadata = metadataTensor;

        const size_t sliceCount =
            tensor.totalLogicalElements() / tensor.sizes()[dim];
        if(sliceCount == 0)
            return;
        const size_t requestedWorkers = std::max<size_t>(
            1, static_cast<size_t>(std::thread::hardware_concurrency()));
        auto hasIndependentSlices = [dim](const Layout& layout) {
            for(size_t dimension = 0;
                dimension < layout.shape().rank();
                ++dimension)
            {
                if(dimension != dim && layout.shape()[dimension] > 1
                   && layout.strides()[dimension] == 0)
                    return false;
            }
            return true;
        };
        const bool independentSlices
            = hasIndependentSlices(denseLayout)
              && hasIndependentSlices(compressedLayout)
              && hasIndependentSlices(metadataTensorLayout);
        const size_t chunkCount
            = independentSlices ? std::min(sliceCount, requestedWorkers) : 1;
#pragma omp parallel for schedule(static)
        for(ptrdiff_t chunk = 0;
            chunk < static_cast<ptrdiff_t>(chunkCount);
            ++chunk)
        {
            const size_t firstSlice =
                sliceCount * static_cast<size_t>(chunk) / chunkCount;
            const size_t endSlice =
                sliceCount * static_cast<size_t>(chunk + 1) / chunkCount;
            roc::host_validation::applyStructuredSparsity(
                problem,
                StructuredSparsitySliceRange{
                    .firstSlice = firstSlice,
                    .sliceCount = endSlice - firstSlice});
        }
        std::ranges::copy(prunedTensor.storage(), prunedStorage.begin());
        std::ranges::copy(compressedTensor.storage(), compressedStorage.begin());
        std::ranges::copy(metadataTensor.storage(), metadataStorage.begin());
    }
} // namespace TensileLite::Client
