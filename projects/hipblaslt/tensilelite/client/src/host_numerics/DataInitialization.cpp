// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from TensileLite initialization descriptors to
// product-independent host-numerics generation recipes.

#include "DataInitialization.hpp"

#include <TensileLite/Client/HostNumerics/HostNumericsBridge.hpp>
#include <TensileLite/Client/HostNumerics/TensileDataGeneration.hpp>
#include <roc/host_numerics/generation.hpp>
#include <roc/host_numerics/structured_sparsity.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace TensileLite::Client
{
    std::uint64_t stableDataInitializationStream(std::string_view semanticName)
    {
        using namespace HostNumerics;

        std::uint64_t hash = dataInitializationFnvLikeOffsetBasis;
        for(const unsigned char character : semanticName)
        {
            hash ^= character;
            hash *= dataInitializationFnvLikePrime;
        }
        return hash;
    }

    namespace
    {
        using roc::host_numerics::GenerationRecipe;
        using roc::host_numerics::GenerationRecipeSettings;
        using roc::host_numerics::Layout;
        using roc::host_numerics::ScalarType;
        using roc::host_numerics::Shape;
        using roc::host_numerics::StructuredSparseTensor;
        using roc::host_numerics::StructuredSparsityPattern;
        using roc::host_numerics::StructuredSparsitySelection;
        using roc::host_numerics::StructuredSparsitySliceRange;
        using roc::host_numerics::Tensor;

        std::optional<GenerationRecipe> generationRecipe(rocisa::DataType      dataType,
                                                         InitMode              mode,
                                                         bool                  problemDependent,
                                                         std::uint64_t         seed,
                                                         std::uint64_t         sequence,
                                                         std::optional<double> freeValue
                                                         = std::nullopt)
        {
            using namespace roc::host_numerics;
            using namespace HostNumerics;

            const GenerationRecipeSettings settings = dataInitializationSettings(seed, sequence);
            auto realOnly = [settings](GenerationRecipe::Component component) {
                return GenerationRecipe::realOnly(std::move(component), settings);
            };
            auto replicated = [settings](GenerationRecipe::Component component) {
                return GenerationRecipe::replicated(std::move(component), settings);
            };
            auto cartesian = [settings](GenerationRecipe::Component component) {
                GenerationRecipe::Component imaginary = component;
                return GenerationRecipe::cartesian(
                    std::move(component), std::move(imaginary), settings);
            };

            switch(mode)
            {
            case InitMode::Zero:
                if(dataType == rocisa::DataType::E8)
                    return realOnly(GenerationRecipe::rawConstant({.bits = 0}));
                return realOnly(GenerationRecipe::zero());
            case InitMode::One:
                return realOnly(GenerationRecipe::constant({.value = 1}));
            case InitMode::Two:
                return realOnly(GenerationRecipe::constant({.value = 2}));
            case InitMode::NegOne:
                return realOnly(GenerationRecipe::constant({.value = -1}));
            case InitMode::Max:
                if(dataType == rocisa::DataType::BFloat16)
                {
                    // Tensile's BF16 conversion rounds Float32 max to +Inf.
                    return replicated(GenerationRecipe::typeInfinity());
                }
                if(dataType == rocisa::DataType::BFloat6)
                {
                    // Tensile uses 7.5 for this mode even though E3M2 has a
                    // larger finite range.
                    return replicated(GenerationRecipe::constant({.value = 7.5}));
                }
                return replicated(GenerationRecipe::typeMaximum());
            case InitMode::DenormMin:
                if(dataType == rocisa::DataType::BFloat6)
                {
                    return replicated(GenerationRecipe::constant({.value = 0.125}));
                }
                return replicated(GenerationRecipe::typeDenormalMinimum());
            case InitMode::DenormMax:
                if(dataType == rocisa::DataType::BFloat6)
                {
                    return replicated(GenerationRecipe::constant({.value = 0.875}));
                }
                return replicated(GenerationRecipe::typeDenormalMaximum());
            case InitMode::NaN:
                return replicated(GenerationRecipe::typeNaN());
            case InitMode::Inf:
                if(dataType == rocisa::DataType::Float8 || dataType == rocisa::DataType::Float8_fnuz
                   || dataType == rocisa::DataType::BFloat8_fnuz)
                    return replicated(GenerationRecipe::typeNaN());
                return replicated(GenerationRecipe::typeInfinity());
            case InitMode::BadInput:
                if(dataType == rocisa::DataType::Int8 || dataType == rocisa::DataType::Int32)
                    return replicated(GenerationRecipe::typeMaximum());
                return replicated(GenerationRecipe::typeNaN());
            case InitMode::BadOutput:
                if(dataType == rocisa::DataType::Int8 || dataType == rocisa::DataType::Int32)
                    return replicated(GenerationRecipe::typeLowest());
                if(dataType == rocisa::DataType::Float8 || dataType == rocisa::DataType::Float8_fnuz
                   || dataType == rocisa::DataType::BFloat8_fnuz || dataType == rocisa::DataType::E8
                   || dataType == rocisa::DataType::E5M3)
                    return replicated(GenerationRecipe::typeNaN());
                return replicated(GenerationRecipe::typeInfinity());
            case InitMode::Random:
                if(dataType == rocisa::DataType::E8)
                {
                    return realOnly(GenerationRecipe::randomEncodedExponent(
                        {.lowerUnbiasedExponent = -3, .upperUnbiasedExponent = 3}));
                }
                if(dataType == rocisa::DataType::E5M3)
                {
                    return cartesian(
                        GenerationRecipe::absoluteUniformInteger({.lower = -3, .upper = 3}));
                }
                if(dataType == rocisa::DataType::Float4)
                {
                    return realOnly(GenerationRecipe::uniformRawInteger({.lower = 0, .upper = 14}));
                }
                if(dataType == rocisa::DataType::Float)
                    return cartesian(
                        GenerationRecipe::uniformInteger({.lower = -100, .upper = 100}));
                if(dataType == rocisa::DataType::Double)
                    return cartesian(
                        GenerationRecipe::uniformInteger({.lower = -1000, .upper = 1000}));
                return cartesian(GenerationRecipe::uniformInteger({.lower = -3, .upper = 3}));
            case InitMode::RandomNarrow:
                if(dataType == rocisa::DataType::E8 || dataType == rocisa::DataType::E5M3
                   || dataType == rocisa::DataType::Float4)
                    return generationRecipe(
                        dataType, InitMode::Random, problemDependent, seed, sequence, freeValue);
                if(dataType == rocisa::DataType::Int8 || dataType == rocisa::DataType::Int32
                   || dataType == rocisa::DataType::Int64)
                {
                    return cartesian(GenerationRecipe::uniformInteger({.lower = -3, .upper = 3}));
                }
                return cartesian(GenerationRecipe::randomEncodedExponent(
                    {.lowerUnbiasedExponent = dataType == rocisa::DataType::Double ? -189 : -100,
                     .upperUnbiasedExponent = 0,
                     .sourceType
                     = dataType == rocisa::DataType::Float6 || dataType == rocisa::DataType::BFloat6
                           ? std::optional<ScalarType>(ScalarType::Float32)
                           : std::nullopt}));
            case InitMode::RandomNegPosLimited:
                if(dataType == rocisa::DataType::E8)
                {
                    return realOnly(
                        GenerationRecipe::uniformRawInteger({.lower = -128, .upper = 128}));
                }
                if(dataType == rocisa::DataType::E5M3)
                {
                    return cartesian(
                        GenerationRecipe::absoluteUniformInteger({.lower = -128, .upper = 128}));
                }
                if(dataType == rocisa::DataType::Int8 || dataType == rocisa::DataType::Int32
                   || dataType == rocisa::DataType::Int64)
                {
                    return cartesian(
                        GenerationRecipe::uniformInteger({.lower = -128, .upper = 128}));
                }
                return cartesian(GenerationRecipe::uniformReal({.lower = -1, .upper = 1}));
            case InitMode::UniformLowPrecision:
                if(dataType == rocisa::DataType::Float4)
                {
                    return realOnly(GenerationRecipe::uniformReal({.lower = -6.0, .upper = 6.0}));
                }
                if(dataType == rocisa::DataType::Float6 || dataType == rocisa::DataType::BFloat6)
                {
                    return cartesian(GenerationRecipe::uniformReal({.lower = -7.5, .upper = 7.5}));
                }
                return std::nullopt;
            case InitMode::Free:
                if(!freeValue)
                    return std::nullopt;
                return realOnly(GenerationRecipe::constant({.value = *freeValue}));
            case InitMode::SerialIdx:
                return replicated(GenerationRecipe::serialIndex());
            case InitMode::SerialDim0:
                if(!problemDependent)
                    return std::nullopt;
                if(dataType == rocisa::DataType::Half)
                    return realOnly(GenerationRecipe::rawSerialDimension({.dimension = 0}));
                return replicated(GenerationRecipe::serialDimension({.dimension = 0}));
            case InitMode::SerialDim1:
                if(!problemDependent)
                    return std::nullopt;
                if(dataType == rocisa::DataType::Half)
                    return realOnly(GenerationRecipe::rawSerialDimension({.dimension = 1}));
                return replicated(GenerationRecipe::serialDimension({.dimension = 1}));
            case InitMode::Identity:
                if(!problemDependent)
                    return std::nullopt;
                return replicated(GenerationRecipe::identity());
            case InitMode::TrigSin:
            case InitMode::TrigIndSin:
                return replicated(GenerationRecipe::sine());
            case InitMode::TrigCos:
            case InitMode::TrigIndCos:
                return replicated(GenerationRecipe::cosine());
            case InitMode::TrigAbsSin:
            case InitMode::TrigIndAbsSin:
                return replicated(GenerationRecipe::absoluteSine());
            case InitMode::TrigAbsCos:
            case InitMode::TrigIndAbsCos:
                return replicated(GenerationRecipe::absoluteCosine());
            default:
                return std::nullopt;
            }
        }

        Layout logicalSparseMetadataLayout(const TensorDescriptor& dense,
                                           const TensorDescriptor& metadata,
                                           size_t                  sparseAxis,
                                           size_t                  metadataAxis)
        {
            if(dense.dimensions() != metadata.dimensions())
                throw std::invalid_argument("TensileLite sparse data and metadata ranks differ.");
            if(sparseAxis >= dense.dimensions())
                throw std::out_of_range("TensileLite sparse axis exceeds the data tensor rank.");
            if(metadataAxis >= metadata.dimensions())
                throw std::out_of_range(
                    "TensileLite metadata axis exceeds the metadata tensor rank.");
            if(dense.sizes()[sparseAxis] == 0)
                throw std::invalid_argument("TensileLite sparse axis extent must be nonzero.");
            if(dense.sizes()[sparseAxis] % 4 != 0)
                throw std::invalid_argument(
                    "TensileLite sparse axis extent must be divisible by four.");

            std::vector<size_t> logicalDimensions = dense.sizes();
            logicalDimensions[sparseAxis]         = (dense.sizes()[sparseAxis] / 4 + 1) / 2;
            const Layout metadataLayout = hostNumericsLayout(metadata);
            std::vector<ptrdiff_t> logicalStrides(dense.dimensions());
            logicalStrides[sparseAxis] = metadataLayout.stride(metadataAxis);
            if(metadata.sizes()[metadataAxis] != logicalDimensions[sparseAxis])
                throw std::invalid_argument("TensileLite sparse metadata axis extent mismatch.");

            size_t metadataDimension = 0;
            for(size_t denseDimension = 0; denseDimension < dense.dimensions(); ++denseDimension)
            {
                if(denseDimension == sparseAxis)
                    continue;
                while(metadataDimension == metadataAxis)
                    ++metadataDimension;
                if(metadataDimension >= metadata.dimensions()
                   || metadata.sizes()[metadataDimension] != logicalDimensions[denseDimension])
                    throw std::invalid_argument(
                        "TensileLite sparse metadata non-axis extent mismatch.");
                logicalStrides[denseDimension] = metadataLayout.stride(metadataDimension);
                ++metadataDimension;
            }
            return Layout(Shape(std::move(logicalDimensions)), std::move(logicalStrides));
        }

        StructuredSparsityPattern sparsePattern(PruneSparseMode mode, size_t sparseAxis)
        {
            StructuredSparsityPattern pattern;
            pattern.axis = sparseAxis;
            switch(mode)
            {
            case PruneSparseMode::PruneRandom:
                pattern.selection = StructuredSparsitySelection::Random;
                pattern.seed      = HostNumerics::sparsePruningSeed;
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
                throw std::invalid_argument("Unsupported TensileLite sparse pruning mode.");
            }
            return pattern;
        }

        void initializeStorageWithHostNumerics(ScalarType            type,
                                               rocisa::DataType      dataType,
                                               InitMode              mode,
                                               Layout                layout,
                                               std::span<std::byte>  storage,
                                               bool                  problemDependent,
                                               DataInitializationKey key,
                                               std::optional<double> freeValue = std::nullopt)
        {
            const std::optional<GenerationRecipe> recipe = generationRecipe(
                dataType, mode, problemDependent, key.seed, key.semanticStream, freeValue);
            if(!recipe)
                throw std::invalid_argument(
                    "TensileLite initialization mode is not representable by host-numerics.");

            Tensor generated = Tensor::copyEncodedBackingStorage(type, std::move(layout), storage);
            roc::host_numerics::generate(generated, *recipe);
            std::ranges::copy(generated.rawEncodedBackingStorage(), storage.begin());
        }
    } // namespace

    void initializeHostBufferWithHostNumerics(rocisa::DataType      dataType,
                                              InitMode              mode,
                                              void*                 array,
                                              size_t                elements,
                                              DataInitializationKey key)
    {
        const ScalarType type   = toHostNumericsScalarType(dataType);
        Layout           layout = Layout::contiguousLastDimensionFastest(Shape{elements});
        const size_t     bytes  = roc::host_numerics::storageBytesForLayout(type, layout);
        if(array == nullptr && bytes != 0)
            throw std::invalid_argument("Null TensileLite host initialization buffer.");
        initializeStorageWithHostNumerics(type,
                                          dataType,
                                          mode,
                                          std::move(layout),
                                          {static_cast<std::byte*>(array), bytes},
                                          false,
                                          key);
    }

    void initializeHostBufferWithHostNumerics(rocisa::DataType      dataType,
                                              InitMode              mode,
                                              void*                 array,
                                              size_t                elements,
                                              DataInitializationKey key,
                                              double                freeValue)
    {
        const ScalarType type   = toHostNumericsScalarType(dataType);
        Layout           layout = Layout::contiguousLastDimensionFastest(Shape{elements});
        const size_t     bytes  = roc::host_numerics::storageBytesForLayout(type, layout);
        if(array == nullptr && bytes != 0)
            throw std::invalid_argument("Null TensileLite host initialization buffer.");
        initializeStorageWithHostNumerics(type,
                                          dataType,
                                          mode,
                                          std::move(layout),
                                          {static_cast<std::byte*>(array), bytes},
                                          false,
                                          key,
                                          freeValue);
    }

    void initializeHostBufferWithHostNumerics(rocisa::DataType        dataType,
                                              InitMode                mode,
                                              void*                   array,
                                              TensorDescriptor const& descriptor,
                                              DataInitializationKey   key)
    {
        const ScalarType type   = toHostNumericsScalarType(dataType);
        Layout           layout = hostNumericsLayout(descriptor);
        const size_t bytes = roc::host_numerics::storageBytesForLayout(type, layout);
        if(array == nullptr && bytes != 0)
            throw std::invalid_argument("Null TensileLite tensor initialization buffer.");
        initializeStorageWithHostNumerics(type,
                                          dataType,
                                          mode,
                                          std::move(layout),
                                          {static_cast<std::byte*>(array), bytes},
                                          true,
                                          key);
    }

    double hostNumericsDoubleValue(InitMode mode, DataInitializationKey key, double freeValue)
    {
        double value = 0;
        initializeHostBufferWithHostNumerics(
            rocisa::DataType::Double, mode, &value, 1, key, freeValue);
        return value;
    }

    double hostNumericsUniformDouble(double lower, double upper, DataInitializationKey key)
    {
        if(lower > upper)
            throw std::invalid_argument("TensileLite uniform lower bound exceeds upper bound.");

        const auto uniformRealRecipe = roc::host_numerics::GenerationRecipe::realOnly(
            roc::host_numerics::GenerationRecipe::uniformReal({.lower = lower, .upper = upper}),
            HostNumerics::dataInitializationSettings(key.seed, key.semanticStream));

        Tensor generated(ScalarType::Float64, Shape{1});
        roc::host_numerics::generate(generated, uniformRealRecipe);
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
        const ScalarType scalarType = toHostNumericsScalarType(tensor.dataType());
        if(tensorC.dataType() != tensor.dataType())
            throw std::invalid_argument(
                "TensileLite sparse data and compressed tensor types differ.");
        if(tensorMeta.dataType() != rocisa::DataType::Int8)
            throw std::invalid_argument("TensileLite sparse metadata must use byte storage.");
        if(dstPruned == nullptr && tensor.totalAllocatedBytes() != 0)
            throw std::invalid_argument("Null TensileLite sparse input buffer.");
        if(dstCompressed == nullptr && tensorC.totalAllocatedBytes() != 0)
            throw std::invalid_argument("Null TensileLite compressed sparse buffer.");
        if(dstMeta == nullptr && tensorMeta.totalAllocatedBytes() != 0)
            throw std::invalid_argument("Null TensileLite sparse metadata buffer.");

        std::span<std::byte> prunedStorage(static_cast<std::byte*>(dstPruned),
                                           tensor.totalAllocatedBytes());
        std::span<std::byte> compressedStorage(static_cast<std::byte*>(dstCompressed),
                                               tensorC.totalAllocatedBytes());
        std::span<std::byte> metadataStorage(static_cast<std::byte*>(dstMeta),
                                             tensorMeta.totalAllocatedBytes());
        const Layout         denseLayout          = hostNumericsLayout(tensor);
        const Layout         compressedLayout     = hostNumericsLayout(tensorC);
        const Layout         metadataTensorLayout = logicalSparseMetadataLayout(
            tensor, tensorMeta, dim, static_cast<size_t>(metadataLayout));
        Tensor prunedTensor
            = Tensor::copyEncodedBackingStorage(scalarType, denseLayout, prunedStorage);
        Tensor compressedTensor
            = Tensor::copyEncodedBackingStorage(scalarType, compressedLayout, compressedStorage);
        Tensor metadataTensor = Tensor::copyEncodedBackingStorage(
            ScalarType::UInt8, metadataTensorLayout, metadataStorage);
        if(tensorC.totalAllocatedElements() > tensorC.totalLogicalElements())
            std::ranges::fill(compressedTensor.rawEncodedBackingStorage(), std::byte{0});
        if(tensorMeta.totalAllocatedElements() > tensorMeta.totalLogicalElements())
            std::ranges::fill(metadataTensor.rawEncodedBackingStorage(), std::byte{0});
        roc::host_numerics::applyStructuredSparsityInto(
            prunedTensor,
            StructuredSparseTensor{.pruned            = prunedTensor,
                                   .compressed        = compressedTensor,
                                   .twoOfFourMetadata = metadataTensor},
            sparsePattern(mode, dim));
        std::ranges::copy(prunedTensor.rawEncodedBackingStorage(), prunedStorage.begin());
        std::ranges::copy(compressedTensor.rawEncodedBackingStorage(), compressedStorage.begin());
        std::ranges::copy(metadataTensor.rawEncodedBackingStorage(), metadataStorage.begin());
    }
} // namespace TensileLite::Client
