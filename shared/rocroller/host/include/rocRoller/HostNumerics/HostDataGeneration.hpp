// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <roc/host_validation/tensor.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/TensorDescriptor.hpp>

namespace rocRoller::HostNumerics
{
    enum class DataInitializationMode
    {
        Bounded,
        BoundedAlternatingSign,
        Unbounded,
        Identity,
        Ones,
        Zeros,
        TrigonometricFromFloat,
        NormalFromFloat,
    };

    struct DataInitialization
    {
        DataInitializationMode mode                    = DataInitializationMode::Bounded;
        double                 normalMean              = 0.0;
        double                 normalStandardDeviation = 1.0;
    };

    std::string toString(DataInitialization const& initialization);

    /**
     * Block-scaled FP8/BF8 storage always uses the OCP encodings required by
     * MX formats. Unscaled storage follows rocRoller's configured F8 mode.
     */
    enum class DataTypeInterpretation
    {
        Unscaled,
        BlockScaled,
    };

    struct BlockScaleGeneration
    {
        DataType type;
        size_t   blockedDimension;
        size_t   blockSize;
    };

    struct GeneratedTensor
    {
        roc::host_validation::Tensor                data;
        std::optional<roc::host_validation::Tensor> scales;
        // Retained only when generateHostTensor(..., includeReference=true).
        std::optional<roc::host_validation::Tensor> reference;
    };

    struct GeneratedGEMMInputs
    {
        roc::host_validation::Tensor                a;
        roc::host_validation::Tensor                b;
        roc::host_validation::Tensor                c;
        std::optional<roc::host_validation::Tensor> scaleA;
        std::optional<roc::host_validation::Tensor> scaleB;
    };

    roc::host_validation::ScalarType hostScalarType(DataType               type,
                                                    DataTypeInterpretation interpretation
                                                    = DataTypeInterpretation::Unscaled);

    roc::host_validation::Layout hostTensorLayout(TensorDescriptor const& descriptor);

    /**
     * Returns the canonical [free dimension, reduction block] scale layout.
     * Its physical order follows the data descriptor so the scale bytes can
     * be uploaded directly before any explicitly requested pre-swizzle.
     */
    roc::host_validation::Layout hostScaleLayout(TensorDescriptor const& descriptor,
                                                 size_t                  blockedDimension,
                                                 size_t                  blockSize);

    GeneratedTensor generateHostTensor(TensorDescriptor const&             descriptor,
                                       DataInitialization const&           initialization,
                                       std::optional<BlockScaleGeneration> blockScale,
                                       float                               minimum,
                                       float                               maximum,
                                       uint32_t                            seed,
                                       bool includeReference = false);

    /**
     * Generates C with seed, A with seed + 1, and B with seed + 2. These
     * offsets are part of the rocRoller client and test reproducibility contract.
     */
    GeneratedGEMMInputs generateGEMMInputs(TensorDescriptor const&   descriptorA,
                                           TensorDescriptor const&   descriptorB,
                                           TensorDescriptor const&   descriptorC,
                                           DataInitialization const& initializationA,
                                           DataInitialization const& initializationB,
                                           DataInitialization const& initializationC,
                                           DataType                  scaleTypeA,
                                           DataType                  scaleTypeB,
                                           size_t                    scaleBlockSize,
                                           float                     minimum,
                                           float                     maximum,
                                           uint32_t                  seed);

    template <typename T>
    std::vector<T> copyTensorStorage(roc::host_validation::Tensor const& tensor)
    {
        static_assert(std::is_trivially_copyable_v<T>);

        auto const storage = tensor.storage();
        if(storage.size() > std::numeric_limits<size_t>::max() - (sizeof(T) - 1))
            throw std::overflow_error("Host tensor storage container count overflow.");

        // host-numerics owns the exact packed byte count. rocRoller upload
        // containers such as FP4x8 and FP6x16 require a complete final
        // container, so preserve every numerical byte and zero only its tail.
        auto const     containerCount = (storage.size() + sizeof(T) - 1) / sizeof(T);
        std::vector<T> result(containerCount);
        if(!result.empty())
            std::memset(result.data(), 0, result.size() * sizeof(T));
        if(!storage.empty())
            std::memcpy(result.data(), storage.data(), storage.size());
        return result;
    }

    template <typename T>
    roc::host_validation::TensorView hostTensorView(TensorDescriptor const& descriptor,
                                                    std::span<const T>      values,
                                                    DataTypeInterpretation  interpretation
                                                    = DataTypeInterpretation::Unscaled)
    {
        static_assert(std::is_trivially_copyable_v<T>);

        auto const type          = hostScalarType(descriptor.dataType(), interpretation);
        auto const layout        = hostTensorLayout(descriptor);
        auto const requiredBytes = roc::host_validation::storageBytesForLayout(type, layout);
        if(requiredBytes > std::numeric_limits<size_t>::max() - (sizeof(T) - 1))
            throw std::overflow_error("rocRoller host tensor storage size overflow.");

        // Packed rocRoller upload containers may include only a zero-filled
        // tail. No other undersized or oversized backing storage is accepted.
        auto const expectedBytes = ((requiredBytes + sizeof(T) - 1) / sizeof(T)) * sizeof(T);
        if(values.size_bytes() != expectedBytes)
            throw std::invalid_argument(
                "rocRoller host tensor storage does not match its descriptor and packing.");

        return roc::host_validation::TensorView(type, layout, std::as_bytes(values));
    }

    template <typename T, typename Allocator>
    roc::host_validation::TensorView hostTensorView(TensorDescriptor const&          descriptor,
                                                    std::vector<T, Allocator> const& values,
                                                    DataTypeInterpretation           interpretation
                                                    = DataTypeInterpretation::Unscaled)
    {
        return hostTensorView(descriptor, std::span<const T>(values), interpretation);
    }
}
