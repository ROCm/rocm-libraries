// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <type_traits>
#include <vector>

#include <roc/host_validation/tensor.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include "client/GEMMParameters.hpp"

namespace rocRoller::Client::GEMMClient
{
    struct GeneratedGEMMInputs
    {
        roc::host_validation::Tensor                a;
        roc::host_validation::Tensor                b;
        roc::host_validation::Tensor                c;
        std::optional<roc::host_validation::Tensor> scaleA;
        std::optional<roc::host_validation::Tensor> scaleB;
    };

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
        // host-numerics owns the exact packed byte count. rocRoller upload
        // containers such as FP4x8 and FP6x16 require a complete final
        // container, so preserve every numerical byte and zero only its tail.
        auto const containerCount
            = storage.size() / sizeof(T) + static_cast<size_t>(storage.size() % sizeof(T) != 0);
        std::vector<T> result(containerCount);
        if(!result.empty())
            std::memset(result.data(), 0, result.size() * sizeof(T));
        if(!storage.empty())
            std::memcpy(result.data(), storage.data(), storage.size());
        return result;
    }
}
