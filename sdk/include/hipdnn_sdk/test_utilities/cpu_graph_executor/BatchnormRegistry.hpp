// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistryKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GenericBatchnormExecutor.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

constexpr std::array<BatchnormSignatureRegistryKey, 2> ALL_BATCHNORM_SIGNATURES
    = {{BatchnormSignatureRegistryKey{hipdnn_sdk::data_objects::DataType::FLOAT,
                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                      hipdnn_sdk::data_objects::DataType::FLOAT},
        BatchnormSignatureRegistryKey{hipdnn_sdk::data_objects::DataType::HALF,
                                      hipdnn_sdk::data_objects::DataType::HALF,
                                      hipdnn_sdk::data_objects::DataType::HALF}}};

inline std::unordered_map<BatchnormSignatureRegistryKey,
                          std::unique_ptr<IGenericBatchnormExecutor>,
                          BatchnormSignatureRegistryKeyHash>&
    batchnormRegistry()
{
    static std::unordered_map<BatchnormSignatureRegistryKey,
                              std::unique_ptr<IGenericBatchnormExecutor>,
                              BatchnormSignatureRegistryKeyHash>
        registry;
    return registry; //
}

/**
 * @brief Registers all batchnorm executors at compile-time using fold expressions
 * 
 * This variadic template function expands a parameter pack of indices to register
 * multiple BatchnormExecutor instances in the batchnormRegistry. Each executor
 * is instantiated with its corresponding signature from ALL_BATCHNORM_SIGNATURES array.
 * 
 * @tparam Is Variadic template parameter pack of indices
 * @param std::index_sequence<Is...> Compile-time sequence of indices used to
 *        iterate through ALL_BATCHNORM_SIGNATURES array
 * 
 * @note Uses C++17 fold expression with comma operator to expand and execute
 *       the registration for each index in the sequence
 */
template <std::size_t... Is>
void registerBatchnormExecutors(std::index_sequence<Is...>)
{
    ((batchnormRegistry()[ALL_BATCHNORM_SIGNATURES[Is]]
      = std::make_unique<BatchnormExecutor<ALL_BATCHNORM_SIGNATURES[Is].inputDataType,
                                           ALL_BATCHNORM_SIGNATURES[Is].scaleBiasDataType,
                                           ALL_BATCHNORM_SIGNATURES[Is].meanVarianceDataType>>()),
     ...);
}

inline void initializeBatchnormRegistry()
{
    registerBatchnormExecutors(std::make_index_sequence<ALL_BATCHNORM_SIGNATURES.size()>{});
}

struct BatchnormRegistryInitializer
{
    BatchnormRegistryInitializer()
    {
        initializeBatchnormRegistry();
    }
};

inline BatchnormRegistryInitializer _batchnormRegistryInitializer;

}
}
