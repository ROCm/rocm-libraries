// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

/**
 * @file PointwiseLaunchValues.hpp
 * @brief The launch state of the pointwise ingestor pack, and its saved form: a versioned
 *        dispatch name and the named, typed values the launch is computed from.
 *
 * The version in a dispatch name identifies the contract of its values. A change to the
 * names, types or meanings of the values needs a new version.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The tensor uids a matched pointwise graph binds, in argument order.
struct PointwiseBinding
{
    int64_t inputA = 0;
    int64_t inputB = 0;
    int64_t output = 0;
};

inline constexpr std::string_view POINTWISE_DISPATCH_SYMBOL_V1 = "hipkernel.pointwise.dispatch.v1";

inline constexpr std::string_view POINTWISE_INPUT_A_UID_VALUE = "input_a.uid";
inline constexpr std::string_view POINTWISE_INPUT_B_UID_VALUE = "input_b.uid";
inline constexpr std::string_view POINTWISE_OUTPUT_UID_VALUE = "output.uid";
inline constexpr std::string_view POINTWISE_BLOCK_SIZE_VALUE = "block_size";

/// The values of the `hipkernel.pointwise.dispatch.v1` contract.
hipdnn_plugin_sdk::ingestor::MetadataValues pointwiseLaunchValues(const PointwiseBinding& binding,
                                                                  int64_t blockSize);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
