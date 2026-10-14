// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

/**
 * @file ConvFwdLaunchValues.hpp
 * @brief The launch state of the conv-forward ingestor pack, and its saved form: a
 *        versioned dispatch name and the named, typed values the launch is computed from.
 *        The pack also reads its saved form back into its launch inputs.
 *
 * The version in a dispatch name identifies the contract of its values. A change to the
 * names, types or meanings of the values needs a new version.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The tensor uids a matched conv graph binds, in kernel argument order.
struct ConvFwdBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
};

/// The seven extents the conv kernel takes as arguments, in argument order.
struct ConvFwdExtents
{
    int n = 0;
    int c = 0;
    int h = 0;
    int width = 0;
    int k = 0;
    int r = 0;
    int s = 0;
};

inline constexpr std::string_view CONV_FWD_DISPATCH_SYMBOL_V1 = "hipkernel.conv_fwd.dispatch.v1";

inline constexpr std::string_view CONV_FWD_X_UID_VALUE = "x.uid";
inline constexpr std::string_view CONV_FWD_W_UID_VALUE = "w.uid";
inline constexpr std::string_view CONV_FWD_Y_UID_VALUE = "y.uid";
inline constexpr std::string_view CONV_FWD_N_VALUE = "n";
inline constexpr std::string_view CONV_FWD_C_VALUE = "c";
inline constexpr std::string_view CONV_FWD_H_VALUE = "h";
inline constexpr std::string_view CONV_FWD_WIDTH_VALUE = "width";
inline constexpr std::string_view CONV_FWD_K_VALUE = "k";
inline constexpr std::string_view CONV_FWD_R_VALUE = "r";
inline constexpr std::string_view CONV_FWD_S_VALUE = "s";
inline constexpr std::string_view CONV_FWD_BLOCK_SIZE_VALUE = "block_size";

/// The values of the `hipkernel.conv_fwd.dispatch.v1` contract.
hipdnn_plugin_sdk::ingestor::MetadataValues convFwdLaunchValues(const ConvFwdBinding& binding,
                                                                const ConvFwdExtents& extents,
                                                                int64_t blockSize);

/// Everything a conv-forward launch is computed from.
struct ConvFwdLaunchInputs
{
    ConvFwdBinding binding;
    ConvFwdExtents extents;
    int64_t blockSize = 0;
};

/// Reads the values of the `hipkernel.conv_fwd.dispatch.v1` contract. Each extent must be
/// in [1, INT_MAX], with `r <= h` and `s <= width`. `block_size` must be in [1, UINT32_MAX],
/// and the grid it gives must fit an unsigned int.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
ConvFwdLaunchInputs
    readConvFwdLaunchInputs(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
