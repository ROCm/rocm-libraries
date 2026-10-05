// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The launch inputs of one prepared dispatch, in a form that can be stored.
struct SavedLaunchInputs
{
    /// The versioned name the dispatch handler is registered under. The version names
    /// the contract that gives `values` their meaning.
    std::string dispatchSymbol;
    /// The named, typed inputs the launch is computed from. Never computed grid or
    /// argument values.
    MetadataValues values;
};

/// A stored kernel code object and the facts that identify it. Owns its bytes, so a
/// restored plan can load them again for another device.
struct SavedKernelCode
{
    KernelSourceKind sourceKind = KernelSourceKind::EMBEDDED_SOURCE;
    std::string symbol;
    /// The GPU target the code object was built for.
    std::string target;
    /// Lowercase hex SHA-256 of `codeObject`.
    std::string sha256;
    /// The argument list recorded for `symbol` when the kernel was packaged.
    std::vector<KernelArgument> recordedSignature;
    std::vector<uint8_t> codeObject;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
