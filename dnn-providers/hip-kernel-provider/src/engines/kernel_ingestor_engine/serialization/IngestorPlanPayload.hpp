// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

/// @file IngestorPlanPayload.hpp
/// The format constants of a saved ingestor plan, and the in-memory form of its contents.
///
/// A payload is a fixed header followed by a FlatBuffer (`ingestor_plan.fbs`). The header
/// holds the marker, the format version, the plan kind, the header size and a SHA-256 of
/// the FlatBuffer body. All header integers are little-endian.
namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

/// The first four bytes of every payload: ASCII `HKIP`.
inline constexpr std::array<uint8_t, 4> INGESTOR_PLAN_MARKER = {0x48, 0x4B, 0x49, 0x50};

/// The format version this provider writes. It reads the same major and any minor up to
/// this one.
inline constexpr uint16_t INGESTOR_PLAN_FORMAT_MAJOR = 1;
inline constexpr uint16_t INGESTOR_PLAN_FORMAT_MINOR = 0;

/// The header size this provider writes. A reader accepts a larger header that keeps the
/// alignment rule.
inline constexpr size_t INGESTOR_PLAN_HEADER_SIZE = 48;

/// Where the body digest sits in the header. The fields before it have a fixed layout in
/// every format version.
inline constexpr size_t INGESTOR_PLAN_DIGEST_OFFSET = 16;
inline constexpr size_t INGESTOR_PLAN_DIGEST_SIZE = 32;

/// The header size is a multiple of this value, so the FlatBuffer body keeps its alignment.
inline constexpr size_t INGESTOR_PLAN_BODY_ALIGNMENT = 16;

static_assert(INGESTOR_PLAN_DIGEST_OFFSET + INGESTOR_PLAN_DIGEST_SIZE == INGESTOR_PLAN_HEADER_SIZE,
              "The body digest must end the header.");
static_assert(INGESTOR_PLAN_HEADER_SIZE % INGESTOR_PLAN_BODY_ALIGNMENT == 0,
              "The header size must keep the body aligned.");

/// The raw SHA-256 of a payload body, as the header stores it.
using IngestorPlanDigest = std::array<uint8_t, INGESTOR_PLAN_DIGEST_SIZE>;

/// What a payload holds. A later kind gets the next value and its own root table. The value
/// 0 is never written.
enum class IngestorPlanKind : uint16_t
{
    SINGLE_KERNEL_PLAN = 1, ///< One selected kernel and one dispatch handler.
};

/// The contents of a saved single-kernel plan.
struct IngestorPlanPayload
{
    int64_t engineId = 0;
    hipdnn_plugin_sdk::ingestor::DescriptorId kernelId{};
    uint64_t workspaceBytes = 0;
    /// UIDs of the tensors the plan takes by value at execution.
    std::vector<int64_t> runtimePassByValueUids;
    /// The versioned name of the dispatch handler that launches the kernel.
    std::string dispatchSymbol;
    /// The handler's launch inputs. The handler contract named by `dispatchSymbol` defines
    /// their names and meanings.
    hipdnn_plugin_sdk::ingestor::MetadataValues launchValues;
    hipdnn_plugin_sdk::ingestor::KernelSourceKind sourceKind{};
    std::string symbol;
    /// The GPU target the code object was built for.
    std::string target;
    /// The kernel's identity digest: 64 lowercase hex characters.
    std::string sha256;
    std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> recordedSignature;
    std::vector<uint8_t> codeObject;
    /// Diagnostic only. It never gates a load.
    std::string providerVersion;
};

namespace detail
{

inline bool sameKernelSignature(const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& lhs,
                                const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& rhs)
{
    return std::equal(lhs.begin(),
                      lhs.end(),
                      rhs.begin(),
                      rhs.end(),
                      [](const hipdnn_plugin_sdk::ingestor::KernelArgument& left,
                         const hipdnn_plugin_sdk::ingestor::KernelArgument& right) {
                          return left.kind == right.kind && left.size == right.size
                                 && left.offset == right.offset && left.name == right.name;
                      });
}

} // namespace detail

inline bool operator==(const IngestorPlanPayload& lhs, const IngestorPlanPayload& rhs)
{
    return lhs.engineId == rhs.engineId && lhs.kernelId == rhs.kernelId
           && lhs.workspaceBytes == rhs.workspaceBytes
           && lhs.runtimePassByValueUids == rhs.runtimePassByValueUids
           && lhs.dispatchSymbol == rhs.dispatchSymbol && lhs.launchValues == rhs.launchValues
           && lhs.sourceKind == rhs.sourceKind && lhs.symbol == rhs.symbol
           && lhs.target == rhs.target && lhs.sha256 == rhs.sha256
           && detail::sameKernelSignature(lhs.recordedSignature, rhs.recordedSignature)
           && lhs.codeObject == rhs.codeObject && lhs.providerVersion == rhs.providerVersion;
}

inline bool operator!=(const IngestorPlanPayload& lhs, const IngestorPlanPayload& rhs)
{
    return !(lhs == rhs);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
