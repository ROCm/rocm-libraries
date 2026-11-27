// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>
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

/// The format version this provider writes at most. It reads the same major and any minor up
/// to this one. The rule for a new format version is in ingestor_plan.fbs. The minor a payload
/// carries depends on its kind: see `ingestorPlanMinorOf`.
inline constexpr uint16_t INGESTOR_PLAN_FORMAT_MAJOR = 1;
inline constexpr uint16_t INGESTOR_PLAN_FORMAT_MINOR = 1;
/// The minor a kind 1 payload carries. Its body is a valid 1.0 body, so a 1.0 reader keeps
/// loading the single-kernel plans a newer provider saves.
inline constexpr uint16_t INGESTOR_PLAN_SINGLE_KERNEL_MINOR = 0;
/// The minor a kind 2 payload carries. A 1.0 reader refuses it.
inline constexpr uint16_t INGESTOR_PLAN_MULTI_KERNEL_MINOR = 1;

/// The SHA-256 of ingestor_plan.fbs, with LF line ends. Set it again after each schema edit.
inline constexpr std::string_view INGESTOR_PLAN_SCHEMA_SHA256
    = "1dfeba6a78103d4397fc347fd8bafdc5169ceffbd84a0053b161fe57bcbb6fba";

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

/// What a payload holds. A later kind gets the next value and its own root table, with its own
/// file identifier. The value 0 is never written.
enum class IngestorPlanKind : uint16_t
{
    SINGLE_KERNEL_PLAN = 1, ///< One selected kernel and one dispatch handler.
    MULTI_KERNEL_PLAN = 2, ///< Ordered stages, each with one or more kernels.
};

/// The format minor a payload of @p kind carries. Throws `HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR`
/// for a value that is not a kind this provider writes.
constexpr uint16_t ingestorPlanMinorOf(IngestorPlanKind kind)
{
    switch(kind)
    {
    case IngestorPlanKind::SINGLE_KERNEL_PLAN:
        return INGESTOR_PLAN_SINGLE_KERNEL_MINOR;
    case IngestorPlanKind::MULTI_KERNEL_PLAN:
        return INGESTOR_PLAN_MULTI_KERNEL_MINOR;
    default:
        break;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "no format minor for ingestor plan kind " + std::to_string(static_cast<uint16_t>(kind)));
}

/// The contents of a saved single-kernel plan.
struct IngestorPlanPayload
{
    hipdnn_plugin_sdk::ingestor::DescriptorId kernelDescriptorId{};
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
    /// The base architecture of the GPU the plan was saved on.
    std::string target;
    /// The kernel's identity digest: 64 lowercase hex characters.
    std::string sha256;
    /// The kernel's argument list.
    std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> recordedSignature;
    std::vector<uint8_t> codeObject;
    /// The version of the provider that saved the plan. Diagnostic only: it never gates a load.
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
    return lhs.kernelDescriptorId == rhs.kernelDescriptorId
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

/// One code object of a multi-kernel plan. Each distinct code object appears once.
struct IngestorCodeObjectPayload
{
    hipdnn_plugin_sdk::ingestor::KernelSourceKind sourceKind{};
    /// The base architecture of the GPU the plan was saved on.
    std::string target;
    /// Lowercase hex SHA-256 of `codeObject`.
    std::string sha256;
    std::vector<uint8_t> codeObject;
};

/// One kernel a stage launches.
struct IngestorStageKernelPayload
{
    /// Empty when the stage launches one kernel.
    std::string launchName;
    std::string symbol;
    std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> recordedSignature;
    /// The index of this kernel's code object in `IngestorMultiKernelPayload::codeObjects`.
    uint32_t codeObjectIndex = 0;
};

/// One stage of a multi-kernel plan: the launches of one dispatch handler.
struct IngestorStagePayload
{
    /// The slot the stage fills. Empty for an unslotted plan.
    std::string slot;
    hipdnn_plugin_sdk::ingestor::DescriptorId kernelDescriptorId{};
    /// The versioned name of the dispatch handler that launches the stage.
    std::string dispatchSymbol;
    /// The handler's launch inputs, as in `IngestorPlanPayload::launchValues`.
    hipdnn_plugin_sdk::ingestor::MetadataValues launchValues;
    std::vector<IngestorStageKernelPayload> kernels;
};

/// The contents of a saved multi-kernel plan.
struct IngestorMultiKernelPayload
{
    /// In slot order. An unslotted plan has one stage with an empty slot.
    std::vector<IngestorStagePayload> stages;
    /// Each code object once.
    std::vector<IngestorCodeObjectPayload> codeObjects;
    /// The aligned shared workspace region. 0 for an unslotted plan.
    uint64_t sharedWorkspaceBytes = 0;
    /// The total workspace of the plan.
    uint64_t workspaceBytes = 0;
    /// UIDs of the tensors the user supplies by value at execution, sorted, unique.
    std::vector<int64_t> runtimePassByValueUids;
    /// The version of the provider that saved the plan. Diagnostic only: it never gates a load.
    std::string providerVersion;
};

inline bool operator==(const IngestorCodeObjectPayload& lhs, const IngestorCodeObjectPayload& rhs)
{
    return lhs.sourceKind == rhs.sourceKind && lhs.target == rhs.target && lhs.sha256 == rhs.sha256
           && lhs.codeObject == rhs.codeObject;
}

inline bool operator!=(const IngestorCodeObjectPayload& lhs, const IngestorCodeObjectPayload& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const IngestorStageKernelPayload& lhs, const IngestorStageKernelPayload& rhs)
{
    return lhs.launchName == rhs.launchName && lhs.symbol == rhs.symbol
           && detail::sameKernelSignature(lhs.recordedSignature, rhs.recordedSignature)
           && lhs.codeObjectIndex == rhs.codeObjectIndex;
}

inline bool operator!=(const IngestorStageKernelPayload& lhs, const IngestorStageKernelPayload& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const IngestorStagePayload& lhs, const IngestorStagePayload& rhs)
{
    return lhs.slot == rhs.slot && lhs.kernelDescriptorId == rhs.kernelDescriptorId
           && lhs.dispatchSymbol == rhs.dispatchSymbol && lhs.launchValues == rhs.launchValues
           && lhs.kernels == rhs.kernels;
}

inline bool operator!=(const IngestorStagePayload& lhs, const IngestorStagePayload& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const IngestorMultiKernelPayload& lhs, const IngestorMultiKernelPayload& rhs)
{
    return lhs.stages == rhs.stages && lhs.codeObjects == rhs.codeObjects
           && lhs.sharedWorkspaceBytes == rhs.sharedWorkspaceBytes
           && lhs.workspaceBytes == rhs.workspaceBytes
           && lhs.runtimePassByValueUids == rhs.runtimePassByValueUids
           && lhs.providerVersion == rhs.providerVersion;
}

inline bool operator!=(const IngestorMultiKernelPayload& lhs, const IngestorMultiKernelPayload& rhs)
{
    return !(lhs == rhs);
}

/// What a saved plan holds, by kind.
using IngestorSavedPlan = std::variant<IngestorPlanPayload, IngestorMultiKernelPayload>;

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
