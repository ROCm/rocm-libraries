// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"

/// @file IngestorPlanCodec.hpp
/// Turns an `IngestorSavedPlan` into the bytes of a saved ingestor plan, and back.
///
/// The codec works for every kernel source kind. It makes no HIP call, and it does not
/// check whether this provider can launch the plan.
namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

/// The bytes of a saved ingestor plan: the header, then the FlatBuffer body.
struct EncodedIngestorPlan
{
    /// The buffer. It is allocated with `new[]`.
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::unique_ptr<uint8_t[]> bytes;
    /// The number of bytes in `bytes`.
    size_t size = 0;
};

/// Writes the header and the FlatBuffer body of `payload` into one buffer. A kind 1 payload
/// (`IngestorPlanPayload`) writes a kind 1 body, and a kind 2 payload
/// (`IngestorMultiKernelPayload`) a kind 2 body.
///
/// The function frees the code bytes of `payload` after it copies them. Move the payload in
/// to keep the number of code object copies low.
///
/// A kind 2 payload writes each code object once: entries with the same digest, source kind
/// and target share one entry, and the kernels name it.
///
/// Throws the same errors as `encodeIngestorPlan`.
EncodedIngestorPlan encodeIngestorPlanToBuffer(IngestorSavedPlan payload);

/// Returns the header and the FlatBuffer body of `payload`.
///
/// Throws `HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR` when the payload breaks a rule the caller
/// owns: a source kind that is not a `KernelSourceKind` value, an empty `providerVersion`,
/// `symbol` or `dispatchSymbol`, or a `sha256` that is not 64 lowercase hex characters. A
/// kind 2 payload also breaks a rule when it has no stage, a stage has no kernel, a kernel
/// names a code object outside `codeObjects`, a stage repeats a launch name, or the shared
/// workspace region is larger than the total workspace.
/// Throws `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE` when the code object is too large for the
/// format.
std::vector<uint8_t> encodeIngestorPlan(const IngestorSavedPlan& payload);

/// Reads the payload bytes at `data`, of either kind. The header kind selects the body. `data`
/// needs no particular alignment.
///
/// The checks run in this order, and the first failure throws:
/// 1. The header checks of `decodeIngestorPlanHeader`.
/// 2. The body size (`INVALID_VALUE`).
/// 3. The body digest in the header (`INVALID_VALUE`).
/// 4. The FlatBuffers verifier and the file identifier of the kind (`INVALID_VALUE`).
/// 5. The structure rules that the verifier does not check (`INVALID_VALUE`). For a kind 1
///    payload one rule is that the kernel SHA-256 text is valid and matches the code object.
///    For a kind 2 payload the rules are: both workspace fields are present and the shared
///    region is not above the total, at least one stage, at least one kernel in each stage,
///    launch names unique within a stage, every code object index in range, and every code
///    object's SHA-256 text valid and matching its bytes.
///
/// A payload that lists runtime pass-by-value UIDs reads like any other: the caller checks
/// the list against the handlers it restores.
///
/// The caller runs the checks that need the provider: GPU, handler, signature and module
/// load.
IngestorSavedPlan decodeIngestorSavedPlan(const uint8_t* data, size_t size);

/// Reads a kind 1 payload. The checks are those of `decodeIngestorSavedPlan`. A kind 2
/// payload throws `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE`.
IngestorPlanPayload decodeIngestorPlan(const uint8_t* data, size_t size);

/// Returns `text` for use in an error message. Text longer than 80 characters is cut, and
/// the result gives the full length.
std::string ingestorPlanMessageText(const std::string& text);

namespace detail
{

// A 16-aligned block. A std::vector of these blocks gives 16-aligned byte storage.
struct alignas(INGESTOR_PLAN_BODY_ALIGNMENT) IngestorPlanAlignedBlock
{
    std::array<uint8_t, INGESTOR_PLAN_BODY_ALIGNMENT> bytes;
};

// The raw SHA-256 of a body, as the header stores it.
IngestorPlanDigest computeIngestorPlanBodyDigest(const uint8_t* body, size_t bodySize);

// Refuses a body that is too small to be a FlatBuffer or too large for the format.
void checkIngestorPlanBodySize(size_t bodySize);

// Refuses a body whose SHA-256 differs from the header digest.
void checkIngestorPlanBodyDigest(const uint8_t* body,
                                 size_t bodySize,
                                 const IngestorPlanDigest& headerDigest);

// Returns body when its address is 16-aligned. Otherwise copies the body into storage and
// returns the copy. The result is valid while body and storage are unchanged.
const uint8_t* alignedIngestorPlanBody(const uint8_t* body,
                                       size_t bodySize,
                                       std::vector<IngestorPlanAlignedBlock>& storage);

} // namespace detail

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
