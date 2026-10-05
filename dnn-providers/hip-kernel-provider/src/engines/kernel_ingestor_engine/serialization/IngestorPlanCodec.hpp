// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"

/// @file IngestorPlanCodec.hpp
/// Turns an `IngestorPlanPayload` into the bytes of a saved ingestor plan, and back.
///
/// The codec works for every kernel source kind. It makes no HIP call, and it does not
/// check whether this provider can launch the plan.
namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

/// Returns the header and the FlatBuffer body of `payload`.
///
/// Throws `HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR` when the payload breaks a rule the caller
/// owns: a source kind that is not a `KernelSourceKind` value, an empty `engineName`,
/// `providerVersion`, `symbol` or `dispatchSymbol`, or a `sha256` that is not 64 lowercase
/// hex characters. Throws `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE` when the code object is too
/// large for the format.
std::vector<uint8_t> encodeIngestorPlan(const IngestorPlanPayload& payload);

/// Reads the payload bytes at `data`. `data` needs no particular alignment.
///
/// The checks run in this order, and the first failure throws:
/// 1. The header checks of `decodeIngestorPlanHeader`.
/// 2. The body size (`INVALID_VALUE`).
/// 3. The body digest in the header (`INVALID_VALUE`).
/// 4. The FlatBuffers verifier and the file identifier (`INVALID_VALUE`).
/// 5. The structure rules that the verifier does not check (`INVALID_VALUE`).
/// 6. The runtime pass-by-value tensor list is empty (`NOT_APPLICABLE`).
///
/// The caller runs the checks that need the provider: identity, GPU, handler, signature and
/// module load.
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
