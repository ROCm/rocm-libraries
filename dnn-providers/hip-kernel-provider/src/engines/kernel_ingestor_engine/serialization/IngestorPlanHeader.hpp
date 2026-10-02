// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"

/// @file IngestorPlanHeader.hpp
/// Writes and reads the fixed header in front of a saved ingestor plan.
namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

/// The header fields of a payload that passed every header check.
struct IngestorPlanHeader
{
    uint16_t formatMajor = 0;
    uint16_t formatMinor = 0;
    IngestorPlanKind kind = IngestorPlanKind::SINGLE_KERNEL_PLAN;
    /// Where the FlatBuffer body starts.
    uint16_t headerSize = 0;
    IngestorPlanDigest bodyDigest{};
};

/// Returns the `INGESTOR_PLAN_HEADER_SIZE` header bytes for this provider's format version.
std::vector<uint8_t> encodeIngestorPlanHeader(IngestorPlanKind kind,
                                              const IngestorPlanDigest& bodyDigest);

/// Reads and checks the header at `data`. `size` is the size of the whole payload.
///
/// The checks run in this order, and the first failure throws:
/// 1. The fixed fields are present (`INVALID_VALUE`).
/// 2. The marker is `HKIP` (`INVALID_VALUE`).
/// 3. This provider reads the format version (`NOT_APPLICABLE`).
/// 4. The plan kind is known (`NOT_APPLICABLE`).
/// 5. The header size is valid for the payload (`INVALID_VALUE`).
///
/// The format version defines the layout after the fixed fields. Checks 3 and 4 therefore
/// run before check 5.
IngestorPlanHeader decodeIngestorPlanHeader(const uint8_t* data, size_t size);

namespace detail
{

// True when a reader at readerMajor.readerMinor reads a payload at major.minor.
bool isReadableIngestorPlanVersion(uint16_t major,
                                   uint16_t minor,
                                   uint16_t readerMajor,
                                   uint16_t readerMinor);

// Converts the 64 hex characters of a SHA-256 to the raw digest.
// Throws HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR when the text is not 64 hex characters.
IngestorPlanDigest ingestorPlanDigestFromHex(const std::string& hex);

// Converts a raw digest to 64 lowercase hex characters.
std::string ingestorPlanDigestToHex(const IngestorPlanDigest& digest);

} // namespace detail

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
