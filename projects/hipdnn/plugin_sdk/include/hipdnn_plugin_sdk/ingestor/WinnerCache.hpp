// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// One benchmarked candidate. `packId`/`dispatchId` travel with `kernelId` as a staleness
/// cross-check: a pack can be replaced between runs, making the same id a different
/// kernel. `timeMs` is diagnostic only; times are never comparable across runs or
/// records.
struct RankedEntry
{
    DescriptorId kernelId{};
    DescriptorId packId{};
    DescriptorId dispatchId{};
    double timeMs = 0.0;
};

/// Digest of the catalog a validated record was measured over; see `catalogDigest()`.
using CatalogDigest = uint64_t;

/// The outcome of one benchmark. `entries` holds every usable candidate in benchmarked
/// order, best first. A failed candidate is never stored, neither ranked last nor listed,
/// so a known-broken kernel is never served as a fallback; a validated record carries
/// coverage in `validated`, the digest of the whole classified catalog. nullopt marks a
/// legacy or unvalidated record.
struct WinnerRecord
{
    std::vector<RankedEntry> entries;
    std::optional<CatalogDigest> validated;
};

/// The identity of one catalog kernel: the kernel plus the pack and dispatch it ships with.
struct KernelTriple
{
    DescriptorId kernelId{};
    DescriptorId packId{};
    DescriptorId dispatchId{};
};

/// FNV-1a 64 (offset 0xcbf29ce484222325, prime 0x100000001b3) over: the triple count as
/// 8 little-endian bytes, then each triple's kernelId, packId and dispatchId as their 16
/// raw UUID bytes each, triples sorted ascending by their 48 concatenated bytes. The
/// result does not depend on the order of @p triples.
inline CatalogDigest catalogDigest(std::vector<KernelTriple> triples)
{
    // std::array compares lexicographically over unsigned bytes, which is memcmp order on
    // the concatenation.
    std::sort(triples.begin(), triples.end(), [](const KernelTriple& lhs, const KernelTriple& rhs) {
        return std::tie(lhs.kernelId, lhs.packId, lhs.dispatchId)
               < std::tie(rhs.kernelId, rhs.packId, rhs.dispatchId);
    });

    CatalogDigest hash = 0xcbf29ce484222325ULL;
    const auto fold = [&hash](uint8_t byte) {
        hash ^= byte;
        hash *= 0x100000001b3ULL;
    };
    for(size_t shift = 0; shift < 64; shift += 8)
    {
        fold(static_cast<uint8_t>(static_cast<uint64_t>(triples.size()) >> shift));
    }
    for(const auto& triple : triples)
    {
        for(const DescriptorId* id : {&triple.kernelId, &triple.packId, &triple.dispatchId})
        {
            std::for_each(id->begin(), id->end(), fold);
        }
    }
    return hash;
}

inline CatalogDigest catalogDigest(const std::vector<const KernelDefinition*>& kernels)
{
    std::vector<KernelTriple> triples;
    triples.reserve(kernels.size());
    for(const auto* kernel : kernels)
    {
        triples.push_back({kernel->kernelId, kernel->packId, kernel->dispatchId});
    }
    return catalogDigest(std::move(triples));
}

/// Exactly 16 lowercase hex digits, zero padded.
inline std::string digestToHex(CatalogDigest digest)
{
    static constexpr char HEX_DIGITS[] = "0123456789abcdef";
    std::string text(16, '0');
    for(size_t i = 16; i-- > 0; digest >>= 4)
    {
        text[i] = HEX_DIGITS[digest & 0xF];
    }
    return text;
}

/// Parses what `digestToHex()` writes; nullopt unless @p text is exactly 16 hex digits
/// (either case).
inline std::optional<CatalogDigest> digestFromHex(std::string_view text)
{
    if(text.size() != 16)
    {
        return std::nullopt;
    }
    CatalogDigest digest = 0;
    for(const char c : text)
    {
        uint64_t nibble = 0;
        if(c >= '0' && c <= '9')
        {
            nibble = static_cast<uint64_t>(c - '0');
        }
        else if(c >= 'a' && c <= 'f')
        {
            nibble = static_cast<uint64_t>(c - 'a' + 10);
        }
        else if(c >= 'A' && c <= 'F')
        {
            nibble = static_cast<uint64_t>(c - 'A' + 10);
        }
        else
        {
            return std::nullopt;
        }
        digest = (digest << 4) | nibble;
    }
    return digest;
}

/// Which records a lookup accepts.
enum class RecordPolicy
{
    /// Validated and legacy records alike.
    ANY,
    /// Validated records only.
    VALIDATED_ONLY,
};

/// Graph content plus device. Knobs are absent: a knob filter narrows which candidates
/// a run considers, never what the graph computes, so runs with different filters share
/// a record, which is why the coverage gate exists.
struct WinnerKey
{
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey graph;
    DeviceKey device;

    bool operator==(const WinnerKey& other) const
    {
        return device == other.device && graph == other.graph;
    }

    bool operator!=(const WinnerKey& other) const
    {
        return !(*this == other);
    }
};

struct WinnerKeyHash
{
    size_t operator()(const WinnerKey& key) const noexcept
    {
        const size_t graphHash
            = std::hash<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey>{}(key.graph);
        const size_t deviceHash = std::hash<DeviceKey>{}(key.device);
        return graphHash
               ^ (deviceHash + 0x9e3779b97f4a7c15ULL + (graphHash << 6) + (graphHash >> 2));
    }
};

/// Why a benchmarked ranking is being written back, which is what separates the two rules
/// below: a fresh miss adopts an existing entry for the key, and a re-benchmark triggered
/// by a record that did not cover the current candidates appends a superseding line.
enum class WinnerWriteCause
{
    /// No record existed for this key when the plan was built.
    FRESH_MISS,
    /// A record existed but was not usable for the current candidate set, so it was re-measured.
    COVERAGE_REBENCHMARK,
};

/// Does @p record carry a measurement for every kernel in @p kernels? One-directional:
/// entries in @p record absent from @p kernels do not fail coverage. Production
/// decisions go through `orderIfFullyCovered` below, not this directly.
inline bool recordCovers(const WinnerRecord& record,
                         const std::vector<const KernelDefinition*>& kernels)
{
    const auto& entries = record.entries;
    return std::all_of(kernels.begin(), kernels.end(), [&entries](const KernelDefinition* kernel) {
        return std::any_of(entries.begin(), entries.end(), [kernel](const RankedEntry& entry) {
            return entry.kernelId == kernel->kernelId;
        });
    });
}

/// Reorders @p kernels into @p record's ranked order, dropping any kernel the record
/// does not carry and any entry whose `packId`/`dispatchId` no longer agree (distinct
/// from `recordCovers`'s coverage check). An empty result means the caller falls back.
inline std::vector<const KernelDefinition*>
    orderByRecord(const WinnerRecord& record, const std::vector<const KernelDefinition*>& kernels)
{
    std::vector<const KernelDefinition*> ordered;
    ordered.reserve(kernels.size());
    for(const auto& entry : record.entries)
    {
        const auto match = std::find_if(
            kernels.begin(), kernels.end(), [&entry](const KernelDefinition* kernel) {
                return kernel->kernelId == entry.kernelId && kernel->packId == entry.packId
                       && kernel->dispatchId == entry.dispatchId;
            });
        if(match != kernels.end())
        {
            ordered.push_back(*match);
        }
    }
    return ordered;
}

/// Returns @p record's order over @p kernels when the record may be served under
/// @p policy; nullopt otherwise, and the caller re-benchmarks or falls back.
///
/// A legacy record (`validated` nullopt) is served under `ANY` only, and only when it
/// both covers every kernel and orders every one of them. The two can diverge --
/// `recordCovers` matches by `kernelId` alone, `orderByRecord` requires the full
/// `(kernelId, packId, dispatchId)` triple -- so an entry covered by id but whose pack
/// has since moved declines the whole record rather than serving what still resolves.
///
/// A validated record is served under either policy when its digest equals the digest
/// of @p kernels and every entry matches a kernel by the full triple. The digest, not
/// the entries, carries coverage: a kernel that failed validation is absent from
/// `entries` yet still in the digest, so it is classified rather than unmeasured and is
/// never served. The result is `entries` order.
inline std::optional<std::vector<const KernelDefinition*>>
    orderIfFullyCovered(const WinnerRecord& record,
                        const std::vector<const KernelDefinition*>& kernels,
                        RecordPolicy policy = RecordPolicy::ANY)
{
    if(!record.validated.has_value())
    {
        if(policy == RecordPolicy::VALIDATED_ONLY || !recordCovers(record, kernels))
        {
            return std::nullopt;
        }
        auto ordered = orderByRecord(record, kernels);
        if(ordered.size() != kernels.size())
        {
            return std::nullopt;
        }
        return ordered;
    }

    if(*record.validated != catalogDigest(kernels))
    {
        return std::nullopt;
    }
    auto ordered = orderByRecord(record, kernels);
    if(ordered.size() != record.entries.size())
    {
        return std::nullopt;
    }
    return ordered;
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
