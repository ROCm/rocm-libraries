// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The device half of a winner-cache key: a fold over every field of
/// `DeviceProperties`, so a benchmarked ranking is never served to a device it was not
/// measured on.
///
/// Field by field rather than a `memcpy` of the struct, for two reasons that are easy
/// to rediscover the hard way: `gcnArchName` is a `std::string`, so its object bytes are
/// a pointer/length/capacity triple rather than the characters anyone means to hash, and
/// the struct's padding bytes are unspecified -- hashing them would make the key depend
/// on whatever the allocator last left there.
///
/// `DeviceId` is deliberately absent. It is a per-process HIP ordinal, so it identifies
/// a slot rather than a device, and including it would key otherwise-identical runs
/// apart. This type takes a `const DeviceProperties&` and never sees a `MatchContext`,
/// so that exclusion holds by construction rather than by remembering.
///
/// Widening `DeviceProperties` does NOT extend the key on its own: a new field is hashed
/// only once `fold()` below emits it. `TestDeviceKey.cpp` pins the field set with a
/// structured binding that fails to compile when the struct grows, which is the reminder.
struct DeviceKey
{
    uint64_t hash = 0;

    DeviceKey() = default;

    explicit DeviceKey(const DeviceProperties& properties)
        : hash(fold(properties))
    {
    }

    bool operator==(const DeviceKey& other) const
    {
        return hash == other.hash;
    }

    bool operator!=(const DeviceKey& other) const
    {
        return !(*this == other);
    }

private:
    /// Emits every field into one byte stream, then folds it once. Lengths precede
    /// variable-width content so that {"gfx9", 42} and {"gfx942", 0} cannot serialize to
    /// the same bytes.
    static uint64_t fold(const DeviceProperties& properties)
    {
        std::vector<uint8_t> stream;
        stream.reserve(properties.gcnArchName.size() + sizeof(size_t) + 2 * sizeof(int));

        appendTrivial(stream, properties.gcnArchName.size());
        stream.insert(stream.end(), properties.gcnArchName.begin(), properties.gcnArchName.end());
        appendTrivial(stream, properties.warpSize);
        appendTrivial(stream, properties.multiProcessorCount);

        return hipdnn_data_sdk::utilities::fnv1aHash(stream.data(), stream.size());
    }

    template <typename T>
    static void appendTrivial(std::vector<uint8_t>& stream, const T& value)
    {
        const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
        stream.insert(stream.end(), bytes, bytes + sizeof(T));
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

namespace std
{

template <>
struct hash<hipdnn_plugin_sdk::ingestor::DeviceKey>
{
    size_t operator()(const hipdnn_plugin_sdk::ingestor::DeviceKey& key) const noexcept
    {
        return static_cast<size_t>(key.hash);
    }
};

} // namespace std

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
