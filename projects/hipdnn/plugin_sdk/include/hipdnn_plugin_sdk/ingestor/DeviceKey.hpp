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
/// measured on. Not a `memcpy` of the struct: `gcnArchName` is a `std::string`, not raw
/// bytes, and the struct has unspecified padding.
///
/// `DeviceId` is absent -- it identifies a slot, not a device.
///
/// Widening `DeviceProperties` does not extend the key on its own; a new field is
/// hashed only once `fold()` below emits it. `TestDeviceKey.cpp` pins the field set
/// with a structured binding that fails to compile when the struct grows.
struct DeviceKey
{
    DeviceKey() = default;

    /// Copies @p properties rather than referencing them: `MatchContext` holds
    /// `DeviceProperties` by reference, and a key outlives its `MatchContext`.
    explicit DeviceKey(DeviceProperties properties)
        : _properties(std::move(properties))
        , _hash(fold(_properties))
    {
    }

    uint64_t hash() const
    {
        return _hash;
    }

    const DeviceProperties& properties() const
    {
        return _properties;
    }

    bool operator==(const DeviceKey& other) const
    {
        return _hash == other._hash && _properties.gcnArchName == other._properties.gcnArchName
               && _properties.warpSize == other._properties.warpSize
               && _properties.multiProcessorCount == other._properties.multiProcessorCount;
    }

    bool operator!=(const DeviceKey& other) const
    {
        return !(*this == other);
    }

protected:
    /// Test seam: forces a hash collision so a test can prove `operator==` still
    /// rejects on the fields when the hash agrees.
    void forceHash(uint64_t hash)
    {
        _hash = hash;
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

    DeviceProperties _properties;
    uint64_t _hash = 0;
};

} // namespace hipdnn_plugin_sdk::ingestor

namespace std
{

template <>
struct hash<hipdnn_plugin_sdk::ingestor::DeviceKey>
{
    size_t operator()(const hipdnn_plugin_sdk::ingestor::DeviceKey& key) const noexcept
    {
        return static_cast<size_t>(key.hash());
    }
};

} // namespace std

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
