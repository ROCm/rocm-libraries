// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

struct hipdnnHandle;
using hipdnnHandle_t = struct hipdnnHandle*;

namespace hipdnn_backend::heuristics
{

/**
 * @brief Query device properties for an explicit device id.
 *
 * Calls hipGetDeviceProperties() for @p deviceId and populates a
 * DevicePropertiesT structure. Use this overload when the caller already
 * knows which device to query (e.g. resolved from a handle's stream).
 *
 * RFC 0007 Reference: Section 6.2
 *
 * @param deviceId HIP device id to query.
 * @return DevicePropertiesT populated from HIP device properties.
 * @throws HipdnnException if the HIP call fails.
 */
hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT queryDeviceProperties(int deviceId);

/**
 * @brief Query device properties for the device bound to a handle's stream.
 *
 * Resolves the device id from @p handle's stream via hipStreamGetDevice and
 * delegates to the explicit-id overload. This is the canonical acquisition
 * path inside the backend: device facts must follow the handle's stream, not
 * whatever device happens to be current on the calling thread.
 *
 * RFC 0007 Reference: Section 6.2
 *
 * @param handle Handle whose stream identifies the target device.
 * @return DevicePropertiesT populated from HIP device properties.
 * @throws HipdnnException if @p handle is null or if any HIP call fails.
 */
hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT
    queryDeviceProperties(hipdnnHandle_t handle);

/**
 * @brief Serialize DevicePropertiesT to FlatBuffer format.
 *
 * Builds a FlatBuffer-serialized representation of the device properties
 * using the Pack method and returns the bytes in a freshly owned vector
 * (the caller owns the storage; the internal FlatBufferBuilder is released
 * before return).
 *
 * The returned vector must remain alive while any hipdnnPluginConstData_t
 * wrapper produced by wrapSerializedDeviceProperties is in use, because the
 * wrapper aliases the vector's buffer.
 *
 * RFC 0007 Reference: Section 13.2
 *
 * @param props Device properties to serialize.
 * @return Vector owning the serialized FlatBuffer bytes.
 */
std::vector<uint8_t>
    serializeDeviceProperties(const hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT& props);

/**
 * @brief Wrap serialized device properties in hipdnnPluginConstData_t.
 *
 * Creates a hipdnnPluginConstData_t wrapper pointing to the serialized buffer.
 * The buffer must remain valid while the wrapper is in use.
 *
 * @param serializedBuffer Reference to the serialized buffer (must outlive the wrapper).
 * @return hipdnnPluginConstData_t wrapper pointing to the buffer.
 */
inline hipdnnPluginConstData_t
    wrapSerializedDeviceProperties(const std::vector<uint8_t>& serializedBuffer)
{
    hipdnnPluginConstData_t wrapper;
    wrapper.ptr = serializedBuffer.data();
    wrapper.size = serializedBuffer.size();
    return wrapper;
}

} // namespace hipdnn_backend::heuristics
