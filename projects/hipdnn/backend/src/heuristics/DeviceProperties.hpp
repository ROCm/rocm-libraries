// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipdnn_backend::heuristics
{

/**
 * @brief Query device properties from HIP.
 *
 * Calls hipGetDevice() and hipGetDeviceProperties() to populate
 * a DevicePropertiesT structure with current device information.
 *
 * RFC 0007 Reference: Section 6.2
 *
 * @return DevicePropertiesT populated from HIP device properties.
 * @throws HipdnnException if HIP calls fail.
 */
hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT queryDeviceProperties();

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
