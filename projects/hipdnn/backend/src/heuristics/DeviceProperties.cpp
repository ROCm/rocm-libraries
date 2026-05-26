// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "DeviceProperties.hpp"

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>

#include <string>

#include "HipdnnException.hpp"

namespace hipdnn_backend::heuristics
{

hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT queryDeviceProperties(int deviceId)
{
    hipDeviceProp_t hipProps;
    auto status = hipGetDeviceProperties(&hipProps, deviceId);
    if(status != hipSuccess)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Failed to get properties for device " + std::to_string(deviceId)
                                  + ": " + hipGetErrorString(status));
    }

    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT devProps;
    devProps.device_id = deviceId;
    devProps.multi_processor_count = hipProps.multiProcessorCount;
    devProps.total_global_mem = hipProps.totalGlobalMem;
    devProps.architecture_name = hipProps.gcnArchName;

    return devProps;
}

hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT queryDeviceProperties()
{
    int currentDevice = 0;
    auto status = hipGetDevice(&currentDevice);
    if(status != hipSuccess)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              std::string{"Failed to get current device: "}
                                  + hipGetErrorString(status));
    }
    return queryDeviceProperties(currentDevice);
}

std::vector<uint8_t>
    serializeDeviceProperties(const hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT& props)
{
    flatbuffers::FlatBufferBuilder builder(256);
    auto offset = hipdnn_flatbuffers_sdk::data_objects::DeviceProperties::Pack(builder, &props);
    builder.Finish(offset, "HDDP");
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

} // namespace hipdnn_backend::heuristics
