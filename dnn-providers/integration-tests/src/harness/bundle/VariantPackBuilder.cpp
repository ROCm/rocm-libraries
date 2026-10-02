// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/VariantPackBuilder.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>

namespace hipdnn_integration_tests::bundle::detail
{

namespace
{

// What TensorBase<T>::fillWithSentinelValue() writes: a quiet NaN where the type has
// one, its largest value where it does not.
template <class T>
T sentinelValue()
{
    if constexpr(std::numeric_limits<T>::has_quiet_NaN)
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
    else
    {
        return std::numeric_limits<T>::max();
    }
}

// Writes the sentinel into the tensor's device memory with a memset, instead of filling
// the host buffer and uploading it. Output buffers are write-only to the engine, so the
// upload moved a full tensor of sentinel values for nothing, and the host fill before
// it is a serial loop. A memset instruction covers elements of 1, 2 and 4 bytes; any
// other type, and any tensor that is not a plain TensorBase<T>, is left to the caller.
//
// The write may still be in flight on return; allocateSentinelOutputs() waits once for
// all of them.
template <class T>
bool tryFillSentinelOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor)
{
    if constexpr(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)
    {
        auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
        if(typed == nullptr)
        {
            return false;
        }

        const T value = sentinelValue<T>();
        auto& memory = typed->memory();

        // Marked first so deviceData() allocates without uploading the host buffer.
        memory.markDeviceModified();
        void* device = memory.deviceData();
        const auto count = typed->elementSpace();

        hipError_t status = hipSuccess;
        if constexpr(sizeof(T) == 1)
        {
            std::uint8_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            status = hipMemsetD8(device, bits, count);
        }
        else if constexpr(sizeof(T) == 2)
        {
            std::uint16_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            status = hipMemsetD16(device, bits, count);
        }
        else
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            status = hipMemsetD32(device, static_cast<int>(bits), count);
        }

        if(status != hipSuccess)
        {
            throw std::runtime_error(std::string("device sentinel fill failed: ")
                                     + hipGetErrorString(status));
        }
        return true;
    }
    else
    {
        static_cast<void>(tensor);
        return false;
    }
}

bool fillSentinelOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor,
                          hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    return std::visit(
        [&tensor]([[maybe_unused]] auto native) {
            return tryFillSentinelOnDevice<std::decay_t<decltype(native)>>(tensor);
        },
        hipdnn_test_sdk::utilities::datatypeToNativeVariant(dataType));
}

} // namespace

VariantPack buildVariantPack(
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    bool useDevice)
{
    VariantPack variantPack;
    const std::set<int64_t> outputUids(outputTensorUids.begin(), outputTensorUids.end());

    for(auto& [uid, tensor] : inputs)
    {
        if(outputUids.count(uid) != 0)
        {
            continue;
        }

        const auto attrIt = tensorAttributes.find(uid);
        const bool isRuntimePassByValue
            = attrIt != tensorAttributes.end() && attrIt->second->is_runtime_pass_by_value();
        variantPack[uid] = hipdnn_test_sdk::utilities::selectVariantPackPointer(
            *tensor, useDevice, isRuntimePassByValue);
    }

    for(auto& [uid, tensor] : outputs)
    {
        variantPack[uid] = hipdnn_test_sdk::utilities::selectVariantPackPointer(
            *tensor, useDevice, /*isRuntimePassByValue=*/false);
    }

    return variantPack;
}

OutputTensors allocateSentinelOutputs(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    bool onDevice)
{
    OutputTensors outputs;
    bool deviceFillPending = false;
    for(const int64_t uid : outputTensorUids)
    {
        const auto& attributes = *tensorAttributes.at(uid);
        outputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(attributes);

        if(onDevice && fillSentinelOnDevice(*outputs[uid], attributes.data_type()))
        {
            deviceFillPending = true;
        }
        else
        {
            outputs[uid]->fillWithSentinelValue();
        }
    }

    // One wait for every device write, so nothing that runs on another stream starts on
    // a half-written buffer.
    if(deviceFillPending)
    {
        const hipError_t status = hipDeviceSynchronize();
        if(status != hipSuccess)
        {
            throw std::runtime_error(std::string("device sentinel fill failed: ")
                                     + hipGetErrorString(status));
        }
    }
    return outputs;
}

void markOutputsModified(OutputTensors& outputs, bool device)
{
    for(auto& [uid, tensor] : outputs)
    {
        if(device)
        {
            tensor->markDeviceModified();
        }
        else
        {
            tensor->markHostModified();
        }
    }
}

} // namespace hipdnn_integration_tests::bundle::detail
