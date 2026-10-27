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
#include <utility>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/NonFiniteMask.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_integration_tests::bundle::detail
{

namespace
{

// The element types the non-finite mask grades, and so the ones whose output sentinel can carry
// a payload (see hipdnn_test_sdk::utilities::payloadNanSentinelBits).
template <class T>
constexpr bool K_HAS_PAYLOAD_SENTINEL = std::is_same_v<T, float> || std::is_same_v<T, double>
                                        || std::is_same_v<T, hipdnn_data_sdk::types::half>
                                        || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

// What TensorBase<T>::fillWithSentinelValue() writes: a quiet NaN where the type has
// one, its largest value where it does not. With `payloadNan`, the types that can carry
// one get the NaN with a distinctive payload instead, so a validator that matches NaN against
// NaN can still tell an element the engine never wrote.
template <class T>
T sentinelValue(bool payloadNan)
{
    if constexpr(K_HAS_PAYLOAD_SENTINEL<T>)
    {
        if(payloadNan)
        {
            return hipdnn_test_sdk::utilities::payloadNanSentinelValue<T>();
        }
    }
    else
    {
        static_cast<void>(payloadNan);
    }

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
// Dispatches on the tensor itself rather than on the attribute's data type: the tensor
// factory builds tensors (UINT8, INT8, INT64, unpacked INT4) that the data type to
// native type mapping has no entry for.
//
// Only the device-touching steps throw DeviceOutputError; anything else is a host
// fault and keeps its own message.
//
// The write may still be in flight on return; allocateSentinelOutputs() waits once for
// all of them.
template <class T>
bool tryFillSentinelOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor, bool payloadNan)
{
    if constexpr(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)
    {
        auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
        if(typed == nullptr)
        {
            return false;
        }

        const T value = sentinelValue<T>(payloadNan);
        auto& memory = typed->memory();

        // Marked first so deviceData() allocates without uploading the host buffer.
        memory.markDeviceModified();
        void* device = nullptr;
        try
        {
            device = memory.deviceData();
        }
        catch(const std::exception& e)
        {
            throw DeviceOutputError(std::string("device output allocation failed: ") + e.what());
        }
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
            throw DeviceOutputError(std::string("device sentinel fill failed: ")
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

// Every element type the tensor factory builds as a plain tensor of 1, 2 or 4 byte
// elements. Wider types (double, INT64) and packed sub-byte tensors fall through to the
// host fill.
bool fillSentinelOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor, bool payloadNan)
{
    using namespace hipdnn_data_sdk::types;
    return tryFillSentinelOnDevice<float>(tensor, payloadNan)
           || tryFillSentinelOnDevice<half>(tensor, payloadNan)
           || tryFillSentinelOnDevice<bfloat16>(tensor, payloadNan)
           || tryFillSentinelOnDevice<std::int32_t>(tensor, payloadNan)
           || tryFillSentinelOnDevice<std::uint8_t>(tensor, payloadNan)
           || tryFillSentinelOnDevice<std::int8_t>(tensor, payloadNan)
           || tryFillSentinelOnDevice<bool>(tensor, payloadNan)
           || tryFillSentinelOnDevice<fp8_e4m3>(tensor, payloadNan)
           || tryFillSentinelOnDevice<fp8_e5m2>(tensor, payloadNan)
           || tryFillSentinelOnDevice<fp8_e8m0>(tensor, payloadNan)
           || tryFillSentinelOnDevice<fp4_e2m1>(tensor, payloadNan)
           || tryFillSentinelOnDevice<fp6_e2m3>(tensor, payloadNan)
           || tryFillSentinelOnDevice<fp6_e3m2>(tensor, payloadNan);
}

// The host fill of the payload sentinel, for the types the device memset does not cover (double)
// and for every host run. A bitwise store of the type's own value, never a conversion from a
// float NaN, which could canonicalise the payload. False when the tensor is not one of the
// types that carry a payload.
template <class T>
bool tryFillPayloadSentinelOnHost(hipdnn_data_sdk::utilities::ITensor& tensor)
{
    auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
    if(typed == nullptr)
    {
        return false;
    }
    typed->fillWithValue(hipdnn_test_sdk::utilities::payloadNanSentinelValue<T>());
    return true;
}

bool fillPayloadSentinelOnHost(hipdnn_data_sdk::utilities::ITensor& tensor)
{
    using namespace hipdnn_data_sdk::types;
    return tryFillPayloadSentinelOnHost<float>(tensor) || tryFillPayloadSentinelOnHost<half>(tensor)
           || tryFillPayloadSentinelOnHost<bfloat16>(tensor)
           || tryFillPayloadSentinelOnHost<double>(tensor);
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
    const TensorMap& loadedTensors,
    bool onDevice,
    bool payloadNanSentinel)
{
    OutputTensors outputs;
    bool deviceFillPending = false;

    for(const int64_t uid : outputTensorUids)
    {
        const auto& attributes = *tensorAttributes.at(uid);
        const auto raggedOffsetUid = attributes.ragged_offset_tensor_uid();
        if(raggedOffsetUid.has_value())
        {
            const auto offsetIt = loadedTensors.find(raggedOffsetUid.value());
            if(offsetIt == loadedTensors.end())
            {
                throw std::invalid_argument(
                    "ragged output " + std::to_string(uid) + " references offset tensor "
                    + std::to_string(raggedOffsetUid.value()) + ", which is not loaded");
            }
            outputs[uid] = hipdnn_test_sdk::detail::createRaggedTensorFromAttributeAndOffset(
                attributes, offsetIt->second);
        }
        else
        {
            outputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(attributes);
        }

        if(onDevice && fillSentinelOnDevice(*outputs[uid], payloadNanSentinel))
        {
            deviceFillPending = true;
        }
        else if(payloadNanSentinel && fillPayloadSentinelOnHost(*outputs[uid]))
        {
            // filled in place
        }
        else
        {
            outputs[uid]->fillWithSentinelValue();
        }
    }

    // One wait for every device write, so nothing that runs on another stream starts
    // on a half-written buffer.
    if(deviceFillPending)
    {
        const hipError_t status = hipDeviceSynchronize();
        if(status != hipSuccess)
        {
            throw DeviceOutputError(std::string("device sentinel fill failed: ")
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
