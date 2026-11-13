// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/VariantPackBuilder.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string>

#include "harness/DeviceCallCheck.hpp"
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

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

        checkDeviceCall<DeviceOutputError>(status, "device sentinel fill");
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
bool fillSentinelOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor)
{
    using namespace hipdnn_data_sdk::types;
    return tryFillSentinelOnDevice<float>(tensor) || tryFillSentinelOnDevice<half>(tensor)
           || tryFillSentinelOnDevice<bfloat16>(tensor)
           || tryFillSentinelOnDevice<std::int32_t>(tensor)
           || tryFillSentinelOnDevice<std::uint8_t>(tensor)
           || tryFillSentinelOnDevice<std::int8_t>(tensor) || tryFillSentinelOnDevice<bool>(tensor)
           || tryFillSentinelOnDevice<fp8_e4m3>(tensor) || tryFillSentinelOnDevice<fp8_e5m2>(tensor)
           || tryFillSentinelOnDevice<fp8_e8m0>(tensor) || tryFillSentinelOnDevice<fp4_e2m1>(tensor)
           || tryFillSentinelOnDevice<fp6_e2m3>(tensor)
           || tryFillSentinelOnDevice<fp6_e3m2>(tensor);
}

} // namespace

VariantPack buildVariantPack(
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    bool useDevice,
    const std::vector<TensorAlias>& aliases)
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

    for(const auto& alias : aliases)
    {
        const std::string what = "buildVariantPack: alias view " + std::to_string(alias.uid)
                                 + " of owner " + std::to_string(alias.ofUid);
        const auto owner = variantPack.find(alias.ofUid);
        if(owner == variantPack.end())
        {
            throw std::invalid_argument(what + ": the owner has no input buffer");
        }
        if(variantPack.count(alias.uid) != 0 || outputs.count(alias.uid) != 0)
        {
            throw std::invalid_argument(what + ": the view already has a buffer of its own");
        }
        void* const ownerPointer = owner->second;
        variantPack[alias.uid] = static_cast<std::byte*>(ownerPointer) + alias.byteOffset;
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
    const std::unordered_map<int64_t, int64_t>& lengthUids,
    bool onDevice)
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
            const auto lengths
                = hipdnn_test_sdk::utilities::findRaggedLengths(lengthUids, loadedTensors, uid);
            outputs[uid] = hipdnn_test_sdk::detail::createRaggedTensorFromAttributeAndOffset(
                attributes, offsetIt->second, lengths);
        }
        else
        {
            outputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(attributes);
        }

        if(onDevice && fillSentinelOnDevice(*outputs[uid]))
        {
            deviceFillPending = true;
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
        checkDeviceCall<DeviceOutputError>(hipDeviceSynchronize(), "device sentinel fill");
    }
    return outputs;
}

StartOffsetStaging::StartOffsetStaging(VariantPack& variantPack,
                                       TensorMap& inputs,
                                       OutputTensors& outputs,
                                       const std::vector<TensorStartOffset>& startOffsets,
                                       const std::vector<TensorAlias>& aliases,
                                       bool useDevice)
    : _useDevice(useDevice)
{
    if(startOffsets.empty())
    {
        return;
    }

    std::set<int64_t> moved;
    for(const auto& entry : startOffsets)
    {
        const std::string what = bundle_metadata_detail::describeStartOffset(entry.uid);
        if(!moved.insert(entry.uid).second)
        {
            throw std::invalid_argument(what + ": the tensor appears in more than one entry");
        }
        hipdnn_data_sdk::utilities::ITensor* tensor = nullptr;
        const auto output = outputs.find(entry.uid);
        const bool isOutput = output != outputs.end();
        if(isOutput)
        {
            tensor = output->second.get();
        }
        else if(const auto input = inputs.find(entry.uid); input != inputs.end())
        {
            tensor = input->second.get();
        }
        if(tensor == nullptr)
        {
            throw std::invalid_argument(what + ": the tensor has no buffer of its own");
        }
        const auto pointer = variantPack.find(entry.uid);
        if(pointer == variantPack.end())
        {
            throw std::invalid_argument(what + ": the tensor has no variant pack entry");
        }

        const std::size_t bytes = tensor->elementSpace() * tensor->elementSize();
        void* const staged = stage(pointer->second, bytes, entry.startOffset);
        if(isOutput)
        {
            _outputs.push_back({pointer->second, staged, bytes});
        }
        pointer->second = staged;
    }

    for(const auto& alias : aliases)
    {
        if(moved.count(alias.ofUid) != 0)
        {
            variantPack[alias.uid]
                = static_cast<std::byte*>(variantPack.at(alias.ofUid)) + alias.byteOffset;
        }
    }

    if(_useDevice)
    {
        checkDeviceCall<DeviceOutputError>(hipDeviceSynchronize(),
                                           "wait for the start-offset copies");
    }
}

void* StartOffsetStaging::stage(void* source, std::size_t bytes, int64_t startOffset)
{
    const auto offset = static_cast<std::size_t>(startOffset);
    if(_useDevice)
    {
        try
        {
            _deviceCopies.emplace_back(bytes + offset);
        }
        catch(const std::bad_alloc&)
        {
            throw DeviceOutputError("device allocation of " + std::to_string(bytes + offset)
                                    + " bytes for a start-offset copy failed");
        }
        void* const staged = static_cast<std::byte*>(_deviceCopies.back().get()) + offset;
        if(bytes != 0)
        {
            checkDeviceCall<DeviceOutputError>(
                hipMemcpy(staged, source, bytes, hipMemcpyDeviceToDevice),
                "device copy into a start-offset buffer");
        }
        return staged;
    }

    const auto alignment = static_cast<std::size_t>(K_START_OFFSET_BASE_ALIGNMENT);
    auto& buffer = _hostCopies.emplace_back(bytes + offset + alignment - 1);
    void* base = buffer.data();
    std::size_t space = buffer.size();
    if(std::align(alignment, bytes + offset, base, space) == nullptr)
    {
        throw std::invalid_argument("no aligned address in a host start-offset buffer");
    }
    void* const staged = static_cast<std::byte*>(base) + offset;
    if(bytes != 0)
    {
        std::memcpy(staged, source, bytes);
    }
    return staged;
}

hipError_t StartOffsetStaging::waitForEngine() const
{
    if(!_useDevice || _deviceCopies.empty())
    {
        return hipSuccess;
    }
    return hipDeviceSynchronize();
}

void StartOffsetStaging::copyOutputsBack()
{
    if(_outputs.empty())
    {
        return;
    }
    if(!_useDevice)
    {
        for(const auto& output : _outputs)
        {
            if(output.bytes != 0)
            {
                std::memcpy(output.tensorData, output.stagedData, output.bytes);
            }
        }
        return;
    }

    for(const auto& output : _outputs)
    {
        if(output.bytes != 0)
        {
            checkDeviceCall<DeviceOutputError>(
                hipMemcpy(
                    output.tensorData, output.stagedData, output.bytes, hipMemcpyDeviceToDevice),
                "device copy out of a start-offset buffer");
        }
    }
    checkDeviceCall<DeviceOutputError>(hipDeviceSynchronize(),
                                       "wait for the start-offset copy-back");
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
