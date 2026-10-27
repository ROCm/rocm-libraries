// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/VariantPackBuilder.hpp"

#include <algorithm>
#include <cstddef>
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

// The element types whose output sentinel can carry a payload: exactly the ones
// hipdnn_test_sdk::utilities::payloadNanSentinelBits reports a pattern for, so this cannot drift
// from the validators' side of the contract.
template <class T>
constexpr bool K_HAS_PAYLOAD_SENTINEL
    = hipdnn_test_sdk::utilities::payloadNanSentinelBits<T>().has_value();

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

std::uintptr_t placedAddress(std::uintptr_t base, std::int64_t alignment, PlacedAt at)
{
    if(alignment <= 0)
    {
        throw std::invalid_argument("placedAddress: alignment must be positive, got "
                                    + std::to_string(alignment));
    }
    if(at == PlacedAt::UNDER && alignment < 2)
    {
        throw std::invalid_argument(
            "placedAddress: an under-aligned pointer needs a declared alignment of at least 2, "
            "got "
            + std::to_string(alignment));
    }

    const auto n = static_cast<std::uintptr_t>(alignment);
    const std::uintptr_t period = 2 * n;
    const std::uintptr_t aligned = (base + period - 1) / period * period;
    return at == PlacedAt::EXACT ? aligned + n : aligned + n / 2;
}

std::uint64_t placementSlack(std::int64_t alignment)
{
    if(alignment <= 0)
    {
        throw std::invalid_argument("placementSlack: alignment must be positive, got "
                                    + std::to_string(alignment));
    }
    return 3 * static_cast<std::uint64_t>(alignment);
}

PlacementPlan planPlacement(
    const VariantPack& variantPack,
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    bool packedInputs)
{
    PlacementPlan plan;

    // Ordered by uid so the plan, and so the order shadows are built in, does not depend on
    // the hash map's iteration order.
    std::set<int64_t> uids;
    for(const auto& entry : variantPack)
    {
        uids.insert(entry.first);
    }

    for(const int64_t uid : uids)
    {
        const std::string name = "uid " + std::to_string(uid);

        const auto outputIt = outputs.find(uid);
        const bool isOutput = outputIt != outputs.end();
        hipdnn_data_sdk::utilities::ITensor* tensor = nullptr;
        if(isOutput)
        {
            tensor = outputIt->second.get();
        }
        else if(const auto inputIt = inputs.find(uid); inputIt != inputs.end())
        {
            tensor = inputIt->second.get();
        }
        if(tensor == nullptr)
        {
            throw std::invalid_argument(name + " is in the variant pack but has no tensor");
        }

        const auto attrIt = tensorAttributes.find(uid);
        if(attrIt != tensorAttributes.end() && attrIt->second->is_runtime_pass_by_value())
        {
            plan.leftInPlace.push_back(name + ": runtime pass-by-value scalar (a host pointer)");
            continue;
        }
        if(!isOutput && packedInputs)
        {
            plan.leftInPlace.push_back(name + ": input of the sub-byte packed set");
            continue;
        }
        if(attrIt == tensorAttributes.end() || attrIt->second->alignment() <= 0)
        {
            plan.leftInPlace.push_back(name + ": no declared alignment");
            continue;
        }

        plan.candidates.push_back({uid, tensor, isOutput, attrIt->second->alignment()});
    }
    return plan;
}

PlacedVariantPack::PlacedVariantPack(VariantPack& variantPack,
                                     const std::vector<PlacementCandidate>& candidates,
                                     CasePlacement placement,
                                     std::optional<int64_t> underUid)
{
    if(placement == CasePlacement::DEFAULT)
    {
        return;
    }
    if(placement == CasePlacement::UNDER)
    {
        if(!underUid.has_value())
        {
            throw PlacementError("placement \"under\" has no placement_uid");
        }
        const bool known
            = std::any_of(candidates.begin(), candidates.end(), [&](const PlacementCandidate& c) {
                  return c.uid == *underUid;
              });
        if(!known)
        {
            throw PlacementError("placement_uid " + std::to_string(*underUid)
                                 + " is not a device buffer of the engine run (a scalar passed by "
                                   "value, a sub-byte packed input, or not a tensor of the graph)");
        }
    }

    const auto check = [](hipError_t status, const std::string& what) {
        if(status != hipSuccess)
        {
            throw PlacementError(what + ": " + hipGetErrorString(status));
        }
    };

    try
    {
        for(const auto& candidate : candidates)
        {
            if(placement == CasePlacement::UNDER && candidate.uid != *underUid)
            {
                continue;
            }

            const auto bytes = candidate.tensor->elementSpace() * candidate.tensor->elementSize();
            if(bytes == 0)
            {
                continue;
            }
            const std::string name = "uid " + std::to_string(candidate.uid);

            // Fetched from the pack, not from the tensor: the pack's pointer is the one the
            // engine would have been given, and the tensor's device data is already current
            // because building the pack uploaded any host-side fill.
            void* original = variantPack.at(candidate.uid);

            Shadow shadow;
            shadow.original = original;
            shadow.bytes = bytes;
            shadow.isOutput = candidate.isOutput;
            check(hipMalloc(&shadow.base, bytes + placementSlack(candidate.alignment)),
                  "placement of " + name + ": device allocation failed");
            _shadows.push_back(shadow);

            const auto placedAt
                = placement == CasePlacement::EXACT ? PlacedAt::EXACT : PlacedAt::UNDER;
            // The offset inside the allocation, so no integer is cast back to a pointer.
            const auto baseAddress = reinterpret_cast<std::uintptr_t>(_shadows.back().base);
            _shadows.back().placed
                = static_cast<char*>(_shadows.back().base)
                  + (placedAddress(baseAddress, candidate.alignment, placedAt) - baseAddress);
            _placedUids.push_back(candidate.uid);

            // Waits for whatever filled the original buffer before it is read.
            check(hipDeviceSynchronize(), "placement of " + name + ": wait failed");
            check(hipMemcpy(_shadows.back().placed, original, bytes, hipMemcpyDeviceToDevice),
                  "placement of " + name + ": copy into the shadow failed");
            variantPack[candidate.uid] = _shadows.back().placed;
        }
        check(hipDeviceSynchronize(), "placement: wait failed");
    }
    catch(...)
    {
        // Nothing has run yet, so the original pointers go back and the shadows are freed
        // before the exception leaves the half-built object (whose destructor never runs).
        for(const auto& shadow : _shadows)
        {
            if(shadow.placed != nullptr)
            {
                for(auto& entry : variantPack)
                {
                    if(entry.second == shadow.placed)
                    {
                        entry.second = shadow.original;
                    }
                }
            }
        }
        release();
        throw;
    }
}

PlacedVariantPack::~PlacedVariantPack()
{
    release();
}

PlacedVariantPack& PlacedVariantPack::operator=(PlacedVariantPack&& other) noexcept
{
    if(this != &other)
    {
        release();
        _shadows = std::move(other._shadows);
        _placedUids = std::move(other._placedUids);
        other._shadows.clear();
        other._placedUids.clear();
    }
    return *this;
}

void PlacedVariantPack::release() noexcept
{
    for(auto& shadow : _shadows)
    {
        if(shadow.base != nullptr)
        {
            // Nothing to do about a failed free here: the allocation is the harness's own
            // and the process is about to move on to the next case.
            static_cast<void>(hipFree(shadow.base));
            shadow.base = nullptr;
        }
    }
    _shadows.clear();
    _placedUids.clear();
}

void PlacedVariantPack::copyBackOutputs()
{
    const auto check = [](hipError_t status, const char* what) {
        if(status != hipSuccess)
        {
            throw PlacementError(std::string(what) + ": " + hipGetErrorString(status));
        }
    };

    // The engine may still be running on its own stream.
    check(hipDeviceSynchronize(), "placement: wait for the engine failed");
    for(const auto& shadow : _shadows)
    {
        if(shadow.isOutput)
        {
            check(hipMemcpy(shadow.original, shadow.placed, shadow.bytes, hipMemcpyDeviceToDevice),
                  "placement: copy back of an output shadow failed");
        }
    }
    check(hipDeviceSynchronize(), "placement: wait for the copy back failed");
}

} // namespace hipdnn_integration_tests::bundle::detail
