// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/BundleInputs.hpp"

#include <cstdint>
#include <exception>
#include <set>
#include <utility>
#include <vector>

#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_integration_tests::bundle
{

namespace
{

BundleInputs unverifiable(std::string reason)
{
    BundleInputs result;
    result.status = BundleInputs::Status::UNVERIFIABLE;
    result.message = std::move(reason);
    return result;
}

BundleInputs generateInputs(const IntegrationTestBundle& bundle,
                            InputFillRecipes& recipes,
                            DeviceInputFiller* deviceFiller)
{
    const auto wrapper = bundle.graphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();
    const std::set<int64_t> outputUids(bundle.outputTensorUids.begin(),
                                       bundle.outputTensorUids.end());

    InputTensorMap inputs;
    std::vector<int64_t> leafInputUids;
    bool anySubByte = false;
    for(const auto& [uid, attrs] : tensorAttrMap)
    {
        if(attrs->virtual_() || outputUids.count(uid) != 0)
        {
            continue;
        }
        inputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(*attrs);
        leafInputUids.push_back(uid);
        anySubByte = anySubByte || hipdnn_test_sdk::detail::isSubByteDataType(attrs->data_type());
    }

    // Sub-byte graphs fill twice, once unpacked and once packed, and the two sets must
    // hold the same values. rocRAND does not fill sub-byte types and its stream differs
    // from the host's, so a graph with any sub-byte tensor is filled on the host
    // throughout. That is a deliberately conservative choice, not a correctness
    // requirement: a large non-sub-byte operand would get the same device values in
    // both sets, so relaxing it would only buy back speed on the large MX shapes.
    DeviceInputFiller* const device = anySubByte ? nullptr : deviceFiller;

    BundleInputs result;
    FillResult fillResult;
    try
    {
        fillResult = hipdnn_integration_tests::fillInputs(
            wrapper.getGraph(), inputs, leafInputUids, recipes, device);
    }
    catch(const DeviceInputError& e)
    {
        result.status = BundleInputs::Status::DEVICE_FILL_FAILED;
        result.message = e.what();
        return result;
    }
    if(!fillResult.filled)
    {
        return unverifiable(fillResult.reason);
    }
    result.deviceFilled = fillResult.deviceFilled;

    if(anySubByte)
    {
        InputTensorMap packed;
        for(const int64_t uid : leafInputUids)
        {
            packed[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(
                *tensorAttrMap.at(uid), /*packSubByteElements=*/true);
        }

        auto packedFill = hipdnn_integration_tests::fillInputs(
            wrapper.getGraph(), packed, leafInputUids, recipes, device);
        if(!packedFill.filled)
        {
            return unverifiable(packedFill.reason);
        }
        result.packedInputs = std::move(packed);
    }

    result.inputs = std::move(inputs);
    return result;
}

BundleInputs loadGoldenInputs(const IntegrationTestBundle& bundle)
{
    // Tensors read from blobs are unpacked, but the engine reads sub-byte operands
    // packed, and only the generated path builds the packed set.
    const auto wrapper = bundle.graphWrapper();
    const std::set<int64_t> outputUids(bundle.outputTensorUids.begin(),
                                       bundle.outputTensorUids.end());
    for(const auto& [uid, attrs] : wrapper.getTensorMap())
    {
        if(!attrs->virtual_() && outputUids.count(uid) == 0
           && hipdnn_test_sdk::detail::isSubByteDataType(attrs->data_type()))
        {
            return unverifiable("sub-byte input " + std::to_string(uid)
                                + " has no packed copy for the engine");
        }
    }

    // Read here, not at registration, so a golden bundle holds its tensors only while
    // its own test runs. A blob that is unreadable or the wrong size fails this test
    // instead of quietly dropping it from the run.
    BundleInputs result;
    try
    {
        result.inputs = bundle.loadTensors();
    }
    catch(const std::exception& e)
    {
        result.status = BundleInputs::Status::TENSOR_LOAD_FAILED;
        result.message = e.what();
    }
    return result;
}

} // namespace

BundleInputs prepareBundleInputs(const IntegrationTestBundle& bundle,
                                 InputFillRecipes& recipes,
                                 DeviceInputFiller* deviceFiller)
{
    return bundle.blobs.has_value() ? loadGoldenInputs(bundle)
                                    : generateInputs(bundle, recipes, deviceFiller);
}

} // namespace hipdnn_integration_tests::bundle
