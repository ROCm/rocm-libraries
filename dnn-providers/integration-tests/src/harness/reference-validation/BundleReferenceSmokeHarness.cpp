// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/reference-validation/BundleReferenceSmokeHarness.hpp"

#include <cmath>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/VariantPackBuilder.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/input-init/InputFillRecipes.hpp"
#include "harness/reference-validation/BundleReferenceValidationHarness.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

template <typename T>
bool tryScan(hipdnn_data_sdk::utilities::ITensor& tensor, NonFiniteScan& scan)
{
    if(dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor) == nullptr)
    {
        return false;
    }
    scan.scanned = true;

    const auto& dims = tensor.dims();
    const auto& strides = tensor.strides();
    for(const auto extent : dims)
    {
        if(extent <= 0)
        {
            return true;
        }
    }
    // rawHostData() migrates a device-modified tensor back to the host first.
    const auto* data = static_cast<const T*>(tensor.rawHostData());

    // Odometer over the logical index space: padding between strided elements is
    // never written and still holds the NaN sentinel, so a flat walk would misreport.
    std::vector<int64_t> index(dims.size(), 0);
    while(true)
    {
        int64_t offset = 0;
        for(size_t axis = 0; axis < dims.size(); ++axis)
        {
            offset += index[axis] * strides[axis];
        }
        // double is read as itself; every narrower float type widens through float.
        double value = 0.0;
        if constexpr(std::is_same_v<T, double>)
        {
            value = data[offset];
        }
        else
        {
            value = static_cast<double>(static_cast<float>(data[offset]));
        }
        ++scan.elements;
        if(std::isnan(value) || std::isinf(value))
        {
            if(std::isnan(value))
            {
                ++scan.nans;
            }
            else
            {
                ++scan.infs;
            }
            if(!scan.firstBadIndex.has_value())
            {
                scan.firstBadIndex = index;
            }
        }

        size_t axis = dims.size();
        while(true)
        {
            if(axis == 0)
            {
                return true; // wrapped the outermost axis, or a rank-0 tensor
            }
            --axis;
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
        }
    }
}

std::string formatIndex(const std::vector<int64_t>& index)
{
    std::ostringstream out;
    out << "[";
    for(size_t i = 0; i < index.size(); ++i)
    {
        out << (i == 0 ? "" : ",") << index[i];
    }
    out << "]";
    return out.str();
}

} // namespace

NonFiniteScan scanNonFinite(hipdnn_data_sdk::utilities::ITensor& tensor)
{
    using namespace hipdnn_data_sdk::types;
    NonFiniteScan scan;
    (void)(tryScan<float>(tensor, scan) || tryScan<half>(tensor, scan)
           || tryScan<bfloat16>(tensor, scan) || tryScan<double>(tensor, scan)
           || tryScan<fp8_e4m3>(tensor, scan) || tryScan<fp8_e5m2>(tensor, scan)
           || tryScan<fp8_e4m3_fnuz>(tensor, scan) || tryScan<fp8_e5m2_fnuz>(tensor, scan));
    return scan;
}

IReferenceGraphExecutor& BundleReferenceSmokeHarness::referenceExecutor() const
{
    return _referenceExecutors->get(_referenceType);
}

bool BundleReferenceSmokeHarness::useDevice() const
{
    return _referenceType == ReferenceExecutorType::GPU
           && referenceExecutor().requiresDeviceMemory();
}

void BundleReferenceSmokeHarness::SetUp()
{
    if(_referenceType == ReferenceExecutorType::GPU)
    {
        SKIP_IF_NO_DEVICES();
    }
    ASSERT_NE(_bundle, nullptr) << "No bundle set";

    if(auto reason
       = checkVramRequirement(_bundle->metadata, TestConfig::get().getCurrentDeviceVramMb()))
    {
        GTEST_SKIP() << *reason;
    }
    if(auto reason = checkArchCompatibility(_bundle->metadata, TestConfig::get().getCurrentArch()))
    {
        GTEST_SKIP() << *reason;
    }
    if(_knownGap.has_value())
    {
        GTEST_SKIP() << "listed in knownReferenceGaps(): " << _knownGap->reason;
    }
}

void BundleReferenceSmokeHarness::TestBody()
{
    const auto label = BundleReferenceValidationHarness::referenceLabel(_referenceType);
    const auto wrapper = _bundle->graphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();

    if(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(tensorAttrMap))
    {
        GTEST_SKIP() << "ragged inputs need golden blobs; the fill recipes cannot build them";
    }

    const std::set<int64_t> outputUids(_bundle->outputTensorUids.begin(),
                                       _bundle->outputTensorUids.end());

    // Inputs exactly as the engine harness generates them for a bundle with no blobs:
    // the bundle's seed and per-tensor recipes, op defaults for the rest.
    InputFillRecipes recipes;
    if(_bundle->metadata.seed.has_value())
    {
        recipes.setGlobalSeed(static_cast<unsigned int>(*_bundle->metadata.seed));
    }
    if(_bundle->metadata.inputs.has_value())
    {
        recipes.loadFromJson(*_bundle->metadata.inputs);
    }

    InputTensorMap inputs;
    std::vector<int64_t> leafInputUids;
    for(const auto& [uid, attrs] : tensorAttrMap)
    {
        if(attrs->virtual_() || outputUids.count(uid) != 0)
        {
            continue;
        }
        if(hipdnn_test_sdk::detail::isSubByteDataType(attrs->data_type()))
        {
            GTEST_SKIP() << "sub-byte input " << uid << " is not supported by the smoke run";
        }
        inputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(*attrs);
        leafInputUids.push_back(uid);
    }

    FillResult fill;
    try
    {
        fill = fillInputs(wrapper.getGraph(), inputs, leafInputUids, recipes, nullptr);
    }
    catch(const std::exception& e)
    {
        FAIL() << "input fill threw: " << e.what() << "\n  bundle: " << _bundlePath;
    }
    ASSERT_TRUE(fill.filled) << "inputs could not be filled: " << fill.reason
                             << "\n  bundle: " << _bundlePath;

    TensorMap tensors(inputs.begin(), inputs.end());
    OutputTensors outputs;
    std::unordered_map<int64_t, void*> variantPack;
    try
    {
        outputs = detail::allocateSentinelOutputs(
            tensorAttrMap, _bundle->outputTensorUids, tensors, useDevice());
        variantPack = detail::buildVariantPack(
            tensors, outputs, tensorAttrMap, _bundle->outputTensorUids, useDevice());
    }
    catch(const std::exception& e)
    {
        FAIL() << label << " was not run: could not prepare its buffers: " << e.what()
               << "\n  bundle: " << _bundlePath;
    }

    IReferenceGraphExecutor& executor = referenceExecutor();
    try
    {
        ASSERT_TRUE(executor.isApplicable(_bundle->graphBuffer.data(), _bundle->graphBuffer.size()))
            << label << " reports this graph is not applicable"
            << (referenceCoversGraph(
                    _referenceType, _bundle->graphBuffer.data(), _bundle->graphBuffer.size())
                    ? ""
                    : " (it has node types outside the reference's op set)")
            << "\n  bundle: " << _bundlePath;

        executor.execute(_bundle->graphBuffer.data(), _bundle->graphBuffer.size(), variantPack);
    }
    catch(const ReferenceCapabilityError& e)
    {
        FAIL() << label << " reported a capability miss: " << e.what()
               << "\n  bundle: " << _bundlePath;
    }
    catch(const std::exception& e)
    {
        FAIL() << label << " threw: " << e.what() << "\n  bundle: " << _bundlePath;
    }

    detail::markOutputsModified(outputs, useDevice());

    std::vector<std::string> unscanned;
    for(const auto uid : _bundle->outputTensorUids)
    {
        auto it = outputs.find(uid);
        if(it == outputs.end() || it->second == nullptr)
        {
            continue;
        }
        const auto scan = scanNonFinite(*it->second);
        if(!scan.scanned)
        {
            unscanned.push_back(std::to_string(uid));
            continue;
        }
        EXPECT_EQ(scan.nans + scan.infs, 0u)
            << label << " output tensor " << uid << ": " << scan.nans << " NaN and " << scan.infs
            << " Inf of " << scan.elements << " elements; first at "
            << formatIndex(scan.firstBadIndex.value_or(std::vector<int64_t>{}))
            << " (a NaN can also be the unwritten-output sentinel)\n  bundle: " << _bundlePath;
    }
    if(!unscanned.empty())
    {
        std::string list;
        for(const auto& uid : unscanned)
        {
            list += (list.empty() ? "" : ",") + uid;
        }
        RecordProperty("unscanned_outputs", list);
    }
}

} // namespace hipdnn_integration_tests::bundle
