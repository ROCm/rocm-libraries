// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormUtilities.hpp"
#include "hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h"
#include "hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h"
#include "hipdnn_plugin_sdk/PluginApiDataTypes.h"
#include "hipdnn_plugin_sdk/PluginException.hpp"
#include <algorithm>
#include <cstdint>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "LayernormApplicabilityChecks.hpp"
#include "core/Utils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/PlanUtils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormBwdKernel.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormFwdKernel.hpp"

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::layernorm
{

namespace
{

// True when the dims of size > 1, taken in increasing stride order, tile memory with no gap
// and no overlap: each stride equals the product of the faster extents. Size-1 dims are
// never stepped over, so their strides are free.
bool hasDenseNonOverlappingStrides(const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides)
{
    std::vector<size_t> order;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(dims[i] > 1)
        {
            order.push_back(i);
        }
    }
    std::sort(
        order.begin(), order.end(), [&](size_t a, size_t b) { return strides[a] < strides[b]; });

    int64_t expected = 1;
    for(const size_t i : order)
    {
        if(strides[i] != expected)
        {
            return false;
        }
        expected *= dims[i];
    }
    return true;
}

std::optional<int64_t> toStdOptional(flatbuffers::Optional<int64_t> value)
{
    return value.has_value() ? std::make_optional(value.value()) : std::nullopt;
}

} // namespace

// --- Type Configuration Helpers ---

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> type_configs::getAllowedIoTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.io);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    type_configs::getAllowedAffineTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.affine);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    type_configs::getAllowedStatTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.stat);
    }
    return types;
}

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>
    type_configs::getAllowedEpsilonTypes()
{
    std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.epsilon);
    }
    return types;
}

// --- Validation Utilities ---

void LayernormValidator::validateNormalizedDim(const std::vector<int64_t>& ioTensorIds,
                                               const std::vector<int64_t>& affineTensorIds,
                                               const std::vector<int64_t>& statTensorIds)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Normalized dimension detection for layernorm requires IO tensors.");
    }

    if(affineTensorIds.empty() && statTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Normalized dimension detection for layernorm requires at least one affine tensor or "
            "one stat tensor.");
    }

    const auto& ioAttr = findTensorAttributes(_tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioAttr.dims()->begin(), ioAttr.dims()->end());

    const size_t affineNormalizedDimMin
        = getMinNormalizedDimFromAffine(ioTensorIds[0], affineTensorIds[0], _tensorMap);
    const size_t affineNormalizedDimMax
        = getMaxNormalizedDimFromAffine(ioTensorIds[0], affineTensorIds[0], _tensorMap);

    const size_t statNormalizedDimMin
        = statTensorIds.empty()
              ? 0
              : getMinNormalizedDimFromStat(ioTensorIds[0], statTensorIds[0], _tensorMap);
    const size_t statNormalizedDimMax
        = statTensorIds.empty()
              ? ioDims.size()
              : getMaxNormalizedDimFromStat(ioTensorIds[0], statTensorIds[0], _tensorMap);

    if(std::max(affineNormalizedDimMin, statNormalizedDimMin)
       > std::min(affineNormalizedDimMax, statNormalizedDimMax))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Affine tensors and stat tensors produce conflicting normalized dimensions for "
            "layernorm.");
    }
}

// --- Component Validators

void LayernormValidator::checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds)
{
    // Skip tensors with embedded scalar values (epsilon, momentum) - they don't have layouts or dimensions to validate
    std::vector<TensorDescriptor> tensors;
    tensors.reserve(tensorIds.size());

    for(const auto& id : tensorIds)
    {
        auto attr = _tensorMap.at(id);
        if(!hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(attr))
        {
            tensors.emplace_back(attr);
        }
    }

    validateConsistentDimensions(tensors);
    validatePackedTensors(tensors);
    validateConsistentLayouts(tensors);
}

void LayernormValidator::checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                                       const std::vector<int64_t>& affineTensorIds,
                                                       const std::vector<int64_t>& statTensorIds,
                                                       const std::vector<int64_t>& epsilonTensorIds)
{
    const auto allowedIoTypes = type_configs::getAllowedIoTypes();
    for(const auto ioTensorId : ioTensorIds)
    {
        const auto& ioTensorAttr = core::utils::findTensorAttributes(_tensorMap, ioTensorId);
        validateDataTypeIsSupported(ioTensorAttr.data_type(),
                                    allowedIoTypes,
                                    "Layernorm implementation supports only FLOAT, HALF and "
                                    "BFLOAT16 data types for x and y tensors.");
    }

    const auto allowedAffineTypes = type_configs::getAllowedAffineTypes();
    validateConsistentDataTypes(
        affineTensorIds,
        allowedAffineTypes,
        "Layernorm implementation supports only FLOAT, HALF and BFLOAT16 data types for scale and "
        "bias tensors.",
        "All affine tensors for layernorm must have the same data type.");

    const auto allowedStatTypes = type_configs::getAllowedStatTypes();
    validateConsistentDataTypes(
        statTensorIds,
        allowedStatTypes,
        "Layernorm implementation supports only FLOAT, HALF and BFLOAT16 data types for mean and "
        "inverse variance tensors.",
        "All stat tensors for layernorm must have the same data type.");

    const auto allowedEpsilonTypes = type_configs::getAllowedEpsilonTypes();
    if(allowedEpsilonTypes.size() == 1)
    {
        validateFixedDataType(
            epsilonTensorIds,
            *allowedEpsilonTypes.begin(),
            "Layernorm implementation supports only FLOAT data type for epsilon tensors.");
    }
    else
    {
        validateConsistentDataTypes(
            epsilonTensorIds,
            allowedEpsilonTypes,
            "Layernorm epsilon tensors use unsupported data type.",
            "All epsilon tensors for layernorm must have the same data type.");
    }
}

void LayernormValidator::checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                                    const std::vector<int64_t>& affineTensorIds,
                                                    const std::vector<int64_t>& statTensorIds)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for layernorm.");
    }

    const auto& ioTensorAttr = findTensorAttributes(_tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validateConsistentShapes(
        ioTensorIds, ioDims, "All IO tensors for layernorm must have the same shape.");
    validateElementCountFitsUint32(
        ioDims,
        "Layernorm kernels index tensors with 32-bit unsigned arithmetic; the IO tensor "
        "has too many elements.");

    const auto& affineTensorAttr = findTensorAttributes(_tensorMap, affineTensorIds[0]);
    const std::vector<int64_t> affineDims(affineTensorAttr.dims()->begin(),
                                          affineTensorAttr.dims()->end());
    validateConsistentShapes(
        affineTensorIds, affineDims, "All affine tensors for layernorm must have the same shape.");

    if(!statTensorIds.empty())
    {
        const auto& statTensorAttr = findTensorAttributes(_tensorMap, statTensorIds[0]);
        const std::vector<int64_t> statDims(statTensorAttr.dims()->begin(),
                                            statTensorAttr.dims()->end());
        validateConsistentShapes(statTensorIds,
                                 statDims,
                                 "Mean and variance tensors for batchnorm must have shape "
                                 "derived from IO tensor shape.");
    }
}

void LayernormValidator::checkTensorMemoryOrderSupported(
    int64_t xTensorId,
    const std::vector<NamedTensorId>& affineTensors,
    const std::vector<NamedTensorId>& statTensors)
{
    const auto& xAttr = findTensorAttributes(_tensorMap, xTensorId);
    const std::vector<int64_t> xStrides(xAttr.strides()->begin(), xAttr.strides()->end());
    const size_t xRank = xAttr.dims()->size();

    const auto check = [&](const NamedTensorId& named, bool alignTrailing) {
        const auto& attr = findTensorAttributes(_tensorMap, named.second);
        if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&attr))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("Layernorm ") + named.first + " tensor must not be pass-by-value.");
        }

        const std::vector<int64_t> dims(attr.dims()->begin(), attr.dims()->end());
        const std::vector<int64_t> strides(attr.strides()->begin(), attr.strides()->end());
        if(dims.size() > xRank)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("Layernorm ") + named.first + " tensor has more dimensions than x.");
        }
        if(!hipdnn_data_sdk::utilities::isTensorPacked(dims, strides))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           std::string("Layernorm ") + named.first
                                                               + " tensor must be packed.");
        }
        if(!hasDenseNonOverlappingStrides(dims, strides))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("Layernorm ") + named.first
                    + " tensor strides must address each element once with no gap.");
        }

        const size_t offset = alignTrailing ? xRank - dims.size() : 0;
        for(size_t a = 0; a < dims.size(); ++a)
        {
            for(size_t b = a + 1; b < dims.size(); ++b)
            {
                if(dims[a] <= 1 || dims[b] <= 1)
                {
                    continue;
                }
                if((strides[a] < strides[b]) != (xStrides[a + offset] < xStrides[b + offset]))
                {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                        std::string("Layernorm ") + named.first
                            + " tensor must have the same memory order as x.");
                }
            }
        }
    };

    for(const auto& named : affineTensors)
    {
        check(named, true);
    }
    for(const auto& named : statTensors)
    {
        check(named, false);
    }
}

void LayernormValidator::checkOperandDimsMatchNormalizedDims(
    int64_t xTensorId,
    const std::vector<NamedTensorId>& affineTensors,
    std::optional<int64_t> meanTensorId,
    const std::vector<NamedTensorId>& statTensors)
{
    const auto& xAttr = findTensorAttributes(_tensorMap, xTensorId);
    const std::vector<int64_t> xDims(xAttr.dims()->begin(), xAttr.dims()->end());
    const size_t xRank = xDims.size();

    const auto* scaleAttr = &findTensorAttributes(_tensorMap, affineTensors.front().second);
    const auto meanAttr = meanTensorId.has_value()
                              ? std::make_optional(&findTensorAttributes(_tensorMap, *meanTensorId))
                              : std::nullopt;
    const size_t normalizedDim = guessNormalizedDim(&xAttr, scaleAttr, meanAttr);

    const auto check = [&](const NamedTensorId& named, bool isAffine) {
        const auto& attr = findTensorAttributes(_tensorMap, named.second);
        const std::vector<int64_t> dims(attr.dims()->begin(), attr.dims()->end());
        const size_t offset = isAffine ? xRank - dims.size() : 0;
        for(size_t i = 0; i < xRank; ++i)
        {
            const bool inOperand = i >= offset && i - offset < dims.size();
            const int64_t dim = inOperand ? dims[i - offset] : 1;
            const bool normalized = i >= normalizedDim;
            const int64_t expected = (normalized == isAffine) ? xDims[i] : 1;
            if(dim != expected)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    std::string("Layernorm ") + named.first + " dim " + std::to_string(i) + " is "
                        + std::to_string(dim) + ", expected " + std::to_string(expected)
                        + " for normalized dims starting at " + std::to_string(normalizedDim)
                        + ".");
            }
        }
    };

    for(const auto& named : affineTensors)
    {
        check(named, true);
    }
    for(const auto& named : statTensors)
    {
        check(named, false);
    }
}

void LayernormValidator::checkDxDataTypeSupported(int64_t xTensorId, int64_t dxTensorId)
{
    const auto& xAttr = findTensorAttributes(_tensorMap, xTensorId);
    const auto& dxAttr = findTensorAttributes(_tensorMap, dxTensorId);
    if(dxAttr.data_type() != xAttr.data_type())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Layernorm backward requires dx to have the same data type as x.");
    }
}

// --- High-level Configuration Validators ---

void LayernormValidator::checkTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& lnAttr)
{
    std::vector<int64_t> ioTensorIds = {lnAttr.x_tensor_uid(), lnAttr.y_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {lnAttr.scale_tensor_uid(), lnAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(lnAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(lnAttr.mean_tensor_uid().value());
    }
    if(lnAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(lnAttr.inv_variance_tensor_uid().value());
    }
    std::vector<NamedTensorId> fwdStatTensors;
    if(lnAttr.mean_tensor_uid().has_value())
    {
        fwdStatTensors.emplace_back("mean", lnAttr.mean_tensor_uid().value());
    }
    if(lnAttr.inv_variance_tensor_uid().has_value())
    {
        fwdStatTensors.emplace_back("inv_variance", lnAttr.inv_variance_tensor_uid().value());
    }
    const std::vector<int64_t> epsilonTensorIds = {lnAttr.epsilon_tensor_uid()};

    std::vector<int64_t> ioAndStatTensorIds
        = std::vector<int64_t>(ioTensorIds.begin(), ioTensorIds.end());
    ioAndStatTensorIds.insert(ioAndStatTensorIds.end(), statTensorIds.begin(), statTensorIds.end());

    {
        std::vector<int64_t> bufferTensorIds = ioAndStatTensorIds;
        bufferTensorIds.insert(
            bufferTensorIds.end(), affineTensorIds.begin(), affineTensorIds.end());
        validateNoVirtualTensors(
            bufferTensorIds, "Layernorm kernel inputs and outputs must not be virtual tensors.");
    }
    checkTensorLayoutsAndDimsSupported(ioAndStatTensorIds);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, epsilonTensorIds);
    checkTensorMemoryOrderSupported(
        lnAttr.x_tensor_uid(),
        {{"scale", lnAttr.scale_tensor_uid()}, {"bias", lnAttr.bias_tensor_uid()}},
        fwdStatTensors);
    checkEpsilonSupported(lnAttr.epsilon_tensor_uid());
    validateNormalizedDim(ioTensorIds, affineTensorIds, statTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds);
    checkOperandDimsMatchNormalizedDims(
        lnAttr.x_tensor_uid(),
        {{"scale", lnAttr.scale_tensor_uid()}, {"bias", lnAttr.bias_tensor_uid()}},
        toStdOptional(lnAttr.mean_tensor_uid()),
        fwdStatTensors);
}

void LayernormValidator::checkTensorConfigSupported(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes& lnAttr)
{
    const std::vector<int64_t> ioTensorIds
        = {lnAttr.dy_tensor_uid(), lnAttr.x_tensor_uid(), lnAttr.dx_tensor_uid()};
    const std::vector<int64_t> affineTensorIds
        = {lnAttr.scale_tensor_uid(), lnAttr.dscale_tensor_uid(), lnAttr.dbias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(lnAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(lnAttr.mean_tensor_uid().value());
    }
    if(lnAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(lnAttr.inv_variance_tensor_uid().value());
    }
    std::vector<NamedTensorId> bwdStatTensors;
    if(lnAttr.mean_tensor_uid().has_value())
    {
        bwdStatTensors.emplace_back("mean", lnAttr.mean_tensor_uid().value());
    }
    if(lnAttr.inv_variance_tensor_uid().has_value())
    {
        bwdStatTensors.emplace_back("inv_variance", lnAttr.inv_variance_tensor_uid().value());
    }
    std::vector<int64_t> epsilonTensorIds;
    if(lnAttr.epsilon_tensor_uid().has_value())
    {
        epsilonTensorIds.push_back(lnAttr.epsilon_tensor_uid().value());
    }

    {
        std::vector<int64_t> bufferTensorIds = ioTensorIds;
        bufferTensorIds.insert(
            bufferTensorIds.end(), affineTensorIds.begin(), affineTensorIds.end());
        bufferTensorIds.insert(bufferTensorIds.end(), statTensorIds.begin(), statTensorIds.end());
        validateNoVirtualTensors(
            bufferTensorIds, "Layernorm kernel inputs and outputs must not be virtual tensors.");
    }
    checkTensorLayoutsAndDimsSupported(ioTensorIds);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, epsilonTensorIds);
    checkTensorMemoryOrderSupported(lnAttr.x_tensor_uid(),
                                    {{"scale", lnAttr.scale_tensor_uid()},
                                     {"dscale", lnAttr.dscale_tensor_uid()},
                                     {"dbias", lnAttr.dbias_tensor_uid()}},
                                    bwdStatTensors);
    if(lnAttr.epsilon_tensor_uid().has_value())
    {
        checkEpsilonSupported(lnAttr.epsilon_tensor_uid().value());
    }
    validateNormalizedDim(ioTensorIds, affineTensorIds, statTensorIds);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds);
    checkOperandDimsMatchNormalizedDims(lnAttr.x_tensor_uid(),
                                        {{"scale", lnAttr.scale_tensor_uid()},
                                         {"dscale", lnAttr.dscale_tensor_uid()},
                                         {"dbias", lnAttr.dbias_tensor_uid()}},
                                        toStdOptional(lnAttr.mean_tensor_uid()),
                                        bwdStatTensors);
    checkDxDataTypeSupported(lnAttr.x_tensor_uid(), lnAttr.dx_tensor_uid());
}

void checkLaunchShapesFitThreadLimit(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& lnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto meanAttr = lnAttr.mean_tensor_uid().has_value()
                              ? std::make_optional(tensorMap.at(lnAttr.mean_tensor_uid().value()))
                              : std::nullopt;
    const ProblemDescription problem(tensorMap.at(lnAttr.x_tensor_uid()),
                                     tensorMap.at(lnAttr.scale_tensor_uid()),
                                     meanAttr,
                                     Direction::FORWARD);
    for(const auto& shape : layernormFwdLaunchShapes(
            problem.outerSize(), problem.stride(), LayernormFwdParams::LOCAL_SIZE))
    {
        checkLaunchFitsThreadLimit(shape);
    }
}

void checkLaunchShapesFitThreadLimit(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes& lnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto meanAttr = lnAttr.mean_tensor_uid().has_value()
                              ? std::make_optional(tensorMap.at(lnAttr.mean_tensor_uid().value()))
                              : std::nullopt;
    const ProblemDescription problem(tensorMap.at(lnAttr.x_tensor_uid()),
                                     tensorMap.at(lnAttr.scale_tensor_uid()),
                                     meanAttr,
                                     Direction::BACKWARD);
    for(const auto& shape : layernormBwdLaunchShapes(problem.outerSize(),
                                                     problem.innerSize(),
                                                     problem.stride(),
                                                     LayernormBwdParams::MAX_LOCAL_SIZE,
                                                     /*isParallel=*/false,
                                                     /*parallelSize=*/0))
    {
        checkLaunchFitsThreadLimit(shape);
    }
}

bool isFwdConfigApplicable(const LayernormFwdParams& params, const LayernormFwdConfig& config)
{
    if(!isLocalSizeSupported(config.localSize))
    {
        return false;
    }
    const auto shapes = layernormFwdLaunchShapes(params, config);
    return std::all_of(shapes.begin(), shapes.end(), [](const KernelLaunchShape& shape) {
        return launchFitsThreadLimit(shape.blocks, shape.threadsPerBlock);
    });
}

bool isBwdConfigApplicable(const LayernormBwdParams& params,
                           int multiProcessorCount,
                           const LayernormBwdConfig& config)
{
    if(!isLocalSizeSupported(config.localSize))
    {
        return false;
    }
    const auto shapes = layernormBwdLaunchShapes(params, multiProcessorCount, config);
    return std::all_of(shapes.begin(), shapes.end(), [](const KernelLaunchShape& shape) {
        return launchFitsThreadLimit(shape.blocks, shape.threadsPerBlock);
    });
}

} // namespace hip_kernel_provider::layernorm
