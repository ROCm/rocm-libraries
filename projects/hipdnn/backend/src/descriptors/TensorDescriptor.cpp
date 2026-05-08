// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TensorDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DataTypeConversion.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <cstring>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void TensorDescriptor::finalize()
{
    THROW_IF_TRUE(_data.dims.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::finalize() failed: dimensions not set");
    THROW_IF_TRUE(_data.strides.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::finalize() failed: strides not set");
    THROW_IF_NE(_data.dims.size(),
                _data.strides.size(),
                HIPDNN_STATUS_BAD_PARAM,
                "TensorDescriptor::finalize() failed: dims and strides size mismatch");
    THROW_IF_TRUE(_data.data_type == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::finalize() failed: data type not set");

    // Pass-by-value tensors are currently required to supply a value at descriptor
    // creation time. In the future, pass-by-value tensors may also support setting
    // values through variant packs.

    HipdnnBackendDescriptorImpl<TensorDescriptor>::finalize();
}

void TensorDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                    hipdnnBackendAttributeType_t attributeType,
                                    int64_t requestedElementCount,
                                    int64_t* elementCount,
                                    void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "TensorDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_TENSOR_UNIQUE_ID:
        getScalar(_data.uid,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "TensorDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_NAME_EXT:
        getName(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_DATA_TYPE:
        getDataType(_data.data_type,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "TensorDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_DIMENSIONS:
        getInt64Vector(_data.dims,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "TensorDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_STRIDES:
        getInt64Vector(_data.strides,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "TensorDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_IS_VIRTUAL:
        getScalar(_data.virtual_,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "TensorDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_VALUE_EXT:
        getTensorValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_IS_BY_VALUE:
    {
        const bool isByValue
            = _data.value.type != hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE;
        getScalar(isByValue,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "TensorDescriptor::getAttribute()");
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "TensorDescriptor::getAttribute: attributeName not supported");
    }
}

void TensorDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                    hipdnnBackendAttributeType_t attributeType,
                                    int64_t elementCount,
                                    const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "TensorDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_TENSOR_UNIQUE_ID:
        setScalar(_data.uid,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "TensorDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_NAME_EXT:
        setName(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_DATA_TYPE:
        setDataType(_data.data_type,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "TensorDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_DIMENSIONS:
        setInt64Vector(_data.dims,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "TensorDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_STRIDES:
        setInt64Vector(_data.strides,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "TensorDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_IS_VIRTUAL:
        setScalar(_data.virtual_,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "TensorDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_TENSOR_VALUE_EXT:
        setTensorValue(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "TensorDescriptor::setAttribute: attributeName not supported");
    }
}

void TensorDescriptor::setName(hipdnnBackendAttributeType_t attributeType,
                               int64_t elementCount,
                               const void* arrayOfElements)
{
    setString(_data.name,
              attributeType,
              elementCount,
              arrayOfElements,
              "TensorDescriptor::setAttribute()");
}

void TensorDescriptor::getName(hipdnnBackendAttributeType_t attributeType,
                               int64_t requestedElementCount,
                               int64_t* elementCount,
                               void* arrayOfElements) const
{
    getString(_data.name,
              attributeType,
              requestedElementCount,
              elementCount,
              arrayOfElements,
              "TensorDescriptor::getAttribute()");
}

void TensorDescriptor::setTensorValue(hipdnnBackendAttributeType_t attributeType,
                                      int64_t elementCount,
                                      const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): attributeType must be HIPDNN_TYPE_CHAR "
                   "for TENSOR_VALUE");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_TRUE(_data.data_type == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::setAttribute(): data type must be set before tensor value");

    auto expectedSize = getDataTypeByteSize(_data.data_type);
    THROW_IF_FALSE(elementCount == expectedSize,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): elementCount (" + std::to_string(elementCount)
                       + ") does not match data type byte size (" + std::to_string(expectedSize)
                       + ")");

    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto bytes = static_cast<const uint8_t*>(arrayOfElements);

    switch(_data.data_type)
    {
    case DataType::FLOAT:
    {
        float val;
        std::memcpy(&val, bytes, sizeof(float));
        _data.value.Set(Float32Value(val));
        break;
    }
    case DataType::DOUBLE:
    {
        double val;
        std::memcpy(&val, bytes, sizeof(double));
        _data.value.Set(Float64Value(val));
        break;
    }
    case DataType::HALF:
    {
        hipdnn_data_sdk::types::half val;
        std::memcpy(&val, bytes, sizeof(val));
        _data.value.Set(Float16Value(static_cast<float>(val)));
        break;
    }
    case DataType::BFLOAT16:
    {
        hipdnn_data_sdk::types::bfloat16 val;
        std::memcpy(&val, bytes, sizeof(val));
        _data.value.Set(BFloat16Value(static_cast<float>(val)));
        break;
    }
    case DataType::INT32:
    {
        int32_t val;
        std::memcpy(&val, bytes, sizeof(int32_t));
        _data.value.Set(Int32Value(val));
        break;
    }
    case DataType::INT64:
    {
        int64_t val;
        std::memcpy(&val, bytes, sizeof(int64_t));
        _data.value.Set(Int64Value(val));
        break;
    }
    case DataType::BOOLEAN:
    {
        bool val;
        std::memcpy(&val, bytes, sizeof(bool));
        _data.value.Set(BoolValue(val));
        break;
    }
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    {
        _data.value.Set(Float8Value(bytes[0]));
        break;
    }
    default:
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            "TensorDescriptor::setAttribute(): unsupported data type for TENSOR_VALUE");
    }
}

void TensorDescriptor::getTensorValue(hipdnnBackendAttributeType_t attributeType,
                                      int64_t requestedElementCount,
                                      int64_t* elementCount,
                                      void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): attributeType must be HIPDNN_TYPE_CHAR "
                   "for TENSOR_VALUE");

    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "TensorDescriptor::getAttribute(): elementCount is null");
        THROW_IF_TRUE(_data.value.type == TensorValue::NONE,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): tensor value is not set");
        *elementCount = getDataTypeByteSize(_data.data_type);
        return;
    }

    THROW_IF_TRUE(_data.value.type == TensorValue::NONE,
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::getAttribute(): tensor value is not set");

    auto byteSize = getDataTypeByteSize(_data.data_type);
    THROW_IF_FALSE(requestedElementCount >= byteSize,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): requestedElementCount ("
                       + std::to_string(requestedElementCount)
                       + ") is less than data type byte size (" + std::to_string(byteSize) + ")");

    auto output = static_cast<uint8_t*>(arrayOfElements);

    switch(_data.data_type)
    {
    case DataType::FLOAT:
    {
        const auto* val = _data.value.AsFloat32Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        auto nativeVal = val->value();
        std::memcpy(output, &nativeVal, sizeof(float));
        break;
    }
    case DataType::DOUBLE:
    {
        const auto* val = _data.value.AsFloat64Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        auto nativeVal = val->value();
        std::memcpy(output, &nativeVal, sizeof(double));
        break;
    }
    case DataType::HALF:
    {
        const auto* val = _data.value.AsFloat16Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        auto nativeVal = hipdnn_data_sdk::types::half(val->value());
        std::memcpy(output, &nativeVal, sizeof(nativeVal));
        break;
    }
    case DataType::BFLOAT16:
    {
        const auto* val = _data.value.AsBFloat16Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        auto nativeVal = hipdnn_data_sdk::types::bfloat16(val->value());
        std::memcpy(output, &nativeVal, sizeof(nativeVal));
        break;
    }
    case DataType::INT32:
    {
        const auto* val = _data.value.AsInt32Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        auto nativeVal = val->value();
        std::memcpy(output, &nativeVal, sizeof(int32_t));
        break;
    }
    case DataType::INT64:
    {
        const auto* val = _data.value.AsInt64Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        auto nativeVal = val->value();
        std::memcpy(output, &nativeVal, sizeof(int64_t));
        break;
    }
    case DataType::BOOLEAN:
    {
        const auto* val = _data.value.AsBoolValue();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        auto nativeVal = val->value();
        std::memcpy(output, &nativeVal, sizeof(bool));
        break;
    }
    case DataType::UINT8:
    case DataType::INT8:
    {
        const auto* val = _data.value.AsFloat8Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        output[0] = val->value();
        break;
    }
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    {
        const auto* val = _data.value.AsFloat8Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch");
        output[0] = val->value();
        break;
    }
    default:
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            "TensorDescriptor::getAttribute(): unsupported data type for TENSOR_VALUE");
    }

    if(elementCount != nullptr)
    {
        *elementCount = byteSize;
    }
}

std::shared_ptr<TensorDescriptor> TensorDescriptor::fromFlatBuffer(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorT)
{
    auto desc = std::make_shared<TensorDescriptor>();
    desc->_data = tensorT;
    desc->finalize();
    return desc;
}

std::shared_ptr<TensorDescriptor> TensorDescriptor::fromFlatBuffer(
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT&& tensorT)
{
    auto desc = std::make_shared<TensorDescriptor>();
    desc->_data = std::move(tensorT);
    desc->finalize();
    return desc;
}

hipdnnBackendDescriptorType_t TensorDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_TENSOR_DESCRIPTOR;
}

std::string TensorDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "TensorDescriptor: {uid=" + std::to_string(_data.uid);
    str += ", name=" + _data.name;
    if(_data.data_type == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
    {
        str += ", dataType=UNSET";
    }
    else
    {
        str += std::string(", dataType=")
               + hipdnnGetDataTypeString(fromSdkDataType(_data.data_type));
    }
    str += ", dims=" + vecToString(_data.dims);
    str += ", strides=" + vecToString(_data.strides);
    str += ", virtual=" + std::string(_data.virtual_ ? "true" : "false");
    using hipdnn_flatbuffers_sdk::data_objects::TensorValue;
    if(_data.value.type != TensorValue::NONE)
    {
        str += ", value=";
        switch(_data.value.type)
        {
        case TensorValue::Float32Value:
            str += std::to_string(_data.value.AsFloat32Value()->value());
            break;
        case TensorValue::Float64Value:
            str += std::to_string(_data.value.AsFloat64Value()->value());
            break;
        case TensorValue::Float16Value:
            str += std::to_string(_data.value.AsFloat16Value()->value());
            break;
        case TensorValue::BFloat16Value:
            str += std::to_string(_data.value.AsBFloat16Value()->value());
            break;
        case TensorValue::Int32Value:
            str += std::to_string(_data.value.AsInt32Value()->value());
            break;
        case TensorValue::Int64Value:
            str += std::to_string(_data.value.AsInt64Value()->value());
            break;
        case TensorValue::BoolValue:
            str += _data.value.AsBoolValue()->value() ? "true" : "false";
            break;
        case TensorValue::Float8Value:
            str += std::to_string(_data.value.AsFloat8Value()->value());
            break;
        default:
            break;
        }
    }
    str += "}";
    return str;
}

} // namespace hipdnn_backend
