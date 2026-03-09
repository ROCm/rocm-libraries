// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "MatmulOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

namespace hipdnn_backend
{

void MatmulOperationDescriptor::finalize()
{
    THROW_IF_NULL(_aDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MatmulOperationDescriptor::finalize() failed: A tensor not set");
    THROW_IF_NULL(_bDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MatmulOperationDescriptor::finalize() failed: B tensor not set");
    THROW_IF_NULL(_cDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MatmulOperationDescriptor::finalize() failed: C tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MatmulOperationDescriptor::finalize() failed: compute data type not set");

    HipdnnBackendDescriptorImpl<MatmulOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void MatmulOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t elementCount,
                                             const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "MatmulOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_MATMUL_A_EXT:
        setTensorDescriptor(_aDesc,
                            _data.a_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MATMUL_B_EXT:
        setTensorDescriptor(_bDesc,
                            _data.b_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MATMUL_C_EXT:
        setTensorDescriptor(_cDesc,
                            _data.c_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_MATMUL_MATH_PREC_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "MatmulOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "MatmulOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void MatmulOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t requestedElementCount,
                                             int64_t* elementCount,
                                             void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "MatmulOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_MATMUL_A_EXT:
        getTensorDescriptor(_aDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MATMUL_B_EXT:
        getTensorDescriptor(_bDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MATMUL_C_EXT:
        getTensorDescriptor(_cDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_MATMUL_MATH_PREC_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "MatmulOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "MatmulOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    MatmulOperationDescriptor::getTensorDescriptors() const
{
    return {_aDesc, _bDesc, _cDesc};
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT> MatmulOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::MatmulAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t MatmulOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR_EXT;
}

std::string MatmulOperationDescriptor::toString() const
{
    std::string str = "MatmulOperationDescriptor: {";
    str += "a_uid=" + std::to_string(_data.a_tensor_uid);
    str += ", b_uid=" + std::to_string(_data.b_tensor_uid);
    str += ", c_uid=" + std::to_string(_data.c_tensor_uid);
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
