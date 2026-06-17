// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/ExecutionPlanDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/VariantDescriptor.hpp"

#include <gmock/gmock.h>

namespace hipdnn_backend
{

class MockDescriptorUtility
{
public:
    template <typename ChildDescriptor>
    static std::shared_ptr<ChildDescriptor> asDescriptorUnsafe(hipdnnBackendDescriptor_t descriptor)
    {
        if(!descriptor)
        {
            return nullptr;
        }

        return std::static_pointer_cast<ChildDescriptor>(descriptor->_impl);
    }
};

template <typename DescType>
class MockDescriptor : public HipdnnBackendDescriptorImpl<MockDescriptor<DescType>>
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, isFinalized, (), (const, override));
    MOCK_METHOD(void,
                setAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElement),
                (override));
    MOCK_METHOD(void,
                getAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements),
                (const, override));

    static hipdnnBackendDescriptorType_t getStaticType()
    {
        return DescType::getStaticType();
    }
};

class MockEngineDescriptor : public EngineDescriptor
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, isFinalized, (), (const, override));
    MOCK_METHOD(void,
                setAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElement),
                (override));
    MOCK_METHOD(void,
                getAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements),
                (const, override));

    MOCK_METHOD(std::shared_ptr<const GraphDescriptor>, getGraph, (), (const, override));
    MOCK_METHOD(int64_t, getEngineId, (), (const, override));

    static hipdnnBackendDescriptorType_t getStaticType()
    {
        return HIPDNN_BACKEND_ENGINE_DESCRIPTOR;
    }
};

class MockEngineConfigDescriptor : public EngineConfigDescriptor
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, isFinalized, (), (const, override));
    MOCK_METHOD(void,
                setAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElement),
                (override));
    MOCK_METHOD(void,
                getAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements),
                (const, override));

    MOCK_METHOD(std::shared_ptr<const EngineDescriptor>, getEngine, (), (const, override));
    MOCK_METHOD(hipdnnPluginConstData_t, getSerializedEngineConfig, (), (const, override));

    static hipdnnBackendDescriptorType_t getStaticType()
    {
        return HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR;
    }
};

class MockGraphDescriptor : public GraphDescriptor
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, isFinalized, (), (const, override));
    MOCK_METHOD(void,
                setAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElement),
                (override));
    MOCK_METHOD(void,
                getAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements),
                (const, override));

    MOCK_METHOD(hipdnnHandle_t, getHandle, (), (const, override));
    MOCK_METHOD(hipdnnPluginConstData_t, getSerializedGraph, (), (const, override));
    MOCK_METHOD(bool, isOverrideShapeEnabled, (), (const, override));

    static hipdnnBackendDescriptorType_t getStaticType()
    {
        return HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
    }
};

class MockExecutionPlanDescriptor : public ExecutionPlanDescriptor
{
public:
    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, isFinalized, (), (const, override));
    MOCK_METHOD(void,
                setAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElement),
                (override));
    MOCK_METHOD(void,
                getAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements),
                (const, override));

    MOCK_METHOD(std::shared_ptr<const EngineConfigDescriptor>,
                getEngineConfig,
                (),
                (const, override));
    MOCK_METHOD(int64_t, getEngineId, (), (const, override));
    MOCK_METHOD(const std::vector<int64_t>&, getTensorUids, (), (const, override));
    MOCK_METHOD(bool, isOverrideShapeEnabled, (), (const, override));
    MOCK_METHOD(hipdnnEnginePluginExecutionContext_t, getExecutionContext, (), (const, override));

    static hipdnnBackendDescriptorType_t getStaticType()
    {
        return HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
    }
};

class MockVariantDescriptor : public VariantDescriptor
{
private:
    // Backing storage for default `ReturnRef` behavior on the override accessors;
    // tests that exercise the override-aware dispatch path program their own
    // EXPECT_CALL / ON_CALL with their own backing vectors.
    std::vector<int64_t> _emptyOverrideStorage;

public:
    MockVariantDescriptor()
    {
        // Make the override accessors return references to an empty vector by
        // default so legacy tests that don't program the override APIs continue
        // to take the non-override dispatch path.
        ON_CALL(*this, getOverrideUniqueIds())
            .WillByDefault(::testing::ReturnRef(_emptyOverrideStorage));
        ON_CALL(*this, getOverrideShapes())
            .WillByDefault(::testing::ReturnRef(_emptyOverrideStorage));
        ON_CALL(*this, getOverrideStrides())
            .WillByDefault(::testing::ReturnRef(_emptyOverrideStorage));
        ON_CALL(*this, getOverrideLengths())
            .WillByDefault(::testing::ReturnRef(_emptyOverrideStorage));
    }

    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, isFinalized, (), (const, override));
    MOCK_METHOD(void,
                setAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElement),
                (override));
    MOCK_METHOD(void,
                getAttribute,
                (hipdnnBackendAttributeName_t attributeName,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements),
                (const, override));

    MOCK_METHOD(void*, getWorkspace, (), (const, override));
    MOCK_METHOD(const std::vector<const void*>&, getDataPointers, (), (const, override));
    MOCK_METHOD(const std::vector<int64_t>&, getTensorIds, (), (const, override));

    MOCK_METHOD(const std::vector<int64_t>&, getOverrideUniqueIds, (), (const, override));
    MOCK_METHOD(const std::vector<int64_t>&, getOverrideShapes, (), (const, override));
    MOCK_METHOD(const std::vector<int64_t>&, getOverrideStrides, (), (const, override));
    MOCK_METHOD(const std::vector<int64_t>&, getOverrideLengths, (), (const, override));

    static hipdnnBackendDescriptorType_t getStaticType()
    {
        return HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
    }
};

ACTION_P(SetArg4ToInt64, value) // NOLINT
{
    *static_cast<int64_t*>(arg4) = value;
}

/// Action helper: program `MockGraphDescriptor::getAttribute()` to write a
/// boolean attribute value into the user-provided buffer (arg4) and report
/// `count == 1` via the inout count pointer (arg3). Mirrors `SetArg4ToInt64`.
ACTION_P(SetArg4ToBool, value) // NOLINT
{
    *static_cast<bool*>(arg4) = value;
    *arg3 = int64_t{1};
}

} // namespace hipdnn_backend
