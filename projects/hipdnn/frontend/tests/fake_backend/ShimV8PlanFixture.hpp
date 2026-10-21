// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ShimV8BuilderPlanFixture plus reading a config's engine index back through
// the backend C API.
#pragma once

#include "ShimV8BuilderPlanFixture.hpp"

#include <cstdint>
#include <vector>

namespace hipdnn_shim_test
{

class ShimV8PlanFixture : public ShimV8BuilderPlanFixture
{
protected:
    // Engine index read back through ENGINECFG_ENGINE and ENGINE_GLOBAL_INDEX.
    static std::int64_t indexOf(cudnnBackendDescriptor_t config)
    {
        cudnnBackendDescriptor_t engineDescriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engineDescriptor),
                  CUDNN_STATUS_SUCCESS);
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(config,
                                           CUDNN_ATTR_ENGINECFG_ENGINE,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &count,
                                           static_cast<void*>(&engineDescriptor)),
                  CUDNN_STATUS_SUCCESS)
            << lastShimError();
        std::int64_t index = -1;
        EXPECT_EQ(cudnnBackendGetAttribute(engineDescriptor,
                                           CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &count,
                                           static_cast<void*>(&index)),
                  CUDNN_STATUS_SUCCESS)
            << lastShimError();
        EXPECT_EQ(cudnnBackendDestroyDescriptor(engineDescriptor), CUDNN_STATUS_SUCCESS);
        return index;
    }

    static std::vector<std::int64_t> indicesOf(const v8fe::EngineConfigList& configs)
    {
        std::vector<std::int64_t> indices;
        indices.reserve(configs.size());
        for(const v8fe::ManagedOpaqueDescriptor& config : configs)
        {
            indices.push_back(indexOf(config->get_backend_descriptor()));
        }
        return indices;
    }
};

} // namespace hipdnn_shim_test
