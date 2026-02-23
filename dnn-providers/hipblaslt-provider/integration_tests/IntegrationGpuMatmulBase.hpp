// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "IntegrationGraphVerificationHarness.hpp"

namespace hipblaslt_plugin::test_utilities
{

template <typename DataType, typename TestParamsType>
class IntegrationGpuMatmulBase
    : public IntegrationGraphVerificationHarness<DataType, TestParamsType>
{
protected:
    void runGraphTest(float tolerance) override
    {
        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name(getGraphName());

        auto dataType = hipdnn_frontend::getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto cAttr = initGraph(this->GetParam(), graphObj);
        cAttr->set_output(true);

        this->registerValidator(cAttr, tolerance);
        this->verifyGraph(graphObj, getSeed(this->GetParam()));
    }

    static std::vector<int64_t> generateInputStrideOrder(const std::vector<int64_t>& dims,
                                                         bool transpose)
    {
        std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);
        if(transpose)
        {
            const size_t rank = dims.size();
            strides[rank - 1] = dims[rank - 2];
            strides[rank - 2] = 1;
        }
        return strides;
    }

    virtual std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
        initGraph(const TestParamsType& testParams, hipdnn_frontend::graph::Graph& graphObj) const
        = 0;
    virtual std::string getGraphName() const = 0;
    virtual unsigned int getSeed(const TestParamsType& testParams) const = 0;
};

} // namespace hipblaslt_plugin::test_utilities
