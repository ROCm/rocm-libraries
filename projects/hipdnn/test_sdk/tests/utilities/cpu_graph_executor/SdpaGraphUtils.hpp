// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildSdpaFwdGraph(SdpaFwdTensorBundle<InputType>& tensorBundle,
                      hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                      bool causalMask = false)
{
    const auto frontendDataType = hipdnn_test_sdk::utilities::sdkToFrontendDataType(dataType);

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("SdpaFwdTest");
    graph->set_io_data_type(frontendDataType)
        .set_compute_data_type(frontendDataType)
        .set_intermediate_data_type(frontendDataType);

    int64_t uid = 1;
    auto qAttr
        = hipdnn_frontend::graph::makeTensorAttributes("Q", frontendDataType, tensorBundle.qTensor);
    qAttr.set_uid(uid++);
    auto qTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(qAttr));

    auto kAttr
        = hipdnn_frontend::graph::makeTensorAttributes("K", frontendDataType, tensorBundle.kTensor);
    kAttr.set_uid(uid++);
    auto kTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(kAttr));

    auto vAttr
        = hipdnn_frontend::graph::makeTensorAttributes("V", frontendDataType, tensorBundle.vTensor);
    vAttr.set_uid(uid++);
    auto vTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(vAttr));

    hipdnn_frontend::graph::SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("SdpaFwd");
    sdpaAttrs.set_causal_mask(causalMask);

    auto [oTensorAttr, statsAttr] = graph->sdpa(qTensorAttr, kTensorAttr, vTensorAttr, sdpaAttrs);

    if(!oTensorAttr->has_uid())
    {
        oTensorAttr->set_uid(uid++);
    }

    const auto oDims = tensorBundle.oTensor.dims();
    const auto oStrides = hipdnn_data_sdk::utilities::generateStrides(oDims);
    oTensorAttr->set_data_type(frontendDataType)
        .set_dim(oDims)
        .set_stride(oStrides)
        .set_is_virtual(false);

    auto variantPack
        = tensorBundle.createVariantPack(*qTensorAttr, *kTensorAttr, *vTensorAttr, *oTensorAttr);

    return std::make_tuple(graph, variantPack);
}

} // namespace hipdnn_sdk_test_utils
