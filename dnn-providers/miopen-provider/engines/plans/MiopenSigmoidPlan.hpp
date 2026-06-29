#pragma once

#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "MiopenActivationDescriptor.hpp"
#include "MiopenTensor.hpp"

namespace miopen_plugin
{

class MiopenSigmoidPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    MiopenSigmoidPlan(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    MiopenSigmoidPlan(const MiopenSigmoidPlan&) = delete;
    MiopenSigmoidPlan& operator=(const MiopenSigmoidPlan&) = delete;

    MiopenSigmoidPlan(MiopenSigmoidPlan&&) = delete;
    MiopenSigmoidPlan& operator=(MiopenSigmoidPlan&&) = delete;

    ~MiopenSigmoidPlan() override = default;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    MiopenTensor _input;
    MiopenTensor _output;
    MiopenActivationDescriptor _activation;
};

} // namespace miopen_plugin
