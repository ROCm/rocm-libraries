// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MiopenApi.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

namespace miopen_plugin
{

class MiopenActivationDescriptor
{
public:
    explicit MiopenActivationDescriptor(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttrs);

    MiopenActivationDescriptor(const MiopenActivationDescriptor&) = delete;
    MiopenActivationDescriptor& operator=(const MiopenActivationDescriptor&) = delete;

    MiopenActivationDescriptor(MiopenActivationDescriptor&& other) noexcept;
    MiopenActivationDescriptor& operator=(MiopenActivationDescriptor&& other) noexcept;

    ~MiopenActivationDescriptor();

    miopenActivationDescriptor_t activationDescriptor() const;

private:
    miopenActivationDescriptor_t _descriptor{nullptr};
};

} // namespace miopen_plugin
