// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <miopen/miopen.h>

namespace miopen_legacy_plugin
{

class MiopenConvDescriptor
{
public:
    MiopenConvDescriptor() = default;
    MiopenConvDescriptor(size_t spatialDimCount,
                         const hipdnn_sdk::data_objects::ConvolutionFwdAttributes& attributes);

    MiopenConvDescriptor(const MiopenConvDescriptor&) = delete;
    MiopenConvDescriptor& operator=(const MiopenConvDescriptor&) = delete;

    MiopenConvDescriptor(MiopenConvDescriptor&& other) noexcept;
    MiopenConvDescriptor& operator=(MiopenConvDescriptor&& other) noexcept;

    ~MiopenConvDescriptor();

    miopenConvolutionDescriptor_t convDescriptor() const;

private:
    miopenConvolutionDescriptor_t _descriptor = nullptr;
};

}
