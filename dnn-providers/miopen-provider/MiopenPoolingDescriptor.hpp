// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

#include "MiopenApi.hpp"

namespace miopen_plugin
{

// RAII owner of an N-d miopenPoolingDescriptor_t. Indices, when requested, are always recorded
// in image mode: a flattened spatial offset into the unpadded input, which is hipDNN's index
// convention.
class MiopenPoolingDescriptor
{
public:
    MiopenPoolingDescriptor(miopenPoolingMode_t mode,
                            const std::vector<int>& window,
                            const std::vector<int>& padding,
                            const std::vector<int>& stride,
                            miopenIndexType_t indexType);

    MiopenPoolingDescriptor(const MiopenPoolingDescriptor&) = delete;
    MiopenPoolingDescriptor& operator=(const MiopenPoolingDescriptor&) = delete;

    MiopenPoolingDescriptor(MiopenPoolingDescriptor&& other) noexcept;
    MiopenPoolingDescriptor& operator=(MiopenPoolingDescriptor&& other) noexcept;

    ~MiopenPoolingDescriptor();

    miopenPoolingDescriptor_t poolingDescriptor() const;

private:
    miopenPoolingDescriptor_t _descriptor{nullptr};
};

} // namespace miopen_plugin
