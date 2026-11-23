// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MiopenApi.hpp"

namespace miopen_plugin
{

// Owns a miopenReduceTensorDescriptor_t configured for legacy miopenReduceTensor with an fp32
// compute type, NaN propagation and no indices output.
class MiopenReduceTensorDescriptor
{
public:
    explicit MiopenReduceTensorDescriptor(miopenReduceTensorOp_t reduceOp);

    MiopenReduceTensorDescriptor(const MiopenReduceTensorDescriptor&) = delete;
    MiopenReduceTensorDescriptor& operator=(const MiopenReduceTensorDescriptor&) = delete;

    MiopenReduceTensorDescriptor(MiopenReduceTensorDescriptor&& other) noexcept;
    MiopenReduceTensorDescriptor& operator=(MiopenReduceTensorDescriptor&& other) noexcept;

    ~MiopenReduceTensorDescriptor();

    miopenReduceTensorDescriptor_t reduceTensorDescriptor() const;

private:
    miopenReduceTensorDescriptor_t _descriptor{nullptr};
};

} // namespace miopen_plugin
