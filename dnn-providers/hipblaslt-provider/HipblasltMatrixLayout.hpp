// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>

namespace hipblaslt_plugin
{

class HipblasltMatrixLayout
{
public:
    HipblasltMatrixLayout() = default;
    HipblasltMatrixLayout(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tensor);

    HipblasltMatrixLayout(const HipblasltMatrixLayout&) = delete;
    HipblasltMatrixLayout& operator=(const HipblasltMatrixLayout&) = delete;

    HipblasltMatrixLayout(HipblasltMatrixLayout&& other) noexcept;
    HipblasltMatrixLayout& operator=(HipblasltMatrixLayout&& other) noexcept;

    ~HipblasltMatrixLayout();

    int64_t uid() const;

    hipblasLtMatrixLayout_t matrixLayout() const;

    void setBatchCount(int64_t count);
    void setStridedBatchOffset(int64_t stride);

private:
    int64_t _uid{0};
    hipblasLtMatrixLayout_t _matrix_layout{nullptr};
};

} // namespace hipblaslt_plugin
