// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fixture-free helpers shared by the cuDNN v8 shim tests.
#pragma once

#include <hipdnn_compatibility/cudnn/detail/v8/error.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_shim_test
{

using Dims = std::vector<std::int64_t>;

inline Dims packedStrides(const Dims& dims)
{
    Dims strides(dims.size(), 1);
    for(std::size_t i = dims.size(); i > 1; --i)
    {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

// The ShimError `body` throws; a body that throws nothing fails the test.
template <typename Body>
hipdnn_frontend::compatibility::cudnn_frontend::detail::v8::ShimError shimErrorOf(Body&& body)
{
    try
    {
        std::forward<Body>(body)();
    }
    catch(const hipdnn_frontend::compatibility::cudnn_frontend::detail::v8::ShimError& error)
    {
        return error;
    }
    ADD_FAILURE() << "no ShimError thrown";
    return {CUDNN_STATUS_SUCCESS, ""};
}

template <typename Body>
void expectShimError(Body&& body, cudnnStatus_t status, const std::string& message)
{
    const auto error = shimErrorOf(std::forward<Body>(body));
    EXPECT_EQ(error.status(), status);
    EXPECT_EQ(std::string(error.what()), message);
}

} // namespace hipdnn_shim_test
