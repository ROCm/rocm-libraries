// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fixture-free helpers shared by the cuDNN v8 compatibility layer tests.
#pragma once

#include <hipdnn_compatibility/cudnn/detail/v8/error.h>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_shim_test
{

using Dims = std::vector<std::int64_t>;

inline Dims packedStrides(const Dims& dims)
{
    return hipdnn_data_sdk::utilities::generateStrides(dims);
}

// The CompatibilityError `body` throws; a body that throws nothing fails the test.
template <typename Body>
hipdnn_frontend::compatibility::cudnn_frontend::detail::v8::CompatibilityError
    compatibilityErrorOf(Body&& body)
{
    try
    {
        std::forward<Body>(body)();
    }
    catch(
        const hipdnn_frontend::compatibility::cudnn_frontend::detail::v8::CompatibilityError& error)
    {
        return error;
    }
    ADD_FAILURE() << "no CompatibilityError thrown";
    return {CUDNN_STATUS_SUCCESS, ""};
}

template <typename Body>
void expectCompatibilityError(Body&& body, cudnnStatus_t status, const std::string& message)
{
    const auto error = compatibilityErrorOf(std::forward<Body>(body));
    EXPECT_EQ(error.status(), status);
    EXPECT_EQ(std::string(error.what()), message);
}

} // namespace hipdnn_shim_test
