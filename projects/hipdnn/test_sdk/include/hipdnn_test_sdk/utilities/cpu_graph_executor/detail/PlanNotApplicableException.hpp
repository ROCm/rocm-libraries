// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdexcept>
#include <string>

namespace hipdnn_test_sdk::utilities::detail
{

class PlanNotApplicableException : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

}
