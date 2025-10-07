// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

// This file contains string constants for tensor names used in CPU graph executor tests.
namespace hipdnn_sdk_test_utils
{
const std::string X_TENSOR_NAME = "X";
const std::string Y_TENSOR_NAME = "Y";
const std::string SCALE_TENSOR_NAME = "Scale";
const std::string BIAS_TENSOR_NAME = "Bias";
const std::string MEAN_TENSOR_NAME = "Mean";
const std::string INV_VARIANCE_TENSOR_NAME = "InvVariance";
};
