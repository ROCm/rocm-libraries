// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "test_moe_gemm1_xdl_common.hpp"

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using EDataType   = ck::half_t;
using AccDataType = float;

#include "test_moe_gemm1_xdl_ut_cases.inc"
