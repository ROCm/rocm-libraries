// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "test_moe_gemm1_xdl_common.hpp"

using ADataType   = ck::bhalf_t;
using BDataType   = ck::bhalf_t;
using EDataType   = ck::bhalf_t;
using AccDataType = float;

#include "test_moe_gemm1_xdl_ut_cases.inc"
