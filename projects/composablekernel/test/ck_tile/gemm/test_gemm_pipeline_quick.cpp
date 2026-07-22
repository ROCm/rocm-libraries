// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineQuick : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineQuick<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

// Keep this list deliberately compact: it spans four pipeline implementations,
// all A/B row/column layout combinations, both schedulers, two input precisions,
// and persistent versus non-persistent execution without compiling the stock
// pipeline suites' full cross-products.
using KernelTypesQuick = ::testing::Types<
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I256, I256, I64, I32, I32, Intrawave, Mem>,
    std::tuple<Row, Col, Row, F8, F8, F32, F16, I256, I256, I64, I32, I32, Interwave, Mem>,
    std::tuple<Col, Row, Row, F16, F16, F32, F16, I256, I256, I64, I32, I32, Interwave, Mem>,
    std::tuple<Col, Col, Row, F8, F8, F32, F16, I256, I256, I64, I32, I32, Intrawave, Mem>,
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I256, I256, I64, I32, I32, Intrawave, CompV3>,
    std::tuple<Row, Col, Row, F8, F8, F32, F16, I256, I256, I64, I32, I32, Intrawave, CompV3>,
    std::tuple<Col, Row, Row, F16, F16, F32, F16, I256, I256, I64, I32, I32, Intrawave, CompV3>,
    std::tuple<Col, Col, Row, F8, F8, F32, F16, I256, I256, I64, I32, I32, Intrawave, CompV3>,
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I256, I256, I32, I32, I32, Intrawave, CompV4>,
    std::tuple<Col, Col, Row, F8, F8, F32, F16, I256, I256, I32, I32, I32, Intrawave, CompV4>,
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I256, I256, I64, I32, I32, Intrawave, CompV6>,
    std::tuple<Col, Row, Row, F8, F8, F32, F16, I256, I256, I64, I32, I32, Intrawave, CompV6>,
    std::tuple<Row,
               Col,
               Row,
               F16,
               F16,
               F32,
               F16,
               I256,
               I256,
               I64,
               I32,
               I32,
               Intrawave,
               CompV3,
               Persistent>,
    std::tuple<Row,
               Col,
               Row,
               F16,
               F16,
               F32,
               F16,
               I256,
               I256,
               I64,
               I32,
               I32,
               Intrawave,
               CompV3,
               NonPersistent>>;

#define TEST_SUITE_NAME TestCkTileGemmPipelineQuick

TYPED_TEST_SUITE(TestCkTileGemmPipelineQuick, KernelTypesQuick);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
