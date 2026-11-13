// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_standalone_dl_test.hpp"

using namespace dl_gemm_test;
using Half = ck::half_t;

int main()
{
    return run_on_native_device([](const auto& arch) {
        // K1=2 and transfer parameters match example/01_gemm/gemm_dl_fp16.cpp.
        bool pass = run_suite<Gemm<Half>, Half, Col, Row, 32>("FP16 Col/Row");
        if(arch == "gfx90c")
        {
            pass =
                expect_rejection<Gemm<Half, Half, Half, float, Row, Row>>("FP16 Row/Row") && pass;
            pass =
                expect_rejection<Gemm<Half, Half, Half, float, Row, Col>>("FP16 Row/Col") && pass;
            pass =
                expect_rejection<Gemm<Half, Half, Half, float, Col, Col>>("FP16 Col/Col") && pass;
            pass = expect_rejection<Gemm<Half, Half, Half, float, Col, Row, Col>>(
                       "FP16 column-major C") &&
                   pass;
            pass = expect_rejection<Gemm<float, Half, Half>>("FP32 A") && pass;
            pass = expect_rejection<Gemm<Half, float, Half>>("FP32 B") && pass;
            pass = expect_rejection<Gemm<Half, Half, float>>("FP32 C") && pass;
            pass = expect_rejection<Gemm<Half, Half, Half, Half>>("FP16 accumulator") && pass;
            pass = expect_rejection<Gemm<ck::bhalf_t>>("BF16 storage") && pass;
            pass = expect_rejection<Gemm<int8_t>>("INT8 storage") && pass;
        }
        return pass ? 0 : 1;
    });
}
