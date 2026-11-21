// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_standalone_dl_test.hpp"

using namespace dl_gemm_test;

template <typename ALayout>
using PaddedGemm = Gemm<float,
                        float,
                        float,
                        float,
                        ALayout,
                        Col,
                        Row,
                        2,
                        ck::tensor_operation::device::GemmSpecialization::MNPadding>;

int main()
{
    return run_on_native_device([](const auto& arch) {
        // Retain the original K1=1 example coverage alongside the factory K1=2 layouts.
        bool pass =
            run_suite<Gemm<float, float, float, float, Col, Row, Row, 1>, float, Col, Row, 16>(
                "FP32 Col/Row");
        pass = run_suite<Gemm<float, float, float, float, Row, Row>, float, Row, Row, 32>(
                   "FP32 Row/Row") &&
               pass;
        pass = run_suite<Gemm<float, float, float, float, Row, Col>, float, Row, Col, 32>(
                   "FP32 Row/Col") &&
               pass;
        pass = run_suite<Gemm<float, float, float, float, Col, Col>, float, Col, Col, 32>(
                   "FP32 Col/Col") &&
               pass;
        if(arch == "gfx90c")
        {
            // Padding removes block divisibility as a rejection reason. A four-wide
            // C store must still reject N=130, whose final vector would overwrite padding.
            pass = expect_rejection<PaddedGemm<Row>>(
                       "MNPadding Row/Col C vector alignment", 128, 130, 32, 32, 32, 136) &&
                   pass;
            pass = expect_rejection<PaddedGemm<Col>>(
                       "MNPadding Col/Col C vector alignment", 128, 130, 32, 128, 32, 136) &&
                   pass;
            // N=124 needs descriptor padding but ends at a complete C vector.
            // run_problem checks active output and the untouched C row padding.
            std::cout << "MNPadding Row/Col M=128 N=124 K=32 strides=32,32,128: " << std::flush;
            pass =
                run_problem<PaddedGemm<Row>, float, Row, Col>({128, 124, 32, 32, 32, 128}) && pass;
            std::cout << "MNPadding Col/Col M=128 N=124 K=32 strides=128,32,128: " << std::flush;
            pass =
                run_problem<PaddedGemm<Col>, float, Col, Col>({128, 124, 32, 128, 32, 128}) && pass;
            pass = expect_rejection<Gemm<float, float, float, float, Col, Row, Col>>(
                       "Column-major C") &&
                   pass;
            pass = expect_rejection<Gemm<ck::half_t, float, float>>("FP16 A") && pass;
            pass = expect_rejection<Gemm<float, ck::half_t, float>>("FP16 B") && pass;
            pass = expect_rejection<Gemm<float, float, ck::half_t>>("FP16 C") && pass;
            pass =
                expect_rejection<Gemm<float, float, float, ck::half_t>>("FP16 accumulator") && pass;
        }
        return pass ? 0 : 1;
    });
}
