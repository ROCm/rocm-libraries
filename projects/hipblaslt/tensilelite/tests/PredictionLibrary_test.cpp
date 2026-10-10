// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Predicates.hpp>
#include <Tensile/PredictionLibrary.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>

using namespace TensileLite;

namespace
{
    using DataType       = rocisa::DataType;
    using AnalyticalType = origami::data_type_t;
    using Architecture   = origami::hardware_t::architecture_t;

    struct MappingCase
    {
        const char*    name;
        DataType       input;
        AnalyticalType expected;
    };

    class AnalyticalDatatypeTest : public ::testing::TestWithParam<MappingCase>
    {
    };

    TEST_P(AnalyticalDatatypeTest, MapsByMeaningNotEnumOrdinal)
    {
        const auto& test = GetParam();
        EXPECT_EQ(datatypeToAnalyticalDatatype(test.input), test.expected);
    }

    INSTANTIATE_TEST_SUITE_P(
        PredictionLibrary,
        AnalyticalDatatypeTest,
        ::testing::Values(
            MappingCase{"Float", DataType::Float, AnalyticalType::Float},
            MappingCase{"Double", DataType::Double, AnalyticalType::Double},
            MappingCase{"Half", DataType::Half, AnalyticalType::Half},
            MappingCase{"Int8x4", DataType::Int8x4, AnalyticalType::Int8x4},
            MappingCase{"Int32", DataType::Int32, AnalyticalType::Int32},
            MappingCase{"BFloat16", DataType::BFloat16, AnalyticalType::BFloat16},
            MappingCase{"Int8", DataType::Int8, AnalyticalType::Int8},
            MappingCase{"Int64", DataType::Int64, AnalyticalType::Int64},
            MappingCase{"XFloat32", DataType::XFloat32, AnalyticalType::XFloat32},
            MappingCase{"Float8_fnuz", DataType::Float8_fnuz, AnalyticalType::Float8_fnuz},
            MappingCase{"BFloat8_fnuz", DataType::BFloat8_fnuz, AnalyticalType::BFloat8_fnuz},
            MappingCase{"Float8BFloat8_fnuz",
                        DataType::Float8BFloat8_fnuz,
                        AnalyticalType::Float8BFloat8_fnuz},
            MappingCase{"BFloat8Float8_fnuz",
                        DataType::BFloat8Float8_fnuz,
                        AnalyticalType::BFloat8Float8_fnuz},
            MappingCase{"Float8", DataType::Float8, AnalyticalType::Float8},
            MappingCase{"BFloat8", DataType::BFloat8, AnalyticalType::BFloat8},
            MappingCase{"Float8BFloat8", DataType::Float8BFloat8, AnalyticalType::Float8BFloat8},
            MappingCase{"BFloat8Float8", DataType::BFloat8Float8, AnalyticalType::BFloat8Float8},
            MappingCase{"None", DataType::None, AnalyticalType::None},
            MappingCase{"ComplexFloat", DataType::ComplexFloat, AnalyticalType::None},
            MappingCase{"ComplexDouble", DataType::ComplexDouble, AnalyticalType::None}),
        [](const ::testing::TestParamInfo<MappingCase>& info) { return info.param.name; });

    struct PredictionCase
    {
        const char*    name;
        Architecture   architecture;
        DataType       a;
        DataType       b;
        DataType       computeInput;
        AnalyticalType expected;
        size_t         miK;
        bool           xf32 = false;
    };

    class PredictionDatatypeTest : public ::testing::TestWithParam<PredictionCase>
    {
    };

    TEST_P(PredictionDatatypeTest, UsesSupportedInstructionLatency)
    {
        const auto&      test = GetParam();
        constexpr size_t m = 256, n = 256, k = 256;
        const auto       output  = test.a == DataType::Int8 ? DataType::Int32 : DataType::Float;
        auto             problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            test.a,
                                                            test.b,
                                                            output,
                                                            output,
                                                            m,
                                                            n,
                                                            k,
                                                            1,
                                                            m,
                                                            m * k,
                                                            k,
                                                            k * n,
                                                            m,
                                                            m * n,
                                                            m,
                                                            m * n,
                                                            0.0);
        problem.setComputeInputType(test.computeInput);
        if(test.xf32)
            problem.setF32XdlMathOp(DataType::XFloat32);

        // Model different architectures on the host; no device initialization,
        // real hardware, kernel files, or GPU allocation is needed.
        hip::HipAMDGPU hardware;
        hardware.analyticalHardware
            = std::make_shared<origami::hardware_t>(test.architecture,
                                                    64,
                                                    65536,
                                                    1,
                                                    4.0,
                                                    2.0,
                                                    1.0,
                                                    8 * 1024 * 1024,
                                                    1.0,
                                                    1,
                                                    std::make_tuple(0.0, 0.17, 0.0));

        // Verify that the chosen instruction is genuinely supported for the
        // expected semantic type, before checking the production selection path.
        const auto& instructions = origami::hardware_t::INSTRUCTION_MAP.at(test.architecture);
        ASSERT_NE(instructions.find(origami::matrix_instruction(16, 16, test.miK, test.expected)),
                  instructions.end());

        ProblemPredictionLibrary<ContractionProblemGemm> library;
        const origami::tile_tuple tile{64, 64, test.miK, 16, 16, test.miK, 1, 1, 0, 0};
        auto                      solution = std::make_shared<ContractionSolution>();
        solution->hardwarePredicate        = std::make_shared<Predicates::True<Hardware>>();
        solution->problemPredicate = std::make_shared<Predicates::True<ContractionProblemGemm>>();
        library.solutionmap.emplace(0, solution);
        library.tile_list.push_back(tile);
        library.tile_map.emplace(tile, 0);

        // A raw cast can still return a solution while falling back to a wrong
        // instruction cost. Test the actual findTopSolutions -> Origami path,
        // not just the mapper (which was already correct before this fix).
        ::testing::internal::CaptureStderr();
        const auto selected    = library.findTopSolutions(problem, hardware, 1);
        const auto diagnostics = ::testing::internal::GetCapturedStderr();
        EXPECT_EQ(diagnostics.find("Latency not found"), std::string::npos) << diagnostics;
        ASSERT_EQ(selected.size(), 1);
        EXPECT_EQ(selected.front(), solution);
    }

    INSTANTIATE_TEST_SUITE_P(PredictionLibrary,
                             PredictionDatatypeTest,
                             ::testing::Values(PredictionCase{"Gfx1201Float8",
                                                              Architecture::gfx1201,
                                                              DataType::Float8,
                                                              DataType::Float8,
                                                              DataType::Float8,
                                                              AnalyticalType::Float8,
                                                              16},
                                               PredictionCase{"Gfx1201BFloat8",
                                                              Architecture::gfx1201,
                                                              DataType::BFloat8,
                                                              DataType::BFloat8,
                                                              DataType::BFloat8,
                                                              AnalyticalType::BFloat8,
                                                              16},
                                               PredictionCase{"Gfx1201Float8BFloat8",
                                                              Architecture::gfx1201,
                                                              DataType::Float8,
                                                              DataType::BFloat8,
                                                              DataType::Float8BFloat8,
                                                              AnalyticalType::Float8BFloat8,
                                                              16},
                                               PredictionCase{"Gfx1201BFloat8Float8",
                                                              Architecture::gfx1201,
                                                              DataType::BFloat8,
                                                              DataType::Float8,
                                                              DataType::BFloat8Float8,
                                                              AnalyticalType::BFloat8Float8,
                                                              16},
                                               PredictionCase{"Gfx950Float8",
                                                              Architecture::gfx950,
                                                              DataType::Float8,
                                                              DataType::Float8,
                                                              DataType::Float8,
                                                              AnalyticalType::Float8,
                                                              128},
                                               PredictionCase{"Gfx950BFloat8",
                                                              Architecture::gfx950,
                                                              DataType::BFloat8,
                                                              DataType::BFloat8,
                                                              DataType::BFloat8,
                                                              AnalyticalType::BFloat8,
                                                              128},
                                               PredictionCase{"Gfx950Float8BFloat8",
                                                              Architecture::gfx950,
                                                              DataType::Float8,
                                                              DataType::BFloat8,
                                                              DataType::Float8BFloat8,
                                                              AnalyticalType::Float8BFloat8,
                                                              128},
                                               PredictionCase{"Gfx950BFloat8Float8",
                                                              Architecture::gfx950,
                                                              DataType::BFloat8,
                                                              DataType::Float8,
                                                              DataType::BFloat8Float8,
                                                              AnalyticalType::BFloat8Float8,
                                                              128},
                                               PredictionCase{"Gfx942Float8Fnuz",
                                                              Architecture::gfx942,
                                                              DataType::Float8_fnuz,
                                                              DataType::Float8_fnuz,
                                                              DataType::Float8_fnuz,
                                                              AnalyticalType::Float8_fnuz,
                                                              32},
                                               PredictionCase{"Gfx942BFloat8Fnuz",
                                                              Architecture::gfx942,
                                                              DataType::BFloat8_fnuz,
                                                              DataType::BFloat8_fnuz,
                                                              DataType::BFloat8_fnuz,
                                                              AnalyticalType::BFloat8_fnuz,
                                                              32},
                                               PredictionCase{"Gfx942Float8BFloat8Fnuz",
                                                              Architecture::gfx942,
                                                              DataType::Float8_fnuz,
                                                              DataType::BFloat8_fnuz,
                                                              DataType::Float8BFloat8_fnuz,
                                                              AnalyticalType::Float8BFloat8_fnuz,
                                                              32},
                                               PredictionCase{"Gfx942BFloat8Float8Fnuz",
                                                              Architecture::gfx942,
                                                              DataType::BFloat8_fnuz,
                                                              DataType::Float8_fnuz,
                                                              DataType::BFloat8Float8_fnuz,
                                                              AnalyticalType::BFloat8Float8_fnuz,
                                                              32},
                                               PredictionCase{"Gfx942Float",
                                                              Architecture::gfx942,
                                                              DataType::Float,
                                                              DataType::Float,
                                                              DataType::Float,
                                                              AnalyticalType::Float,
                                                              4},
                                               PredictionCase{"Gfx1201Half",
                                                              Architecture::gfx1201,
                                                              DataType::Half,
                                                              DataType::Half,
                                                              DataType::Half,
                                                              AnalyticalType::Half,
                                                              16},
                                               PredictionCase{"Gfx1201BFloat16",
                                                              Architecture::gfx1201,
                                                              DataType::BFloat16,
                                                              DataType::BFloat16,
                                                              DataType::BFloat16,
                                                              AnalyticalType::BFloat16,
                                                              16},
                                               PredictionCase{"Gfx1201Int8",
                                                              Architecture::gfx1201,
                                                              DataType::Int8,
                                                              DataType::Int8,
                                                              DataType::Int8,
                                                              AnalyticalType::Int8,
                                                              16},
                                               PredictionCase{"Gfx942XFloat32Override",
                                                              Architecture::gfx942,
                                                              DataType::Float,
                                                              DataType::Float,
                                                              DataType::Float,
                                                              AnalyticalType::XFloat32,
                                                              32,
                                                              true}),
                             [](const ::testing::TestParamInfo<PredictionCase>& info) {
                                 return info.param.name;
                             });
}
