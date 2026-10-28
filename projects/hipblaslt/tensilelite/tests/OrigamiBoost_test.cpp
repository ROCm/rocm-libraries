// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Origami Boost in TensileLite: the Stream-K launch the hook passes the re-rank, the switch, and
// the dtype classes of logic files and problems.

#include <Tensile/OrigamiBoost.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>
#include <origami/nn/nn.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
    // Host-only gfx950 device (as StreamKLaunchSummary_test.cpp builds it): 256 CUs, 8 XCDs.
    TensileLite::hip::HipAMDGPU makeDevice()
    {
        using arch_t = origami::hardware_t::architecture_t;
        origami::hardware_t hw(arch_t::gfx950, 256, 163840, 262144, 8, 1.0, 1.0, 1.0, 4000000,
                               1.2, 1, std::make_tuple(0.0, 0.008, 0.0));
        TensileLite::hip::HipAMDGPU device;
        device.processor          = TensileLite::AMDGPU::Processor::gfx950;
        device.computeUnitCount   = 256;
        device.deviceName         = "test-gfx950";
        device.analyticalHardware = std::make_shared<origami::hardware_t>(hw);
        return device;
    }

    // A static (SK3) Stream-K solution, 128x128x64 macro tile.
    void makeStreamK(TensileLite::ContractionSolution& s, bool streamK = true)
    {
        auto& sm                  = s.sizeMapping;
        sm.tileProcessingStrategy = streamK ? TensileLite::TileProcessingStrategy::StreamK
                                            : TensileLite::TileProcessingStrategy::None;
        sm.workAssignment         = TensileLite::WorkAssignment::StaticGrid;
        sm.streamKAtomic          = 0;
        sm.macroTile              = TensileLite::dim3(128, 128, 1);
        sm.workGroupSize          = TensileLite::dim3(256, 1, 1);
        sm.threadTile             = TensileLite::dim3(1, 1, 1);
        sm.depthU                 = 64;
        sm.matrixInstruction      = {16, 16, 32, 1};
        sm.CUOccupancy            = 2;
        sm.workspaceSizePerElemC  = 4;
    }

    TensileLite::ContractionProblemGemm makeProblem(size_t m, size_t n, size_t k)
    {
        auto p = TensileLite::ContractionProblemGemm::GEMM(false, false, m, n, k, m, n, m, 1.0,
                                                            false, 1);
        p.setComputeInputTypeA(rocisa::DataType::Float);
        p.setComputeInputTypeB(rocisa::DataType::Float);
        p.setWorkspaceSize(std::numeric_limits<size_t>::max());
        return p;
    }

    void expectSameLaunch(origami::nn::boost::streamk_launch_t const&     sk,
                          TensileLite::ContractionSolution const&           s,
                          TensileLite::ContractionProblemGemm const&        p,
                          TensileLite::hip::HipAMDGPU const&                device)
    {
        const auto d = s.computeStreamKDecisions(p, device);
        EXPECT_TRUE(sk.valid);
        EXPECT_TRUE(sk.is_streamk);
        EXPECT_EQ(sk.final_grid, d.finalGrid);
        EXPECT_EQ(sk.tiles, d.tiles);
        EXPECT_EQ(sk.sk_tiles, d.skTiles);
        EXPECT_EQ(sk.sk_split, d.skSplit);
        EXPECT_EQ(sk.reduction, d.reduction);
        EXPECT_EQ(sk.is_dynamic, d.isDynamic);
        EXPECT_EQ(sk.dp_only, d.dpOnly);
        EXPECT_EQ(sk.iters_per_tile, std::max<size_t>(1, p.getItersPerTile(s.sizeMapping)));
        EXPECT_EQ(sk.mt_x, 128u);
        EXPECT_EQ(sk.mt_y, 128u);
        EXPECT_EQ(sk.workspace_size_per_elem_c, 4u);
        EXPECT_EQ(sk.cu_occupancy, 2);
        EXPECT_EQ(sk.cu_count, 256u);
    }
} // namespace

// The Stream-K launch the hook hands Origami Boost is computeStreamKDecisions on the call's own
// problem: it follows the problem's CU budget and workspace, as the launch does.
TEST(OrigamiBoost, StreamKLaunchIsTheDecision)
{
    namespace TL      = TensileLite;
    const auto device = makeDevice();
    TL::ContractionSolution s;
    makeStreamK(s);
    int withPartials = 0, budgetMoved = 0, fellBack = 0;
    for(auto [m, n, k] : std::vector<std::array<size_t, 3>>{{4096, 4224, 512},
                                                           {1024, 1024, 8192},
                                                           {512, 512, 16384},
                                                           {2048, 1024, 4096},
                                                           {1536, 1536, 2048},
                                                           {768, 768, 12288},
                                                           {3072, 3200, 1024},
                                                           {256, 4096, 30000}})
    {
        SCOPED_TRACE(std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k));
        auto       p    = makeProblem(m, n, k);
        const auto full = TL::OrigamiBoost::streamKLaunch(s, p, device);
        expectSameLaunch(full, s, p, device);
        const bool partials = origami::nn::boost::streamk_partials(full).partials > 0;
        withPartials += partials;

        // CU budget: smCountTarget reaches the grid selection.
        p.setParams().setSmCountTarget(224);
        const auto budget = TL::OrigamiBoost::streamKLaunch(s, p, device);
        expectSameLaunch(budget, s, p, device);
        budgetMoved += budget.final_grid != full.final_grid;

        // No workspace: no partials fit, so the launch is data parallel (Origami's grid
        // selection reads the workspace, and the workspace fallback backs it) -- what a launch
        // given less workspace than the heuristic saw would run instead.
        p.setParams().setSmCountTarget(0);
        p.setWorkspaceSize(0);
        const auto none = TL::OrigamiBoost::streamKLaunch(s, p, device);
        expectSameLaunch(none, s, p, device);
        EXPECT_EQ(origami::nn::boost::streamk_partials(none).partials, 0u);
        if(partials)
        {
            EXPECT_EQ(none.final_grid, none.tiles);
            ++fellBack;
        }
    }
    EXPECT_GT(withPartials, 0);
    EXPECT_GT(budgetMoved, 0);
    EXPECT_GT(fellBack, 0);
}

TEST(OrigamiBoost, StreamKLaunchOfOtherKernels)
{
    namespace TL      = TensileLite;
    const auto device = makeDevice();
    TL::ContractionSolution s;
    makeStreamK(s, false);
    const auto p  = makeProblem(4096, 4096, 4096);
    const auto sk = TL::OrigamiBoost::streamKLaunch(s, p, device);
    EXPECT_TRUE(sk.valid);
    EXPECT_FALSE(sk.is_streamk);
    EXPECT_EQ(sk.final_grid, 0u);
    EXPECT_EQ(sk.mt_x, 128u);

    // Not an AMDGPU: no launch to describe; a model that needs one falls back.
    struct OtherHardware : TL::Hardware
    {
        std::string description() const override
        {
            return "other";
        }
        size_t id() const override
        {
            return 0;
        }
        std::string archName() const override
        {
            return "other";
        }
    } other;
    makeStreamK(s);
    EXPECT_FALSE(TL::OrigamiBoost::streamKLaunch(s, p, other).valid);
}

// Off unless HIPBLASLT_ORIGAMI_BOOST=1: deserialization loads no model (nn_models.boost stays
// invalid_handle) and findTopSolutions asks for no re-rank, so stock Origami ranking is untouched.
TEST(OrigamiBoost, OffByDefault)
{
    const char* v = std::getenv("HIPBLASLT_ORIGAMI_BOOST");
    if(v != nullptr && std::string(v) == "1")
        GTEST_SKIP() << "HIPBLASLT_ORIGAMI_BOOST=1 is set";
    EXPECT_FALSE(TensileLite::OrigamiBoost::enabled());
}

// The dtype class of a logic file (its index stems) and of a problem (the call).
TEST(OrigamiBoost, LogicAndProblemClasses)
{
    namespace OB = TensileLite::OrigamiBoost;
    EXPECT_STREQ(
        OB::logicClass("TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_"
                       "Bljk_Cijk_Dijk_ID75a0_gfx950"),
        "bf16");
    EXPECT_STREQ(OB::logicClass("TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Alik_"
                                "Bljk_Cijk_Dijk_ID75a0_gfx950"),
                 "tf32");
    EXPECT_EQ(OB::logicClass("TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_"
                             "Alik_Bljk_Cijk_Dijk_ID75a0_gfx950"),
              nullptr);
    EXPECT_EQ(OB::logicClass("TensileLiteLibrary_lazy_gfx950_Mapping"), nullptr);

    EXPECT_STREQ(OB::logicClass("TensileLibrary_F8F8_BF8_HA_Bias_SAB_SAV_UA_Type_F8B_HPA_"
                                "Contraction_l_Ailk_Bjlk_Cijk_Dijk_ID75a0_gfx950"),
                 "fp8");

    auto p = makeProblem(512, 512, 512);
    EXPECT_EQ(OB::problemClass(p), nullptr); // strict f32
    p.setF32XdlMathOp(rocisa::DataType::XFloat32);
    EXPECT_STREQ(OB::problemClass(p), "tf32");

    // fp8 is plain F8BS: Float8 compute input, unset or scalar A/B scales, no alpha vector, MX
    // blocks or swizzled operands.
    using rocisa::DataType;
    auto f8 = [](DataType a, DataType b, DataType cd) {
        return TensileLite::ContractionProblemGemm::GEMM_Strides(
            false, false, a, b, cd, cd, 512, 512, 512, 1, 512, 512 * 512, 512, 512 * 512, 512,
            512 * 512, 512, 512 * 512, 0.0);
    };
    auto q = f8(DataType::Float8, DataType::Float8, DataType::BFloat16);
    EXPECT_STREQ(OB::problemClass(q), "fp8");
    q.setUseScaleAB("Scalar");
    EXPECT_STREQ(OB::problemClass(q), "fp8");
    q.setUseScaleAB("Vector");
    EXPECT_EQ(OB::problemClass(q), nullptr);
    q.setUseScaleAB("");
    q.setMXScaleA(DataType::E8, 32);
    q.setMXScaleB(DataType::E8, 32);
    EXPECT_EQ(OB::problemClass(q), nullptr);
    EXPECT_EQ(OB::problemClass(f8(DataType::Float8, DataType::BFloat8, DataType::BFloat16)),
              nullptr);
    auto bf16Math = f8(DataType::Float8, DataType::Float8, DataType::BFloat16);
    bf16Math.setComputeInputTypeA(DataType::BFloat16);
    bf16Math.setComputeInputTypeB(DataType::BFloat16);
    EXPECT_EQ(OB::problemClass(bf16Math), nullptr);
    auto alphaVec = f8(DataType::Float8, DataType::Float8, DataType::BFloat16);
    alphaVec.setUseScaleAlphaVec(1);
    EXPECT_EQ(OB::problemClass(alphaVec), nullptr);
    auto swizzled = f8(DataType::Float8, DataType::Float8, DataType::BFloat16);
    swizzled.setSwizzleTensorA(true);
    EXPECT_EQ(OB::problemClass(swizzled), nullptr);
    EXPECT_EQ(OB::problemClass(f8(DataType::Float8, DataType::Float8, DataType::Float)), nullptr);
}
