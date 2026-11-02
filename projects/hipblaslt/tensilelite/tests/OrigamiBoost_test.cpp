// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Origami Boost in TensileLite: the Stream-K launch the hook passes the re-rank, the switch, the
// dtype classes of logic files and problems, and the shipped models against the built library.

#include <Tensile/OrigamiBoost.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>
#include <origami/nn/boost/load_validate.hpp>
#include <origami/nn/nn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

namespace
{
    namespace fs = std::filesystem;

    // The library root (holding <arch>/) to check: HIPBLASLT_ORIGAMI_BOOST_TEST_LIBRARY, else
    // this build's, else the one installed with the tests (<prefix>/bin/hipblaslt-tensilelite/).
    fs::path builtLibraryRoot()
    {
        if(const char* env = std::getenv("HIPBLASLT_ORIGAMI_BOOST_TEST_LIBRARY"))
            return env;
        std::vector<fs::path> dirs{TENSILELITE_TEST_BUILT_LIBRARY_DIR};
#ifdef __linux__
        std::error_code ec;
        const auto      exe = fs::read_symlink("/proc/self/exe", ec);
        if(!ec)
            dirs.push_back(exe.parent_path().parent_path().parent_path()
                           / TENSILELITE_TEST_INSTALLED_LIBRARY_DIR);
#endif
        for(auto const& dir : dirs)
            if(fs::is_directory(dir))
                return dir;
        return {};
    }

    bool hasKernelLibrary(fs::path const& archDir, std::string const& arch)
    {
        std::error_code ec;
        for(fs::directory_iterator it(archDir, ec), end; !ec && it != end; it.increment(ec))
        {
            const std::string name = it->path().filename().string();
            if(name.rfind("TensileLibrary_", 0) == 0 && name.find(arch) != std::string::npos)
                return true;
        }
        return false;
    }

    std::vector<fs::path> modelsIn(fs::path const& dir)
    {
        std::vector<fs::path> out;
        std::error_code       ec;
        for(fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
            if(it->path().extension() == ".orr")
                out.push_back(it->path());
        std::sort(out.begin(), out.end());
        return out;
    }

    std::string fileBytes(fs::path const& p)
    {
        std::ifstream f(p, std::ios::binary);
        return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    }
} // namespace

// Every shipped Origami Boost model (shared/origami/data/nn/boost/<arch>/*.orr; installed tests:
// the copies next to the library) must load the way hipBLASLt loads it with
// HIPBLASLT_ORIGAMI_BOOST=1 (OrigamiBoost::loadForLogic): this build's Origami stamps, and the
// manifest diff against the kernel library this build made. A library change the model was not
// trained for (RETRAIN-REQUIRED) fails here. Each library-directory copy must be the shipped file.
TEST(OrigamiBoostLibrary, ShippedModelsAcceptBuiltLibrary)
{
    namespace ob       = origami::nn::boost;
    const fs::path lib = builtLibraryRoot();
    if(lib.empty())
        GTEST_SKIP() << "no built kernel library (set HIPBLASLT_ORIGAMI_BOOST_TEST_LIBRARY to a "
                        "library directory holding <arch>/)";
    // Deserializing the library's predicates asks the current device for its CU count.
    try
    {
        TensileLite::hip::GetCurrentDevice();
    }
    catch(std::exception const& e)
    {
        GTEST_SKIP() << "no HIP device to deserialize the kernel library with: " << e.what();
    }
    const fs::path shipped     = ORIGAMI_BOOST_SHIPPED_MODELS;
    const bool     fromSources = fs::is_directory(shipped);
    const fs::path modelRoot   = fromSources ? shipped : lib;

    int tested = 0;
    for(auto const& entry : fs::directory_iterator(modelRoot))
    {
        const std::string arch    = entry.path().filename().string();
        const fs::path    archDir = lib / arch;
        if(!entry.is_directory() || !hasKernelLibrary(archDir, arch))
            continue;
        for(auto const& model : modelsIn(entry.path()))
        {
            SCOPED_TRACE(model.string() + " against " + archDir.string());
            ob::status_t status;
            std::string  detail;
            auto         m = ob::ensemble_t::load_file(model.string(), status, detail);
            ASSERT_NE(m, nullptr) << ob::to_string(status) << ": " << detail;
            const std::string dtype = m->head_string("dtype");

            const auto                        t0 = std::chrono::steady_clock::now();
            std::vector<ob::manifest_entry_t> library;
            std::string                       error;
            if(!TensileLite::OrigamiBoost::libraryManifest(
                   archDir.string(), arch, dtype, library, error))
            {
                ADD_FAILURE() << error;
                continue;
            }
            const auto diff = ob::diff_library(*m, library);
            EXPECT_TRUE(diff.decision != ob::library_decision_t::retrain_required)
                << "the kernel library has " << dtype
                << " kernels whose structural values the model never saw in training: "
                << diff.summary()
                << ". hipBLASLt would refuse the model and keep stock Origami ranking. Retrain "
                   "the model on this library (OrigamiML, outside this repository; see "
                   "shared/origami/README.md, \"Shipped Origami Boost models\") or drop the "
                   "kernels from the PredictionMatching rows.";

            auto booster = ob::booster_t::create(std::move(m),
                                                 ob::deployment_t{arch, dtype, &library},
                                                 ob::build_identity(),
                                                 status,
                                                 detail);
            EXPECT_TRUE(booster != nullptr)
                << "refused (" << ob::to_string(status) << "): " << detail
                << " (origami-tests \"[shipped]\" says how to fix a stamp mismatch)";
            const double ms
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                      .count();
            std::cout << "[origami boost] " << model.filename().string() << " on " << archDir
                      << ": " << diff.summary() << " (" << ms << " ms)" << std::endl;

            if(fromSources)
            {
                const fs::path copy = archDir / model.filename();
                EXPECT_TRUE(fs::is_regular_file(copy) && fileBytes(copy) == fileBytes(model))
                    << copy << " is missing or not the shipped model: rebuild "
                    << "origami-nn-colocate-" << arch;
            }
            ++tested;
        }
    }
    if(tested == 0)
        GTEST_SKIP() << "no shipped model has a kernel library for its arch in " << lib;
}
