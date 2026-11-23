// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Origami Learned Correction (OLC) in TensileLite: the Stream-K launch OLC reads from Origami's
// context against the launch hipBLASLt takes, the switch, the dtype classes of logic files and
// problems, and the shipped models against the built library.

#include <Tensile/OrigamiOLC.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/gemm_intermediates.hpp>
#include <origami/hardware.hpp>
#include <origami/nn/olc/load_validate.hpp>
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

} // namespace

// OLC reads the Stream-K launch from Origami's context (gemm_intermediates_t::launch): with the
// inputs the prediction row passes (the kernel's per-element workspace and per-tile-extras
// capability, the call's CU budget and uniform summation order), it is the launch
// computeStreamKDecisions takes.
TEST(OrigamiOLC, ContextLaunchIsTheDecision)
{
    namespace TL      = TensileLite;
    const auto device = makeDevice();
    const auto& hw    = *device.analyticalHardware;
    TL::ContractionSolution s;
    makeStreamK(s);
    int withPartials = 0, compared = 0;
    for(int uso : {0, 1})
        for(int perTile : {0, 1})
            for(int budget : {0, 224})
                for(auto [m, n, k] : std::vector<std::array<size_t, 3>>{{4096, 4224, 512},
                                                                       {1024, 1024, 8192},
                                                                       {512, 512, 16384},
                                                                       {2048, 1024, 4096},
                                                                       {1536, 1536, 2048},
                                                                       {768, 768, 12288},
                                                                       {3072, 3200, 1024},
                                                                       {256, 4096, 30000}})
                {
                    SCOPED_TRACE(std::to_string(m) + "x" + std::to_string(n) + "x"
                                 + std::to_string(k) + " uso " + std::to_string(uso) + " perTile "
                                 + std::to_string(perTile) + " budget " + std::to_string(budget));
                    s.internalArgsSupport.perTileExtraIters = perTile != 0;
                    auto p = TL::ContractionProblemGemm::GEMM(
                        false, false, m, n, k, m, n, m, 1.0, false, 1);
                    p.setComputeInputTypeA(rocisa::DataType::Float);
                    p.setComputeInputTypeB(rocisa::DataType::Float);
                    p.setWorkspaceSize(std::numeric_limits<size_t>::max());
                    p.setParams().setUniformSummationOrder(uso);
                    p.setParams().setSmCountTarget(budget);
                    const auto d = s.computeStreamKDecisions(p, device);

                    origami::problem_t op;
                    op.size                    = {m, n, k};
                    op.num_cus                 = static_cast<size_t>(budget);
                    op.uniform_summation_order = uso != 0;
                    origami::config_t c;
                    c.mt                        = {128, 128, 64};
                    c.mi                        = {16, 16, 32};
                    c.occupancy                 = 2;
                    c.workspace_size            = std::numeric_limits<size_t>::max();
                    c.workspace_size_per_elem_c = 4;
                    c.stream_k                  = 3;
                    origami::tensile_params_t tp;
                    tp.stream_k_per_tile_extra_iters = perTile != 0;
                    c.backend                        = tp;
                    const origami::gemm::context_t ctx(op, hw, c);
                    const auto L = origami::gemm::compute_context_launch(op, hw, c, ctx);
                    EXPECT_EQ(L.final_grid, d.grid());
                    EXPECT_EQ(L.tiles, d.tiles);
                    EXPECT_EQ(L.reduction, d.reduction);
                    EXPECT_EQ(L.sk_tiles, d.skTiles());
                    EXPECT_EQ(L.sk_split, d.skSplit());
                    EXPECT_EQ(L.is_dynamic, d.isDynamic);
                    EXPECT_EQ(L.iters_per_tile,
                              std::max<size_t>(1, p.getItersPerTile(s.sizeMapping)));
                    withPartials += origami::streamk::launch_partials(L).partials > 0;
                    ++compared;
                }
    EXPECT_EQ(compared, 64);
    EXPECT_GT(withPartials, 0);
}

// Off unless HIPBLASLT_ORIGAMI_OLC=1: deserialization loads no model (nn_models.olc stays
// invalid_handle) and findTopSolutions asks for no re-rank, so stock Origami ranking is untouched.
TEST(OrigamiOLC, OffByDefault)
{
    const char* v = std::getenv("HIPBLASLT_ORIGAMI_OLC");
    if(v != nullptr && std::string(v) == "1")
        GTEST_SKIP() << "HIPBLASLT_ORIGAMI_OLC=1 is set";
    EXPECT_FALSE(TensileLite::OrigamiOLC::enabled());
}

// The dtype class of a logic file (its index stems) and of a problem (the call).
TEST(OrigamiOLC, LogicAndProblemClasses)
{
    namespace OLC = TensileLite::OrigamiOLC;
    EXPECT_STREQ(
        OLC::logicClass("TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_"
                       "Bljk_Cijk_Dijk_ID75a0_gfx950"),
        "bf16");
    EXPECT_STREQ(OLC::logicClass("TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Alik_"
                                "Bljk_Cijk_Dijk_ID75a0_gfx950"),
                 "tf32");
    EXPECT_EQ(OLC::logicClass("TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_"
                             "Alik_Bljk_Cijk_Dijk_ID75a0_gfx950"),
              nullptr);
    EXPECT_EQ(OLC::logicClass("TensileLiteLibrary_lazy_gfx950_Mapping"), nullptr);

    EXPECT_STREQ(OLC::logicClass("TensileLibrary_F8F8_BF8_HA_Bias_SAB_SAV_UA_Type_F8B_HPA_"
                                "Contraction_l_Ailk_Bjlk_Cijk_Dijk_ID75a0_gfx950"),
                 "fp8");

    auto p = makeProblem(512, 512, 512);
    EXPECT_EQ(OLC::problemClass(p), nullptr); // strict f32
    p.setF32XdlMathOp(rocisa::DataType::XFloat32);
    EXPECT_STREQ(OLC::problemClass(p), "tf32");

    // fp8 is plain F8BS: Float8 compute input, unset or scalar A/B scales, no alpha vector, MX
    // blocks or swizzled operands.
    using rocisa::DataType;
    auto f8 = [](DataType a, DataType b, DataType cd) {
        return TensileLite::ContractionProblemGemm::GEMM_Strides(
            false, false, a, b, cd, cd, 512, 512, 512, 1, 512, 512 * 512, 512, 512 * 512, 512,
            512 * 512, 512, 512 * 512, 0.0);
    };
    auto q = f8(DataType::Float8, DataType::Float8, DataType::BFloat16);
    EXPECT_STREQ(OLC::problemClass(q), "fp8");
    q.setUseScaleAB("Scalar");
    EXPECT_STREQ(OLC::problemClass(q), "fp8");
    q.setUseScaleAB("Vector");
    EXPECT_EQ(OLC::problemClass(q), nullptr);
    q.setUseScaleAB("");
    q.setMXScaleA(DataType::E8, 32);
    q.setMXScaleB(DataType::E8, 32);
    EXPECT_EQ(OLC::problemClass(q), nullptr);
    EXPECT_EQ(OLC::problemClass(f8(DataType::Float8, DataType::BFloat8, DataType::BFloat16)),
              nullptr);
    auto bf16Math = f8(DataType::Float8, DataType::Float8, DataType::BFloat16);
    bf16Math.setComputeInputTypeA(DataType::BFloat16);
    bf16Math.setComputeInputTypeB(DataType::BFloat16);
    EXPECT_EQ(OLC::problemClass(bf16Math), nullptr);
    auto alphaVec = f8(DataType::Float8, DataType::Float8, DataType::BFloat16);
    alphaVec.setUseScaleAlphaVec(1);
    EXPECT_EQ(OLC::problemClass(alphaVec), nullptr);
    auto swizzled = f8(DataType::Float8, DataType::Float8, DataType::BFloat16);
    swizzled.setSwizzleTensorA(true);
    EXPECT_EQ(OLC::problemClass(swizzled), nullptr);
    EXPECT_EQ(OLC::problemClass(f8(DataType::Float8, DataType::Float8, DataType::Float)), nullptr);
}

namespace
{
    namespace fs = std::filesystem;

    // The library root (holding <arch>/) to check: HIPBLASLT_ORIGAMI_OLC_TEST_LIBRARY, else
    // this build's, else the one installed with the tests (<prefix>/bin/hipblaslt-tensilelite/).
    fs::path builtLibraryRoot()
    {
        if(const char* env = std::getenv("HIPBLASLT_ORIGAMI_OLC_TEST_LIBRARY"))
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

// Every shipped OLC model (shared/origami/data/nn/olc/<arch>/*.orr; installed tests:
// the copies next to the library) must load the way hipBLASLt loads it with
// HIPBLASLT_ORIGAMI_OLC=1 (OrigamiOLC::loadForLogic): this build's Origami stamps, and the
// manifest diff against the kernel library this build made. A library change the model was not
// trained for (RETRAIN-REQUIRED) fails here. Each library-directory copy must be the shipped file.
TEST(OrigamiOLCLibrary, ShippedModelsAcceptBuiltLibrary)
{
    namespace olc      = origami::nn::olc;
    const fs::path lib = builtLibraryRoot();
    if(lib.empty())
        GTEST_SKIP() << "no built kernel library (set HIPBLASLT_ORIGAMI_OLC_TEST_LIBRARY to a "
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
    const fs::path shipped     = ORIGAMI_OLC_SHIPPED_MODELS;
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
            olc::status_t status;
            std::string  detail;
            auto         m = olc::ensemble_t::load_file(model.string(), status, detail);
            ASSERT_NE(m, nullptr) << olc::to_string(status) << ": " << detail;
            const std::string dtype = m->head_string("dtype");

            const auto                        t0 = std::chrono::steady_clock::now();
            std::vector<olc::manifest_entry_t> library;
            std::string                       error;
            if(!TensileLite::OrigamiOLC::libraryManifest(
                   archDir.string(), arch, dtype, library, error))
            {
                ADD_FAILURE() << error;
                continue;
            }
            const auto diff = olc::diff_library(*m, library);
            EXPECT_TRUE(diff.decision != olc::library_decision_t::retrain_required)
                << "the kernel library has " << dtype
                << " kernels whose structural values the model never saw in training: "
                << diff.summary()
                << ". hipBLASLt would refuse the model and keep stock Origami ranking. Retrain "
                   "the model on this library (OrigamiML, outside this repository; see "
                   "shared/origami/README.md, \"Shipped OLC models\") or drop the "
                   "kernels from the PredictionMatching rows.";

            auto corrector = olc::corrector_t::create(std::move(m),
                                                 olc::deployment_t{arch, dtype, &library},
                                                 olc::build_identity(),
                                                 status,
                                                 detail);
            EXPECT_TRUE(corrector != nullptr)
                << "refused (" << olc::to_string(status) << "): " << detail
                << " (origami-tests \"[shipped]\" says how to fix a stamp mismatch)";
            const double ms
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                      .count();
            std::cout << "[origami olc] " << model.filename().string() << " on " << archDir
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
