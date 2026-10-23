// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Oracle test for Tensile/RocRollerStreamKGrid.hpp, the rocRoller-off port of
// the Stream-K grid solve.
//
// SOURCE OF TRUTH:
//   projects/hipblaslt/library/src/amd_detail/rocblaslt/src/rocroller/
//       runtime_args_selection.cpp  ::  chooseStreamKGridSize
//
// That function is what the rocRoller JIT path actually launches with, and it
// cannot be called from here: it needs a JIT-compiled GemmKernel. So this file
// carries a verbatim transcription of its struct-building and its two origami
// calls, parameterised on plain values instead of `gemm` / `prob`. The test
// then asserts that the port and the transcription build field-identical
// origami::problem_t and origami::config_t objects, and return the same grid.
//
// Struct equality is the real property under test: both sides go on to call the
// same origami functions, so the port's entire failure surface is the
// marshalling -- which fields are set, and which are deliberately left at
// origami's defaults.
//
// WHAT THIS TEST CANNOT DO: it proves the port matches the transcription below,
// not that the transcription still matches rocRoller. If
// runtime_args_selection.cpp changes its own marshalling, this test keeps
// passing while the rocRoller-off path silently diverges from the JIT path. The
// two must be re-diffed by hand whenever that file changes.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/RocRollerStreamKGrid.hpp>

#include <origami/hardware.hpp>
#include <origami/streamk.hpp>
#include <origami/types.hpp>

using namespace TensileLite;

namespace
{
    // ---------------------------------------------------------------------
    // The verbatim transcription.
    // ---------------------------------------------------------------------

    // The values chooseStreamKGridSize reads off `gemm` and `prob`.
    //
    // The real function derives a_dtype / b_dtype with
    // hipDataType_to_analytical_type() on the kernel's hipDataTypes; here they
    // arrive already converted, because the port reaches the same
    // origami::data_type_t enumerators through
    // TensileLite::datatypeToAnalyticalDatatype() (verified equal for every MX
    // type) and this test is about the marshalling, not the type mapping.
    struct RocRollerCallSite
    {
        size_t               m;
        size_t               n;
        size_t               k;
        size_t               batch_count;
        int                  sm_count_target;
        int                  workgroupTile_m;
        int                  workgroupTile_n;
        int                  workgroupTile_k;
        int                  occupancy;
        size_t               workspaceSize;
        origami::data_type_t typeA;
        origami::data_type_t typeB;
        size_t               elementSizeA_bits;
        size_t               elementSizeB_bits;
        size_t               elementSizeAcc; // bytes
    };

    // Transcribed line for line from chooseStreamKGridSize. Do not tidy it:
    // divergence from the original is the bug this file exists to catch.
    //
    // out_problem / out_config receive the two structs exactly as the original
    // builds them, before select_reduction overwrites reduction_strategy.
    int chooseStreamKGridSizeTranscription(RocRollerCallSite const&   call,
                                           origami::hardware_t const& analytical_hardware,
                                           origami::problem_t&        out_problem,
                                           origami::config_t&         out_config)
    {
        const origami::grid_selection_t DEFAULT_DYNAMIC_MODE
            = origami::grid_selection_t::k_split_aware;

        size_t elementSizeA_bits = call.elementSizeA_bits;
        size_t elementSizeB_bits = call.elementSizeB_bits;
        size_t elementSizeAcc    = call.elementSizeAcc;

        origami::problem_t origami_problem = {
            .size  = {call.m, call.n, call.k},
            .batch = call.batch_count,
            // 0 = use all CUs;
            .num_cus  = static_cast<size_t>(call.sm_count_target),
            .a_dtype  = call.typeA,
            .b_dtype  = call.typeB,
            .mi_dtype = elementSizeA_bits < elementSizeB_bits ? call.typeB : call.typeA,
        };
        origami::config_t origami_config = {
            .mt = {static_cast<size_t>(call.workgroupTile_m),
                   static_cast<size_t>(call.workgroupTile_n),
                   static_cast<size_t>(call.workgroupTile_k)},
            .occupancy                 = call.occupancy,
            .workspace_size            = call.workspaceSize,
            .workspace_size_per_elem_c = elementSizeAcc,
        };

        out_problem = origami_problem;
        out_config  = origami_config;

        auto reduction_type = origami::streamk::select_reduction(
            origami_problem, analytical_hardware, origami_config, DEFAULT_DYNAMIC_MODE);

        origami_config.reduction_strategy = reduction_type;

        // origami_problem.num_cus carries the caller's budget (sm_count_target);
        // select_grid_size derives the CU budget from it internally.
        auto result = origami::streamk::select_grid_size(
            origami_problem, analytical_hardware, origami_config, DEFAULT_DYNAMIC_MODE);

        return result;
    }

    // ---------------------------------------------------------------------
    // Harness
    // ---------------------------------------------------------------------

    // CPU-only: get_hardware_for_arch() derives everything from origami's
    // per-arch constant tables and never touches HIP, so this test needs no
    // device. (get_hardware_for_device(0), which chooseStreamKGridSize itself
    // uses, would.)
    origami::hardware_t makeGfx950Hardware()
    {
        return origami::hardware_t::get_hardware_for_arch(
            origami::hardware_t::architecture_t::gfx950,
            256, // N_CU
            163840, // LDS bytes
            262144, // RF bytes
            4 * 1024 * 1024, // L2 bytes
            2100000); // compute clock, kHz
    }

    origami::hardware_t makeGfx942Hardware()
    {
        return origami::hardware_t::get_hardware_for_arch(
            origami::hardware_t::architecture_t::gfx942,
            304,
            65536,
            262144,
            4 * 1024 * 1024,
            2100000);
    }

    // The port's inputs, built from the same call-site values the transcription
    // gets. This is the only place the two descriptions are tied together.
    RocRollerStreamKGridInputs portInputs(RocRollerCallSite const& call)
    {
        RocRollerStreamKGridInputs in;
        in.m                     = call.m;
        in.n                     = call.n;
        in.k                     = call.k;
        in.batch                 = call.batch_count;
        in.numCUs                = static_cast<size_t>(call.sm_count_target);
        in.tileM                 = static_cast<size_t>(call.workgroupTile_m);
        in.tileN                 = static_cast<size_t>(call.workgroupTile_n);
        in.tileK                 = static_cast<size_t>(call.workgroupTile_k);
        in.occupancy             = call.occupancy;
        in.workspaceSize         = call.workspaceSize;
        in.workspaceSizePerElemC = call.elementSizeAcc;
        in.aDtype                = call.typeA;
        in.bDtype                = call.typeB;
        in.aElementBits          = call.elementSizeA_bits;
        in.bElementBits          = call.elementSizeB_bits;
        return in;
    }

    std::string describe(RocRollerCallSite const& c)
    {
        return "m=" + std::to_string(c.m) + " n=" + std::to_string(c.n) + " k="
               + std::to_string(c.k) + " batch=" + std::to_string(c.batch_count) + " cus="
               + std::to_string(c.sm_count_target) + " mt=" + std::to_string(c.workgroupTile_m)
               + "x" + std::to_string(c.workgroupTile_n) + "x"
               + std::to_string(c.workgroupTile_k) + " occ=" + std::to_string(c.occupancy)
               + " a=" + std::to_string(static_cast<int>(c.typeA))
               + " b=" + std::to_string(static_cast<int>(c.typeB));
    }

    // Every field of problem_t, including the ones neither side sets: default
    // drift in origami is exactly as dangerous as a missing assignment.
    void expectProblemsEqual(origami::problem_t const& port,
                             origami::problem_t const& oracle,
                             std::string const&        what)
    {
        EXPECT_EQ(port.size.m, oracle.size.m) << what;
        EXPECT_EQ(port.size.n, oracle.size.n) << what;
        EXPECT_EQ(port.size.k, oracle.size.k) << what;
        EXPECT_EQ(port.batch, oracle.batch) << what;
        EXPECT_EQ(port.num_cus, oracle.num_cus) << what;
        EXPECT_EQ(port.q_heads, oracle.q_heads) << what;
        EXPECT_EQ(port.a_transpose, oracle.a_transpose) << what;
        EXPECT_EQ(port.b_transpose, oracle.b_transpose) << what;
        EXPECT_EQ(port.a_dtype, oracle.a_dtype) << what;
        EXPECT_EQ(port.b_dtype, oracle.b_dtype) << what;
        EXPECT_EQ(port.c_dtype, oracle.c_dtype) << what;
        EXPECT_EQ(port.d_dtype, oracle.d_dtype) << what;
        EXPECT_EQ(port.mi_dtype, oracle.mi_dtype) << what;
        EXPECT_EQ(port.a_mx_block_size, oracle.a_mx_block_size) << what;
        EXPECT_EQ(port.b_mx_block_size, oracle.b_mx_block_size) << what;
    }

    // config_t::operator== skips occupancy, workspace_size,
    // workspace_size_per_elem_c and index, so those are checked by hand --
    // three of the four are fields the port is responsible for setting.
    void expectConfigsEqual(origami::config_t const& port,
                            origami::config_t const& oracle,
                            std::string const&       what)
    {
        EXPECT_EQ(port.occupancy, oracle.occupancy) << what;
        EXPECT_EQ(port.workspace_size, oracle.workspace_size) << what;
        EXPECT_EQ(port.workspace_size_per_elem_c, oracle.workspace_size_per_elem_c) << what;
        EXPECT_EQ(port.index, oracle.index) << what;
        EXPECT_EQ(port.mt.m, oracle.mt.m) << what;
        EXPECT_EQ(port.mt.n, oracle.mt.n) << what;
        EXPECT_EQ(port.mt.k, oracle.mt.k) << what;
        EXPECT_EQ(port.mi.m, oracle.mi.m) << what;
        EXPECT_EQ(port.mi.n, oracle.mi.n) << what;
        EXPECT_EQ(port.mi.k, oracle.mi.k) << what;
        EXPECT_EQ(port.workgroup_mapping, oracle.workgroup_mapping) << what;
        EXPECT_EQ(port.stream_k, oracle.stream_k) << what;
        EXPECT_EQ(port.reduction_strategy, oracle.reduction_strategy) << what;
        EXPECT_EQ(port.grid_selection, oracle.grid_selection) << what;
        // Catches every remaining field in one shot (hand_optimized_main_loop,
        // subtile, cache hints, prediction_mode, target, vector widths,
        // cluster_dim, backend, ...).
        EXPECT_TRUE(port == oracle) << what;
    }

    // Both sides of the sweep for one call site.
    void checkOnePoint(RocRollerCallSite const&   call,
                       origami::hardware_t const& hw,
                       std::string const&         hwName)
    {
        std::string const what = hwName + " " + describe(call);

        origami::problem_t oracleProblem{};
        origami::config_t  oracleConfig{};
        int const oracleGrid = chooseStreamKGridSizeTranscription(
            call, hw, oracleProblem, oracleConfig);

        RocRollerStreamKGridInputs const in = portInputs(call);

        expectProblemsEqual(rocRollerStreamKGridProblem(in), oracleProblem, what);
        expectConfigsEqual(rocRollerStreamKGridConfig(in), oracleConfig, what);

        EXPECT_EQ(rocRollerStreamKGrid(in, hw), static_cast<uint32_t>(oracleGrid)) << what;
    }

    struct Tile
    {
        int m, n, k;
    };

    struct TypePair
    {
        origami::data_type_t a;
        origami::data_type_t b;
        size_t               aBits;
        size_t               bBits;
    };

    // FP8xFP8, FP4xFP4 and both mixed orderings; the mixed pairs are the only
    // ones that exercise the mi_dtype larger-element-bits rule.
    std::vector<TypePair> const& typePairs()
    {
        static std::vector<TypePair> const pairs = {
            {origami::data_type_t::Float8, origami::data_type_t::Float8, 8, 8},
            {origami::data_type_t::Float4, origami::data_type_t::Float4, 4, 4},
            {origami::data_type_t::Float8, origami::data_type_t::Float4, 8, 4},
            {origami::data_type_t::Float4, origami::data_type_t::Float8, 4, 8},
        };
        return pairs;
    }

    // Small / medium / large, with non-multiples of every tile in the sweep.
    std::vector<Tile> const& sizes()
    {
        static std::vector<Tile> const s = {
            {64, 64, 256}, // small, tile-aligned
            {131, 97, 513}, // small, aligned to nothing
            {1024, 1024, 4096}, // medium
            {1537, 769, 2049}, // medium, non-multiples
            {8192, 8192, 8192}, // large
            {8193, 6145, 16385}, // large, non-multiples
        };
        return s;
    }

    std::vector<Tile> const& macrotiles()
    {
        static std::vector<Tile> const t = {
            {16, 16, 128},
            {32, 32, 128},
            {128, 128, 128},
            {256, 192, 128}, // MT_N = 192 is not a power of two
        };
        return t;
    }

    // 0 = all CUs, then a masked partition, a typical gfx950 budget, and the
    // full gfx950 CU count.
    std::vector<int> const& cuBudgets()
    {
        static std::vector<int> const c = {0, 64, 224, 256};
        return c;
    }

    size_t sweep(origami::hardware_t const& hw, std::string const& hwName)
    {
        // Byte size of the accumulate type: Float, as every rocRoller MX kernel
        // in this path accumulates in fp32.
        constexpr size_t kAccBytes = 4;

        size_t points = 0;
        for(auto const& size : sizes())
        {
            for(auto const& mt : macrotiles())
            {
                for(auto const& tp : typePairs())
                {
                    for(int cus : cuBudgets())
                    {
                        for(size_t batch : {size_t{1}, size_t{7}})
                        {
                            for(int occ = 1; occ <= 20; ++occ)
                            {
                                RocRollerCallSite call{};
                                call.m                 = static_cast<size_t>(size.m);
                                call.n                 = static_cast<size_t>(size.n);
                                call.k                 = static_cast<size_t>(size.k);
                                call.batch_count       = batch;
                                call.sm_count_target   = cus;
                                call.workgroupTile_m   = mt.m;
                                call.workgroupTile_n   = mt.n;
                                call.workgroupTile_k   = mt.k;
                                call.occupancy         = occ;
                                // Exercise both the "no workspace" and the
                                // "plenty of workspace" branches of the
                                // reduction choice.
                                call.workspaceSize     = (occ % 2 == 0) ? 0 : (1u << 28);
                                call.typeA             = tp.a;
                                call.typeB             = tp.b;
                                call.elementSizeA_bits = tp.aBits;
                                call.elementSizeB_bits = tp.bBits;
                                call.elementSizeAcc    = kAccBytes;

                                checkOnePoint(call, hw, hwName);
                                ++points;

                                if(::testing::Test::HasFatalFailure())
                                    return points;
                            }
                        }
                    }
                }
            }
        }
        return points;
    }
} // namespace

TEST(RocRollerStreamKGrid, MatchesRocRollerTranscriptionOnGfx950)
{
    origami::hardware_t const hw = makeGfx950Hardware();
    size_t const              n  = sweep(hw, "gfx950");
    EXPECT_EQ(n, sizes().size() * macrotiles().size() * typePairs().size() * cuBudgets().size()
                     * 2u * 20u);
}

TEST(RocRollerStreamKGrid, MatchesRocRollerTranscriptionOnGfx942)
{
    origami::hardware_t const hw = makeGfx942Hardware();
    size_t const              n  = sweep(hw, "gfx942");
    EXPECT_EQ(n, sizes().size() * macrotiles().size() * typePairs().size() * cuBudgets().size()
                     * 2u * 20u);
}

// The mi_dtype rule, stated directly rather than inferred from the sweep:
// whichever of A/B has the larger element width wins, and a tie keeps A.
TEST(RocRollerStreamKGrid, MiDtypeTakesTheWiderOperand)
{
    RocRollerStreamKGridInputs in;
    in.aDtype = origami::data_type_t::Float8;
    in.bDtype = origami::data_type_t::Float4;

    in.aElementBits = 8;
    in.bElementBits = 4;
    EXPECT_EQ(rocRollerStreamKGridProblem(in).mi_dtype, origami::data_type_t::Float8);

    in.aDtype       = origami::data_type_t::Float4;
    in.bDtype       = origami::data_type_t::Float8;
    in.aElementBits = 4;
    in.bElementBits = 8;
    EXPECT_EQ(rocRollerStreamKGridProblem(in).mi_dtype, origami::data_type_t::Float8);

    in.aDtype       = origami::data_type_t::Float8;
    in.bDtype       = origami::data_type_t::BFloat8;
    in.aElementBits = 8;
    in.bElementBits = 8;
    EXPECT_EQ(rocRollerStreamKGridProblem(in).mi_dtype, origami::data_type_t::Float8);
}

// The fields chooseStreamKGridSize does NOT set, asserted against a
// default-constructed origami object. getPersistentGridImpl()'s adapter sets
// four of these; a copy-paste from there would fail here.
TEST(RocRollerStreamKGrid, LeavesUnsetFieldsAtOrigamiDefaults)
{
    RocRollerStreamKGridInputs in;
    in.m      = 4096;
    in.n      = 4096;
    in.k      = 8192;
    in.tileM  = 256;
    in.tileN  = 192;
    in.tileK  = 128;
    in.aDtype = origami::data_type_t::Float8;
    in.bDtype = origami::data_type_t::Float8;

    origami::problem_t const defaultProblem{};
    origami::problem_t const built = rocRollerStreamKGridProblem(in);
    EXPECT_EQ(built.a_transpose, defaultProblem.a_transpose);
    EXPECT_EQ(built.b_transpose, defaultProblem.b_transpose);
    EXPECT_EQ(built.c_dtype, defaultProblem.c_dtype);
    EXPECT_EQ(built.d_dtype, defaultProblem.d_dtype);

    origami::config_t const defaultConfig{};
    origami::config_t const builtConfig = rocRollerStreamKGridConfig(in);
    EXPECT_EQ(builtConfig.mi.m, defaultConfig.mi.m);
    EXPECT_EQ(builtConfig.mi.n, defaultConfig.mi.n);
    EXPECT_EQ(builtConfig.mi.k, defaultConfig.mi.k);
    EXPECT_EQ(builtConfig.workgroup_mapping, defaultConfig.workgroup_mapping);
    // mt.k is the kernel's tile K, not depthU.
    EXPECT_EQ(builtConfig.mt.k, in.tileK);
}

// ContractionSolution::rocRollerStreamKGrid clamps this solve's answer to
// StreamKFlagElements because the flag region holds exactly that many ints.
//
// These inputs overshoot that clamp, but note what they are: every one of them
// has far more tiles than the device has CUs, and the ranker only picks a
// Stream-K kernel when numTiles < usableCUs (chooseSolutionIndexParameters in
// rocroller/solution_selection.cpp). So none of them can reach a Stream-K
// kernel in practice -- swept over 76464 points that do satisfy that gate, the
// grid never exceeded 256 and the clamp never fired.
//
// They are pinned anyway because they fix what the clamp is for: it guards the
// unreachable region, and if the ranker's gate is ever loosened these become
// live, at which point truncating is the wrong answer and the solution should
// be refused at selection time instead.
TEST(RocRollerStreamKGrid, OvershootsTheStreamKFlagRegion)
{
    origami::hardware_t const hw = makeGfx950Hardware(); // 256 CUs

    RocRollerStreamKGridInputs in;
    in.m                     = 8192;
    in.n                     = 8192;
    in.k                     = 8192;
    in.batch                 = 1;
    in.tileK                 = 128;
    in.workspaceSize         = size_t{1} << 28;
    in.workspaceSizePerElemC = 4;
    in.aDtype                = origami::data_type_t::Float4;
    in.bDtype                = origami::data_type_t::Float4;
    in.aElementBits          = 4;
    in.bElementBits          = 4;

    // 1. The Stream-K answer is num_cus * occupancy, so 256 CUs at occupancy 16
    //    is already 4096.
    in.tileM     = 128;
    in.tileN     = 128;
    in.numCUs    = 256;
    in.occupancy = 16;
    EXPECT_GT(rocRollerStreamKGrid(in, hw), StreamKFlagElements);

    // 2. When k_split_aware picks data parallel it returns the whole tile
    //    count, which small macrotiles blow far past.
    in.tileM     = 16;
    in.tileN     = 16;
    in.numCUs    = 224;
    in.occupancy = 1;
    EXPECT_GT(rocRollerStreamKGrid(in, hw), StreamKFlagElements);

    // The configurations that do fit, for contrast: 256x192 on the same shape.
    in.tileM     = 256;
    in.tileN     = 192;
    in.numCUs    = 256;
    in.occupancy = 1;
    EXPECT_LE(rocRollerStreamKGrid(in, hw), StreamKFlagElements);
}

// ContractionSolution::rocRollerStreamKGrid derives aElementBits/bElementBits
// from rocisa::GetElementSize(). Pin the two MX widths the mi_dtype rule turns
// on, so that derivation cannot drift from the 8/4 the sweep above assumes.
TEST(RocRollerStreamKGrid, TensileLiteElementWidthsMatchRocRoller)
{
    auto bits = [](rocisa::DataType type) -> size_t {
        return static_cast<size_t>(rocisa::GetElementSize(type) * 8.0f + 0.5f);
    };
    EXPECT_EQ(bits(rocisa::DataType::Float8), 8u);
    EXPECT_EQ(bits(rocisa::DataType::BFloat8), 8u);
    EXPECT_EQ(bits(rocisa::DataType::Float4), 4u);
}
