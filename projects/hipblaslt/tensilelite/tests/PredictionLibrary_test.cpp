/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Host-only tests for ProblemPredictionLibrary deserialization: the Prediction
// node stores a table of solution indices, then MappingTraits copies each
// resolved ContractionSolution's SizeMapping into an aligned origami::config_t.
// These tests pin clusterDim x/y/z -> origami cluster_dim m/n/k, including the
// default {1,1,1} when SizeMapping never sets the field.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>

#if defined(TENSILELITE_HAS_TILEWRIGHT)
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/TilewrightRanker.hpp>
#endif

#include "FallbackTestUtils.hpp"

#if defined(TILEWRIGHT_TEST_MODEL_WRITER)
#include "model_writer.hpp"
#endif

#if defined(TENSILE_MSGPACK)
#include <Tensile/msgpack/MessagePack.hpp>
#include <msgpack.hpp>
#elif defined(TENSILE_YAML)
#include <Tensile/llvm/YAML.hpp>
#else
#error "PredictionLibrary_test requires TENSILE_MSGPACK or TENSILE_YAML"
#endif

using namespace TensileLite;

namespace
{
    std::shared_ptr<ContractionSolution> makeMappedSolution(int index)
    {
        auto solution            = std::make_shared<ContractionSolution>();
        solution->index          = index;
        solution->kernelName     = "cluster-dim-probe";
        solution->sizeMapping.macroTile         = TensileLite::dim3(256, 256, 1);
        solution->sizeMapping.depthU            = 64;
        solution->sizeMapping.matrixInstruction = {16, 16, 16, 1};
        solution->sizeMapping.CUOccupancy       = 4;
        solution->sizeMapping.workGroupMapping  = 1;
        return solution;
    }

    std::shared_ptr<ContractionSolution> makeMappedSolution(int index, TensileLite::dim3 clusterDim)
    {
        auto solution                    = makeMappedSolution(index);
        solution->sizeMapping.clusterDim = clusterDim;
        return solution;
    }

    void expectClusterDim(origami::config_t const& cfg, TensileLite::dim3 const& clusterDim)
    {
        // Tensile SizeMapping is {x,y,z}; Origami dim3_t is {m,n,k}.
        EXPECT_EQ(cfg.cluster_dim.m, clusterDim.x);
        EXPECT_EQ(cfg.cluster_dim.n, clusterDim.y);
        EXPECT_EQ(cfg.cluster_dim.k, clusterDim.z);
    }

    // Fills `lib` in place: ProblemPredictionLibrary is not copyable (std::atomic).
    void loadPredictionLibrary(std::vector<int> const&              table,
                               SolutionMap<ContractionSolution>&    solutions,
                               ContractionProblemPredictionLibrary& lib)
    {
        LibraryIOContext<ContractionSolution> ctx{"", {}, &solutions};

#if defined(TENSILE_MSGPACK)
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, std::map<std::string, std::vector<int>>{{"table", table}});
        auto handle = msgpack::unpack(buffer.data(), buffer.size());

        Serialization::MessagePackInput input(handle.get(), &ctx);
        input.input(lib);

        std::string errors;
        for(auto const& err : input.error)
        {
            if(!errors.empty())
                errors += "; ";
            errors += err;
        }
        EXPECT_TRUE(input.error.empty()) << errors;
#elif defined(TENSILE_YAML)
        std::ostringstream yaml;
        yaml << "table: [";
        for(size_t i = 0; i < table.size(); ++i)
        {
            if(i != 0)
                yaml << ", ";
            yaml << table[i];
        }
        yaml << "]\n";

        llvm::yaml::Input yin(llvm::StringRef(yaml.str()), &ctx);
        yin >> lib;
        EXPECT_FALSE(yin.error()) << yin.error().message();
#endif
    }

    hip::HipAMDGPU makeGfx950Device(int computeUnits = 256)
    {
        using arch_t = origami::hardware_t::architecture_t;
        hip::HipAMDGPU device;
        device.processor        = AMDGPU::Processor::gfx950;
        device.computeUnitCount = computeUnits;
        device.analyticalHardware
            = std::make_shared<origami::hardware_t>(arch_t::gfx950,
                                                    computeUnits,
                                                    163840,
                                                    262144,
                                                    8,
                                                    1.0,
                                                    1.0,
                                                    1.0,
                                                    4000000,
                                                    1.2,
                                                    1,
                                                    std::make_tuple(0.0, 0.008, 0.0));
        return device;
    }
}

TEST(PredictionLibraryTest, CopiesClusterDimIntoOrigamiConfig)
{
    auto solution = makeMappedSolution(42, TensileLite::dim3(2, 4, 1));

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.solution_list.size(), 1u);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);
    EXPECT_EQ(lib.solution_list[0].first, 42);
    EXPECT_EQ(lib.solution_list[0].second, solution);

    expectClusterDim(lib.origami_config_list[0], solution->sizeMapping.clusterDim);
    EXPECT_EQ(lib.origami_config_list[0].cluster_dim, (origami::dim3_t{2, 4, 1}));
}

TEST(PredictionLibraryTest, DefaultClusterDimWhenUnset)
{
    auto solution = makeMappedSolution(42);
    EXPECT_EQ(solution->sizeMapping.clusterDim.x, 1u);
    EXPECT_EQ(solution->sizeMapping.clusterDim.y, 1u);
    EXPECT_EQ(solution->sizeMapping.clusterDim.z, 1u);

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);
    expectClusterDim(lib.origami_config_list[0], TensileLite::dim3(1, 1, 1));
}

TEST(PredictionLibraryTest, ClusterDimAxesAreNotSwapped)
{
    auto solution = makeMappedSolution(42, TensileLite::dim3(2, 1, 1));

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);

    auto const& cfg = lib.origami_config_list[0];
    expectClusterDim(cfg, TensileLite::dim3(2, 1, 1));
    EXPECT_NE(cfg.cluster_dim, (origami::dim3_t{1, 2, 1}));
}

TEST(PredictionLibraryTest, ClusterDimStaysIndexAlignedWithSolutionList)
{
    auto clustered = makeMappedSolution(42, TensileLite::dim3(2, 4, 1));
    auto plain     = makeMappedSolution(7);

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, clustered);
    solutions.emplace(7, plain);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42, 7}, solutions, lib);
    ASSERT_EQ(lib.solution_list.size(), 2u);
    ASSERT_EQ(lib.origami_config_list.size(), 2u);

    EXPECT_EQ(lib.solution_list[0].first, 42);
    EXPECT_EQ(lib.solution_list[1].first, 7);
    expectClusterDim(lib.origami_config_list[0], TensileLite::dim3(2, 4, 1));
    expectClusterDim(lib.origami_config_list[1], TensileLite::dim3(1, 1, 1));
}

namespace
{
    using ProblemTerm = Predicates::Predicate<ContractionProblemGemm>;

    std::shared_ptr<ProblemTerm> allOf(std::vector<std::shared_ptr<ProblemTerm>> const& terms)
    {
        return std::make_shared<Predicates::And<ContractionProblemGemm>>(terms);
    }

    std::shared_ptr<ProblemTerm> floatTypes()
    {
        auto types   = std::make_shared<Predicates::Contraction::TypesEqual>();
        types->value = {rocisa::DataType::Float,
                        rocisa::DataType::Float,
                        rocisa::DataType::Float,
                        rocisa::DataType::Float,
                        rocisa::DataType::Float,
                        rocisa::DataType::Float};
        return types;
    }

    std::shared_ptr<ProblemTerm> accumulate(bool high)
    {
        return std::make_shared<Predicates::Contraction::HighPrecisionAccumulateEqual>(high);
    }

    std::shared_ptr<ProblemTerm> leadingM(size_t atLeast)
    {
        return std::make_shared<Predicates::Contraction::LeadingFree0SizesGreaterOrEqual>(atLeast);
    }

    std::vector<int> solutionIndicesOf(SolutionVector<ContractionSolution> const& solutions)
    {
        std::vector<int> indices;
        for(auto const& solution : solutions)
            indices.push_back(solution->index);
        return indices;
    }

    ContractionProblemGemm floatProblem(size_t m)
    {
        auto problem
            = ContractionProblemGemm::GEMM(false, false, m, 64, 64, m, 64, m, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }
}

TEST(PredictionLibraryTest, ARowPredicateLeavesTheSameSolutionsAsTheWholeProblemPredicate)
{
    auto large                  = makeMappedSolution(1);
    auto otherHpa               = makeMappedSolution(2);
    auto unchecked              = makeMappedSolution(3);
    auto alsoLarge              = makeMappedSolution(4);
    large->problemPredicate     = allOf({floatTypes(), accumulate(false), leadingM(128)});
    otherHpa->problemPredicate  = allOf({floatTypes(), accumulate(true)});
    alsoLarge->problemPredicate = allOf({leadingM(128), floatTypes()});

    SolutionMap<ContractionSolution> solutions;
    for(auto const& solution : {large, otherHpa, unchecked, alsoLarge})
        solutions.emplace(solution->index, solution);
    auto lib = std::make_shared<ContractionProblemPredictionLibrary>();
    loadPredictionLibrary({1, 2, 3, 4}, solutions, *lib);

    ContractionProblemSelectionLibrary rows;
    rows.rows.emplace_back(ContractionProblemPredicate(allOf({floatTypes(), accumulate(false)})),
                           lib);

    auto const device = makeGfx950Device();
    for(size_t m : {64, 256})
    {
        SCOPED_TRACE(m);
        auto const problem = floatProblem(m);
        ASSERT_FALSE(problem.highPrecisionAccumulate());
        auto const direct     = solutionIndicesOf(lib->findTopSolutions(problem, device, 4));
        auto const throughRow = solutionIndicesOf(rows.findTopSolutions(problem, device, 4));
        EXPECT_EQ(throughRow, direct);
        EXPECT_EQ(std::count(direct.begin(), direct.end(), 2), 0);
        EXPECT_EQ(std::count(direct.begin(), direct.end(), 1), m >= 128 ? 1 : 0);
        EXPECT_EQ(std::count(direct.begin(), direct.end(), 4), m >= 128 ? 1 : 0);
    }
}

namespace
{
    // True, and counts its evaluations.
    struct CountedTrue : public Predicates::Predicate_CRTP<CountedTrue, ContractionProblemGemm>
    {
        enum
        {
            HasIndex = false,
            HasValue = true
        };
        int value = 0;

        explicit CountedTrue(int value)
            : value(value)
        {
        }
        static std::string Type()
        {
            return "CountedTrue";
        }
        bool operator()(ContractionProblemGemm const&) const override
        {
            evaluations()++;
            return true;
        }
        static int& evaluations()
        {
            static int count = 0;
            return count;
        }
    };
}

TEST(PredictionLibraryTest, ARowPredicateTermIsNotEvaluatedAgainForTheSolution)
{
    auto solution              = makeMappedSolution(1);
    solution->problemPredicate = allOf({std::make_shared<CountedTrue>(7), floatTypes()});
    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(1, solution);
    auto lib = std::make_shared<ContractionProblemPredictionLibrary>();
    loadPredictionLibrary({1}, solutions, *lib);

    ContractionProblemSelectionLibrary rows;
    rows.rows.emplace_back(
        ContractionProblemPredicate(allOf({floatTypes(), std::make_shared<CountedTrue>(7)})), lib);

    auto const device  = makeGfx950Device();
    auto const problem = floatProblem(256);

    CountedTrue::evaluations() = 0;
    ASSERT_EQ(solutionIndicesOf(lib->findTopSolutions(problem, device, 1)), std::vector<int>{1});
    EXPECT_EQ(CountedTrue::evaluations(), 1);

    CountedTrue::evaluations() = 0;
    ASSERT_EQ(solutionIndicesOf(rows.findTopSolutions(problem, device, 1)), std::vector<int>{1});
    EXPECT_EQ(CountedTrue::evaluations(), 1) << "only the row's term is evaluated";
}

TEST(PredictionLibraryTest, ZeroRequestedSolutionsReturnsNone)
{
    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, makeMappedSolution(42));
    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    ASSERT_EQ(lib.findTopSolutions(problem, device, 1).size(), 1u);
    EXPECT_TRUE(lib.findTopSolutions(problem, device, 0).empty());
    EXPECT_FALSE(lib.lastFindTopAlreadyRetAll());
}

#if defined(TENSILELITE_HAS_TILEWRIGHT)
namespace
{
    class ScratchDir
    {
    public:
        explicit ScratchDir(std::string const& name)
            : m_path(
                  std::filesystem::temp_directory_path()
                  / (name + "_"
                     + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
        {
            std::filesystem::create_directories(m_path);
        }

        ~ScratchDir()
        {
            std::error_code ec;
            std::filesystem::remove_all(m_path, ec);
        }

        std::filesystem::path const& path() const
        {
            return m_path;
        }

    private:
        std::filesystem::path m_path;
    };

    // Sets TENSILE_STREAMK5_FORCE_MODE (unsets it for nullptr) for the
    // lifetime of the object.
    class ScopedStreamK5ForceMode
    {
    public:
        explicit ScopedStreamK5ForceMode(char const* value)
        {
            if(char const* previous = std::getenv("TENSILE_STREAMK5_FORCE_MODE"))
                m_previous = previous;
            if(value)
                setenv("TENSILE_STREAMK5_FORCE_MODE", value, 1);
            else
                unsetenv("TENSILE_STREAMK5_FORCE_MODE");
            Debug::Instance().reloadDebugBitsForTest();
        }

        ~ScopedStreamK5ForceMode()
        {
            if(m_previous)
                setenv("TENSILE_STREAMK5_FORCE_MODE", m_previous->c_str(), 1);
            else
                unsetenv("TENSILE_STREAMK5_FORCE_MODE");
            Debug::Instance().reloadDebugBitsForTest();
        }

    private:
        std::optional<std::string> m_previous;
    };

    std::optional<std::int64_t> findAttribute(tilewright::Config const& config,
                                              std::string const&        name)
    {
        for(auto const& attribute : config.attributes)
        {
            if(attribute.name == name)
                return attribute.value;
        }
        return std::nullopt;
    }

    // Every attribute source holds a value no other one holds, except the
    // booleans, whose A/B pairs differ.
    SizeMapping attributeProbe()
    {
        SizeMapping s{};
        s.waveNum                                = 101;
        s.workGroupSize                          = TensileLite::dim3(102, 103, 104);
        s.threadTile                             = TensileLite::dim3(105, 106, 107);
        s.gwvwC                                  = 108;
        s.staggerU                               = 109;
        s.staggerUMapping                        = 110;
        s.globalSplitUPGR                        = 111;
        s.globalSplitU                           = 112;
        s.staggerStrideShift                     = 113;
        s.workGroupMapping                       = 114;
        s.workGroupMappingXCC                    = 115;
        s.workGroupMappingXCCGroup               = 116;
        s.globalSplitUCoalesced                  = true;
        s.globalSplitUWorkGroupMappingRoundRobin = false;
        s.packBatchDims                          = 117;
        s.packSummationDims                      = 118;
        s.magicDivAlg                            = 119;
        s.streamKAtomic                          = 120;
        s.tileProcessingStrategy                 = TileProcessingStrategy::StreamK;
        s.workAssignment                         = WorkAssignment::DynamicWorkQueue;
        s.prefetchAcrossPersistent               = 121;
        s.persistentKernel                       = 122;
        s.persistentKernelAlongBatch             = true;
        s.sourceKernel                           = false;
        s.globalAccumulation                     = 123;
        s.adaptiveGemmGSUA                       = 124;
        s.activationFused                        = false;
        s.PrefetchGlobalRead                     = 125;
        s.MathClocksUnrolledLoop                 = 126;
        s.nonTemporalA                           = 127;
        s.nonTemporalB                           = 128;
        s.temporalHintA                          = 129;
        s.temporalHintB                          = 130;
        s.hasTemporalHint                        = true;
        s.adaptiveGemmNTAB                       = 131;
        s.customMainLoopScheduling               = 132;
        s.useSubtileImpl                         = true;
        s.SourceSwap                             = false;
        s.NonTemporalD                           = 133;
        s.WaveSeparateGlobalReadA                = 134;
        s.WaveSeparateGlobalReadB                = 135;
        s.UnrollLoopSwapGlobalReadOrder          = 136;
        s.DirectToVgprA                          = true;
        s.DirectToVgprB                          = false;
        s.NumLoadsCoalescedA                     = 137;
        s.NumLoadsCoalescedB                     = 138;
        s.waveGroup                              = {139, 140};
        s.VectorWidthA                           = 141;
        s.VectorWidthB                           = 142;
        s.LocalSplitU                            = 143;
        s.DirectToLdsA                           = false;
        s.DirectToLdsB                           = true;
        s.expertSchedulingMode                   = 144;
        s.clusterDim                             = TensileLite::dim3(145, 146, 147);
        return s;
    }
}

TEST(PredictionLibraryTest, TilewrightIsSilentlyUnusedWithoutAnIndex)
{
    ScratchDir dir("tensilelite_tilewright_no_index");

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, makeMappedSolution(42));
    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);

    ::testing::internal::CaptureStderr();
    lib.loadTilewright((dir.path() / "TensileLibrary_Probe_gfx950.dat").string());
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
    EXPECT_EQ(lib.tilewright_ranker, nullptr);
}

TEST(PredictionLibraryTest, TilewrightWarnsAndIsUnusedWhenTheIndexedModelCannotLoad)
{
    ScratchDir dir("tensilelite_tilewright_bad_model");
    std::ofstream(dir.path() / "tilewright_index")
        << "TensileLibrary_Probe_gfx950\tmissing.tilewright.bin\n";

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, makeMappedSolution(42));
    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);

    ::testing::internal::CaptureStderr();
    lib.loadTilewright((dir.path() / "TensileLibrary_Probe_gfx950.dat").string());
    std::string const warning = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(lib.tilewright_ranker, nullptr);
    EXPECT_NE(warning.find("TensileLibrary_Probe_gfx950"), std::string::npos) << warning;
    EXPECT_NE(warning.find("ranking with origami"), std::string::npos) << warning;
}

TEST(PredictionLibraryTest, TilewrightBlocklistIsReadNextToTheIndex)
{
    ScratchDir        dir("tensilelite_tilewright_blocklist");
    std::string const logic = (dir.path() / "TensileLibrary_Probe_gfx950.dat").string();
    EXPECT_EQ(TilewrightRanker::loadBlocklist(logic), nullptr);

    std::ofstream(dir.path() / "TensileLibrary_Probe_gfx950.tilewright.blocklist")
        << "# a comment\n\n   \nCijk_A\n  Cijk_B\t\r\n#Cijk_C\n";
    auto const blocklist = TilewrightRanker::loadBlocklist(logic);
    ASSERT_NE(blocklist, nullptr);
    EXPECT_EQ(*blocklist, (std::unordered_set<std::string>{"Cijk_A", "Cijk_B"}));
    EXPECT_EQ(
        TilewrightRanker::loadBlocklist((dir.path() / "TensileLibrary_Other_gfx950.dat").string()),
        nullptr);

    std::filesystem::create_directory(dir.path()
                                      / "TensileLibrary_Dir_gfx950.tilewright.blocklist");
    EXPECT_THROW(
        TilewrightRanker::loadBlocklist((dir.path() / "TensileLibrary_Dir_gfx950.dat").string()),
        std::runtime_error);
}

TEST(PredictionLibraryTest, TilewrightConfigKeepsTheCoreKernelFields)
{
    SizeMapping sizeMapping{};
    sizeMapping.macroTile         = TensileLite::dim3(256, 128, 1);
    sizeMapping.depthU            = 64;
    sizeMapping.matrixInstruction = {16, 16, 32, 1};
    sizeMapping.CUOccupancy       = 3;
    sizeMapping.nonTemporalA      = 4;
    sizeMapping.grvwA             = 8;
    sizeMapping.grvwB             = 4;
    sizeMapping.gwvwD             = 2;

    auto const config = TilewrightRanker::makeConfig(sizeMapping, 7);
    EXPECT_EQ(config.mt.m, 256u);
    EXPECT_EQ(config.mt.n, 128u);
    EXPECT_EQ(config.mt.k, 64u);
    EXPECT_EQ(config.mi.m, 16u);
    EXPECT_EQ(config.mi.n, 16u);
    EXPECT_EQ(config.mi.k, 32u);
    EXPECT_EQ(config.occupancy, 3);
    EXPECT_EQ(config.cache_hints_a, 4);
    EXPECT_EQ(config.cache_hints_b, 0);
    EXPECT_EQ(config.grvw_a, 8u);
    EXPECT_EQ(config.grvw_b, 4u);
    EXPECT_EQ(config.gwvw_d, 2u);
    EXPECT_EQ(config.index, 7u);

    sizeMapping.matrixInstruction = {0, 0, 0, 0};
    sizeMapping.CUOccupancy       = -1;
    sizeMapping.hasTemporalHint   = true;
    sizeMapping.temporalHintA     = 2;
    sizeMapping.temporalHintB     = 3;

    auto const dot2 = TilewrightRanker::makeConfig(sizeMapping, 0);
    EXPECT_EQ(dot2.mi.m, 1u);
    EXPECT_EQ(dot2.mi.n, 1u);
    EXPECT_EQ(dot2.mi.k, 64u);
    EXPECT_EQ(dot2.occupancy, 1);
    EXPECT_EQ(dot2.cache_hints_a, 0);
    EXPECT_EQ(dot2.cache_hints_b, 4);
}

TEST(PredictionLibraryTest, TilewrightConfigCarriesEveryKernelAttributeOnce)
{
    std::map<std::string, std::int64_t> const expected = {
        {"wave_num", 101},
        {"work_group_x", 102},
        {"work_group_y", 103},
        {"work_group_z", 104},
        {"thread_tile_x", 105},
        {"thread_tile_y", 106},
        {"thread_tile_z", 107},
        {"gwvw_c", 108},
        {"stagger_u", 109},
        {"stagger_u_mapping", 110},
        {"global_split_u_pgr", 111},
        {"global_split_u", 112},
        {"stagger_stride_shift", 113},
        {"workgroup_mapping", 114},
        {"workgroup_mapping_xcc", 115},
        {"workgroup_mapping_xcc_group", 116},
        {"global_split_u_coalesced", 1},
        {"global_split_u_wgm_round_robin", 0},
        {"pack_batch_dims", 117},
        {"pack_summation_dims", 118},
        {"magic_div_alg", 119},
        {"stream_k_atomic", 120},
        {"tile_processing_strategy", 2},
        {"work_assignment", 1},
        {"prefetch_across_persistent", 121},
        {"persistent_kernel", 122},
        {"persistent_kernel_along_batch", 1},
        {"source_kernel", 0},
        {"global_accumulation", 123},
        {"adaptive_gemm_gsua", 124},
        {"activation_fused", 0},
        {"prefetch_global_read", 125},
        {"math_clocks_unrolled_loop", 126},
        {"non_temporal_a", 127},
        {"non_temporal_b", 128},
        {"temporal_hint_a", 129},
        {"temporal_hint_b", 130},
        {"has_temporal_hint", 1},
        {"adaptive_gemm_ntab", 131},
        {"custom_main_loop_scheduling", 132},
        {"use_subtile_impl", 1},
        {"source_swap", 0},
        {"non_temporal_d", 133},
        {"wave_separate_global_read_a", 134},
        {"wave_separate_global_read_b", 135},
        {"unroll_loop_swap_global_read_order", 136},
        {"direct_to_vgpr_a", 1},
        {"direct_to_vgpr_b", 0},
        {"num_loads_coalesced_a", 137},
        {"num_loads_coalesced_b", 138},
        {"wave_group_0", 139},
        {"wave_group_1", 140},
        {"vector_width_a", 141},
        {"vector_width_b", 142},
        {"local_split_u", 143},
        {"direct_to_lds_a", 0},
        {"direct_to_lds_b", 1},
        {"expert_scheduling_mode", 144},
        {"cluster_dim_x", 145},
        {"cluster_dim_y", 146},
        {"cluster_dim_z", 147},
    };

    auto const config = TilewrightRanker::makeConfig(attributeProbe(), 0);

    std::map<std::string, std::vector<std::int64_t>> values;
    for(auto const& attribute : config.attributes)
        values[attribute.name].push_back(attribute.value);

    EXPECT_EQ(values.size(), expected.size());
    for(auto const& [name, value] : expected)
    {
        auto const found = values.find(name);
        if(found == values.end())
            ADD_FAILURE() << "no attribute " << name;
        else
            EXPECT_EQ(found->second, std::vector<std::int64_t>{value}) << name;
    }
}

TEST(PredictionLibraryTest, TilewrightAttributesCodeTheExecutionPolicy)
{
    SizeMapping sizeMapping{};
    for(auto const& [strategy, code] : std::vector<std::pair<TileProcessingStrategy, std::int64_t>>{
            {TileProcessingStrategy::None, 0},
            {TileProcessingStrategy::DataParallel, 1},
            {TileProcessingStrategy::StreamK, 2},
        })
    {
        sizeMapping.tileProcessingStrategy = strategy;
        EXPECT_EQ(
            findAttribute(TilewrightRanker::makeConfig(sizeMapping, 0), "tile_processing_strategy"),
            code)
            << toString(strategy);
    }
    for(auto const& [assignment, code] : std::vector<std::pair<WorkAssignment, std::int64_t>>{
            {WorkAssignment::StaticGrid, 0},
            {WorkAssignment::DynamicWorkQueue, 1},
            {WorkAssignment::Hybrid, 2},
        })
    {
        sizeMapping.workAssignment = assignment;
        EXPECT_EQ(findAttribute(TilewrightRanker::makeConfig(sizeMapping, 0), "work_assignment"),
                  code)
            << toString(assignment);
    }
}

TEST(PredictionLibraryTest, TilewrightAttributesOfAMinimalSolution)
{
    auto const config = TilewrightRanker::makeConfig(makeMappedSolution(42)->sizeMapping, 0);
    EXPECT_EQ(findAttribute(config, "workgroup_mapping"), 1);
    EXPECT_EQ(findAttribute(config, "magic_div_alg"), 1);
    EXPECT_EQ(findAttribute(config, "activation_fused"), 1);
    EXPECT_EQ(findAttribute(config, "prefetch_global_read"), 2);
    EXPECT_EQ(findAttribute(config, "local_split_u"), 1);
    EXPECT_EQ(findAttribute(config, "vector_width_a"), 1);
    EXPECT_EQ(findAttribute(config, "vector_width_b"), 1);
    EXPECT_EQ(findAttribute(config, "tile_processing_strategy"), 0);
    EXPECT_EQ(findAttribute(config, "work_assignment"), 0);
    EXPECT_EQ(findAttribute(config, "stream_k_atomic"), 0);
    EXPECT_EQ(findAttribute(config, "cluster_dim_x"), 1);
    EXPECT_EQ(findAttribute(config, "cluster_dim_y"), 1);
    EXPECT_EQ(findAttribute(config, "cluster_dim_z"), 1);
}

TEST(PredictionLibraryTest, TilewrightContextCarriesTheCuBudget)
{
    auto problem = TensileLite::testing::dummyProblem();
    EXPECT_EQ(TilewrightRanker::executionContext(problem).cu_budget, 0u);

    problem.setParams().setSmCountTarget(128);
    EXPECT_EQ(TilewrightRanker::executionContext(problem).cu_budget, 128u);
}

TEST(PredictionLibraryTest, TilewrightContextFollowsStreamKTileScheduling)
{
    ScopedStreamK5ForceMode respectApi(nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule, tilewright::Schedule::Default);

    problem.setParams().setStreamKTileSchedulingMode(1);
    EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule, tilewright::Schedule::Dynamic);

    problem.setParams().setStreamKTileSchedulingMode(2);
    EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule, tilewright::Schedule::Auto);
}

TEST(PredictionLibraryTest, TilewrightContextHonorsTheStreamK5ForceMode)
{
    auto problem = TensileLite::testing::dummyProblem();
    {
        ScopedStreamK5ForceMode forceDynamic("1");
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Dynamic);
        problem.setParams().setStreamKTileSchedulingMode(2);
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Dynamic);
    }
    {
        ScopedStreamK5ForceMode forceStatic("0");
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Default);
        problem.setParams().setStreamKTileSchedulingMode(1);
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Default);
    }
    {
        ScopedStreamK5ForceMode respectApi("-1");
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Dynamic);
    }
}

#if defined(TILEWRIGHT_TEST_WEIGHTS_DIR)
namespace
{
    std::filesystem::path shippedGfx950Model()
    {
        return std::filesystem::path(TILEWRIGHT_TEST_WEIGHTS_DIR) / "gfx950" / "gfx950"
               / "gfx950_Cijk_Ailk_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.tilewright.bin";
    }

    // Loads `table` and the model at `model` as the library of `stem`.
    void loadWithTilewright(ScratchDir const&                    dir,
                            std::filesystem::path const&         model,
                            std::vector<int> const&              table,
                            SolutionMap<ContractionSolution>&    solutions,
                            ContractionProblemPredictionLibrary& lib,
                            std::string const& stem = "TensileLibrary_Probe_gfx950")
    {
        std::filesystem::copy_file(model,
                                   dir.path() / "probe.tilewright.bin",
                                   std::filesystem::copy_options::overwrite_existing);
        std::ofstream(dir.path() / "tilewright_index") << stem << "\tprobe.tilewright.bin\n";
        loadPredictionLibrary(table, solutions, lib);
        lib.loadTilewright((dir.path() / (stem + ".dat")).string());
    }

    std::shared_ptr<ContractionSolution> makeDot2Solution(int index)
    {
        auto solution                           = makeMappedSolution(index);
        solution->sizeMapping.matrixInstruction = {0, 0, 0, 0};
        return solution;
    }

    std::vector<int> const contestedTable = {13, 11, 12};

    // The shipped model and origami order the blocky and the skinny tile
    // differently at 1024^3; neither ranks the Dot2 kernel there.
    void addContestedKernels(SolutionMap<ContractionSolution>& solutions)
    {
        auto blocky                   = makeMappedSolution(11);
        blocky->sizeMapping.macroTile = TensileLite::dim3(128, 64, 1);
        auto skinny                   = makeMappedSolution(12);
        skinny->sizeMapping.macroTile = TensileLite::dim3(16, 256, 1);

        solutions.emplace(11, blocky);
        solutions.emplace(12, skinny);
        solutions.emplace(13, makeDot2Solution(13));
    }

    std::vector<int> solutionIndices(SolutionVector<ContractionSolution> const& solutions)
    {
        std::vector<int> indices;
        for(auto const& solution : solutions)
            indices.push_back(solution->index);
        return indices;
    }

    // The kernels the model scores for dummyProblem() on the whole device, best first.
    std::vector<int> tilewrightOrder(ContractionProblemPredictionLibrary const& lib)
    {
        tilewright::Problem const problem{
            .size     = {1024, 1024, 1024},
            .batch    = 1,
            .a_dtype  = tilewright::DataType::Float,
            .b_dtype  = tilewright::DataType::Float,
            .c_dtype  = tilewright::DataType::Float,
            .d_dtype  = tilewright::DataType::Float,
            .mi_dtype = tilewright::DataType::Float,
        };
        std::vector<int> order;
        for(auto const& r :
            lib.tilewright_ranker->candidates().rank(problem, {256, 163840, 4000000}, 3))
        {
            if(r.scored)
                order.push_back(lib.solution_list[r.config_index].first);
        }
        return order;
    }

    // What the contested library returns for `problem` without a model.
    std::vector<int> origamiOrder(ContractionProblemGemm const& problem)
    {
        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);
        ContractionProblemPredictionLibrary lib;
        loadPredictionLibrary(contestedTable, solutions, lib);
        return solutionIndices(lib.findTopSolutions(problem, makeGfx950Device(), 3));
    }

    void expectOrigamiOrder(ContractionProblemPredictionLibrary const& lib,
                            ContractionProblemGemm const&              problem)
    {
        auto const device = makeGfx950Device();
        EXPECT_TRUE(lib.tilewright_ranker->rank(problem, device, 3).empty());

        auto const origami = origamiOrder(problem);
        EXPECT_NE(origami, tilewrightOrder(lib));
        EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), origami);
    }

    void expectTilewrightOrder(ContractionProblemPredictionLibrary const& lib,
                               ContractionProblemGemm const&              problem)
    {
        auto const device = makeGfx950Device();
        EXPECT_NE(origamiOrder(problem), tilewrightOrder(lib));
        EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), tilewrightOrder(lib));
    }
}

TEST(PredictionLibraryTest, TilewrightOrdersTheKernelsItScores)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_rank");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const expected = tilewrightOrder(lib);
    ASSERT_EQ(expected.size(), 2u);

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    EXPECT_NE(origamiOrder(problem), expected);
    EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), expected);
    EXPECT_TRUE(lib.lastFindTopAlreadyRetAll());

    auto const best = lib.findTopSolutions(problem, device, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0]->index, expected[0]);
    EXPECT_FALSE(lib.lastFindTopAlreadyRetAll());
}

TEST(PredictionLibraryTest, TilewrightPoolPositionsAreSolutionListPositions)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_pool");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const& configs = lib.tilewright_ranker->candidates().configs();
    auto const  names   = tilewright::attribute_names(lib.tilewright_ranker->candidates().model());
    ASSERT_EQ(configs.size(), lib.solution_list.size());
    for(size_t i = 0; i < configs.size(); ++i)
    {
        auto const expected
            = TilewrightRanker::makeConfig(lib.solution_list[i].second->sizeMapping, i, &names);
        EXPECT_EQ(configs[i].index, i);
        EXPECT_EQ(configs[i].mt.m, expected.mt.m) << i;
        EXPECT_EQ(configs[i].mt.n, expected.mt.n) << i;
        EXPECT_EQ(configs[i].mi.m, expected.mi.m) << i;
        ASSERT_EQ(configs[i].attributes.size(), expected.attributes.size()) << i;
        for(size_t a = 0; a < expected.attributes.size(); ++a)
        {
            EXPECT_EQ(configs[i].attributes[a].name, expected.attributes[a].name) << i;
            EXPECT_EQ(configs[i].attributes[a].value, expected.attributes[a].value) << i;
        }
    }
}

TEST(PredictionLibraryTest, OrigamiRanksWhenTilewrightScoresNothing)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(13, makeDot2Solution(13));

    ScratchDir                          dir("tensilelite_tilewright_fallback");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), {13}, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const top
        = lib.findTopSolutions(TensileLite::testing::dummyProblem(), makeGfx950Device(), 1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0]->index, 13);
}

TEST(PredictionLibraryTest, OrigamiRanksCallsWithACuBudgetBelowTheDevice)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_cu_budget");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    problem.setParams().setSmCountTarget(224);
    expectOrigamiOrder(lib, problem);
}

TEST(PredictionLibraryTest, TilewrightNamesTheCompilerTargetOfTheLibrary)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    for(auto const& [suffix, arch] :
        std::vector<std::pair<std::string, std::string>>{{"gfx950", "gfx950"},
                                                         {"CU256_ID75a8_gfx950", "gfx950"},
                                                         {"gfx1250", "gfx1250"},
                                                         {"gfx1250-strict", "gfx1250"},
                                                         {"gfx1250v0", "gfx1250"},
                                                         {"gfx942:xnack+", "gfx942"}})
    {
        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);

        ScratchDir                          dir("tensilelite_tilewright_arch");
        ContractionProblemPredictionLibrary lib;
        loadWithTilewright(dir,
                           shippedGfx950Model(),
                           contestedTable,
                           solutions,
                           lib,
                           "TensileLibrary_Probe_" + suffix);
        ASSERT_NE(lib.tilewright_ranker, nullptr) << suffix;
        EXPECT_EQ(lib.tilewright_ranker->arch(), arch) << suffix;
    }
}

TEST(PredictionLibraryTest, TilewrightNotesEachContextTheModelDoesNotSupportOnce)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_unsupported");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    tilewright::Hardware const hardware{256, 163840, 4000000};
    auto const&                ranker = *lib.tilewright_ranker;
    EXPECT_FALSE(ranker.unsupportedContextNote({}, hardware));
    EXPECT_FALSE(ranker.unsupportedContextNote({.cu_budget = 256}, hardware));

    auto const note = ranker.unsupportedContextNote({.cu_budget = 224}, hardware);
    ASSERT_TRUE(note);
    EXPECT_EQ(*note,
              "[TILEWRIGHT_DIAG TENSILELITE] arch=gfx950 cu_budget=224 schedule=0 n_cu=256: the "
              "model does not support this context; ranking with origami");
    EXPECT_FALSE(ranker.unsupportedContextNote({.cu_budget = 224}, hardware));
    EXPECT_TRUE(ranker.unsupportedContextNote(
        {.cu_budget = 224, .schedule = tilewright::Schedule::Dynamic}, hardware));
}

TEST(PredictionLibraryTest, TilewrightRanksTheTopKernelsOrEveryScoredKernel)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_every_scored");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    auto const top     = lib.tilewright_ranker->rank(problem, device, 1);
    auto const every   = lib.tilewright_ranker->rank(problem, device, 1, true);
    ASSERT_EQ(top.size(), 1u);
    ASSERT_EQ(every.size(), tilewrightOrder(lib).size());
    EXPECT_EQ(every.front(), top.front());
    EXPECT_EQ(lib.tilewright_ranker->rank(problem, device, 3), every);
}

TEST(PredictionLibraryTest, TilewrightRanksCallsWithACuBudgetOfTheWholeDevice)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_full_budget");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    problem.setParams().setSmCountTarget(256);
    expectTilewrightOrder(lib, problem);
}

TEST(PredictionLibraryTest, OrigamiRanksCallsWithDynamicOrAutoStreamKScheduling)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    ScopedStreamK5ForceMode respectApi(nullptr);

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_streamk_mode");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    for(int mode : {1, 2})
    {
        SCOPED_TRACE("streamKTileSchedulingMode " + std::to_string(mode));
        auto problem = TensileLite::testing::dummyProblem();
        problem.setParams().setStreamKTileSchedulingMode(mode);
        expectOrigamiOrder(lib, problem);
    }
}

TEST(PredictionLibraryTest, StreamK5ForceModeDecidesBetweenTilewrightAndOrigami)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_force_mode");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    {
        ScopedStreamK5ForceMode forceDynamic("1");
        expectOrigamiOrder(lib, problem);
    }
    problem.setParams().setStreamKTileSchedulingMode(1);
    {
        ScopedStreamK5ForceMode forceStatic("0");
        expectTilewrightOrder(lib, problem);
    }
}

namespace
{
    // Sets TENSILE_TILEWRIGHT_BANK (unsets it for nullptr) for the lifetime of
    // the object.
    class ScopedTilewrightBank
    {
    public:
        explicit ScopedTilewrightBank(char const* value)
        {
            if(char const* previous = std::getenv("TENSILE_TILEWRIGHT_BANK"))
                m_previous = previous;
            if(value)
                setenv("TENSILE_TILEWRIGHT_BANK", value, 1);
            else
                unsetenv("TENSILE_TILEWRIGHT_BANK");
            Debug::Instance().reloadDebugBitsForTest();
        }

        ~ScopedTilewrightBank()
        {
            if(m_previous)
                setenv("TENSILE_TILEWRIGHT_BANK", m_previous->c_str(), 1);
            else
                unsetenv("TENSILE_TILEWRIGHT_BANK");
            Debug::Instance().reloadDebugBitsForTest();
        }

    private:
        std::optional<std::string> m_previous;
    };

    // Returns its solutions from findTopSolutions, like an Equality row that
    // matches the problem exactly.
    class ExactRow : public ContractionLibrary
    {
    public:
        explicit ExactRow(SolutionVector<ContractionSolution> solutions)
            : m_solutions(std::move(solutions))
        {
        }

        std::shared_ptr<ContractionSolution>
            getSolutionByIndex(ContractionProblemGemm const&, Hardware const&, int) const override
        {
            return {};
        }

        std::shared_ptr<ContractionSolution> findBestSolution(ContractionProblemGemm const&,
                                                              Hardware const&,
                                                              double* = nullptr) const override
        {
            return m_solutions.empty() ? nullptr : m_solutions.front();
        }

        SolutionSet<ContractionSolution>
            findAllSolutions(ContractionProblemGemm const&,
                             Hardware const&,
                             SolutionLibrarySearchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            return SolutionSet<ContractionSolution>(m_solutions.begin(), m_solutions.end());
        }

        SolutionSet<ContractionSolution>
            findAllSolutionsGroupedGemm(std::vector<ContractionProblemGemm> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return {};
        }

        std::string type() const override
        {
            return "ExactRow";
        }

        std::string description() const override
        {
            return "ExactRow";
        }

        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                             Hardware const&,
                                                             int count) const override
        {
            return SolutionVector<ContractionSolution>(
                m_solutions.begin(),
                m_solutions.begin() + std::min<size_t>(m_solutions.size(), std::max(count, 0)));
        }

    private:
        SolutionVector<ContractionSolution> m_solutions;
    };

    using Bank = TilewrightBank<ContractionSolution>;

    std::shared_ptr<ContractionSolution> withTile(int index, size_t m, size_t n)
    {
        auto solution                   = makeMappedSolution(index);
        solution->sizeMapping.macroTile = TensileLite::dim3(m, n, 1);
        return solution;
    }

    // Runs only on devices with `computeUnits` CUs.
    std::shared_ptr<ContractionSolution> onlyFor(std::shared_ptr<ContractionSolution> solution,
                                                 int                                  computeUnits)
    {
        solution->hardwarePredicate = std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(
            std::make_shared<Predicates::GPU::CUCountEqual>(computeUnits));
        return solution;
    }

    // Each solution as a bank source of its own, the way Equality rows hold
    // their solutions in leaves.
    void setBankSources(ContractionProblemPredictionLibrary&                     lib,
                        std::vector<std::shared_ptr<ContractionSolution>> const& bank)
    {
        std::vector<std::shared_ptr<ContractionLibrary>> sources;
        for(auto const& solution : bank)
            sources.push_back(std::make_shared<SingleContractionLibrary>(solution));
        lib.tilewright_bank_sources = [sources]() { return sources; };
    }

    // The bank pool built for `device`; fails when there is none yet.
    Bank::Pool const& builtPool(ContractionProblemPredictionLibrary const& lib,
                                AMDGPU const&                              device)
    {
        return lib.tilewright_bank->pool(device, []() {
            ADD_FAILURE() << "no bank pool for the device yet";
            return Bank::Pool{};
        });
    }

    std::vector<int> bankIndices(Bank::Pool const& pool)
    {
        std::vector<int> indices;
        for(auto const& solution : pool.bank)
            indices.push_back(solution->index);
        return indices;
    }
}

TEST(PredictionLibraryTest, TilewrightBankFollowsTheSettingAndTheModel)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    for(auto [setting, bank] :
        std::vector<std::pair<char const*, bool>>{{nullptr, false}, {"0", false}, {"1", true}})
    {
        SCOPED_TRACE(setting ? setting : "unset");
        ScopedTilewrightBank scoped(setting);

        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);
        ScratchDir                          dir("tensilelite_tilewright_bank_setting");
        ContractionProblemPredictionLibrary lib;
        loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
        ASSERT_NE(lib.tilewright_ranker, nullptr);
        // The shipped models were not trained on the bank and read no kernel
        // attributes.
        EXPECT_EQ(lib.tilewright_bank != nullptr, bank);
    }
}

TEST(PredictionLibraryTest, TilewrightBankFollowsThePoolInSolutionIndexOrder)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();
    ScopedTilewrightBank bankOn("1");

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);
    ScratchDir                          dir("tensilelite_tilewright_bank_order");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_bank, nullptr);

    auto const bank30 = withTile(30, 256, 128);
    auto const bank21 = withTile(21, 64, 64);
    setBankSources(
        lib, {bank30, bank21, solutions.at(11), onlyFor(withTile(25, 128, 128), 128), bank21});

    auto const device = makeGfx950Device();
    lib.findTopSolutions(TensileLite::testing::dummyProblem(), device, 1);
    auto const& pool = builtPool(lib, device);
    EXPECT_EQ(bankIndices(pool), (std::vector<int>{21, 30}));
    ASSERT_NE(pool.ranker, nullptr);

    auto const& configs = pool.ranker->candidates().configs();
    ASSERT_EQ(configs.size(), lib.solution_list.size() + 2);
    for(size_t i = 0; i < configs.size(); i++)
    {
        auto const& solution = i < lib.solution_list.size()
                                   ? lib.solution_list[i].second
                                   : pool.bank[i - lib.solution_list.size()];
        EXPECT_EQ(configs[i].index, i);
        EXPECT_EQ(configs[i].mt.m, solution->sizeMapping.macroTile.x) << i;
        EXPECT_EQ(configs[i].mt.n, solution->sizeMapping.macroTile.y) << i;
    }
}

TEST(PredictionLibraryTest, TilewrightBankOffersEqualityKernelsForAnySize)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    auto const blocky  = withTile(21, 128, 64);
    for(char const* setting : {"0", "1"})
    {
        SCOPED_TRACE(setting);
        ScopedTilewrightBank scoped(setting);

        // The model scores no kernel of this pool, so origami ranks it alone.
        SolutionMap<ContractionSolution> solutions;
        solutions.emplace(13, makeDot2Solution(13));
        ScratchDir                          dir("tensilelite_tilewright_bank_any_size");
        ContractionProblemPredictionLibrary lib;
        loadWithTilewright(dir, shippedGfx950Model(), {13}, solutions, lib);
        ASSERT_NE(lib.tilewright_ranker, nullptr);
        setBankSources(lib, {blocky});

        auto const top = lib.findTopSolutions(problem, device, 2);
        if(std::string(setting) == "1")
            EXPECT_EQ(solutionIndices(top), (std::vector<int>{21, 13}));
        else
            EXPECT_EQ(solutionIndices(top), (std::vector<int>{13}));
    }
}

TEST(PredictionLibraryTest, TilewrightBankLeavesExactMatchesToTheirEqualityRow)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();
    ScopedTilewrightBank bankOn("1");

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(13, makeDot2Solution(13));
    ScratchDir dir("tensilelite_tilewright_bank_exact");
    auto       lib = std::make_shared<ContractionProblemPredictionLibrary>();
    loadWithTilewright(dir, shippedGfx950Model(), {13}, solutions, *lib);
    ASSERT_NE(lib->tilewright_bank, nullptr);
    setBankSources(*lib, {withTile(21, 128, 64), withTile(22, 256, 256), withTile(23, 64, 256)});

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    auto const ranked  = lib->findTopSolutions(problem, device, 3);
    ASSERT_EQ(ranked.size(), 3u);

    ContractionProblemSelectionLibrary rows;
    rows.rows.emplace_back(
        ContractionProblemPredicate(std::make_shared<Predicates::Contraction::EqualityMatching>()),
        std::make_shared<ExactRow>(SolutionVector<ContractionSolution>{ranked[0]}));
    rows.rows.emplace_back(ContractionProblemPredicate(
                               std::make_shared<Predicates::Contraction::PredictionMatching>()),
                           lib);

    EXPECT_EQ(solutionIndices(rows.findTopSolutions(problem, device, 3)),
              (std::vector<int>{ranked[0]->index, ranked[1]->index, ranked[2]->index}));
    EXPECT_EQ(solutionIndices(lib->findTopSolutions(problem, device, 3)), solutionIndices(ranked));
}

TEST(PredictionLibraryTest, TilewrightBankPoolsAreKeptPerDevice)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();
    ScopedTilewrightBank bankOn("1");

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);
    ScratchDir                          dir("tensilelite_tilewright_bank_devices");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_bank, nullptr);
    setBankSources(lib, {onlyFor(withTile(21, 128, 64), 256), onlyFor(withTile(22, 128, 64), 128)});

    auto const problem = TensileLite::testing::dummyProblem();
    auto const full    = makeGfx950Device(256);
    auto const half    = makeGfx950Device(128);
    lib.findTopSolutions(problem, full, 1);
    lib.findTopSolutions(problem, half, 1);
    EXPECT_EQ(bankIndices(builtPool(lib, full)), (std::vector<int>{21}));
    EXPECT_EQ(bankIndices(builtPool(lib, half)), (std::vector<int>{22}));
}

TEST(PredictionLibraryTest, TilewrightBankWithoutKernelsRanksThePredictionPool)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();

    std::vector<int> withoutBank;
    {
        ScopedTilewrightBank             bankOff("0");
        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);
        ScratchDir                          dir("tensilelite_tilewright_bank_off");
        ContractionProblemPredictionLibrary lib;
        loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
        withoutBank = solutionIndices(lib.findTopSolutions(problem, device, 3));
    }

    ScopedTilewrightBank             bankOn("1");
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);
    ScratchDir                          dir("tensilelite_tilewright_bank_empty");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_bank, nullptr);
    setBankSources(lib, {solutions.at(12)});

    EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), withoutBank);
    auto const& pool = builtPool(lib, device);
    EXPECT_TRUE(pool.bank.empty());
    EXPECT_EQ(pool.ranker, nullptr);
}

namespace
{
    // Sets TENSILE_USE_TILEWRIGHT (unsets it for nullptr) for the lifetime of
    // the object.
    class ScopedTilewright
    {
    public:
        explicit ScopedTilewright(char const* value)
        {
            if(char const* previous = std::getenv("TENSILE_USE_TILEWRIGHT"))
                m_previous = previous;
            if(value)
                setenv("TENSILE_USE_TILEWRIGHT", value, 1);
            else
                unsetenv("TENSILE_USE_TILEWRIGHT");
            Debug::Instance().reloadDebugBitsForTest();
        }

        ~ScopedTilewright()
        {
            if(m_previous)
                setenv("TENSILE_USE_TILEWRIGHT", m_previous->c_str(), 1);
            else
                unsetenv("TENSILE_USE_TILEWRIGHT");
            Debug::Instance().reloadDebugBitsForTest();
        }

    private:
        std::optional<std::string> m_previous;
    };

    // The rows of a problem type in a lazily loaded library: a row per library
    // file, each a loaded placeholder of that file's rows.
    std::shared_ptr<ContractionProblemSelectionLibrary>
        loadedFiles(std::vector<std::vector<ContractionProblemSelectionLibrary::Row>> const& files)
    {
        using Placeholder = PlaceholderLibrary<ContractionProblemGemm, ContractionSolution>;
        std::vector<ContractionProblemSelectionLibrary::Row> rows;
        for(size_t i = 0; i < files.size(); i++)
        {
            auto file        = std::make_shared<Placeholder>();
            file->filePrefix = "TensileLibrary_Probe" + std::to_string(i) + "_gfx950";
            file->library    = std::make_shared<ContractionProblemSelectionLibrary>(files[i]);
            rows.emplace_back(ContractionProblemPredicate(
                                  std::make_shared<Predicates::True<ContractionProblemGemm>>()),
                              file);
        }
        return std::make_shared<ContractionProblemSelectionLibrary>(rows);
    }
}

TEST(PredictionLibraryTest, ALoadedModelIsSearchedBeforeTheGridBasedRowsOfEarlierFiles)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);
    ScratchDir dir("tensilelite_tilewright_row_order");
    auto       withModel = std::make_shared<ContractionProblemPredictionLibrary>();
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, *withModel);
    ASSERT_NE(withModel->tilewright_ranker, nullptr);
    EXPECT_TRUE(withModel->hasTilewrightModel());

    SolutionMap<ContractionSolution> plain;
    addContestedKernels(plain);
    auto withoutModel = std::make_shared<ContractionProblemPredictionLibrary>();
    loadPredictionLibrary(contestedTable, plain, *withoutModel);
    EXPECT_FALSE(withoutModel->hasTilewrightModel());

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    auto const table   = solutionIndices(withModel->findTopSolutions(problem, device, 4));
    ASSERT_FALSE(table.empty());
    ASSERT_EQ(table.front(), tilewrightOrder(*withModel).front());
    int const all = static_cast<int>(table.size()) + 1;

    // A file with a GridBased row of one kernel before the file of the
    // Prediction table.
    auto const grid = SolutionVector<ContractionSolution>{withTile(31, 512, 512)};
    auto       rows = [&](std::shared_ptr<ContractionProblemPredictionLibrary> const& prediction) {
        return loadedFiles(
            {{ContractionProblemSelectionLibrary::Row(
                 ContractionProblemPredicate(
                     std::make_shared<Predicates::Contraction::GridBasedMatching>()),
                 std::make_shared<ExactRow>(grid))},
             {ContractionProblemSelectionLibrary::Row(
                 ContractionProblemPredicate(
                     std::make_shared<Predicates::Contraction::PredictionMatching>()),
                 prediction)}});
    };
    auto expected = table;
    expected.push_back(31);
    {
        ScopedTilewright on("1");

        auto const lib = rows(withModel);
        EXPECT_EQ(solutionIndices(lib->findTopSolutions(problem, device, 1)),
                  (std::vector<int>{table.front()}));
        EXPECT_EQ(solutionIndices(lib->findTopSolutions(problem, device, all)), expected);
        EXPECT_EQ(solutionIndices(rows(withoutModel)->findTopSolutions(problem, device, 1)),
                  (std::vector<int>{31}));
    }
    ScopedTilewright off(nullptr);
    EXPECT_EQ(solutionIndices(rows(withModel)->findTopSolutions(problem, device, 1)),
              (std::vector<int>{31}));
}

#if defined(TILEWRIGHT_TEST_MODEL_WRITER)
namespace
{
    // MLREC_v3 serving flag of a model trained on the Prediction kernels and
    // the bank.
    constexpr std::uint32_t TrainedOnBank = 0x40;

    // An MLREC_v3 model with `servingFlags` that reads `attributes`. Its one
    // cell scores every problem, with the launch model of gfx950 libraries.
    std::filesystem::path writeV3Model(ScratchDir const&                             dir,
                                       std::uint32_t                                 servingFlags,
                                       std::vector<tilewright_test::AttrSpec> const& attributes)
    {
        tilewright_test::CellSpec cell;
        cell.label = "Large|Large|LargeK|Bnone";

        tilewright_test::ModelSpec spec;
        spec.arch             = "gfx950";
        spec.cells            = {cell};
        spec                  = tilewright_test::as_v3(spec, attributes);
        spec.v3_serving_flags = servingFlags;
        spec.provenance
            = {{"launch_model", TilewrightLaunch<ContractionSolution>("gfx950", {}).id()}};

        auto const image = tilewright_test::write_model(spec);
        auto const path  = dir.path() / "v3.tilewright.bin";
        std::ofstream(path, std::ios::binary)
            .write(reinterpret_cast<char const*>(image.bytes.data()),
                   static_cast<std::streamsize>(image.bytes.size()));
        return path;
    }

    // Ranks the contested table with a writeV3Model model, with
    // withTile(21, 128, 128) as the bank, and expects findTopSolutions to
    // return the bank kernel exactly when `bank` is set: origami ranks only
    // the table, so only tilewright can return it.
    void expectBankRanked(std::uint32_t                                 servingFlags,
                          std::vector<tilewright_test::AttrSpec> const& attributes,
                          bool                                          bank)
    {
        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);
        ScratchDir                          dir("tensilelite_tilewright_bank_v3");
        ContractionProblemPredictionLibrary lib;
        loadWithTilewright(
            dir, writeV3Model(dir, servingFlags, attributes), contestedTable, solutions, lib);
        ASSERT_NE(lib.tilewright_ranker, nullptr);
        setBankSources(lib, {withTile(21, 128, 128)});

        auto const device = makeGfx950Device();
        auto       top    = solutionIndices(
            lib.findTopSolutions(TensileLite::testing::dummyProblem(), device, 4));
        std::sort(top.begin(), top.end());
        EXPECT_EQ(top, (bank ? std::vector<int>{11, 12, 13, 21} : std::vector<int>{11, 12, 13}));
        ASSERT_EQ(lib.tilewright_bank != nullptr, bank);
        if(bank)
            EXPECT_EQ(bankIndices(builtPool(lib, device)), (std::vector<int>{21}));
    }
}

TEST(PredictionLibraryTest, TilewrightRanksTheBankOfAModelTrainedOnIt)
{
    ScopedTilewrightBank unset(nullptr);
    {
        SCOPED_TRACE("trained on the bank");
        expectBankRanked(TrainedOnBank, {}, true);
    }
    {
        SCOPED_TRACE("trained on the Prediction kernels");
        expectBankRanked(0, {}, false);
    }
}

TEST(PredictionLibraryTest, TilewrightBankSettingOverridesTheModel)
{
    {
        ScopedTilewrightBank bankOff("0");
        expectBankRanked(TrainedOnBank, {}, false);
    }
    ScopedTilewrightBank bankOn("1");
    expectBankRanked(0, {}, true);
}

TEST(PredictionLibraryTest, TilewrightBankFollowsTheAttributesOfAModelNotTrainedOnIt)
{
    ScopedTilewrightBank unset(nullptr);
    expectBankRanked(0, {{"work_group_x", 0}}, true);
}

namespace
{
    std::shared_ptr<ContractionSolution> named(std::shared_ptr<ContractionSolution> solution)
    {
        solution->solutionName = "probe_" + std::to_string(solution->index);
        return solution;
    }

    // 31 and 32 share one signature, so the model scores them as one
    // candidate; 33 is another.
    std::vector<int> const keyTable = {31, 32, 33};

    void addKeyKernels(SolutionMap<ContractionSolution>& solutions)
    {
        solutions.emplace(31, named(withTile(31, 128, 128)));
        solutions.emplace(32, named(withTile(32, 128, 128)));
        solutions.emplace(33, named(withTile(33, 64, 256)));
    }

    // Loads keyTable with a writeV3Model model and a blocklist of `blocked`
    // (none for an empty list).
    void loadBlocked(ScratchDir const&                    dir,
                     std::vector<std::string> const&      blocked,
                     SolutionMap<ContractionSolution>&    solutions,
                     ContractionProblemPredictionLibrary& lib,
                     std::uint32_t                        servingFlags = 0)
    {
        if(!blocked.empty())
        {
            std::ofstream out(dir.path() / "TensileLibrary_Probe_gfx950.tilewright.blocklist");
            out << "# kernels that compute wrong results\n\n";
            for(auto const& name : blocked)
                out << "  " << name << " \n";
        }
        addKeyKernels(solutions);
        loadWithTilewright(dir, writeV3Model(dir, servingFlags, {}), keyTable, solutions, lib);
    }

    std::vector<size_t> configPositions(TilewrightRanker const& ranker)
    {
        std::vector<size_t> positions;
        for(auto const& config : ranker.candidates().configs())
            positions.push_back(config.index);
        return positions;
    }
}

TEST(PredictionLibraryTest, TilewrightServesTheNextMemberOfACandidateWhoseFirstIsBlocked)
{
    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();

    SolutionMap<ContractionSolution>    plain;
    ScratchDir                          plainDir("tensilelite_tilewright_unblocked");
    ContractionProblemPredictionLibrary unblocked;
    loadBlocked(plainDir, {}, plain, unblocked);
    ASSERT_NE(unblocked.tilewright_ranker, nullptr);
    EXPECT_FALSE(unblocked.tilewright_ranker->blocksAny());
    auto const before = unblocked.tilewright_ranker->rank(problem, device, -1);
    ASSERT_EQ(before.size(), 3u);
    tilewright::Problem const twProblem{
        .size     = {1024, 1024, 1024},
        .batch    = 1,
        .a_dtype  = tilewright::DataType::Float,
        .b_dtype  = tilewright::DataType::Float,
        .c_dtype  = tilewright::DataType::Float,
        .d_dtype  = tilewright::DataType::Float,
        .mi_dtype = tilewright::DataType::Float,
    };
    std::map<size_t, double> score;
    {
        TilewrightLaunchScope<ContractionProblemGemm> const scope(problem, device);
        for(auto const& r :
            unblocked.tilewright_ranker->candidates().rank(twProblem, {256, 163840, 4000000}, 3))
            score[r.config_index] = r.score;
    }
    ASSERT_EQ(score.size(), 3u);
    EXPECT_EQ(score.at(0), score.at(1)) << "31 and 32 are one candidate";
    EXPECT_NE(score.at(0), score.at(2));

    SolutionMap<ContractionSolution>    solutions;
    ScratchDir                          dir("tensilelite_tilewright_blocked_member");
    ContractionProblemPredictionLibrary lib;
    loadBlocked(dir, {"probe_31"}, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);
    auto const& ranker = *lib.tilewright_ranker;
    EXPECT_EQ(configPositions(ranker), (std::vector<size_t>{1, 2}));
    EXPECT_TRUE(ranker.blocked(0));
    EXPECT_FALSE(ranker.blocked(1));
    EXPECT_FALSE(ranker.blocked(5));

    std::vector<size_t> expected;
    for(size_t position : before)
        if(position != 0)
            expected.push_back(position);
    EXPECT_EQ(ranker.rank(problem, device, -1), expected);

    auto const top = solutionIndices(lib.findTopSolutions(problem, device, 3));
    EXPECT_EQ(top.size(), 2u);
    EXPECT_EQ(std::count(top.begin(), top.end(), 31), 0);
    EXPECT_EQ(std::count(top.begin(), top.end(), 32), 1);
}

TEST(PredictionLibraryTest, TilewrightDropsACandidateWhoseMembersAreAllBlocked)
{
    SolutionMap<ContractionSolution>    solutions;
    ScratchDir                          dir("tensilelite_tilewright_blocked_key");
    ContractionProblemPredictionLibrary lib;
    loadBlocked(dir, {"probe_31", "probe_32", "probe_of_another_library"}, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);
    EXPECT_EQ(configPositions(*lib.tilewright_ranker), (std::vector<size_t>{2}));

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    EXPECT_EQ(lib.tilewright_ranker->rank(problem, device, -1), (std::vector<size_t>{2}));
    // Origami fills the remaining slots without the blocked kernels too.
    EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), (std::vector<int>{33}));
}

TEST(PredictionLibraryTest, TilewrightIsUnusedWhenItsBlocklistCannotBeRead)
{
    SolutionMap<ContractionSolution> solutions;
    ScratchDir                       dir("tensilelite_tilewright_bad_blocklist");
    std::filesystem::create_directory(dir.path()
                                      / "TensileLibrary_Probe_gfx950.tilewright.blocklist");
    addKeyKernels(solutions);
    ContractionProblemPredictionLibrary lib;
    ::testing::internal::CaptureStderr();
    loadWithTilewright(dir, writeV3Model(dir, 0, {}), keyTable, solutions, lib);
    std::string const warning = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(lib.tilewright_ranker, nullptr);
    EXPECT_NE(warning.find("cannot read the blocklist"), std::string::npos) << warning;
    EXPECT_NE(warning.find("ranking with origami"), std::string::npos) << warning;
}

TEST(PredictionLibraryTest, TilewrightLeavesBlockedKernelsOutOfTheBank)
{
    ScopedTilewrightBank                unset(nullptr);
    SolutionMap<ContractionSolution>    solutions;
    ScratchDir                          dir("tensilelite_tilewright_blocked_bank");
    ContractionProblemPredictionLibrary lib;
    loadBlocked(dir, {"probe_31", "probe_21"}, solutions, lib, TrainedOnBank);
    ASSERT_NE(lib.tilewright_bank, nullptr);
    setBankSources(lib, {named(withTile(21, 128, 64)), named(withTile(22, 256, 64))});

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    auto       top     = solutionIndices(lib.findTopSolutions(problem, device, 5));
    std::sort(top.begin(), top.end());
    EXPECT_EQ(top, (std::vector<int>{22, 32, 33}));
    auto const& pool = builtPool(lib, device);
    EXPECT_EQ(bankIndices(pool), (std::vector<int>{21, 22}));
    ASSERT_NE(pool.ranker, nullptr);
    EXPECT_EQ(configPositions(*pool.ranker), (std::vector<size_t>{1, 2, 4}));
}
#endif
#endif

TEST(PredictionLibraryTest, LibraryProblemTypeKeyDropsTheHardwareParts)
{
    std::string const type = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_"
                             "Bljk_Cijk_Dijk";
    for(auto const& hardware : {"", "_CU256", "_ID75a0", "_CU256_ID75a0", "_ID75a3-75a2"})
        EXPECT_EQ(libraryProblemTypeKey(type + hardware + "_gfx950"), type + "_gfx950") << hardware;
    EXPECT_EQ(libraryProblemTypeKey(type + "_CU128_gfx1250-strict"), type + "_gfx1250-strict");
    EXPECT_EQ(libraryProblemTypeKey("TensileLibrary_Probe_IDx_gfx950"),
              "TensileLibrary_Probe_IDx_gfx950");
    EXPECT_EQ(libraryProblemTypeKey("TensileLibrary_Probe_CU_gfx950"),
              "TensileLibrary_Probe_CU_gfx950");
    EXPECT_NE(libraryProblemTypeKey(type + "_ID75a0_gfx950"),
              libraryProblemTypeKey(type + "_ID75a0_gfx942"));
}

TEST(PredictionLibraryTest, PlaceholderRegistryListsTheEqualityRowsOfTheProblemType)
{
    using Placeholder = PlaceholderLibrary<ContractionProblemGemm, ContractionSolution>;

    auto row = [](auto predicate, int index) {
        return ContractionProblemSelectionLibrary::Row(
            ContractionProblemPredicate(predicate),
            std::make_shared<SingleContractionLibrary>(makeMappedSolution(index)));
    };
    auto equality   = std::make_shared<Predicates::Contraction::EqualityMatching>();
    auto range      = std::make_shared<Predicates::Contraction::RangeMatching>();
    auto prediction = std::make_shared<Predicates::Contraction::PredictionMatching>();

    std::string const type = "TensileLibrary_Probe_Contraction_l_Alik_Bljk_Cijk_Dijk";
    std::vector<std::pair<std::string, std::shared_ptr<ContractionProblemSelectionLibrary>>> files{
        {type + "_ID75a0_gfx950",
         std::make_shared<ContractionProblemSelectionLibrary>(
             std::vector<ContractionProblemSelectionLibrary::Row>{
                 row(range, 1), row(equality, 2), row(prediction, 3)})},
        {type + "_CU256_ID75a0_gfx950",
         std::make_shared<ContractionProblemSelectionLibrary>(
             std::vector<ContractionProblemSelectionLibrary::Row>{row(equality, 4)})},
        {type + "_ID75a3-75a2_gfx950",
         std::make_shared<ContractionProblemSelectionLibrary>(
             std::vector<ContractionProblemSelectionLibrary::Row>{row(equality, 5)})},
        {"TensileLibrary_Other_Contraction_l_Alik_Bljk_Cijk_Dijk_CU256_ID75a0_gfx950",
         std::make_shared<ContractionProblemSelectionLibrary>(
             std::vector<ContractionProblemSelectionLibrary::Row>{row(equality, 6)})},
    };

    PlaceholderRegistry<ContractionProblemGemm, ContractionSolution> registry;
    std::vector<std::unique_ptr<Placeholder>>                        placeholders;
    for(auto const& [prefix, tree] : files)
    {
        auto placeholder        = std::make_unique<Placeholder>();
        placeholder->filePrefix = prefix;
        placeholder->library    = tree;
        registry.byPrefix.emplace(prefix, placeholder.get());
        placeholders.push_back(std::move(placeholder));
    }

    std::vector<int> indices;
    auto const       problem = TensileLite::testing::dummyProblem();
    auto const       device  = makeGfx950Device();
    for(auto const& library : registry.equalityLibraries(type + "_ID75a0_gfx950"))
    {
        for(auto const& solution :
            library->findAllSolutions(problem, device, SolutionLibrarySearchType::HARDWARE_ONLY))
            indices.push_back(solution->index);
    }
    // In file-prefix order: _CU256_ID75a0, _ID75a0, _ID75a3-75a2.
    EXPECT_EQ(indices, (std::vector<int>{4, 2, 5}));
}
#endif
