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

TEST(PredictionLibraryTest, LegacyNonTemporalBecomesOrigamiCacheHint)
{
    auto solution                      = makeMappedSolution(42);
    solution->sizeMapping.nonTemporalA = 4;
    solution->sizeMapping.nonTemporalB = 0;

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);
    EXPECT_EQ(lib.origami_config_list[0].cache_hints_a, 4);
    EXPECT_EQ(lib.origami_config_list[0].cache_hints_b, 0);
}

TEST(PredictionLibraryTest, TemporalHintsOverrideLegacyNonTemporal)
{
    auto solution                         = makeMappedSolution(42);
    solution->sizeMapping.hasTemporalHint = true;
    solution->sizeMapping.temporalHintA   = 1;
    solution->sizeMapping.temporalHintB   = 2;
    solution->sizeMapping.nonTemporalA    = 0;
    solution->sizeMapping.nonTemporalB    = 4;

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);
    EXPECT_EQ(lib.origami_config_list[0].cache_hints_a, 4);
    EXPECT_EQ(lib.origami_config_list[0].cache_hints_b, 0);
}

// cacheHintA/B only treat temporal hints 1 (TH_NT) and 3 (TH_LU / TH_WB) as
// non-temporal. TH_NT_RT, TH_RT_NT, TH_NT_HT, and TH_NT_WB (4-7) stay cached,
// and that mapping overrides a legacy non-temporal value.
TEST(PredictionLibraryTest, UnmappedTemporalHintsStayCached)
{
    struct Case
    {
        int hint;
        int cache;
    };
    // 3 is the other non-temporal encoding. 4-7 currently collapse to 0.
    Case const cases[] = {{3, 4}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};

    for(auto const& c : cases)
    {
        auto solution                         = makeMappedSolution(42);
        solution->sizeMapping.hasTemporalHint = true;
        solution->sizeMapping.temporalHintA   = c.hint;
        solution->sizeMapping.temporalHintB   = c.hint;
        solution->sizeMapping.nonTemporalA    = 4;
        solution->sizeMapping.nonTemporalB    = 4;

        SolutionMap<ContractionSolution> solutions;
        solutions.emplace(42, solution);

        ContractionProblemPredictionLibrary lib;
        loadPredictionLibrary({42}, solutions, lib);
        ASSERT_EQ(lib.origami_config_list.size(), 1u) << "temporalHint " << c.hint;
        EXPECT_EQ(lib.origami_config_list[0].cache_hints_a, c.cache) << "temporalHint " << c.hint;
        EXPECT_EQ(lib.origami_config_list[0].cache_hints_b, c.cache) << "temporalHint " << c.hint;
    }
}

// The per-tile flags are written once while the library is loaded.
// An NT-B solution at 128x256 must not mark the 256x256 solution.
TEST(PredictionLibraryTest, NonTemporalAvailabilityIsMemoizedPerMacroTile)
{
    auto ntB                        = makeMappedSolution(42);
    ntB->sizeMapping.macroTile      = TensileLite::dim3(128, 256, 1);
    ntB->sizeMapping.nonTemporalB   = 4;
    auto temporal                   = makeMappedSolution(7);
    temporal->sizeMapping.macroTile = TensileLite::dim3(256, 256, 1);

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, ntB);
    solutions.emplace(7, temporal);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42, 7}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 2u);

    EXPECT_FALSE(lib.origami_config_list[0].non_temporal_a_available);
    EXPECT_TRUE(lib.origami_config_list[0].non_temporal_b_available);
    EXPECT_FALSE(lib.origami_config_list[1].non_temporal_a_available);
    EXPECT_FALSE(lib.origami_config_list[1].non_temporal_b_available);
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

TEST(PredictionLibraryTest, TilewrightIsSilentlyUnusedWhenTheIndexListsNoModelForTheLibrary)
{
    // A library subtree that ships no model keeps its tilewright_index, with
    // its header lines alone or with the entries of other libraries: Origami
    // ranks its libraries as it ranks those of a subtree without an index.
    auto const addKernels = [](SolutionMap<ContractionSolution>& solutions) {
        for(auto const& [index, m, n] : std::vector<std::tuple<int, size_t, size_t>>{
                {11, 128, 64}, {12, 16, 256}, {13, 256, 256}})
        {
            auto solution                   = makeMappedSolution(index);
            solution->sizeMapping.macroTile = TensileLite::dim3(m, n, 1);
            solutions.emplace(index, solution);
        }
    };
    auto const indices = [](SolutionVector<ContractionSolution> const& top) {
        std::vector<int> out;
        for(auto const& solution : top)
            out.push_back(solution->index);
        return out;
    };
    std::vector<int> const table   = {13, 11, 12};
    auto const             problem = TensileLite::testing::dummyProblem();
    auto const             device  = makeGfx950Device();

    SolutionMap<ContractionSolution> plainSolutions;
    addKernels(plainSolutions);
    ContractionProblemPredictionLibrary plain;
    loadPredictionLibrary(table, plainSolutions, plain);
    auto const origami = indices(plain.findTopSolutions(problem, device, 3));
    ASSERT_EQ(origami.size(), table.size());

    for(std::string const& indexFile : std::vector<std::string>{
            "# tilewright per-library weights index: <logic_stem>\\t<weights.bin>\n"
            "# auto-generated by the training pipeline's deploy stage\n",
            "TensileLibrary_Other_gfx1250\tother.tilewright.bin\n"})
    {
        ScratchDir dir("tensilelite_tilewright_no_entry");
        std::ofstream(dir.path() / "tilewright_index") << indexFile;

        SolutionMap<ContractionSolution> solutions;
        addKernels(solutions);
        ContractionProblemPredictionLibrary lib;
        loadPredictionLibrary(table, solutions, lib);

        ::testing::internal::CaptureStderr();
        lib.loadTilewright((dir.path() / "TensileLibrary_Probe_gfx1250.dat").string());
        EXPECT_EQ(::testing::internal::GetCapturedStderr(), "") << indexFile;
        EXPECT_EQ(lib.tilewright_ranker, nullptr) << indexFile;
        EXPECT_FALSE(lib.hasTilewrightModel()) << indexFile;
        EXPECT_EQ(indices(lib.findTopSolutions(problem, device, 3)), origami) << indexFile;
    }
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

TEST(PredictionLibraryTest, TilewrightBlocklistKeysDropTheIaAndPkaTokens)
{
    // The training pipeline's blocklist_key gives these keys too.
    std::vector<std::pair<std::string, std::string>> const keys = {
        {"Cijk_Ailk_MT64x64x32_IA0_K1_PKA1_SU0", "Cijk_Ailk_MT64x64x32_K1_SU0"},
        {"Cijk_Ailk_MT64x64x32_K1_PKA0_SU0", "Cijk_Ailk_MT64x64x32_K1_SU0"},
        {"IA2_Cijk_IA10_PKAn1", "Cijk"},
        {"Cijk_SIA3_IA_PKA_IAx1_PKA1x_MIA0_IAn_ia0", "Cijk_SIA3_IA_PKA_IAx1_PKA1x_MIA0_IAn_ia0"},
        {"Cijk__IA0_", "Cijk__"},
        {"PKA1", ""},
        {"", ""},
        {"Custom_Cijk_Ailk_Bljk_S_SK3_UserArgs_MT256x256x32_shortname0",
         "Custom_Cijk_Ailk_Bljk_S_SK3_UserArgs_MT256x256x32_shortname0"},
        {"Cijk_Ailk_Bjlk_S_MT64x64x64_MI32x32x1_GRVWA1_GRVWB4_IA0_K1_MIAV1_NTA7_NTB2_PKA1_SIA3_SU0",
         "Cijk_Ailk_Bjlk_S_MT64x64x64_MI32x32x1_GRVWA1_GRVWB4_K1_MIAV1_NTA7_NTB2_SIA3_SU0"},
    };
    for(auto const& [name, key] : keys)
        EXPECT_EQ(TilewrightRanker::blocklistKey(name), key) << name;
    // Kernels whose names differ in any other token keep different keys.
    EXPECT_NE(TilewrightRanker::blocklistKey("Cijk_MT64x64x32_IA0_NTA0_NTB0_PKA1"),
              TilewrightRanker::blocklistKey("Cijk_MT64x64x32_IA0_NTA4_NTB0_PKA1"));
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

    auto const config = TilewrightRanker::makeConfig(sizeMapping, "", 7);
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

    auto const dot2 = TilewrightRanker::makeConfig(sizeMapping, "", 0);
    EXPECT_EQ(dot2.mi.m, 1u);
    EXPECT_EQ(dot2.mi.n, 1u);
    EXPECT_EQ(dot2.mi.k, 64u);
    EXPECT_EQ(dot2.occupancy, 1);
    EXPECT_EQ(dot2.cache_hints_a, 0);
    EXPECT_EQ(dot2.cache_hints_b, 4);
}

TEST(PredictionLibraryTest, TilewrightReadsTdmInstFromTheSolutionName)
{
    EXPECT_EQ(TilewrightRanker::tdmInst(""), 0);
    EXPECT_EQ(TilewrightRanker::tdmInst("Cijk_MT64x64x64_TDMI3_TDMIM0_THA1"), 3);
    EXPECT_EQ(TilewrightRanker::tdmInst("Cijk_MT64x64x64_THA1_THB1_TDMI2"), 2);
    // TDMIterateMode is another token.
    EXPECT_EQ(TilewrightRanker::tdmInst("Cijk_MT64x64x64_TDMIM3_THA1"), 0);
    EXPECT_EQ(TilewrightRanker::tdmInst("Cijk_MT64x64x64_TDMI_THA1"), 0);
    EXPECT_EQ(TilewrightRanker::tdmInst("Cijk_MT64x64x64_TDMI3x_THA1"), 0);
    // A name that spells one value twice gives it; two different values none.
    EXPECT_EQ(TilewrightRanker::tdmInst("Cijk_TDMI1_MT64x64x64_TDMI1"), 1);
    EXPECT_EQ(TilewrightRanker::tdmInst("Cijk_TDMI3_MT64x64x64_TDMI1"), 0);
}

TEST(PredictionLibraryTest, TilewrightConfigZeroesTheCacheHintOfTdmLoadedOperands)
{
    // A kernel with a temporal hint on both operands: TensileLite gives each a
    // cache hint, which does not reach the code of an operand the tensor data
    // mover loads.
    SizeMapping sizeMapping{};
    sizeMapping.macroTile       = TensileLite::dim3(128, 128, 1);
    sizeMapping.depthU          = 64;
    sizeMapping.hasTemporalHint = true;
    sizeMapping.temporalHintA   = 1;
    sizeMapping.temporalHintB   = 1;
    ASSERT_EQ(sizeMapping.cacheHintA(), 4);
    ASSERT_EQ(sizeMapping.cacheHintB(), 4);

    std::string const prefix = "Cijk_Alik_Bljk_BBS_BH_MT128x128x64_MI16x16x1_";
    for(auto const& [tokens, hintA, hintB] :
        std::vector<std::tuple<std::string, int, int>>{{"TDMI3_TDMIM0_THA1_THB1", 0, 0},
                                                       {"TDMI1_TDMIM0_THA1_THB1", 0, 4},
                                                       {"TDMI2_TDMIM0_THA1_THB1", 4, 0},
                                                       {"TDMI0_TDMIM0_THA1_THB1", 4, 4},
                                                       {"TDMIM3_THA1_THB1", 4, 4}})
    {
        auto const config = TilewrightRanker::makeConfig(sizeMapping, prefix + tokens, 0);
        EXPECT_EQ(config.cache_hints_a, hintA) << tokens;
        EXPECT_EQ(config.cache_hints_b, hintB) << tokens;
        // The attributes keep the hints the logic file gives.
        EXPECT_EQ(findAttribute(config, "temporal_hint_a"), 1) << tokens;
        EXPECT_EQ(findAttribute(config, "temporal_hint_b"), 1) << tokens;
    }

    // Without temporal hints the non-temporal fields give the cache hints; a
    // loaded operand's is 0 all the same.
    sizeMapping.hasTemporalHint = false;
    sizeMapping.nonTemporalA    = 4;
    sizeMapping.nonTemporalB    = 2;
    auto const loaded = TilewrightRanker::makeConfig(sizeMapping, prefix + "NTA4_NTB2_TDMI3", 0);
    EXPECT_EQ(loaded.cache_hints_a, 0);
    EXPECT_EQ(loaded.cache_hints_b, 0);
    auto const plain = TilewrightRanker::makeConfig(sizeMapping, prefix + "NTA4_NTB2", 0);
    EXPECT_EQ(plain.cache_hints_a, 4);
    EXPECT_EQ(plain.cache_hints_b, 2);
}

#if defined(TILEWRIGHT_TDM_CACHE_HINT_CASES)
TEST(PredictionLibraryTest, TilewrightCacheHintsFollowTheTrainingPipelineCases)
{
    // training_pipeline/tests/data/tdm_cache_hint_cases.tsv, which the
    // pipeline's tests read with lib/dat.py: a solution name, its sizeMapping's
    // hint fields, the TDMInst the name spells and the cache hints of A and B.
    std::ifstream in(TILEWRIGHT_TDM_CACHE_HINT_CASES);
    ASSERT_TRUE(in) << TILEWRIGHT_TDM_CACHE_HINT_CASES;
    std::string line;
    int         cases = 0;
    while(std::getline(in, line))
    {
        if(line.empty() || line[0] == '#')
            continue;
        std::istringstream fields(line);
        std::string        name;
        int hasTemporalHint = 0, temporalA = 0, temporalB = 0, nonTemporalA = 0, nonTemporalB = 0;
        int tdm = 0, hintA = 0, hintB = 0;
        ASSERT_TRUE(fields >> name >> hasTemporalHint >> temporalA >> temporalB >> nonTemporalA
                    >> nonTemporalB >> tdm >> hintA >> hintB)
            << line;
        SizeMapping sizeMapping{};
        sizeMapping.hasTemporalHint = hasTemporalHint != 0;
        sizeMapping.temporalHintA   = temporalA;
        sizeMapping.temporalHintB   = temporalB;
        sizeMapping.nonTemporalA    = nonTemporalA;
        sizeMapping.nonTemporalB    = nonTemporalB;
        EXPECT_EQ(TilewrightRanker::tdmInst(name), tdm) << name;
        auto const config = TilewrightRanker::makeConfig(sizeMapping, name, 0);
        EXPECT_EQ(config.cache_hints_a, hintA) << name;
        EXPECT_EQ(config.cache_hints_b, hintB) << name;
        ++cases;
    }
    EXPECT_GT(cases, 0);
}
#endif

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

    auto const config = TilewrightRanker::makeConfig(attributeProbe(), "", 0);

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
        EXPECT_EQ(findAttribute(TilewrightRanker::makeConfig(sizeMapping, "", 0),
                                "tile_processing_strategy"),
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
        EXPECT_EQ(
            findAttribute(TilewrightRanker::makeConfig(sizeMapping, "", 0), "work_assignment"),
            code)
            << toString(assignment);
    }
}

TEST(PredictionLibraryTest, TilewrightAttributesOfAMinimalSolution)
{
    auto const config = TilewrightRanker::makeConfig(makeMappedSolution(42)->sizeMapping, "", 0);
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

#if defined(TILEWRIGHT_TEST_MODEL_WRITER)
namespace
{
    // An MLREC_v2 model, which the tests below expect: it records no launch
    // model, so they rank its candidates with no device to plan launches for
    // and load it as the model of other architectures' libraries; it supports
    // the whole device with the default schedule alone; and it orders the
    // probe kernels unlike origami. Its one cell, that of dummyProblem()'s
    // size class, scores a kernel by the log2 of its tile's N size: the
    // skinny tile before the blocky one. The tests read a shipped MLREC_v2
    // model until no gfx950 library shipped one.
    std::filesystem::path writeV2Model(ScratchDir const& dir)
    {
        tilewright_test::CellSpec cell;
        cell.label = tilewright_test::base_label(1024, 1024, 1024, 1);
        cell.edit  = [](tilewright_test::CellTensors& t) {
            for(auto* weights : {&t.q_w0,
                                 &t.q_b0,
                                 &t.q_w2,
                                 &t.q_b2,
                                 &t.q_w4,
                                 &t.q_b4,
                                 &t.i_w0,
                                 &t.i_b0,
                                 &t.i_w2,
                                 &t.i_b2,
                                 &t.x_w0,
                                 &t.x_b0,
                                 &t.x_w2,
                                 &t.x_b2})
                std::fill(weights->begin(), weights->end(), 0.0f);
            // The query embedding is (1, 0, ...) and the item embedding's
            // first value is item feature 1, the log2 of the tile's N size,
            // which mean 0 and std 1 leave as is: their product is the score.
            t.temperature = 1.0f;
            t.q_b4[0]     = 1.0f;
            t.i_mean[1]   = 0.0f;
            t.i_std[1]    = 1.0f;
            t.i_w0[1]     = 1.0f;
            t.i_w2[0]     = 1.0f;
        };

        tilewright_test::ModelSpec spec;
        spec.arch  = "gfx950";
        spec.cells = {cell};

        auto const image = tilewright_test::write_model(spec);
        auto const path  = dir.path() / "v2.tilewright.bin";
        std::ofstream(path, std::ios::binary)
            .write(reinterpret_cast<char const*>(image.bytes.data()),
                   static_cast<std::streamsize>(image.bytes.size()));
        return path;
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

    // writeV2Model's model and origami order the blocky and the skinny tile
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
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_rank");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
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
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_pool");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const& configs = lib.tilewright_ranker->candidates().configs();
    auto const  names   = tilewright::attribute_names(lib.tilewright_ranker->candidates().model());
    ASSERT_EQ(configs.size(), lib.solution_list.size());
    for(size_t i = 0; i < configs.size(); ++i)
    {
        auto const& solution = *lib.solution_list[i].second;
        auto const  expected
            = TilewrightRanker::makeConfig(solution.sizeMapping, solution.name(), i, &names);
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

TEST(PredictionLibraryTest, TilewrightRanksTdmLoadedOperandsWithoutTheirCacheHint)
{
    // The ranker reads TDMInst from each kernel's solution name: a TDMI3
    // kernel with temporal hints on both operands reaches the engine with
    // cache hint 0 on both, its TDMI0 twin with TensileLite's hints, which
    // Origami keeps for both.
    SolutionMap<ContractionSolution> solutions;
    for(auto const& [index, name] : std::vector<std::pair<int, std::string>>{
            {21, "Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_TDMI3_THA1_THB1"},
            {22, "Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_TDMI0_THA1_THB1"}})
    {
        auto solution                         = makeMappedSolution(index);
        solution->solutionName                = name;
        solution->sizeMapping.hasTemporalHint = true;
        solution->sizeMapping.temporalHintA   = 1;
        solution->sizeMapping.temporalHintB   = 1;
        solutions.emplace(index, solution);
    }

    ScratchDir                          dir("tensilelite_tilewright_tdm_hints");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), {21, 22}, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const& configs = lib.tilewright_ranker->candidates().configs();
    ASSERT_EQ(configs.size(), 2u);
    EXPECT_EQ(configs[0].cache_hints_a, 0);
    EXPECT_EQ(configs[0].cache_hints_b, 0);
    EXPECT_EQ(configs[1].cache_hints_a, 4);
    EXPECT_EQ(configs[1].cache_hints_b, 4);
    ASSERT_EQ(lib.origami_config_list.size(), 2u);
    for(auto const& config : lib.origami_config_list)
    {
        EXPECT_EQ(config.cache_hints_a, 4);
        EXPECT_EQ(config.cache_hints_b, 4);
    }
}

TEST(PredictionLibraryTest, OrigamiRanksWhenTilewrightScoresNothing)
{
    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(13, makeDot2Solution(13));

    ScratchDir                          dir("tensilelite_tilewright_fallback");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), {13}, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const top
        = lib.findTopSolutions(TensileLite::testing::dummyProblem(), makeGfx950Device(), 1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0]->index, 13);
}

TEST(PredictionLibraryTest, TilewrightRanksCallsWithACuBudgetBelowTheDevice)
{
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_cu_budget");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    // A v2 model was trained for every CU only: a smaller budget is ranked as
    // every CU, never by origami.
    auto const device = makeGfx950Device();
    for(int budget : {224, 31, 1})
    {
        SCOPED_TRACE("smCountTarget " + std::to_string(budget));
        auto problem = TensileLite::testing::dummyProblem();
        problem.setParams().setSmCountTarget(budget);
        EXPECT_FALSE(lib.tilewright_ranker->rank(problem, device, 3).empty());
        EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), tilewrightOrder(lib));
    }
    auto problem = TensileLite::testing::dummyProblem();
    problem.setParams().setSmCountTarget(224);
    expectTilewrightOrder(lib, problem);
}

TEST(PredictionLibraryTest, TilewrightNamesTheCompilerTargetOfTheLibrary)
{
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
                           writeV2Model(dir),
                           contestedTable,
                           solutions,
                           lib,
                           "TensileLibrary_Probe_" + suffix);
        ASSERT_NE(lib.tilewright_ranker, nullptr) << suffix;
        EXPECT_EQ(lib.tilewright_ranker->arch(), arch) << suffix;
    }
}

TEST(PredictionLibraryTest, TilewrightNotesEachContextOutsideTheModelOnce)
{
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_unsupported");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    tilewright::Hardware const hardware{256, 163840, 4000000};
    auto const&                ranker = *lib.tilewright_ranker;
    EXPECT_FALSE(ranker.unsupportedContextNote({}, hardware));
    EXPECT_FALSE(ranker.unsupportedContextNote({.cu_budget = 256}, hardware));
    EXPECT_FALSE(ranker.nearestContextNote({}, hardware));
    EXPECT_FALSE(ranker.nearestContextNote({.cu_budget = 256}, hardware));

    // A budget outside the model's range is ranked at the nearest one in it
    // (every CU, for a v2 model), never by origami.
    EXPECT_FALSE(ranker.unsupportedContextNote({.cu_budget = 224}, hardware));
    auto const nearest = ranker.nearestContextNote({.cu_budget = 224}, hardware);
    ASSERT_TRUE(nearest);
    EXPECT_EQ(*nearest,
              "[TILEWRIGHT_DIAG TENSILELITE] arch=gfx950 cu_budget=224 schedule=0 n_cu=256: "
              "outside the model's CU range; ranked as cu_budget=0, the kernels "
              "launch with cu_budget=224");
    EXPECT_FALSE(ranker.nearestContextNote({.cu_budget = 224}, hardware));

    // A schedule the model does not list is ranked by origami.
    tilewright::ExecutionContext const dynamic{.cu_budget = 224,
                                               .schedule  = tilewright::Schedule::Dynamic};
    EXPECT_FALSE(ranker.nearestContextNote(dynamic, hardware));
    auto const note = ranker.unsupportedContextNote(dynamic, hardware);
    ASSERT_TRUE(note);
    EXPECT_EQ(*note,
              "[TILEWRIGHT_DIAG TENSILELITE] arch=gfx950 cu_budget=224 schedule=1 n_cu=256: the "
              "model does not support this context; ranking with origami");
    EXPECT_FALSE(ranker.unsupportedContextNote(dynamic, hardware));
    EXPECT_TRUE(ranker.unsupportedContextNote(
        {.cu_budget = 0, .schedule = tilewright::Schedule::Auto}, hardware));
}

namespace
{
    // Ranks dummyProblem() at SM count target `smCountTarget` on the gfx950
    // device with writeV2Model's model and exits: 0 when the model ranked it.
    [[noreturn]] void rankAtSmCountTargetAndExit(int smCountTarget)
    {
        bool ranked = false;
        {
            SolutionMap<ContractionSolution> solutions;
            addContestedKernels(solutions);

            ScratchDir                          dir("tensilelite_tilewright_note");
            ContractionProblemPredictionLibrary lib;
            loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
            auto problem = TensileLite::testing::dummyProblem();
            problem.setParams().setSmCountTarget(smCountTarget);
            ranked = lib.tilewright_ranker != nullptr
                     && !lib.tilewright_ranker->rank(problem, makeGfx950Device(), 3).empty();
        }
        std::exit(ranked ? 0 : 1);
    }
}

TEST(PredictionLibraryTest, TilewrightNotesACallItRanksAtAnotherCuBudget)
{
    // TensileLite reads TILEWRIGHT_DIAG once per process, so the call is
    // ranked in a newly started copy of this binary that sets it, and the
    // test matches the note rank() prints to the copy's stderr.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            setenv("TILEWRIGHT_DIAG", "1", 1);
            rankAtSmCountTargetAndExit(224);
        },
        ::testing::ExitedWithCode(0),
        "cu_budget=224 schedule=0 n_cu=256: outside the model's CU range; ranked as "
        "cu_budget=0, the kernels launch with cu_budget=224");
}

TEST(PredictionLibraryTest, TilewrightRanksTheTopKernelsOrEveryScoredKernel)
{
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_every_scored");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
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
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_full_budget");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    problem.setParams().setSmCountTarget(256);
    expectTilewrightOrder(lib, problem);
}

TEST(PredictionLibraryTest, OrigamiRanksCallsWithDynamicOrAutoStreamKScheduling)
{
    ScopedStreamK5ForceMode respectApi(nullptr);

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_streamk_mode");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
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
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_force_mode");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
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
    for(auto [setting, bank] :
        std::vector<std::pair<char const*, bool>>{{nullptr, false}, {"0", false}, {"1", true}})
    {
        SCOPED_TRACE(setting ? setting : "unset");
        ScopedTilewrightBank scoped(setting);

        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);
        ScratchDir                          dir("tensilelite_tilewright_bank_setting");
        ContractionProblemPredictionLibrary lib;
        loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
        ASSERT_NE(lib.tilewright_ranker, nullptr);
        // An MLREC_v2 model is not trained on the bank and reads no kernel
        // attributes.
        EXPECT_EQ(lib.tilewright_bank != nullptr, bank);
    }
}

TEST(PredictionLibraryTest, TilewrightBankFollowsThePoolInSolutionIndexOrder)
{
    ScopedTilewrightBank bankOn("1");

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);
    ScratchDir                          dir("tensilelite_tilewright_bank_order");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
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
        loadWithTilewright(dir, writeV2Model(dir), {13}, solutions, lib);
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
    ScopedTilewrightBank bankOn("1");

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(13, makeDot2Solution(13));
    ScratchDir dir("tensilelite_tilewright_bank_exact");
    auto       lib = std::make_shared<ContractionProblemPredictionLibrary>();
    loadWithTilewright(dir, writeV2Model(dir), {13}, solutions, *lib);
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
    ScopedTilewrightBank bankOn("1");

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);
    ScratchDir                          dir("tensilelite_tilewright_bank_devices");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
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
    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();

    std::vector<int> withoutBank;
    {
        ScopedTilewrightBank             bankOff("0");
        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);
        ScratchDir                          dir("tensilelite_tilewright_bank_off");
        ContractionProblemPredictionLibrary lib;
        loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
        withoutBank = solutionIndices(lib.findTopSolutions(problem, device, 3));
    }

    ScopedTilewrightBank             bankOn("1");
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);
    ScratchDir                          dir("tensilelite_tilewright_bank_empty");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, lib);
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
    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);
    ScratchDir dir("tensilelite_tilewright_row_order");
    auto       withModel = std::make_shared<ContractionProblemPredictionLibrary>();
    loadWithTilewright(dir, writeV2Model(dir), contestedTable, solutions, *withModel);
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

namespace
{
    // MLREC_v3 serving flag of a model trained on the Prediction kernels and
    // the bank.
    constexpr std::uint32_t TrainedOnBank = 0x40;

    // An MLREC_v3 model with `servingFlags` that reads `attributes`. Its one
    // cell scores every problem, with the launch model `launchModel`, by
    // default the one gfx950 models train with.
    std::filesystem::path writeV3Model(ScratchDir const&                             dir,
                                       std::uint32_t                                 servingFlags,
                                       std::vector<tilewright_test::AttrSpec> const& attributes,
                                       std::string launchModel = {})
    {
        tilewright_test::CellSpec cell;
        cell.label = "Large|Large|LargeK|Bnone";

        tilewright_test::ModelSpec spec;
        spec.arch             = "gfx950";
        spec.cells            = {cell};
        spec                  = tilewright_test::as_v3(spec, attributes);
        spec.v3_serving_flags = servingFlags;
        if(launchModel.empty())
            launchModel = TilewrightLaunch<ContractionSolution>("gfx950", {}).id();
        spec.provenance = {{"launch_model", launchModel}};

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

// A model trained on launch rule version 1 ranks with that version's plans
// whatever version gfx950 models train on now; one that records a launch model
// TensileLite does not serve it with is not used, and origami ranks.
TEST(PredictionLibraryTest, TilewrightPlansAModelUnderTheLaunchRuleVersionItRecords)
{
    EXPECT_EQ(TilewrightLaunch<ContractionSolution>::ruleVersion("gfx950"), 2);
    struct Case
    {
        std::string recorded;
        std::string served;
    };
    for(auto const& c : {Case{"hipblaslt:gfx950:1", "hipblaslt:gfx950:1"},
                         Case{"hipblaslt:gfx950:2", "hipblaslt:gfx950:2"},
                         Case{"hipblaslt:gfx950:3", ""},
                         Case{"hipblaslt:gfx942:1", ""}})
    {
        SCOPED_TRACE(c.recorded);
        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);
        ScratchDir                          dir("tensilelite_tilewright_rule_version");
        ContractionProblemPredictionLibrary lib;
        loadWithTilewright(
            dir, writeV3Model(dir, 0, {}, c.recorded), contestedTable, solutions, lib);
        if(c.served.empty())
        {
            EXPECT_EQ(lib.tilewright_ranker, nullptr);
            continue;
        }
        ASSERT_NE(lib.tilewright_ranker, nullptr);
        EXPECT_EQ(lib.tilewright_ranker->launchModel(), c.served);
        auto const top = solutionIndices(
            lib.findTopSolutions(TensileLite::testing::dummyProblem(), makeGfx950Device(), 4));
        EXPECT_EQ(top.size(), 3u);
    }
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

namespace
{
    // Loads keyTable, its kernels 31, 32 and 33 named `names`, with a
    // writeV3Model model and a blocklist of `blocked`; returns what the load
    // wrote to stderr.
    std::string loadNamed(ScratchDir const&                    dir,
                          std::vector<std::string> const&      names,
                          std::vector<std::string> const&      blocked,
                          ContractionProblemPredictionLibrary& lib,
                          std::uint32_t                        servingFlags = 0)
    {
        SolutionMap<ContractionSolution> solutions;
        addKeyKernels(solutions);
        for(size_t i = 0; i < keyTable.size(); i++)
            solutions.at(keyTable[i])->solutionName = names.at(i);
        {
            std::ofstream out(dir.path() / "TensileLibrary_Probe_gfx950.tilewright.blocklist");
            out << "# kernels that compute wrong results\n";
            for(auto const& name : blocked)
                out << name << "\n";
        }
        ::testing::internal::CaptureStderr();
        loadWithTilewright(dir, writeV3Model(dir, servingFlags, {}), keyTable, solutions, lib);
        return ::testing::internal::GetCapturedStderr();
    }
}

TEST(PredictionLibraryTest, TilewrightBlocklistNamesKernelsWhoseIaAndPkaTokensDiffer)
{
    // The first name comes from a build that names the kernels with PKA0 and
    // no IA token, the second, exact, from the build that loads them.
    ScratchDir                          dir("tensilelite_tilewright_blocklist_tokens");
    ContractionProblemPredictionLibrary lib;
    EXPECT_EQ(loadNamed(dir,
                        {"Cijk_probe_31_IA0_K1_PKA1_SU0",
                         "Cijk_probe_32_IA0_K1_PKA1_SU0",
                         "Cijk_probe_33_IA0_K1_PKA1_SU0"},
                        {"Cijk_probe_31_K1_PKA0_SU0", "Cijk_probe_32_IA0_K1_PKA1_SU0"},
                        lib),
              "");
    ASSERT_NE(lib.tilewright_ranker, nullptr);
    EXPECT_EQ(configPositions(*lib.tilewright_ranker), (std::vector<size_t>{2}));
    EXPECT_TRUE(lib.tilewright_ranker->unresolvedBlocklistNames().empty());

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    EXPECT_EQ(lib.tilewright_ranker->rank(problem, device, -1), (std::vector<size_t>{2}));
    EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), (std::vector<int>{33}));
}

TEST(PredictionLibraryTest, TilewrightBlocklistNameBlocksEveryKernelOfItsKey)
{
    // 31 and 32 differ in their PKA token alone: a name of either leaves both
    // out rather than offer the one it was written for.
    ScratchDir                          dir("tensilelite_tilewright_blocklist_variants");
    ContractionProblemPredictionLibrary lib;
    EXPECT_EQ(loadNamed(dir,
                        {"Cijk_probe_IA0_PKA0", "Cijk_probe_IA0_PKA1", "Cijk_other_IA0_PKA1"},
                        {"Cijk_probe_PKA1"},
                        lib),
              "");
    ASSERT_NE(lib.tilewright_ranker, nullptr);
    EXPECT_EQ(configPositions(*lib.tilewright_ranker), (std::vector<size_t>{2}));
}

TEST(PredictionLibraryTest, TilewrightWarnsAboutBlocklistNamesThatMatchNoKernel)
{
    // probe_31 names its kernel exactly; the others name no kernel of the
    // library, also without their IA and PKA tokens.
    ScratchDir                          dir("tensilelite_tilewright_blocklist_unresolved");
    ContractionProblemPredictionLibrary lib;
    std::string const                   warning
        = loadNamed(dir,
                    {"probe_31", "probe_32", "probe_33"},
                    {"probe_31", "probe_32_SU1", "probe_of_another_library_PKA0"},
                    lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);
    EXPECT_EQ(configPositions(*lib.tilewright_ranker), (std::vector<size_t>{1, 2}));
    EXPECT_EQ(lib.tilewright_ranker->unresolvedBlocklistNames(),
              (std::vector<std::string>{"probe_32_SU1", "probe_of_another_library_PKA0"}));
    EXPECT_NE(warning.find("hipBLASLt Warning: 2 of the 3 names in the tilewright blocklist of "
                           "TensileLibrary_Probe_gfx950 match no kernel"),
              std::string::npos)
        << warning;
    EXPECT_NE(warning.find("without their IA/PKA tokens (the first: probe_32_SU1)"),
              std::string::npos)
        << warning;
    EXPECT_EQ(std::count(warning.begin(), warning.end(), '\n'), 1) << warning;
}

TEST(PredictionLibraryTest, TilewrightWarnsAboutBlocklistNamesTheBankPoolLacks)
{
    // probe_21 names a bank kernel, so the library's own pool lacking it is
    // no warning when the bank pool ranks in its place; probe_gone names none.
    ScopedTilewrightBank                unset(nullptr);
    ScratchDir                          dir("tensilelite_tilewright_blocklist_bank_unresolved");
    ContractionProblemPredictionLibrary lib;
    EXPECT_EQ(loadNamed(dir,
                        {"probe_31", "probe_32", "probe_33"},
                        {"probe_31", "probe_21", "probe_gone"},
                        lib,
                        TrainedOnBank),
              "");
    ASSERT_NE(lib.tilewright_bank, nullptr);
    EXPECT_EQ(lib.tilewright_ranker->unresolvedBlocklistNames(),
              (std::vector<std::string>{"probe_21", "probe_gone"}));
    setBankSources(lib, {named(withTile(21, 128, 64)), named(withTile(22, 256, 64))});

    auto const device = makeGfx950Device();
    ::testing::internal::CaptureStderr();
    lib.findTopSolutions(TensileLite::testing::dummyProblem(), device, 5);
    std::string const warning = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(warning.find("1 of the 3 names in the tilewright blocklist of "
                           "TensileLibrary_Probe_gfx950"),
              std::string::npos)
        << warning;
    EXPECT_NE(warning.find("(the first: probe_gone)"), std::string::npos) << warning;
    auto const& pool = builtPool(lib, device);
    ASSERT_NE(pool.ranker, nullptr);
    EXPECT_EQ(pool.ranker->unresolvedBlocklistNames(), (std::vector<std::string>{"probe_gone"}));
    EXPECT_EQ(configPositions(*pool.ranker), (std::vector<size_t>{1, 2, 4}));
}

namespace
{
    // MLREC_v3 serving flag of a model trained with kernel guards.
    constexpr std::uint32_t TrainedWithGuards = 0x800;

    // An M x N x 64 GEMM of the probe kernels' types, neither operand
    // transposed.
    ContractionProblemGemm gemmOf(size_t m, size_t n)
    {
        auto problem
            = ContractionProblemGemm::GEMM(false, false, m, n, 64, m, 64, m, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }

    // `text` as the guards of TensileLibrary_Probe_gfx950 in `dir`.
    void writeGuards(ScratchDir const& dir, std::string const& text)
    {
        std::ofstream(dir.path() / "TensileLibrary_Probe_gfx950.tilewright.guards") << text;
    }
}

TEST(PredictionLibraryTest, TilewrightGuardsAreReadNextToTheIndex)
{
    ScratchDir        dir("tensilelite_tilewright_guards");
    std::string const logic = (dir.path() / "TensileLibrary_Probe_gfx950.dat").string();
    EXPECT_EQ(TilewrightRanker::loadGuards(logic), nullptr);

    writeGuards(dir,
                "# kernels offered only from a free size on\n\n   \nCijk_A M 4\n"
                "  Cijk_B\tN\t16 \r\nCijk_A N 2\n#Cijk_C M 4\n");
    auto const guards = TilewrightRanker::loadGuards(logic);
    ASSERT_NE(guards, nullptr);
    ASSERT_EQ(guards->size(), 3u);
    EXPECT_EQ((*guards)[0].name, "Cijk_A");
    EXPECT_EQ((*guards)[0].free, 0);
    EXPECT_EQ((*guards)[0].minimum, 4u);
    EXPECT_EQ((*guards)[1].name, "Cijk_B");
    EXPECT_EQ((*guards)[1].free, 1);
    EXPECT_EQ((*guards)[1].minimum, 16u);
    EXPECT_EQ((*guards)[2].name, "Cijk_A");
    EXPECT_EQ((*guards)[2].free, 1);
    EXPECT_EQ((*guards)[2].minimum, 2u);
    EXPECT_EQ(
        TilewrightRanker::loadGuards((dir.path() / "TensileLibrary_Other_gfx950.dat").string()),
        nullptr);

    for(std::string const malformed : {"Cijk_A M\n",
                                       "Cijk_A\n",
                                       "Cijk_A K 4\n",
                                       "Cijk_A m 4\n",
                                       "Cijk_A M 1\n",
                                       "Cijk_A M 0\n",
                                       "Cijk_A M -4\n",
                                       "Cijk_A M 4x\n",
                                       "Cijk_A M four\n",
                                       "Cijk_A M 1234567890\n",
                                       "Cijk_A M 4 8\n",
                                       "Cijk_A M 4\nCijk_A M 8\n",
                                       "Cijk_A_PKA1 N 4\nCijk_A_PKA0 N 8\n"})
    {
        writeGuards(dir, malformed);
        EXPECT_THROW(TilewrightRanker::loadGuards(logic), std::runtime_error) << malformed;
    }

    std::filesystem::create_directory(dir.path() / "TensileLibrary_Dir_gfx950.tilewright.guards");
    EXPECT_THROW(
        TilewrightRanker::loadGuards((dir.path() / "TensileLibrary_Dir_gfx950.dat").string()),
        std::runtime_error);
}

TEST(PredictionLibraryTest, TilewrightOffersAGuardedKernelOnlyAtOrAboveItsGuard)
{
    auto const device = makeGfx950Device();
    for(auto const& [free, guards] :
        std::vector<std::pair<int, std::string>>{{0, "probe_33 M 4\n"}, {1, "probe_33 N 4\n"}})
    {
        SCOPED_TRACE(guards);
        // The blocklist names 33, as the sidecar of a model trained with the
        // guard does: a model with the flag narrows it to the guard.
        SolutionMap<ContractionSolution>    solutions;
        ScratchDir                          dir("tensilelite_tilewright_guarded");
        ContractionProblemPredictionLibrary lib;
        writeGuards(dir, guards);
        loadBlocked(dir, {"probe_33"}, solutions, lib, TrainedWithGuards);
        ASSERT_NE(lib.tilewright_ranker, nullptr);
        auto const& ranker = *lib.tilewright_ranker;
        EXPECT_EQ(configPositions(ranker), (std::vector<size_t>{0, 1, 2}));
        EXPECT_FALSE(ranker.blocked(2));
        EXPECT_TRUE(ranker.guardsAny());
        EXPECT_TRUE(ranker.unresolvedBlocklistNames().empty());

        // A model without the flag never reads the guards: the blocklist
        // leaves 33 out whole.
        SolutionMap<ContractionSolution>    plainSolutions;
        ScratchDir                          plainDir("tensilelite_tilewright_guards_unflagged");
        ContractionProblemPredictionLibrary plain;
        writeGuards(plainDir, guards);
        loadBlocked(plainDir, {"probe_33"}, plainSolutions, plain);
        ASSERT_NE(plain.tilewright_ranker, nullptr);
        EXPECT_EQ(configPositions(*plain.tilewright_ranker), (std::vector<size_t>{0, 1}));
        EXPECT_TRUE(plain.tilewright_ranker->blocked(2));
        EXPECT_FALSE(plain.tilewright_ranker->guardsAny());
        EXPECT_EQ(plain.tilewright_ranker->guards(), nullptr);

        for(size_t m = 1; m <= 8; m++)
        {
            for(size_t n = 1; n <= 8; n++)
            {
                SCOPED_TRACE(std::to_string(m) + " x " + std::to_string(n));
                auto const problem  = gemmOf(m, n);
                bool const admitted = (free == 0 ? m : n) >= 4;
                EXPECT_EQ(ranker.guardDenies(2, m, n), !admitted);
                EXPECT_EQ(ranker.guardDenied(m, n), admitted ? 0u : 1u);
                auto const ranked = ranker.rank(problem, device, -1);
                EXPECT_EQ(ranked.size(), admitted ? 3u : 2u);
                EXPECT_EQ(std::count(ranked.begin(), ranked.end(), size_t{2}), admitted ? 1 : 0);
                for(int k = 1; k <= 3; k++)
                {
                    auto top = ranker.rank(problem, device, k);
                    EXPECT_EQ(top.size(), std::min<size_t>(k, ranked.size())) << k;
                    EXPECT_TRUE(std::equal(top.begin(), top.end(), ranked.begin())) << k;
                }
                // Below the guard the analytical ranking that fills the
                // remaining slots does not offer 33 either.
                auto const served = solutionIndices(lib.findTopSolutions(problem, device, 3));
                EXPECT_EQ(served.size(), admitted ? 3u : 2u);
                EXPECT_EQ(std::count(served.begin(), served.end(), 33), admitted ? 1 : 0);
                auto const whole = solutionIndices(plain.findTopSolutions(problem, device, 3));
                EXPECT_EQ(whole.size(), 2u);
                EXPECT_EQ(std::count(whole.begin(), whole.end(), 33), 0);
            }
        }
    }
}

TEST(PredictionLibraryTest, TilewrightServesTheNextMemberOfACandidateWhoseGuardDeniesItsFirst)
{
    // 31 and 32 are one candidate (keyTable); a guard of 31 hands the
    // candidate to 32 below it.
    auto const                          device = makeGfx950Device();
    SolutionMap<ContractionSolution>    plainSolutions;
    ScratchDir                          plainDir("tensilelite_tilewright_guard_unblocked");
    ContractionProblemPredictionLibrary plain;
    loadBlocked(plainDir, {}, plainSolutions, plain);
    ASSERT_NE(plain.tilewright_ranker, nullptr);
    EXPECT_FALSE(plain.tilewright_ranker->guardsAny());

    SolutionMap<ContractionSolution>    solutions;
    ScratchDir                          dir("tensilelite_tilewright_guarded_member");
    ContractionProblemPredictionLibrary lib;
    writeGuards(dir, "probe_31 M 4\n");
    loadBlocked(dir, {"probe_31"}, solutions, lib, TrainedWithGuards);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    for(size_t m = 1; m <= 8; m++)
    {
        SCOPED_TRACE(m);
        auto const problem = gemmOf(m, 256);
        auto const all     = plain.tilewright_ranker->rank(problem, device, -1);
        ASSERT_EQ(all.size(), 3u);
        auto const at31 = std::find(all.begin(), all.end(), size_t{0});
        ASSERT_NE(at31, all.end());
        EXPECT_EQ(*(at31 + 1), size_t{1}) << "32 follows 31, its candidate's first member";
        std::vector<size_t> expected;
        for(size_t position : all)
            if(m >= 4 || position != 0)
                expected.push_back(position);
        EXPECT_EQ(lib.tilewright_ranker->rank(problem, device, -1), expected);
        auto const top = solutionIndices(lib.findTopSolutions(problem, device, 1));
        ASSERT_EQ(top.size(), 1u);
        EXPECT_EQ(top[0], lib.solution_list[expected[0]].first);
        if(*all.begin() == 0)
            EXPECT_EQ(top[0], m >= 4 ? 31 : 32);
    }
}

TEST(PredictionLibraryTest, TilewrightAsksForAsManyMoreKernelsAsTheGuardsDeny)
{
    // Whichever kernel ranks first at M = 2, guarded on M: a request for k
    // kernels still gets the next k below the guard.
    auto const                          device = makeGfx950Device();
    SolutionMap<ContractionSolution>    plainSolutions;
    ScratchDir                          plainDir("tensilelite_tilewright_guard_depth_plain");
    ContractionProblemPredictionLibrary plain;
    loadBlocked(plainDir, {}, plainSolutions, plain);
    ASSERT_NE(plain.tilewright_ranker, nullptr);
    auto const first = plain.tilewright_ranker->rank(gemmOf(2, 256), device, 1);
    ASSERT_EQ(first.size(), 1u);
    std::string const top = "probe_" + std::to_string(plain.solution_list[first[0]].first);

    SolutionMap<ContractionSolution>    solutions;
    ScratchDir                          dir("tensilelite_tilewright_guard_depth");
    ContractionProblemPredictionLibrary lib;
    writeGuards(dir, top + " M 4\n");
    loadBlocked(dir, {top}, solutions, lib, TrainedWithGuards);
    ASSERT_NE(lib.tilewright_ranker, nullptr);
    for(size_t m : {1, 2, 3, 4, 8})
    {
        SCOPED_TRACE(m);
        auto const          problem = gemmOf(m, 256);
        auto const          all     = plain.tilewright_ranker->rank(problem, device, -1);
        std::vector<size_t> expected;
        for(size_t position : all)
            if(m >= 4 || position != first[0])
                expected.push_back(position);
        for(int k = 1; k <= 3; k++)
        {
            std::vector<size_t> const want(expected.begin(),
                                           expected.begin() + std::min<size_t>(k, expected.size()));
            EXPECT_EQ(lib.tilewright_ranker->rank(problem, device, k), want) << k;
        }
        auto const served = solutionIndices(lib.findTopSolutions(problem, device, 1));
        ASSERT_EQ(served.size(), 1u);
        EXPECT_EQ(served[0], lib.solution_list[expected[0]].first);
    }
}

TEST(PredictionLibraryTest, TilewrightRefusesAGuardedModelWithoutUsableGuards)
{
    // Missing, malformed and unreadable guards make a flagged model
    // unusable; a model without the flag never reads them, and its
    // blocklist leaves 33 out whole.
    for(std::string const guards : {std::string(), std::string("probe_33 M 4\nprobe_33 M 1\n")})
    {
        for(std::uint32_t const flags : {TrainedWithGuards, 0u})
        {
            SCOPED_TRACE(guards + " flags " + std::to_string(flags));
            SolutionMap<ContractionSolution>    solutions;
            ScratchDir                          dir("tensilelite_tilewright_bad_guards");
            ContractionProblemPredictionLibrary lib;
            if(!guards.empty())
                writeGuards(dir, guards);
            ::testing::internal::CaptureStderr();
            loadBlocked(dir, {"probe_33"}, solutions, lib, flags);
            std::string const warning = ::testing::internal::GetCapturedStderr();
            if(flags == 0)
            {
                ASSERT_NE(lib.tilewright_ranker, nullptr);
                EXPECT_EQ(warning, "");
                EXPECT_TRUE(lib.tilewright_ranker->blocked(2));
                EXPECT_FALSE(lib.tilewright_ranker->guardsAny());
                continue;
            }
            EXPECT_EQ(lib.tilewright_ranker, nullptr);
            EXPECT_NE(warning.find("ranking with origami"), std::string::npos) << warning;
            EXPECT_NE(warning.find(guards.empty()
                                       ? "trained with kernel guards and there is no "
                                         "TensileLibrary_Probe_gfx950.tilewright.guards"
                                       : "line 2 is not `<kernel name> <M or N> <minimum"),
                      std::string::npos)
                << warning;
        }
    }

    SolutionMap<ContractionSolution> solutions;
    ScratchDir                       dir("tensilelite_tilewright_guards_directory");
    std::filesystem::create_directory(dir.path() / "TensileLibrary_Probe_gfx950.tilewright.guards");
    ContractionProblemPredictionLibrary lib;
    ::testing::internal::CaptureStderr();
    loadBlocked(dir, {"probe_33"}, solutions, lib, TrainedWithGuards);
    std::string const warning = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(lib.tilewright_ranker, nullptr);
    EXPECT_NE(warning.find("cannot read the guards"), std::string::npos) << warning;
}

TEST(PredictionLibraryTest, TilewrightGuardsTheBankPoolToo)
{
    ScopedTilewrightBank                unset(nullptr);
    SolutionMap<ContractionSolution>    solutions;
    ScratchDir                          dir("tensilelite_tilewright_guarded_bank");
    ContractionProblemPredictionLibrary lib;
    writeGuards(dir, "probe_22 M 4\n");
    loadBlocked(dir, {"probe_22"}, solutions, lib, TrainedOnBank | TrainedWithGuards);
    ASSERT_NE(lib.tilewright_bank, nullptr);
    setBankSources(lib, {named(withTile(21, 128, 64)), named(withTile(22, 256, 64))});

    auto const device = makeGfx950Device();
    for(size_t m : {1, 2, 3, 4, 8})
    {
        SCOPED_TRACE(m);
        auto top = solutionIndices(lib.findTopSolutions(gemmOf(m, 256), device, 5));
        std::sort(top.begin(), top.end());
        EXPECT_EQ(top,
                  m >= 4 ? (std::vector<int>{21, 22, 31, 32, 33})
                         : (std::vector<int>{21, 31, 32, 33}));
    }
    auto const& pool = builtPool(lib, device);
    ASSERT_NE(pool.ranker, nullptr);
    EXPECT_EQ(pool.ranker->guards(), lib.tilewright_ranker->guards());
    EXPECT_EQ(configPositions(*pool.ranker), (std::vector<size_t>{0, 1, 2, 3, 4}));
    EXPECT_FALSE(pool.ranker->blocked(4));
    EXPECT_TRUE(pool.ranker->guardDenies(4, 3, 256));
    EXPECT_FALSE(pool.ranker->guardDenies(4, 4, 256));
}
#endif

#if defined(TILEWRIGHT_TEST_MODEL_WRITER)
namespace
{
    // An MLREC_v3 model of one cell that records `configId`, with the CU
    // floor `minEffectiveCus` when it is not 0.
    void writeTaggedModel(std::filesystem::path const& path,
                          std::string const&           configId,
                          std::uint32_t                minEffectiveCus = 0)
    {
        tilewright_test::CellSpec cell;
        cell.label = "Large|Large|LargeK|Bnone";

        tilewright_test::ModelSpec spec;
        spec.arch       = "gfx1250";
        spec.cells      = {cell};
        spec            = tilewright_test::as_v3(spec, {});
        spec.provenance = {{"config_id", configId}};
        if(minEffectiveCus != 0)
        {
            spec.v3_serving_flags  = 0x100;
            spec.min_effective_cus = minEffectiveCus;
        }
        auto const image = tilewright_test::write_model(spec);
        std::ofstream(path, std::ios::binary)
            .write(reinterpret_cast<char const*>(image.bytes.data()),
                   static_cast<std::streamsize>(image.bytes.size()));
    }
}

TEST(PredictionLibraryTest, TilewrightLoadsTheModelOfTheDevicesPciChipId)
{
    // Parts of an architecture that load the same library file can still get
    // different models from the index, by chip ID.
    ScratchDir        dir("tensilelite_tilewright_chip_id");
    std::string const stem  = "TensileLibrary_Probe_gfx1250";
    std::string const logic = (dir.path() / (stem + ".dat")).string();
    writeTaggedModel(dir.path() / "any.tilewright.bin", "any");
    writeTaggedModel(dir.path() / "chip.tilewright.bin", "chip");
    std::ofstream(dir.path() / "tilewright_index") << stem << "\tchip.tilewright.bin\tchip=0x1a2b\n"
                                                   << stem << "\tany.tilewright.bin\n";

    auto const configId = [&](std::uint32_t chipId) {
        auto const model = TilewrightRanker::loadModel(logic, chipId);
        return model ? tilewright::describe(*model).provenance.config_id : std::string("<none>");
    };
    EXPECT_EQ(configId(0), "any");
    EXPECT_EQ(configId(0x1A2B), "chip");
    EXPECT_EQ(configId(0x1A2C), "any");
    EXPECT_EQ(TilewrightRanker::indexChipId("gfx950"), 0u);

    // A row of the stem the engine cannot read leaves the library to origami.
    std::ofstream(dir.path() / "tilewright_index") << stem << "\tany.tilewright.bin\n"
                                                   << stem << "\tchip.tilewright.bin\tchip=1a2b\n";
    ::testing::internal::CaptureStderr();
    EXPECT_EQ(TilewrightRanker::loadModel(logic, 0x1A2B), nullptr);
    std::string const warning = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(warning.find("malformed row"), std::string::npos) << warning;
    EXPECT_NE(warning.find("ranking with origami"), std::string::npos) << warning;
}

TEST(PredictionLibraryTest, TilewrightNotesTheCuFloorOfADeviceSmallerThanIt)
{
    ScratchDir const            dir("tensilelite_tilewright_cu_floor");
    std::filesystem::path const path = dir.path() / "floor.tilewright.bin";
    writeTaggedModel(path, "floor", 64);
    std::string error;
    auto        model = tilewright::load_model(path.string(), &error);
    ASSERT_NE(model, nullptr) << error;
    EXPECT_EQ(tilewright::describe(*model).min_effective_cus, 64u);

    TilewrightRanker const             ranker(model, {}, nullptr, "gfx1250");
    tilewright::ExecutionContext const whole{};
    // A device with fewer CUs than the floor is ranked as a whole, never by
    // origami.
    EXPECT_FALSE(ranker.unsupportedContextNote(whole, {32, 163840, 4000000}).has_value());
    EXPECT_FALSE(ranker.nearestContextNote(whole, {64, 163840, 4000000}).has_value());
    auto const note = ranker.nearestContextNote(whole, {32, 163840, 4000000});
    ASSERT_TRUE(note.has_value());
    EXPECT_NE(note->find("n_cu=32 min_effective_cus=64"), std::string::npos) << *note;
    EXPECT_NE(note->find("ranked as cu_budget=0"), std::string::npos) << *note;
}
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
