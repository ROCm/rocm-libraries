// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Round trips through the msgpack writer. Every library is written, read back
// with the existing loader and written again: the two byte streams must match,
// and the reloaded library must hold the same solutions and tree and select
// the same solutions.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include <msgpack.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/FreeSizeLibrary.hpp>
#include <Tensile/GranularitySelectionLibrary.hpp>
#include <Tensile/MLPClassificationLibrary.hpp>
#include <Tensile/MapLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/MatchingLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/PropertyMatching.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <Tensile/msgpack/Writing.hpp>

#include "TestData.hpp"

namespace fs = std::filesystem;

using namespace TensileLite;

namespace
{
    using Gemm            = ContractionProblemGemm;
    using Library         = SolutionLibrary<Gemm, ContractionSolution>;
    using Master          = MasterSolutionLibrary<Gemm, ContractionSolution>;
    using Caching         = CachingLibrary<Gemm, ContractionSolution>;
    using Holder          = PlaceholderLibrary<Gemm, ContractionSolution>;
    using FreeSize        = ContractionProblemFreeSizeLibrary;
    using Granular        = ContractionGranularitySelectionLibrary;
    using MLP             = MLPClassificationLibrary<Gemm, ContractionSolution>;
    using ProblemMatching = ContractionProblemMatchingLibrary;

    using ProblemPredicatePtr = std::shared_ptr<Predicates::Predicate<Gemm>>;

    constexpr const char* kArch = "gfx9999";

    std::shared_ptr<Master> loadData(std::vector<uint8_t> const& bytes)
    {
        return std::dynamic_pointer_cast<Master>(
            LoadLibraryData<Gemm, ContractionSolution>(bytes));
    }

    /// Loads a msgpack `solutions` array through a library whose tree is one
    /// leaf naming `leafIndex`.
    std::shared_ptr<Master> loadSolutions(std::vector<uint8_t> const& solutions, int leafIndex)
    {
        msgpack::sbuffer                  doc;
        msgpack::packer<msgpack::sbuffer> packer(doc);
        packer.pack_map(2);
        packer.pack(std::string("solutions"));
        doc.write(reinterpret_cast<const char*>(solutions.data()), solutions.size());
        packer.pack(std::string("library"));
        packer.pack_map(2);
        packer.pack(std::string("type"));
        packer.pack(std::string("Single"));
        packer.pack(std::string("index"));
        packer.pack(leafIndex);

        auto data = reinterpret_cast<const uint8_t*>(doc.data());
        return loadData(std::vector<uint8_t>(data, data + doc.size()));
    }

    std::vector<uint8_t> readFile(fs::path const& path)
    {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if(!in.good())
            return {};
        std::vector<uint8_t> bytes(static_cast<size_t>(in.tellg()));
        in.seekg(0);
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        return bytes;
    }

    void writeFile(fs::path const& path, std::vector<uint8_t> const& bytes)
    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good()) << "cannot open " << path;
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(out.good()) << "write failed for " << path;
    }

    template <typename Object>
    std::string describe(std::shared_ptr<Predicates::Predicate<Object>> const& predicate);

    template <typename Object>
    std::string describe(std::vector<std::shared_ptr<Predicates::Predicate<Object>>> const& all)
    {
        std::string rv;
        for(auto const& predicate : all)
            rv += describe(predicate) + ", ";
        return rv;
    }

    /// Predicate::toString() prints the address of the predicate nested in a
    /// Not or IsSubclass; this spells it out instead.
    template <typename Object>
    std::string describe(std::shared_ptr<Predicates::Predicate<Object>> const& predicate)
    {
        if(auto all = std::dynamic_pointer_cast<Predicates::And<Object>>(predicate))
            return "And(" + describe(all->value) + ")";
        if(auto any = std::dynamic_pointer_cast<Predicates::Or<Object>>(predicate))
            return "Or(" + describe(any->value) + ")";
        if(auto inverse = std::dynamic_pointer_cast<Predicates::Not<Object>>(predicate))
            return "Not(" + describe(inverse->value) + ")";
        if constexpr(std::is_same<Object, Hardware>::value)
        {
            if(auto gpu
               = std::dynamic_pointer_cast<Predicates::IsSubclass<Hardware, AMDGPU>>(predicate))
                return "AMDGPU(" + describe(gpu->value) + ")";
        }
        return predicate->toString();
    }

    /// The tree as the loader built it: node types, row predicates, matching
    /// properties and the solution index of every leaf. Built from the
    /// in-memory objects, not from the writer.
    std::string outline(std::shared_ptr<Library> const& lib)
    {
        if(lib == nullptr)
            return "null";

        if(auto caching = std::dynamic_pointer_cast<Caching>(lib))
            return outline(caching->library());

        if(auto single = std::dynamic_pointer_cast<SingleContractionLibrary>(lib))
            return concatenate("Single(", single->solutionIndex(), ")");

        if(auto hardware = std::dynamic_pointer_cast<ContractionHardwareSelectionLibrary>(lib))
        {
            std::string rv = "Hardware[";
            for(auto const& row : hardware->rows)
                rv += concatenate(describe(row.first.value), " -> ", outline(row.second), "; ");
            return rv + "]";
        }

        if(auto problem = std::dynamic_pointer_cast<ContractionProblemSelectionLibrary>(lib))
        {
            std::string rv = "Problem[";
            for(auto const& row : problem->rows)
                rv += concatenate(describe(row.first.value), " -> ", outline(row.second), "; ");
            return rv + "]";
        }

        if(auto map = std::dynamic_pointer_cast<ContractionProblemMapLibrary>(lib))
        {
            std::map<std::string, std::string> sorted;
            for(auto const& entry : map->map)
                sorted[entry.first] = outline(entry.second);

            std::string rv = concatenate("ProblemMap(", map->property->toString(), ")[");
            for(auto const& entry : sorted)
                rv += concatenate(entry.first, " -> ", entry.second, "; ");
            return rv + "]";
        }

        if(auto matching = std::dynamic_pointer_cast<ProblemMatching>(lib))
        {
            std::string rv = concatenate("Matching(",
                                         matching->table->properties,
                                         ", ",
                                         matching->table->distanceType(),
                                         ")[");
            for(auto const& value : matching->table->GetAll())
                rv += outline(value) + "; ";
            return rv + "]";
        }

        if(auto holder = std::dynamic_pointer_cast<Holder>(lib))
            return concatenate("Placeholder(", holder->filePrefix, ")");

        if(auto freeSize = std::dynamic_pointer_cast<FreeSize>(lib))
        {
            std::string rv = "FreeSize[";
            for(auto const& entry : *freeSize->table)
                rv += outline(entry.value) + "; ";
            return rv + "]";
        }

        if(auto granular = std::dynamic_pointer_cast<Granular>(lib))
        {
            std::string rv = "Granularity[";
            for(auto const& entry : granular->solutions)
                rv += concatenate(entry.first, " ");
            rv += "|";
            for(auto const& entry : granular->exactMap)
            {
                for(size_t size : entry.first)
                    rv += concatenate(size, ",");
                rv += concatenate("->", entry.second, " ");
            }
            return rv + "]";
        }

        if(auto mlp = std::dynamic_pointer_cast<MLP>(lib))
        {
            std::string rv = "MLPClassification[";
            for(auto const& entry : mlp->solutionmap)
                rv += concatenate(entry.first, " ");
            auto const& net = *mlp->model;
            rv += concatenate("| dense ", net.dense.size_in, "x", net.dense.size_out);
            rv += concatenate(", blocks ", net.res_blocks.size());
            return rv + "]";
        }

        return lib->type();
    }

    void expectSameSolution(ContractionSolution const& a, ContractionSolution const& b)
    {
        SCOPED_TRACE(concatenate("solution ", a.index));

        EXPECT_EQ(a.index, b.index);
        EXPECT_EQ(a.solutionName, b.solutionName);
        EXPECT_EQ(a.kernelName, b.kernelName);
        EXPECT_EQ(a.libraryLogicIndex, b.libraryLogicIndex);

        EXPECT_EQ(describe(a.hardwarePredicate), describe(b.hardwarePredicate));
        EXPECT_EQ(describe(a.problemPredicate), describe(b.problemPredicate));
        EXPECT_EQ(describe(a.taskPredicate), describe(b.taskPredicate));

        auto const& sa = a.sizeMapping;
        auto const& sb = b.sizeMapping;
        EXPECT_EQ(sa.waveNum, sb.waveNum);
        EXPECT_EQ(sa.workGroupSize.x, sb.workGroupSize.x);
        EXPECT_EQ(sa.workGroupSize.y, sb.workGroupSize.y);
        EXPECT_EQ(sa.workGroupSize.z, sb.workGroupSize.z);
        EXPECT_EQ(sa.macroTile.x, sb.macroTile.x);
        EXPECT_EQ(sa.macroTile.y, sb.macroTile.y);
        EXPECT_EQ(sa.threadTile.x, sb.threadTile.x);
        EXPECT_EQ(sa.threadTile.y, sb.threadTile.y);
        EXPECT_EQ(sa.matrixInstruction, sb.matrixInstruction);
        EXPECT_EQ(sa.depthU, sb.depthU);
        EXPECT_EQ(sa.globalSplitU, sb.globalSplitU);
        EXPECT_EQ(sa.staggerU, sb.staggerU);
        EXPECT_EQ(sa.workGroupMapping, sb.workGroupMapping);
        EXPECT_EQ(sa.tileProcessingStrategy, sb.tileProcessingStrategy);
        EXPECT_EQ(sa.workAssignment, sb.workAssignment);
        EXPECT_EQ(sa.streamKAtomic, sb.streamKAtomic);
        EXPECT_EQ(sa.persistentKernel, sb.persistentKernel);
        EXPECT_EQ(sa.CUOccupancy, sb.CUOccupancy);
        EXPECT_EQ(sa.waveGroup, sb.waveGroup);

        auto const& pa = a.problemType;
        auto const& pb = b.problemType;
        EXPECT_EQ(pa.operationIdentifier, pb.operationIdentifier);
        EXPECT_EQ(pa.transA, pb.transA);
        EXPECT_EQ(pa.transB, pb.transB);
        EXPECT_EQ(pa.aType, pb.aType);
        EXPECT_EQ(pa.bType, pb.bType);
        EXPECT_EQ(pa.cType, pb.cType);
        EXPECT_EQ(pa.dType, pb.dType);
        EXPECT_EQ(pa.computeInputTypeA, pb.computeInputTypeA);
        EXPECT_EQ(pa.computeInputTypeB, pb.computeInputTypeB);
        EXPECT_EQ(pa.computeType, pb.computeType);
        EXPECT_EQ(pa.highPrecisionAccumulate, pb.highPrecisionAccumulate);
        EXPECT_EQ(pa.useBias, pb.useBias);
        EXPECT_EQ(pa.activationType, pb.activationType);
        EXPECT_EQ(pa.biasSrcWhiteList, pb.biasSrcWhiteList);
        EXPECT_EQ(pa.biasDataTypeWhiteList, pb.biasDataTypeWhiteList);
        EXPECT_EQ(pa.stridedBatched, pb.stridedBatched);
        EXPECT_EQ(pa.supportDeviceUserArguments, pb.supportDeviceUserArguments);

        auto const& ia = a.internalArgsSupport;
        auto const& ib = b.internalArgsSupport;
        EXPECT_EQ(ia.version, ib.version);
        EXPECT_EQ(ia.persistentLoopArgsVersion, ib.persistentLoopArgsVersion);
        EXPECT_EQ(ia.gsu, ib.gsu);
        EXPECT_EQ(ia.wgm, ib.wgm);
        EXPECT_EQ(ia.staggerU, ib.staggerU);
        EXPECT_EQ(ia.perTileExtraIters, ib.perTileExtraIters);
        EXPECT_EQ(ia.useUniversalArgs, ib.useUniversalArgs);
        EXPECT_EQ(ia.useSFC, ib.useSFC);

        EXPECT_EQ(a.ideals, b.ideals);
        EXPECT_EQ(a.linearModel.slope, b.linearModel.slope);
        EXPECT_EQ(a.linearModel.intercept, b.linearModel.intercept);
        EXPECT_EQ(a.linearModel.max, b.linearModel.max);

        EXPECT_EQ(MessagePackWriteSolutionData(a), MessagePackWriteSolutionData(b));
    }

    void expectSameSolutions(Master const& a, Master const& b)
    {
        ASSERT_EQ(a.solutions.size(), b.solutions.size());
        for(auto const& entry : a.solutions)
        {
            auto found = b.solutions.find(entry.first);
            ASSERT_NE(found, b.solutions.end()) << "solution " << entry.first << " was lost";
            expectSameSolution(*entry.second, *found->second);
        }
    }

    int selectedIndex(Library const& lib, Gemm const& problem, Hardware const& hardware)
    {
        auto solution = lib.findBestSolution(problem, hardware);
        return solution ? solution->index : -1;
    }

    /// Writes `lib`, loads the bytes and writes the reloaded library. The two
    /// written streams must be identical, and the reloaded library must carry
    /// the same solutions and tree.
    std::shared_ptr<Master> roundTrip(Master const& lib, std::vector<uint8_t>* written = nullptr)
    {
        auto first  = MessagePackWriteLibraryData(lib);
        auto loaded = loadData(first);
        EXPECT_NE(loaded, nullptr) << "the loader rejected the written library";
        if(loaded == nullptr)
            return nullptr;

        auto second = MessagePackWriteLibraryData(*loaded);
        EXPECT_EQ(first, second) << "rewriting a reloaded library changed its bytes";

        expectSameSolutions(lib, *loaded);
        EXPECT_EQ(outline(lib.library), outline(loaded->library));

        if(written)
            *written = std::move(first);
        return loaded;
    }

    // ------------------------------------------------------------------
    // Libraries built in code
    // ------------------------------------------------------------------

    Gemm gemm(size_t m, size_t n, size_t k)
    {
        auto problem = Gemm::GEMM(false, false, m, n, k, m, k, m, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Half);
        problem.setComputeInputTypeB(rocisa::DataType::Half);
        return problem;
    }

    std::string gemmOperation()
    {
        return gemm(64, 64, 64).operationIdentifier();
    }

    /// A per-solution predicate with nested values, shaped like the ones
    /// TensileCreateLibrary emits. Rejects K < 8 and K divisible by 3.
    ProblemPredicatePtr solutionPredicate()
    {
        using namespace Predicates::Contraction;

        BufferLoadCheckPacket packet{0, 8, 64, 128};

        auto operation   = std::make_shared<OperationIdentifierEqual>();
        operation->value = gemmOperation();

        return std::make_shared<Predicates::And<Gemm>>(std::initializer_list<ProblemPredicatePtr>{
            std::make_shared<BufferLoadOffsetLimitCheck>(packet),
            std::make_shared<Free0SizeMultiple>(0, 8),
            std::make_shared<Predicates::Not<Gemm>>(
                std::make_shared<Predicates::Or<Gemm>>(std::initializer_list<ProblemPredicatePtr>{
                    std::make_shared<SizeInRange>(3, Range{0, 8}),
                    std::make_shared<BoundSizeMultiple>(-1, 3)})),
            operation});
    }

    std::shared_ptr<ContractionSolution> makeSolution(int index)
    {
        auto s          = std::make_shared<ContractionSolution>();
        s->index        = index;
        s->solutionName = concatenate("Solution_", index);
        s->kernelName   = concatenate("Kernel_", index);

        s->problemPredicate = solutionPredicate();
        s->hardwarePredicate
            = std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(
                std::make_shared<Predicates::GPU::ProcessorEqual>(AMDGPU::Processor::gfx942));

        auto& m             = s->sizeMapping;
        m.waveNum           = 4;
        m.workGroupSize     = {256, 1, 1};
        m.threadTile        = {1, 1, 1};
        m.macroTile         = {size_t(64 * (index % 4 + 1)), 128, 1};
        m.matrixInstruction = {16, 16, 16, 1};
        m.waveGroup         = {2, 2};
        m.depthU            = 64;
        m.globalSplitU      = 1;
        m.workGroupMapping  = 8;

        s->problemType.operationIdentifier = gemmOperation();
        s->problemType.aType               = rocisa::DataType::Half;
        s->problemType.bType               = rocisa::DataType::Half;
        s->problemType.cType               = rocisa::DataType::Half;
        s->problemType.dType               = rocisa::DataType::Half;
        s->problemType.computeInputTypeA   = rocisa::DataType::Half;
        s->problemType.computeInputTypeB   = rocisa::DataType::Half;
        s->problemType.computeType         = rocisa::DataType::Float;
        s->problemType.biasSrcWhiteList    = {0, 1, 3};
        s->problemType.biasDataTypeWhiteList
            = {rocisa::DataType::Half, rocisa::DataType::Float};

        s->ideals[1024]          = 0.5;
        s->ideals[4096]          = -0.0;
        s->linearModel.slope     = 2.0;
        s->linearModel.intercept = -0.0;

        return s;
    }

    ProblemPredicatePtr sizeRange(size_t index, size_t min, size_t max)
    {
        return std::make_shared<Predicates::Contraction::SizeInRange>(
            index, Predicates::Contraction::Range{min, max});
    }

    std::shared_ptr<Library> single(Master& master, int index)
    {
        return std::make_shared<SingleContractionLibrary>(master.solutions.at(index));
    }

    /// A Euclidean matching table over (M, N, batch, K).
    std::shared_ptr<Library> matching(Master&                                      master,
                                      std::map<std::array<int64_t, 4>, int> const& entries)
    {
        using Key   = std::array<int64_t, 4>;
        using Table = Matching::DistanceMatchingTable<Key,
                                                      Gemm,
                                                      ProblemMatching::Element,
                                                      std::shared_ptr<ContractionSolution>,
                                                      Matching::EuclideanDistance<Key>>;

        auto freeSizeA   = std::make_shared<Contraction::FreeSizeA>();
        auto freeSizeB   = std::make_shared<Contraction::FreeSizeB>();
        auto batchSize   = std::make_shared<Contraction::BatchSize>();
        auto boundSize   = std::make_shared<Contraction::BoundSize>();
        freeSizeA->index = freeSizeB->index = batchSize->index = boundSize->index = 0;

        auto table        = std::make_shared<Table>();
        table->properties = {freeSizeA, freeSizeB, batchSize, boundSize};
        for(auto const& [key, index] : entries)
            table->table.push_back({key, single(master, index), double(index % 7)});

        auto lib   = std::make_shared<ProblemMatching>();
        lib->table = table;
        return lib;
    }

    AMDGPU mi300()
    {
        return AMDGPU(AMDGPU::Processor::gfx942, 304, "AMD Instinct MI300X");
    }

    /// A hardware row holding a ProblemMap, whose main entry holds Problem
    /// rows over single solutions and a matching table.
    std::shared_ptr<Master> buildSelectionLibrary()
    {
        auto master = std::make_shared<Master>();
        for(int index : {10, 11, 12, 13, 14, 15})
            master->solutions[index] = makeSolution(index);

        auto opId = gemmOperation();
        auto max  = std::numeric_limits<size_t>::max();

        auto rows = std::make_shared<ContractionProblemSelectionLibrary>();
        rows->rows.emplace_back(
            ProblemPredicate<Gemm>(std::make_shared<Predicates::And<Gemm>>(
                std::initializer_list<ProblemPredicatePtr>{sizeRange(0, 0, 1024),
                                                           sizeRange(1, 0, 1024)})),
            single(*master, 10));
        rows->rows.emplace_back(ProblemPredicate<Gemm>(sizeRange(0, 1024, 8192)),
                                matching(*master,
                                         {{{2048, 256, 1, 256}, 11},
                                          {{4096, 4096, 1, 1024}, 12},
                                          {{8000, 64, 1, 8192}, 13}}));
        rows->rows.emplace_back(ProblemPredicate<Gemm>(sizeRange(1, 1024, max)),
                                single(*master, 14));
        rows->rows.emplace_back(
            ProblemPredicate<Gemm>(std::make_shared<Predicates::True<Gemm>>()),
            single(*master, 15));

        auto map      = std::make_shared<ContractionProblemMapLibrary>();
        map->property = std::make_shared<Contraction::OperationIdentifier>();

        map->map[opId] = rows;
        map->map["Contraction_l_Alik_Bljk_Cijk_Dijk"] = single(*master, 10);

        std::shared_ptr<Predicates::Predicate<AMDGPU>> isGfx942
            = std::make_shared<Predicates::GPU::ProcessorEqual>(AMDGPU::Processor::gfx942);
        std::shared_ptr<Predicates::Predicate<AMDGPU>> notOneCU
            = std::make_shared<Predicates::Not<AMDGPU>>(
                std::make_shared<Predicates::GPU::CUCountEqual>(1));

        auto hardware = std::make_shared<ContractionHardwareSelectionLibrary>();
        hardware->rows.emplace_back(
            HardwarePredicate(std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(
                std::make_shared<Predicates::And<AMDGPU>>(
                    std::initializer_list<std::shared_ptr<Predicates::Predicate<AMDGPU>>>{
                        isGfx942, notOneCU}))),
            map);

        master->library = hardware;
        master->version = "test";
        return master;
    }

    struct TempDirTest : public ::testing::Test
    {
        fs::path tmpDir;

        void SetUp() override
        {
            tmpDir = fs::temp_directory_path()
                     / fs::path("hipblaslt-msgpack-writing-"
                                + std::to_string(static_cast<long long>(getpid())) + "-"
                                + ::testing::UnitTest::GetInstance()->current_test_info()->name());
            fs::create_directories(tmpDir);
        }

        void TearDown() override
        {
            std::error_code ec;
            fs::remove_all(tmpDir, ec);
        }
    };

    // ------------------------------------------------------------------
    // The committed KernelsLite fixture
    // ------------------------------------------------------------------

    std::shared_ptr<Master> loadKernelsLite()
    {
        auto fixture = TestData::Instance().file("KernelsLite.dat");
        if(!fs::is_regular_file(fixture))
        {
            ADD_FAILURE() << "KernelsLite.dat is part of the test data and was not found at "
                          << fixture;
            return nullptr;
        }
        return std::dynamic_pointer_cast<Master>(
            LoadLibraryFile<Gemm, ContractionSolution>(fixture.string()));
    }

    /// The compute input types `lib`'s solutions accept. The committed fixture
    /// predates the sixth TypesEqual slot, which then loads as the zero type,
    /// so they are read off the predicate rather than the problem type.
    std::pair<rocisa::DataType, rocisa::DataType> kernelsLiteComputeTypes(Master const& lib)
    {
        using TypesEqual = Predicates::Contraction::TypesEqual;

        auto all = std::dynamic_pointer_cast<Predicates::And<Gemm>>(
            lib.solutions.begin()->second->problemPredicate);
        if(all)
        {
            for(auto const& predicate : all->value)
                if(auto types = std::dynamic_pointer_cast<TypesEqual>(predicate))
                    return {types->value[4], types->value[5]};
        }
        return {rocisa::DataType::Half, rocisa::DataType::Half};
    }

    /// A TN Half x Float8_fnuz problem in the shape KernelsLite was tuned for.
    Gemm kernelsLiteProblem(Master const& lib, size_t m, size_t n, size_t k)
    {
        auto computeTypes = kernelsLiteComputeTypes(lib);
        auto problem      = Gemm::GEMM_Strides(true,
                                          false,
                                          rocisa::DataType::Half,
                                          rocisa::DataType::Float8_fnuz,
                                          rocisa::DataType::Half,
                                          rocisa::DataType::Half,
                                          m,
                                          n,
                                          k,
                                          1,
                                          k,
                                          m * k,
                                          k,
                                          k * n,
                                          m,
                                          m * n,
                                          m,
                                          m * n,
                                          2.0);
        problem.setComputeInputTypeA(computeTypes.first);
        problem.setComputeInputTypeB(computeTypes.second);
        problem.setHighPrecisionAccumulate(true);
        problem.setWorkspaceSize(size_t(1) << 30);
        return problem;
    }

} // namespace

TEST(MessagePackWritingTest, FixtureRoundTripIsStable)
{
    auto lib = loadKernelsLite();
    if(lib == nullptr)
    {
        if(!HasFailure())
            GTEST_SKIP() << "KernelsLite.dat did not load";
        return;
    }
    ASSERT_FALSE(lib->solutions.empty());

    auto reloaded = roundTrip(*lib);
    ASSERT_NE(reloaded, nullptr);

    auto row = outline(reloaded->library);
    EXPECT_NE(row.find("Problem[EqualityMatching"), std::string::npos) << row;
    EXPECT_NE(row.find("Equality)["), std::string::npos) << row;

    auto hardware = mi300();
    int  selected = 0;
    for(size_t n : {16, 2048, 8192})
    {
        for(size_t k : {1024, 8192, 65536})
        {
            auto problem = kernelsLiteProblem(*lib, 16, n, k);
            int  index   = selectedIndex(*lib, problem, hardware);
            EXPECT_EQ(selectedIndex(*reloaded, problem, hardware), index)
                << "n=" << n << " k=" << k;
            selected += index >= 0;
        }
    }
    EXPECT_EQ(selected, 9) << "every tuned size should select a solution";

    auto unmatched = kernelsLiteProblem(*lib, 16, 16, 1000);
    EXPECT_EQ(selectedIndex(*lib, unmatched, hardware), -1);
    EXPECT_EQ(selectedIndex(*reloaded, unmatched, hardware), -1);
}

TEST(MessagePackWritingTest, IndexedLibraryWritesAsItsEagerForm)
{
    auto lib = loadKernelsLite();
    if(lib == nullptr)
    {
        if(!HasFailure())
            GTEST_SKIP() << "KernelsLite.dat did not load";
        return;
    }

    auto eager   = MessagePackWriteLibraryData(*lib);
    auto indexed = loadData(MessagePackWriteIndexedLibraryData(*lib));
    ASSERT_NE(indexed, nullptr) << "solution slices from the writer did not load as a blob";
    ASSERT_NE(indexed->blobCache, nullptr);
    EXPECT_EQ(indexed->blobCache->materializedCount(), 0u);

    EXPECT_EQ(MessagePackWriteLibraryData(*indexed), eager)
        << "an indexed library must write the same document as its eager form";
    EXPECT_EQ(indexed->blobCache->materializedCount(), lib->solutions.size());
}

TEST(MessagePackWritingTest, ProblemLibraryWithSeveralRows)
{
    auto lib = buildSelectionLibrary();

    auto reloaded = roundTrip(*lib);
    ASSERT_NE(reloaded, nullptr);
    EXPECT_EQ(reloaded->version, "test");

    auto tree = outline(reloaded->library);
    EXPECT_NE(tree.find("Hardware[AMDGPU"), std::string::npos) << tree;
    EXPECT_NE(tree.find("Euclidean"), std::string::npos) << tree;

    auto const& solution = *reloaded->solutions.at(12);
    EXPECT_NE(solution.problemPredicate->toString().find("DU/MT0:64"), std::string::npos)
        << solution.problemPredicate->toString();
    EXPECT_TRUE(std::signbit(solution.ideals.at(4096)));
    EXPECT_TRUE(std::signbit(solution.linearModel.intercept));
    EXPECT_EQ(solution.linearModel.slope, 2.0);

    struct Case
    {
        size_t m, n, k;
        int    expected;
    };
    std::vector<Case> cases = {{512, 512, 64, 10},
                               {2048, 256, 256, 11},
                               {4096, 4000, 1024, 12},
                               {8000, 64, 8192, 13},
                               {512, 2048, 64, 14},
                               {16384, 64, 64, 15},
                               {512, 512, 4, -1}};

    auto hardware = mi300();
    for(auto const& c : cases)
    {
        auto problem = gemm(c.m, c.n, c.k);
        EXPECT_EQ(selectedIndex(*lib, problem, hardware), c.expected)
            << c.m << "x" << c.n << "x" << c.k;
        EXPECT_EQ(selectedIndex(*reloaded, problem, hardware), c.expected)
            << c.m << "x" << c.n << "x" << c.k;
    }

    AMDGPU other(AMDGPU::Processor::gfx950, 256, "AMD Instinct MI350");
    EXPECT_EQ(selectedIndex(*reloaded, gemm(512, 512, 64), other), -1);
}

TEST(MessagePackWritingTest, OtherNodeTypesRoundTrip)
{
    auto master = std::make_shared<Master>();
    for(int index : {1, 2, 3, 4})
        master->solutions[index] = makeSolution(index);

    auto freeSize   = std::make_shared<FreeSize>();
    freeSize->table = std::make_shared<FreeSize::Table>();
    for(int index : {2, 1})
        freeSize->table->push_back({single(*master, index)});

    auto granular = std::make_shared<Granular>();
    for(int index : {3, 4})
        granular->solutions[index] = master->solutions.at(index);
    granular->exactMap[{128, 128, 1, 64}] = 3;
    granular->exactMap[{4096, 64, 2, 64}] = 4;

    auto rows = std::make_shared<ContractionProblemSelectionLibrary>();
    rows->rows.emplace_back(ProblemPredicate<Gemm>(sizeRange(0, 0, 100)), freeSize);
    rows->rows.emplace_back(
        ProblemPredicate<Gemm>(std::make_shared<Predicates::True<Gemm>>()), granular);
    master->library = rows;

    auto reloaded = roundTrip(*master);
    ASSERT_NE(reloaded, nullptr);

    auto tree = outline(reloaded->library);
    EXPECT_NE(tree.find("FreeSize[Single(2); Single(1); ]"), std::string::npos) << tree;
    EXPECT_NE(tree.find("Granularity[3 4 |128,128,1,64,->3 4096,64,2,64,->4 ]"),
              std::string::npos)
        << tree;
}

TEST(MessagePackWritingTest, MLPClassificationRoundTrips)
{
    std::vector<float> mean = {0.5f, -1.0f, 2.0f}, scale = {1.0f, 0.25f, 3.0f};
    std::vector<float> linear1(12, 0.125f), linear1Bias(4, -0.375f);
    std::vector<float> linear2(16, 1.0f), linear2Bias(4, 0.5f);
    std::vector<float> res(12, -2.0f), resBias(4, 0.0f);
    std::vector<float> dense = {1, 2, 3, 4, 5, 6, 7, 8}, denseBias = {0.1f, 0.2f};

    auto lib = buildSelectionLibrary();
    std::vector<std::shared_ptr<ContractionSolution>> solutions
        = {lib->solutions.at(10), lib->solutions.at(13)};
    auto solutionBytes = MessagePackWriteSolutionListData(solutions);

    msgpack::sbuffer                  doc;
    msgpack::packer<msgpack::sbuffer> packer(doc);
    auto layer = [&](std::vector<float> const& weight, std::vector<float> const& bias) {
        packer.pack_map(2);
        packer.pack(std::string("weight"));
        packer.pack(weight);
        packer.pack(std::string("bias"));
        packer.pack(bias);
    };
    packer.pack_map(2);
    packer.pack(std::string("solutions"));
    doc.write(reinterpret_cast<const char*>(solutionBytes.data()), solutionBytes.size());
    packer.pack(std::string("library"));
    packer.pack_map(3);
    packer.pack(std::string("type"));
    packer.pack(std::string("MLPClassification"));
    packer.pack(std::string("table"));
    packer.pack(std::vector<int>{10, 13});
    packer.pack(std::string("mlp"));
    packer.pack_map(3);
    packer.pack(std::string("scaler"));
    packer.pack_map(2);
    packer.pack(std::string("mean"));
    packer.pack(mean);
    packer.pack(std::string("scale"));
    packer.pack(scale);
    packer.pack(std::string("res_blocks"));
    packer.pack_array(1);
    packer.pack_map(3);
    packer.pack(std::string("linear1"));
    layer(linear1, linear1Bias);
    packer.pack(std::string("linear2"));
    layer(linear2, linear2Bias);
    packer.pack(std::string("res"));
    layer(res, resBias);
    packer.pack(std::string("dense"));
    layer(dense, denseBias);

    auto data   = reinterpret_cast<const uint8_t*>(doc.data());
    auto master = loadData(std::vector<uint8_t>(data, data + doc.size()));
    ASSERT_NE(master, nullptr) << "the hand-written MLP library did not load";

    std::vector<uint8_t> written;
    auto                 reloaded = roundTrip(*master, &written);
    ASSERT_NE(reloaded, nullptr);

    auto tree = outline(reloaded->library);
    EXPECT_EQ(tree, "MLPClassification[10 13 | dense 4x2, blocks 1]");

    auto net = std::dynamic_pointer_cast<MLP>(
                   std::dynamic_pointer_cast<Caching>(reloaded->library)->library())
                   ->model;
    EXPECT_EQ(net->scaler.mean, mean);
    EXPECT_EQ(net->scaler.scale, scale);
    EXPECT_EQ(net->dense.B, denseBias);
    EXPECT_EQ(net->res_blocks.at(0).linear1.B, linear1Bias);

    // Weights live behind a type private to the library; check them in the
    // written document.
    auto handle = msgpack::unpack(reinterpret_cast<const char*>(written.data()), written.size());
    std::unordered_map<std::string, msgpack::object> top, node, mlp, layerMap, block;
    Serialization::objectToMap(handle.get(), top);
    Serialization::objectToMap(top.at("library"), node);
    Serialization::objectToMap(node.at("mlp"), mlp);
    Serialization::objectToMap(mlp.at("dense"), layerMap);
    EXPECT_EQ(layerMap.at("weight").as<std::vector<float>>(), dense);
    Serialization::objectToMap(mlp.at("res_blocks").via.array.ptr[0], block);
    layerMap.clear();
    Serialization::objectToMap(block.at("linear2"), layerMap);
    EXPECT_EQ(layerMap.at("weight").as<std::vector<float>>(), linear2);

    EXPECT_EQ(MessagePackWriteLibraryData(*master), written)
        << "writing must not modify the library";
}

using MessagePackWritingLazyTest = TempDirTest;

namespace
{
    std::vector<std::string> mapKeys(msgpack::object const& object)
    {
        std::vector<std::string> keys;
        keys.reserve(object.via.map.size);
        for(uint32_t i = 0; i < object.via.map.size; ++i)
        {
            std::string key;
            object.via.map.ptr[i].key.convert(key);
            keys.push_back(std::move(key));
        }
        return keys;
    }

    msgpack::object const& mapValue(msgpack::object const& object, std::string const& key)
    {
        for(uint32_t i = 0; i < object.via.map.size; ++i)
        {
            std::string found;
            object.via.map.ptr[i].key.convert(found);
            if(found == key)
                return object.via.map.ptr[i].val;
        }
        ADD_FAILURE() << "missing key " << key;
        return object.via.map.ptr[0].val;
    }
} // namespace

TEST_F(MessagePackWritingLazyTest, IndexedDocumentKeepsFormatVersion2)
{
    auto lib = buildSelectionLibrary();
    ASSERT_NE(lib, nullptr);

    auto bytes  = MessagePackWriteIndexedLibraryData(*lib);
    auto handle = msgpack::unpack(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    auto const& doc = handle.get();
    ASSERT_EQ(doc.type, msgpack::type::object_type::MAP);

    EXPECT_EQ(mapKeys(doc),
              (std::vector<std::string>{"format_version",
                                        "version",
                                        "solutions_index",
                                        "solutions_blob",
                                        "library"}));
    EXPECT_EQ(mapValue(doc, "format_version").as<int>(), 2);

    auto table = mapValue(doc, "solutions_index").as<std::vector<int64_t>>();
    ASSERT_EQ(table.size() % 3, 0u);
    ASSERT_EQ(table.size() / 3, lib->solutions.size());

    auto const& blobObj = mapValue(doc, "solutions_blob");
    ASSERT_EQ(blobObj.type, msgpack::type::object_type::BIN);
    std::vector<uint8_t> blob(blobObj.via.bin.ptr, blobObj.via.bin.ptr + blobObj.via.bin.size);

    std::vector<int64_t> indices;
    size_t               cursor = 0;
    for(size_t i = 0; i < table.size(); i += 3)
    {
        indices.push_back(table[i]);
        auto offset = static_cast<size_t>(table[i + 1]);
        auto length = static_cast<size_t>(table[i + 2]);
        EXPECT_EQ(offset, cursor);
        ASSERT_LE(offset + length, blob.size());
        std::vector<uint8_t> slice(blob.begin() + static_cast<std::ptrdiff_t>(offset),
                                   blob.begin() + static_cast<std::ptrdiff_t>(offset + length));
        EXPECT_EQ(slice, MessagePackWriteSolutionData(*lib->solutions.at(static_cast<int>(table[i]))));
        cursor += length;
    }
    EXPECT_EQ(cursor, blob.size());
    EXPECT_TRUE(std::is_sorted(indices.begin(), indices.end()));

    auto path = tmpDir / "library.dat";
    writeFile(path, bytes);

    auto loaded = std::dynamic_pointer_cast<Master>(
        LoadLibraryFile<Gemm, ContractionSolution>(path.string()));
    ASSERT_NE(loaded, nullptr);
    ASSERT_NE(loaded->blobCache, nullptr);
    EXPECT_EQ(loaded->version, "test");
    EXPECT_TRUE(loaded->solutions.empty());
    EXPECT_EQ(loaded->blobCache->size(), lib->solutions.size());
    EXPECT_EQ(loaded->blobCache->materializedCount(), 0u);

    auto solution = loaded->blobCache->get(12);
    ASSERT_NE(solution, nullptr);
    EXPECT_EQ(solution->index, 12);
    EXPECT_EQ(loaded->blobCache->materializedCount(), 1u);

    auto bare     = buildSelectionLibrary();
    bare->version = "";
    auto eagerBare = MessagePackWriteLibraryData(*bare);
    auto eagerHandle
        = msgpack::unpack(reinterpret_cast<const char*>(eagerBare.data()), eagerBare.size());
    auto eagerKeys = mapKeys(eagerHandle.get());
    // Both keys carry their absent value here, so the optional mapping with a
    // default omits them rather than writing the default back out.
    EXPECT_EQ(std::find(eagerKeys.begin(), eagerKeys.end(), "version"), eagerKeys.end());
    EXPECT_EQ(std::find(eagerKeys.begin(), eagerKeys.end(), "format_version"), eagerKeys.end());

    auto bareBytes
        = MessagePackWriteIndexedLibraryData(*bare);
    auto bareHandle
        = msgpack::unpack(reinterpret_cast<const char*>(bareBytes.data()), bareBytes.size());
    EXPECT_EQ(mapKeys(bareHandle.get()),
              (std::vector<std::string>{
                  "format_version", "solutions_index", "solutions_blob", "library"}));
}

TEST_F(MessagePackWritingLazyTest, PlaceholderMasterWithMapping)
{
    const std::string small = std::string("TensileLibrary_small_") + kArch;
    const std::string large = std::string("TensileLibrary_large_") + kArch;

    auto source = buildSelectionLibrary();
    auto opId   = gemmOperation();
    auto max    = std::numeric_limits<size_t>::max();

    // Shards: each a master with its own solutions and tree.
    auto smallShard = std::make_shared<Master>();
    for(int index : {10, 11})
        smallShard->solutions[index] = source->solutions.at(index);
    auto smallRows = std::make_shared<ContractionProblemSelectionLibrary>();
    smallRows->rows.emplace_back(ProblemPredicate<Gemm>(sizeRange(1, 0, 1024)),
                                 single(*smallShard, 10));
    smallRows->rows.emplace_back(
        ProblemPredicate<Gemm>(std::make_shared<Predicates::True<Gemm>>()),
        single(*smallShard, 11));
    smallShard->library = smallRows;

    auto largeShard = std::make_shared<Master>();
    for(int index : {14, 15})
        largeShard->solutions[index] = source->solutions.at(index);
    largeShard->library
        = matching(*largeShard, {{{16384, 64, 1, 64}, 15}, {{2048, 4096, 1, 64}, 14}});

    // The lazy master holds no solutions; each placeholder names a shard.
    auto holderFor = [](std::string const& prefix) {
        auto holder        = std::make_shared<Holder>();
        holder->filePrefix = prefix;
        return holder;
    };
    auto lazyRows = std::make_shared<ContractionProblemSelectionLibrary>();
    lazyRows->rows.emplace_back(ProblemPredicate<Gemm>(sizeRange(0, 0, 1024)),
                                holderFor(small));
    lazyRows->rows.emplace_back(ProblemPredicate<Gemm>(sizeRange(0, 1024, max)),
                                holderFor(large));
    auto lazyMap       = std::make_shared<ContractionProblemMapLibrary>();
    lazyMap->property  = std::make_shared<Contraction::OperationIdentifier>();
    lazyMap->map[opId] = lazyRows;

    Master lazy;
    lazy.library = lazyMap;

    std::map<int, std::string> mapping = {{0, small}, {14, large}};

    auto masterPath  = tmpDir / (std::string("TensileLibrary_lazy_") + kArch + ".dat");
    auto mappingPath = tmpDir / (std::string("TensileLiteLibrary_lazy_") + kArch + "_Mapping.dat");
    auto masterBytes = MessagePackWriteLibraryData(lazy);
    writeFile(masterPath, masterBytes);
    writeFile(tmpDir / (small + ".dat"), MessagePackWriteLibraryData(*smallShard));
    writeFile(tmpDir / (large + ".dat"), MessagePackWriteLibraryData(*largeShard));
    writeFile(mappingPath, MessagePackWriteLibraryMapping(mapping));

    auto loadedMapping = LoadLibraryMapping(mappingPath.string());
    EXPECT_EQ(loadedMapping, mapping);
    EXPECT_EQ(MessagePackWriteLibraryMapping(loadedMapping), readFile(mappingPath));

    auto loaded = std::dynamic_pointer_cast<Master>(
        LoadLibraryFile<Gemm, ContractionSolution>(masterPath.string()));
    ASSERT_NE(loaded, nullptr);
    EXPECT_TRUE(loaded->solutions.empty());
    EXPECT_EQ(outline(loaded->library), outline(lazy.library));
    EXPECT_EQ(MessagePackWriteLibraryData(*loaded), masterBytes)
        << "a lazy master must rewrite to the same placeholders";

    ASSERT_TRUE(loaded->initLibraryMapping(masterPath.string()));
    for(int index : {10, 11, 14, 15})
    {
        auto solution = loaded->resolveSolutionByIndex(index);
        ASSERT_NE(solution, nullptr) << "index " << index << " did not resolve through its shard";
        expectSameSolution(*source->solutions.at(index), *solution);
    }

    auto hardware = mi300();
    EXPECT_EQ(selectedIndex(*loaded, gemm(512, 512, 64), hardware), 10);
    EXPECT_EQ(selectedIndex(*loaded, gemm(512, 2048, 64), hardware), 11);
    EXPECT_EQ(selectedIndex(*loaded, gemm(16000, 64, 64), hardware), 15);
    EXPECT_EQ(selectedIndex(*loaded, gemm(2048, 4000, 64), hardware), 14);
}

TEST(MessagePackWritingTest, SolutionsRoundTripOnTheirOwn)
{
    auto lib = buildSelectionLibrary();

    auto const& original = *lib->solutions.at(12);
    auto        bytes    = MessagePackWriteSolutionData(original);

    std::vector<uint8_t> array = {0x91}; // a msgpack array of one element
    array.insert(array.end(), bytes.begin(), bytes.end());

    auto one = loadSolutions(array, 12);
    ASSERT_NE(one, nullptr) << "a written solution is not a valid solutions element";
    ASSERT_EQ(one->solutions.size(), 1u);
    expectSameSolution(original, *one->solutions.at(12));

    std::vector<std::shared_ptr<ContractionSolution>> list;
    for(auto const& entry : lib->solutions)
        list.push_back(entry.second);
    auto listBytes = MessagePackWriteSolutionListData(list);

    auto all = loadSolutions(listBytes, 10);
    ASSERT_NE(all, nullptr) << "a written solution list is not a valid solutions array";
    ASSERT_EQ(all->solutions.size(), list.size());

    std::vector<std::shared_ptr<ContractionSolution>> listBack;
    for(auto const& solution : list)
    {
        listBack.push_back(all->solutions.at(solution->index));
        expectSameSolution(*solution, *listBack.back());
    }
    EXPECT_EQ(MessagePackWriteSolutionListData(listBack), listBytes);
}

TEST(MessagePackWritingTest, ReportsWhatCannotBeWritten)
{
    auto lib = buildSelectionLibrary();

    EXPECT_THROW(MessagePackWriteLibraryData(SingleContractionLibrary(lib->solutions.at(10))),
                 std::runtime_error);

    Master empty;
    try
    {
        MessagePackWriteLibraryData(empty);
        FAIL() << "a master without a library must not be written";
    }
    catch(std::runtime_error const& err)
    {
        EXPECT_NE(std::string(err.what()).find("library: Null pointer"), std::string::npos)
            << err.what();
    }

    Master unresolved;
    unresolved.solutions = lib->solutions;
    auto rows            = std::make_shared<ContractionProblemSelectionLibrary>();
    rows->rows.emplace_back(
        ProblemPredicate<Gemm>(std::make_shared<Predicates::True<Gemm>>()),
        std::make_shared<SingleContractionLibrary>());
    unresolved.library = rows;
    try
    {
        MessagePackWriteLibraryData(unresolved);
        FAIL() << "a leaf without a solution must not be written";
    }
    catch(std::runtime_error const& err)
    {
        EXPECT_NE(std::string(err.what()).find("library.rows[0].library: SingleSolutionLibrary"),
                  std::string::npos)
            << err.what();
    }

    std::vector<std::shared_ptr<ContractionSolution>> withNull = {lib->solutions.at(10), nullptr};
    EXPECT_THROW(MessagePackWriteSolutionListData(withNull), std::runtime_error);
}
