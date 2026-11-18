// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "BenchmarkDatabaseSetup.hpp"

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/FusedA2A.hpp>

#ifdef TENSILELITE_HAVE_MSGPACK_WRITER
#include <Tensile/msgpack/Loading.hpp>
#include <msgpack.hpp>
#endif
#ifdef TENSILE_YAML
#include <Tensile/llvm/Writing.hpp>
#include <Tensile/llvm/YAML.hpp>
#endif

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

static std::shared_ptr<ContractionSolution> makeTestSolution(int index = 12)
{
    std::shared_ptr<ContractionSolution> solution = std::make_shared<ContractionSolution>();
    solution->index                               = index;
    solution->solutionName                        = "Solution_" + std::to_string(index);
    solution->kernelName                          = "Kernel_" + std::to_string(index);

    SizeMapping& mapping = solution->sizeMapping;
    mapping.waveNum           = 4;
    mapping.workGroupSize     = {256, 1, 1};
    mapping.threadTile        = {1, 1, 1};
    mapping.macroTile         = {128, 128, 1};
    mapping.matrixInstruction = {16, 16, 16, 1};
    mapping.waveGroup         = {2, 2};
    mapping.depthU            = 64;
    mapping.globalSplitU      = 1;
    mapping.workGroupMapping  = 8;

    solution->problemType.operationIdentifier = "Contraction_l_Ajlk_Blik_Cijk_Dijk";
    solution->problemType.aType               = rocisa::DataType::Half;
    solution->problemType.bType               = rocisa::DataType::Half;
    solution->problemType.cType               = rocisa::DataType::Half;
    solution->problemType.dType               = rocisa::DataType::Half;
    return solution;
}

TEST(BenchmarkDatabaseSetup, SolutionSpecUsesMsgpackWhenTheHostCanWriteIt)
{
    std::shared_ptr<ContractionSolution> solution = makeTestSolution();

    SolutionSpec spec = serializeSolution(*solution);
#ifdef TENSILELITE_HAVE_MSGPACK_WRITER
    EXPECT_EQ(spec.format, "msgpack");
#else
    EXPECT_EQ(spec.format, "yaml");
#endif
    EXPECT_FALSE(spec.bytes.empty());
    EXPECT_EQ(spec.bytes, serializeSolution(*solution).bytes);
}

#ifdef TENSILELITE_HAVE_MSGPACK_WRITER
TEST(BenchmarkDatabaseSetup, MsgpackSolutionSpecLoadsBackToTheSameSolution)
{
    std::shared_ptr<ContractionSolution> solution = makeTestSolution();

    SolutionSpec spec = serializeSolution(*solution);

    // The spec is an element of a library's solutions array.
    msgpack::sbuffer                  doc;
    msgpack::packer<msgpack::sbuffer> packer(doc);
    packer.pack_map(2);
    packer.pack(std::string("solutions"));
    packer.pack_array(1);
    doc.write(spec.bytes.data(), spec.bytes.size());
    packer.pack(std::string("library"));
    packer.pack_map(2);
    packer.pack(std::string("type"));
    packer.pack(std::string("Single"));
    packer.pack(std::string("index"));
    packer.pack(solution->index);

    uint8_t const* data = reinterpret_cast<uint8_t const*>(doc.data());
    std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> loaded
        = std::dynamic_pointer_cast<MasterSolutionLibrary<ContractionProblemGemm>>(
            MessagePackLoadLibraryData<ContractionProblemGemm, ContractionSolution>(
                std::vector<uint8_t>(data, data + doc.size())));
    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->solutions.size(), 1u);

    ContractionSolution const& back = *loaded->solutions.at(solution->index);
    EXPECT_EQ(back.index, solution->index);
    EXPECT_EQ(back.name(), solution->name());
    EXPECT_EQ(serializeSolution(back).bytes, spec.bytes);
}
#endif

#ifdef TENSILE_YAML
TEST(BenchmarkDatabaseSetup, YamlSolutionSpecLoadsBackToTheSameSolution)
{
    std::shared_ptr<ContractionSolution> solution = makeTestSolution();

    std::string text = LLVMWriteSolutionData(*solution);
    ASSERT_FALSE(text.empty());

    llvm::yaml::Input                    yin(text);
    std::shared_ptr<ContractionSolution> back;
    yin >> back;
    ASSERT_FALSE(yin.error()) << yin.error().message();
    ASSERT_NE(back, nullptr);
    EXPECT_EQ(back->index, solution->index);
    EXPECT_EQ(back->name(), solution->name());
    EXPECT_EQ(LLVMWriteSolutionData(*back), text);
}
#endif

// The options checkSupportedOptions reads, at values the client accepts. The keys are kept in
// sync with checkSupportedOptions by hand.
static po::variables_map supportedArgs()
{
    po::variables_map vm;
    vm["num-benchmarks"].value()       = 1;
    vm["fused-a2a"].value()            = FusedA2A::Disabled;
    vm["selection-only"].value()       = false;
    vm["icache-flush-args"].value()    = std::vector<bool>(1, false);
    return vm;
}

// Expects the options to be rejected with a message that contains `text`.
static void expectRejected(po::variables_map const& vm, std::string const& text)
{
    try
    {
        checkSupportedOptions(vm);
        FAIL() << "expected an exception";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find(text), std::string::npos) << error.what();
    }
}

TEST(BenchmarkDatabaseSetup, DefaultOptionsAreSupported)
{
    EXPECT_NO_THROW(checkSupportedOptions(supportedArgs()));
}

// Pairs the prediction does not select are recorded as NotSelected, so every threshold is valid.
TEST(BenchmarkDatabaseSetup, EveryPredictionThresholdIsSupported)
{
    for(double threshold : {0.0, 0.5, 1.0, 1.5, 2.0})
    {
        po::variables_map vm               = supportedArgs();
        vm["prediction-threshold"].value() = threshold;
        EXPECT_NO_THROW(checkSupportedOptions(vm)) << threshold;
    }
}

TEST(BenchmarkDatabaseSetup, SeveralBenchmarksAreRejected)
{
    po::variables_map vm         = supportedArgs();
    vm["num-benchmarks"].value() = 2;
    expectRejected(vm, "num-benchmarks");
}

TEST(BenchmarkDatabaseSetup, FusedA2AIsRejected)
{
    po::variables_map vm      = supportedArgs();
    vm["fused-a2a"].value()   = FusedA2A::GemmA2A;
    expectRejected(vm, "fused-a2a");
}

TEST(BenchmarkDatabaseSetup, SelectionOnlyIsRejected)
{
    po::variables_map vm         = supportedArgs();
    vm["selection-only"].value() = true;
    expectRejected(vm, "selection-only");
}

TEST(BenchmarkDatabaseSetup, SeveralICacheFlushValuesAreRejected)
{
    po::variables_map vm            = supportedArgs();
    vm["icache-flush-args"].value() = std::vector<bool>{false, true};
    expectRejected(vm, "icache-flush-args");
}

namespace
{
    // Unique temporary directory, removed with the files it holds.
    class TempDir
    {
    public:
        TempDir()
        {
            std::string pattern
                = (std::filesystem::temp_directory_path() / "tensilelite_setup_XXXXXX").string();
            char* created = mkdtemp(pattern.data());
            if(!created)
                throw std::runtime_error("cannot create temporary directory");
            m_dir = created;
        }

        ~TempDir()
        {
            std::error_code error;
            std::filesystem::remove_all(m_dir, error);
        }

        TempDir(TempDir const&)            = delete;
        TempDir& operator=(TempDir const&) = delete;

        std::string file(std::string const& name) const
        {
            return (m_dir / name).string();
        }

    private:
        std::filesystem::path m_dir;
    };
}

using Library = MasterSolutionLibrary<ContractionProblemGemm>;

// A library of the solutions with the given indices, in a form the iterators accept.
static std::shared_ptr<Library> makeLibrary(std::vector<int> const& indices)
{
    std::shared_ptr<Library> library = std::make_shared<Library>();
    for(int index : indices)
        library->solutions[index] = makeTestSolution(index);
    return library;
}

static po::variables_map openArgs(std::string const& path)
{
    po::variables_map vm                  = supportedArgs();
    vm["benchmark-db"].value()            = path;
    vm["performance-metric"].value()      = PerformanceMetric::CUEfficiency;
    vm["csv-export-extra-cols"].value()   = true;
    vm["csv-merge-same-problems"].value() = false;
    return vm;
}

static std::string metaValue(SqliteConnection& db, std::string const& key)
{
    SqliteStatement statement(db, "SELECT value FROM meta WHERE key = ?1");
    statement.bindText(1, key);
    EXPECT_TRUE(statement.step()) << key;
    return statement.columnText(0);
}

TEST(BenchmarkDatabaseSetup, EmptyDatabasePathOpensNothing)
{
    po::variables_map vm;
    vm["benchmark-db"].value() = std::string();

    std::shared_ptr<Library> library = makeLibrary({1});
    AllSolutionsIterator     iterator(library, std::shared_ptr<Hardware>{}, 2.0, -1, -1, false);
    EXPECT_EQ(openBenchmarkDatabase(vm, "", {0}, *library, iterator), nullptr);
}

TEST(BenchmarkDatabaseSetup, OpenStoresTheIteratedSolutionsAndTheCsvOptions)
{
    TempDir                  dir;
    std::string const        path    = dir.file("bench.db");
    std::shared_ptr<Library> library = makeLibrary({10, 11, 12, 13});
    // Solutions 11 and 12 only.
    AllSolutionsIterator iterator(library, std::shared_ptr<Hardware>{}, 2.0, 11, 2, false);

    {
        std::shared_ptr<BenchmarkDatabase> database = openBenchmarkDatabase(
            openArgs(path), "count=1\n", {0, 1}, *library, iterator);
        ASSERT_NE(database, nullptr);
    }

    SqliteConnection db(path);
    SqliteStatement  solutions(db, "SELECT library_index, name FROM solutions ORDER BY 1");
    ASSERT_TRUE(solutions.step());
    EXPECT_EQ(solutions.columnInt(0), 11);
    EXPECT_EQ(solutions.columnText(1), "Solution_11");
    ASSERT_TRUE(solutions.step());
    EXPECT_EQ(solutions.columnInt(0), 12);
    EXPECT_FALSE(solutions.step());

    std::ostringstream metric;
    metric << PerformanceMetric::CUEfficiency;
    EXPECT_EQ(metaValue(db, "performance_metric"), metric.str());
    EXPECT_EQ(metaValue(db, "csv_export_extra_cols"), "true");
    EXPECT_EQ(metaValue(db, "csv_merge_same_problems"), "false");

    SqliteStatement problems(db, "SELECT COUNT(*) FROM init_configs");
    ASSERT_TRUE(problems.step());
    EXPECT_EQ(problems.columnInt(0), 2);
}

TEST(BenchmarkDatabaseSetup, OpenRejectsAnIteratorThatIsNotAllSolutions)
{
    TempDir                  dir;
    std::shared_ptr<Library> library = makeLibrary({1});
    TopSolutionIterator iterator(library, std::shared_ptr<Hardware>{}, 2.0, 1, false, false);

    try
    {
        openBenchmarkDatabase(openArgs(dir.file("bench.db")), "", {0}, *library, iterator);
        FAIL() << "expected an exception";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("all-solutions"), std::string::npos)
            << error.what();
    }
    EXPECT_FALSE(std::filesystem::exists(dir.file("bench.db")));
}
