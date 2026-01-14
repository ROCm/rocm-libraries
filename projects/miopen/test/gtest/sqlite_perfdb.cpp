// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <miopen/conv/problem_description.hpp>
#include <miopen/sqlite_db.hpp>
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/process.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/filesystem.hpp>

#include <boost/thread.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace miopen;

namespace {
static std::optional<fs::path>& thread_logs_root()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::optional<fs::path> path{std::nullopt};
    return path;
}

static bool& full_set()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool full_set = false;
    return full_set;
}

class Random
{
public:
    Random(unsigned int seed = 0) : rng(seed), dist_positive(1) {}

    std::mt19937::result_type Next() { return dist(rng); }
    auto NextNonNegative() { return dist_non_negative(rng); }
    auto NextPositive() { return dist_positive(rng); }

private:
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist;
    std::uniform_int_distribution<> dist_non_negative;
    std::uniform_int_distribution<> dist_positive;
};

static Random& Rnd()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static Random rnd;
    return rnd;
}

struct ProblemData : SQLiteSerializable<ProblemData>
{
    conv::ProblemDescription prob;

    ProblemData() : ProblemData(Rnd()) {}
    ProblemData(Random& rnd)
    {
        const int n_inputs          = rnd.NextPositive();
        const int in_height         = rnd.NextPositive();
        const int in_width          = rnd.NextPositive();
        const int kernel_size_h     = rnd.NextPositive();
        const int kernel_size_w     = rnd.NextPositive();
        const int n_outputs         = rnd.NextPositive();
        const int batch_sz          = rnd.NextPositive();
        const int pad_h             = rnd.NextNonNegative();
        const int pad_w             = rnd.NextNonNegative();
        const int kernel_stride_h   = rnd.NextPositive();
        const int kernel_stride_w   = rnd.NextPositive();
        const int kernel_dilation_h = rnd.NextPositive();
        const int kernel_dilation_w = rnd.NextPositive();
        const int bias              = rnd.Next();

        const TensorDescriptor in        = {miopenFloat, {batch_sz, n_inputs, in_height, in_width}};
        const TensorDescriptor weights   = {miopenFloat, {1, 1, kernel_size_h, kernel_size_w}};
        const TensorDescriptor out       = {miopenFloat, {1, n_outputs, 1, 1}};
        const ConvolutionDescriptor conv = {{pad_h, pad_w},
                                            {kernel_stride_h, kernel_stride_w},
                                            {kernel_dilation_h, kernel_dilation_w}};

        prob = {in, weights, out, conv, conv::Direction::Forward, bias};
    }
    ProblemData(int i)
    {
        i += 1;

        const int n_inputs          = i;
        const int in_height         = i;
        const int in_width          = i;
        const int kernel_size_h     = i;
        const int kernel_size_w     = i;
        const int n_outputs         = i;
        const int batch_sz          = i;
        const int pad_h             = i;
        const int pad_w             = i;
        const int kernel_stride_h   = i;
        const int kernel_stride_w   = i;
        const int kernel_dilation_h = i;
        const int kernel_dilation_w = i;
        const int bias              = i;

        const TensorDescriptor in        = {miopenFloat, {batch_sz, n_inputs, in_height, in_width}};
        const TensorDescriptor weights   = {miopenFloat, {1, 1, kernel_size_h, kernel_size_w}};
        const TensorDescriptor out       = {miopenFloat, {1, n_outputs, 1, 1}};
        const ConvolutionDescriptor conv = {{pad_h, pad_w},
                                            {kernel_stride_h, kernel_stride_w},
                                            {kernel_dilation_h, kernel_dilation_w}};

        prob = {in, weights, out, conv, conv::Direction::Forward, bias};
    }

    static std::string table_name() { return "config"; }
    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        conv::ProblemDescription::Visit(self.prob, f);
    }
};

struct SolverData
{
    int x;
    int y;

    struct NoInit
    {
    };

    SolverData(NoInit) : x(0), y(0) {}
    SolverData(Random& rnd) : x(rnd.Next()), y(rnd.Next()) {}
    SolverData() : x(Rnd().Next()), y(Rnd().Next()) {}
    SolverData(int x_, int y_) : x(x_), y(y_) {}

    template <unsigned int seed>
    static SolverData Seeded()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static Random rnd(seed);
        return {static_cast<int>(rnd.Next()), static_cast<int>(rnd.Next())};
    }

    void Serialize(std::ostream& s) const
    {
        static const auto sep = ',';
        s << x << sep << y;
    }

    bool Deserialize(const std::string& s)
    {
        static const auto sep = ',';
        SolverData t(NoInit{});
        std::istringstream ss(s);

        const auto success = DeserializeField(ss, &t.x, sep) && DeserializeField(ss, &t.y, sep);

        if(!success)
            return false;

        *this = t;
        return true;
    }

    bool operator==(const SolverData& other) const { return x == other.x && y == other.y; }

private:
    static bool DeserializeField(std::istream& from, int* ret, char separator)
    {
        std::string part;

        if(!std::getline(from, part, separator))
            return false;

        const auto start = part.c_str();
        char* end;
        const auto value = std::strtol(start, &end, 10);

        if(start == end)
            return false;

        *ret = value;
        return true;
    }
};

std::ostream& operator<<(std::ostream& s, const SolverData& td)
{
    s << "x: " << td.x << ", y: " << td.y;
    return s;
}

#if defined(MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB)
// Helper function for multi-process test child processes
fs::path LockFilePath(const fs::path& db_path) { return db_path + ".test.lock"; }

static fs::path& exe_path()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static fs::path path = []() { return fs::canonical("/proc/self/exe"); }();
    return path;
}
#endif // MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB

} // namespace

// Test classes and TEST_P macros must be outside anonymous namespace per GTest guidelines
// See: https://github.com/ROCm/MIOpen/wiki/GTest-development
// Note: These classes can still access helper functions/types from the anonymous namespace above
class DbTest : public ::testing::TestWithParam<int>
{
public:
    DbTest() : temp_file("miopen.tests.perfdb"), db_inst{DbKinds::PerfDb, temp_file, false} {}

    virtual ~DbTest() {}

    void SetUp() override
    {
        // Reset internal environment values to ensure tests are order-agnostic
        // See: https://github.com/ROCm/MIOpen/wiki/GTest-development
        // Reset static variables that can be modified between tests:
        thread_logs_root() = std::nullopt; // Reset thread logs root path
        full_set()         = false;        // Reset full_set flag
        Rnd()              = Random(0);    // Reset Random seed for order-agnostic tests
    }

    // Public static helper functions for use by test helpers and test cases
    static const std::array<std::pair<std::string, SolverData>, 2>& common_data()
    {
        static const std::array<std::pair<std::string, SolverData>, 2> data{{
            {id1(), value1()},
            {id0(), value0()},
        }};

        return data;
    }

    static const ProblemData& key()
    {
        static const ProblemData p(0);
        return p;
    }
    static const SolverData& value0()
    {
        static const SolverData data(3, 4);
        return data;
    }

    static const SolverData& value1()
    {
        static const SolverData data(5, 6);
        return data;
    }

    static const SolverData& value2()
    {
        static const SolverData data(7, 8);
        return data;
    }

    static const std::string& id0()
    {
        static const std::string id0_ = "Solver0";
        return id0_;
    }
    static const std::string& id1()
    {
        static const std::string id1_ = "Solver1";
        return id1_;
    }
    static const std::string& id2()
    {
        static const std::string id2_ = "Solver2";
        return id2_;
    }
    static const std::string& missing_id()
    {
        static const std::string missing_id_ = "UnknownSolver";
        return missing_id_;
    }

    template <class TDb, class TKey, class TValue, size_t count>
    static void ValidateSingleEntry(TKey& key,
                                    const std::array<std::pair<std::string, TValue>, count> values,
                                    TDb db)
    {
        auto record = db.FindRecord(key);

        EXPECT_TRUE(record);

        for(const auto& id_value : values)
        {
            TValue read;
            EXPECT_TRUE(record->GetValues(id_value.first, read));
            EXPECT_EQ(id_value.second, read);
        }
    }

    template <class TKey, class TValue, size_t count>
    static void RawWrite(const fs::path& db_path,
                         const TKey& key,
                         const std::array<std::pair<std::string, TValue>, count> values)
    {
        SQLitePerfDb tmp_inst(DbKinds::PerfDb, db_path, false);
        for(const auto& id_values : values)
        {
            tmp_inst.UpdateUnsafe(key, id_values.first, id_values.second);
        }
    }

protected:
    TempFile temp_file;
    SQLitePerfDb db_inst;

    void ClearDb(SQLitePerfDb& db) const
    {
        db.sql.Exec("delete from config; delete from perf_db;");
    }

    void ResetDb() const { db_inst.sql.Exec("delete from config; delete from perf_db;"); }
};

// Typedef for test class name matching GTest naming convention
using CPU_SqlitePerfDb_NONE = DbTest;

// SchemaTest converted to GTest
TEST_P(CPU_SqlitePerfDb_NONE, Schema)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    // check if the config and perf_db tables exist
    SQLite::result_type res = db_inst.sql.Exec(
        // clang-format off
            "SELECT name, sql "
            "FROM sqlite_master "
            "WHERE type='table' "
            "AND name = 'config';"
        // clang-format on
    );
    EXPECT_EQ(res.size(), 1);
    res = db_inst.sql.Exec(
        // clang-format off
            "SELECT name, sql "
            "FROM sqlite_master "
            "WHERE type='table' "
            "AND name = 'perf_db';"
        // clang-format on
    );
    EXPECT_EQ(res.size(), 1);
    // TODO: check for indices
}

// DbFindTest converted to GTest
TEST_P(CPU_SqlitePerfDb_NONE, Find)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    ResetDb();

    const ProblemData p;
    db_inst.InsertConfig(p);

    auto no_rec = db_inst.FindRecord(p);
    EXPECT_FALSE(no_rec);

    auto id = db_inst.GetConfigIDs(p);
    const SolverData sol;
    std::ostringstream ss;
    sol.Serialize(ss);
    db_inst.sql.Exec(
        // clang-formagt off
        "INSERT OR IGNORE INTO perf_db(config, solver, params) "
        "VALUES( " +
        id + ", '" + id0() + "', '" + ss.str() + "');");
    // clang-fromat on

    auto sol_res = db_inst.FindRecord(p);
    EXPECT_TRUE(sol_res);
}

// DbOperationsTest converted to GTest
TEST_P(CPU_SqlitePerfDb_NONE, Operations)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Testing different db operations db..." << std::endl;

    ProblemData p;
    const SolverData to_be_rewritten(7, 8);

    {
        SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);

        EXPECT_TRUE(db.Update(p, id0(), to_be_rewritten));
        EXPECT_TRUE(db.Update(p, id1(), to_be_rewritten));

        // Rewritting existing value with other.
        EXPECT_TRUE(db.Update(p, id1(), value1()));

        // Rewritting existing value with same. In fact no DB manipulation should be performed
        // inside of store in such case.
        EXPECT_TRUE(db.Update(p, id1(), value1()));
    }

    {
        SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);

        // Rewriting existing value to store it to file.
        EXPECT_TRUE(db.Update(p, id0(), value0()));
    }

    {
        SolverData read0, read1, read_missing;
        const auto read_missing_cmp(read_missing);
        SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);

        // Loading by id not present in record should execute well but return false as nothing
        // was read.
        EXPECT_FALSE(db.Load(p, missing_id(), read_missing));

        // In such case value should not be changed.
        EXPECT_EQ(read_missing, read_missing_cmp);

        EXPECT_TRUE(db.Load(p, id0(), read0));
        EXPECT_TRUE(db.Load(p, id1(), read1));

        EXPECT_EQ(read0, value0());
        EXPECT_EQ(read1, value1());

        EXPECT_TRUE(db.Remove(p, id0()));

        read0 = read_missing_cmp;

        EXPECT_FALSE(db.Load(p, id0(), read0));
        EXPECT_TRUE(db.Load(p, id1(), read1));

        EXPECT_EQ(read0, read_missing_cmp);
        EXPECT_EQ(read1, value1());
    }

    {
        SolverData read0, read1;
        const auto read_missing_cmp(read0);
        SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);

        EXPECT_FALSE(db.Load(p, id0(), read0));
        EXPECT_TRUE(db.Load(p, id1(), read1));

        EXPECT_EQ(read0, read_missing_cmp);
        EXPECT_EQ(read1, value1());
    }
}

// DbParallelTest converted to GTest
TEST_P(CPU_SqlitePerfDb_NONE, Parallel)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Testing db for using two objects targeting one file existing in one scope..."
              << std::endl;

    ProblemData p;

    SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);
    EXPECT_TRUE(db.Update(p, id0(), value0()));

    {
        SQLitePerfDb db0(DbKinds::PerfDb, temp_file, false);
        SQLitePerfDb db1(DbKinds::PerfDb, temp_file, false);

        auto r0 = db0.FindRecord(p);
        auto r1 = db1.FindRecord(p);

        EXPECT_TRUE(r0);
        EXPECT_TRUE(r1);

        EXPECT_TRUE(r0->SetValues(id1(), value1()));
        EXPECT_TRUE(r1->SetValues(id2(), value2()));
    }

    const std::array<std::pair<std::string, SolverData>, 3> data{{
        {id0(), value0()},
        {id1(), value1()},
        {id2(), value2()},
    }};
    EXPECT_TRUE(db.Update(p, id1(), value1()));
    EXPECT_TRUE(db.Update(p, id2(), value2()));

    ValidateSingleEntry(p, data, SQLitePerfDb(DbKinds::PerfDb, temp_file, false));
}

class DBMultiThreadedTestWork
{
public:
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int threads_count;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int common_part_size;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int unique_part_size;
    static constexpr unsigned int ids_per_key      = 16;
    static constexpr unsigned int common_part_seed = 435345;

    static const std::vector<SolverData>& common_part()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        static const auto& ref = common_part_init();
        return ref;
    }

    static void Initialize() { (void)common_part(); }

    template <class TDbConstructor>
    static void
    WorkItem(unsigned int id, const TDbConstructor& db_constructor, const std::string& log_postfix)
    {
        RedirectLogs(id, log_postfix, [id, &db_constructor]() {
            CommonPart(db_constructor);
            UniquePart(id, db_constructor);
        });
    }

    template <class TDbConstructor>
    static void ReadWorkItem(unsigned int id,
                             const TDbConstructor& db_constructor,
                             const std::string& log_postfix)
    {
        RedirectLogs(id, log_postfix, [&db_constructor]() { ReadCommonPart(db_constructor); });
    }

    template <class TDbConstructor>
    static void FillForReading(const TDbConstructor& db_constructor)
    {
        CommonPartSection(0u, common_part_size, db_constructor);
    }

    template <class TDbConstructor>
    static void ValidateCommonPart(const TDbConstructor& db_constructor)
    {
        auto db       = db_constructor();
        const auto cp = common_part();

        for(unsigned int i = 0u; i < common_part_size; i++)
        {
            ProblemData p(static_cast<int>(i / ids_per_key));
            const auto id   = std::to_string(i % ids_per_key);
            const auto data = cp[i];
            SolverData read(SolverData::NoInit{});

            EXPECT_TRUE(db.Load(p, id, read));
            EXPECT_EQ(read, data);
        }
    }

private:
    template <class TWorker>
    static void RedirectLogs(unsigned int id, const std::string& log_postfix, const TWorker& worker)
    {
        std::ofstream log;
        std::ofstream log_err;
        std::streambuf *cout_buf = nullptr, *cerr_buf = nullptr;

        if(thread_logs_root())
        {
            fs::path out_path;
            if(thread_logs_root())
                out_path = *thread_logs_root();

            out_path /= "thread-" + std::to_string(id) + "_" + log_postfix + ".log";

            fs::path err_path;
            if(thread_logs_root())
                err_path = *thread_logs_root();

            err_path /= "thread-" + std::to_string(id) + "_" + log_postfix + ".err";

            fs::remove(out_path);
            fs::remove(err_path);

            log.open(out_path);
            log_err.open(err_path);
            cout_buf = std::cout.rdbuf();
            cerr_buf = std::cerr.rdbuf();
            std::cout.rdbuf(log.rdbuf());
            std::cerr.rdbuf(log_err.rdbuf());
        }

        worker();

        if(thread_logs_root())
        {
            std::cout.rdbuf(cout_buf);
            std::cerr.rdbuf(cerr_buf);
        }
    }

    template <class TDbConstructor>
    static void ReadCommonPart(const TDbConstructor& db_constructor)
    {
        std::cout << "Common part. Section with common db instance." << std::endl;
        {
            // auto db = db_constructor();
            // ReadCommonPartSection(0u, common_part_size / 2, [&db]() { return db; });
            ReadCommonPartSection(0u, common_part_size / 2, db_constructor);
        }

        std::cout << "Common part. Section with separate db instances." << std::endl;
        ReadCommonPartSection(common_part_size / 2, common_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void
    ReadCommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        const auto cp = common_part();

        for(unsigned int i = start; i < end; i++)
        {
            ProblemData p(static_cast<int>(i / ids_per_key));
            const auto id   = std::to_string(i % ids_per_key);
            const auto data = cp[i];
            SolverData read(SolverData::NoInit{});

            EXPECT_TRUE(db_getter().Load(p, id, read));
            EXPECT_EQ(read, data);
        }
    }

    template <class TDbConstructor>
    static void CommonPart(const TDbConstructor& db_constructor)
    {
        std::cout << "Common part. Section with common db instance." << std::endl;
        {
            CommonPartSection(0u, common_part_size / 2, db_constructor);
        }

        std::cout << "Common part. Section with separate db instances." << std::endl;
        CommonPartSection(common_part_size / 2, common_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void CommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        const auto cp = common_part();

        for(unsigned int i = start; i < end; i++)
        {
            ProblemData p(static_cast<int>(i / ids_per_key));
            // const auto key  = i / ids_per_key;
            const auto id   = i % ids_per_key;
            const auto data = cp[i];

            db_getter().Update(p, std::to_string(id), data);
        }
    }

    template <class TDbConstructor>
    static void UniquePart(unsigned int id, const TDbConstructor& db_constructor)
    {
        Random rnd(123123 + id);

        std::cout << "Unique part. Section with common db instance." << std::endl;
        {
            UniquePartSection(rnd, 0, unique_part_size / 2, db_constructor);
        }

        std::cout << "Unique part. Section with separate db instances." << std::endl;
        UniquePartSection(rnd, unique_part_size / 2, unique_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void
    UniquePartSection(Random& rnd, unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        for(auto i = start; i < end; i++)
        {
            auto id = LimitedRandom(rnd, ids_per_key + 1);
            SolverData data(rnd);
            ProblemData p;

            db_getter().Update(p, std::to_string(id), data);
        }
    }

    static std::mt19937::result_type LimitedRandom(Random& rnd, std::mt19937::result_type min)
    {
        std::mt19937::result_type key;

        do
            key = rnd.Next();
        while(key < min);

        return key;
    }

    static const std::vector<SolverData>& common_part_init()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static std::vector<SolverData> data(common_part_size, SolverData::NoInit{});

        for(auto i = 0u; i < common_part_size; i++)
            data[i] = SolverData::Seeded<common_part_seed>();

        return data;
    }
};

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::threads_count = 16;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::common_part_size = 16;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::unique_part_size = 16;

// DbMultiThreadedTest converted to GTest
TEST_P(CPU_SqlitePerfDb_NONE, MultiThreaded)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Testing db for multithreaded write access..." << std::endl;

    ResetDb();
    std::shared_mutex mutex;
    std::vector<std::thread> threads;

    std::cout << "Initializing test data..." << std::endl;
    DBMultiThreadedTestWork::Initialize();

    std::cout << "Launching test threads..." << std::endl;
    threads.reserve(DBMultiThreadedTestWork::threads_count);
    const auto c = [this]() { return SQLitePerfDb(DbKinds::PerfDb, temp_file, false); };

    {
        std::unique_lock<std::shared_mutex> lock(mutex);

        for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
        {
            threads.emplace_back([c, &mutex, i]() {
                std::shared_lock<std::shared_mutex> lock(mutex);
                DBMultiThreadedTestWork::WorkItem(i, c, "mt");
            });
        }
    }

    std::cout << "Waiting for test threads..." << std::endl;
    for(auto& thread : threads)
        thread.join();

    std::cout << "Validating results..." << std::endl;
    DBMultiThreadedTestWork::ValidateCommonPart(c);
    std::cout << "Validation passed..." << std::endl;
}
// DbMultiThreadedReadTest converted to GTest
TEST_P(CPU_SqlitePerfDb_NONE, MultiThreadedRead)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Testing db for multithreaded read access..." << std::endl;

    std::shared_mutex mutex;
    std::vector<std::thread> threads;

    std::cout << "Initializing test data..." << std::endl;
    const auto c = [this]() { return SQLitePerfDb(DbKinds::PerfDb, temp_file, false); };
    DBMultiThreadedTestWork::FillForReading(c);

    std::cout << "Launching test threads..." << std::endl;
    threads.reserve(DBMultiThreadedTestWork::threads_count);

    {
        std::unique_lock<std::shared_mutex> lock(mutex);

        for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
        {
            threads.emplace_back([c, &mutex, i]() {
                std::shared_lock<std::shared_mutex> lock(mutex);
                DBMultiThreadedTestWork::ReadWorkItem(i, c, "mt");
            });
        }
    }

    std::cout << "Waiting for test threads..." << std::endl;
    for(auto& thread : threads)
        thread.join();
}

#if defined(MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB)
// WorkItem function for multi-process test child processes
static void MultiProcessWorkItem(unsigned int id, const std::string& db_path, bool write)
{
    {
        auto& file_lock = LockFile::Get(LockFilePath(db_path).c_str());
        std::lock_guard<LockFile> lock(file_lock);
    }

    const auto c = [&db_path]() { return SQLitePerfDb(DbKinds::PerfDb, db_path, false); };

    if(write)
        DBMultiThreadedTestWork::WorkItem(id, c, "mp");
    else
        DBMultiThreadedTestWork::ReadWorkItem(id, c, "mp");
}
#endif // MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB

// DbMultiProcessTest converted to GTest
// Note: Multi-process tests are disabled by default as they require command-line argument parsing
// which is not standard GTest practice. Multi-threaded tests (MultiThreaded, MultiThreadedRead)
// provide similar concurrency testing within the same process.
// To enable multi-process tests, define MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB
#if defined(MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB)
TEST_P(CPU_SqlitePerfDb_NONE, MultiProcess)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Testing db for multiprocess write access..." << std::endl;

    ResetDb();
    std::vector<ProcessAsync> children{};
    const auto lock_file_path = LockFilePath(temp_file);

    std::cout << "Initializing test data..." << std::endl;
    DBMultiThreadedTestWork::Initialize();

    std::cout << "Launching test processes..." << std::endl;
    {
        auto& file_lock = LockFile::Get(lock_file_path.c_str());
        boost::shared_lock<LockFile> lock(file_lock);

        auto id = 0;
        for(auto i = 0; i < DBMultiThreadedTestWork::threads_count; ++i)
        {
            ProcessEnvironmentMap env;
            env[MIOPEN_MP_CHILD_ID_ENV]    = std::to_string(id++);
            env[MIOPEN_MP_CHILD_PATH_ENV]  = temp_file;
            env[MIOPEN_MP_CHILD_WRITE_ENV] = "1"; // Write mode

            if(thread_logs_root())
            {
                env[MIOPEN_MP_THREAD_LOGS_ROOT_ENV] = *thread_logs_root();
            }

            children.emplace_back(exe_path(), "", "", nullptr, env);
        }
    }

    std::cout << "Waiting for test processes..." << std::endl;
    for(auto&& child : children)
    {
        EXPECT_EQ(child.Wait(), 0);
    }

    fs::remove(lock_file_path);

    const auto c = [this]() { return SQLitePerfDb(DbKinds::PerfDb, temp_file, false); };

    std::cout << "Validating results..." << std::endl;
    DBMultiThreadedTestWork::ValidateCommonPart(c);
    std::cout << "Validation passed..." << std::endl;
}
#endif // MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB

// DbMultiProcessReadTest converted to GTest
#if defined(MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB)
TEST_P(CPU_SqlitePerfDb_NONE, MultiProcessRead)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Testing db for multiprocess read access..." << std::endl;

    std::vector<ProcessAsync> children{};

    const auto lock_file_path = LockFilePath(temp_file);

    std::cout << "Initializing test data..." << std::endl;
    const auto c = [this]() { return SQLitePerfDb(DbKinds::PerfDb, temp_file, false); };
    DBMultiThreadedTestWork::FillForReading(c);

    std::cout << "Launching test processes..." << std::endl;
    {
        auto& file_lock = LockFile::Get(lock_file_path.c_str());
        boost::shared_lock<LockFile> lock(file_lock);

        auto id = 0;
        for(auto i = 0; i < DBMultiThreadedTestWork::threads_count; ++i)
        {
            ProcessEnvironmentMap env;
            env[MIOPEN_MP_CHILD_ID_ENV]    = std::to_string(id++);
            env[MIOPEN_MP_CHILD_PATH_ENV]  = temp_file;
            env[MIOPEN_MP_CHILD_WRITE_ENV] = "0"; // Read mode

            if(thread_logs_root())
            {
                env[MIOPEN_MP_THREAD_LOGS_ROOT_ENV] = *thread_logs_root();
            }

            children.emplace_back(exe_path(), "", "", nullptr, env);
        }
    }

    std::cout << "Waiting for test processes..." << std::endl;
    for(auto&& child : children)
    {
        EXPECT_EQ(child.Wait(), 0);
    }

    fs::remove(lock_file_path);
}
#endif // MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB

INSTANTIATE_TEST_SUITE_P(Full, CPU_SqlitePerfDb_NONE, testing::Values(0));

class DbMultiFileTest : public DbTest
{
protected:
    const fs::path user_db_path = temp_file.Path() + ".user";

    void ResetDb() const
    {
        DbTest::ResetDb();
        // (void)std::ofstream(user_db_path);
    }

    // Helper methods for Operations test
    void PrepareDb() const;
    void UpdateTest() const;
    void LoadTest() const;
    void RemoveTest() const;
    void RemoveRecordTest() const;

    template <class TDb>
    void ValidateData(TDb& db, const SolverData& id1Value) const;
};

// Helper methods for DbMultiFileReadTest (template converted to separate test fixtures)
template <bool merge_records>
void DbMultiFileReadTestHelper()
{
    std::cout << "Running multifile read test";
    if(merge_records)
        std::cout << " with merge";
    std::cout << "..." << std::endl;

    TempFile temp_file("miopen.tests.perfdb");
    const fs::path user_db_path = temp_file.Path() + ".user";

    auto ResetDb = [&temp_file]() {
        SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);
        db.sql.Exec("delete from config; delete from perf_db;");
    };

    auto single_item_data = []() {
        static const std::array<std::pair<std::string, SolverData>, 1> data{
            {{DbTest::id0(), DbTest::value2()}}};
        return data;
    };

    auto MergedAndMissing = [&temp_file, &user_db_path, &single_item_data]() {
        DbTest::RawWrite(temp_file, DbTest::key(), DbTest::common_data());
        DbTest::RawWrite(user_db_path, DbTest::key(), single_item_data());

        static const std::array<std::pair<std::string, SolverData>, 2> merged_data{{
            {DbTest::id1(), DbTest::value1()},
            {DbTest::id0(), DbTest::value2()},
        }};

        MultiFileDb<SQLitePerfDb, SQLitePerfDb, merge_records> db(
            DbKinds::PerfDb, temp_file, user_db_path);
        if(merge_records)
            DbTest::ValidateSingleEntry(DbTest::key(), merged_data, std::move(db));
        else
            DbTest::ValidateSingleEntry(DbTest::key(), single_item_data(), std::move(db));

        MultiFileDb<SQLitePerfDb, SQLitePerfDb, merge_records> db1(
            DbKinds::PerfDb, temp_file, user_db_path);
        ProblemData p;
        auto record1 = db1.FindRecord(p);
        EXPECT_FALSE(record1);
    };

    auto ReadUser = [&temp_file, &user_db_path, &single_item_data]() {
        DbTest::RawWrite(user_db_path, DbTest::key(), single_item_data());
        DbTest::ValidateSingleEntry(DbTest::key(),
                                    single_item_data(),
                                    MultiFileDb<SQLitePerfDb, SQLitePerfDb, merge_records>(
                                        DbKinds::PerfDb, temp_file, user_db_path));
    };

    auto ReadInstalled = [&temp_file, &user_db_path, &single_item_data]() {
        DbTest::RawWrite(temp_file, DbTest::key(), single_item_data());
        DbTest::ValidateSingleEntry(DbTest::key(),
                                    single_item_data(),
                                    MultiFileDb<SQLitePerfDb, SQLitePerfDb, merge_records>(
                                        DbKinds::PerfDb, temp_file, user_db_path));
    };

    auto ReadConflict = [&temp_file, &ReadUser, &single_item_data]() {
        DbTest::RawWrite(temp_file, DbTest::key(), single_item_data());
        ReadUser();
    };

    ResetDb();
    MergedAndMissing();

    ResetDb();
    ReadUser();

    ResetDb();
    ReadInstalled();

    ResetDb();
    ReadConflict();
}

// Typedef for multi-file test class name matching GTest naming convention
using CPU_SqlitePerfDbMultiFile_NONE = DbMultiFileTest;

// DbMultiFileReadTest converted to GTest (with merge)
TEST_P(CPU_SqlitePerfDbMultiFile_NONE, ReadWithMerge)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    DbMultiFileReadTestHelper<true>();
}

// DbMultiFileReadTest converted to GTest (without merge)
#if !MIOPEN_DISABLE_USERDB
TEST_P(CPU_SqlitePerfDbMultiFile_NONE, ReadWithoutMerge)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    DbMultiFileReadTestHelper<false>();
}
#endif
// DbMultiFileWriteTest converted to GTest
TEST_P(CPU_SqlitePerfDbMultiFile_NONE, Write)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Running multifile write test..." << std::endl;

    ResetDb();

    {
        MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
        EXPECT_TRUE(db.StoreRecord(key(), id0(), value0()));
        EXPECT_TRUE(db.Update(key(), id1(), value1()));
    }
    EXPECT_FALSE(SQLitePerfDb(DbKinds::PerfDb, temp_file, false).FindRecord(key()));
    EXPECT_TRUE(SQLitePerfDb(DbKinds::PerfDb, user_db_path, false).FindRecord(key()));

    ValidateSingleEntry(
        key(),
        common_data(),
        MultiFileDb<SQLitePerfDb, SQLitePerfDb, true>(DbKinds::PerfDb, temp_file, user_db_path));
}

// DbMultiFileOperationsTest converted to GTest
TEST_P(CPU_SqlitePerfDbMultiFile_NONE, Operations)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    ResetDb();
    PrepareDb();
    UpdateTest();
    LoadTest();
    RemoveTest();
    RemoveRecordTest();
}

// Helper methods for DbMultiFileOperationsTest
void DbMultiFileTest::PrepareDb() const
{
    std::cout << "Running multifile operations test..." << std::endl;

    {
        SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);
        EXPECT_TRUE(db.StoreRecord(key(), id0(), value0()));
        EXPECT_TRUE(db.Update(key(), id1(), value2()));
    }
}

void DbMultiFileTest::UpdateTest() const
{
    std::cout << "Update test..." << std::endl;

    {
        MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
        EXPECT_TRUE(db.Update(key(), id1(), value1()));
    }

    {
        SQLitePerfDb db(DbKinds::PerfDb, user_db_path, false);
        SolverData read(SolverData::NoInit{});
        EXPECT_FALSE(db.Load(key(), id0(), read));
        EXPECT_TRUE(db.Load(key(), id1(), read));
        EXPECT_EQ(read, value1());
    }

    {
        SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);
        ValidateData(db, value2());
    }
}

void DbMultiFileTest::LoadTest() const
{
    std::cout << "Load test..." << std::endl;

    MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
    ValidateData(db, value1());
}

void DbMultiFileTest::RemoveTest() const
{
    std::cout << "Remove test..." << std::endl;

    MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
    EXPECT_TRUE(db.Remove(key(), id0()));
    EXPECT_TRUE(db.Remove(key(), id1()));

    ValidateData(db, value2());
}

void DbMultiFileTest::RemoveRecordTest() const
{
    std::cout << "Remove record test..." << std::endl;

    MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
    EXPECT_TRUE(db.Update(key(), id1(), value1()));
    EXPECT_TRUE(db.Remove(key(), id1()));

    ValidateData(db, value2());
}

template <class TDb>
void DbMultiFileTest::ValidateData(TDb& db, const SolverData& id1Value) const
{
    SolverData read(SolverData::NoInit{});
    EXPECT_TRUE(db.Load(key(), id0(), read));
    EXPECT_EQ(read, value0());
    EXPECT_TRUE(db.Load(key(), id1(), read));
    EXPECT_EQ(read, id1Value);
}

// DbMultiFileMultiThreadedReadTest converted to GTest
#if !MIOPEN_DISABLE_USERDB
TEST_P(CPU_SqlitePerfDbMultiFile_NONE, MultiThreadedRead)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Testing db for multifile multithreaded read access..." << std::endl;

    std::shared_mutex mutex;
    std::vector<std::thread> threads;

    std::cout << "Initializing test data..." << std::endl;
    const auto c = [this]() {
        return MultiFileDb<SQLitePerfDb, SQLitePerfDb, true>(
            DbKinds::PerfDb, temp_file, user_db_path);
    };
    ResetDb();
    DBMultiThreadedTestWork::FillForReading(c);

    std::cout << "Launching test threads..." << std::endl;
    threads.reserve(DBMultiThreadedTestWork::threads_count);

    {
        std::unique_lock<std::shared_mutex> lock(mutex);

        for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
        {
            threads.emplace_back([c, &mutex, i]() {
                std::shared_lock<std::shared_mutex> lock(mutex);
                DBMultiThreadedTestWork::ReadWorkItem(i, c, "mt");
            });
        }
    }

    std::cout << "Waiting for test threads..." << std::endl;
    for(auto& thread : threads)
        thread.join();
}
#endif

// DbMultiFileMultiThreadedTest converted to GTest
#if !MIOPEN_DISABLE_USERDB
TEST_P(CPU_SqlitePerfDbMultiFile_NONE, MultiThreaded)
{
    (void)GetParam(); // Parameter unused but required for TEST_P pattern
    std::cout << "Testing db for multifile multithreaded write access..." << std::endl;

    ResetDb();
    std::shared_mutex mutex;
    std::vector<std::thread> threads;

    std::cout << "Initializing test data..." << std::endl;
    DBMultiThreadedTestWork::Initialize();

    std::cout << "Launching test threads..." << std::endl;
    threads.reserve(DBMultiThreadedTestWork::threads_count);
    const auto c = [this]() {
        return MultiFileDb<SQLitePerfDb, SQLitePerfDb, true>(
            DbKinds::PerfDb, temp_file, user_db_path);
    };

    {
        std::unique_lock<std::shared_mutex> lock(mutex);

        for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
        {
            threads.emplace_back([c, &mutex, i]() {
                std::shared_lock<std::shared_mutex> lock(mutex);
                DBMultiThreadedTestWork::WorkItem(i, c, "mt");
            });
        }
    }

    std::cout << "Waiting for test threads..." << std::endl;
    for(auto& thread : threads)
        thread.join();

    std::cout << "Validating results..." << std::endl;
    DBMultiThreadedTestWork::ValidateCommonPart(c);
    std::cout << "Validation passed..." << std::endl;
}

INSTANTIATE_TEST_SUITE_P(Full, CPU_SqlitePerfDbMultiFile_NONE, testing::Values(0));
#endif

#if defined(MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB)
// Configuration constants for multi-process tests
namespace {
// Environment variable names for child process communication
constexpr const char* MIOPEN_MP_CHILD_ID_ENV         = "MIOPEN_MP_CHILD_ID";
constexpr const char* MIOPEN_MP_CHILD_PATH_ENV       = "MIOPEN_MP_CHILD_PATH";
constexpr const char* MIOPEN_MP_CHILD_WRITE_ENV      = "MIOPEN_MP_CHILD_WRITE";
constexpr const char* MIOPEN_MP_THREAD_LOGS_ROOT_ENV = "MIOPEN_MP_THREAD_LOGS_ROOT";

// Default configuration values (replaces --all flag behavior)
constexpr unsigned int MP_THREADS_COUNT    = 16;
constexpr unsigned int MP_COMMON_PART_SIZE = 16;
constexpr unsigned int MP_UNIQUE_PART_SIZE = 16;
} // namespace

// Check if we're being called as a child process for multi-process tests
// Uses environment variables instead of command-line arguments (more GTest-friendly)
static bool HandleChildProcessIfNeeded()
{
    const char* child_id_str = std::getenv(MIOPEN_MP_CHILD_ID_ENV);
    const char* db_path_str  = std::getenv(MIOPEN_MP_CHILD_PATH_ENV);
    const char* write_str    = std::getenv(MIOPEN_MP_CHILD_WRITE_ENV);

    if(child_id_str && db_path_str)
    {
        int child_id = std::stoi(child_id_str);
        std::string db_path(db_path_str);
        bool write_mode = (write_str && std::string(write_str) == "1");

        MultiProcessWorkItem(child_id, db_path, write_mode);
        return true; // Indicates we handled child process and should exit
    }

    return false; // Normal GTest execution
}

namespace {
class MultiProcessEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        if(HandleChildProcessIfNeeded())
        {
            // We handled a child process execution, exit now to avoid running GTests
            std::exit(0);
        }
    }
};

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
::testing::Environment* const sqlite_env =
    ::testing::AddGlobalTestEnvironment(new MultiProcessEnvironment);
} // namespace
#endif // MIOPEN_ENABLE_MULTIPROCESS_TEST_SQLITE_PERFDB
