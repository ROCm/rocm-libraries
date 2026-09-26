// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests for the tuning file's row format and store: writing and
// reading rows, which problems a row serves, and which rows are trusted. They
// link the store directly, so they need neither a device nor the hooks
// hipblaslt-test uses.

#include "TuningCacheStore.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#ifdef WIN32
static int setenv(const char* name, const char* value, int)
{
    return _putenv_s(name, value);
}
static int unsetenv(const char* name)
{
    return _putenv_s(name, "");
}
#endif

using namespace TensileLite;

namespace
{
    const std::string kStamp = "abc1234";

    const TuningRowTypes kHalfTypes{"f16_r", "f16_r", "f16_r", "f16_r", "f32_r"};

    ProblemOverride halfKey(size_t m = 1024, size_t n = 512, size_t k = 1024)
    {
        ProblemOverride key;
        key.m           = m;
        key.n           = n;
        key.k           = k;
        key.batchSize   = 1;
        key.inputTypeA  = rocisa::DataType::Half;
        key.inputTypeB  = rocisa::DataType::Half;
        key.outputTypeC = rocisa::DataType::Half;
        key.outputTypeD = rocisa::DataType::Half;
        key.computeType = rocisa::DataType::Float;
        key.colStrideA  = m;
        key.colStrideB  = k;
        key.colStrideC  = m;
        key.colStrideD  = m;
        key.archName    = "gfx950:sramecc+:xnack-";
        key.cuCount     = 256;
        return key;
    }

    TuningSearch rankedSearch(int32_t maxCandidates)
    {
        TuningSearch search;
        search.allKernels     = false;
        search.maxCandidates  = maxCandidates;
        search.workspaceBytes = 32 << 20;
        search.coldIters      = 2;
        search.hotIters       = 5;
        search.flushICache    = true;
        search.rotatingMb     = 160;
        return search;
    }

    TunedEntry tunedEntry(int32_t             index,
                          const std::string&  kernel,
                          bool                complete = true,
                          int64_t             budgetMs = 0,
                          const TuningSearch& search   = rankedSearch(16))
    {
        TunedEntry entry;
        entry.solutionIndex = index;
        entry.kernelName    = kernel;
        entry.schemaVersion = TuningSchemaVersion::Current;
        entry.complete      = complete;
        entry.budgetMs      = budgetMs;
        entry.search        = search;
        entry.winnerTimeUs  = 12.5;
        return entry;
    }

    std::string tunedRow(const ProblemOverride& key, const TunedEntry& entry)
    {
        return formatTuningRow(key, kHalfTypes, entry, kStamp);
    }

    // A file as the writer leaves it: the build stamp, then each row.
    std::string fileOf(const std::vector<std::string>& rows, const std::string& stamp = kStamp)
    {
        std::string text = "Git Version: " + stamp + "\n";
        for(const auto& row : rows)
            text += row;
        return text;
    }

    TuningRowsLoaded loadInto(OverrideMap& map, const std::string& text)
    {
        std::istringstream in(text);
        return loadTuningRows(in, map, kStamp);
    }

    std::vector<int32_t> indexesOf(const std::vector<TunedEntry>& entries)
    {
        std::vector<int32_t> out;
        for(const auto& entry : entries)
            out.push_back(entry.solutionIndex);
        return out;
    }

    // A row's header and value lines as cells, so a test can damage one.
    struct RowCells
    {
        std::vector<std::string> names;
        std::vector<std::string> values;
    };

    std::vector<std::string> splitCells(const std::string& line)
    {
        std::vector<std::string> cells;
        std::stringstream        ss(line);
        std::string              cell;
        while(std::getline(ss, cell, ','))
            cells.push_back(cell);
        if(!line.empty() && line.back() == ',')
            cells.emplace_back();
        return cells;
    }

    RowCells cellsOf(const std::string& row)
    {
        std::istringstream in(row);
        std::string        header, value;
        std::getline(in, header);
        std::getline(in, value);
        header.erase(0, header.find_first_not_of(' '));
        return {splitCells(header), splitCells(value)};
    }

    std::string joinCells(const std::vector<std::string>& cells)
    {
        std::string out;
        for(size_t i = 0; i < cells.size(); i++)
            out += (i ? "," : "") + cells[i];
        return out;
    }

    std::string rowOf(const RowCells& cells)
    {
        return "    " + joinCells(cells.names) + "\n" + joinCells(cells.values) + "\n";
    }

    size_t columnIndex(const RowCells& cells, const std::string& column)
    {
        for(size_t i = 0; i < cells.names.size(); i++)
            if(cells.names[i] == column)
                return i;
        ADD_FAILURE() << "the writer emits no " << column << " column";
        return 0;
    }

    std::string
        withCell(const std::string& row, const std::string& column, const std::string& value)
    {
        auto cells                               = cellsOf(row);
        cells.values[columnIndex(cells, column)] = value;
        return rowOf(cells);
    }

    std::string withoutColumn(const std::string& row, const std::string& column)
    {
        auto         cells = cellsOf(row);
        const size_t at    = columnIndex(cells, column);
        cells.names.erase(cells.names.begin() + at);
        cells.values.erase(cells.values.begin() + at);
        return rowOf(cells);
    }

    // What a process that died partway through the append leaves behind.
    std::string cutBefore(const std::string& row, const std::string& column)
    {
        auto cells = cellsOf(row);
        cells.values.resize(columnIndex(cells, column));
        return rowOf(cells);
    }

    // A row in the format hipblaslt-bench writes: no schema version, and only
    // the problem columns the historical key used.
    std::string legacyRow(const ProblemOverride&            key,
                          int32_t                           index,
                          const std::optional<std::string>& kernel = std::nullopt)
    {
        std::ostringstream names, values;
        names << "    transA,transB,batch_count,m,n,k,lda,ldb,ldc,ldd,a_type,b_type,c_type,d_type,"
                 "compute_type,solution_index";
        values << (key.transA ? "T" : "N") << ',' << (key.transB ? "T" : "N") << ','
               << key.batchSize << ',' << key.m << ',' << key.n << ',' << key.k << ','
               << key.colStrideA << ',' << key.colStrideB << ',' << key.colStrideC << ','
               << key.colStrideD << ",f16_r,f16_r,f16_r,f16_r,f32_r," << index;
        if(kernel)
        {
            names << ",kernel_name";
            values << ',' << *kernel;
        }
        return names.str() + "\n" + values.str() + "\n";
    }

    class TuningStore : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            for(const char* name : {"HIPBLASLT_TUNING_MODE", "HIPBLASLT_TUNING_CACHE_PATH"})
            {
                const char* value = std::getenv(name);
                m_savedEnv.emplace_back(name,
                                        value ? std::optional<std::string>(value) : std::nullopt);
            }
        }

        void TearDown() override
        {
            for(const auto& [name, value] : m_savedEnv)
            {
                if(value)
                    setenv(name.c_str(), value->c_str(), 1);
                else
                    unsetenv(name.c_str());
            }
        }

        std::vector<std::pair<std::string, std::optional<std::string>>> m_savedEnv;
    };

    // Every column the writer emits reads back into the key and entry it came
    // from, including the fields legacy rows cannot express.
    TEST_F(TuningStore, WrittenRowReadsBackAsItsKeyAndEntry)
    {
        ProblemOverride key       = halfKey(768, 384, 640);
        key.transB                = true;
        key.conjugateB            = true;
        key.batchSize             = 4;
        key.batchStrideA          = 768 * 640;
        key.batchStrideB          = 640 * 384;
        key.batchStrideC          = 768 * 384;
        key.batchStrideD          = 768 * 384;
        key.colStrideE            = 768;
        key.batchStrideE          = 768 * 384;
        key.epilogue              = 8;
        key.hasBias               = true;
        key.biasType              = 2;
        key.hasScaleA             = true;
        key.hasScaleAlphaVec      = true;
        key.swizzleA              = true;
        key.streamkTileScheduling = 1;
        key.smCountTarget         = 80;
        key.uniformSummationOrder = true;

        TunedEntry written = tunedEntry(4321, "Cijk_Alik_Bljk_HHS_MT64x32x64", false, 1000);
        written.requiredWorkspaceBytes = 4096;

        OverrideMap map;
        const auto  loaded = loadInto(map, fileOf({tunedRow(key, written)}));
        ASSERT_EQ(loaded.accepted, 1u);
        EXPECT_FALSE(loaded.readError);

        const auto found = map.find(key);
        ASSERT_EQ(found.size(), 1u) << "the row did not read back as the key it was written from";

        const TunedEntry& read = found[0];
        EXPECT_EQ(read.solutionIndex, 4321);
        EXPECT_EQ(read.kernelName, written.kernelName);
        EXPECT_EQ(read.schemaVersion, TuningSchemaVersion::Current);
        EXPECT_EQ(read.buildStamp, kStamp);
        EXPECT_EQ(read.requiredWorkspaceBytes, 4096u);
        EXPECT_DOUBLE_EQ(read.winnerTimeUs, 12.5);
        EXPECT_FALSE(read.complete);
        EXPECT_EQ(read.budgetMs, 1000);
        ASSERT_TRUE(read.search.has_value());
        EXPECT_TRUE(*read.search == *written.search);
    }

    // The FNUZ FP8 types have their own spelling. Written as the plain FP8
    // spelling they would read back as the OCP types, and the row would never
    // match the problem it was tuned for.
    TEST_F(TuningStore, FnuzSpellingsReadBackAsFnuzTypes)
    {
        ProblemOverride key = halfKey();
        key.inputTypeA      = rocisa::DataType::Float8_fnuz;
        key.inputTypeB      = rocisa::DataType::BFloat8_fnuz;

        const TuningRowTypes fnuz{"f8_fnuz_r", "bf8_fnuz_r", "f16_r", "f16_r", "f32_r"};
        const TuningRowTypes ocp{"f8_r", "bf8_r", "f16_r", "f16_r", "f32_r"};

        OverrideMap fnuzMap;
        loadInto(fnuzMap, fileOf({formatTuningRow(key, fnuz, tunedEntry(7, "kernel"), kStamp)}));
        EXPECT_EQ(fnuzMap.find(key).size(), 1u);

        OverrideMap ocpMap;
        loadInto(ocpMap, fileOf({formatTuningRow(key, ocp, tunedEntry(7, "kernel"), kStamp)}));
        EXPECT_TRUE(ocpMap.find(key).empty()) << "an OCP spelling served an FNUZ problem";
    }

    // A newer schema keys on fields this parser cannot see, so reading one as
    // the current version would match on the subset it recognises.
    TEST_F(TuningStore, RowFromANewerSchemaIsIgnored)
    {
        OverrideMap map;
        const auto  row
            = withCell(tunedRow(halfKey(), tunedEntry(7, "kernel")), "schema_version", "99");
        EXPECT_EQ(loadInto(map, fileOf({row})).accepted, 0u);
        EXPECT_EQ(map.size(), 0u);
    }

    // Some spellings name a hipDataType that is not a GEMM tensor type, and must
    // be refused before the assert-based Tensile converter sees them.
    TEST_F(TuningStore, DatatypeThatIsNotAGemmTypeIsRejected)
    {
        OverrideMap map;
        const auto  row = withCell(tunedRow(halfKey(), tunedEntry(7, "kernel")), "a_type", "e8_r");
        EXPECT_EQ(loadInto(map, fileOf({row})).accepted, 0u);
    }

    // An empty cell is indistinguishable from a real zero once defaulted, and
    // this shape's real batch stride is zero, so a lenient parser would replay
    // the row.
    TEST_F(TuningStore, BlankKeyCellIsRejected)
    {
        OverrideMap map;
        const auto  row = withCell(tunedRow(halfKey(), tunedEntry(7, "kernel")), "stride_a", "");
        EXPECT_EQ(loadInto(map, fileOf({row})).accepted, 0u);
    }

    TEST_F(TuningStore, NumberWithTrailingTextIsRejected)
    {
        OverrideMap map;
        const auto  row = withCell(tunedRow(halfKey(), tunedEntry(7, "kernel")), "m", "1024junk");
        EXPECT_EQ(loadInto(map, fileOf({row})).accepted, 0u);
    }

    // A current row missing a key column would read it back as a default and
    // describe some other problem.
    TEST_F(TuningStore, RowWithoutAKeyColumnIsRejected)
    {
        for(const char* column : {"lda", "stride_e", "epilogue", "gcnArchName", "CUs"})
        {
            SCOPED_TRACE(std::string("row without ") + column);
            OverrideMap map;
            const auto  row = withoutColumn(tunedRow(halfKey(), tunedEntry(7, "kernel")), column);
            EXPECT_EQ(loadInto(map, fileOf({row})).accepted, 0u);
        }
    }

    // Whether tune mode revisits a row depends on its completion column, and a
    // missing value can no longer be told apart from a finished search.
    TEST_F(TuningStore, RowWithoutItsCompletionColumnIsRejected)
    {
        OverrideMap map;
        const auto  row = withoutColumn(tunedRow(halfKey(), tunedEntry(7, "kernel")), "complete");
        EXPECT_EQ(loadInto(map, fileOf({row})).accepted, 0u);
    }

    // A value line cut short keeps a well-formed prefix, which must not be read
    // as the whole row, least of all as a finished search.
    TEST_F(TuningStore, RowCutShortIsRejected)
    {
        const auto row = tunedRow(halfKey(), tunedEntry(7, "kernel", false, 1000));
        for(const char* column :
            {"kernel_name", "required_workspace", "complete", "budget_ms", "rotating_mb"})
        {
            SCOPED_TRACE(std::string("value row cut before ") + column);
            OverrideMap map;
            EXPECT_EQ(loadInto(map, fileOf({cutBefore(row, column)})).accepted, 0u);
        }
    }

    // A process that dies between a header and its value row leaves an orphan
    // header, which must not swallow the next record's header as its values.
    TEST_F(TuningStore, OrphanHeaderDoesNotHideTheNextRow)
    {
        const auto  torn   = tunedRow(halfKey(), tunedEntry(7, "kernel"));
        const auto  whole  = tunedRow(halfKey(), tunedEntry(9, "other"));
        const auto  orphan = torn.substr(0, torn.find('\n') + 1);
        OverrideMap map;

        EXPECT_EQ(loadInto(map, fileOf({orphan, whole})).accepted, 1u);
        EXPECT_EQ(indexesOf(map.find(halfKey())), std::vector<int32_t>{9});
    }

    // Each of these is part of the key: a row that differs from the problem in
    // one of them is loaded, and serves the problem it describes, not this one.
    TEST_F(TuningStore, RowDifferingInAKeyColumnServesOnlyItsOwnProblem)
    {
        const ProblemOverride problem = halfKey();
        const auto            row     = tunedRow(problem, tunedEntry(7, "kernel"));

        struct Change
        {
            const char*     column;
            const char*     value;
            ProblemOverride described;
        };
        std::vector<Change> changes;
        {
            ProblemOverride e = problem;
            e.colStrideE      = 12345;
            changes.push_back({"lde", "12345", e});
        }
        {
            ProblemOverride e = problem;
            e.batchStrideE    = 12345;
            changes.push_back({"stride_e", "12345", e});
        }
        {
            ProblemOverride u       = problem;
            u.uniformSummationOrder = true;
            changes.push_back({"uniform_summation_order", "1", u});
        }
        {
            ProblemOverride d = problem;
            d.archName        = "gfx942:sramecc+:xnack-";
            changes.push_back({"gcnArchName", "gfx942:sramecc+:xnack-", d});
        }
        {
            // Both are "not N", which is all the historical key recorded.
            ProblemOverride c = problem;
            c.transA          = true;
            c.conjugateA      = true;
            changes.push_back({"transA", "C", c});
        }

        for(const auto& change : changes)
        {
            SCOPED_TRACE(std::string(change.column) + "=" + change.value);
            OverrideMap map;
            EXPECT_EQ(loadInto(map, fileOf({withCell(row, change.column, change.value)})).accepted,
                      1u)
                << "the row was rejected rather than keyed on the column";
            EXPECT_TRUE(map.find(problem).empty());
            EXPECT_EQ(map.find(change.described).size(), 1u);
        }

        ProblemOverride transposed = problem;
        transposed.transA          = true;
        OverrideMap map;
        loadInto(map, fileOf({withCell(row, "transA", "C")}));
        EXPECT_TRUE(map.find(transposed).empty()) << "a conjugate-transpose row served a transpose";
    }

    // A row without a kernel or solution name has nothing to check it against
    // at replay, so the build that wrote it is its only credential.
    TEST_F(TuningStore, UnnamedRowNeedsTheBuildThatWroteIt)
    {
        const ProblemOverride key = halfKey();

        OverrideMap sameBuild;
        EXPECT_EQ(loadInto(sameBuild, fileOf({legacyRow(key, 7)}, kStamp)).accepted, 1u);

        OverrideMap otherBuild;
        EXPECT_EQ(loadInto(otherBuild, fileOf({legacyRow(key, 7)}, "0000000")).accepted, 0u);

        OverrideMap named;
        EXPECT_EQ(loadInto(named, fileOf({legacyRow(key, 7, "kernel")}, "0000000")).accepted, 1u)
            << "a named row from another build is validated by its name, not the build";

        OverrideMap current;
        const auto  unnamed = withCell(tunedRow(key, tunedEntry(7, "kernel")), "kernel_name", "");
        EXPECT_EQ(loadInto(current, fileOf({unnamed})).accepted, 0u)
            << "a current-schema row must carry a name";
    }

    // A build made outside a git checkout has an empty stamp, as does a file
    // with no version line or an empty one. Two empty stamps are not a match.
    TEST_F(TuningStore, BuildWithoutAStampTrustsNoUnnamedRow)
    {
        const std::string row = legacyRow(halfKey(), 7);

        for(const std::string& text : {row, "Git Version:\n" + row, "Git Version: \n" + row})
        {
            SCOPED_TRACE(text);
            OverrideMap        map;
            std::istringstream in(text);
            const auto         loaded = loadTuningRows(in, map, "");
            EXPECT_EQ(loaded.accepted, 0u);
            EXPECT_EQ(loaded.skippedUnnamed, 1u);
        }

        OverrideMap        named;
        std::istringstream in(legacyRow(halfKey(), 7, "kernel"));
        EXPECT_EQ(loadTuningRows(in, named, "").accepted, 1u)
            << "a named row does not depend on the build stamp";
    }

    TEST_F(TuningStore, VersionLineIsReadTrimmed)
    {
        OverrideMap map;
        const auto  loaded
            = loadInto(map, "Git Version:   " + kStamp + "  \n" + legacyRow(halfKey(), 7));
        EXPECT_EQ(loaded.fileBuildStamp, kStamp);
        EXPECT_EQ(loaded.accepted, 1u);
    }

    // Index 0 is a real solution; only a negative index is meaningless.
    TEST_F(TuningStore, SolutionIndexZeroIsAccepted)
    {
        const ProblemOverride key = halfKey();

        OverrideMap zero;
        EXPECT_EQ(loadInto(zero, fileOf({legacyRow(key, 0, "kernel")})).accepted, 1u);
        EXPECT_EQ(indexesOf(zero.findLegacy(key)), std::vector<int32_t>{0});

        OverrideMap negative;
        EXPECT_EQ(loadInto(negative, fileOf({legacyRow(key, -1, "kernel")})).accepted, 0u);
    }

    // Legacy rows keep matching on the fields their format records, and only
    // on those.
    TEST_F(TuningStore, LegacyRowMatchesOnTheHistoricalFields)
    {
        const ProblemOverride key = halfKey();
        OverrideMap           map;
        ASSERT_EQ(loadInto(map, fileOf({legacyRow(key, 7, "kernel")})).accepted, 1u);

        EXPECT_TRUE(map.find(key).empty()) << "a legacy row was filed under the full key";

        ProblemOverride unrecorded = key;
        unrecorded.colStrideA      = 4096;
        unrecorded.epilogue        = 8;
        unrecorded.archName        = "gfx942";
        EXPECT_EQ(indexesOf(map.findLegacy(unrecorded)), std::vector<int32_t>{7});

        ProblemOverride otherShape = key;
        otherShape.m               = 2048;
        EXPECT_TRUE(map.findLegacy(otherShape).empty());
    }

    // A hand-written file's order is the only preference it can express.
    TEST_F(TuningStore, LegacyRowsKeepFileOrder)
    {
        const ProblemOverride key = halfKey();
        OverrideMap           map;
        loadInto(map, fileOf({legacyRow(key, 1, "first"), legacyRow(key, 2, "second")}));

        EXPECT_EQ(indexesOf(map.findLegacy(key)), (std::vector<int32_t>{1, 2}));
    }

    // Reading the same row again refreshes it rather than stacking a copy. A
    // row naming another kernel at the same index is a different entry, so a
    // stale row cannot hide the replacement written after it.
    TEST_F(TuningStore, RepeatedRowRefreshesAndRenamedRowIsKept)
    {
        const ProblemOverride key = halfKey();

        TunedEntry first                 = tunedEntry(7, "kernel");
        TunedEntry refreshed             = first;
        refreshed.requiredWorkspaceBytes = 8192;

        OverrideMap map;
        EXPECT_EQ(loadInto(map, fileOf({tunedRow(key, first), tunedRow(key, refreshed)})).accepted,
                  1u);
        ASSERT_EQ(map.find(key).size(), 1u);
        EXPECT_EQ(map.find(key)[0].requiredWorkspaceBytes, 8192u);

        OverrideMap renamed;
        EXPECT_EQ(loadInto(renamed,
                           fileOf({tunedRow(key, tunedEntry(7, "NotARealKernelName")),
                                   tunedRow(key, tunedEntry(7, "kernel"))}))
                      .accepted,
                  2u);

        const auto found = renamed.find(key);
        ASSERT_EQ(found.size(), 2u);
        EXPECT_EQ(found[0].kernelName, std::optional<std::string>("kernel"));
    }

    // Rows written before the search and completion columns existed are
    // finished searches with none recorded, so they close the gate like any
    // other finished row.
    TEST_F(TuningStore, VersionOneRowsReadAsFinishedSearches)
    {
        const ProblemOverride key = halfKey();

        std::string row = withCell(tunedRow(key, tunedEntry(7, "kernel")), "schema_version", "1");
        for(const char* column : {"complete",
                                  "budget_ms",
                                  "search_all_kernels",
                                  "search_max_candidates",
                                  "search_workspace",
                                  "cold_iters",
                                  "hot_iters",
                                  "flush_icache",
                                  "rotating_mb"})
            row = withoutColumn(row, column);

        OverrideMap map;
        ASSERT_EQ(loadInto(map, fileOf({row})).accepted, 1u);

        const auto found = map.find(key);
        ASSERT_EQ(found.size(), 1u);
        EXPECT_EQ(found[0].schemaVersion, TuningSchemaVersion::FullKey);
        EXPECT_TRUE(found[0].complete);
        EXPECT_FALSE(found[0].search.has_value());
        EXPECT_FALSE(map.needsRetune(key, rankedSearch(16), 0));
    }

    // The file is append-only, so the run that finishes a truncated search
    // leaves its winner behind the partial one. Replay takes the first entry
    // that validates, so the completed row has to come first, whichever of the
    // two was written last.
    TEST_F(TuningStore, CompleteRowWinsOverPartialRows)
    {
        const ProblemOverride key = halfKey();

        OverrideMap finishedLater;
        loadInto(finishedLater,
                 fileOf({tunedRow(key, tunedEntry(1, "partial", false, 1000)),
                         tunedRow(key, tunedEntry(2, "finished", true, 0))}));
        EXPECT_EQ(indexesOf(finishedLater.find(key)), (std::vector<int32_t>{2, 1}));

        OverrideMap partialLater;
        loadInto(partialLater,
                 fileOf({tunedRow(key, tunedEntry(1, "finished", true, 0)),
                         tunedRow(key, tunedEntry(2, "partial", false, 1000))}));
        EXPECT_EQ(indexesOf(partialLater.find(key)), (std::vector<int32_t>{1, 2}));
    }

    // Among rows of the same completeness the newest wins, whatever ceilings
    // they were written under.
    TEST_F(TuningStore, NewestRowWinsWithinEachGroup)
    {
        const ProblemOverride key = halfKey();

        OverrideMap partials;
        loadInto(partials,
                 fileOf({tunedRow(key, tunedEntry(1, "older", false, 60000)),
                         tunedRow(key, tunedEntry(2, "newer", false, 1000))}));
        EXPECT_EQ(indexesOf(partials.find(key)), (std::vector<int32_t>{2, 1}));

        OverrideMap completes;
        loadInto(
            completes,
            fileOf({tunedRow(key, tunedEntry(1, "older")), tunedRow(key, tunedEntry(2, "newer"))}));
        EXPECT_EQ(indexesOf(completes.find(key)), (std::vector<int32_t>{2, 1}));
    }

    // Re-tuning a partial row is worth a stall only when this run can get
    // further: a more generous ceiling, or a different search. Under the same
    // ceiling it would stop in the same place and append an identical row.
    TEST_F(TuningStore, PartialRowIsRetunedOnlyWhenThisRunCanGetFurther)
    {
        const ProblemOverride key    = halfKey();
        const TuningSearch    search = rankedSearch(16);
        OverrideMap           map;
        loadInto(map, fileOf({tunedRow(key, tunedEntry(7, "kernel", false, 1000, search))}));

        EXPECT_FALSE(map.needsRetune(key, search, 1000));
        EXPECT_FALSE(map.needsRetune(key, search, 500));
        EXPECT_TRUE(map.needsRetune(key, search, 2000));
        EXPECT_TRUE(map.needsRetune(key, search, 0)) << "an unlimited run can finish any search";
        EXPECT_TRUE(map.needsRetune(key, rankedSearch(32), 1000));
    }

    // Once a completed row covers the search, the superseded partial row beside
    // it must not keep the shape looking untuned.
    TEST_F(TuningStore, PartialRowBesideCompleteRowDoesNotRetune)
    {
        const ProblemOverride key    = halfKey();
        const TuningSearch    search = rankedSearch(16);
        OverrideMap           map;
        loadInto(map,
                 fileOf({tunedRow(key, tunedEntry(1, "partial", false, 1000, search)),
                         tunedRow(key, tunedEntry(2, "finished", true, 0, search))}));

        EXPECT_FALSE(map.needsRetune(key, search, 1000));
    }

    // A finished search of a ranked prefix says nothing about the kernels past
    // it, so a wider run re-tunes the shape; a narrower one has nothing to add.
    TEST_F(TuningStore, FinishedSearchIsWidenedButNotRepeated)
    {
        const ProblemOverride key = halfKey();

        OverrideMap narrow;
        loadInto(narrow,
                 fileOf({tunedRow(key, tunedEntry(7, "kernel", true, 0, rankedSearch(2)))}));
        EXPECT_TRUE(narrow.needsRetune(key, rankedSearch(16), 0));
        EXPECT_FALSE(narrow.needsRetune(key, rankedSearch(2), 0));

        OverrideMap wide;
        loadInto(wide, fileOf({tunedRow(key, tunedEntry(7, "kernel", true, 0, rankedSearch(16)))}));
        EXPECT_FALSE(wide.needsRetune(key, rankedSearch(2), 0));

        TuningSearch everyKernel = rankedSearch(0);
        everyKernel.allKernels   = true;
        EXPECT_TRUE(wide.needsRetune(key, everyKernel, 0));

        TuningSearch moreWorkspace   = rankedSearch(16);
        moreWorkspace.workspaceBytes = 256 << 20;
        EXPECT_TRUE(wide.needsRetune(key, moreWorkspace, 0))
            << "a result filtered by a smaller workspace was treated as final";
    }

    // A row that records no search, and a legacy row, are final: reading them as
    // unfinished would re-tune every shape of an existing file.
    TEST_F(TuningStore, RowsWithoutARecordedSearchAreFinal)
    {
        const ProblemOverride key = halfKey();

        OverrideMap unrecorded;
        TunedEntry  entry = tunedEntry(7, "kernel");
        entry.search.reset();
        unrecorded.add(key, entry);
        EXPECT_FALSE(unrecorded.needsRetune(key, rankedSearch(16), 0));

        OverrideMap legacy;
        loadInto(legacy, fileOf({legacyRow(key, 7, "kernel")}));
        EXPECT_FALSE(legacy.needsRetune(key, rankedSearch(16), 0));
    }

    TEST_F(TuningStore, BudgetComparison)
    {
        EXPECT_TRUE(tuningBudgetIsMoreGenerous(1000, -1)) << "a row that did not record one";
        EXPECT_TRUE(tuningBudgetIsMoreGenerous(0, 1000));
        EXPECT_TRUE(tuningBudgetIsMoreGenerous(2000, 1000));
        EXPECT_FALSE(tuningBudgetIsMoreGenerous(1000, 1000));
        EXPECT_FALSE(tuningBudgetIsMoreGenerous(500, 1000));
        EXPECT_FALSE(tuningBudgetIsMoreGenerous(0, 0)) << "nothing beats an unlimited run";
        EXPECT_FALSE(tuningBudgetIsMoreGenerous(5000, 0));
    }

    TEST_F(TuningStore, SearchCoverage)
    {
        TuningSearch everyKernel = rankedSearch(0);
        everyKernel.allKernels   = true;

        EXPECT_TRUE(tuningSearchCovers(everyKernel, rankedSearch(16)));
        EXPECT_TRUE(tuningSearchCovers(rankedSearch(16), rankedSearch(2)));
        EXPECT_FALSE(tuningSearchCovers(rankedSearch(2), rankedSearch(16)));
        EXPECT_FALSE(tuningSearchCovers(rankedSearch(16), everyKernel));

        auto less = [](auto change) {
            TuningSearch search = rankedSearch(16);
            change(search);
            return search;
        };
        EXPECT_FALSE(tuningSearchCovers(less([](TuningSearch& s) { s.workspaceBytes = 0; }),
                                        rankedSearch(16)));
        EXPECT_FALSE(
            tuningSearchCovers(less([](TuningSearch& s) { s.coldIters = 0; }), rankedSearch(16)));
        EXPECT_FALSE(
            tuningSearchCovers(less([](TuningSearch& s) { s.hotIters = 1; }), rankedSearch(16)));
        EXPECT_FALSE(tuningSearchCovers(less([](TuningSearch& s) { s.flushICache = false; }),
                                        rankedSearch(16)));
        EXPECT_FALSE(
            tuningSearchCovers(less([](TuningSearch& s) { s.rotatingMb = 0; }), rankedSearch(16)));
    }

    // A process in a secure execution context (set-user-ID and the like) must
    // not let an inherited environment choose a file for it to write and
    // minutes of GPU work for it to spend.
    TEST_F(TuningStore, PrivilegedProcessIgnoresTheTuningEnvironment)
    {
        setenv("HIPBLASLT_TUNING_MODE", "tune", 1);
        setenv("HIPBLASLT_TUNING_CACHE_PATH", "tuning.txt", 1);

        const auto ordinary = TuningModeConfig::fromEnvironment(false);
        EXPECT_EQ(ordinary.mode, TuningMode::Tune);
        EXPECT_EQ(ordinary.cachePath, "tuning.txt");
        EXPECT_TRUE(ordinary.writes());
        EXPECT_FALSE(ordinary.suppressedForSecurity);

        const auto privileged = TuningModeConfig::fromEnvironment(true);
        EXPECT_EQ(privileged.mode, TuningMode::Off);
        EXPECT_TRUE(privileged.cachePath.empty());
        EXPECT_FALSE(privileged.reads());
        EXPECT_TRUE(privileged.suppressedForSecurity);
    }

    // An unknown mode is off, only tune writes, and a mode with no path does
    // nothing.
    TEST_F(TuningStore, ModeNeedsAKnownValueAndAPath)
    {
        setenv("HIPBLASLT_TUNING_MODE", "online", 1);
        setenv("HIPBLASLT_TUNING_CACHE_PATH", "tuning.txt", 1);
        EXPECT_EQ(TuningModeConfig::fromEnvironment(false).mode, TuningMode::Off);

        setenv("HIPBLASLT_TUNING_MODE", "cache", 1);
        const auto cache = TuningModeConfig::fromEnvironment(false);
        EXPECT_TRUE(cache.reads());
        EXPECT_FALSE(cache.writes());

        setenv("HIPBLASLT_TUNING_MODE", "tune", 1);
        unsetenv("HIPBLASLT_TUNING_CACHE_PATH");
        const auto noPath = TuningModeConfig::fromEnvironment(false);
        EXPECT_EQ(noPath.mode, TuningMode::Tune);
        EXPECT_FALSE(noPath.reads());
        EXPECT_FALSE(noPath.writes());
        EXPECT_FALSE(noPath.suppressedForSecurity);
    }

    // A shape is only tuned when none of its entries is usable, and its winner
    // replaces them, even one that reuses a dead row's index. Legacy rows are
    // matched separately and stay.
    TEST_F(TuningStore, ReplaceAllInstallsTheWinnerAlone)
    {
        const ProblemOverride key = halfKey();

        OverrideMap map;
        loadInto(map,
                 fileOf({tunedRow(key, tunedEntry(7, "NotARealKernelName")),
                         tunedRow(key, tunedEntry(9, "other")),
                         legacyRow(key, 3, "legacy")}));
        ASSERT_EQ(map.find(key).size(), 2u);

        map.replaceAll(key, tunedEntry(7, "kernel"));

        const auto found = map.find(key);
        ASSERT_EQ(found.size(), 1u);
        EXPECT_EQ(found[0].kernelName, std::optional<std::string>("kernel"));
        EXPECT_EQ(indexesOf(map.findLegacy(key)), std::vector<int32_t>{3});
    }

    // The file starts with the build stamp once, and every appended row reads
    // back.
    TEST_F(TuningStore, AppendedRowsReadBackFromTheFile)
    {
#ifdef WIN32
        const auto pid = _getpid();
#else
        const auto pid = getpid();
#endif
        const std::string path
            = ::testing::TempDir() + "hipblaslt_tuning_store_" + std::to_string(pid) + ".txt";
        std::remove(path.c_str());

        ASSERT_TRUE(
            appendTuningRow(path, tunedRow(halfKey(256, 256, 256), tunedEntry(1, "a")), kStamp));
        ASSERT_TRUE(
            appendTuningRow(path, tunedRow(halfKey(512, 256, 256), tunedEntry(2, "b")), kStamp));

        std::ifstream in(path);
        std::string   first;
        std::getline(in, first);
        EXPECT_EQ(first, "Git Version: " + kStamp);

        std::stringstream rest;
        rest << in.rdbuf();
        EXPECT_EQ(rest.str().find("Git Version:"), std::string::npos)
            << "the build stamp was repeated for a later row";

        std::ifstream again(path);
        OverrideMap   map;
        EXPECT_EQ(loadTuningRows(again, map, kStamp).accepted, 2u);
        EXPECT_EQ(indexesOf(map.find(halfKey(512, 256, 256))), std::vector<int32_t>{2});

        std::remove(path.c_str());
    }
} // namespace
