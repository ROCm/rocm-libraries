// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/config.h>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../tensor_holder.hpp"

#if MIOPEN_EMBED_DB
#include <miopen_data.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/db.hpp>
#include <miopen/sqlite_db.hpp>
#include <miopen/find_db.hpp>
#include <miopen/filesystem.hpp>
#endif

namespace {

struct CPU_EmbedSQLite_NONE : public ::testing::TestWithParam<int>
{
    void SetUp() override
    {
        prng::reset_seed();
#if MIOPEN_EMBED_DB
        x = tensor<float>{128, 1024, 14, 14};
        w = tensor<float>{2048, 1024, 1, 1};
        y = tensor<float>{filter.GetForwardOutputTensor(x.desc, w.desc)};
#endif
    }

#if MIOPEN_EMBED_DB
    tensor<float> x;
    tensor<float> w;
    tensor<float> y;
    miopen::ConvolutionDescriptor filter = {2,
                                            miopenConvolution,
                                            miopenPaddingDefault,
                                            {0, 0},
                                            {2, 2},
                                            { 1,
                                              1 }};
#endif
};

} // namespace

TEST_P(CPU_EmbedSQLite_NONE, EmbedSQLite)
{
#if MIOPEN_EMBED_DB
    auto&& handle = get_handle();

    // create a context/problem descriptor
    const auto problem = miopen::conv::ProblemDescription{
        x.desc, w.desc, y.desc, filter, miopen::conv::Direction::Forward};
    miopen::ExecutionContext ctx{};
    ctx.SetStream(&handle);

    // Check PerfDb
    {
        // Get filename for the sys db
        // Check it in miopen_data()
        miopen::fs::path pdb_path(ctx.GetPerfDbPath());
        const auto& it_p = miopen_data().find(miopen::make_object_file_name(pdb_path.filename()));
        EXPECT_NE(it_p, miopen_data().end());
        // find all the entries in perf db
        // Assert result is non-empty
        auto pdb = miopen::GetDb(ctx);
        EXPECT_TRUE(pdb.FindRecord(problem));
    }

    // Check FindDb
    {
        // FindDb will throw if the file is not present
        miopen::FindDbRecord rec{handle, problem};
        EXPECT_FALSE(rec.empty());
    }
#else
    GTEST_SKIP() << "Test requires MIOPEN_EMBED_DB to be enabled";
#endif
}

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_EmbedSQLite_NONE, testing::ValuesIn({0}));
