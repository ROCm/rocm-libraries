// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <miopen/mlo_internal.hpp>
#include <miopen/solver_id.hpp>

// Regression guard against Find-list/registry drift.
//
// Every convolution solver reachable through the Find solver lists in
// mlo_dir_conv.cpp must also be registered in the solver Id registry
// (solver.cpp). If a solver is dropped from the registry (e.g. by a revert) but
// left behind in a Find list, Find can still select it and MIOpen then throws
// miopenStatusInternalError from solver::Id::GetAlgo() because the id is
// unknown. This test fails fast on that drift instead of letting it ship and
// surface as a runtime crash in a workload.
TEST(CPU_SolverFindListRegistry_NONE, AllFindSolversAreRegistered)
{
    const auto db_ids = GetAllFindSolverDbIds();
    ASSERT_FALSE(db_ids.empty()) << "Find solver lists returned no solvers; test wiring is broken.";

    for(const auto& db_id : db_ids)
    {
        const miopen::solver::Id id{db_id};
        EXPECT_TRUE(id.IsValid())
            << "Solver '" << db_id
            << "' is present in a Find solver list (mlo_dir_conv.cpp) but is not registered in the "
               "solver Id registry (solver.cpp). This drift makes Find throw "
               "miopenStatusInternalError.";

        // GetAlgo() is the exact call that throws in the field when the id is
        // unregistered. The production Find paths (ShrinkToFind10Results,
        // EvaluateConvSolutions, FillFindReturnParameters in convolutionocl.cpp)
        // call it on found solver ids with no IsValid() guard, so call it
        // unguarded here too to exercise that real failure mode.
        EXPECT_NO_THROW((void)id.GetAlgo())
            << "Solver '" << db_id << "': solver::Id::GetAlgo() threw.";
    }
}
