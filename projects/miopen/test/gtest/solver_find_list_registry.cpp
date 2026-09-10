// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

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

// Regression guard for the opposite direction of the same drift.
//
// Every convolution solver in the Id registry (solver.cpp) must also appear in
// one of the Find solver lists in mlo_dir_conv.cpp. A solver that is registered
// but absent from every list is dead code: Find never enumerates it, so it never
// gets a workspace size and can never be selected. Transposing solvers report
// IsDynamic() == false, which excludes them from immediate mode as well, so for
// those the Find lists are the only path in and the solver is unreachable
// outright. This test fails on that drift instead of letting a solver ship
// silently disabled.
TEST(CPU_SolverFindListRegistry_NONE, AllRegisteredConvSolversAreFindReachable)
{
    const auto db_ids = GetAllFindSolverDbIds();
    ASSERT_FALSE(db_ids.empty()) << "Find solver lists returned no solvers; test wiring is broken.";

    const std::unordered_set<std::string> find_list_ids(db_ids.begin(), db_ids.end());

    const auto& registered =
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution);
    ASSERT_FALSE(registered.empty())
        << "Solver registry returned no convolution solvers; test wiring is broken.";

    for(const auto& id : registered)
    {
        EXPECT_TRUE(find_list_ids.count(id.ToString()) != 0)
            << "Solver '" << id.ToString() << "' (id " << id.Value()
            << ") is registered in the solver Id registry (solver.cpp) but is absent from every "
               "Find solver list (mlo_dir_conv.cpp), so Find can never enumerate or select it. "
               "Add it to the appropriate SolverContainer in mlo_dir_conv.cpp.";
    }
}
