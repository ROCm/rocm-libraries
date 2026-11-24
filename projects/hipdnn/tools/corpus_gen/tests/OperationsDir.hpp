// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/PointFilter.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace hipdnn_corpus_gen::test
{

/// The shipped operation declarations, staged with the test data at the same offset from
/// the test binary in the build and install trees, so an installed run reads its own copy
/// rather than the source tree, which an artifact-only test runner does not have.
inline std::string operationsDir()
{
    return (hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / HIPDNN_TEST_DATA_RELDIR
            / HIPDNN_CORPUS_GEN_OPERATIONS_SUBDIR)
        .string();
}

/// The shipped declaration of @p operation, parsed as the generator parses it. A file that is
/// missing or does not load fails the calling test and yields an empty declaration.
inline OperationMetadata shippedOperation(const std::string& operation)
{
    const auto path = operationsDir() + "/" + operation + ".opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    return parsed.metadata.value_or(OperationMetadata{});
}

/// @p point with every enum and bool parameter of @p metadata that it does not bind bound to the
/// operation's own default: its `model_shape_defaults` entry, else the first value of an enum
/// and false for a bool. A test names the facets it is about and leaves the rest to the
/// declaration, so a parameter the declaration gains needs no edit in it. Numeric parameters are
/// the caller's.
inline ProblemPoint withDefaultCategoricals(const OperationMetadata& metadata, ProblemPoint point)
{
    for(const auto& parameter : metadata.parameters)
    {
        if(point.count(parameter.name) != 0)
        {
            continue;
        }
        const auto declared = metadata.modelShapeDefaults.find(parameter.name);
        if(declared != metadata.modelShapeDefaults.end())
        {
            point[parameter.name] = declared->second;
        }
        else if(parameter.type == ParameterType::ENUM)
        {
            point[parameter.name] = parameter.values.front();
        }
        else if(parameter.type == ParameterType::BOOL)
        {
            point[parameter.name] = false;
        }
    }
    return point;
}

/// The facets a narrow `--keep` names on the shipped forward: one value of each of form, layout,
/// scale, window, dtype and value_head_dim, which leaves is_causal x alignment x generate_stats
/// (8 combinations, 6 of them able to hold a problem) of the 53760.
inline std::vector<KeepClause> narrowSdpaFwdKeeps()
{
    return {{"form", "plain"},
            {"layout", "bshd"},
            {"scale", "inv_sqrt_d"},
            {"window", "none"},
            {"dtype", "fp16"},
            {"value_head_dim", "same"}};
}

/// What the tool's search admits for a keep: only the points the clauses keep. @p keep must
/// outlive the oracle.
inline std::function<bool(const ProblemPoint&)>
    admitsOnlyWhatKeepKeeps(const std::vector<KeepClause>& keep)
{
    return [&keep](const ProblemPoint& point) { return keeps(keep, point); };
}

} // namespace hipdnn_corpus_gen::test
