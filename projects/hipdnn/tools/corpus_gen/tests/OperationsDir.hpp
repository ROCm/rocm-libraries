// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

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

} // namespace hipdnn_corpus_gen::test
