// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

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

} // namespace hipdnn_corpus_gen::test
