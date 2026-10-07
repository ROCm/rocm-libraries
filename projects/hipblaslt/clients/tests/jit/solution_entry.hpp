// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-replay-catalog.hpp"
#include "test_helpers.hpp"

// The committed-kernel descriptions live with the replay backend. Tests use
// these names.
namespace hipblaslt_jit_test
{
    using hipblaslt_jit::committed::Assembly;
    using hipblaslt_jit::committed::Prepared;
    using hipblaslt_jit::committed::isDescribedReplayRoot;
    using hipblaslt_jit::committed::prepareSolutions;
    using hipblaslt_jit::committed::readAssembly;
    using hipblaslt_jit::committed::stageDescribedSources;
}
