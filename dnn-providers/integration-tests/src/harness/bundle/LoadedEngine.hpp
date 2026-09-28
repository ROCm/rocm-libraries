// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace hipdnn_integration_tests::bundle
{

/// One engine this build loaded: what to pin, and what to call it.
///
/// Its own header so the collaborator seams can name an engine without including
/// LoadedEngineTable.hpp, which needs <hipdnn_backend.h> and a live handle to build
/// the inventory. Those seams are what the deviceless unit binary links, and the
/// backend C API has no business in it.
struct LoadedEngine
{
    int64_t id = 0;
    std::string name;
};

} // namespace hipdnn_integration_tests::bundle
