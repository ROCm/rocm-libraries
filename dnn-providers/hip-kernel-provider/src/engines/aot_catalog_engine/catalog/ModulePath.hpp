// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Self-location: resolve the directory of the shared object this engine is
// compiled into (the loaded hip_kernel_provider plugin). Used so the catalog is
// found BESIDE the plugin binary that is actually loaded -- a locally-built /
// force-loaded plugin then reads its own build-tree catalog and never crosses
// over to a system install's catalog (and vice-versa).

#pragma once

#include <string>

namespace aot_catalog_engine::catalog
{

// Absolute, canonicalized directory of the shared object containing this
// engine's code (i.e. the loaded plugin .so/.dll), or an empty string if it
// cannot be determined. Anchored on a symbol in this translation unit, so it
// resolves the plugin binary rather than the backend or any other library.
std::string thisModuleDir();

} // namespace aot_catalog_engine::catalog
