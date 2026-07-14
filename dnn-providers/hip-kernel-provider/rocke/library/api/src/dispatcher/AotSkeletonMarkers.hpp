// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocke_client::dispatcher
{

// Stable, greppable substrings emitted by AotCatalog::loadDefault() to mark
// the outcome of the AOT skeleton load probe. Integration tests match these
// exact substrings via SharedLogRecorder::hasLogContaining so they can verify
// the load path was actually exercised, not silently skipped.
//
// TODO(kpack-fastfollow): temporary skeleton observability — remove these
// markers and the log-capture integration test once real selection/execution +
// result validation land (at which point a live graph submission proves the path
// far more strongly than a log match).

/// Emitted at INFO level when kpack_open -> kpack_get_kernel ->
/// hipModuleLoadData -> hipModuleGetFunction all succeed.
static constexpr const char* AOT_SKELETON_LOAD_OK = "rocke-client AOT skeleton: LOAD OK";

/// Emitted at ERROR level immediately before throwing HipdnnPluginException
/// on any failure in the E2E load sequence.
static constexpr const char* AOT_SKELETON_LOAD_FAILED = "rocke-client AOT skeleton: LOAD FAILED";

} // namespace rocke_client::dispatcher
