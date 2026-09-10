// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <miopen/config.hpp>

/// \file
/// Runtime queries for the database kill switches. These live in their own header, rather than in
/// db_path.hpp, because db.hpp and find_controls.hpp need them: db_path.hpp declares a
/// miopen::testing namespace under MIOPEN_BUILD_TESTING, and pulling that into widely-included
/// headers makes an unqualified `testing::` inside `namespace miopen` ambiguous with gtest's.

namespace miopen {

/// True when reads of the system (installed) find-db and perf-db must be skipped, either because
/// the library was built with MIOPEN_DISABLE_SYSDB or because MIOPEN_DEBUG_DISABLE_SYSTEM_DB is set
/// in the environment. The user databases are unaffected, and so are the files that merely live
/// alongside the system databases, such as the AI heuristic models.
MIOPEN_INTERNALS_EXPORT bool IsSystemDbDisabled();

/// True when all file I/O against the user find-db and perf-db must be skipped, either because the
/// library was built with MIOPEN_DISABLE_USERDB or because MIOPEN_DEBUG_DISABLE_USER_DB is set in
/// the environment. This suppresses both lookups and writes, so tuning results are not persisted.
/// The system databases and the kernel cache are unaffected.
MIOPEN_INTERNALS_EXPORT bool IsUserDbDisabled();

/// \note Both predicates are process-wide switches: env::enabled() samples the environment once,
/// on first use, and only the debug override (miopen::debug::env::UpdateEnvVariable) can change
/// the answer afterwards. Database objects latch it in their constructors rather than calling
/// these per operation, so no single instance can straddle a change.

} // namespace miopen
