// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

namespace ck_dsl_provider {

/// Plugin-specific execution settings.
///
/// Empty for the I-1 skeleton. The runtime DSL knob/spec surface
/// (block size, tile sizes, pipeline mode, ...) lands in later
/// milestones once the C++ adapter and Python compile bridge are
/// wired up.
struct CkDslSettings {};

}  // namespace ck_dsl_provider
