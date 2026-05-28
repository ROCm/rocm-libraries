// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Common PipelineScheduler enum. No runtime, no CK deps.

#pragma once

namespace ck_common {

/// Instruction scheduling strategy within a wavefront.
///
/// Default:   Let the runtime or pipeline choose.
/// Intrawave: Synchronous — all waves in a workgroup synchronize after each
///     k-iteration. Memory loads and compute are interleaved within a single wave.
/// Interwave: Asynchronous — waves proceed independently with minimal
///     synchronization. Overlaps compute from one wave with memory loads from another.
enum class PipelineScheduler
{
    Default,
    Intrawave,
    Interwave
};

} // namespace ck_common
