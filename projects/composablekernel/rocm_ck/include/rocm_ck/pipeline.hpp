// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Thin wrapper -- canonical definition lives in ck_common/pipeline.hpp.
// NOTE: prefer ck_common::Pipeline in new code; this header is a compatibility re-export.
//
// TODO(#7280): rocm_ck::Pipeline (V1, V3, V4, Memory, Preshuffle) defined in
// gemm_spec.hpp uses a different naming/ordering than ck_common::Pipeline
// (Mem, CompV1..V5, PreShuffleV1/V2). It is intentionally left in place for
// pt.1 because the two enumerations have non-overlapping semantics; pt.2
// should consolidate by extending the ck_common enum and migrating call sites.

#pragma once

#include <ck_common/pipeline.hpp>

namespace rocm_ck {

// Common-canonical Pipeline lives under this alias to avoid colliding with
// rocm_ck::Pipeline in <rocm_ck/gemm_spec.hpp> until the pt.2 consolidation.
using CommonPipeline = ck_common::Pipeline;

} // namespace rocm_ck
