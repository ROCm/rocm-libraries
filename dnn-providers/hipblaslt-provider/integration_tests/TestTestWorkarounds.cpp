// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestWorkarounds.hpp"

#include <gtest/gtest.h>

// The arch-gate half of IS_HIPBLASLT_GFX12_FP16_TT_BIAS can't be unit-tested
// cleanly without selecting a device, so these tests cover only the
// error-code half of the predicate. On a non-gfx12 device the predicate
// returns false regardless of the error code; on a gfx12 device the
// predicate returns true iff the error code is GRAPH_NOT_SUPPORTED. Either
// way the rejection cases below remain valid: a non-GRAPH_NOT_SUPPORTED
// result must never be skipped, on any arch.

TEST(TestWorkaroundsHipblasltGfx12Fp16TtBias, PredicateRejectsBackendError)
{
    const hipdnn_frontend::Error result(hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                                        "real backend failure");
    EXPECT_FALSE(IS_HIPBLASLT_GFX12_FP16_TT_BIAS(result));
}

TEST(TestWorkaroundsHipblasltGfx12Fp16TtBias, PredicateRejectsOk)
{
    const hipdnn_frontend::Error result;
    EXPECT_FALSE(IS_HIPBLASLT_GFX12_FP16_TT_BIAS(result));
}
