// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

// Mirrors the `union Entry` Treelite emits in lgbm_models/rank/header.h.
// We can't include that header directly from C++ (it defines its own
// `predict` symbol with C linkage), so we declare the wrapper + a matching
// union here. The TUs that compile the generated C are built with
// -Dpredict=lgbm_rank_predict to expose a distinct symbol name.
//
// v5: rank-only. The applicability model was dropped in v4.
extern "C" {

union LgbmEntry
{
    int missing;
    double fvalue;
    int qvalue;
};

void lgbm_rank_predict(union LgbmEntry* data, int pred_margin, double* result);

} // extern "C"

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP
