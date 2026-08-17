// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

// One feature cell fed to the forest walker (lgbm_forest.hpp): `missing == -1`
// marks an absent/NaN feature; otherwise the value -- a real number, or a
// categorical code cast to double -- lives in `fvalue`. `qvalue` is an unused
// quantized slot retained for layout compatibility.
union LgbmEntry
{
    int missing;
    double fvalue;
    int qvalue;
};

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP
