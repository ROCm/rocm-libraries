#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

// Mirrors the `union Entry` Treelite emits in lgbm_models/{rank,appl}/header.h.
// We can't include those headers directly from C++ (each defines its own
// `predict` symbol with C linkage), so we declare the wrappers + a
// matching union here. The TUs that compile the generated C are built with
// -Dpredict=lgbm_{rank,appl}_predict to expose distinct symbol names.
extern "C" {

union LgbmEntry
{
    int missing;
    double fvalue;
    int qvalue;
};

void lgbm_rank_predict(union LgbmEntry* data, int pred_margin, double* result);
void lgbm_appl_predict(union LgbmEntry* data, int pred_margin, double* result);

} // extern "C"

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PREDICT_HPP
