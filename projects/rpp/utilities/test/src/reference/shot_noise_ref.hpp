#ifndef RPP_TEST_SHOT_NOISE_REF_H
#define RPP_TEST_SHOT_NOISE_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_shot_noise, derived from the op's public API doc
// ("adds Poisson/shot noise based on a user defined shotNoiseFactor ... shotNoiseFactorTensor[i]
// >= 0"), NOT from the RPP kernel.
//
// This reference covers only the shotNoiseFactor = 0 corner: shot noise is conventionally
// modeled by scaling a pixel's intensity into a photon count by 1/factor, drawing a Poisson
// sample, then scaling back by factor -- a formula that is undefined (division by zero) at
// factor = 0 and therefore degenerates to a no-op passthrough at that corner. The general-case
// Poisson scaling for factor > 0 is not derivable from the public API doc comment alone (the
// exact photon-count normalization is unspecified), so it is deliberately left unmodeled here;
// factor > 0 is instead covered by runtime invariants in the test file (valid-range sanity and
// seed determinism), not by this golden. Used as the reference for both backends so kernel bugs
// at the factor = 0 corner surface as diffs.
template <typename T>
void shot_noise_identity_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                                   RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_SHOT_NOISE_REF_H
