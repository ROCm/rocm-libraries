#ifndef RPP_TEST_GAUSSIAN_NOISE_REF_H
#define RPP_TEST_GAUSSIAN_NOISE_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_gaussian_noise, derived from the op's public API doc
// ("adds Gaussian noise based on user defined means and standard deviations ... meanTensor[i] >= 0
// ... stdDevTensor[i] >= 0"), NOT from the RPP kernel.
//
// This reference covers only the (mean = 0, stdDev = 0) corner. The noise is additive and drawn
// from N(mean, stdDev); at stdDev = 0 that distribution collapses to a point mass at mean, so with
// mean = 0 every sample is exactly 0 and the op degenerates to a passthrough with no RNG involved.
// That makes the corner bit-exact and reproducible whatever the seed. Away from it the output
// depends on the kernel's Box-Muller stream, whose exact per-element consumption order is not
// described by the public API, so the general case is deliberately left unmodeled here rather than
// guessed -- it is covered by parameter-validation checks in the test file, not by a golden. Used
// as the reference for both backends so kernel bugs at the degenerate corner surface as diffs.
template <typename T>
void gaussian_noise_identity_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                                       RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_GAUSSIAN_NOISE_REF_H
