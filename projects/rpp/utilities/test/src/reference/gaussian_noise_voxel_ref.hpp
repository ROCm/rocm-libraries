#ifndef RPP_TEST_GAUSSIAN_NOISE_VOXEL_REF_H
#define RPP_TEST_GAUSSIAN_NOISE_VOXEL_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/voxel_tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_gaussian_noise_voxel, derived from the op's public API
// doc ("adds gaussian noise to a batch of 4D tensors ... meanTensor[i] >= 0 ... stdDevTensor[i]
// >= 0"), NOT from the RPP kernel.
//
// The voxel counterpart of gaussian_noise_ref.hpp and scoped the same way: only the
// (mean = 0, stdDev = 0) corner is modeled, where N(mean, stdDev) collapses to a point mass at 0
// and the additive noise degenerates to a passthrough with no RNG involved. The general case
// depends on the kernel's Box-Muller stream, which the public API does not describe. Used as the
// reference for both backends.
template <typename T>
void gaussian_noise_voxel_identity_reference(const T* src, T* dst, const RpptGenericDesc& desc,
                                             const RpptROI3D* roi, Roi3D roiType) {
    for_each_voxel_roi_io(desc, roi, roiType,
                          [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx,
                              std::size_t dstIdx) { dst[dstIdx] = src[srcIdx]; });
}

}  // namespace rpptest

#endif  // RPP_TEST_GAUSSIAN_NOISE_VOXEL_REF_H
