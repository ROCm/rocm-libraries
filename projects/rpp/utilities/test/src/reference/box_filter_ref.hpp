#ifndef RPP_TEST_BOX_FILTER_REF_H
#define RPP_TEST_BOX_FILTER_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Independent host golden model for rppt_box_filter, derived from the box-filter definition
// (per-channel arithmetic mean over a KxK window with a clamp-to-edge / REPLICATE border), NOT
// from the RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
// The uniform kernel weights (each 1/(K*K)) are handed to convolve_reference, which owns the
// shared window/border/quantization math (see reference/filter_common.hpp).
template <typename T>
void box_filter_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                          RpptRoiType type, Rpp32u kernelSize) {
    const std::size_t kk = static_cast<std::size_t>(kernelSize) * kernelSize;
    const std::vector<double> weights(kk, 1.0 / static_cast<double>(kk));
    convolve_reference<T>(src, dst, d, dt, roi, type, kernelSize, weights);
}

}  // namespace rpptest

#endif  // RPP_TEST_BOX_FILTER_REF_H
