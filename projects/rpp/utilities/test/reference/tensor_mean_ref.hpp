#ifndef RPP_TEST_TENSOR_MEAN_REF_H
#define RPP_TEST_TENSOR_MEAN_REF_H

#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/reduction.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_tensor_mean, derived from the op's definition (the
// channel-wise mean R/G/B and the total mean, per image, over the ROI in raw intensity space),
// NOT from the RPP kernel. Used as the reference for both backends so kernel bugs surface as
// diffs.
//
// Per image, with N = roiWidth*roiHeight pixels per channel:
//   out[R/G/B] = sum(channel) / N
//   out[total] = (sum(R)+sum(G)+sum(B)) / (3*N)   (mean of every pixel across the 3 channels)
// For a 1-channel image the single result is sum / N. The mean is in the stored intensity space
// (U8 [0,255], I8 [-128,127], F16/F32 [0,1]) -- consistent with the channel means; the golden
// accumulates in double.
template <typename T>
std::vector<double> tensor_mean_reference(const T* src, const RpptDesc& d, const RpptROI* roi,
                                          RpptRoiType type) {
    const std::size_t stride = reduction_stride(d);
    const std::vector<std::size_t> N = roi_pixel_counts(d, roi, type);
    std::vector<double> sum(reduction_length(d), 0.0);
    for_each_roi_value(src, d, roi, type,
                       [&](Rpp32u n, Rpp32u c, double v) { sum[n * stride + c] += v; });

    std::vector<double> out(reduction_length(d), 0.0);
    for (Rpp32u n = 0; n < d.n; ++n) {
        for (Rpp32u c = 0; c < d.c; ++c)
            out[n * stride + c] = sum[n * stride + c] / static_cast<double>(N[n]);
        if (d.c == 3)
            out[n * stride + 3] =
                (sum[n * stride + 0] + sum[n * stride + 1] + sum[n * stride + 2]) /
                (3.0 * static_cast<double>(N[n]));
    }
    return out;
}

}  // namespace rpptest

#endif  // RPP_TEST_TENSOR_MEAN_REF_H
