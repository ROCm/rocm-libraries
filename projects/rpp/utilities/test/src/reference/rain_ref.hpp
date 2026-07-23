#ifndef RPP_TEST_RAIN_REF_H
#define RPP_TEST_RAIN_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Kernel-derived REGRESSION golden for rppt_rain -- restricted to the DEGENERATE case
// rainPercentage == 0 (no rain drops).
//
// The rain kernel seeds its rain-layer generator with std::random_device
// (src/modules/tensor/cpu/kernel/rain.cpp: create_rain_layer), so the drop pattern -- and
// therefore the output for any rainPercentage > 0 -- is NON-DETERMINISTIC: the kernel's own
// output differs run to run and no golden can match it. The only reproducible configuration
// is rainPercentage == 0, where zero drops are drawn and the rain layer stays at its memset
// background value. The op then reduces to a per-element alpha blend of the source toward
// that background:
//     out = (background - src) * alpha + src
// with background = 0 for U8/F16/F32 and -127 (0x81) for I8. This LOCKS the blend + clamp +
// round + per-layout/dtype store paths (a regression test), which is all that is
// deterministically testable; drop placement is intentionally not covered. The model is
// transcribed from the kernel with the user's explicit authorization (snow/rain/fog have no
// documented per-element spec) and serves both HOST and HIP.
//
// Per dtype (native intensity units; rain does NOT normalize): U8/I8 round to nearest and
// clamp ([0,255] / [-128,127]); F16/F32 clamp to [0,1]. (rain's I8 store rounds, unlike snow.)

namespace rain_detail {

inline double blend_store(double v, DType dt, double alpha) {
    const double bg = (dt == DType::I8) ? -127.0 : 0.0;
    const float out = (static_cast<float>(bg) - static_cast<float>(v)) * static_cast<float>(alpha) +
                      static_cast<float>(v);
    switch (dt) {
        case DType::U8: return clampd(std::nearbyint(static_cast<double>(out)), 0.0, 255.0);
        case DType::I8: return clampd(std::nearbyint(static_cast<double>(out)), -128.0, 127.0);
        default:        return clampd(static_cast<double>(out), 0.0, 1.0);  // F16/F32
    }
}

}  // namespace rain_detail

// alpha is the per-image rain blend value (rainPercentage is fixed at 0 by the test).
template <typename T>
void rain_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                    RpptRoiType roiType, double alpha) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] =
                            from_double<T>(rain_detail::blend_store(to_double(src[srcIdx]), dt, alpha));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_RAIN_REF_H
