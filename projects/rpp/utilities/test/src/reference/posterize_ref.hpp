#ifndef RPP_TEST_POSTERIZE_REF_H
#define RPP_TEST_POSTERIZE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_posterize, derived from the op's definition (reduce each
// channel to the requested number of bits by keeping the top `levelBits` most-significant bits of
// its 8-bit representation and zeroing the rest -- the canonical posterize bit-mask), NOT from the
// kernel. The same reference serves both HOST and HIP backends.
//
// The bit-mask lives in 8-bit unsigned intensity space; each dtype is mapped in and back out:
//   U8      : i = round(v);                   out = i & mask
//   I8      : i = round(v) + 128;             out = (i & mask) - 128
//   F16/F32 : i = round(v * 255) in [0,255];  out = (i & mask) / 255
// masking is an exact integer operation, so no rounding of a product is involved for U8/I8.
inline int posterize_mask(int levelBits) { return (0xFF << (8 - levelBits)) & 0xFF; }

inline double posterize_scalar(double v, DType dt, int levelBits) {
    const int mask = posterize_mask(levelBits);
    switch (dt) {
        case DType::U8: {
            const int i = static_cast<int>(std::lround(v));
            return static_cast<double>(i & mask);
        }
        case DType::I8: {
            const int i = static_cast<int>(std::lround(v)) + 128;
            return static_cast<double>((i & mask) - 128);
        }
        case DType::F16:
        case DType::F32: {
            const int i = static_cast<int>(clampd(std::lround(v * 255.0), 0.0, 255.0));
            return static_cast<double>(i & mask) / 255.0;
        }
        default: return v;
    }
}

template <typename T>
void posterize_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                         RpptRoiType roiType, int levelBits) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] =
                            from_double<T>(posterize_scalar(to_double(src[srcIdx]), dt, levelBits));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_POSTERIZE_REF_H
