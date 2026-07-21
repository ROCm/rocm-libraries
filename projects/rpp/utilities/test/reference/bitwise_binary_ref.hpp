#ifndef RPP_TEST_BITWISE_BINARY_REF_H
#define RPP_TEST_BITWISE_BINARY_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Host golden models for the two-source bitwise ops (rppt_bitwise_and / _or / _xor).
// Computed once on the host and used as the reference for both the HOST and HIP backends.
//
// All three are U8-only (the ops reject any other dtype): each output byte is the bitwise
// combination of the two source bytes. Both sources are read at the ROI offset and the
// output is written packed at the destination origin (matching the kernel). Results are
// bit-exact, so the caller compares with zero tolerance.

enum class BitwiseOp { And, Or, Xor };

inline double bitwise_binary_scalar(double a, double b, BitwiseOp op) {
    const Rpp8u x = static_cast<Rpp8u>(a);
    const Rpp8u y = static_cast<Rpp8u>(b);
    Rpp8u r = 0;
    switch (op) {
        case BitwiseOp::And: r = x & y; break;
        case BitwiseOp::Or: r = x | y; break;
        case BitwiseOp::Xor: r = x ^ y; break;
    }
    return static_cast<double>(r);
}

template <typename T>
void bitwise_binary_reference(const T* src1, const T* src2, T* dst, const RpptDesc& d,
                              const RpptROI* roi, RpptRoiType roiType, BitwiseOp op) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(bitwise_binary_scalar(
                            to_double(src1[srcIdx]), to_double(src2[srcIdx]), op));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_BITWISE_BINARY_REF_H
