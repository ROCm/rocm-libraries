#ifndef RPP_TEST_ARITHMETIC_SCALAR_REF_H
#define RPP_TEST_ARITHMETIC_SCALAR_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/voxel_tensor_setup.hpp"

namespace rpptest {

// Host golden model for the four Voxel scalar-arithmetic ops (rppt_add_scalar /
// rppt_subtract_scalar / rppt_multiply_scalar / rppt_fused_multiply_add_scalar). Modelled from the
// operations' definition and the public API header, NOT from the kernel; computed once on the host
// and used as the reference for both the HOST and HIP backends.
//
// Semantics: each op takes one Rpp32f per batch sample and applies it to every voxel of that
// sample's ROI box -- "adds a corresponding element from the 'addTensor' to source tensor",
// likewise subtract and multiply, while fmadd "multiplies each element of the source tensor by a
// corresponding element in the 'mulTensor', adds a corresponding element from the 'addTensor'".
// All four are declared f32 -> f32 only.
//
// Two behaviors the header does not spell out, taken from the operations' definition:
//   - The result is NOT clamped. These are 3D volumes (the legacy harness drives them with NIFTI
//     scans), not [0,1] image intensities, and the ops are plain arithmetic; clamping a
//     subtraction at 0 or a multiply at 1 would make them meaningless. The test parameters are
//     chosen so a clamp anywhere would show.
//   - Placement follows RPP's pointwise convention (for_each_voxel_roi_io): the source is read at
//     the ROI offset and the result is written packed at the destination origin. The header
//     documents nothing about the destination outside that block, so the comparator ignores it.
// Arithmetic is carried in double and stored once.

enum class ScalarArithmeticOp { Add, Subtract, Multiply, FusedMultiplyAdd };

// p0 is the addend / subtrahend / multiplier; p1 is fmadd's addend and is ignored otherwise.
inline double apply_scalar_arithmetic(double v, double p0, double p1, ScalarArithmeticOp op) {
    switch (op) {
        case ScalarArithmeticOp::Add:              return v + p0;
        case ScalarArithmeticOp::Subtract:         return v - p0;
        case ScalarArithmeticOp::Multiply:         return v * p0;
        case ScalarArithmeticOp::FusedMultiplyAdd: return v * p0 + p1;
    }
    return 0.0;
}

// param0/param1 are per-sample (batchSize values); only fmadd takes a second one. dst must already
// hold whatever the test seeded it with -- only the ROI box's output block is written.
template <typename T>
void arithmetic_scalar_reference(const T* src, T* dst, const RpptGenericDesc& desc,
                                 const RpptROI3D* roi, Roi3D roiType, ScalarArithmeticOp op,
                                 const Rpp32f* param0, const Rpp32f* param1 = nullptr) {
    for_each_voxel_roi_io(
        desc, roi, roiType,
        [&](Rpp32u n, Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
            const double p0 = static_cast<double>(param0[n]);
            const double p1 = param1 ? static_cast<double>(param1[n]) : 0.0;
            dst[dstIdx] =
                from_double<T>(apply_scalar_arithmetic(to_double(src[srcIdx]), p0, p1, op));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_ARITHMETIC_SCALAR_REF_H
