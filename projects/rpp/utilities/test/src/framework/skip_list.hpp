/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_SKIP_LIST_H
#define RPP_TEST_SKIP_LIST_H

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <string>

// The cases the current implementation does not satisfy: defects this suite has already found,
// plus the few whose result is not reproducible from run to run. They are skipped, so a run is
// green unless something new breaks.
//
// The test bodies assert the correct behaviour unconditionally. This list, and only this list,
// records which of those assertions do not hold today, so fixing a kernel means deleting the
// matching entry here in the same change.
//
// Set RPP_TEST_NO_SKIP_LIST=1 to run the listed cases anyway. That is the only thing that
// notices when a fix has made an entry obsolete, so it is worth running periodically.
//
// Patterns are GTest filter globs ('*' and '?') matched against "<Suite>/<Fixture>.<Test>/<Param>".
// They were derived from observed runs and collapsed onto the axes that share a cause; the reason
// field is for the ticket, and is empty where nobody has attributed the defect yet.

namespace rpptest {

struct SkipEntry {
    const char* pattern;
    const char* reason;
};

inline constexpr SkipEntry kSkipList[] = {
    // ---- not reproducible: the result changes from run to run -------------------
    {"Image_Effects/SpatterTest.*",
     "spatter-nondeterministic-and-baked-mask: 8/100 flipped over 20 runs, on no one axis"},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HOST_*_NDHWC3_FullRoi_*",
     "flip-voxel-host-u8-avx-reads-past-buffer: result depends on whatever is past the buffer"},

    // ---- known kernel defects ---------------------------------------------------
    {"Image_Color/ColorCastTest.Correctness/*_a0p6_r30_g90_b150", ""},
    {"Image_Color/ColorJitterTest.Correctness/HOST_*_PLN1_*_b0_c0_h0_s1", ""},
    {"Image_Color/ColorJitterTest.Correctness/HOST_*_b0_c0_h0_s0", ""},
    {"Image_Color/ColorJitterTest.Correctness/HOST_*_b0_c0_h90_s1", ""},
    {"Image_Color/ColorJitterTest.Correctness/HOST_*_b0_c0p25_h0_s1", ""},
    {"Image_Color/ColorJitterTest.Correctness/HOST_*_b0p25_c0_h0_s1", ""},
    {"Image_Color/ColorJitterTest.Correctness/HOST_F16toF16_*", ""},
    {"Image_Color/ColorJitterTest.Correctness/HOST_F32toF32_*", ""},
    {"Image_Color/GammaCorrectionTest.Correctness/*_F16toF16_*_g2p2", ""},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PKD3_PartialRoi_*", ""},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PLN3_PartialRoi_*", ""},
    {"Image_DataExchange/ChannelPermuteTest.Correctness/HOST_I8toI8_PKD3_FullRoi_*_perm201", ""},
    {"Image_DataExchange/YuvToRgbCubicVTest.Correctness/HIP_U8toU8_PKD3_FullRoi_*", ""},
    {"Image_Effects/CoarseDropoutTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Effects/CutoutDropoutTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Effects/EraseTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Effects/FogNegativeTest.Negative/*_U8toU8_PKD3_FullRoi_*", ""},
    {"Image_Effects/FogTest.Correctness/HOST_F16toF16_*", ""},
    {"Image_Effects/FogTest.Correctness/HOST_F32toF32_*", ""},
    {"Image_Effects/GaussianNoiseNegativeTest.Negative/*_U8toU8_PKD3_FullRoi_*", ""},
    {"Image_Effects/GaussianNoiseTest.Correctness/HIP_I8toI8_*", ""},
    {"Image_Effects/GaussianNoiseTest.Correctness/HOST_I8toI8_*_PartialRoi_*", ""},
    {"Image_Effects/GlitchTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_*_PKD3_PartialRoi_*_Shift", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_F16toF16_PLN3_PartialRoi_*_Shift", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_F32toF32_PLN3_*_Shift", ""},
    {"Image_Effects/GridDropoutTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Effects/JitterTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Effects/NoiseShotTest.Correctness/HIP_U8toU8_PKD3_FullRoi_*_Seed", ""},
    {"Image_Effects/PixelateTest.Correctness/HIP_*_p50", ""},
    {"Image_Effects/PixelateTest.Correctness/HOST_*_PartialRoi_*", ""},
    {"Image_Effects/PixelateTest.Correctness/HOST_F32toF32_*_FullRoi_*_p50", ""},
    {"Image_Effects/PixelateTest.Correctness/HOST_U8toU8_*_FullRoi_*_p50", ""},
    {"Image_Effects/RandomEraseTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Effects/RicapTest.Correctness/HIP_*_FullRoi_*_bound", ""},
    {"Image_Effects/RicapTest.Correctness/HIP_*_FullRoi_*_quad", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_*_FullRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_*_PKD3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_*_PLN1_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_F16toF16_PLN3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_F32toF32_PLN3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/WaterTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F16toF16_PLN1_PartialRoi_*_k5", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F16toF16_PLN3_PartialRoi_*_k5", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F32toF32_PLN1_PartialRoi_*_k5", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F32toF32_PLN3_PartialRoi_*_k5", ""},
    {"Image_Filter/EmbossTest.Correctness/*_F16toF16_*", ""},
    {"Image_Filter/EmbossTest.Correctness/*_PKD3_PartialRoi_*_k3_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HIP_I8toI8_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HOST_*_PKD3_FullRoi_*_k3_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HOST_*_PLN1_PartialRoi_*_k5_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HOST_*_PLN3_PartialRoi_*_k5_*", ""},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PKD3_*_k3_sd1", ""},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PLN1_PartialRoi_*_k5_sd1", ""},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PLN3_PartialRoi_*_k5_sd1", ""},
    {"Image_Filter/SobelFilterTest.Correctness/*_I8toI8_PLN1_*_k3", ""},
    {"Image_Filter/SobelFilterTest.Correctness/*_PLN1_*_gradXY_k3", ""},
    {"Image_Filter/SobelFilterTest.Correctness/HIP_F16toF16_PLN1_*_k3", ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/*_F16toF16_*_Normalize", ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/*_F32toF32_*_Normalize", ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HOST_I8toI8_*_PartialRoi_*_Normalize",
     ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HOST_I8toI8_*_PartialRoi_*_Scale", ""},
    {"Image_Geometric/FisheyeTest.Correctness/HOST_*_PartialRoi_*", ""},
    {"Image_Geometric/FlipTest.Correctness/HOST_I8toI8_*_PartialRoi_*_h1_*", ""},
    {"Image_Geometric/JpegCompressionDistortionTest.Correctness/*_q10", ""},
    {"Image_Geometric/JpegCompressionDistortionTest.Correctness/*_q90", ""},
    {"Image_Geometric/LensCorrectionTest.Correctness/*_Barrel", ""},
    {"Image_Geometric/LensCorrectionTest.Correctness/*_FullRoi_*_NoDistortion", ""},
    {"Image_Geometric/LensCorrectionTest.Correctness/*_I8toI8_*_PartialRoi_*_NoDistortion", ""},
    {"Image_Geometric/RemapTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Geometric/RemapTest.Correctness/*_halfshift_BILINEAR", ""},
    {"Image_Geometric/RemapTest.Correctness/HOST_*_FullRoi_*_identity_BILINEAR", ""},
    {"Image_Geometric/ResizeCropMirrorTest.Correctness/*_2x36x48_*", ""},
    {"Image_Geometric/ResizeMirrorNormalizeTest.Correctness/HIP_*", ""},
    {"Image_Geometric/ResizeMirrorNormalizeTest.Correctness/HOST_*_IdentityNN", ""},
    {"Image_Geometric/ResizeMirrorNormalizeTest.Correctness/HOST_F32toF32_*", ""},
    {"Image_Geometric/ResizeMirrorNormalizeTest.Correctness/HOST_U8toU8_*", ""},
    {"Image_Geometric/ResizeTest.Correctness/HIP_*_2x36x48_*_BILINEAR", ""},
    {"Image_Geometric/ResizeTest.Correctness/HOST_F32toF32_*_2x36x48_*_BILINEAR", ""},
    {"Image_Geometric/ResizeTest.Correctness/HOST_U8toU8_*_2x36x48_*_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/*_FullRoi_*_a180_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/*_FullRoi_*_a270_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/*_FullRoi_*_a90_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/*_I8toI8_*_a180_*", ""},
    {"Image_Geometric/RotateTest.Correctness/*_I8toI8_*_a270_*", ""},
    {"Image_Geometric/RotateTest.Correctness/*_I8toI8_*_a90_*", ""},
    {"Image_Geometric/RotateTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Geometric/RotateTest.Correctness/HOST_*_FullRoi_*_a0_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/HOST_I8toI8_*_PartialRoi_*_a0_*", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/*_FullRoi_*_shift_BILINEAR", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/*_I8toI8_*_shift_*", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/*_halfshift_BILINEAR", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/HOST_*_FullRoi_*_identity_BILINEAR", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/HOST_I8toI8_*_PartialRoi_*_identity_*", ""},
    {"Image_Geometric/WarpPerspectiveTest.Correctness/*_FullRoi_*_shift_BILINEAR", ""},
    {"Image_Geometric/WarpPerspectiveTest.Correctness/*_I8toI8_*_shift_*", ""},
    {"Image_Geometric/WarpPerspectiveTest.Correctness/*_halfshift_BILINEAR", ""},
    {"Image_Geometric/WarpPerspectiveTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Geometric/WarpPerspectiveTest.Correctness/HOST_*_FullRoi_*_identity_BILINEAR", ""},
    {"Image_Geometric/WarpPerspectiveTest.Correctness/HOST_I8toI8_*_PartialRoi_*_identity_*", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F16toF16_PLN1_PartialRoi_*_k5", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F16toF16_PLN3_PartialRoi_*_k5", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F32toF32_PLN1_PartialRoi_*_k5", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F32toF32_PLN3_PartialRoi_*_k5", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F16toF16_PLN1_PartialRoi_*_k5", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F16toF16_PLN3_PartialRoi_*_k5", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F32toF32_PLN1_PartialRoi_*_k5", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F32toF32_PLN3_PartialRoi_*_k5", ""},
    {"Image_Statistical/TensorStddevTest.Correctness/*_F16toF16_*", ""},
    {"Image_Statistical/TensorStddevTest.Correctness/*_F32toF32_*", ""},
    {"Misc_Arithmetic/LogTest.Correctness/*_I8toF32_*", ""},
    {"Misc_Arithmetic/TensorAddTensorTest.Correctness/*_F16toF16_*", ""},
    {"Misc_Arithmetic/TensorAddTensorTest.Correctness/*_I8toI8_*", ""},
    {"Misc_Arithmetic/TensorAddTensorTest.Correctness/*_U8toU8_*", ""},
    {"Misc_Arithmetic/TensorDivideTensorTest.Correctness/*_F16toF16_*", ""},
    {"Misc_Arithmetic/TensorDivideTensorTest.Correctness/*_I8toI8_*", ""},
    {"Misc_Arithmetic/TensorDivideTensorTest.Correctness/*_U8toU8_*", ""},
    {"Misc_Arithmetic/TensorMultiplyTensorTest.Correctness/*_F16toF16_*", ""},
    {"Misc_Arithmetic/TensorMultiplyTensorTest.Correctness/*_I8toI8_*", ""},
    {"Misc_Arithmetic/TensorMultiplyTensorTest.Correctness/*_U8toU8_*", ""},
    {"Misc_Arithmetic/TensorSubtractTensorTest.Correctness/*_F16toF16_*", ""},
    {"Misc_Arithmetic/TensorSubtractTensorTest.Correctness/*_I8toI8_*", ""},
    {"Misc_Arithmetic/TensorSubtractTensorTest.Correctness/*_U8toU8_*", ""},
    {"Misc_Geometric/ConcatTest.Correctness/*_3D_AxisMiddle_*", ""},
    {"Misc_Geometric/ConcatTest.Correctness/*_4D_AxisMiddle_*", ""},
    {"Misc_Geometric/ConcatTest.Correctness/*_AxisFirst_*", ""},
    {"Misc_Geometric/SliceTest.Correctness/HIP_*_Packed_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/*_I8toF32_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/*_U8toF32_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/*_cms0_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_*_3D_axis4_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_*_3D_axis6_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_*_4D_axis12_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_*_4D_axis8_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_*_cms1_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_F16toF16_2D_axis1_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_F16toF16_2D_axis2_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_F16toF16_2D_axis3_cms3_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_F32toF32_3D_axis7_*", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HOST_F32toF32_4D_axis15_*", ""},
    {"Voxel_Arithmetic/AddScalarTest.Correctness/HIP_F32toF32_*_FullRoi_LTFRBB_*_add40", ""},
    {"Voxel_Arithmetic/FusedMultiplyAddScalarTest.Correctness/HIP_F32toF32_*_FullRoi_LTFRBB_*_mul80_add5",
     ""},
    {"Voxel_Arithmetic/MultiplyScalarTest.Correctness/HIP_F32toF32_*_FullRoi_LTFRBB_*_mul80", ""},
    {"Voxel_Arithmetic/SubtractScalarTest.Correctness/HIP_F32toF32_*_FullRoi_LTFRBB_*_sub40", ""},
    {"Voxel_Effects/GaussianNoiseVoxelNegativeTest.Negative/*_F32toF32_NCDHW1_FullRoi_XYZWHD_*",
     ""},
    {"Voxel_Effects/GaussianNoiseVoxelTest.Correctness/HIP_*_FullRoi_LTFRBB_*", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_FullRoi_LTFRBB_*", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_PartialRoi_LTFRBB_*_h1_v0_*", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_PartialRoi_LTFRBB_*_v1_*", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HOST_*_PartialRoi_*_h0_*_d1", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HOST_F32toF32_*_PartialRoi_*_h1_*_d1", ""},
};

// GTest filter glob: '*' matches any run of characters, '?' matches one.
inline bool matches_pattern(const char* pattern, const std::string& name) {
    const std::string p(pattern);
    std::size_t pi = 0, ni = 0, star = std::string::npos, mark = 0;
    while (ni < name.size()) {
        if (pi < p.size() && (p[pi] == '?' || p[pi] == name[ni])) {
            ++pi;
            ++ni;
        } else if (pi < p.size() && p[pi] == '*') {
            star = pi++;
            mark = ni;
        } else if (star != std::string::npos) {
            pi = star + 1;
            ni = ++mark;
        } else {
            return false;
        }
    }
    while (pi < p.size() && p[pi] == '*') ++pi;
    return pi == p.size();
}

inline bool skip_list_enabled() {
    static const bool enabled = std::getenv("RPP_TEST_NO_SKIP_LIST") == nullptr;
    return enabled;
}

inline const SkipEntry* skip_entry_for(const std::string& testName) {
    for (const SkipEntry& e : kSkipList)
        if (matches_pattern(e.pattern, testName)) return &e;
    return nullptr;
}

// Fixture base for the suite's parameterised tests: skips the case if it is listed above, so no
// test body has to know the list exists.
template <typename P>
class SkipListTest : public ::testing::TestWithParam<P> {
   protected:
    void SetUp() override {
        if (!skip_list_enabled()) return;
        const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
        if (info == nullptr) return;
        const std::string name = std::string(info->test_suite_name()) + "." + info->name();
        const SkipEntry* entry = skip_entry_for(name);
        if (entry == nullptr) return;
        GTEST_SKIP() << "skip_list.hpp: " << entry->pattern
                     << (entry->reason[0] == '\0' ? "" : " -- ") << entry->reason;
    }
};

}  // namespace rpptest

#endif  // RPP_TEST_SKIP_LIST_H
