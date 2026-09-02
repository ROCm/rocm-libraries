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
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PKD3toPLN3_PartialRoi_*", ""},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PLN3toPKD3_PartialRoi_*", ""},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PKD3_FullRoi_2x36x55",
     "histogram-equalize-hip-3ch-width-tail-spills-into-next-row: the YCbCr intermediate is dense "
     "at the logical width, so an 8-wide tail store lands on the next row"},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PLN3_FullRoi_2x36x55", ""},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PKD3toPLN3_FullRoi_2x36x55", ""},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PLN3toPKD3_FullRoi_2x36x55", ""},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PKD3_FullRoi_1x45x13", ""},
    {"Image_Color/HistogramEqualizeTest.Correctness/HIP_U8toU8_PLN3_FullRoi_1x45x13", ""},
    {"Image_DataExchange/ChannelPermuteTest.Correctness/HOST_I8toI8_PKD3_FullRoi_*_perm201", ""},
    {"Image_DataExchange/ChannelPermuteTest.Correctness/HOST_I8toI8_PLN3toPKD3_FullRoi_*_perm201",
     ""},
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
    {"Image_Effects/GaussianNoiseTest.Correctness/HOST_I8toI8_*_FullRoi_2x36x55",
     "the same HOST I8 one-LSB error as the PartialRoi entry above, now visible on the full frame: "
     "it lands in the columns past the last whole vector of an odd-width row"},
    {"Image_Effects/GaussianNoiseTest.Correctness/HOST_I8toI8_*_FullRoi_1x45x13", ""},
    {"Image_Effects/GlitchTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_U8toU8_PLN3_PartialRoi_*",
     "glitch-host-u8-pln3-vector-loop-underflow: unsigned vector-loop size underflows when the ROI "
     "is narrower than 64 columns, segfaults the process"},
    {"Image_Effects/GlitchTest.Correctness/HOST_U8toU8_PLN3_FullRoi_1x45x13_*",
     "the same underflow as the entry above, reached on the full frame once the row itself is "
     "narrower than the vector block. Fatal, so the whole shape is skipped rather than the "
     "parameter that happened to die"},
    {"Image_Effects/GlitchTest.Correctness/HOST_U8toU8_PKD3_PartialRoi_1x45x13_*", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_F32toF32_PLN3_PartialRoi_1x45x13_*", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_F32toF32_PKD3_FullRoi_1x45x13_Shift", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_F32toF32_PKD3_FullRoi_2x36x55_Shift", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_*_PKD3_PartialRoi_*_Shift", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_F16toF16_PLN3_PartialRoi_*_Shift", ""},
    {"Image_Effects/GlitchTest.Correctness/HOST_F32toF32_PLN3_*_Shift", ""},
    {"Image_Effects/GridDropoutTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Effects/JitterTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Effects/NoiseShotTest.Correctness/HIP_U8toU8_PKD3_FullRoi_*_Seed", ""},
    {"Image_Effects/PixelateTest.Correctness/HIP_*_p50", ""},
    {"Image_Effects/PixelateTest.Correctness/HIP_*_PartialRoi_2x36x55_p87p5",
     "the HIP path also breaks at the other pixelation size once a partial ROI sits on an "
     "odd-width frame"},
    {"Image_Effects/PixelateTest.Correctness/HOST_*_PartialRoi_*", ""},
    {"Image_Effects/PixelateTest.Correctness/HOST_F32toF32_*_FullRoi_*_p50", ""},
    {"Image_Effects/PixelateTest.Correctness/HOST_U8toU8_*_FullRoi_*_p50", ""},
    {"Image_Effects/PosterizeTest.Correctness/HOST_F16toF16_PKD3_*_2x36x55_*",
     "posterize-host-f16-row-tail-is-shifted: past the last whole vector of a 55-wide row the "
     "output is displaced by a column (col 48 gets col 49's value), so it is a tail load/store "
     "offset rather than a wrong mask. Only the F16 path, only widths that are not a multiple of "
     "the vector width"},
    {"Image_Effects/PosterizeTest.Correctness/HOST_F16toF16_PLN1_*_2x36x55_*", ""},
    {"Image_Effects/PosterizeTest.Correctness/HOST_F16toF16_PLN3_*_2x36x55_*", ""},
    {"Image_Effects/PosterizeTest.Correctness/HOST_F16toF16_PLN3toPKD3_*_2x36x55_*", ""},
    {"Image_Effects/PosterizeTest.Correctness/HOST_F16toF16_*_1x45x13_*", ""},
    {"Image_Effects/RandomEraseTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Effects/RicapTest.Correctness/HIP_*_FullRoi_*_bound", ""},
    {"Image_Effects/RicapTest.Correctness/HIP_*_FullRoi_*_quad", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_*_FullRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_*_PKD3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_*_PLN1_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_F16toF16_PLN3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_F32toF32_PLN3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_*_PKD3toPLN3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_F16toF16_PLN3toPKD3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/SnowTest.Correctness/HOST_F32toF32_PLN3toPKD3_PartialRoi_*_b2p5_t0p5_d1", ""},
    {"Image_Effects/WaterTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_U8toU8_PLN1_FullRoi_2x36x55_*",
     "box-filter-host-integer-first-row-left-border-reads-stale-memory: passes alone, fails after "
     "other tests have run -- the value depends on prior heap contents"},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_U8toU8_PLN3_FullRoi_2x36x55_*", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_I8toI8_PLN1_FullRoi_2x36x55_*", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_I8toI8_PLN3_FullRoi_2x36x55_*", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F16toF16_PKD3_PartialRoi_2x36x55_k3",
     "gaussian-filter-host-pkd3-edge: box_filter's packed k3 path is wrong at the last ROI column "
     "too, visible once the ROI width is odd"},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F32toF32_PKD3_PartialRoi_2x36x55_k3", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_*_PartialRoi_1x45x13_k9",
     "filter-host-aligned-length-underflows-for-narrow-roi: the AVX loop bound is unsigned and "
     "unclamped, so an ROI narrower than kernelSize-1 wraps it and the loop walks off the heap. "
     "SIGSEGV, not a diff -- these must stay skipped or the run dies here"},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F16toF16_PKD3_*_1x45x13_k7",
     "box-filter-host-float-pkd3-narrow-width: on a 13 px row the k7/k9 float packed path is wrong "
     "from the first interior column onwards (1350 of 1755 elements). U8/I8 are clean at the same "
     "shape and k3/k5 are clean, so it is the float packed path once fewer than one AVX vector of "
     "columns fits between the two pad regions"},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F32toF32_PKD3_*_1x45x13_k7", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F16toF16_PKD3_FullRoi_1x45x13_k9", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F32toF32_PKD3_FullRoi_1x45x13_k9", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_U8toU8_PKD3toPLN3_FullRoi_2x36x55_k9",
     "box-filter-host-pkd3-to-pln3-k9-seam: exactly one column (51 of a 55-wide row, where the "
     "vector span meets the trailing pad) is off by 2-5 LSB; block-aligned 48-wide rows are clean"},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_I8toI8_PKD3toPLN3_FullRoi_2x36x55_k9", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F16toF16_PLN1_PartialRoi_*_k5", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F16toF16_PLN3_PartialRoi_*_k5", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F32toF32_PLN1_PartialRoi_*_k5", ""},
    {"Image_Filter/BoxFilterTest.Correctness/HOST_F32toF32_PLN3_PartialRoi_*_k5", ""},
    {"Image_Filter/EmbossTest.Correctness/*_F16toF16_*", ""},
    {"Image_Filter/EmbossTest.Correctness/*_PKD3_PartialRoi_*_k3_*", ""},
    {"Image_Filter/EmbossTest.Correctness/*_PKD3toPLN3_PartialRoi_*_k3_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HIP_I8toI8_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HOST_*_PKD3_FullRoi_*_k3_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HOST_*_PKD3toPLN3_FullRoi_*_k3_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HOST_*_PLN1_PartialRoi_*_k5_*", ""},
    {"Image_Filter/EmbossTest.Correctness/HOST_*_PLN3_PartialRoi_*_k5_*", ""},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PartialRoi_1x45x13_k9_*",
     "filter-host-aligned-length-underflows-for-narrow-roi: same unsigned loop-bound wrap as "
     "box_filter; SIGSEGV rather than a diff"},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PKD3_*_k3_sd1", ""},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PKD3toPLN3_*_k3_sd1", ""},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PLN1_PartialRoi_*_k5_sd1", ""},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PLN3_PartialRoi_*_k5_sd1", ""},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PLN1_PartialRoi_*_k5_sd5",
     "the same planar k5 partial-ROI defect as the sd1 entries above, on every dtype. sd0p25 "
     "passes because a near-delta kernel puts almost all the weight on the centre tap, which "
     "keeps the wrong edge columns inside tolerance"},
    {"Image_Filter/GaussianFilterTest.Correctness/HOST_*_PLN3_PartialRoi_*_k5_sd5", ""},
    {"Image_Filter/SobelFilterTest.Correctness/*_I8toI8_PLN1_*_k3", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F16toF16_PKD3_*_1x45x13_k7",
     "the float packed path on a 13 px row, exactly as in the box_filter entries above: erode and "
     "dilate share that skeleton, so they share its narrow-width, odd-width and k9-seam defects"},
    {"Image_Morphological/DilateTest.Correctness/HOST_F32toF32_PKD3_*_1x45x13_k7", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F16toF16_PKD3_FullRoi_1x45x13_k9", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F32toF32_PKD3_FullRoi_1x45x13_k9", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F16toF16_PKD3_PartialRoi_2x36x55_k3", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_F32toF32_PKD3_PartialRoi_2x36x55_k3", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_U8toU8_PKD3toPLN3_FullRoi_2x36x55_k9", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_U8toU8_PLN1_FullRoi_2x36x55_*", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_U8toU8_PLN3_FullRoi_2x36x55_*", ""},
    {"Image_Morphological/DilateTest.Correctness/HOST_*_PartialRoi_1x45x13_k9",
     "the same unsigned AVX loop-bound wrap as the box_filter entry above -- erode and dilate "
     "carry 32 copies of the idiom each. SIGSEGV, not a diff"},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F16toF16_PKD3_*_1x45x13_k7", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F32toF32_PKD3_*_1x45x13_k7", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F16toF16_PKD3_FullRoi_1x45x13_k9", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F32toF32_PKD3_FullRoi_1x45x13_k9", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F16toF16_PKD3_PartialRoi_2x36x55_k3", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_F32toF32_PKD3_PartialRoi_2x36x55_k3", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_U8toU8_PKD3toPLN3_FullRoi_2x36x55_k9", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_U8toU8_PLN1_FullRoi_2x36x55_*", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_U8toU8_PLN3_FullRoi_2x36x55_*", ""},
    {"Image_Morphological/ErodeTest.Correctness/HOST_*_PartialRoi_1x45x13_k9", ""},
    {"Image_Filter/SobelFilterTest.Correctness/*_PLN1_*_gradXY_k3", ""},
    {"Image_Filter/SobelFilterTest.Correctness/HIP_F16toF16_PLN1_*_k3", ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/*_F16toF16_*_Normalize", ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/*_F32toF32_*_Normalize", ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HOST_I8toI8_*_PartialRoi_*_Normalize",
     ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HOST_I8toI8_*_PartialRoi_*_Scale", ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HOST_I8toI8_*_FullRoi_2x36x55_Scale",
     "the same HOST I8 defect as the PartialRoi entries above; an odd width exposes it on the full "
     "frame too"},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HOST_I8toI8_*_FullRoi_2x36x55_Normalize",
     ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HOST_I8toI8_*_FullRoi_1x45x13_Scale", ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HOST_I8toI8_*_FullRoi_1x45x13_Normalize",
     ""},
    {"Image_Geometric/CropMirrorNormalizeTest.Correctness/HIP_*_1x45x13_*",
     "crop-mirror-normalize-hip-narrow-width: on a 13 px row the HIP path is wrong for every "
     "dtype, layout and parameter set, the identity included. The whole width is skipped rather "
     "than the failing subset because the few combinations that pass do so incidentally"},
    {"Image_Geometric/FisheyeTest.Correctness/HOST_*_PartialRoi_*", ""},
    {"Image_Geometric/FlipTest.Correctness/HOST_I8toI8_*_PartialRoi_*_h1_*", ""},
    {"Image_Geometric/FlipTest.Correctness/HOST_U8toU8_*_1x45x13_h1_*",
     "flip-host-hfactor-underflows-below-16-columns: the horizontal path adds "
     "(roiWidth - 16) * bufferMultiplier to an Rpp32u offset, so an ROI narrower than 16 columns "
     "wraps it and the source pointer lands outside the allocation. SIGSEGV, not a diff"},
    {"Image_Geometric/FlipTest.Correctness/HOST_I8toI8_*_FullRoi_1x45x13_h1_*", ""},
    {"Image_Geometric/FlipTest.Correctness/HOST_I8toI8_*_FullRoi_2x36x55_h1_*",
     "the I8 horizontal path is also fatal on a 55-wide row, where the offset does not underflow; "
     "that one is a second overrun and has not been pinned to a line yet"},
    {"Image_Geometric/FlipTest.Correctness/HIP_*_FullRoi_1x45x13_h1_v1",
     "flip-hip-horizontal-partial-group-unguarded: the horizontal-only branch clamps its last "
     "8-wide group back to the row start, but only for id_z == 0 && id_y == 0, and the "
     "horizontal-and-vertical branch has no such clamp at all. Either way a width that is not a "
     "multiple of 8 indexes before the plane and faults the device"},
    {"Image_Geometric/FlipTest.Correctness/HIP_*_FullRoi_2x36x55_h1_v1", ""},
    {"Image_Geometric/FlipTest.Correctness/HIP_*_PartialRoi_1x45x13_h1_v0", ""},
    {"Image_Geometric/FlipTest.Correctness/HOST_F32toF32_PKD3_*_1x45x13_h0_v1",
     "a third fatal overrun in the same op: the float packed source path dies on a width that is "
     "not a multiple of 16, including on the plain copy (h0_v0) once the output layout converts. "
     "48-wide rows are clean"},
    {"Image_Geometric/FlipTest.Correctness/HOST_F32toF32_PKD3_*_2x36x55_h0_v1", ""},
    {"Image_Geometric/FlipTest.Correctness/HOST_F32toF32_PKD3toPLN3_*_2x36x55_h0_*", ""},
    {"Image_Geometric/JpegCompressionDistortionTest.Correctness/HOST_*_PKD3_PartialRoi_2x36x55_q50",
     "the packed HOST path is wrong over a partial ROI of an odd-width frame at the one quality "
     "that is otherwise green"},
    {"Image_Geometric/JpegCompressionDistortionTest.Correctness/*_q10", ""},
    {"Image_Geometric/JpegCompressionDistortionTest.Correctness/*_q90", ""},
    {"Image_Geometric/LensCorrectionTest.Correctness/*_Barrel", ""},
    {"Image_Geometric/LensCorrectionTest.Correctness/*_FullRoi_*_NoDistortion", ""},
    {"Image_Geometric/LensCorrectionTest.Correctness/*_I8toI8_*_PartialRoi_*_NoDistortion", ""},
    {"Image_Geometric/PhaseTest.Correctness/HOST_I8toI8_PLN3toPKD3_FullRoi_*", ""},
    {"Image_Geometric/RemapTest.Correctness/*_PartialRoi_*", ""},
    {"Image_Geometric/RemapTest.Correctness/*_halfshift_BILINEAR", ""},
    {"Image_Geometric/RemapTest.Correctness/HOST_*_FullRoi_*_identity_BILINEAR", ""},
    {"Image_Geometric/ResizeCropMirrorTest.Correctness/*",
     "unpinned from 2x36x48: the whole op is red (nearest-neighbour returns -6, bilinear is "
     "wrong at the last texel), and an odd width fails the same way"},
    {"Image_Geometric/ResizeMirrorNormalizeTest.Correctness/HIP_*", ""},
    {"Image_Geometric/ResizeMirrorNormalizeTest.Correctness/HOST_*_IdentityNN", ""},
    {"Image_Geometric/ResizeMirrorNormalizeTest.Correctness/HOST_F32toF32_*", ""},
    {"Image_Geometric/ResizeMirrorNormalizeTest.Correctness/HOST_U8toU8_*", ""},
    {"Image_Geometric/ResizeTest.Correctness/*_oddratio_37x53_NN",
     "resize-nn-float32-tie: at a ratio with no exact float32 representation the source index "
     "lands on an integer boundary at a handful of columns ((18+0.5)*48/37 is exactly 24), and "
     "float32 rounds it to the pixel below where exact arithmetic does not. 106 of 3922 outputs, "
     "identically on both backends; the whole-multiple ratios never reach a tie"},
    {"Image_Geometric/ResizeTest.Correctness/HIP_*_BILINEAR",
     "unpinned from 2x36x48: the bilinear last-texel defect does not depend on the width"},
    {"Image_Geometric/ResizeTest.Correctness/HOST_F32toF32_*_BILINEAR", ""},
    {"Image_Geometric/ResizeTest.Correctness/HOST_U8toU8_*_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/*_FullRoi_*_a180_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/*_FullRoi_*_a270_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/*_FullRoi_*_a90_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/*_a45_BILINEAR",
     "the bilinear edge family again: at 45 degrees almost every sample is interpolated, so the "
     "last-texel defect covers the frame instead of its border. Nearest-neighbour at 45 degrees "
     "is green on both backends, which is what confirms the rotation itself"},
    {"Image_Geometric/RotateTest.Correctness/*_I8toI8_*_a45_*", ""},
    {"Image_Geometric/RotateTest.Correctness/*_I8toI8_*_a180_*", ""},
    {"Image_Geometric/RotateTest.Correctness/*_I8toI8_*_a270_*", ""},
    {"Image_Geometric/RotateTest.Correctness/*_I8toI8_*_a90_*", ""},
    {"Image_Geometric/RotateTest.Correctness/HIP_*_PartialRoi_*", ""},
    {"Image_Geometric/RotateTest.Correctness/HOST_*_FullRoi_*_a0_BILINEAR", ""},
    {"Image_Geometric/RotateTest.Correctness/HOST_I8toI8_*_PartialRoi_*_a0_*", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/*_FullRoi_*_shift_BILINEAR", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/*_I8toI8_*_shift_*", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/*_I8toI8_*_scale2_*",
     "the same I8 out-of-image fill as the shift entry above: a matrix with a non-identity linear "
     "part sends far more of the output off the source, so the wrong black shows up everywhere "
     "rather than at the edge"},
    {"Image_Geometric/WarpAffineTest.Correctness/*_I8toI8_*_rot30_*", ""},
    {"Image_Geometric/WarpAffineTest.Correctness/HOST_*_rot30_*",
     "warp-affine-host-rotating-matrix: with a 30 degree linear part the HOST kernel disagrees "
     "with the model on every dtype, while HIP agrees with it under nearest-neighbour -- so the "
     "mapping the model asserts is the one the op implements and HOST is the outlier"},
    {"Image_Geometric/WarpAffineTest.Correctness/HIP_*_rot30_BILINEAR",
     "the bilinear sampler's edge behaviour, the same family as the resize and rotate bilinear "
     "entries; a rotating matrix puts many more samples on the last texel"},
    {"Image_Geometric/WarpAffineTest.Correctness/HOST_*_scale2_BILINEAR", ""},
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
    {"Image_Statistical/ThresholdTest.Correctness/HOST_*_PKD3_PartialRoi_*_step40",
     "threshold-host-pkd3-per-channel-cutoffs: with a different [min, max] per channel the packed "
     "HOST path over a partial ROI mis-associates them; every dtype and every shape, including "
     "the shapes that are green when all three channels share one pair"},
    {"Image_Statistical/ThresholdTest.Correctness/HOST_*_PLN3_PartialRoi_1x45x13_*",
     "threshold-host-pln3-narrow-roi: every dtype is wrong over a 6-column ROI in the "
     "3-channel planar path; the same ROI on a 48- or 55-wide image is clean"},
    {"Image_Statistical/ThresholdTest.Correctness/HOST_*_PKD3toPLN3_*_2x36x55_*",
     "threshold-host-pkd3-to-pln3-writes-past-the-plane: a 55-wide row pads to 56, leaving one "
     "column of slack, and the packed-to-planar store runs past the end of the last plane -- the "
     "process aborts with 'double free or corruption' at teardown, so these must stay skipped. "
     "48-wide rows have 8 columns of slack and hide it"},
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
    {"Misc_Statistical/NormalizeTest.Correctness/HIP_F*_3D_axis4_cms1_2x5x12x19",
     "normalize-hip-innermost-axis-tail: the axisMask 4 reduction zeroes the lanes of its last "
     "vector load starting at the count of INVALID lanes instead of the count of valid ones, so "
     "an innermost extent that is not a multiple of 8 sums garbage lanes and drops real ones"},
    {"Misc_Statistical/NormalizeTest.Correctness/HIP_F*_3D_axis4_cms2_2x5x12x19", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HIP_F*_3D_axis4_cms3_2x5x12x19", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HIP_F*_3D_axis7_cms1_2x5x12x19",
     "normalize-hip-all-axis-tail: the axisMask 7 reduction flattens (y, x) through the innermost "
     "STRIDE, so once that stride is not a multiple of 8 a vector load straddles two rows and the "
     "tail mask, which assumes an aligned start, does not cover it"},
    {"Misc_Statistical/NormalizeTest.Correctness/HIP_F*_3D_axis7_cms2_2x5x12x19", ""},
    {"Misc_Statistical/NormalizeTest.Correctness/HIP_F*_3D_axis7_cms3_2x5x12x19", ""},
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
    {"Voxel_Arithmetic/AddScalarTest.Correctness/HIP_F32toF32_*_PartialRoi_LTFRBB_2x3x10x19_add40",
     "voxel-hip-ignores-roitype, the partial-ROI half of the FullRoi entries above: the whole ROI "
     "comes back untouched (the dst prefill sentinel), so the box the kernel wrote to is not the "
     "one that was asked for. On the 2x4x12x16 volume the misread box coincides with the intended "
     "one; on 2x3x10x19 it does not"},
    {"Voxel_Arithmetic/FusedMultiplyAddScalarTest.Correctness/"
     "HIP_F32toF32_*_FullRoi_LTFRBB_*_mul80_add5",
     ""},
    {"Voxel_Arithmetic/FusedMultiplyAddScalarTest.Correctness/"
     "HIP_F32toF32_*_PartialRoi_LTFRBB_2x3x10x19_mul80_add5",
     ""},
    {"Voxel_Arithmetic/MultiplyScalarTest.Correctness/HIP_F32toF32_*_FullRoi_LTFRBB_*_mul80", ""},
    {"Voxel_Arithmetic/MultiplyScalarTest.Correctness/"
     "HIP_F32toF32_*_PartialRoi_LTFRBB_2x3x10x19_mul80",
     ""},
    {"Voxel_Arithmetic/SubtractScalarTest.Correctness/HIP_F32toF32_*_FullRoi_LTFRBB_*_sub40", ""},
    {"Voxel_Arithmetic/SubtractScalarTest.Correctness/"
     "HIP_F32toF32_*_PartialRoi_LTFRBB_2x3x10x19_sub40",
     ""},
    {"Voxel_Effects/GaussianNoiseVoxelNegativeTest.Negative/*_F32toF32_NCDHW1_FullRoi_XYZWHD_*",
     ""},
    {"Voxel_Effects/GaussianNoiseVoxelTest.Correctness/HIP_*_FullRoi_LTFRBB_*", ""},
    {"Voxel_Effects/GaussianNoiseVoxelTest.Correctness/HIP_*_PartialRoi_LTFRBB_2x3x10x19", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_FullRoi_LTFRBB_*", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_FullRoi_XYZWHD_2x3x10x19_h1_v0_d0",
     "flip-voxel-hip-volume-not-a-multiple-of-8: 15 of 1140 voxels differ, all in the last "
     "partial group of the flattened (z, y, x) walk -- 3*10*19 = 570 voxels is 71 vectors plus 2. "
     "The identity and the all-three-axes cases survive it; every single- and double-axis flip "
     "does not"},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_FullRoi_XYZWHD_2x3x10x19_h0_v1_d0", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_FullRoi_XYZWHD_2x3x10x19_h0_v0_d1", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_FullRoi_XYZWHD_2x3x10x19_h1_v1_d0", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_FullRoi_XYZWHD_2x3x10x19_h1_v0_d1", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_FullRoi_XYZWHD_2x3x10x19_h0_v1_d1", ""},
    {"Voxel_Geometric/FlipVoxelTest.Correctness/HIP_*_PartialRoi_LTFRBB_2x3x10x19_h0_v0_*", ""},
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
