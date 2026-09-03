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

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/gridmask_ref.hpp"

using namespace rpptest;

namespace {

// tileWidth <= min(h, w) as the API doc requires; 0 <= gridRatio <= 1; gridAngle in radians.
// These are single scalars applying to the whole batch (not per-image tensors), so they are baked
// into the param struct rather than allocated as pinned tensors.
struct GridmaskParams {
    Rpp32u tileWidth;
    float gridRatio;
    float gridAngle;
    Rpp32u tx, ty;
    std::string tag;
    std::string name() const {
        return tag;
    }
};

// gridmask is a pure copy/mask: kept pixels are copied bit-exact, masked pixels are set to an
// exact constant (black). No arithmetic, so every dtype is bit-exact.

template <typename T>
void run_gridmask(const TestConfig& cfg, const GridmaskParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u n = 0; n < cfg.size.n; ++n) roi[n] = roiVec[n];

    RpptUintVector2D translateVector{op.tx, op.ty};

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    gridmask_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                          XYWH, op.tileWidth, op.gridRatio, op.gridAngle, op.tx, op.ty);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(
        rppt_gridmask(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.tileWidth, op.gridRatio,
                      op.gridAngle, translateVector, roi.data(), XYWH, handle.get(), cfg.backend),
        RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare within tolerance over the ROI.
    EXPECT_TRUE(
        compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, kExact(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Effects/GridmaskTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Params>
class GridmaskTest : public SkipListTest<WithParams<GridmaskParams>> {};

TEST_P(GridmaskTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_gridmask<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Four param sets. gridRatio is chosen so the black-square edge l = gridRatio * tileWidth lands
// strictly between pixel centres (4.0 and 5.5); a ratio putting l exactly on an integer boundary
// (e.g. 0.6f * 10, which is 6.0000002 in float) makes the edge column a coin flip, not a test.
//   "axis"  -- no rotation, no translation: the grid aligned to the region origin.
//   "shift" -- translation only, so the translate direction is exercised without rotation.
//   "wide"  -- a different tileWidth/gridRatio, still axis-aligned.
//   "rot"   -- rotation only (translateVector {0,0}), so the rotate/translate order -- which the
//              API doc does not specify -- cannot affect the result.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, GridmaskTest,
    ::testing::ValuesIn(with_params<GridmaskParams>(
        concat_configs({
            make_configs({DType::U8, DType::F16, DType::F32}, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs({DType::U8, DType::F16, DType::F32}, presets::kLayoutsFull,
                         {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
        {GridmaskParams{8, 0.5f, 0.0f, 0, 0, "axis"}, GridmaskParams{8, 0.5f, 0.0f, 3, 2, "shift"},
         GridmaskParams{10, 0.55f, 0.0f, 0, 0, "wide"},
         GridmaskParams{10, 0.55f, 0.5f, 0, 0, "rot"}})),
    op_config_name<GridmaskParams>);
