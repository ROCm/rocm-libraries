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

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/jitter_ref.hpp"

using namespace rpptest;

namespace {

// The three properties checked here take disjoint parameter sets (Identity only holds at k=1, the
// others only mean anything at k>1), so the check is an axis of the grid rather than a separate
// TEST_P body: GTest instantiates every body of a fixture against the whole param set, which would
// generate (and skip) each check across the other checks' points as well.
enum class Check { Identity, Window, Seed };

// kernelSize is an odd square window size (3/5/7 per the API doc; kernelSize=1 is the RNG-free
// identity corner used for the bit-exact golden).
struct JitterParams {
    Rpp32u kernelSize;
    Check check;
    std::string name() const {
        std::string token = "k" + std::to_string(kernelSize);
        switch (check) {
            case Check::Identity: return token + "_Identity";
            case Check::Window: return token + "_Window";
            case Check::Seed: return token + "_Seed";
        }
        return token;
    }
};

// -------- Correctness: kernelSize=1 forces identity, independent of seed ----------------------

template <typename T>
void run_jitter_identity(const TestConfig& cfg, const JitterParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32u> kernelSize(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        kernelSize[i] = op.kernelSize;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input so the untouched
    // (outside-ROI) region is defined; only the ROI is overwritten by the reference.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    jitter_identity_reference<T>(input.data(), golden.data(), desc, roi.data(), XYWH);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_jitter(src.ptr(), &desc, dst.ptr(), &desc, kernelSize.data(), /*seed=*/42u,
                          roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();
    dst.read(actual.data(), bytes);

    // (4) kernelSize=1 collapses the offset window to {0,0} regardless of seed/RNG draw, so the
    // op is forced to identity: bit-exact, tolerance 0.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH, 0.0));
}

// -------- ReachableWithinWindow: structural membership invariant for kernelSize > 1 ------------
//
// Runs the real kernel (no golden model call -- kernelSize > 1 is genuinely RNG-driven) and checks
// that every output pixel equals the SOURCE value at *some* clamped offset within the
// [-r,r]x[-r,r] window around that pixel's own coordinate, i.e. it is one of the (2r+1)^2 legally
// reachable source pixels post-clamp-into-ROI. The check additionally requires the SAME candidate
// offset to work for every channel of a given pixel (jitter must not desync channels).

template <typename T>
void run_jitter_window(const TestConfig& cfg, const JitterParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32u> kernelSize(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        kernelSize[i] = op.kernelSize;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_jitter(src.ptr(), &desc, dst.ptr(), &desc, kernelSize.data(), /*seed=*/123u,
                          roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);
    handle.sync();
    dst.read(actual.data(), bytes);

    const int r = static_cast<int>(op.kernelSize / 2);
    ASSERT_GT(r, 0) << "ReachableWithinWindow requires kernelSize > 1";

    for (Rpp32u n = 0; n < desc.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], XYWH);
        const int xlo = static_cast<int>(b.x0), xhi = static_cast<int>(b.x0 + b.w - 1);
        const int ylo = static_cast<int>(b.y0), yhi = static_cast<int>(b.y0 + b.h - 1);
        for (Rpp32u j = 0; j < b.h; ++j) {
            for (Rpp32u i = 0; i < b.w; ++i) {
                const int sx = static_cast<int>(b.x0 + i);
                const int sy = static_cast<int>(b.y0 + j);
                const std::size_t dstPix = plane_index(desc, plane_base(desc, n, 0), j, i);

                bool found = false;
                for (int dy = -r; dy <= r && !found; ++dy) {
                    for (int dx = -r; dx <= r && !found; ++dx) {
                        const int csy = clamp_coord(sy + dy, ylo, yhi);
                        const int csx = clamp_coord(sx + dx, xlo, xhi);
                        bool allMatch = true;
                        for (Rpp32u c = 0; c < desc.c; ++c) {
                            const std::size_t srcIdx =
                                plane_index(desc, plane_base(desc, n, c),
                                           static_cast<std::size_t>(csy),
                                           static_cast<std::size_t>(csx));
                            const std::size_t dstIdx = channel_index(desc, dstPix, c);
                            if (to_double(actual[dstIdx]) != to_double(input[srcIdx])) {
                                allMatch = false;
                                break;
                            }
                        }
                        if (allMatch) found = true;
                    }
                }
                if (!found) {
                    ADD_FAILURE() << "n=" << n << " row=" << j << " col=" << i
                                  << ": output pixel is not any clamped candidate within the ["
                                  << -r << "," << r << "] window (kernelSize=" << op.kernelSize
                                  << ")";
                    return;
                }
            }
        }
    }
}

// -------- SeedInvariant: same seed -> bit-identical output; different seed -> some difference ---
//
// jitter takes a single explicit Rpp32u seed for the whole call (unlike e.g. rain, which seeds
// from std::random_device and is therefore never reproducible). A real per-call seed parameter must
// make repeated calls with the same seed bit-identical, and different seeds should (with
// overwhelming probability, over this many pixels) produce a divergent result somewhere in the ROI.

template <typename T>
void run_jitter_seed_invariant(const TestConfig& cfg, const JitterParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32u> kernelSize(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        kernelSize[i] = op.kernelSize;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    auto run = [&](Rpp32u seed) {
        DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
        src.write(input.data(), bytes);
        dst.write(input.data(), bytes);
        RppHandle handle(cfg.backend, shape.n);
        RppStatus status = rppt_jitter(src.ptr(), &desc, dst.ptr(), &desc, kernelSize.data(), seed,
                                       roi.data(), XYWH, handle.get(), cfg.backend);
        EXPECT_EQ(status, RPP_SUCCESS);
        handle.sync();
        std::vector<T> out(count);
        dst.read(out.data(), bytes);
        return out;
    };

    const std::vector<T> outA1 = run(42u);
    const std::vector<T> outA2 = run(42u);
    const std::vector<T> outB = run(1337u);

    // Same seed -> bit-identical over the ROI.
    EXPECT_TRUE(compare_roi<T>(outA1.data(), outA2.data(), desc, roi.data(), XYWH, 0.0));

    // Different seed -> at least one element differs somewhere within the ROI.
    bool anyDiffer = false;
    for_each_roi_io(desc, roi.data(), XYWH,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t, std::size_t dstIdx) {
                        if (outA1[dstIdx] != outB[dstIdx]) anyDiffer = true;
                    });
    EXPECT_TRUE(anyDiffer) << "different seeds produced identical output over the whole ROI";
}

// Identity gets the full dtype/layout/ROI grid at k=1; the RNG-active checks are narrower on
// purpose -- Seed runs three full kernel invocations per case, and the window invariant is a
// per-pixel candidate search over one integer and one float dtype rather than all four.
std::vector<WithParams<JitterParams>> jitter_configs() {
    std::vector<WithParams<JitterParams>> configs = with_params<JitterParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {JitterParams{1, Check::Identity}});
    const std::vector<WithParams<JitterParams>> window = with_params<JitterParams>(
        make_configs({DType::U8, DType::F32}, {Layout::PKD3, Layout::PLN1},
                     {Roi::Full, Roi::Partial}),
        {JitterParams{3, Check::Window}, JitterParams{5, Check::Window}});
    const std::vector<WithParams<JitterParams>> seed = with_params<JitterParams>(
        make_configs({DType::U8}, {Layout::PKD3}, {Roi::Full}), {JitterParams{5, Check::Seed}});
    configs.insert(configs.end(), window.begin(), window.end());
    configs.insert(configs.end(), seed.begin(), seed.end());
    return configs;
}

}  // namespace

// Full name:
// Image_Effects/JitterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_k<N>_<Check>
class JitterTest : public ::testing::TestWithParam<WithParams<JitterParams>> {};

TEST_P(JitterTest, Correctness) {
    const auto& p = GetParam();
    switch (p.op.check) {
        case Check::Identity:
            dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
                p.cfg.dtype,
                [&](auto tag) { run_jitter_identity<Element<decltype(tag)>>(p.cfg, p.op); });
            break;
        case Check::Window:
            if (p.cfg.dtype == DType::U8)
                run_jitter_window<Rpp8u>(p.cfg, p.op);
            else
                run_jitter_window<Rpp32f>(p.cfg, p.op);
            break;
        case Check::Seed:
            run_jitter_seed_invariant<Rpp8u>(p.cfg, p.op);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, JitterTest, ::testing::ValuesIn(jitter_configs()),
                         op_config_name<JitterParams>);
