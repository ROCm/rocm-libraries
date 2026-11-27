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

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "framework/audio_config_param.hpp"
#include "framework/audio_setup.hpp"
#include "framework/audio_signal.hpp"
#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/skip_list.hpp"
#include "reference/resample_ref.hpp"

using namespace rpptest;

namespace {

struct Rates {
    Rpp32f in;
    Rpp32f out;
};

// rates[n % size] is sample n's rate pair.
struct ResampleParams {
    std::string label;
    std::vector<Rates> rates;
    Rpp32s lobes;
    Bound bound;
    std::string name() const {
        return label + "_lobes" + std::to_string(lobes);
    }
};

// The windowed-sinc table the legacy suite and MIVisionX build: a Hann-windowed sinc with `lobes`
// zero crossings each side. HOST reads lookup, HIP reads lookupPinned.
class ResamplingWindow {
   public:
    ResamplingWindow(RppBackend backend, Rpp32s lobes)
        : coeffs_(lobes * 64 + 1), window_(backend, 1), pinnedLookup_(backend, coeffs_ + 5) {
        RpptResamplingWindow& w = window_[0];
        const Rpp32s center = (coeffs_ - 1) / 2;
        const double sincStep = 2.0 * lobes / (coeffs_ - 1);
        const double hannStep = 2.0 / coeffs_;
        w.lobes = lobes;
        w.coeffs = coeffs_;
        w.lookupSize = coeffs_ + 5;
        w.lookup.assign(w.lookupSize, 0.0f);
        for (Rpp32s i = 0; i < coeffs_; ++i) {
            const double x = M_PI * (i - center) * sincStep;
            const double sinc = x == 0.0 ? 1.0 : std::sin(x) / x;
            const double hann = 0.5 * (1.0 + std::cos(M_PI * (i - center) * hannStep));
            w.lookup[i + 1] = static_cast<Rpp32f>(sinc * hann);
        }
        w.center = center + 1;
        w.scale = 1.0f / static_cast<Rpp32f>(sincStep);
        w.pCenter = _mm_set1_ps(w.center);
        w.pScale = _mm_set1_ps(w.scale);
        std::copy(w.lookup.begin(), w.lookup.end(), pinnedLookup_.data());
        if (backend == RPP_HIP_BACKEND) w.lookupPinned = pinnedLookup_.data();
    }

    RpptResamplingWindow& get() {
        return window_[0];
    }

   private:
    Rpp32s coeffs_;
    PinnedArray<RpptResamplingWindow> window_;  // HIP reads the struct itself on the device
    PinnedArray<Rpp32f> pinnedLookup_;
};

void run_resample(const AudioConfig& cfg, const ResampleParams& p) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    RpptDesc srcDesc = make_audio_descriptor(cfg);
    const Rpp32u n = srcDesc.n;

    std::vector<Rpp32f> inRates(n), outRates(n);
    std::vector<Rpp32s> outLengths(n);
    for (Rpp32u i = 0; i < n; ++i) {
        inRates[i] = p.rates[i % p.rates.size()].in;
        outRates[i] = p.rates[i % p.rates.size()].out;
        outLengths[i] = resampled_length(lengths[i], inRates[i], outRates[i]);
    }
    RpptDesc dstDesc =
        make_audio_descriptor(n, static_cast<Rpp32u>(max_audio_length(outLengths)), cfg.channels);
    const std::size_t srcBytes = audio_byte_size(srcDesc);
    const std::size_t dstBytes = audio_byte_size(dstDesc);

    ResamplingWindow window(cfg.backend, p.lobes);

    const std::vector<Rpp32f> input = make_audio_input(srcDesc, lengths, Tones{});
    std::vector<Rpp32f> golden = make_audio_output(dstDesc);
    std::vector<Rpp32f> actual = golden;
    resample_reference(input.data(), srcDesc, golden.data(), dstDesc, lengths, inRates, outRates,
                       window.get());

    const std::vector<Rpp32s> dims = audio_dims_tensor(lengths, cfg.channels);
    PinnedArray<Rpp32s> srcDims(cfg.backend, dims.size());
    PinnedArray<Rpp32f> inRateTensor(cfg.backend, n), outRateTensor(cfg.backend, n);
    std::copy(dims.begin(), dims.end(), srcDims.data());
    std::copy(inRates.begin(), inRates.end(), inRateTensor.data());
    std::copy(outRates.begin(), outRates.end(), outRateTensor.data());

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(actual.data(), dstBytes);

    RppHandle handle(cfg.backend, n);
    ASSERT_EQ(rppt_resample(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, inRateTensor.data(),
                            outRateTensor.data(), srcDims.data(), window.get(), handle.get(),
                            cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), dstBytes);

    EXPECT_TRUE(compare_audio(actual.data(), golden.data(), dstDesc, outLengths, p.bound));
    EXPECT_TRUE(compare_audio_padding(actual.data(), dstDesc, outLengths));
}

}  // namespace

// Full name:
// Audio_Filter/ResampleTest.Correctness/<Backend>_F32toF32_<C>ch_<Shape>_<NxMaxLen>_<Rates>_<Lobes>
class ResampleTest : public SkipListTest<AudioWithParams<ResampleParams>> {};

TEST_P(ResampleTest, Correctness) {
    const auto& p = GetParam();
    run_resample(p.cfg, p.op);
}

// Both kernels step the input position in float across each block and resync it from double at
// the next block. When inRate / outRate is not exact in float the position drifts within a block,
// which moves the output by the drift times the signal's slope. An exact step (Up2x) leaves only
// float sums.
// Channels cover the mono and interleaved paths on each backend, up to RPPT_MAX_AUDIO_CHANNELS.
// Mixed passes one sample per batch through unchanged (the copy path) beside two resampled ones,
// with a short window: one SSE step plus a scalar tail on HOST, scalar only on HIP.
INSTANTIATE_TEST_SUITE_P(
    Audio_Filter, ResampleTest,
    ::testing::ValuesIn(audio_with_params<ResampleParams>(
        make_audio_configs({1, 2, 16}),
        {{"Up2x", {{16000, 32000}}, 16, Bound{1e-6}},
         {"Up1p15", {{16000, 18400}}, 16, Bound{5e-4}},
         {"Down", {{48000, 44100}}, 16, Bound{5e-4}},
         {"Mixed", {{16000, 16000}, {22050, 16000}, {16000, 44100}}, 3, Bound{5e-4}}})),
    audio_op_config_name<ResampleParams>);
