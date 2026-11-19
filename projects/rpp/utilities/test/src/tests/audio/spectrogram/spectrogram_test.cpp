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
#include "reference/spectrogram_ref.hpp"

using namespace rpptest;

namespace {

struct SpectrogramParams {
    RpptLayout layout;
    bool centerWindows;
    bool reflectPadding;
    Rpp32s power;
    Rpp32s windowLength;
    Rpp32s windowStep;
    Rpp32s nfft;        // 0: windowLength
    bool customWindow;  // a Hamming window passed in, instead of the default Hann
    std::string name() const {
        return std::string(layout == NFT ? "NFT" : "NTF") + (centerWindows ? "_center" : "") +
               (reflectPadding ? "_reflect" : "_zero") + "_pow" + std::to_string(power) + "_win" +
               std::to_string(windowLength) + "_step" + std::to_string(windowStep) + "_nfft" +
               std::to_string(nfft) + (customWindow ? "_hamming" : "_hann");
    }
};

std::vector<double> hamming_window(Rpp32s windowLength) {
    std::vector<double> window(windowLength);
    for (Rpp32s j = 0; j < windowLength; ++j)
        window[j] = 0.54 - 0.46 * std::cos(2.0 * M_PI * j / (windowLength - 1));
    return window;
}

void run_spectrogram(const AudioConfig& cfg, const SpectrogramParams& p) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    RpptDesc srcDesc = make_audio_descriptor(cfg);
    const Rpp32u n = srcDesc.n;

    SpectrogramArgs args{
        p.centerWindows,
        p.reflectPadding,
        p.customWindow ? hamming_window(p.windowLength) : spectrogram_hann_window(p.windowLength),
        p.nfft == 0 ? p.windowLength : p.nfft,
        p.power,
        p.windowStep};
    const Rpp32u numBins = args.nfft / 2 + 1;
    std::vector<AudioPlane> planes(n);
    for (Rpp32u i = 0; i < n; ++i) {
        const Rpp32u numWindows =
            spectrogram_num_windows(lengths[i], p.windowLength, p.windowStep, p.centerWindows);
        planes[i] =
            p.layout == NFT ? AudioPlane{numBins, numWindows} : AudioPlane{numWindows, numBins};
    }
    RpptDesc dstDesc = make_audio_plane_descriptor(planes, p.layout);
    const std::size_t srcBytes = audio_byte_size(srcDesc);
    const std::size_t dstBytes = audio_byte_size(dstDesc);

    // White noise puts energy in every bin. Tones leave most bins in window leakage far below the
    // peak-relative bound, where any value would pass.
    const std::vector<Rpp32f> input = make_audio_input(srcDesc, lengths, Noise{});
    std::vector<Rpp32f> golden = make_audio_output(dstDesc);
    std::vector<Rpp32f> actual = golden;
    spectrogram_reference(input.data(), srcDesc, golden.data(), dstDesc, lengths, args);

    PinnedArray<Rpp32s> srcLengths(cfg.backend, n);
    std::copy(lengths.begin(), lengths.end(), srcLengths.data());
    PinnedArray<Rpp32f> window(cfg.backend, p.windowLength);
    std::copy(args.window.begin(), args.window.end(), window.data());

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(actual.data(), dstBytes);

    RppHandle handle(cfg.backend, n);
    ASSERT_EQ(rppt_spectrogram(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, srcLengths.data(),
                               p.centerWindows, p.reflectPadding,
                               p.customWindow ? window.data() : nullptr, p.nfft, p.power,
                               p.windowLength, p.windowStep, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), dstBytes);

    // A float FFT's rounding error scales with the frame's energy rather than with each bin, so
    // the bound is relative to the largest output.
    double peak = 0.0;
    for_each_audio_plane_element(dstDesc, planes, [&](Rpp32u, Rpp32u, Rpp32u, std::size_t i) {
        peak = std::max(peak, static_cast<double>(golden[i]));
    });
    EXPECT_TRUE(
        compare_audio_plane(actual.data(), golden.data(), dstDesc, planes, Bound{2e-6 * peak}));
}

}  // namespace

// Full name:
// Audio_Spectrogram/SpectrogramTest.Correctness/<Backend>_F32toF32_1ch_<Shape>_<NxMaxLen>_<Params>
class SpectrogramTest : public SkipListTest<AudioWithParams<SpectrogramParams>> {};

TEST_P(SpectrogramTest, Correctness) {
    const auto& p = GetParam();
    run_spectrogram(p.cfg, p.op);
}

// Mono only: HIP rejects other numDims. Covers both layouts, both padding modes with and without
// centered windows, both powers, nfft larger than the window and equal to it, a non-power-of-2
// nfft (Bluestein on HOST) and a caller-supplied window.
INSTANTIATE_TEST_SUITE_P(Audio_Spectrogram, SpectrogramTest,
                         ::testing::ValuesIn(audio_with_params<SpectrogramParams>(
                             make_audio_configs({1}), {{NFT, true, true, 2, 320, 160, 512, false},
                                                       {NTF, true, false, 1, 320, 160, 512, false},
                                                       {NFT, false, true, 1, 256, 128, 0, true},
                                                       {NTF, false, false, 2, 400, 100, 0, false},
                                                       {NTF, true, true, 2, 200, 64, 256, true}})),
                         audio_op_config_name<SpectrogramParams>);
