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
#include "framework/config_param.hpp"
#include "framework/skip_list.hpp"
#include "reference/mel_filter_bank_ref.hpp"

using namespace rpptest;

namespace {

// The spectrogram of each audio sample, with the frame count a centered spectrogram gives for the
// config's lengths.
constexpr Rpp32u kNumBins = 512 / 2 + 1;
constexpr Rpp32s kWindowStep = 160;

struct MelFilterBankParams {
    MelFilterBankArgs args;
    std::string name() const {
        return std::string(args.formula == RpptMelScaleFormula::HTK ? "HTK" : "Slaney") + "_f" +
               std::to_string(args.numFilter) + "_" + num_token(args.minFreq) + "to" +
               num_token(args.maxFreq) + "Hz_sr" + num_token(args.sampleRate) +
               (args.normalize ? "_norm" : "");
    }
};

// A power spectrum: squared noise, so non-negative, with a different level per sample.
struct Power {
    Rpp32f operator()(Rpp32u n, Rpp32u k, Rpp32u t) const {
        const Rpp32f v = Noise{}(n, k * 1000 + t, 0);
        return (0.5f + 0.75f * n) * v * v;
    }
};

void run_mel_filter_bank(const AudioConfig& cfg, const MelFilterBankParams& p) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    const Rpp32u n = static_cast<Rpp32u>(lengths.size());
    std::vector<AudioPlane> srcPlanes(n), dstPlanes(n);
    for (Rpp32u i = 0; i < n; ++i) {
        const Rpp32u numFrames = lengths[i] / kWindowStep + 1;
        srcPlanes[i] = {kNumBins, numFrames};
        dstPlanes[i] = {static_cast<Rpp32u>(p.args.numFilter), numFrames};
    }
    RpptDesc srcDesc = make_audio_plane_descriptor(srcPlanes, NFT);
    RpptDesc dstDesc = make_audio_plane_descriptor(dstPlanes, NFT);
    const std::size_t srcBytes = audio_byte_size(srcDesc);
    const std::size_t dstBytes = audio_byte_size(dstDesc);

    std::vector<Rpp32f> input = make_audio_output(srcDesc);
    for_each_audio_plane_element(
        srcDesc, srcPlanes,
        [&](Rpp32u s, Rpp32u k, Rpp32u t, std::size_t idx) { input[idx] = Power{}(s, k, t); });
    std::vector<Rpp32f> golden = make_audio_output(dstDesc);
    std::vector<Rpp32f> actual = golden;
    mel_filter_bank_reference(input.data(), srcDesc, golden.data(), dstDesc, srcPlanes, p.args);

    PinnedArray<Rpp32s> srcDims(cfg.backend, 2 * n);
    for (Rpp32u i = 0; i < n; ++i) {
        srcDims[2 * i] = static_cast<Rpp32s>(srcPlanes[i].rows);
        srcDims[2 * i + 1] = static_cast<Rpp32s>(srcPlanes[i].cols);
    }

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(actual.data(), dstBytes);

    RppHandle handle(cfg.backend, n);
    ASSERT_EQ(rppt_mel_filter_bank(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, srcDims.data(),
                                   p.args.maxFreq, p.args.minFreq, p.args.formula, p.args.numFilter,
                                   p.args.sampleRate, p.args.normalize, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), dstBytes);

    // RPP converts the filter edges to and from mel in float, which moves each edge slightly; on a
    // narrow filter that visibly shifts the weights. The bound is relative to the largest output.
    // Padding is not checked: HOST zeroes frames past each sample's end.
    double peak = 0.0;
    for_each_audio_plane_element(dstDesc, dstPlanes, [&](Rpp32u, Rpp32u, Rpp32u, std::size_t i) {
        peak = std::max(peak, static_cast<double>(golden[i]));
    });
    EXPECT_TRUE(
        compare_audio_plane(actual.data(), golden.data(), dstDesc, dstPlanes, Bound{2e-5 * peak}));
}

}  // namespace

// Full name:
// Audio_Spectrogram/MelFilterBankTest.Correctness/<Backend>_F32toF32_1ch_<Shape>_<NxMaxLen>_<Params>
class MelFilterBankTest : public SkipListTest<AudioWithParams<MelFilterBankParams>> {};

TEST_P(MelFilterBankTest, Correctness) {
    const auto& p = GetParam();
    run_mel_filter_bank(p.cfg, p.op);
}

// Both mel formulas, with and without normalization, over the full band and a sub-band. The
// largest filter count leaves the lowest filters narrower than one bin.
INSTANTIATE_TEST_SUITE_P(Audio_Spectrogram, MelFilterBankTest,
                         ::testing::ValuesIn(audio_with_params<MelFilterBankParams>(
                             make_audio_configs({1}),
                             {{{0.0f, 0.0f, RpptMelScaleFormula::SLANEY, 128, 16000.0f, true}},
                              {{0.0f, 0.0f, RpptMelScaleFormula::HTK, 40, 16000.0f, false}},
                              {{300.0f, 6000.0f, RpptMelScaleFormula::SLANEY, 40, 16000.0f, false}},
                              {{100.0f, 9000.0f, RpptMelScaleFormula::HTK, 64, 22050.0f, true}}})),
                         audio_op_config_name<MelFilterBankParams>);
