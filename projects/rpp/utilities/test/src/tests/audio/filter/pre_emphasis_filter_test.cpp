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

#include "framework/audio_config_param.hpp"
#include "framework/audio_setup.hpp"
#include "framework/audio_signal.hpp"
#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/skip_list.hpp"
#include "reference/pre_emphasis_filter_ref.hpp"

using namespace rpptest;

namespace {

// One coefficient per sample: a typical one, a weaker one, and a negative one.
const std::vector<Rpp32f> kCoeffs{0.97f, 0.5f, -0.3f};

struct PreEmphasisParams {
    RpptAudioBorderType border;
    std::string name() const {
        switch (border) {
            case RpptAudioBorderType::ZERO:
                return "Zero";
            case RpptAudioBorderType::CLAMP:
                return "Clamp";
            case RpptAudioBorderType::REFLECT:
                return "Reflect";
        }
        return "";
    }
};

void run_pre_emphasis_filter(const AudioConfig& cfg, const PreEmphasisParams& p) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    RpptDesc desc = make_audio_descriptor(cfg);
    const std::size_t bytes = audio_byte_size(desc);

    const std::vector<Rpp32f> input = make_audio_input(desc, lengths, Noise{});
    std::vector<Rpp32f> golden = make_audio_output(desc);
    std::vector<Rpp32f> actual = golden;
    pre_emphasis_filter_reference(input.data(), golden.data(), desc, lengths, kCoeffs, p.border);

    PinnedArray<Rpp32s> srcLength(cfg.backend, lengths.size());
    PinnedArray<Rpp32f> coeffs(cfg.backend, kCoeffs.size());
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        srcLength[i] = lengths[i];
        coeffs[i] = kCoeffs[i];
    }

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(actual.data(), bytes);

    RppHandle handle(cfg.backend, desc.n);
    ASSERT_EQ(rppt_pre_emphasis_filter(src.ptr(), &desc, dst.ptr(), &desc, srcLength.data(),
                                       coeffs.data(), p.border, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // A float multiply and subtract (or one FMA), so at most two float roundings. Padding is not
    // checked: the HIP kernel stores whole vectors.
    EXPECT_TRUE(compare_audio(actual.data(), golden.data(), desc, lengths, Bound{5e-7}));
}

}  // namespace

// Full name:
// Audio_Filter/PreEmphasisFilterTest.Correctness/<Backend>_F32toF32_1ch_<Shape>_<NxMaxLen>_<Border>
class PreEmphasisFilterTest : public SkipListTest<AudioWithParams<PreEmphasisParams>> {};

TEST_P(PreEmphasisFilterTest, Correctness) {
    const auto& p = GetParam();
    run_pre_emphasis_filter(p.cfg, p.op);
}

// Mono only: the HIP backend rejects numDims other than 2.
INSTANTIATE_TEST_SUITE_P(Audio_Filter, PreEmphasisFilterTest,
                         ::testing::ValuesIn(audio_with_params<PreEmphasisParams>(
                             make_audio_configs({1}), {{RpptAudioBorderType::ZERO},
                                                       {RpptAudioBorderType::CLAMP},
                                                       {RpptAudioBorderType::REFLECT}})),
                         audio_op_config_name<PreEmphasisParams>);
