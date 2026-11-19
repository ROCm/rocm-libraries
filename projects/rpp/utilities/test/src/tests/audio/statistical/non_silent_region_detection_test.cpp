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

#include <cmath>
#include <string>
#include <vector>

#include "framework/audio_config_param.hpp"
#include "framework/audio_setup.hpp"
#include "framework/audio_signal.hpp"
#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/skip_list.hpp"
#include "reference/non_silent_region_detection_ref.hpp"

using namespace rpptest;

namespace {

// Full-scale noise between loudBegin[n] and loudEnd[n], and low-level noise elsewhere, so
// the silence is not exact zeros. The loud span is placed differently in each sample.
struct Burst {
    std::vector<Rpp32s> loudBegin, loudEnd;

    Rpp32f operator()(Rpp32u n, Rpp32u t, Rpp32u c) const {
        const bool loud =
            static_cast<Rpp32s>(t) >= loudBegin[n] && static_cast<Rpp32s>(t) < loudEnd[n];
        return (loud ? 1.0f : 1e-4f) * Noise{}(n, t, c);
    }
};

Burst make_burst(const std::vector<Rpp32s>& lengths) {
    Burst b;
    for (std::size_t n = 0; n < lengths.size(); ++n) {
        b.loudBegin.push_back(lengths[n] * static_cast<Rpp32s>(1 + n) / 8);
        b.loudEnd.push_back(lengths[n] * static_cast<Rpp32s>(5 + n) / 8);
    }
    return b;
}

struct NonSilentParams {
    float cutOffDB;
    Rpp32s windowLength;
    float referencePower;  // 0: the maximum of the moving mean square
    Rpp32s resetInterval;  // -1: never reset
    std::string name() const {
        return "cut" + num_token(cutOffDB) + "_w" + std::to_string(windowLength) + "_ref" +
               num_token(referencePower) + "_reset" + num_token(static_cast<float>(resetInterval));
    }
};

// The result is a pair of indices, compared exactly, so the backends' float rounding of the
// moving mean square must not move any value across the cut-off. True when every value is clear of
// it by a small relative margin.
bool clear_of_cutoff(const std::vector<double>& mms, double cutOff) {
    for (double v : mms)
        if (std::fabs(v - cutOff) < 1e-3 * cutOff) return false;
    return true;
}

void run_non_silent_region_detection(const AudioConfig& cfg, const NonSilentParams& p) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    RpptDesc desc = make_audio_descriptor(cfg);
    const std::size_t bytes = audio_byte_size(desc);

    const std::vector<Rpp32f> input = make_audio_input(desc, lengths, make_burst(lengths));

    std::vector<NonSilentRegion> golden;
    for (Rpp32u n = 0; n < desc.n; ++n) {
        const Rpp32f* sample = input.data() + n * desc.strides.nStride;
        const std::vector<double> mms = moving_mean_square(sample, lengths[n], p.windowLength);
        ASSERT_TRUE(clear_of_cutoff(mms, non_silent_cutoff(mms, p.cutOffDB, p.referencePower)))
            << "test input for sample " << n << " lies too close to the cut-off";
        golden.push_back(non_silent_region_reference(sample, lengths[n], p.cutOffDB, p.windowLength,
                                                     p.referencePower));
    }

    PinnedArray<Rpp32s> srcLength(cfg.backend, lengths.size());
    PinnedArray<Rpp32s> begin(cfg.backend, lengths.size());
    PinnedArray<Rpp32s> length(cfg.backend, lengths.size());
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        srcLength[i] = lengths[i];
        begin[i] = -1;
        length[i] = -1;
    }

    DeviceTensor src(cfg.backend, bytes);
    src.write(input.data(), bytes);

    RppHandle handle(cfg.backend, desc.n);
    ASSERT_EQ(rppt_non_silent_region_detection(
                  src.ptr(), &desc, srcLength.data(), begin.data(), length.data(), p.cutOffDB,
                  p.windowLength, p.referencePower, p.resetInterval, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();

    for (Rpp32u n = 0; n < desc.n; ++n) {
        EXPECT_EQ(begin[n], golden[n].begin) << "begin of sample " << n;
        EXPECT_EQ(length[n], golden[n].length) << "length of sample " << n;
    }
}

}  // namespace

// Full name:
// Audio_Statistical/NonSilentRegionDetectionTest.Correctness/<Backend>_F32toF32_1ch_<Shape>_<NxMaxLen>_<Params>
class NonSilentRegionDetectionTest : public SkipListTest<AudioWithParams<NonSilentParams>> {};

TEST_P(NonSilentRegionDetectionTest, Correctness) {
    const auto& p = GetParam();
    run_non_silent_region_detection(p.cfg, p.op);
}

// A cut-off relative to the maximum, with no reset and with a reset interval shorter than the
// sample (which also shrinks the HIP tile size), and a cut-off relative to a fixed reference power.
// Mono only: the HIP backend rejects numDims other than 2.
INSTANTIATE_TEST_SUITE_P(Audio_Statistical, NonSilentRegionDetectionTest,
                         ::testing::ValuesIn(audio_with_params<NonSilentParams>(
                             make_audio_configs({1}), {{-60.0f, 2048, 0.0f, -1},
                                                       {-20.0f, 256, 0.0f, 1000},
                                                       {-30.0f, 100, 0.5f, -1}})),
                         audio_op_config_name<NonSilentParams>);
