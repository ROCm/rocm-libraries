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
#include "framework/config_param.hpp"
#include "framework/skip_list.hpp"
#include "reference/to_decibels_ref.hpp"

using namespace rpptest;

namespace {

// A power signal: squared noise, so non-negative and spread over several decades, scaled per
// sample so each sample has a different maximum.
struct Power {
    Rpp32f operator()(Rpp32u n, Rpp32u t, Rpp32u c) const {
        const Rpp32f v = Noise{}(n, t, c);
        return (0.5f + 0.75f * n) * v * v;
    }
};

struct ToDecibelsParams {
    float cutOffDB;
    float multiplier;
    float referenceMagnitude;  // 0: each sample's maximum
    std::string name() const {
        return "cut" + num_token(cutOffDB) + "_m" + num_token(multiplier) + "_ref" +
               num_token(referenceMagnitude);
    }
};

void run_to_decibels(const AudioConfig& cfg, const ToDecibelsParams& p) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    RpptDesc desc = make_audio_descriptor(cfg);
    const std::size_t bytes = audio_byte_size(desc);

    const std::vector<Rpp32f> input = make_audio_input(desc, lengths, Power{});
    std::vector<Rpp32f> golden = make_audio_output(desc);
    std::vector<Rpp32f> actual = golden;
    to_decibels_reference(input.data(), golden.data(), desc, lengths, p.cutOffDB, p.multiplier,
                          p.referenceMagnitude);

    const std::vector<RpptImagePatch> patches = audio_patch_tensor(lengths);
    PinnedArray<RpptImagePatch> srcDims(cfg.backend, patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i) srcDims[i] = patches[i];

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(actual.data(), bytes);

    RppHandle handle(cfg.backend, desc.n);
    ASSERT_EQ(rppt_to_decibels(src.ptr(), &desc, dst.ptr(), &desc, srcDims.data(), p.cutOffDB,
                               p.multiplier, p.referenceMagnitude, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // HOST computes the ratio and log in float, so both the ratio's and the result's rounding show
    // up. Padding is not checked: the HIP kernel stores whole vectors.
    EXPECT_TRUE(compare_audio(actual.data(), golden.data(), desc, lengths, Bound{1e-5, 1e-6}));
}

}  // namespace

// Full name:
// Audio_Arithmetic/ToDecibelsTest.Correctness/<Backend>_F32toF32_1ch_<Shape>_<NxMaxLen>_<Params>
class ToDecibelsTest : public SkipListTest<AudioWithParams<ToDecibelsParams>> {};

TEST_P(ToDecibelsTest, Correctness) {
    const auto& p = GetParam();
    run_to_decibels(p.cfg, p.op);
}

// Power dB against each sample's maximum, and amplitude dB against a fixed reference. Both
// cut-offs floor a few percent of the signal. Mono only: see the reference model.
INSTANTIATE_TEST_SUITE_P(Audio_Arithmetic, ToDecibelsTest,
                         ::testing::ValuesIn(audio_with_params<ToDecibelsParams>(
                             make_audio_configs({1}),
                             {{-30.0f, 10.0f, 0.0f}, {-40.0f, 20.0f, 2.0f}})),
                         audio_op_config_name<ToDecibelsParams>);
