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

#include "framework/audio_config_param.hpp"
#include "framework/audio_setup.hpp"
#include "framework/audio_signal.hpp"
#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/skip_list.hpp"
#include "reference/audio_arithmetic_ref.hpp"

using namespace rpptest;

namespace {

// One addend per sample, each different and one negative, so a kernel that applies one sample's
// addend to another is caught.
const std::vector<Rpp32f> kAddends{0.25f, -0.5f, 1.75f};

void run_audio_tensor_add_tensor(const AudioConfig& cfg) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    RpptDesc desc = make_audio_descriptor(cfg);
    const std::size_t bytes = audio_byte_size(desc);
    const std::size_t addendBytes = desc.n * sizeof(Rpp32f);

    const std::vector<Rpp32f> input = make_audio_input(desc, lengths, Noise{});
    std::vector<Rpp32f> golden = make_audio_output(desc);
    std::vector<Rpp32f> actual = golden;
    audio_add_tensor_reference(input.data(), kAddends.data(), golden.data(), desc, lengths);

    // The op's length is a float count, as for mul_scalar.
    PinnedArray<Rpp32s> srcLength(cfg.backend, lengths.size());
    for (std::size_t i = 0; i < lengths.size(); ++i)
        srcLength[i] = lengths[i] * static_cast<Rpp32s>(cfg.channels);

    DeviceTensor src(cfg.backend, bytes), addends(cfg.backend, addendBytes),
        dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    addends.write(kAddends.data(), addendBytes);
    dst.write(actual.data(), bytes);

    RppHandle handle(cfg.backend, desc.n);
    ASSERT_EQ(rppt_audio_tensor_add_tensor(src.ptr(), addends.ptr(), &desc, dst.ptr(), &desc,
                                           srcLength.data(), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // One correctly rounded float add, which the double golden reproduces exactly. Padding is not
    // checked: the HIP kernel stores whole vectors over each sample's tail.
    EXPECT_TRUE(compare_audio(actual.data(), golden.data(), desc, lengths, Bound{}));
}

}  // namespace

// Full name:
// Audio_Arithmetic/AudioTensorAddTensorTest.Correctness/<Backend>_F32toF32_<C>ch_<Shape>_<NxMaxLen>
class AudioTensorAddTensorTest : public SkipListTest<AudioConfig> {};

TEST_P(AudioTensorAddTensorTest, Correctness) {
    run_audio_tensor_add_tensor(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Audio_Arithmetic, AudioTensorAddTensorTest,
                         ::testing::ValuesIn(make_audio_configs({1, 2})), audio_config_param_name);
