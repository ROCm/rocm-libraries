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
#include "reference/audio_arithmetic_ref.hpp"

using namespace rpptest;

namespace {

struct MulScalarParams {
    float scalar;
    std::string name() const {
        return "s" + num_token(scalar);
    }
};

void run_audio_tensor_mul_scalar(const AudioConfig& cfg, const MulScalarParams& p) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    RpptDesc desc = make_audio_descriptor(cfg);
    const std::size_t bytes = audio_byte_size(desc);

    const std::vector<Rpp32f> input = make_audio_input(desc, lengths, Noise{});
    std::vector<Rpp32f> golden = make_audio_output(desc);
    std::vector<Rpp32f> actual = golden;
    audio_mul_scalar_reference(input.data(), golden.data(), desc, lengths, p.scalar);

    // The op's length is a float count (see the reference model).
    PinnedArray<Rpp32s> srcLength(cfg.backend, lengths.size());
    for (std::size_t i = 0; i < lengths.size(); ++i)
        srcLength[i] = lengths[i] * static_cast<Rpp32s>(cfg.channels);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(actual.data(), bytes);

    RppHandle handle(cfg.backend, desc.n);
    ASSERT_EQ(rppt_audio_tensor_mul_scalar(src.ptr(), p.scalar, &desc, dst.ptr(), &desc,
                                           srcLength.data(), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // One correctly rounded float multiply, which the double golden reproduces exactly. Padding is
    // not checked: the HIP kernel stores whole 8-float vectors over each sample's tail.
    EXPECT_TRUE(compare_audio(actual.data(), golden.data(), desc, lengths, Bound{}));
}

}  // namespace

// Full name:
// Audio_Arithmetic/AudioTensorMulScalarTest.Correctness/<Backend>_F32toF32_<C>ch_<Shape>_<NxMaxLen>_<Scalar>
class AudioTensorMulScalarTest : public SkipListTest<AudioWithParams<MulScalarParams>> {};

TEST_P(AudioTensorMulScalarTest, Correctness) {
    const auto& p = GetParam();
    run_audio_tensor_mul_scalar(p.cfg, p.op);
}

// A negative scalar catches a kernel that drops the sign; stereo checks the length covers every
// interleaved channel.
INSTANTIATE_TEST_SUITE_P(Audio_Arithmetic, AudioTensorMulScalarTest,
                         ::testing::ValuesIn(audio_with_params<MulScalarParams>(
                             make_audio_configs({1, 2}), {{2.5f}, {-0.75f}})),
                         audio_op_config_name<MulScalarParams>);
