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

#include "framework/audio_setup.hpp"

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/audio_config_param.hpp"
#include "framework/audio_signal.hpp"
#include "framework/compare.hpp"

using namespace rpptest;

// The audio harness itself, checked without calling RPP: every audio test trusts the descriptor,
// the traversals and the generators, so a fault here would hide or invent op failures.

namespace {

const Rpp32u kChannelCounts[] = {1, 2, 3};

}  // namespace

TEST(AudioSetupTest, DescriptorMatchesTheApiContract) {
    for (Rpp32u channels : kChannelCounts) {
        const RpptDesc d = make_audio_descriptor(3, 4099, channels);
        EXPECT_EQ(d.numDims, channels == 1 ? 2u : 3u) << channels << " channels";
        EXPECT_EQ(d.dataType, F32);
        EXPECT_EQ(d.w, 4099u);
        EXPECT_EQ(d.h, 1u);
        EXPECT_EQ(d.strides.wStride, channels);
        EXPECT_EQ(d.strides.cStride, 1u);
        // Room for a full 8-float store over the last frame, from any 8-aligned start.
        EXPECT_GT(d.strides.nStride, 4099u * channels);
        EXPECT_EQ(d.strides.nStride % 8, 0u);
    }
}

// Each element of the allocation is visited exactly once across the two traversals, and the
// in-length traversal stays inside its own sample's frames.
TEST(AudioSetupTest, TraversalsPartitionTheAllocation) {
    const std::vector<Rpp32s> lengths = audio_lengths(AudioShape::Ragged);
    for (Rpp32u channels : kChannelCounts) {
        const RpptDesc d = make_audio_descriptor(static_cast<Rpp32u>(lengths.size()),
                                                 max_audio_length(lengths), channels);
        std::vector<int> visits(audio_element_count(d), 0);

        for_each_audio_element(d, lengths, [&](Rpp32u n, Rpp32u t, Rpp32u c, std::size_t idx) {
            ASSERT_LT(t, static_cast<Rpp32u>(lengths[n]));
            ASSERT_LT(c, channels);
            ASSERT_GE(idx, n * d.strides.nStride);
            ASSERT_LT(idx, n * d.strides.nStride + lengths[n] * channels);
            ++visits[idx];
        });
        for_each_audio_padding(d, lengths, [&](Rpp32u, std::size_t idx) { ++visits[idx]; });

        for (std::size_t i = 0; i < visits.size(); ++i)
            ASSERT_EQ(visits[i], 1) << "element " << i << ", " << channels << " channels";
    }
}

TEST(AudioSetupTest, LengthArrays) {
    const std::vector<Rpp32s> lengths = {7, 5};
    EXPECT_EQ(audio_dims_tensor(lengths, 2), (std::vector<Rpp32s>{7, 2, 5, 2}));

    const std::vector<RpptImagePatch> patches = audio_patch_tensor(lengths);
    ASSERT_EQ(patches.size(), 2u);
    EXPECT_EQ(patches[0].height, 7u);
    EXPECT_EQ(patches[0].width, 1u);
    EXPECT_EQ(patches[1].height, 5u);
}

// The input carries the signal inside the lengths and the sentinel everywhere else, which is what
// compare_audio_padding() relies on; a fresh output passes it and fails compare_audio().
TEST(AudioSetupTest, BuffersAndComparators) {
    const std::vector<Rpp32s> lengths = audio_lengths(AudioShape::Ragged);
    const RpptDesc d =
        make_audio_descriptor(static_cast<Rpp32u>(lengths.size()), max_audio_length(lengths), 2);
    const std::vector<Rpp32f> input = make_audio_input(d, lengths, Tones{});
    const std::vector<Rpp32f> output = make_audio_output(d);

    EXPECT_TRUE(compare_audio_padding(input.data(), d, lengths));
    EXPECT_TRUE(compare_audio(input.data(), input.data(), d, lengths, Bound{}));
    EXPECT_TRUE(compare_audio_padding(output.data(), d, lengths));
    EXPECT_FALSE(compare_audio(output.data(), input.data(), d, lengths, Bound{}));

    std::vector<Rpp32f> overrun = input;
    overrun[audio_index(d, 1, lengths[1], 0)] = 0.0f;
    EXPECT_FALSE(compare_audio_padding(overrun.data(), d, lengths));
}

TEST(AudioSignalTest, BoundedAndDeterministic) {
    const Tones tones;
    const Noise noise{7};
    for (Rpp32u n = 0; n < 3; ++n)
        for (Rpp32u c = 0; c < 3; ++c)
            for (Rpp32u t = 0; t < 4099; ++t) {
                ASSERT_LE(std::fabs(tones(n, t, c)), 1.0f);
                ASSERT_LE(std::fabs(noise(n, t, c)), 1.0f);
                ASSERT_EQ(noise(n, t, c), Noise{7}(n, t, c));
            }
}

// A kernel that reads the wrong sample or channel must change its output, so no two (n, c)
// streams may be equal.
TEST(AudioSignalTest, SamplesAndChannelsDiffer) {
    auto stream = [](auto signal, Rpp32u n, Rpp32u c) {
        std::vector<Rpp32f> s;
        for (Rpp32u t = 0; t < 256; ++t) s.push_back(signal(n, t, c));
        return s;
    };
    for (Rpp32u a = 0; a < 9; ++a)
        for (Rpp32u b = a + 1; b < 9; ++b) {
            EXPECT_NE(stream(Tones{}, a / 3, a % 3), stream(Tones{}, b / 3, b % 3));
            EXPECT_NE(stream(Noise{}, a / 3, a % 3), stream(Noise{}, b / 3, b % 3));
        }
    EXPECT_NE(stream(Noise{1}, 0, 0), stream(Noise{2}, 0, 0));
}
