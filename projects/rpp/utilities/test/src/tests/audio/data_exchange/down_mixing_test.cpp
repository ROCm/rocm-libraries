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
#include "reference/down_mixing_ref.hpp"

using namespace rpptest;

namespace {

struct DownMixingParams {
    bool normalizeWeights;
    std::string name() const {
        return normalizeWeights ? "NormWeights" : "RawWeights";
    }
};

void run_down_mixing(const AudioConfig& cfg, const DownMixingParams& p) {
    const std::vector<Rpp32s> lengths = audio_lengths(cfg.shape);
    RpptDesc srcDesc = make_audio_descriptor(cfg);
    RpptDesc dstDesc = make_audio_descriptor(srcDesc.n, srcDesc.w, 1);
    const std::size_t srcBytes = audio_byte_size(srcDesc);
    const std::size_t dstBytes = audio_byte_size(dstDesc);

    const std::vector<Rpp32f> input = make_audio_input(srcDesc, lengths, Noise{});
    std::vector<Rpp32f> golden = make_audio_output(dstDesc);
    std::vector<Rpp32f> actual = golden;
    down_mixing_reference(input.data(), srcDesc, golden.data(), dstDesc, lengths);

    const std::vector<Rpp32s> dims = audio_dims_tensor(lengths, cfg.channels);
    PinnedArray<Rpp32s> srcDims(cfg.backend, dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) srcDims[i] = dims[i];

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);
    dst.write(actual.data(), dstBytes);

    RppHandle handle(cfg.backend, srcDesc.n);
    ASSERT_EQ(rppt_down_mixing(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, srcDims.data(),
                               p.normalizeWeights, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), dstBytes);

    // A float sum of weighted values in [-1, 1], in a different order on each backend.
    EXPECT_TRUE(compare_audio(actual.data(), golden.data(), dstDesc, lengths, Bound{1e-6}));
    // Both kernels write one value per frame, so the padding stays untouched.
    EXPECT_TRUE(compare_audio_padding(actual.data(), dstDesc, lengths));
}

}  // namespace

// Full name:
// Audio_DataExchange/DownMixingTest.Correctness/<Backend>_F32toF32_<C>ch_<Shape>_<NxMaxLen>_<Weights>
class DownMixingTest : public SkipListTest<AudioWithParams<DownMixingParams>> {};

TEST_P(DownMixingTest, Correctness) {
    const auto& p = GetParam();
    run_down_mixing(p.cfg, p.op);
}

// The channel counts cover each kernel path: the mono copy, HOST's scalar, SSE and AVX sums with a
// scalar tail, and HIP's vector sums with a tail.
INSTANTIATE_TEST_SUITE_P(Audio_DataExchange, DownMixingTest,
                         ::testing::ValuesIn(audio_with_params<DownMixingParams>(
                             make_audio_configs({1, 2, 5, 9}), {{false}, {true}})),
                         audio_op_config_name<DownMixingParams>);
