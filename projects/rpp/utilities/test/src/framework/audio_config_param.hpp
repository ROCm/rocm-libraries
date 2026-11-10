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

#ifndef RPP_TEST_AUDIO_CONFIG_PARAM_H
#define RPP_TEST_AUDIO_CONFIG_PARAM_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <algorithm>
#include <ostream>
#include <string>
#include <vector>

#include "framework/backend_param.hpp"

// Grid axes for the Audio ops: a batch of interleaved F32 signals, each with its own length.
// Labels read "<Backend>_F32toF32_<Channels>ch_<Shape>_<N>x<MaxLength>". Audio ops are F32 only;
// the dtype token is kept so every domain's label starts the same way.

namespace rpptest {

// Per-sample lengths in frames. Uniform fills the whole buffer at a vector-aligned length; Ragged
// gives every sample a different length with a tail, so a kernel that applies one sample's length
// to the batch, or mishandles the remainder, is caught.
enum class AudioShape { Uniform, Ragged };

inline std::string audio_shape_name(AudioShape s) {
    return s == AudioShape::Uniform ? "Uniform" : "Ragged";
}

// 4099 frames spans more than two 2048-float HIP blocks plus a tail.
inline std::vector<Rpp32s> audio_lengths(AudioShape s) {
    if (s == AudioShape::Uniform) return {4096, 4096, 4096};
    return {4099, 2051, 1037};
}

inline Rpp32s max_audio_length(const std::vector<Rpp32s>& lengths) {
    return *std::max_element(lengths.begin(), lengths.end());
}

// Every sample in a batch shares the channel count.
struct AudioConfig {
    RppBackend backend;
    Rpp32u channels;
    AudioShape shape;
};

inline std::string audio_config_name(const AudioConfig& c) {
    const std::vector<Rpp32s> lengths = audio_lengths(c.shape);
    return backend_name(c.backend) + "_F32toF32_" + std::to_string(c.channels) + "ch_" +
           audio_shape_name(c.shape) + "_" + std::to_string(lengths.size()) + "x" +
           std::to_string(max_audio_length(lengths));
}

namespace presets {
inline const std::vector<AudioShape> kAudioShapes{AudioShape::Uniform, AudioShape::Ragged};
}  // namespace presets

inline std::vector<AudioConfig> make_audio_configs(
    const std::vector<Rpp32u>& channels,
    const std::vector<AudioShape>& shapes = presets::kAudioShapes) {
    std::vector<AudioConfig> configs;
    for (RppBackend backend : available_backends())
        for (Rpp32u c : channels)
            for (AudioShape shape : shapes) configs.push_back({backend, c, shape});
    return configs;
}

inline std::string audio_config_param_name(const ::testing::TestParamInfo<AudioConfig>& info) {
    return audio_config_name(info.param);
}

// The audio counterpart of WithParams. P must provide std::string name() const.
template <typename P>
struct AudioWithParams {
    AudioConfig cfg;
    P op;
};

template <typename P>
inline std::vector<AudioWithParams<P>> audio_with_params(const std::vector<AudioConfig>& base,
                                                         const std::vector<P>& params) {
    std::vector<AudioWithParams<P>> out;
    out.reserve(base.size() * params.size());
    for (const auto& c : base)
        for (const auto& p : params) out.push_back({c, p});
    return out;
}

template <typename P>
inline std::string audio_op_config_name(const ::testing::TestParamInfo<AudioWithParams<P>>& info) {
    const std::string suffix = info.param.op.name();
    return audio_config_name(info.param.cfg) + (suffix.empty() ? "" : "_" + suffix);
}

inline void PrintTo(const AudioConfig&, std::ostream*) {}

template <typename P>
void PrintTo(const AudioWithParams<P>&, std::ostream*) {}

}  // namespace rpptest

#endif  // RPP_TEST_AUDIO_CONFIG_PARAM_H
