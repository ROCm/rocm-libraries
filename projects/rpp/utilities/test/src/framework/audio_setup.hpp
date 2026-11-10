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

#ifndef RPP_TEST_AUDIO_SETUP_H
#define RPP_TEST_AUDIO_SETUP_H

#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/audio_config_param.hpp"
#include "framework/tensor_setup.hpp"

// Audio tensors use the 4D RpptDesc: n samples of up to w frames with c channels interleaved
// (wStride = c, cStride = 1) and h = 1. numDims is 2 for mono and 3 for multi-channel, as the API
// documents. Each sample's real length travels beside the descriptor, in whichever form the op
// takes (see the length arrays below).

namespace rpptest {

// Fills everything outside the samples' lengths. Far outside the generators' [-1, 1], so a kernel
// that reads padding gives visibly wrong output, and one that writes it is caught.
inline constexpr Rpp32f kAudioSentinel = -1024.0f;

// ---- descriptors ----------------------------------------------------------

// d.w is the longest sample; the frame stride is padded like an image row, so a kernel storing a
// full vector over the last sample's tail stays inside the allocation.
inline RpptDesc make_audio_descriptor(Rpp32u n, Rpp32u maxLength, Rpp32u channels) {
    RpptDesc d{};
    d.numDims = channels == 1 ? 2 : 3;
    d.offsetInBytes = 0;
    d.dataType = F32;
    d.layout = NHWC;  // interleaved channels; the 1D ops do not read it
    d.n = n;
    d.c = channels;
    d.h = 1;
    d.w = maxLength;
    d.strides.wStride = channels;
    d.strides.cStride = 1;
    d.strides.hStride = channels * padded_width(maxLength);
    d.strides.nStride = d.strides.hStride;
    return d;
}

inline RpptDesc make_audio_descriptor(const AudioConfig& c) {
    const std::vector<Rpp32s> lengths = audio_lengths(c.shape);
    return make_audio_descriptor(static_cast<Rpp32u>(lengths.size()),
                                 static_cast<Rpp32u>(max_audio_length(lengths)), c.channels);
}

inline std::size_t audio_element_count(const RpptDesc& d) {
    return static_cast<std::size_t>(d.n) * d.strides.nStride;
}

inline std::size_t audio_byte_size(const RpptDesc& d) {
    return audio_element_count(d) * sizeof(Rpp32f);
}

// ---- length arrays ----------------------------------------------------------
//
// The ops take a batch's lengths as srcLengthTensor (one Rpp32s per sample, the lengths vector
// itself), srcDimsTensor or RpptImagePatch. Copy the result into a PinnedArray for the call.

// srcDimsTensor: (length, channels) per sample.
inline std::vector<Rpp32s> audio_dims_tensor(const std::vector<Rpp32s>& lengths, Rpp32u channels) {
    std::vector<Rpp32s> dims;
    dims.reserve(2 * lengths.size());
    for (Rpp32s length : lengths) {
        dims.push_back(length);
        dims.push_back(static_cast<Rpp32s>(channels));
    }
    return dims;
}

// RpptImagePatch form: the signal as a column, height = length and width = 1.
inline std::vector<RpptImagePatch> audio_patch_tensor(const std::vector<Rpp32s>& lengths) {
    std::vector<RpptImagePatch> patches;
    patches.reserve(lengths.size());
    for (Rpp32s length : lengths) patches.push_back({1, static_cast<Rpp32u>(length)});
    return patches;
}

// ---- traversal --------------------------------------------------------------

// Element (t, c) of sample n.
inline std::size_t audio_index(const RpptDesc& d, Rpp32u n, Rpp32u t, Rpp32u c) {
    return static_cast<std::size_t>(n) * d.strides.nStride +
           static_cast<std::size_t>(t) * d.strides.wStride + c;
}

// fn(n, t, c, idx) for every frame and channel within each sample's length.
template <typename Fn>
void for_each_audio_element(const RpptDesc& d, const std::vector<Rpp32s>& lengths, Fn fn) {
    for (Rpp32u n = 0; n < d.n; ++n)
        for (Rpp32u t = 0; t < static_cast<Rpp32u>(lengths[n]); ++t)
            for (Rpp32u c = 0; c < d.c; ++c) fn(n, t, c, audio_index(d, n, t, c));
}

// fn(n, idx) for every element of the allocation past its sample's length: the complement of
// for_each_audio_element.
template <typename Fn>
void for_each_audio_padding(const RpptDesc& d, const std::vector<Rpp32s>& lengths, Fn fn) {
    for (Rpp32u n = 0; n < d.n; ++n) {
        const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride;
        const std::size_t used = static_cast<std::size_t>(lengths[n]) * d.c;
        for (std::size_t i = used; i < d.strides.nStride; ++i) fn(n, base + i);
    }
}

// ---- buffers ----------------------------------------------------------------

// The sentinel everywhere, then signal(n, t, c) within each sample's length.
template <typename Signal>
std::vector<Rpp32f> make_audio_input(const RpptDesc& d, const std::vector<Rpp32s>& lengths,
                                     Signal signal) {
    std::vector<Rpp32f> buf(audio_element_count(d), kAudioSentinel);
    for_each_audio_element(d, lengths, [&](Rpp32u n, Rpp32u t, Rpp32u c, std::size_t idx) {
        buf[idx] = signal(n, t, c);
    });
    return buf;
}

// Output buffers start as the sentinel, so an element the op never writes cannot pass as zero.
inline std::vector<Rpp32f> make_audio_output(const RpptDesc& d) {
    return std::vector<Rpp32f>(audio_element_count(d), kAudioSentinel);
}

}  // namespace rpptest

#endif  // RPP_TEST_AUDIO_SETUP_H
