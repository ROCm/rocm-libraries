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

#ifndef RPP_TEST_AUDIO_ARITHMETIC_REF_H
#define RPP_TEST_AUDIO_ARITHMETIC_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/audio_setup.hpp"

namespace rpptest {

/*
Reference model: audio arithmetic

RPP op
  rppt_audio_tensor_mul_scalar   (Audio / Arithmetic)

Description
  Multiplies every value of each sample by one scalar shared by the whole
  batch. F32 only. Computed in double and stored once.

Expression
  dst = src * scalar

Notes
  - srcLengthTensor counts floats, not frames: both kernels walk that many
    contiguous values from the start of the sample. The tests pass
    frames * channels, which covers every channel of an interleaved signal.
  - Only each sample's length is written here; the destination past it is
    not modelled.
*/

inline void audio_mul_scalar_reference(const Rpp32f* src, Rpp32f* dst, const RpptDesc& desc,
                                       const std::vector<Rpp32s>& lengths, Rpp32f scalar) {
    for_each_audio_element(desc, lengths, [&](Rpp32u, Rpp32u, Rpp32u, std::size_t idx) {
        dst[idx] = static_cast<Rpp32f>(static_cast<double>(src[idx]) * scalar);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_AUDIO_ARITHMETIC_REF_H
