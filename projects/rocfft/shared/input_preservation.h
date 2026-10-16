// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef INPUT_PRESERVATION_H
#define INPUT_PRESERVATION_H

#include "fft_enums.h"
#include "hostbuf.h"
#include "params_gen.h"
#include "rocfft_hip.h"
#include "stable_hash.h"
#include "test_params.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// 64-bit digest of a raw byte range, chaining the same splitmix64 mix that
// fft_hash.h uses per element.  The element-wise hash there is deliberately
// layout-aware and skips padding, which this must not do - see the note on
// input_preservation_check below.
static inline uint64_t raw_byte_digest(const void* data, const size_t size_bytes)
{
    const auto* bytes  = static_cast<const unsigned char*>(data);
    uint64_t    digest = size_bytes;

    const size_t whole_words = size_bytes / sizeof(uint64_t);
    for(size_t i = 0; i < whole_words; ++i)
    {
        uint64_t word;
        memcpy(&word, bytes + i * sizeof(uint64_t), sizeof(word));
        digest = stable_hash(digest ^ word);
    }

    if(const size_t tail_bytes = size_bytes % sizeof(uint64_t))
    {
        uint64_t tail = 0;
        memcpy(&tail, bytes + whole_words * sizeof(uint64_t), tail_bytes);
        digest = stable_hash(digest ^ tail);
    }

    return digest;
}

// Digests a transform's input buffer(s) before it runs, so that the same
// digest can be taken afterwards to confirm the transform left the input
// alone.
//
// Only out-of-place C2C and R2C transforms are checked.  Those never get the
// input allocation offered to them as a place to write, so no byte of it may
// change - which is why the whole allocation is digested rather than only the
// logically addressed elements.  In-place transforms write their output over
// the input by definition, and out-of-place C2R transforms are documented to
// be allowed to overwrite their input; both are skipped.
//
// Only a sampled fraction of eligible cases is checked, governed by
// input_preservation_prob; see the constructor.
//
// The check also skips itself, rather than failing, when it cannot run: on
// bricked or multi-device layouts (where the input pointers refer to per-brick
// staging buffers that the input layout no longer describes), on a params
// object whose isize has not been computed yet, and when there is not enough
// host memory to stage a copy.
template <class Tparams>
class input_preservation_check
{
public:
    // An unarmed check, for callers that have to declare one before they are in
    // a position to populate it.  Reports nothing.
    input_preservation_check() = default;

    input_preservation_check(const Tparams& params, const std::vector<void*>& pibuffer)
    {
        if(!input_must_be_preserved(params))
            return;

        // Confirming the input survived costs two full copies of it back to
        // the host, which is too much to pay on every case, so only a sampled
        // fraction is checked.  The token is salted so this roll is
        // independent of the one that decided to run the test at all - that
        // one hashes the bare token, and reusing it would confine the check to
        // whichever cases happened to roll low there.
        const double roll = hash_prob(random_seed, "input_preservation " + params.token());
        if(roll > input_preservation_prob)
        {
            if(verbose > 4)
            {
                std::cout << "Input preservation check skipped: (roll=" << roll << " > "
                          << input_preservation_prob << ")\n";
            }
            return;
        }

        sizes   = params.ibuffer_sizes();
        buffers = pibuffer;
        if(sizes.empty() || sizes.size() != buffers.size())
            return;

        try
        {
            staging.resize(sizes.size());
            for(unsigned int i = 0; i < sizes.size(); ++i)
                staging[i].alloc(sizes[i]);
        }
        catch(const HOSTBUF_MEM_USAGE&)
        {
            staging.clear();
            return;
        }
        catch(const std::bad_alloc&)
        {
            staging.clear();
            return;
        }

        before = digest_device_buffers();
    }

    // Captures the post-transform image of the input.  This has to happen
    // while the input allocation still holds what the transform left there,
    // which is why it is separate from reporting: the round-trip inverse
    // writes its own output into the forward transform's input buffer.
    void snapshot_after()
    {
        if(staging.empty())
            return;

        after = digest_device_buffers();
        // The staging copies are no longer needed, and releasing them keeps the
        // harness' host memory accounting honest for whatever runs next.
        staging.clear();
    }

    // Describes how the input buffer changed, or returns an empty string when
    // the input is intact or the check did not apply.
    std::string verify() const
    {
        if(after.empty())
            return {};

        std::stringstream msg;
        for(unsigned int i = 0; i < after.size(); ++i)
        {
            if(after[i] == before[i])
                continue;
            msg << "input buffer " << i << " (" << sizes[i]
                << " bytes) was modified by the transform: digest " << before[i] << " -> "
                << after[i] << "\n";
        }
        return msg.str();
    }

private:
    static bool input_must_be_preserved(const Tparams& params)
    {
        if(params.placement != fft_placement_notinplace)
            return false;

        if(!params.ifields.empty() || !params.ofields.empty() || params.multiGPU > 1)
            return false;

        return params.transform_type == fft_transform_type_complex_forward
               || params.transform_type == fft_transform_type_complex_inverse
               || params.transform_type == fft_transform_type_real_forward
               || params.transform_type == fft_transform_type_real_inverse; // TEMP: probe C2R
    }

    std::vector<uint64_t> digest_device_buffers()
    {
        auto hip_status = hipDeviceSynchronize();
        if(hip_status != hipSuccess)
            throw hip_runtime_error("hipDeviceSynchronize failure", hip_status);

        std::vector<uint64_t> digests(sizes.size());
        for(unsigned int i = 0; i < sizes.size(); ++i)
        {
            hip_status = hipMemcpy(staging[i].data(), buffers[i], sizes[i], hipMemcpyDeviceToHost);
            if(hip_status != hipSuccess)
                throw hip_runtime_error("hipMemcpy failure", hip_status);
            digests[i] = raw_byte_digest(staging[i].data(), sizes[i]);
        }
        return digests;
    }

    std::vector<size_t>   sizes;
    std::vector<void*>    buffers;
    std::vector<hostbuf>  staging;
    std::vector<uint64_t> before;
    std::vector<uint64_t> after;
};

#endif // INPUT_PRESERVATION_H
