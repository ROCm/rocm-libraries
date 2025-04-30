
// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef REPO_H
#define REPO_H

#include "../../../shared/gpubuf.h"
#include <map>
#include <mutex>

class Repo
{
    Repo() {}

    // key structure for 1D twiddles - these are the arguments to
    // twiddle creation
    struct repo_twd_key_1D_t
    {
        // twiddle table length
        size_t           length       = 0;
        size_t           length_limit = 0;
        rocfft_precision precision    = rocfft_precision_single;
        // large twiddle base (0 for non-large twiddle)
        size_t              large_twiddle_base = 0;
        bool                attach_halfN       = false;
        std::vector<size_t> radices;
        // buffers are in device memory, so we need per-device
        // twiddles
        int deviceId = 0;

        bool operator<(const repo_twd_key_1D_t& other) const
        {
            if(length != other.length)
                return length < other.length;
            if(length_limit != other.length_limit)
                return length_limit < other.length_limit;
            if(precision != other.precision)
                return precision < other.precision;
            if(large_twiddle_base != other.large_twiddle_base)
                return large_twiddle_base < other.large_twiddle_base;
            if(attach_halfN != other.attach_halfN)
                return attach_halfN < other.attach_halfN;
            if(radices != other.radices)
                return radices < other.radices;
            return deviceId < other.deviceId;
        }
    };
    // key structure for 2D twiddles
    struct repo_twd_key_2D_t
    {
        size_t              length0   = 0;
        size_t              length1   = 0;
        rocfft_precision    precision = rocfft_precision_single;
        std::vector<size_t> radices1;
        std::vector<size_t> radices2;
        // buffers are in device memory, so we need per-device
        // twiddles
        int deviceId = 0;

        bool operator<(const repo_twd_key_2D_t& other) const
        {
            if(length0 != other.length0)
                return length0 < other.length0;
            if(length1 != other.length1)
                return length1 < other.length1;
            if(precision != other.precision)
                return precision < other.precision;
            if(radices1 != other.radices1)
                return radices1 < other.radices1;
            if(radices2 != other.radices2)
                return radices2 < other.radices2;
            return deviceId < other.deviceId;
        }
    };
    // key structure for partial-pass twiddles
    struct repo_pp_twd_key_t
    {
        size_t           length    = 0;
        rocfft_precision precision = rocfft_precision_single;
        // buffers are in device memory, so we need per-device
        // twiddles
        int deviceId = 0;

        bool operator<(const repo_pp_twd_key_t& other) const
        {
            if(length != other.length)
                return length < other.length;
            if(precision != other.precision)
                return precision < other.precision;
            return deviceId < other.deviceId;
        }
    };
    // key structure for chirp table
    struct repo_chirp_key_t
    {
        size_t           length    = 0;
        rocfft_precision precision = rocfft_precision_single;
        // buffers are in device memory, so we need per-device
        // chirps
        int deviceId = 0;

        bool operator<(const repo_chirp_key_t& other) const
        {
            if(length != other.length)
                return length < other.length;
            if(precision != other.precision)
                return precision < other.precision;
            return deviceId < other.deviceId;
        }
    };

    // twiddle tables are buffers in device memory, along with a
    // reference count
    //
    // NOTE: some buffers might be more shareable here (e.g. simple
    // 1D might match half of a 2D twiddle, or a simple 1D might be
    // shareable with a same-length attach_halfN buffer)
    std::map<repo_twd_key_1D_t, std::pair<gpubuf, unsigned int>> twiddles_1D;
    std::map<repo_twd_key_2D_t, std::pair<gpubuf, unsigned int>> twiddles_2D;
    std::map<repo_pp_twd_key_t, std::pair<gpubuf, unsigned int>> twiddles_PP;

    std::map<repo_chirp_key_t, std::pair<gpubuf, unsigned int>> chirp;

    // reverse-map the device pointers back to the keys so users can
    // free the pointer they were given
    std::map<void*, repo_twd_key_1D_t> twiddles_1D_reverse;
    std::map<void*, repo_twd_key_2D_t> twiddles_2D_reverse;
    std::map<void*, repo_pp_twd_key_t> twiddles_PP_reverse;

    std::map<void*, repo_chirp_key_t> chirp_reverse;

    static std::mutex mtx;

    // internal helpers to get and free twiddles
    template <typename KeyType>
    static std::pair<void*, size_t>
        GetTwiddlesInternal(KeyType,
                            std::map<KeyType, std::pair<gpubuf, unsigned int>>&,
                            std::map<void*, KeyType>&,
                            std::function<gpubuf(unsigned int)>);
    template <typename KeyType>
    static void ReleaseTwiddlesInternal(void* ptr,
                                        std::map<KeyType, std::pair<gpubuf, unsigned int>>&,
                                        std::map<void*, KeyType>&);

    // internal helpers to get and free chirp table
    template <typename KeyType>
    static std::pair<void*, size_t>
        GetChirpInternal(KeyType,
                         std::map<KeyType, std::pair<gpubuf, unsigned int>>&,
                         std::map<void*, KeyType>&,
                         std::function<gpubuf(unsigned int)>);
    template <typename KeyType>
    static void ReleaseChirpInternal(void* ptr,
                                     std::map<KeyType, std::pair<gpubuf, unsigned int>>&,
                                     std::map<void*, KeyType>&);

public:
    // repo is a singleton, so no copying or assignment
    Repo(const Repo&) = delete;
    Repo& operator=(const Repo&) = delete;

    static Repo& GetRepo()
    {
        static Repo repo;
        return repo;
    }

    ~Repo()
    {
        repoDestroyed = true;
    }

    static std::pair<void*, size_t> GetTwiddles1D(size_t                     length,
                                                  size_t                     length_limit,
                                                  rocfft_precision           precision,
                                                  const hipDeviceProp_t&     deviceProp,
                                                  size_t                     largeTwdBase,
                                                  bool                       attach_halfN,
                                                  const std::vector<size_t>& radices);
    static std::pair<void*, size_t> GetTwiddles2D(size_t                     length0,
                                                  size_t                     length1,
                                                  rocfft_precision           precision,
                                                  const hipDeviceProp_t&     deviceProp,
                                                  bool                       attach_halfN1,
                                                  bool                       attach_halfN2,
                                                  const std::vector<size_t>& radices1,
                                                  const std::vector<size_t>& radices2);
    static std::pair<void*, size_t>
        GetTwiddlesPP(size_t length, rocfft_precision precision, const hipDeviceProp_t& deviceProp);
    static std::pair<void*, size_t>
        GetChirp(size_t length, rocfft_precision precision, const hipDeviceProp_t& deviceProp);
    static void ReleaseTwiddle1D(void* ptr);
    static void ReleaseTwiddle2D(void* ptr);
    static void ReleaseTwiddlePP(void* ptr);
    static void ReleaseChirp(void* ptr);
    // remove cached twiddles/chirp
    static void Clear();

    // Repo is a singleton that should only be destroyed on static
    // deinitialization.  But it's possible for other things to want to
    // destroy plans at static deinitialization time.  So keep track of
    // whether the repo has been destroyed, so we can avoid wanting it
    // again.
    static std::atomic<bool> repoDestroyed;
};

#endif // REPO_H
