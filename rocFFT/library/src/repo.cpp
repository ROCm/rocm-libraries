/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include <assert.h>
#include <iostream>
#include <numeric>
#include <vector>

#include "chirp.h"
#include "logging.h"
#include "node_factory.h"
#include "plan.h"
#include "repo.h"
#include "rocfft/rocfft.h"
#include "twiddles.h"
// Implementation of Class Repo

std::mutex        Repo::mtx;
std::atomic<bool> Repo::repoDestroyed(false);

template <typename KeyType>
std::pair<void*, size_t>
    Repo::GetTwiddlesInternal(KeyType                                             key,
                              std::map<KeyType, std::pair<gpubuf, unsigned int>>& twiddles,
                              std::map<void*, KeyType>&                           twiddles_reverse,
                              std::function<gpubuf(unsigned int)>                 create_twiddle)
{
    if(repoDestroyed)
    {
        throw std::runtime_error("Repo prematurely destroyed.");
    }

    // see if the repo has already stored the plan or not
    if(hipGetDevice(&key.deviceId) != hipSuccess)
    {
        throw std::runtime_error("hipGetDevice failed.");
    }

    auto it = twiddles.find(key);
    if(it != twiddles.end())
    {
        // already had this length
        it->second.second += 1;
        return {it->second.first.data(), it->second.first.size()};
    }

    // otherwise, need to allocate
    auto buf = create_twiddle(key.deviceId);
    // if allocation failed, don't update maps
    if(buf.data() == nullptr)
        return {nullptr, 0};
    it = twiddles.insert({key, std::make_pair(std::move(buf), 1)}).first;
    twiddles_reverse.insert({it->second.first.data(), key});
    return {it->second.first.data(), it->second.first.size()};
}

template <typename KeyType>
std::pair<void*, size_t>
    Repo::GetChirpInternal(KeyType                                             key,
                           std::map<KeyType, std::pair<gpubuf, unsigned int>>& chirp,
                           std::map<void*, KeyType>&                           chirp_reverse,
                           std::function<gpubuf(unsigned int)>                 create_chirp)
{
    if(repoDestroyed)
    {
        throw std::runtime_error("Repo prematurely destroyed.");
    }

    // see if the repo has already stored the plan or not
    if(hipGetDevice(&key.deviceId) != hipSuccess)
    {
        throw std::runtime_error("hipGetDevice failed.");
    }

    auto it = chirp.find(key);
    if(it != chirp.end())
    {
        // already had this length
        it->second.second += 1;
        return {it->second.first.data(), it->second.first.size()};
    }

    // otherwise, need to allocate
    auto buf = create_chirp(key.deviceId);
    // if allocation failed, don't update maps
    if(buf.data() == nullptr)
        return {nullptr, 0};
    it = chirp.insert({key, std::make_pair(std::move(buf), 1)}).first;
    chirp_reverse.insert({it->second.first.data(), key});
    return {it->second.first.data(), it->second.first.size()};
}

template <typename KeyType>
void Repo::ReleaseTwiddlesInternal(void*                                               ptr,
                                   std::map<KeyType, std::pair<gpubuf, unsigned int>>& twiddles,
                                   std::map<void*, KeyType>& twiddles_reverse)
{
    if(repoDestroyed)
    {
        return;
    }

    auto reverse_it = twiddles_reverse.find(ptr);
    if(reverse_it == twiddles_reverse.end())
        return;
    auto forward_it = twiddles.find(reverse_it->second);
    if(forward_it == twiddles.end())
    {
        // orphaned reverse entry?
        twiddles_reverse.erase(reverse_it);
        return;
    }
    forward_it->second.second -= 1;
    if(forward_it->second.second == 0)
    {
        // remove from both maps
        twiddles.erase(forward_it);
        twiddles_reverse.erase(reverse_it);
    }
}

template <typename KeyType>
void Repo::ReleaseChirpInternal(void*                                               ptr,
                                std::map<KeyType, std::pair<gpubuf, unsigned int>>& chirp,
                                std::map<void*, KeyType>&                           chirp_reverse)
{
    if(repoDestroyed)
    {
        return;
    }

    auto reverse_it = chirp_reverse.find(ptr);
    if(reverse_it == chirp_reverse.end())
        return;
    auto forward_it = chirp.find(reverse_it->second);
    if(forward_it == chirp.end())
    {
        // orphaned reverse entry?
        chirp_reverse.erase(reverse_it);
        return;
    }
    forward_it->second.second -= 1;
    if(forward_it->second.second == 0)
    {
        // remove from both maps
        chirp.erase(forward_it);
        chirp_reverse.erase(reverse_it);
    }
}

std::pair<void*, size_t> Repo::GetTwiddles1D(size_t                     length,
                                             size_t                     length_limit,
                                             rocfft_precision           precision,
                                             const hipDeviceProp_t&     deviceProp,
                                             size_t                     largeTwdBase,
                                             bool                       attach_halfN,
                                             const std::vector<size_t>& radices)
{
    std::lock_guard<std::mutex> lck(mtx);
    Repo&                       repo = Repo::GetRepo();

    repo_twd_key_1D_t key{length, length_limit, precision, largeTwdBase, attach_halfN, radices};
    return GetTwiddlesInternal(
        key, repo.twiddles_1D, repo.twiddles_1D_reverse, [&](unsigned int deviceId) {
            return twiddles_create(length,
                                   length_limit,
                                   precision,
                                   deviceProp,
                                   largeTwdBase,
                                   attach_halfN,
                                   radices,
                                   deviceId);
        });
}

std::pair<void*, size_t> Repo::GetTwiddles2D(size_t                     length0,
                                             size_t                     length1,
                                             rocfft_precision           precision,
                                             const hipDeviceProp_t&     deviceProp,
                                             bool                       attach_halfN1,
                                             bool                       attach_halfN2,
                                             const std::vector<size_t>& radices1,
                                             const std::vector<size_t>& radices2)
{
    std::lock_guard<std::mutex> lck(mtx);
    Repo&                       repo = Repo::GetRepo();

    repo_twd_key_2D_t key{length0, length1, precision, radices1, radices2};
    return GetTwiddlesInternal(
        key, repo.twiddles_2D, repo.twiddles_2D_reverse, [&](unsigned int deviceId) {
            return twiddles_create_2D(length0,
                                      length1,
                                      precision,
                                      deviceProp,
                                      attach_halfN1,
                                      attach_halfN2,
                                      radices1,
                                      radices2,
                                      deviceId);
        });
}

std::pair<void*, size_t> Repo::GetTwiddlesPP(size_t                 length,
                                             rocfft_precision       precision,
                                             const hipDeviceProp_t& deviceProp)
{
    std::lock_guard<std::mutex> lck(mtx);
    Repo&                       repo = Repo::GetRepo();

    repo_pp_twd_key_t key{length, precision};
    return GetTwiddlesInternal(
        key, repo.twiddles_PP, repo.twiddles_PP_reverse, [&](unsigned int deviceId) {
            return twiddles_pp_create(length, precision, deviceProp, deviceId);
        });
}

std::pair<void*, size_t>
    Repo::GetChirp(size_t length, rocfft_precision precision, const hipDeviceProp_t& deviceProp)
{
    std::lock_guard<std::mutex> lck(mtx);
    Repo&                       repo = Repo::GetRepo();

    repo_chirp_key_t key{length, precision};
    return GetChirpInternal(key, repo.chirp, repo.chirp_reverse, [&](unsigned int deviceId) {
        return chirp_create(length, precision, deviceId, deviceProp);
    });
}

void Repo::ReleaseTwiddle1D(void* ptr)
{
    std::lock_guard<std::mutex> lck(mtx);

    Repo& repo = Repo::GetRepo();
    return ReleaseTwiddlesInternal(ptr, repo.twiddles_1D, repo.twiddles_1D_reverse);
}

void Repo::ReleaseTwiddle2D(void* ptr)
{
    std::lock_guard<std::mutex> lck(mtx);

    Repo& repo = Repo::GetRepo();
    return ReleaseTwiddlesInternal(ptr, repo.twiddles_2D, repo.twiddles_2D_reverse);
}

void Repo::ReleaseTwiddlePP(void* ptr)
{
    std::lock_guard<std::mutex> lck(mtx);

    Repo& repo = Repo::GetRepo();
    return ReleaseTwiddlesInternal(ptr, repo.twiddles_PP, repo.twiddles_PP_reverse);
}

void Repo::ReleaseChirp(void* ptr)
{
    std::lock_guard<std::mutex> lck(mtx);

    Repo& repo = Repo::GetRepo();
    return ReleaseChirpInternal(ptr, repo.chirp, repo.chirp_reverse);
}

void Repo::Clear()
{
    std::lock_guard<std::mutex> lck(mtx);
    if(repoDestroyed)
        return;
    Repo& repo = Repo::GetRepo();

    repo.twiddles_1D.clear();
    repo.twiddles_2D.clear();
    repo.twiddles_PP.clear();
    twiddle_streams_cleanup();

    repo.chirp.clear();
    chirp_streams_cleanup();
}
