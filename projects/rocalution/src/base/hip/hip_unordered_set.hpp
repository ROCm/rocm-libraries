/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_UNORDERED_SET_HPP_
#define ROCALUTION_HIP_HIP_UNORDERED_SET_HPP_

#include "hip_atomics.hpp"

#include <hip/hip_runtime.h>
#include <limits>

namespace rocalution
{
    template <typename KeyType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL = 31232527,
              KeyType      EMPTY   = std::numeric_limits<KeyType>::max()>
    class unordered_set
    {
    public:
        // Constructor
        __device__ __forceinline__ explicit unordered_set(KeyType* skeys);
        // Destructor
        __device__ __forceinline__ ~unordered_set(void) {}

        // Erases all keys from the set
        __device__ __forceinline__ void clear(void);
        // Returns the empty key
        __device__ __forceinline__ constexpr KeyType empty_key(void) const
        {
            return EMPTY;
        }

        // Inserts a key into the set, if the key does not already exists, returns true
        __device__ __forceinline__ bool insert(const KeyType& key);
        // Checks, whether the set contains key
        __device__ __forceinline__ bool contains(const KeyType& key) const;
        // Returns the raw key at set position i, regardless its state
        __device__ __forceinline__ KeyType get_key(unsigned int i) const;

    private:
        // Array to hold keys
        KeyType* keys_;
    };

    template <typename KeyType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__
        unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::unordered_set(KeyType* skeys)
    {
        this->keys_ = skeys;

        // Initialize the set to be empty
        this->clear();
    }

    template <typename KeyType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ void
        unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::clear(void)
    {
        unsigned int tid = threadIdx.x & (NTHREADS - 1);

#pragma unroll 4
        for(unsigned int i = tid; i < SIZE; i += NTHREADS)
        {
            this->keys_[i] = EMPTY;
        }

        // Wait for all threads to finish clearing
        if(NTHREADS < warpSize)
        {
            __threadfence_block();
        }
        else
        {
            __syncthreads();
        }
    }

    template <typename KeyType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ bool
        unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::insert(const KeyType& key)
    {
        // Quick return
        if(key == EMPTY)
        {
            return false;
        }

        // Compute hash
        unsigned int hash     = (key * HASHVAL) & (SIZE - 1);
        unsigned int hash_inc = 1 + 2 * (key & (NTHREADS - 1));

        // Loop
        while(true)
        {
            KeyType temp = this->keys_[hash];

            if(temp == key)
            {
                // Key is already inserted, done
                return false;
            }
            else if(temp == EMPTY)
            {
                // If empty, add element with atomic
                if(atomicCAS(&this->keys_[hash], EMPTY, key) == EMPTY)
                {
                    // Element has been inserted
                    return true;
                }
            }
            else
            {
                // Collision, compute new hash
                hash = (hash + hash_inc) & (SIZE - 1);
            }
        }

        return false;
    }

    template <typename KeyType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ bool
        unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::contains(const KeyType& key) const
    {
        // Quick return
        if(key == EMPTY)
        {
            return false;
        }

        // Compute hash
        unsigned int hash     = (key * HASHVAL) & (SIZE - 1);
        unsigned int hash_inc = 1 + 2 * (key & (NTHREADS - 1));

        // Loop
        while(true)
        {
            if(this->keys_[hash] == EMPTY)
            {
                // Key not present, done
                return false;
            }
            else if(this->keys_[hash] == key)
            {
                // Key is present, done
                return true;
            }
            else
            {
                // Collision, compute new hash
                hash = (hash + hash_inc) & (SIZE - 1);
            }
        }

        return false;
    }

    template <typename KeyType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ KeyType
        unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::get_key(unsigned int i) const
    {
        return (i >= 0 && i < SIZE) ? this->keys_[i] : EMPTY;
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_UNORDERED_SET_HPP_
