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

#ifndef ROCALUTION_HIP_HIP_UNORDERED_MAP_HPP_
#define ROCALUTION_HIP_HIP_UNORDERED_MAP_HPP_

#include "hip_atomics.hpp"

#include <hip/hip_runtime.h>
#include <limits>
#include <utility>

namespace rocalution
{
    template <typename KeyType,
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL = 31232527,
              KeyType      EMPTY   = std::numeric_limits<KeyType>::max()>
    class unordered_map
    {
    public:
        // Constructor
        __device__ __forceinline__ explicit unordered_map(KeyType* skeys, ValType* svals);
        // Destructor
        __device__ __forceinline__ ~unordered_map(void) {}

        // Erases all elements from the map
        __device__ __forceinline__ void clear(void);
        // Returns the empty key
        __device__ __forceinline__ constexpr KeyType empty_key(void) const
        {
            return EMPTY;
        }

        // Inserts an element into the map, if the map does not already contain an element
        // with an equivalent key
        __device__ __forceinline__ bool insert(const KeyType& key,
                                               const ValType& val = static_cast<ValType>(0));
        // Adds an element into the map, if the map already contains an element with an equivalent key
        // Inserts the element otherwise
        // Returns true, if a new element has been inserted
        // Returns false, if the element has been added
        __device__ __forceinline__ bool insert_or_add(const KeyType& key, const ValType& val);
        // Adds an element into the map, if the map already contains an element with an equivalent key
        __device__ __forceinline__ bool add(const KeyType& key, const ValType& val);
        // Checks, whether there is an element key in the map
        __device__ __forceinline__ bool contains(const KeyType& key) const;
        // Returns the raw key at map position i, regardless its state
        __device__ __forceinline__ KeyType get_key(unsigned int i) const;
        // Returns the raw val at map position i, regardless its state
        __device__ __forceinline__ ValType get_val(unsigned int i) const;
        // Returns the raw key val pair at map position i, regardless its state
        __device__ __forceinline__ std::pair<KeyType, ValType> get_pair(unsigned int i) const;

        // Store the key val pairs sorted
        __device__ __forceinline__ void store_sorted(KeyType* keys, ValType* vals) const;
        // Store the key val pairs sorted with a given permutation and alpha, such that
        // sorted_keys = perm[keys] and sorted_vals = alpha * vals
        __device__ __forceinline__ void store_sorted_with_perm(const int* perm,
                                                               ValType    alpha,
                                                               KeyType*   keys,
                                                               ValType*   vals) const;

        // Sorts the map, this will destroy the map
        __device__ __forceinline__ void sort(void);

    private:
        // Array to hold keys
        KeyType* keys_;
        // Array to hold vals
        ValType* vals_;
    };

    template <typename KeyType,
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::unordered_map(
            KeyType* skeys, ValType* svals)
    {
        this->keys_ = skeys;
        this->vals_ = svals;

        // Initialize the map to be empty
        this->clear();
    }

    template <typename KeyType,
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ void
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::clear(void)
    {
        unsigned int tid = threadIdx.x & (NTHREADS - 1);

#pragma unroll 4
        for(unsigned int i = tid; i < SIZE; i += NTHREADS)
        {
            this->keys_[i] = EMPTY;
            this->vals_[i] = static_cast<ValType>(0);
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
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ bool
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::insert(const KeyType& key,
                                                                                const ValType& val)
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
                    // Set value exclusively
                    this->vals_[hash] = val;

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
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ bool
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::insert_or_add(
            const KeyType& key, const ValType& val)
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
                // Map contains the key
                atomicAdd(&this->vals_[hash], val);

                // val has been added to the map
                return false;
            }
            else if(temp == EMPTY)
            {
                // If empty, add element with atomic
                if(atomicCAS(&this->keys_[hash], EMPTY, key) == EMPTY)
                {
                    // Add value
                    atomicAdd(&this->vals_[hash], val);

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
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ bool
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::add(const KeyType& key,
                                                                             const ValType& val)
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
            if(this->keys_[hash] == key)
            {
                // Map contains the key
                atomicAdd(&this->vals_[hash], val);

                // val has been added to the map
                return true;
            }
            else if(this->keys_[hash] == EMPTY)
            {
                // Key is not contained in map
                return false;
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
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ bool
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::contains(
            const KeyType& key) const
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
            if(this->keys_[hash] == key)
            {
                // Map contains key
                return true;
            }
            else if(this->keys_[hash] == EMPTY)
            {
                // Map does not contain key
                return false;
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
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ KeyType
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::get_key(
            unsigned int i) const
    {
        return (i >= 0 && i < SIZE) ? this->keys_[i] : EMPTY;
    }

    template <typename KeyType,
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ ValType
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::get_val(
            unsigned int i) const
    {
        return (i >= 0 && i < SIZE) ? this->vals_[i] : static_cast<ValType>(0);
    }

    template <typename KeyType,
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ std::pair<KeyType, ValType>
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::get_pair(
            unsigned int i) const
    {
        return ((i >= 0 && i < SIZE) ? std::make_pair(this->keys_[i], this->vals_[i])
                                     : std::make_pair(EMPTY, static_cast<ValType>(0)));
    }

    template <typename KeyType,
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ void
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::store_sorted(
            KeyType* keys, ValType* vals) const
    {
        unsigned int tid = threadIdx.x & (NTHREADS - 1);

#pragma unroll 4
        for(unsigned int i = tid; i < SIZE; i += NTHREADS)
        {
            // Get next key entry
            KeyType key = this->keys_[i];

            // Check, whether the key is valid or not
            if(key == EMPTY)
            {
                continue;
            }

            // Index into output arrays
            int idx = 0;

            // Index counter
            unsigned int cnt = 0;

            // Go through the map, until we reach its end
            while(cnt < SIZE)
            {
                // We are going through the map to determine the insertion slot
                // for this key val pair in order to obtain a sorted output
                if(key > this->keys_[cnt])
                {
                    ++idx;
                }

                // Go to next key
                ++cnt;
            }

            keys[idx] = key;
            vals[idx] = this->vals_[i];
        }
    }

    template <typename KeyType,
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ void
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::store_sorted_with_perm(
            const int* perm, ValType alpha, KeyType* keys, ValType* vals) const
    {
        unsigned int tid = threadIdx.x & (NTHREADS - 1);

#pragma unroll 4
        for(unsigned int i = tid; i < SIZE; i += NTHREADS)
        {
            // Get next key entry
            KeyType key = this->keys_[i];

            // Check, whether the key is valid or not
            if(key == EMPTY)
            {
                continue;
            }

            // Index into output arrays
            int idx = 0;

            // Index counter
            unsigned int cnt = 0;

            // Go through the map, until we reach its end
            while(cnt < SIZE)
            {
                // We are going through the map to determine the insertion slot
                // for this key val pair in order to obtain a sorted output
                if(key > this->keys_[cnt])
                {
                    ++idx;
                }

                // Go to next key
                ++cnt;
            }

            keys[idx] = perm[key];
            vals[idx] = alpha * this->vals_[i];
        }
    }

    template <typename KeyType,
              typename ValType,
              unsigned int SIZE,
              unsigned int NTHREADS,
              unsigned int HASHVAL,
              KeyType      EMPTY>
    __device__ __forceinline__ void
        unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::sort(void)
    {
        // Sync writes before sorting
        if(NTHREADS < warpSize)
        {
            __threadfence_block();
        }
        else
        {
            __syncthreads();
        }

        // Thread id
        unsigned int tid = threadIdx.x & (NTHREADS - 1);

        KeyType      keys[SIZE / NTHREADS];
        ValType      vals[SIZE / NTHREADS];
        unsigned int idx[SIZE / NTHREADS];

        // Each thread grabs its key val pairs
        for(unsigned int i = 0; i < SIZE / NTHREADS; ++i)
        {
            keys[i] = this->keys_[tid + NTHREADS * i];
            vals[i] = this->vals_[tid + NTHREADS * i];
        }

#pragma unroll 4
        for(unsigned int i = 0; i < SIZE / NTHREADS; ++i)
        {
            // Check, whether the key is valid or not
            if(keys[i] == EMPTY)
            {
                continue;
            }

            // Index into output arrays
            idx[i] = 0;

            // Index counter
            unsigned int cnt = 0;

            // Go through the map, until we reach its end
            while(cnt < SIZE)
            {
                // We are going through the map to determine the insertion slot
                // for this key val pair in order to obtain a sorted output
                if(keys[i] > this->keys_[cnt])
                {
                    ++idx[i];
                }

                // Go to next key
                ++cnt;
            }
        }

        // Clear map
        for(unsigned int i = tid; i < SIZE; i += NTHREADS)
        {
            this->keys_[i] = EMPTY;
        }

        // Write back key val pairs
        for(unsigned int i = 0; i < SIZE / NTHREADS; ++i)
        {
            if(keys[i] == EMPTY)
            {
                continue;
            }

            this->keys_[idx[i]] = keys[i];
            this->vals_[idx[i]] = vals[i];
        }

        // Sync writes after sorting
        if(NTHREADS < warpSize)
        {
            __threadfence_block();
        }
        else
        {
            __syncthreads();
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_UNORDERED_MAP_HPP_
