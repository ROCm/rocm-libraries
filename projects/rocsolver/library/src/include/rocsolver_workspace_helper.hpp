/* **************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common_host_helpers.hpp"
#include "init_scalars.hpp"
#include "lib_host_helpers.hpp"
#include "rocsolver/rocsolver.h"

#include <unordered_map>
#include <vector>

ROCSOLVER_BEGIN_NAMESPACE

#define MIN_CHUNK_SIZE 64

/* ROCSOLVER_WORKSPACE_HELPER provides a wrapper for workspace arrays, allowing workspace arrays to be passed
   between functions using a single object rather than individually, and improving device memory utilization. */
class rocsolver_workspace_helper
{
private:
    bool has_scalars_r, has_scalars_c, optim_mem;
    size_t num_excl, size_excl, size_shared;
    std::unordered_map<std::string, size_t> indices, n_indices;
    std::vector<uint8_t*> pointers;
    std::vector<size_t> sizes;
    std::vector<rocsolver_workspace_helper*> nested;

    void assign_buffer(uint8_t* scalars_r, uint8_t* scalars_c, uint8_t* curr_ptr)
    {
        pointers.reserve(sizes.size() + has_scalars_r + has_scalars_c);

        if(has_scalars_r)
            pointers.push_back(scalars_r);
        if(has_scalars_c)
            pointers.push_back(scalars_c);

        uint8_t* nested_ptr = curr_ptr;
        for(size_t i = 0; i < sizes.size(); i++)
        {
            pointers.push_back(curr_ptr);
            curr_ptr += sizes[i];
            if(i < num_excl)
                nested_ptr = curr_ptr;
        }

        for(size_t i = 0; i < nested.size(); i++)
            nested[i]->assign_buffer(scalars_r, scalars_c, nested_ptr);
    }

    size_t get_internal_size(bool* has_scalars_r, bool* has_scalars_c)
    {
        if(this->has_scalars_r)
            *has_scalars_r = 1;
        if(this->has_scalars_c)
            *has_scalars_c = 1;

        size_t shared = this->size_shared;

        for(size_t i = 0; i < nested.size(); i++)
            shared = std::max(shared, nested[i]->get_internal_size(has_scalars_r, has_scalars_c));

        return this->size_excl + shared;
    }

public:
    /* Constructor */
    rocsolver_workspace_helper()
        : has_scalars_r(0)
        , has_scalars_c(0)
        , optim_mem(0)
        , num_excl(0)
        , size_excl(0)
        , size_shared(0)
        , pointers(0)
        , sizes(0)
        , nested(0)
    {
    }

    /* Disallow copying. */
    rocsolver_workspace_helper(const rocsolver_workspace_helper&) = delete;

    /* Disallow assigning. */
    rocsolver_workspace_helper& operator=(const rocsolver_workspace_helper&) = delete;

    /* Destructor */
    ~rocsolver_workspace_helper()
    {
        for(size_t i = 0; i < nested.size(); i++)
            delete nested[i];
    }

    /* Assigns the given workspace sizes to the workspace helper. Workspace sizes are passed in two separate lists:
       sizes_excl for workspaces that must not be overwritten by nested functions, and sizes_shared for workspaces that
       may be overwritten by nested functions. Also rounds the sizes up to the nearest MIN_CHUNK_SIZE for better
       alignment. */
    void assign_sizes(std::initializer_list<size_t> sizes_excl,
                      std::initializer_list<size_t> sizes_shared = {})
    {
        this->num_excl = sizes_excl.size();
        this->size_excl = 0;
        this->size_shared = 0;

        this->indices.clear();

        this->pointers.clear();

        this->sizes.clear();
        this->sizes.reserve(sizes_excl.size() + sizes_shared.size());

        this->sizes.insert(this->sizes.end(), sizes_excl);
        this->sizes.insert(this->sizes.end(), sizes_shared);

        // round up sizes to nearest MIN_CHUNK_SIZE
        for(size_t i = 0; i < sizes.size(); i++)
        {
            this->sizes[i]
                = ((this->sizes[i] + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;

            if(i < this->num_excl)
                this->size_excl += this->sizes[i];
            else
                this->size_shared += this->sizes[i];
        }
    }
    /* Assigns the given workspace sizes to the workspace helper with associated names. Workspace sizes are passed in
       two separate lists: sizes_excl for workspaces that must not be overwritten by nested functions, and sizes_shared
       for workspaces that may be overwritten by nested functions. Also rounds the sizes up to the nearest MIN_CHUNK_SIZE
       for better alignment. */
    void assign_sizes(std::initializer_list<std::pair<std::string, size_t>> sizes_excl,
                      std::initializer_list<std::pair<std::string, size_t>> sizes_shared = {})
    {
        this->num_excl = sizes_excl.size();
        this->size_excl = 0;
        this->size_shared = 0;

        this->indices.clear();

        this->pointers.clear();

        this->sizes.clear();
        this->sizes.reserve(sizes_excl.size() + sizes_shared.size());

        // round up sizes to nearest MIN_CHUNK_SIZE
        for(auto pair : sizes_excl)
        {
            this->indices[pair.first] = sizes.size();

            size_t size = pair.second;
            size = ((size + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
            this->sizes.push_back(size);

            this->size_excl += size;
        }

        for(auto pair : sizes_shared)
        {
            this->indices[pair.first] = sizes.size();

            size_t size = pair.second;
            size = ((size + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
            this->sizes.push_back(size);

            this->size_shared += size;
        }
    }
    /* Gets the size of the workspace array at position i, rounded up to a multiple of MIN_CHUNK_SIZE. */
    size_t get_size(size_t i)
    {
        return sizes[i];
    }
    /* Gets the size of the named workspace array, rounded up to a multiple of MIN_CHUNK_SIZE. */
    size_t get_size(std::string name)
    {
        return sizes[indices[name]];
    }

    /* Sets the capacity of the internal vector that holds workspace helpers for nested functions. Optional,
       but should be called before add_nested. */
    void set_nested_capacity(size_t capacity)
    {
        nested.reserve(capacity);
    }
    /* Adds a nested workspace helper to manage workspaces for a nested function. May be called before or
       after assign_sizes. */
    rocsolver_workspace_helper* add_nested()
    {
        rocsolver_workspace_helper* result = new rocsolver_workspace_helper();
        nested.push_back(result);
        return result;
    }
    /* Adds a nested workspace helper to manage workspaces for a nested function. May be called before or
       after assign_sizes. */
    rocsolver_workspace_helper* add_nested(std::string name)
    {
        this->n_indices[name] = nested.size();
        rocsolver_workspace_helper* result = new rocsolver_workspace_helper();
        nested.push_back(result);
        return result;
    }
    /* Gets a pointer to the nested workspace helper at position i, which was created by the ith call to
       add_nested. */
    rocsolver_workspace_helper* get_nested(size_t i)
    {
        return nested[i];
    }
    /* Gets a pointer to the named nested workspace helper. */
    rocsolver_workspace_helper* get_nested(std::string name)
    {
        return nested[n_indices[name]];
    }

    /* Assigns device memory to the workspace helper, to be partitioned into individual workspace arrays
       based on the assigned sizes. Only called after assign_sizes (and, if applicable, add_nested). */
    template <typename T>
    rocblas_status assign_buffer(rocblas_handle handle, void* buffer)
    {
        using S = decltype(std::real(T{}));

        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        bool has_scalars_r = 0;
        bool has_scalars_c = 0;
        size_t size_scalars_r = 0;
        size_t size_scalars_c = 0;
        size_t size_mem = get_internal_size(&has_scalars_r, &has_scalars_c);

        if(has_scalars_r)
        {
            size_scalars_r = sizeof(S) * 3;
            size_scalars_r
                = ((size_scalars_r + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
        }
        if(has_scalars_c)
        {
            size_scalars_c = sizeof(T) * 3;
            size_scalars_c
                = ((size_scalars_c + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
        }

        // zero the workspace
        HIP_CHECK(hipMemsetAsync(buffer, 0, size_scalars_r + size_scalars_c + size_mem, stream));

        // assign pointers
        uint8_t* scalars_r = (uint8_t*)buffer;
        uint8_t* scalars_c = scalars_r + size_scalars_r;
        uint8_t* curr_ptr = scalars_c + size_scalars_c;
        assign_buffer(scalars_r, scalars_c, curr_ptr);

        // initialize scalars
        if(has_scalars_r)
            ROCSOLVER_LAUNCH_KERNEL(iota_n<S>, dim3(1), dim3(IOTA_MAX_THDS), 0, stream,
                                    (S*)scalars_r, 3, -1);
        if(has_scalars_c)
            ROCSOLVER_LAUNCH_KERNEL(iota_n<T>, dim3(1), dim3(IOTA_MAX_THDS), 0, stream,
                                    (T*)scalars_c, 3, -1);

        return rocblas_status_success;
    }
    /* Gets a pointer to the workspace array at position i, whose size is >= the assigned size at position i.
       Only called after assign_buffer. */
    void* operator[](size_t i)
    {
        return pointers[i + has_scalars_r + has_scalars_c];
    }
    /* Gets a pointer to the named workspace array. Only called after assign_buffer. */
    void* operator[](std::string name)
    {
        return pointers[indices[name] + has_scalars_r + has_scalars_c];
    }

    /* Sets a value indicating that the function requires the device-side scalar array. Only called before
       assign_buffer. */
    template <typename T>
    void add_scalars()
    {
        using S = decltype(std::real(T{}));

        if(std::is_same<T, S>::value)
            this->has_scalars_r = 1;
        else
            this->has_scalars_c = 1;
    }
    /* Gets a pointer to the device-side scalar array. Only called after assign_buffer. */
    template <typename T>
    T* get_scalars()
    {
        using S = decltype(std::real(T{}));

        if(std::is_same<T, S>::value)
        {
            if(has_scalars_r && pointers.size() > 0)
                return (T*)pointers[0];
            else
                return nullptr;
        }
        else
        {
            if(has_scalars_c && pointers.size() > has_scalars_r)
                return (T*)pointers[has_scalars_r];
            else
                return nullptr;
        }
    }

    /* Sets a value indicating if the optimal amount of memory for TRSM is available. May be called anytime. */
    void set_optim_mem(bool optim_mem)
    {
        this->optim_mem = optim_mem;
    }
    /* Gets a value indicating if the optimal amount of memory for TRSM is available. May be called anytime. */
    bool get_optim_mem()
    {
        return this->optim_mem;
    }

    /* Returns the total amount of device memory required by the workspaces assigned to this helper and its
       nested helpers. Only called after assign_sizes (and, if applicable, add_nested). */
    template <typename T>
    size_t get_total_size()
    {
        using S = decltype(std::real(T{}));

        bool has_scalars_r = 0;
        bool has_scalars_c = 0;
        size_t size_scalars_r = 0;
        size_t size_scalars_c = 0;
        size_t size_mem = get_internal_size(&has_scalars_r, &has_scalars_c);

        if(has_scalars_r)
        {
            size_scalars_r = sizeof(S) * 3;
            size_scalars_r
                = ((size_scalars_r + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
        }
        if(has_scalars_c)
        {
            size_scalars_c = sizeof(T) * 3;
            size_scalars_c
                = ((size_scalars_c + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
        }

        return size_scalars_r + size_scalars_c + size_mem;
    }

    void print_debug()
    {
        bool has_scalars_r = 0;
        bool has_scalars_c = 0;

        std::cerr << "Num Sizes: " << sizes.size() << std::endl;
        for(size_t i = 0; i < sizes.size(); i++)
        {
            std::cerr << sizes[i] << " bytes";
            if(i < pointers.size())
                std::cerr << " (" << (void*)pointers[i] << ")";
            if(i >= num_excl)
                std::cerr << " shared";
            std::cerr << std::endl;
        }
        std::cerr << "Total Size (Self): " << size_excl + size_shared << std::endl;
        if(nested.size() > 0)
            std::cerr << "Total Size (Self+Nested): "
                      << get_internal_size(&has_scalars_r, &has_scalars_c) << std::endl;
        if(has_scalars_r)
            std::cerr << "Scalar array requested" << std::endl;
        if(has_scalars_c)
            std::cerr << "Complex scalar array requested" << std::endl;
    }
};

#undef MIN_CHUNK_SIZE

ROCSOLVER_END_NAMESPACE
