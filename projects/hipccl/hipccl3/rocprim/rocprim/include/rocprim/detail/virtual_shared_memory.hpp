// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_VSMEM_HPP_
#define ROCPRIM_VSMEM_HPP_

#include "../config.hpp"
#include "../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

namespace detail
{

/// \brief Helper struct to wrap all the information needed to implement
///        virtual shared memory that's passed to a kernel.
struct vsmem_t
{
    void* gmem_ptr;
};

// The maximum amount of static shared memory available
// per thread block (64K)
static constexpr std::size_t max_smem_per_block = 1 << 16;

// Per-block virtual shared memory may be padded to make sure
// vsmem is an integer multiple of `cache_line_size`
static constexpr std::size_t cache_line_size = 128;

/// \brief Class template that helps to prevent exceeding the available
///        shared memory per thread block.
///
/// \tparam AgentT The agent for which we check whether per-thread block
///         shared memory is sufficient or whether virtual
///         shared memory is needed.
template<typename AgentT>
class vsmem_helper_impl
{
private:
    // The amount of shared memory or virtual shared memory
    // required by the algorithm
    static constexpr std::size_t required_smem = sizeof(typename AgentT::storage_type);

    // Whether we need to allocate global memory-backed virtual shared memory
    static constexpr bool needs_vsmem = required_smem > max_smem_per_block;

    // Padding bytes to an integer multiple of `cache_line_size`.
    // Only applies to virtual shared memory
    static constexpr std::size_t padding_bytes
        = (required_smem % cache_line_size == 0)
              ? 0
              : (cache_line_size - (required_smem % cache_line_size));

public:
    // Type alias to be used for static temporary storage
    // declaration within the algorithm's kernel
    using static_temp_storage_t =
        typename std::conditional<needs_vsmem, empty_type, typename AgentT::storage_type>::type;

    // The amount of global memory-backed virtual shared memory needed,
    // padded to an integer multiple of 128 bytes
    static constexpr std::size_t vsmem_per_block
        = needs_vsmem ? (required_smem + padding_bytes) : 0;

    /// \brief Used from within the device algorithm's kernel to get the temporary storage that can be
    /// passed to the agent, specialized for the case when we can use native shared memory as temporary
    /// storage.
    static ROCPRIM_DEVICE __forceinline__
    typename AgentT::storage_type&
        get_temp_storage(typename AgentT::storage_type& static_temp_storage, vsmem_t&)
    {
        return static_temp_storage;
    }

    /// \brief Used from within the device algorithm's kernel to get the temporary storage that can be
    /// passed to the agent, specialized for the case when we can use native shared memory as temporary
    /// storage and taking a linear block id.
    static ROCPRIM_DEVICE __forceinline__
    typename AgentT::storage_type&
        get_temp_storage(typename AgentT::storage_type& static_temp_storage, vsmem_t&, std::size_t)
    {
        return static_temp_storage;
    }

    /// \brief Used from within the device algorithm's kernel to get the temporary storage that can be
    /// passed to the agent, specialized for the case when we have to use global memory-backed
    /// virtual shared memory as temporary storage.
    static ROCPRIM_DEVICE __forceinline__
    typename AgentT::storage_type& get_temp_storage(empty_type&, vsmem_t& vsmem)
    {
        return *reinterpret_cast<typename AgentT::storage_type*>(static_cast<char*>(vsmem.gmem_ptr)
                                                                 + (vsmem_per_block * blockIdx.x));
    }

    /// \brief Used from within the device algorithm's kernel to get the temporary storage that can be
    /// passed to the agent, specialized for the case when we have to use global memory-backed
    /// virtual shared memory as temporary storage and taking a linear block id.
    static ROCPRIM_DEVICE __forceinline__
    typename AgentT::storage_type&
        get_temp_storage(empty_type&, vsmem_t& vsmem, std::size_t linear_block_id)
    {
        return *reinterpret_cast<typename AgentT::storage_type*>(
            static_cast<char*>(vsmem.gmem_ptr) + (vsmem_per_block * linear_block_id));
    }
};

} // namespace detail

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_VSMEM_HPP_
