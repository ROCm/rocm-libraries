// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_INTRINSICS_THREAD_HPP_
#define ROCPRIM_INTRINSICS_THREAD_HPP_

#include <atomic>

#include "../config.hpp"
#include "../detail/various.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup intrinsicsmodule
/// @{

// Sizes

/// \brief Returns a number of threads in a hardware warp.
///
/// It is constant for a device.
ROCPRIM_HOST_DEVICE inline
constexpr unsigned int warp_size()
{
    #ifdef ROCPRIM_HC_API
        // Using marco allows contexpr, but we may have to
        // change it to hc::__wavesize() for safety
        return __HSA_WAVEFRONT_SIZE__;
    #else // HIP
        return warpSize;
    #endif
}

/// \brief Returns flat size of a multidimensional block (tile).
ROCPRIM_DEVICE inline
unsigned int flat_block_size()
{
    #ifdef ROCPRIM_HC_API
        return hc_get_group_size(2) * hc_get_group_size(1) * hc_get_group_size(0);
    #else // HIP
        return hipBlockDim_z * hipBlockDim_y * hipBlockDim_x;
    #endif
}

/// \brief Returns flat size of a multidimensional tile (block).
ROCPRIM_DEVICE inline
unsigned int flat_tile_size()
{
    return flat_block_size();
}

// IDs

/// \brief Returns thread identifier in a warp.
ROCPRIM_DEVICE inline
unsigned int lane_id()
{
    #ifdef ROCPRIM_HC_API
        return hc::__lane_id();
    #else // HIP
        return ::__lane_id();
    #endif
}

/// \brief Returns flat (linear, 1D) thread identifier in a multidimensional block (tile).
ROCPRIM_DEVICE inline
unsigned int flat_block_thread_id()
{
    #ifdef ROCPRIM_HC_API
        return (hc_get_workitem_id(2) * hc_get_group_size(1) * hc_get_group_size(0))
            + (hc_get_workitem_id(1) * hc_get_group_size(0))
            + hc_get_workitem_id(0);
    #else // HIP
        return (hipThreadIdx_z * hipBlockDim_y * hipBlockDim_x)
            + (hipThreadIdx_y * hipBlockDim_x)
            + hipThreadIdx_x;
    #endif
}

/// \brief Returns flat (linear, 1D) thread identifier in a multidimensional tile (block).
ROCPRIM_DEVICE inline
unsigned int flat_tile_thread_id()
{
    return flat_block_thread_id();
}

/// \brief Returns warp id in a block (tile).
ROCPRIM_DEVICE inline
unsigned int warp_id()
{
    return flat_block_thread_id()/warp_size();
}

/// \brief Returns flat (linear, 1D) block identifier in a multidimensional grid.
ROCPRIM_DEVICE inline
unsigned int flat_block_id()
{
    #ifdef ROCPRIM_HC_API
        return (hc_get_group_id(2) * hc_get_num_groups(1) * hc_get_num_groups(0))
            + (hc_get_group_id(1) * hc_get_num_groups(0))
            + hc_get_group_id(0);
    #else // HIP
        return (hipBlockIdx_z * hipGridDim_y * hipGridDim_x)
            + (hipBlockIdx_y * hipGridDim_x)
            + hipBlockIdx_x;
    #endif
}

// Sync

/// \brief Synchronize all threads in a block (tile)
ROCPRIM_DEVICE inline
void syncthreads()
{
    #ifdef ROCPRIM_HC_API
        hc_barrier(CLK_LOCAL_MEM_FENCE);
    #else // HIP
        __syncthreads();
    #endif
}

namespace detail
{
    /// \brief Returns thread identifier in a multidimensional block (tile) by dimension.
    template<unsigned int Dim>
    ROCPRIM_DEVICE inline
    unsigned int block_thread_id()
    {
        #ifdef ROCPRIM_HC_API
            return hc_get_workitem_id(Dim);
        #else
            static_assert(Dim > 2, "Dim must be 0, 1 or 2");
            // dummy return, correct values handled by specializations
            return 0;
        #endif
    }

    /// \brief Returns block identifier in a multidimensional grid by dimension.
    template<unsigned int Dim>
    ROCPRIM_DEVICE inline
    unsigned int block_id()
    {
        #ifdef ROCPRIM_HC_API
            return hc_get_group_id(Dim);
        #else
            static_assert(Dim > 2, "Dim must be 0, 1 or 2");
            // dummy return, correct values handled by specializations
            return 0;
        #endif
    }

    /// \brief Returns block size in a multidimensional grid by dimension.
    template<unsigned int Dim>
    ROCPRIM_DEVICE inline
    unsigned int block_size()
    {
        #ifdef ROCPRIM_HC_API
            return hc_get_group_size(Dim);
        #else
            static_assert(Dim > 2, "Dim must be 0, 1 or 2");
            // dummy return, correct values handled by specializations
            return 0;
        #endif
    }

    /// \brief Returns grid size by dimension.
    template<unsigned int Dim>
    ROCPRIM_DEVICE inline
    unsigned int grid_size()
    {
        #ifdef ROCPRIM_HC_API
            return hc_get_num_groups(Dim);
        #else
            static_assert(Dim > 2, "Dim must be 0, 1 or 2");
            // dummy return, correct values handled by specializations
            return 0;
        #endif
    }

    #define ROCPRIM_DETAIL_CONCAT(A, B) A ## B
    #define ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, dim, suffix) \
        template<> \
        ROCPRIM_DEVICE inline \
        unsigned int name<dim>() \
        { \
            return ROCPRIM_DETAIL_CONCAT(prefix, suffix); \
        }
    #define ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(name, prefix) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 0, x) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 1, y) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 2, z)

    #ifdef ROCPRIM_HIP_API
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_thread_id, hipThreadIdx_)
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_id, hipBlockIdx_)
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_size, hipBlockDim_)
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(grid_size, hipGridDim_)
    #endif

    #undef ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS
    #undef ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC
    #undef ROCPRIM_DETAIL_CONCAT

    // Return thread id in a "logical warp", which can be smaller than a hardware warp size.
    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE inline
    auto logical_lane_id()
        -> typename std::enable_if<detail::is_power_of_two(LogicalWarpSize), unsigned int>::type
    {
        return lane_id() & (LogicalWarpSize-1); // same as land_id()%WarpSize
    }

    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE inline
    auto logical_lane_id()
        -> typename std::enable_if<!detail::is_power_of_two(LogicalWarpSize), unsigned int>::type
    {
        return lane_id()%LogicalWarpSize;
    }

    template<>
    ROCPRIM_DEVICE inline
    unsigned int logical_lane_id<warp_size()>()
    {
        return lane_id();
    }

    // Return id of "logical warp" in a block
    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE inline
    unsigned int logical_warp_id()
    {
        return flat_block_thread_id()/LogicalWarpSize;
    }

    template<>
    ROCPRIM_DEVICE inline
    unsigned int logical_warp_id<warp_size()>()
    {
        return warp_id();
    }

    #ifdef ROCPRIM_HIP_API
    ROCPRIM_DEVICE inline
    void memory_fence_system(void)
    {
        ::__threadfence_system();
    }

    ROCPRIM_DEVICE inline
    void memory_fence_block(void)
    {
        ::__threadfence_block();
    }

    ROCPRIM_DEVICE inline
    void memory_fence_device(void)
    {
        ::__threadfence();
    }
    #else
    // __threadfence_system()
    ROCPRIM_DEVICE inline
    void memory_fence_system(void)
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    // Works like __threadfence_block()
    extern __attribute__((const)) ROCPRIM_DEVICE void memory_fence_block() __asm("__llvm_fence_sc_wg");

    // Works like __threadfence()
    extern __attribute__((const)) ROCPRIM_DEVICE void memory_fence_device() __asm("__llvm_fence_sc_dev");
    #endif
}

/// @}
// end of group intrinsicsmodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_THREAD_HPP_
