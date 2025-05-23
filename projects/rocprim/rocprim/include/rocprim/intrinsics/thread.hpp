// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../intrinsics/arch.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup intrinsicsmodule
/// @{

// Sizes

/// \brief Returns flat size of a multidimensional block (tile).
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_block_size()
{
    return blockDim.z * blockDim.y * blockDim.x;
}

/// \brief Returns flat size of a multidimensional tile (block).
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_tile_size()
{
    return flat_block_size();
}

// IDs

/// \brief Returns thread identifier in a warp.
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int lane_id()
{
    return ::__lane_id();
}

/// \brief Returns flat (linear, 1D) thread identifier in a multidimensional block (tile).
/// \ingroup intrinsicsmodule_flat_id
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_block_thread_id()
{
    return (threadIdx.z * blockDim.y * blockDim.x)
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// \brief Returns flat (linear, 1D) thread identifier in a multidimensional block (tile). Use template parameters to optimize 1D or 2D kernels.
template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_thread_id()
    -> typename std::enable_if<(BlockSizeY == 1 && BlockSizeZ == 1), unsigned int>::type
{
    return threadIdx.x;
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_thread_id()
    -> typename std::enable_if<(BlockSizeY > 1 && BlockSizeZ == 1), unsigned int>::type
{
    return threadIdx.x + (threadIdx.y * blockDim.x);
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_thread_id()
    -> typename std::enable_if<(BlockSizeY > 1 && BlockSizeZ > 1), unsigned int>::type
{
    return threadIdx.x + (threadIdx.y * blockDim.x) +
           (threadIdx.z * blockDim.y * blockDim.x);
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Returns flat (linear, 1D) thread identifier in a multidimensional tile (block).
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_tile_thread_id()
{
    return flat_block_thread_id();
}

/// \brief Returns warp id in a block (tile).
/// \ingroup intrinsicsmodule_warp_id
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int warp_id()
{
    return flat_block_thread_id() / arch::wavefront::size();
}

/// \brief Returns warp id in a block (tile), given the flat (linear, 1D) thread identifier in a multidimensional tile (block).
/// \param flat_id the flat id that should be used to compute the warp id.
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int warp_id(unsigned int flat_id)
{
    return flat_id / arch::wavefront::size();
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// \brief Returns warp id in a block (tile). Use template parameters to optimize 1D or 2D kernels.
/// \ingroup intrinsicsmodule_warp_id
template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int warp_id()
{
    return flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>() / arch::wavefront::size();
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Returns flat (linear, 1D) block identifier in a multidimensional grid.
/// \ingroup intrinsicsmodule_flat_id
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_block_id()
{
    return (blockIdx.z * gridDim.y * gridDim.x)
        + (blockIdx.y * gridDim.x)
        + blockIdx.x;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_id()
    -> typename std::enable_if<(BlockSizeY == 1 && BlockSizeZ == 1), unsigned int>::type
{
    return blockIdx.x;
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_id()
    -> typename std::enable_if<(BlockSizeY > 1 && BlockSizeZ == 1), unsigned int>::type
{
    return blockIdx.x + (blockIdx.y * gridDim.x);
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_id()
    -> typename std::enable_if<(BlockSizeY > 1 && BlockSizeZ > 1), unsigned int>::type
{
    return blockIdx.x + (blockIdx.y * gridDim.x) +
           (blockIdx.z * gridDim.y * gridDim.x);
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

// Sync

/// \brief Synchronize all threads in a block (tile)
ROCPRIM_DEVICE ROCPRIM_INLINE
void syncthreads()
{
    __syncthreads();
}

/// \brief Synchronize all threads in the wavefront.
///
/// Wait for all threads in the wavefront before continuing execution.
/// Memory ordering is guaranteed by this function between threads in the same wavefront.
/// Threads can communicate by storing to global / shared memory, executing wave_barrier()
/// then reading values stored by other  threads in the same wavefront.
///
/// wave_barrier() should be executed by all threads in the wavefront in convergence, this means
/// that if the function is executed in a conditional statement all threads in the wave must enter
/// the conditional statement.
///
/// \note On SIMT architectures all lanes come to a convergence point simultaneously, thus no
/// special instruction is needed in the ISA.
ROCPRIM_DEVICE ROCPRIM_INLINE
void wave_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

}

namespace detail
{
    /// \brief Returns thread identifier in a multidimensional block (tile) by dimension.
    template<unsigned int Dim>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int block_thread_id()
    {
        static_assert(Dim > 2, "Dim must be 0, 1 or 2");
        // dummy return, correct values handled by specializations
        return 0;
    }

    /// \brief Returns block identifier in a multidimensional grid by dimension.
    template<unsigned int Dim>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int block_id()
    {
        static_assert(Dim > 2, "Dim must be 0, 1 or 2");
        // dummy return, correct values handled by specializations
        return 0;
    }

    /// \brief Returns block size in a multidimensional grid by dimension.
    template<unsigned int Dim>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int block_size()
    {
        static_assert(Dim > 2, "Dim must be 0, 1 or 2");
        // dummy return, correct values handled by specializations
        return 0;
    }

    /// \brief Returns grid size by dimension.
    template<unsigned int Dim>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int grid_size()
    {
        static_assert(Dim > 2, "Dim must be 0, 1 or 2");
        // dummy return, correct values handled by specializations
        return 0;
    }

    #define ROCPRIM_DETAIL_CONCAT(A, B) A B
    #define ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, dim, suffix) \
        template<> \
        ROCPRIM_DEVICE ROCPRIM_INLINE \
        unsigned int name<dim>() \
        { \
            return ROCPRIM_DETAIL_CONCAT(prefix, suffix); \
        }
    #define ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(name, prefix) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 0, x) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 1, y) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 2, z)

    ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_thread_id, threadIdx.)
    ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_id, blockIdx.)
    ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_size, blockDim.)
    ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(grid_size, gridDim.)

    #undef ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS
    #undef ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC
    #undef ROCPRIM_DETAIL_CONCAT

    // Return thread id in a "logical warp", which can be smaller than a hardware warp size.
    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto logical_lane_id()
        -> typename std::enable_if<detail::is_power_of_two(LogicalWarpSize), unsigned int>::type
    {
        return lane_id() & (LogicalWarpSize-1); // same as land_id()%WarpSize
    }

    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto logical_lane_id()
        -> typename std::enable_if<!detail::is_power_of_two(LogicalWarpSize), unsigned int>::type
    {
        return lane_id()%LogicalWarpSize;
    }

    // Return id of "logical warp" in a block
    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int logical_warp_id()
    {
        return flat_block_thread_id()/LogicalWarpSize;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void memory_fence_system()
    {
        ::__threadfence_system();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void memory_fence_block()
    {
        ::__threadfence_block();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void memory_fence_device()
    {
        ::__threadfence();
    }
}

/// @}
// end of group intrinsicsmodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_THREAD_HPP_
