/*
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef SOFTWARE_GRID_SYNC_HPP
#define SOFTWARE_GRID_SYNC_HPP

#include <hip/hip_runtime.h>
namespace softwareSync
{
//     extern "C" __device__ int llvm_amdgcn_wave_id()
//         asm("llvm.amdgcn.wave.id");
extern "C" __attribute__((const)) size_t __device__ __ockl_get_local_linear_id(void);
} //softwareSync

#if defined(__GFX8__) || defined(__GFX9__)
  #define SW_WARP_SIZE 64
#else
  #define SW_WARP_SIZE 32
#endif

// Required allocation size in bytes for a kernel that calls sync() num_syncs times
// across num_blocks blocks.  Each sync() call uses its own row of num_blocks slots so
// no generation can be overwritten before all blocks have finished reading it.
// The buffer must be zeroed before the kernel launch (hipMemset to 0).
static __host__ __device__ __forceinline__ size_t softwareGridSync_buf_bytes(size_t num_blocks,
                                                                              size_t num_syncs)
{
    return num_blocks * num_syncs;
}

class SoftwareGridSync
{
    public:
    // syncBuffer must point to a zeroed buffer of softwareGridSync_buf_bytes() bytes.
    __device__ SoftwareGridSync(uint8_t* syncBuffer)
        : m_size( gridDim.x * gridDim.y * gridDim.z), m_syncIdx(0)
        {
            m_waveID = softwareSync::__ockl_get_local_linear_id() / SW_WARP_SIZE;
            m_blockIDFlat = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

            // The buffer has m_size bytes per sync call.  We store the base pointer as a
            // raw buffer descriptor covering the full allocation; each sync() advances the
            // voffset by m_size to use a fresh, zero-initialised row of slots.
            // A large num_records is used so we don't need to recompute the descriptor.
            m_syncBuffer = __builtin_amdgcn_make_buffer_rsrc( syncBuffer, // base pointer
                                                              0,          // stride (0 for raw buffer)
                                                              0xffffffff, // num_records: cover whole buffer
                                                              0x00027000  // OOB_SELECT=2 TODO: verify these flags
            );
        }

    // delete copy and move construction
    SoftwareGridSync(const SoftwareGridSync&) = delete;
    SoftwareGridSync& operator=(const SoftwareGridSync&) = delete;
    SoftwareGridSync(SoftwareGridSync&&) = delete;
    SoftwareGridSync& operator=(SoftwareGridSync&&) = delete;

    // full single grid sync with all memory fences
    void __device__ sync()
    {
        // Each sync() call uses a fresh row of slots (offset m_syncIdx * m_size) so a
        // fast block advancing to the next sync cannot overwrite a slot that a slow
        // block is still waiting on from the previous round.
        size_t row_offset = (size_t)m_syncIdx * m_size;

        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        //elect a wave to do the l2 flush (maybe we can do it once per xcc?)
        if(m_waveID == 0)
        {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

            __builtin_amdgcn_raw_buffer_store_b8( // todo, scalar buffer store?
            1,                              // write 1 (non-zero) into this block's slot
            m_syncBuffer,
            row_offset + m_blockIDFlat,    // byte offset into this sync round's row
            0,
            0x10 // set sc1 for device scope write
            );
            spinLoop(row_offset);

            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        }

        m_syncIdx++;
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    }

    // simple execution barrier, no memory fences, user responsible for memory state
    void __device__ barrier()
    {
        size_t row_offset = (size_t)m_syncIdx * m_size;
        //elect a wave to do the grid sync
        if(m_waveID == 0)
        {
            __builtin_amdgcn_raw_buffer_store_b8( // todo, scalar buffer store?
            1,
            m_syncBuffer,
            row_offset + m_blockIDFlat,
            0,
            0x10 // set sc1 for device scope write
            );
            spinLoop(row_offset);
        }
        m_syncIdx++;
        __builtin_amdgcn_s_barrier();
    }

private:
    void __device__ spinLoop(size_t row_offset)
    {
        while (true)
        {
            // use lanes to reduce loop size, each lane checks a subset of blocks then use
            // ballot to check completion across the wave.
            // Each lane in the wave checks a subset of blocks
            unsigned int lane_id = __lane_id();

            unsigned int lane_completed = 1;
            for (unsigned int i = lane_id; i < m_size; i += SW_WARP_SIZE) // TODO: can this be done better
            {
                uint8_t flag = __builtin_amdgcn_raw_buffer_load_b8(m_syncBuffer,
                                                                   row_offset + i, // byte offset into this round's row
                                                                   0,
                                                                   0x10 //set sc1 for device scope read
                                                                   );
                __builtin_amdgcn_wave_barrier(); // to avoid this branch being optimized away
                if (flag == 0) // slot is 0 until the block writes 1
                {
                    lane_completed = 0;
                    break;
                }
                lane_completed = 1;
            }

            // If lane has no blocks to check, it's trivially complete
            if (lane_id >= m_size)
            {
                lane_completed = 1;
            }

            // Use __all cross-lane function to check if all lanes report completion
            if (__all(lane_completed == 1))
            {
                break;
            }
        }
    }

    size_t  m_syncIdx;   // index of the current sync round (advances each sync call)
    size_t  m_size;      // total number of blocks (gridDim.x * y * z)
    size_t  m_waveID;
    size_t  m_blockIDFlat;
    __amdgpu_buffer_rsrc_t m_syncBuffer;
};
#endif // SOFTWARE_GRID_SYNC_HPP
