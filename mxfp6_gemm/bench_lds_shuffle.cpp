#include "mxfp6_asm_utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

using namespace mxfp6;

// Each WG (64 threads) loads tiles in a strided loop over the entire buffer.
// This way we fix the buffer size and vary only WG count for occupancy.
__global__ void lds_shuffle_bench(
    const char* __restrict__ A_packed,
    uint32_t* __restrict__ sink,
    int total_tiles)
{
    __shared__ uint32_t lds[512];
    void* smem = (void*)lds;
    asm volatile("" : : "r"(smem) : "memory");

    int tid = threadIdx.x;
    int wg_id = blockIdx.x;
    int num_wgs = gridDim.x;

    const int TILE_BYTES = 1536;
    uint32_t accum = 0;

    // Strided loop: WG i processes tiles i, i+num_wgs, i+2*num_wgs, ...
    for (int tile_idx = wg_id; tile_idx < total_tiles; tile_idx += num_wgs) {
        const char* src = A_packed + (size_t)tile_idx * TILE_BYTES;

        // Pass 1
        asm volatile("s_mov_b32 m0, %0" :: "s"((uint32_t)0));
        asm volatile("global_load_lds_dwordx4 %1, off offset:0"
                     : "=r"(smem) : "v"((const void*)(src + tid * 16)) : "memory");
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();

        // Pass 2
        if (tid < 32) {
            asm volatile("s_mov_b32 m0, %0" :: "s"((uint32_t)1024));
            asm volatile("global_load_lds_dwordx4 %1, off offset:0"
                         : "=r"(smem) : "v"((const void*)(src + 1024 + tid * 16)) : "memory");
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();

        // DS_READ_B96 × 2
        int lane = tid & 0xF;
        int group = (tid >> 4) & 3;
        int row = lane + ((group & 1) << 4);
        int k_half = (group >> 1) & 1;
        uint32_t lds_addr = static_cast<uint32_t>(row * 48 + k_half * 24);

        v3i h0, h1;
        asm volatile("ds_read_b96 %0, %1 offset:0"  : "=v"(h0) : "v"(lds_addr) : "memory");
        asm volatile("ds_read_b96 %0, %1 offset:12" : "=v"(h1) : "v"(lds_addr) : "memory");
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        int* p0 = (int*)&h0;
        accum += p0[0];
    }

    if (tid == 0)
        sink[wg_id] = accum;
}

int main() {
    // Fixed buffer: 1 GB
    const size_t BUF_SIZE = (size_t)1 << 30;
    const int TILE_BYTES = 1536;
    int total_tiles = BUF_SIZE / TILE_BYTES;
    double total_gb = (double)total_tiles * TILE_BYTES / 1e9;

    char* d_A;
    hipMalloc(&d_A, BUF_SIZE);
    hipMemset(d_A, 0x42, BUF_SIZE);

    printf("Buffer: %.2f GB (%d tiles)\n\n", total_gb, total_tiles);
    printf("%8s  %8s  %10s  %10s\n", "WGs", "waves/CU", "time(ms)", "BW(GB/s)");
    printf("--------------------------------------------\n");

    // Sweep WG count (= wave count, since 1 wave/WG)
    int wg_counts[] = {256, 512, 1024, 2048, 4096, 8192};

    for (int wg_count : wg_counts) {
        uint32_t* d_sink;
        hipMalloc(&d_sink, wg_count * 4);

        // Warmup
        lds_shuffle_bench<<<wg_count, 64>>>(d_A, d_sink, total_tiles);
        hipDeviceSynchronize();

        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        const int ITERS = 10;
        hipEventRecord(start);
        for (int i = 0; i < ITERS; i++)
            lds_shuffle_bench<<<wg_count, 64>>>(d_A, d_sink, total_tiles);
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        double avg_ms = ms / ITERS;
        double bw = total_gb / (avg_ms / 1000.0);

        printf("%8d  %8.1f  %10.3f  %10.1f\n",
               wg_count, (double)wg_count / 256, avg_ms, bw);

        hipEventDestroy(start);
        hipEventDestroy(stop);
        hipFree(d_sink);
    }

    hipFree(d_A);
    return 0;
}
