#include "mxfp6_asm_utils.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_preprocess.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace mxfp6;

__global__ void lds_shuffle_kernel(
    const void* __restrict__ A_packed,
    uint32_t* __restrict__ out_regs)
{
    __shared__ uint32_t lds[512];  // 2048 bytes

    int tid = threadIdx.x;
    const char* src = reinterpret_cast<const char*>(A_packed);
    void* smem = (void*)lds;

    // Zero-cost anchor: forces compiler to allocate LDS, emits no instructions
    asm volatile("" : : "r"(smem) : "memory");

    // Pass 1: 64 threads × 16 bytes = 1024 bytes → LDS[0..1023]
    asm volatile("s_mov_b32 m0, %0" :: "s"((uint32_t)0));
    asm volatile("global_load_lds_dwordx4 %1, off offset:0"
                 : "=r"(smem) : "v"((const void*)(src + tid * 16)) : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();

    // Pass 2: 32 threads × 16 bytes = 512 bytes → LDS[1024..1535]
    if (tid < 32) {
        asm volatile("s_mov_b32 m0, %0" :: "s"((uint32_t)1024));
        asm volatile("global_load_lds_dwordx4 %1, off offset:0"
                     : "=r"(smem) : "v"((const void*)(src + 1024 + tid * 16)) : "memory");
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();

    // Compute LDS read address for MFMA lane mapping
    int lane = tid & 0xF;
    int group = (tid >> 4) & 3;
    int row = lane + ((group & 1) << 4);
    int k_half = (group >> 1) & 1;
    uint32_t lds_addr = static_cast<uint32_t>(row * 48 + k_half * 24);

    // DS_READ_B96 × 2
    v3i half0, half1;
    asm volatile("ds_read_b96 %0, %1 offset:0"  : "=v"(half0) : "v"(lds_addr) : "memory");
    asm volatile("ds_read_b96 %0, %1 offset:12" : "=v"(half1) : "v"(lds_addr) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

    // Write 6 DWORDs per thread
    uint32_t* my_out = out_regs + tid * 6;
    __builtin_memcpy(my_out,     &half0, 12);
    __builtin_memcpy(my_out + 3, &half1, 12);
}

int main() {
    const int M = 32, K = 64;
    const int packed_row_bytes = mxfp6::fp6_packed_bytes(K);
    const int tile_bytes = M * packed_row_bytes;

    std::vector<uint8_t> fp6_vals(M * K);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            fp6_vals[m * K + k] = ((m * 7 + k * 3) % 31) + 1;

    std::vector<uint8_t> A_packed(tile_bytes);
    for (int m = 0; m < M; m++)
        pack_fp6(fp6_vals.data() + m * K, K, A_packed.data() + m * packed_row_bytes);

    void* d_A;
    uint32_t* d_out;
    hipMalloc(&d_A, tile_bytes);
    hipMalloc(&d_out, 64 * 6 * sizeof(uint32_t));
    hipMemcpy(d_A, A_packed.data(), tile_bytes, hipMemcpyHostToDevice);
    hipMemset(d_out, 0, 64 * 6 * sizeof(uint32_t));

    lds_shuffle_kernel<<<1, 64>>>(d_A, d_out);
    hipDeviceSynchronize();

    std::vector<uint32_t> h_out(64 * 6);
    hipMemcpy(h_out.data(), d_out, 64 * 6 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    int pass = 0, fail = 0;
    for (int tid = 0; tid < 64; tid++) {
        int lane = tid & 0xF;
        int group = (tid >> 4) & 3;
        int row = lane + ((group & 1) << 4);
        int k_start = ((group >> 1) & 1) * 32;

        uint8_t got_packed[24];
        memcpy(got_packed, &h_out[tid * 6], 24);
        uint8_t got_fp6[32];
        unpack_fp6(got_packed, 32, got_fp6);

        bool ok = true;
        for (int i = 0; i < 32; i++) {
            uint8_t expected = fp6_vals[row * K + k_start + i];
            if (got_fp6[i] != expected) {
                if (fail < 10)
                    printf("FAIL: tid=%d (row=%d, k=%d): expected %d, got %d\n",
                           tid, row, k_start + i, expected, got_fp6[i]);
                ok = false;
                fail++;
            }
        }
        if (ok) pass++;
    }

    printf("\n========================================\n");
    printf("LDS Shuffle Test: %d/64 threads correct", pass);
    if (fail > 0) printf(", %d value mismatches", fail);
    printf("\n========================================\n");

    hipFree(d_A);
    hipFree(d_out);
    return fail > 0 ? 1 : 0;
}
