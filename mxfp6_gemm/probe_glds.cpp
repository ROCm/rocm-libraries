// Probe: determine the exact LDS write layout of global_load_lds_dwordx4 on gfx950.
// Each lane reads its own 16B (4 ints) from global; g[i]=i. We zero LDS, issue the
// async load with M0=0, then dump LDS. out[j] tells us which global index landed at
// LDS int-slot j -> reveals the per-lane LDS striding (lane*16 vs lane*4 vs ...).
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void probe(const int* __restrict__ g, int* __restrict__ out, int n_ints) {
    extern __shared__ int lds[];
    int tid = threadIdx.x;
    for (int i = tid; i < n_ints; i += blockDim.x) lds[i] = -1;
    __syncthreads();

    // M0 = LDS byte offset base (this is the only shared alloc -> base 0).
    asm volatile("s_mov_b32 m0, 0" ::: "memory");
    // FLAT async load: lane reads 16B from &g[tid*4]; LDS dest implied by M0.
    const void* gaddr = (const void*)&g[tid * 4];
    int dummy;
    asm volatile("global_load_lds_dwordx4 %1, off offset:0"
                 : "=r"(dummy) : "v"(gaddr) : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();

    for (int i = tid; i < n_ints; i += blockDim.x) out[i] = lds[i];
}

int main() {
    const int W = 64;          // 1 wave
    const int n_ints = W * 8;  // generous: 512 ints of LDS
    int *g, *out;
    int hg[W * 4], hout[n_ints];
    for (int i = 0; i < W * 4; i++) hg[i] = i;
    hipMalloc(&g, sizeof(int) * W * 4);
    hipMalloc(&out, sizeof(int) * n_ints);
    hipMemcpy(g, hg, sizeof(int) * W * 4, hipMemcpyHostToDevice);
    hipMemset(out, 0xff, sizeof(int) * n_ints);
    probe<<<1, W, sizeof(int) * n_ints>>>(g, out, n_ints);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("ERR %s\n", hipGetErrorString(e)); return 1; }
    hipMemcpy(hout, out, sizeof(int) * n_ints, hipMemcpyDeviceToHost);

    printf("LDS int-slot -> global index loaded (-1 = untouched):\n");
    for (int i = 0; i < n_ints; i++) {
        if (hout[i] != -1) printf("  lds[%3d] = %d\n", i, hout[i]);
    }
    // Infer pattern: for lane*16 (=lane*4 ints) layout, lds[i]==i identity for i<256.
    bool identity = true;
    for (int i = 0; i < W * 4; i++) if (hout[i] != i) { identity = false; break; }
    printf("\nlane*16 (contiguous 16B/lane) layout: %s\n", identity ? "YES" : "NO");
    return 0;
}
