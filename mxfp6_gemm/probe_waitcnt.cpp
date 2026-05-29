// Probe: does the compiler insert s_waitcnt for loads, depending on how the
// load is expressed? Three variants, same consumer (MFMA in inline asm).
#include <hip/hip_runtime.h>
#include <cstdint>

using v3i  = __attribute__((__vector_size__(3 * 4))) int;
using v6i  = __attribute__((__vector_size__(6 * 4))) int;
using v16f = __attribute__((__vector_size__(16 * 4))) float;

// ---- Variant A: ds_read inside inline asm, NO self-wait ----
// Question: will the compiler insert lgkmcnt before the MFMA that uses lo/hi?
__global__ void varA(const void* gB, float* out, int off) {
    __shared__ int lds[4096];
    asm volatile("" : : "r"(lds) : "memory");
    v3i lo, hi;
    asm volatile("ds_read_b96 %0, %2\n ds_read_b96 %1, %2 offset:12"
                 : "=v"(lo), "=v"(hi) : "v"(off) : "memory");
    v6i a = v6i{lo[0],lo[1],lo[2],hi[0],hi[1],hi[2]};
    v6i b = *reinterpret_cast<const v6i*>(gB);
    v16f acc = v16f{};
    asm volatile("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0 cbsz:2 blgp:2"
                 : "+v"(acc) : "v"(b), "v"(a));
    *reinterpret_cast<v16f*>(out) = acc;
}

// ---- Variant B: ds_read via plain __shared__ pointer load ----
// Compiler sees a real DS_READ machine instruction → should manage waitcnt.
__global__ void varB(const void* gB, float* out, int off) {
    __shared__ int lds[4096];
    asm volatile("" : : "r"(lds) : "memory");
    const v6i* p = reinterpret_cast<const v6i*>(
        reinterpret_cast<const char*>(lds) + off);
    v6i a = *p;
    v6i b = *reinterpret_cast<const v6i*>(gB);
    v16f acc = v16f{};
    asm volatile("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0 cbsz:2 blgp:2"
                 : "+v"(acc) : "v"(b), "v"(a));
    *reinterpret_cast<v16f*>(out) = acc;
}

// ---- Variant C: two plain LDS loads + global load, check relative counts ----
__global__ void varC(const void* gB0, const void* gB1, float* out, int o0, int o1) {
    __shared__ int lds[4096];
    asm volatile("" : : "r"(lds) : "memory");
    const char* base = reinterpret_cast<const char*>(lds);
    v6i a0 = *reinterpret_cast<const v6i*>(base + o0);
    v6i a1 = *reinterpret_cast<const v6i*>(base + o1);
    v6i b0 = *reinterpret_cast<const v6i*>(gB0);
    v6i b1 = *reinterpret_cast<const v6i*>(gB1);
    v16f acc0 = v16f{}, acc1 = v16f{};
    asm volatile("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0 cbsz:2 blgp:2"
                 : "+v"(acc0) : "v"(b0), "v"(a0));
    asm volatile("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0 cbsz:2 blgp:2"
                 : "+v"(acc1) : "v"(b1), "v"(a1));
    *reinterpret_cast<v16f*>(out) = acc0 + acc1;
}
