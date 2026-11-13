// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fused Winograd F(2,3), 3x3 stride 1, NCHW, forward and backward data (forward form, see
// ConvWinogradArgs.h). HIPRTC-compatible: no standard headers.
//
// One persistent kernel (grid = CUs x occupancy) loops over work items (group, K-block of BK out
// channels, tile-block of BT tiles). Per item, over C in chunks of CCV dword-channels (fp16: pairs):
//   - the next chunk's input patches and pre-transformed filter U are loaded while the Hadamard of
//     the current chunk runs (registers, or LDS-DMA on gfx9), then B^T d B is applied (fp32, rounded
//     once to the operand type) into the other half of a double-buffered LDS: one barrier per chunk;
//   - Hadamard: per bin a BK x BT register-blocked GEMM over the chunk, fp32 accumulate: fp32 FMA
//     (fp32, bf16 converted), or v_dot2_f32_f16 on channel pairs (fp16);
//   - epilogue: accumulators through LDS (4 passes of BK/4 out channels), A^T M A straight to y.
// Outputs whose 3x3 window lies entirely in the padding are written as exact 0 (direct-conv parity).
//
// Filter pre-pass (k5FusedFilterTransform): U as dwords [G][P][ceil(cgi/VW)][cgo] where a dword is
// one fp32 (fp32 / bf16 paths; bf16 values pre-rounded to bf16) or a half2 of two consecutive
// in-channels (fp16 path, odd C zero-padded). Size P * ceil(cgi/VW) * cgo * G * 4 bytes.
#pragma once

#include "ConvArchFamily.h"
#include "ConvWinogradCore.h"

typedef _Float16 k5h2 __attribute__((ext_vector_type(2)));

template <int V>
struct K5Par
{
    static constexpr int value = V;
};

// Unroll of the Hadamard channel loop. Full unroll lets the scheduler hoist every LDS read of the
// chunk (128 VGPRs of operands on top of 64 accumulators); 2 keeps operands to two steps.
// LDS row swizzle for the two bins of a wave (measured: no gain on gfx906; off by default).
#ifndef K5_LDS_SWIZZLE
#define K5_LDS_SWIZZLE 0
#endif
#ifndef K5_PIPE
#define K5_PIPE 0
#endif
#ifndef K5_HADAMARD_UNROLL
#define K5_HADAMARD_UNROLL 1
#endif

template <class T>
__device__ inline float k5Round(float v)
{
    return K5Elem<T>::load(K5Elem<T>::store(v));
}

// Offset that is out of range for every buffer (all < 1 GiB); sums of up to three such offsets
// stay below 2^32, so they never wrap back into range.
#define K5_OOB 0x40000000

// Buffer loads: 32-bit byte offsets from a per-tensor resource (tensor < 2 GiB; the host slices
// the batch otherwise); per-chunk and per-row offsets go in the scalar offset.
#if HKP_ARCH_GFX11 || HKP_ARCH_GFX12
#define K5_RSRC_FLAGS 0x31004000
#elif defined(__gfx1010__) || defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
#define K5_RSRC_FLAGS 0x31014000
#else
#define K5_RSRC_FLAGS 0x00020000
#endif

// Callers pass wave-uniform values; readfirstlane says so, so the descriptor stays in SGPRs and
// no buffer access becomes a waterfall loop (seen in ConvWinogradCoop with TheRock 10.3 HIPRTC).
__device__ inline __amdgpu_buffer_rsrc_t k5Rsrc(const void* base, long long bytes)
{
    const int records = bytes < 0x7fffffffLL ? static_cast<int>(bytes) : 0x7fffffff;
    const unsigned long long p = reinterpret_cast<unsigned long long>(base);
    const unsigned lo = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p));
    const unsigned hi = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p >> 32));
    return __builtin_amdgcn_make_buffer_rsrc(
        reinterpret_cast<void*>((static_cast<unsigned long long>(hi) << 32) | lo),
        static_cast<short>(0),
        __builtin_amdgcn_readfirstlane(records),
        K5_RSRC_FLAGS);
}
template <class T>
__device__ inline float k5BufLoad(__amdgpu_buffer_rsrc_t r, unsigned voff, unsigned soff)
{
    if constexpr(sizeof(T) == 4)
        return __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(r, voff, soff, 0));
    else
    {
        const unsigned short b = __builtin_amdgcn_raw_buffer_load_b16(r, voff, soff, 0);
        return K5Elem<T>::load(__builtin_bit_cast(T, b));
    }
}

// Filter pre-pass for the fused kernel. One thread per (g, in-channel dword row iv, out channel o),
// o fastest. Grid: ceil(G * ceil(cgi/VW) * cgo / 256), 256 threads.
template <class T, int M, int LAYOUT, int DIR, bool DOT2>
__device__ void k5FusedFilterTransform(const K5Args& a)
{
    constexpr int A = K5Wino<M>::A, P = A * A, VW = DOT2 ? 2 : 1;
    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiV = (cgi + VW - 1) / VW;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= static_cast<long long>(a.groups) * cgiV * cgo)
        return;
    const int o = static_cast<int>(idx % cgo);
    const int iv = static_cast<int>((idx / cgo) % cgiV);
    const int g = static_cast<int>(idx / (static_cast<long long>(cgo) * cgiV));
    constexpr int R = K5Taps<M>::R;
    float u[VW][P];
#pragma unroll
    for(int v = 0; v < VW; ++v)
    {
        const int i = iv * VW + v;
        float f[R * R];
#pragma unroll
        for(int q = 0; q < R * R; ++q)
            f[q] = 0.0f;
        if(i < cgi)
            k5GatherFilter<T, LAYOUT, DIR, R>(a, g, o, i, f);
        k5FilterTile<M>(f, u[v]);
    }
    unsigned* U = static_cast<unsigned*>(a.u);
    const long long ps = static_cast<long long>(cgiV) * cgo;
    const long long base
        = static_cast<long long>(g) * P * ps + static_cast<long long>(iv) * cgo + o;
#pragma unroll
    for(int p = 0; p < P; ++p)
    {
        unsigned word;
        if constexpr(DOT2)
        {
            k5h2 h;
            h.x = static_cast<_Float16>(u[0][p]);
            h.y = static_cast<_Float16>(u[VW - 1][p]);
            word = __builtin_bit_cast(unsigned, h);
        }
        else
            word = __float_as_uint(k5Round<T>(u[0][p]));
        U[base + p * ps] = word;
    }
}

// ---------------------------------------------------------------------------------------------
// Fused kernel v2 (the default): v21-style, double-buffered.
//
// Same work items and math as v1, restructured to hide the staging behind the Hadamard:
// - LDS holds two chunk buffers (U and V each); one barrier per chunk. While the Hadamard runs on
//   chunk ch, the global loads of chunk ch+1 are in flight (registers), and after it the chunk is
//   transformed into the other buffer.
// - 8 x 8 micro-tile per lane over a BK x BT bin tile; P bins / (64 / lanes-per-bin) waves.
//   F(2,3) 64 x 32 (or 32 x 64): 32 lanes per bin, 2 bins per wave, 8 waves (512 threads), at most
//   128 VGPRs (2 waves per SIMD, one workgroup per CU).
// - Input patches are one task per (channel, tile), 16-bit types write their half of the half2
//   word (ds_write_b16), so a lane holds one 4x4 raw patch (16 VGPRs) whatever the dtype.
// - U is loaded with one lane offset and a uniform per-slot stride.
// DMA (gfx9 / CDNA only, "vmem-to-lds-load-insts"): the next chunk's U and raw input patches are
// loaded straight into LDS (buffer_load ... lds), so neither occupies VGPRs while the Hadamard runs;
// each lane later transforms the patch it loaded itself (own LDS slots: no extra barrier). This is
// what lets 8 x 8 micro-tiles (64 accumulators) run at 2 waves per SIMD.
template <class T,
          int M,
          bool DOT2,
          int BK,
          int BT,
          int CCV,
          int TM = 8,
          int TN = 8,
          bool DMA = false>
struct K5F2Cfg
{
    static constexpr int A = K5Wino<M>::A, P = A * A, VW = DOT2 ? 2 : 1;
    static constexpr int TYN = BK / TM, TXN = BT / TN, LB = TYN * TXN, BPW = 64 / LB;
    static constexpr int THREADS = P / BPW * 64;
    static constexpr int NTASK = CCV * VW * BT, NPS = (NTASK + THREADS - 1) / THREADS;
    // U slots per thread. Regular: THREADS a multiple of BK * CCV, every thread the same (k,
    // channel) at UQ-strided bins. Otherwise (e.g. BK 24) each slot decodes its own (bin,
    // channel, k), the last ones guarded.
    static constexpr bool UREG = THREADS % (BK * CCV) == 0 && P * CCV * BK % THREADS == 0;
    static constexpr int NU = (P * CCV * BK + THREADS - 1) / THREADS;
    static constexpr int BUF = P * CCV * (BK + BT); // dwords per buffer
    static constexpr int RPP = (TM / 4) * TYN, LDM = BT + 1;
    static constexpr int RAW
        = DMA ? A * A * ((NTASK + 63) / 64 * 64) : 0; // raw patch staging (dwords)
    static constexpr int LDS = (2 * BUF + RAW > P * RPP * LDM ? 2 * BUF + RAW : P * RPP * LDM);
};

// PFD (prefetch depth): 1 loads chunk ch + 1 while chunk ch's Hadamard runs; 2 loads chunk ch + 2
// (two register sets of staged input and U), so a chunk's global loads have two Hadamards to land.
template <class T,
          int M,
          bool DOT2,
          int BK,
          int BT,
          int CCV,
          int TM,
          int TN,
          bool DMA = false,
          int PFD = 1>
__device__ void k5FusedConv2(const K5Args& a)
{
    typedef K5F2Cfg<T, M, DOT2, BK, BT, CCV, TM, TN, DMA> C;
    static_assert(!DMA || C::NPS == 1, "DMA: one patch task per lane");
    constexpr int A = C::A, P = C::P, VW = C::VW, TYN = C::TYN, TXN = C::TXN, LB = C::LB;
    constexpr int THREADS = C::THREADS, NPS = C::NPS, NU = C::NU, BUF = C::BUF, RPP = C::RPP,
                  LDM = C::LDM;
    // QUAD (F(2,3), 4 lanes per patch task): lane j of a quad loads and column-transforms
    // column j of the 4x4 patch; the row pass reads the quad's other columns through DPP, so the
    // input transform is spread over every lane (MIOpen's quad_perm trick) instead of 1 in 4.
    constexpr bool QUAD = A == 4 && C::NTASK * 4 == THREADS;
    constexpr int RAWS
        = QUAD ? THREADS : (C::NTASK + 63) / 64 * 64; // raw region stride per element
    static_assert(LB * C::BPW == 64 && P % C::BPW == 0, "wave layout");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "4-runs");
    static_assert(C::UREG || (!DMA && !QUAD), "irregular U slots: plain staging only");
    static_assert(PFD == 1 || (PFD == 2 && !DMA), "prefetch depth 2: register staging only");
    // Software-pipelined Hadamard (8 x 8 micro-tiles; K5_PIPE=0 for the plain loop).
    constexpr bool PIPE = K5_PIPE && TM == 8 && TN == 8 && !K5_LDS_SWIZZLE;
    constexpr int UQ = C::UREG ? THREADS / (BK * CCV) : 1;
    // LDS bank swizzle: the two bins of a wave are adjacent (even, odd); their rows of U and V are
    // CCV*BK (or CCV*BT) dwords apart, a multiple of the 32 banks, so their 16-byte reads would hit
    // the same banks. Odd bins store row index ^ 16 (when the row has 32+ entries).
    constexpr int SWU = (BK >= 32 && K5_LDS_SWIZZLE && !DMA) ? 16 : 0,
                  SWV = (BT >= 32 && K5_LDS_SWIZZLE) ? 16 : 0;

    __shared__ __attribute__((aligned(16))) unsigned lds[C::LDS];

    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiV = (cgi + VW - 1) / VW;
    const int nKB = (cgo + BK - 1) / BK, nTB = (a.tiles + BT - 1) / BT;
    const long long items = static_cast<long long>(a.groups) * nKB * nTB;
    const int nCh = (cgiV + CCV - 1) / CCV;
    // Stride-2 forward reads the original x through phase-gathered virtual channels (4 per
    // channel; the chunks are whole channels).
    const bool phased = a.mode == 1;
    const int inH = phased ? a.xh : a.hi, inW = phased ? a.xw : a.wi,
              inC = phased ? a.ci >> 2 : a.ci;
    const int chanStride = inH * inW;
    const __amdgpu_buffer_rsrc_t rIn = k5Rsrc(
        a.in, static_cast<long long>(a.n) * inC * inH * inW * static_cast<long long>(sizeof(T)));
    const __amdgpu_buffer_rsrc_t rU
        = k5Rsrc(a.u, static_cast<long long>(a.groups) * P * cgiV * cgo * 4);
    const int tid = threadIdx.x;
    const int bin = tid / LB, sub = tid % LB, ty = sub / TXN, tx = sub % TXN;
    T* out = static_cast<T*>(a.out);
    const int uKk = tid % BK, uCv = (tid / BK) % CCV, uP = tid / (BK * CCV);

    for(long long item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int kb = static_cast<int>(item % nKB);
        const long long r0 = item / nKB;
        const int tb = static_cast<int>(r0 % nTB);
        const int g = static_cast<int>(r0 / nTB);
        const int k0 = kb * BK, t0 = tb * BT;

        // Patch tasks: e -> (channel cl in chunk, tile tl), tile fastest.
        int sRow[NPS][A], sCol[NPS][A], sCl[NPS];
#pragma unroll
        for(int s = 0; s < NPS; ++s)
        {
            const int e = QUAD ? tid / 4 : tid + s * THREADS;
            const int cl = e / BT, tl = e % BT;
            const int t = t0 + tl;
            sCl[s] = cl;
            const bool live = e < C::NTASK && t < a.tiles;
            const int tw = t % a.tw, rr = t / a.tw;
            const int th = rr % a.th, img = rr / a.th;
            const int phase = phased ? (cl & 3) : 0, step = phased ? 2 : 1;
            const int ih0 = phased ? 2 * th * M + (phase >> 1) - a.xph : th * M - a.ph;
            const int iw0 = phased ? 2 * tw * M + (phase & 1) - a.xpw : tw * M - a.pw;
            const int chan = phased ? g * (cgi >> 2) + (cl >> 2) : g * cgi + cl;
            const int base
                = (((img * inC + chan) * inH + ih0) * inW + iw0) * static_cast<int>(sizeof(T));
#pragma unroll
            for(int i = 0; i < A; ++i)
            {
                sRow[s][i] = live && ih0 + step * i >= 0 && ih0 + step * i < inH
                                 ? base + step * i * inW * static_cast<int>(sizeof(T))
                                 : K5_OOB;
                sCol[s][i] = iw0 + step * i >= 0 && iw0 + step * i < inW
                                 ? step * i * static_cast<int>(sizeof(T))
                                 : K5_OOB;
            }
        }
        const bool uKok = k0 + uKk < cgo;

        // A chunk's staged input and U in registers (PFD sets).
        struct Staged
        {
            float raw[(DMA || QUAD) ? 1 : NPS][P];
            unsigned ureg[DMA ? 1 : NU];
            float qraw[4];
        };
        Staged sets[PFD];
        unsigned* rawL = lds + 2 * BUF; // DMA: [A*A][task]
        const int wave
            = __builtin_amdgcn_readfirstlane(tid / 64); // uniform: LDS DMA bases go to SGPRs/M0
        auto load = [&](int ch, Staged& S) {
            auto& raw = S.raw;
            auto& ureg = S.ureg;
            auto& qraw = S.qraw;
            (void)raw;
            (void)ureg;
            (void)qraw;
            const int c0 = ch * CCV * VW;
            if constexpr(QUAD && !DMA)
            {
                const int coff = c0 + sCl[0] < cgi ? (phased ? c0 >> 2 : c0) * chanStride
                                                         * static_cast<int>(sizeof(T))
                                                   : K5_OOB;
                const int col = sCol[0][tid & 3] + coff;
#pragma unroll
                for(int i = 0; i < 4; ++i)
                    qraw[i] = k5BufLoad<T>(rIn, static_cast<unsigned>(sRow[0][i] + col), 0);
                const int cvg = ch * CCV + uCv;
                const unsigned uoff = uKok && cvg < cgiV
                                          ? static_cast<unsigned>(
                                                (((g * P + uP) * cgiV + cvg) * cgo + k0 + uKk) * 4)
                                          : static_cast<unsigned>(K5_OOB);
#pragma unroll
                for(int q = 0; q < NU; ++q)
                    ureg[q] = __builtin_amdgcn_raw_buffer_load_b32(
                        rU, uoff, static_cast<unsigned>(q * UQ * cgiV * cgo * 4), 0);
                return;
            }
            if constexpr(DMA)
            {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX9__)
                __builtin_trap(); // no LDS DMA on this arch; the host never launches DMA configs here
#else
                using lds_ptr = __attribute__((address_space(3))) void*;
                if constexpr(QUAD)
                {
                    const int coff = c0 + sCl[0] < cgi ? (phased ? c0 >> 2 : c0) * chanStride
                                                             * static_cast<int>(sizeof(T))
                                                       : K5_OOB;
                    const int col = sCol[0][tid & 3] + coff;
#pragma unroll
                    for(int i = 0; i < A; ++i)
                        __builtin_amdgcn_raw_ptr_buffer_load_lds(
                            rIn,
                            (lds_ptr)(rawL + i * RAWS + wave * 64),
                            sizeof(T),
                            static_cast<unsigned>(sRow[0][i] + col),
                            0,
                            0,
                            0);
                }
                else if(tid < C::NTASK)
                {
                    const int coff = c0 + sCl[0] < cgi ? (phased ? c0 >> 2 : c0) * chanStride
                                                             * static_cast<int>(sizeof(T))
                                                       : K5_OOB;
#pragma unroll
                    for(int i = 0; i < A; ++i)
                    {
                        int row = sRow[0][i] + coff;
                        asm volatile("" : "+v"(row));
#pragma unroll
                        for(int j = 0; j < A; ++j)
                            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                                rIn,
                                (lds_ptr)(rawL + (i * A + j) * RAWS + wave * 64),
                                sizeof(T),
                                static_cast<unsigned>(row + sCol[0][j]),
                                0,
                                0,
                                0);
                    }
                }
                const int cvg = ch * CCV + uCv;
                const unsigned uoff = uKok && cvg < cgiV
                                          ? static_cast<unsigned>(
                                                (((g * P + uP) * cgiV + cvg) * cgo + k0 + uKk) * 4)
                                          : static_cast<unsigned>(K5_OOB);
                unsigned* Ub = lds + (ch & 1) * BUF;
#pragma unroll
                for(int q = 0; q < NU; ++q)
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        rU,
                        (lds_ptr)(Ub + q * THREADS + wave * 64),
                        4,
                        uoff,
                        static_cast<unsigned>(q * UQ * cgiV * cgo * 4),
                        0,
                        0);
#endif
                return;
            }
#pragma unroll
            for(int s = 0; s < NPS; ++s)
            {
                const int coff = c0 + sCl[s] < cgi ? (phased ? c0 >> 2 : c0) * chanStride
                                                         * static_cast<int>(sizeof(T))
                                                   : K5_OOB;
#pragma unroll
                for(int i = 0; i < A; ++i)
                {
                    // Opaque per chunk: stops the A*A loop-invariant row+column sums from being
                    // hoisted out of the chunk loop (A*A extra live VGPRs).
                    int row = sRow[s][i] + coff;
                    asm volatile("" : "+v"(row));
#pragma unroll
                    for(int j = 0; j < A; ++j)
                        raw[s][i * A + j]
                            = k5BufLoad<T>(rIn, static_cast<unsigned>(row + sCol[s][j]), 0);
                }
            }
            if constexpr(!C::UREG)
            {
#pragma unroll
                for(int q = 0; q < NU; ++q)
                {
                    const int sl = tid + q * THREADS;
                    const int sp = sl / (CCV * BK), scv = (sl / BK) % CCV, skk = sl % BK;
                    const int cvg = ch * CCV + scv;
                    const bool ok = sl < P * CCV * BK && k0 + skk < cgo && cvg < cgiV;
                    ureg[q] = __builtin_amdgcn_raw_buffer_load_b32(
                        rU,
                        ok ? static_cast<unsigned>((((g * P + sp) * cgiV + cvg) * cgo + k0 + skk)
                                                   * 4)
                           : static_cast<unsigned>(K5_OOB),
                        0,
                        0);
                }
                return;
            }
            const int cvg = ch * CCV + uCv;
            const unsigned uoff
                = uKok && cvg < cgiV
                      ? static_cast<unsigned>((((g * P + uP) * cgiV + cvg) * cgo + k0 + uKk) * 4)
                      : static_cast<unsigned>(K5_OOB);
#pragma unroll
            for(int q = 0; q < NU; ++q)
                ureg[q] = __builtin_amdgcn_raw_buffer_load_b32(
                    rU, uoff, static_cast<unsigned>(q * UQ * cgiV * cgo * 4), 0);
        };
        auto stage = [&](int buf, Staged& S) {
            auto& raw = S.raw;
            auto& ureg = S.ureg;
            auto& qraw = S.qraw;
            (void)raw;
            (void)ureg;
            (void)qraw;
            unsigned* Us = lds + buf * BUF;
            unsigned* Vs = Us + P * CCV * BK;
            if constexpr(QUAD)
            {
                const int j = tid & 3, e = tid / 4, cl = e / BT, tl = e % BT;
                float d[4], t[4];
                if constexpr(DMA)
                {
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                    {
                        const unsigned w = rawL[i * RAWS + tid];
                        if constexpr(sizeof(T) == 4)
                            d[i] = __uint_as_float(w);
                        else
                            d[i] = K5Elem<T>::load(
                                __builtin_bit_cast(T, static_cast<unsigned short>(w)));
                    }
                }
                else
                {
#pragma unroll
                    for(int q = 0; q < NU; ++q)
                        Us[((uP + q * UQ) * CCV + uCv) * BK + (uKk ^ (((uP + q * UQ) & 1) * SWU))]
                            = ureg[q];
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        d[i] = qraw[i];
                }
                K5Wino<2>::bt(d, 1, t, 1); // column j: B^T over the rows
                // Row pass, lane j: V[i][j] = t[i][p(j)] + sgn(j) * t[i][q(j)] with
                // (p, q, sgn) = (0,2,-) (1,2,+) (2,1,-) (1,3,-).
                const float sgn = j == 1 ? 1.0f : -1.0f;
#pragma unroll
                for(int i = 0; i < 4; ++i)
                {
                    const float pv = __int_as_float(
                        __builtin_amdgcn_update_dpp(0, __float_as_int(t[i]), 0x64, 0xF, 0xF, true));
                    const float qv = __int_as_float(
                        __builtin_amdgcn_update_dpp(0, __float_as_int(t[i]), 0xDA, 0xF, 0xF, true));
                    const float o = __builtin_fmaf(sgn, qv, pv);
                    const int w
                        = ((i * 4 + j) * CCV + cl / VW) * BT + (tl ^ (((i * 4 + j) & 1) * SWV));
                    if constexpr(DOT2)
                        reinterpret_cast<_Float16*>(Vs)[w * 2 + cl % 2] = static_cast<_Float16>(o);
                    else
                        Vs[w] = __float_as_uint(k5Round<T>(o));
                }
                return;
            }
            if constexpr(DMA)
            {
                // Own DMA'd raw patch: this lane's loads must have landed (no cross-lane reads).
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                if(tid < C::NTASK)
                {
#pragma unroll
                    for(int e = 0; e < P; ++e)
                    {
                        const unsigned w = rawL[e * RAWS + tid];
                        if constexpr(sizeof(T) == 4)
                            raw[0][e] = __uint_as_float(w);
                        else
                            raw[0][e] = K5Elem<T>::load(
                                __builtin_bit_cast(T, static_cast<unsigned short>(w)));
                    }
                }
            }
            if constexpr(!C::UREG)
            {
#pragma unroll
                for(int q = 0; q < NU; ++q)
                {
                    const int sl = tid + q * THREADS;
                    const int sp = sl / (CCV * BK), scv = (sl / BK) % CCV, skk = sl % BK;
                    if(sl < P * CCV * BK)
                        Us[(sp * CCV + scv) * BK + (skk ^ ((sp & 1) * SWU))] = ureg[q];
                }
            }
            else
            {
#pragma unroll
                for(int q = 0; q < (DMA ? 0 : NU); ++q)
                    Us[((uP + q * UQ) * CCV + uCv) * BK + (uKk ^ (((uP + q * UQ) & 1) * SWU))]
                        = ureg[q];
            }
#pragma unroll
            for(int s = 0; s < NPS; ++s)
            {
                const int e = tid + s * THREADS;
                if(e >= C::NTASK)
                    continue;
                const int cl = e / BT, tl = e % BT;
                float* d = raw[(DMA || QUAD) ? 0 : s];
#pragma unroll
                for(int j = 0; j < A; ++j)
                {
                    float t[A];
                    K5Wino<M>::bt(d + j, A, t, 1);
#pragma unroll
                    for(int i = 0; i < A; ++i)
                        d[i * A + j] = t[i];
                }
#pragma unroll
                for(int i = 0; i < A; ++i)
                {
                    float o[A];
                    K5Wino<M>::bt(d + i * A, 1, o, 1);
#pragma unroll
                    for(int j = 0; j < A; ++j)
                    {
                        const int w
                            = ((i * A + j) * CCV + cl / VW) * BT + (tl ^ (((i * A + j) & 1) * SWV));
                        if constexpr(DOT2)
                            reinterpret_cast<_Float16*>(Vs)[w * 2 + cl % 2]
                                = static_cast<_Float16>(o[j]);
                        else
                            Vs[w] = __float_as_uint(k5Round<T>(o[j]));
                    }
                }
            }
        };

        float acc[TM][TN];
#pragma unroll
        for(int i = 0; i < TM; ++i)
#pragma unroll
            for(int j = 0; j < TN; ++j)
                acc[i][j] = 0.0f;

        load(0, sets[0]);
        if(PFD == 2 && nCh > 1)
            load(1, sets[PFD - 1]);
        stage(0, sets[0]);
        if constexpr(DMA)
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        // The Hadamard of chunk ch from LDS buffer ch & 1.
        auto hadamard = [&](int ch) __attribute__((always_inline))
        {
            const unsigned* Us = lds + (ch & 1) * BUF;
            const unsigned* Vs = Us + P * CCV * BK;
            if constexpr(PIPE)
            {
                // Software-pipelined: the next channel's operands are loaded into the registers
                // freed by each quadrant of the 8 x 8 product (no extra VGPRs).
                unsigned av[8], bv[8];
                auto ld = [&](const unsigned* base, int off, unsigned* dst) {
                    const uint4 x = *reinterpret_cast<const uint4*>(base + off);
                    dst[0] = x.x, dst[1] = x.y, dst[2] = x.z, dst[3] = x.w;
                };
                auto mac = [&](int i0, int j0) {
#pragma unroll
                    for(int i = i0; i < i0 + 4; ++i)
#pragma unroll
                        for(int j = j0; j < j0 + 4; ++j)
                        {
                            if constexpr(DOT2)
                                acc[i][j] = __builtin_amdgcn_fdot2(__builtin_bit_cast(k5h2, av[i]),
                                                                   __builtin_bit_cast(k5h2, bv[j]),
                                                                   acc[i][j],
                                                                   false);
                            else
                                acc[i][j] = __builtin_fmaf(
                                    __uint_as_float(av[i]), __uint_as_float(bv[j]), acc[i][j]);
                        }
                };
                const int aOff0 = ty * 4, aOff1 = (TYN + ty) * 4, bOff0 = tx * 4,
                          bOff1 = (TXN + tx) * 4;
                ld(Us + bin * CCV * BK, aOff0, av);
                ld(Us + bin * CCV * BK, aOff1, av + 4);
                ld(Vs + bin * CCV * BT, bOff0, bv);
                ld(Vs + bin * CCV * BT, bOff1, bv + 4);
#pragma unroll 1
                for(int cv = 0; cv < CCV; ++cv)
                {
                    const int nx = cv + 1 < CCV ? cv + 1 : cv; // last step reloads (harmless)
                    const unsigned* ua = Us + (bin * CCV + nx) * BK;
                    const unsigned* vb = Vs + (bin * CCV + nx) * BT;
                    mac(0, 0);
                    mac(0, 4);
                    ld(ua, aOff0, av);
                    mac(4, 0);
                    ld(vb, bOff0, bv);
                    mac(4, 4);
                    ld(ua, aOff1, av + 4);
                    ld(vb, bOff1, bv + 4);
                }
            }
            else
#pragma unroll K5_HADAMARD_UNROLL
                for(int cv = 0; cv < CCV; ++cv)
                {
                    const unsigned* ua = Us + (bin * CCV + cv) * BK;
                    const unsigned* vb = Vs + (bin * CCV + cv) * BT;
                    unsigned av[TM], bv[TN];
#pragma unroll
                    for(int q = 0; q < TM / 4; ++q)
                    {
                        const uint4 x = *reinterpret_cast<const uint4*>(
                            ua + (((q * TYN + ty) * 4) ^ ((bin & 1) * SWU)));
                        av[q * 4 + 0] = x.x, av[q * 4 + 1] = x.y, av[q * 4 + 2] = x.z,
                                   av[q * 4 + 3] = x.w;
                    }
#pragma unroll
                    for(int q = 0; q < TN / 4; ++q)
                    {
                        const uint4 x = *reinterpret_cast<const uint4*>(
                            vb + (((q * TXN + tx) * 4) ^ ((bin & 1) * SWV)));
                        bv[q * 4 + 0] = x.x, bv[q * 4 + 1] = x.y, bv[q * 4 + 2] = x.z,
                                   bv[q * 4 + 3] = x.w;
                    }
#pragma unroll
                    for(int i = 0; i < TM; ++i)
#pragma unroll
                        for(int j = 0; j < TN; ++j)
                        {
                            if constexpr(DOT2)
                                acc[i][j] = __builtin_amdgcn_fdot2(__builtin_bit_cast(k5h2, av[i]),
                                                                   __builtin_bit_cast(k5h2, bv[j]),
                                                                   acc[i][j],
                                                                   false);
                            else
                                acc[i][j] = __builtin_fmaf(
                                    __uint_as_float(av[i]), __uint_as_float(bv[j]), acc[i][j]);
                        }
                }
        };
        // PFD 1: chunk ch + 1 loads into the one set. PFD 2: chunk ch + 2 loads into set PAR (chunk
        // ch's, staged already) and chunk ch + 1 stages from set 1 - PAR.
        auto chunk = [&](int ch, auto parTag) __attribute__((always_inline))
        {
            constexpr int PAR = decltype(parTag)::value;
            Staged& cur = sets[PFD == 2 ? PAR : 0];
            Staged& nxt = sets[PFD == 2 ? 1 - PAR : 0];
            const bool more = ch + 1 < nCh;
            if(PFD == 1 && more)
                load(ch + 1, cur);
            if(PFD == 2 && ch + 2 < nCh)
                load(ch + 2, cur);
            hadamard(ch);
            if(more)
                stage((ch + 1) & 1, nxt);
            if constexpr(DMA)
                asm volatile("s_waitcnt vmcnt(0)" ::
                                 : "memory"); // U DMA landed before others read it
            __syncthreads();
        };
        if constexpr(PFD == 1)
        {
            for(int ch = 0; ch < nCh; ++ch)
            {
                const bool more = ch + 1 < nCh;
                if(more)
                    load(ch + 1, sets[0]);
                const unsigned* Us = lds + (ch & 1) * BUF;
                const unsigned* Vs = Us + P * CCV * BK;
                if constexpr(PIPE)
                {
                    // Software-pipelined: the next channel's operands are loaded into the registers
                    // freed by each quadrant of the 8 x 8 product (no extra VGPRs).
                    unsigned av[8], bv[8];
                    auto ld = [&](const unsigned* base, int off, unsigned* dst) {
                        const uint4 x = *reinterpret_cast<const uint4*>(base + off);
                        dst[0] = x.x, dst[1] = x.y, dst[2] = x.z, dst[3] = x.w;
                    };
                    auto mac = [&](int i0, int j0) {
#pragma unroll
                        for(int i = i0; i < i0 + 4; ++i)
#pragma unroll
                            for(int j = j0; j < j0 + 4; ++j)
                            {
                                if constexpr(DOT2)
                                    acc[i][j]
                                        = __builtin_amdgcn_fdot2(__builtin_bit_cast(k5h2, av[i]),
                                                                 __builtin_bit_cast(k5h2, bv[j]),
                                                                 acc[i][j],
                                                                 false);
                                else
                                    acc[i][j] = __builtin_fmaf(
                                        __uint_as_float(av[i]), __uint_as_float(bv[j]), acc[i][j]);
                            }
                    };
                    const int aOff0 = ty * 4, aOff1 = (TYN + ty) * 4, bOff0 = tx * 4,
                              bOff1 = (TXN + tx) * 4;
                    ld(Us + bin * CCV * BK, aOff0, av);
                    ld(Us + bin * CCV * BK, aOff1, av + 4);
                    ld(Vs + bin * CCV * BT, bOff0, bv);
                    ld(Vs + bin * CCV * BT, bOff1, bv + 4);
#pragma unroll 1
                    for(int cv = 0; cv < CCV; ++cv)
                    {
                        const int nx = cv + 1 < CCV ? cv + 1 : cv; // last step reloads (harmless)
                        const unsigned* ua = Us + (bin * CCV + nx) * BK;
                        const unsigned* vb = Vs + (bin * CCV + nx) * BT;
                        mac(0, 0);
                        mac(0, 4);
                        ld(ua, aOff0, av);
                        mac(4, 0);
                        ld(vb, bOff0, bv);
                        mac(4, 4);
                        ld(ua, aOff1, av + 4);
                        ld(vb, bOff1, bv + 4);
                    }
                }
                else
#pragma unroll K5_HADAMARD_UNROLL
                    for(int cv = 0; cv < CCV; ++cv)
                    {
                        const unsigned* ua = Us + (bin * CCV + cv) * BK;
                        const unsigned* vb = Vs + (bin * CCV + cv) * BT;
                        unsigned av[TM], bv[TN];
#pragma unroll
                        for(int q = 0; q < TM / 4; ++q)
                        {
                            const uint4 x = *reinterpret_cast<const uint4*>(
                                ua + (((q * TYN + ty) * 4) ^ ((bin & 1) * SWU)));
                            av[q * 4 + 0] = x.x, av[q * 4 + 1] = x.y, av[q * 4 + 2] = x.z,
                                       av[q * 4 + 3] = x.w;
                        }
#pragma unroll
                        for(int q = 0; q < TN / 4; ++q)
                        {
                            const uint4 x = *reinterpret_cast<const uint4*>(
                                vb + (((q * TXN + tx) * 4) ^ ((bin & 1) * SWV)));
                            bv[q * 4 + 0] = x.x, bv[q * 4 + 1] = x.y, bv[q * 4 + 2] = x.z,
                                       bv[q * 4 + 3] = x.w;
                        }
#pragma unroll
                        for(int i = 0; i < TM; ++i)
#pragma unroll
                            for(int j = 0; j < TN; ++j)
                            {
                                if constexpr(DOT2)
                                    acc[i][j]
                                        = __builtin_amdgcn_fdot2(__builtin_bit_cast(k5h2, av[i]),
                                                                 __builtin_bit_cast(k5h2, bv[j]),
                                                                 acc[i][j],
                                                                 false);
                                else
                                    acc[i][j] = __builtin_fmaf(
                                        __uint_as_float(av[i]), __uint_as_float(bv[j]), acc[i][j]);
                            }
                    }
                if(more)
                    stage((ch + 1) & 1, sets[0]);
                if constexpr(DMA)
                    asm volatile("s_waitcnt vmcnt(0)" ::
                                     : "memory"); // U DMA landed before others read it
                __syncthreads();
            }
        }
        else
        {
            for(int ch = 0; ch < nCh; ch += 2)
            {
                chunk(ch, K5Par<0>{});
                if(ch + 1 < nCh)
                    chunk(ch + 1, K5Par<1>{});
            }
        }

        // Epilogue: 4 passes; pass e stages out channels k0 + rr*4 + e (rr < RPP) for every bin.
        float* Ms = reinterpret_cast<float*>(lds);
#pragma unroll
        for(int e = 0; e < 4; ++e)
        {
            if(e > 0)
                __syncthreads();
#pragma unroll
            for(int q = 0; q < TM / 4; ++q)
            {
                const int rr = q * TYN + ty;
#pragma unroll
                for(int j = 0; j < TN; ++j)
                    Ms[(bin * RPP + rr) * LDM + ((j / 4) * TXN + tx) * 4 + j % 4]
                        = acc[q * 4 + e][j];
            }
            __syncthreads();
            for(int pi = tid; pi < RPP * BT; pi += THREADS)
            {
                const int rr = pi / BT, tl = pi % BT;
                const int k = k0 + rr * 4 + e, t = t0 + tl;
                if(k >= cgo || t >= a.tiles)
                    continue;
                float mm[P], tmp[M * A], y[M * M];
#pragma unroll
                for(int p = 0; p < P; ++p)
                    mm[p] = Ms[(p * RPP + rr) * LDM + tl];
#pragma unroll
                for(int j = 0; j < A; ++j)
                    K5Wino<M>::at(mm + j, A, tmp + j, A);
#pragma unroll
                for(int i = 0; i < M; ++i)
                    K5Wino<M>::at(tmp + i * A, 1, y + i * M, 1);
                const int tw = t % a.tw, r1 = t / a.tw;
                const int th = r1 % a.th, img = r1 / a.th;
#pragma unroll
                for(int i = 0; i < M; ++i)
                {
                    const int oh = th * M + i;
                    if(oh >= a.ho)
                        continue;
#pragma unroll
                    for(int j = 0; j < M; ++j)
                    {
                        const int ow = tw * M + j;
                        if(ow < a.wo)
                            k5StoreOut<T, K5_NCHW>(a, out, img, g * cgo + k, oh, ow, y[i * M + j]);
                    }
                }
            }
        }
        __syncthreads();
    }
}
