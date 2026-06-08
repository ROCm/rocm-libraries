// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/*
 * fp64_emulation.cpp
 *
 * FP64 GEMM emulation via Ozaki Scheme II (accurate mode) using INT8 Tensor Cores.
 *
 * Algorithm (paper: Ozaki, Uchino, Imamura, arXiv:2504.08009)
 * -----------------------------------------------------------
 * Given  D = alpha * op(A) * op(B) + beta * C   (A,B,C,D in FP64)
 *
 * Part 1 – Accurate scaling (OS II-accu):
 *   1a. Per-row 6-bit extraction of op(A) → A8i_high, with per-row shifts sftA[i].
 *   1b. Per-col 6-bit extraction of op(B) → B8i_high, with per-col shifts sftB[j].
 *   1c. Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high  (one extra GEMM).
 *   1d. Refine sftA[i] from per-row max of |C32i_prelim| (tighter bound than Cauchy-Schwarz).
 *   1e. Refine sftB[j] from per-col max of |C32i_prelim|.
 *   1f. Final scaling: A8i[t], B8i[t] for t=0..num_moduli-1 using refined sftA, sftB.
 *
 * Part 2 – For each of s coprime moduli m_t (including implicit m_0=256):
 *   2a.  A'_t = symmetric_mod(A', m_t)  → INT8  in [-m_t/2, m_t/2]
 *   2b.  B'_t = symmetric_mod(B', m_t)  → INT8
 *   2c.  C'_t = A'_t × B'_t             → INT32  (INT8 tensor cores)
 *   2d.  Z    += C'_t * qPi_t           (CRT accumulation, double-double)
 *
 * Part 3 – Range reduction: X = Z mod M  (unique because |X| < M/2)
 *
 * Part 4 – Per-element inverse scale: D[i,j] = alpha * X[i,j] * 2^-(sftA[i]+sftB[j])
 *                                              + beta * C[i,j]
 *
 * The number of moduli s (= number of INT8 GEMMs) is configurable at runtime via
 * HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT (default: s=16, ~125 bits of CRT
 * capacity, sufficient for guaranteed FP64-equivalent results on all inputs).
 *
 * Constants (tables) are taken verbatim from the open-source GEMMul8 implementation
 * (Y. Uchino, RIKEN R-CCS, https://github.com/RIKEN-RCCS/GEMMul8).
 *
 * This file MUST be compiled as HIP (LANGUAGE HIP in CMakeLists.txt).
 * Inner INT8 GEMMs use hipblasLtMatmul (INT8 tensor cores, INT32 accumulate).
 */

#include "fp64_emulation.hpp"
#include "handle.h"   /* _rocblaslt_handle */

#include "hipblaslt/hipblaslt.h"
#include <hip/hip_runtime.h>

#include <cstdlib>   // std::getenv
#include <cstdio>    // std::fopen / std::fprintf / std::fclose / std::ftell
#include <cstring>   // std::strcmp
#include <cmath>     // std::log2, std::floor, etc.

/* =========================================================================
 * Tuning constants
 * ========================================================================= */
static constexpr double FP64_EMUL_AI_THRESHOLD = 32.0;

/* Maximum number of moduli supported (s = 2..OZ2_S_MAX).
 * Constant memory and table arrays are sized for the maximum. */
static constexpr unsigned OZ2_S_MAX = 20;

/* Alignment for INT8 arrays (128 bytes = 128 INT8 elements) */
static constexpr size_t OZ2_ALIGN = 128;

static __host__ __device__ size_t oz2_pad(size_t n)
{
    return (n + OZ2_ALIGN - 1) / OZ2_ALIGN * OZ2_ALIGN;
}

/* oz2_compute_chunk_size — automatic chunked-accumulation parameter.
 *
 * Divides the s moduli into chunks of this size.  For each chunk, all
 * chunk_size GEMMs run back-to-back (storing separate C32i slices), then ONE
 * oz2_chunk_accum_kernel folds them into Zhi/Zlo in a single register loop.
 * This reduces kernel launches from (s+1) to (ceil(s/k)+1), which matters
 * most for small N where GEMM latency is low relative to launch overhead.
 *
 * The target keeps the C32i batch (chunk_size × ldc32i × n × 4 B) below
 * OZ2_CHUNK_TARGET_BYTES.  For large N this naturally falls back to k=1
 * (current behaviour, no extra memory); for small N it picks larger k.     */
static constexpr size_t OZ2_CHUNK_TARGET_BYTES = 8ull << 30;   /* 8 GiB    */

/* oz2_compute_scale_chunk_size — number of moduli covered by ONE pair of
 * scale kernel invocations (oz2_scaleA_kernel + oz2_scaleB_kernel).
 * A and B are each read from HBM once per scale chunk, so a larger value
 * means fewer re-reads of the FP64 input matrices.
 *
 * The target keeps the combined A8i + B8i buffer below
 * OZ2_SCALE_CHUNK_TARGET_BYTES.  The result is clamped to be at least
 * gemm_chunk_sz (so every GEMM chunk fits inside one scale chunk) and at
 * most num_moduli.  For small/medium matrices the result equals num_moduli
 * (A and B are read exactly once); for very large matrices it gracefully
 * degrades toward gemm_chunk_sz.                                           */
static constexpr size_t OZ2_SCALE_CHUNK_TARGET_BYTES = 8ull << 30;  /* 8 GiB */

static unsigned oz2_compute_chunk_size(int64_t m, int64_t n, unsigned num_moduli)
{
    const size_t mn4 = static_cast<size_t>(m) * static_cast<size_t>(n) * 4u;
    if(mn4 == 0u) return num_moduli;
    const size_t k = std::max(size_t(1), OZ2_CHUNK_TARGET_BYTES / mn4);
    return static_cast<unsigned>(std::min(static_cast<size_t>(num_moduli), k));
}

static unsigned oz2_compute_scale_chunk_size(int64_t m, int64_t n, int64_t k,
                                              unsigned num_moduli,
                                              unsigned gemm_chunk_sz)
{
    const size_t lda8i       = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i      = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i       = lda8i;
    /* Combined A8i + B8i bytes per modulus slice. */
    const size_t slice_bytes = lda8i * cola8i                             /* A8i slice */
                             + ldb8i * static_cast<size_t>(n);            /* B8i slice */
    if(slice_bytes == 0u) return num_moduli;
    const size_t s = std::min(static_cast<size_t>(num_moduli),
                              std::max(static_cast<size_t>(gemm_chunk_sz),
                                       OZ2_SCALE_CHUNK_TARGET_BYTES / slice_bytes));
    return static_cast<unsigned>(s);
}

/* =========================================================================
 * GPU-side constant memory (sized for OZ2_S_MAX moduli; only the first
 * num_moduli entries are used for a given call).
 * ========================================================================= */
static __constant__ double cNegMod[OZ2_S_MAX]; /* -m_t                 */
static __constant__ double cInvMod[OZ2_S_MAX]; /* 1/m_t  (double)      */
static __constant__ float  cInvModF[OZ2_S_MAX];/* 1/m_t  (float, for 2nd-pass refinement) */
static __constant__ double cQpiHi[OZ2_S_MAX];  /* qPi high part        */
static __constant__ double cQpiLo[OZ2_S_MAX];  /* qPi correction (lo)  */
static __constant__ double cP_hi;               /* high part of -M      */
static __constant__ double cP_lo;               /* low  part of -M      */
static __constant__ double cInvP;               /* 1/M                  */

/* =========================================================================
 * Host-side tables (source: GEMMul8/GEMMul8/src/table.hpp)
 *
 * Moduli in order: 256 (implicit), 255, 253, 251, 247, 241, 239, 233,
 *                  229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173
 *                  (OZ2_S_MAX = 20 total)
 *
 * All arrays indexed by table_idx = s - 2  (s = number of moduli, 2..20).
 * ========================================================================= */

/* Per-modulus constants — same for all s, only first s entries used */
static const double h_neg_mod[OZ2_S_MAX] = {
    -256.0, -255.0, -253.0, -251.0, -247.0, -241.0, -239.0,
    -233.0, -229.0, -227.0, -223.0, -217.0, -211.0, -199.0,
    -197.0, -193.0, -191.0, -181.0, -179.0, -173.0
};
static const double h_inv_mod[OZ2_S_MAX] = {
    0x1.0000000000000p-8,   /* 1/256 */
    0x1.0101010101010p-8,   /* 1/255 */
    0x1.03091b51f5e1ap-8,   /* 1/253 */
    0x1.05197f7d73404p-8,   /* 1/251 */
    0x1.0953f39010954p-8,   /* 1/247 */
    0x1.0fef010fef011p-8,   /* 1/241 */
    0x1.12358e75d3033p-8,   /* 1/239 */
    0x1.19453808ca29cp-8,   /* 1/233 */
    0x1.1e2ef3b3fb874p-8,   /* 1/229 */
    0x1.20b470c67c0d9p-8,   /* 1/227 */
    0x1.25e22708092f1p-8,   /* 1/223 */
    0x1.2e025c04b8097p-8,   /* 1/217 */
    0x1.3698df3de0748p-8,   /* 1/211 */
    0x1.49539e3b2d067p-8,   /* 1/199 */
    0x1.4cab88725af6ep-8,   /* 1/197 */
    0x1.5390948f40febp-8,   /* 1/193 */
    0x1.571ed3c506b3ap-8,   /* 1/191 */
    0x1.6a13cd1537290p-8,   /* 1/181 */
    0x1.6e1f76b4337c7p-8,   /* 1/179 */
    0x1.7ad2208e0ecc3p-8    /* 1/173 */
};
/* Float versions of 1/m_t — used in the 2nd-pass FMA refinement.
 * Values taken verbatim from GEMMul8/GEMMul8/src/table.hpp (moduli_f[].y),
 * prepended with the exact value for m_0=256. */
static const float h_inv_mod_f[OZ2_S_MAX] = {
    0x1.000000p-8F,   /* 1/256  (exact) */
    0x1.010102p-8F,   /* 1/255  */
    0x1.03091cp-8F,   /* 1/253  */
    0x1.051980p-8F,   /* 1/251  */
    0x1.0953f4p-8F,   /* 1/247  */
    0x1.0fef02p-8F,   /* 1/241  */
    0x1.12358ep-8F,   /* 1/239  */
    0x1.194538p-8F,   /* 1/233  */
    0x1.1e2ef4p-8F,   /* 1/229  */
    0x1.20b470p-8F,   /* 1/227  */
    0x1.25e228p-8F,   /* 1/223  */
    0x1.2e025cp-8F,   /* 1/217  */
    0x1.3698e0p-8F,   /* 1/211  */
    0x1.49539ep-8F,   /* 1/199  */
    0x1.4cab88p-8F,   /* 1/197  */
    0x1.539094p-8F,   /* 1/193  */
    0x1.571ed4p-8F,   /* 1/191  */
    0x1.6a13cep-8F,   /* 1/181  */
    0x1.6e1f76p-8F,   /* 1/179  */
    0x1.7ad220p-8F    /* 1/173  */
};

/* -M (high part) for s = 2..20 */
static const double h_P_hi_all[OZ2_S_MAX - 1] = {
    -6.5280000000000000e+04,     /* s=2  */
    -1.6515840000000000e+07,     /* s=3  */
    -4.1454758400000000e+09,     /* s=4  */
    -1.0239325324800000e+12,     /* s=5  */
    -2.4676774032768000e+14,     /* s=6  */
    -5.8977489938315520e+16,     /* s=7  */
    -1.3741755155627516e+19,     /* s=8  */
    -3.1468619306387012e+21,     /* s=9  */
    -7.1433765825498518e+23,     /* s=10 */
    -1.5929729779086169e+26,     /* s=11 */
    -3.4567513620616985e+28,     /* s=12 */
    -7.2937453739501847e+30,     /* s=13 */
    -1.4514553294160867e+33,     /* s=14 */
    -2.8593669989496909e+35,     /* s=15 */
    -5.5185783079729035e+37,     /* s=16 */
    -1.0540484568228245e+40,     /* s=17 */
    -1.9078277068493124e+42,     /* s=18 */
    -3.4150115952602691e+44,     /* s=19 */
    -5.9079700598002656e+46,     /* s=20 */
};
/* -M (low part) — 0 for s <= 7 (P fits in one double) */
static const double h_P_lo_all[OZ2_S_MAX - 1] = {
     0.0,                        /* s=2  */
     0.0,                        /* s=3  */
     0.0,                        /* s=4  */
     0.0,                        /* s=5  */
     0.0,                        /* s=6  */
     0.0,                        /* s=7  */
    -2.5600000000000000e+02,     /* s=8  */
     3.1488000000000000e+04,     /* s=9  */
     4.5263360000000000e+06,     /* s=10 */
    -2.6145057280000000e+09,     /* s=11 */
    -2.0448164928000000e+12,     /* s=12 */
     3.3380381295129600e+14,     /* s=13 */
     1.0131963435176704e+16,     /* s=14 */
     1.7272206732770533e+19,     /* s=15 */
     3.2597489231298749e+21,     /* s=16 */
    -2.5574812149594794e+23,     /* s=17 */
     4.6796878119559867e+25,     /* s=18 */
     6.3951593786758970e+26,     /* s=19 */
     4.2754890730815036e+29,     /* s=20 */
};
/* 1/M for s = 2..20 */
static const double h_inv_P_all[OZ2_S_MAX - 1] = {
    1.5318627450980392e-05,      /* s=2  */
    6.0547934588855299e-08,      /* s=3  */
    2.4122683103129606e-10,      /* s=4  */
    9.7662684628055072e-13,      /* s=5  */
    4.0523935530313311e-15,      /* s=6  */
    1.6955621560800549e-17,      /* s=7  */
    7.2770907986268441e-20,      /* s=8  */
    3.1777689076973120e-22,      /* s=9  */
    1.3998981972234855e-24,      /* s=10 */
    6.2775703911367061e-27,      /* s=11 */
    2.8928895811689891e-29,      /* s=12 */
    1.3710377161938337e-31,      /* s=13 */
    6.8896367647931339e-34,      /* s=14 */
    3.4972775455802713e-36,      /* s=15 */
    1.8120609044457363e-38,      /* s=16 */
    9.4872298662080431e-41,      /* s=17 */
    5.2415634619933945e-43,      /* s=18 */
    2.9282477441303881e-45,      /* s=19 */
    1.6926287538325940e-47,      /* s=20 */
};
/* accu::log2P = log2(P-1)/2 - 0.5 for s = 2..20 (used for shift refinement) */
static const float h_accu_log2P_all[OZ2_S_MAX - 1] = {
    7.49716566e+00F,   /* s=2  */
    1.14886734e+01F,   /* s=3  */
    1.54744452e+01F,   /* s=4  */
    1.94486288e+01F,   /* s=5  */
    2.34050735e+01F,   /* s=6  */
    2.73555069e+01F,   /* s=7  */
    3.12876000e+01F,   /* s=8  */
    3.52072019e+01F,   /* s=9  */
    3.91204761e+01F,   /* s=10 */
    4.30209261e+01F,   /* s=11 */
    4.69017017e+01F,   /* s=12 */
    5.07622513e+01F,   /* s=13 */
    5.45805636e+01F,   /* s=14 */
    5.83915895e+01F,   /* s=15 */
    6.21878180e+01F,   /* s=16 */
    6.59765324e+01F,   /* s=17 */
    6.97264554e+01F,   /* s=18 */
    7.34683633e+01F,   /* s=19 */
    7.71856774e+01F,   /* s=20 */
};

/* qPi high parts for each s (row = table_idx = s-2, col = modulus index t).
 * For s <= 7 these come from qPi_1 (single double, exact product guaranteed).
 * For s >= 8 these are the hi halves of qPi_2 (double-double).             */
static const double h_qpi_hi_all[OZ2_S_MAX - 1][OZ2_S_MAX] = {
    /* s=2  (qPi_1[0]) */
    {0x1.fc02000000000p+15, 0x1.0000000000000p+8,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=3  (qPi_1[1]) */
    {0x1.50ac020000000p+23, 0x1.f60c000000000p+22, 0x1.a45a000000000p+23,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=4  (qPi_1[2]) */
    {0x1.0688601000000p+28, 0x1.f01e000000000p+28, 0x1.4826900000000p+28,
     0x1.6654440000000p+31,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=5  (qPi_1[3]) */
    {0x1.99c1435808000p+37, 0x1.d553914600000p+39, 0x1.cf9d0d8400000p+38,
     0x1.2ff09e4000000p+38, 0x1.dae0172c00000p+39,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=6  (qPi_1[4]) */
    {0x1.24d0f0aa6c020p+47, 0x1.00ffb685c4000p+47, 0x1.7820600df8000p+45,
     0x1.b28fb528de000p+47, 0x1.765c060a1c000p+47, 0x1.56b441a210000p+47,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=7  (qPi_1[5]) */
    {0x1.49071d4742060p+55, 0x1.5fae947039b40p+55, 0x1.42fdb9e1948e0p+55,
     0x1.187c8ee783700p+55, 0x1.e89ef222a1c00p+52, 0x1.0316493fe27a0p+55,
     0x1.1f8e561d65780p+53,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=8  (qPi_2[0]) */
    {0x1.4f3952ae32400p+63, 0x1.f094cf17cf000p+61, 0x1.0f5bef8d36400p+63,
     0x1.e02e9274c5000p+62, 0x1.a403bd5c1a000p+61, 0x1.a1cf7b99c2800p+62,
     0x1.a54e8a8f42000p+60, 0x1.787fdcb9fa000p+62,
     0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=9  (qPi_2[1]) */
    {0x1.9a7c80fe96000p+69, 0x1.43ca2f89db000p+71, 0x1.40f4871424000p+70,
     0x1.2c6790ef15000p+71, 0x1.24d66e4d76000p+70, 0x1.459c5b1ee5800p+71,
     0x1.d43c2b2519000p+70, 0x1.ab93da2aca000p+70, 0x1.dfbe1fda93000p+70,
     0.0,0.0,0.0,0.0,0.0},
    /* s=10 (qPi_2[2]) */
    {0x1.1ba01a9548000p+75, 0x1.b499060d20000p+76, 0x1.8d00367a82000p+77,
     0x1.348f721e1e000p+77, 0x1.09c9ed1acf000p+79, 0x1.6988bc8c28000p+75,
     0x1.4e2df779b8000p+77, 0x1.54302cc6b7000p+78, 0x1.675767107c000p+76,
     0x1.1fdfa04826000p+77,
     0.0,0.0,0.0,0.0},
    /* s=11 (qPi_2[3]) */
    {0x1.ae4dbe76d7000p+86, 0x1.258185fdee000p+86, 0x1.76fdabbf54000p+85,
     0x1.73ade1f823000p+86, 0x1.0cdeb7fb80000p+85, 0x1.0671178918000p+87,
     0x1.c416fd0741000p+86, 0x1.5350d862f8000p+86, 0x1.52567e0ff5000p+86,
     0x1.d0611c1cae000p+85, 0x1.814201f9be000p+86,
     0.0,0.0,0.0},
    /* s=12 (qPi_2[4]) */
    {0x1.42dd4f0c25000p+94, 0x1.71af2232d1000p+94, 0x1.b5f1f25063000p+93,
     0x1.0e8e8784ac000p+93, 0x1.0477c23ba5000p+93, 0x1.ac3c7c8760800p+94,
     0x1.507ba57edc000p+92, 0x1.2b20ca473f000p+93, 0x1.5f2d33fd22000p+92,
     0x1.ab17cae65c800p+94, 0x1.408e48b610000p+90, 0x1.32c582e2cf000p+94,
     0.0,0.0},
    /* s=13 (qPi_2[5]) */
    {0x1.187ecea5a8800p+102, 0x1.71af223280000p+94,  0x1.5a685a078a000p+102,
     0x1.48a0e93cba000p+102, 0x1.6d422253da000p+102, 0x1.ec015f50a0000p+101,
     0x1.27d31b1920000p+99,  0x1.7b4d942fe0000p+100, 0x1.68332a1fe8000p+101,
     0x1.7859de7afc000p+99,  0x1.317d98db46800p+102, 0x1.08b9be1306800p+102,
     0x1.411e88bd34000p+100, 0.0},
    /* s=14 (qPi_2[6]) */
    {0x1.4af9bb23b8000p+107, 0x1.e0730f7df3000p+109, 0x1.9e197740a0000p+109,
     0x1.11b44daf38000p+106, 0x1.959dba1ed5000p+109, 0x1.d3f9c70059000p+109,
     0x1.c71fc39610000p+108, 0x1.6e1a9ef495000p+109, 0x1.067fc962e0800p+110,
     0x1.81de6aed04000p+109, 0x1.086d6ad9bc800p+110, 0x1.66ccfaf43f000p+109,
     0x1.d2ae54e567000p+109, 0x1.98842ba66f000p+109},
    /* s=15 (qPi_2[7]) */
    {0x1.8334edf0c0800p+117, 0x1.d9618469e1000p+116, 0x1.4c97d49af8800p+117,
     0x1.3db0f47816800p+117, 0x1.ac11e30d56000p+116, 0x1.d3f9c70000000p+109,
     0x1.0210da6024000p+117, 0x1.2e86f6e52b000p+116, 0x1.f43197eee2000p+115,
     0x1.e913152bf0000p+115, 0x1.775c686f24000p+116, 0x1.44d556f611000p+116,
     0x1.90e2677038000p+115, 0x1.1b5f498bca000p+117, 0x1.9702ab51fa000p+116},
    /* s=16 (qPi_2[8]) */
    {0x1.568442b104000p+122, 0x1.23c286bfdb000p+125, 0x1.fffd89ae2f000p+124,
     0x1.9f80a3facf000p+124, 0x1.6b10abb2b0000p+124, 0x1.b90322c900000p+119,
     0x1.ff687bb9b9000p+124, 0x1.494950989a000p+125, 0x1.5c176f9414000p+122,
     0x1.6dca3fa2e7000p+124, 0x1.951e4290e0000p+122, 0x1.a671255128000p+123,
     0x1.b2745cf9ae000p+124, 0x1.2c6cfd90da000p+123, 0x1.a57e7d4e8e000p+124,
     0x1.8f40d0ef24000p+124},
    /* s=17 (qPi_2[9]) */
    {0x1.e01f9407c4000p+129, 0x1.e201959d63000p+131, 0x1.31982160c4000p+132,
     0x1.7f0fe22eef000p+132, 0x1.00d5bf9f80000p+126, 0x1.8ad801f1a0000p+129,
     0x1.2a9c662802000p+130, 0x1.d836977997000p+131, 0x1.85903a5f3c000p+132,
     0x1.a3320451ba800p+132, 0x1.ce462d2242000p+132, 0x1.d67cf11ca9800p+132,
     0x1.add7c7ba40000p+132, 0x1.57b0afae95000p+131, 0x1.e30840c0e8000p+128,
     0x1.5aabc9d4bf800p+132, 0x1.82a0ee308b800p+132},
    /* s=18 (qPi_2[10]) */
    {0x1.06cf388320000p+134, 0x1.a1bf2dfdc0000p+136, 0x1.bb35a9d83c000p+137,
     0x1.b0c7cfa209000p+139, 0x1.4921eae073800p+140, 0x1.172ab95fd6000p+139,
     0x1.68acfd38e8000p+139, 0x1.f34ce4f4e8000p+138, 0x1.01123dfc72000p+140,
     0x1.9db3f73893000p+139, 0x1.f6d5907a7e000p+138, 0x1.e7abc6d98b000p+139,
     0x1.8e92d65018000p+136, 0x1.1d42b11e83800p+140, 0x1.0579b3ad70800p+140,
     0x1.0cb5cec87c000p+138, 0x1.2009162ca2800p+140, 0x1.3d803cbad1800p+140},
    /* s=19 (qPi_2[11]) */
    {0x1.b09acf4b80000p+146, 0x1.6f0acc1cea000p+147, 0x1.d8992594f0000p+145,
     0x1.4be496434a000p+146, 0x1.8cc9189a96000p+147, 0x1.c776b470b0000p+143,
     0x1.cd534fe2dc000p+147, 0x1.82fa017336000p+147, 0x1.946f7304e8000p+147,
     0x1.551407a0b7000p+147, 0x1.034c6790f6000p+146, 0x1.452e68b9f4000p+145,
     0x1.407e5f3ab7000p+147, 0x1.a514c77360000p+147, 0x1.840b4e6816000p+147,
     0x1.7a503c2406000p+147, 0x1.9fa0adbac0000p+147, 0x1.0c070c3e0c000p+147,
     0x1.952a21ca4c000p+145},
    /* s=20 (qPi_2[12]) */
    {0x1.b7d0145780000p+153, 0x1.22e534dde0000p+150, 0x1.157cefeb34000p+153,
     0x1.3ca3f6e300000p+151, 0x1.016a241f28000p+152, 0x1.e66c961dd0000p+154,
     0x1.1945b982ed000p+155, 0x1.3e5ca23c80000p+152, 0x1.1ce0e51379000p+155,
     0x1.98788b5ce0000p+154, 0x1.b19e1bb310000p+154, 0x1.df2e1fa0ce000p+154,
     0x1.8b801f14e4000p+153, 0x1.38d9254eb8000p+153, 0x1.354ce8cdbc000p+154,
     0x1.94eef4587e000p+154, 0x1.3b8c91b979000p+155, 0x1.0b1e374058000p+155,
     0x1.088d7305d7000p+155, 0x1.67ddaa2cae000p+154},
};

/* qPi low parts: 0 for s <= 7 (single-double is exact), double-double lo for s >= 8 */
static const double h_qpi_lo_all[OZ2_S_MAX - 1][OZ2_S_MAX] = {
    /* s=2..7: lo = 0 */
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=8  (qPi_2[0]) */
    {0x1.16f0100000000p+20, 0x1.89a0000000000p+19, 0x1.8880000000000p+19,
     0x1.d740000000000p+19, 0x1.0b80000000000p+19, 0x1.2880000000000p+19,
     0x1.bcf0000000000p+20, 0x1.2d80000000000p+17,
     0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=9  (qPi_2[1]) */
    {0x1.008cc04000000p+26, 0x1.eca4600000000p+28, 0x1.9a00780000000p+29,
     0x1.e855180000000p+29, 0x1.e9c7f00000000p+29, 0x1.38caf00000000p+29,
     0x1.d6d0600000000p+29, 0x1.e459400000000p+27, 0x1.9cd8200000000p+27,
     0.0,0.0,0.0,0.0,0.0},
    /* s=10 (qPi_2[2]) */
    {0x1.c6c29fa008000p+37, 0x1.4ddc380000000p+30, 0x1.5e72640800000p+37,
     0x1.5939d00000000p+34, 0x1.acce161000000p+36, 0x1.d3148d7000000p+37,
     0x1.1bca621000000p+37, 0x1.be65b8a000000p+35, 0x1.43b8ee6000000p+36,
     0x1.940b60e000000p+36,
     0.0,0.0,0.0,0.0},
    /* s=11 (qPi_2[3]) */
    {0x1.c311739de0100p+44, 0x1.3f5c901690000p+45, 0x1.de7087e210000p+45,
     0x1.6bfc28bd30000p+44, 0x1.de9bee2d48000p+45, 0x1.5646b56780000p+45,
     0x1.5ee3b89260000p+43, 0x1.77449328c0000p+43, 0x1.2e0367d338000p+45,
     0x1.c1e3b22c60000p+45, 0x1.4dba603168000p+45,
     0.0,0.0,0.0},
    /* s=12 (qPi_2[4]) */
    {0x1.f5cc036fee804p+50, 0x1.9502088c71500p+52, 0x1.f27fe97ac8c00p+52,
     0x1.6a7fd4fb91000p+50, 0x1.9f4e1d77bb800p+52, 0x1.541aed8de8f00p+52,
     0x1.cad9eee787600p+51, 0x1.b754bf1ae1c00p+51, 0x1.d1793fc3ce200p+51,
     0x1.f0f2278772b00p+52, 0x1.59f94c68de600p+52, 0x1.f1f52b3aa8500p+52,
     0.0,0.0},
    /* s=13 (qPi_2[5]) */
    {0x1.2a800bf67755ap+60, 0x1.459502088c715p+60, 0x1.73141ccb58410p+57,
     0x1.956a7a15d56e0p+60, 0x1.c5f9191e4aa91p+60, 0x1.c69660c475d7bp+60,
     0x1.1a2f5dd7b0278p+60, 0x1.f7e2f13271df4p+60, 0x1.008e6afbfbd20p+59,
     0x1.45eaf7cf70b15p+60, 0x1.b2d9a1321591ap+59, 0x1.2a3c04a60a8a8p+59,
     0x1.25d2c634e54f0p+57, 0.0},
    /* s=14 (qPi_2[6]) */
    {0x1.ed366131bfd87p+61, 0x1.2a2b688425b37p+67, 0x1.ea31249d190dbp+66,
     0x1.57f8ce0e05580p+65, 0x1.34d662a4fdd1cp+66, 0x1.5013290076958p+67,
     0x1.3e7351822d438p+66, 0x1.1ebdf25941f8bp+67, 0x1.89ef93ae85687p+67,
     0x1.84d8fc60d93d4p+68, 0x1.1340b8f1c34bfp+67, 0x1.590ded2a35e12p+68,
     0x1.34cf70f07ae33p+67, 0x1.d80e799d28f38p+68},
    /* s=15 (qPi_2[7]) */
    {0x1.a4b62a6fdb1e1p+75, 0x1.c75bd2f612d0fp+75, 0x1.3f90fd4ad5142p+74,
     0x1.9e3c7f45d92bfp+75, 0x1.c9413fbd969ffp+75, 0x1.6550132900769p+75,
     0x1.a05bb4379a4c1p+75, 0x1.dae820f5ffc00p+74, 0x1.6405781ac87d9p+75,
     0x1.175dbeffba9cdp+75, 0x1.fe04b43a93e73p+71, 0x1.67335f8e813b8p+73,
     0x1.1bfe09769edb0p+75, 0x1.f1ee6037f1f5dp+71, 0x1.0c08e68bbfe9cp+75},
    /* s=16 (qPi_2[8]) */
    {0x1.195bce21a4a4cp+82, 0x1.d368142940c54p+83, 0x1.6961d82d67d29p+81,
     0x1.19861fe645aefp+78, 0x1.4aa2ee5f58c0cp+82, 0x1.42e6d8398ebf0p+83,
     0x1.09590940ec246p+83, 0x1.4ed69939f54a5p+83, 0x1.bccd986816af6p+83,
     0x1.38fff5b887f40p+83, 0x1.8c2bed86953acp+81, 0x1.2544a485cce86p+81,
     0x1.c88c3ec2fb90fp+83, 0x1.a84f9682c93f7p+83, 0x1.0beb05e6abcdfp+81,
     0x1.aeb1b1661a570p+81},
    /* s=17 (qPi_2[9]) */
    {0x1.1e87e3b708c22p+90, 0x1.4160efcbeef78p+90, 0x1.4480003b19f81p+89,
     0x1.0b25d1ed6a121p+87, 0x1.747bf6c0d8b31p+90, 0x1.bcbc193dd346cp+88,
     0x1.fddf745f1ee5ap+88, 0x1.d1f525311dabfp+90, 0x1.1660b883eb1a4p+90,
     0x1.41dedd270b797p+88, 0x1.79125e4f2418ap+90, 0x1.0272e6220fc37p+90,
     0x1.d2e0f92de9773p+87, 0x1.ea083b704edc0p+90, 0x1.eba48f8e2a378p+90,
     0x1.a977e531befa8p+90, 0x1.2ed72602864a3p+88},
    /* s=18 (qPi_2[10]) */
    {0x1.928222f7c81d9p+98, 0x1.7691a1e475ec2p+97, 0x1.50297632195fep+97,
     0x1.08e60e4fc6baep+96, 0x1.3586f9a06cbf4p+98, 0x1.7a70fdd8b6610p+98,
     0x1.5e172302320e2p+98, 0x1.1c4557a753b6cp+98, 0x1.4622df275a365p+95,
     0x1.cf7c96f698830p+95, 0x1.ebd8e9c0e37a5p+97, 0x1.f1a79e989b457p+97,
     0x1.1c8ffe978c39ep+98, 0x1.c0c39f95f19abp+92, 0x1.f55e10b41e4a2p+97,
     0x1.a546c43a54205p+98, 0x1.c22d132ce1471p+97, 0x1.6f3d636bc541bp+95},
    /* s=19 (qPi_2[11]) */
    {0x1.e0958b3fc5a41p+105, 0x1.586ae0321d89fp+104, 0x1.fd46b49aa2b42p+106,
     0x1.846cf4df36408p+106, 0x1.479c156f25f6cp+106, 0x1.020640df24636p+104,
     0x1.ddd5bd56c8a86p+106, 0x1.9e0161aac9805p+105, 0x1.a622eb926525ap+105,
     0x1.78923f9483982p+106, 0x1.cb97659eca409p+106, 0x1.e91af6550ad66p+105,
     0x1.58efe3be140c5p+106, 0x1.ef9fd35ae1d32p+103, 0x1.a8df1a86177b9p+106,
     0x1.a20f1e945f461p+105, 0x1.03cca141443f9p+106, 0x1.71f2fcf80933dp+106,
     0x1.7334b3dff2d8bp+105},
    /* s=20 (qPi_2[12]) */
    {0x1.4ca65aa6e2e69p+113, 0x1.3e4218a70dc2fp+114, 0x1.3349341ba5ba9p+114,
     0x1.a8a4cf94c4963p+113, 0x1.a9ef27a2e8284p+113, 0x1.f9453ff49eeb9p+114,
     0x1.54038a5103c5fp+114, 0x1.7e6af65bdb8e7p+114, 0x1.feec500f6cd99p+110,
     0x1.ccccb6fa6a5aep+113, 0x1.e8165d05819c5p+114, 0x1.481a67408850bp+111,
     0x1.d767562c372cdp+112, 0x1.c6ebbea0ef5f4p+113, 0x1.d062d71a7af94p+112,
     0x1.4a1e8a895454cp+111, 0x1.77cf77e873cd7p+114, 0x1.d1d5597316f21p+111,
     0x1.ed07530f9a7fap+114, 0x1.3929cf709cf74p+114},
};

/* =========================================================================
 * One-time constant-memory initialisation (parameterised by num_moduli)
 * ========================================================================= */
static hipError_t oz2_init_constants(unsigned num_moduli)
{
    static unsigned done_for = 0u;
    if(done_for == num_moduli) return hipSuccess;

    const unsigned idx = num_moduli - 2;   /* table_idx */

#define OZ2_CHECK(expr)              \
    do {                             \
        hipError_t _e = (expr);      \
        if(_e != hipSuccess) {       \
            done_for = 0u;           \
            return _e;               \
        }                            \
    } while(0)

    /* Per-modulus tables: always upload all OZ2_S_MAX entries; kernels
     * only read the first num_moduli of them.                           */
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cNegMod),  h_neg_mod,   sizeof(h_neg_mod)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvMod),  h_inv_mod,   sizeof(h_inv_mod)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvModF), h_inv_mod_f, sizeof(h_inv_mod_f)));

    /* qPi and P/invP depend on the chosen num_moduli */
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cQpiHi), h_qpi_hi_all[idx],
                                num_moduli * sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cQpiLo), h_qpi_lo_all[idx],
                                num_moduli * sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cP_hi),  &h_P_hi_all[idx],  sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cP_lo),  &h_P_lo_all[idx],  sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvP),  &h_inv_P_all[idx], sizeof(double)));

#undef OZ2_CHECK

    done_for = num_moduli;
    return hipSuccess;
}

/* =========================================================================
 * fp64EmulationIsEnabled / fp64EmulationAICheck / eager / mask / numModuli
 * ========================================================================= */
bool fp64EmulationIsEnabled()
{
    static const bool enabled = []() -> bool {
        const char* v = std::getenv("HIPBLASLT_EMULATE_DOUBLE_PRECISION");
        return (v != nullptr && std::strcmp(v, "1") == 0);
    }();
    return enabled;
}

bool fp64EmulationAICheck(int64_t m, int64_t n, int64_t k)
{
    const double flops = 2.0 * static_cast<double>(m)
                             * static_cast<double>(n)
                             * static_cast<double>(k);
    const double bytes = 8.0 * (  static_cast<double>(m) * static_cast<double>(k)
                                 + static_cast<double>(k) * static_cast<double>(n)
                                 + static_cast<double>(m) * static_cast<double>(n));
    return (flops / bytes) >= FP64_EMUL_AI_THRESHOLD;
}

bool fp64EmulationIsEager()
{
    static const bool eager = []() -> bool {
        const char* v = std::getenv("HIPBLASLT_EMULATION_STRATEGY");
        return (v != nullptr && std::strcmp(v, "eager") == 0);
    }();
    return eager;
}

uint32_t fp64EmulationSpecialValuesMask()
{
    static const uint32_t mask = []() -> uint32_t {
        const char* v = std::getenv("HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK");
        if(v == nullptr) return 0x3u;  /* default: Inf (bit 0) + NaN (bit 1) detection */
        return static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
    }();
    return mask;
}

/* =========================================================================
 * fp64EmulationWouldApply / fp64EmulationEffectiveNumModuli
 *
 * These centralise the emulation-eligibility check and num_moduli resolution
 * so that rocblaslt_matmul_impl and hipblasLtMatmulAlgoGetHeuristic share
 * exactly the same logic without duplication.
 * ========================================================================= */

/* Cumulative log2 of the product of the first s moduli (s=2..20). */
static constexpr double oz2_cum_bits[19] = {
     15.994,  /* s=2  */  23.976,  /* s=3  */  31.945,  /* s=4  */
     39.894,  /* s=5  */  47.807,  /* s=6  */  55.708,  /* s=7  */
     63.572,  /* s=8  */  71.411,  /* s=9  */  79.238,  /* s=10 */
     87.040,  /* s=11 */  94.801,  /* s=12 */ 102.522,  /* s=13 */
    110.160,  /* s=14 */ 117.782,  /* s=15 */ 125.374,  /* s=16 */
    132.949,  /* s=17 */ 140.448,  /* s=18 */ 147.931,  /* s=19 */
    155.365,  /* s=20 */
};

bool fp64EmulationWouldApply(const _rocblaslt_handle* h,
                              hipDataType              type_a,
                              int64_t                  m,
                              int64_t                  n,
                              int64_t                  k,
                              int                      batch_count)
{
    if(type_a != HIP_R_64F || batch_count != 1)
        return false;

    const bool emulEnabled = (h->emulation.enabled == 1)
                           || (h->emulation.enabled != 0 && fp64EmulationIsEnabled());
    if(!emulEnabled)
        return false;

    const bool eager = (h->emulation.strategy == 2)
                     || (h->emulation.strategy != 1 && fp64EmulationIsEager());
    return eager || fp64EmulationAICheck(m, n, k);
}

unsigned fp64EmulationEffectiveNumModuli(const _rocblaslt_handle* h)
{
    if(h->emulation.mantissa_control == 1 /* FIXED */
       && h->emulation.max_mantissa_bits >= 0)
    {
        const unsigned target = static_cast<unsigned>(h->emulation.max_mantissa_bits);
        for(unsigned s = 2u; s <= 20u; ++s) {
            if(oz2_cum_bits[s - 2u] >= static_cast<double>(target))
                return s;
        }
        return 20u;
    }
    return fp64EmulationNumModuli();
}

/* =========================================================================
 * fp64EmulationWorkspaceSize
 * ========================================================================= */
size_t fp64EmulationWorkspaceSize(int64_t m, int64_t n, int64_t k, unsigned num_moduli)
{
    const size_t lda8i       = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i      = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i       = lda8i;
    const size_t ldc32i      = cola8i;
    const size_t padn        = oz2_pad(static_cast<size_t>(n));
    const size_t szC32i      = ldc32i * static_cast<size_t>(n);
    const unsigned gemm_sz   = oz2_compute_chunk_size(m, n, num_moduli);
    const unsigned scale_sz  = oz2_compute_scale_chunk_size(m, n, k, num_moduli, gemm_sz);

    return   scale_sz * lda8i * cola8i * sizeof(int8_t)                   /* A8i batch  */
           + scale_sz * ldb8i * static_cast<size_t>(n) * sizeof(int8_t)   /* B8i batch  */
           + gemm_sz  * szC32i * sizeof(int32_t)                           /* C32i batch */
           + szC32i * sizeof(double) * 2                                   /* Zhi + Zlo  */
           + cola8i * sizeof(int16_t)                                      /* sftA       */
           + padn   * sizeof(int16_t)                                      /* sftB       */
           + sizeof(uint32_t)                                              /* nan_flag   */
           + cola8i * sizeof(int32_t);                                     /* row_max    */
}

/**
 * fp64EmulationNumModuli
 *
 * Returns the number of INT8 GEMMs (moduli) to use, in the range [2, OZ2_S_MAX].
 *
 * Reads HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT.  The value
 * specifies the total CRT capacity in bits: minimum s such that
 *   log2(prod(moduli 0..s-1)) >= target_bits.
 *
 * Default (env var absent or 0): 16 moduli (~125 bits of CRT capacity).
 * The full range [2, 20] is still reachable via the env var.
 *
 * Notable values (from design document §2.5):
 *   55 bits → s=7  ("fixed-mode default")
 *   79 bits → s=10 ("ADP max")
 */
unsigned fp64EmulationNumModuli()
{
    static const unsigned num_moduli = []() -> unsigned {
        const char* v = std::getenv("HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT");
        if(v == nullptr) return 16u;   /* default: 16 moduli (~125 bits) */

        const unsigned target = static_cast<unsigned>(std::strtoul(v, nullptr, 0));
        if(target == 0u) return OZ2_S_MAX;

        /* Cumulative log2 of the product of the first s moduli, for s=2..OZ2_S_MAX.
         * Derived from the exact moduli: 256, 255, 253, 251, 247, 241, 239, 233,
         * 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173. */
        static constexpr double cum_bits[OZ2_S_MAX - 1] = {
            15.994,   /* s=2  */
            23.976,   /* s=3  */
            31.945,   /* s=4  */
            39.894,   /* s=5  */
            47.807,   /* s=6  */
            55.708,   /* s=7  ← design doc "55 bits"  */
            63.572,   /* s=8  */
            71.411,   /* s=9  */
            79.238,   /* s=10 ← design doc "79 bits" */
            87.040,   /* s=11 */
            94.801,   /* s=12 */
           102.522,   /* s=13 */
           110.160,   /* s=14 */
           117.782,   /* s=15 */
           125.374,   /* s=16 */
           132.949,   /* s=17 */
           140.448,   /* s=18 */
           147.931,   /* s=19 */
           155.365,   /* s=20 */
        };

        for(unsigned s = 2u; s <= OZ2_S_MAX; ++s) {
            if(cum_bits[s - 2u] >= static_cast<double>(target))
                return s;
        }
        return OZ2_S_MAX;  /* target exceeds max capacity; use maximum */
    }();
    return num_moduli;
}

/* =========================================================================
 * Device helper: warp-level reductions
 * ========================================================================= */

static __device__ __forceinline__ double warp_reduce_max_abs_d(double val)
{
    val = fabs(val);
    unsigned long long bits; __builtin_memcpy(&bits, &val, 8);
    for(int off = warpSize >> 1; off > 0; off >>= 1) {
        unsigned long long other = __shfl_down(bits, off);
        if(other > bits) bits = other;
    }
    double res; __builtin_memcpy(&res, &bits, 8);
    return res;
}

static __device__ __forceinline__ int32_t warp_reduce_max_abs_i32(int32_t val)
{
    if(val < 0) val = -val;
    for(int off = warpSize >> 1; off > 0; off >>= 1) {
        int32_t other = __shfl_down(val, off);
        if(other > val) val = other;
    }
    return val;
}

static __device__ __forceinline__ double
block_reduce_max_d(double warp_max, double* __restrict__ s_wmax)
{
    if(threadIdx.x % warpSize == 0)
        s_wmax[threadIdx.x / warpSize] = warp_max;
    __syncthreads();
    double result = 0.0;
    if(threadIdx.x == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        result = s_wmax[0];
        for(int w = 1; w < num_warps; ++w)
            if(s_wmax[w] > result) result = s_wmax[w];
    }
    return result;
}

static __device__ __forceinline__ int32_t
block_reduce_max_i32(int32_t warp_max, int32_t* __restrict__ s_wmax)
{
    if(threadIdx.x % warpSize == 0)
        s_wmax[threadIdx.x / warpSize] = warp_max;
    __syncthreads();
    int32_t result = 0;
    if(threadIdx.x == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        result = s_wmax[0];
        for(int w = 1; w < num_warps; ++w)
            if(s_wmax[w] > result) result = s_wmax[w];
    }
    return result;
}

/* =========================================================================
 * GPU kernels — accu mode Part 1: preliminary extraction (A and B fused)
 *
 * oz2_accu_prelim_AB_kernel processes both A rows and B columns in a single
 * dispatch.  Grid = dim3(m + n, 1), Block = dim3(256, 1).
 *   blockIdx.x <  m_blocks → A row processing   (row = blockIdx.x)
 *   blockIdx.x >= m_blocks → B column processing (col = blockIdx.x - m_blocks)
 *
 * The branch is scalar-uniform (all threads in a block share blockIdx.x),
 * so there is zero warp divergence and zero EXEC-mask overhead.
 * Combining both dispatches into one reduces kernel-launch overhead by one
 * and gives the GPU scheduler m+n independent blocks simultaneously, which
 * improves utilisation for small problem sizes.
 * ========================================================================= */
__global__ static void
oz2_accu_prelim_AB_kernel(const double* __restrict__ A,
                          int64_t m, int64_t k, int64_t lda, bool transA,
                          int8_t*   __restrict__ A8i_high, size_t lda8i,
                          int16_t*  __restrict__ sftA,
                          const double* __restrict__ B,
                          int64_t n, int64_t ldb, bool transB,
                          int8_t*   __restrict__ B8i_high, size_t ldb8i,
                          int16_t*  __restrict__ sftB,
                          uint32_t* __restrict__ nan_flag,
                          unsigned m_blocks)
{
    /* Shared memory is reused by both branches (only one executes per block). */
    __shared__ double  s_wmax[8];
    __shared__ int16_t s_sft;

    if(blockIdx.x < m_blocks) {
        /* ── A row processing ──────────────────────────────────────────── */
        const int64_t row = static_cast<int64_t>(blockIdx.x);
        /* Guard is redundant (m_blocks == m) but kept for safety. */

        double local_max = 0.0;
        for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
            double val = transA ? A[row * lda + j] : A[j * lda + row];
            if(!isfinite(val))
                (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
            double av = fabs(val);
            if(av > local_max) local_max = av;
        }
        local_max = warp_reduce_max_abs_d(local_max);
        local_max = block_reduce_max_d(local_max, s_wmax);

        if(threadIdx.x == 0) {
            if(local_max < 1e-300) local_max = 1.0;
            /* sft = 6 - floor(log2(amax)): scales amax to ≤64 as INT8.
             * Negative sft is valid for large amax — see GEMMul8 maxUFP<INT8>=6. */
            int sft   = 6 - static_cast<int>(floor(log2(local_max)));
            s_sft     = static_cast<int16_t>(sft);
            sftA[row] = s_sft;
        }
        __syncthreads();
        const int16_t sft = s_sft;

        for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
            double val    = transA ? A[row * lda + j] : A[j * lda + row];
            double scaled = ceil(ldexp(fabs(val), static_cast<int>(sft)));
            A8i_high[static_cast<size_t>(j) + static_cast<size_t>(row) * lda8i] =
                static_cast<int8_t>(static_cast<int32_t>(scaled));
        }

    } else {
        /* ── B column processing ───────────────────────────────────────── */
        const int64_t col = static_cast<int64_t>(blockIdx.x - m_blocks);

        double local_max = 0.0;
        for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
            double val = transB ? B[col + j * ldb] : B[j + col * ldb];
            if(!isfinite(val))
                (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
            double av = fabs(val);
            if(av > local_max) local_max = av;
        }
        local_max = warp_reduce_max_abs_d(local_max);
        local_max = block_reduce_max_d(local_max, s_wmax);

        if(threadIdx.x == 0) {
            if(local_max < 1e-300) local_max = 1.0;
            int sft   = 6 - static_cast<int>(floor(log2(local_max)));
            s_sft     = static_cast<int16_t>(sft);
            sftB[col] = s_sft;
        }
        __syncthreads();
        const int16_t sft = s_sft;

        for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
            double val    = transB ? B[col + j * ldb] : B[j + col * ldb];
            double scaled = ceil(ldexp(fabs(val), static_cast<int>(sft)));
            B8i_high[static_cast<size_t>(j) + static_cast<size_t>(col) * ldb8i] =
                static_cast<int8_t>(static_cast<int32_t>(scaled));
        }
    }
}

/* =========================================================================
 * GPU kernels — accu mode Part 1: shift refinement from preliminary GEMM
 *
 * sftA refinement uses two passes to combine high occupancy with coalesced
 * access to the column-major C32i array:
 *
 * oz2_refine_sftA_partial_kernel — pass 1 (partial per-row max):
 *   Block = dim3(64), Grid = dim3(ceil(m/64), ceil(n/64)).
 *   Thread tx in each wavefront handles row = blockIdx.x*64 + tx.
 *   For each of the 64 columns covered by blockIdx.y, the wavefront reads
 *   64 consecutive rows of the same column → 256-byte coalesced burst.
 *   Result written to row_max[row] via atomicMax (workspace-resident buffer).
 *
 * oz2_refine_sftA_apply_kernel — pass 2 (apply refinement):
 *   Block = dim3(64), Grid = dim3(ceil(m/64)).
 *   Reads row_max[row], computes sftA[row] update, zeroes row_max[row].
 *
 * oz2_refine_sftB_kernel — per-column max of C32i → update sftB.
 *   Block = dim3(256), Grid = dim3(n).  One block per column; the column
 *   elements C32i[i + col*ldc32i] for i=0..m-1 are contiguous → coalesced.
 * ========================================================================= */
__global__ static void
oz2_refine_sftA_partial_kernel(const int32_t* __restrict__ C32i,
                                int64_t m, int64_t n, size_t ldc32i,
                                int32_t* __restrict__ row_max)
{
    /* One thread per row (tx dimension), one column-tile per block (y dim). */
    const int64_t row      = static_cast<int64_t>(blockIdx.x) * 64
                           + static_cast<int64_t>(threadIdx.x);
    const int64_t col_base = static_cast<int64_t>(blockIdx.y) * 64;
    if(row >= m) return;

    int32_t local_max = 0;

    /* Iterate over ≤64 columns for this tile.  For each column, the 64
     * threads in the wavefront access 64 consecutive rows → COALESCED.     */
    const int64_t col_end = (col_base + 64 < n) ? col_base + 64 : n;
    for(int64_t col = col_base; col < col_end; ++col) {
        int32_t v  = C32i[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }

    /* Write partial per-row max to the temporary buffer. */
    if(local_max > 0)
        atomicMax(row_max + static_cast<size_t>(row), local_max);
}

__global__ static void
oz2_refine_sftA_apply_kernel(const int32_t* __restrict__ row_max,
                              int16_t* __restrict__ sftA,
                              int64_t m, float log2P)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 64
                      + static_cast<int64_t>(threadIdx.x);
    if(row >= m) return;

    int32_t max_val = row_max[row];
    if(max_val < 1) max_val = 1;
    float refinement = floorf(-0.5f * log2f(static_cast<float>(max_val)) + log2P);
    sftA[row] += static_cast<int16_t>(refinement);
}

__global__ static void
oz2_refine_sftB_kernel(const int32_t* __restrict__ C32i,
                       int64_t m, int64_t n, size_t ldc32i,
                       int16_t* __restrict__ sftB,
                       float log2P)
{
    __shared__ int32_t s_wmax[8];

    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if(col >= n) return;

    int32_t local_max = 0;
    for(int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        int32_t v  = C32i[static_cast<size_t>(i) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_i32(local_max);
    local_max = block_reduce_max_i32(local_max, s_wmax);

    if(threadIdx.x == 0) {
        if(local_max < 1) local_max = 1;
        float refinement = floorf(-0.5f * log2f(static_cast<float>(local_max)) + log2P);
        sftB[col] += static_cast<int16_t>(refinement);
    }
}

/* =========================================================================
 * GPU kernels — Part 1f: full multi-modulus scaling (A and B fused)
 *
 * oz2_scaleAB_kernel_rt: runtime-count fallback used when do_pass3=true
 * (num_moduli >= 19) or for any other partial-chunk case not covered by the
 * template specialisations below.
 *
 * oz2_scaleAB_kernel<T_COUNT, DO_PASS3>: compile-time template version.
 * With T_COUNT known at compile time the compiler fully unrolls the inner
 * moduli loop.  All T_COUNT iterations are mutually independent (each
 * depends only on ival, not on any prior iteration), so the unrolled code
 * exposes instruction-level parallelism across all T_COUNT FP64 computation
 * chains simultaneously, hiding their individual ~5-10 cycle latencies.
 *
 * Both variants use __builtin_nontemporal_store for INT8 writes to A8i and
 * B8i.  These buffers are write-only during scaling and together occupy
 * ~8 GiB (2 × num_moduli × m × k bytes), far exceeding L2 capacity.
 * Streaming stores bypass the L2, eliminating read-for-ownership traffic and
 * keeping the cache warm for the FP64 reads of A and B.
 *
 * Grid = dim3(ceil(k/64), ceil(m/4) + ceil(n/4)), Block = dim3(64, 4).
 *   blockIdx.y <  m_y_blocks → A scaling: i = blockIdx.y*4 + threadIdx.y
 *   blockIdx.y >= m_y_blocks → B scaling: l = (blockIdx.y - m_y_blocks)*4 + threadIdx.y
 *
 * threadIdx.x covers the k-index (fast) in both branches → A8i and B8i writes
 * remain coalesced (64 consecutive k-elements per wavefront).
 * The branch on blockIdx.y is scalar-uniform: zero warp divergence.
 * ========================================================================= */
__global__ static void
oz2_scaleAB_kernel_rt(const double* __restrict__ A,
                      int64_t m, int64_t lda, bool transA,
                      int8_t*  __restrict__       A8i,
                      size_t                      lda8i,
                      size_t                      cola8i,
                      const int16_t* __restrict__ sftA,
                      const double* __restrict__ B,
                      int64_t n, int64_t ldb, bool transB,
                      int8_t*  __restrict__       B8i,
                      size_t                      ldb8i,
                      const int16_t* __restrict__ sftB,
                      int64_t k,
                      unsigned t_start, unsigned t_count, bool do_pass3,
                      unsigned m_y_blocks)
{
    const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; /* k-index */

    if(blockIdx.y < m_y_blocks) {
        /* ── A scaling ─────────────────────────────────────────────────── */
        const int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
        if(i >= m || j >= k) return;

        const double val  = transA ? A[i * lda + j] : A[j * lda + i];
        const int16_t sft = sftA[i];
        const double ival = trunc(ldexp(val, static_cast<int>(sft)));

        const size_t stride = lda8i * cola8i;
        const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(i) * lda8i;

        for(unsigned t_local = 0; t_local < t_count; ++t_local) {
            const unsigned t = t_start + t_local;
            const double  r   = fma(cNegMod[t], rint(ival * cInvMod[t]), ival);
            const float   rf  = static_cast<float>(r);
            float rf2 = fmaf(rintf(rf * cInvModF[t]),
                             static_cast<float>(cNegMod[t]), rf);
            if(do_pass3)
                rf2 = fmaf(rintf(rf2 * cInvModF[t]),
                           static_cast<float>(cNegMod[t]), rf2);
            __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                        A8i + t_local * stride + offset);
        }

    } else {
        /* ── B scaling ─────────────────────────────────────────────────── */
        const int64_t l = static_cast<int64_t>(blockIdx.y - m_y_blocks) * blockDim.y
                        + threadIdx.y;
        if(l >= n || j >= k) return;

        const double val  = transB ? B[j * ldb + l] : B[l * ldb + j];
        const int16_t sft = sftB[l];
        const double ival = trunc(ldexp(val, static_cast<int>(sft)));

        const size_t stride = ldb8i * static_cast<size_t>(n);
        const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(l) * ldb8i;

        for(unsigned t_local = 0; t_local < t_count; ++t_local) {
            const unsigned t = t_start + t_local;
            const double  r   = fma(cNegMod[t], rint(ival * cInvMod[t]), ival);
            const float   rf  = static_cast<float>(r);
            float rf2 = fmaf(rintf(rf * cInvModF[t]),
                             static_cast<float>(cNegMod[t]), rf);
                rf2 = fmaf(rintf(rf2 * cInvModF[t]),
                           static_cast<float>(cNegMod[t]), rf2);
            __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                        B8i + t_local * stride + offset);
        }
    }
}

/* Template version: T_COUNT moduli are all mutually independent → the compiler
 * can issue all T_COUNT FP64 chains in parallel after full loop unrolling.
 * DO_PASS3 controls the optional third FP32 refinement pass (only needed
 * when num_moduli >= 19; evaluated at compile time via if constexpr).      */
template <unsigned T_COUNT, bool DO_PASS3>
__global__ static void
oz2_scaleAB_kernel(const double* __restrict__ A,
                   int64_t m, int64_t lda, bool transA,
                   int8_t*  __restrict__       A8i,
                   size_t                      lda8i,
                   size_t                      cola8i,
                   const int16_t* __restrict__ sftA,
                   const double* __restrict__ B,
                   int64_t n, int64_t ldb, bool transB,
                   int8_t*  __restrict__       B8i,
                   size_t                      ldb8i,
                   const int16_t* __restrict__ sftB,
                   int64_t k,
                   unsigned t_start, unsigned m_y_blocks)
{
    const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; /* k-index */

    if(blockIdx.y < m_y_blocks) {
        /* ── A scaling ─────────────────────────────────────────────────── */
        const int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
        if(i >= m || j >= k) return;

        const double val  = transA ? A[i * lda + j] : A[j * lda + i];
        const int16_t sft = sftA[i];
        const double ival = trunc(ldexp(val, static_cast<int>(sft)));

        const size_t stride = lda8i * cola8i;
        const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(i) * lda8i;

        #pragma unroll
        for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
            const unsigned t = t_start + t_local;
            const double  r   = fma(cNegMod[t], rint(ival * cInvMod[t]), ival);
            const float   rf  = static_cast<float>(r);
            float rf2 = fmaf(rintf(rf * cInvModF[t]),
                             static_cast<float>(cNegMod[t]), rf);
            if constexpr (DO_PASS3)
                rf2 = fmaf(rintf(rf2 * cInvModF[t]),
                           static_cast<float>(cNegMod[t]), rf2);
            __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                        A8i + t_local * stride + offset);
        }

    } else {
        /* ── B scaling ─────────────────────────────────────────────────── */
        const int64_t l = static_cast<int64_t>(blockIdx.y - m_y_blocks) * blockDim.y
                        + threadIdx.y;
        if(l >= n || j >= k) return;

        const double val  = transB ? B[j * ldb + l] : B[l * ldb + j];
        const int16_t sft = sftB[l];
        const double ival = trunc(ldexp(val, static_cast<int>(sft)));

        const size_t stride = ldb8i * static_cast<size_t>(n);
        const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(l) * ldb8i;

        #pragma unroll
        for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
            const unsigned t = t_start + t_local;
            const double  r   = fma(cNegMod[t], rint(ival * cInvMod[t]), ival);
            const float   rf  = static_cast<float>(r);
            float rf2 = fmaf(rintf(rf * cInvModF[t]),
                             static_cast<float>(cNegMod[t]), rf);
                rf2 = fmaf(rintf(rf2 * cInvModF[t]),
                           static_cast<float>(cNegMod[t]), rf2);
            __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                        B8i + t_local * stride + offset);
        }
    }
}

/* =========================================================================
 * GPU kernels — Part 2d: chunked CRT accumulation
 *
 * oz2_chunk_accum_kernel_rt<HAS_LO>: runtime-count fallback, used for
 * chunk sizes not covered by the template specialisations below.
 *
 * oz2_chunk_accum_kernel<HAS_LO, CHUNK_SIZE, IS_FIRST_CHUNK>: template
 * version with CHUNK_SIZE known at compile time, enabling #pragma unroll.
 * With the loop unrolled the compiler pre-issues all CHUNK_SIZE C32i_batch
 * loads simultaneously, overlapping their ~100-cycle HBM latencies instead
 * of serialising them one after the other.  IS_FIRST_CHUNK=true additionally
 * enables __builtin_nontemporal_store for the Zhi/Zlo writes (write-only
 * in that path, so bypassing L2 keeps the cache warm for C32i reads).
 *
 * oz2_accum_finalize_kernel_rt<HAS_LO> / oz2_accum_finalize_kernel<…>:
 * same pattern for the fused last-chunk kernel.
 *
 * Template parameter HAS_LO:
 *   true  — s ≥ 8: cQpiLo[t] is non-zero; fuse dc×cQpiLo[t] into a FMA
 *            with the local_lo update, saving one multiply vs the separate
 *            lo = dc*cQpiLo[t] + two-add form.
 *   false — s ≤ 7: cQpiLo[t] = 0 for all t (single-double qPi suffices);
 *            skip the lo multiply entirely and reduce local_lo += err,
 *            saving two FP64 ops per modulus.
 * ========================================================================= */
template <bool HAS_LO>
__global__ static void
oz2_chunk_accum_kernel_rt(const int32_t* __restrict__ C32i_batch,
                       double*        __restrict__ Zhi,
                       double*        __restrict__ Zlo,
                       int64_t                     m,
                       int64_t                     n,
                       size_t                      ldc32i,
                       unsigned                    chunk_start,
                       unsigned                    chunk_size,
                       bool                        is_first_chunk)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;

    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);

    /* Register-level double-double accumulator for this chunk */
    double local_hi = 0.0;
    double local_lo = 0.0;

    for(unsigned t_local = 0; t_local < chunk_size; ++t_local) {
        const unsigned t     = chunk_start + t_local;
        const double dc_raw  = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        /* Reduce to symmetric residue mod m_t (exact dc * qPiHi[t] in double) */
        const double dc      = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi      = dc * cQpiHi[t];
        /* 2Sum accumulation of hi into local_hi */
        const double new_hi  = local_hi + hi;
        const double err     = hi - (new_hi - local_hi);
        local_hi = new_hi;
        /* lo update — two variants selected at compile time:
         *   HAS_LO=true:  fuse dc×cQpiLo[t] and the lo+err accumulation into
         *                 one FMA, saving one multiply over the split form.
         *   HAS_LO=false: cQpiLo[t] == 0 for all t (s ≤ 7); skip entirely. */
        if constexpr (HAS_LO)
            local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else
            local_lo += err;
    }

    /* Merge local double-double into global Zhi/Zlo */
    if(is_first_chunk) {
        Zhi[idx] = local_hi;
        Zlo[idx] = local_lo;
    } else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zhi[idx] = s_hi;
        Zlo[idx] += err + local_lo;
    }
}

/* =========================================================================
 * GPU kernels — Parts 3+4: fused last-chunk CRT accumulation + finalize
 *
 * oz2_accum_finalize_kernel is dispatched for the LAST GEMM chunk only.
 * It fuses the CRT accumulation and the per-element inverse-scale step into
 * a single pass, eliminating an intermediate Zhi/Zlo global write+read.
 *
 * When is_first_chunk is also true (single chunk = most common case),
 * Zhi_in/Zlo_in are never read and the C32i slices flow directly to D
 * without any Zhi/Zlo traffic.
 * ========================================================================= */
template <bool HAS_LO>
__global__ static void
oz2_accum_finalize_kernel_rt(const int32_t* __restrict__ C32i_batch,
                          const double*  __restrict__ Zhi_in,  /* prev hi; ignored if is_first */
                          const double*  __restrict__ Zlo_in,  /* prev lo; ignored if is_first */
                          const double*  __restrict__ C,
                          double*        __restrict__ D,
                          int64_t m, int64_t n, size_t ldc32i,
                          int64_t ldc, int64_t ldd,
                          double alpha, double beta,
                          const int16_t* __restrict__ sftA,
                          const int16_t* __restrict__ sftB,
                          unsigned chunk_start, unsigned chunk_size,
                          bool is_first_chunk)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;

    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);

    /* Phase 1: accumulate this chunk's C32i slices in registers. */
    double local_hi = 0.0;
    double local_lo = 0.0;

    for(unsigned t_local = 0; t_local < chunk_size; ++t_local) {
        const unsigned t     = chunk_start + t_local;
        const double dc_raw  = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc      = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi      = dc * cQpiHi[t];
        const double new_hi  = local_hi + hi;
        const double err     = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO)
            local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else
            local_lo += err;
    }

    /* Phase 2: merge with previous Zhi/Zlo if this is not the first chunk. */
    double Zh, Zl;
    if(is_first_chunk) {
        Zh = local_hi;
        Zl = local_lo;
    } else {
        const double old_hi = Zhi_in[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zh = s_hi;
        Zl = Zlo_in[idx] + err + local_lo;
    }

    /* Phase 3: range reduction + inverse scale — write directly to D. */
    const double q = rint((Zh + Zl) * cInvP);
    const double X = fma(cP_lo, q, fma(cP_hi, q, Zh) + Zl);

    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
    D[d_idx] = alpha * ldexp(X, inv_sft) + beta * C[c_idx];
}

/* Template versions: CHUNK_SIZE known at compile time enables #pragma unroll,
 * which lets the compiler pre-issue all CHUNK_SIZE C32i_batch loads
 * simultaneously, hiding their ~100-cycle HBM latencies instead of stalling
 * one load at a time.  IS_FIRST_CHUNK eliminates the runtime branch and,
 * for oz2_chunk_accum_kernel, uses __builtin_nontemporal_store for the
 * write-only Zhi/Zlo path (bypasses L2, keeps cache warm for C32i reads). */
template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
__global__ static void
oz2_chunk_accum_kernel(const int32_t* __restrict__ C32i_batch,
                       double*        __restrict__ Zhi,
                       double*        __restrict__ Zlo,
                       int64_t                     m,
                       int64_t                     n,
                       size_t                      ldc32i,
                       unsigned                    chunk_start)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;

    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);

    double local_hi = 0.0;
    double local_lo = 0.0;

    #pragma unroll
    for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local) {
        const unsigned t     = chunk_start + t_local;
        const double dc_raw  = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc      = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi      = dc * cQpiHi[t];
        const double new_hi  = local_hi + hi;
        const double err     = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO)
            local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else
            local_lo += err;
    }

    if constexpr (IS_FIRST_CHUNK) {
        __builtin_nontemporal_store(local_hi, Zhi + idx);
        __builtin_nontemporal_store(local_lo, Zlo + idx);
    } else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zhi[idx] = s_hi;
        Zlo[idx] += err + local_lo;
    }
}

template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
__global__ static void
oz2_accum_finalize_kernel(const int32_t* __restrict__ C32i_batch,
                          const double*  __restrict__ Zhi_in,
                          const double*  __restrict__ Zlo_in,
                          const double*  __restrict__ C,
                          double*        __restrict__ D,
                          int64_t m, int64_t n, size_t ldc32i,
                          int64_t ldc, int64_t ldd,
                          double alpha, double beta,
                          const int16_t* __restrict__ sftA,
                          const int16_t* __restrict__ sftB,
                          unsigned chunk_start)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;

    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);

    double local_hi = 0.0;
    double local_lo = 0.0;

    #pragma unroll
    for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local) {
        const unsigned t     = chunk_start + t_local;
        const double dc_raw  = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc      = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi      = dc * cQpiHi[t];
        const double new_hi  = local_hi + hi;
        const double err     = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO)
            local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else
            local_lo += err;
    }

    double Zh, Zl;
    if constexpr (IS_FIRST_CHUNK) {
        Zh = local_hi;
        Zl = local_lo;
    } else {
        const double old_hi = Zhi_in[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zh = s_hi;
        Zl = Zlo_in[idx] + err + local_lo;
    }

    const double q = rint((Zh + Zl) * cInvP);
    const double X = fma(cP_lo, q, fma(cP_hi, q, Zh) + Zl);

    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
    D[d_idx] = alpha * ldexp(X, inv_sft) + beta * C[c_idx];
}

/* =========================================================================
 * Optional per-call profiling — activated by setting
 *   HIPBLASLT_EMULATION_PROFILE=/path/to/output.csv
 * One CSV row is appended per fp64EmulatedGemm call.  When the env var is
 * not set the only overhead is a single const-pointer check — effectively
 * free for GPU-bound workloads.
 * ========================================================================= */
static const char* oz2_profile_file()
{
    static const char* const fn = std::getenv("HIPBLASLT_EMULATION_PROFILE");
    return fn;
}

/* =========================================================================
 * fp64EmulatedGemm  (OS II accurate mode, variable number of moduli)
 * ========================================================================= */
rocblaslt_status fp64EmulatedGemm(hipblasOperation_t           opA,
                                  hipblasOperation_t           opB,
                                  int64_t                      m,
                                  int64_t                      n,
                                  int64_t                      k,
                                  const double*                alpha,
                                  const double*                A,
                                  int64_t                      lda,
                                  const double*                B,
                                  int64_t                      ldb,
                                  const double*                beta,
                                  const double*                C,
                                  int64_t                      ldc,
                                  double*                      D,
                                  int64_t                      ldd,
                                  hipStream_t                  stream,
                                  const Fp64EmulationSettings& settings)
{
    /* Resolve settings: handle overrides first, then env vars. */
    const unsigned num_moduli = (settings.num_moduli >= 2u && settings.num_moduli <= OZ2_S_MAX)
                                    ? settings.num_moduli
                                    : fp64EmulationNumModuli();

    if(oz2_init_constants(num_moduli) != hipSuccess)
        return rocblaslt_status_internal_error;

    /* ------------------------------------------------------------------
     * Optional profiling — zero overhead when HIPBLASLT_EMULATION_PROFILE
     * is not set (a single nullptr check, perfectly predicted branch).
     * ------------------------------------------------------------------ */
    const char* const _pf   = oz2_profile_file();
    const bool        _prof = (_pf != nullptr);
    hipEvent_t _ev0{}, _ev1{}, _ev_tot{};
    float _t_prelim = 0, _t_prelim_gemm = 0, _t_refine = 0,
          _t_scale  = 0, _t_int8 = 0, _t_accum = 0, _t_finalize = 0, _t_total = 0;
    if(_prof) {
        (void)hipEventCreate(&_ev0);
        (void)hipEventCreate(&_ev1);
        (void)hipEventCreate(&_ev_tot);
    }
    auto _pstart = [&]() noexcept { if(_prof) (void)hipEventRecord(_ev0, stream); };
    auto _pstop  = [&](float& t) noexcept {
        if(_prof) {
            (void)hipEventRecord(_ev1, stream);
            (void)hipStreamSynchronize(stream);
            float ms = 0.f;
            (void)hipEventElapsedTime(&ms, _ev0, _ev1);
            t += ms;
        }
    };

    /* ------------------------------------------------------------------
     * Chunk size: number of consecutive moduli batched into one
     * oz2_chunk_accum_kernel call.  Reduces non-GEMM kernel launches from
     * (s+1) to (ceil(s/chunk_size)+1) while keeping C32i batch ≤ 256 MiB.
     * ------------------------------------------------------------------ */
    const unsigned chunk_size       = oz2_compute_chunk_size(m, n, num_moduli);
    const unsigned scale_chunk_size = oz2_compute_scale_chunk_size(m, n, k, num_moduli, chunk_size);

    /* ------------------------------------------------------------------
     * Padding / stride constants — needed both for the INT8 workspace
     * query below and for the workspace layout that follows.
     * ------------------------------------------------------------------ */
    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = oz2_pad(static_cast<size_t>(n));
    const size_t szC32i = ldc32i * static_cast<size_t>(n);  /* one INT32 slice */

    /* ------------------------------------------------------------------
     * Workspace layout
     *
     *   lda8i      = padding(k)           — INT8 leading dim for A slices
     *   cola8i     = padding(m)           — INT8 col count for A slices
     *   ldb8i      = lda8i               — INT8 leading dim for B slices
     *   ldc32i     = cola8i              — INT32 leading dim for GEMM output
     *   padn       = padding(n)          — padded n for sftB alignment
     *   chunk_size = oz2_compute_chunk_size(m,n,s) — number of C32i slices
     *
     *   A8i        [scale_chunk_size × lda8i × cola8i]  INT8  (pos 0 = A8i_high)
     *   B8i        [scale_chunk_size × ldb8i × n]       INT8  (pos 0 = B8i_high)
     *   C32i_batch [chunk_size × ldc32i × n]            INT32 (batch, re-used each GEMM chunk)
     *   Zhi        [ldc32i × n]                   FP64  CRT accumulator hi
     *   Zlo        [ldc32i × n]                   FP64  CRT accumulator lo
     *   sftA       [cola8i]                        INT16 per-row shifts for A
     *   sftB       [padn]                          INT16 per-col shifts for B
     *   nan_flag   [1]                             UINT32 Inf/NaN detection
     *   row_max    [cola8i]                        INT32  partial per-row max (sftA refine)
     * ------------------------------------------------------------------ */
    const size_t szA8i    = scale_chunk_size * lda8i * cola8i;
    const size_t szB8i    = scale_chunk_size * ldb8i * static_cast<size_t>(n);
    // szC32i already computed above
    const size_t szZhi    = szC32i;
    const size_t szZlo    = szC32i;
    const size_t szSftA   = cola8i;
    const size_t szSftB   = padn;
    const size_t szNanFlag = 1;
    const size_t szRowMax  = cola8i;  /* partial per-row max for sftA refinement */

    const size_t wsBytes =
          szA8i             * sizeof(int8_t)
        + szB8i             * sizeof(int8_t)
        + chunk_size * szC32i * sizeof(int32_t)   /* C32i batch */
        + szZhi             * sizeof(double)
        + szZlo             * sizeof(double)
        + szSftA            * sizeof(int16_t)
        + szSftB            * sizeof(int16_t)
        + szNanFlag         * sizeof(uint32_t)
        + szRowMax          * sizeof(int32_t);    /* row_max    */

    /* Use caller-provided workspace if large enough; otherwise allocate. */
    bool   ws_owned = false;
    char*  ws       = nullptr;
    if(settings.workspace != nullptr && settings.workspace_bytes >= wsBytes) {
        ws = static_cast<char*>(settings.workspace);
    } else {
        ws_owned = true;
        if(hipMallocAsync(&ws, wsBytes, stream) != hipSuccess)
            return rocblaslt_status_memory_error;
    }

    int8_t*   const A8i        = reinterpret_cast<int8_t*>(ws);
    int8_t*   const B8i        = A8i + szA8i;
    int32_t*  const C32i_batch = reinterpret_cast<int32_t*>(B8i + szB8i);  /* chunk_size slices */
    double*   const Zhi        = reinterpret_cast<double*>(C32i_batch + chunk_size * szC32i);
    double*   const Zlo        = Zhi + szZhi;
    int16_t*  const sftA       = reinterpret_cast<int16_t*>(Zlo + szZlo);
    int16_t*  const sftB       = sftA + szSftA;
    uint32_t* const nan_flag   = reinterpret_cast<uint32_t*>(sftB + szSftB);
    int32_t*  const row_max    = reinterpret_cast<int32_t*>(nan_flag + szNanFlag);
    /* First C32i slice (used by preliminary GEMM and shift refinement) */
    int32_t*  const C32i       = C32i_batch;

    if(_prof) (void)hipEventRecord(_ev_tot, stream);
    if(hipMemsetAsync(nan_flag, 0, sizeof(uint32_t), stream) != hipSuccess) {
        (void)hipFreeAsync(ws, stream);
        return rocblaslt_status_internal_error;
    }

    int8_t* const A8i_high = A8i;
    int8_t* const B8i_high = B8i;

    const bool tA = (opA != HIPBLAS_OP_N);
    const bool tB = (opB != HIPBLAS_OP_N);

    /* hipBLASLt matmul descriptors */
    hipblasLtMatrixLayout_t layoutA  = nullptr;
    hipblasLtMatrixLayout_t layoutB  = nullptr;
    hipblasLtMatrixLayout_t layoutCD = nullptr;
    hipblasLtMatmulDesc_t   matmulDesc = nullptr;

    hipblasLtMatrixLayoutCreate(&layoutA,  HIP_R_8I,
                                static_cast<uint64_t>(k), static_cast<uint64_t>(m),
                                static_cast<int64_t>(lda8i));
    hipblasLtMatrixLayoutCreate(&layoutB,  HIP_R_8I,
                                static_cast<uint64_t>(k), static_cast<uint64_t>(n),
                                static_cast<int64_t>(ldb8i));
    hipblasLtMatrixLayoutCreate(&layoutCD, HIP_R_32I,
                                static_cast<uint64_t>(m), static_cast<uint64_t>(n),
                                static_cast<int64_t>(ldc32i));

    hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32I, HIP_R_32I);
    {
        hipblasOperation_t opT = HIPBLAS_OP_T;
        hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                        &opN, sizeof(opN));
    }

    const int32_t one_i  = 1;
    const int32_t zero_i = 0;

    const unsigned m_blks = static_cast<unsigned>(m);
    _pstart();
    hipLaunchKernelGGL(oz2_accu_prelim_AB_kernel,
                       dim3(m_blks + static_cast<unsigned>(n)), dim3(256), 0, stream,
                       A, m, k, lda, tA, A8i_high, lda8i, sftA,
                       B, n, ldb, tB, B8i_high, ldb8i, sftB,
                       nan_flag, m_blks);
    _pstop(_t_prelim);

    /* Inf/NaN check: use setting if not sentinel, else env var */
    const uint32_t svmask = (settings.sv_mask != ~0u)
                                ? settings.sv_mask
                                : fp64EmulationSpecialValuesMask();
    if(svmask != 0u) {
        if(hipStreamSynchronize(stream) != hipSuccess) {
            (void)hipFreeAsync(ws, stream);
            return rocblaslt_status_internal_error;
        }
        uint32_t detected = 0u;
        if(hipMemcpy(&detected, nan_flag, sizeof(uint32_t), hipMemcpyDeviceToHost)
               != hipSuccess) {
            (void)hipFreeAsync(ws, stream);
            return rocblaslt_status_internal_error;
        }
        if(detected & svmask) {
            if(ws_owned) (void)hipFreeAsync(ws, stream);
            return rocblaslt_status_invalid_value;
        }
    }

    _pstart();
    hipblasLtMatmul(settings.handle, matmulDesc,
                    &one_i, A8i_high, layoutA, B8i_high, layoutB,
                    &zero_i, C32i, layoutCD, C32i, layoutCD,
                    nullptr, nullptr, 0, stream);
    _pstop(_t_prelim_gemm);

    const float accu_log2P = h_accu_log2P_all[num_moduli - 2];
    _pstart();
    /* sftA refinement — two-pass coalesced approach:
     *   Pass 1: tile C32i into 64-row × 64-column blocks; each wavefront reads
     *   64 consecutive rows per column (coalesced), accumulates per-row maxima
     *   into row_max via atomicMax (workspace-resident, no extra allocation).
     *   Pass 2: computes sftA[row] update from row_max.                     */
    const unsigned sftA_m_blks = static_cast<unsigned>((m + 63) / 64);
    const unsigned sftA_n_blks = static_cast<unsigned>((n + 63) / 64);
    (void)hipMemsetAsync(row_max, 0, szRowMax * sizeof(int32_t), stream);
    hipLaunchKernelGGL(oz2_refine_sftA_partial_kernel,
                       dim3(sftA_m_blks, sftA_n_blks), dim3(64), 0, stream,
                       C32i, m, n, ldc32i, row_max);
    hipLaunchKernelGGL(oz2_refine_sftA_apply_kernel,
                       dim3(sftA_m_blks), dim3(64), 0, stream,
                       row_max, sftA, m, accu_log2P);
    /* sftB refinement — one block per column, coalesced (unchanged). */
    hipLaunchKernelGGL(oz2_refine_sftB_kernel,
                       dim3(static_cast<unsigned>(n)), dim3(256), 0, stream,
                       C32i, m, n, ldc32i, sftB, accu_log2P);
    _pstop(_t_refine);

    /* Parts 1f + 2a-d: two-level loop — scale chunks (outer) × GEMM chunks (inner).
     *
     * Outer loop (scale chunks of size scale_chunk_size):
     *   oz2_scaleAB_kernel reads A and B ONCE and writes scale_chunk_size INT8
     *   slices into A8i[0..actual_scale-1] / B8i[0..actual_scale-1].
     *
     * Inner loop (GEMM chunks of size chunk_size within one scale chunk):
     *   A strided-batch INT8 GEMM over chunk_size slices (A8i_gemm / B8i_gemm
     *   point into the pre-computed scale buffer at offset gemm_local).
     *
     * Scale kernel: Block (64,4), threadIdx.x=k-index (fast) for both A and B.
     * A8i/B8i writes are always coalesced.  For transA=N the FP64 reads from A
     * have stride=lda (non-coalesced), but keeping INT8 writes coalesced is the
     * bandwidth-optimal trade-off.  The A and B branches are fused into a single
     * kernel dispatch using a combined y-dimension grid.                      */
    const dim3 blk_scale(64u, 4u);
    const unsigned m_y_blks = static_cast<unsigned>((m + 3) / 4);
    const unsigned n_y_blks = static_cast<unsigned>((n + 3) / 4);
    const dim3 gAB_scale(static_cast<unsigned>((k + 63) / 64),
                         m_y_blks + n_y_blks);

    const bool do_pass3 = (num_moduli >= 19u);

    /* Parts 2a-d (continued): chunk loop configuration.
     * Block (64,8) = 512 threads = 8 wavefronts on MI300/MI350.  Each 64-thread
     * wavefront covers 64 consecutive m-elements (one full row of C32i/Zhi/Zlo
     * in the column-major layout), yielding two clean 256-byte HBM transactions
     * per wavefront per slice — more efficient than (32,16) which spans two
     * l-rows and touches separate cache-line groups per warp.               */
    const dim3 blk_acc(64, 8);
    const dim3 grid_acc((m + 63) / 64, (n + 7) / 8);
    /* strideA8i / strideB8i: stride between consecutive moduli slices within
     * the scale buffer (in INT8 elements).  Used by the scale kernels to
     * write the t_local-th slice, and by the inner GEMM loop to compute
     * A8i_gemm / B8i_gemm offsets into the pre-computed scale chunk.       */
    const size_t strideA8i = lda8i * cola8i;
    const size_t strideB8i = ldb8i * static_cast<size_t>(n);

    /* Create batch layout objects once before the outer scale loop.
     * BATCH_COUNT is updated only when actual_gemm changes (i.e. the last
     * partial inner GEMM chunk, or when the next outer iteration resets to
     * chunk_size).  The three Create/Destroy pairs are avoided for all but
     * the final partial GEMM batch.                                         */
    hipblasLtMatrixLayout_t layoutA_b  = nullptr;
    hipblasLtMatrixLayout_t layoutB_b  = nullptr;
    hipblasLtMatrixLayout_t layoutCD_b = nullptr;

    hipblasLtMatrixLayoutCreate(&layoutA_b,  HIP_R_8I,
                                static_cast<uint64_t>(k), static_cast<uint64_t>(m),
                                static_cast<int64_t>(lda8i));
    hipblasLtMatrixLayoutCreate(&layoutB_b,  HIP_R_8I,
                                static_cast<uint64_t>(k), static_cast<uint64_t>(n),
                                static_cast<int64_t>(ldb8i));
    hipblasLtMatrixLayoutCreate(&layoutCD_b, HIP_R_32I,
                                static_cast<uint64_t>(m), static_cast<uint64_t>(n),
                                static_cast<int64_t>(ldc32i));

    /* HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET is in elements, not bytes. */
    int32_t       batch_cur  = static_cast<int32_t>(chunk_size);
    const int64_t stride_A_b = static_cast<int64_t>(strideA8i);  /* int8 elements  */
    const int64_t stride_B_b = static_cast<int64_t>(strideB8i);  /* int8 elements  */
    const int64_t stride_C_b = static_cast<int64_t>(szC32i);     /* int32 elements */

    hipblasLtMatrixLayoutSetAttribute(layoutA_b,
        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &batch_cur,  sizeof(batch_cur));
    hipblasLtMatrixLayoutSetAttribute(layoutA_b,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_A_b, sizeof(stride_A_b));
    hipblasLtMatrixLayoutSetAttribute(layoutB_b,
        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &batch_cur,  sizeof(batch_cur));
    hipblasLtMatrixLayoutSetAttribute(layoutB_b,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_B_b, sizeof(stride_B_b));
    hipblasLtMatrixLayoutSetAttribute(layoutCD_b,
        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &batch_cur,  sizeof(batch_cur));
    hipblasLtMatrixLayoutSetAttribute(layoutCD_b,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_C_b, sizeof(stride_C_b));

    for(unsigned scale_start = 0; scale_start < num_moduli; scale_start += scale_chunk_size) {
        const unsigned actual_scale = (scale_start + scale_chunk_size <= num_moduli)
                                      ? scale_chunk_size : (num_moduli - scale_start);

        /* Part 1f: extract INT8 slices for all actual_scale moduli in this
         * scale chunk.  A and B are each read from HBM exactly once here,
         * writing scale_chunk_size (or fewer) slices into A8i and B8i.
         * When scale_chunk_size == num_moduli A and B are read once in total;
         * otherwise ceil(num_moduli / scale_chunk_size) reads of each.      */
        _pstart();
        /* Dispatch oz2_scaleAB_kernel<T_COUNT, false>: with T_COUNT fixed at
         * compile time, #pragma unroll fully unrolls the inner moduli loop and
         * the compiler can issue all T_COUNT independent FP64 chains in parallel.
         * The runtime fallback (oz2_scaleAB_kernel_rt) handles the rare
         * do_pass3=true case (num_moduli >= 19).                             */
#define OZ2_SCALE_LAUNCH(TC) \
        hipLaunchKernelGGL((oz2_scaleAB_kernel<(TC), false>), gAB_scale, blk_scale, 0, stream, \
                           A, m, lda, tA, A8i, lda8i, cola8i, sftA, \
                           B, n, ldb, tB, B8i, ldb8i, sftB, k, scale_start, m_y_blks)
        if(!do_pass3) {
            switch(actual_scale) {
                case  1: OZ2_SCALE_LAUNCH( 1); break;
                case  2: OZ2_SCALE_LAUNCH( 2); break;
                case  3: OZ2_SCALE_LAUNCH( 3); break;
                case  4: OZ2_SCALE_LAUNCH( 4); break;
                case  5: OZ2_SCALE_LAUNCH( 5); break;
                case  6: OZ2_SCALE_LAUNCH( 6); break;
                case  7: OZ2_SCALE_LAUNCH( 7); break;
                case  8: OZ2_SCALE_LAUNCH( 8); break;
                case  9: OZ2_SCALE_LAUNCH( 9); break;
                case 10: OZ2_SCALE_LAUNCH(10); break;
                case 11: OZ2_SCALE_LAUNCH(11); break;
                case 12: OZ2_SCALE_LAUNCH(12); break;
                case 13: OZ2_SCALE_LAUNCH(13); break;
                case 14: OZ2_SCALE_LAUNCH(14); break;
                case 15: OZ2_SCALE_LAUNCH(15); break;
                case 16: OZ2_SCALE_LAUNCH(16); break;
                case 17: OZ2_SCALE_LAUNCH(17); break;
                case 18: OZ2_SCALE_LAUNCH(18); break;
                default:
                    hipLaunchKernelGGL(oz2_scaleAB_kernel_rt, gAB_scale, blk_scale, 0, stream,
                                       A, m, lda, tA, A8i, lda8i, cola8i, sftA,
                                       B, n, ldb, tB, B8i, ldb8i, sftB,
                                       k, scale_start, actual_scale, false, m_y_blks);
            }
        } else {
            /* num_moduli >= 19 (do_pass3=true): use runtime fallback kernel. */
            hipLaunchKernelGGL(oz2_scaleAB_kernel_rt, gAB_scale, blk_scale, 0, stream,
                               A, m, lda, tA, A8i, lda8i, cola8i, sftA,
                               B, n, ldb, tB, B8i, ldb8i, sftB,
                               k, scale_start, actual_scale, true, m_y_blks);
        }
#undef OZ2_SCALE_LAUNCH
        _pstop(_t_scale);

        /* Parts 2c-d: GEMM + accumulate for each GEMM chunk within this
         * scale chunk.  A8i_gemm / B8i_gemm are offset pointers into the
         * pre-computed scale-chunk buffers — A and B are NOT re-read here.  */
        for(unsigned gemm_local = 0; gemm_local < actual_scale; gemm_local += chunk_size) {
            const unsigned actual_gemm = (gemm_local + chunk_size <= actual_scale)
                                         ? chunk_size : (actual_scale - gemm_local);

            /* Update batch count only when the GEMM batch size changes. */
            if(static_cast<int32_t>(actual_gemm) != batch_cur) {
                batch_cur = static_cast<int32_t>(actual_gemm);
                hipblasLtMatrixLayoutSetAttribute(layoutA_b,
                    HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
                hipblasLtMatrixLayoutSetAttribute(layoutB_b,
                    HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
                hipblasLtMatrixLayoutSetAttribute(layoutCD_b,
                    HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
            }

            /* Offset into the pre-computed scale chunk for this GEMM batch. */
            const int8_t* const A8i_gemm = A8i + gemm_local * strideA8i;
            const int8_t* const B8i_gemm = B8i + gemm_local * strideB8i;

            _pstart();
            hipblasLtMatmul(settings.handle, matmulDesc,
                            &one_i,
                            A8i_gemm, layoutA_b,
                            B8i_gemm, layoutB_b,
                            &zero_i,
                            C32i_batch, layoutCD_b, C32i_batch, layoutCD_b,
                            nullptr, nullptr, 0, stream);
            _pstop(_t_int8);

            const unsigned global_chunk_start = scale_start + gemm_local;
            const bool is_first = (global_chunk_start == 0);
            /* Detect the final chunk: when true, dispatch the fused accum+finalize
             * kernel that writes D directly and skips the Zhi/Zlo global write. */
            const bool is_last  = (global_chunk_start + actual_gemm == num_moduli);
            _pstart();
            /* Dispatch template kernels with compile-time CHUNK_SIZE and
             * IS_FIRST_CHUNK: #pragma unroll in the kernel body pre-issues all
             * CHUNK_SIZE C32i_batch loads simultaneously.
             * _rt fallbacks handle chunk sizes not in the switch table.      */
#define OZ2_FARGS C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd, *alpha, *beta, sftA, sftB, global_chunk_start
#define OZ2_AARGS C32i_batch, Zhi, Zlo, m, n, ldc32i, global_chunk_start
#define OZ2_FINALIZE(HL, CS) \
            do { if(is_first) hipLaunchKernelGGL((oz2_accum_finalize_kernel<(HL),(CS),true>),  grid_acc, blk_acc, 0, stream, OZ2_FARGS); \
                 else         hipLaunchKernelGGL((oz2_accum_finalize_kernel<(HL),(CS),false>), grid_acc, blk_acc, 0, stream, OZ2_FARGS); } while(0)
#define OZ2_ACCUM(HL, CS) \
            do { if(is_first) hipLaunchKernelGGL((oz2_chunk_accum_kernel<(HL),(CS),true>),  grid_acc, blk_acc, 0, stream, OZ2_AARGS); \
                 else         hipLaunchKernelGGL((oz2_chunk_accum_kernel<(HL),(CS),false>), grid_acc, blk_acc, 0, stream, OZ2_AARGS); } while(0)
            if(is_last) {
                if(num_moduli <= 7u) {
                    switch(actual_gemm) {
                        case  1: OZ2_FINALIZE(false,  1); break;
                        case  2: OZ2_FINALIZE(false,  2); break;
                        case  3: OZ2_FINALIZE(false,  3); break;
                        case  4: OZ2_FINALIZE(false,  4); break;
                        case  5: OZ2_FINALIZE(false,  5); break;
                        case  6: OZ2_FINALIZE(false,  6); break;
                        case  7: OZ2_FINALIZE(false,  7); break;
                        case  8: OZ2_FINALIZE(false,  8); break;
                        case  9: OZ2_FINALIZE(false,  9); break;
                        case 10: OZ2_FINALIZE(false, 10); break;
                        case 11: OZ2_FINALIZE(false, 11); break;
                        case 12: OZ2_FINALIZE(false, 12); break;
                        case 13: OZ2_FINALIZE(false, 13); break;
                        case 14: OZ2_FINALIZE(false, 14); break;
                        case 15: OZ2_FINALIZE(false, 15); break;
                        case 16: OZ2_FINALIZE(false, 16); break;
                        case 17: OZ2_FINALIZE(false, 17); break;
                        case 18: OZ2_FINALIZE(false, 18); break;
                        case 19: OZ2_FINALIZE(false, 19); break;
                        case 20: OZ2_FINALIZE(false, 20); break;
                        default: hipLaunchKernelGGL((oz2_accum_finalize_kernel_rt<false>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd,
                                     *alpha, *beta, sftA, sftB, global_chunk_start, actual_gemm, is_first);
                    }
                } else {
                    switch(actual_gemm) {
                        case  1: OZ2_FINALIZE(true,  1); break;
                        case  2: OZ2_FINALIZE(true,  2); break;
                        case  3: OZ2_FINALIZE(true,  3); break;
                        case  4: OZ2_FINALIZE(true,  4); break;
                        case  5: OZ2_FINALIZE(true,  5); break;
                        case  6: OZ2_FINALIZE(true,  6); break;
                        case  7: OZ2_FINALIZE(true,  7); break;
                        case  8: OZ2_FINALIZE(true,  8); break;
                        case  9: OZ2_FINALIZE(true,  9); break;
                        case 10: OZ2_FINALIZE(true, 10); break;
                        case 11: OZ2_FINALIZE(true, 11); break;
                        case 12: OZ2_FINALIZE(true, 12); break;
                        case 13: OZ2_FINALIZE(true, 13); break;
                        case 14: OZ2_FINALIZE(true, 14); break;
                        case 15: OZ2_FINALIZE(true, 15); break;
                        case 16: OZ2_FINALIZE(true, 16); break;
                        case 17: OZ2_FINALIZE(true, 17); break;
                        case 18: OZ2_FINALIZE(true, 18); break;
                        case 19: OZ2_FINALIZE(true, 19); break;
                        case 20: OZ2_FINALIZE(true, 20); break;
                        default: hipLaunchKernelGGL((oz2_accum_finalize_kernel_rt<true>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd,
                                     *alpha, *beta, sftA, sftB, global_chunk_start, actual_gemm, is_first);
                    }
                }
            } else {
                if(num_moduli <= 7u) {
                    switch(actual_gemm) {
                        case  1: OZ2_ACCUM(false,  1); break;
                        case  2: OZ2_ACCUM(false,  2); break;
                        case  3: OZ2_ACCUM(false,  3); break;
                        case  4: OZ2_ACCUM(false,  4); break;
                        case  5: OZ2_ACCUM(false,  5); break;
                        case  6: OZ2_ACCUM(false,  6); break;
                        case  7: OZ2_ACCUM(false,  7); break;
                        case  8: OZ2_ACCUM(false,  8); break;
                        case  9: OZ2_ACCUM(false,  9); break;
                        case 10: OZ2_ACCUM(false, 10); break;
                        case 11: OZ2_ACCUM(false, 11); break;
                        case 12: OZ2_ACCUM(false, 12); break;
                        case 13: OZ2_ACCUM(false, 13); break;
                        case 14: OZ2_ACCUM(false, 14); break;
                        case 15: OZ2_ACCUM(false, 15); break;
                        case 16: OZ2_ACCUM(false, 16); break;
                        case 17: OZ2_ACCUM(false, 17); break;
                        case 18: OZ2_ACCUM(false, 18); break;
                        case 19: OZ2_ACCUM(false, 19); break;
                        case 20: OZ2_ACCUM(false, 20); break;
                        default: hipLaunchKernelGGL((oz2_chunk_accum_kernel_rt<false>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, m, n, ldc32i,
                                     global_chunk_start, actual_gemm, is_first);
                    }
                } else {
                    switch(actual_gemm) {
                        case  1: OZ2_ACCUM(true,  1); break;
                        case  2: OZ2_ACCUM(true,  2); break;
                        case  3: OZ2_ACCUM(true,  3); break;
                        case  4: OZ2_ACCUM(true,  4); break;
                        case  5: OZ2_ACCUM(true,  5); break;
                        case  6: OZ2_ACCUM(true,  6); break;
                        case  7: OZ2_ACCUM(true,  7); break;
                        case  8: OZ2_ACCUM(true,  8); break;
                        case  9: OZ2_ACCUM(true,  9); break;
                        case 10: OZ2_ACCUM(true, 10); break;
                        case 11: OZ2_ACCUM(true, 11); break;
                        case 12: OZ2_ACCUM(true, 12); break;
                        case 13: OZ2_ACCUM(true, 13); break;
                        case 14: OZ2_ACCUM(true, 14); break;
                        case 15: OZ2_ACCUM(true, 15); break;
                        case 16: OZ2_ACCUM(true, 16); break;
                        case 17: OZ2_ACCUM(true, 17); break;
                        case 18: OZ2_ACCUM(true, 18); break;
                        case 19: OZ2_ACCUM(true, 19); break;
                        case 20: OZ2_ACCUM(true, 20); break;
                        default: hipLaunchKernelGGL((oz2_chunk_accum_kernel_rt<true>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, m, n, ldc32i,
                                     global_chunk_start, actual_gemm, is_first);
                    }
                }
            }
#undef OZ2_FARGS
#undef OZ2_AARGS
#undef OZ2_FINALIZE
#undef OZ2_ACCUM
            _pstop(_t_accum);
        }
    }

    hipblasLtMatrixLayoutDestroy(layoutCD_b);
    hipblasLtMatrixLayoutDestroy(layoutB_b);
    hipblasLtMatrixLayoutDestroy(layoutA_b);

    /* The last GEMM chunk is handled by oz2_accum_finalize_kernel which writes
     * D directly; _t_finalize remains 0 (cost is included in _t_accum).    */

    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(layoutCD);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutA);

    if(ws_owned) {
        if(hipFreeAsync(ws, stream) != hipSuccess)
            return rocblaslt_status_internal_error;
    }

    if(_prof) {
        (void)hipEventRecord(_ev1, stream);
        (void)hipStreamSynchronize(stream);
        (void)hipEventElapsedTime(&_t_total, _ev_tot, _ev1);
        std::FILE* _f = std::fopen(_pf, "a");
        if(_f) {
            if(std::ftell(_f) == 0)
            std::fprintf(_f,
                "m,n,k,num_moduli,scale_chunk_size,gemm_chunk_size,"
                "workspace_bytes,"
                "t_prelim_ms,t_prelim_gemm_ms,t_refine_ms,"
                "t_scale_ms,t_int8_gemm_ms,t_accum_ms,"
                "t_finalize_ms,t_total_ms\n");
            std::fprintf(_f,
                "%lld,%lld,%lld,%u,%u,%u,"
                "%llu,"
                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                (long long)m, (long long)n, (long long)k,
                num_moduli, scale_chunk_size, chunk_size,
                (unsigned long long)wsBytes,
                _t_prelim, _t_prelim_gemm, _t_refine,
                _t_scale, _t_int8, _t_accum, _t_finalize, _t_total);
            std::fclose(_f);
        }
        (void)hipEventDestroy(_ev_tot);
        (void)hipEventDestroy(_ev1);
        (void)hipEventDestroy(_ev0);
    }
    return rocblaslt_status_success;
}
