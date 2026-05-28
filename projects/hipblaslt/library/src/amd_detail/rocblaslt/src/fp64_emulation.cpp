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
 * HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT (default: s=14, ~110 bits of CRT
 * capacity, sufficient for guaranteed FP64-equivalent results on HPL-like inputs).
 *
 * Constants (tables) are taken verbatim from the open-source GEMMul8 implementation
 * (Y. Uchino, RIKEN R-CCS, https://github.com/RIKEN-RCCS/GEMMul8).
 *
 * This file MUST be compiled as HIP (LANGUAGE HIP in CMakeLists.txt).
 * Inner INT8 GEMMs use hipblasLtMatmul (INT8 tensor cores, INT32 accumulate).
 */

#include "fp64_emulation.hpp"

#include "hipblaslt/hipblaslt.h"
#include <hip/hip_runtime.h>

#include <cstdlib>   // std::getenv
#include <cstring>   // std::strcmp
#include <cmath>     // std::log2, std::floor, etc.

/* =========================================================================
 * Tuning constants
 * ========================================================================= */
static constexpr double FP64_EMUL_AI_THRESHOLD = 32.0;

/* Maximum number of moduli supported (s = 2..OZ2_S_MAX).
 * Constant memory and table arrays are sized for the maximum. */
static constexpr unsigned OZ2_S_MAX = 14;

/* Alignment for INT8 arrays (128 bytes = 128 INT8 elements) */
static constexpr size_t OZ2_ALIGN = 128;

static __host__ __device__ size_t oz2_pad(size_t n)
{
    return (n + OZ2_ALIGN - 1) / OZ2_ALIGN * OZ2_ALIGN;
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
 *                  229, 227, 223, 217, 211, 199  (OZ2_S_MAX = 14 total)
 *
 * All arrays indexed by table_idx = s - 2  (s = number of moduli, 2..14).
 * ========================================================================= */

/* Per-modulus constants — same for all s, only first s entries used */
static const double h_neg_mod[OZ2_S_MAX] = {
    -256.0, -255.0, -253.0, -251.0, -247.0, -241.0, -239.0,
    -233.0, -229.0, -227.0, -223.0, -217.0, -211.0, -199.0
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
    0x1.49539e3b2d067p-8    /* 1/199 */
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
    0x1.49539ep-8F    /* 1/199  */
};

/* -M (high part) for s = 2..14 */
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
};
/* 1/M for s = 2..14 */
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
};
/* accu::log2P = log2(P-1)/2 - 0.5 for s = 2..14 (used for shift refinement) */
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
 * fp64EmulationWorkspaceSize
 * ========================================================================= */
size_t fp64EmulationWorkspaceSize(int64_t m, int64_t n, int64_t k, unsigned num_moduli)
{
    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = oz2_pad(static_cast<size_t>(n));

    return   num_moduli * lda8i * cola8i * sizeof(int8_t)
           + num_moduli * ldb8i * static_cast<size_t>(n) * sizeof(int8_t)
           + ldc32i * static_cast<size_t>(n) * sizeof(int32_t)
           + ldc32i * static_cast<size_t>(n) * sizeof(double) * 2   /* Zhi + Zlo */
           + cola8i * sizeof(int16_t)                               /* sftA */
           + padn   * sizeof(int16_t)                               /* sftB */
           + sizeof(uint32_t);                                       /* nan_flag */
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
 * Default (env var absent or 0): use all OZ2_S_MAX=14 moduli (~110 bits).
 *
 * Notable values (from design document §2.5):
 *   55 bits → s=7  ("fixed-mode default")
 *   79 bits → s=10 ("ADP max")
 */
unsigned fp64EmulationNumModuli()
{
    static const unsigned num_moduli = []() -> unsigned {
        const char* v = std::getenv("HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT");
        if(v == nullptr) return OZ2_S_MAX;

        const unsigned target = static_cast<unsigned>(std::strtoul(v, nullptr, 0));
        if(target == 0u) return OZ2_S_MAX;

        /* Cumulative log2 of the product of the first s moduli, for s=2..OZ2_S_MAX.
         * Derived from the exact moduli: 256, 255, 253, 251, 247, 241, 239, 233,
         * 229, 227, 223, 217, 211, 199. */
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
 * GPU kernels — accu mode Part 1: preliminary extraction
 * ========================================================================= */

/**
 * oz2_accu_prelim_A_kernel
 * Grid: (m, 1), Block: (128, 1)
 */
__global__ static void
oz2_accu_prelim_A_kernel(const double* __restrict__ A,
                         int64_t m, int64_t k, int64_t lda, bool transA,
                         int8_t*   __restrict__ A8i_high, size_t lda8i,
                         int16_t*  __restrict__ sftA,
                         uint32_t* __restrict__ nan_flag)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if(row >= m) return;

    double local_max = 0.0;
    for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
        double val = transA ? A[row * lda + j] : A[j * lda + row];
        if(!isfinite(val))
            (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
        double av = fabs(val);
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_d(local_max);

    __shared__ double s_wmax[4];
    local_max = block_reduce_max_d(local_max, s_wmax);

    __shared__ int16_t s_sft;
    if(threadIdx.x == 0) {
        if(local_max < 1e-300) local_max = 1.0;
        int sft = 5 - static_cast<int>(floor(log2(local_max)));
        if(sft < 0) sft = 0;
        s_sft     = static_cast<int16_t>(sft);
        sftA[row] = s_sft;
    }
    __syncthreads();
    const int16_t sft = s_sft;

    for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
        double val    = transA ? A[row * lda + j] : A[j * lda + row];
        /* Use fabs(val): the prelim GEMM computes Σ|A8|·|B8| ≥ |D|·2^(sftA+sftB),
         * which is a valid upper bound for the refinement (matches GEMMul8 T2int_8i). */
        double scaled = trunc(ldexp(fabs(val), static_cast<int>(sft)));
        A8i_high[static_cast<size_t>(j) + static_cast<size_t>(row) * lda8i] =
            static_cast<int8_t>(static_cast<int32_t>(scaled));
    }
}

/**
 * oz2_accu_prelim_B_kernel
 * Grid: (n, 1), Block: (128, 1)
 */
__global__ static void
oz2_accu_prelim_B_kernel(const double* __restrict__ B,
                         int64_t k, int64_t n, int64_t ldb, bool transB,
                         int8_t*   __restrict__ B8i_high, size_t ldb8i,
                         int16_t*  __restrict__ sftB,
                         uint32_t* __restrict__ nan_flag)
{
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if(col >= n) return;

    double local_max = 0.0;
    for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
        double val = transB ? B[col + j * ldb] : B[j + col * ldb];
        if(!isfinite(val))
            (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
        double av = fabs(val);
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_d(local_max);

    __shared__ double s_wmax[4];
    local_max = block_reduce_max_d(local_max, s_wmax);

    __shared__ int16_t s_sft;
    if(threadIdx.x == 0) {
        if(local_max < 1e-300) local_max = 1.0;
        int sft = 5 - static_cast<int>(floor(log2(local_max)));
        if(sft < 0) sft = 0;
        s_sft     = static_cast<int16_t>(sft);
        sftB[col] = s_sft;
    }
    __syncthreads();
    const int16_t sft = s_sft;

    for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
        double val    = transB ? B[col + j * ldb] : B[j + col * ldb];
        /* Same abs-value extraction as the A prelim kernel. */
        double scaled = trunc(ldexp(fabs(val), static_cast<int>(sft)));
        B8i_high[static_cast<size_t>(j) + static_cast<size_t>(col) * ldb8i] =
            static_cast<int8_t>(static_cast<int32_t>(scaled));
    }
}

/* =========================================================================
 * GPU kernels — accu mode Part 1: shift refinement from preliminary GEMM
 * ========================================================================= */

/**
 * oz2_accu_refine_sftA_kernel
 * Grid: (m, 1), Block: (128, 1)
 */
__global__ static void
oz2_accu_refine_sftA_kernel(const int32_t* __restrict__ C32i,
                             int64_t m, int64_t n, size_t ldc32i,
                             int16_t* __restrict__ sftA,
                             float log2P)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if(row >= m) return;

    int32_t local_max = 0;
    for(int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        int32_t v  = C32i[static_cast<size_t>(row) + static_cast<size_t>(j) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_i32(local_max);

    __shared__ int32_t s_wmax[4];
    local_max = block_reduce_max_i32(local_max, s_wmax);

    if(threadIdx.x == 0) {
        if(local_max < 1) local_max = 1;
        float refinement = floorf(-0.5f * log2f(static_cast<float>(local_max)) + log2P);
        sftA[row] += static_cast<int16_t>(refinement);
    }
}

/**
 * oz2_accu_refine_sftB_kernel
 * Grid: (n, 1), Block: (128, 1)
 */
__global__ static void
oz2_accu_refine_sftB_kernel(const int32_t* __restrict__ C32i,
                             int64_t m, int64_t n, size_t ldc32i,
                             int16_t* __restrict__ sftB,
                             float log2P)
{
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if(col >= n) return;

    int32_t local_max = 0;
    for(int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        int32_t v  = C32i[static_cast<size_t>(i) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_i32(local_max);

    __shared__ int32_t s_wmax[4];
    local_max = block_reduce_max_i32(local_max, s_wmax);

    if(threadIdx.x == 0) {
        if(local_max < 1) local_max = 1;
        float refinement = floorf(-0.5f * log2f(static_cast<float>(local_max)) + log2P);
        sftB[col] += static_cast<int16_t>(refinement);
    }
}

/* =========================================================================
 * GPU kernels — Part 1f: full multi-modulus scaling with per-row/col shifts
 * ========================================================================= */

/**
 * oz2_scaleA_kernel
 * Grid: ((m+15)/16, (k+15)/16), Block: (16,16)
 */
__global__ static void
oz2_scaleA_kernel(const double* __restrict__ A,
                  int64_t m, int64_t k, int64_t lda, bool transA,
                  int8_t*  __restrict__       A8i,
                  size_t                      lda8i,
                  size_t                      cola8i,
                  const int16_t* __restrict__ sftA,
                  unsigned                    num_moduli)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || j >= k) return;

    const double val  = transA ? A[i * lda + j] : A[j * lda + i];
    const int16_t sft = sftA[i];
    const double ival = trunc(ldexp(val, static_cast<int>(sft)));

    const size_t stride = lda8i * cola8i;
    const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(i) * lda8i;

    for(unsigned t = 0; t < num_moduli; ++t) {
        /* Pass 1 – double-precision FMA: r ≈ ival mod m_t, exact for t=0 (m=256),
         *          possibly off by ±m_t for t≥1 when sft_final ≥ ~53.          */
        const double  r   = fma(cNegMod[t], rint(ival * cInvMod[t]), ival);
        /* Pass 2 – float-precision refinement (GEMMul8 ITER=2):
         *          corrects any off-by-m_t error from pass 1.                  */
        const float   rf  = static_cast<float>(r);
        const float   rf2 = fmaf(rintf(rf * cInvModF[t]),
                                 static_cast<float>(cNegMod[t]), rf);
        A8i[t * stride + offset] = static_cast<int8_t>(static_cast<int32_t>(rf2));
    }
}

/**
 * oz2_scaleB_kernel
 * Grid: ((k+15)/16, (n+15)/16), Block: (16,16)
 */
__global__ static void
oz2_scaleB_kernel(const double* __restrict__ B,
                  int64_t k, int64_t n, int64_t ldb, bool transB,
                  int8_t*  __restrict__       B8i,
                  size_t                      ldb8i,
                  const int16_t* __restrict__ sftB,
                  unsigned                    num_moduli)
{
    const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(j >= k || l >= n) return;

    const double val  = transB ? B[j * ldb + l] : B[l * ldb + j];
    const int16_t sft = sftB[l];
    const double ival = trunc(ldexp(val, static_cast<int>(sft)));

    const size_t stride = ldb8i * static_cast<size_t>(n);
    const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(l) * ldb8i;

    for(unsigned t = 0; t < num_moduli; ++t) {
        /* Pass 1 – double-precision FMA (same logic as oz2_scaleA_kernel). */
        const double  r   = fma(cNegMod[t], rint(ival * cInvMod[t]), ival);
        /* Pass 2 – float-precision refinement. */
        const float   rf  = static_cast<float>(r);
        const float   rf2 = fmaf(rintf(rf * cInvModF[t]),
                                 static_cast<float>(cNegMod[t]), rf);
        B8i[t * stride + offset] = static_cast<int8_t>(static_cast<int32_t>(rf2));
    }
}

/* =========================================================================
 * GPU kernels — Part 2d: CRT accumulation
 * ========================================================================= */

__global__ static void
oz2_accum_kernel(const int32_t* __restrict__ C32i,
                 double*        __restrict__ Zhi,
                 double*        __restrict__ Zlo,
                 int64_t                     m,
                 int64_t                     n,
                 size_t                      ldc32i,
                 unsigned                    t)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;

    const size_t idx = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const double dc  = static_cast<double>(C32i[idx]);

    const double hi = dc * cQpiHi[t];
    const double lo = dc * cQpiLo[t];

    if(t == 0) {
        Zhi[idx] = hi;
        Zlo[idx] = lo;
    } else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + hi;
        const double err    = hi - (s_hi - old_hi);   /* standard 2Sum round-off */
        Zhi[idx] = s_hi;
        Zlo[idx] += err + lo;
    }
}

/* =========================================================================
 * GPU kernels — Parts 3+4: finalize with per-element inverse scale
 * ========================================================================= */

__global__ static void
oz2_finalize_kernel(const double* __restrict__ Zhi,
                    const double* __restrict__ Zlo,
                    const double* __restrict__ C,
                    double*       __restrict__ D,
                    int64_t                    m,
                    int64_t                    n,
                    size_t                     ldc32i,
                    int64_t                    ldc,
                    int64_t                    ldd,
                    double                     alpha,
                    double                     beta,
                    const int16_t* __restrict__ sftA,
                    const int16_t* __restrict__ sftB)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;

    const size_t z_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);

    const double Zh = Zhi[z_idx];
    const double Zl = Zlo[z_idx];
    /* Use the double-double (Zh, Zl) for q estimation, then apply the
     * GEMMul8-style nested FMA for range reduction:
     *   X = fma(P_lo, q, fma(P_hi, q, Zh) + Zl)
     * This avoids catastrophic cancellation in fma(P_hi, q, Zh) and
     * preserves the Zl contribution, giving a more accurate result
     * than the collapsed form  Z + cP_hi*q + cP_lo*q. */
    const double q  = rint((Zh + Zl) * cInvP);
    const double X  = fma(cP_lo, q, fma(cP_hi, q, Zh) + Zl);

    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    D[d_idx] = alpha * ldexp(X, inv_sft) + beta * C[c_idx];
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
     * Workspace layout
     *
     *   lda8i  = padding(k)        — INT8 leading dim for A slices
     *   cola8i = padding(m)        — INT8 col count for A slices
     *   ldb8i  = lda8i             — INT8 leading dim for B slices
     *   ldc32i = cola8i            — INT32 leading dim for GEMM output
     *   padn   = padding(n)        — padded n for sftB alignment
     *
     *   A8i  [num_moduli × lda8i × cola8i]  INT8  (pos 0 = A8i_high)
     *   B8i  [num_moduli × ldb8i × n]       INT8  (pos 0 = B8i_high)
     *   C32i [ldc32i × n]                   INT32 (reused)
     *   Zhi  [ldc32i × n]                   FP64  CRT accumulator hi
     *   Zlo  [ldc32i × n]                   FP64  CRT accumulator lo
     *   sftA [cola8i]                        INT16 per-row shifts for A
     *   sftB [padn]                          INT16 per-col shifts for B
     *   nan_flag [1]                         UINT32 Inf/NaN detection
     * ------------------------------------------------------------------ */
    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = oz2_pad(static_cast<size_t>(n));

    const size_t szA8i    = num_moduli * lda8i * cola8i;
    const size_t szB8i    = num_moduli * ldb8i * static_cast<size_t>(n);
    const size_t szC32i   = ldc32i * static_cast<size_t>(n);
    const size_t szZhi    = szC32i;
    const size_t szZlo    = szC32i;
    const size_t szSftA   = cola8i;
    const size_t szSftB   = padn;
    const size_t szNanFlag = 1;

    const size_t wsBytes =
          szA8i    * sizeof(int8_t)
        + szB8i    * sizeof(int8_t)
        + szC32i   * sizeof(int32_t)
        + szZhi    * sizeof(double)
        + szZlo    * sizeof(double)
        + szSftA   * sizeof(int16_t)
        + szSftB   * sizeof(int16_t)
        + szNanFlag * sizeof(uint32_t);

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

    int8_t*   const A8i      = reinterpret_cast<int8_t*>(ws);
    int8_t*   const B8i      = A8i + szA8i;
    int32_t*  const C32i     = reinterpret_cast<int32_t*>(B8i + szB8i);
    double*   const Zhi      = reinterpret_cast<double*>(C32i + szC32i);
    double*   const Zlo      = Zhi + szZhi;
    int16_t*  const sftA     = reinterpret_cast<int16_t*>(Zlo + szZlo);
    int16_t*  const sftB     = sftA + szSftA;
    uint32_t* const nan_flag = reinterpret_cast<uint32_t*>(sftB + szSftB);

    if(hipMemsetAsync(nan_flag, 0, sizeof(uint32_t), stream) != hipSuccess) {
        (void)hipFreeAsync(ws, stream);
        return rocblaslt_status_internal_error;
    }

    int8_t* const A8i_high = A8i;
    int8_t* const B8i_high = B8i;

    const bool tA = (opA != HIPBLAS_OP_N);
    const bool tB = (opB != HIPBLAS_OP_N);

    /* Thread-local hipBLASLt handle */
    static thread_local hipblasLtHandle_t t_ltHandle = nullptr;
    if(t_ltHandle == nullptr) {
        if(hipblasLtCreate(&t_ltHandle) != HIPBLAS_STATUS_SUCCESS) {
            (void)hipFreeAsync(ws, stream);
            return rocblaslt_status_memory_error;
        }
    }

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

    /* Part 1a-b: per-row/col 6-bit preliminary extraction + Inf/NaN detection */
    hipLaunchKernelGGL(oz2_accu_prelim_A_kernel,
                       dim3(static_cast<unsigned>(m)), dim3(128), 0, stream,
                       A, m, k, lda, tA, A8i_high, lda8i, sftA, nan_flag);
    hipLaunchKernelGGL(oz2_accu_prelim_B_kernel,
                       dim3(static_cast<unsigned>(n)), dim3(128), 0, stream,
                       B, k, n, ldb, tB, B8i_high, ldb8i, sftB, nan_flag);

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

    /* Part 1c: preliminary INT8 GEMM */
    hipblasLtMatmul(t_ltHandle, matmulDesc,
                    &one_i, A8i_high, layoutA, B8i_high, layoutB,
                    &zero_i, C32i, layoutCD, C32i, layoutCD,
                    nullptr, nullptr, 0, stream);

    /* Parts 1d-e: refine shifts using the accu_log2P for the chosen num_moduli */
    const float accu_log2P = h_accu_log2P_all[num_moduli - 2];
    hipLaunchKernelGGL(oz2_accu_refine_sftA_kernel,
                       dim3(static_cast<unsigned>(m)), dim3(128), 0, stream,
                       C32i, m, n, ldc32i, sftA, accu_log2P);
    hipLaunchKernelGGL(oz2_accu_refine_sftB_kernel,
                       dim3(static_cast<unsigned>(n)), dim3(128), 0, stream,
                       C32i, m, n, ldc32i, sftB, accu_log2P);

    /* Part 1f: full multi-modulus scaling */
    {
        const dim3 blk(16, 16);
        const dim3 gA((m + 15) / 16, (k + 15) / 16);
        const dim3 gB((k + 15) / 16, (n + 15) / 16);
        hipLaunchKernelGGL(oz2_scaleA_kernel, gA, blk, 0, stream,
                           A, m, k, lda, tA, A8i, lda8i, cola8i, sftA, num_moduli);
        hipLaunchKernelGGL(oz2_scaleB_kernel, gB, blk, 0, stream,
                           B, k, n, ldb, tB, B8i, ldb8i, sftB, num_moduli);
    }

    /* Parts 2a-d: for each modulus — INT8 GEMM + CRT accumulation */
    const dim3 blk_acc(16, 16);
    const dim3 grid_acc((m + 15) / 16, (n + 15) / 16);
    const size_t strideA8i = lda8i * cola8i;
    const size_t strideB8i = ldb8i * static_cast<size_t>(n);

    for(unsigned t = 0; t < num_moduli; ++t) {
        const int8_t* At = A8i + t * strideA8i;
        const int8_t* Bt = B8i + t * strideB8i;

        hipblasLtMatmul(t_ltHandle, matmulDesc,
                        &one_i, At, layoutA, Bt, layoutB,
                        &zero_i, C32i, layoutCD, C32i, layoutCD,
                        nullptr, nullptr, 0, stream);

        hipLaunchKernelGGL(oz2_accum_kernel, grid_acc, blk_acc, 0, stream,
                           C32i, Zhi, Zlo, m, n, ldc32i, t);
    }

    /* Parts 3+4: collapse, range-reduce, per-element inverse scale */
    hipLaunchKernelGGL(oz2_finalize_kernel, grid_acc, blk_acc, 0, stream,
                       Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd,
                       *alpha, *beta, sftA, sftB);

    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(layoutCD);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutA);

    if(ws_owned) {
        if(hipFreeAsync(ws, stream) != hipSuccess)
            return rocblaslt_status_internal_error;
    }
    return rocblaslt_status_success;
}
