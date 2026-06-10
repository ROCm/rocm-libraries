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
/* Maximum number of moduli supported (s = 2..OZ2_S_MAX).
 * Capped at 18 (log2(M) ≈ 140 bits, far more than needed for any practical
 * FP64 input).  This keeps the symmetric modular reduction to 2 FP32
 * refinement passes for both A and B, matching the GEMMul8 reference
 * implementation, and eliminates the runtime-fallback scale kernel.          */
static constexpr unsigned OZ2_S_MAX = 18;

/* Alignment for INT8 arrays (128 bytes = 128 INT8 elements) */
static constexpr size_t OZ2_ALIGN = 128;

static __host__ __device__ size_t oz2_pad(size_t n)
{
    return (n + OZ2_ALIGN - 1) / OZ2_ALIGN * OZ2_ALIGN;
}

static constexpr size_t OZ2_CHUNK_TARGET_BYTES = 8ull << 30;   /* 8 GiB    */
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
    const size_t slice_bytes = lda8i * cola8i + ldb8i * static_cast<size_t>(n);
    if(slice_bytes == 0u) return num_moduli;
    const size_t s = std::min(static_cast<size_t>(num_moduli),
                              std::max(static_cast<size_t>(gemm_chunk_sz),
                                       OZ2_SCALE_CHUNK_TARGET_BYTES / slice_bytes));
    return static_cast<unsigned>(s);
}

/* =========================================================================
 * GPU-side constant memory
 * ========================================================================= */
static __constant__ double cNegMod[OZ2_S_MAX];
static __constant__ double cInvMod[OZ2_S_MAX];
static __constant__ float  cInvModF[OZ2_S_MAX];
static __constant__ double cQpiHi[OZ2_S_MAX];
static __constant__ double cQpiLo[OZ2_S_MAX];
static __constant__ double cP_hi;
static __constant__ double cP_lo;
static __constant__ double cInvP;

/* =========================================================================
 * Host-side tables (source: GEMMul8/GEMMul8/src/table.hpp)
 *
 * Moduli in order: 256 (implicit), 255, 253, 251, 247, 241, 239, 233,
 *                  229, 227, 223, 217, 211, 199, 197, 193, 191, 181
 *                  (OZ2_S_MAX = 18 total)
 *
 * All arrays indexed by table_idx = s - 2  (s = number of moduli, 2..18).
 * ========================================================================= */
static const double h_neg_mod[OZ2_S_MAX] = {
    -256.0, -255.0, -253.0, -251.0, -247.0, -241.0, -239.0,
    -233.0, -229.0, -227.0, -223.0, -217.0, -211.0, -199.0,
    -197.0, -193.0, -191.0, -181.0
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
    0x1.6a13cd1537290p-8    /* 1/181 */
};
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
    0x1.6a13cep-8F    /* 1/181  */
};

/* -M (high part) for s = 2..18 */
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
};
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
};
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
};
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
};

static const double h_qpi_hi_all[OZ2_S_MAX - 1][OZ2_S_MAX] = {
    /* s=2  (qPi_1[0]) */
    {0x1.fc02000000000p+15, 0x1.0000000000000p+8,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=3  (qPi_1[1]) */
    {0x1.50ac020000000p+23, 0x1.f60c000000000p+22, 0x1.a45a000000000p+23,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=4  (qPi_1[2]) */
    {0x1.0688601000000p+28, 0x1.f01e000000000p+28, 0x1.4826900000000p+28,
     0x1.6654440000000p+31,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=5  (qPi_1[3]) */
    {0x1.99c1435808000p+37, 0x1.d553914600000p+39, 0x1.cf9d0d8400000p+38,
     0x1.2ff09e4000000p+38, 0x1.dae0172c00000p+39,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=6  (qPi_1[4]) */
    {0x1.24d0f0aa6c020p+47, 0x1.00ffb685c4000p+47, 0x1.7820600df8000p+45,
     0x1.b28fb528de000p+47, 0x1.765c060a1c000p+47, 0x1.56b441a210000p+47,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=7  (qPi_1[5]) */
    {0x1.49071d4742060p+55, 0x1.5fae947039b40p+55, 0x1.42fdb9e1948e0p+55,
     0x1.187c8ee783700p+55, 0x1.e89ef222a1c00p+52, 0x1.0316493fe27a0p+55,
     0x1.1f8e561d65780p+53,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=8  (qPi_2[0]) */
    {0x1.4f3952ae32400p+63, 0x1.f094cf17cf000p+61, 0x1.0f5bef8d36400p+63,
     0x1.e02e9274c5000p+62, 0x1.a403bd5c1a000p+61, 0x1.a1cf7b99c2800p+62,
     0x1.a54e8a8f42000p+60, 0x1.787fdcb9fa000p+62,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=9  (qPi_2[1]) */
    {0x1.9a7c80fe96000p+69, 0x1.43ca2f89db000p+71, 0x1.40f4871424000p+70,
     0x1.2c6790ef15000p+71, 0x1.24d66e4d76000p+70, 0x1.459c5b1ee5800p+71,
     0x1.d43c2b2519000p+70, 0x1.ab93da2aca000p+70, 0x1.dfbe1fda93000p+70,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=10 (qPi_2[2]) */
    {0x1.1ba01a9548000p+75, 0x1.b499060d20000p+76, 0x1.8d00367a82000p+77,
     0x1.348f721e1e000p+77, 0x1.09c9ed1acf000p+79, 0x1.6988bc8c28000p+75,
     0x1.4e2df779b8000p+77, 0x1.54302cc6b7000p+78, 0x1.675767107c000p+76,
     0x1.1fdfa04826000p+77,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=11 (qPi_2[3]) */
    {0x1.ae4dbe76d7000p+86, 0x1.258185fdee000p+86, 0x1.76fdabbf54000p+85,
     0x1.73ade1f823000p+86, 0x1.0cdeb7fb80000p+85, 0x1.0671178918000p+87,
     0x1.c416fd0741000p+86, 0x1.5350d862f8000p+86, 0x1.52567e0ff5000p+86,
     0x1.d0611c1cae000p+85, 0x1.814201f9be000p+86,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=12 (qPi_2[4]) */
    {0x1.42dd4f0c25000p+94, 0x1.71af2232d1000p+94, 0x1.b5f1f25063000p+93,
     0x1.0e8e8784ac000p+93, 0x1.0477c23ba5000p+93, 0x1.ac3c7c8760800p+94,
     0x1.507ba57edc000p+92, 0x1.2b20ca473f000p+93, 0x1.5f2d33fd22000p+92,
     0x1.ab17cae65c800p+94, 0x1.408e48b610000p+90, 0x1.32c582e2cf000p+94,
     0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=13 (qPi_2[5]) */
    {0x1.187ecea5a8800p+102, 0x1.71af223280000p+94,  0x1.5a685a078a000p+102,
     0x1.48a0e93cba000p+102, 0x1.6d422253da000p+102, 0x1.ec015f50a0000p+101,
     0x1.27d31b1920000p+99,  0x1.7b4d942fe0000p+100, 0x1.68332a1fe8000p+101,
     0x1.7859de7afc000p+99,  0x1.317d98db46800p+102, 0x1.08b9be1306800p+102,
     0x1.411e88bd34000p+100, 0.0,0.0,0.0,0.0,0.0},
    /* s=14 (qPi_2[6]) */
    {0x1.4af9bb23b8000p+107, 0x1.e0730f7df3000p+109, 0x1.9e197740a0000p+109,
     0x1.11b44daf38000p+106, 0x1.959dba1ed5000p+109, 0x1.d3f9c70059000p+109,
     0x1.c71fc39610000p+108, 0x1.6e1a9ef495000p+109, 0x1.067fc962e0800p+110,
     0x1.81de6aed04000p+109, 0x1.086d6ad9bc800p+110, 0x1.66ccfaf43f000p+109,
     0x1.d2ae54e567000p+109, 0x1.98842ba66f000p+109,
     0.0,0.0,0.0,0.0},
    /* s=15 (qPi_2[7]) */
    {0x1.8334edf0c0800p+117, 0x1.d9618469e1000p+116, 0x1.4c97d49af8800p+117,
     0x1.3db0f47816800p+117, 0x1.ac11e30d56000p+116, 0x1.d3f9c70000000p+109,
     0x1.0210da6024000p+117, 0x1.2e86f6e52b000p+116, 0x1.f43197eee2000p+115,
     0x1.e913152bf0000p+115, 0x1.775c686f24000p+116, 0x1.44d556f611000p+116,
     0x1.90e2677038000p+115, 0x1.1b5f498bca000p+117, 0x1.9702ab51fa000p+116,
     0.0,0.0,0.0},
    /* s=16 (qPi_2[8]) */
    {0x1.568442b104000p+122, 0x1.23c286bfdb000p+125, 0x1.fffd89ae2f000p+124,
     0x1.9f80a3facf000p+124, 0x1.6b10abb2b0000p+124, 0x1.b90322c900000p+119,
     0x1.ff687bb9b9000p+124, 0x1.494950989a000p+125, 0x1.5c176f9414000p+122,
     0x1.6dca3fa2e7000p+124, 0x1.951e4290e0000p+122, 0x1.a671255128000p+123,
     0x1.b2745cf9ae000p+124, 0x1.2c6cfd90da000p+123, 0x1.a57e7d4e8e000p+124,
     0x1.8f40d0ef24000p+124,
     0.0,0.0},
    /* s=17 (qPi_2[9]) */
    {0x1.e01f9407c4000p+129, 0x1.e201959d63000p+131, 0x1.31982160c4000p+132,
     0x1.7f0fe22eef000p+132, 0x1.00d5bf9f80000p+126, 0x1.8ad801f1a0000p+129,
     0x1.2a9c662802000p+130, 0x1.d836977997000p+131, 0x1.85903a5f3c000p+132,
     0x1.a3320451ba800p+132, 0x1.ce462d2242000p+132, 0x1.d67cf11ca9800p+132,
     0x1.add7c7ba40000p+132, 0x1.57b0afae95000p+131, 0x1.e30840c0e8000p+128,
     0x1.5aabc9d4bf800p+132, 0x1.82a0ee308b800p+132,
     0.0},
    /* s=18 (qPi_2[10]) */
    {0x1.06cf388320000p+134, 0x1.a1bf2dfdc0000p+136, 0x1.bb35a9d83c000p+137,
     0x1.b0c7cfa209000p+139, 0x1.4921eae073800p+140, 0x1.172ab95fd6000p+139,
     0x1.68acfd38e8000p+139, 0x1.f34ce4f4e8000p+138, 0x1.01123dfc72000p+140,
     0x1.9db3f73893000p+139, 0x1.f6d5907a7e000p+138, 0x1.e7abc6d98b000p+139,
     0x1.8e92d65018000p+136, 0x1.1d42b11e83800p+140, 0x1.0579b3ad70800p+140,
     0x1.0cb5cec87c000p+138, 0x1.2009162ca2800p+140, 0x1.3d803cbad1800p+140},
};

static const double h_qpi_lo_all[OZ2_S_MAX - 1][OZ2_S_MAX] = {
    /* s=2..7: lo = 0 */
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=8  (qPi_2[0]) */
    {0x1.16f0100000000p+20, 0x1.89a0000000000p+19, 0x1.8880000000000p+19,
     0x1.d740000000000p+19, 0x1.0b80000000000p+19, 0x1.2880000000000p+19,
     0x1.bcf0000000000p+20, 0x1.2d80000000000p+17,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=9  (qPi_2[1]) */
    {0x1.008cc04000000p+26, 0x1.eca4600000000p+28, 0x1.9a00780000000p+29,
     0x1.e855180000000p+29, 0x1.e9c7f00000000p+29, 0x1.38caf00000000p+29,
     0x1.d6d0600000000p+29, 0x1.e459400000000p+27, 0x1.9cd8200000000p+27,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=10 (qPi_2[2]) */
    {0x1.c6c29fa008000p+37, 0x1.4ddc380000000p+30, 0x1.5e72640800000p+37,
     0x1.5939d00000000p+34, 0x1.acce161000000p+36, 0x1.d3148d7000000p+37,
     0x1.1bca621000000p+37, 0x1.be65b8a000000p+35, 0x1.43b8ee6000000p+36,
     0x1.940b60e000000p+36,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=11 (qPi_2[3]) */
    {0x1.c311739de0100p+44, 0x1.3f5c901690000p+45, 0x1.de7087e210000p+45,
     0x1.6bfc28bd30000p+44, 0x1.de9bee2d48000p+45, 0x1.5646b56780000p+45,
     0x1.5ee3b89260000p+43, 0x1.77449328c0000p+43, 0x1.2e0367d338000p+45,
     0x1.c1e3b22c60000p+45, 0x1.4dba603168000p+45,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=12 (qPi_2[4]) */
    {0x1.f5cc036fee804p+50, 0x1.9502088c71500p+52, 0x1.f27fe97ac8c00p+52,
     0x1.6a7fd4fb91000p+50, 0x1.9f4e1d77bb800p+52, 0x1.541aed8de8f00p+52,
     0x1.cad9eee787600p+51, 0x1.b754bf1ae1c00p+51, 0x1.d1793fc3ce200p+51,
     0x1.f0f2278772b00p+52, 0x1.59f94c68de600p+52, 0x1.f1f52b3aa8500p+52,
     0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=13 (qPi_2[5]) */
    {0x1.2a800bf67755ap+60, 0x1.459502088c715p+60, 0x1.73141ccb58410p+57,
     0x1.956a7a15d56e0p+60, 0x1.c5f9191e4aa91p+60, 0x1.c69660c475d7bp+60,
     0x1.1a2f5dd7b0278p+60, 0x1.f7e2f13271df4p+60, 0x1.008e6afbfbd20p+59,
     0x1.45eaf7cf70b15p+60, 0x1.b2d9a1321591ap+59, 0x1.2a3c04a60a8a8p+59,
     0x1.25d2c634e54f0p+57, 0.0,0.0,0.0,0.0,0.0},
    /* s=14 (qPi_2[6]) */
    {0x1.ed366131bfd87p+61, 0x1.2a2b688425b37p+67, 0x1.ea31249d190dbp+66,
     0x1.57f8ce0e05580p+65, 0x1.34d662a4fdd1cp+66, 0x1.5013290076958p+67,
     0x1.3e7351822d438p+66, 0x1.1ebdf25941f8bp+67, 0x1.89ef93ae85687p+67,
     0x1.84d8fc60d93d4p+68, 0x1.1340b8f1c34bfp+67, 0x1.590ded2a35e12p+68,
     0x1.34cf70f07ae33p+67, 0x1.d80e799d28f38p+68,
     0.0,0.0,0.0,0.0},
    /* s=15 (qPi_2[7]) */
    {0x1.a4b62a6fdb1e1p+75, 0x1.c75bd2f612d0fp+75, 0x1.3f90fd4ad5142p+74,
     0x1.9e3c7f45d92bfp+75, 0x1.c9413fbd969ffp+75, 0x1.6550132900769p+75,
     0x1.a05bb4379a4c1p+75, 0x1.dae820f5ffc00p+74, 0x1.6405781ac87d9p+75,
     0x1.175dbeffba9cdp+75, 0x1.fe04b43a93e73p+71, 0x1.67335f8e813b8p+73,
     0x1.1bfe09769edb0p+75, 0x1.f1ee6037f1f5dp+71, 0x1.0c08e68bbfe9cp+75,
     0.0,0.0,0.0},
    /* s=16 (qPi_2[8]) */
    {0x1.195bce21a4a4cp+82, 0x1.d368142940c54p+83, 0x1.6961d82d67d29p+81,
     0x1.19861fe645aefp+78, 0x1.4aa2ee5f58c0cp+82, 0x1.42e6d8398ebf0p+83,
     0x1.09590940ec246p+83, 0x1.4ed69939f54a5p+83, 0x1.bccd986816af6p+83,
     0x1.38fff5b887f40p+83, 0x1.8c2bed86953acp+81, 0x1.2544a485cce86p+81,
     0x1.c88c3ec2fb90fp+83, 0x1.a84f9682c93f7p+83, 0x1.0beb05e6abcdfp+81,
     0x1.aeb1b1661a570p+81,
     0.0,0.0},
    /* s=17 (qPi_2[9]) */
    {0x1.1e87e3b708c22p+90, 0x1.4160efcbeef78p+90, 0x1.4480003b19f81p+89,
     0x1.0b25d1ed6a121p+87, 0x1.747bf6c0d8b31p+90, 0x1.bcbc193dd346cp+88,
     0x1.fddf745f1ee5ap+88, 0x1.d1f525311dabfp+90, 0x1.1660b883eb1a4p+90,
     0x1.41dedd270b797p+88, 0x1.79125e4f2418ap+90, 0x1.0272e6220fc37p+90,
     0x1.d2e0f92de9773p+87, 0x1.ea083b704edc0p+90, 0x1.eba48f8e2a378p+90,
     0x1.a977e531befa8p+90, 0x1.2ed72602864a3p+88,
     0.0},
    /* s=18 (qPi_2[10]) */
    {0x1.928222f7c81d9p+98, 0x1.7691a1e475ec2p+97, 0x1.50297632195fep+97,
     0x1.08e60e4fc6baep+96, 0x1.3586f9a06cbf4p+98, 0x1.7a70fdd8b6610p+98,
     0x1.5e172302320e2p+98, 0x1.1c4557a753b6cp+98, 0x1.4622df275a365p+95,
     0x1.cf7c96f698830p+95, 0x1.ebd8e9c0e37a5p+97, 0x1.f1a79e989b457p+97,
     0x1.1c8ffe978c39ep+98, 0x1.c0c39f95f19abp+92, 0x1.f55e10b41e4a2p+97,
     0x1.a546c43a54205p+98, 0x1.c22d132ce1471p+97, 0x1.6f3d636bc541bp+95},
};

/* =========================================================================
 * One-time constant-memory initialisation
 * ========================================================================= */
static hipError_t oz2_init_constants(unsigned num_moduli)
{
    static unsigned done_for = 0u;
    if(done_for == num_moduli) return hipSuccess;
    const unsigned idx = num_moduli - 2;
#define OZ2_CHECK(expr) do { hipError_t _e=(expr); if(_e!=hipSuccess){done_for=0u;return _e;} } while(0)
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cNegMod),  h_neg_mod,   sizeof(h_neg_mod)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvMod),  h_inv_mod,   sizeof(h_inv_mod)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvModF), h_inv_mod_f, sizeof(h_inv_mod_f)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cQpiHi), h_qpi_hi_all[idx], num_moduli*sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cQpiLo), h_qpi_lo_all[idx], num_moduli*sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cP_hi),  &h_P_hi_all[idx],  sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cP_lo),  &h_P_lo_all[idx],  sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvP),  &h_inv_P_all[idx], sizeof(double)));
#undef OZ2_CHECK
    done_for = num_moduli;
    return hipSuccess;
}

/* =========================================================================
 * Host-side emulation control functions
 * ========================================================================= */
bool fp64EmulationIsEnabled()
{
    static const bool enabled = []() -> bool {
        const char* v = std::getenv("HIPBLASLT_EMULATE_DOUBLE_PRECISION");
        return (v != nullptr && std::strcmp(v, "1") == 0);
    }();
    return enabled;
}

bool fp64EmulationPerformanceCheck(int64_t m, int64_t n, int64_t k, unsigned num_moduli)
{
    static constexpr double HBM_BW         = 6.4e12;
    static constexpr double INT8_PEAK       = 3.05e15;
    static constexpr double FP64_EFF        = 7.0e13;
    static constexpr double LATENCY_KERNEL  = 5.0e-6;
    static constexpr double LATENCY_MATMUL  = 10.0e-6;
    static constexpr double LATENCY_MEMSET  = 2.0e-6;
    static constexpr double CHUNK_BYTES_D   = static_cast<double>(OZ2_CHUNK_TARGET_BYTES);

    const double s   = static_cast<double>(num_moduli);
    const double mn  = static_cast<double>(m) * static_cast<double>(n);
    const double mk  = static_cast<double>(m) * static_cast<double>(k);
    const double kn  = static_cast<double>(k) * static_cast<double>(n);
    const double mnk = mn * static_cast<double>(k);

    const double chunk_sz = std::max(1.0, std::min(s, CHUNK_BYTES_D / (mn * 4.0)));
    const double n_chunks = std::ceil(s / chunk_sz);

    const double t_int8_gemm_bw = (mk + kn + 4.0 * mn) / HBM_BW;
    const double t_prelim_gemm  = std::max(2.0 * mnk / INT8_PEAK, t_int8_gemm_bw);
    const double t_prelim_kern  = (mk + kn) * 17.0 / HBM_BW;
    const double t_refine_kern  = mn * 8.0 / HBM_BW;
    const double t_scale_kern   = (mk + kn) * (8.0 + s) / HBM_BW;
    const double t_int8_gemms   = s * std::max(2.0 * mnk / INT8_PEAK, t_int8_gemm_bw);
    const double t_accum_kern  = mn * (4.0 * s + 32.0 * n_chunks - 16.0) / HBM_BW;
    const double t_launch = (5.0 + n_chunks) * LATENCY_KERNEL
                          + (1.0 + n_chunks) * LATENCY_MATMUL
                          + LATENCY_MEMSET;

    const double t_emul = t_prelim_gemm + t_prelim_kern + t_refine_kern
                        + t_scale_kern  + t_int8_gemms  + t_accum_kern + t_launch;
    const double t_native = std::max(2.0 * mnk / FP64_EFF,
                                     8.0 * (mk + kn + mn) / HBM_BW) + LATENCY_MATMUL;
    return t_emul <= t_native;
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
        if(v == nullptr) return 0x3u;
        return static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
    }();
    return mask;
}

static constexpr double oz2_cum_bits[OZ2_S_MAX - 1] = {
     15.994,  /* s=2  */  23.976,  /* s=3  */  31.945,  /* s=4  */
     39.894,  /* s=5  */  47.807,  /* s=6  */  55.708,  /* s=7  */
     63.572,  /* s=8  */  71.411,  /* s=9  */  79.238,  /* s=10 */
     87.040,  /* s=11 */  94.801,  /* s=12 */ 102.522,  /* s=13 */
    110.160,  /* s=14 */ 117.782,  /* s=15 */ 125.374,  /* s=16 */
    132.949,  /* s=17 */ 140.448,  /* s=18 */
};

bool fp64EmulationWouldApply(const _rocblaslt_handle* h, hipDataType type_a,
                              int64_t m, int64_t n, int64_t k, int batch_count)
{
    if(type_a != HIP_R_64F || batch_count != 1) return false;
    const bool emulEnabled = (h->emulation.enabled == 1)
                           || (h->emulation.enabled != 0 && fp64EmulationIsEnabled());
    if(!emulEnabled) return false;
    const bool eager = (h->emulation.strategy == 2)
                     || (h->emulation.strategy != 1 && fp64EmulationIsEager());
    const unsigned s = fp64EmulationEffectiveNumModuli(h);
    return eager || fp64EmulationPerformanceCheck(m, n, k, s);
}

unsigned fp64EmulationEffectiveNumModuli(const _rocblaslt_handle* h)
{
    if(h->emulation.mantissa_control == 1 && h->emulation.max_mantissa_bits >= 0) {
        const unsigned target = static_cast<unsigned>(h->emulation.max_mantissa_bits);
        for(unsigned s = 2u; s <= OZ2_S_MAX; ++s)
            if(oz2_cum_bits[s - 2u] >= static_cast<double>(target)) return s;
        return OZ2_S_MAX;
    }
    return fp64EmulationNumModuli();
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
    const size_t szC32i = ldc32i * static_cast<size_t>(n);
    const unsigned gemm_sz  = oz2_compute_chunk_size(m, n, num_moduli);
    const unsigned scale_sz = oz2_compute_scale_chunk_size(m, n, k, num_moduli, gemm_sz);

    return   scale_sz * lda8i * cola8i * sizeof(int8_t)
           + scale_sz * ldb8i * static_cast<size_t>(n) * sizeof(int8_t)
           + gemm_sz  * szC32i * sizeof(int32_t)
           + szC32i * sizeof(double) * 2
           + cola8i * sizeof(int16_t)
           + padn   * sizeof(int16_t)
           + sizeof(uint32_t)
           + cola8i * sizeof(int32_t);
}

unsigned fp64EmulationNumModuli()
{
    static const unsigned num_moduli = []() -> unsigned {
        const char* v = std::getenv("HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT");
        if(v == nullptr) return 16u;
        const unsigned target = static_cast<unsigned>(std::strtoul(v, nullptr, 0));
        if(target == 0u) return OZ2_S_MAX;
        for(unsigned s = 2u; s <= OZ2_S_MAX; ++s)
            if(oz2_cum_bits[s - 2u] >= static_cast<double>(target)) return s;
        return OZ2_S_MAX;
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
    if(threadIdx.x % warpSize == 0) s_wmax[threadIdx.x / warpSize] = warp_max;
    __syncthreads();
    double result = 0.0;
    if(threadIdx.x == 0) {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        result = s_wmax[0];
        for(int w = 1; w < nw; ++w) if(s_wmax[w] > result) result = s_wmax[w];
    }
    return result;
}

static __device__ __forceinline__ int32_t
block_reduce_max_i32(int32_t warp_max, int32_t* __restrict__ s_wmax)
{
    if(threadIdx.x % warpSize == 0) s_wmax[threadIdx.x / warpSize] = warp_max;
    __syncthreads();
    int32_t result = 0;
    if(threadIdx.x == 0) {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        result = s_wmax[0];
        for(int w = 1; w < nw; ++w) if(s_wmax[w] > result) result = s_wmax[w];
    }
    return result;
}

/* =========================================================================
 * GPU kernel — fused preliminary shift computation + INT8 extraction
 *
 * oz2_accu_prelim_kernel<TRANS_A, TRANS_B, CHECK_NAN>
 *
 * Fuses per-row/col shift computation (sftA, sftB) and preliminary INT8
 * extraction (A8i_high, B8i_high) into a SINGLE kernel dispatch, eliminating
 * the extra launch overhead of the previous two-kernel approach and achieving
 * coalesced HBM reads AND writes for all four (TRANS_A, TRANS_B) combinations.
 *
 *   TRANS_A=true  (op(A)=A^T, A stored k×m col-major): k-fast (threadIdx.x=j).
 *     Loop 1: read A[row*lda+j] (COALESCED) → block-reduce → sft.
 *     Loop 2: read A[row*lda+j] (COALESCED) → scale → write A8i_high[j+row*lda8i] (COALESCED).
 *
 *   TRANS_A=false (op(A)=A, A stored m×k col-major): tiled SHMEM.
 *     One block per TILE-row tile; two passes over k:
 *       Pass 1: load m-fast tiles into SHMEM (COALESCED), accumulate per-row max → sft.
 *       Pass 2: load m-fast tiles into SHMEM (COALESCED), scale, write A8i_high k-fast
 *               from SHMEM via SHMEM transposition (COALESCED).
 *
 *   TRANS_B=false / TRANS_B=true: symmetric to TRANS_A=true / TRANS_A=false.
 *
 * Grid  = dim3(m_blks + n_blks, 1)
 *   m_blks = TRANS_A ? m       : ceil(m / OZ2_PRELIM_TILE)
 *   n_blks = TRANS_B ? ceil(n / OZ2_PRELIM_TILE) : n
 * Block = dim3(OZ2_PRELIM_TILE * OZ2_PRELIM_TILE, 1) = dim3(256, 1)
 * ========================================================================= */
static constexpr int OZ2_PRELIM_TILE_K = 64;  /* k-tile size (reduces k-tile loop count) */
static constexpr int OZ2_PRELIM_TILE_M = 4;   /* rows/cols per tile (= blockDim.x / TILE_K) */
/* blockDim.x = TILE_K × TILE_M = 256 threads */

template <bool TRANS_A, bool TRANS_B, bool CHECK_NAN>
__global__ static void
oz2_accu_prelim_kernel(const double* __restrict__ A,
                        int64_t m, int64_t k, int64_t lda,
                        int8_t*  __restrict__  A8i_high, size_t lda8i,
                        int16_t* __restrict__  sftA,
                        const double* __restrict__ B,
                        int64_t n, int64_t ldb,
                        int8_t*  __restrict__  B8i_high, size_t ldb8i,
                        int16_t* __restrict__  sftB,
                        uint32_t* __restrict__ nan_flag,
                        unsigned m_blks)
{
    static constexpr int TILE_K = OZ2_PRELIM_TILE_K;  /* k-tile size (256/TILE_M iterations) */
    static constexpr int TILE_M = OZ2_PRELIM_TILE_M;  /* rows/cols per block */

    /* Shared memory:
     *   shmem[TILE_K][TILE_M+1] — FP64 tile (TILE_M+1 padding avoids bank conflicts)
     *   s_sft[TILE_M]           — per-row/col sft broadcast
     *   s_wmax[4]               — warp maxes for block_reduce_max_d (4 warps of 64)
     *
     * TILE_K=64, TILE_M=4 → 4× fewer k-tile iterations (k/64 vs k/16 for TILE=16),
     * 4× fewer __syncthreads() in Pass 2, 4× more A/B blocks → better occupancy. */
    __shared__ double  shmem[TILE_K][TILE_M + 1];
    __shared__ int16_t s_sft[TILE_M];
    __shared__ double  s_wmax[4];

    if(blockIdx.x < m_blks) {
        /* ── A block ────────────────────────────────────────────────────── */
        if constexpr (TRANS_A) {
            /* k-fast: one block per op(A) row; two k-loops.               */
            const int64_t row = static_cast<int64_t>(blockIdx.x);

            /* Loop 1: compute per-row max */
            double local_max = 0.0;
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val = A[row * lda + j];                     /* COALESCED */
                if constexpr (CHECK_NAN)
                    if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                double av = fabs(val);
                if(av > local_max) local_max = av;
            }
            local_max = warp_reduce_max_abs_d(local_max);
            local_max = block_reduce_max_d(local_max, s_wmax);
            if(threadIdx.x == 0) {
                if(local_max < 1e-300) local_max = 1.0;
                s_sft[0] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(local_max))));
                sftA[row] = s_sft[0];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[0]);

            /* Loop 2: scale and write A8i_high */
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val    = A[row * lda + j];                  /* COALESCED */
                double scaled = ceil(ldexp(fabs(val), sft));
                A8i_high[static_cast<size_t>(j)
                         + static_cast<size_t>(row) * lda8i] =
                    static_cast<int8_t>(static_cast<int32_t>(scaled));  /* COALESCED */
            }

        } else {
            /* Tiled SHMEM: one block per TILE_M-row tile.
             * Thread t: k_local = t/TILE_M (k-index in tile, 0..TILE_K-1),
             *           m_local = t%TILE_M (row-index in tile, 0..TILE_M-1).
             * Adjacent threads (same k_local, consecutive m_local) access
             * consecutive rows of A → COALESCED loads.  k-tile iterations =
             * k/TILE_K (e.g. 16 for k=1024, TILE_K=64).                    */
            const int64_t m_base = static_cast<int64_t>(blockIdx.x) * TILE_M;
            const int t       = static_cast<int>(threadIdx.x);
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int m_local = t % TILE_M;   /* 0..TILE_M-1 */
            const int64_t i   = m_base + m_local;

            /* Pass 1: accumulate per-row max across all k-tiles */
            double thr_max = 0.0;
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;
                if(i < m && j < k) {
                    /* A[i + j*lda]: adjacent i (= m_local, varies) → COALESCED */
                    double val = A[i + j * lda];
                    if constexpr (CHECK_NAN)
                        if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                    double av = fabs(val);
                    if(av > thr_max) thr_max = av;
                }
            }
            /* Reduce across k_local dimension (TILE_K=64 values per row) */
            shmem[k_local][m_local] = thr_max;
            __syncthreads();
            if(k_local == 0) {   /* TILE_M=4 threads finalise, one per row */
                double row_max = 0.0;
                for(int kl = 0; kl < TILE_K; ++kl)
                    if(shmem[kl][m_local] > row_max) row_max = shmem[kl][m_local];
                if(row_max < 1e-300) row_max = 1.0;
                s_sft[m_local] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(row_max))));
                if(i < m) sftA[i] = s_sft[m_local];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[m_local]);  /* row-specific */

            /* Pass 2: coalesced loads (m-fast) → SHMEM → coalesced writes (k-fast)
             * k_tile iterations = k/TILE_K (32 syncs for k=2048 vs 256 with TILE=16) */
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;

                /* Load: A[i + j*lda], i=m_base+t%TILE_M varies → COALESCED */
                double scaled = 0.0;
                if(i < m && j < k)
                    scaled = ceil(ldexp(fabs(A[i + j * lda]), sft));
                shmem[k_local][m_local] = scaled;
                __syncthreads();

                /* Write: k_write = t%TILE_K varies fast → COALESCED */
                const int k_write = t % TILE_K;
                const int m_write = t / TILE_K;
                const int64_t j_out = k_base  + k_write;
                const int64_t i_out = m_base  + m_write;
                if(i_out < m && j_out < k)
                    A8i_high[static_cast<size_t>(j_out)
                             + static_cast<size_t>(i_out) * lda8i] =
                        static_cast<int8_t>(static_cast<int32_t>(
                            shmem[k_write][m_write]));
                __syncthreads();
            }
        }

    } else {
        /* ── B block (symmetric to A, with TRANS_B) ─────────────────── */
        if constexpr (!TRANS_B) {
            /* j-fast: one block per op(B) col, threadIdx.x = j */
            const int64_t col = static_cast<int64_t>(blockIdx.x - m_blks);

            /* Loop 1: compute per-col max */
            double local_max = 0.0;
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val = B[j + col * ldb];                     /* COALESCED */
                if constexpr (CHECK_NAN)
                    if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                double av = fabs(val);
                if(av > local_max) local_max = av;
            }
            local_max = warp_reduce_max_abs_d(local_max);
            local_max = block_reduce_max_d(local_max, s_wmax);
            if(threadIdx.x == 0) {
                if(local_max < 1e-300) local_max = 1.0;
                s_sft[0] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(local_max))));
                sftB[col] = s_sft[0];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[0]);

            /* Loop 2: scale and write B8i_high */
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val    = B[j + col * ldb];                  /* COALESCED */
                double scaled = ceil(ldexp(fabs(val), sft));
                B8i_high[static_cast<size_t>(j)
                         + static_cast<size_t>(col) * ldb8i] =
                    static_cast<int8_t>(static_cast<int32_t>(scaled));  /* COALESCED */
            }

        } else {
            /* Tiled SHMEM for TRANS_B=T (B stored n×k, B[col,j]=B[col+j*ldb]).
             * One block per TILE_M-col tile.                                */
            const int64_t n_base = static_cast<int64_t>(blockIdx.x - m_blks) * TILE_M;
            const int t       = static_cast<int>(threadIdx.x);
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int l_local = t % TILE_M;   /* col-index within tile */
            const int64_t col = n_base + l_local;

            /* Pass 1: accumulate per-col max */
            double thr_max = 0.0;
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;
                if(col < n && j < k) {
                    /* B[col + j*ldb]: adjacent col (= l_local, varies) → COALESCED */
                    double val = B[col + j * ldb];
                    if constexpr (CHECK_NAN)
                        if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                    double av = fabs(val);
                    if(av > thr_max) thr_max = av;
                }
            }
            shmem[k_local][l_local] = thr_max;
            __syncthreads();
            if(k_local == 0) {
                double col_max = 0.0;
                for(int kl = 0; kl < TILE_K; ++kl)
                    if(shmem[kl][l_local] > col_max) col_max = shmem[kl][l_local];
                if(col_max < 1e-300) col_max = 1.0;
                s_sft[l_local] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(col_max))));
                if(col < n) sftB[col] = s_sft[l_local];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[l_local]);

            /* Pass 2: coalesced loads (col-fast) → SHMEM → coalesced writes (k-fast) */
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;

                double scaled = 0.0;
                if(col < n && j < k)
                    scaled = ceil(ldexp(fabs(B[col + j * ldb]), sft));  /* COALESCED */
                shmem[k_local][l_local] = scaled;
                __syncthreads();

                const int k_write = t % TILE_K;
                const int l_write = t / TILE_K;
                const int64_t j_out   = k_base  + k_write;
                const int64_t col_out = n_base  + l_write;
                if(col_out < n && j_out < k)
                    B8i_high[static_cast<size_t>(j_out)
                             + static_cast<size_t>(col_out) * ldb8i] =
                        static_cast<int8_t>(static_cast<int32_t>(
                            shmem[k_write][l_write]));
                __syncthreads();
            }
        }
    }
}

/* =========================================================================
 * GPU kernels — accu mode Part 1: shift refinement from preliminary GEMM
 * ========================================================================= */
__global__ static void
oz2_refine_sftA_partial_kernel(const int32_t* __restrict__ C32i,
                                int64_t m, int64_t n, size_t ldc32i,
                                int32_t* __restrict__ row_max)
{
    const int64_t row      = static_cast<int64_t>(blockIdx.x) * 64
                           + static_cast<int64_t>(threadIdx.x);
    const int64_t col_base = static_cast<int64_t>(blockIdx.y) * 64;
    if(row >= m) return;
    int32_t local_max = 0;
    const int64_t col_end = (col_base + 64 < n) ? col_base + 64 : n;
    for(int64_t col = col_base; col < col_end; ++col) {
        int32_t v  = C32i[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    if(local_max > 0) atomicMax(row_max + static_cast<size_t>(row), local_max);
}

__global__ static void
oz2_refine_sftA_apply_kernel(const int32_t* __restrict__ row_max,
                              int16_t* __restrict__ sftA, int64_t m, float log2P)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 64
                      + static_cast<int64_t>(threadIdx.x);
    if(row >= m) return;
    int32_t max_val = row_max[row];
    if(max_val < 1) max_val = 1;
    sftA[row] += static_cast<int16_t>(floorf(-0.5f * log2f(static_cast<float>(max_val)) + log2P));
}

__global__ static void
oz2_refine_sftB_kernel(const int32_t* __restrict__ C32i,
                       int64_t m, int64_t n, size_t ldc32i,
                       int16_t* __restrict__ sftB, float log2P)
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
        sftB[col] += static_cast<int16_t>(floorf(-0.5f * log2f(static_cast<float>(local_max)) + log2P));
    }
}

/* =========================================================================
 * GPU kernels — Part 1f: full multi-modulus scaling (A and B fused)
 *
 * oz2_scaleAB_kernel<T_COUNT, TRANS_A, TRANS_B>: compile-time template.
 *
 * Both A and B branches apply the same 2-pass symmetric modular reduction
 * (FP64 pass + 1 FP32 refinement pass), matching the GEMMul8 reference.
 * With OZ2_S_MAX=18 there is no need for a 3rd FP32 pass or a runtime-
 * fallback kernel.
 *
 * For TRANS_A=true / TRANS_B=false (coalesced reads):
 *   j = t%TILE_K varies fast within warp → stride-1 HBM reads (COALESCED).
 *
 * For TRANS_A=false / TRANS_B=true (non-coalesced reads):
 *   Tiled SHMEM transposition (same structure as oz2_accu_prelim_kernel):
 *     Load:  k_local=t/TILE_M, m_local=t%TILE_M → A[i+j*lda] (m_local fast → COALESCED)
 *     Store raw val in shmem[k_local][m_local]; store per-row sft in s_sft[m_local].
 *     __syncthreads()
 *     Write: k_write=t%TILE_K, m_write=t/TILE_K → shmem transposed read
 *            → A8i[j_out + i_out*lda8i + t_local*stride]  (k_write fast → COALESCED)
 *
 * Grid  = dim3(ceil(k/TILE_K), ceil(m/TILE_M) + ceil(n/TILE_M))  [unchanged]
 * Block = dim3(TILE_K × TILE_M) = dim3(256)                      [was dim3(64,4)]
 *   blockIdx.y <  m_y_blocks → A scaling
 *   blockIdx.y >= m_y_blocks → B scaling
 * ========================================================================= */
static constexpr unsigned OZ2_SCALE_TILE_K = 64;
static constexpr unsigned OZ2_SCALE_TILE_M = 4;

template <unsigned T_COUNT, bool TRANS_A, bool TRANS_B>
__global__ static void
oz2_scaleAB_kernel(const double* __restrict__ A,
                   int64_t m, int64_t lda,
                   int8_t*  __restrict__       A8i, size_t lda8i, size_t cola8i,
                   const int16_t* __restrict__ sftA,
                   const double* __restrict__ B,
                   int64_t n, int64_t ldb,
                   int8_t*  __restrict__       B8i, size_t ldb8i,
                   const int16_t* __restrict__ sftB,
                   int64_t k, unsigned t_start, unsigned m_y_blocks)
{
    static constexpr int TILE_K = static_cast<int>(OZ2_SCALE_TILE_K);
    static constexpr int TILE_M = static_cast<int>(OZ2_SCALE_TILE_M);

    /* SHMEM used only in non-coalesced paths (TRANS_A=false / TRANS_B=true).
     * Declared unconditionally; coalesced paths skip it via if constexpr.   */
    __shared__ double  shmem[TILE_K][TILE_M + 1];  /* +1 avoids bank conflicts */
    __shared__ int16_t s_sft[TILE_M];

    const int t = static_cast<int>(threadIdx.x);

    if(static_cast<unsigned>(blockIdx.y) < m_y_blocks) {
        /* ── A block ─────────────────────────────────────────────────────── */
        const int64_t m_base = static_cast<int64_t>(blockIdx.y) * TILE_M;

        if constexpr (TRANS_A) {
            /* Coalesced: A stored k×m, A[i,j] = A[j + i*lda].
             * j = t%TILE_K varies fast within warp → stride-1 reads.       */
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + (t % TILE_K);
            const int64_t i = m_base + (t / TILE_K);
            if(i >= m || j >= k) return;
            const double val  = A[i * lda + j];                         /* COALESCED */
            const double ival = trunc(ldexp(val, static_cast<int>(sftA[i])));
            const size_t stride = lda8i * cola8i;
            const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(i) * lda8i;
            #pragma unroll
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                const unsigned tidx = t_start + t_local;
                const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                const float   rf = static_cast<float>(r);
                const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                        static_cast<float>(cNegMod[tidx]), rf);
                __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                            A8i + t_local * stride + offset);
            }
        } else {
            /* Non-coalesced load: A stored m×k, A[i,j] = A[i + j*lda].
             * Use SHMEM transposition for coalesced reads AND writes.       */
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int m_local = t % TILE_M;   /* 0..TILE_M-1 */
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + k_local;
            const int64_t i = m_base + m_local;

            /* Load raw val (m_local varies fast → COALESCED) */
            shmem[k_local][m_local] = (i < m && j < k) ? A[i + j * lda] : 0.0;
            if(k_local == 0 && i < m) s_sft[m_local] = sftA[i];
            __syncthreads();

            /* Write: k_write varies fast → COALESCED writes */
            const int k_write = t % TILE_K;
            const int m_write = t / TILE_K;
            const int64_t j_out = static_cast<int64_t>(blockIdx.x) * TILE_K + k_write;
            const int64_t i_out = m_base + m_write;
            if(i_out < m && j_out < k) {
                const double val  = shmem[k_write][m_write];
                const double ival = trunc(ldexp(val, static_cast<int>(s_sft[m_write])));
                const size_t stride = lda8i * cola8i;
                const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i;
                #pragma unroll
                for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                    const unsigned tidx = t_start + t_local;
                    const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                    const float   rf = static_cast<float>(r);
                    const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                            static_cast<float>(cNegMod[tidx]), rf);
                    __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                                A8i + t_local * stride + offset);
                }
            }
        }
    } else {
        /* ── B block (symmetric to A, with TRANS_B) ─────────────────────── */
        const int64_t n_base = static_cast<int64_t>(blockIdx.y - m_y_blocks) * TILE_M;

        if constexpr (!TRANS_B) {
            /* Coalesced: B stored k×n, B[l,j] = B[j + l*ldb].
             * j = t%TILE_K varies fast → stride-1 reads.                   */
            const int64_t j   = static_cast<int64_t>(blockIdx.x) * TILE_K + (t % TILE_K);
            const int64_t col = n_base + (t / TILE_K);
            if(col >= n || j >= k) return;
            const double val  = B[col * ldb + j];                       /* COALESCED */
            const double ival = trunc(ldexp(val, static_cast<int>(sftB[col])));
            const size_t stride = ldb8i * static_cast<size_t>(n);
            const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(col) * ldb8i;
            #pragma unroll
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                const unsigned tidx = t_start + t_local;
                const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                const float   rf = static_cast<float>(r);
                const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                        static_cast<float>(cNegMod[tidx]), rf);
                __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                            B8i + t_local * stride + offset);
            }
        } else {
            /* Non-coalesced: B stored n×k, B[l,j] = B[l + j*ldb].
             * Use SHMEM transposition.                                       */
            const int k_local = t / TILE_M;
            const int l_local = t % TILE_M;
            const int64_t j   = static_cast<int64_t>(blockIdx.x) * TILE_K + k_local;
            const int64_t col = n_base + l_local;

            /* Load raw val (l_local varies fast → COALESCED) */
            shmem[k_local][l_local] = (col < n && j < k) ? B[col + j * ldb] : 0.0;
            if(k_local == 0 && col < n) s_sft[l_local] = sftB[col];
            __syncthreads();

            /* Write: k_write varies fast → COALESCED writes */
            const int k_write = t % TILE_K;
            const int l_write = t / TILE_K;
            const int64_t j_out   = static_cast<int64_t>(blockIdx.x) * TILE_K + k_write;
            const int64_t col_out = n_base + l_write;
            if(col_out < n && j_out < k) {
                const double val  = shmem[k_write][l_write];
                const double ival = trunc(ldexp(val, static_cast<int>(s_sft[l_write])));
                const size_t stride = ldb8i * static_cast<size_t>(n);
                const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(col_out) * ldb8i;
                #pragma unroll
                for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                    const unsigned tidx = t_start + t_local;
                    const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                    const float   rf = static_cast<float>(r);
                    const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                            static_cast<float>(cNegMod[tidx]), rf);
                    __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                                B8i + t_local * stride + offset);
                }
            }
        }
    }
}

/* =========================================================================
 * GPU kernels — Part 2d: chunked CRT accumulation
 * ========================================================================= */
template <bool HAS_LO>
__global__ static void
oz2_chunk_accum_kernel_rt(const int32_t* __restrict__ C32i_batch,
                           double* __restrict__ Zhi, double* __restrict__ Zlo,
                           int64_t m, int64_t n, size_t ldc32i,
                           unsigned chunk_start, unsigned chunk_size, bool is_first_chunk)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    for(unsigned t_local = 0; t_local < chunk_size; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    if(is_first_chunk) { Zhi[idx] = local_hi; Zlo[idx] = local_lo; }
    else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zhi[idx] = s_hi; Zlo[idx] += err + local_lo;
    }
}

template <bool HAS_LO>
__global__ static void
oz2_accum_finalize_kernel_rt(const int32_t* __restrict__ C32i_batch,
                              const double* __restrict__ Zhi_in, const double* __restrict__ Zlo_in,
                              const double* __restrict__ C, double* __restrict__ D,
                              int64_t m, int64_t n, size_t ldc32i, int64_t ldc, int64_t ldd,
                              double alpha, double beta,
                              const int16_t* __restrict__ sftA, const int16_t* __restrict__ sftB,
                              unsigned chunk_start, unsigned chunk_size, bool is_first_chunk)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    for(unsigned t_local = 0; t_local < chunk_size; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    double Zh, Zl;
    if(is_first_chunk) { Zh = local_hi; Zl = local_lo; }
    else {
        const double old_hi = Zhi_in[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zh = s_hi; Zl = Zlo_in[idx] + err + local_lo;
    }
    const double q = rint((Zh + Zl) * cInvP);
    const double X = fma(cP_lo, q, fma(cP_hi, q, Zh) + Zl);
    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
    D[d_idx] = alpha * ldexp(X, inv_sft) + beta * C[c_idx];
}

template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
__global__ static void
oz2_chunk_accum_kernel(const int32_t* __restrict__ C32i_batch,
                       double* __restrict__ Zhi, double* __restrict__ Zlo,
                       int64_t m, int64_t n, size_t ldc32i, unsigned chunk_start)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    #pragma unroll
    for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    if constexpr (IS_FIRST_CHUNK) {
        __builtin_nontemporal_store(local_hi, Zhi + idx);
        __builtin_nontemporal_store(local_lo, Zlo + idx);
    } else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zhi[idx] = s_hi; Zlo[idx] += err + local_lo;
    }
}

template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
__global__ static void
oz2_accum_finalize_kernel(const int32_t* __restrict__ C32i_batch,
                          const double* __restrict__ Zhi_in, const double* __restrict__ Zlo_in,
                          const double* __restrict__ C, double* __restrict__ D,
                          int64_t m, int64_t n, size_t ldc32i, int64_t ldc, int64_t ldd,
                          double alpha, double beta,
                          const int16_t* __restrict__ sftA, const int16_t* __restrict__ sftB,
                          unsigned chunk_start)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    #pragma unroll
    for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    double Zh, Zl;
    if constexpr (IS_FIRST_CHUNK) { Zh = local_hi; Zl = local_lo; }
    else {
        const double old_hi = Zhi_in[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zh = s_hi; Zl = Zlo_in[idx] + err + local_lo;
    }
    const double q = rint((Zh + Zl) * cInvP);
    const double X = fma(cP_lo, q, fma(cP_hi, q, Zh) + Zl);
    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
    D[d_idx] = alpha * ldexp(X, inv_sft) + beta * C[c_idx];
}

static const char* oz2_profile_file()
{
    static const char* const fn = std::getenv("HIPBLASLT_EMULATION_PROFILE");
    return fn;
}

/* =========================================================================
 * fp64EmulatedGemm
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
    const unsigned num_moduli = (settings.num_moduli >= 2u && settings.num_moduli <= OZ2_S_MAX)
                                    ? settings.num_moduli : fp64EmulationNumModuli();
    if(oz2_init_constants(num_moduli) != hipSuccess)
        return rocblaslt_status_internal_error;

    const char* const _pf   = oz2_profile_file();
    const bool        _prof = (_pf != nullptr);
    hipEvent_t _ev0{}, _ev1{}, _ev_tot{};
    float _t_prelim = 0, _t_prelim_gemm = 0, _t_extract = 0, _t_refine = 0,
          _t_scale  = 0, _t_int8 = 0, _t_accum = 0, _t_finalize = 0, _t_total = 0;
    if(_prof) { (void)hipEventCreate(&_ev0); (void)hipEventCreate(&_ev1); (void)hipEventCreate(&_ev_tot); }
    auto _pstart = [&]() noexcept { if(_prof) (void)hipEventRecord(_ev0, stream); };
    auto _pstop  = [&](float& t) noexcept {
        if(_prof) {
            (void)hipEventRecord(_ev1, stream); (void)hipStreamSynchronize(stream);
            float ms = 0.f; (void)hipEventElapsedTime(&ms, _ev0, _ev1); t += ms;
        }
    };

    const unsigned chunk_size       = oz2_compute_chunk_size(m, n, num_moduli);
    const unsigned scale_chunk_size = oz2_compute_scale_chunk_size(m, n, k, num_moduli, chunk_size);

    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = oz2_pad(static_cast<size_t>(n));
    const size_t szC32i = ldc32i * static_cast<size_t>(n);

    const size_t szA8i   = scale_chunk_size * lda8i * cola8i;
    const size_t szB8i   = scale_chunk_size * ldb8i * static_cast<size_t>(n);
    const size_t szZhi   = szC32i;
    const size_t szZlo   = szC32i;
    const size_t szSftA  = cola8i;
    const size_t szSftB  = padn;
    const size_t szNanFlag = 1;
    const size_t szRowMax  = cola8i;

    const size_t wsBytes =
          szA8i    * sizeof(int8_t)
        + szB8i    * sizeof(int8_t)
        + chunk_size * szC32i * sizeof(int32_t)
        + szZhi    * sizeof(double)
        + szZlo    * sizeof(double)
        + szSftA   * sizeof(int16_t)
        + szSftB   * sizeof(int16_t)
        + szNanFlag * sizeof(uint32_t)
        + szRowMax  * sizeof(int32_t);

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
    int32_t*  const C32i_batch = reinterpret_cast<int32_t*>(B8i + szB8i);
    double*   const Zhi        = reinterpret_cast<double*>(C32i_batch + chunk_size * szC32i);
    double*   const Zlo        = Zhi + szZhi;
    int16_t*  const sftA       = reinterpret_cast<int16_t*>(Zlo + szZlo);
    int16_t*  const sftB       = sftA + szSftA;
    uint32_t* const nan_flag   = reinterpret_cast<uint32_t*>(sftB + szSftB);
    int32_t*  const row_max    = reinterpret_cast<int32_t*>(nan_flag + szNanFlag);
    int32_t*  const C32i       = C32i_batch;

    if(_prof) (void)hipEventRecord(_ev_tot, stream);

    int8_t* const A8i_high = A8i;
    int8_t* const B8i_high = B8i;

    const bool tA = (opA != HIPBLAS_OP_N);
    const bool tB = (opB != HIPBLAS_OP_N);

    const uint32_t svmask = (settings.sv_mask != ~0u)
                                ? settings.sv_mask : fp64EmulationSpecialValuesMask();

    if(svmask != 0u) {
        if(hipMemsetAsync(nan_flag, 0, sizeof(uint32_t), stream) != hipSuccess) {
            (void)hipFreeAsync(ws, stream); return rocblaslt_status_internal_error;
        }
    }

    hipblasLtMatrixLayout_t layoutA  = nullptr;
    hipblasLtMatrixLayout_t layoutB  = nullptr;
    hipblasLtMatrixLayout_t layoutCD = nullptr;
    hipblasLtMatmulDesc_t   matmulDesc = nullptr;

    hipblasLtMatrixLayoutCreate(&layoutA,  HIP_R_8I, static_cast<uint64_t>(k), static_cast<uint64_t>(m), static_cast<int64_t>(lda8i));
    hipblasLtMatrixLayoutCreate(&layoutB,  HIP_R_8I, static_cast<uint64_t>(k), static_cast<uint64_t>(n), static_cast<int64_t>(ldb8i));
    hipblasLtMatrixLayoutCreate(&layoutCD, HIP_R_32I, static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<int64_t>(ldc32i));
    hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32I, HIP_R_32I);
    {
        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    }

    const int32_t one_i = 1, zero_i = 0;

    /* Fused preliminary shift + extraction (oz2_accu_prelim_kernel).
     * Grid adapts to TRANS_A/TRANS_B:
     *   m_blks = TRANS_A ? m           : ceil(m / OZ2_PRELIM_TILE_M)
     *   n_blks = TRANS_B ? ceil(n/TM)  : n
     * Block = OZ2_PRELIM_TILE_K × OZ2_PRELIM_TILE_M = 256 threads.    */
    const unsigned m_blks_prelim = tA
        ? static_cast<unsigned>(m)
        : static_cast<unsigned>((m + OZ2_PRELIM_TILE_M - 1) / OZ2_PRELIM_TILE_M);
    const unsigned n_blks_prelim = tB
        ? static_cast<unsigned>((n + OZ2_PRELIM_TILE_M - 1) / OZ2_PRELIM_TILE_M)
        : static_cast<unsigned>(n);
    _pstart();
#define OZ2_PRELIM(TA, TB, CN) \
    hipLaunchKernelGGL((oz2_accu_prelim_kernel<(TA),(TB),(CN)>), \
                       dim3(m_blks_prelim + n_blks_prelim), \
                       dim3(OZ2_PRELIM_TILE_K * OZ2_PRELIM_TILE_M), 0, stream, \
                       A, m, k, lda, A8i_high, lda8i, sftA, \
                       B, n, ldb, B8i_high, ldb8i, sftB, nan_flag, m_blks_prelim)
    if(svmask == 0u) {
        if(tA && !tB)       OZ2_PRELIM(true,  false, false);
        else if(!tA && !tB) OZ2_PRELIM(false, false, false);
        else if(!tA && tB)  OZ2_PRELIM(false, true,  false);
        else                OZ2_PRELIM(true,  true,  false);
    } else {
        if(tA && !tB)       OZ2_PRELIM(true,  false, true);
        else if(!tA && !tB) OZ2_PRELIM(false, false, true);
        else if(!tA && tB)  OZ2_PRELIM(false, true,  true);
        else                OZ2_PRELIM(true,  true,  true);
    }
#undef OZ2_PRELIM
    _pstop(_t_prelim);
    /* _t_extract remains 0: extraction is now fused into _t_prelim */

    if(svmask != 0u) {
        if(hipStreamSynchronize(stream) != hipSuccess) {
            (void)hipFreeAsync(ws, stream); return rocblaslt_status_internal_error;
        }
        uint32_t detected = 0u;
        if(hipMemcpy(&detected, nan_flag, sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess) {
            (void)hipFreeAsync(ws, stream); return rocblaslt_status_internal_error;
        }
        if(detected & svmask) {
            if(ws_owned) (void)hipFreeAsync(ws, stream);
            return rocblaslt_status_invalid_value;
        }
    }

    /* Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high */
    _pstart();
    hipblasLtMatmul(settings.handle, matmulDesc,
                    &one_i, A8i_high, layoutA, B8i_high, layoutB,
                    &zero_i, C32i, layoutCD, C32i, layoutCD, nullptr, nullptr, 0, stream);
    _pstop(_t_prelim_gemm);

    const float accu_log2P = h_accu_log2P_all[num_moduli - 2];
    _pstart();
    const unsigned sftA_m_blks = static_cast<unsigned>((m + 63) / 64);
    const unsigned sftA_n_blks = static_cast<unsigned>((n + 63) / 64);
    (void)hipMemsetAsync(row_max, 0, szRowMax * sizeof(int32_t), stream);
    hipLaunchKernelGGL(oz2_refine_sftA_partial_kernel, dim3(sftA_m_blks, sftA_n_blks), dim3(64), 0, stream,
                       C32i, m, n, ldc32i, row_max);
    hipLaunchKernelGGL(oz2_refine_sftA_apply_kernel, dim3(sftA_m_blks), dim3(64), 0, stream,
                       row_max, sftA, m, accu_log2P);
    hipLaunchKernelGGL(oz2_refine_sftB_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, stream,
                       C32i, m, n, ldc32i, sftB, accu_log2P);
    _pstop(_t_refine);

    const dim3 blk_scale(OZ2_SCALE_TILE_K * OZ2_SCALE_TILE_M);   /* dim3(256) */
    const unsigned m_y_blks = static_cast<unsigned>((m + OZ2_SCALE_TILE_M - 1) / OZ2_SCALE_TILE_M);
    const unsigned n_y_blks = static_cast<unsigned>((n + OZ2_SCALE_TILE_M - 1) / OZ2_SCALE_TILE_M);
    const dim3 gAB_scale(static_cast<unsigned>((k + OZ2_SCALE_TILE_K - 1) / OZ2_SCALE_TILE_K),
                         m_y_blks + n_y_blks);
    const dim3 blk_acc(64, 8);
    const dim3 grid_acc((m + 63) / 64, (n + 7) / 8);
    const size_t strideA8i = lda8i * cola8i;
    const size_t strideB8i = ldb8i * static_cast<size_t>(n);

    hipblasLtMatrixLayout_t layoutA_b  = nullptr;
    hipblasLtMatrixLayout_t layoutB_b  = nullptr;
    hipblasLtMatrixLayout_t layoutCD_b = nullptr;
    hipblasLtMatrixLayoutCreate(&layoutA_b,  HIP_R_8I,  static_cast<uint64_t>(k), static_cast<uint64_t>(m), static_cast<int64_t>(lda8i));
    hipblasLtMatrixLayoutCreate(&layoutB_b,  HIP_R_8I,  static_cast<uint64_t>(k), static_cast<uint64_t>(n), static_cast<int64_t>(ldb8i));
    hipblasLtMatrixLayoutCreate(&layoutCD_b, HIP_R_32I, static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<int64_t>(ldc32i));

    int32_t       batch_cur  = static_cast<int32_t>(chunk_size);
    const int64_t stride_A_b = static_cast<int64_t>(strideA8i);
    const int64_t stride_B_b = static_cast<int64_t>(strideB8i);
    const int64_t stride_C_b = static_cast<int64_t>(szC32i);
    hipblasLtMatrixLayoutSetAttribute(layoutA_b,  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &batch_cur,  sizeof(batch_cur));
    hipblasLtMatrixLayoutSetAttribute(layoutA_b,  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_A_b, sizeof(stride_A_b));
    hipblasLtMatrixLayoutSetAttribute(layoutB_b,  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &batch_cur,  sizeof(batch_cur));
    hipblasLtMatrixLayoutSetAttribute(layoutB_b,  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_B_b, sizeof(stride_B_b));
    hipblasLtMatrixLayoutSetAttribute(layoutCD_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &batch_cur,  sizeof(batch_cur));
    hipblasLtMatrixLayoutSetAttribute(layoutCD_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_C_b, sizeof(stride_C_b));

    for(unsigned scale_start = 0; scale_start < num_moduli; scale_start += scale_chunk_size) {
        const unsigned actual_scale = (scale_start + scale_chunk_size <= num_moduli)
                                      ? scale_chunk_size : (num_moduli - scale_start);
        _pstart();
        /* Scale kernel: OZ2_S_MAX=18, so actual_scale <= 18 always.
         * TRANS_A/TRANS_B are compile-time template params for branch-free
         * coalesced vs SHMEM-transposed paths.  No runtime fallback needed. */
#define OZ2_SCALE_LAUNCH(TC, TA, TB) \
        hipLaunchKernelGGL((oz2_scaleAB_kernel<(TC),(TA),(TB)>), gAB_scale, blk_scale, 0, stream, \
                           A, m, lda, A8i, lda8i, cola8i, sftA, \
                           B, n, ldb, B8i, ldb8i, sftB, k, scale_start, m_y_blks)
#define OZ2_SCALE_DISPATCH(TA, TB) \
        switch(actual_scale) { \
            case  1: OZ2_SCALE_LAUNCH( 1,(TA),(TB)); break; \
            case  2: OZ2_SCALE_LAUNCH( 2,(TA),(TB)); break; \
            case  3: OZ2_SCALE_LAUNCH( 3,(TA),(TB)); break; \
            case  4: OZ2_SCALE_LAUNCH( 4,(TA),(TB)); break; \
            case  5: OZ2_SCALE_LAUNCH( 5,(TA),(TB)); break; \
            case  6: OZ2_SCALE_LAUNCH( 6,(TA),(TB)); break; \
            case  7: OZ2_SCALE_LAUNCH( 7,(TA),(TB)); break; \
            case  8: OZ2_SCALE_LAUNCH( 8,(TA),(TB)); break; \
            case  9: OZ2_SCALE_LAUNCH( 9,(TA),(TB)); break; \
            case 10: OZ2_SCALE_LAUNCH(10,(TA),(TB)); break; \
            case 11: OZ2_SCALE_LAUNCH(11,(TA),(TB)); break; \
            case 12: OZ2_SCALE_LAUNCH(12,(TA),(TB)); break; \
            case 13: OZ2_SCALE_LAUNCH(13,(TA),(TB)); break; \
            case 14: OZ2_SCALE_LAUNCH(14,(TA),(TB)); break; \
            case 15: OZ2_SCALE_LAUNCH(15,(TA),(TB)); break; \
            case 16: OZ2_SCALE_LAUNCH(16,(TA),(TB)); break; \
            case 17: OZ2_SCALE_LAUNCH(17,(TA),(TB)); break; \
            case 18: OZ2_SCALE_LAUNCH(18,(TA),(TB)); break; \
            default: break; /* unreachable with OZ2_S_MAX=18 */ \
        }
        if(tA && !tB)       { OZ2_SCALE_DISPATCH(true,  false) }
        else if(!tA && !tB) { OZ2_SCALE_DISPATCH(false, false) }
        else if(!tA && tB)  { OZ2_SCALE_DISPATCH(false, true)  }
        else                { OZ2_SCALE_DISPATCH(true,  true)  }
#undef OZ2_SCALE_DISPATCH
#undef OZ2_SCALE_LAUNCH
        _pstop(_t_scale);

        for(unsigned gemm_local = 0; gemm_local < actual_scale; gemm_local += chunk_size) {
            const unsigned actual_gemm = (gemm_local + chunk_size <= actual_scale)
                                         ? chunk_size : (actual_scale - gemm_local);
            if(static_cast<int32_t>(actual_gemm) != batch_cur) {
                batch_cur = static_cast<int32_t>(actual_gemm);
                hipblasLtMatrixLayoutSetAttribute(layoutA_b,  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
                hipblasLtMatrixLayoutSetAttribute(layoutB_b,  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
                hipblasLtMatrixLayoutSetAttribute(layoutCD_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
            }
            const int8_t* const A8i_gemm = A8i + gemm_local * strideA8i;
            const int8_t* const B8i_gemm = B8i + gemm_local * strideB8i;
            _pstart();
            hipblasLtMatmul(settings.handle, matmulDesc,
                            &one_i, A8i_gemm, layoutA_b, B8i_gemm, layoutB_b,
                            &zero_i, C32i_batch, layoutCD_b, C32i_batch, layoutCD_b,
                            nullptr, nullptr, 0, stream);
            _pstop(_t_int8);

            const unsigned global_chunk_start = scale_start + gemm_local;
            const bool is_first = (global_chunk_start == 0);
            const bool is_last  = (global_chunk_start + actual_gemm == num_moduli);
            _pstart();
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
                        case  1: OZ2_FINALIZE(false,  1); break; case  2: OZ2_FINALIZE(false,  2); break;
                        case  3: OZ2_FINALIZE(false,  3); break; case  4: OZ2_FINALIZE(false,  4); break;
                        case  5: OZ2_FINALIZE(false,  5); break; case  6: OZ2_FINALIZE(false,  6); break;
                        case  7: OZ2_FINALIZE(false,  7); break; case  8: OZ2_FINALIZE(false,  8); break;
                        case  9: OZ2_FINALIZE(false,  9); break; case 10: OZ2_FINALIZE(false, 10); break;
                        case 11: OZ2_FINALIZE(false, 11); break; case 12: OZ2_FINALIZE(false, 12); break;
                        case 13: OZ2_FINALIZE(false, 13); break; case 14: OZ2_FINALIZE(false, 14); break;
                        case 15: OZ2_FINALIZE(false, 15); break; case 16: OZ2_FINALIZE(false, 16); break;
                        case 17: OZ2_FINALIZE(false, 17); break; case 18: OZ2_FINALIZE(false, 18); break;
                        default: hipLaunchKernelGGL((oz2_accum_finalize_kernel_rt<false>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd,
                                     *alpha, *beta, sftA, sftB, global_chunk_start, actual_gemm, is_first);
                    }
                } else {
                    switch(actual_gemm) {
                        case  1: OZ2_FINALIZE(true,  1); break; case  2: OZ2_FINALIZE(true,  2); break;
                        case  3: OZ2_FINALIZE(true,  3); break; case  4: OZ2_FINALIZE(true,  4); break;
                        case  5: OZ2_FINALIZE(true,  5); break; case  6: OZ2_FINALIZE(true,  6); break;
                        case  7: OZ2_FINALIZE(true,  7); break; case  8: OZ2_FINALIZE(true,  8); break;
                        case  9: OZ2_FINALIZE(true,  9); break; case 10: OZ2_FINALIZE(true, 10); break;
                        case 11: OZ2_FINALIZE(true, 11); break; case 12: OZ2_FINALIZE(true, 12); break;
                        case 13: OZ2_FINALIZE(true, 13); break; case 14: OZ2_FINALIZE(true, 14); break;
                        case 15: OZ2_FINALIZE(true, 15); break; case 16: OZ2_FINALIZE(true, 16); break;
                        case 17: OZ2_FINALIZE(true, 17); break; case 18: OZ2_FINALIZE(true, 18); break;
                        default: hipLaunchKernelGGL((oz2_accum_finalize_kernel_rt<true>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd,
                                     *alpha, *beta, sftA, sftB, global_chunk_start, actual_gemm, is_first);
                    }
                }
            } else {
                if(num_moduli <= 7u) {
                    switch(actual_gemm) {
                        case  1: OZ2_ACCUM(false,  1); break; case  2: OZ2_ACCUM(false,  2); break;
                        case  3: OZ2_ACCUM(false,  3); break; case  4: OZ2_ACCUM(false,  4); break;
                        case  5: OZ2_ACCUM(false,  5); break; case  6: OZ2_ACCUM(false,  6); break;
                        case  7: OZ2_ACCUM(false,  7); break; case  8: OZ2_ACCUM(false,  8); break;
                        case  9: OZ2_ACCUM(false,  9); break; case 10: OZ2_ACCUM(false, 10); break;
                        case 11: OZ2_ACCUM(false, 11); break; case 12: OZ2_ACCUM(false, 12); break;
                        case 13: OZ2_ACCUM(false, 13); break; case 14: OZ2_ACCUM(false, 14); break;
                        case 15: OZ2_ACCUM(false, 15); break; case 16: OZ2_ACCUM(false, 16); break;
                        case 17: OZ2_ACCUM(false, 17); break; case 18: OZ2_ACCUM(false, 18); break;
                        default: hipLaunchKernelGGL((oz2_chunk_accum_kernel_rt<false>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, m, n, ldc32i, global_chunk_start, actual_gemm, is_first);
                    }
                } else {
                    switch(actual_gemm) {
                        case  1: OZ2_ACCUM(true,  1); break; case  2: OZ2_ACCUM(true,  2); break;
                        case  3: OZ2_ACCUM(true,  3); break; case  4: OZ2_ACCUM(true,  4); break;
                        case  5: OZ2_ACCUM(true,  5); break; case  6: OZ2_ACCUM(true,  6); break;
                        case  7: OZ2_ACCUM(true,  7); break; case  8: OZ2_ACCUM(true,  8); break;
                        case  9: OZ2_ACCUM(true,  9); break; case 10: OZ2_ACCUM(true, 10); break;
                        case 11: OZ2_ACCUM(true, 11); break; case 12: OZ2_ACCUM(true, 12); break;
                        case 13: OZ2_ACCUM(true, 13); break; case 14: OZ2_ACCUM(true, 14); break;
                        case 15: OZ2_ACCUM(true, 15); break; case 16: OZ2_ACCUM(true, 16); break;
                        case 17: OZ2_ACCUM(true, 17); break; case 18: OZ2_ACCUM(true, 18); break;
                        default: hipLaunchKernelGGL((oz2_chunk_accum_kernel_rt<true>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, m, n, ldc32i, global_chunk_start, actual_gemm, is_first);
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
    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(layoutCD);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutA);

    if(ws_owned) {
        if(hipFreeAsync(ws, stream) != hipSuccess)
            return rocblaslt_status_internal_error;
    }

    if(_prof) {
        (void)hipEventRecord(_ev1, stream); (void)hipStreamSynchronize(stream);
        (void)hipEventElapsedTime(&_t_total, _ev_tot, _ev1);
        std::FILE* _f = std::fopen(_pf, "a");
        if(_f) {
            if(std::ftell(_f) == 0)
                std::fprintf(_f,
                    "m,n,k,num_moduli,scale_chunk_size,gemm_chunk_size,"
                    "workspace_bytes,"
                    "t_prelim_ms,t_prelim_gemm_ms,t_extract_ms,t_refine_ms,"
                    "t_scale_ms,t_int8_gemm_ms,t_accum_ms,"
                    "t_finalize_ms,t_total_ms\n");
            std::fprintf(_f,
                "%lld,%lld,%lld,%u,%u,%u,"
                "%llu,"
                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                (long long)m, (long long)n, (long long)k,
                num_moduli, scale_chunk_size, chunk_size,
                (unsigned long long)wsBytes,
                _t_prelim, _t_prelim_gemm, _t_extract, _t_refine,
                _t_scale, _t_int8, _t_accum, _t_finalize, _t_total);
            std::fclose(_f);
        }
        (void)hipEventDestroy(_ev_tot);
        (void)hipEventDestroy(_ev1);
        (void)hipEventDestroy(_ev0);
    }
    return rocblaslt_status_success;
}
