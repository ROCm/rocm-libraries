# AMD Matrix Instruction (MFMA / WMMA / SWMMA / SMFMAC) Support by Architecture

Summary of matrix-multiply-accumulate instructions documented in the ISA / shader
programming guides in this folder. Each table gives the **Opcode** (the VOP3P `OP`
field, bits [22:16]) and the **PDF page** where the instruction is defined. A secondary
page is given in each section header for the pseudocode and the opcode-encoding summary
table.

Instruction naming convention: `V_<engine>_<Cfmt>_<M>X<N>X<K>_<ABfmt>`
- **MFMA** – Matrix Fused Multiply-Add (CDNA / Instinct)
- **SMFMAC** – Sparse MFMA with Compression (structured sparsity, CDNA3+)
- **WMMA** – Wave Matrix Multiply-Accumulate (RDNA)
- **SWMMAC** – Sparse WMMA with Compression (RDNA4 / 5)

| Architecture | Engine(s) | Source PDF |
|---|---|---|
| CDNA1 | MFMA | `instinct-mi100-cdna1-shader-instruction-set-architecture.pdf` |
| CDNA2 | MFMA (+F64) | `instinct-mi200-cdna2-instruction-set-architecture.pdf` |
| CDNA3 | MFMA, SMFMAC | `instinct-mi300-cdna3-instruction-set-architecture.pdf` |
| CDNA4 | MFMA, SMFMAC | `instinct-mi350-cdna4-instruction-set-architecture.pdf` |
| RDNA3 | WMMA | `rdna3-shader-instruction-set-architecture-feb-2023_0.pdf` |
| RDNA3.5 | WMMA | `rdna35_instruction_set_architecture.pdf` |
| RDNA4 | WMMA, SWMMAC | `rdna4-instruction-set-architecture.pdf` |
| RDNA5 | WMMA, SWMMAC | `rdna5 Shader Programming Guide.pdf` |

> Opcodes are taken from each document's authoritative "VOP3P Opcodes" table. Page
> numbers are the **PDF page index** (not the printed page label).

---

## CDNA1 — (`instinct-mi100-cdna1-shader-instruction-set-architecture.pdf`)

Instruction list p.11; pseudocode pp.156–157; opcode table & definitions pp.258–260.

| Opcode | Instruction | Def. page |
|---:|---|---:|
| 65 | V_MFMA_F32_16X16X1F32 | 259 |
| 64 | V_MFMA_F32_32X32X1F32 | 259 |
| 66 | V_MFMA_F32_4X4X1F32 | 259 |
| 68 | V_MFMA_F32_32X32X2F32 | 259 |
| 69 | V_MFMA_F32_16X16X4F32 | 259 |
| 73 | V_MFMA_F32_16X16X4F16 | 259 |
| 77 | V_MFMA_F32_16X16X16F16 | 259 |
| 72 | V_MFMA_F32_32X32X4F16 | 259 |
| 76 | V_MFMA_F32_32X32X8F16 | 259 |
| 74 | V_MFMA_F32_4X4X4F16 | 259 |
| 105 | V_MFMA_F32_16X16X2BF16 | 260 |
| 109 | V_MFMA_F32_16X16X8BF16 | 260 |
| 104 | V_MFMA_F32_32X32X2BF16 | 260 |
| 108 | V_MFMA_F32_32X32X4BF16 | 260 |
| 107 | V_MFMA_F32_4X4X2BF16 | 260 |
| 81 | V_MFMA_I32_16X16X4I8 | 259 |
| 85 | V_MFMA_I32_16X16X16I8 | 260 |
| 80 | V_MFMA_I32_32X32X4I8 | 259 |
| 84 | V_MFMA_I32_32X32X8I8 | 260 |
| 82 | V_MFMA_I32_4X4X4I8 | 259 |

**Data types:** F32 (from F32/F16/BF16 inputs), I32 (from I8 inputs). No F64. BF16 is the
original 2-element (non-1K) form.

---

## CDNA2 — (`instinct-mi200-cdna2-instruction-set-architecture.pdf`)

Pseudocode pp.156–157; opcode table & definitions pp.252–254. Adds **FP64** matrix ops
and the **BF16 "1K"** (full-width bf16) variants relative to CDNA1.

| Opcode | Instruction | Def. page |
|---:|---|---:|
| 65 | V_MFMA_F32_16X16X1F32 | 253 |
| 64 | V_MFMA_F32_32X32X1F32 | 253 |
| 66 | V_MFMA_F32_4X4X1F32 | 253 |
| 68 | V_MFMA_F32_32X32X2F32 | 253 |
| 69 | V_MFMA_F32_16X16X4F32 | 253 |
| 73 | V_MFMA_F32_16X16X4F16 | 253 |
| 77 | V_MFMA_F32_16X16X16F16 | 254 |
| 72 | V_MFMA_F32_32X32X4F16 | 253 |
| 76 | V_MFMA_F32_32X32X8F16 | 254 |
| 74 | V_MFMA_F32_4X4X4F16 | 253 |
| 105 | V_MFMA_F32_16X16X2BF16 | 254 |
| 109 | V_MFMA_F32_16X16X8BF16 | 254 |
| 104 | V_MFMA_F32_32X32X2BF16 | 254 |
| 108 | V_MFMA_F32_32X32X4BF16 | 254 |
| 107 | V_MFMA_F32_4X4X2BF16 | 254 |
| 100 | V_MFMA_F32_16X16X4BF16_1K | 254 |
| 103 | V_MFMA_F32_16X16X16BF16_1K | 254 |
| 99 | V_MFMA_F32_32X32X4BF16_1K | 254 |
| 102 | V_MFMA_F32_32X32X8BF16_1K | 254 |
| 101 | V_MFMA_F32_4X4X4BF16_1K | 254 |
| 81 | V_MFMA_I32_16X16X4I8 | 254 |
| 85 | V_MFMA_I32_16X16X16I8 | 254 |
| 80 | V_MFMA_I32_32X32X4I8 | 254 |
| 84 | V_MFMA_I32_32X32X8I8 | 254 |
| 82 | V_MFMA_I32_4X4X4I8 | 254 |
| 110 | V_MFMA_F64_16X16X4F64 | 254 |
| 111 | V_MFMA_F64_4X4X4F64 | 254 |

**Data types:** F32, I32, **F64** (new), BF16 with 1K variants (new).

---

## CDNA3 — (`instinct-mi300-cdna3-instruction-set-architecture.pdf`)

Instruction definitions pp.269–289; opcode summary table (Table) pp.541–542. Adds
**XF32** (tf32-like), **FP8/BF8** inputs, and the **SMFMAC** sparse engine.

### MFMA
| Opcode | Instruction | Def. page |
|---:|---|---:|
| 62 | V_MFMA_F32_16X16X8_XF32 | 269 |
| 63 | V_MFMA_F32_32X32X4_XF32 | 270 |
| 65 | V_MFMA_F32_16X16X1_4B_F32 | 271 |
| 69 | V_MFMA_F32_16X16X4_F32 | 272 |
| 64 | V_MFMA_F32_32X32X1_2B_F32 | 270 |
| 68 | V_MFMA_F32_32X32X2_F32 | 271 |
| 66 | V_MFMA_F32_4X4X1_16B_F32 | 271 |
| 73 | V_MFMA_F32_16X16X4_4B_F16 | 273 |
| 77 | V_MFMA_F32_16X16X16_F16 | 274 |
| 72 | V_MFMA_F32_32X32X4_2B_F16 | 272 |
| 76 | V_MFMA_F32_32X32X8_F16 | 273 |
| 74 | V_MFMA_F32_4X4X4_16B_F16 | 273 |
| 94 | V_MFMA_F32_16X16X4_4B_BF16 | 277 |
| 97 | V_MFMA_F32_16X16X16_BF16 | 278 |
| 93 | V_MFMA_F32_32X32X4_2B_BF16 | 276 |
| 96 | V_MFMA_F32_32X32X8_BF16 | 278 |
| 95 | V_MFMA_F32_4X4X4_16B_BF16 | 277 |
| 112 | V_MFMA_F32_16X16X32_BF8_BF8 | 282 |
| 113 | V_MFMA_F32_16X16X32_BF8_FP8 | 283 |
| 114 | V_MFMA_F32_16X16X32_FP8_BF8 | 283 |
| 115 | V_MFMA_F32_16X16X32_FP8_FP8 | 283 |
| 116 | V_MFMA_F32_32X32X16_BF8_BF8 | 284 |
| 117 | V_MFMA_F32_32X32X16_BF8_FP8 | 284 |
| 118 | V_MFMA_F32_32X32X16_FP8_BF8 | 285 |
| 119 | V_MFMA_F32_32X32X16_FP8_FP8 | 285 |
| 81 | V_MFMA_I32_16X16X4_4B_I8 | 275 |
| 87 | V_MFMA_I32_16X16X32_I8 | 276 |
| 80 | V_MFMA_I32_32X32X4_2B_I8 | 274 |
| 86 | V_MFMA_I32_32X32X16_I8 | 275 |
| 82 | V_MFMA_I32_4X4X4_16B_I8 | 275 |
| 110 | V_MFMA_F64_16X16X4_F64 | 281 |
| 111 | V_MFMA_F64_4X4X4_4B_F64 | 282 |

### SMFMAC (sparse)
| Opcode | Instruction | Def. page |
|---:|---|---:|
| 98 | V_SMFMAC_F32_16X16X32_F16 | 278 |
| 100 | V_SMFMAC_F32_32X32X16_F16 | 279 |
| 102 | V_SMFMAC_F32_16X16X32_BF16 | 279 |
| 104 | V_SMFMAC_F32_32X32X16_BF16 | 280 |
| 106 | V_SMFMAC_I32_16X16X64_I8 | 280 |
| 108 | V_SMFMAC_I32_32X32X32_I8 | 281 |
| 120 | V_SMFMAC_F32_16X16X64_BF8_BF8 | 285 |
| 121 | V_SMFMAC_F32_16X16X64_BF8_FP8 | 286 |
| 122 | V_SMFMAC_F32_16X16X64_FP8_BF8 | 286 |
| 123 | V_SMFMAC_F32_16X16X64_FP8_FP8 | 287 |
| 124 | V_SMFMAC_F32_32X32X32_BF8_BF8 | 287 |
| 125 | V_SMFMAC_F32_32X32X32_BF8_FP8 | 288 |
| 126 | V_SMFMAC_F32_32X32X32_FP8_BF8 | 288 |
| 127 | V_SMFMAC_F32_32X32X32_FP8_FP8 | 289 |

**Data types:** F32, F64, I32/I8, **XF32** (CDNA3-only among these parts), **FP8/BF8**;
sparse via SMFMAC.

---

## CDNA4 —  (`instinct-mi350-cdna4-instruction-set-architecture.pdf`)

Instruction definitions pp.286–316; opcode summary Table 90 pp.596–597. Adds the
**F8F6F4** mixed narrow-precision ops (with **scale** variants) and wider-K formats.
**Note:** CDNA4 does **not** provide XF32 — those opcode slots (62/63) are removed.

### MFMA
| Opcode | Instruction | Def. page |
|---:|---|---:|
| 65 | V_MFMA_F32_16X16X1_4B_F32 | 293 |
| 69 | V_MFMA_F32_16X16X4_F32 | 295 |
| 64 | V_MFMA_F32_32X32X1_2B_F32 | 292 |
| 68 | V_MFMA_F32_32X32X2_F32 | 294 |
| 66 | V_MFMA_F32_4X4X1_16B_F32 | 293 |
| 73 | V_MFMA_F32_16X16X4_4B_F16 | 296 |
| 77 | V_MFMA_F32_16X16X16_F16 | 298 |
| 84 | V_MFMA_F32_16X16X32_F16 | 301 |
| 72 | V_MFMA_F32_32X32X4_2B_F16 | 296 |
| 76 | V_MFMA_F32_32X32X8_F16 | 298 |
| 85 | V_MFMA_F32_32X32X16_F16 | 302 |
| 74 | V_MFMA_F32_4X4X4_16B_F16 | 297 |
| 94 | V_MFMA_F32_16X16X4_4B_BF16 | 304 |
| 97 | V_MFMA_F32_16X16X16_BF16 | 306 |
| 53 | V_MFMA_F32_16X16X32_BF16 | 288 |
| 93 | V_MFMA_F32_32X32X4_2B_BF16 | 304 |
| 96 | V_MFMA_F32_32X32X8_BF16 | 305 |
| 55 | V_MFMA_F32_32X32X16_BF16 | 289 |
| 95 | V_MFMA_F32_4X4X4_16B_BF16 | 305 |
| 112 | V_MFMA_F32_16X16X32_BF8_BF8 | 310 |
| 113 | V_MFMA_F32_16X16X32_BF8_FP8 | 310 |
| 114 | V_MFMA_F32_16X16X32_FP8_BF8 | 310 |
| 115 | V_MFMA_F32_16X16X32_FP8_FP8 | 311 |
| 116 | V_MFMA_F32_32X32X16_BF8_BF8 | 311 |
| 117 | V_MFMA_F32_32X32X16_BF8_FP8 | 312 |
| 118 | V_MFMA_F32_32X32X16_FP8_BF8 | 312 |
| 119 | V_MFMA_F32_32X32X16_FP8_FP8 | 312 |
| 45 | V_MFMA_F32_16X16X128_F8F6F4 | 286 |
| 46 | V_MFMA_F32_32X32X64_F8F6F4 | 286 |
| 81 | V_MFMA_I32_16X16X4_4B_I8 | 300 |
| 87 | V_MFMA_I32_16X16X32_I8 | 302 |
| 54 | V_MFMA_I32_16X16X64_I8 | 289 |
| 80 | V_MFMA_I32_32X32X4_2B_I8 | 299 |
| 86 | V_MFMA_I32_32X32X16_I8 | 302 |
| 56 | V_MFMA_I32_32X32X32_I8 | 289 |
| 82 | V_MFMA_I32_4X4X4_16B_I8 | 300 |
| 110 | V_MFMA_F64_16X16X4_F64 | 309 |
| 111 | V_MFMA_F64_4X4X4_4B_F64 | 309 |

### MFMA with scale (F8F6F4 microscaling)
Encoding uses the scale form; described pp.59, 64 (definitions p.286).

| Instruction | Ref. page |
|---|---:|
| V_MFMA_SCALE_F32_16X16X128_F8F6F4 | 59, 64 |
| V_MFMA_SCALE_F32_32X32X64_F8F6F4 | 59, 64 |

### SMFMAC (sparse)
| Opcode | Instruction | Def. page |
|---:|---|---:|
| 98 | V_SMFMAC_F32_16X16X32_F16 | 306 |
| 100 | V_SMFMAC_F32_32X32X16_F16 | 306 |
| 90 | V_SMFMAC_F32_16X16X64_F16 | 303 |
| 91 | V_SMFMAC_F32_32X32X32_F16 | 303 |
| 102 | V_SMFMAC_F32_16X16X32_BF16 | 307 |
| 104 | V_SMFMAC_F32_32X32X16_BF16 | 307 |
| 57 | V_SMFMAC_F32_16X16X64_BF16 | 290 |
| 70 | V_SMFMAC_F32_32X32X32_BF16 | 295 |
| 106 | V_SMFMAC_I32_16X16X64_I8 | 308 |
| 108 | V_SMFMAC_I32_32X32X32_I8 | 308 |
| 58 | V_SMFMAC_I32_16X16X128_I8 | 290 |
| 71 | V_SMFMAC_I32_32X32X64_I8 | 295 |
| 120 | V_SMFMAC_F32_16X16X64_BF8_BF8 | 313 |
| 121 | V_SMFMAC_F32_16X16X64_BF8_FP8 | 313 |
| 122 | V_SMFMAC_F32_16X16X64_FP8_BF8 | 314 |
| 123 | V_SMFMAC_F32_16X16X64_FP8_FP8 | 314 |
| 124 | V_SMFMAC_F32_32X32X32_BF8_BF8 | 315 |
| 125 | V_SMFMAC_F32_32X32X32_BF8_FP8 | 315 |
| 126 | V_SMFMAC_F32_32X32X32_FP8_BF8 | 316 |
| 127 | V_SMFMAC_F32_32X32X32_FP8_FP8 | 316 |
| 59 | V_SMFMAC_F32_16X16X128_BF8_BF8 | 291 |
| 60 | V_SMFMAC_F32_16X16X128_BF8_FP8 | 291 |
| 61 | V_SMFMAC_F32_16X16X128_FP8_BF8 | 292 |
| 67 | V_SMFMAC_F32_16X16X128_FP8_FP8 | 294 |
| 75 | V_SMFMAC_F32_32X32X64_BF8_BF8 | 297 |
| 78 | V_SMFMAC_F32_32X32X64_BF8_FP8 | 298 |
| 79 | V_SMFMAC_F32_32X32X64_FP8_BF8 | 299 |
| 83 | V_SMFMAC_F32_32X32X64_FP8_FP8 | 301 |

**Data types:** F32, F64, I32/I8, FP8/BF8, and **F8F6F4** mixed narrow formats with
per-block **scaling** (microscaling / MX). **No XF32.**

---

## RDNA3 (`rdna3-shader-instruction-set-architecture-feb-2023_0.pdf`)

First RDNA generation with WMMA. Encoding p.75; pseudocode pp.82–84; definitions
pp.367–368.

| Opcode | Instruction | Def. page |
|---:|---|---:|
| 64 | V_WMMA_F32_16X16X16_F16 | 367 |
| 65 | V_WMMA_F32_16X16X16_BF16 | 367 |
| 66 | V_WMMA_F16_16X16X16_F16 | 367 |
| 67 | V_WMMA_BF16_16X16X16_BF16 | 368 |
| 68 | V_WMMA_I32_16X16X16_IU8 | 368 |
| 69 | V_WMMA_I32_16X16X16_IU4 | 368 |

**Data types:** F16, BF16, IU8, IU4. Single 16x16x16 shape. No sparsity.

---

## RDNA3.5 (`rdna35_instruction_set_architecture.pdf`)

Same WMMA set and opcodes as RDNA3. Encoding p.77; pseudocode pp.84–86; definitions
pp.390–392.

| Opcode | Instruction | Def. page |
|---:|---|---:|
| 64 | V_WMMA_F32_16X16X16_F16 | 390 |
| 65 | V_WMMA_F32_16X16X16_BF16 | 390 |
| 66 | V_WMMA_F16_16X16X16_F16 | 391 |
| 67 | V_WMMA_BF16_16X16X16_BF16 | 391 |
| 68 | V_WMMA_I32_16X16X16_IU8 | 392 |
| 69 | V_WMMA_I32_16X16X16_IU4 | 392 |

**Data types:** F16, BF16, IU8, IU4. Identical WMMA capability to RDNA3.

---

## RDNA4 (`rdna4-instruction-set-architecture.pdf`)

Adds **FP8/BF8** WMMA, larger-K IU4, and the **SWMMAC** sparse engine. Encoding p.100;
opcode table (Table 98) p.198; definitions pp.409–419.

### WMMA
| Opcode | Instruction | Def. page |
|---:|---|---:|
| 64 | V_WMMA_F32_16X16X16_F16 | 409 |
| 65 | V_WMMA_F32_16X16X16_BF16 | 409 |
| 66 | V_WMMA_F16_16X16X16_F16 | 410 |
| 67 | V_WMMA_BF16_16X16X16_BF16 | 410 |
| 68 | V_WMMA_I32_16X16X16_IU8 | 411 |
| 69 | V_WMMA_I32_16X16X16_IU4 | 411 |
| 70 | V_WMMA_F32_16X16X16_FP8_FP8 | 411 |
| 71 | V_WMMA_F32_16X16X16_FP8_BF8 | 412 |
| 72 | V_WMMA_F32_16X16X16_BF8_FP8 | 412 |
| 73 | V_WMMA_F32_16X16X16_BF8_BF8 | 413 |
| 74 | V_WMMA_I32_16X16X32_IU4 | 413 |

### SWMMAC (sparse)
| Opcode | Instruction | Def. page |
|---:|---|---:|
| 80 | V_SWMMAC_F32_16X16X32_F16 | 414 |
| 81 | V_SWMMAC_F32_16X16X32_BF16 | 414 |
| 82 | V_SWMMAC_F16_16X16X32_F16 | 415 |
| 83 | V_SWMMAC_BF16_16X16X32_BF16 | 415 |
| 84 | V_SWMMAC_I32_16X16X32_IU8 | 416 |
| 85 | V_SWMMAC_I32_16X16X32_IU4 | 416 |
| 86 | V_SWMMAC_I32_16X16X64_IU4 | 417 |
| 87 | V_SWMMAC_F32_16X16X32_FP8_FP8 | 417 |
| 88 | V_SWMMAC_F32_16X16X32_FP8_BF8 | 418 |
| 89 | V_SWMMAC_F32_16X16X32_BF8_FP8 | 419 |
| 90 | V_SWMMAC_F32_16X16X32_BF8_BF8 | 419 |

**Data types:** F16, BF16, IU8, IU4, **FP8/BF8** (new); **sparsity via SWMMAC** (new).
No XF32.

---

## CNDA5 (`Shader Programming Guide.pdf`)

CNDA5 uses the RDNA-style **WMMA / SWMMAC** naming (not MFMA). It expands supported
shapes and formats, adds **F64 WMMA**, **F8F6F4/FP4** with per-block scaling
(`V_WMMA_SCALE*` / `V_WMMA_LD_SCALE*`). Instruction list pp.15–17; register/operation
details pp.154–168; opcode summary table pp.330–331.

> **CDNA5 explicitly removes TF32/XF32 WMMA support** (p.10) and adds WMMA MXFP
> block-size-16 support.

### WMMA (dense)
| Opcode | Instruction | Def. page |
|---:|---|---:|
| 93 | V_WMMA_F32_16X16X4_F32 | 154 |
| 96 | V_WMMA_F32_16X16X32_F16 | 154 |
| 97 | V_WMMA_F16_16X16X32_F16 | 154 |
| 98 | V_WMMA_F32_16X16X32_BF16 | 154 |
| 99 | V_WMMA_BF16_16X16X32_BF16 | 154 |
| 100 | V_WMMA_BF16F32_16X16X32_BF16 | 154 |
| 114 | V_WMMA_I32_16X16X64_IU8 | 154 |
| 106 | V_WMMA_F32_16X16X64_FP8_FP8 | 154 |
| 107 | V_WMMA_F32_16X16X64_FP8_BF8 | 154 |
| 108 | V_WMMA_F32_16X16X64_BF8_FP8 | 154 |
| 109 | V_WMMA_F32_16X16X64_BF8_BF8 | 154 |
| 110 | V_WMMA_F16_16X16X64_FP8_FP8 | 154 |
| 111 | V_WMMA_F16_16X16X64_FP8_BF8 | 154 |
| 112 | V_WMMA_F16_16X16X64_BF8_FP8 | 154 |
| 113 | V_WMMA_F16_16X16X64_BF8_BF8 | 154 |
| 128 | V_WMMA_F32_16X16X128_FP8_FP8 | 154 |
| 129 | V_WMMA_F32_16X16X128_FP8_BF8 | 154 |
| 130 | V_WMMA_F32_16X16X128_BF8_FP8 | 154 |
| 131 | V_WMMA_F32_16X16X128_BF8_BF8 | 154 |
| 132 | V_WMMA_F16_16X16X128_FP8_FP8 | 154 |
| 133 | V_WMMA_F16_16X16X128_FP8_BF8 | 154 |
| 134 | V_WMMA_F16_16X16X128_BF8_FP8 | 154 |
| 135 | V_WMMA_F16_16X16X128_BF8_BF8 | 154 |
| 51 | V_WMMA_F32_16X16X128_F8F6F4 | 155 |
| 136 | V_WMMA_F32_32X16X128_F4 | 155 |
| 91 | V_WMMA_F64_16X16X4_F64 | 154 |

### SWMMAC (sparse)
| Opcode | Instruction | Def. page |
|---:|---|---:|
| 101 | V_SWMMAC_F32_16X16X64_F16 | 154 |
| 102 | V_SWMMAC_F32_16X16X64_BF16 | 154 |
| 103 | V_SWMMAC_F16_16X16X64_F16 | 154 |
| 104 | V_SWMMAC_BF16_16X16X64_BF16 | 154 |
| 105 | V_SWMMAC_BF16F32_16X16X64_BF16 | 154 |
| 123 | V_SWMMAC_I32_16X16X128_IU8 | 154 |
| 115 | V_SWMMAC_F32_16X16X128_FP8_FP8 | 154 |
| 116 | V_SWMMAC_F32_16X16X128_FP8_BF8 | 154 |
| 117 | V_SWMMAC_F32_16X16X128_BF8_FP8 | 154 |
| 118 | V_SWMMAC_F32_16X16X128_BF8_BF8 | 154 |
| 119 | V_SWMMAC_F16_16X16X128_FP8_FP8 | 154 |
| 120 | V_SWMMAC_F16_16X16X128_FP8_BF8 | 154 |
| 121 | V_SWMMAC_F16_16X16X128_BF8_FP8 | 154 |
| 122 | V_SWMMAC_F16_16X16X128_BF8_BF8 | 154 |

### WMMA scale helpers (F8F6F4 / microscaling — separate encoding)
| Instruction | Ref. page |
|---|---:|
| V_WMMA_SCALE_F32_16X16X128_F8F6F4 | 163, 166 |
| V_WMMA_SCALE_F32_32X16X128_F4 | 163, 166 |
| V_WMMA_SCALE16_F32_16X16X128_F8F6F4 | 163, 166 |
| V_WMMA_SCALE16_F32_32X16X128_F4 | 163, 166 |
| V_WMMA_LD_SCALE_B32 | 167 |
| V_WMMA_LD_SCALE_PAIRED_B32 | 167 |
| V_WMMA_LD_SCALE16_B64 | 167 |
| V_WMMA_LD_SCALE16_PAIRED_B64 | 167 |

**Data types:** F16, BF16, F32, **F64**, IU8, IU4, FP8/BF8, and **F8F6F4 & FP4** with
per-block **scale** operands. **No XF32/TF32.** (p.15 also lists reduced-K legacy-shape
16X16X16 / 16X16X32 variants for compatibility.)

---

## Cross-architecture capability summary

| Feature | CDNA1 | CDNA2 | CDNA3 | CDNA4 | RDNA3 | RDNA3.5 | RDNA4 | RDNA5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Engine | MFMA | MFMA | MFMA | MFMA | WMMA | WMMA | WMMA | WMMA |
| F32 accum | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| F16 in | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| BF16 in | ✅ | ✅ (1K) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT4 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| FP64 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **XF32 / TF32** | ❌ | ❌ | **✅** | ❌ | ❌ | ❌ | ❌ | ❌ |
| FP8/BF8 | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| F8F6F4 / FP4 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Block scaling (MX) | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Sparse (SMFMAC/SWMMAC) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |

**XF32 note:** XF32 (AMD's tf32-equivalent, `V_MFMA_F32_16X16X8_XF32` = opcode 62 and
`V_MFMA_F32_32X32X4_XF32` = opcode 63) is documented **only in CDNA3**. CDNA4
removes it, and the CDNA5 guide explicitly states TF32/XF32 WMMA support is
removed. RDNA parts never had it.

*Opcodes are the VOP3P `OP` field from each ISA's opcode table. Page references are to
the PDF page index (not the printed page label).*
