# AMD Matrix Instruction (MFMA / WMMA / SWMMA / SMFMAC) Support by Architecture

A convenience index of AMD matrix instructions. The public XML instruction and
encoding records are the source of truth. The tables are not exhaustive and do
not imply that rocKE exposes every instruction.

## Contents

- [Sources and reproduction](#sources-and-reproduction)
- [CDNA1](#cdna1)
- [CDNA2](#cdna2)
- [CDNA3](#cdna3)
- [CDNA4](#cdna4)
- [RDNA3](#rdna3)
- [RDNA3.5](#rdna35)
- [RDNA4](#rdna4)
- [CDNA5](#cdna5)
- [Cross-architecture capability summary](#cross-architecture-capability-summary)

## Sources and reproduction

The [official AMD machine-readable ISA page](https://gpuopen.com/machine-readable-isa/)
provides the [XML archive](https://gpuopen.com/download/machine-readable-isa/latest/).
The archive checked on 2026-09-23 has XML `Document/ReleaseDate` **2026-08-03**,
schema **1.2.0**, and SHA-256
`82404f1126761b7877595b622afa7e1f311f2f41e89a3abe9aaf8ad045c082e2`.
The `latest` URL changes over time; verify the hash when reproducing this index.
All 232 numbered mnemonic/opcode rows below match the corresponding XML records,
including official aliases. This check does not establish table completeness.

| Architecture / target | Engine(s) | XML file in the public archive |
|---|---|---|
| CDNA1 | MFMA | `amdgpu_isa_cdna1.xml` |
| CDNA2 | MFMA | `amdgpu_isa_cdna2.xml` |
| CDNA3 / gfx942 | MFMA, SMFMAC | `amdgpu_isa_cdna3.xml` |
| CDNA4 / gfx950 | MFMA, SMFMAC | `amdgpu_isa_cdna4.xml` |
| RDNA3 | WMMA | `amdgpu_isa_rdna3.xml` |
| RDNA3.5 / gfx1151 | WMMA | `amdgpu_isa_rdna3_5.xml` |
| RDNA4 / gfx1201 | WMMA, SWMMAC | `amdgpu_isa_rdna4.xml` |
| CDNA5 / gfx1250 | WMMA, SWMMAC | `amdgpu_isa_cdna5.xml` |

The last XML identifies itself as **AMD CDNA 5** in
`ISA/Architecture/ArchitectureName`. rocKE's gfx1250 catalog entry uses
`family: "cdna"` and `target_family: "gfx12_cdna"`.

To locate a row, find its mnemonic under
`ISA/Instructions/Instruction/InstructionName` (or an official alias), then read
`InstructionEncodings/InstructionEncoding/Opcode` and `EncodingName`.
The encoding layout is under `ISA/Encodings/Encoding`. These locators replace
unversioned PDF page references. Scale helpers have separate encoding records.
For gfx1250, `ENC_VOP3P` has an eight-bit `OP` field at **[23:16]**; use each
architecture's encoding record rather than a universal seven-bit mask.

Compiler target restrictions are cross-checked against public
[ROCm LLVM commit 0586239bb02ac04051622636bcca1c6d34642ad2](https://github.com/ROCm/llvm-project/commit/0586239bb02ac04051622636bcca1c6d34642ad2):

- [XF32 selection predicates](https://github.com/ROCm/llvm-project/blob/0586239bb02ac04051622636bcca1c6d34642ad2/llvm/lib/Target/AMDGPU/VOP3PInstructions.td#L1405)
  and [gfx942 features](https://github.com/ROCm/llvm-project/blob/0586239bb02ac04051622636bcca1c6d34642ad2/llvm/lib/Target/AMDGPU/AMDGPU.td#L2106).
- [F64 WMMA selection predicate](https://github.com/ROCm/llvm-project/blob/0586239bb02ac04051622636bcca1c6d34642ad2/llvm/lib/Target/AMDGPU/VOP3PInstructions.td#L2415)
  and [gfx1250/gfx1251 features](https://github.com/ROCm/llvm-project/blob/0586239bb02ac04051622636bcca1c6d34642ad2/llvm/lib/Target/AMDGPU/AMDGPU.td#L2516).
- [VOP3P encoding](https://github.com/ROCm/llvm-project/blob/0586239bb02ac04051622636bcca1c6d34642ad2/llvm/lib/Target/AMDGPU/VOPInstructions.td#L653).

Instruction names follow `V_<engine>_<Cfmt>_<M>X<N>X<K>_<ABfmt>`.
MFMA and WMMA are dense matrix operations; SMFMAC and SWMMAC are their sparse
counterparts. Target support depends on processor features and selection
predicates, not just the presence of an intrinsic.

---

## CDNA1

| Opcode | Instruction |
|---:|---|
| 65 | V_MFMA_F32_16X16X1F32 |
| 64 | V_MFMA_F32_32X32X1F32 |
| 66 | V_MFMA_F32_4X4X1F32 |
| 68 | V_MFMA_F32_32X32X2F32 |
| 69 | V_MFMA_F32_16X16X4F32 |
| 73 | V_MFMA_F32_16X16X4F16 |
| 77 | V_MFMA_F32_16X16X16F16 |
| 72 | V_MFMA_F32_32X32X4F16 |
| 76 | V_MFMA_F32_32X32X8F16 |
| 74 | V_MFMA_F32_4X4X4F16 |
| 105 | V_MFMA_F32_16X16X2BF16 |
| 109 | V_MFMA_F32_16X16X8BF16 |
| 104 | V_MFMA_F32_32X32X2BF16 |
| 108 | V_MFMA_F32_32X32X4BF16 |
| 107 | V_MFMA_F32_4X4X2BF16 |
| 81 | V_MFMA_I32_16X16X4I8 |
| 85 | V_MFMA_I32_16X16X16I8 |
| 80 | V_MFMA_I32_32X32X4I8 |
| 84 | V_MFMA_I32_32X32X8I8 |
| 82 | V_MFMA_I32_4X4X4I8 |

**Data types:** F32 (from F32/F16/BF16 inputs), I32 (from I8 inputs). No F64. BF16 is the
original 2-element (non-1K) form.

---

## CDNA2

Adds **FP64** matrix ops
and the **BF16 "1K"** (full-width bf16) variants relative to CDNA1.

| Opcode | Instruction |
|---:|---|
| 65 | V_MFMA_F32_16X16X1F32 |
| 64 | V_MFMA_F32_32X32X1F32 |
| 66 | V_MFMA_F32_4X4X1F32 |
| 68 | V_MFMA_F32_32X32X2F32 |
| 69 | V_MFMA_F32_16X16X4F32 |
| 73 | V_MFMA_F32_16X16X4F16 |
| 77 | V_MFMA_F32_16X16X16F16 |
| 72 | V_MFMA_F32_32X32X4F16 |
| 76 | V_MFMA_F32_32X32X8F16 |
| 74 | V_MFMA_F32_4X4X4F16 |
| 105 | V_MFMA_F32_16X16X2BF16 |
| 109 | V_MFMA_F32_16X16X8BF16 |
| 104 | V_MFMA_F32_32X32X2BF16 |
| 108 | V_MFMA_F32_32X32X4BF16 |
| 107 | V_MFMA_F32_4X4X2BF16 |
| 100 | V_MFMA_F32_16X16X4BF16_1K |
| 103 | V_MFMA_F32_16X16X16BF16_1K |
| 99 | V_MFMA_F32_32X32X4BF16_1K |
| 102 | V_MFMA_F32_32X32X8BF16_1K |
| 101 | V_MFMA_F32_4X4X4BF16_1K |
| 81 | V_MFMA_I32_16X16X4I8 |
| 85 | V_MFMA_I32_16X16X16I8 |
| 80 | V_MFMA_I32_32X32X4I8 |
| 84 | V_MFMA_I32_32X32X8I8 |
| 82 | V_MFMA_I32_4X4X4I8 |
| 110 | V_MFMA_F64_16X16X4F64 |
| 111 | V_MFMA_F64_4X4X4F64 |

**Data types:** F32, I32, **F64** (new), BF16 with 1K variants (new).

---

## CDNA3

Adds **XF32** (tf32-like), **FP8/BF8** inputs, and the **SMFMAC** sparse engine.

### MFMA
| Opcode | Instruction |
|---:|---|
| 62 | V_MFMA_F32_16X16X8_XF32 |
| 63 | V_MFMA_F32_32X32X4_XF32 |
| 65 | V_MFMA_F32_16X16X1_4B_F32 |
| 69 | V_MFMA_F32_16X16X4_F32 |
| 64 | V_MFMA_F32_32X32X1_2B_F32 |
| 68 | V_MFMA_F32_32X32X2_F32 |
| 66 | V_MFMA_F32_4X4X1_16B_F32 |
| 73 | V_MFMA_F32_16X16X4_4B_F16 |
| 77 | V_MFMA_F32_16X16X16_F16 |
| 72 | V_MFMA_F32_32X32X4_2B_F16 |
| 76 | V_MFMA_F32_32X32X8_F16 |
| 74 | V_MFMA_F32_4X4X4_16B_F16 |
| 94 | V_MFMA_F32_16X16X4_4B_BF16 |
| 97 | V_MFMA_F32_16X16X16_BF16 |
| 93 | V_MFMA_F32_32X32X4_2B_BF16 |
| 96 | V_MFMA_F32_32X32X8_BF16 |
| 95 | V_MFMA_F32_4X4X4_16B_BF16 |
| 112 | V_MFMA_F32_16X16X32_BF8_BF8 |
| 113 | V_MFMA_F32_16X16X32_BF8_FP8 |
| 114 | V_MFMA_F32_16X16X32_FP8_BF8 |
| 115 | V_MFMA_F32_16X16X32_FP8_FP8 |
| 116 | V_MFMA_F32_32X32X16_BF8_BF8 |
| 117 | V_MFMA_F32_32X32X16_BF8_FP8 |
| 118 | V_MFMA_F32_32X32X16_FP8_BF8 |
| 119 | V_MFMA_F32_32X32X16_FP8_FP8 |
| 81 | V_MFMA_I32_16X16X4_4B_I8 |
| 87 | V_MFMA_I32_16X16X32_I8 |
| 80 | V_MFMA_I32_32X32X4_2B_I8 |
| 86 | V_MFMA_I32_32X32X16_I8 |
| 82 | V_MFMA_I32_4X4X4_16B_I8 |
| 110 | V_MFMA_F64_16X16X4_F64 |
| 111 | V_MFMA_F64_4X4X4_4B_F64 |

### SMFMAC (sparse)
| Opcode | Instruction |
|---:|---|
| 98 | V_SMFMAC_F32_16X16X32_F16 |
| 100 | V_SMFMAC_F32_32X32X16_F16 |
| 102 | V_SMFMAC_F32_16X16X32_BF16 |
| 104 | V_SMFMAC_F32_32X32X16_BF16 |
| 106 | V_SMFMAC_I32_16X16X64_I8 |
| 108 | V_SMFMAC_I32_32X32X32_I8 |
| 120 | V_SMFMAC_F32_16X16X64_BF8_BF8 |
| 121 | V_SMFMAC_F32_16X16X64_BF8_FP8 |
| 122 | V_SMFMAC_F32_16X16X64_FP8_BF8 |
| 123 | V_SMFMAC_F32_16X16X64_FP8_FP8 |
| 124 | V_SMFMAC_F32_32X32X32_BF8_BF8 |
| 125 | V_SMFMAC_F32_32X32X32_BF8_FP8 |
| 126 | V_SMFMAC_F32_32X32X32_FP8_BF8 |
| 127 | V_SMFMAC_F32_32X32X32_FP8_FP8 |

**Data types:** F32, F64, I32/I8, **XF32** (CDNA3-only among these parts), **FP8/BF8**;
sparse via SMFMAC.

---

## CDNA4

Adds **F8F6F4** mixed narrow-precision ops (with **scale** variants) and wider-K formats.
**Note:** CDNA4 does **not** provide XF32 — those opcode slots (62/63) are removed.

### MFMA
| Opcode | Instruction |
|---:|---|
| 65 | V_MFMA_F32_16X16X1_4B_F32 |
| 69 | V_MFMA_F32_16X16X4_F32 |
| 64 | V_MFMA_F32_32X32X1_2B_F32 |
| 68 | V_MFMA_F32_32X32X2_F32 |
| 66 | V_MFMA_F32_4X4X1_16B_F32 |
| 73 | V_MFMA_F32_16X16X4_4B_F16 |
| 77 | V_MFMA_F32_16X16X16_F16 |
| 84 | V_MFMA_F32_16X16X32_F16 |
| 72 | V_MFMA_F32_32X32X4_2B_F16 |
| 76 | V_MFMA_F32_32X32X8_F16 |
| 85 | V_MFMA_F32_32X32X16_F16 |
| 74 | V_MFMA_F32_4X4X4_16B_F16 |
| 94 | V_MFMA_F32_16X16X4_4B_BF16 |
| 97 | V_MFMA_F32_16X16X16_BF16 |
| 53 | V_MFMA_F32_16X16X32_BF16 |
| 93 | V_MFMA_F32_32X32X4_2B_BF16 |
| 96 | V_MFMA_F32_32X32X8_BF16 |
| 55 | V_MFMA_F32_32X32X16_BF16 |
| 95 | V_MFMA_F32_4X4X4_16B_BF16 |
| 112 | V_MFMA_F32_16X16X32_BF8_BF8 |
| 113 | V_MFMA_F32_16X16X32_BF8_FP8 |
| 114 | V_MFMA_F32_16X16X32_FP8_BF8 |
| 115 | V_MFMA_F32_16X16X32_FP8_FP8 |
| 116 | V_MFMA_F32_32X32X16_BF8_BF8 |
| 117 | V_MFMA_F32_32X32X16_BF8_FP8 |
| 118 | V_MFMA_F32_32X32X16_FP8_BF8 |
| 119 | V_MFMA_F32_32X32X16_FP8_FP8 |
| 45 | V_MFMA_F32_16X16X128_F8F6F4 |
| 46 | V_MFMA_F32_32X32X64_F8F6F4 |
| 81 | V_MFMA_I32_16X16X4_4B_I8 |
| 87 | V_MFMA_I32_16X16X32_I8 |
| 54 | V_MFMA_I32_16X16X64_I8 |
| 80 | V_MFMA_I32_32X32X4_2B_I8 |
| 86 | V_MFMA_I32_32X32X16_I8 |
| 56 | V_MFMA_I32_32X32X32_I8 |
| 82 | V_MFMA_I32_4X4X4_16B_I8 |
| 110 | V_MFMA_F64_16X16X4_F64 |
| 111 | V_MFMA_F64_4X4X4_4B_F64 |

### MFMA with scale (F8F6F4 microscaling)
See the scale encoding records in the XML.

| Instruction |
|---|
| V_MFMA_SCALE_F32_16X16X128_F8F6F4 |
| V_MFMA_SCALE_F32_32X32X64_F8F6F4 |

### SMFMAC (sparse)
| Opcode | Instruction |
|---:|---|
| 98 | V_SMFMAC_F32_16X16X32_F16 |
| 100 | V_SMFMAC_F32_32X32X16_F16 |
| 90 | V_SMFMAC_F32_16X16X64_F16 |
| 91 | V_SMFMAC_F32_32X32X32_F16 |
| 102 | V_SMFMAC_F32_16X16X32_BF16 |
| 104 | V_SMFMAC_F32_32X32X16_BF16 |
| 57 | V_SMFMAC_F32_16X16X64_BF16 |
| 70 | V_SMFMAC_F32_32X32X32_BF16 |
| 106 | V_SMFMAC_I32_16X16X64_I8 |
| 108 | V_SMFMAC_I32_32X32X32_I8 |
| 58 | V_SMFMAC_I32_16X16X128_I8 |
| 71 | V_SMFMAC_I32_32X32X64_I8 |
| 120 | V_SMFMAC_F32_16X16X64_BF8_BF8 |
| 121 | V_SMFMAC_F32_16X16X64_BF8_FP8 |
| 122 | V_SMFMAC_F32_16X16X64_FP8_BF8 |
| 123 | V_SMFMAC_F32_16X16X64_FP8_FP8 |
| 124 | V_SMFMAC_F32_32X32X32_BF8_BF8 |
| 125 | V_SMFMAC_F32_32X32X32_BF8_FP8 |
| 126 | V_SMFMAC_F32_32X32X32_FP8_BF8 |
| 127 | V_SMFMAC_F32_32X32X32_FP8_FP8 |
| 59 | V_SMFMAC_F32_16X16X128_BF8_BF8 |
| 60 | V_SMFMAC_F32_16X16X128_BF8_FP8 |
| 61 | V_SMFMAC_F32_16X16X128_FP8_BF8 |
| 67 | V_SMFMAC_F32_16X16X128_FP8_FP8 |
| 75 | V_SMFMAC_F32_32X32X64_BF8_BF8 |
| 78 | V_SMFMAC_F32_32X32X64_BF8_FP8 |
| 79 | V_SMFMAC_F32_32X32X64_FP8_BF8 |
| 83 | V_SMFMAC_F32_32X32X64_FP8_FP8 |

**Data types:** F32, F64, I32/I8, FP8/BF8, and **F8F6F4** mixed narrow formats with
per-block **scaling** (microscaling / MX). **No XF32.**

---

## RDNA3

First RDNA generation with WMMA.

| Opcode | Instruction |
|---:|---|
| 64 | V_WMMA_F32_16X16X16_F16 |
| 65 | V_WMMA_F32_16X16X16_BF16 |
| 66 | V_WMMA_F16_16X16X16_F16 |
| 67 | V_WMMA_BF16_16X16X16_BF16 |
| 68 | V_WMMA_I32_16X16X16_IU8 |
| 69 | V_WMMA_I32_16X16X16_IU4 |

**Data types:** F16, BF16, IU8, IU4. Single 16x16x16 shape. No sparsity.

---

## RDNA3.5

Same WMMA set and opcodes as RDNA3.

| Opcode | Instruction |
|---:|---|
| 64 | V_WMMA_F32_16X16X16_F16 |
| 65 | V_WMMA_F32_16X16X16_BF16 |
| 66 | V_WMMA_F16_16X16X16_F16 |
| 67 | V_WMMA_BF16_16X16X16_BF16 |
| 68 | V_WMMA_I32_16X16X16_IU8 |
| 69 | V_WMMA_I32_16X16X16_IU4 |

**Data types:** F16, BF16, IU8, IU4. Identical WMMA capability to RDNA3.

---

## RDNA4

Adds **FP8/BF8** WMMA, larger-K IU4, and the **SWMMAC** sparse engine.

### WMMA
| Opcode | Instruction |
|---:|---|
| 64 | V_WMMA_F32_16X16X16_F16 |
| 65 | V_WMMA_F32_16X16X16_BF16 |
| 66 | V_WMMA_F16_16X16X16_F16 |
| 67 | V_WMMA_BF16_16X16X16_BF16 |
| 68 | V_WMMA_I32_16X16X16_IU8 |
| 69 | V_WMMA_I32_16X16X16_IU4 |
| 70 | V_WMMA_F32_16X16X16_FP8_FP8 |
| 71 | V_WMMA_F32_16X16X16_FP8_BF8 |
| 72 | V_WMMA_F32_16X16X16_BF8_FP8 |
| 73 | V_WMMA_F32_16X16X16_BF8_BF8 |
| 74 | V_WMMA_I32_16X16X32_IU4 |

### SWMMAC (sparse)
| Opcode | Instruction |
|---:|---|
| 80 | V_SWMMAC_F32_16X16X32_F16 |
| 81 | V_SWMMAC_F32_16X16X32_BF16 |
| 82 | V_SWMMAC_F16_16X16X32_F16 |
| 83 | V_SWMMAC_BF16_16X16X32_BF16 |
| 84 | V_SWMMAC_I32_16X16X32_IU8 |
| 85 | V_SWMMAC_I32_16X16X32_IU4 |
| 86 | V_SWMMAC_I32_16X16X64_IU4 |
| 87 | V_SWMMAC_F32_16X16X32_FP8_FP8 |
| 88 | V_SWMMAC_F32_16X16X32_FP8_BF8 |
| 89 | V_SWMMAC_F32_16X16X32_BF8_FP8 |
| 90 | V_SWMMAC_F32_16X16X32_BF8_BF8 |

**Data types:** F16, BF16, IU8, IU4, **FP8/BF8** (new); **sparsity via SWMMAC** (new).
No XF32.

---

## CDNA5

The gfx1250 target uses the **WMMA / SWMMAC** naming (not MFMA), consistent with rocKE's
`arch_specs.json` entry for `gfx1250` (`family: "cdna"`, `target_family:
"gfx12_cdna"`, `has_mfma: false`, `has_wmma: true`). It expands supported shapes and
formats, and adds **F8F6F4/FP4** with per-block scaling (`V_WMMA_SCALE*` /
`V_WMMA_LD_SCALE*`).

> No native XF32 or F64 matrix instruction is present in this XML. LLVM gates
> `V_WMMA_F64_16X16X4_F64` on `gfx1251-gemm-insts`, which gfx1250 lacks
> (see the pinned sources above). It is not an unwired gfx1250 capability.

### WMMA (dense)
| Opcode | Instruction |
|---:|---|
| 93 | V_WMMA_F32_16X16X4_F32 |
| 96 | V_WMMA_F32_16X16X32_F16 |
| 97 | V_WMMA_F16_16X16X32_F16 |
| 98 | V_WMMA_F32_16X16X32_BF16 |
| 99 | V_WMMA_BF16_16X16X32_BF16 |
| 100 | V_WMMA_BF16F32_16X16X32_BF16 |
| 114 | V_WMMA_I32_16X16X64_IU8 |
| 106 | V_WMMA_F32_16X16X64_FP8_FP8 |
| 107 | V_WMMA_F32_16X16X64_FP8_BF8 |
| 108 | V_WMMA_F32_16X16X64_BF8_FP8 |
| 109 | V_WMMA_F32_16X16X64_BF8_BF8 |
| 110 | V_WMMA_F16_16X16X64_FP8_FP8 |
| 111 | V_WMMA_F16_16X16X64_FP8_BF8 |
| 112 | V_WMMA_F16_16X16X64_BF8_FP8 |
| 113 | V_WMMA_F16_16X16X64_BF8_BF8 |
| 128 | V_WMMA_F32_16X16X128_FP8_FP8 |
| 129 | V_WMMA_F32_16X16X128_FP8_BF8 |
| 130 | V_WMMA_F32_16X16X128_BF8_FP8 |
| 131 | V_WMMA_F32_16X16X128_BF8_BF8 |
| 132 | V_WMMA_F16_16X16X128_FP8_FP8 |
| 133 | V_WMMA_F16_16X16X128_FP8_BF8 |
| 134 | V_WMMA_F16_16X16X128_BF8_FP8 |
| 135 | V_WMMA_F16_16X16X128_BF8_BF8 |
| 51 | V_WMMA_F32_16X16X128_F8F6F4 |
| 136 | V_WMMA_F32_32X16X128_F4 |

### SWMMAC (sparse)
| Opcode | Instruction |
|---:|---|
| 101 | V_SWMMAC_F32_16X16X64_F16 |
| 102 | V_SWMMAC_F32_16X16X64_BF16 |
| 103 | V_SWMMAC_F16_16X16X64_F16 |
| 104 | V_SWMMAC_BF16_16X16X64_BF16 |
| 105 | V_SWMMAC_BF16F32_16X16X64_BF16 |
| 123 | V_SWMMAC_I32_16X16X128_IU8 |
| 115 | V_SWMMAC_F32_16X16X128_FP8_FP8 |
| 116 | V_SWMMAC_F32_16X16X128_FP8_BF8 |
| 117 | V_SWMMAC_F32_16X16X128_BF8_FP8 |
| 118 | V_SWMMAC_F32_16X16X128_BF8_BF8 |
| 119 | V_SWMMAC_F16_16X16X128_FP8_FP8 |
| 120 | V_SWMMAC_F16_16X16X128_FP8_BF8 |
| 121 | V_SWMMAC_F16_16X16X128_BF8_FP8 |
| 122 | V_SWMMAC_F16_16X16X128_BF8_BF8 |

### WMMA scale helpers (F8F6F4 / microscaling — separate encoding)
| Instruction |
|---|
| V_WMMA_SCALE_F32_16X16X128_F8F6F4 |
| V_WMMA_SCALE_F32_32X16X128_F4 |
| V_WMMA_SCALE16_F32_16X16X128_F8F6F4 |
| V_WMMA_SCALE16_F32_32X16X128_F4 |
| V_WMMA_LD_SCALE_B32 |
| V_WMMA_LD_SCALE_PAIRED_B32 |
| V_WMMA_LD_SCALE16_B64 |
| V_WMMA_LD_SCALE16_PAIRED_B64 |

**Data types:** F16, BF16, F32, IU8, FP8/BF8, and **F8F6F4 & FP4** with
per-block **scale** operands. **No native XF32/TF32 or F64 matrix instruction.**

> **IU4 is not listed above on purpose.** The dense opcode table for this part
> enumerates `V_WMMA_I32_16X16X64_IU8` and `V_SWMMAC_I32_16X16X128_IU8` only — no
> `IU4` form was located. Treat INT4 on this part as **unconfirmed** and check the
> official ISA document before relying on it.

---

## Cross-architecture capability summary

| Feature | CDNA1 | CDNA2 | CDNA3 | CDNA4 | RDNA3 | RDNA3.5 | RDNA4 | CDNA5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Engine | MFMA | MFMA | MFMA | MFMA | WMMA | WMMA | WMMA | WMMA |
| F32 accum | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| F16 in | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| BF16 in | ✅ | ✅ (1K) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT4 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❓ |
| FP64 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **XF32 / TF32** | ❌ | ❌ | **✅** | ❌ | ❌ | ❌ | ❌ | ❌ |
| FP8/BF8 | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| F8F6F4 / FP4 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Block scaling (MX) | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Sparse (SMFMAC/SWMMAC) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |

`❓` = not located in the source document's opcode table; see the per-section note.

**XF32 note:** `V_MFMA_F32_16X16X8_XF32` (opcode 62) and
`V_MFMA_F32_32X32X4_XF32` (opcode 63) are present on gfx942, a supported rocKE
target. The catalog does not yet expose XF32 atoms. gfx950 and gfx1250 have no
native XF32 instruction in the public XML or pinned LLVM target features.
BF16 compute has a different precision contract; native XF32 absence does not
rule out software implementations of TF32 semantics.
