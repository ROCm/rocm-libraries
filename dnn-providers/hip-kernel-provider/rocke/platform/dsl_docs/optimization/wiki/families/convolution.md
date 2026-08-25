---
id: family-convolution
title: "Convolution family — common tech × architecture"
type: family
operator_families: [convolution]
architecture_families: [cdna, rdna, gfx12]
tags: [convolution, routing]
related:
  - family-overview
  - kernel-conv-implicit-gemm
  - technique-async-copy
  - technique-epilogue
  - technique-mfma-atom
  - process-escape-hatch
sources:
  - project-rocke
  - project-miopen
  - project-hipdnn
  - project-composablekernel
---

# Convolution

Rocke: `conv_implicit_gemm` (NHWC × KYXC as GEMM), `conv_direct_grouped`
(16c / 4c), `img2col`, `pooling`. There is **no** `projects/hipconv` tree on
`develop`; convolution sources in this monorepo are MIOpen
(`projects/miopen`), hipDNN convolution descriptors
(`projects/hipdnn/backend/...Convolution*`), CK Tile grouped conv, and rocke.

## Common levers

| Lever | Rocke | Typical direction |
|---|---|---|
| Implicit GEMM vs direct | `ImplicitGemmConvSpec` vs `DirectConv16cSpec` / `4c` | 3×3 hero → implicit; tiny C → direct |
| Async DRAM→LDS | `async_dma` + `compv4` | overlap input/weight fetch |
| LDS K-pad | `lds_k_pad` | bank conflicts on packed K |
| Atom / K-fold | 32×32×16 or 16×16×32 fold | fewer K trips on gfx950 |
| Epilogue vectorization | wide `buffer_store` / direct | store bound after compute is healthy |
| `block_groups` / `block_q` | direct grouped | sweep ±1 around the known-good; not copied blindly from CK |

## Architecture columns

| | gfx942 | gfx950 | gfx1151 | gfx1250 |
|---|---|---|---|---|
| Implicit GEMM | yes | yes | WMMA `mem`+`default`, `groups=1` | per-instance |
| `conv_implicit_gemm_auto` | MFMA autotune | yes | no (raw MfmaAtom) | — |
| Direct 16c (`fold_k32`) | no (needs 16×16×32 f16) | yes | no | — |
| Direct 4c (`4×4×4`) | yes | yes | no WMMA 4×4×4 | — |

MIOpen and hipDNN own host-side algorithm selection (FFT / Winograd /
implicit GEMM). Rocke owns the explicit tile/pipeline/epilogue levers on the
instances above. Switching algorithm *class* (implicit ↔ direct, or a
mapping MIOpen has that this spec cannot name) after the table stalls is
`process-escape-hatch`, not another `block_q` tweak.
