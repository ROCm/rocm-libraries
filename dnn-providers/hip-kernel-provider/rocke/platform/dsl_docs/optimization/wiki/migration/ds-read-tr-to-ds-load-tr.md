---
id: migration-ds-read-tr-to-ds-load-tr
title: "ds_read_*_tr_* to ds_load_tr16_b128"
type: migration
from_arch: gfx950
to_arch: gfx1250
architectures: [gfx950, gfx1250]
architecture_families: [cdna, gfx12]
tags: [migration, ds-load-tr, gfx1250]
confidence: verified
reproducibility: snippet
related:
  - hw-ds-load-tr
  - technique-ds-read-tr
  - technique-gfx1250-ds-load-tr
  - pattern-silent-ds-load-tr
sources: [project-rocke]
---

# Transpose-read ABI

Same Python helper (`ds_read_tr16_b128`), different llvm intrinsic and
lane distribution. gfx950: `llvm.amdgcn.ds.read.tr16.b128` → `<8 x i16>`
then bitcast. gfx1250: element-typed `llvm.amdgcn.ds.load.tr16.b128.v8f16`
/ `.v8bf16`.

Rebuild the LDS store layout for wave32 + the gfx1250 B-operand map. A
gfx950 conflict-free swizzle plus this opcode is not automatically legal.
If WMMA inputs are garbage with a healthy async DMA, check
`pattern-silent-ds-load-tr` (LLVM inserting the transpose on a row-major
tile) before retuning pads.
