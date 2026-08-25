---
id: pattern-silent-ds-load-tr
title: "LLVM inserted ds_load_tr16_b128 on row-major LDS"
type: pattern
tags: [ds-load-tr, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
symptoms: [silent-ds-load-tr]
candidate_techniques:
  - technique-gfx1250-ds-load-tr
  - technique-isa-inspect
related:
  - hw-ds-load-tr
  - migration-ds-read-tr-to-ds-load-tr
sources: [project-rocke]
---

# Silent transpose substitution

Correct async DMA, garbage WMMA inputs. The AMDGPU backend may replace
`load <8 x half>` from LDS with `ds_load_tr16_b128` when the load feeds
WMMA, assuming column-major LDS. Conv/GEMM wavelet stores are row-major.

Rocke’s fix: volatile LDS loads when `blocks_ds_load_tr16` (`Gfx1250Backend`).
Confirm ISA shows plain `ds_read_b128` unless you *intended* the transpose
layout. Documented on `instances/common/README_conv_implicit_gemm.md`.
