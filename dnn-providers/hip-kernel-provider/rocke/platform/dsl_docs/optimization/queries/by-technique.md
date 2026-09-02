# Query: By Technique

> Auto-generated. Do not edit manually.

| Technique | Scope | Architectures | Operators |
|-----------|-------|---------------|-----------|
| [AGPR accumulators (CDNA)](../wiki/techniques/cdna/agpr-acc.md) | arch-specific | gfx90a, gfx942, gfx950 | gemm, attention, convolution |
| [Async DRAM to LDS](../wiki/techniques/common/async-copy.md) | common | cdna, gfx12 | gemm, attention, convolution |
| [Chiplet / XCD grid swizzle (CDNA)](../wiki/techniques/cdna/chiplet-swizzle.md) | arch-specific | gfx942, gfx950 | gemm |
| [Direct vs cshuffle epilogue](../wiki/techniques/common/epilogue.md) | common | cdna, rdna | gemm, convolution |
| [Emit split loadcnt/dscnt instead of s_waitcnt](../wiki/techniques/gfx1250/split-waitcnt.md) | arch-specific | gfx1250 | gemm, attention, convolution, moe, small-ops |
| [GFX12 async global→LDS (gfx1250)](../wiki/techniques/gfx1250/async-lds.md) | arch-specific | gfx1250 | gemm, attention |
| [GFX12 async load cachepolicy (th/scope)](../wiki/techniques/gfx1250/cache-policy.md) | arch-specific | gfx1250 | gemm, attention |
| [ISA and resource inspection](../wiki/techniques/common/isa-inspect.md) | common | cdna, rdna, gfx12 | gemm, attention, convolution, moe, small-ops |
| [Kernel and epilogue fusion](../wiki/techniques/common/fusion.md) | common | cdna, rdna, gfx12 | gemm, attention, convolution, moe, small-ops |
| [LDS padding and bank swizzle](../wiki/techniques/common/lds-swizzle.md) | common | cdna, rdna, gfx12 | gemm, attention, convolution |
| [MFMA atom selection (CDNA)](../wiki/techniques/cdna/mfma-atom.md) | arch-specific | gfx90a, gfx942, gfx950 | gemm, attention, convolution |
| [Multi-stage async DMA with s_wait_asynccnt](../wiki/techniques/gfx1250/asynccnt-pipeline.md) | arch-specific | gfx1250 | gemm, attention |
| [Occupancy vs resource budget](../wiki/techniques/common/occupancy.md) | common | cdna, rdna, gfx12 | gemm, attention, convolution, moe |
| [Persistent kernels and Stream-K](../wiki/techniques/common/persistent-streamk.md) | common | cdna | gemm |
| [Producer / consumer waves (warp-specialization analog)](../wiki/techniques/gfx1250/producer-consumer.md) | arch-specific | gfx1250 | gemm, attention |
| [Prototype a new algorithm (escape hatch)](../wiki/techniques/common/algorithm-break.md) | common | cdna, rdna, gfx12 | gemm, attention, convolution, moe, small-ops |
| [Software pipeline and double buffering](../wiki/techniques/common/software-pipeline.md) | common | cdna, rdna, gfx12 | gemm, attention, convolution |
| [Tile and CTA geometry](../wiki/techniques/common/tiling.md) | common | cdna, rdna, gfx12 | gemm, attention, convolution, moe |
| [Tile scheduling and persistent CTAs (no CLC)](../wiki/techniques/gfx1250/tile-schedule.md) | arch-specific | gfx1250 | gemm, moe, attention |
| [Transpose LDS read (gfx950)](../wiki/techniques/cdna/ds-read-tr.md) | arch-specific | gfx950 | gemm, attention |
| [Use ds_load_tr16_b128 with a matching LDS layout](../wiki/techniques/gfx1250/ds-load-tr.md) | arch-specific | gfx1250 | gemm, attention |
| [WMMA atom selection (RDNA)](../wiki/techniques/rdna/wmma-atom.md) | arch-specific | gfx1151, gfx1201 | gemm, attention |
| [Wave32 reductions and launch](../wiki/techniques/rdna/wave32.md) | arch-specific | gfx1151, gfx1201, gfx1250 | small-ops, attention, gemm |
| [Wide global loads and stores](../wiki/techniques/common/vectorized-io.md) | common | cdna, rdna, gfx12 | gemm, attention, convolution, moe, small-ops |
| [fp8 block-scaled GEMM on gfx1250](../wiki/techniques/gfx1250/block-scale.md) | arch-specific | gfx1250 | gemm, moe |
| [gfx1250 WMMA 16×16×32](../wiki/techniques/gfx1250/wmma-k32.md) | arch-specific | gfx1250 | gemm, attention, moe |
