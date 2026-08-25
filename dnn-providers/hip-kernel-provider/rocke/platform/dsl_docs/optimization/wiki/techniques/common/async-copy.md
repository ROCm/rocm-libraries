---
id: technique-async-copy
title: "Async DRAM to LDS"
type: technique
tags: [async-copy, async-lds, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, gfx12]
operator_families: [gemm, attention, convolution]
rocke_primitive: "AsyncTileLoader / async_buffer_load_lds"
related: [technique-software-pipeline, technique-gfx12-async-lds, pattern-missing-waitcnt]
sources: [project-rocke]
symptoms: [memory-bound, missing-waitcnt]
---

# Async copy

Issue global loads that write LDS directly, then wait once per stage.
CDNA gfx942+: `buffer_load_lds` / `raw_ptr_buffer_load_lds` (`has_async_lds`).
gfx1151/gfx1201: catalog flag false — do not expect this opcode.
gfx1250: different opcode family (`technique-gfx12-async-lds`).

```python
from rocke.helpers.loads import AsyncTileLoader
```

Always pair with `s_waitcnt(vmcnt=…)` before the consumer. Intermittent
corruption is `pattern-missing-waitcnt`.
