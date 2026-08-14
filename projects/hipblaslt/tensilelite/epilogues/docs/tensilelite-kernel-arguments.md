# TensileLite Kernel Argument Handling

## 1. Argument Layout

Defined in `tensilelite/Tensile/Components/Signature.py`. The layout is strictly sequential in the kernarg segment:

- **Bytes 0–15** — common 16-byte header: `GemmInfo`, `KernelInfo0`, `KernelInfo1`, `numWG` (4 × u32)
- **Bytes 16+** — `SizesFree[]`, `SizesSum[]`, then the matrix pointers `D/C/A/B` (8 bytes each), their strides (u32 each), then `alpha`/`beta`, then epilogue-specific fields (bias, activation args, RMSNorm/quant pointers, etc.)

The C++ host builds this buffer in `tensilelite/src/ContractionSolution.cpp:singleCallArgs()` in the same order.

---

## 2. SGPR Loading

The authoritative loader is a C++ class `ArgumentLoader` in `rocisa/rocisa/include/functions/argument.hpp` (lines 31–204). It maintains a running `kernArgOffset` and exposes two methods:

- **`loadKernArg(dst, srcAddr, sgprOffset, dword, ...)`** — emits a single `SLoadB{N}` from `KernArgAddress` (SGPR 0–1) + byte offset.
- **`loadAllKernArg(sgprStart, srcAddr, numSgpr, numPreload)`** — greedy packer that emits the widest aligned load that fits at each step (`SLoadB512` → `SLoadB256` → `SLoadB128` → `SLoadB64` → `SLoadB32`).

A Python mirror lives in `rocisa_stinkytofu_adaptor/rocisa_stinkytofu_adaptor/functions.py` lines 44–134.

---

## 3. Argument References in Codegen

`KernelWriter.py:defineSgpr()` (lines 602–627) is a pool allocator that maps named arguments to physical SGPR indices, stored in `self.sgprs["<name>"]`. Key allocations (always in this order):

```
KernArgAddress  → SGPR 0–1 (asserted, never changes)
SizesFree       → next N SGPRs (align=4)
AddressD/C/A/B  → 2 SGPRs each
StridesD/C/A/B  → N SGPRs each
Alpha, Beta     → N SGPRs, alignment padded
```

Codegen then calls `sgpr("AddressA", 2)` to produce the `s[n:n+1]` operand text — the physical SGPR number is looked up at that point.

Epilogue/store-path arguments (bias, activation, quant scale pointers, etc.) are tracked in `self.states.numStoreSgprNames` and loaded in a **second** `loadAllKernArg` call after the main GEMM loop.

---

## 4. On-Demand Loading

There are no `.s` macro files — everything is Python generating rocisa IR. The entry point is:

**`KernelWriterAssembly.py:getKernelArgLoadModule()`** (line 2228) — assembles the full prologue argument-load sequence into a `Module` object inserted at kernel entry.

Usage pattern:

```python
# Single field at explicit offset (does not advance cursor):
module.add(self.argLoader.loadKernArg(
    "AddressTD", "KernArgAddress",
    sgprOffset=hex(commonArgsSize + 16), dword=2))

# Bulk greedy load (advances cursor):
module.add(self.argLoader.loadAllKernArg(
    sgprStartIndex, "KernArgAddress", numSgprToLoad, numSgprPreload))
```

The store-block saves and restores the `argLoader` cursor (lines 8408–8415 and 8501–8508) so the two load phases don't interfere.

---

## 5. Argument Passing Convention

**Hardware mechanism**: The HSA kernel descriptor (in `rocisa_stinkytofu_adaptor/code.py`) emits `.amdhsa_user_sgpr_kernarg_segment_ptr 1`, which tells the hardware to place the 64-bit kernarg buffer address in SGPR 0–1 at dispatch. This is why `KernArgAddress` is always asserted to be SGPR 0.

**Metadata**: `SignatureCodeMeta` emits `.kernarg_segment_size` (total bytes, 8-byte rounded) and `.kernarg_segment_align: 8`.

**Preload optimization**: When `PreloadKernArgs` is enabled, `.amdhsa_user_sgpr_kernarg_preload_length/offset` directives push the first N argument words directly into user SGPRs at launch — the kernel then copies them into canonical named positions via `SMovB32`/`SMovB64` (lines 2643–2669).

**Grouped GEMM**: The 16-byte header holds the GEMM count + a pointer to a per-GEMM argument array. After loading the header, `KernArgAddress` is advanced by `commonArgsSize`, then offset by `gemmIdx * userArgsInfo.totalSize` to reach each GEMM's block.

---

## Key File Map

| File | Role |
|---|---|
| `Tensile/Components/Signature.py` | Argument order, field sizes, `.amdhsa_kernel` metadata |
| `rocisa/rocisa/include/functions/argument.hpp` | `ArgumentLoader` C++ — offset tracker + `SLoadB*` emitter |
| `rocisa_stinkytofu_adaptor/functions.py` | Python mirror of `ArgumentLoader` |
| `rocisa_stinkytofu_adaptor/code.py` | `SignatureCodeMeta` + `SignatureKernelDescriptor` |
| `Tensile/KernelWriter.py` | `defineSgpr()` pool, `numStoreSgprNames`, preload logic |
| `Tensile/KernelWriterAssembly.py` | `getKernelArgLoadModule()`, `argLoader` usage, grouped GEMM ptr arithmetic |
| `tensilelite/src/ContractionSolution.cpp` | `singleCallArgs()` — host-side kernarg buffer builder |
