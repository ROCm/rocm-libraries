<!-- Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT -->

# Descriptor-driven fragment transport migration

The gfx950 and gfx1250 block-scaled GEMM builders and the gfx942 TF32 probe
now use the public tiling `load_fragment` and `store_fragment` verbs, including
the existing native mirrors. The old `mma_io` import and native include are
removed. Raw byte-padded rows and wider slots still use the raw-layout tiling
adapter rather than a second transport implementation. New kernels should use
the descriptor-driven verbs directly.

HIP loads, stores, and global atomics add the memory operation's element index
using the access type's byte width, even when the pointer is represented as a
byte base. The rebase itself is still a byte offset. Packed FP16/BF16 atomics
index scalar halfwords and require four-byte alignment for the pair.
This applies beyond packed MMA fragments: grouped MoE rebases FP16/BF16 weights
by expert before issuing typed loads. Address emission is covered in both HIP
lowerers; the full grouped FP16 kernel is tested through the Python HIP route
with canonical and preshuffled weights on both vector and scalar load paths.

Even without rebasing, HIP scalar loads and stores whose IR alignment is weaker
than the access width use a byte copy. Naturally aligned multi-byte accesses
through matching typed pointers still use direct dereferences. As a codegen check,
Clang 24 at `-O3` emitted identical kernel instructions before and after the
scalar alignment change for FP16 MoE gather with hidden width and workgroup size
both 256, on `gfx942`, `gfx950`, and `gfx1250`. This does not measure runtime
performance or establish that other kernels have unchanged codegen.

The inventory below records the migration from `f29f15d3f7ec` and the validation
requirements. Its old paths and line numbers refer to that starting revision.

## Scope and invariants

- Migrate the gfx950 and gfx1250 block-scaled GEMM builders and the gfx942 TF32
  probe, including existing native builder counterparts.
- Keep instruction selection, numerical conversions, scale semantics, tensor
  factors, prefetch policy, launch geometry, signatures, and kernel identity
  unchanged. No new instruction support or convolution LDS migration.
- A fragment contains logical data in its live storage registers. Trailing
  words required by an instruction ABI never belong to the memory descriptor
  or the fragment's storage capacity.
- Preserve bit ordering, lane/register mapping, signed address arithmetic,
  clipping and allocation safety, alignment guarantees, and input refusals.
- Do not replace a native builder with a Python-authored serialized kernel.
- Preserve existing accepted raw-storage contracts through a compatibility
  adapter until equivalent tests and migration exist. Do not erase their tests
  merely to remove the old module.

## Temporary scaled-MMA ABI bridge

`prepare_scaled_mma_operand` in both engines is a temporary bridge. It checks
format and logical element count, then reinterprets and zero-pads the live
payload as needed before issuing the MMA op. It preserves the current full-carrier
requirements of gfx950 scaled MFMA and the gfx1250 scaled WMMA HIP route. The
gfx1250 scaled WMMA LLVM route can already adapt live payloads.

The long-term owner of reinterpretation and ABI zero-padding is the shared
lowering operand forms. Remove the bridge once the builder and verifier admit
the catalog-typed logical payload and an explicit full carrier, and the LLVM
and HIP lowerers of both engines handle both forms. Wrong format and width
must still be rejected. This migration does not change operand admission or
require the lowering work to land first.

## Implementation inventory

Paths in the tables are relative to `platform/`.

| Concern | Before migration | Ownership after migration |
| --- | --- | --- |
| Pointer, storage element, and storage vector types | `python/rocke/helpers/mma_io.py:31-59` | New `python/rocke/core/ir_storage.py` with native core counterpart; retain `core/storage.py` as the bit-storage rule owner |
| MMA logical operand type | `python/rocke/helpers/mma_io.py:62-74` | `python/rocke/core/mma_operand.py`; preserve existing admission in this migration |
| Encoded-pattern packing | `python/rocke/helpers/mma_io.py:182-240`, `cpp/helpers/mma_io.cpp:159-240` | New `helpers/packing.py` and native helper/header; no numerical quantization |
| Atom operand ABI bridge | Consumer-side carrier vectors | Temporary `helpers/mma_operand.py` and native helper/header; shared lowering forms are the long-term owner |
| Descriptor and alignment contracts | `helpers/tiling/descriptors.py:52-180` | Add explicit conservative base-alignment support without changing existing default emission |
| Load/store policy | `helpers/tiling/emit.py:508-725,958-1209` | One shared transport implementation; retain safe whole-group clipping and unsupported swizzle/repacking refusals |
| Native fragment I/O | No tiling authoring twin | Add bounded descriptor/window/tile/fragment and load/store equivalents under `cpp/helpers/tiling/`; reuse distribution and IR primitives |
| Obsolete transport module | `helpers/mma_io.py`, `cpp/helpers/mma_io.cpp`, matching public header | Remove independent transport after all callers and tests are migrated; any temporary compatibility wrapper delegates, never duplicates loads |

### Consumer locations

| Consumer | Input and output sites before migration | Constraints |
| --- | --- | --- |
| `python/rocke/instances/gfx950/block_scaled_gemm.py` | A/B loads 154/164, scale reads 177/178, C store 183 | All five formats and independent A/B; live FP6/FP4 carrier counts 6/4 padded to 8 only for the atom |
| `cpp/instances/gfx950/block_scaled_gemm.cpp` | A/B loads 111/113 and adjacent scale/output code | Author natively using the same descriptor and operand bridge |
| `python/rocke/instances/gfx1250/block_scaled_gemm.py` | `_load_frag` 491-509, scale packing 511-530, output stores 553/590 | Preserve SCALE/SCALE16 and legacy branches; FP6/FP4 counts 12/8 padded to 16; preserve prefetch/tensor-scale order |
| `python/rocke/instances/gfx942/tf32_mma_probe.py` | Input/C/preparation/output transport 118-170 | Preserve raw/carrier/RNE/prepacked/FP32 modes and batch offsets; conversions stay explicit |
| `cpp/instances/gfx942/tf32_mma_probe.cpp` | Matching input/preparation/output block | Same native contract and emission order |

### Import-only migrations

Update tiling `emit.py`, `fragments.py`, and `mma/driver.py`; tiling tests
`mma/test_mma_k_coupling.py` and `traits/test_mma_backend_coverage.py`; and core
tests `test_dtypes.py`, `test_optimization_barrier.py`, and the type tests split
out of `test_mma_io.py`. These are utility users, not transport consumers.

## Test and build migration inventory

- Split `tests/core/test_mma_io.py` by contract: storage types, encoded packing,
  generic vector-access lowering, and descriptor-driven fragment transport.
- Keep the bit-storage fixtures in `tests/core/test_storage.cpp`; migrate its
  native transport emitters without dropping invalid-input and alignment cases.
- Extend `python/rocke/tests/helpers/tiling/test_fragment_storage.py` and add a
  focused native-authored fragment-I/O emitter test. Cover nonzero origins,
  independent A/B, whole-group clipping, 12/24-byte loads, 20/36-byte tails,
  conservative alignment, ABI zero tails, and rejection before emission.
- Extend operand tests near `tests/core/test_mma_operand_register_type.py` for
  atom-owned padding and format/width refusals.
- Reuse `test_gfx950_scaled_mfma.py`, the gfx1250 scaled/FP4/FP6/mixed/NVFP4
  tests, and `test_tf32.py` for full builder coverage.
- Reuse native parity pairs `tests/instances/parity/gfx950_block_scaled_gemm_emit.*`
  and `tf32_mma_emit.*`; add native helper parity independently of lowerer parity.
- Register new native tests in `tests/cmake/CoreTests.cmake` or the matching
  helper-test registration. Keep installed test environment contracts in
  `CMakeLists.txt` synchronized; do not turn a missing native lane into a pass.
- Refresh only intentionally affected golden entries, including
  `tests/core/gfx950_scaled_mfma_llvm23.json` and affected representative GEMM
  entries, after parity and codegen review. Untouched families remain unchanged.

## Baseline-versus-migrated ISA gate

Capture a baseline before emission changes. Use identical specs, LLVM flavor,
compiler release, optimization flags, compile route, and target on both arms.
Record source and artifact hashes independently of the branch names.

Inspect small and multi-step cases on gfx950, gfx1250, and gfx942:

1. Expected MFMA/WMMA opcodes, selectors, instruction counts, and operand widths.
2. Extra `v_mov` zero materialization around ABI padding: compare with the old
   helper, which already zero-pads. Do not assume vector construction is free.
3. Shifts, masks, permutes, conversions, and integer address arithmetic.
4. Global load/store counts and widths, especially FP6 24-byte transport versus
   12-byte groups and the 20/36-byte tail policy.
5. Waits/barriers, VGPR/SGPR/accumulator usage, scratch, spills, and preserved
   prefetch ordering.

Reuse the gfx950 disassembly extraction in
`tests/instances/gfx950_scaled_numeric.py:31-100`; add a reusable comparison tool
and its parser tests rather than hand-maintaining unrelated shell scripts.
ISA comparison is codegen evidence, not numeric correctness or performance
measurement. Investigate an added instruction/resource tax before accepting it.

## Execution sequence

1. Capture baseline and make the existing contract matrix reproducible.
2. Separate storage typing and encoded packing from matrix I/O, with tests.
3. Add the temporary scaled-MMA ABI bridge and explicit transport alignment semantics.
4. Add the minimal mirrored native descriptor-driven fragment transport.
5. Migrate each Python/native consumer pair and its parity tests; keep numerical
   code and public kernel signatures stable.
6. Consolidate tests/docs and remove obsolete independent transport only after
   a repository-wide reference audit.
7. Run host contracts, native authoring parity, LLVM/HIP lowering parity,
   compiler/ISA comparisons, numeric GPU gates, and the full byte-identity gate.

GPU execution is a separate lane. A missing device/compiler or a skipped test
is not a pass. Remote/Slurm submission requires separate user authorization.

## Decisions that must not be hidden in cleanup

- The old helper accepts raw byte-padded rows and wider logical slots. A logical
  dense `Fragment` cannot represent arbitrary physical carrier capacity. Keep
  generic bit-storage coverage and a delegating compatibility path until the
  storage/view representation and API compatibility decision are explicit.
- Native I/O needs a bounded public subset, not a wholesale port of the tiling
  MMA driver, catalog, layout optimizer, recorder, or solver.
- Preserve safe weak-alignment behavior even if it requires a narrower load.
  Any performance claim needs generated-code evidence plus correctness-gated
  measurement; an opcode count alone is insufficient.
