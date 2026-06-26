---
name: isa-inspection-rocke
description: >
  Inspect generated AMDGPU ISA for CK DSL kernels before and after optimization.
  Use this when validating that the intended matrix instruction, packed dtype
  path, LDS layout, wait/barrier schedule, or occupancy-related code shape was
  actually emitted. Includes architecture-agnostic checks plus CDNA and RDNA
  notes.
  Usage: /isa-inspection-rocke
allowed-tools: Read Edit Bash Grep Glob Agent
---

# ISA Inspection (CK DSL)

Use ISA inspection to answer one question: "Did the compiler emit the hardware
work we intended, with no hidden scalar, conversion, memory, or synchronization
tax dominating the hot loop?"

Run it early. A benchmark delta without an ISA check can be misleading: a kernel
may be "faster" because it skipped work, or "slower" because the compiler
inserted conversions, spills, extra waits, or a fallback instruction path.

---

## When To Use

Use this skill when:

- Adding or changing an MMA/WMMA/MFMA atom.
- Switching precision, especially fp16 emulation -> native integer, fp8/bf8,
  packed int4/int8, or block-scaled formats.
- Adding LDS staging, async global-to-LDS copies, cshuffle epilogues, or
  cross-lane shuffles.
- Changing schedule hints, wait placement, barriers, launch bounds, or
  occupancy attributes.
- A roofline estimate says matrix compute should be fast, but measured latency
  is still dominated by scalar or coordination work.

---

## Minimal Workflow

1. Compile a small shape and a production-like shape with the exact same kernel
   options you intend to benchmark.
2. Extract disassembly from the HSACO or code object.
3. Search for the expected matrix instruction family.
4. Count hot-loop memory, LDS, conversion, wait, barrier, and scalar arithmetic
   instructions.
5. Compare the hot-loop code shape before and after the change.
6. Only then benchmark and explain the movement.

Example commands from the CK DSL repo root:

```bash
export PYTHONPATH=Python

# If the kernel artifact is already built, inspect the HSACO/code object.
llvm-objdump -d --mcpu=<gfx_arch> path/to/kernel.hsaco > /tmp/kernel.isa

# Quick opcode survey. Prefer rg over manual browsing for first pass.
rg "v_mfma|v_wmma|buffer_load|raw_buffer_load|ds_read|ds_write|s_waitcnt|s_barrier|v_cvt|v_perm|v_med|v_lshr|v_ashr" /tmp/kernel.isa
```

If no disassembler is available, or if you want the optimized compiler schedule
directly from a CK DSL kernel builder, lower the kernel to LLVM IR and ask
`amdclang++` to emit assembly. Use `-O3`; unoptimized assembly can make a
pipelined kernel look serialized.

```bash
export PYTHONPATH=<repo>/dnn-providers/hip-kernel-provider/rocKE/Python

python3 - <<'PY'
from pathlib import Path
from rocke.instances.common.moe_fused_mega_fp8 import (
    FusedMegaKernelSpecFp8,
    build_moe_fused_mega_gemm_fp8,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm

spec = FusedMegaKernelSpecFp8(name="isa_probe")
kernel = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950", persistent=False)
Path("/tmp/isa_probe.ll").write_text(lower_kernel_to_llvm(kernel, arch="gfx950"))
PY

amdclang++ -O3 -x ir -target amdgcn-amd-amdhsa -mcpu=gfx950 \
  -S /tmp/isa_probe.ll -o /tmp/isa_probe.s

rg "global_load_lds|global_load|buffer_load|ds_read|ds_write|v_mfma|v_wmma|s_waitcnt|s_barrier|v_exp|v_rcp|v_cvt|global_atomic" \
  /tmp/isa_probe.s
```

For fused kernels, inspect stage boundaries, not just the hot GEMM loop. A load
pipeline may be strong inside each matmul while still missing cross-stage
overlap. For example, in a gate/up -> activation/quant -> down fused MoE, check
whether `WDown` loads begin during the SiLU/dynamic-quant VALU window or only
after the quant barrier. The latter means the down loop is internally prefetched,
but the full fused chain is not completely overlapped.

If available, use CK DSL probes first because they package common extraction and
counting steps:

```bash
python dsl_docs/optimization/utilities/tools/dsl_probes/probe_isa_inspect.py \
  --demo <demo_name>

python dsl_docs/optimization/utilities/tools/dsl_probes/probe_intrinsic_counts.py \
  --demo <demo_name>
```

---

## First-Pass Checklist

### 1. Expected Matrix Instruction

Confirm the intended matrix instruction appears.

Good signs:

- CDNA MFMA kernels contain the expected `v_mfma_*` opcode.
- RDNA WMMA kernels contain the expected `v_wmma_*` opcode.
- The opcode suffix matches the intended datatype, K width, and accumulator
  type.

Red flags:

- A precision-switch kernel still contains the old opcode, such as fp16 WMMA in
  a supposed native-int path.
- Matrix instructions are present but surrounded by conversion chains that
  recreate the previous emulation path.
- The expected intrinsic exists in LLVM IR but not in ISA; this usually means
  lowering selected a fallback, folded the op unexpectedly, or the atom was not
  actually used by the builder.

### 2. Hidden Conversion Tax

Precision work often moves from matrix units to VALU. Search for conversion and
packing opcodes in the hot loop:

```text
v_cvt_*        float/int conversion
v_pk_*         packed conversion or packed arithmetic
v_perm_b32     byte/nibble packing and unpacking
v_med3_*       clamp/select idioms
v_lshrrev_*    unsigned bitfield extraction
v_ashrrev_*    signed nibble/byte sign extension
v_and_b32      mask low bits/nibbles
```

Optimization lesson: replacing fp16-emulated integer math with native integer
MMA/WMMA only pays if the surrounding pack, unpack, quantize, and coordinate code
also shrinks. If `v_wmma_i32_*` or `v_mfma_i32_*` appears but the loop still
contains fp16 dequantization, `rint`, or f32 clamp chains, the kernel may remain
VALU-bound.

### 3. Waits And Barriers

Look for where `s_waitcnt` and `s_barrier` land relative to the instructions they
wait for.

Good signs:

- VMEM loads for a future tile are issued before current tile compute.
- LDS reads are batched far enough ahead of the matrix instruction that
  `lgkmcnt` wait cycles are small.
- Barriers gate real cross-wave handoffs, not every local bookkeeping step.

Red flags:

- `buffer_load` immediately followed by `s_waitcnt vmcnt(0)` and use.
- `ds_read` immediately followed by `s_waitcnt lgkmcnt(0)` and MFMA/WMMA.
- Multiple `s_barrier` in a tiny K-loop with only a few matrix instructions
  between them.
- A "latency hiding" change increases wait/barrier count more than matrix count.

Small-K kernels need special skepticism. If K has only two to five matrix atoms,
deep software pipelines and many schedule hints may have too little compute to
hide their own overhead.

### 4. LDS Handoff Versus Cross-Lane Shuffle

Do not assume "register only" beats LDS. Inspect the required data movement.

An LDS handoff is often the cheap path when:

- A producer's accumulator lane layout is not the consumer's operand fragment
  layout.
- The transformation is a true cross-lane transpose.
- A cross-lane primitive would need many `ds_bpermute`, DPP, or permute
  operations per output slot.

Optimization lesson: an LDS round trip that moves each value once can beat a
register shuffle that moves each logical value many times through the same
cross-lane or LDS-backed path. Count the shuffle instructions you would add
before deleting a cshuffle or producer-consumer LDS handoff.

### 5. Coordinate Arithmetic

For convolution, gather/scatter, pooling, and fused kernels, count scalar and
vector integer address work:

```text
s_mul_i32 / v_mul_lo_u32
s_div_* / v_div_* or compiler-generated reciprocal sequences
v_lshr / v_and / v_add / v_sub address decomposition
v_cmp / v_cndmask bounds predication
readfirstlane / scalarized block-coordinate setup
```

Hoist or specialize before changing matrix atoms:

- Hoist row/column decode out of per-element loops.
- Replace div/mod by powers of two with shift/mask.
- Keep static problem constants as compile-time constants.
- Specialize fixed filters and strides instead of carrying generic descriptor
  transforms through the hot path.
- Use OOB-safe buffer resources for tails when they remove branchy per-lane
  guards.

### 6. Occupancy And Register Side Effects

If an optimization changes launch bounds, unrolling, pipeline depth, or fragment
count, inspect resource metadata and spills.

Red flags:

- Extra scratch or spill memory operations appear in the hot path.
- VGPR count drops because a waves-per-EU/max register attribute forced
  spilling or recomputation.
- Occupancy improves on paper but latency regresses because each wave lost too
  many registers or gained too much scalar packing work.

Architecture nuance:

- On CDNA MFMA kernels, inspect both architectural VGPR and accumulator VGPR
  usage. AGPRs are not freely exchangeable with VGPRs: MFMA accumulates there,
  but ordinary VALU, VMEM, and LDS instructions generally need VGPR operands.
  Consuming accumulators in epilogues or non-MFMA code can require moves back to
  VGPRs, so high AGPR pressure can still limit occupancy or create dependency
  latency even when VGPR count looks acceptable.
- On CDNA3-class accounting, arch VGPR and accum VGPR pressure can limit
  occupancy independently; check the worse limiter, not only the total.
- On CDNA4-class accounting, the pool is more flexible, but instruction operands
  still have register-class constraints. Treat lower VGPR count plus higher
  accumulator pressure as a trade-off to verify, not an automatic win.
- On RDNA WMMA kernels, there is usually no separate CDNA-style AGPR signal in
  the CK DSL workflow. Focus on total VGPR pressure, scratch/spills, repeated
  coordinate recomputation, and whether occupancy hints forced the compiler to
  lengthen the VALU path.

Optimization lesson: forcing more resident waves/workgroups is not free. Verify
the ISA and metadata show no spills and no large recomputation before accepting
an occupancy hint.

---

## Architecture-Agnostic Patterns

### Pattern: Native Low-Bit MMA But VALU-Bound Kernel

Signal:

- Correct native low-bit matrix opcode appears.
- Matrix instruction count is small relative to address, pack/unpack, clamp, and
  wait instructions.
- Roofline says matrix floor is low, but latency remains far above it.

Actions:

- Remove old emulation conversions first.
- Add packed fragment helpers so the hot loop loads the ABI-ready fragment type.
- Fast-path power-of-two quant scales with integer shifts instead of f32
  multiply/round/clamp.
- Specialize fixed shapes and filter sizes to cut coordinate arithmetic.

### Pattern: Positive ISA Change But No Benchmark Gain

Signal:

- Desired opcode appears and old opcode disappears.
- Runtime barely moves.

Actions:

- Compare instruction mix, not just matrix opcode.
- Check whether waits, barriers, or LDS traffic increased.
- Inspect VGPR/SGPR/LDS metadata for occupancy or spill regressions.
- Re-run a same-process interleaved A/B benchmark; small ISA wins can be inside
  run-to-run spread.

### Pattern: Schedule Hints Help Only Slightly

Signal:

- `sched_group_barrier` or scheduling changes move instructions but gain is tiny.
- K-loop has very few matrix atoms.

Actions:

- Keep the hint only if correctness-neutral and repeatedly positive.
- Do not stack it with unrelated occupancy or branch-masking changes unless each
  composition is measured.
- Prefer removing instructions over rescheduling a tiny instruction window.

---

## CDNA Notes

CDNA kernels use MFMA. Inspect for `v_mfma_*`.

Common checks:

- Verify MFMA shape and datatype suffix: f16/bf16, fp8/bf8, int8, fp4/fp6/fp8,
  or scaled variants.
- Check accumulator VGPR usage separately from architectural VGPR usage where
  the architecture exposes separate accounting.
- Async global-to-LDS copies and larger LDS capacity are architecture and
  generation dependent; confirm `raw.ptr.buffer.load.lds` or equivalent appears
  before assuming the async path is active.
- For cshuffle epilogues, compare global store coalescing against extra
  `ds_write`, `ds_read`, `s_waitcnt`, and `s_barrier` cost.

Typical CDNA red flags:

- High `s_waitcnt lgkmcnt(0)` after LDS reads feeding MFMA.
- Too many MFMA dependency bubbles because there are not enough independent
  accumulators.
- Padding/swizzle changes that reduce bank conflict but exceed LDS budget or
  reduce occupancy.

---

## RDNA Notes

RDNA kernels use WMMA. Inspect for `v_wmma_*`.

Common checks:

- Confirm wave size and fragment ABI. RDNA3/3.5 and RDNA4 can have different
  operand fragment widths for similar WMMA shapes.
- For integer WMMA, confirm signedness and packed operand ABI in the lowering
  path, then verify with a standalone bit-exact probe before using the atom in a
  fused kernel.
- RDNA WMMA and VALU scheduling can make coordinate arithmetic especially
  visible. A native WMMA opcode alone does not imply the kernel is matrix-bound.
- Cross-lane shuffles may route through LDS-like paths. Count them before
  replacing a simple LDS handoff.

Typical RDNA red flags:

- A supposed integer WMMA kernel still has fp16 conversion/dequantization in the
  hot loop.
- Small K loops have only a few `v_wmma_*` instructions per tile and many
  address/pack/quantize instructions.
- Occupancy hints improve theoretical residency but regress due to VGPR pressure
  or recomputation.

---

## Reporting Template

When summarizing ISA inspection, include:

```text
Expected matrix op:
  wanted: v_wmma_i32_16x16x16_iu8
  found:  yes/no, count=N

Old path removed:
  fp16/f32 conversion opcodes in hot loop: yes/no
  dequant/pack opcodes remaining: summary

Synchronization:
  s_waitcnt count: before -> after
  s_barrier count: before -> after
  wait directly before matrix op: yes/no

Memory/LDS:
  buffer_load / global_load count: before -> after
  ds_read / ds_write count: before -> after
  cshuffle or handoff cost: summary

Resources:
  VGPR/SGPR/LDS: before -> after
  spills/scratch: yes/no

Verdict:
  keep/revert/measure more, with the benchmark command and correctness status
```

Never report a speedup from ISA inspection alone. ISA tells you whether the
change is plausible; correctness and timing decide whether to keep it.

---

## See Also

- `/kernel-trace-analysis` - ATT trace stall attribution after ISA inspection
- `/gemm-optimization-rocke` - GEMM tiling and pipeline choices
- `/lds-optimization-rocke` - LDS bank conflict and handoff analysis
- `/prefetch-data-load-rocke` - VMEM/LDS prefetch and pipeline overlap
- `probe_isa_inspect.py` - CK DSL ISA extraction helper
- `probe_intrinsic_counts.py` - LLVM intrinsic count helper
