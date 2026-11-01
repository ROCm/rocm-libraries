<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Known issues and traps (gfx950)

These are three compiler or ISA traps we hit while porting and tuning this megakernel. Two of them corrupted data
silently. A1 is not affected by any of them: the kernel avoids each one, as described below. None of them has been
reported upstream yet.

Toolchain: ROCm 10.0 LLVM (`23.0.0git`), comgr `-O3`, `-mcpu=gfx950`.

## 1. A constant was rematerialized under partial EXEC (probable LLVM AMDGPU bug)

**What happened.** A loop-invariant VGPR constant was used by every lane after a divergent region. The compiler
rematerialized it (`v_mov_b32 vN, imm`) inside the loop-latch flow block, **before** the
`s_or_b64 exec, exec, s[..]` that turns the masked lanes back on. So the lanes that were masked off never got the
value. The register was shared with another live range, so those lanes kept that range's data.

```
v_and_b32  v247, 15, v0          ; another live range uses v247
s_mov_b64  exec, s[4:5]          ; waves 1-3 now have EXEC = 0
...        branch to latch
v_mov_b32  v247, 0x7f            ; remat under partial EXEC
s_or_b64   exec, exec, s[..]     ; restore
...        use of v247 with full EXEC: waves 1-3 read tid & 15, not 0x7f
```

**Symptoms seen.**
- Wrong padding bytes: exactly the bytes owned by waves 1–3.
- In another build, `ds_add_u32` to an address taken from the wrong value, which raised a memory aperture
  violation.

**Trigger.** A wave-uniform predicate computed in VGPRs, such as `tid // 64 == 0`. That makes the CFG divergent,
so the loop body ends in an EXEC-masked flow block. The IR itself was legal.

**Workaround used in `a1_kernel.py`:**
- every wave-uniform decision is an SGPR value (`readfirstlane(tid // 64)`, and LDS status words read with
  `readfirstlane`);
- the lane-0 counter publications narrow and restore EXEC inside one inline-asm block (`lane0_atomic_add_nw`).

So no lane-dependent region ends a loop body. The workaround keeps float order and the MFMA sequence.

**Not yet done:** a reduced reproducer, and a re-test on current LLVM main.

## 2. Inline-asm store, then compiler VALU write to the store's data registers (hazard not covered)

**What happened.** On gfx940/gfx950, a VMEM store wider than 8 bytes reads its data VGPRs late. A VALU write to
those VGPRs needs about 2 wait states after the store issues. LLVM inserts these wait states after its *own*
stores. It does not insert them after an **inline-asm** store, because the hazard recognizer cannot look inside
asm.

**Symptom seen.** An asm `global_store_dwordx4 … v[4:7] sc0 sc1` was followed directly by a compiler-generated
`v_lshl_add_u64 v[4:5], …`. The store wrote address bits into the data for a fixed subset of lanes.

**In this kernel.** The inline-asm stores in `a1_kernel.py` are at most 8 bytes (`global_store_dwordx2`), so they
are outside this hazard. If you add an inline-asm store wider than 8 bytes, pad it yourself (`s_nop 1` after the
store) before compiled code can write its data registers.

A related case: a `v_readlane_b32` that restores a spilled SGPR right before an inline-asm buffer store using
that SGPR. gfx9 needs 5 wait states between a VALU write of an SGPR and a VMEM read of it, and LLVM did not insert
them before the asm. This is the suspected cause of one memory fault in a microbenchmark (not in this kernel); the
padded fix (`s_nop 4`) was built but not re-run, so the cause is not confirmed.

## 3. `v_lshl_add_u64`: the shift operand is only 3 bits

`v_lshl_add_u64 D, S0, S1, S2` computes `D = (S0 << S1[2:0]) + S2`. Only bits [2:0] of the shift are used. So an
immediate shift of 9 **executes as 1**.

The assembler accepts `v_lshl_add_u64 v[0:1], v[2:3], 9, v[4:5]` (and 31 or 64) with no warning.

We hit this in hand-written microbenchmark asm, as silently wrong addresses. The fix: shift first with
`v_lshlrev_b32` (or `v_lshlrev_b64`), then use `v_lshl_add_u64` with a shift of 0–7.

Lesson for ISA audits: check what an opcode does with its operands (field widths, masking), not the operand
text.
