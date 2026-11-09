# XNACK Hazard Protection

This note summarizes the gfx1250 replay-safety rules StinkyTofu must preserve
for XNACK replay in single-VMEM-group mode.

## Core Rule

Unless an instruction class has a specific safe-overwrite rule below, do not
overwrite any register that an in-flight replayable memory operation would need
on replay until all outstanding translations have acknowledged. That includes
address registers, scalar bases, and data sources used by other instructions in
the same replay group.

In this document, an **open VMEM group** means:

`VMEM` means Global, Buffer, Scratch, Image, or FLAT memory instructions.
**TDM and LDS are not treated as VMEM here.**
Exception: `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` is treated as VMEM.

```text
XCNT is known to be 0
Non-VMEM
VMEM        # the VMEM group opens
VMEM        # still the same open VMEM group
```

An **open SMEM group** is the same idea for consecutive SMEM instructions.
StinkyTofu uses the sources from the open VMEM or SMEM group when deciding
whether a later destination needs an explicit drain.

Use one of these protections before the overwrite:

- Choose a destination that does not overlap any live replay source.
- Reorder the clobber after the memory group is known to be safe.
- Insert `s_wait_xcnt 0` to end the group and wait for all translations to
  acknowledge.

Example:

```asm
; Unsafe: v0 is still part of the first flat address.
flat_load_b32 v9, v[0:1]
flat_load_b32 v0, v[2:3]

; Safe: the explicit XCNT drain ends the replay group first.
flat_load_b32 v9, v[0:1]
s_wait_xcnt 0
flat_load_b32 v0, v[2:3]
```

## Protection Rules

Insert or preserve `s_wait_xcnt 0` for these cases:

- Before an `SMEM` operation whose destination would overwrite a source used by
  the open group. A wide load such as `s_load_b256` must keep its destination
  disjoint from its address registers.
- Before a `FLAT` operation whose destination would overwrite a source used by
  the open VMEM group. A FLAT operation may overwrite its own source only when
  it is the only instruction in the open VMEM group.
- Before a global atomic store/RMW, or a consecutive run of global atomic
  store/RMW operations, when a VMEM group is open:
  ```text
  open VMEM group
  XCNT == 0
  global atomic store/RMW
  global atomic store/RMW
  ...
  ```
- Before `s_set_vgpr_msb` when a VMEM or SMEM group is open:
  ```text
  VMEM
  XCNT == 0
  s_set_vgpr_msb

  SMEM
  XCNT == 0
  s_set_vgpr_msb
  ```
- Before an unbounded `s_sleep`.
- Before `s_prefetch_*` when an SMEM group is open:
  ```text
  SMEM
  XCNT == 0
  s_prefetch_*
  s_prefetch_*
  ...
  ```

No source-clobber drain is needed for global, buffer, scratch, or image VMEM
within the same group.

## TODO

The current StinkyTofu protection is intentionally scoped to the
single-VMEM-group model implemented by `Gfx1250XnackHazardPass`. Before extending the
rule set, resolve these coverage questions:

- When adding multi-group support, model data waits by counter. For example,
  `s_wait_loadcnt 0` proves completion only for operations tracked by LOADCNT;
  it does not generally prove `XCNT == 0`.
- Confirm whether `global_prefetch_b8` and `flat_prefetch_b8` need any
  single-VMEM-group XNACK source-clobber protection.
- Confirm whether `global_store_async_from_lds_*` needs the same
  source-clobber protection.
- Handle `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` as the LDS exception that is
  treated as VMEM for these rules.
- Add coverage for a self-overlapping FLAT followed by another VMEM instruction,
  to pin the single-instruction exception.
- Add FileCheck coverage for any newly classified instruction family, including
  source-clobber and explicit-protection cases.
