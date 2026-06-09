# NGL pack / ds_read ordering — the normative question behind rocm-libraries-uvrl

This doc captures the one design question that gates `rocm-libraries-uvrl`
(the load-bearing half of the original h7lo investigation). It is written so a
reviewer who has never seen the C-chain can reproduce the issue and answer the
question in one sitting.

Related beads:
- `rocm-libraries-h7lo` (CLOSED) — fixed the *messaging* artifact (validator was
  citing the wrong body's reference instruction).
- `rocm-libraries-uvrl` (OPEN, P0) — the *substantive* routing divergence
  described here. Blocks `rocm-libraries-r62g` (Phase 3 go/no-go gate).

## Committed artifacts (stable, line numbers below resolve into these)

A frozen snapshot is committed alongside this doc so the line citations never
go stale:
- `Tensile/Components/h7lo_uvrl_artifacts/kernel.s` — the CMS-emitted assembly
  (line numbers in this doc are into this file).
- `Tensile/Components/h7lo_uvrl_artifacts/compare_graphs_failures.txt` — the 16 failures.
- `Tensile/Components/h7lo_uvrl_artifacts/H7LO_INVESTIGATION_MEMO.md` — full
  investigation with live graph dumps (the reference-side timeline that is NOT in
  the assembly text lives in §4).

The `hxcx_artifacts/` paths in the repro section below regenerate identical
content (they are gitignored build output; the committed copy above is the
permanent record).

---

## How to reproduce (exact commands)

From the worktree, with rocisa freshly built (the cached `build_tmp` binary
predates the C-chain and will give stale results):

```bash
cd /home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite
pip install -e ./rocisa
python Tensile/Tests/unit/_dump_hxcx_assembly.py
```

That regenerates `hxcx_artifacts/`:
- `hxcx_artifacts/kernel.s` — the full CMS-emitted assembly (~7,200 lines). This
  is the SUBJECT (CMS) schedule. Line numbers cited below are into this file.
- `hxcx_artifacts/compare_graphs_failures.txt` — the 16 EdgeRoutedDifferentlyFailures.
- `hxcx_artifacts/validator_failures.txt` — 0 entries (the timing failures were
  fixed by hxcx; this confirms the rebuild is clean on that axis).

To see the 16 failures as the validator reports them:

```bash
cat hxcx_artifacts/compare_graphs_failures.txt
```

Each reads like:
```
Subject's consumer PackA3[9] @ idx=43 reads from subject's producer
PackA3[14] @ idx=45 at byte_keys (('v', 4),), but reference routes through
LRA3[1] @ idx=39 (of next iteration).
```

(Pre-h7lo-FixA these messages cited `PackA0[9] @ idx=-1 (PRO body)` — that was
the misattribution h7lo fixed. They now correctly cite the NGL/NLL-era
reference instruction.)

---

## The fixture

`CANONICAL_KERNEL_CONFIG` in `Tensile/Tests/unit/test_cross_subiter_alu_carveout_real_kernel.py`
and `_dump_hxcx_assembly.py`: BPG#11 TF32 4x4 TN, gfx950, MI=[16,16,32,1,1,4,4,2,2],
MacroTile 128x128, DepthU 32, `PrefetchGlobalRead=2`, `PrefetchLocalRead=1`,
`UseCustomMainLoopSchedule=1`, `UsePLRPack=True`, `UseMFMAF32XEmulation=True`.

CMS schedule dispatched: `Tensile/Components/CustomSchedule/gfx950/_128x128x32_TF32.py`
(`_get_schedule_128x128x32_TF32`, the `('TN', False, 1)` branch).

---

## The issue, concisely

The validator compares two captures of the same kernel:
- **subject** = the CMS schedule (`customMainLoopSchedule`)
- **reference** = the default schedule (`_captureDefaultSchedule` / SHADOW)

Design contract (`DEFAULT_SCHEDULER_REFERENCE_DESIGN.md` §3): the two may differ
**only in scheduling**, never in dataflow content. The validator enforces this by
building a per-byte latest-writer dataflow graph for each and set-diffing the edges.

In the no-global-load region, the two schedules place the **`ds_read` of the next
tile's A/B fragment** and the **pack chain** (`v_cvt_pk_bf16_f32` +
`v_mfma_f32_4x4x4` that re-derive the rotating pack buffer `ValuA/B_X0_I0`) in
**opposite order**:

| | order in that region | last writer of `X0_I0+15` (= byte `('v',14)`) before the next-iter consumer |
|---|---|---|
| **subject (CMS)** | `ds_read` → pack mfma → pack cvt | the **pack cvt** |
| **reference (default)** | pack mfma → pack cvt → `ds_read` | the **`ds_read`** |

Both are correct on hardware: the `ds_read` reloads the same logical value the
pack cvt re-derives, so the rotating buffer holds the right bytes before the
consumer either way. But the validator's per-byte latest-writer model treats
`X0_I0+15` as one flat physical register, so it sees two *different* producers for
the same consumer byte and reports a topology divergence — the 16
EdgeRoutedDifferentlyFailures.

### Where to see it in the assembly (`Tensile/Components/h7lo_uvrl_artifacts/kernel.s`)

The subject (CMS) ordering — `ds_read` BEFORE the pack chain — is directly visible:

```
kernel.s:1814  ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], ...   ; ds_read FIRST
kernel.s:1845  v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+10], ...                   ; pack chain (low bits)
kernel.s:1846  v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+11], ...
kernel.s:1848  v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:+3], ...        ; pack mfma
kernel.s:1856  v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[...+14], v[...+15]  ; pack cvt — final writer of v14
```

The reference (default) ordering is NOT in `kernel.s` — the SHADOW schedule is
captured but not emitted as assembly text. Its timeline is reproduced from the
live reference graph in `hxcx_artifacts/H7LO_INVESTIGATION_MEMO.md` §4, which
shows the default placing `pack mfma → pack cvt → ds_read` (the `ds_read` lands
last and becomes the consumer's source).

---

## The normative question (what a reviewer must decide)

Is the CMS ordering (`ds_read` → pack chain) **intended**, or is the default
ordering (pack chain → `ds_read`) authoritative?

- **If both orderings are legitimate (expected answer):** the validator's per-byte
  model is wrong to flag this. `uvrl` becomes "teach the latest-writer model that,
  for the rotating pack buffer, a pack-cvt that re-derives byte K and the
  `ds_read` that loads byte K are interchangeable producers." This is a
  validator-modeling change — no kernel-writer change. It is consistent with the
  "differ ONLY in scheduling" contract: a value-equivalent reorder of the
  rotating buffer is a *scheduling* difference the validator must tolerate.

- **If one ordering is actually wrong:** it is a real `customMainLoopSchedule` (or
  default) scheduler bug and the validator is correctly catching it; the fix is
  in the schedule, not the validator. This would contradict the "differ only in
  scheduling" premise, so it is the less likely answer — but it must be ruled out
  by whoever owns the NGL pack-emission ordering in
  `_get_schedule_128x128x32_TF32` / `customMainLoopSchedule`.

The person who owns `customMainLoopSchedule`'s NGL emission ordering can confirm
which case applies in one read of the schedule. Until that is answered, `uvrl`
cannot be planned (the two answers lead to fixes in different layers).

---

## Why this is benign today

These 16 failures are inside the inline xj16 validation assertion, which fires
only on `UseCustomMainLoopSchedule=1` builds. They do not affect emitted kernel
correctness (the assembly is correct on hardware). They are validator
false-positives pending the `uvrl` modeling decision. The kernel itself is fine.
