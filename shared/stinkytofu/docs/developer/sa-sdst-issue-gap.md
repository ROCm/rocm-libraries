# SA_SDST Issue Gap

`dagFeatures.saSdstIssueGap` (module option `SaSdstIssueGap`, CLI
`--sa-sdst-issue-gap=N`) makes the CDNA5 DAG scheduler hide the SA_SDST drain:
it keeps a VALU that waits for SA_SDST at least N SQ issue cycles behind the last
SALU/SMEM. `0` (default) turns it off and leaves the schedule unchanged.

```yaml
- StinkyTofuParameters: [{SaSdstIssueGap: 8}]
```

## The hazard

```asm
s_cmp_le_u32  s[sgprLoopCounterL], 4
s_cmov_b32    s[sgprGL2PrefetchIncA], 0
v_add_co_u32  v[vgprGL2PrefetchAddrA_0], vcc_lo, v[vgprGL2PrefetchAddrA_0], s[sgprGL2PrefetchIncA]  ; stalls
```

- Every SALU (and SMEM, until it has read its sources) counts SA_SDST up until it
  retires, about 8 instructions later.
- A VALU that **writes an SGPR or VCC** (here the `vcc_lo` carry-out), or **reads
  SCC**, waits for SA_SDST == 0.
- The wait is not register-matched: any in-flight SALU stalls any such VALU. Reading
  the SGPR the SALU just wrote is forwarded and is not the cause.
- SCHED_MODE 2 still checks SA_SDST in hardware, so no `s_wait_alu` is emitted and the
  code is correct either way. This is latency hiding only, which is why it lives in the
  scheduler and not in `InsertWaitAluPass` / `WaitAluTracker`.

Affected VALUs (`waitsForSaSdst`, `HazardRules.hpp`): `v_add_co*` / `v_sub_co*` /
`v_add_co_ci*`, `v_cmp*` to SGPR/VCC, `v_readfirstlane`, `v_readlane`, and any VALU
reading SCC. EXEC-only destinations (`v_cmpx`) are excluded.

## Unit: SQ issue cycles, not `clock_`

The gap is counted in issue cycles of instructions the wave actually issues, never in
`CDNA5ReadyQueue::clock_`:

- `clock_` jumps to a WMMA window end, waits out queue/pacing stalls and skips non
  co-issue slots for VALU. A VALU co-issued under a WMMA reaches its SA_SDST check long
  before `clock_` says so, so a gate that decays on `clock_` would believe the drain is
  hidden when it is not.
- A VALU co-issued with a prior WMMA on the SP side still costs its own SQ issue cycle,
  so it counts.

| Issued instruction | Effect on the gap |
|---|---|
| SALU, SMEM (`incrementsSaSdst`) | reset to 0 |
| WMMA, VALU, DS, other | `+ issueCycles` (WMMA = 1, not its latency) |
| `s_set_vgpr_msb`, `s_wait_alu`, `s_delay_alu`, `s_wait*cnt`, pseudo (`isSequencerOnly`) | `+ 0`: issued by the sequencer next to other work |
| VMEM / tensor / SCC- or VCC-branch (`drainsSaSdst`) | drained: they wait for SA_SDST == 0 themselves |
| a waiting VALU | drained after it issues; if the gap was short, the rest of the drain is charged to `clock_` as elapsed time |

Since elapsed time is never less than issue time, an issue-cycle gap of N guarantees the
drain is hidden.

## DAG scheduler (`src/transforms/asm/dag/CDNA5.hpp`)

**State.** `saSdstGap_` is stepped by `stepSaSdst()` from `onScheduled()`, which the
scheduler calls for every instruction of the BB's final order (picked nodes, coexec
`v_nop` fillers and the non-movable side-effect instructions between regions). It is
per BB and carries across region cuts with no prefix seeding. `onInit()` starts each BB
drained: loop blocks are entered through SCC branches, which wait for SA_SDST.

**Holds.** Both go through `heldForAlu()`, which replaces `needsWaitAlu()` at every hold
site (`pickFreeBest`, the critical-filler loops, `findOldestFallbackNonWmma`,
`decidePromote`). Held work is skipped while anything else can issue and is still taken
last, so the fallback always makes progress.

- `saSdstHeld`: a waiting VALU while `saSdstGap_ < saSdstIssueGap`.
- `saluHeldForSaSdst`: a non-critical SALU while a waiting VALU is ready and the gap is
  partly built (`0 < gap < N`). Issuing it would restart the drain. Right after a SALU
  (`gap == 0`) nothing has built up, so it is not held.

**Grouping.** In `findSmallestPickableNonWmma`, right after a SALU (`gap == 0`) a ready
SALU that feeds a waiting VALU (`feedsSaSdstWaiter`) may issue even past the even-spread
filler quota: it costs nothing then and spares its VALU a drain of its own. This is what
puts `s_cmp`, `s_cmov IncA`, `s_cmov IncB` back to back.

## Repair pass (`WaitAwareScheduleRepairPass`)

The repair pass runs after the scheduler and gives each wait-anchored WMMA window one
fewer non-memory slot; memory producers go first, so ordinary fillers are carried past
anchor after anchor. Without help, the SALUs the scheduler spaced out drift down onto
their VALUs and the stall comes back.

With the knob on, `addSaSdstPinEdges()` (`WaitAnchoredReadyQueue.hpp`) pins every
SA_SDST waiter, and every SALU/SMEM up to the last waiter of the segment, ahead of its
next matrix instruction -- the same mechanism `addPrefetchPinEdges()` uses for prefetch
address math. All SALUs are pinned, not only the producers of a waiter, because any SALU
restarts the drain (e.g. `s_sub_u32 s[sgprLoopCounterL]` drifting onto
`v_add_co_ci_u32`).

## Result (`f8_tn_maf_henry`, gfx1250, codegen only)

Main loop (`label_LoopBeginL` .. `label_LoopEndL`), SA_SDST waiters with a gap under 8
issue slots after the last SALU:

| | waiters with gap < 8 | `v_add_co_u32 ... IncA` gap |
|---|---|---|
| `SaSdstIssueGap: 0` | 14 (most at 0) | 0 |
| `SaSdstIssueGap: 8` | 0 | 9 |

WMMA / ds_load / `s_wait_alu` / total instruction counts are unchanged, and with the
knob off the output matches the baseline schedule exactly.

```asm
s_cmp_le_u32 s[sgprLoopCounterL], 4
s_cmov_b32 s[sgprGL2PrefetchIncA], 0
s_cmov_b32 s[sgprGL2PrefetchIncB], 0
s_sub_u32  s[sgprLoopCounterL], s[sgprLoopCounterL], 1
  ... 3 WMMA + 6 ds_load (9 issue slots) ...
v_add_co_u32 v[vgprGL2PrefetchAddrA_0], vcc_lo, v[vgprGL2PrefetchAddrA_0], s[sgprGL2PrefetchIncA]
```

## Limits and open items

- Not yet measured on hardware. The threshold (about 8, the SALU + SDST FIFO depth in
  the data-dependency spec) should be swept (6/8/10) against a trace.
- Prologue / epilogue address math (`s_and_b32` -> `v_add_co_u32` in the store path) is
  not covered.
- No unit tests yet for the counter, the holds or the repair pin edges.
