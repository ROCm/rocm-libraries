# DAG placement of prefetches and wait-prone fillers (gfx1250)

How the CDNA5 DAG scheduler places global/flat prefetches, SALU/VALU fillers that would cost
an `s_wait_alu`, and the rest of the fillers, the knobs that control it, and what it measured.

## Why the DAG needs rules for these

The DAG models hazards in **cycles**. InsertWaitAlu, which runs after the DAG, decides
waits by **op counts** (`HWModel::WaitHide`). When the DAG thinks a gap is covered and the
pass does not, the pass inserts an `s_wait_alu`.

Prefetch placement is not a hazard but a **performance** placement. Left alone, the DAG
issues a prefetch as soon as its address is ready, far ahead of the `tensor_load_to_lds` it
belongs with. Develop avoided that by accident: WaitAwareScheduleRepairPass carried the
prefetch forward to the segment end, the barrier right before the tensor_load.

## Rules, in order of measured impact

1. **Prefetch placement** (`PrefetchLeadWmmas`). Each prefetch is grouped with the
   tensor_load that follows it; the load's window is the planned window of the stage
   barrier above it (`barrierWmmaThresholds_`).
   - Multi-stage loop body (tensilelite `HalfPLR`, 3 TDM stages): the k-th of n prefetches is
     held until window `load - lead + k * lead / n`.
   - Single-stage loop body (lead < 8): the whole group is released at the load window and
     the stage barrier waits until the ready group has issued, so the group lands just
     before `s_barrier_wait -3`. A group that is not fully ready never holds the barrier.
   - With a lead set, prefetches have their own ready queue; they are not fillers and never
     count against the filler quota. WaitAwareScheduleRepairPass pins each prefetch, and the
     VALU chain computing its address, ahead of the next WMMA so the repair does not carry
     them off.
2. **Wait-prone fillers** (`WaitAluHoldStrictCount`, only with ESM2, the only mode in which
   InsertWaitAlu runs). `WaitAluTracker` is InsertWaitAlu's
   own scoreboard (shared `stepInstruction()`, so the answers cannot diverge); `query(inst)`
   returns the `s_wait_alu` the pass would emit if `inst` issued next. A filler whose wait
   would be strict (count ≤ N, and in multi-stage loops any `vm_vsrc` wait, which drains
   older LDS reads whatever its count) is held until the two windows before the next
   `s_barrier_wait`. Not held when it is the last pending predecessor of real work, or when
   it feeds a prefetch's address.
3. **ds_load pacing** (`DsSlotFirst`). In a ds stream of 2+ ds_loads per WMMA window, a
   ds_load that still fits the window goes before fillers and prefetches, and a hazard-hoisted
   VALU that cannot co-issue now is not forced. Otherwise each stolen slot shifts the stream,
   and the tensor_load, one window late.
4. **Critical chains.** A filler whose consumer (prefetch, tensor_load, ds_load) is planned
   within 20 windows skips the filler quota and takes the earliest window with a free
   co-issue slot, or issues as soon as it is ready once every remaining window is full.
5. **Even spread** (`EvenSpreadFillers`) of the remaining fillers: each window is owed
   `ceil(fillers / WMMAs)` and closes once that quota is met; the quota never holds a
   ds_load or tensor_load.

The Phase G fallback skips held prefetches and fillers while anything else can go, and
takes them last, so a hold can delay work but never strand it.

## Knobs

All are `ModuleOptions`, settable from tensilelite. The full list of scheduler tuning values
is in [Scheduler tuning parameters](../user/scheduler-tuning-parameters.md). `stinkytofu-opt` flags set the matching
`dagFeatures` field (default off there).

| ModuleOption | default | stinkytofu-opt flag | meaning |
|---|---|---|---|
| `PrefetchLeadWmmas` | 25 (KernelWriter: stage < 64 WMMAs 0; no HalfPLR 4; else 25 sub-byte A, 40 otherwise) | `--prefetch-lead-wmmas=N` | prefetch lead; < 8 = single-stage grouping; 0 = off |
| `WaitAluHoldStrictCount` | 2 | `--wait-alu-hold-strict-count=N` | strict-wait threshold; < 0 = no hold |
| `DsSlotFirst` | true | `--ds-slot-first` | ds_load before fillers in a saturated ds stream |
| `EvenSpreadFillers` | true | – | filler quota per window |
| `WarGateWmmas` | -1 | `--war-gate-wmmas=N` | WMMA-src → ds_load overwrite gap; -1 = derived |

## Measured effect

gfx1250, b8-3, develop 40b5cbed as REF, 3 reps with REF before and after each kernel,
median of the 6 comparisons (medium mxf4: 10). MAF = 4096×4096×65536 TN, medium =
2048×2048×65536 TN, henry = F8BS 256×256×256.

| | % vs develop |
|---|---|
| henry | +2.3 (handwritten golden: +1.4) |
| MAF bbs / f8 / mxf4 / mxf8 / nvf4 | +0.4 / +0.5 / +0.2 / +0.5 / +0.15 |
| medium bbs / f8 / mxf4 / mxf8 | +2.2 / −0.1 / ≈ +0.4 / −1.2 |

SQTT WMMA-issue bubbles per main-loop iteration, henry: develop 687, this change 497,
handwritten 578. Most of what remains is the stage-barrier and tensorcnt-barrier waits
(waves waiting for each other).

Findings behind the rules:

- A prefetch early in its stage costs 2–5%; a single-stage loop wants the group right
  before the stage barrier (+2% over develop on medium mxf4 in a hand edit), a multi-stage
  loop a staggered lead. A lead of 80 on henry is −14%.
- A stage under 64 WMMAs (MAB, medium mxf8) has no room for a lead: the address chain is
  ready only a few windows before the tensor_load, and a prefetch issued there is worse
  than one grouped at the tensor_load (lead 0, develop's placement).
- An LDS-address swap (`v_xor`) issued while ds_loads still read the old address pays a
  `vm_vsrc` wait (about 100 cycles/iteration on medium mxf4). Moving it after the barrier
  is worse: the next stage's first ds_loads read the new address right after the barrier.
- A filler or a forced VALU that takes a ds_load's slot in a saturated stream pushes the
  tensor_load one window late, which is −3% on medium f8.
- Even spread helps the f4 MAF kernels (+1.5 vs off) and is neutral elsewhere once the
  critical chains are exempt.

## Open

- The lead per loop shape (25/40, single-stage grouping) is a measured default, not a
  derivation; a prefetch-distance sweep per kernel family is the next step.
