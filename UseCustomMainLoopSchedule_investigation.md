# `UseCustomMainLoopSchedule` — Investigation

Scope: tensilelite (`projects/hipblaslt/tensilelite/Tensile/`). Question: for a given Tensile
input YAML, what changes when the boolean `UseCustomMainLoopSchedule` (UCMLS) is flipped
True ↔ False — which other flags get auto-set, and which code sections run differently.

## Key disambiguation

UCMLS is **not** the same as a "custom kernel". The custom-kernel / benchmark / library-IO
machinery (`CustomKernels.py`, `TensileLogic/HandleCustomKernel.py`, `BenchmarkStructs.py`,
`BenchmarkProblems.py`, `LibraryIO.py`) is keyed entirely on `CustomKernelName` / `CustomKernels`.
UCMLS does **not** propagate into any of those files. Toggling UCMLS in a YAML changes nothing
in the custom-kernel-loading path. The two layers that matter are **solution construction** and
**kernel codegen**.

## Default value & resolution

- Default: `-1` (auto). Valid values `[-1, 0, 1]`.
  - `GlobalParameters.py:536` (default), `ValidParameters.py:1015` (valid set).
- Resolution: `Solution.py:2416–2484` (`assignDerivedParameters`).
  - `-1` → resolves to `1` if `hasCustomSchedule(state)` finds a CMS schedule, else `0`
    (`Solution.py:2424,2427`; probe at `Components/CustomSchedule.py:518`).
  - `1` → rejected/errored if not supported.
  - `-1` → silently demotes to `0` if unsupported (no rejection).
- `hasCustomSchedule()` short-circuits to False unless: UCMLS truthy AND
  `EnableMatrixInstruction` AND ISA == gfx950 (9,5,0) AND non-mixed dtype
  (`CustomSchedule.py:523`). Even with UCMLS=True the custom schedule only *engages* when a
  registered schedule matches; otherwise it falls back. Codegen branches key off the **flag**,
  not off a successful match.

---

## 1. Flags auto-set / forced / constrained by UCMLS

### Solution-construction layer (`Solution.py`)

When UCMLS is being considered (`-1` or `1`), set before the availability probe:

| Affected param | Resolved `1` (CMS) | Resolved `0` (non-CMS) |
|---|---|---|
| `SwapGlobalReadOrder` | forced `0` (`:2421`) | re-derived: forced `0` if A/B GR-mode mismatch (`:2479–2481`) or `UnrollLoopSwapGlobalReadOrder` truthy (`:2483–2484`) |
| `UsePLRPack` | forced `0`; YAML value ignored (`:2422`; cf. `ValidParameters.py:1047`) | restored from `backup_UsePLRPack` then knocked to `0` unless 7-condition gauntlet passes (`:2448–2467`) |
| `MfmaInitCVgprs` | left CMS-initialized (non-CMS block skipped) | set `True` if `UseMFMAF32XEmulation` (`:2444–2445`) |
| `ForceUnrollSubIter` / `numSubTiles` | left CMS-initialized | F32X SubIter-disable runs (`:2442–2443`) |

`UsePLRPack` is backed up first (`:2417`) so the original YAML value survives into the non-CMS path.

**The 7-condition `UsePLRPack` gauntlet** (non-CMS, `:2448–2467`): restored to `1`, then forced
`0` if ANY of: not `EnableMatrixInstruction`; not `UseF32XEmulation`; `_ScheduleIterAlg != 3`;
not `ForceUnrollSubIter`; `DirectToLds != 1`; `PrefetchGlobalRead == 0`; `PrefetchLocalRead == 0`.

**Rejection rules** (only when explicitly `== 1`):
- incompatible with `TailloopInNll=True` (`:2429–2431`)
- incompatible with `UseSubtileImpl=True` (`:2433–2434`)
- `==1` requested but `hasCMS` False → reject (`:2425–2426`)

### Kernel-codegen layer (`KernelWriter.py`) — internal state forced when UCMLS=True

| State | Effect | Citation |
|---|---|---|
| `doFullPackCodePrefetch` | `= UsePLRPack and not UCMLS` → always False under UCMLS | `:8235` |
| `doPackPreSchedulingThisLoop/NextLoop` | prePack enable block gated `(not UCMLS) and numItersPLR` → not enabled | `:8238–8243` |
| `lrvwTileA/B` | CMS exempt from XF32 forcing to 1; keep `VectorWidthA/B` | `:6547–6564` |
| `checkVregOverflowTF32Emu` | returns False → no VGPR-overflow adjustment | `:8316–8317` |
| `useCommonSgprSwap` | StoreSwapAddr route gated `and not UCMLS` | `:6685–6689` |

A selected CMS schedule can itself mutate the kernel flag `SwapGlobalReadOrder = True`
(`CustomSchedule.py:1040`), reachable only when UCMLS=True.

---

## 2. Divergent code sections (True vs False)

Fundamental swap = **main-loop scheduling strategy**:

- **UCMLS=False** (default): each iteration scheduled inline by `_makeSubIterSchedule()`;
  NLL/NGLL emit full codegen; `closeLoop` emitted directly.
- **UCMLS=True**: all per-iteration instruction streams (MFMA, local read/write, pack, swaps)
  are accumulated and handed to `customMainLoopSchedule()` (`CustomSchedule.py:311`), which emits
  one parameterized `MAINLOOP` rocisa macro from a hand-tuned index schedule in
  `_SCHEDULE_REGISTRY`. The no-load loops then invoke that macro with different guard args.

Specific divergent sections:

| Section | True vs False | Citation |
|---|---|---|
| Main-loop body | stream accumulation + `customMainLoopSchedule` vs inline `_makeSubIterSchedule` | `KernelWriter.py:3936–4437` |
| closeLoop | folded into macro as `loopCounterCode` vs emitted directly | `:4431–4461` |
| NoLoadLoopBody | single `MacroInstruction("MAINLOOP", …)` vs full body codegen | `:3053–3062` |
| NLL pre-body waits/sync | suppressed (`not UCMLS` guard) | `:3656–3665` |
| open-loop PGR wait/sync | skipped | `:3704–3713` |
| directToLds M0-update / 2nd-GR wait | `skipWait=UCMLS`; skipped | `:3214,3731,3741–3742` |
| PLR=0 prefetch | dedicated half-local-read / half-pack branch | `:5217–5247` |
| SIMD-spec dispatch tail | called only on True path | `:4438` (impl `KernelWriterAssembly.py:17637`) |
| `MAINLOOP` macro generation | only runs under True: strips comments, validates via `cmsv.isValid`, interleaves at registry index positions wrapped in `ValueIf` guards (`\useGR`, `\usePLR`, `\useGRInc`, `\useLoop`, `\ID`) so one body serves main loop / NGL / NLL | `CustomSchedule.py:311–515` |

### Naming / data carry-through
- Kernel/solution name: token `'CMS'` appended when truthy (`Naming.py:188–189`).
- Resolved value carried into contraction Solution as `customMainLoopScheduling`
  (`Contractions.py:648,741`) — data only, no branching.

---

## Bottom line

Flipping the flag changes two things:
1. A cascade forcing `SwapGlobalReadOrder=0` and `UsePLRPack=0` (plus codegen-internal
   prefetch/XF32 state) when CMS resolves on; the non-CMS path instead re-derives those from
   other params.
2. The entire main-loop instruction-scheduling approach: default per-iteration scheduler vs. a
   pre-baked hand-tuned `MAINLOOP` macro, with the no-load loops reduced to macro invocations and
   several default waits/syncs skipped.
