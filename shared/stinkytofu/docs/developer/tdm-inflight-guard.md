# TDM In-Flight Guard Pass

`createTDMInflightGuardPass` keeps every wave at or below a fixed number of
outstanding tensor (TDM) operations.

```cpp
STINKYTOFU_EXPORT std::unique_ptr<Pass> createTDMInflightGuardPass(
    int limit = kDefaultTDMInflightLimit, std::vector<Function*> functions = {});
```

## Why

On gfx1250 B0 a wave with more than 11 TDM operations outstanding at once can
deadlock the TDM (LCOMPILER-2572). The hardware requirement is an
`s_wait_tensorcnt 10` before every TDM issue unless the compiler can prove the
wave has no more than 10 outstanding at that point.

Nothing upstream enforces this. TensileLite's parameter space happens to stay at
or below 8 in flight per wave today (two LDS stages, at most four TDM ops per
stage per wave), and `StinkyWaitCntInsertionPass` places tensor waits for
correctness only. Deeper LDS rings (three or four TDM stages) can exceed the
limit, so this pass is the compiler-side backstop.

## Rule

The pass computes, for every TDM issue, an upper bound on this wave's tensor
counter immediately before the issue, over every CFG path:

| Instruction | Effect on the bound |
|---|---|
| `tensor_load_to_lds` | +1. A load whose descriptor was nulled still issues and retires on the counter, so it counts too. |
| `s_wait_tensorcnt N` | `min(bound, N)`. An immediate that cannot be decoded caps nothing. |
| `s_swappc_b64` (call) | Unchanged when the kernel's function list shows that no callable function issues a TDM op; otherwise the bound saturates. |
| CFG join | Maximum over the incoming paths. |

The counter is per wave and in order, so the bound is the number in flight.
Loops are iterated to a fixpoint, saturating at 64, which is more than the 6-bit
counter can hold. A kernel starts at 0; a callable function starts saturated
because its caller is unknown.

Where the bound before an issue exceeds `limit - 1`, the pass inserts

```asm
    s_wait_tensorcnt 10    // at most 11 TDM ops in flight per wave
    tensor_load_to_lds ...
```

immediately before that issue (shown for the default limit of 11). The
dataflow treats every issue as if such a wait stood before it; a wait at an
issue that is already within the limit changes nothing, so placing waits only
where the bound exceeds `limit - 1` gives the same fixpoint. That set of waits
is the smallest one that keeps every issue within the limit, and later issues
see the waits placed before them. A second run inserts nothing.

`tensor_store_from_lds` increments the same counter but is not modelled in
StinkyTofu yet; it belongs in `isTensorCounterOp` once it is.

## Control flow

The pass builds its own CFG from the function's instruction stream in emission
order, from `LABEL` instructions (or block names, for `.stir` input) and branch
targets. It does not use `BasicBlock` edges: `RegionClonePass` emits its clone as
a single block that keeps the region's internal labels and branches without
edges for them.

- A conditional branch has its target and its fall-through as successors; an
  unconditional branch only its target.
- A branch whose target the function does not define, including a
  register-target `s_setpc_b64` in a kernel, may land on any label: its bound
  flows into every label.
- `s_endpgm`, and a register-target `s_setpc_b64` in a callable function, end
  the path.
- A `FUNCTION_ASM_PLACEMENT_MARKER` is treated like a call, since
  `FlattenCalleesPass` later places a callable body there.
- Code no path reaches is neither analysed nor guarded.

## Pipeline placement

`buildGfx1250Pipeline` adds the pass, through `createFunctionToModuleAdaptor`,
right after the entry-only bucket that ends in `RegionClonePass`, at every
OptLevel. By then every pass that schedules, inserts or clones a TDM op or an
`s_wait_tensorcnt` (DAG scheduler, waitcnt insertion, cluster barrier, region
clone) has run, so the bound covers the stream that is emitted. The pass
receives `StinkyAsmModule::getFunctions()` so calls to activation bodies, which
issue no TDM op, keep the bound.

## Option

| Module option | Default | Meaning |
|---|---|---|
| `TDMInflightLimit` | 11 | Most TDM ops one wave may have in flight. 0 disables the pass. |

In `stinkytofu-opt` the pass is `--TDMInflightGuardPass`, with an optional
`limit=<n>` argument (`--TDMInflightGuardPass=limit=4`); a malformed value is
rejected. The tool does not hand the pass the module's callable functions, so
every call there is treated as unknown.

## Effect on today's kernels

None. Every gfx1250 kernel TensileLite generates today stays at or below 8 in
flight per wave, and when no issue exceeds the limit the pass does not touch the
IR at all and preserves every analysis. Its emitted assembly is byte-identical
to the pipeline without the pass.

When the pass does insert, it emits one analysis remark per function
(`StinkyTofuEnableRemarks`) with the number of waits, the limit, and the worst
bound it found before an issue.

## Tests

- `tests/unit/asm/TDMInflightGuardPassTest.cpp`
- `tests/filecheck/tdm_inflight_guard.stir`, `tests/filecheck/tdm_inflight_guard_limit.stir`
