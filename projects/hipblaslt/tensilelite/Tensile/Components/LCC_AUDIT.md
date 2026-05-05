# LCC Instruction Inventory Audit (bead 2bu.1)

Source revision: validator branch tip `5b4afa4ba0` (`users/alvasile/validator_long_term_plans`).

Files inspected:
- `Tensile/Components/CustomSchedule.py` (LCC slot tables, `customMainLoopSchedule`, `removeComments`, `scheduleInst`)
- `Tensile/Components/ScheduleCapture.py` (`build_idmap`, `_min_issue_quad_cycles_for`, `_make_node`, `build_dataflow_graph`, `_SCC_OPCODE_FLAGS`, `_class_tag_from_category`)
- `Tensile/Components/CMSValidator.py` (`ValidatorInstruction.min_issue_quad_cycles_base`)
- `Tensile/KernelWriter.py` (`_loopBody`, the `closeLoop(..., False)` callsite that supplies LCC)
- `Tensile/KernelWriterAssembly.py` (`closeLoop` definition, lines 6730-6906)

## How LCC reaches the captured macro

1. `_loopBody` calls `customMainLoopSchedule(..., self.closeLoop(kernel, tPA, tPB, unrollIdx, False))` (`KernelWriter.py:4569-4573`). The last positional argument becomes `loopCounterCode` inside CMS.
2. `customMainLoopSchedule` immediately runs `removeComments(loopCounterCode)` (`CustomSchedule.py:325`). `removeComments` filters out `TextBlock`, `SCBranchSCC1` and `SNop` (`CustomSchedule.py:276-281`). **The branch instruction is therefore not part of the LCC bucket** — it is consumed elsewhere (the macro-walker emits the loop-edge separately).
3. `build_idmap(..., loopCounterCode=loopCounterCode, ...)` registers the surviving instruction list under category `'LCC'` (`ScheduleCapture.py:945`).
4. `scheduleInst('LCC', [start, end], instructionList)` walks the schedule entry: each integer in the list places the next remaining instruction at that miIndex slot (`CustomSchedule.py:400-423`). For LCC every entry has length 2, so exactly two instructions are placed.
5. `build_dataflow_graph` then constructs a node for each LCC instruction in the per-body sidecar but excludes them from `nodes_by_identity` because of the explicit guard `node.identity[0] != "LCC"` (`ScheduleCapture.py:2766-2767`). That is the exclusion `2bu.2` must remove.

## Step 1 — LCC instruction inventory

`closeLoop(kernel, tPA, tPB, unrollIdx, finalLoop=False)` produces the loopCounterCode that flows into LCC. The relevant branch is the non-tail-loop, `finalLoop=False` path in `KernelWriterAssembly.py:6814-6892, 6895-6905`. There are two sub-branches:

### Sub-branch A — common path (`KernelWriterAssembly.py:6883-6892`)

Triggered when EITHER `kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) != 0`, OR `endCounter == 0`, OR `kernel["PrefetchGlobalRead"] > 2`. Emits, in order:

| # | rocisa class | Operands | Reads | Writes | Notes |
|---|--------------|----------|-------|--------|-------|
| 1 | `SSubU32`    | `dst=loopCounter, src0=loopCounter, src1=1` | sgpr loopCounter | sgpr loopCounter, SCC | dec counter |
| 2 | `SCmpEQI32`  | `src0=loopCounter, src1=hex(endCounter)`    | sgpr loopCounter | SCC                  | counter == endCounter? |

The `SCBranchSCC1` that is conditionally added afterwards (`KernelWriterAssembly.py:6900-6902`) is filtered out by `removeComments` and never reaches LCC.

### Sub-branch B — `endCounter > 0` AND `ASEM % (DepthU*2) == 0` AND `PGR <= 2` (`KernelWriterAssembly.py:6837-6882`)

For `PrefetchGlobalRead == 2`:

| # | rocisa class    | Operands | Reads | Writes | Notes |
|---|-----------------|----------|-------|--------|-------|
| 1 | `SCmpEQU32`     | `src0=sgpr("StaggerU"), src1=0` | sgpr StaggerU | SCC | choose decrement of 1 vs 2 |
| 2 | `SCSelectB32`   | `dst=tmp, src0=hex(2), src1=hex(1)` | SCC | sgpr tmp | (SCC reader; not in `_SCC_OPCODE_FLAGS` — already covered there as `SCSelectB32`) |
| 3 | `SSubU32`       | `dst=loopCounter, src0=loopCounter, src1=sgpr(tmp)` | sgpr loopCounter, sgpr tmp | sgpr loopCounter, SCC | dec counter (variable) |
| 4 | `SCmpEQI32`     | `src0=loopCounter, src1=hex(endCounter)` | sgpr loopCounter | SCC | counter == endCounter? |

For `PrefetchGlobalRead == 1`, items 1-3 collapse to a single `SSubU32` with `src1=1`.

The `noExit` sub-case (lines 6863-6879) drops both `decCode` and `condCode`; the LCC bucket then becomes empty for that codepath. None of the captured CMS schedules currently hit this branch (see Step 3).

### Class summary

Surveying both sub-branches, the union of rocisa classes that LCC can carry is:

| Class           | Default issue cycles (quad-cycles) | Wait-state notes |
|-----------------|------------------------------------|------------------|
| `SSubU32`       | 1                                  | SCC writer; default 1 (matches `ValidatorInstruction.min_issue_quad_cycles_base = 1`, `CMSValidator.py:143`) |
| `SCmpEQI32`     | 1                                  | SCC writer (no sgpr dst); default 1 |
| `SCmpEQU32`     | 1                                  | SCC writer; default 1 |
| `SCSelectB32`   | 1                                  | SCC reader; default 1 |
| `SBranch`       | 1                                  | unconditional jump; only emitted in the `noExit && finalLoop` corner (does not reach captured LCC in current schedules) |

`SCBranchSCC1` and `SCBranchSCC0` are intentionally NOT in the LCC bucket — `removeComments` strips `SCBranchSCC1`, and `SCBranchSCC0` is only emitted by the `finalJump` path with `finalLoop=True` which is not the path supplying LCC under CMS.

`SAddU32`, `SAddCU32`, `SSubBU32`, `SCmpLeI32`, `SCmpLeU32` (mentioned as "likely candidates" in the bead description) are NOT emitted into LCC by `closeLoop(..., finalLoop=False)`. `SAddU32` and `SCmpLeU32` appear only in the tail-loop branch (`tailLoop=True`, `KernelWriterAssembly.py:6789-6813`) which is invoked from a different callsite and never feeds LCC.

## Step 2 — Per-class issue cycles

Authoritative source: `CMSValidator.py:137-158`.

```
class ValidatorInstruction:
    min_issue_quad_cycles_base: ClassVar[int] = 1
    def min_issue_quad_cycles(self) -> int:
        return self.min_issue_quad_cycles_base
```

Only `SNop` (`CMSValidator.py:447-452`) overrides this with `base + wait_state`. Every other validator class — including all the SALU classes that LCC emits — inherits the base of 1 quad-cycle. `_min_issue_quad_cycles_for` in `ScheduleCapture.py:2632-2649` mirrors the same dispatch (special-case SNop, default 1).

**Assumption (flag for verification by `2bu.2` implementer):** the dataflow model treats every SALU class as a 1-quad-cycle issue. CDNA 4 ISA section 7.6 wait-state tables for SCC chaining (e.g. SCC producer-to-consumer latency) are not modelled in either CMSValidator or ScheduleCapture today. The structural validator that previously approximated this was removed (per the comment at `ScheduleCapture.py:2628-2629`: "Post-`8nz` this helper is the canonical per-instruction cost table — the structural-side simulators are deleted."). For the purposes of `2bu.2` cross-body cycle counting, treating every LCC instruction as 1 quad-cycle is the conservative, behaviourally-consistent choice. If hardware-accurate wait-states are needed later, the table in `_SCC_OPCODE_FLAGS` (`ScheduleCapture.py:2158-2220`) already enumerates the SCC producer/consumer relationships and can be augmented with per-class wait-state counts in a follow-up bead.

## Step 3 — Per-kernel uniformity check

Methodology: parsed every `'LCC' :` literal in `Tensile/Components/CustomSchedule.py` (98 occurrences, line numbers per bead description plus the rest spanning lines 924..7757). For each entry I tabulated the per-codepath slot-list length (= number of instructions emitted into LCC for that codepath).

| Per-codepath slot-list length | Number of occurrences |
|-------------------------------|-----------------------|
| 2                             | 98                    |
| any other length              | 0                     |

Span (max slot − min slot + 1) within a single codepath:
- 73 schedules use `[N, N]` (both instructions placed at the same miIndex; e.g. line 924 `'LCC' : [[47, 47]]`).
- 10 schedules use `[N, N+1]` (one instruction per consecutive miIndex; e.g. line 1781 `'LCC' : [[45, 46]]`, line 4308 `'LCC' : [[189, 190]]`).
- The remaining 15 use the `numMfma-K` symbolic forms (e.g. `[numMfma-2, numMfma-1]`, `[n_mfma-1, n_mfma-1]`) which resolve to one of the two patterns above.

Multi-codepath cases (`numCodePaths == 2`) such as line 1032 `'LCC': [[46,46], [45,46]]` carry the same 2-instruction inventory in both codepaths but place them at different miIndex slots. No kernel deviates from "exactly 2 LCC instructions per codepath."

**Conclusion: the LCC inventory is uniform.** Every captured kernel — across TF32, BF16, F8, F32, dot2, sparse and conversion variants — emits the same `(SSubU32, SCmpEQI32)` pair into LCC under sub-branch A. No CMS-scheduled kernel currently exercises sub-branch B (the slot tables would have to be 3- or 4-wide to absorb the extra `SCmpEQU32` + `SCSelectB32`, and none are). If a future schedule activates sub-branch B, both `_class_tag_from_category` and `_min_issue_quad_cycles_for` already cover the additional classes (default 1 quad-cycle each).

## Step 4 — Cycle-cost summary (typical kernel)

Per body, per codepath:

```
SSubU32    : 1 quad-cycle
SCmpEQI32  : 1 quad-cycle
-----------
LCC total  : 2 quad-cycles
```

Multiplied across the 4 captured bodies (`ml_prev`, `ml`, `n_gl`, `n_ll`), the aggregate cross-body LCC cost is at most `4 * 2 = 8 quad-cycles` per outer iteration — but note that the `n_gl` and `n_ll` bodies have `\useLoop == 0` (`CustomSchedule.py:459-460`), so the macro guard suppresses the LCC instructions in those bodies. **Effective cost: 2 quad-cycles in `ml_prev` + 2 quad-cycles in `ml` = 4 quad-cycles per outer iteration.** This matches the existing test invariant `LCC not in nll_categories` at `Tests/unit/test_ScheduleCapture.py:754`.

## Step 5 — Implementation hints for `2bu.2`

### Drop the LCC exclusion in `build_dataflow_graph`

`Tensile/Components/ScheduleCapture.py:2766-2767` currently reads:

```python
if (not (_is_swait(inst) or _is_sbarrier(inst) or _is_snop(inst))
        and node.identity[0] != "LCC"):
    nodes_by_identity[node.identity] = node
```

To bring LCC into the cross-graph identity set, simply drop the `node.identity[0] != "LCC"` clause (and update the comment at `ScheduleCapture.py:2682-2684` and `2763-2767`). Because every LCC instruction already gets a `GraphNode` constructed (via `_make_node` at `ScheduleCapture.py:2748`) and is appended to `nodes_per_body`, no separate construction work is required.

### `_min_issue_quad_cycles_for` — no new dispatch entries needed

Today the function (`ScheduleCapture.py:2615-2649`) only special-cases SNop and returns 1 for everything else. Since every LCC class (`SSubU32`, `SCmpEQI32`, plus the latent `SCmpEQU32`, `SCSelectB32`, `SBranch`) has a base cost of 1 quad-cycle, the existing default already produces the correct value. **No code change is required in `_min_issue_quad_cycles_for` for the current schedule inventory.**

If `2bu.2` wants to make the dispatch explicit (e.g. for documentation), the table to add right above the final `return 1` would be:

```python
# SALU/SCC classes that LCC emits (and other SALU classes already covered
# by _SCC_OPCODE_FLAGS). All default to 1 quad-cycle base; add wait-state
# overrides here if/when CDNA 4 ISA section 7.6 numbers are wired in.
_SALU_ISSUE_QUAD_CYCLES = {
    "SSubU32":      1,
    "SCmpEQI32":    1,
    "SCmpEQU32":    1,
    "SCSelectB32":  1,
    "SBranch":      1,
    "SAddU32":      1,   # used by tail-loop variant; not in current LCC
    "SCBranchSCC0": 1,   # not in LCC today (stripped/finalLoop-only)
    "SCBranchSCC1": 1,   # not in LCC today (removeComments strips it)
}
cls_name = type(rocisa_inst).__name__
if cls_name in _SALU_ISSUE_QUAD_CYCLES:
    return _SALU_ISSUE_QUAD_CYCLES[cls_name]
```

### Edge formation

LCC-instruction reads/writes are already covered by the `_SCC_OPCODE_FLAGS` rule registry (`ScheduleCapture.py:2158-2220`) — `SSubU32`, `SCmpEQI32`, `SCmpEQU32`, `SCSelectB32` all have entries that emit the correct `(reads, writes)` tuples including SCC. So once the exclusion is dropped, Phase 2 (resource resolution) will start emitting:

- An SCC-write edge from LCC's `SSubU32` to LCC's `SCmpEQI32` is overwritten — both write SCC, so per-byte latest-writer just makes the second the new writer for SCC. No data hazard inside LCC.
- An sgpr-read edge from LCC's `SCmpEQI32` to LCC's `SSubU32` (both reference `loopCounter`).
- Cross-body chains: the LCC `SSubU32` in `ml_prev` writes the loop counter; the LCC `SSubU32` in `ml` then reads + writes it. This is precisely the "cross-body cycle" the parent bead `2bu` wants to surface.

### Test surface to extend

- `Tests/unit/test_ScheduleCapture.py:748-754` currently asserts `"LCC" not in nll_categories`. Keep that — `n_ll` still has `\useLoop == 0` so LCC truly does not appear there. But add a positive-presence assertion for `ml_prev` and `ml` once the exclusion drops.
- Consider an LCC-specific identity-set assertion: `assert any(identity[0] == "LCC" for identity in graph.nodes_by_identity)` in a TF32-style fixture body.

## Step 6 — Open questions

1. **CDNA 4 ISA wait-state values for SALU SCC chaining.** I could not find authoritative CDNA 4 numbers in code or comments. CMSValidator and ScheduleCapture both default to 1 quad-cycle for SALU. If `2bu.2` (or a follow-up) wants hardware-faithful counts, the SAddCU32-after-SAddU32 (carry) and SCBranchSCC*-after-SCmp* (SCC) latencies should be looked up in section 7.6 of the CDNA 4 ISA reference and folded into the `_SALU_ISSUE_QUAD_CYCLES` table sketched above. None of the LCC classes currently emitted (`SSubU32`, `SCmpEQI32`) are SCC consumers feeding another SALU SCC consumer in the same bucket, so the practical impact today is nil.

2. **Sub-branch B activation.** Sub-branch B (PGR=2 + ASEM-divisible) would emit 3-4 LCC instructions, exceeding the 2-slot budget every CMS schedule currently allocates. It is unclear whether this branch is statically unreachable for CMS-eligible kernels, or whether it could be activated by a future tuning decision and silently break `scheduleInst`'s slot accounting. Worth a defensive assertion in `customMainLoopSchedule` (or a comment in `closeLoop`) that the LCC slot count must match the number of remaining-after-removeComments instructions.

3. **`SBranch` emission.** Sub-branch B's `noExit && finalLoop` corner converts the conditional branch to an unconditional `SBranch`. Since LCC supplies `finalLoop=False` exclusively, this corner is unreachable for LCC, but `SBranch` should still be added to the SALU coverage table if it ever appears in the captured macro elsewhere (the structural validator removal noted at `ScheduleCapture.py:2628-2629` may have left coverage gaps).

4. **`SCBranchSCC1` placement.** The branch is stripped from LCC by `removeComments` and re-emitted by the macro walker. `2bu.2` should confirm that the cross-body cycle the validator is checking includes (or correctly excludes) the branch's effective issue cycle. The branch reads SCC produced by `SCmpEQI32`; if that read-after-write must be counted, `2bu.2` may need to track the branch as a synthetic post-LCC node rather than treating it as part of the captured stream.
