# WaitCnt Insertion Test

This document describes each test in `tests/unit/asm/WaitCntInsertionTest.cpp`, which validates the `StinkyWaitCntInsertionPass`. The pass analyzes def-use chains across basic blocks and inserts `s_wait_dscnt` (and optionally `s_wait_tensorcnt`) instructions to ensure memory operations complete before their results are consumed.

---

## Test Infrastructure

### Test Fixture: `WaitCntInsertionTest`

All tests extend a shared GoogleTest fixture that provides:

- **`getArch()`** -- Returns the GFX1250 architecture triple `{12, 5, 0}` from the `GemmTileConfig`.
- **`parseIR(irString, converter)`** -- Parses an MLIR-style IR string into a `Function` using `StinkyIRConverter` and sets the GFX1250 `GemmTileConfig`.
- **`runInsertionPass(func)`** -- Runs `StinkyWaitCntInsertionPass` on the given `Function`.
- **Assertion helpers** -- `countWaitCnt(bb)`, `getAllWaitCnts(bb)`, `findWaitCntBefore(bb, inst)`, `findNthInst(bb, opcode, n)`, etc., all parameterized on a `BasicBlock&` reference.

### IR String Format

Tests define IR using raw string literals in the `st.func` format:

```
st.func @test_name() {
^block_label:
  dest = "st.opcode"(src0, src1) { issueCycles = N, latencyCycles = M }
  Successors: ^other_block
}
```

Key mnemonics: `st.ds_load_b32`, `st.ds_load_b64`, `st.ds_load_b128`, `st.v_add_f32`, `st.v_wmma_f32_16x16x32_bf16`, `st.tensor_load_to_lds`, `st.s_barrier`.

Register format: `v0` (single VGPR), `v[0:3]` (4-VGPR range), `s[0:3]` (SGPR range), `a[10:17]` (AGPR range).

---

## Test Summary

| Suite | Test Name | CFG | Scenario | Expected Waitcnts |
|-------|-----------|-----|----------|--------------------|
| 1 | BarrierWithDSRead | Single block | Unconsumed `ds_load` + barrier | None |
| 1 | BarrierWithDSReadTensorLoad | Single block | Unconsumed `tensor_load` + `ds_load` + barrier | None |
| 2 | DSReadBeforeWMMA | Single block | 4 `ds_load`s feeding 2 WMMAs | `dlcnt=2` before wmma1, `dlcnt=0` before wmma2 |
| 3 | CompleteTest | Single block | Full GEMM preloop: 8 `ds_load`s + `tensor_load` + 2 WMMAs + 2 barriers | `dlcnt=4` before each WMMA, none before barriers |
| 4 | NoLoop | Single block (no back-edge) | Uses before definitions, no loop | None |
| 4 | LoopOnly | Self-loop | Cross-iteration def-use via back-edge | `dlcnt=2` before 1st `v_add`, `dlcnt=0` before 2nd |
| 4 | TwoBlockChain | entry + self-loop | Cross-block loads (order `v0,v1,v2,v3`) consumed in loop | `dlcnt=1` before 1st `v_add`, `dlcnt=0` before 2nd |
| 4 | TwoBlockChain2 | entry + self-loop | Same as above, different load order (`v0,v2,v1,v3`) | `dlcnt=1` before 1st `v_add`, `dlcnt=0` before 2nd |
| 4 | MultiPredecessorMerge | Diamond (b1,b2 -> b3) | Per-path analysis at merge point | `dlcnt=0` before `v_add` in b3 |
| 4 | MultiPredecessorMerge2 | Chained diamond (b1,b2 -> b3 -> b4) | Cross-block propagation through merge | `dlcnt=2` in b3, `dlcnt=1` in b4 |

---

## Test Suite 1: Barrier Wait Insertion

Tests that barriers do NOT trigger waitcnts when there are no def-use dependencies. `StinkyWaitCntInsertionPass` uses def-use chain analysis rather than config-based barrier policies.

### BarrierWithDSRead

**Input IR:**
```
ds_load_b64 v[0:1], v10
s_barrier
```

**What is tested:** A `ds_load` whose result (`v[0:1]`) is never consumed by any subsequent instruction, followed by a barrier.

**Expected output:** No `s_wait_dscnt` or `s_wait_tensorcnt` inserted. Since no instruction uses `v[0:1]`, there is no def-use dependency to satisfy.

### BarrierWithDSReadTensorLoad

**Input IR:**
```
tensor_load_to_lds s[0:3], s[10:17]
ds_load_b64 v[0:1], v10
s_barrier
```

**What is tested:** A `tensor_load_to_lds` and `ds_load` with no consumer, followed by a barrier. Neither instruction has `MemTokenData`, so the pass's tensor-wait heuristic does not fire.

**Expected output:** No `s_wait_dscnt` or `s_wait_tensorcnt` inserted.

---

## Test Suite 2: DS Read Insertion Before WMMA

Tests that waitcnts are inserted before instructions that consume `ds_load` results through def-use dependencies.

### DSReadBeforeWMMA

**Input IR:**
```
ds_load_b64 v[20:21], v0   -- load 0
ds_load_b64 v[30:31], v0   -- load 1
ds_load_b64 v[40:41], v0   -- load 2
ds_load_b64 v[50:51], v0   -- load 3
v_add_f32 v60, v61, v62    -- independent ALU
wmma a[10:17], v[20:27], v[30:37], a[10:17]   -- consumes loads 0,1
v_add_f32 v60, v61, v62    -- independent ALU
wmma a[10:17], v[40:47], v[50:57], a[10:17]   -- consumes loads 2,3
```

**What is tested:** Two groups of `ds_load`s feed two `v_wmma` instructions. The pass must insert separate waitcnts before each WMMA, waiting only for the loads it depends on.

**Expected output:**
- `s_wait_dscnt 2` before wmma1 (wait for loads 0,1; leave loads 2,3 outstanding)
- `s_wait_dscnt 0` before wmma2 (wait for all remaining loads 2,3)

The other counter fields (`vlcnt`, `vscnt`, `dscnt`, `kmcnt`) are all `-1` (unused).

---

## Test Suite 3: Complete Preloop + WMMA + Barrier Pattern

### CompleteTest

**Input IR:**
```
ds_load_b128 v[0:3], v40      -- preloop load 0
ds_load_b128 v[4:7], v40      -- preloop load 1
ds_load_b128 v[8:11], v40     -- preloop load 2
ds_load_b128 v[12:15], v40    -- preloop load 3
tensor_load_to_lds s[0:3], s[10:17]
ds_load_b128 v[16:19], v40    -- load 4
ds_load_b128 v[20:23], v40    -- load 5
ds_load_b128 v[24:27], v40    -- load 6
ds_load_b128 v[28:31], v40    -- load 7
wmma a[50:57], v[0:7], v[8:15], a[50:57]     -- consumes preloop loads 0-3
s_barrier
ds_load_b128 v[0:3], v40      -- load 8
ds_load_b128 v[4:7], v40      -- load 9
ds_load_b128 v[8:11], v40     -- load 10
ds_load_b128 v[12:15], v40    -- load 11
wmma a[50:57], v[16:23], v[24:31], a[50:57]  -- consumes loads 4-7
s_barrier
```

**What is tested:** The full GEMM mainloop pattern with preloop data loads, tensor prefetch, two WMMA consumers, and barrier synchronization points. This is the most realistic test case.

**Expected output:**
- `s_wait_dscnt 4` before wmma1 (wait for preloop loads 0-3; leave loads 4-7 outstanding)
- `s_wait_dscnt 4` before wmma2 (wait for loads 4-7; leave loads 8-11 outstanding)
- No waitcnt before either barrier (no `MemTokenData`, no def-use dependency through barriers)

---

## Test Suite 4: Basic Block State Tracking

Tests cross-block def-use dependency tracking across various CFG patterns: non-loop, self-loop, preloop+loop, and diamond CFGs.

### BasicBlockStateTracking_NoLoop

**CFG:** Single block (no loop, no back-edge)

**Input IR:**
```
^entry:
  v_add_f32 v4, v0, v2   -- uses v0, v2 (not yet loaded)
  v_add_f32 v4, v1, v3   -- uses v1, v3 (not yet loaded)
  ds_load_b32 v0, v10    -- defines v0 AFTER use
  ds_load_b32 v2, v10
  ds_load_b32 v1, v10
  ds_load_b32 v3, v10
```

**What is tested:** Uses appear before definitions in program order. Since this is not a loop, there is no cross-iteration dependency. The def-use chain builder does not link the `v_add` uses to the later `ds_load` definitions.

**Expected output:** No waitcnt inserted.

### BasicBlockStateTracking_LoopOnly

**CFG:** `entry -> loop_start -> loop_start` (self-loop)

**Input IR (loop_start):**
```
^loop_start:
  v_add_f32 v4, v0, v2   -- uses v0, v2 from previous iteration
  v_add_f32 v4, v1, v3   -- uses v1, v3 from previous iteration
  ds_load_b32 v0, v10    -- loads for next iteration
  ds_load_b32 v2, v10
  ds_load_b32 v1, v10
  ds_load_b32 v3, v10
  Successors: ^loop_start
```

**What is tested:** In a loop, the `ds_load`s from the end of one iteration must complete before the `v_add`s at the start of the next iteration consume their results. The pass detects this cross-iteration dependency through the loop back-edge.

**Expected output:**
- `s_wait_dscnt 2` before first `v_add` (wait for v0, v2 from ds_load order `[v0, v2, v1, v3]`; leave v1, v3 outstanding)
- `s_wait_dscnt 0` before second `v_add` (wait for all remaining: v1, v3)

### BasicBlockStateTracking_TwoBlockChain

**CFG:** `entry -> loop_start -> loop_start` (self-loop)

**Input IR:**
```
^entry:                           ^loop_start:
  ds_load_b32 v0, v10              v_add_f32 v4, v0, v2
  ds_load_b32 v1, v10              v_add_f32 v4, v1, v3
  ds_load_b32 v2, v10              ds_load_b32 v0, v10
  ds_load_b32 v3, v10              ds_load_b32 v2, v10
  Successors: ^loop_start          ds_load_b32 v1, v10
                                   ds_load_b32 v3, v10
                                   Successors: ^loop_start
```

**What is tested:** Cross-block dependency where entry block's `ds_load`s define registers consumed by the loop block's `v_add`s. The entry block loads in order `[v0, v1, v2, v3]`.

**Expected output:**
- `s_wait_dscnt 1` before first `v_add` (uses v0 at pos 0, v2 at pos 2 -> wait through pos 2, leave v3 -> dlcnt=1)
- `s_wait_dscnt 0` before second `v_add` (wait for v3, the last remaining)

### BasicBlockStateTracking_TwoBlockChain2

Same structure as TwoBlockChain but with different load order in both blocks.

**Entry loads:** `v0, v2, v1, v3` (different from TwoBlockChain's `v0, v1, v2, v3`)
**Loop loads:** `v0, v1, v2, v3`

**Expected output:**
- `s_wait_dscnt 1` before first `v_add` -> dlcnt=1
- `s_wait_dscnt 0` before second `v_add` -> dlcnt=0

This tests that the pass correctly handles different load orders across blocks and loop iterations.

### BasicBlockStateTracking_MultiPredecessorMerge

**CFG (diamond):**
```
    entry
     / \
    b1   b2
     \ /
      b3
```

**Input IR:**
```
b1: ds_load v0, ds_load v1
b2: ds_load v2, ds_load v3, ds_load v4
b3: v_add v5, v0, v1
```

**What is tested:** Multi-predecessor merge where b3 has two incoming paths with different outstanding loads. The pass performs per-path analysis:
- Path b1: `[v0, v1]` -> `v_add` uses v0, v1 -> needs `dlcnt=0`
- Path b2: `[v2, v3, v4]` -> `v_add` uses v0, v1 (not present) -> no wait needed
- Final: `min(0, IGNORE) = 0`

**Expected output:** `s_wait_dscnt 0` before `v_add` in b3.

### BasicBlockStateTracking_MultiPredecessorMerge2

**CFG (chained diamond):**
```
    entry
     / \
    b1   b2
     \ /
      b3
      |
      b4
```

**Input IR:**
```
b1: ds_load v0, ds_load v1, ds_load v2
b2: ds_load v3, ds_load v4, ds_load v5
b3: v_add v6, v3, v3
b4: v_add v7, v0, v1
```

**What is tested:** Chained multi-predecessor analysis. Block3 uses v3 (from b2 path only), and block4 uses v0, v1 (from b1 path, propagated through b3).

**Block3 analysis:**
- Path b2: `[v3, v4, v5]` -> uses v3 -> `dlcnt=2` (leaves v4, v5)
- Path b1: `[v0, v1, v2]` -> uses v3 (not present) -> no wait
- Result: `dlcnt=2`

**Block4 analysis:**
- b4's predecessor b3 propagates outstanding loads from both paths
- v0, v1 from b1 path still outstanding -> needs wait
- Result: `dlcnt=1`

**Expected output:**
- `s_wait_dscnt 2` before `v_add` in b3
- `s_wait_dscnt 1` before `v_add` in b4
