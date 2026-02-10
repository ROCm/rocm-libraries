# StinkyConfigurableWaitCntPass - Precise Wait Count Insertion

'StinkyConfigurableWaitCntPass' inserts 's_waitcnt' instructions to ensure memory consistency on AMD GPUs. It tracks outstanding memory operations at the register level and computes precise wait counts based on actual dependencies.

> **For Developers:** To extend this pass for new instructions, see [Adding WaitCnt Instructions](../developer-guide/adding-waitcnt-instructions.md).

## Key Features

- **Precise tracking for loads**: Tracks specific registers with outstanding loads
- **Optimal wait counts**: Only waits for operations that are actually needed
- **Cross-block analysis**: Maintains state across basic block boundaries
- **Loop support**: Iterative dataflow analysis for convergence
- **Conservative stores**: Stores use "wait all" at barriers (no address analysis)

## How It Works

### Pass Flow

'''
+------------------------------------------------------------+
|              StinkyConfigurableWaitCntPass Flow            |
+------------------------------------------------------------+

              Function Entry
                    |
                    v
        +-----------------------+
        |  Detect Loops in CFG  |
        +-----------+-----------+
                    |
                    v
        +-----------------------+
        | Traverse CFG in RPO   |<-----+ Iterate until
        | (Reverse Post-Order)  |      | convergence
        +-----------+-----------+      | (for loops)
                    |                  |
                    v                  |
    +---------------------------+     |
    | For each BasicBlock:      |     |
    |                           |     |
    | 1. Compute Entry State    |     |
    |    (merge predecessors)   |     |
    |                           |     |
    | 2. Insert WaitCnts        |     |
    |    - Cross-block deps     |     |
    |    - Barriers             |     |
    |    - Intra-block deps     |     |
    |                           |     |
    | 3. Compute Exit State     |     |
    +-----------+---------------+     |
                |                     |
                v                     |
        +---------------+             |
        | State Changed?+-----Yes-----+
        +-------+-------+
                | No
                v
           Function Exit
'''

### Memory Operation State

The pass tracks outstanding operations in 'MemoryOperationState':

'''cpp
struct MemoryOperationState {
    // Scalar counters
    int globalLoadCount;   // Outstanding global loads
    int dsLoadCount;       // Outstanding LDS loads
    int globalStoreCount;  // Outstanding global stores (count only)
    int dsStoreCount;      // Outstanding LDS stores (count only)

    // Register-level tracking (LOADS ONLY)
    vector<StinkyRegister> outstandingGlobalLoads;  // Precise tracking
    vector<StinkyRegister> outstandingDSLoads;      // Precise tracking

    // Stores: count only (no register tracking)
    // Reason: Without address analysis, can't determine if stores alias
};
'''

**Key difference:**
- **Loads**: Precise register tracking -> Optimal wait counts
- **Stores**: Scalar count only -> Conservative "wait all"

## Register-Level Tracking

### Outstanding Load Queue

'''
+--------------------------------------------------------+
|         Outstanding Loads (FIFO Order)                 |
+--------------------------------------------------------+

  Oldest <------------------------> Youngest
  (issued first)                     (issued last)

  [v0] [v1] [v2] [v3]
   ^    ^    ^    ^
   |    |    |    +-- Most recent
   |    |    +------- 3rd issued
   |    +------------ 2nd issued
   +----------------- 1st issued
'''

### Computing Precise 'dlcnt'

When an instruction uses registers, find the **latest** outstanding load it needs:

'''
Example: v_fmac_f32 v4, v0, v2

Outstanding: [v0, v1, v2, v3]
             idx:0  1   2   3

Step 1: Find needed loads
    - Uses v0 (index 0) [x]
    - Uses v2 (index 2) [x]

Step 2: Find latest needed
    - latestNeededIndex = max(0, 2) = 2

Step 3: Compute dlcnt
    - Total: 4 loads outstanding
    - Wait up to index 2 -> wait for 3 loads (v0, v1, v2)
    - Can leave: 4 - 3 = 1
    - dlcnt = 1 [x]

Result: s_waitcnt dlcnt=1
'''

**Visual:**
'''
Before:  [v0] [v1] [v2] [v3]
          ^         ^
          +---------+--- Instruction needs v0 and v2
                         Must wait for v0, v1, v2

After s_waitcnt dlcnt=1:  [v3]
                          Only v3 remains
'''

## Cross-Block State Propagation

### State Flow Between Blocks

'''
+-------------------------------------------------------+
|                    Block 1 (Preloop)                  |
|                                                       |
|  Entry: { dsLoadCount=0, outstandingDSLoads=[] }    |
|                                                       |
|  Instructions:                                        |
|    ds_read_b32 v0, v10  --> [v0]                    |
|    ds_read_b32 v1, v10  --> [v0, v1]                |
|    ds_read_b32 v2, v10  --> [v0, v1, v2]            |
|    ds_read_b32 v3, v10  --> [v0, v1, v2, v3]        |
|                                                       |
|  Exit: { dsLoadCount=4, outstandingDSLoads=[...] }  |
+-------------------+-----------------------------------+
                    | State flows down
                    v
+-------------------------------------------------------+
|                    Block 2 (Loop)                     |
|                                                       |
|  Entry: { dsLoadCount=4, outstandingDSLoads=[...] } |
|                                                       |
|  Instructions:                                        |
|    s_waitcnt dlcnt=1    <--- Inserted!              |
|    v_fmac_f32 v4, v0, v2  <--- Safe to use         |
|    ds_read_b32 v0, v10                              |
|    ...                                               |
|    s_cbranch loop       <--- Back-edge              |
+-------------------+-----------------------------------+
                    | Feeds back (requires iteration)
                    +----------+
                               +-> Entry State
'''

### State Merging at Join Points: Multi-Path Analysis

The pass uses **multi-path analysis** to achieve optimal precision for GPU performance:

#### Strategy: Keep Predecessor States Separate

Instead of merging states early, the pass:
1. Collects all predecessor states into a vector
2. For each instruction, computes wait requirements from **each path separately**
3. Takes **minimum** (most restrictive) across all paths
4. Only then merges states for internal block processing

#### Single Predecessor -> Precise [x]

'''
     Block 1
Exit: [v0, v1, v2, v3]
          |
          v
     Block 2
  predecessorStates = [[v0, v1, v2, v3]]  <-- One state
'''

**Result:** Full precision maintained

#### Multiple Predecessors -> Multi-Path Analysis [x]

'''
  Block A          Block B
Exit: [v0, v1]  Exit: [v2, v3, v4]
      |              |
      +-----+--------+
            v
        Block C
  predecessorStates = [[v0, v1], [v2, v3, v4]]  <-- Kept separate!

  For instruction using v0:
    Path A: [v0, v1] -> v0 at index 0 -> dlcnt = 1
    Path B: [v2, v3, v4] -> v0 not present -> no wait
    Final: min(1, IGNORE) = 1  <- Optimal!
'''

**Result:** Optimal 'dlcnt' even with multiple paths [x]

**Why This Works:** Each path has its own temporal ordering. We compute what's needed for each path independently, then take the most restrictive to satisfy all paths.

#### Preloop + Loop -> Multi-Path Analysis [x]

'''
Iteration 1: Skip back-edge
  Block 1 (preloop)
Exit: [v0, v1, v2, v3]
          |
          v
  Block 2 (loop) <-----+ back-edge (skipped)
  predecessorStates = [[v0, v1, v2, v3]]  <-- Only preloop
          |              |
          +--------------+

Iteration 2+: Both paths analyzed separately
  Block 1 (preloop)      Block 2 (loop back)
Exit: [v0, v1, v2, v3]  Exit: [v0, v2, v1, v3]
          |                      |
          +----------+-----------+
                     v
              Block 2 (loop)
  predecessorStates = [[v0, v1, v2, v3], [v0, v2, v1, v3]]

  For instruction using v0, v2:
    Preloop path: v2 at index 2 -> dlcnt = 1
    Loop path:    v2 at index 1 -> dlcnt = 2
    Final: min(1, 2) = 1  <- Most restrictive!
'''

**Strategy:**
1. **Iteration 1:** Only preloop path -> Precise
2. **Iteration 2+:** Both paths analyzed -> Takes minimum -> Optimal for both!

### Why Multi-Path Analysis is Critical for GPUs

**GPU Performance Insight:** Every cycle of unnecessary waiting reduces throughput.

**Conservative merge (old approach):**
- Merge early -> Lose precision -> 'dlcnt=0' fallback
- Cost: 3-4 extra wait cycles per block entry
- Impact: Cumulative across all blocks and iterations

**Multi-path analysis (current):**
- Keep states separate -> Compute per-path -> Take minimum
- Cost: None! Optimal waits
- Impact: Maximum GPU utilization

## Loop Handling with Iteration

Loops require multiple passes until convergence, with special handling for preloop + loop patterns:

'''
Preloop + Loop Pattern:
------------------------------------------------
Iteration 1: (Skip back-edge)
  Preloop Exit: [v0, v1, v2, v3]
  --> Loop Entry: [v0, v1, v2, v3]  (from preloop only)
  --> Loop Exit: [v0, v2, v1, v3]   (reordered by loads in loop)

Iteration 2: (Merge with canonical ordering)
  Preloop Exit: [v0, v1, v2, v3]
  Loop Exit:    [v0, v2, v1, v3]
  --> Merge: Sort by register index -> [v0, v1, v2, v3]
  --> Loop Entry: [v0, v1, v2, v3]  (canonical order)
  --> Loop Exit: [v0, v1, v2, v3]   (canonical order maintained)

Iteration 3:
  --> Entry == Exit -> Converged [x]
'''

**Convergence:** When entry and exit states stop changing (register lists must match exactly, not just sizes).

## Example: Preloop + Loop

### Input Code

'''assembly
entry:
    ds_read_b32 v0, v10
    ds_read_b32 v1, v10
    ds_read_b32 v2, v10
    ds_read_b32 v3, v10
    s_branch loop

loop:
    v_fmac_f32 v4, v0, v2   ; Uses v0, v2
    v_fmac_f32 v4, v1, v3   ; Uses v1, v3
    ds_read_b32 v0, v10
    ds_read_b32 v2, v10
    ds_read_b32 v1, v10
    ds_read_b32 v3, v10
    s_cbranch loop
'''

### Analysis

'''
Block: entry
  Exit State: [v0, v1, v2, v3]

Block: loop
  Entry State: [v0, v1, v2, v3]  <-- From entry

  At v_fmac v4, v0, v2:
    Outstanding: [v0, v1, v2, v3]
    Latest needed: index 2 (v2)
    dlcnt = 4 - 3 = 1 [x]
'''

### Output Code

'''assembly
entry:
    ds_read_b32 v0, v10
    ds_read_b32 v1, v10
    ds_read_b32 v2, v10
    ds_read_b32 v3, v10
    s_branch loop

loop:
    s_waitcnt dlcnt=1       ; <-- Inserted! Wait for v0, v1, v2
    v_fmac_f32 v4, v0, v2
    s_waitcnt dlcnt=0       ; <-- Wait for v1, v3
    v_fmac_f32 v4, v1, v3
    ds_read_b32 v0, v10
    ds_read_b32 v2, v10
    ds_read_b32 v1, v10
    ds_read_b32 v3, v10
    s_cbranch loop
'''

## Configuration

'''cpp
#include "stinkytofu/core/stinkytofu.hpp"

using namespace stinkytofu;

// Create custom policy
WaitCntConfig config = WaitCntConfig::standard();
config.dependencyPolicy.trackLoadDependencies = true;
config.dependencyPolicy.trackStoreDependencies = true;
config.barrierPolicy.waitDSRead = true;

// Create and run pass
auto pass = createStinkyCustomWaitCntPass(config);
pass->run(function, passContext);
'''

## Performance Characteristics

| Aspect | Current Implementation |
|--------|----------------------|
| **Time Complexity** | O(iterations x blocks x instructions) |
| **Space Complexity** | O(blocks x registers) |
| **Iterations** | Typically 2-3 for loops, 1 for acyclic CFG |
| **Precision** | Optimal for loads, conservative for stores |

## Implementation Notes

### Register Reissue Handling

When a load reissues to a register with an outstanding load, the **old load is replaced**:

'''
State: [v0, v1, v2]

After: ds_read_b32 v0, v20  (reissue to v0)
    --> [v1, v2, v0]  (old v0 removed, new v0 added at end)
'''

This matches hardware behavior and prevents count inflation.

### Store Handling (Conservative)

Without address analysis, stores can't be tracked precisely:

'''assembly
ds_write [v0], v2  ; Address in v0
ds_write [v1], v3  ; Does [v0] == [v1]? Unknown!
'''

**Solution:** Stores use scalar counts and "wait all" ('dscnt=0') at barriers.

## Limitations

1. **Store precision**: No address tracking -> Conservative "wait all"
2. **Multi-predecessor merges**: Register lists cleared when paths differ
3. **Compilation time**: Multiple iterations for complex loops

## See Also

- [Developer Guide: Adding New Instructions](../developer-guide/adding-waitcnt-instructions.md)
- [AMD GCN ISA: s_waitcnt Instruction](https://www.amd.com/en/support/tech-docs)
- [Dataflow Analysis](https://en.wikipedia.org/wiki/Data-flow_analysis)
