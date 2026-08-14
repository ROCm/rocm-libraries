# Kernel Selection in hipBLASLt / TensileLite

How a runtime matmul call is routed to a compiled kernel — from the hipBLASLt API
down to `hipModuleGetFunction`.

## Overview

Selection has two stages:

1. **Predicate filtering** — hard-eliminates kernels whose epilogue flags, hardware
   requirements, or tile constraints do not match the runtime problem.
2. **Size matching** — among the survivors, a KD-tree nearest-neighbour search on
   (M, N, K) picks the kernel that was benchmarked closest to the runtime shape.

---

## Dispatch chain

### 1. hipBLASLt host entry — `tensile_host.cpp:4329`

`hipblasLtMatmulAlgoGetHeuristic` → `rocblaslt_matmul_algo_get_heuristic` →
`getBestSolutions()` → `getSolutions()` →
`library->findTopSolutions(problem, hardware, N)`

`getBestRawSolutions()` (`tensile_host.cpp:4290`) is a separate helper called
from the algo-inspection path in `rocblaslt_auxiliary.cpp`, not from the
heuristic selection entry point. The actual `hipblasLtMatmul` → `rocblaslt_matmul`
call uses an algo that was already selected by the heuristic step.

The runtime `ContractionProblemGemm` carries epilogue flags derived from the
matmul descriptor: `usePartialRMS()`, `partialRMSResidualAdd()`.

### 2. Library deserialisation

The device library (`.dat.zlib`) is a zlib-compressed msgpack blob. On first use
`MasterSolutionLibrary` (line 151 of `MasterSolutionLibrary.hpp`) decompresses it
via `readCompressedMsgObject` (`MessagePack.cpp:102`) and inflates the tree into
memory. Lazy-load libraries also have a mapping shard
(`TensileLiteLibrary_lazy_<arch>_Mapping.dat`) that records which `.co` file
holds each solution index, so unused code objects are never loaded.
`fileToMsgObject()` probes the `.zlib` compressed variant of the shard
automatically, so callers always pass the bare `.dat` name.

### 3. Library tree walk — `MasterSolutionLibrary.hpp:366`

The in-memory structure is a recursive tree:

```
MasterSolutionLibrary
└── MatchingLibrary            ← KD-tree indexed on (M, N, K, ...)
    └── entries[]
        └── SingleSolutionLibrary    ← leaf node, one kernel
            ├── hardwarePredicate    ← GPU model / VRAM
            ├── problemPredicate     ← epilogue flags, data types, ...
            └── taskPredicate        ← tile-size alignment, GSU constraints, ...
```

`findTopSolutions` (`MatchingLibrary.hpp:145`) walks the tree and collects
candidates, calling `findBestSolution` on each leaf.

### 4. Predicate evaluation — `SingleSolutionLibrary.hpp:107`

For each leaf, three predicate chains are evaluated in order. All three must pass
or the candidate is discarded:

| Chain | Examples |
|---|---|
| `hardwarePredicate` | `IsaEqual`, VRAM threshold |
| `problemPredicate` | `UsePartialRMSEqual`, `UsePartialRMSResidualAddEqual`, `UsePartialRMSQuantEqual`, `DataTypeEqual`, `TransposeEqual` |
| `taskPredicate` | tile-size divisibility, `GlobalSplitU` validity |

The PartialRMS predicates are defined in `ContractionProblemPredicates.hpp`:

| Predicate | Line | Checks |
|---|---|---|
| `UsePartialRMSEqual` | 2089 | `problem.usePartialRMS() == value` |
| `UsePartialRMSResidualAddEqual` | 2157 | `problem.partialRMSResidualAdd() == value` |
| `UsePartialRMSQuantEqual` | 2191 | `problem.partialRMSQuant() == value` |

A plain GEMM problem (all flags false) is routed only to the plain kernel; a
PartialRMS + residual problem (usePartialRMS=true, partialRMSResidualAdd=true)
matches only the v3 kernel. There is no ambiguity between variants.

### 5. Size matching — `MatchingLibrary.hpp:85`

After predicate filtering, the `MatchingTable` (`PropertyMatching.hpp:81`) runs a
KD-tree nearest-neighbour search on the benchmarked (M, N, K) size points to find
the candidate whose tuned shape is closest to the runtime problem. This is the
output of the tuning runs: a mapping from size → best kernel for that size.

`findBestMatch` (`PropertyMatching.hpp:94`) returns the single closest entry;
`findTopMatch` returns the top-N ranked by distance for the heuristic fallback.

### 6. Code object loading — `HipSolutionAdapter.cpp:100`

The winning `ContractionSolution` carries two strings:

- `kernelName` — the HIP kernel function name
- `codeObjectFilename` — the `.co` file that contains it

`loadCodeObjectFile` calls `hipModuleLoad` on the `.co` once and caches the
`hipModule_t`. `getKernel` (`HipSolutionAdapter.cpp:329`) then calls
`hipModuleGetFunction` to extract the function handle, which is also cached in
`m_kernels[kernelName]` for subsequent calls.

---

## Key file index

| Component | File | Lines | Function |
|---|---|---|---|
| Heuristic entry | `tensile_host.cpp` | 4329–4399 | `getBestSolutions` |
| Raw solutions (aux path) | `tensile_host.cpp` | 4290–4327 | `getBestRawSolutions` |
| Solution search | `tensile_host.cpp` | 4277–4288 | `getSolutions` |
| Master dispatch | `MasterSolutionLibrary.hpp` | 366–388 | `findTopSolutions` |
| Lazy shard load | `MasterSolutionLibrary.hpp` | 151–213 | `loadLibrary` |
| Size matching | `MatchingLibrary.hpp` | 145–171 | `findTopSolutions` |
| KD-tree search | `PropertyMatching.hpp` | 81–110 | `MatchingTable` |
| Predicate eval | `SingleSolutionLibrary.hpp` | 89–123 | `findBestSolution` |
| `UsePartialRMSEqual` | `ContractionProblemPredicates.hpp` | 2089 | `operator()` |
| `UsePartialRMSResidualAddEqual` | `ContractionProblemPredicates.hpp` | 2157 | `operator()` |
| `UsePartialRMSQuantEqual` | `ContractionProblemPredicates.hpp` | 2191 | `operator()` |
| Msgpack inflate | `MessagePack.cpp` | 102–191 | `readCompressedMsgObject` |
| Code object load | `HipSolutionAdapter.cpp` | 100–196 | `loadCodeObjectFile` |
| Kernel extraction | `HipSolutionAdapter.cpp` | 329–374 | `getKernel` |
