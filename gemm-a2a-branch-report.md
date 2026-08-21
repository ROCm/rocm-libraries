# `users/yiding12/gemm-a2a`: fused GEMM plus feature-sharded all-to-all via SDMA

## Executive summary

This branch is an end-to-end **TensileLite prototype for fusing a very specific all-to-all into a GEMM epilogue**.  Each GPU computes its own bf16 GEMM, stores the first `AM` entries along its feature (`M`) dimension to its ordinary `D` allocation, and then has the GPU build and submit SDMA packets that copy the appropriate feature shard to each peer.  The remaining features (`[AM, M)`) stay local in `D`.  A final work-group elected over each GPU's own GEMM grid optionally waits until every inbound peer shard is present before that kernel exits.

The important distinction is scope: this is **not yet a general collective API or a shipped hipBLASLt dispatch route**.  The code is opt-in in two places:

- A logic solution must set `FusedGemmA2A=1`; its default is `0`.
- The only launch implementation that appends the required private kernarg tail is the special `tensilelite-client --fused-a2a` path, whose SDMA dependencies are behind `-DTENSILELITE_ENABLE_SDMA_A2A=ON` (default `OFF`).

There is no checked-in logic/configuration with `FusedGemmA2A: 1`, no public hipBLASLt API integration, and no checked-in automated GPU test that renders and runs an enabled fused kernel.  The branch is therefore best read as a working, instrumented proof-of-path for gfx950-class hardware, plus the generator and host contracts needed to develop it further.

The branch is at `c6a0236269c79f0cce80a6d0c0ae0ee9257a44b4` (2026-08-20).  Its last merged `develop` point is `54dcb4f36c6762cb544250f1254b2c29e8b3806f`; from there it contains 12 feature commits and two `origin/develop` merges.  The locally available `origin/develop` ref is eight commits beyond that point, including `0e896704e4` (`feat(hipblaslt): Gemm-From-Anywhere custom kernel framework V1 (#9304)`), so integration/rebase work remains material.  Network DNS prevented refreshing remote refs during this investigation, so this is a precise local-tracking-ref comparison rather than a claim about the live server at report-read time.

All file references below describe the checked-out final state at `c6a0236269`; commit references identify where the design step entered the branch.

### Evidence labels used in this report

- **Direct investigation:** I inspected the checked-out source, branch graph, feature commit messages, build/test wiring, and the tracked-tree search for enabled logic/configuration.  Those observations are described as present/absent in the branch.
- **Author-recorded:** The one hardware result quoted later is copied from the `90036432c526` commit message and is labeled as such.  It was not independently rerun here.
- **Not performed here:** I did not build this branch or run the HSA/KFD/GPU path; in particular, no fresh 4x gfx950 result is claimed by this report.

## What computation is being fused?

The client treats the first free dimension as **feature** (`M`) and the second as **token** (`N`).  For each source rank/GPU `s`, it computes:

```text
D_s[m, n] = GEMM(A[m, k], B_s[k, n])
```

where the harness shares `A` across GPUs and gives each rank its own `B_s`.  It divides only the leading `AM` feature rows across `W` ranks:

```text
n_shard = AM / W

recv_p[s, n, f] = D_s[p * n_shard + f, n]
    for destination rank p, source rank s, 0 <= f < n_shard

local_out_s[m, n] = D_s[m, n]
    for AM <= m < M
```

The staging `D_s` store for the leading `AM` rows is still required: SDMA reads those completed rows from HBM.  The `recv` buffers, not the leading portion of `D`, are the cross-card result interface.  In the client, each destination rank owns a `recv[W, padded_N, n_shard]` buffer; slot `s` holds data that originated on source rank `s`.

This is consequently a **feature-shard all-to-all** specialized to the GEMM output layout.  It is neither a generic all-to-all primitive nor a result-layout conversion exposed through the usual hipBLASLt API.  The layout and intent are stated explicitly in `Tensile/Components/GlobalWriteBatch.py:2560-2605` and validated on the host in `client/src/FusedA2AClient.cpp:121-276`.  The initial skeleton was introduced by `caa38aebe84f`; the final transfer mechanism is SDMA rather than CU-issued remote stores, following `ca2acc5a1670`.

## End-to-end execution flow

```text
host: W GPUs + P2P + one SDMA queue per (source GPU, destination rank)
  │
  ├─ append a fixed fused-A2A kernarg segment to each GEMM launch
  │
  └─ launch the same selected GEMM solution on every GPU before synchronizing
       │
       ├─ remap work-groups so outbound M tiles issue before local M tiles
       ├─ PUSH work-groups: store their D tile with SLC/sc1 so SDMA sees HBM data
       ├─ last WG for each (destination rank, token tile):
       │    build COPY_SUBWIN + flag-atomic packets, reserve/publish one ring entry
       └─ globally last WG: optionally poll all inbound source flags, then exit
             │
             └─ host validates the receive shards, local tail, guard tail, and wptrs
```

The ordering is deliberate:

1. `FusedA2AWgRemap` changes the work-group bijection so the PUSH region—M tiles `[0, AM/MT0)`—is issued as one front segment.  That moves the last outbound tile earlier in dispatch order without changing the grid cardinality or ownership.  See `Tensile/Components/WorkGroupMappingAlgos.py:1116-1203`; this performance change is `099c20ec96fe`.
2. The assembly writer generates the store body twice, once for PUSH and once for LOCAL, under a single runtime gate `WorkGroup0 < FusedAM / MT0`.  The PUSH path forces the SLC/sc1 store property so the SDMA engine reads HBM rather than a stale XCD-local L2 line.  See `Tensile/KernelWriterAssembly.py:16571-16649` and `Tensile/Components/GlobalWriteBatch.py:1681-1686`.  The hoisted dispatch design is `6b822c2da770`.
3. Once a PUSH work-group has completed its stores, all its waves wait and synchronize; wave 0/lane 0 alone increments a counter indexed by `(destination rank, token tile)`.  The work-group that observes the final count for that cell creates the SDMA packet pair.  `Tensile/Components/GlobalWriteBatch.py:2849-2986` is the core handshake.
4. The packet pair is one 13-dword `COPY_SUBWIN` followed immediately by one 8-dword `ATOMIC ADD_RTN_32`.  The copy moves the completed feature band into the target rank's receive slot; the atomic increments that target rank's flag for the source rank.  The two packets share one ring reservation so the flag cannot overtake its own copy.  See `Tensile/Components/GlobalWriteBatch.py:2679-2828` and `Tensile/Components/SdmaPacketEmitter.py:98-184`.
5. Every surviving work-group then contributes to `counter3`; the one that observes the final increment in that *device's GEMM grid* becomes the only DRAIN owner for that kernel launch.  With `FusedDrain=1`, it uses `W` lanes to poll its own `W` flag slots until every one equals `tokenTiles`.  That means all source ranks have completed all token-tile packet pairs into this GPU.  See `Tensile/Components/GlobalWriteBatch.py:3029-3192`.  The single global-within-a-grid owner replaced per-peer drain behavior in `54c962009f4c`.

The host must enqueue every GPU before synchronizing any one of them, because a DRAIN owner may be waiting on data produced by every other GPU.  The client does exactly that at `client/src/FusedA2AClient.cpp:575-603`.

## Design by subsystem

### 1. Solution parameter, admission rules, and code generation

`FusedGemmA2A` is added to the global default parameter list as `0` and accepted as a binary tuning parameter.  Its semantics are documented as “PUSH along the feature dimension” with the remainder stored locally.  Sources: `Tensile/Common/GlobalParameters.py:577-585`, `Tensile/Common/ValidParameters.py:943-947`; introduced in `caa38aebe84f` and snapshot-adjusted by `d16c1114e49`.

When enabled, solution validation rejects:

- non-bf16 destination data;
- any Stream-K mode;
- macro tiles other than 128 or 256 in both dimensions; and
- `GlobalSplitU != 1` (including runtime-resolved `GSU=-1`).

It also disables user-selectable GSU because the grid-wide DRAIN-owner election depends on a fixed work-group count.  Source: `Tensile/SolutionStructs/Solution.py:1805-1823`; the important constraints were tightened in `c0d882d5a0f7` and `54c962009f4c`.

The writer allocates persistent SGPR state for the counter base, `n_shard`, `tokenTiles`, the peer-group offset, and the total work-group count.  The prologue latches these once rather than reloading them in the epilogue.  Sources: `Tensile/KernelWriter.py:9519-9537`, `Tensile/KernelWriterAssembly.py:3390-3405`.  This matters both for code size and for the last-work-group election.

### 2. The private kernarg ABI

An enabled kernel receives a **fixed-size tail segment** after the normal GEMM arguments.  The generator emits eight peer groups regardless of runtime `W`, followed by the counter pointer and four `u32` scalars:

```text
peer[j] = { flagPtr, recvPtr, queueBuf, rptr, wptr, doorbell }  for j = 0..7
counter_ptr
FusedMyRank, FusedW, FusedDrain, FusedAM
```

At eight ranks, this is `8 * 6 * 8 + 8 + 4 * 4 = 408` bytes.  Unused peer groups are null-filled by the host.  The tail is metadata-only: the assembly reads fields on demand from absolute kernarg offsets instead of spending permanent argument SGPRs on every pointer.  Sources: `Tensile/Components/Signature.py:33-104,423-442` and `client/include/FusedA2AKernArg.hpp:22-108`.  The ABI was first added in `caa38aebe84f`, consolidated in `72c6195e3351`, and moved to its final host-known layout in `90036432c526`.

The same contract exists in Python and C++.  The host packer appends all eight groups in the generator's order and throws if the appended segment is not exactly 408 bytes.  That catches a size/alignment mismatch, but not every possible semantic mismatch in field order or offset arithmetic.  This is a conscious cross-language ABI surface and should be treated as such during future edits.

The counter allocation is also a host/kernel contract.  Its current layout—finalized by `c6a0236269c7`—is:

```text
+0      8 cursor pairs      cached and committed 64-bit SDMA wptrs
+128    8 counter2 slots    u32
+160    counter3            u32 global last-WG election
+164    counter1            W * tokenTiles u32 cells
+...    64-byte sentinel
```

The leading regions are sized by `FUSED_A2A_MAX_RANKS`, not runtime `W`, making their offsets compile-time immediates.  The sentinel sits directly after the only runtime-indexed region (`counter1`) and is checked by the client after every iteration.  Sources: `Tensile/Components/Signature.py:64-69`, `client/include/FusedA2ACounterSentinel.hpp:6-90`, and the final relayout rationale in `c6a0236269c7`.

### 3. SDMA packet creation and ring publication

`SdmaPacketEmitter.py` deliberately knows only the packet format.  It accepts already-scaled geometry and builds packet dwords in SGPRs:

- a 13-dword `COPY_SUBWIN` rectangular copy; and
- an 8-dword `ATOMIC ADD_RTN_32` used as arrival notification.

The only hardware-validated element size is 16 bytes, and the packet layout is marked gfx9xx/gfx95x-only; gfx12+ has a different layout.  The packet encoder leaves many range checks to the caller because its bit fields are packed unmasked.  Source: `Tensile/Components/SdmaPacketEmitter.py:1-94,98-184`; introduced in `012205db2c65`.

`GlobalWriteBatchWriter._fusedA2AComputeCopyFields` translates GEMM coordinates into that packet's layout.  It folds token and feature coordinates into 64-bit source and destination base addresses, uses `D`'s actual token stride as source pitch, uses `n_shard` for the receive pitch/width, clamps only the final token-tile height, and converts bf16 element counts to 16-byte packet elements.  This is why the D layout restrictions are semantic correctness conditions, not merely performance preferences.  Source: `Tensile/Components/GlobalWriteBatch.py:2560-2677`.

`SdmaRingEmitter.py` implements a GPU-side producer for a 256 KiB KFD SDMA ring.  It:

1. raises software cursors to at least the hardware write pointer with idempotent `s_atomic_umax_x2` after the per-launch counter memset;
2. reserves enough contiguous ring space with a CAS loop, including tail NOP padding on wrap;
3. writes the copy and atomic packets using scalar stores with `glc`;
4. serializes commits so `wptr`, doorbell, and software committed cursor are published in order.

The cursor repair is especially important: publishing a write pointer behind the hardware pointer can make the engine interpret a huge bogus packet range.  Sources: `Tensile/Components/SdmaRingEmitter.py:17-170,156-269,402-483`; the per-launch cursor repair and pointer-in-kernarg redesign are `90036432c526`.

The branch extends rocisa to render the scalar atomic and memory forms needed by that producer, including `s_atomic_cmpswap_x2` and `s_atomic_umax_x2`.  Sources: `rocisa/rocisa/include/instruction/mem.hpp:3621-3710` and `Tensile/Components/SdmaRingEmitter.py:122-150,238-252`.  The first commit also added cache-scope/atomic support used by the prototype (`caa38aebe84f`).

### 4. Host-side queue, topology, and launch orchestration

The special client path is selected by `--fused-a2a`; it returns before the normal single-GPU benchmark loop.  It exposes `--fused-a2a-world`, `--fused-a2a-am`, `--fused-a2a-drain`, iteration/warmup settings, and a `--fused-a2a-validate` switch.  Defaults are `W=4`, `AM=1024`, drain on, 100 iterations, 10 warmups, and numeric validation on.  Source: `client/main.cpp:84-90,295-301,1136-1144`; harness introduction `e5cb0c25023a`.

The harness resolves a normal Tensile solution, then appends this private tail, but it does not visibly assert that the selected solution was generated with `FusedGemmA2A=1`.  Its intended use therefore requires a matching externally generated logic/code object; a normal solution will not consume the appended metadata.  This is another reason the path is demonstrably a prototype harness rather than a general user-facing launch contract.  Source: `client/src/FusedA2AClient.cpp:90-120,520-572`.

For each GPU, the client:

- checks pairwise HIP P2P support and enables it;
- allocates a fine-grained peer allocation containing its flag area and receive buffer, plus a separate local counter allocation;
- creates one `SdmaQueue` for each destination rank, including a loopback/self queue;
- creates a per-device `SolutionAdapter` and stream;
- constructs the normal GEMM invocation once and appends the 408-byte fused segment.

The queue wrapper uses HSA/KFD topology IDs, maps HIP ordinal to HSA agent by PCI BDF, selects a recommended SDMA engine for each IO link where available, and creates a high-priority KFD SDMA queue backed by uncached 256 KiB ring memory.  Sources: `client/src/FusedA2AClient.cpp:336-445,520-573`; `client/include/SdmaQueue.hpp:35-299`.

The branch does not create a process-distributed communicator or a multi-node route.  Its own latency output calls it a “single-process … on ONE node” datum rather than multi-node xGMI evidence.  Source: `client/src/FusedA2AClient.cpp:861-866`.

### 5. Build and package impact

The ordinary client build remains free of HSA/KFD/NUMA dependencies unless `TENSILELITE_ENABLE_SDMA_A2A=ON`.  When enabled on non-Windows hosts, CMake finds `hsa-runtime64`, `hsakmt`, and `libnuma`, links those libraries privately into `tensilelite-client-common`, and defines `TENSILELITE_ENABLE_SDMA_A2A`.  Without that macro, invoking `--fused-a2a` fails fast rather than launching with null queue handles.  Source: `client/CMakeLists.txt:24-113` and `client/src/FusedA2AClient.cpp:959-967`.

`cmake/HsakmtLinkInterface.cmake` patches known ROCm 7.x hsakmt export problems: an invalid absolute libc path and dead build-host `-L` directories for sysdeps.  The workaround is pragmatic but environment-sensitive, and the branch itself notes no dedicated test for the helper.  Source: `cmake/HsakmtLinkInterface.cmake:4-62`; implementation added with `012205db2c65` and documented/trimmed in `c0d882d5a0f7`.

## Explicit constraints and their consequences

| Constraint | Where enforced | Why it matters |
| --- | --- | --- |
| `1 <= W <= 8` | Host check and Python/C++ ABI constants: `client/src/FusedA2AClient.cpp:50-63`, `Signature.py:43-60` | Eight fixed peer groups are compiled into the kernarg segment.  The DRAIN mask is deliberately bounded below the wave32 `s_bfm` wrap hazard. |
| bf16 `D` only | Solution validation: `Solution.py:1805-1810` | Copy-field arithmetic assumes two-byte elements and the 16-byte packet element conversion. |
| `StreamK=0`, `GlobalSplitU=1`, MT0/MT1 in `{128,256}` | `Solution.py:1809-1823` | The per-tile election and global owner count rely on an ordinary data-parallel grid with a known work-group population. |
| One plain, non-batched GEMM | Client checks: `FusedA2AClient.cpp:76-131` | The special harness expects one GEMM kernel and its grid/counting model. |
| `AM <= M`, `AM % W == 0`, `n_shard % MT0 == 0`, tile alignment | `FusedA2AClient.cpp:133-160` | Every output M tile must belong to exactly one destination rank; otherwise the DRAIN protocol can wait for a tile that no work-group can produce. |
| Feature-contiguous D (`dMStride == 1`) and packet-addressable token stride | `FusedA2AClient.cpp:240-276` | The SDMA rectangle is contiguous in feature dimension and carries source pitch in a finite packet field. |
| 16-byte SDMA element, rect fields < 14 bits | `FusedA2AClient.cpp:162-196`, `SdmaPacketEmitter.py:50-53` | Over-range fields are OR-packed into neighboring packet bits; they are not safely truncated. |
| P2P-capable, one-node HIP/HSA/KFD system | `FusedA2AClient.cpp:382-420`, `SdmaQueue.hpp:74-210` | This is direct peer SDMA using KFD queue resources, not a portable communication backend. |
| SDMA feature build enabled | `client/CMakeLists.txt:24-113` | It is default-off and unsupported on Windows by this wiring. |
| `FusedDrain=1` for “results are ready when kernel exits” semantics | DRAIN code: `GlobalWriteBatch.py:3086-3172`; client default: `main.cpp:297` | With drain off, the kernel does not wait for inbound packet completion; it is a different completion contract. |

There is one notable boundary mismatch to resolve before generic use: the **client** rejects batched GEMM, but the shown `FusedGemmA2A` solution-level rejection block does not.  The global tally is explicitly `NumWorkGroups0 * NumWorkGroups1` (`KernelWriterAssembly.py:3396-3404`), which does not obviously include a batch/work-group-2 dimension.  This is safe on the harness's restricted path, but an enabled logic solution must not be exposed through a generic batched launcher until that accounting is proven or constrained at generation time.

## Branch narrative and commit sequence

| Commit | Contribution to the final design |
| --- | --- |
| `caa38aebe84f` | Adds the `FusedGemmA2A` skeleton, solution flag, basic kernarg metadata, store-path split, initial counters/DRAIN, and rocisa support. |
| `e5cb0c25023a` | Adds the single-process four-GPU validation harness and fixes several store-path permutations it exercises. |
| `6b822c2da770` | Hoists the PUSH/local gate out of the store loop and regenerates the two modes cleanly. |
| `012205db2c65` | Adds packet and ring emitters, KFD queue ownership, and build wiring for GPU-initiated SDMA. |
| `ca2acc5a1670` | Replaces CU-driven remote stores with SDMA copies from the natural local-D layout; adds overflow guards and counter sentinel support. |
| `54c962009f4c` | Replaces per-peer DRAIN owners with one globally last work-group and formalizes rank-mask bounds. |
| `099c20ec96fe` | Front-loads outgoing feature tiles with a bijective work-group remap to improve overlap. |
| `72c6195e3351` | Consolidates peer pointers into grouped kernarg data and improves runtime reporting/setup. |
| `c0d882d5a0f7` | Trims obsolete surfaces, derives duplicated metadata, and turns bf16 assumptions into solution rejection. |
| `d16c1114e49` | Re-records characterization snapshots caused by adding the solution parameter/default/naming abbreviation. |
| `90036432c526` | Moves queue handles and SDMA cursors into the fixed kernarg/counter contracts; adds lazy cursor repair and wptr monotonicity checking. |
| `c6a0236269c7` | Moves cursor pairs to the front of the counter block for constant offsets and a better-positioned overflow guard. |

The two merge commits in the feature range (`91329656fa2a` and `467f1ef6296a`) only merge contemporaneous `origin/develop`; they are not part of the intended fused-A2A design itself.

## Validation: what is present, what was reported, and what is absent

### In-tree mechanisms

The client has unusually strong runtime observability for a prototype:

- It creates an independent host bf16 GEMM golden for every source rank.
- It checks every received `[destination, source, token, feature-within-shard]` value and every local-tail value on every iteration.
- It re-zeroes counters, flags, and receive buffers each iteration so stale correct data cannot hide a scatter race.
- It checks the counter sentinel tail and verifies that every SDMA queue write pointer is monotonic and dword-aligned.
- It records per-GPU timing, max-across-GPU p50/p90, and a slowest-card distribution.

Sources: `client/src/FusedA2AClient.cpp:305-334,447-518,575-705,707-958`.

The normal TensileLite unit coverage was updated only at the registration/snapshot level.  The snapshots show the new default value (`FusedGemmA2A: 0`), and `gpu_test_helpers.py` also defaults it to zero.  A tracked-file search finds no enabled `FusedGemmA2A: 1` logic/configuration and no dedicated packet/ring unit test.  Sources: `Tensile/Tests/unit/gpu_test_helpers.py:192`; snapshot updates in `d16c1114e49`.

As a lightweight final hygiene check for this investigation, `git diff --check 54dcb4f36c..HEAD -- projects/hipblaslt/tensilelite` completed cleanly.  That checks patch whitespace only; it is not a build, code-generation, or GPU validation result.

### Author-recorded hardware evidence

The commit message for `90036432c526` records this hardware result:

> 4x gfx950, MED/N2000/N2047/N2048, 50 iterations each; 8/8 PASS and race 50/50 per shape with numeric host-golden comparison.  It also records an SDMA-off default client build and a 6,210-test / 747-snapshot unit-suite pass.

That is useful behavioral evidence for the path **through `90036432c526`**, but it is not a substitute for current verification.  The final cursor-layout change `c6a0236269c7` follows it and its commit message does not include a new hardware-run record.  I did not run a 4x gfx950 SDMA experiment in this inspection; this report distinguishes the recorded author result from a fresh result.

### What a merge/readiness conversation should require

The highest-value missing evidence is not another default-parameter snapshot.  It is a reproducible enabled-kernel path that:

1. builds with SDMA both off and on;
2. generates/selects a real `FusedGemmA2A=1` solution;
3. runs the four-GPU numeric harness after `c6a0236269c7` across partial and full token tiles; and
4. exercises enough repeated traffic to cross a 256 KiB ring boundary, including the room-refresh and NOP-padding paths.

The last item matters because the ring code itself notes that a short validation run may not exercise the “ring full” refresh path (`SdmaRingEmitter.py:221-235`), and the current checked-in unit suite does not do so.

## Risks and discussion questions

1. **Prototype versus product seam.**  Who owns the user-facing contract?  Today the only producer of the required 408-byte tail is `FusedA2AClient.cpp`, and no enabled logic is checked in.  A normal solution launcher that selects an enabled kernel without this specialized host packing would supply incompatible/missing metadata.  Decide whether this stays a client experiment, becomes a Tensile runtime ABI, or is exposed through a higher-level hipBLASLt/consumer API.

2. **Generic-use correctness gap.**  The host bans batch mode and expects exactly one kernel, while the generator-level validation shown does not make those restrictions universal.  The `counter3` owner count is 2-D.  This needs either a proof that every enabled use has that grid shape or explicit solution-side rejections for unsupported problem forms.

3. **No automated enabled path.**  The checked-in tests exercise defaults and snapshots, not the generated SDMA instructions or cross-card dataflow.  The author-recorded hardware run is valuable but predates the final counter relayout.  This is the most direct validation gap.

4. **Platform and dependency narrowness.**  Packet encoding is gfx9xx/gfx95x-specific; the queue path needs KFD/HSA, P2P, and NUMA/hsakmt packaging that is patched for ROCm 7.x quirks.  There is no Windows or multi-node implementation.  The scope needs to be explicitly documented as a supported hardware matrix rather than inferred from source comments.

5. **Architecture admission remains implicit.**  `SdmaPacketEmitter.py` explicitly excludes gfx12+ packet layout, but the `FusedGemmA2A` solution-validation block does not visibly reject incompatible ISAs.  Today the actual architecture gate appears to be the generated solution/library selected for the harness.  A general route should make that constraint explicit at selection or generation time.

6. **Liveness failure mode.**  DRAIN and ring-space spins have no user-visible timeout inside the kernel.  A topology, packet, cache-scope, or counter bug is likely to become a hang/watchdog event rather than a clean error.  The harness's fail-fast geometry checks and host-side diagnostics reduce setup errors but cannot turn a GPU-side liveness fault into a bounded failure.

7. **Cross-language ABI drift.**  The rank maximum, peer-field order, counter offsets, and argument order are mirrored across Python and C++.  There is a runtime total-size check, but no generated/shared schema or compile-time cross-language offset test.  Any extension should make that contract mechanically verifiable.

8. **`counter2` intent should be clarified.**  In the final code, `counter2` is incremented and compared against `tokenTiles`, but both comparison paths immediately converge at `skipReleaseLabel` before the global `counter3` tally (`GlobalWriteBatch.py:2988-3025`).  That may be intentionally retained accounting or it may be dead synchronization work left after the single-global-owner redesign.  It is worth resolving explicitly before treating the protocol as settled.

9. **Upstream integration risk.**  The branch's last merged upstream point is `54dcb4f36c67`; the locally available `origin/develop` is eight commits ahead.  One of those is the hipBLASLt Gemm-From-Anywhere framework (`0e896704e4`), a likely conceptual overlap around custom-kernel launch/dispatch.  A rebase should be treated as a design integration, not just a mechanical conflict resolution.

## Bottom-line assessment

The branch successfully assembles all layers needed to demonstrate its narrow idea: generator gating, an explicit private ABI, GPU-side SDMA packet/ring production, a DRAIN completion protocol, KFD queue management, and a rigorous numeric/race harness.  The most sophisticated portions are the memory-ordering and ownership protocol: SLC stores make D visible to SDMA, one work-group owns each `(destination, token tile)` submit, one reservation keeps copy and flag ordered, lazy cursor repair survives per-launch memset, and one final work-group turns distributed arrival counters into a kernel-exit condition.

What it does **not** yet establish is production adoption: no enabled solution is shipped, no normal API can launch it, no automated enabled regression exists, and the final post-validation refactor has no fresh hardware evidence in the commit record.  Those are not incidental polish items; they define the next decision about whether this remains an experimental fused-A2A vehicle or becomes a supported hipBLASLt capability.
