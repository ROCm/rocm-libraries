# DESIGN-CCL: rocKE multi-GPU foundation (IPC peer mailboxes + in-kernel collectives), gfx950

Status: design, revision 1 (W3, 2026-10-30, user decision 4 of `PLAN.md`: "Multi-GPU is in scope: rocKE gets IPC peer
mailboxes and an in-kernel CCL (all-reduce / all-gather) so TP2 (and up) works"). Companion of `DESIGN-LB.md` (the
large-batch true mono that is the first consumer) and of `DESIGN.md` rev3 (the small-batch true mono, which uses the
same pieces at rows 1-16). No rocKE or ATOM source is edited by this document. Ratios, shares and byte counts only;
absolute times live in `hackathon-prep/private-perf/qwen38-true-mono/` (subdir `large-batch/`).

**Revision 1.1 (W6 synthesis, 2026-11-01): section 10 records what the measurements (`ccl/BUILD.md`,
`large-batch/CCL-LM.md`, `large-batch/WP-C5.md`, with their Verification and Fix rounds) changed: the TP2 defaults
(lamport one-shot in situ, not lines), the WP-C5 gate (not met; the per-exchange 0.25 target moves to DESIGN-LB WP-L5
as an overlap target), the MoE-down exchange, TP4, and two correctness rules. Where section 10 and an earlier section
disagree, section 10 wins.**

Abbreviations (as `DESIGN.md`): `R` = `copy-rocm-lib/dnn-providers/hip-kernel-provider/rocke`, `P` =
`R/platform/python/rocke`, `L` = `R/library`, `Q` = `L/kernels/gfx950/qwen38_mono`, `F` =
`dev/flytape-main/flytape-main/flytape`, `FD` = `dev/flytape-main/flytape-main/docs`, `AT` = `dev/ATOM-truemono/atom`
(read-only for W3). "Rank" = one process driving one GPU. "Use" = one numbered instance of one exchange.

---

## 0. Summary

1. **Push model, one hop, no extra phase.** A producer stage pushes its partial straight from its epilogue into every
   rank's receive buffer (its own included) with system-scope stores through **64-bit global addresses**; the consumer
   stage polls only its own (local) memory and adds the ranks' partials **in rank order in f32**, so every rank gets
   the same bits. The exchange is the cross-device sync: no barrier, no grid phase, no host step per layer (FlyTape's
   "adds no phases", `FD/multi-gpu.md`, made dataflow-granular: the consumer polls only the lines its own unit needs).
2. **Three wire forms, one interface** (`Q/ccl/`): tagged pairs (8 B = 32-bit value + 32-bit tag; small payloads,
   headers, argmax/LSE merges, rows <= 4), 1 KB lines + flag (bulk all-reduce/all-gather, FlyTape's tiled-tape form),
   and a Lamport sentinel form (payload only, three-slot rotation, aiter's one-shot) as a knob.
3. **Collectives:** one-shot all-reduce (default up to a measured size threshold), two-shot all-reduce (reduce-scatter
   + all-gather on the same buffers; default above the threshold at TP >= 4), all-gather (vocab-parallel argmax, top-k
   merge, logsumexp merge, full logits for `logits=out`), reduce-scatter (two-shot, `hc=sp`). Every reduction is a
   fixed-order tree; no float atomics anywhere.
4. **No per-launch reset.** Tags continue across launches from a device-resident epoch word that the kernel itself
   advances (graph-replay safe, no host memset or barrier per step). The host resets only at group creation and
   after an error. A launch header exchange (epoch, rows, schedule hash) turns a desynchronised launch into an error
   code instead of a hang or wrong data.
5. **Failure is an error word, never a hang:** every poll has an `s_memrealtime` deadline; the first error wins in a
   host-mapped control block (code, stage, layer, peer, expected/seen tag, CTA); a timed-out rank pushes ABORT into
   every peer's control block so peers stop within one poll round instead of their own deadline.
6. **Host runtime** `P/runtime/peer.py` (host-only ctypes over `libamdhip64`, no IR, so every existing build stays
   byte-identical): uncached / fine-grained / coarse allocation, IPC export/open, peer access, peer-base kernargs,
   control block, epoch, rendezvous through `torch.distributed` gloo or a file rendezvous, one process per GPU (plus a
   single-process multi-device mode for microbenchmarks).
7. **Targets to beat:** FlyTape's exchange (`FD/multi-gpu.md` "Measured": the reduce phase including its grid sync is
   about 0.1-0.3 of a toy layer at TP1-TP4; the lines exchange plus its waits were about 0.12 of the TP2 B128 step),
   RCCL `all_reduce` through `torch.distributed` (graph-captured), and aiter's custom all-reduce (what ATOM's TP path
   calls, `AT/model_ops/communication_op.py`), at decode sizes 1-128 rows x 2560 / 10240 x bf16 / f32, TP 2 / 4 / 8.
   Win rule per size: per-collective latency in a 96-collective chain below every arm's, AND exposed cost inside a
   GEMM + all-reduce pipeline at or below 0.25 of FlyTape's per-layer exchange share.
8. **Measured (section 10):** standalone, the CCL-LM selection beats aiter's graph from 80 KiB to 2.5 MiB per rank
   (0.71-0.98) and ties at 5 MiB (link-bound). In situ at TP2 rows 5-128 the default is lamport one-shot with a
   consumer pass (0.83-0.93 of GEMM + aiter, 0.62-0.73 of GEMM + RCCL); the 96 exchanges of a step cost 0.048-0.061
   of FlyTape's TP2 step, below FlyTape's own 0.060-0.082. The 0.25 per-exchange target is not met (0.49-0.77 at
   B128: the link time alone is 0.65-0.87 of FlyTape's per-exchange cost); it needs overlap with the next layer.

Work packages (section 8): WP-C0 topology and platform facts | WP-C1 host runtime | WP-C2 device emitters | WP-C3
tests | WP-C4 microbenchmarks vs RCCL / aiter / FlyTape | WP-C5 in-situ GEMM + all-reduce | WP-C6 memory-type,
cache-policy and homing levers | WP-C7 model integration hooks (consumed by `DESIGN-LB.md` WP-L5 and `DESIGN.md`).

---

## 1. Requirements and non-goals

Requirements:
- TP2 for the Qwen3.8 true mono at every batch (rows 1-128), TP4 / TP8 for the collectives (model TP4 needs KV-head
  replication, `DESIGN-LB.md` 7.3).
- Inside a persistent rocKE kernel (one CTA per CU, all resident) and inside `layers=1` graph-launched kernels; the
  same emitters, the same wire format.
- Bitwise agreement of every rank's reduced values; a CPU reference reproduces them bit for bit (rank-ordered f32 sum
  of the exact pushed values).
- Deterministic, deadlock-free, timeout-bounded; negative controls that must fail do fail.
- Works under ATOM's process model (one process per TP rank, its gloo CPU group, HIP graph capture of the decode step).
- Python-only (user rule: no C++ engine work); existing builds byte-identical.

Non-goals: prefill collectives (ATOM's), inter-node, expert-parallel all-to-all (not needed for the chosen sharding;
section 4.5 notes when it would be), replacing RCCL outside the decode kernels.

---

## 2. Hardware facts and assumptions (each one verified by a WP before it is relied on)

| # | fact / assumption | status | verified by |
|---|---|---|---|
| H1 | MI355X 8-GPU UBB node: every GPU pair is joined by a direct xGMI link (fully connected mesh, 7 links per GPU), so one-shot push is one hop for every pair at TP <= 8 | assumption (platform spec) | WP-C0: `amd-smi topology` (link type, hops, weight) and `rocm-smi --showtopo` on both rocke nodes; per-pair push RTT matrix |
| H2 | Per-pair bandwidth is one link's (lambda); a one-shot exchange of K bytes costs about K / lambda on each link in parallel, independent of TP; two-shot costs 2K / (N lambda) per link plus a second hop | assumption | WP-C4 bandwidth sweep per pair, 1 vs N-1 concurrent peers |
| H3 | An aligned 8-byte `global_store_dwordx2 sc0 sc1` to peer memory is single-copy atomic for a reader doing an 8-byte system-scope load (the tag-in-pair premise; FlyTape, aiter and the vLLM mono rely on it; the memory model does not promise it) | hardware behaviour, untested by us | WP-C3 rank-agreement + torn-pair detector (value carries a hash of its tag in the stress test) |
| H4 | Uncached (`hipDeviceMallocUncached`) memory written over xGMI and read with system-scope loads is coherent without fences | FlyTape-tested | WP-C3 positive and negative controls (plain loads must be able to read stale data in the negative arm, or the arm is reported as "not discriminating") |
| H5 | The receiving GPU homes each address on one of its IODs (bit 13 XOR a per-2-MiB-page constant, the NUMA lever's model); a remote push lands through whichever IOD owns the target address; consumers poll faster from a copy homed on their own IOD (1.48-1.50x round trip remote vs local, `research/q38-learnings.md` 1.1) | measured on device-local coarse memory only | WP-C6: the ping-pong calibration repeated on uncached and fine-grained allocations, 4 KiB vs 2 MiB pages |
| H6 | `!amdgpu.no.remote.memory` on the f32 `atomicrmw fadd` that `P/core/lower_llvm.py` always emits (lines ~3422-3434, 3514, 6650) is a promise the compiler acts on (native `global_atomic_add_f32` instead of a CAS loop); on peer memory it is false | code fact | section 4.8: never used on peer memory; a lint enforces it |
| H7 | Buffer resources (V#) on a peer address: `R/platform/dsl_docs/runtime/limitations.md` "Multi-GPU and RCCL" reports peer pointers in a kernel arg's buffer descriptor read as out of range and clamp to zero; FlyTape uses buffer stores on peer bases successfully | conflicting | not needed: peer access uses 64-bit global addressing only (4.1); WP-C3 keeps one diagnostic test that records which is true and corrects `limitations.md` |
| H8 | `s_memrealtime` runs at a constant 100 MHz | FlyTape-documented | reused from the residency ticket |
| H9 | A persistent kernel that shares its GPU with another process can hang or time out (FlyTape pitfall 1) | known | every multi-GPU job checks `squeue -p rocke` and `amd-smi process` on its GPUs before launching and refuses to time on a shared GPU |

Every new fact (H1, H2, H5, the topology per node, which GPU ordinals pair with which bus ids, idle-GPU rules) is
written into the alola-cluster skill the day it is measured.

---

## 3. Host runtime: `P/runtime/peer.py`

Host-only (ctypes; it emits no IR and no existing module imports it), so the cpp-parity / byte-identity invariant is
untouched. It follows `P/runtime/hip_module.py`'s lazy binding (`_b(...)` over the same resolved `libamdhip64`
handle, so torch and rocKE share one HIP runtime, `runtime_coexistence.py`).

### 3.1 Process model and bootstrap

- **One process per GPU** (`mode="ipc"`, default): the ATOM model, FlyTape and RCCL all run ranks this way. Each rank
  constructs `PeerGroup(rank, ranks, nbytes, rendezvous=...)` before any rank launches.
- **Rendezvous:** `rendezvous="gloo"` (an existing CPU `torch.distributed` group: ATOM's TP group's CPU twin, or one
  created from env `MASTER_ADDR/PORT`), handles exchanged by `all_gather_object`, barriers by `dist.barrier`; or
  `rendezvous="file:<dir>"` (torch-free: each rank writes `<dir>/rank<r>.handle` atomically (write + rename), polls
  for the others, barrier = a generation counter file per rank). The file mode lets the pure-rocKE tests and the
  venv microbenchmarks run without a torch process group.
- **Single process, several devices** (`mode="p2p"`, microbenchmarks only): one `KernelLauncher` per device under
  `torch.cuda.device(d)` (works per `limitations.md`), `hipDeviceCanAccessPeer` checked, `hipDeviceEnablePeerAccess`
  for every ordered pair, plain device pointers instead of IPC. It removes process skew from latency measurements and
  is how the RTT matrix of WP-C0 is taken.

### 3.2 Allocation, IPC, peer access

- One symmetric arena per rank (`nbytes` rounded up to 2 MiB, default 64 MiB), holding every exchange buffer, the
  per-rank control block and the epoch word. Allocation by `mem=`:
  - `uncached` (default): `hipExtMallocWithFlags(&p, n, hipDeviceMallocUncached)`, the FlyTape / aiter / vLLM choice;
  - `fine`: `hipDeviceMallocFinegrained` (cached in the remote-coherent MTYPE);
  - `coarse`: `hipMalloc` (requires the system-scope cache bits on every access; negative-control and lever arm).
  WP-C6 times all three (FlyTape open question "cached versus uncached").
- Export `hipIpcGetMemHandle` on the arena base; open the peers' with `hipIpcOpenMemHandle(..,
  hipIpcMemLazyEnablePeerAccess)`; keep the mapped bases; `hipIpcCloseMemHandle` on close (collective, barrier-
  bracketed, as `F/peer.py`). The arena is 2 MiB aligned so the NUMA page calibration (H5) applies page by page.
- **Peer bases are kernel arguments**, not a table read inside loops: `peer0..peer{N-1}` (i64) + `rank` (i32) +
  `ctrl` (i64), loaded once at kernel entry into SGPRs (kernarg preload where the build uses it). FlyTape's lesson: a
  table load between two pushes made every push wait for the previous store (`FD/multi-gpu.md` "Deviations"). Under
  graph capture these kernargs are constants of the group, so capture and replay are safe.

### 3.3 Control block, error word, epoch

Per rank, two pieces:
- **`ctrl_dev`** (in the arena, written by the kernel and by peers): `epoch` (u32, the tag base of the next launch),
  `abort` (u32, written by any peer that aborts, system scope), `launch_count`, the launch-header slot (pairs).
- **`ctrl_host`** (`hipHostMalloc(mapped | coherent)`, 64 B, read by the host after each launch and by the watchdog):
  `status`, `code`, `stage`, `layer`, `peer`, `tag_expected`, `tag_seen`, `cta`, `realtime`. First error wins
  (`atomic_cmpswap`-free: a system-scope `atomic_add` on a claim word, the winner writes the record).

Error codes (shared with FlyTape's numbering where it overlaps): -1 RESIDENCY_TIMEOUT, -2 GRID_SYNC_TIMEOUT,
-3 PEER_TIMEOUT, -4 QSA_TIMEOUT, -5 CCL_MISMATCH (launch header differs), -6 PEER_ABORT (a peer aborted), -7
TAG_AHEAD (a pair carries a tag ahead of the expected one by more than the slot window: a WAR violation or a
desynchronised schedule).

### 3.4 Reset protocol

- **Group creation and recovery only:** `reset()` = synchronize, barrier, memset arena, write `epoch = 1`,
  synchronize, barrier (FlyTape's order: the first barrier waits out a peer that may still push into this arena).
- **Between launches: nothing.** Tags are `epoch + use` (4.3). Each launch reads `epoch` at entry (written by the
  previous launch's last CTA, coherent across the kernel boundary) and the last CTA to finish (exit counter, agent
  scope) writes `epoch + uses_this_launch`. Every rank runs the same launch sequence (ATOM keeps TP ranks in
  lockstep; the header check catches a divergence), so epochs agree without host traffic. A HIP graph that captures
  the step therefore replays correctly.
- **Why no WAR barrier is needed across launches** (the cross-launch case of 4.9's rule): a rank R's first push of
  launch L+1 into peer P is data-dependent on R having polled P's last push of launch L, which P issued after all its
  readers of the slot being reused were done; the kernel boundary on R orders the rest. WP-C3's "consecutive launches
  without reset, ranks skewed by random host sleeps" test covers it.
- **After an error:** the host sees `code != 0` after the launch, every rank calls `reset()` (collective; the
  watchdog of 3.5 forces it if one rank is stuck), the epoch restarts at 1, and the launch is retried or the request
  failed by the caller.

### 3.5 API sketch

```python
class PeerGroup:                       # P/runtime/peer.py, host only
    def __init__(self, rank, ranks, nbytes=64 << 20, *, mode="ipc", mem="uncached",
                 rendezvous="gloo", group=None, device=None, timeout_s=10.0): ...
    bases: tuple[int, ...]             # every rank's arena as mapped in this process
    ctrl_host: int                     # mapped error record
    def kernargs(self) -> dict          # {"peer0".., "rank", "ctrl"} for the launcher
    def reset(self) -> None             # collective; creation / recovery only
    def check(self) -> None             # raise PeerError(record) if the kernel recorded an error
    def barrier(self) -> None
    def close(self) -> None             # collective
def topology(device) -> dict           # hipDeviceCanAccessPeer matrix, link type / hops when amd-smi is present
```
A watchdog thread (optional) polls `ctrl_host` during long launches and reports the first error early.

---

## 4. Device side: `Q/ccl/`

Library family (like `mailbox_ir`), Python engine for anything that needs a Python-only opcode (`int_to_ptr`,
`tagged_poll`, `buffer_*_cp`); the mirrored asm path of `rocke.helpers.mailbox` (`MailboxKnobs(scope="system")`,
`global_store_dwordx{2,4} ... sc0 sc1`, asm poll loop) is reused unchanged where it suffices. Nothing is added to
`rocke.helpers` (`platform/AGENTS.md`: no helper without a C++ port path).

### 4.1 Addressing

- Peer accesses are `global_*` with an SGPR-pair base (the peer base kernarg, uniform) and a 32-bit VGPR offset
  (`saddr` form), never buffer descriptors (H7) and never `flat_*` (flat would also go through the LDS aperture check
  and a scalar-path load of a mailbox is forbidden anyway). The arena is < 4 GiB, so 32-bit offsets suffice.
- **Dead lanes** (a push whose `live` is false) get the offset of a per-rank junk line in the sender's own arena
  instead of a branch: the store stays unconditional and convergent, so the waitcnt pass does not merge a store path
  with a store-less one (FlyTape's reason for its out-of-extent buffer trick, done with global addressing).
- `Q/ccl/layout.py` owns the arena map (offsets of every exchange, slot strides, control block), computed on the host
  and passed as constants; `validate()` checks overlaps and alignment (8 B pairs, 1 KB lines, 16 KiB blocks for NUMA
  copies).

### 4.2 Wire forms

| form | layout per (slot, src rank) | push | ready test | payload share on the link | use |
|---|---|---|---|---|---|
| `pair` | `count` x i64 `(tag << 32) | value32`; bf16 data packs two columns per value | one `global_store_dwordx2 sc0 sc1` (or `x4`: two pairs) per element per rank | every pair's tag == expected | 0.50 | headers, cancel, argmax / LSE / top-k merges, all-reduce at rows <= 4 (narrow path) |
| `line` | `count` payload dwords + `count / 256` flag dwords; a line = 1 KB written by one wave (lane l stores 16 B) | payload `x4` stores, then a publish of the line's flag (4.2.1) | the flags of the needed lines == expected tag; payload loads gated by the flag difference (FlyTape's index trick) so they cannot issue before the poll | 0.996 | bulk all-reduce / all-gather, rows >= 8 |
| `lamport` | payload only; slots S = 3; empty = a sentinel (bf16 `0xFFFF`, f32 `0xFFFFFFFF`, NaN patterns the producer never emits because it canonicalises NaN outputs) | payload stores only | no sentinel left in the needed payload | 1.0 | knob for bulk data (aiter's one-shot form) |

#### 4.2.1 Publishing a line without draining the weight pipeline

On gfx950 `vmcnt` counts loads and stores together, so FlyTape's `wait_stores()` before the flag is a `vmcnt(0)` that
also waits for every weight load the producer already issued for its next unit. Three variants (`publish=`):
- `unit_end` (default for correctness bring-up): publish at the end of a unit, before the next unit's loads are issued
  (one pipeline bubble per unit; FlyTape's choice).
- `comm_wave`: the producer waves write the tile into LDS; one dedicated wave of the CTA (8 waves at 512 threads)
  copies it to every rank with `x4` stores, `vmcnt(0)` on its own stores only, then the flags. The compute waves never
  drain. LDS cost one tile (2-8 KB).
- `counted`: payload stores, next unit's loads, then `s_waitcnt vmcnt(n_loads_issued_after)` (stores retire in issue
  order), then the flag. Needs exact waitcnt control: inline asm plus a lint that the compiler did not insert an
  earlier `vmcnt(0)`, or the Rust direct backend (`DESIGN.md` 12.6). Default-off lever.
`pair` and `lamport` need no publish step: the data is its own ready flag.

### 4.3 Tags, slots, epochs, wrap

- `tag = epoch + use_in_launch + 1` (u32; 0 is skipped on wrap, so zeroed memory never matches), `slot = use mod S`.
  `use_in_launch` = the static index of this exchange instance in the launch's schedule (layer slot x exchanges per
  layer + offset), computed at trace time from the schedule; `uses_this_launch` is a launch constant (depends on rows
  bucket and layer count only).
- `S = 2` by default where the WAR rule of 4.9 is proven by the checker, `S = 3` for `lamport`, `S >= 3` wherever the
  checker cannot prove the chain (it raises unless the caller raises S).
- **Exact match only.** A pair whose tag is in `(expected, expected + S]` raises TAG_AHEAD; anything else is "not yet".
- **Wrap:** u32 tags wrap after about 2^32 uses (tens of millions of steps at about 100 uses per step). Wrap is safe
  because a slot is rewritten every S uses and acceptance is exact-match; the only hazard is an element that was never
  rewritten (a larger-batch launch's rows that a smaller-batch launch does not read), which by construction is never
  read. The stress test starts the epoch at `2^32 - 5000` and runs through the wrap (7.3).
- The fused-MoE private-tag rule of the narrow path (`tag ^ 2^31`, `research/q38-learnings.md` pitfall 6) does not
  apply here: CCL tags live in their own arena.

### 4.4 Deterministic reduction

- Every consumer reads all ranks' partials for an element and adds them in f32 in rank order 0..N-1 (unrolled; N is a
  trace-time constant, part of the kernel name). Each rank adds the same bits in the same order: rank-agreement is
  bitwise, and a numpy f32 loop in rank order reproduces it bit for bit (the CPU reference of every test).
- Partials are pushed as `ar_dtype=bf16` (default; one extra rounding vs f32 partials, FlyTape's choice, accuracy-gated
  in `DESIGN-LB.md` 9) or `f32` (2x bytes; the bitwise-vs-TP1-golden twin for drift measurements) or `fp8` (MXFP8 per
  32 columns, 0.53x bytes, tolerance class, default off).
- Two-shot: the reduce-scatter owner of a column block adds in rank order, then all-gathers the reduced block; every
  rank receives the same bits (one rounding to `ar_dtype` at the all-gather unless `ag_dtype=f32`).
- The sum is consumed in f32 by the consumer's own epilogue (HC combine `r + s y`), so no rounding sits between the
  rank sum and its first use unless the consumer needs bf16.

### 4.5 Collectives

| collective | where it is used | algorithm | default form |
|---|---|---|---|
| all-reduce (sum) | GDN / QSA out-projection, MoE combine (two per layer), PLE reduce | **one-shot**: every producer tile pushes its partial to all N ranks; consumers poll and sum. **two-shot**: producer pushes column block b only to its owner rank b mod N (reduce-scatter), the owner sums in rank order and pushes the reduced block to all ranks (all-gather) | one-shot `line` at rows >= 8, `pair` at rows <= 4; two-shot chosen by `ar=auto` from WP-C4's crossover table (expected: TP >= 4 at rows >= 64; never at TP2 where it moves the same bytes over the one link with a second hop) |
| all-gather | vocab-parallel head: per-row (max, argmax) and its ATOM tie rule; per-row (max, sum-exp) for logprobs; per-row local top-K' for non-greedy sampling; full logits rows for `logits=out` | push to all, consumer merges in rank order | `pair` for merges, `line` for logits |
| merge rules | argmax: larger value wins, equal values take the lower **global** token id (ATOM `topk_select(..., tie="low")`); LSE: `M = max_r m_r`, `S = sum_r (in rank order) s_r exp(m_r - M)`; top-K': K'-way merge by (value desc, global id asc) | bitwise across ranks, CPU reference | - |
| reduce-scatter | two-shot all-reduce; `hc=sp` (token-sharded HC residual at TP >= 4, `DESIGN-LB.md` 7.4) | push block b to owner b | `line` |
| broadcast | launch header, cancel word, rank-0-only host words | push to all | `pair` |

Expert-parallel all-to-all is not needed: the activations entering the MoE are replicated (the HC mixers and the
router are replicated), and the intermediate-split MoE ends in the same all-reduce as the attention. It would become
relevant only for an EP layout at TP8, where MXFP4's 32-column groups no longer divide the per-rank intermediate (640
/ 8 = 80): `DESIGN-LB.md` 7.3 notes the TP2 x EP4 hybrid as a later option.

### 4.6 Timeouts, errors, abort propagation, launch header

- Every poll loop reads `s_memrealtime` once per round; past its deadline it records PEER_TIMEOUT (with stage, layer,
  peer of the first missing element, expected and seen tag), sets the CTA's ABORTED word, pushes `abort = 1` into every
  peer's `ctrl_dev` (system scope, unconditional), and returns zeros. Later polls of an aborted CTA skip their spin.
- Every poll also checks the local `abort` word once every 32 rounds (one extra local load); a set word ends the poll
  with PEER_ABORT. A dead peer therefore stops the group within about one deadline, not one deadline per exchange.
- Deadlines (`deadline_first` for the first exchange of a launch, which absorbs host launch skew between ranks:
  default 10 s; `deadline` for the rest: default 1 s), both kernel constants derived from the 100 MHz clock.
- **Launch header:** the first exchange of every launch is a `pair` broadcast of `(epoch, rows, uses_this_launch,
  schedule hash)`; a mismatch on any rank records CCL_MISMATCH on every rank (all see the same pairs) and the launch
  aborts before any layer runs.
- The host side of ATOM may hold one rank in Python for tens of ms (GC); `deadline_first` covers that, and a rank that
  never launches turns into PEER_TIMEOUT on the others (the "absent rank" test).

### 4.7 Memory type, cache policy, homing

| item | default | levers (WP-C6) |
|---|---|---|
| receive arena memory | uncached | `mem=fine`, `mem=coarse` (with system-scope bits on every access) |
| push cache bits | `sc0 sc1` (system scope) | `+nt` (streaming: bulk lines are read once by the receiver); `sc1` only (agent scope: must fail the coherence negative control on coarse memory, measured on uncached) |
| poll / payload loads | system-scope `global_load_dwordx2/x4 sc0 sc1` (monotonic atomic loads in the `ir` impl, never hoisted) | `sc1` only on uncached memory; 16-byte polls (two pairs per lane) and one-wave-per-CTA polling (the mailbox `pre_poll` throttle) at TP8 (FlyTape's poll-traffic open question) |
| homing | flags and pairs: **dual copies on the receiver** (one per receiver IOD, written by the remote producer: two remote stores per flag), consumers poll the copy on their own IOD (`local_copy`, calibrated per rank by the existing ping-pong on the receive arena at group creation, constants kept on device) | `homing=single` (one copy homed on the IOD of the consumer set when it is known, e.g. a fixed reducer CTA set); payload lines are never duplicated (bulk bytes), they are striped across both IODs by address |
| producer side | push from registers or the LDS tile, no local store first (the GLM / vLLM mono pattern); the rank's own copy goes through the same path (writes its own arena) so the consumer code is rank-agnostic | `self=local` (own partial stays in registers / LDS for producer-consumer co-placed units; changes nothing numerically) |

### 4.8 Atomics policy and lints

- The protocol is **push-only**: no peer atomics at all. Exit counters, claim words and the epoch advance are local
  (agent scope) except the error-claim word (local) and the ABORT store (a plain system-scope store).
- **No float atomics on peer memory, ever** (and none in reductions anywhere: determinism). `P/core/lower_llvm.py`
  attaches `!amdgpu.no.remote.memory` / `!amdgpu.no.fine.grained.memory` to every f32 and bf16x2 `atomicrmw fadd`
  (H6); on a peer or fine-grained address that would license the native `global_atomic_add_f32`, whose result over
  xGMI is not guaranteed. The lowering is NOT changed (byte identity). Instead `Q/ccl/lint.py` walks a `KernelDef`'s
  def-use graph from every value derived from a peer-base kernarg (names `peer*`, `ctrl`) and fails the build on:
  any atomic RMW whose address derives from it; any `buffer_*` op or `rsrc` built from it (H7); any scalar
  (`s_load`) read of it; any load of it without system-scope bits. If an atomic on peer memory is ever needed, it is a
  new Python-only opcode (`atomic_add_remote`, no metadata, CAS-free i32 only) in `ir.PYTHON_ONLY_OPCODES`, never an
  attribute on a mirrored op (`research/q38-learnings.md` pitfall 6: the C++ parser may drop attributes).
- The ISA lint of `DESIGN.md` 6.3 (spills, exec-join spills, scratch) applies to every CCL build; the CCL lint adds:
  every peer store carries `sc0 sc1` (or the knob's bits), no `s_waitcnt vmcnt(0)` between a unit's weight issue and
  its first MFMA except the `unit_end` publish points.

### 4.9 Inside the persistent kernel

- **Producer epilogue pushes, consumer prologue polls, no phase:** the out-projection fold / the MoE combine pushes
  its finished tile (or column group x token block) to all ranks as soon as it is final; the consumer unit (the next
  HC combine + down-projection K-slice) issues its own weight loads first, then polls exactly the lines its K-slice and
  token block need (`DESIGN-LB.md` 5). On one GPU the same edge is a tile-flag mailbox; with TP the producer set simply
  spans ranks.
- **Push before poll inside a unit** (FlyTape's deadlock rule): a unit never polls an exchange after pushing into one
  it also reads in the same stage; across stages, every CTA runs its units in the stage order of the schedule, all
  dependencies point to strictly earlier stages, and every CTA of every rank is resident (residency ticket per rank).
  By induction on stage order no cycle exists, so no deadlock; a missing rank is a timeout, not a hang.
- **WAR rule (slot reuse) without grid syncs.** A push of use u + S into slot (u mod S) on rank P is safe iff every
  consumer unit of use u on P is a transitive ancestor of the pushing unit on the pushing rank, through local edges and
  exchanges (exchanges are all-to-all, so a dependence on use u+1 on rank R implies a dependence on P's push of u+1,
  which follows P's consumers of u wherever P's data flow makes them ancestors). `Q/ccl/war.py` checks this over the
  launch's task graph (the LB `TaskGraph`, or `atom_layout`'s for the narrow path) for every exchange, including the
  wrap into the next launch; it raises with the offending unit, and the fix is a larger S or an explicit edge. A
  mutant schedule with one consumer moved off the chain must fail the checker (CPU test) and, with the checker
  bypassed, must produce TAG_AHEAD or a mismatch on hardware (GPU negative control).
- **Grid-sync mode** (`sync=grid` knob of `DESIGN-LB.md` 5): exchanges still add no phase; RAW is the tag, WAR is
  covered by the existing grid syncs exactly as in FlyTape.
- **Narrow path** (`DESIGN.md`, rows 1-16): the existing mono mailboxes become rank-spanning by giving their put a
  second (peer) base: `put` to the own copy and every peer copy with `MailboxKnobs(scope="system")`, poll the own
  copy. C1 / M5 tasks are the producers, A1 / M1 / C1 / P2 the consumers, so TP2 at small batch needs no new
  structure, only the K-split weights and the rank-ordered sum in the consumer.

### 4.10 Knobs (`CclKnobs`; the kernel name names only non-defaults)

`form` (pair / line / lamport, per exchange or `auto`), `ar` (oneshot / twoshot / auto), `ar_dtype` (bf16 / f32 / fp8),
`slots` (2 / 3 / 4), `publish` (unit_end / comm_wave / counted), `mem` (uncached / fine / coarse), `push_cpol`
(sc0sc1 / sc0sc1nt / sc1), `poll_cpol`, `poll_width` (8 / 16), `poll_wave` (all / one), `homing` (dual / single),
`self` (push / local), `deadline`, `deadline_first`, `header` (on / off: off only for microbenchmarks). Every value
is built (no knob value without a build); losers stay registered default-off knobs.

---

## 5. Repo placement and byte identity

| piece | path | engine | byte-identity |
|---|---|---|---|
| host runtime | `P/runtime/peer.py` (+ `P/runtime/__init__` export only if the runtime exports are listed there) | host | not IR; no existing module imports it |
| device emitters | `Q/ccl/` (`layout.py`, `wire.py` pairs / lines / lamport, `collectives.py`, `merge.py`, `war.py`, `lint.py`, `errors.py`) | Python engine where a Python-only opcode is used (`_pyonly` suffix, `backend="python"` explicit, tests assert the backend); asm-mailbox-only builds stay mirrored | outside the gate only when they use Python-only ops (the `Q/README.md` exemption); `test_toy_cpp_engine_byte_identical` and `test_day1_cpp_engine_byte_identical_for_asm_impl` keep passing |
| tests | `platform/tests/test_peer_runtime.py` (CPU: validation, handle packing, file rendezvous with two local processes), `library/tests/test_qwen38_ccl.py` (CPU: layout, tags, WAR checker, lints, CPU references; GPU-marked multi-process tests run in the harness) | - | - |
| harness | `hackathon-prep/qwen38/truemono-lb/ccl/` (multi-process launcher, RCCL / aiter / FlyTape arms, jobs) | - | - |
| docs | `R/platform/dsl_docs/runtime/limitations.md` "Multi-GPU and RCCL": updated when WP-C1/C2 land (what now works, H7's answer) | - | - |

---

## 6. Benchmarks

### 6.1 What to beat

| arm | what | how it runs |
|---|---|---|
| FlyTape exchange | `F/runtime.py` `PeerAccessor` (pairs) and `PeerLines` (lines) in the toy (`scripts/bench_tp.py`) and in the TP2 qwen4 tiled tape (`--trace`, exchange phases and `mix_combine` waits) | container mode, flydsl 0.3.4.1, `ft_bench_tp1.sh` with `TP=2` |
| RCCL | `torch.distributed.all_reduce` (backend "nccl" = RCCL) on a dedicated stream, HIP-graph-captured chain of 96 collectives with a tiny kernel between them; also `all_gather_into_tensor` | venv mode (torch 2.12 + ROCm 7.2) and container mode (ROCm 10) |
| aiter custom all-reduce | `aiter.dist.communication_op.tensor_model_parallel_all_reduce` (ATOM's path), graph-captured | container mode |
| rocKE CCL | `Q/ccl` one-shot / two-shot x pair / line / lamport, isolated persistent microkernel (chain of 96 all-reduces, each consumer feeding the next producer) and in situ (6.3) | venv mode (pure rocKE + torch for the process group) |

### 6.2 Sizes and metrics

- Rows {1, 2, 4, 8, 16, 32, 64, 128} x columns {2560, 10240} x {bf16, f32}; TP {2, 4, 8} (TP4 / TP8 only when that
  many idle GPUs are available on one node: never `--exclusive`, opportunistic; TP2 always).
- Metrics: (a) per-collective latency in the chain (median of per-step device stamps / 96); (b) exposed cost in situ:
  `T(GEMM + AR pipeline) - T(GEMM alone)` with the same tile schedule; (c) link utilisation at 64-128 rows (bytes /
  time vs WP-C0's per-pair bandwidth); (d) for RCCL / aiter also the per-call host-side cost, labelled separately.
- Discipline (`DESIGN.md` 9.1, `research/q38-learnings.md` section 2): every arm in one job, interleaved shuffled
  order per round, 12+ rounds, two controls (a second instance of the baseline arm, and a recompiled rocKE build),
  numerics before timing (every rank's result bitwise vs the CPU rank-order reference for the rocKE arms, and within
  RCCL's documented tolerance for the others), node + GPU bus ids + xGMI pair recorded per row, SCLK sampled.
  Absolute results go to `private-perf/qwen38-true-mono/large-batch/ccl-*.md`; this tree gets ratios.

### 6.3 In-situ benchmark (the one that decides defaults)

A persistent microkernel with the LB out-projection shape (2560 x 3072 MXFP4 at TP2, rows 8-128): tiles of the GEMM
push their bf16 partial as they finish; a consumer stage (the HC down-projection K-slice shape) polls and sums.
Arms: no exchange (TP1 shape, the floor), FlyTape-style (push in the last arriver's fold, consumer in a following
phase behind a grid sync), rocKE `publish=unit_end / comm_wave / counted`, `form=line / lamport`, `ar=oneshot /
twoshot`. Decides `form`, `publish` and the `ar=auto` thresholds per (rows, TP).

Run as WP-C5 (W6); results and the chosen defaults in section 10.2.

---

## 7. Tests

### 7.1 CPU (no GPU)
- Layout: no overlaps, alignment, arena size per (TP, rows bucket, schedule).
- Tags: unique within S consecutive uses of every exchange across the launch boundary and the u32 wrap; tag 0 never
  live; `uses_this_launch` matches the schedule walk.
- WAR checker: every real schedule passes; mutants (a consumer moved off the chain; S=1; a launch boundary without the
  epoch advance) fail with the right message.
- Lints: a synthetic kernel with an f32 atomic on a peer-derived address, a buffer resource on a peer base, a scalar
  load of a mailbox and a peer store without system bits each fail.
- CPU references: rank-ordered f32 sums of bf16 / f32 partials, argmax with planted ties across ranks (lower global id
  wins), LSE merge, top-K' merge.
- Host runtime: argument validation, file rendezvous with two local processes (no GPU: a fake allocator).

### 7.2 One GPU (`ranks = 1` and `mode="p2p"` self-peer)
- `ranks=1` runs the whole exchange path (push to self, poll, sum of one) and must be bitwise equal to the no-exchange
  build's result for every collective.
- Self-peer IPC (open this process's own handle in a child process) for the IPC plumbing on a 1-GPU allocation.

### 7.3 Multi-process (2 GPUs always; 4 / 8 when available)
| test | expectation |
|---|---|
| bitwise rank agreement, every collective, every form, sizes of 6.2, odd rows (1, 3, 5, 17, 127) | every rank's output identical and equal to the CPU rank-order reference bit for bit |
| GPU permutation (rank r on a different physical GPU) | identical bits |
| consecutive launches without reset (1,000 launches), ranks skewed by random host sleeps 0-50 ms | all correct; epochs agree; no error |
| tag wrap: epoch initialised to `2^32 - 5000`, 200 launches straddling the wrap | all correct |
| stress: random per-CTA `s_sleep` delay injection at every push and poll point (`DESIGN.md` 7.8 gate 5), 10,000 launches | all correct; a torn-pair detector (each value carries a checksum of its tag) never fires |
| **dropped peer** (one rank joins the reset but does not launch) | the others record PEER_TIMEOUT within `deadline_first` (+ one round), then PEER_ABORT propagates; the next launch after `reset()` is correct |
| **aborting peer** (one rank's kernel is built to abort at layer 7) | the others record PEER_ABORT, not PEER_TIMEOUT, well inside the deadline |
| **stale tag** (receive slots pre-filled with tags u - S, u - 1, u + S) | never accepted; u + S raises TAG_AHEAD |
| **mutant acceptance** (`>=` instead of `==` on the tag; slot index off by one; rank-order sum reversed on one rank) | the bitwise tests fail (must-fail controls) |
| **coherence negative control** (`poll_cpol` plain loads on `mem=coarse`) | expected to read stale data or time out; recorded as discriminating or not |
| **launch mismatch** (one rank launched with a different rows bucket) | CCL_MISMATCH on every rank before any layer runs |
| cancel on one rank (head exchange carries the cancel word, OR over ranks) | every rank stops at the same step |

---

## 8. Work packages

Cost: G = GPU-hours (multi-GPU jobs count every GPU), C = CPU core-hours. All GPU work on the alola rocke partition
via `tools/q38/q38_job.sh` (`--gpus` up to 7, `Q38_MAX_GPUS=10` exported), stages named `w3-*`, at most 4 of our
jobs queued; every multi-GPU job checks that its GPUs are idle (`squeue -p rocke`, `amd-smi process`) and records node
and bus ids per row.

| WP | title | depends | gate | cost |
|---|---|---|---|---|
| WP-C0 | topology and platform facts: `amd-smi topology` / `rocm-smi --showtopo` on both rocke nodes, `hipDeviceCanAccessPeer` matrix, IPC on uncached / fine / coarse in venv and container, single-process p2p, per-pair push RTT and bandwidth (one peer and N-1 concurrent), receive-arena IOD calibration on each memory type, H7 diagnostic | - | facts written into the alola-cluster skill; H1/H2/H5/H7 answered | 3 G |
| WP-C1 | host runtime `P/runtime/peer.py` (3.1-3.5), file + gloo rendezvous, watchdog | - | CPU tests 7.1; 2-GPU smoke (alloc, open, push one pair each way, reset, close) | 1 G, 4 C |
| WP-C2 | device emitters `Q/ccl/`: wire forms + publish variants, collectives, merges, epoch / header / timeouts / abort, WAR checker, lints | WP-C1 | CPU tests 7.1; one-GPU 7.2 bitwise | 2 G, 30 C |
| WP-C3 | multi-process test suite 7.3 (TP2; TP4 when 4 idle GPUs are free) | WP-C2 | every must-pass passes, every must-fail fails, stress clean | 6 G |
| WP-C4 | microbenchmarks 6.1-6.2 vs FlyTape / RCCL / aiter, crossover table for `ar=auto` and `form=auto` | WP-C3 | numerics first; per-size win table (ratios here, absolutes private) | 6 G |
| WP-C5 | in-situ GEMM + all-reduce pipeline 6.3 | WP-C3, `DESIGN-LB.md` WP-L2a (tiled GEMM core) | exposed cost <= 0.25 of FlyTape's per-layer exchange share at TP2 B128; defaults chosen | 4 G |
| WP-C6 | levers 4.7: memory type, cache bits, homing, poll width / wave, `self=local`, TP8 poll traffic | WP-C4 | WIN rule per lever (`DESIGN.md` 9.1); losers stay knobs | 4 G |
| WP-C7 | integration hooks: `CclKnobs` in the LB framework and the narrow framework (`put` with peer bases), rank-local weight sharding helpers (host), `ctrl` error reporting in the harness and in ATOM-side experiments (`hackathon-prep/qwen38/truemono-lb/`) | WP-C2 | the LB TP2 layer gate (`DESIGN-LB.md` WP-L5) uses it unchanged | 1 G, 10 C |

Order: WP-C0 || WP-C1 -> WP-C2 -> WP-C3 -> WP-C4 || WP-C5 -> WP-C6; WP-C7 after WP-C2. Total about 27 G, 44 C.
Status per WP after W3 / W4 / W6 and the WP-C5 gate as restated: section 10.6.
`DESIGN-LB.md` needs WP-C2 + WP-C3 before its first TP2 layer (WP-L5) and WP-C5 before quoting TP2 budgets.

---

## 9. Risks

| risk | effect | mitigation |
|---|---|---|
| H3 false (torn 8-byte pairs over xGMI) | silent wrong sums | lines (flag after drained payload) are immune; the torn-pair detector in the stress test; pairs used only where the detector stayed clean |
| `publish=unit_end` bubbles dominate at small tiles | exposed exchange cost | `comm_wave`; `lamport`; bigger tiles per publish |
| rank launch skew in serving (Python GC, scheduler jitter) | CTAs of the early rank spin at the first exchange (GPU busy, no harm) | `deadline_first`; the spin is at the first exchange only, later exchanges are data-paced |
| two-shot never wins at TP2 | none (one-shot default) | `ar=auto` table from WP-C4 |
| TP8 poll traffic and SGPR pressure (FlyTape: SGPR spills grew with TP) | slower polls, spills | 16-byte polls, one-wave polling, peer bases in a small LDS table read once per stage (not per push) if SGPRs run out; ISA lint fails on spills |
| uncached arena too slow for bulk lines (no MALL benefit for the receiver's repeated reads) | slower consumers | `mem=fine`; consumer copies the reduced tile to its LDS once per unit |
| other users' jobs on the partition | multi-GPU jobs wait or hang | idle check at job start, TP2 on the two most idle GPUs, never `--exclusive`; TP4 / TP8 opportunistic |
| byte-identity regression | repo invariant broken | nothing in `rocke.helpers` changes; the two byte-identity tests run in every WP gate |

---

## 10. Measured (W3 / W4 / W6): what the results change

W6 synthesis, 2026-11-01, from `ccl/BUILD.md` (WP-C0-C3, WP-C4 part), `large-batch/CCL-LM.md` (large-message
all-reduce, W4, with its fix round) and `large-batch/WP-C5.md` (in situ, W6, with its fix round). Ratios only;
< 1 = rocKE faster. Every number is DESIGN 9.1 on both rocke nodes unless marked as one run. The large-batch budget
that uses these numbers is `DESIGN-LB.md` section 15.

### 10.1 Standalone collective (WP-C4, CCL-LM)

- The W3 grid-synced line one-shot lost to aiter's custom all-reduce from 160 KiB per rank (1.12-1.22). The CCL-LM
  selection (`layout.lm_select(..., layer=True)`) beats aiter's captured graph at every size from 80 KiB to 2.5 MiB
  per rank, bf16 and f32, on both nodes, with the persistent launch counted:
  - lamport one-shot at 80-320 KiB and at 640 KiB f32 (0.66-0.88; 4 waves per CTA at 80 KiB);
  - pull with 2-4 KiB units on 32-64 CTAs at 640 KiB bf16 and 1.25-2.5 MiB (0.90-0.99; 2.5 MiB bf16 a WIN in 5 of 6
    runs), the one-word grid sync above 1.25 MiB;
  - counted publish only in the 1.25 MiB f32 band.
- 5 MiB per rank (f32) is a link-bound tie (0.983-1.002 over 5 runs): both stacks are bound by the one xGMI link.
- Two-shot never wins at TP2 (it moves the same bytes over the one link with one more hop), as section 9 expected.
- Job-level spread of the in-kernel exchanges on one GPU pair is 1.5-3 % (alola rule 167): single-job verdicts near
  1.0 need a second job.

### 10.2 In situ at TP2 (WP-C5, 6.3): the defaults

WP-C5 built every value 6.3 asks for (lines with `publish=unit_end / comm_wave / counted`, lamport, pull, `ar=oneshot /
twoshot / auto`, the consumer as a pass, inside the fold, or behind a grid barrier as FlyTape does) in one persistent
launch with P-L3's mx out-proj GEMM, rows 5-128, and raced them against the default arm, GEMM + aiter and GEMM + RCCL.

| knob | 0.2 / 4.5 / 4.10 said | measured default at TP2 rows 5-128 | the other values (ratio to the default, both nodes) |
|---|---|---|---|
| `form` | `line` at rows >= 8, `pair` at rows <= 4 | **`lamport`** | `line` 1.06-1.16 for every publish variant; `pull` 1.02-1.07 (CCL-LM's standalone pull win does not transfer) |
| `publish` | `unit_end` for bring-up, `comm_wave` / `counted` as levers | not applicable (lamport needs no publish) | `counted` and `comm_wave` shave the line drain by 0.01-0.06 at 5-16 rows but cannot remove the flag round trip |
| `ar` | one-shot, two-shot from a crossover table | **one-shot** | two-shot 1.01-1.07 (loss shrinks with size) |
| consumer placement | the next stage's prologue polls (4.9) | **`cons=pass`** (a pass after the layer's last unit) | `cons=fold` 1.04-1.14 at rows 5-64, a tie at 128 (0.991-0.996 in 6 runs); FlyTape's `cons=phase` (consumer behind a grid barrier) 1.07-1.54 |
| grid barrier | - | the GEMM's own (`gb=gemm`) | `gb=lite` ties or loses |
| poll pause | - | `s_nop` | `s_sleep 1` / `16`: noise |

- End to end per layer the default beats GEMM + aiter at every TP2 size from 5 to 128 rows (0.83-0.93, every round
  below 1) and GEMM + RCCL's best protocol (0.62-0.73). Below 5 rows (outside the large-batch scope) it ties or loses
  to aiter at 1 row: the phase barrier dominates.
- `insitu.AUTO` (`ar=auto`) holds this table. Every other value stays a registered default-off knob (4.10).
- Consequence for section 0 item 2 and 4.5: at large batch the `line` form is not a default anywhere it was measured.
  It stays the form for all-gathers of full logits rows (not measured) and a knob for all-reduce. `pair` stays the
  narrow-path form (rows <= 4), which W5's small-batch path decides.
- FlyTape's own structure (consumer behind a grid sync) is itself a large part of why the in-situ exchange beats
  FlyTape's: `cons=phase` costs 0.11-0.41 of a layer with lamport.

### 10.3 Exposed cost against FlyTape, and the WP-C5 gate

- **Where the exposed cost sits.** From 16 rows up it is 0.89-1.01 of the time the bf16 partial needs to cross the
  one xGMI link: every unit of the out-proj layer finishes within one unit's time, so the whole payload crosses after
  the last fold. Below 16 rows it is a fixed push-to-poll round trip (rows 16 / rows 1 = 0.95-0.99), and aiter's
  separate kernel pays 2.3-2.9x it.
- **FlyTape's exchange cost, bracketed** (both ranks traced in one job, three estimates: old proxy, restricted to the
  exchange phases, skew floor): 0.060-0.082 (skew floor) to 0.078-0.107 (old proxy) of its TP2 step at B8-B128.
- **Per step (the number the budget uses):** with both exchanges measured in situ (out-proj `lam` + MoE-down `lamMu`),
  the 96 exchanges cost 0.048-0.061 of FlyTape's TP2 step: below FlyTape's skew floor at every batch, by 0.003-0.025.
- **The gate of section 8 is not met.** Exposed cost per exchange over FlyTape's per-exchange cost at TP2 B128:
  out-proj 0.58-0.77, MoE-down 0.49-0.74 (default layout), under every estimate, against a target of 0.25. The link
  time alone is 0.65-0.87 of FlyTape's per-exchange cost, so no epilogue knob can reach 0.25 on this schedule:
  `ar_dtype=fp8` (0.53x bytes, tolerance class, not built) would give about 0.34-0.46.
- **Restated gate.** WP-C5 closes with its defaults chosen at TP2 and the per-step exchange share below FlyTape's own
  (both met). The 0.25 per-exchange target moves to `DESIGN-LB.md` WP-L5 as an overlap target: the next layer's
  units start on the blocks whose exchange has landed (DESIGN-LB 5's tile-flag dataflow), so the link time hides
  behind work instead of being cut. `ar_dtype=fp8` is the bytes lever behind it (WP-L8, accuracy-gated).

### 10.4 The MoE-down exchange is not the out-proj exchange

- WP-C5 first assumed both exchanges cost the same. Measured in situ on P-L4's MoE (`insitu_moe.py`): with the
  token-major arena it cost 1.3-8.3x the out-proj exchange. A combine-unit-major arena (`lay=unit`, 512 contiguous
  bytes per push instruction; now the default) wins 0.92-0.97 and leaves it at 3.3-4.3x (rows 8-16), 1.8-2.9x
  (32-64) and 0.84-0.96x (128).
- Likely cause (not isolated): the MoE's combine units finish at very different times, and each push waits for its
  unit's down-projection arrivals. Poll pressure is ruled out (pause levers are noise), and so is the separate pass
  (`cons=combine` ties).
- The in-situ MoE-down exchange alone is as cheap as aiter's at 8-32 rows (0.98-1.05) and cheaper at 64-128
  (0.49-0.74), but MoE + exchange in one launch still loses to MoE + aiter at 8-32 rows (1.05-1.08), because P-L4's
  persistent MoE walk costs 1.04-1.08 of one launch per layer. The in-situ MoE exchange pays off only on a MoE whose
  layer walk is no slower than per-layer launches: DESIGN-LB's rebuilt MoE (WP-L2d, DESIGN-LB 15.4).

### 10.5 TP4 (one run, n07-17 only; provisional)

- Every value built and bitwise on 4 ranks; all 65 must-fail twins detected. TP4 differs from TP2: two-shot wins at
  64-128 rows (0.88-0.96 of one-shot; it halves the bytes per link) and `cons=fold` at 32-128 rows.
- **Every in-situ value loses to GEMM + aiter at 32-128 rows (1.01-1.12)**, wins at 8-16 (0.87-0.94), and beats RCCL
  everywhere (0.57-0.75). Two-shot with the fold-side consumer is not built (`ar=twoshot` refuses `cons=fold`).
- `insitu.AUTO` holds this run's TP4 winners marked provisional. TP4 is not a plan-of-record configuration
  (DESIGN-LB WP-L11); a second node and the twoshot + fold combination are its open items.

### 10.6 Correctness rules added, and WP status

Rules (both from hardware findings):
- **Inline-asm VMEM hazard.** An asm memory instruction that reads an SGPR base the compiler restored from a spill
  (`v_readlane`) 0-2 cycles earlier needs 5 wait states, and LLVM does not pad inside inline asm. It broke P-L4's
  grid barrier and hit 6 of WP-C5's 135 measured builds (twins and one lever; no default arm) and most TP4 builds.
  Knob `hz` (default on) pads those asm blocks with `s_nop 4` (timing: a tie); `hz=False` reproduces the old IR. The
  ISA scanner (`p-l4/fix/asm_hazard.py`) runs as a compile gate on every CCL / in-situ / LB build: allocation decides
  whether the hazard appears, so a clean build today does not clear the next one (alola rules 176, 182, 191).
- **Numerics gates must be able to see stale data.** A bench whose chained payloads repeat lets a stale-partial read
  pass (CCL-LM issue 1: fixed with growth chains, `ref.LM_GROWTH`). Every in-situ arm carries must-fail stale-read
  twins (`noreset` per lamport arm, `slot` per line arm, `kskip` per GEMM). Plain-load coherence twins (`noacq`) are
  not a reliable must-fail check (P-L4 F2): they stay informational.
- Open: a rare `PEER_ABORT` of W3's pair form at 2.5 MiB (never selected at that size; every rank's record is now
  kept); aiter's `hipIpcOpenMemHandle` failures inside its own registration (re-run in a new process); RCCL at TP4 is
  not bitwise (tolerance arm only).

| WP | status | what remains |
|---|---|---|
| WP-C0, WP-C1, WP-C2, WP-C3 | done (W3, `ccl/BUILD.md`); suite 20 / 20 at TP2, permuted TP2 and TP4 (CCL-LM first round; the fix round changed no device code, IR of all 2284 earlier builds unchanged) | - |
| WP-C4 | done at TP2 (CCL-LM); TP4 standalone partly | crossover table at TP4 / TP8 when idle GPUs exist |
| WP-C5 | **partial: defaults chosen at TP2 rows 5-128, per-step share below FlyTape's own; the 0.25 per-exchange gate not met and moved to DESIGN-LB WP-L5 (10.3)** | TP4 second node; MoE-down exchange on the rebuilt MoE |
| WP-C6 | open: memory type, cache bits and homing were tested only as 5 MiB levers (all within 0.993-1.02) | in situ, on the LB layer (WP-L5) |
| WP-C7 | open; the WAR checker runs on microkernel schedules only | `CclKnobs` / `insitu` defaults wired into the LB framework (DESIGN-LB WP-L1 task graph, WP-L5), host error reporting in the LB harness; needed before WP-L5 |

### 10.7 What DESIGN-LB takes from this

- Budget: the 96 exchanges of a TP2 step at 0.048-0.061 of FlyTape's step (DESIGN-LB 15.2 / 15.3), not FlyTape's
  0.060-0.082 to 0.078-0.107.
- WP-L5 starts from: lamport one-shot, `cons=pass`, `gb=gemm`, `pause=s_nop`, `hz` on, `lay=unit` for the MoE-down
  push; re-decides `cons=fold` in the layer context (where the consumer is the next stage's prologue, 4.9) and owns
  the overlap target of 10.3.
- PLAN decision 5: the exchange is only worth having if LB TP2 beats LB DP2 (two TP1 replicas, no CCL). One DP2 GPU
  reads 1.14-1.40x a TP2 rank's bytes (L0c-DP2), the exchanges cost about 0.05-0.06 of the step, and the TP2 rank shape
  pays more sync than TP1 (P-L2); DESIGN-LB WP-L4 / WP-L5 measure both per cell.
