# LDS Banks & Bank Conflicts — Source of Truth

This is the complete reference for how the Local Data Store (LDS / shared memory) serializes accesses into
banks, what a bank conflict actually is, how to measure it, how to fix it, and — crucially — **when a fix is
worth making**. It is the SOT the **LDS Expert** (`.claude/team_members/lds_expert.md`) and the
**`bank-conflict` skill** cite; they hold judgment and procedure, this holds the model and derivations.

Peer docs: `mma_is_machinery.md` (the MMA position/label machinery) and `tiling_interleaving_design.md` (the
interleaving + vectorization contract). LDS layout choices interact with those — see §8.

> **Arch note.** Concrete numbers below are for **gfx90a / CDNA2 (wave64, NB=32 banks × 4 B)**. `NB`, bank
> width, wave width, and the exact serialization are **architecture-dependent — confirm per target** (CDNA
> vs RDNA differ in bank count and wave width). Never hardcode 32; treat the model as a template to validate.

---

## 0. The empirical discipline (the method this whole doc serves)

LDS bank conflicts are an **empirically verifiable statistic**. Do NOT state a conflict count, factor, or
"is/isn't a conflict" verdict from a diagram or hand-argument.

1. **No conflict claim without BOTH** (a) rocprof hardware counters AND (b) a simulator that reproduces those
   exact counters from the address map. If the simulator doesn't match, the model is wrong — fix it, don't
   narrate around it. The GPU is the arbiter.
2. **Report the model as a hypothesis until validated** ("predicted, pending counter confirmation"); state the
   mechanism as fact only once the simulator matches.
3. **Never store per-kernel results.** Every case (arch, tile, dtype, layout, pad, pipeline) differs —
   regenerate the data. This doc stores the MODEL and METHOD, never the answers.
4. **Conflict reduction is SUBORDINATE to the binding stage** (see §7). `SQ_LDS_BANK_CONFLICT` is a diagnostic,
   not the objective — wall-time is.

(This discipline exists because hand-reasoned claims about a real kernel flip-flopped and wasted time.)

---

## 1. The bank model

- LDS is `NB` banks, each **4 bytes = 1 dword** wide, serving **one dword per bank per cycle**. Byte address
  `a` → `bank = (a/4) mod NB`; for a dword index `d`, `bank = d mod NB`.
- **A wave64 LDS instruction is NOT issued for all 64 lanes at once.** It is served in **groups of ≤32 lanes
  (a half-wave)** against the `NB` banks, and a **wide** access (`ds_*_b64`/`b128` = 2/4 dwords per lane) is
  further split into **per-dword phases** (a `b128` = **4 phases**, one dword-position each).
- **A bank conflict is evaluated PER served group × per phase.** Within one (half-wave, phase), if two lanes
  target the **same bank at different addresses**, the bank can serve only one/cycle → the instruction
  **replays** the extra ones. Each replay is one extra cycle. `SQ_LDS_BANK_CONFLICT` counts replay cycles.
  - *N-way conflict* = N distinct addresses stacked on one bank in one served group/phase = N cycles (N−1
    replays). **This naive per-address replay rule is the READ-side model. For WRITES it OVER-COUNTS
    (~8–9× on K-aliased coop stores) — the write port has a different serialization; see §1.4.**
  - **Same-address** in a group is a **broadcast** (free) — a *separate* pathology counted by
    `SQ_LDS_ADDR_CONFLICT`, not bank conflict. Report it separately (it should be ~0 in a pure bank study).

### 1.4 The CDNA2 LDS **write-port** rule (validated gfx90a MI210, `tmp/lds_sim.py`)
The write datapath is NOT the naive `Σ_bank (distinct − 1)` replay counter (that over-counts stores ~8–9×).
It has **two hardware constants**, both confirmed by the fit being integer-exact on all measured store
configs **and** by independently reproducing the measured `productive` (= footprint/NB) it was not fit to:
- **Write-port width = 8 banks/cycle** — `min(banks_used, 8)`. The store port covers an 8-bank stripe per
  cycle, not all NB=32 banks.
- **Write-combine depth = 4** — same-bank stacked stores drain 4 per cycle (`depth / 4`).

  ```
  served(half-wave, phase) = min(banks_used, 8) × max_bank_depth / 4
  SQ_LDS_IDX_ACTIVE (IDX)  = Σ_half-waves MAX_phases served      # the instruction's dword-phases PIPELINE
  productive (floor)       = distinct_dwords_in_instruction / NB  # per instruction, once
  SQ_LDS_BANK_CONFLICT     = IDX − productive
  ```
  The dword-phases of ONE instruction pipeline through the port → MAX over phases per half-wave (this holds
  for `ds_write_b64`/`b128` and for the two dwords of `ds_write2_b32`). The counter is **per instruction**;
  a store forced to a narrow width issues several instructions, each measured with only its own footprint.
  Validated bit-exact (IDX and BC) on A/B × {b64, b128, b32} × {pad0, pad8} and the masked active-lane sweep.
  **This is a WRITE rule** — reads use the §1 per-address replay model; confirm the read port separately.

**Conflict-free CONDITION — the stripe-alignment rule (validated over a full 17-point pad sweep, both operands).**
The naive `bank = dword mod 32` map is not just imprecise for these padded K-aliased f16 stores — it is
*inverted* (it gives byte-identical served-group histograms for pad8 and pad16 and predicts the opposite
verdict). The validated condition: with `s = stride_dwords mod 32` and `W = dwords/lane` (b64→W=2, b128→W=4),
the K-aliased store is **conflict-free (BC=0) iff `s` is an ODD multiple of the half-stripe unit `4·W`**.
Worked: A (b64, W=2, unit 8) → pad+16 f16 gives stride 72 dwords, `s=8=1×8` → BC=0; B (b128, W=4, unit 16) →
pad+32 f16 gives stride 144 dwords, `s=16=1×16` → BC=0. `s=0` (any bank-multiple stride) is the fully-aliased
worst case. **A `b32` narrowing swizzle does NOT fix a b128 store** — measured `conflicts/access = 3.00`
(worse, 4× the LDS instructions): HW is the arbiter, and padding to the odd-half-stripe is the cheap fix.

### 1.1 The classic modeling trap
Summing a lane's whole `b128` (all 4 dwords) into ONE bank histogram makes conflicts *vanish* (looks like an
even floor). **Wrong** — the hardware arbitrates per phase. A correct simulator MUST iterate half-waves and
phases. A per-instruction-total model will confidently report "no conflict" when there is one.

### 1.2 Determine the served-group SIZE empirically (don't assume 32)
Run the simulator at group ∈ {8, 16, 32, 64} and see which reproduce the measured `conflicts/access`. In
practice a given pattern **rules out group=64** (whole-wave-at-once lowers the predicted conflict, because the
per-phase floor becomes `ceil(64/NB)=2` instead of 1) but often **cannot distinguish 8 vs 16 vs 32** (a
pile-up contained in a smaller sub-block is caught by any grouping ≥ that block). Report what you ruled out;
the CDNA default is the half-wave (32).

### 1.3 Order-independence
`SQ_LDS_BANK_CONFLICT` is a SUM over (group × phase) replays, so the **order** groups/phases issue in (e.g.
"all phases of half-wave 0 before half-wave 1" vs interleaved) does **not** change it — and the counter cannot
tell you the order. Determining issue order needs an order-sensitive signal (per-instruction latency / ATT
thread trace), not conflict counts.

---

## 2. Floor vs fixable conflict; the conflict-free condition

- **Throughput floor (unavoidable):** 64 lanes must share `NB` banks, and a wide access moves >1 dword/lane —
  so an access takes ≥ `(lanes × dwords/lane) / NB` cycles no matter what. This is NOT a conflict; it's the
  minimum. It cannot be "fixed."
- **Fixable conflict:** an *uneven* pile-up (some banks stacked while others idle) that costs cycles beyond the
  floor. This is what padding/swizzle/relayout remove.
- **`conflicts/access` = `SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE − SQ_LDS_BANK_CONFLICT)`** = replay cycles
  per productive cycle. `SQ_LDS_IDX_ACTIVE − SQ_LDS_BANK_CONFLICT` (productive) is the floor and is constant
  across pads for the same traffic.
- **Conflict-free ⇔ a per-phase bank PERMUTATION:** in every served group, each phase the lanes map to `NB`
  **distinct** banks (one each) — every bank busy once per phase = the floor. This is what "0 conflict"
  *means*; §7 says pursue it only when it pays.

---

## 3. K-stride aliasing (the common mechanism)

For an LDS tile addressed `addr(K, m) = K·stride + m` (free dim `m` innermost, contraction `K` the row):
`bank = ((K·stride + m)/2) mod NB`. **If `stride/2` is a multiple of `NB`** (i.e. the row stride is `k·NB`
dwords), then `K·stride/2 mod NB = 0` and **the bank index is independent of K** — every K-row of a given `m`
aliases the SAME bank. Any served group that carries the same free coordinate `m` with different `K` then
piles those lanes onto one bank.

*Worked structure (not a measurement):* a coop store where consecutive lanes step `K` by 4 and free-dim blocks
by 8, into a tile whose row stride is `2·NB` dwords → in each phase the lanes sharing a free-dim block (same
`m`, different `K`) collapse onto one bank → an 8-way pile, repeated across the group. Padding or a swizzle
that makes `stride/2 mod NB ≠ 0` re-introduces the K spread.

---

## 4. The contiguity floor and the width ladder (why "just swizzle to 0" isn't free)

A **valid** swizzle/pad must **preserve vector contiguity**: a `ds_*_bW` access is `W/2` **consecutive** dwords,
so a contiguity-preserving transform may only relocate whole `W`-blocks (XOR/shift the address bits *above* the
vector's dword span) — never split them. A transform that "reaches 0" in a sim but scatters the vector is
**invalid** (it is no longer a wide access).

Consequence: with the vector pinned at width `W` dwords, there are only `NB / W` distinct block-start banks, so
a contiguity-preserving swizzle (and padding) **floors at `group / (NB/W)`-way**. On gfx90a (NB=32, 32-lane
group):

| store width | dwords/lane (W) | valid-swizzle floor | LDS instrs (rel. b128) |
|-------------|-----------------|---------------------|------------------------|
| `b128`      | 4               | 4-way               | 1×                     |
| `b64`       | 2               | 2-way               | 2×                     |
| `b32`       | 1               | 0-way (conflict-free) | 4×                   |

**Reading it:** conflicts ↓ and instruction count ↑ *together*. Zero conflict is only reachable by narrowing to
`b32` (no block to preserve → each dword places freely → full permutation) — at 4× the LDS instructions. The
best width is **bottleneck-dependent** (§7), not "narrowest."

---

## 5. Fixes (levers), and the `vw_elems` mechanism

Ranked by cost (cheapest first). Apply the cheapest that relieves the binding stage (§7), then re-measure.

1. **Zero-instruction levers (keep full bandwidth):**
   - **Free relabel / symmetry** that lands the coalesced axis on distinct banks (best — costs nothing).
   - **Contiguity-preserving swizzle:** XOR the block bits *above* the vector's dword span. Keeps the wide
     access; de-aliases partially (floors per §4). rocWMMA/CK's move.
   - **LDS padding:** pad the row stride by a multiple of the vector's dword width so it stops aliasing banks;
     costs LDS capacity (occupancy). Also floors per §4.
2. **Narrowing swizzle (`b64`→`b32`):** removes the contiguity constraint so the swizzle reaches a finer/full
   permutation — but each halving of width **doubles LDS instructions** and can flip the kernel into
   issue-bound. Last resort; **measure**, don't assume lower conflict = faster (it often isn't — §7).
3. **Redistribute** the store/read so consecutive lanes step the free dim by 1–2 (natural bijection) *while
   keeping wide access* — the "proper" fix, but it changes the whole layout chain (global load, MMA read, C),
   so it can just move the conflict; measure end-to-end.

### 5.1 The swizzle-width family (worked example)
For the K-aliased coop store, the de-alias function `perm(K) = kk ^ (kk<<2)` with `kk = K>>2`, applied as
`m ← m ^ (perm(K) << shift)` with `shift = log2(width_elems)`, hits each width's floor:
`b128` (`<<3`) → 4-way, `b64` (`<<2`) → 2-way, `b32` (`<<1`) → 0-way. Same de-alias, shifted by the block
granularity. (Validated: conflict-free at `b32`, and a bijection over the tile → bit-exact.)

### 5.2 `vw_elems` — how the emit picks the width
A swizzle is a callable `(builder, positions) -> positions` passed as `lds_swizzle=` to
`store_fragment`/`load_fragment` (`helpers/tiling/emit.py`). It carries a **`vw_elems`** attribute = the
contiguity it preserves (its block width in elements). The emit resolves the access width as
`min(vw_elems, natural_contiguous_run)` and **range-checks it to `[1, natural_run]`** (`_swizzle_vw`), so a
swizzle can only relocate whole blocks of its declared granularity, never widen past the layout. `bool True`
selects the built-in block-preserving swizzle (keeps the natural run); `None/False` = off. Factory:
`_bank_swizzle(width_elems)` in `kernels/tiling_gemm_interleaved_demo.py` (`b32_swizzle`, `b64_swizzle`).

---

## 6. Padding always has a floor too

If the lane→bank offsets are all multiples of some `g` (e.g. consecutive lanes step the free dim by 8 and `K`
by 4, and a block-aligned pad preserves the ×`g` structure), only `NB/g` banks are reachable per phase, so
padding can only reduce the pile-up to `group/(NB/g)`-way — **not** to zero. Padding is the cheap *partial*
win; a swizzle (§4/§5) or a redistribution is required for a full permutation. Always MEASURE where the pad
floors out — and note padding costs LDS capacity (occupancy), which can itself cost more than the conflict.

---

## 7. The binding stage decides — conflict reduction is subordinate to wall-time

`SQ_LDS_BANK_CONFLICT` being large is **only** a problem if the LDS stage is on the critical path. The decision
is driven by the binding stage, not the conflict count:

- **Collect binding-stage counters alongside BC:**
  - `SQ_WAIT_INST_LDS` — cycles waves stall waiting on LDS. **This**, not BC, tells you LDS is *exposed*.
  - `SQ_VALU_MFMA_BUSY_CYCLES` — matrix-core (compute) saturation. Rising after an LDS fix = you un-starved it.
  - `SQ_INSTS_LDS` / `SQ_INSTS_VALU` — issue pressure (a narrowing swizzle multiplies `SQ_INSTS_LDS`).
  - `TCC_EA_RDREQ`/`TCC_EA_WRREQ` (×64 B) — HBM traffic. If HBM-bound, LDS conflicts are irrelevant; fix
    reuse/tiling. (GEMM at scale is usually NOT HBM-bound — it reuses on-chip.)
- **Decide:**
  - LDS **hidden** / MFMA- or HBM-bound → **do nothing** (fewer instructions = lower power; a "fix" only hurts).
  - LDS **exposed** → fix with the **cheapest bandwidth-keeping lever** (free relabel / pad / contiguity-
    preserving swizzle), then **RE-MEASURE — the bottleneck migrates** (fixing LDS exposure often makes the
    kernel MFMA- or issue-bound, at which point more LDS work regresses).
- **The trap, proven on hardware:** a fully **conflict-free** narrow (`b32`/`b64`) swizzle variant ran *slower*
  than a partially-fixed (4-way) padded one, because it quadrupled LDS instructions once compute was the
  limiter. **Lower conflict ≠ faster.** Report whether **TFLOPS actually moved**, not just whether BC dropped.
- **Occupancy/pipeline is often the higher-leverage lever than any LDS micro-opt** — a deep double-buffered
  pipeline can hide the LDS stage entirely (a conflict present in the fastest config). Ask "can I hide it?"
  before "can I remove it?".

---

## 8. Interaction with the MMA / interleaving layout chain

LDS is the pivot shared by the global→LDS **store** and the LDS→register **wave read**, which want different
things; someone eats the conflict. The store/read layouts come from the interleaving design (see
`tiling_interleaving_design.md`): the free-dim-contiguous store, the K-order wave read, and the vectorization
contract (a physical `VW>1` is valid only along a stride-1 axis). A/B are **separate** LDS regions — isolate
them (A-only/B-only, store-only/read-only) to attribute a conflict. The MMA machinery (position ≠ label, sound
contraction) is in `mma_is_machinery.md`; the LDS bank model here is orthogonal to it (banks are physical
addresses; the machinery is logical labels).

---

## 9. Counter reference (rocprofv3, gfx9)

| counter | meaning |
|---|---|
| `SQ_LDS_BANK_CONFLICT` | replay cycles from bank conflicts (0 for conflict-free; proves the counter is live) |
| `SQ_LDS_ADDR_CONFLICT` | same-address/broadcast stalls — a different pathology; report separately |
| `SQ_LDS_IDX_ACTIVE` | total LDS-busy cycles; `productive = IDX − BC` = the floor |
| `SQ_WAIT_INST_LDS` | waves stalled on LDS → is the LDS stage exposed? |
| `SQ_VALU_MFMA_BUSY_CYCLES` | MFMA (compute) busy → compute saturation |
| `SQ_INSTS_LDS`, `SQ_INSTS_VALU` | instruction counts → issue pressure |
| `TCC_EA_RDREQ`, `TCC_EA_WRREQ` | HBM requests (×64 B) → is it HBM-bound? |
| `GRBM_GUI_ACTIVE` | kernel active cycles (wall-time proxy) |

`conflicts/access` is a RATIO — robust even if the compiler drops loop iterations. SQ counters are per-SE
aggregates; sum/compare consistently across variants.

---

## 10. Measurement method (the how; the `bank-conflict` skill owns the exact recipe)

1. **Isolate the access** — the aggregate counter mixes A-store + A-read + B + C, and gfx90a has no
   read/write-split LDS conflict counter (ATT decoder may be absent). Build a **micro-kernel that does ONLY
   that access with the kernel's EXACT descriptor** (store-mirror = loop{store; sync; read}; read-only =
   store-once/loop-reads). Keep every probe **bit-exact**.
2. **Sweep a control** (pad / swizzle width) — if BC moves, the counter is live and the conflict is real; if a
   setting drives BC→0, that layout was conflicted and the setting fixes it. If nothing moves BC, question the
   measurement.
3. **Build + validate the per-half-wave/per-phase simulator** against the measured `conflicts/access` for every
   config. Gate: simulator == hardware to the number, `ADDR_CONFLICT ≈ 0`, bit-exact — else the model is wrong.
4. **Then, and only then**, state the mechanism, locate the colliding group (for the diagram), and evaluate the
   binding stage (§7) to decide whether/what to fix.

The container + rocprofv3 command recipe, the isolation-probe construction, and the register→LDS visualization
live in the **`bank-conflict` skill** — the operational hands. This doc is the model they execute against.
