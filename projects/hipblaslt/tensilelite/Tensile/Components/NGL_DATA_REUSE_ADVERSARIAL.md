# Adversarial investigation: does the SHADOW NGL pack re-use ML's ds_read data?

Read-only investigation. No source code changed. Every fragment-identity claim
cites `file:line` in
`Tensile/Components/kernel_comparison/shadow_capture_listing.txt`,
`.../cms1_capture_listing.txt`, or `.../cms0_kernel.s`.

Trace scripts used (throwaway, under the gitignored `hxcx_artifacts/`):
`_frag_trace.py`, `_frag_trace2.py`, `_frag_trace3.py`,
`_frag_trace2_cms1.py`.

---

## SUMMARY — VERDICT

**The user is WRONG.** I tried hard to prove the NGL pack repeats ML's
fragment and could not; the evidence is decisive and points the other way.

The SHADOW NGL real-math chain computes on LDS read-buffer **parity 0**
(`('A', 0, 0)`, fed by `shadow:440` ← ds_read `shadow:378`), while the
immediately preceding ML body computes on **parity 1** (`('A', 1, 0)`, fed by
`shadow:201` ← ds_read `shadow:184`). Different buffer parity = different
physical LDS half = a **different k-fragment**. NGL consumes the fragment that
ML *prefetched but never computed on* (ML's second-wave ds_read at `shadow:378`,
parity 0). That is textbook PrefetchLocalRead=1 software pipelining (benign
drain), **not** a repeat of ML's data.

This is identical in the real CMS kernel (`cms1`: ML parity 1 at
`cms1:247`, NGL parity 0 at `cms1:443`) and the read-buffer alternation is
confirmed in real emitted assembly (`cms0_kernel.s:1957/2285`,
`// swap Red Blk`). No correctness bug exists — the NGL math operates on fresh,
never-before-computed data.

---

## §1 The claim + the benign-drain-vs-repeat distinction

User's claim: in SHADOW, the NGL body's pack re-uses ds_read data from the ML
body, so NGL's math is a *repeat* of math ML already did.

The two competing readings (both must be tested against the actual LDS
read-buffer state, not register names — register names like `ValuA_X0_I0+0` are
reused every body):

- **Benign drain (prior analyses):** ML computes on fragment K and prefetches
  K+1; NGL computes on K+1 — fresh data, never computed before. NO repeat.
- **Repeat (user):** NGL computes on fragment K *again* — the same data ML
  already computed on — duplicating math.

Decisive discriminator: a *fragment identity* = `(side, read-buffer parity,
LDS offset)`. The offset alone is NOT enough — every body issues the same
offsets (0/64/128/192 and 256/320/384/448), so identity hinges on the
double-buffer parity advanced by `v_xor_b32 v[vgprLocalReadAddrA], 0x10000`
(the Red/Black block toggle).

---

## §2 LDS read-pointer / buffer-swap accounting (SHADOW, single timeline)

The unrolled timeline is PRO → ML_PREV → ML → NGL → NLL
(`shadow:6/50/244/438/618`). PRO is pure packing of the GR-prefetched tile
(`shadow:7-48`), no ds_read, no swap. Initial read-buffer parity = 0.

Read-buffer swaps (`v_xor LocalReadAddrA`), one per body, in linear order:

| swap | `shadow:line` | body | parityA transition |
|------|---------------|------|--------------------|
| 1 | `shadow:89`  | ML_PREV | 0 → 1 |
| 2 | `shadow:283` | ML      | 1 → 0 |
| 3 | `shadow:521` | NGL     | 0 → 1 |

(B-side swaps mirror at `shadow:90/284/522`.) NLL has **no** read-buffer swap
(`grep v_xor.*LocalReadAddr` returns only the six lines above). The read
buffer therefore alternates every body — exactly the AMD double-buffer (Red/Black)
LDS scheme. Confirmed in real emitted asm: `cms0_kernel.s:1957` and `:2285`
both carry the `// swap Red Blk` comment.

Within each body the ds_reads come in two waves:
- wave-1 (mfma_idx 0–7), offsets 256/320/384/448, **before** that body's swap →
  current parity;
- wave-2 (mfma_idx 38–45), offsets 0/64/128/192, **after** the swap →
  flipped parity (these are the PREFETCH of the next body's fragment).

---

## §3 Which fragment each SHADOW body's pack/math consumes

Traced by latest-writer over the linear timeline (`_frag_trace2.py`,
`_frag_trace3.py`). Focus on `ValuA_X0_I0+0` (the failure-relevant A operand);
all other lanes behave identically.

The raw LDS-load register `ValuA_T0_I0+0` (which is packed into
`ValuA_X0_I0+0`) is written by exactly three ds_reads, in order
(`_frag_trace3.py`):

| `shadow:line` | body | offset | parityA | role |
|---|---|---|---|---|
| `shadow:184` | ML_PREV | 0 | **1** | prefetch consumed by ML |
| `shadow:378` | ML      | 0 | **0** | prefetch consumed by NGL |
| `shadow:600` | NGL     | 0 | **1** | prefetch consumed by NLL |

Resulting real-math (`v_mfma_f32_16x16x32_bf16`) fragment consumption
(`_frag_trace2.py`):

| body | real-mfma | feeding pack | fragment `(side,parity,off)` |
|---|---|---|---|
| ML       | `shadow:246` | cvt `shadow:201` (ML_PREV) | **`('A', 1, 0)`** |
| **NGL**  | `shadow:453` | cvt `shadow:440` (NGL head-flush) | **`('A', 0, 0)`** |
| NLL      | `shadow:633` | cvt `shadow:620` (NLL) | **`('A', 1, 0)`** |

The NGL head-flush pack at `shadow:440`
(`v_cvt_pk_bf16_f32 v[ValuA_X0_I0+0], v[ValuA_T0_I0+0], v[ValuA_T0_I0+1]`)
consumes `ValuA_T0_I0+0`, whose latest writer at that point is the ML
second-wave ds_read `shadow:378` (parity **0**). ML's own math at `shadow:246`
consumes the pack `shadow:201`, whose source ds_read is `shadow:184`
(parity **1**).

**ML math parity 1 ≠ NGL math parity 0.** The adjacent bodies compute on
opposite LDS blocks. NGL consumes the fragment ML *prefetched* (`shadow:378`),
not the fragment ML *computed on* (`shadow:184`). This is the benign-drain case,
explicitly distinct from the repeat case.

(NLL also shows parity 1, same numeral as ML — but NLL is non-adjacent to ML,
with an intervening read swap `shadow:521` AND fresh LDS writes
[`s_xor LocalWriteAddr` at `shadow:179/373`] that overwrote the block. Parity-1
at ML and parity-1 at NLL are physically different k-data. Irrelevant to the
claim, which is specifically about NGL re-using ML.)

---

## §4 cms=0 / cms=1 baseline: do the real kernels advance per body?

Yes — identically.

- **cms1 capture** (`_frag_trace2_cms1.py`): ML real-mfma `cms1:247` →
  `('A', 1, 0)`; NGL real-mfma `cms1:443` → `('A', 0, 0)`; NLL `cms1:637` →
  `('A', 1, 0)`. One read swap per body at `cms1:117` (ML_PREV), `cms1:313`
  (ML), `cms1:509` (NGL); none in NLL. Byte-for-byte the same advance pattern
  as SHADOW.
- **cms0 emitted assembly**: `cms0_kernel.s:1957` and `:2285` are the two
  in-loop `v_xor LocalReadAddrA ... // swap Red Blk` toggles — real hardware
  double-buffer alternation, one per loop body. The parity model in §2/§3 is
  not a capture artifact; it is the emitted Red/Black swap.

The "correct behavior" baseline (real cms kernels) advances the fragment every
body, and SHADOW's NGL advances it the same way. There is no SHADOW-specific
repeat.

---

## §5 Verdict: user is WRONG

Decisive citation chain:

1. ML math operand `ValuA_X0_I0+0` is sourced from ds_read `shadow:184`,
   parity **1** (`_frag_trace2.py` → `last_writer=('cvt',201,'ML_PREV',('A',1,0))`).
2. NGL math operand `ValuA_X0_I0+0` is sourced from ds_read `shadow:378`,
   parity **0** (`_frag_trace2.py` → `last_writer=('cvt',440,'NGL',('A',0,0))`).
3. `shadow:184` and `shadow:378` are *different physical LDS reads* — different
   read-buffer parity (separated by the swap `shadow:283`) — i.e. different
   k-fragments (`_frag_trace3.py`).

Therefore the SHADOW NGL pack/math operates on **fresh** current-position data
(the fragment ML prefetched but never computed on). It does NOT repeat ML's
fragment. The benign cross-body software-pipelining (drain) reading is correct;
the user's "math is repeated" reading is not supported by the assembly.

I attempted the strongest pro-user framings — (a) same register names across
bodies, (b) the suspicious `mfma_index=-1` head-flush pack block at
`shadow:439-451`, (c) identical per-body offsets — and each dissolves once the
read-buffer parity is tracked: the head-flush at `shadow:440` is precisely what
consumes ML's *prefetch* (parity 0), making NGL's data fresh.

---

## §6 Is there a real bug? — No.

No duplicated math in any kernel. The behavior is identical in:
- the synthetic SHADOW reference capture (§3),
- the real CMS capture cms1 (§4),
- the real emitted cms0 assembly's Red/Black swaps (§4).

All three advance the LDS read fragment one block per loop body, so adjacent
bodies never recompute the same fragment. This is the intended PLR=1 pipeline.

**No P0 correctness bead warranted** (the trigger condition — a confirmed
math-duplication repeat — did not materialize). No new defect surfaced during
this investigation, so no bead is filed. (The pre-existing stale-citation bead
noted in the prior NGL memo §8 is unrelated to this question and already filed.)
