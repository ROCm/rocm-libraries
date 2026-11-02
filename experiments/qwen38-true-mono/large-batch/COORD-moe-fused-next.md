# COORD: adopting `combine=fused_next` from the large-batch MoE (`moe_lb2`) in the next stage

From WP-L2d (large-batch MoE) to WP-L2b / WP-L3 (large-batch HC mixers and layers). Ratios and evidence are in
`P-L4b.md` sections 7.2 and 7.5.

## What it is

With `fnext=1` (and `cmb=flow`), the MoE kernel stops after the experts. It writes no `y` and runs no combine pass.
The next stage, the consumer of the MoE output, reads the slot partials directly and forms

    y[t, c] = sum over slots s = 0..10 (in order) of  w[t, s] * ( sum over segments j < nseg[t, s] (in order) of P[t, s, j, c] )

then uses `y` (residual add, HC mix, ...) without a bf16 round trip.

- **Timing.** The modelled fused chain (MoE + consumer summing partials) is 0.92-0.97 of the unfused chain (MoE with
  combine + consumer reading `y`). Against FlyTape's MoE plus the same consumer read, it is 0.79-0.87.
- **Reference implementation.** `_Moe2.combine()` in `large_batch/moe_lb2.py` (also built as the `stage=cons_part`
  consumer kernel, which adds the residual and writes `out`). Copy its read and sum order exactly.

## Buffers and layout (per layer; all addresses are kernel parameters)

| buffer | layout | written by | notes |
|---|---|---|---|
| `part` | `[bucket tokens][SLOTS=11][MAXSEG=6][H=2560]`; f32, or bf16 under `pdt=bf16` (row stride H x 2 bytes) | the expert segments (sc1 stores) | one buffer, reused by every layer |
| `desc` | `[bucket][DSTR=16]` i32; word `s < 11` = `nseg[t, s]` (1..6) | the segment with `jseg == 0` of each (token, slot) | the number of segment rows to sum |
| `pr` (route pairs) | `[layer][bucket][PRS=16][2]` i32: (expert id \| tag << 16, weight f32 bits); slot 10 = (512, sigmoid shared gate) | the router epilogue (`entry=route`) or the in-kernel route (`entry=logits/router`, `pr_lg`) | `w[t, s]` = the weight word |
| `cnt` | i32 words; `W_TOKC + parity * 128 + t` (W_TOKC = 64) | `+ 12 / nseg` per finished segment | the per-token flow counter |

- A slot's segment rows are `P[t, s, j, :]` for `j < nseg[t, s]`. Rows `j >= nseg` hold stale data and must not be
  read; mask them (the reference uses an OOB offset).
- Tokens `t >= rows` have no partials and no counter.

## Ordering and acquire rules

1. **Wait for the token.** Poll `cnt[W_TOKC + parity * 128 + t]` until it reaches `11 * 12 = 132` (all 11 slots,
   each slot contributing 12 in total over its segments).
   - Inside one persistent kernel, `parity = (ep + layer) & 1`, where `ep` is the epoch word `cnt[16]` read at kernel
     start. A separate consumer launch after a MoE launch uses the previous launch's parity (`ep` has advanced by the
     MoE launch's layer count): see `flow_par()` with `stage != moe`.
   - Poll from one lane per unit with the agent-scope poll (`gpoll`, `sc0 sc1` loads). A longer pause per miss
     (`s_sleep 16` to `s_sleep 64`) was a measured win: the spinning competes with the expert streams.
2. **Acquire.** After the counter reaches 132, read `desc`, `part` (and `pr` under `entry=logits/router`) with
   **sc1** loads. A plain load can return a line this XCD's L2 cached before the producer wrote it (observed in 1 of
   20 cells with a pretouch twin, `flowsc`).
3. **Release (producer side, already in `moe_lb2`).** Each segment does `s_waitcnt vmcnt(0)` in every wave, then a
   CTA barrier, then the counter add. With non-sc1 partial stores (`pst=nt/def`), a `buffer_wbl2` comes before the add.
4. **Reset.** Exactly one reader per token zeroes the token's counter of the *next* layer's parity
   (`W_TOKC + (1 - parity) * 128 + t`, sc1 store). The reference uses the column-block-0 unit. Never zero the
   current parity: other readers of the same token may still be polling it.
5. **Reuse.** `part` / `desc` are rewritten by the next layer's experts, so every consumer read of layer `l` must
   complete before any CTA starts layer `l + 1`'s experts. The layer-boundary grid barrier (`gsync`) provides that;
   if the consumer moves past it, double-buffer `part` / `desc` by layer parity.
6. **Order of the sums.** Slots 0..10 in order; inside a slot, segments 0..nseg-1 in order, f32; the weight is
   applied once per slot, after its segment sum. This order makes the result bit for bit the unfused combine's f32
   sum (before its bf16 rounding of `y`). Keep it: the numerics gates compare bitwise.

## Numerics gates the consumer must pass (see `pl4b_bench.chain_check`)

- **Unfused consumer of `y`:** out == bf16(res + y) bit for bit.
- **Fused consumer:** every element within ulp(y) + ulp(out) of res + y, with `y` from the same MoE build with its
  combine. The fused sum skips `y`'s bf16 rounding, so it is not bitwise equal to the unfused chain.
- **Must-fail mutant `flowrel`** (`mutant=flowrel` with `cspread=1`). It moves the counter bump to the segment start,
  so readers see 132 before any partial exists. It is caught in 1 launch at every cell, r8 included, because the
  readers then sum poisoned partials. A consumer port must keep this mutant failing; it proves the reader really
  waits on the counter.
- **Must-fail mutant `flow0`** (the reader does not wait at all). Also deterministic.
- `bar` / `flow` (a one-slot-early threshold) are races, reported only.

## Knobs that change what the consumer reads

| knob | effect on the consumer |
|---|---|
| `pdt=bf16` | `part` rows are bf16 (2 bytes, row stride H x 2); widen to f32 before summing |
| `pc`, `pcal`, deal knobs | change `nseg` (1..6) per (token, slot); always read it from `desc` |
| `shs` | shared-expert passes of 4 / 8 tokens; slot 10 is unchanged |
| `entry=logits/router` | the pairs live in the in-kernel `pr_lg` area, written with sc1; the consumer reads them after the counter (they precede the experts) |

Do not edit `moe_lb2.py` for the port. Ask WP-L2d for any new producer-side hook (for example a second partial
buffer by layer parity).
