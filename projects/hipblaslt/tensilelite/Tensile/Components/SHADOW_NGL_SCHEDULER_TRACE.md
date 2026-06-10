# SHADOW (default) scheduler NGL trace — is the NGL ds_read/pack flip a bug?

Trace of the TensileLite DEFAULT (SHADOW) scheduler's handling of the
no-global-load (NGL) body on the BPG#11 TF32 4x4 TN canonical kernel, to settle
whether the default scheduler produces a wrong/surprising ordering in NGL or
whether the ordering is correct and only the per-byte validator model
over-reports it.

All claims below cite either a source file+line, the committed listings
(`Tensile/Components/h7lo_uvrl_artifacts/{cms,shadow}_capture_listing.txt`,
`kernel.s`, `compare_graphs_failures.txt`), or a logged value from the live
dataflow graph reproduced by the throwaway probes in `hxcx_artifacts/`
(`_ngl_decisive_probe.py`, `_ngl_packmfma_edge.py`, `_params.py`).

---

## SUMMARY (verdict)

**The default scheduler's NGL ordering is CORRECT. The ML/NGL asymmetry is
benign software-pipelining, not a bug. B1 still holds — but the prior memo's
*reasoning* for B1 was partly wrong and is corrected here.**

1. The "flip" the user saw is real but is NOT the ds_read moving. The SHADOW
   `+12:+15` ds_read sits at the **same mfma slot (mfma_index 43)** in both ML
   and NGL. What moves is the **pack chain**: in ML it lands in the trailing
   leftover-pack flush (`mfma_index = -1`, end of body); in NGL it is hoisted
   into the body head (`mfma_index 1-2`). Hence ML reads "ds_read before pack",
   NGL reads "pack before ds_read".
2. Cause: `_makeSubIterSchedule` splits local reads into *this-loop* reads
   (scheduled early) and *next-loop* reads (scheduled late, after the barrier)
   at `isBarrier = LoopIters - numItersPLR = 3` (`KernelWriter.py:1004,1365-1366,
   2010-2015`). The `+12:+15` read is the last sub-iteration's read
   (iteration 3 >= isBarrier) → it is a **next-loop prefetch**, scheduled late
   in EVERY body. NGL additionally carries the *previous* body's prefetched pack
   work into its head as a leftover-pack flush; that flush is the early NGL pack.
3. **Decisive ds_read-count test (the user's KEY test):** at the *NGL
   pack-mfma* level the two schedules DO consume different ds_reads — CMS NGL
   pack reads the **3rd (NGL) ds_read**, SHADOW NGL pack reads the **2nd (ML)
   ds_read** (different k-fragments). The user is right about that. BUT the
   validator does **not** flag the NGL pack-mfma edge: its edge-key is
   byte-identical on both sides and cancels in set-diff
   (`_ngl_packmfma_edge.py`). All 16 flagged failures have their consumer in the
   **NLL** body, and at that consumer **both** schedules root at the **3rd (NGL)
   ds_read**, the same physical LDS fragment. So the validator's flagged edges
   are genuine false-positives; B1 holds for the failures that actually fire.
4. SHADOW NGL reads live, un-clobbered data on hardware (§6). No scheduler bug.

---

## §1 The anomaly (ML agrees, NGL diverges)

Per-body within-body order of the `ValuA_X0_I0+12..15` writers, read from the
committed listings (columns `file_line | body | body_idx | mfma_index | seq |
render`; `body_idx` is the in-body scheduled position):

CMS (`cms_capture_listing.txt`) — ds_read FIRST in every body:
- ML : ds_read `:402` body_idx 155 (mfma37) → pack-mfma `:424` body_idx 177 (mfma43)
- NGL: ds_read `:584` body_idx 141 (mfma37) → pack-mfma `:604` body_idx 161 (mfma43)

SHADOW (`shadow_capture_listing.txt`):
- ML : ds_read `:388` body_idx 143 (mfma43) → pack-mfma `:410` body_idx 165 (mfma **-1**)  [ds_read first]
- NGL: pack-mfma `:477` body_idx 38 (mfma **1**) → ds_read `:610` body_idx 171 (mfma43)   [pack first — FLIPPED]

So both schedules agree in ML (ds_read first); they diverge only in NGL, where
SHADOW puts the pack first. This is the exact anomaly under investigation.

Crucial refinement (live-listing fact, `shadow_capture_listing.txt:388 vs 477`):
the SHADOW `+12:+15` **ds_read is at mfma_index 43 in BOTH ML and NGL** — it does
NOT move between bodies. What moves is the pack: ML pack at `mfma_index -1`
(trailing flush) vs NGL pack at `mfma_index 1` (leading flush). The flip is a
**pack-placement** difference, not a ds_read-placement difference.

---

## §2 NGL instruction generation (default path) — where + order before scheduling

SHADOW NGL/NLL are emitted by `_noLoadLoopBodyDefault`
(`KernelWriter.py:3454`), called from `noLoadLoopBody` (`:3447`) with the CMS
macro suppressed; the per-iteration interleave is done by `_makeSubIterSchedule`
(`:4041`, `capture_body_label = "n_gl" if isNGLL else "n_ll"` at `:4057`). SHADOW
ML/ML_PREV are emitted by `_loopBody` (`:4260`, `_makeSubIterSchedule` at `:5248`,
`capture_body_label="main_loop"`); ML_PREV is a clone of ML
(`main_loop_prev={0: clone_loop_body(main)}`, `:6249/:6440`).

Per-body parameters (`_params.py`): `LoopIters=4`, `numItersPLR=1`,
`isBarrier = LoopIters - numItersPLR = 3`, `numMfmaPerIter=12`,
`numMfmaForNextLoopLR=9`, `syncPlrMfmaIndex=38`, `usePLRPack=True`,
`ClusterLocalRead=1`.

Local-read generation in `_noLoadLoopBodyDefault` (`:3508-3778`):
- `isLastLoop = not isNGLL` (`:3510`); for NGL `isLastLoop=False`, for NLL `True`.
- `hasLiveLdsData = PrefetchGlobalRead and not isLastLoop` (`:3635-3636`): truthy
  for NGL, **False** for NLL. So NGL still issues next-loop-prefetch reads; NLL
  does not — confirmed: SHADOW NLL has **zero** `+12:+15` offset:192 ds_reads.
- `doReadA = (u < LoopIters/coalesce - numItersPLR) or (hasLiveLdsData and
  doNext)`, `doNext = uNext > localWriteEndIter` (`:3645,3657-3658`). The
  `+12:+15` read belongs to the last sub-iteration; it is issued as a next-loop
  read. `localReadDo` (`:3690`) emits the `ds_read_b128 ... offset:192`.

Before scheduling, the local-read code is just a flat module; the early-vs-late
placement is decided entirely in `_makeSubIterSchedule` (§3).

---

## §3 NGL scheduling (default SIA) — where the ds_read gets placed late

`_makeSubIterSchedule` (`KernelWriter.py:983`) builds the per-mfma slot schedule.
The load-bearing split:

- `isBarrier = LoopIters - numItersPLR` (`:1004`) = 3.
- `localReadItemsThisLoop = localReadItems if iteration < isBarrier else []`
  and `localReadItemsNextLoop = localReadItems if iteration >= isBarrier else []`
  (`:1365-1366`).
- **This-loop reads** are scheduled EARLY, interleaved ahead of the iteration's
  mfma work (`:1799-1890`, the `localReadItemsThisLoop.pop(0)` loop).
- **Next-loop reads** are scheduled LATE — the block headed
  `"scheduled local read for next loop / localReads for next loop should after
  barrier"` (`:2009-2015`), gated `if numItersPLR and iteration >= isBarrier`.

The `+12:+15` read is sub-iteration 3's read (3 >= isBarrier 3) → a **next-loop
read** → emitted in the post-barrier late block at `mfma_index 43`. This is why
the SHADOW ds_read sits at mfma43 (late within the body) in BOTH ML and NGL.

The pack hoist that makes NGL look flipped is the leftover-pack / pack-pre carry.
With `usePLRPack` active the pack of a next-loop read is deferred to the FOLLOWING
body and flushed at its head (`:2026-2083` schedule the next-loop local read /
pack-pre starting at `mfma 1`; `startPrePackIndex = min(numMfmaForNextLoopLR+1,
numMfmaPerIter-1)` at `:1663`). In ML the analogous flush stays in the trailing
`mfma_index=-1` block; in NGL it appears as the head block
(`shadow_capture_listing.txt:439-452`, then the `+12:+15` pack-mfma at `:477`).

LDS address double-buffering is one `v_xor_b32 v[vgprLocalReadAddrA], 0x10000`
per body (`kernel.s:2003`, `usePLR==1` guard; the base is computed once at
`kernel.s:549/552`). The xor↔ds_read interleave is identical on both schedules:
`shadow_capture_listing.txt:89/194 (ML_PREV), 283/388 (ML), 521/610 (NGL)` and
`cms_capture_listing.txt:117/206, 313/402, 509/584`. So the Nth ds_read targets
the same physical fragment on both sides.

---

## §4 The decisive ds_read-count test (CMS vs SHADOW)

ds_read inventory of byte `('v',14)` = `ValuA_X0_I0+15`, from the live graph
(`_ngl_decisive_probe.py`); both sides identical in count and body:

| | ds_read#0 | ds_read#1 | ds_read#2 (NGL) |
|---|---|---|---|
| CMS    | ML-1 pos 197 | ML pos 391 | **NGL pos 571** |
| SHADOW | ML-1 pos 225 | ML pos 417 | **NGL pos 597** |

### Which ds_read does each NGL *pack-mfma* consume?

From the latest-writer trace (`_uvrl_lifetime_trace.py`) and the edge probe
(`_ngl_packmfma_edge.py`):

- **CMS NGL pack-mfma** (`cms:604`, pos 591) ← **NGL ds_read** (`cms:584`,
  pos 571) = the **3rd** ds_read. (CMS keeps ds_read first within NGL.)
- **SHADOW NGL pack-mfma** (`shadow:477`, pos 464) ← **ML ds_read** (pos 417) =
  the **2nd** ds_read. (SHADOW's NGL ds_read at pos 597 is LATER than the pack,
  so the pack reads the previous body's read, routed through ML's pack chain.)

`_ngl_packmfma_edge.py` confirms both NGL-pack-mfma edges have the **identical**
edge-key
`(None, 0, ((v11),(v12),(v13),(v14)), (...), 'raw_intrawave', (0,1,2,3), 0, 2)`
but different producer NODES (CMS → NGL ds_read pos 571; SHADOW → ML ds_read pos
417). **Different ds_read, different k-fragment.** *This is exactly the
divergence the user predicted.* Because the edge-keys are byte-identical (iter-
blind), this edge **cancels in set-diff** and is NEVER reported as a failure.

### Which ds_read does the FLAGGED consumer (NLL pack-mfma) consume?

All 16 `EdgeRoutedDifferentlyFailure`s (`compare_graphs_failures.txt`) have their
subject consumer in the **NLL** body, NOT NGL. Live-graph proof
(`_ngl_decisive_probe.py`, every failure):

- SUBJ consumer = NLL pack-mfma (e.g. A-side `+12:+15` at pos 737, B-side at
  pos 702/704).
- SUBJ producer = NGL pack-cvt (e.g. pos 599).
- REF-side resolved consumer = same NLL pack-mfma (cons_pos 636 A-side).
- REF closest-prior writer of the byte = **NGL ds_read** (pos 597, `offset:192`,
  `LocalReadAddrA+0`), the **3rd** ds_read.

And CMS's NLL consumer roots, through its NGL pack chain
(`:752 ← :612 ← :604 ← :584`), at the **CMS NGL ds_read** (pos 571), also the
**3rd**. Both NLL consumers therefore root at their respective **3rd (NGL)
ds_read**. Per §3 the 3rd ds_read targets the same physical LDS fragment on both
sides (3 identical xor swaps precede it). **Same logical k-fragment, different
producer-node identity.** That identity difference (NGL pack-cvt vs NGL ds_read)
is the entire content of the 16 failures.

### Net answer to the KEY test

- At the **NGL pack-mfma** (NOT flagged): CMS=3rd, SHADOW=2nd ds_read → DIFFERENT
  fragments. Real, benign pipeline-depth shift; invisible to the validator.
- At the **NLL pack-mfma** (the 16 flagged failures): CMS and SHADOW BOTH root at
  the 3rd (NGL) ds_read → SAME fragment. B1's "same logical data" claim is true
  *for the flagged edges*.

The prior memo (`H7LO_UVRL_NORMATIVE_ORDERING.md`) reached the right verdict but
muddied two things: (a) it asserted SHADOW "each body's pack chain reads the
PRIOR body's pack output" as if uniform — that is true ONLY in NGL; in ML the
SHADOW pack reads the SAME body's ds_read (`shadow:410 ← shadow:388`,
body_idx 165 > 143). (b) It framed the divergence as a clean ds_read-ordering
swap; the real swap is pack placement (trailing flush vs head flush), the ds_read
slot being constant.

---

## §5 Why ML and NGL differ on SHADOW

They differ because the **leftover-pack flush** lands in a different place, not
because the scheduler treats the ds_read differently:

- In **ML** (`_loopBody`), the body is a steady-state loop iteration. The
  `+12:+15` next-loop read's pack work is emitted as the trailing
  `mfma_index=-1` flush at the END of the captured body
  (`shadow:407-414`, body_idx 162-169), which is *after* the ds_read at body_idx
  143. → ds_read-first in the ML capture.
- In **NGL** (`_noLoadLoopBodyDefault`), the body begins by flushing the
  *previous* (ML) body's prefetched pack work as a leading `mfma_index=-1`/early
  block (`shadow:439-452` head, then the `+12:+15` pack-mfma at body_idx 38),
  while NGL's OWN next-loop read still schedules late at mfma43 (body_idx 171). →
  pack-first in the NGL capture.

Both placements are produced by the same `isBarrier` split (`:1365-1366`) and the
same next-loop-late block (`:2010-2015`); the only difference is which body owns
the head-flush of a given next-loop read's pack. This is ordinary cross-body
software pipelining at PLR depth 1, encoded structurally — not a special-cased
NGL reorder and not a bug. CMS, by contrast, uses its `lra3`/`pack_a3` slot
arrays (`CustomSchedule/gfx950/_128x128x32_TF32.py:82-95`) to keep the read and
its pack inside the same body, so CMS shows ds_read-first uniformly.

---

## §6 Hardware-correctness check (does SHADOW NGL read live, un-clobbered data?)

Trace of `ValuA_X0_I0+12..15` across the SHADOW unrolled timeline
(`_uvrl_lifetime_trace.py`):

- ML ds_read (`shadow:388`) writes +12..15 (fp32 high fragment, 2nd read).
- ML trailing pack chain (`shadow:410-414`) reads that fp32 and overwrites
  +12..15 in place with packed bf16 (the pack-mfma at `:410` reads +12..15 from
  the ML ds_read; the cvts at `:411-414` finish the pack).
- NGL leading pack-mfma (`shadow:477`) reads +12..15 — its latest writer is the
  **ML pack chain output** (`:411-414`), i.e. the already-packed bf16, NOT raw
  fp32. Live and correct: the value it consumes is exactly what the previous
  body produced for this k-fragment.
- NGL late ds_read (`shadow:610`) overwrites +12..15 with the NEXT fragment;
  its consumer is the NLL pack chain (`shadow:648-651`). No +12..15 writer sits
  between `:610` and `:648` → no clobber; NLL reads the freshly-loaded value.
- NLL emits no next-loop +12..15 ds_read (`hasLiveLdsData=False` for the last
  loop), correctly terminating the prefetch chain.

Every read in the SHADOW chain has a well-defined, immediately-prior writer of
the same bytes; no read-before-write, no overwrite of still-live data. The
schedule is hardware-correct.

---

## §7 Verdict

**Scheduler CORRECT — B1 holds, now with a per-body mechanism.**

- The default scheduler does not do anything wrong in NGL. The ds_read sits at
  the same mfma slot as in ML; the visible flip is the leftover-pack flush moving
  from the trailing position (ML) to the body head (NGL), a direct consequence of
  the `isBarrier` this-loop/next-loop read split (`KernelWriter.py:1365-1366`,
  `:2010-2015`) plus cross-body pack-pre carry. Software pipelining, not a bug.
- The user's prediction is literally correct at the NGL-pack-mfma level (CMS
  consumes the 3rd ds_read, SHADOW the 2nd — different fragments) — but that edge
  is byte-key-identical and cancels in set-diff, so it never becomes a failure.
- The 16 reported failures are at the NLL pack-mfma, where both schedules root at
  the 3rd (NGL) ds_read = the same physical fragment. They are validator
  false-positives caused by the flat-register latest-writer model treating an
  NGL pack-cvt and the NGL ds_read as distinct producers of the rotating buffer.
- Therefore `uvrl` remains a validator-modeling fix (B1), not a scheduler fix
  (B2). No scheduler change is warranted.

Uncertainty remaining: none material to B1-vs-B2. The only nuance is cosmetic —
the prior memo over-generalized "pack reads prior body" to all bodies; corrected
in §4/§5. That does not change the verdict.

---

## §8 Debug logging added + revert confirmation

**No debug logging was added to any committed source file.** The entire trace was
done through existing public APIs (`build_dataflow_graph`, `compare_graphs`,
`KernelWriterAssembly._getKernelSource`, `_last_cms_capture`,
`_last_default_capture`) plus three throwaway probe scripts written into the
gitignored build-output dir `hxcx_artifacts/`:
`_ngl_decisive_probe.py`, `_ngl_packmfma_edge.py`, `_params.py`.

`git status` shows only untracked `hxcx_artifacts/` and `oplb_artifacts/`;
`git diff --stat` is empty. The committed artifacts under
`Tensile/Components/h7lo_uvrl_artifacts/` are untouched. **Net source change is
zero.** Nothing to revert.

## §9 Bead filed

None. Verdict is scheduler-correct (B1), so no P0 scheduler bug bead against
`r62g` is warranted. `uvrl` (validator-modeling fix) stands as already filed.
