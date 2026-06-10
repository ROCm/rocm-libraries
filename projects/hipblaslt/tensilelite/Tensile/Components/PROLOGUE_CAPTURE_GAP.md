# Prologue Capture Gap: global-read / local-read prefetch absent from the PRO body

## SUMMARY (plain verdict)

The TensileLite SHADOW/CMS capture pipeline does **not** capture the prologue's
global-read prefetch (`buffer_load ... lds`) or local-read prefetch (`ds_read`)
instructions. The captured PRO body contains **only the PACK stream** (`v_cvt_pk_*`
+ pack-`v_mfma`). This is **by design, not an oversight** — both the source comment
(`KernelWriter.py:5849-5852`) and the `build_prologue_capture` docstring
(`ScheduleCapture.py:2370-2374`) explicitly state that LR/LW/GR prologue capture is
out of scope, tracked as bead `rocm-libraries-6jbr`. The reason given: capturing
those producers without also capturing the prologue's `s_waitcnt` + `s_barrier`
would surface uncovered prologue→mainloop edges in `validate_edge_wait_coverage`.

**Does it invalidate the 16-failure / uvrl / B1 / synthetic-reference conclusions?
No.** Those conclusions are entirely about NGL/NLL **loop-body** local reads
(producers `LRA3`/`LRB3` placed in the NGL body, consumed by NGL `MFMA`), which **are
captured** (NGL body has 16 `ds_read`). The 16-edge divergence is a scheduler
reordering of stream position, not a capture-layer gap. Both CMS and SHADOW traverse
the **same** prologue capture path, so the CMS-vs-SHADOW comparison stays
**apples-to-apples**. What is genuinely incomplete is the **absolute** dataflow
graph: it has zero prologue global-read / local-read-prefetch nodes, so every
prologue→mainloop RAW edge from a warmup load is invisible.

**Severity: P1** — incomplete-but-apples-to-apples. The comparison conclusions stand;
the absolute dataflow graph is incomplete and must be completed. Bead
`rocm-libraries-6jbr` already existed at P3; raised to P1 with the empirical findings
below.

The gap is **prologue-only**. Loop-body capture (ML/ML_PREV/NGL/NLL) is complete:
raw `.s` loop-body GR/LR counts match the captured-body counts exactly.

---

## §1 The confirmed gap (per-body counts)

Fixture: BPG#11 TF32 4x4 TN canonical kernel, cms=1 build.

### Raw prologue region of the assembly
`Tensile/Components/kernel_comparison/cms1_kernel.s`, region lines `1 ..
label_openLoopL:1868`:

| instruction class                  | count | line range        |
|------------------------------------|------:|-------------------|
| `buffer_load_dwordx4 ... lds` (GR prefetch) |    16 | first 1578, last 1795 |
| `ds_read` (LR prefetch)            |     9 | first 1575, last 1820 |

### Captured PRO body
`Tensile/Components/kernel_comparison/cms1_capture_listing.txt`, header
`### body PRO: 42 instructions` (line 6), body rows lines 7-48:

| instruction class | count |
|-------------------|------:|
| `buffer_load`     |     0 |
| `ds_read`         |     0 |
| total             |    42 |

All 42 PRO instructions are `v_cvt_pk_bf16_f32` (CVT/pack) and
`v_mfma_f32_4x4x4_16b_bf16` (pack-mfma), plus 2 `s_nop` pads (rows at listing lines
19 and 40). The prologue's global-read and local-read-prefetch instructions are
absent.

### Per-body captured counts (all five bodies)
From `cms1_capture_listing.txt`:

| body    | total | buffer_load | ds_read |
|---------|------:|------------:|--------:|
| PRO     |    42 |           0 |       0 |
| ML_PREV |   194 |           8 |      16 |
| ML      |   194 |           8 |      16 |
| NGL     |   174 |           0 |      16 |
| NLL     |   146 |           0 |       8 |

PRO is the only body missing its GR/LR.

---

## §2 Root cause: how the prologue is captured, what the snapshot includes/excludes

### Where the prologue is emitted
The prologue spans from the `setupNewTile` call (`KernelWriter.py:5664`) through the
prefetch-local block (`KernelWriter.py:5826` onward), ending at the prologue
checkpoint (`KernelWriter.py:6011-6025`).

Three classes of prologue producer are emitted into the physical `module`:

1. **Global-read prefetch** — `globalReadDo(...)` at `KernelWriter.py:5768` and
   `:5789`, plus the PGR≥2 block `KernelWriter.py:5736-5817`. These emit the
   `buffer_load ... lds` instructions. They are `module.add(...)`'d directly into the
   physical stream.

2. **Local-read prefetch** — `localReadDo(...)` returns `localReadCodeA` /
   `localReadCodeB` at `KernelWriter.py:5866` and `:5905`; these are
   `module.add(localReadCodeA/B)`'d at `:5867` and `:5906`. These emit the prologue
   `ds_read` instructions. Added directly to the physical stream.

3. **Pack prefetch** — `packPreA/packCodeA/packPreB/packCodeB` accumulate into
   `packPrePrefetchA` / `packPrePrefetchB`, then are interleaved by
   `_interleavePackAB(...)` into `packPrePrefetchItems` at `KernelWriter.py:5956`.

### What the snapshot actually receives
`prologue_interleaved_items` (the sole input to the prologue capture) is populated at
**exactly one site**: `KernelWriter.py:5966`:

```python
self._capture_context.prologue_interleaved_items.extend(_prologue_snap)
```

where `_prologue_snap` (`KernelWriter.py:5962-5965`) is built **only** from
`packPrePrefetchItems` — the post-interleave **PACK** stream. The `id->category`
dict that tags it (`KernelWriter.py:5948-5955`) maps only `packPrePrefetchA` /
`packPrePrefetchB` leaves to `PackA{plrIdx}` / `PackB{plrIdx}`; everything else
becomes `"SNOP"`.

`grep prologue_interleaved_items KernelWriter.py` returns only the comment lines and
the single `.extend` at `:5966` and the read at `:6024`. **No other producer class is
ever threaded into the snapshot.** The global-read code (class 1) and the local-read
code (class 2) live in different modules (`globalReadDo` output, `localReadCodeA/B`)
that are `module.add`'d to the physical stream but never enter
`prologue_interleaved_items`.

### What `build_prologue_capture` can build
`build_prologue_capture(*, prologue_interleaved_items=None)`
(`ScheduleCapture.py:2340`) iterates exactly that one list (`ScheduleCapture.py:2385`)
and appends each non-`TextBlock`/non-`Label` leaf as a `SLOT_KIND_PRE_LOOP` node.
Its signature takes **no** GR or LR input. It can therefore only ever produce a PACK
prologue. The PRO body's 42 entries are precisely the pack stream and the SNOP pads.

---

## §3 By-design or oversight?

**By design.** Two independent in-tree statements say so explicitly:

- `KernelWriter.py:5849-5852`:
  > NOTE: LR/LW/GR prologue capture is deferred — tracked as rocm-libraries-6jbr
  > (requires concurrent capture of the prologue's s_waitcnt + s_barrier to keep
  > validate_edge_wait_coverage clean).

- `ScheduleCapture.py:2370-2374`:
  > NOTE: LR/LW/GR prologue capture is intentionally out of scope for
  > rocm-libraries-oram Phase 2. Capturing those producers without also capturing
  > the prologue's s_waitcnt + s_barrier instructions surfaces uncovered
  > prologue->mainloop edges in `validate_edge_wait_coverage`. Tracked as a
  > follow-up: rocm-libraries-6jbr.

The recent hxcx fix (`KernelWriter.py:5942-5966`, "rocm-libraries-oram Phase 2
(hxcx)") changed the prologue capture to snapshot the **post-interleave** PACK stream
so the prefetch's pack leaves enter `PackCodeAAllIters` instead of landing in
UNKNOWN. That fix was scoped to PACK only. The bead `rocm-libraries-6jbr` already
records the deferral and its acceptance criteria (capture LWA/LWB, prefetch LRA/LRB,
and the prologue `s_waitcnt`/`s_barrier` with stream positions, then prove
`validate_edge_wait_coverage` stays clean).

So: the prologue capture was only ever **designed** to capture the PACK prologue, by
explicit decision. It is not a silent miss.

---

## §4 Impact on the validator dataflow graph (absolute incompleteness)

Because the PRO body has zero `buffer_load` and zero `ds_read` nodes:

- The dataflow graph has **no prologue global-read node** (no LDS-fill warmup load)
  and **no prologue local-read-prefetch node** (no first-fragment `ds_read`).
- Any RAW edge rooted at a prologue prefetch is **invisible**:
  - prologue `ds_read` (first-fragment LR) → first mainloop pack/MFMA consumer:
    absent.
  - prologue `buffer_load ... lds` (global-read filling LDS) → mainloop `ds_read`
    via LDS: absent (and LDS-mediated edges are not modeled here anyway).
- The per-byte `latest_writer` / `byte_key_writers` map
  (`CMSValidator.py:2106`, `:2133`) never sees a prologue GR/LR write. For a register
  whose first writer is a prologue first-fragment local read, the validator's
  "closest-prior writer" walk will resolve to the **next** writer it does see (a
  mainloop producer), not the true prologue root.

This is a genuine **absolute** incompleteness of the dataflow graph. It is the
problem `rocm-libraries-6jbr` is filed to fix.

---

## §5 Impact on the 16-failure / uvrl / B1 / synthetic-reference conclusions

**These conclusions stand. The gap does not invalidate them.**

### The 16 EdgeRoutedDifferently failures are NGL-body, not prologue
`n7og_PROBE_REPORT.md:300`:
> The 16 missing-in-SHADOW edges in NGL are the same mechanism mirrored: there CMS
> happens to place LRA3/LRB3 producers BEFORE the LRA3→MFMA / LRB3→MFMA consumers,
> so CMS emits those edges and SHADOW does not.

`n7og_PROBE_REPORT.md:300` (continued):
> This is a SCHEDULER REORDERING DIVERGENCE measured by `compare_graphs`, not a
> CAPTURE-LAYER granularity divergence.

`UNROLLED_VALIDATION_ANSWERS.md:127`:
> the 16 NGL "missing in SHADOW" edges all have producer category `LRA3`/`LRB3` and
> consumer category `MFMA` (8 each per operand).

The producers (`LRA3`/`LRB3` — local-read sub-iter 3) and consumers (`MFMA`) both
live in the **NGL body**, which **is captured** (NGL `ds_read = 16` in §1). The
divergence is purely which stream-position the `latest_writer` walk reaches first on
the CMS vs SHADOW side — not whether the instruction was captured. The missing
**prologue** first-fragment LR/GR is upstream of and disjoint from these NGL/NLL
consumers, so it cannot change the 16-edge result.

### uvrl / byte_key_writers analysis is unaffected
The uvrl (unrolled-validation) reverse-index work
(`UNROLLED_VALIDATION_ANSWERS.md:289-298`, `CMSValidator.py:2106-2133`) builds
`byte_key_writers` from the **captured** nodes. The 16/192 edge residuals it reasons
about are NGL/NLL-body byte_keys (NLL extras: producers `PackA0`/`PackB0`, consumers
`MFMA`, body pair `(NLL, NLL)` — `n7og_PROBE_REPORT.md:366-368`). None of these roots
in the prologue. The "closest-prior writer" resolution for the
EdgeRoutedDifferently/SpuriousEdge classifier operates over the same captured set on
both graphs.

The only way the prologue gap could move a `byte_key_writers` root is if the *true*
closest-prior writer of an NGL/NLL consumer's register were a **prologue** GR/LR. For
the registers involved in the 16/192 residuals the closest-prior writer is an
NGL/NLL pack or LR that **is** captured (the divergence is its stream-position, and
both graphs contain it). So the missing prologue root does not relocate these edges.

### B1 / synthetic-reference conclusions
The B1 / single-source-of-truth refactor (`SVB1_IMPLEMENTATION.md`) governs the
category schema shared by both public factories; it is structural and does not depend
on prologue read capture. The synthetic-reference reasoning compares CMS against a
SHADOW reference that traverses the **same** prologue capture path — both omit the
same prologue instructions — so the reference and subject graphs are missing the
identical prologue nodes. The comparison remains apples-to-apples.

### What is NOT safe to claim
- It is **not** safe to claim the validator's dataflow graph is a complete model of
  the kernel's RAW dependencies. It omits all prologue→mainloop GR/LR edges.
- It is **not** safe to use this graph for any absolute (single-side) dataflow
  property that depends on prologue warmup loads (e.g., "every mainloop consumer's
  first-fragment producer is present"). Those producers are missing.
- It **is** safe to use the graph for CMS-vs-SHADOW differential comparison, because
  the omission is symmetric across both sides.

---

## §6 Loop-body completeness confirmation (gap is prologue-only)

Raw `.s` first mainloop-body iteration, `cms1_kernel.s` lines
`label_LoopBeginL:1874 .. label_LoopBeginL_0:2238`:

| instruction class            | raw count | captured ML count |
|------------------------------|----------:|------------------:|
| `buffer_load` (any)          |         8 |                 8 |
| `buffer_load ... lds`        |         8 |                 8 |
| `ds_read`                    |        16 |                16 |

The raw loop body's GR/LR counts match the captured ML body exactly (8 + 16). The
NGL and NLL bodies likewise carry their `ds_read` (16 and 8). **Loop-body capture is
complete; the gap is confined to the prologue.**

The mechanism difference: loop-body GR/LR are routed through
`makeSchedule` / `perIterGlobalRead[]` / `perIterLocalReads` and the
`_loopBody` capture path (`KernelWriter.py:763`, `:999`, the `LoopBodyCaptureBuilder`
sites at `:4195` and `:6044`), which captures the actual `buffer_load`/`ds_read`
leaves. The prologue path snapshots only `packPrePrefetchItems` and so captures only
the pack stream.

---

## §7 Verdict + bead

**Verdict:** The prologue GR/LR-prefetch is uncaptured **by design** — the prologue
capture snapshots only the post-interleave PACK stream (`KernelWriter.py:5966`,
fed solely from `packPrePrefetchItems`), and `build_prologue_capture`
(`ScheduleCapture.py:2340`) takes no GR/LR input. The global-read (`globalReadDo`,
`KernelWriter.py:5768`/`:5789`) and local-read-prefetch (`localReadCodeA/B`,
`KernelWriter.py:5867`/`:5906`) leaves are `module.add`'d to the physical stream but
never threaded into the snapshot.

This produces an **absolute incompleteness** in the dataflow graph (no
prologue→mainloop GR/LR edges) but does **not** invalidate the CMS-vs-SHADOW
comparison: both sides traverse the same prologue path and omit the identical
instructions. The 16-failure / uvrl / B1 / synthetic-reference conclusions are about
NGL/NLL-body reads that **are** captured, and so are unaffected. The gap is
**prologue-only**; loop-body capture is complete.

**Bead:** `rocm-libraries-6jbr` (already existed, P3 → **raised to P1**). It carries
the acceptance criteria to capture the prologue LWA/LWB, prefetch LRA/LRB, and the
prologue `s_waitcnt`/`s_barrier` (with stream positions), and to prove
`validate_edge_wait_coverage` stays clean for the new prologue-rooted edges. The
empirical findings above were added as a comment on the bead.

**Severity reasoning (P1, not P0):** Per the rubric, P0 is reserved for a gap that
*invalidates the validator comparison*. This gap does not — the comparison is
apples-to-apples. It is the P1 case: an incomplete-but-apples-to-apples prologue gap
that should be fixed (the absolute graph is incomplete) but does not change the
existing differential conclusions.
