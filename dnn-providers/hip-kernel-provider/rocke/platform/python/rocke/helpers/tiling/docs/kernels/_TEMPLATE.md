# Design Report: <kernel name>

- **Status:** <in-design | building | tuned | shipped>   **Owner / started:** <name> / <YYYY-MM-DD>
- **Kernel:** `<path to the kernel file / build fn>`

> A per-kernel design report, kept **only when the user opts in**. It is a **reproduction-complete** report:
> everything another agent needs to rebuild the kernel and its layouts bit-for-bit, plus the full performance
> record. Sections: **1 Introduction · 2 Theory · 3 Experimental setup (all FIXED constants + the sweep space) ·
> 4 Runs & results · 5 Data & kernel description of the WINNER only · 6 Appendices** (per-iteration/verbose
> captures). Fill every section; mark N/A explicitly. (Internal artifact — follow the repo compliance policy; do
> not publish. Units on EVERY number. Conflict numbers need `bank-conflict` validation before blessed.)
>
> **BE VISUALLY EXPRESSIVE — visual > verbose.** Prefer **tables, pictograms/ASCII diagrams, and images** over
> prose; put constants in tables, the layout in a pictogram, the pipeline in a stage table, results in tables,
> and link every `/layout-viz` render. Keep prose to short captions. Wordy paragraphs are the fallback, not the
> default.
>
> **Kernel folder layout (self-contained under `kernels/<kernel>/`):** kernel code in the folder
> (`__init__.py` or module) · **`docs/`** — this report (`design_report.md`) + **`docs/viz/`** renders ·
> **`tmp/`** — throwaway scripts/verbose captures/scratch code. Only rocke-CORE changes touch the rocke source
> tree; everything kernel-specific stays in this folder. On completion, run a **cleanup task** — decide which
> docs/code to keep, move, or summarize, and clear `tmp/`.

---

## 1. Introduction

- **Goal:** <correctness only | peak TFLOPS | specific target | learning>.
- **Problem:** <algorithm, e.g. GEMM C = A·B>; dtypes input <..> → accumulate <..> → output <..>.
- **What & why:** <one paragraph — what this kernel is, why it was designed this way, headline result>.

## 2. Theory (the design rationale — why it works)

- **Layout/pipeline reasoning:** <the design story — majors → coalescing, soundness, interleaving, the
  transform chain and where each transpose lives, the binding-stage argument>.
- **Expert reasoning (verbatim):** Architect (pipeline), MMA (operand/C layouts, soundness, C-shuffle,
  symmetry), LDS (bank exposure, budget, binding stage). Include their **tables verbatim**.
- **SOT references:** <`tiling_interleaving_design.md` §.., `mma_is_machinery.md`, `lds_banks.md`, …>.
- **Alternatives ruled out & why:** <..>.

## 3. Experimental setup (ALL fixed constants + the sweep space)

**Constants — the FIXED spec (prefer a table):**

| aspect | value |
|--------|-------|
| goal | <correctness \| peak TFLOPS \| target \| learning> |
| algorithm / dtypes | GEMM `C=A·B`; f16 in → f32 acc → <..> out |
| A=(M×K) / B=(K×N) / C=(M×N) | strides + stride-1 axis per tensor (a **layout pictogram** here is great) |
| atom / wave size | `<MxNxK>` / `<64\|32>` |
| pipeline | <interleaved · double-buffered · …> |
| arch / GPU | <gfx90a/gfx942 · MI210/MI300> |

**Environment & reproduction:** commit/branch; exact build+run command; correctness method (integer inputs vs
numpy golden, `max_abs_diff==0.0`); rocprof caveats (ROCm-7.14 container). *(a compact table works well.)*

**Pipeline (FIXED structure — same stage chain for every swept config; a stage table):**

  | # | stage (verb) | granularity | register/LDS state (vector axis) | transition → next (tier) |
  |---|--------------|-------------|----------------------------------|--------------------------|
  | 1 | … | | | |

**Sweep space (a table — axis × values, then the LDS-budget screen + valid-config count):**

  | axis | values swept |
  |------|--------------|
  | macro / waves / tile_k / flags | … |

**Flags / knobs (range swept + peak value in bold):**

  | knob | range | peak | effect |
  |------|-------|------|--------|
  | ab_swap / mac_prio / lds_swizzle / … | | | |

## 4. Runs & results

> Order: correctness → iteration journal → performance progression → **winning-kernel stats only**. Keep this
> section COMPACT (tables) — per-iteration verbose captures (rocprof, resource dumps) live in Appendix A, one
> section per iteration named by its iteration # + config.

- **Correctness:** `max_abs_diff = <..>` table (<PASS / skipped-no-GPU>), shapes <..>.
- **Iteration journal:** one bullet per iteration — `<YYYY-MM-DD> iter <n> (<config>):` changed <..> →
  <perf delta, units> → **finding:** <..>. *(point to Appendix A.<n> for the verbose data.)*
- **Performance progression** (compact table; one row per iteration; units on every number):

  | iter | date | config (macro / waves / Kt / flags) | wall-time (ms) | TFLOPS | notes |
  |------|------|-------------------------------------|----------------|--------|-------|
  | 0 | | | | | baseline |

- **Winning kernel — stats** (the peak config only): perf (TFLOPS / ms, bit-exact); **rocprof** (MfmaUtil,
  SQ_WAIT_INST_LDS, SQ_LDS_BANK_CONFLICT, conflicts/access, HBM); **kernel resources** (occupancy, VGPR/lane,
  SGPR, scratch, LDS alloc, ASM lines). ONLY the winner here — losing/intermediate runs go to Appendix A.

## 5. Data & kernel description — WINNER only (reproducible to the encoding)

Describe the **winning config's** kernel (you only know the winner after §4). Reproducible to the encoding:

- **Tiling hierarchy:** macro <Mt×Nt> ⊃ wave grid <waves_m × waves_n> (wave <..>) ⊃ atom <MxNxK>; buffering.
- **Ordered tensors:** A=(M×K) s=(..) → <..>-contig; B=(K×N) s=(..) → <..>-contig; C=(M×N) s=(..) → <..>-contig.
- **EXACT static distributions** — for each of A, B, C (+ staging/LDS descs), the full `WarpDistributionEncoding`
  (a table): **Rs** (`replication_lengths`), **Hs** (`hierarchical_lengths`), **Ps** (`lane_to_rh_*`),
  **Ys** (`register_to_rh_*`); flag any Kt-/tile-dependent rows. Plus the **`make_tile_desc` recipe**,
  the **ownership cross-check** (`lane(M,N)=…`, verified via `RegisterMapper`), and the **C-shuffle**
  (`interleave_idx<...>` + issue order + tier).

## 6. Appendices

### Appendix A — Per-iteration detailed captures
One subsection per iteration (or per notable run), **named by iteration # + the config description** from the
progression table. Each holds that run's verbose data — the full **rocprof** counter table + reading, the
**kernel resource** table, and any other verbose capture. (Conflict numbers need `bank-conflict` validation
before blessed; report `conflicts/access` as measured.)

#### A.<n> — Iteration <n>: `<config>` — rocprof + resources
| counter / metric | value | ... |
|---|---|---|

### Appendix B — Visualizations
The `/layout-viz` renders — A/B/C tee, C-shuffle before/after, **coalescing (two images/major)**, and the
**stepwise pipeline dataflow**. List every image path (under `docs/viz/`) + a one-line caption.

### Appendix C — API-gap proposals & open items
- **API-gap proposals raised:** link `../../../docs/api_proposals/<name>.md`.
- **Open items / next:** <what to try next; unresolved questions>.
- **Other:** <anything else needed to reproduce or understand the kernel>.
