# Rust track: integration plan for the direct code-object backend (for the D0 / WP1 owners, W2)

Ratios only (direct time / COMGR time, paired in one process, shuffled interleaved rounds, win rule of DESIGN 9.1 on
both rocke nodes). Evidence: `LEDGER.md` (this directory); absolute numbers in
`hackathon-prep/private-perf/qwen38-true-mono/rust-track/`. Nothing here changed git state anywhere.

## 1. Recommended default per kernel family

| family | recommendation | lever set (`attrs["direct_levers"]`) | evidence (n01-09 / n07-17) |
|---|---|---|---|
| `stage_b3` (all variants, tuned and mono) | **direct on** | B3 set (below) | r1 0.970-0.975 / 0.967-0.973; r4s 0.979-0.981 / 0.976-0.979; r2s 0.962 / 0.960, r3s 0.966 / 0.965, r4 0.986 / 0.990, r4g2 0.989 / 0.988, mono r1 0.992 / 0.984, mono r4s 0.950 / 0.948 |
| `moe_fused` rows 1 | **direct on** | MoE r1 set | 0.970-0.976 / 0.976 |
| `moe_fused` rows 4 | direct neutral: keep COMGR until a win | MoE r4 set | 0.995-1.005 / 1.008-1.009 |
| `mx_decode_gemv` head (fused / logits, rows 4 / 16) | keep COMGR | (F sets) | 1.003-1.012 (neutral to small loss) |
| toy chain (R2 toy) | **direct on** for d8w0; neutral d4w8l2 | B3 set | d8w0 0.988 / 0.989 WIN; d4w8l2 0.988 / 0.994 neutral; no levers 1.02-1.10 LOSS |
| `stage_q2`, framework layer kernels | keep COMGR (no paired verdict yet) | - | compile directly |
| probes P2 / P4 / P5, CCL collectives | keep COMGR; direct available | `acqfence` | compile directly (strict); P4 / P5 / CCL suites pass with the direct objects (P5's controls need `acqfence`) |

Lever sets (comma or `+` joined names of `ROCKE_DIRECT_LEVERS`):
- B3: `dppcombine, dppcombinef, dppmirror, ldspairs, ldsquads, litreuse, litshare, minmax, nopause, selxor,
  storewait, subadd, vmemfirst, wgfence, xorcmp, scalesep, dsrebase, loadorder` (the r4s G1 set; best or within noise on
  every B3 build measured).
- MoE r1: `dppmirror, ldsquads, litreuse, litshare, minmax, nofence, selxor, subadd, wgfence, xorcmp, scalesep, dsrebase,
  loadorder` (+ `bufoffany`, neutral).
- MoE r4 (if direct): all 20 levers, or the r1 set + `bufoffany`.
- Never for the head: `nofence` (+0.9 to +4.3%), `loadorder` with the F sets (+1.2%).

## 2. Mechanism (already in the overlay)

- `KernelDef.attrs["direct_levers"]` (list or `,` / `+` string): `helpers/compile.py` joins it to
  `ROCKE_DIRECT_LEVERS` for that compile only and strips it before serialization (the COMGR path never sees it).
  Test: `platform/tests/runtime/test_direct_targets.py::test_a_kernels_direct_levers_apply_to_its_compile_only`.
- The backend per family: the existing `request_code_object_backend(kernel, "direct")` mark (or
  `code_object_backend="direct"` at the call site). A refusal falls back to COMGR (`artifact.direct_fallback`);
  `ROCKE_DIRECT_STRICT=1` turns a refusal into an error for validation runs.
- In the true mono (one kernel, all stages) a lever applies to the whole kernel. The levers that differ by stage are
  `nofence` (MoE wants it, the head does not) and `loadorder` / `vmemfirst`; the fix is per-op scoping (a stage
  emitter marking its `tile.sched_barrier`s soft instead of the global `nofence`), not yet implemented.

## 3. What to port (files of this track, in `copy-rocm-lib-rust`)

Crate `platform/rust/rocke-codeobj`:
- `isa/gfx950.isa` (+ `src/target/gfx9/gen_isa.rs`, regenerate: `PYTHONPATH=python python tools/isa_tables.py
  generate --isa gfx950` from `platform/`): new rows (scaled converts, block-scaled MFMA with two scale operands,
  maximum3, DPP old-tied / combined forms, f64 converts / rsq, bcnt, ff1, memtime, raw DS ops, read2, v_bitop3,
  v_bfi, s_buffer_load, s_dcache_inv, WaitRelease pseudo), modsets (abs, glc on loads / stores / s_load).
- `src/rir/`: `ast.rs`, `parse.rs`, `validate.rs`, `analysis.rs`, `opt.rs`, `isel.rs`, `isel_asm.rs` (new: the
  inline-asm assembler), `post.rs`, `mir.rs` (`Uses` holds 5), `mod.rs` (levers, `with_levers` for tests, `wpecap`).
- `src/target/gfx9/`: `q38.rs` (tests), `mfma.rs`, `asm.rs`, `dpp.rs`, `lds_pairs.rs` / `minmax.rs` / `bitop3.rs`
  (new), `isel_hooks.rs`, `waitcnt.rs`, `legalize.rs`, `lits.rs`, `mod.rs`; `src/target/mod.rs`; `src/hsaco_writer.rs`.
- `examples/isa_xcheck.rs`; `platform/tools/isa_xcheck.py`.
Python: `platform/python/rocke/runtime/direct_lift.py` (new; the asm lifting, with the memory-clobber fences),
`platform/python/rocke/helpers/compile.py` (lift hook, `direct_levers`, strict mode),
`platform/tests/runtime/test_direct_targets.py`.

Port as one unit: the lift fence fix and the assembler are correctness fixes that the lifting and the CCL / probe
kernels depend on. Every lever is default-off: with no levers the census objects stay byte-identical to the R2
crate (2,017 + 18), which is the no-regression gate.

## 4. Gates to run after the port (all scripted here)

- `jobs/rt_cpu_submit.sh <v>`: crate build (0 warnings), `cargo test` (lib / all targets / 1 thread), ISA table
  checks, the census vs R2 (byte-identical).
- `isa_xcheck.py --isa gfx950` vs llvm-mc: 0 mismatches.
- `rt_mutants.py cpu` (local) and the GPU mutants (`jobs/rt_gpu.sh` steps `mut:*`): every mutant killed.
- `jobs/rt_gpu.sh` numerics + paired timing for the families switched to direct, both nodes (win rule).
- `jobs/rt_asm.sh` (P4, P5 with `acqfence`, CCL on 2 GPUs): the suites' own gates pass with direct objects.

## 5. Proposed default changes (each a separate decision)

1. `acqfence` on by default: an acquire-only fence then matches LLVM (no write-back); no stage kernel has one, so
   only census objects with acquire fences change, and P5's coherence controls need it.
2. `bufoffany` on by default (LLVM's buffer offset split): neutral to small win everywhere measured.
3. The B3 lever set as the default for B3 builds (attribute on the B3 emitter's kernels).

## 6. Open items

- Head: 0.3-1.2% behind COMGR after every lever; ATT of one CU shows the direct object stalling less per wave, so
  the loss is outside the per-CU instruction stream (placement / memory-system interplay); next probe: whole-grid
  stamps of the head (it has none) or a WaveScope capture over more CUs.
- MoE r4: neutral; the remaining phases are down / combine, which include waits on other CTAs.
- Q2 and framework layer kernels: paired timing harnesses not built (they compile directly). The toy chain is timed
  (`r2` step): the B3 set wins on d8w0, neutral on d4w8l2.
- Probe / CCL timing direct vs COMGR: the suites validate behavior; their benches time one backend per process.
- Per-op lever scoping for the true mono (section 2).

## 7. WP-R3 status (2026-11-01)

Ported into the copy-rocm-lib working tree (`wave1/WP-R3.md`): the crate, the direct runtime and lifting, WP1's named
lever scopes consumed as `dlevers`, the backend defaults `acqfence` + `bufoffany`, and D0.6's per-family default
(`persist.knobs.rust_family` for standalone kernels; `FrameworkKnobs(code_object_backend="auto")` reads
`RUST_FRAMEWORK_DIRECT_FAMILIES`, empty after the fix round: framework kernels stay COMGR until booked) for B3 and
fused MoE rows 1 only; the fix round confirmed both on both nodes under DESIGN 8.1 (`wave1/WP-R3.md` "Fix round"). The
COMGR path is unchanged and every existing build is byte-identical. Section 1's table is superseded by
`wave1/WP-R3.md` section 8 (Q2 / framework: LOSS; P4 xl, P5 scalar re-read: WIN; head gap: memory-level parallelism,
`loadfirst` closes most of logits r16).
