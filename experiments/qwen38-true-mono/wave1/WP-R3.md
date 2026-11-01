# WP-R3: the Rust direct code-object backend in the copy-rocm-lib tree (W5, wave 1b, 2026-11-01)

D0.6, D0 7.4 m5, `rust-track/INTEGRATION.md` 2-5, PLAN decisions 6, 9-11, 13. Ratios only (direct or lever arm time
/ COMGR time of the same KernelDef, paired in one process; < 1 is faster). Absolute times and the job index:
`hackathon-prep/private-perf/qwen38-true-mono/wave1/WP-R3/`. Harness and port tooling:
`hackathon-prep/qwen38/truemono/WPR3/`. No git state was changed anywhere.

**Gate: see "Fix round" at the end** (it supersedes this paragraph). Crate tests, census, scope check, byte identity
of every existing COMGR build and the both-node WINs of the D0.6 families hold. ~~The D0.6 confirmation passed the 9.1
win rule three times on n07-17 / n01-09 under the 8.1 neighbour rule~~ (fix round: overstated; on n01-09 only B3 r1 /
r4s tuned had an 8.1-compliant run before the fix round). The other families' first runs on n01-09 (Q2, probes, CCL,
two lookbacks) had 3-4 of our neighbours; their verdicts rest on n07-17.

## 1. What was ported

`port.py` builds an overlay of exactly the files WP-R3 adds or changes (157: the crate's 118 files, 22 other new
files, 17 changed files), records each target's pre-port hash, and applies it only if no target changed since and
none was edited in the last 30 minutes. Classes:

| class | files |
|---|---|
| new: the crate | `platform/rust/rocke-codeobj/` (rust-track tree v21: isa tables, `rir/`, `target/gfx9/`, examples) |
| new: Python | `runtime/{code_object,code_object_elf,direct_session,direct_lift}.py`; `tools/{isa_tables,isa_xcheck,isa_burs,lever_scope_check}.py`; `qwen38_mono/{irops,direct_pilot}.py`; docs `dsl_docs/runtime/{direct_compiler,code_object_compiler}.md` |
| new: tests | `platform/tests/runtime/test_{code_object,code_object_elf,direct_session,direct_targets}.py`, `platform/tests/tools/test_isa_{burs,tables,xcheck}.py`, `library/tests/test_qwen38_{lever_scopes,direct_pilot,wpr3}.py` |
| rust side only (mono file unchanged since HEAD) | `runtime/comgr.py` (errors subclass `CodeObjectError`; error-path cleanup), `core/passes.py` (`dlevers` outside the CSE key), `helpers/mailbox.py` (`xcc_id` hook), `mx_decode_gemv.py` / `stamps.py` / `toy_chain.py` (`irops` knobs, default off), `test_persistent_mem_ops.py`, `env_flags.md`, `pyproject.toml` (msgpack test dep) |
| clean 3-way merges (base HEAD) | `core/ir.py` (`IRBuilder.lever_scope` / `set_lever_scope`, the device-op IR ops), `core/lower_llvm.py` (their lowering = the same asm), `qwen38_mono/README.md` |
| hand merges | `helpers/compile.py`, `framework.py`, `moe_fused.py`, `persist/knobs.py`, `persist/kv.py` (below) |

Left out on purpose (not needed by the direct backend, and each would touch the COMGR path): the rust branch's
compile.py refactor of the COMGR route through the code-object facade (`_code_object_llvm_flavor`, the
`CodeObjectInput` rename), the `ir_serialize.py` memoization, `isa/backend.py`'s unused `splits_global_address`,
`core/backend.py`'s C++ flavor move, the package `__init__` re-exports and the import renames in examples /
portable-IR drivers, and the gfx1151 `helpers/direct_gemm.py`. **The COMGR path of `compile_kernel` is the same code
as before**; the direct backend is a branch taken only when `code_object_backend="direct"`, the kernel's
`request_code_object_backend` mark, or `ROCKE_CODEOBJ_BACKEND=direct` asks for it (default: COMGR; fix round
correction: `runtime/comgr.py` imports `rocke.runtime.code_object` when it loads, so the facade module is imported
by every COMGR compile; the direct `.so` still loads only on a direct compile). A refusal falls back to COMGR (`artifact.direct_fallback`); `ROCKE_DIRECT_STRICT=1`
raises instead.

New Python is Python-only (no C++ engine change). `framework.py` kept every WP1 hook; the rust-tree knobs came in as
`FrameworkKnobs` fields with `repr=False` (`code_object_backend`, `irops`, `stage_levers`, `direct_levers`), and
`FusedKnobs.phase_levers` with `repr=False` (the fused MoE's pair id is a hash of `repr(knobs)`: a repr field would
change every pair id).

## 2. Per-op lever scoping (WP1's hook, consumed)

- WP1's `RustScope` tags every op of a scoped stage with `attrs["rocke.scope"] = <stage>` and the kernel with
  `attrs["direct_scopes"] = "<stage>:<lever>+..."`. The direct backend (`compile.direct_source`) turns the named set
  into the crate's per-op scope `dlevers` on a copy (the KernelDef keeps its attributes for a COMGR fallback),
  before the asm lifting (lifted ops inherit the scope), drops every `rocke.*` op marker (`rocke.kv` too, which the
  crate would not parse) and the kernel's `direct_scopes` / `direct_levers`.
- An op's explicit `dlevers` (`IRBuilder.lever_scope`: the rust tree's `stage_levers` / `phase_levers`) wins over
  the named set. `stage_levers` is applied by the same `_RustScope` context, so the persistent loop driver gets both.
- `_RustScope` now covers the whole stage (barrier, entry, prologue, tasks; was tasks only) and scopes the stage's
  constants too (`kv.marked(constants=True)`), as `lever_scope` does: a named scope and the same explicit scope give
  the same object (tested with the direct library: `test_named_scopes_build_what_explicit_scopes_build`).
- None of it reaches the LLVM IR: a kernel with any of these knobs lowers to the same LLVM text and name (tested).

## 3. Defaults (D0.6)

- **Backend defaults `acqfence` + `bufoffany`** (`compile.DIRECT_DEFAULT_LEVERS`, equal to
  `persist.knobs.RUST_BACKEND_DEFAULTS`: tested). Every direct compile adds them to `ROCKE_DIRECT_LEVERS` and the
  kernel's `direct_levers`; `ROCKE_DIRECT_DEFAULT_LEVERS` replaces the set (`""` = none). They change 26 of the 85
  directly compiled true-mono objects (fused MoE r1 / r4 / stamped, the framework layers holding the MoE, the heads,
  P2, P5 kv / host) and none of the B3, Q2, toy, P4 or CCL objects (direct census, section 5).
- **Direct as the default per family where both nodes WIN** (D0 7.4 m5: only in new true-mono builds; existing
  B3 / MoE builds stay COMGR and byte-identical):
  - `persist.knobs.RUST_DIRECT_FAMILIES = ("b3", "moe_r1")` (the B3 set; the fused-MoE rows-1 set);
  - `FrameworkKnobs(code_object_backend="auto")` + `rust_scope`: direct when every stage of the kernel is scoped to
    one of those families, else COMGR (a head or rows-4 MoE stage keeps the kernel on COMGR). **Fix round: framework
    kernels now read `persist.knobs.RUST_FRAMEWORK_DIRECT_FAMILIES` (empty), so `auto` builds COMGR until a
    framework-hosted family is booked (see "Fix round" item 2).**
  - `persist.knobs.rust_family(kernel, family)` for a standalone kernel of one family: the family's set as
    `direct_levers` and the direct mark (COMGR families: unchanged). Its objects are byte-identical to the
    environment-lever objects that were timed (B3p = B3, M1p = M1 in every confirmation job).
  - New named set `moe_r4` (the r1 set; not a direct family).
- The true-mono layer kernels (WP3) do not exist yet, so nothing is switched by default today; D0.6's in-situ
  `layers=1` paired test stays the condition for booking it (unchanged).

## 4. Gates (CPU, CPUONLY&MARKHAM, final overlay; jobs c3 / c6 / c7 / i3)

| gate | result |
|---|---|
| crate release build lib + examples | 0 warnings |
| `cargo test` lib / all targets / 1 thread | 481 / 481 each |
| `isa_tables.py check` (both ISAs), ISA tool pytest | ok, 25 passed |
| `librocke_codeobj_ffi.so` from the ported crate | byte-identical to the rust-track v21 library |
| census vs the R2 crate (2,029 gfx950 + 18 gfx1151 IRs) | 2,017 / 2,017 and 18 / 18 objects byte-identical, 12 same errors, CENSUS OK |
| lever-scope check (`dlevers`, 2,029 lifted IRs): scoped on == env levers; env levers scoped off == plain | 2,029 / 2,029 each, SCOPES OK |
| direct census through `compile_kernel` (93 rt_builds labels): rust tree + v21 library vs ported tree, defaults off | 92 identical objects, 1 label whose source IR differs between the trees (`ccl_linkbw`: the mono tree's lm_probe is newer), 0 differences; 85 direct, 0 fallbacks, 8 errors (heads rows 1 / 8 are not valid specs) |
| **byte identity of every existing build, base (pre-port snapshot) vs base + overlay** | q38 / true-mono: **87 / 87 builds byte-identical in LLVM IR, serialized IR and COMGR `.text`** (261 files; 10 new-only builds are the `irops` / head labels the base cannot build); parity families: gfx950 821, gfx942 768, gfx1151 860, gfx1201 500 files, all identical |
| must-fail mutant D (the default backend flipped to direct) | caught: 84 of 87 `.text` differ |
| targeted tests on the ported tree (with the built library) | 1,032 passed, 130 skipped (library); 230 passed, 73 skipped (platform); the skips need a GPU (/dev/kfd) |
| local must-fail mutants of the port (`mutants.py`: scope mapping, markers, defaults, env restore, backend selection, family table, any / all, constants, stage levers, auto, repr) | 18 / 18 killed |

The two obsolete R1 census tests of `test_qwen38_direct_pilot.py` (the R1 crate's first-refusal tables) were
replaced by a coverage test: every representative build compiles direct in strict mode, except the scaled-MFMA
probe (pointer-to-vector argument), which is refused cleanly.

## 5. On hardware

### 5.1 D0.6 confirmation (rt_b3 / rt_fmoe, 32 shuffled rounds, controls C2 / R, numerics first: every arm bitwise)

| kernel | arm | n07-17 (g1) | n07-17 (g1c, fresh allocation) | n01-09 (g1q, fresh seed, 8.1) | n01-09 (g1, 4 of ours busy) |
|---|---|---|---|---|---|
| B3 r1 tuned | B3 set (+ defaults; = policy object) | 0.979 WIN | 0.971 WIN | 0.974 WIN | 0.979 WIN |
| | plain direct | 0.985 WIN | 0.977 WIN | 0.982 WIN | 0.987 WIN |
| B3 r4s tuned | B3 set | 0.984 WIN | 0.976 WIN | 0.978 WIN | 0.987 WIN |
| | plain direct | 0.999 neutral | 0.999 neutral | 0.992 WIN | 1.001 neutral |
| B3 r4s mono | B3 set | 0.944 WIN | 0.952 WIN | 0.937 WIN | 0.940 WIN (controls 0.983-0.986) |
| fused MoE r1 | MoE r1 set (+ defaults) | 0.974 WIN | 0.978 WIN | 0.972 WIN | 0.969 WIN |
| | MoE r1 set, no defaults | 0.973 WIN | 0.975 WIN | 0.972 WIN | 0.972 WIN |
| | plain direct | 1.020 LOSS | 1.020 LOSS | 1.020 LOSS | 1.022 LOSS |

The defaults change no B3 object (B3n = B3) and are neutral on the fused MoE r1 (0.998-1.003 relative). Every arm
32 / 32 rounds below 1; controls 0.991-1.006 except g1's mono r4s on n01-09. g1q ran with one job of ours on n01-09
(a second one started for its last 40 s, during the fused MoE step); g1 / g1c on n07-17 with none. Buses: n07-17 5,
n01-09 5 (g1q).

### 5.2 Suites with the direct defaults (strict, n07-17, 2 GPUs: asm1)

P4 (numerics, negative controls, mutants), P5 (positives; every negative control and the early-rewrite mutant fail
as required: the controls that needed `acqfence` pass with the default) and the CCL multi-process suite (bitwise
all-reduce / all-gather / argmax, tag wrap, stress, dropped / aborting peers, stale tags, mutants, coherence,
lm_bitwise / canon / mutants): all PASS.

### 5.3 Paired timing of the other families (new harness runs; n07-17 is the 8.1-compliant node)

| kernel | best arm | n07-17 | n01-09 (non-8.1) | verdict |
|---|---|---|---|---|
| Q2 in the framework layer kernel r1 / r4s (`rt_q2`, first run) | plain direct | 1.058 / 1.130 LOSS | 1.084 / 1.132 LOSS | keep COMGR |
| | B3 set / MoE r1 set | 1.057-1.060 / 1.113-1.117 LOSS | 1.086-1.088 / 1.118-1.125 | |
| P4 xl hand-off (`rt_probe_ab`) | plain direct | **0.860 WIN** | 0.858 WIN | direct wins |
| P4 counter / a2a / ticket | plain direct | 1.018 / 1.012 / 0.867 neutral | 1.018 / 1.007 / 0.962 neutral | neutral (ticket controls spread 0.77-1.04) |
| P5 scalar re-read (sc) | plain direct | **0.939 WIN** | 0.944 WIN | direct wins |
| P5 kv hand-off | direct (with `acqfence`) / without | 1.088 / 1.268 LOSS | 1.085 / 1.257 LOSS | keep COMGR; the default cuts the loss 4x |
| CCL all-reduce chain `line_os` 1 / 4 / 16 x 2560 (`wpr3_ccl_ab`, device latency) | plain direct | 1.018 / 0.978 / 0.945 neutral | 1.016 / 1.014 / 0.968 neutral | neutral |
| CCL `pair_os` 1 / 4 / 16 x 2560 | plain direct | 0.972 / **0.972 WIN** / 0.999 | 0.949 / **0.953 WIN** / 0.979 | 4 x 2560 wins on both nodes; the others neutral |

### 5.4 Combination lookback over every lever, losers included (`rt_tune`, 65 configs + confirmation each, the
backend defaults off so `acqfence` / `bufoffany` are explicit dimensions)

| family | start | node | best confirmed | verdict |
|---|---|---|---|---|
| head logits r16 | loadfirst | n07-17 | 1.003 (bitop3, bufoffany, dppcombine, litreuse, loadfirst, loadorder, minmax, scalesep, storewait, vmemfirst, wpecap) | neutral |
| head fused r4 | vmemfirst + loadfirst | n07-17 | 1.003 (bufoffany, dppcombinef, litreuse, loadfirst, nopause, vmemfirst) | neutral |
| fused MoE r4 | MoE r1 set | n01-09 (exploratory) | 1.002 (the r1 set + nopause) | neutral |
| Q2 r1 | none | n01-09 (exploratory) | 1.064 | LOSS: no lever combination recovers it |

The rust track's own B3 r1 lookback (rt-tb3r1, n01-09, 10-31) confirmed its start set G0 at 0.968 WIN; not repeated.

## 6. The head gap (open item 1): what it is, and what closes part of it

- **Not cross-CU placement, dispatch or clocks.** Whole-grid CTA stamps (`wpr3_headstamps.py`, both nodes; stamped
  builds are diagnostics, not timing): direct CTAs start no later (start-ramp p90 is 0.3-0.4% of the kernel span in
  both objects), the tail share is unchanged, and the per-CTA duration is longer by the same 1.1% (fused r4) / 1.4-1.5%
  (logits r16) on every one of the 8 XCDs. SCLK sampled per arm is equal (1.000). Occupancy is equal (the LDS
  allows one CTA per CU in logits r16; VGPRs allow the same waves in fused r4).
- **It is memory-level parallelism.** The direct main loop issues fewer instructions (936 vs 973) and half the nop
  cycles, but it splits COMGR's groups of five weight loads and drains `vmcnt` to 8 (fused r4) or 5 (logits r16)
  where COMGR never goes below 9: fewer loads in flight per wave on every CU, hence a uniform per-CTA slowdown that a
  one-CU ATT trace reads as "fewer stalls".
- **`loadfirst` restores COMGR's load schedule exactly** (groups of five, waits 13 -> 10) and moves logits r16 from
  1.014-1.017 to 1.004 (m1 1.001), neutral. Fused r4 stays at 1.003-1.008: its remaining 0.3-0.5% is outside the
  main loop: with `loadfirst` the loop matches COMGR's schedule, while the prologue and the argmax fold carry 65 more
  instructions than COMGR's (+18 LDS ops, +24 SALU); not yet attributed further. `loadfirst` needs `bufoffany` (without it 1.014-1.020).
- **Decision:** the head stays COMGR (no WIN on any build); the head's candidate set is now `loadfirst + vmemfirst
  + bufoffany` (+ the lookback's F sets, all neutral). `loadorder` alone still loses (1.013-1.036).

## 7. Other open items

- **MoE rows 4**: neutral. The lookback (65 configs, defaults off, n01-09) confirmed 1.002 at best. With the backend
  defaults the r1 set reads 0.998 (24 rounds) and 0.995 (48 rounds, fresh seed; controls 0.997 / 0.999, so neutral
  by the win rule), against 0.999 without the defaults in the same job: `bufoffany` + `acqfence` are worth about
  0.4% here. Phase-scoped levers (`phase_levers`, ported; n07-17, 24 rounds): `nofence` / `loadorder` off in down and
  combine 1.005, `wgfence` off there 1.003, `nopause` everywhere 1.002 or outside xchg / route 0.998, `loadfirst` in
  issue / gu / down / combine changes no byte. No phase scope beats the plain set; the rust track's phase stamps put
  the residue in down / combine, which wait on other CTAs (a schedule matter, not a code-generation one).
- **Q2 / framework layer**: a 6-13% loss that no lever combination recovers (lookback). The ISA differs in the
  mailbox polls (one dwordx4 tagged poll with a per-lane exec exit vs COMGR's two dwordx2 loads and a uniform exit)
  and in the scalar address code (101 vs 63 SALU): a compiler item, not a lever.
- **Probes / CCL**: the P4 xl hand-off and the P5 scalar re-read win on both nodes; they are probes, not served
  kernels, so nothing switches; the in-kernel xl / scalar-reread code of the true mono is where it can matter.
- **8.1**: the n01-09 runs of g1, g2, ccl1 and t2 had 3-4 of our jobs on the node.

## 8. Recommended default per family (updates `rust-track/INTEGRATION.md` 1)

| family | default in new true-mono builds | lever set |
|---|---|---|
| B3 (tuned and mono, all variants) | **direct** for standalone kernels (`rust_family(k, "b3")`); framework kernels: COMGR under `auto` until booked (fix round) | B3 set + backend defaults |
| fused MoE rows 1 | **direct** (standalone; framework: as B3) | MoE r1 set + backend defaults |
| fused MoE rows 4 | COMGR | `moe_r4` named set + defaults (0.995-0.998, neutral) |
| head (fused / logits, rows 4 / 16) | COMGR | candidate: loadfirst + vmemfirst + bufoffany (neutral) |
| Q2 / framework layer stages | COMGR | none recovers the loss |
| probes P4 xl, P5 scalar re-read | direct wins (probes) | plain + defaults |
| P5 kv, CCL collectives | COMGR (kv loses; CCL neutral except pair_os 4 x 2560) | plain + defaults |

## 9. Verification and gaps

- Self-verification (no separate verifier agent in this run): every ratio above is the harness's own per-round
  median against two controls with the 9.1 rule; the confirmation verdicts were reproduced on a fresh allocation
  (g1c); the policy objects are byte-identical to the timed lever objects; the identity verdict rests on a dump of
  the pre-port snapshot plus exactly the overlay, so concurrent edits elsewhere in the tree cannot leak in; each
  identity half has a must-fail mutant (D, caught). (Fix round correction: D exercised only the `.text` half; the IR
  half got its mutant F from the verifier and in the fix round.)
- The n01-09 confirmation under 8.1 was submitted by `WPR3/wait_n0109.sh` (fresh seed 29) once at most one of our
  jobs ran there (g1q, section 5.1). The first n01-09 runs of Q2, probes, CCL and the MoE r4 / Q2 lookbacks were not
  repeated under 8.1; their verdicts rest on n07-17 (and every n01-09 verdict agrees).
- Not done: the mono tuner catalog's `DIRECT_LEVERS` / `direct_levers()` (the tuner's own working-tree file was being
  edited; `rt_tune.py` builds the same space itself); M5 (the code-object backend in the tuner's cache fingerprint)
  stays deferred to the tuner owner.

## 10. Files and jobs

- copy-rocm-lib rocKE tree: the 157 overlay files (`hackathon-prep/qwen38/truemono/WPR3/overlay.json` lists each
  with its pre-port and applied sha256).
- Harness (`hackathon-prep/qwen38/truemono/WPR3/`): `port.py`, `edits/`, `stage_cpu/{wpr3_cpu.sh,wpr3_dcensus.py,
  ident_existing.py,census_cmp.py,wpr3_meta.py}`, `h/rust-track/{wpr3_headstamps.py,wpr3_ccl_ab.py,jobs/wpr3_ccl.sh,
  jobs/wpr3_tune.sh}` and the patched `rt_b3.py` (`~` no-default and `%family` policy arms), `rt_compile.py`,
  `rt_tune.py` (own lever space), `jobs/rt_gpu.sh` (`hst` step, bus print, `RT_SEED`); `loops.py`, `mutants.py`.
- Jobs: CPU 68236197 (c1), 68236198 (i1), 68236243 (c2), 68236278 (m1), 68236376 (c3), 68236377 (i3), 68236413 (c6),
  68236435 (c7); GPU 68236274 / 68236276 (g1), 68236525 (g1c), 68236315 / 68236318 (g2), 68236341 (h1),
  68236326 / 68236329 (ccl1), 68236365 (asm1), 68236367 / 68236369 (t1 / t2), 68236557 (m4s), 68236567 (m4c),
  68236738 (g1q).

## Verification (independent verifier vWPR3, W5, 2026-11-01)

**Verdict: partial.** The hard-rule gates hold and I re-proved the byte identity myself with a wider net. The CPU
gate logs match the report. Two things fail or are unsupported. First, the both-node confirmation of fused MoE r1
and B3 r4s mono never had a run on n01-09 that met DESIGN 8.1. Second, the framework-kernel half of the `auto`
policy has no measurement behind it. Harness and results: `hackathon-prep/qwen38/truemono/vWPR3/` (`stage/`,
`results/`). I made no git or tree edits; only this section was added.

**Re-run (CPU job 68236880 `w5-vWPR3-cpu1`, CPUONLY&MARKHAM, ctr-smc-s26-005):**
- **Base:** I checked that WP-R3's base snapshot (`w5wpr3-base`) really is the pre-port tree. All 157 of 157 overlay
  paths pass: every new file is absent, and every changed file hashes to `overlay.json`'s `mono_sha` (LF or CRLF).
- **New tree:** base plus the 157 files copied from the copy-rocm-lib working tree as it is now. Every file matches
  WP-R3's applied sha, and `port.py check` reports nothing pending or recent. The tree files equal the c7 overlay up
  to EOL. c3, c6 and i3 differ from c7 only in `test_qwen38_wpr3.py`, so the c3 test failure was fixed in the test,
  not the code. The GPU stage `rocke-wpr3-v1` equals the tree except that same test file.
- **Identity, q38 set:** LLVM IR, serialized IR, COMGR `.text` and now the **whole code object (hsaco)** were compared
  for every build. **87 / 87 existing builds are byte-identical (348 files).** The 10 new-only labels are the ones
  whose spec has no `irops` in base. Each `head_*` new-only object equals the matching base `headplain_*` object.
- **Identity, W6 large-batch builds** (not in the q38 set, although `wp1_identity.py`'s docstring says they are):
  - The 7 large-batch builds that exist in the base are identical in IR, serialized IR and hsaco.
  - `e1`, `lbf` and `lbf_stale` are newer than the base. I checked them locally instead, against the current tree
    with the pre-port `persist/knobs.py`, `persist/kv.py` and `helpers/mailbox.py` loaded over it. LLVM IR and
    serialized IR are identical for e1, lbf 4, lbf 8 and stale.
  - Caveat: the local check overrides modules after the package imports. The changes are additive with defaults off.
- **Must-fail mutants:**
  - **D** (default backend flipped to direct): caught, 0 / 87 identical.
  - **F** (op order swapped in `mailbox.iod_id`, one of the edited functions): caught on the IR half (11 builds).
    WP-R3's own mutant D tested only the `.text` half, so the IR half had no mutant until now.
  - **E** (`kv.marked` marks constants by default): not caught. No existing build calls `kv.marked` (I counted 0
    calls while building p5_kv, p5_sc, fw_q2_r1, fw_b3_r1, fmoe_r1 and b3_r1_tuned), so that change cannot reach an
    existing build. This is not a blind spot of the identity check.
- **Code review:** the COMGR branch of `compile_kernel` is unchanged; the direct path is a guarded early branch.
  `passes._cse_key` drops only `dlevers`. The `comgr.py` edits change only the exception base class and rename
  variables on the error path. The `kv.marked` / `mailbox` / `knobs` changes keep their defaults.
- **One claim is wrong:** "an unmarked kernel never imports the facade". `runtime/comgr.py` now imports
  `rocke.runtime.code_object` when the module loads. The `.so` still loads lazily, so this is harmless.

**Logs re-read:**
- **CPU gates:** c3, c6, c7 and i3 show what the report says. Crate 0 warnings, 481 tests three ways, `.so` sha equals
  v21, census OK, scopes OK. The direct census is 92 same plus 1 source-IR difference. i3's parity identity is
  correct, but i3's q38 line says DIFFERS because that run had no existing-builds filter; c6's filtered result is the
  valid one.
- **Suites (asm1):** they ran with `ROCKE_CODEOBJ_BACKEND=direct` and `ROCKE_DIRECT_STRICT=1`. Every
  control and mutant fails as required.
- **Timing harness (`rt_b3.py`):**
  - Numerics run before timing in the same job, every arm bitwise.
  - 4 placements per arm, shuffled in every round; 32 rounds.
  - Two controls (C2 = a second graph of the baseline, R = a recompiled copy).
  - The win rule is coded as in DESIGN 9.1.
  - The policy arms (`%b3`, `%moe_r1`) build through the tree's `rust_family` and alias to the timed lever objects.

**Problems found:**
1. **Not every D0.6 confirmation on n01-09 meets 8.1.**
   - The whole g1 n01-09 run had 4 of our jobs on the node.
   - In g1q, `qk-qd1` (68236742) started at 03:37:33 UTC, 4 s into the B3 r4s mono step (03:37:29). It ran through
     that step and the whole fused MoE r1 step (03:37:39 to 03:38:11), next to `qk-lbJ`: 2 of ours (sacct).
   - So **fused MoE r1 and B3 r4s mono have no 8.1-compliant run on the second node.** B3 r1 and r4s tuned do
     (g1q, 1 of ours).
   - Every run agrees: fused MoE r1 0.969-0.978, r4s mono 0.937-0.952. The WIN is probably real, but the gate as
     written was not exercised.
   - My re-run (`wait_g1v.sh`, seed 71) could not start. n01-09 had 2-4 of our jobs from 04:00 to 04:27 UTC, so
     nothing was submitted. **Re-run fused MoE r1 and B3 r4s mono on n01-09 under 8.1 before booking them as
     direct defaults.**
2. **The `FrameworkKnobs(code_object_backend="auto")` rule has no measurement behind it.**
   - Its rule: direct for a framework kernel whose stages are all B3 / moe_r1.
   - No framework-hosted B3 or MoE kernel was timed with the direct backend. The WINs were measured on standalone
     B3 and fused-MoE kernels.
   - The only framework-hosted direct timing, fw_q2 r1 / r4s, loses 6-13%, with the B3 and MoE r1 sets too. Section
     7 traces that loss to framework code (mailbox polls, SALU address code), which a framework-hosted B3 or MoE
     kernel would also carry.
   - D0.6's in-situ test still gates booking, but `auto` should stay COMGR for framework kernels until
     fw_b3 / fw_moe direct runs win.
3. **The B3 ratios against C are biased low by a placement effect.**
   - In g1q's B3 r1 tuned, both controls read 0.991 with 32 / 32 rounds below 1. Against the pooled COMGR arms the B3
     set reads 0.980, not 0.974.
   - The win rule still passes, but the table overstates that row by about 0.6%. The pooled view (`vs_pooled`) is
     the fairer magnitude.
4. **Minor.**
   - The first g1 runs have no bus recorded.
   - `amdsmi.jsonl` is recorded, but no throttled-round drop is applied or reported (DESIGN 9.1 clocks).

**Confirmed:**
- No git changes.
- Python-only.
- Byte identity of every existing COMGR build: q38 and W6 large-batch re-run by me; parity on gfx950 / gfx942 /
  gfx1151 / gfx1201 from the i3 and c6 logs, whose overlay differs from the final one only in a test file.
- Default COMGR, and the strict and fallback behaviour (code read).
- Crate and census gates.
- Both-node WINs of B3 r1 / r4s tuned under 8.1.
- The open-item conclusions: head, MoE r4 and Q2 stay COMGR.

## Fix round (W5, 2026-11-01)

Every verifier issue is addressed below. Harness and tools: `hackathon-prep/qwen38/truemono/WPR3/` (`fixround.py`,
`fixcpu/fixcpu.sh`, `fix_wait.sh`, `wpr3_fix_analyze.py`, `edits/{make_rt_fw,patch_fix_harness,patch_fix_clock}.py`,
`h/rust-track/{rt_fw,rt_clock}.py`). Private records: `private-perf/.../wave1/WP-R3/fixround/`. No git state changed.

**Gate: met for the D0.6 families; the framework-kernel half of `auto` stays COMGR (not booked).** B3 (r1 / r4s
tuned, r4s mono) and fused MoE rows 1 now each have an 8.1-compliant WIN on both rocke nodes, every round
clock-checked. Byte identity holds for every existing build, including the W6 large-batch builds, with a must-fail
mutant on each half.

### 1. 8.1-compliant confirmation on both nodes (issue 1)

Harness v10 (`wpr3-h-v10`) on tree stage `rocke-wpr3-v2` (the current tree with the fix-round edits). Each job's 8.1
audit is per timed step: `TZ=UTC sacct` on login-01 lists every job of ours on the node that overlaps the step window.
A step counts only when at most one of our jobs ran next to it. 32 shuffled rounds, two controls, every arm bitwise.
The table gives the lever-set arm (B3 set / MoE r1 set, the backend defaults on; = the policy object), vs C and
(pooled) vs the mean of the three COMGR arms:

| kernel | n07-17 (job 68237921, seed 93, bus 21; 1 of ours) | n01-09 (8.1-compliant step) |
|---|---|---|
| B3 r1 tuned | 0.975 WIN (pooled 0.975) | 0.984 WIN (pooled 0.982; job 68238252, seed 85, bus 21; 1 of ours) |
| B3 r4s tuned | 0.980 WIN (0.979) | 0.978 WIN (0.979; 68238252) |
| **B3 r4s mono** | 0.943 WIN (0.943) | **0.940 WIN (0.942; 68238252)** |
| **fused MoE r1** | 0.974 WIN (0.974); without the defaults 0.973 | **0.968 WIN (0.968; job 68240239, seed 87, bus 5; 0 of ours)**; without the defaults 0.971 |

Every arm is below 1 in 32 / 32 rounds. Plain direct is unchanged: B3 r1 and r4s mono WIN, r4s tuned neutral, fused
MoE r1 LOSS (1.018-1.022). Getting the compliant steps took several tries. Other lanes' jobs started next to our runs
mid-job, so the non-compliant steps of 68237402, 68238252 (MoE and framework steps) and 68240266 were not booked. n07-17
went down at about 05:45 UTC and came back unseeded; W5 reseeded it (`q38/seed.sh`, job 68237780). Every
non-compliant step agrees with the table within 0.01.

### 2. `FrameworkKnobs(code_object_backend="auto")` for framework kernels (issue 2)

- **Code:** `persist.knobs.RUST_FRAMEWORK_DIRECT_FAMILIES = ()`. `rust_scope_backend` reads it by default, so
  `auto` builds framework kernels on COMGR until a family is added there. `rust_family` (standalone kernels) still
  uses `RUST_DIRECT_FAMILIES`. Applied through `fixround.py`, which updates only the 3 overlay entries (knobs.py,
  the framework.py docstring, test_qwen38_wpr3.py) and refuses targets edited in the last 30 min. Three new
  must-fail mutants of the port (the default reads the standalone set, the families are booked, the argument is
  ignored) are killed, along with the 18 earlier ones: **21 / 21 killed**.
- **Measurement:** the new `rt_fw.py` (generalized from `rt_q2.py`) times the framework-hosted stages that `auto`
  would switch. These are `fw_b3_<v>` (delta layer, native B3) and `fw_moe_r1` (M3..M5 through the adapter), plus
  `fw_q2_r1`. Each runs after ATOM's FlyDSL prefix on the mono rig, with numerics first: bitwise against C, twice,
  and the stage must write (`rec` + scratch for B3, `h_out` + scratch for the MoE). The arm `S=^scoped` is the
  exact `auto` object (`rust_scope` over the stages + direct); its COMGR object equals C's bytes (checked in-job).
  Results (lever-set arm; S is byte-identical to it in every job, so the framework code outside the stage is
  unaffected by those sets):

| kernel | n07-17 (68237921, compliant) | n01-09 (68240239) |
|---|---|---|
| fw B3 r1, B3 set | 0.942 WIN (pooled 0.946) | 0.949 WIN (0.947), compliant step |
| fw B3 r4s, B3 set | 0.915 WIN (0.915) | 0.915 WIN (0.914), **not compliant** (2 of ours) |
| fw MoE r1, MoE r1 set | 0.980 WIN (0.978) | 0.967 WIN (0.967), **not compliant** (3 of ours) |
| fw Q2 r1, plain / B3 set scoped | 1.084 / 1.072 LOSS | 1.089 / 1.088 LOSS |
| plain direct: fw B3 r1 / r4s / fw MoE r1 | 0.970 neutral (pooled WIN) / 0.944 WIN / 1.031 LOSS | 0.977 neutral / 0.941 WIN / 1.020 LOSS |

- **Reading:** framework-hosted B3 and MoE r1 win direct with their sets on both nodes. Q2's loss belongs to the Q2
  stage's own code, not to the framework's shared code, which all four kernels carry. Booking B3 / `moe_r1` in
  `RUST_FRAMEWORK_DIRECT_FAMILIES` still waits on two things: compliant n01-09 steps for fw B3 r4s and fw MoE r1
  (n01-09 held 3-5 of our long jobs while the waiter ran), then D0.6's in-situ `layers=1` paired test. Until then
  `auto` stays COMGR for framework kernels, as the verifier asked.

### 3. Pooled view (issue 3)

`rt_fmoe.py` now reports `vs_pooled` too. A harness bug (the alias loop shadowed the times variable) had written the
alias name instead of the per-round times to every earlier fused-MoE `raw.json` that had an alias, so the first runs'
MoE rows cannot be re-read; the new runs carry them (fixed). Every table above gives C and pooled. B3 set rows of the
first runs, vs C / vs pooled: g1 n07-17 r1 0.979 / 0.979, r4s tuned 0.984 / 0.984, r4s mono 0.944 / 0.942; g1c
0.971 / 0.973, 0.976 / 0.975, 0.952 / 0.947; **g1q 0.974 / 0.980**, 0.978 / 0.980, 0.937 / 0.937; g1 n01-09
0.979 / 0.975, 0.987 / 0.984, 0.940 / 0.950. The bias of C changes sign from run to run (up to 0.6%, as the
verifier found in g1q); every verdict holds in both views (`wpr3_fix_analyze.py` also applies the win rule to the
pooled ratios).

### 4. Identity: the IR half's mutant and the W6 large-batch builds (issues 4, 5)

- `wp1_identity.py` (WP1's and WP-R3's copies) now builds the 13 W6 large-batch labels its docstring promised;
  `lb_insitu` / `lb_insitu_moe` take their sub-plan's default knobs. A module missing from a tree is listed as
  "skip", never as identical.
- **Base2 instead of the old snapshot.** CPU job 68237240 (`w5-wpr3fx-cpu2`, ctr-smc-s26-005) builds base2 from one
  snapshot of the current tree: the WP-R3 files are reverted (new files removed; changed files restored from the
  verified pre-port base, hash-checked; 23 of the 157 lie in the snapshot's `library/` + `platform/python/`). Base2
  therefore carries every concurrent edit, W6's newer `e1` / `lbf` / `lbf_stale` included.
- Base2 vs the tree, with LLVM IR, serialized IR, COMGR `.text` and the whole code object: **100 / 100 existing
  builds byte-identical (400 files; 87 q38 + 13 large-batch)**. The 14 new-only builds are the fix-round
  `fw_*_scoped` labels and the `irops` heads and toys.
- Must-fail mutants: **D** (default backend direct; the code-object half) caught, 7 / 100 stay identical (the
  kernels the direct compiler refuses, which fall back to COMGR). **F** (op order in `mailbox.iod_id`; the IR half)
  caught, 11 builds differ.
- Targeted tests of the snapshot with the direct library: 507 passed, 95 skipped (GPU).

### 5. Corrections and clocks (issues 6, 7)

- Section 1's claim is corrected in place: `runtime/comgr.py` imports the facade module when it loads.
- **Clocks (DESIGN 9.1):** `rt_clock.py` reads the GPU's SCLK through amdsmi at the end of every timed round,
  outside the timed events (the container's amdsmi package sits in the ROCm wheel: alola rule 199).
  `wpr3_fix_analyze.py` drops rounds below 0.97 x the step's median and recomputes the verdict. In every
  fix-round job the round clocks stayed within 0.998-1.002 of the median, so **0 rounds were dropped**. Bus and
  GRES are recorded in every job (21 / 21 / 5 above); amd-smi samples are timestamped now.
- The first runs had no per-round windows. Their `amdsmi.jsonl` samples (about one per 2 s, no timestamps) peak at
  1.000-1.002 of the median busy SCLK. Lower samples (down to 0.58) are ramp samples between steps and cannot be
  mapped to rounds, so the first runs stay unfiltered; their verdicts are now backed by the clock-checked runs above.
  g1's missing bus stays missing.

### 6. Remaining

Before `RUST_FRAMEWORK_DIRECT_FAMILIES` is booked:
- compliant n01-09 steps for fw B3 r4s and fw MoE r1;
- D0.6's in-situ `layers=1` test.

The other open items of sections 6-7 are unchanged. The Q2, probe, CCL and lookback verdicts still rest on n07-17
alone.
