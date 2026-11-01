# WP0a: ATOM-truemono, the `ATOM_Q38_TRUE_MONO` skeleton and the `layer` hook

Status: done, gate met (2026-10-30); verified PARTIAL, then fixed (see "Fix round" at the end: it corrects the ISA
non-reproducibility facts, replaces the variant-set ISA gate and the served-path claim below, where marked).
Ratios only; absolute times are in `hackathon-prep/private-perf/qwen38-true-mono/probes/WP0a.md`. Jobs: `68217479`
(gate, smci355-ccs-aus-n07-17), `68217581` (ISA determinism, n07-17), `68217480` (e2e, smci355-ccs-aus-n01-09);
fix round `68218140`, `68218067`, `68218220`, `68218460`, `68218461`.

## What was built

**`dev/ATOM-truemono`** is a plain copy (no `.git`, no `__pycache__`) of the tree the baseline staged: stage
`atom-rkrepro` = `dev/ATOM`'s hybrid working tree (`352229d6` + the rocKE / sched / numa / head files). `dev/ATOM` had
moved since then only in `recipes/Qwen3.8-Flash-Next-rocKE.md`; that file was restored from the stage tarball, so the
pristine copy packs to the stage's exact manifest (1987 files, every sha256 equal).

On top of it (host side only, nothing under `atom/models/qwen4_exp_mono/kernels/`):

| file | what |
|---|---|
| `atom/models/qwen4_exp_mono/true_mono.py` (new, stdlib only) | the switch `ATOM_Q38_TRUE_MONO=off\|layer\|tail\|pass\|round\|loop` (unset/empty = `off`, any other value raises); `layer_launcher(launch, kind, rows, experts=, seq=, group=, where=)` returns the hybrid's own launcher object when off and imports nothing else; `announce()` logs the mode at install |
| `atom/models/qwen4_exp_mono/truemono/hook.py` (new) | the `layer` hook: a *layer provider* `ATOM_Q38_TRUE_MONO_LAYER=<module>:<callable>` (default `kernels.gfx950.qwen38_mono.truemono_layer:provider`, the rocKE module WP3 is to write) is called as `provider(LayerRequest(kind, rows, seq, group, experts, where), hybrid_launcher)` and returns a launcher of the hybrid ABI `(step counter, slot, scratch, *LAYER_ARGS[1:], stream=)` or None. A refused width, a provider that does not import and the granularities not built yet (`tail` WP4B, `pass` WP4, `round` WP5c, `loop` WP12) keep the hybrid's launcher, logged once each (per-width fallback, DESIGN 7.4); `ATOM_Q38_TRUE_MONO_STRICT=1` raises `TrueMonoRefused` instead (for gates). `ROUTES` records what served each (runner, kind, variant) |
| `atom/models/qwen4_exp_mono/truemono/slots.py` (new) | the round's instance names and tag slots (DESIGN 6.2): `target.<0..47>` 0-47, `draft.<j>` 48-50, `tail.<f>` 52-55, `front.<f>` 56-59, `meta` 60; `round_instances(k)`, `validate()` (disjoint, below 63) |
| `runner.py`, `group_runner.py`, `draft_runner.py` | each `build_layer(...)` call is wrapped in `true_mono.layer_launcher(...)`: target single / seq / grouped widths and the drafter's QSA layer (step 0 and later steps) |
| `dispatch.py` | `true_mono.announce(logger)` after the config check |
| `tests/test_qwen4_exp_mono_truemono.py` (new, 43 tests) | off = the build itself and imports nothing (subprocess check), bad mode raises, no true-mono file under or referenced from `kernels/`, every host `build_layer` call goes through the switch (AST), provider served / refused / missing / non-callable / strict, unbuilt modes, variant names = `sched` / `extern_stages`, slot plan incl. a reused-slot mutant, `LAYER_SLOTS` = the kernels' |

Stage naming (DESIGN 8.1): `hackathon-prep/qwen38/truemono/stage_atom.sh <suffix>` packs ATOM-truemono as
`atom-truemono-<suffix>` (an `atom*` stage: `q38_job.sh` mounts it at `/atom`) and makes `hb-atom-<suffix>` a
server-side byte copy of `atom-rkrepro` (the pinned hybrid, reached with `Q38_ATOM_DIR` + `PYTHONPATH`). The e2e job
below ran on exactly these two stages.

Probe code: `hackathon-prep/qwen38/truemono/WP0a/` (`job_gate.sh`, `keys_isa.py`, `keys_cmp.py`, `nondet.sh`,
`numerics.py`, `num_cmp.py`, `wp0a_providers.py`, `rig_time.py`, `e2e.sh`, `e2e_sum.py`, `mut_unit.py`, `sync.sh`).

## Gate results

| gate | result |
|---|---|
| tree diff on the node (`/atom` = atom-rkrepro vs the copy) | 5 files added, 4 host files changed, 0 removed, 0 under `kernels/` |
| ATOM CPU tests, both trees (`tests/test_qwen4_exp_mono_*.py`, `test_mono_*.py`, `test_qwen4_exp_mtp.py`, q38 container) | hybrid 577 pass / 6 skip; copy 620 pass (the same 577 + the 43 new) / 6 skip; 0 fail; no test changed outcome |
| unit-test mutation check (`mut_unit.py`, 12 mutants of switch / hook / slots / call sites / kernels placement) | 12 of 12 fail the new test file, the unmutated copy passes (local and on the node) |
| build keys, plain env, all 48 layer builds (3 kinds x r1-r4, r2s-r4s, r4g2, r6g2, r8g2, r6g3, r9g3, r12g3, r8g4, r12g4, r16g4) | the `keyed` tuple, FlyDSL's JIT key, every full cache key and all 48 disk-cache entries identical |
| build keys + ISA, hybrid ALL env (`SCHED=best`, rocKE fused MoE + B3 + B3-prologue functions, `EXTERN_MISSING=mono`, rocKE head), all 48 builds | keys, code objects (ELF, `.text`) and resources identical; 25 disk entries identical (extern-linked builds are never disk-cached) |
| ISA, plain env | first run: 47 of 48 `.text` identical, `qsa` r9g3 differed with identical keys. **Corrected in the fix round:** 5 of the 48 builds (not 2) compile to more than one binary in EITHER tree, the mix depends on process context, and the variant-set comparison used here is not a sound gate. With the pooled gate (24 compiles per build and tree) the copy PASSES 48 of 48 (and 14 of 14 served builds in the ALL env) |
| mutant M-key (a comment appended to `kernels/layout.py`) | every key differs, every `.text` identical (`key-only` as expected); the identity gate FAILS as it must |
| mutant M-isa (`HEAD_SLEEP` 64 -> 65 in `kernels/kernel.py`) | every key differs; `.text` differs for exactly the builds starting with A1 (delta, qsa) and not for delta_ple (starts with P1); the identity gate FAILS as it must, also with variant sets |
| numerics, rig (48 layers, synthetic weights, cells r1, r4, r4s, r8g4; fingerprint of every layer output and every state pool) | bitwise equal to the hybrid tree: a second hybrid process, the copy with the switch unset, `layer` + passthrough provider, `layer` + wrapping provider, `layer` without a provider (fallback), `round` (fallback); in the ALL env: the copy unset and `layer` + passthrough equal the hybrid. Every arm deterministic over two steps. ALL equals plain bitwise in all four cells. `layer` + strict without a provider refuses every cell |
| numerics mutants (must fail) | `mutant_prev_weights` (layer table row off by one): 46-48 of 48 layer outputs differ; `mutant_kind` (the delta_ple slot runs the delta body): 47 of 48 differ, from layer 1 (the PLE layer) on; the gate run as an equality rule FAILS on both. (`mutant_prev_weights` was itself not deterministic here, an artifact of its own state; fixed in the fix round) |
| served path (e2e, ALL env) | the copy unset and `layer` + passthrough serve with mono on, rocKE head on, 0 tracebacks; the hook engaged 15 (runner, kind, variant) builds per server: target delta / delta_ple / qsa at r4s, r8g4, r12g4, r16g4 and the drafter's qsa r4s, r4, r1 |

## Timing (DESIGN 9.1)

**Rig, the 48-layer pass** (ATOM-truemono, ALL env, every layer its own weights = cold, n07-17, unstamped): arms H
(switch off), TL (`layer` + a provider returning its own Python launcher, strict), H2 (second graph of H), RC (the same
builds compiled again into a fresh cache: a real compile per kind). Numerics first: all four bitwise equal from the
same state. 20 rounds, arm order shuffled per round, 5 graph replays + 3 eager passes per arm per round.

| cell | metric | TL / H median [min-max] | control band (H2, RC) | verdict |
|---|---|---|---|---|
| r4s (K=3 verify) | graph replay | 0.9996 [0.9972-1.0043] | 0.9966-1.0062 | NEUTRAL |
| r4s | eager pass | 1.0010 [0.9940-1.0029] | 0.9953-1.0041 | NEUTRAL |
| r1 (K=0) | graph replay | 0.9992 [0.9944-1.0036] | 0.9945-1.0027 | NEUTRAL |
| r1 | eager pass | 0.9999 [0.9963-1.0035] | 0.9969-1.0086 | NEUTRAL |

The eager 48-layer pass takes 3.35x (r4s) and 5.07x (r1) the graph-replayed pass in this harness: eager FlyDSL
launches are host-bound. The hook does not change that.

**E2E** (K=3, CONC 1, GSM8K 16 prompts x 2048 tokens, 4 pairs, one fixed arm permutation rotated per pair so that
every arm takes every position, one server per arm-run, n01-09 with one busy neighbour of ours): H = the pinned
hybrid (`hb-atom-w2-wp0a`), H2 = H again, T = the copy (switch unset), TL = the copy, `layer` + passthrough, strict.
All four in the hybrid ALL env.

| arm | ms per verify round vs H, median [min-max] | token-weighted tok/s vs H | acceptance vs H | GSM8K right (64) | texts identical to H's (64) | verdict |
|---|---|---|---|---|---|---|
| H2 (A-vs-A) | 1.0030 [0.9990-1.0051] | 1.0064 | 1.008 | 62 | 3 | control |
| T | 0.9992 [0.9963-1.0062] | 1.0020 | 1.000 | 64 | 3 | NEUTRAL |
| TL | 1.0026 [0.9971-1.0062] | 1.0066 | 1.008 | 61 | 1 | NEUTRAL |

H scored 64 of 64. TL's 61 of 64 is not identical accuracy: it sits 3 below H and 1 below the A-vs-A control H2
(62). These are 4 repeats of the same 16 prompts on a server that is not deterministic, so the difference is within
noise (Fisher exact, treating the 64 as independent: p = 0.24 against H, 1.0 against H2), but it is not shown to be
zero. The served path is not token-deterministic run to run: H against H2 gives 3 of 64 identical texts
(same prompts, greedy), first divergence after 32-455 characters. The same holds for plain mono without any knob in
the baseline's own runs (`tmb-k3`, `tmb-k0`: 0-4 of 64 identical between pairs). **Corrected in the fix round:** the
original sentence here said the source "is outside the layer kernels". That is not shown. The fix round widened the
rig evidence (every served width, three contexts, real weights, separately compiled processes, and served builds
that compile to one binary), and the layer pass was deterministic in every case. The source is still not isolated,
and the layer kernels on served inputs are not excluded.

## What D0 needs from WP0a (the answer)

1. **A copy to edit that is the baseline.** ATOM-truemono is byte-identical to the pinned hybrid (`atom-rkrepro`)
   except for the hook; with the switch unset it builds the same keys and the same layer binaries and serves at
   parity (T: NEUTRAL inside the A-vs-A band). P1, P3 and P9 can edit it; WP4h starts from its call sites.
2. **Two identity levels for every later ATOM-truemono edit.** ATOM keys every mono build by the digest of all of
   `kernels/` (`sources.SOURCES`): a host-only edit keeps keys and ISA identical (this WP), but any kernel-side
   knob (P1's `xl` / `xhold`, P3's spread, both in `kernels/`) changes every build key even when off. Their
   "off = hybrid" gate is `keys_cmp.py --expect key-only` (ISA and resources identical, keys not), the form the
   M-key mutant validated.
3. **(Replaced in the fix round; see "Fix round", item 3.)** ~~ISA identity needs variant sets~~. The hybrid's own
   builds are not bit-reproducible (5 of 48 plain-env builds), and a recompiled control can be a different binary.
   The ISA gate is NOT set equality over a few compiles. Use build-key identity for host-only edits. For
   kernel-side knobs, use the pooled distribution gate (`WP0a-fix/isa_pool.sh` + `isa_dist.py`).
4. **The Path-B / WP3 attachment point exists and is inert.** A rocKE `layers=1` kernel enters through one provider
   module (`kernels/gfx950/qwen38_mono/truemono_layer.py:provider` in rocKE `Q/`, on PYTHONPATH as the extern
   stages are), per runner (target single / seq / grouped, drafter), per kind and variant, with per-width fallback.
   Graph-captured, the hook costs nothing (TL NEUTRAL, inside both controls); in eager use (the drafter) a provider
   that wraps a launcher in Python is still NEUTRAL. The served path engaged it at all 15 (runner, kind, variant)
   builds of a K=3 server.
5. **A per-layer bitwise harness with proven teeth.** `numerics.py` + `num_cmp.py` compare every layer output and
   every state pool of the 48-layer pass across processes and catch a layer-table row off by one and a wrong body
   for a kind at the first wrong layer; WP3's "bitwise vs the mono layer" gate and D0's P1 arms can use it as is
   (providers are the arms).
6. **The noise floor for rig A/B at pass level:** controls span 0.9945-1.0086 of the pass (graph 0.9945-1.0062,
   eager 0.9953-1.0086) over 20 rounds. A P1 arm must clear the controls' extreme by 0.002 to count; the stack loss
   P1 has to reproduce (1.03-1.09 per layer on `best`) is far outside it.
7. **Bitwise references are offline only.** In the rig the hybrid ALL pass equals plain mono bitwise at r1, r4,
   r4s, r8g4 (the fix round adds r12g4, r16g4 and real weights), so either is the per-layer reference. Where the
   served divergence comes from is open; it is not shown to be outside the layer kernels. Served e2e text is not deterministic even hybrid-vs-hybrid, so
   Tier G "round bitwise vs the hybrid round" (DESIGN 7.8 gate 4) cannot be checked through a server: it needs an
   offline driver with deterministic prefill (or a captured-input replay), and e2e gates stay on round time,
   acceptance and accuracy within the A-vs-A spread.
8. **The tag plan is checked code** (`truemono/slots.py`): WP1's `validate_stack` over the round schedule can import
   it.

Open (not WP0a's gate, recorded for WP4h / Tier G): the source of the served path's run-to-run token divergence.

## Verification (independent verifier, 2026-10-30)

Verdict: **PARTIAL**. The DESIGN 8.2 gate for WP0a ("ATOM CPU tests pass; hook unset = ISA-identical build keys")
is met. The numerics and timing results hold. The report's description of the ISA non-reproducibility is wrong,
though, and so is the variant-set ISA gate it recommends to D0 (point 3). One sentence about where the served
output's non-determinism comes from is stated more strongly than the evidence supports.

### What was re-checked and holds

- **The copy is the copy.** The current `dev/ATOM-truemono` matches the `w2-wp0a-tm` and `atom-truemono-w2-wp0a`
  stage manifests file for file (1992 files, all sha256 equal after CRLF->LF; no one has edited it since). The four
  host diffs against `dev/ATOM` only wrap `build_layer(...)` calls in `true_mono.layer_launcher(...)` with the same
  build arguments, plus an import and `announce`. `layer_launcher` returns the `launch` object itself when the switch
  is off. `tree_diff.json`: 0 files under `kernels/`, 0 removed.
- **CPU tests and unit mutants.** `unit_H.log`: 577 passed, 6 skipped. `unit_T.log`: 620 passed, 6 skipped. Both
  rc=0. `mut_unit.log`: the unmutated copy passes 43/43, and each of the 12 mutants fails 1-10 tests.
- **Build keys (the DESIGN 8.2 gate).** `keys_cmp_plain.json` and `keys_cmp_all.json`: `keyed`, FlyDSL key, memory
  keys and disk entries are identical for 48/48 builds in both envs. Verifier job 68217853 recomputed the `keyed`
  sha for all 48 builds in 216 fresh processes; the keys were identical across trees, with one key per build. The
  key-mutant (M-key) and ISA-mutant (M-isa) gate runs FAIL as they must (`keys_gate_mk.log`, `keys_gate_mi.log`;
  6 builds each, r1/r4s).
- **Numerics ran on hardware and the mutants fail.** All arms ran on n07-17. The T arms import
  `/job/stages/w2-wp0a-tm/atom`, the H arms import `/atom/atom`, so they really are two trees. I recompared the raw
  fingerprints in `num/*.json` myself:
  - Copy and passthrough arms vs H_plain or H_all: 0/99 items differ in every cell. That covers H_plain2, T_plain,
    T_layer_pass, T_layer_wrap, T_layer_noprov, T_round, T_all and T_all_layer_pass.
  - H_all vs H_plain: 0/99 in all 4 cells.
  - `ROUTES` shows the provider really served each arm: "provider wp0a_providers:..." for the layer arms, and the
    fallback reason for noprov and round.
  - T_strict_noprov refused every cell.
  - Mutants: `mutant_kind` differs in 86-97 of 99 items, `mutant_prev_weights` in 84-98 of 99; `num_cmp` FAILs on
    both.
  - Minor: `mutant_prev_weights` also shows up as "not deterministic". That comes from its module-level `_prev`
    dict carrying over between the two steps. It is an artifact of the mutant, not of the harness.
- **Timing discipline.**
  - Rig (`rig_time.py`): arm order shuffled per round with a seed, 20 rounds; two controls, H2 (a second graph) and
    RC (a real recompile: the `build_layer` cache is cleared and a fresh FlyDSL cache dir is used); numerics are
    checked before timing; the WIN/LOSS rule is "beyond both controls' extremes by 0.002 and >= 3/4 of rounds";
    unstamped; node and SCLK recorded.
  - E2E: 4 pairs, rotations of one seeded permutation, so every arm takes every position; one server per arm-run;
    H2 is the A-vs-A control. The job log shows each arm's package path (H/H2 = `hb-atom-w2-wp0a`, T/TL = `/atom` =
    the copy). TL logged 15 provider-served builds per server; H/H2/T logged 0 true-mono lines; no tracebacks.
  - The NEUTRAL verdicts follow from the logged ratios.
  - Minor: TL's GSM8K was 61/64 against H's 64 and H2's 62. On 16 prompts x 4 pairs with non-deterministic serving,
    this is within the noise.

### What does not hold

1. **Non-reproducible ISA affects at least 4 of the 48 builds, not 2.** Verifier job **68217853** (n01-09, stage
   `w2-vWP0a-solo`, script `hackathon-prep/qwen38/truemono/WP0a-verify/solo.sh`) compiled each of the 48 plain-env
   builds alone in a fresh process. Each build ran twice per tree, with 6 more solo runs per tree of qsa r9g3 and
   delta_ple r3s, for 216 processes, using `AMD_COMGR_CACHE=0` and fresh caches. Pooled with WP0a's nd job and gate
   chunks (at least 6 compiles per build per tree):

   | build | H variants (count) | T variants (count) | in WP0a's report |
   |---|---|---|---|
   | qsa r9g3 | 2 (23:5) | 2 (25:3) | yes |
   | delta_ple r3s | 2 (9:3) | 2 (8:4) | yes |
   | **qsa r12g3** | 2 (12:2) | 2 (13:1) | **no, reported as "one variant every time"** |
   | **delta_ple r2s** | 2 (4:2) | **1** (6) | **no, reported as "one variant every time"** |

   The report's "the other 46 builds give one variant every time", and rule 130's "46 stable", are false.
2. **"Not reproducible" is not just in-process compile history, and the mix depends on compile context.**
   - Solo compiles still produce two variants for delta_ple r3s (11:5).
   - The minority variants of qsa r12g3 and delta_ple r2s appeared only in solo processes, never in 24 and 8
     chunked compiles.
   - qsa r9g3 gave one variant in all 32 solo compiles; its second variant appears only in multi-build processes.

   So the variant distribution depends on process context, and variants are not sampled from one fixed set.
3. **The variant-set gate (`keys_cmp.py --variants`, `nondet.sh`: set(H) == set(T)) is statistically unsound at
   3-8 compiles per tree.**
   - On the pooled data it FAILS for delta_ple r2s: H has a variant that T never showed in 6 compiles, even though
     the two trees have identical full build keys. That is a false failure caused by sampling.
   - The verifier job's own solo-only set comparison flagged 2 builds as "H != T sets" for the same reason.
   - The report's "48 of 48 with variant sets" is an artifact of which compiles happened to be sampled. It is not a
     robust property.
   - `nondet.sh`'s header says the verdict is "T's variants within H's", but its code checks set equality in both
     directions.
4. **Point 7's "the source is outside the layer kernels" is not shown.** The rig's evidence is narrower than the
   served case:
   - It shows that the 48-layer pass is bitwise deterministic for identical inputs, synthetic weights, one
     context, and cells r1/r4/r4s/r8g4.
   - Serving also uses r12g4 and r16g4, real routing and growing contexts, and it recompiles the extern-linked
     builds at every server start.
   - Given finding 2, different server starts can run different binaries.

   The divergence source is open, as the report's last line says. The "outside the layer kernels" claim should be
   read as unproven.

### What this changes for D0 (replacing point 3 of "What D0 needs")

- **Host-only edits.** Use full build-key identity: `keyed`, FlyDSL key, memory keys and disk entries. With an
  unchanged compiler and keys, the binary distribution is the same by construction, so this is the decisive gate.
  WP0a passes it 48/48 in both envs. Any ISA check here is a sanity check, not the gate.
- **Kernel-side knobs, where the keys change by design.** "Off = hybrid" needs a distribution-level ISA test:
  - pool solo and chunked compiles, at least 8 per tree, per build;
  - pass if every T variant lies within the pooled H variants, and flag a build only when a T variant is absent
    from a much larger H pool;
  - or compare after a canonicalising disassembly.

  Set equality on a few compiles must not be used. M-isa-style mutants still fail such a test, because their
  variants are new.
- **Recompiled controls.** "A recompiled control can be a different binary" holds for at least 4 builds. None of them
  is among the 15 builds a K=3 server makes, but all 4 are in the build matrix. That supports RC as a real control.

The CPU tests, the key gate, the numerics with mutants and the timing all stand as WP0a's deliverables. D0 must not
reuse the variant-set ISA gate as specified. Rule 133 of the alola-cluster skill records the corrected facts. The
verifier's summary JSON and log are in
`hackathon-prep/private-perf/qwen38-true-mono/probes/WP0a-raw/verify/`; they contain no absolute performance numbers.

## Fix round (2026-10-30)

This section answers each of the verifier's findings. Everything was re-run on the cluster; no git state changed.
The body above was corrected in place where it was wrong; each such spot is marked "Corrected in the fix round" or
"Replaced in the fix round". Probe code is in `hackathon-prep/qwen38/truemono/WP0a-fix/` (stage `w2-wp0a-fix-h`).
Raw results are in `hackathon-prep/private-perf/qwen38-true-mono/probes/WP0a-raw/fix/`, and the logs are
`hackathon-prep/logs/q38/w2-wp0a-fx-*.out`.

| job | node | what |
|---|---|---|
| `68218140` (`w2-wp0a-fx-pool`) | n01-09 | 2,300 compile-only processes, `isa_pool.sh`: pooled ISA distributions (plain env, both trees), the two mutant trees, the source arms, the 14 served builds in the ALL env, and the gates |
| `68218067` (`w2-wp0a-fx-num`) | n01-09 | rig numerics, 19 arms, each in its own fresh FlyDSL cache: six cells, three contexts, real weights, the fixed mutant |
| `68218220` (`w2-wp0a-fx-real`) | n01-09 | real-weight dump (a K=3 mono server of the pinned hybrid) and 6 real-weight numerics arms |
| `68218460` / `68218461` (`w2-wp0a-fx-proto7` / `-proto1`) | n07-17 / n01-09 | a deterministic compile protocol: 768 solo compiles per node (H, T, MK, MI) |
| `68218055` (`w2-wp0a-fx-smoke`) | n01-09 | a small smoke run of `isa_pool.sh` (it does not count toward any result) |

### 1. How many builds compile to more than one binary (finding 1)

There were 24 compiles per build and tree (H = `atom-rkrepro`, T = the copy), with the same contexts for both: 12
solo (one build in a fresh process), 6 in WP0a's 8 gate chunks in gate order, and 6 in random 6-build chunks in
random order. All used `AMD_COMGR_CACHE=0` and fresh caches. **5 of the 48 plain-env builds give two `.text`
variants in either tree. The other 43 give one in all 48 pooled compiles.**

| build | H solo | H gate chunks | H random chunks | T solo | T gate chunks | T random chunks |
|---|---|---|---|---|---|---|
| `delta_ple` r2s | 6 : 6 | 6 : 0 | 6 : 0 | 9 : 3 | 6 : 0 | 6 : 0 |
| `delta_ple` r3s | 6 : 6 | 4 : 2 | 5 : 1 | 8 : 4 | 4 : 2 | 6 : 0 |
| `delta_ple` r4s | 12 : 0 | 5 : 1 | 5 : 1 | 12 : 0 | 6 : 0 | 6 : 0 |
| `qsa` r9g3 | 12 : 0 | 4 : 2 | 6 : 0 | 12 : 0 | 4 : 2 | 5 : 1 |
| `qsa` r12g3 | 10 : 2 | 6 : 0 | 6 : 0 | 10 : 2 | 6 : 0 | 6 : 0 |

(Counts are majority : minority variant, in the same variant order within each row.) `delta_ple` r4s is new; no
earlier sample had caught it. It is the plain-env K=3 verify width, so it is a served build when mono is served
without the ALL knobs. Rule 130's "46 stable" and WP0a's "the other 46 give one variant every time" were false.
Rule 130 is corrected and points to the new rule 137.

The **14 builds a K=3 server makes, in the hybrid ALL env** (extern-linked: delta / delta_ple / qsa at r4s, r8g4,
r12g4, r16g4, and qsa r4 and r1) gave one binary each over 18 compiles per tree (12 solo plus 6 one-process runs of
all 14 in server order). H and T gave the same binary for all 14.

### 2. It is not compile history; what it is (finding 2)

- **Context changes the mix, in both directions.** Two builds show their minority variant only in solo processes
  (`delta_ple` r2s, `qsa` r12g3). Two show it only in multi-build processes (`delta_ple` r4s, `qsa` r9g3). One
  shows it in both (`delta_ple` r3s). Any ISA sample must therefore cover solo and chunked processes, in the same
  proportions for both trees.
- **The source is process-level randomness.** Arms on H, 16 solo compiles per build, for the 4 builds known
  before and 2 stable controls:

  | build | base (as served) | `PYTHONHASHSEED=0` | ASLR off (`setarch -R`) | both pinned |
  |---|---|---|---|---|
  | `delta_ple` r2s | 2 variants | 2 (11 : 5) | 2 (12 : 4) | **1** (16) |
  | `delta_ple` r3s | 2 variants | 2 (14 : 2) | 2 (12 : 4) | **1** (16; and 8 of 8 in its gate chunk) |
  | `qsa` r12g3 | 2 variants | 2 (9 : 7) | 2 (13 : 3) | **1** (16) |
  | `qsa` r9g3 | 2 variants (chunked only) | 1 | 1 | **1** (16; and 8 of 8 in its gate chunk) |
  | `delta` r1, `qsa` r4s (controls) | 1 | 1 | 1 | 1 |

  Pinning either one alone leaves the variation; pinning both removes it. The variation therefore comes from
  hash-seed- and address-dependent ordering in the compile, not from the build key.
- **The protocol makes compiles repeatable, but only in place.** Jobs `68218460` and `68218461` compiled every build
  of H, T, MK and MI 4 times under `PYTHONHASHSEED=0` plus `setarch -R`, one build per process. In every tree on
  each node, each of the 48 builds was ONE binary (4 trees x 2 nodes, 48 of 48 each). But WHICH binary still
  depended on where the compile ran:
  - Across nodes: T's `qsa` r12g3 (same path, `/job/stages/w2-wp0a-tm`) differed between n01-09 and n07-17. MK's and
    MI's `delta_ple` r2s and r3s differed too. H matched on both nodes.
  - Across trees on one node: H and T differed for `qsa` r12g3 on n01-09, and H and MK for `delta_ple` r2s (n01-09)
    and r3s (n07-17), all with identical sources for those builds.

  Every binary the protocol produced is one of the base pool's variants. So the protocol can reproduce a binary
  for bisection or debugging, but it cannot be a cross-tree identity gate: its strict cross-tree comparison
  false-FAILs. The first run of the `proto` step expected cross-tree identity, and its recorded expectations fail
  for exactly that reason. The step now checks in-tree reproducibility and only reports cross-tree differences.
- **`FLYDSL_DUMP_IR=1`** (FlyDSL 0.3.4.1 runs the pipeline stage by stage and dumps 23 files) gave one binary per
  build over 12 solo compiles each. Every dumped stage, from the origin MLIR through the LLVM IR to the ISA, was
  identical across compiles. The dump path does not show the divergence, so it cannot be used to locate it.

### 3. The ISA gate that replaces variant sets (finding 3)

- **Host-only edits (anything outside `kernels/`): full build-key identity is the gate.** That means the `keyed`
  tuple, FlyDSL's key, the memory keys and the disk entries (`keys_cmp.py --expect identical`; WP0a passes 48/48
  plain and ALL). With equal keys and the same compiler, the binary distribution is the same by construction, so
  an ISA comparison is only a sanity check here.
- **Kernel-side knobs (keys change by design): `WP0a-fix/isa_pool.sh` + `isa_dist.py`, gate
  `REF:TEST:ARM:KEYS`.**
  - Every TEST variant must be among REF's pooled variants, with at least 8 compiles per build and tree (the job
    uses 24), drawn from the same mix of solo and chunked processes.
  - A TEST variant REF has not shown is PENDING, not FAIL. REF is then compiled again (48 solo plus 24 chunked for
    that build), and only a variant still missing after that FAILs (`--final`).
  - REF-only variants are reported, with the chance that TEST missed them by sampling.
  - `KEYS=same` requires one build key across both trees; `KEYS=differ` requires a changed key.
- **Results on the pooled data:**
  - T vs H, plain: **PASS 48/48**.
  - T vs H, ALL env served builds: **PASS 14/14**.
  - Key-only mutant MK (a comment in `kernels/layout.py`): **PASS 48/48**.
  - ISA mutant MI (`HEAD_SLEEP` 64 -> 65): **FAILs exactly the 32 delta + qsa builds**, and PASSES the 16
    delta_ple builds, whose ISA it does not change. Both expectations hold.
  - On the same data, set equality would have false-FAILed T vs H on `delta_ple` r4s (H 22 : 2, T 24 : 0) and MK on
    2 builds.
- **How often each rule false-FAILs a correct tree** (bootstrap from the pooled 48 compiles per build; chance that
  at least one of the 48 builds false-FAILs):

  | rule | 3 per tree | 4 per tree | 8 per tree |
  |---|---|---|---|
  | set(H) == set(T) (WP0a's gate) | 0.94 | 0.94 | 0.94 |
  | T within H, H pool 16 | 0.20 | 0.23 | 0.37 |
  | T within H, H pool 24 | 0.09 | 0.12 | 0.20 |
  | T within H, H pool 40 | 0.014 | 0.021 | 0.027 |

  The within rule's remaining risk is why a missing variant is resampled before it FAILs, rather than failing on
  the first pool.
- **The tools are tested.** `selftest_isa_dist.py` passes 10 of 10 checks on synthetic pools, and all 8 mutants of
  `isa_dist.py`'s rules fail it (`--mutants`). `keys_cmp.py --variants` now reads the pooled summary, and its help
  says the pooled gate is `isa_dist.py`.

### 4. `nondet.sh` header vs code (finding 4)

The header now says what the code does. It reports set inequality (descriptive only) and a new `t_outside_h`
(T variants H never showed), states that neither is a gate at its sample size, and points to `isa_dist.py`.

### 5. The served path's non-determinism (finding 5)

The claim that the source "is outside the layer kernels" is retracted, in the body (E2E paragraph and D0 point 7)
and here. What the fix round adds:

- **The layer pass is bitwise equal and deterministic across separately compiled processes in every case tried.**
  Each arm compiled into its own fresh FlyDSL and comgr cache (in WP0a, H_plain2 had reused H_plain's binaries). The
  cases cover:
  - all six served target widths: r1, r4, r4s, r8g4, and the new r12g4 and r16g4 (the rig now sizes the group
    runner for the widest grouped cell);
  - contexts 1,027, 8,192 and 32,768;
  - synthetic weights, and real weights: every layer on its kind's real dumped layer (0 delta, 1 delta_ple,
    3 qsa), 48 of 48 layers real;
  - the plain and the ALL env.

  That is 72 equality comparisons in job `68218067` plus 24 in job `68218220` (99 fingerprinted items per cell),
  with 0 items differing. Every arm was deterministic over two steps. ALL equals plain bitwise, with synthetic and
  with real weights.
- **The ALL env's 14 served builds compile to one binary** (item 1). So with the ALL knobs, different server starts
  should run the same layer binaries; this is not proven for every in-server compile context.
- **In the plain env, served `delta_ple` r4s has a second binary** in multi-build processes (2 of 12 in H). A
  plain-mono server (like the baseline's `tmb-k3` runs) compiles many builds in one process, so different server
  starts can run different r4s binaries. Whether the two binaries compute different bits was not measured.
- **What is still not covered:** real hidden states and routing (the rig feeds random inputs), contexts that grow
  step to step, the drafter's own path, prefill, the rocKE head, sampling and scheduling. The source of the served
  divergence stays open for WP4h / Tier G (the offline deterministic driver from D0 point 7). The layer kernels on
  served inputs are not excluded.

### 6. GSM8K wording (finding 6)

The E2E paragraph now states TL's 61 of 64 against 64 (H) and 62 (H2, A-vs-A). That is within noise for 4
repeats of 16 prompts on a non-deterministic server (Fisher exact p = 0.24 vs H, treating the 64 as independent),
but it is not identical accuracy.

### 7. The `mutant_prev_weights` artifact (finding 7)

The provider (`WP0a/wp0a_providers.py`) now keys the weights it borrows by layer index. Each layer takes the
weights of the nearest earlier layer of its kind, and the first layer of a kind keeps its own, so every pass is
the same. Result: deterministic in 6 of 6 cells (it was 0 of 4). It still FAILs the numerics gate as it must: 46
of 48 layer outputs differ from layer 2 on (84-96 of 99 items). `mutant_kind` FAILs too: 47 differ, from layer 1.

### What D0 takes from WP0a now (replacing points 3 and 7 of "What D0 needs")

- **Point 3 (ISA identity).**
  - Host-only edits: build-key identity, which is decisive.
  - Kernel-side knobs (P1 `xl` / `xhold`, P3 spread): `isa_pool.sh` + `isa_dist.py --gate H:X:base:differ`.
    The pool must contain solo and chunked compiles, at least 8 per tree (24 used here), and a missing variant is
    resampled before it FAILs. Expect MI-style mutants to FAIL exactly the changed kinds.
  - Never use set equality over a few compiles.
  - A recompiled control really can be another binary, for at least 5 builds (and `delta_ple` r4s is served in
    plain mono).
  - To pin one binary for bisection, use `PYTHONHASHSEED=0` plus `setarch -R` in the same tree on the same node.
    It does not give identity across trees.
- **Point 7 (bitwise references).**
  - The rig's per-layer reference holds at every served width, at three contexts and with real weights. ALL equals
    plain bitwise.
  - The served divergence is unexplained and the layer kernels are not excluded. Tier G's round-level bitwise
    check needs the offline deterministic driver. Prefer the ALL env, whose served builds compile to one binary,
    or pin binaries when comparing server starts.

### Fix-round verdict

- WP0a's DESIGN 8.2 gate stands: CPU tests pass, and with the hook unset the build keys are identical.
- The ISA facts are corrected. The variant-set gate is replaced by a tested distribution gate that PASSES the copy
  (48/48 plain, 14/14 served ALL) and has teeth (MI FAILs exactly the 32 changed builds; MK PASSES).
- The served-path claim is retracted, with wider rig evidence.
- The GSM8K wording and the mutant artifact are fixed.
- Rule 137 of the alola-cluster skill records the new cluster facts, and rule 130's ISA bullet now points to rules
  133 and 137.
