# WP2c: QSA Q3-Q7 as rocKE stage functions (W8, wave 2, 2026-11-01)

D0 section 5 (wave 2b, WP2c) and DESIGN 5 / 7.8 / 9.1 / 12.3-12.4; PLAN decisions 13, 15, 16. This file gives ratios,
shares and counts only. Absolute device times and tok/s: `hackathon-prep/private-perf/qwen38-true-mono/wave2/WP2c.md`.
No git state was changed, no file was deleted, nothing in `dev/ATOM` or `dev/rocm-libraries` was touched; ATOM-truemono
was not edited (the ports link through its existing `ATOM_Q38_EXTERN_STAGES`).

**Verdict: gate met for the bitwise port (served L AND 32K / 128K / max, every narrow width, ties, mutants); the
port beats the FlyDSL stages in situ on the served stack once its catalogued knobs are climbed; N4a / N4b pass T0.**
**Fix round (2026-11-02, section "Fix round"):** every sub-1 % verdict re-timed with 4 separately loaded copies per
arm on both nodes, clocks sampled; the DESIGN 9.1 E2E rule applied (6 rotated pairs, A-vs-A twins): K=3 ms per round
WIN vs the consolidated stack (0.992-0.997) and the hybrid (0.936-0.939), K=3 request tok/s NEUTRAL vs the stack;
K=0 request tok/s WIN vs the stack (1.015) and the hybrid (1.116). The served r4s set becomes FlyDSL's Q3 / Q4 +
the rocKE Q5-Q7 (0.961, resolved better on both nodes); the r1 default loss at 128K (0.7-0.9 %) is real.

- **Port.** Q3, Q4, Q5, Q6, Q7 of the FlyDSL mono are rocKE stage functions (new package
  `kernels/gfx950/qwen38_mono/qsa2/`), linked in place of the FlyDSL stages for every narrow variant
  (r1-r4, r2s-r4s), plain and on ATOM's `numa=1` mailbox layout.
- **Parity by construction (decision 16).** Per stage and variant, the f32 / f64 operation multiset, every DPP /
  permlane / readlane, every LDS access, atomic and barrier of the rocKE function equals the FlyDSL stage's final
  `.ll` (21 of 21 stage-variant pairs checked); after the `fly` mailbox (section 3.2) the ISA memory forms match too.
- **Bitwise numerics.** 85 + 35 + 161 + 55 in-situ cells (the layer kernel with the functions linked vs the FlyDSL
  build from one snapshot and tag) are bitwise equal in every byte of the scratch (both halves), the QSA caches and
  the layer output: 7 variants, contexts 639 to 262,136 (the maximum), planted Q4 ties across chunk boundaries,
  `numa=1`, the served stack's schedule, every exact knob of the climbs. 14 must-fail mutants (9 numeric: `q5_scale`,
  `q5_cur`, `q6_den`, `q7_gate`, `q7_tail`, `q3_relu`, `q3_key_row`, `q4_rank`, `q4_tie`; 5 T0: `fuse_den`,
  `splitkv_norescale`, `splitkv_drop`, `tile_norescale`, `qdense_drop`) differ where they must (fix round: this line
  said 10).
- **Speed in situ (DESIGN 9.1, two controls, WIN rule).** Defaults on the served stack (`stackmoe` = the stack's
  schedule + its rocKE fused MoE): r1 0.999 (parity), r4s **0.980 WIN**. Climbed (bitwise): r1 **0.948**, r4s
  **0.959-0.964** of the FlyDSL stage's layer (accepted or confirmed sets; fix round: this line said 0.956-0.963, and
  0.956 was an arm the climb did not accept), the WIN repeated on the other node (section 6.3). Fix round, with every
  arm built 4 times as separate code objects (alola rule 215): r1 0.947-0.949, r4s 0.965 (all five functions) /
  **0.961** (FlyDSL's Q3 / Q4 + the rocKE Q5-Q7, now resolved better on both nodes).
- **End to end (decision 14, quick track, 3 passes, same job as the hybrid; section 6.4).** The consolidated full
  stack plus the functions with the climbed picks (`numa=1,qdense=3,prepoll=q5+q6+q7`, r1 `q5_kpre`, `qsa.r1`
  `hold.q7=64`): K=3 median ms per round **0.991** of the full stack (0.939 of the hybrid); K=3 tok/s per request
  per pass 0.970 / 1.002 / 0.999 (median 1.000: no gain; fix round: this line said 1.010, a pooled aggregate that
  one slow request moved); K=0 tok/s per request **1.017** of the full stack (1.114 of the hybrid); GSM8K correct
  within the arms' spread. A first job with `prepoll` only gave 0.990 / 1.013. These 3-pass runs had no A-vs-A, so
  no e2e WIN follows from them; the DESIGN 9.1 E2E rule is applied in the fix round (section "Fix round", F2).
- **N4a (qdense, P13) is a knob** of the port (levels 1 / 2 / 3, mutant), bitwise to ATOM's `qdense` lever at every
  level (section 4.2). **N4b** (fused split-KV softmax, Q6 removed) is built in three forms; all pass T0 (tolerance,
  the Q7 pack at 0.995-1.003 of the mono's drift vs the fp64 golden, 72-79 % of the pack codes unchanged) with
  must-fail mutants caught; in situ it wins against the FlyDSL chain on the plain base (r1 0.982) and on the served
  stack at r4s (0.971-0.975), but re-tested on the climbed exact set it is 0.9-1.5 % behind at r1 and within 0.004 at
  r4s (section 5): kept default-off; T2 / T4 (served KL, RULER) not spent on a form with no gain (section 9).
- **WaveScope ATT** (section 8.1): the climbed set removes a third of the traced wave's barrier-stall cycles.

## 1. What was built

| file (rocKE `library/kernels/gfx950/qwen38_mono/qsa2/`, new) | content |
|---|---|
| `lower.py` | device-function lowering for these functions (a subclass of `device_function._DeviceFunctionLowerer`): `q38.intr` (any LLVM intrinsic with immediates: `update.dpp` with row / bank masks, `maximum.f32`, `fma.f64`, `umax`), `q38.ifr` / `q38.yield` (runtime if with results, an LLVM phi), `q38.bin` (`ashr`), `q38.poll` (FlyDSL's poll shape: peeled first load, pause + reload loop, 32-bit tag compares); `arith.fma` as `llvm.fma` and f32 `arith.fmax` as `llvm.maximum` (the FlyDSL ops; knobs `fma` / `fmax`). No other rocKE build sees these ops |
| `layout.py` | the FlyDSL QSA layout from the ATOM checkout (`layout.for_rows`, `config`, `STATE` / `PAGE` literals), `qsa.py` / `qsa_fast.py` constants recomputed (`check_against_atom` compares them in the container: equal) |
| `ctx.py` | `QCtx`: rows, mailboxes (plain / `numa=1`; `fly` / `asm` / `irf` puts and polls), the mono's LDS arena offsets, buffer loads, FlyDSL `xshfl` / `xred` / `wave_max` / `wave_sum` / `natural_sum`, `mexp_fma`, `sigmoid_acc`, `e8m0_fp8`, `fp8_word`, `bf16_pair` |
| `q567.py` | Q5 / Q6 / Q7 (`qsa._q5`, `q6`, `_q7`; `qsa_dense._q5d` / `_q7d` under `qdense`), knobs `q5_kpre`, `q7_pp`, `kv_nt`, N4b `fuse` (`single`, `splitkv`, `tile`), `n4b_e` |
| `q34.py` | Q3 / Q4 (`qsa_fast.q3`, `q4`: f64 scores, cached-score batches, radix select, bitonic rank; `qsa_dense` Q3 key-only / Q4 publish), knobs `q3_batch`, `q3_switch`, `q4_chunks`, `q4_thr`, `q4_umax`, `q4_scan` |
| `knobs.py` | `QsaKnobs` (default = the FlyDSL stages), validation, suffix, the 14 test mutants |
| `build.py`, `gen.py` | the `stage_fn` ABI around a body; the generator (`python -m kernels.gfx950.qwen38_mono.qsa2.gen --out DIR [--knobs ...] [--variant-knobs ...] [--mutate M]`), sidecars with `stage_knobs`, `tolerance`, `atom_check`, `mailboxes: numa1` / `numa_hdr` |
| `library/tests/test_qwen38_qsa2.py` (new) | 32 CPU tests: knobs, layout, every stage and variant lowers with the ABI, the `q38.*` lowering, default text stable under off knobs, numa copies, every mutant changes its stage, a fused Q6 idles |

Harness (new): `hackathon-prep/qwen38/truemono/WP2c/` (`ab_qsa.py` bitwise in-situ A/B, `t0_qsa.py` tolerance gate,
`bench_qsa.py` 9.1 timing + stage stamps, `climb_qsa.py` knob climb, `ll_cmp.py` .ll parity, `wp2c_job.sh` phases
gen / ab / abx / abtie / absched / stress / t0 / isa / att / bench / climb / confirm, `sync_wp2c.sh`, `sync_qk.sh`,
`summarize_ab.py`); quick-track arm file `hackathon-prep/qwen38/truemono-quick/arms.d/w8c.sh` (new: `full3_w8c`,
`full0_w8c`, knob / per-variant knob / schedule-override plumbing).

**Existing builds are untouched:** no existing rocKE or ATOM file was edited (the package and the tests are new
files; ATOM links them through its existing extern mechanism, default off). Byte identity of existing builds holds
trivially; inside the package, every refactor was checked to leave the default functions' text equal (sections 3.2,
7).

## 2. Method (decision 16), per stage

1. **FlyDSL's final .ll and ISA** of the QSA layer at r1 and r4s (and all other narrow variants for the .ll) from the
   reverse toolkit's `rv-dump4` (`mono_qsa_*.pre.ll`, marked) and our own GPU-less `rv_dump` jobs (`isa3`, `isa4`).
2. **Port + parity.** `ll_cmp.py` compares, per stage, the FlyDSL stage region of the marked `.pre.ll` with the
   function's `.ll` by operation class; `rv_isa.py split` + `rv_diff.py` compare the linked code objects per stage.
   Bitwise numerics in the layer kernel (`ab_qsa.py`, Tier B in situ: the function's own upstream mailboxes are the
   FlyDSL stages' outputs from the same snapshot), with must-fail mutants.
3. **Knobs** for every hard-coded choice and every catalogued FlyDSL inefficiency (section 8), default off.
4. **Climb** (`climb_qsa.py`): every value of every knob on the current best (losers included), lookback pairs by the
   product of single gains plus seeded random pairs, the bitwise gate before timing, schedule holds of the
   `qsa.<variant>` cell as knobs, two acceptance orders, every rejected value re-tested on each new base.
5. **Timing in situ** (`bench_qsa.py`, `climb_qsa.py`): 96-launch HIP graphs over 16 cold copies, arms shuffled per
   round, controls `base2` (second graph) and `base_rc` (recompiled), WIN = median below both controls' minimum by
   0.002 and >= 3/4 rounds below 1; on the plain mono, the served stack's schedule, and that schedule with the stack's
   rocKE fused MoE (the served qsa layer); confirmation on the other rocke node.
6. **WaveScope**: run (job n4b2, section 8.1; fix round: this line said "not run"), plus the stamped stage views
   (section 6.2).

## 3. Parity by construction

### 3.1 .ll level (`ll_cmp.py`, FlyDSL `rv-dump4` marked `.pre.ll` vs the function's `.ll`)

For all 21 (stage, variant) pairs of Q5 / Q6 / Q7 at r1, r2, r3, r4, r2s, r3s, r4s and for Q3 / Q4 at r1 and r4s:
the multiset of float operations (`fadd` / `fmul` / `fdiv` / `fsub` / `fcmp` / `fptrunc` / `fpext` on f32, f64,
bf16) and float intrinsics (`fma.f32`, `fma.f64`, `maximum.f32`, `exp2`, `exp`, `sqrt.f64`, `fabs`, the scaled fp8
convert) is **equal**. Lane exchanges are equal too (`update.dpp` 1 / 12 / 137 / 139 / 212 in Q5 / Q6 / Q7 / Q3 / Q4
at r1, `permlane16/32.swap`, `readlane`), and so are LDS loads / stores by width (a `<4 x float>` store is a
`<4 x i32>` one here), the 240 LDS atomics of Q4, the barriers (4 / 5 / 5 / 3 / 33) and the poll-loop pauses.
What differs by design: integer index arithmetic (FlyDSL's floor-division idiom vs shifts) and, before 3.2, the
mailbox instruction forms.

### 3.2 ISA level (`rv_isa` split + `rv_diff`, layer kernels with the five functions linked vs FlyDSL)

| stage | first port (`asm` mailbox, rocKE's inline-asm global loads / stores, 64-bit tag compares) | default (`fly` mailbox) |
|---|---|---|
| Q3 | memory forms differ (global vs buffer); SALU 0.61x | buffer forms; SALU 0.72x (r1) / 0.75x (r4s) |
| Q4 | vmem loads 0.58x, waitcnts 0.47x, branches 0.89x | buffer forms; branches 0.90x, loops 10 vs 16 |
| Q5 | SALU 0.46x | SALU 0.63x (r1) / 0.46x (r4s) |
| Q6 | waitcnt 0.62x, vmem 0.80x | buffer forms; hint polls load only the tag dword |
| Q7 | SALU 0.44x | SALU 0.62x (r1) / 0.49x (r4s) |

Every non-Q stage of the layer (A1, A2, Q1, Q2, C1, M1-M5) matches (r4s) or differs only in nop / waitcnt counts
moved by code layout (r1). The `asm` first port was 1.009-1.030 of the FlyDSL layer (section 6.1): the poll shape
was the difference (FlyDSL peels the first load, retries only the lanes still waiting, one 16-byte load per two
pairs, 32-bit compares). The `fly` mailbox (`q38.poll` + `buffer_store` sc1) restores FlyDSL's shape and is the
default; `asm` and `irf` stay knobs (losers on every base, section 7).

Q4's s_nop budget is FlyDSL's to the cycle: 180 cycles in the 18 inline-asm `v_max_u32_dpp` chains
(`wave_umax_dpp`) and 158 in the DPP inclusive scans = the catalogue's 338 of 547. The knobs `q4_umax=ir |
readlane` and `q4_scan=lds` remove them (bitwise); the climbs found them near-neutral at served L (section 7).

## 4. Numerics gates

### 4.1 Bitwise, in situ (`ab_qsa.py`; one MonoRig per context; random layer input; every byte compared)

| job | functions | cells (variant x context) | contexts | result |
|---|---|---|---|---|
| a1 + a2 (`asm` port) | all five + each alone | 85 (7 variants) | 639, 1023, 2047, 2051, 8191, 8195, 32767, 32771, 65540, 131072, 262136 | bitwise in every cell and arm, deterministic |
| a2 tie | all five | 15 (r2, r3, r4) | planted ties: 2051 .. 262136 | bitwise; `q4_tie` differs in 15 / 15 |
| v1 (`fly` default) | all five + each alone | 35 (7 variants) | 639, 2051, 8195, 32771, 262136 | bitwise |
| v3 tie | all five | 35 (7 variants) | planted ties 2051, 8195, 32771, 65540, 262136 | bitwise; `q4_tie` differs in 35 / 35 |
| v3 abx | `q3_switch=0,q4_chunks=32`; `q3_switch=4,q4_chunks=16`; `q5_kpre`; `prepoll=q5+q6+q7`; `asm` | 66 (+ ties) | 639 .. 262136 | bitwise except Q4's own chunk mailboxes (`fch` / `fcand`, written by its chunk tasks and read only by its final task; their count is the knob) in the 24 `q4_chunks` cells above 8192 blocks: `sel` and everything downstream bitwise |
| v3 absched | `numa=1`; the served stack's schedule (`numa=1,qdense=3` functions); ATOM `qdense=1 / 2 / 3` vs the port's `qdense` | 55 | 639 .. 131072 | bitwise (N4a: the port equals ATOM's lever at every level) |
| climbs | every arm (~600 knob sets) | r1, r4s at 1024 | 1024 | bitwise, 0 failures |

Planted ties (`--tie`): every cached compressed key of block b % 3 == 0 is one vector and of the others another, so
tie groups straddle every Q4 chunk boundary and the 512th margin falls inside one. Mutants (must differ):
`q5_scale` (35/35), `q5_cur` (only where the current token is in the tile), `q6_den`, `q7_gate`, `q7_tail`,
`q3_relu`, `q3_key_row`, `q4_rank`, `q4_tie`: each caught in every run.

Q4's private chunk mailboxes: `ab_qsa.py --mask-q4-private` / `bench_qsa` exclude exactly the second-half bytes
`[2048, isc)` of each row (`fch`, `fcand`); `isc` (Q3's scores) and `sel` are compared.

### 4.2 Tolerance (T0, DESIGN 12.4; `t0_qsa.py`)

T0 per request row: the fp64 golden softmax(q k / 16) v x sigmoid(gate) over the row's attended token list (the arm's
own `sel`, position order for qdense-3 rows under the budget) from the kernel's own upstream values; PASS = arm drift
<= max(1.25 x the FlyDSL mono's drift, the MXFP8 floor), max and relative L2, every row. Contexts 639, 2047, 8195,
32771, 131072 (and 2051), 2 fresh draws each, r1 and r4s, plain base and the served stack's schedule.

| arm | rows (r1 / r4s cells) | worst arm / mono drift (rel L2) | Q7 pack codes equal to the mono's | T0 |
|---|---|---|---|---|
| N4a `qdense=3` | 10 / 10 | 1.0000 | >= 0.999 | PASS |
| N4b `single` | 10 / 10 | 0.9995 / 1.0017 | >= 0.761 | PASS |
| N4b `splitkv` | 10 / 10 | 0.9995 / 1.0017 | >= 0.761 | PASS |
| N4b `tile` (f32 e) | 10 / 10 | 0.9995 / 1.0017 | >= 0.761 | PASS |
| N4b `tile`, bf16 e | 10 / 10 | 1.0020 / 1.0024 | >= 0.717 | PASS |
| the same on the stack (`numa=1,qdense=3` + N4b) | 10 / 10 | <= 1.0034 | >= 0.719 | PASS |
| mutants: `fuse_den`, `splitkv_norescale`, `splitkv_drop`, `tile_norescale`, `qdense_drop` | | 5-38x | <= 0.01 | FAIL (caught) |

N4b's f32 probabilities are closer to the fp64 golden than the mono's bf16 ones before the pack (P13 measured ~5,000x
by emulation); after the MXFP8 pack the drift equals the mono's (0.995-1.003), 21-28 % of the pack codes move.

### 4.3 Race stress (DESIGN 7.8 gate 5)

ATOM's deterministic stage delays (`ATOM_Q38_DELAY`, both arms) that make one producer late at a time: Q1, Q2, Q3
(64 tasks), Q4 (and its exit), Q5 (16 tiles), Q6 exit, Q7. With the climbed knob set (`numa=1,qdense=3,prepoll=q5+q6+q7,q5_kpre`: hint pre-polls removed, key rows prefetched)
on the served stack's schedule, job cf1 (n07-17): 7 delay patterns x r1 / r4s / r2s x L 639 / 8195 / 32771 = **63
cells, bitwise in every cell, three launches each deterministic, no hang** (`q4_tie`-style private Q4 regions
masked as in 4.1).

## 5. N4b: the fused split-KV softmax (DESIGN 12, tolerance class)

Three forms, all with Q6 idle (one hop fewer):
- `single`: Q7 polls Q5's scores of its 4 heads, CTA-wide max, `mexp` in LDS, fixed-order denominator, PV with f32
  probabilities, one `1 / den` multiply.
- `splitkv`: as `single`, each wave a split (its 17 x 16 positions), exp inside PV, the 8 splits merged in wave
  order (`mexp(m_w - M)`).
- `tile`: Q5's 32-position tiles are the splits (per head max / exp / sum in Q5; e published f32 or bf16 in `ss`, the
  tile's (m, l) in `pp`); Q7 merges the 65 tiles per head in wave order and runs PV with p = e x exp(m_t - M) / L.

In situ (L 1024; layer ratio vs the FlyDSL base; WIN / LOSS per 9.1):

| base | `single` | `splitkv` | `tile` f32 | `tile` bf16 |
|---|---|---|---|---|
| plain r1 | 1.005 (asm port) | 1.087 (asm) | **0.987 WIN** | **0.982 WIN** |
| plain r4s | 1.014 (asm) | 1.114 (asm) | 1.001 | **0.996 WIN** |
| stack r1 / r4s | 1.040 / 1.014 (asm) | 1.196 / 1.156 (asm) | 1.017 LOSS / 1.003 | 1.012 LOSS / 0.995 |
| stackmoe r1 / r4s | | | 1.024 LOSS / **0.975 WIN** | 1.017 LOSS / **0.971 WIN** |

Stage view (stamped twins): `single` / `splitkv` move all of Q6's work into each Q7 CTA (4 heads x 2080 positions per
CTA, 2x the poll volume): Q7's increment grows by about Q6's, so the saved hop does not show. `tile` keeps the
exp work in the 130 Q5 CTAs; on the stack's r1 cell its loss comes with the cell's `hold.q7` tuned for the FlyDSL
chain.

**Re-test on the climbed exact set (job n4b4, n01-09, stackmoe, L 1024, 24 rounds, controls within [0.996, 1.001]).**
The fused forms carry Q5-Q7 only, so the exact comparator is the climbed set on Q5-Q7 with FlyDSL's Q3 / Q4
(`pick57`); `pick` = all five functions. Layer / FlyDSL base:

| variant | `pick` (5 fns) | `pick57` | `tile` f32 | `tile` bf16 | `tile`, `hold.q7` off | no-`q5_kpre`, no-hold set: exact / `tile` / `tile` bf16 |
|---|---|---|---|---|---|---|
| r1 | **0.948** | 0.955 | 0.970 | 0.964 | 0.967 | 0.983 / 1.023 LOSS / 1.023 LOSS |
| r4s | 0.964 | **0.958** | 0.962 | 0.959 | 0.962 | 0.962 / 0.964 / **0.958** |

On the climbed set N4b no longer pays: at r1 it is 1.5 % (f32) / 0.9 % (bf16) behind its exact comparator, at r4s
within 0.004 of it (bf16 0.959 vs 0.958). Stamps (r1): Q6's hop is gone, but Q7's increment (the 65-tile merge per
head) is 0.170 of the launch vs 0.126 + 0.032 (Q7 + Q6) for the exact chain; the pre-poll removal and the key
pre-issue already took the slack the fused form was exploiting. N4b stays default-off (tolerance class, no gain on
the served set); T2 / T4 are therefore not spent on it (section 9). All exact arms bitwise, all tile arms in the
tolerance class as built (T0 section 4.2).

A side finding of the same job: at r4s, the climbed Q5-Q7 with **FlyDSL's Q3 / Q4** (`pick57`, 0.958) beat all five
functions (`pick`, 0.964), consistently with b3's solo `rk_q3` at r4s (1.005 / base); at r1 the five functions are
better (0.948 vs 0.955). The functions are bitwise to the stages, so this is a code-layout / scheduling difference of
the r4s Q3 inside the layer; the extern list can take it per width (`q3`/`q4` omitted at r4s) and it is named in
section 9.

## 6. Timing in situ (DESIGN 9.1)

### 6.1 The port at its defaults (L 1024; job b3, n01-09; 24 shuffled rounds; controls within [0.995, 1.004])

The `asm` column is the first port's mailbox as a knob arm in the same job.

| base | variant | `asm` first port | `fly` default | solo stages (fly) | `prepoll=q5+q6+q7` | `q5_kpre` |
|---|---|---|---|---|---|---|
| plain | r1 | 1.009 LOSS | **0.994 WIN** | q4 1.009 LOSS, others noise | **0.945 WIN** | 0.993 WIN |
| plain | r4s | 1.025 LOSS | **0.991 WIN** | q5 0.996, q7 0.991 WIN | **0.969 WIN** | 0.990 WIN |
| stack | r1 | 1.023 LOSS | 0.999 | q7 0.994 WIN | **0.973 WIN** | 0.998 |
| stack | r4s | 1.030 LOSS | **0.988 WIN** | q7 0.991 WIN | **0.960 WIN** | 0.986 WIN |
| stackmoe | r1 | 1.028 LOSS | 0.999 | noise | **0.971 WIN** | 0.998 |
| stackmoe | r4s | 1.022 LOSS | **0.980 WIN** | q7 0.986 WIN | **0.968 WIN** | 0.977 WIN |

### 6.2 Long contexts (P8: the parity gate at 32K / 128K / max; stack base; 16 rounds)

| variant, L | `fly` default | solo Q3 / Q4 | `prepoll` | `q4_chunks=16` | `q3_switch=0` |
|---|---|---|---|---|---|
| r1, 32,768 | **0.995 WIN** | 0.993 / 0.996 WIN | **0.965 WIN** | 1.071 LOSS | 0.999 |
| r1, 131,072 | 1.006 LOSS | 1.003 / 1.007 LOSS | **0.979 WIN** | **0.992 WIN** | 0.999 |
| r1, 262,136 | 1.000 | 1.003 / 0.999 | **0.973 WIN** | 1.000 | 1.005 LOSS |
| r4s, 32,768 | 0.996 | 1.003 / 1.008 LOSS | **0.991 WIN** | 1.000 | **0.985 WIN** |
| r4s, 131,072 | **0.991 WIN** | 1.001 / 1.000 | **0.976 WIN** | **0.982 WIN** | **0.984 WIN** |
| r4s, 262,136 | 0.992 | 1.000 / 1.000 | **0.977 WIN** | **0.989 WIN** | **0.985 WIN** |

The default port is within [0.991, 1.006] of the FlyDSL chain at every long context (one LOSS: r1 at 131K, 1.006, Q3
and Q4 each about 1.003-1.007 there); the climbed exact set wins at every L. `q4_chunks` alone is context-dependent
(it turns a single-level selection two-level at 32K): `q4_thr=8192` (16 chunks of 4096 only above 8192 blocks, the
mono's single level below) was added for that and is in the long-L climb (section 7.2). Stamped views (r1, 131K):
Q4's increment 0.205 -> 0.066 of the launch with 16 chunks, Q5's wait absorbs part of it.

### 6.3 Confirmation on the other node (n07-17; job cf1)

Fresh process, seed 7, the other node (n07-17, GPU bus 101), 24 shuffled rounds, served-stack base (`stackmoe`):

| variant | knob set | n07-17 / base | n01-09 climb / base |
|---|---|---|---|
| r1 | gain pick: `prepoll=q5+q6+q7`, `q5_kpre`, `hold.q7=64` | **0.947 WIN** | 0.948 |
| r1 | random pick: + `q3_batch=4`, `inline=default`, `hold.q6=64` | **0.948 WIN** | 0.949 |
| r1 | `prepoll=q5+q6+q7` only | **0.971 WIN** | 0.971 (b3) |
| r4s | `prepoll=q5+q6+q7` (gain pick) | **0.964 WIN** | 0.963 |
| r4s | + `q4_umax=ir` (random pick) | **0.961 WIN** | 0.959 |
| r4s | + `q5_kpre` | **0.960 WIN** | 0.956 (cl2 it2, not accepted: 0.997 / cur) |

Controls within [0.996, 1.003] in every run. The picks reproduce on both nodes within 0.004.

### 6.4 End to end (decision 14; `truemono-quick` arms `full3_w8c` / `full0_w8c`, new file `arms.d/w8c.sh`)

Same job, same node (n01-09), arms shuffled per pass, 3 passes, 32 GSM8K prompts, concurrency 1. `full3` / `full0`
= the consolidated stack (cs.sh); `_w8c` = the same + the five QSA functions on every narrow QSA build (the grouped /
wide widths keep FlyDSL's). Ratios of the 3-pass medians:

| job, function knobs | K=3 ms / round (median) vs full3 | K=3 tok/s per request vs full3 | K=0 tok/s per request vs full0 | vs hybrid (K=3 round / K=0 tok/s) |
|---|---|---|---|---|
| e1: `numa=1,qdense=3,prepoll=q5+q6+q7` | 0.990 | 1.025 | 1.013 | 0.939 / 1.102 |
| e2: + r1 `q5_kpre`, `qsa.r1` `hold.q7=64` (the climb's picks) | **0.991** | 1.010 | **1.017** | 0.939 / 1.114 |

(Fix round: the "K=3 tok/s per request" column is a pooled aggregate; per pass, e2 gives 0.970 / 1.002 / 0.999, median
1.000, so it shows no gain. The robust K=3 metric is the median ms per round below. No A-vs-A was run in e1 / e2, so
none of these ratios is an e2e WIN; see "Fix round" F2 for the DESIGN 9.1 E2E rule.)
Per-pass median round ratios (e2, K=3) 0.993 / 0.992 / 0.989; the pass-1 mean round of `full3_w8c` holds one slow
request (slow share 0.16, the median unaffected). GSM8K correct: 29-30 / 32 (`full3_w8c`) vs 30-31 (`full3`) vs
29-30 (hybrid) at K=3, 30-31 vs 28-31 vs 30-31 at K=0: within the arms' spread (the functions are bitwise to the
stages, so the difference is the sampled-request spread, not numerics). The in-situ gain is on the narrow QSA
layers only (the MoE / GDN / grouped-width layers are untouched), hence 0.95-0.96 per layer shows as about 0.99 per
round.

## 7. The climbs (decisions 13 / 15)

### 7.1 Served L (stackmoe base, L 1024; job cl2; 16 rounds per iteration; every arm bitwise-checked first)

Space: `pause` (3), `pbar` (2), `mailbox_impl` (fly / asm / irf), `q3_switch` (1 / 2 / 4 / 0), `q4_chunks` (8 / 16 /
32), `put_nt`, `kv_nt`, `tbl`, `q7_pp`, `q4_scan`, `q4_umax` (3), `q3_batch` (8 / 4 / 16), `fma`, `inline`,
`prepoll` (6 stage sets), `q5_kpre`, and the `qsa.<variant>` schedule levers `hold.q7` (off / 64 / 192 / 320),
`hold.q5`, `hold.q6`, `hold.q2`; 42-56 arms per iteration (singles over the whole space on the current best, 14
lookback pairs by single-gain product, 4 seeded random pairs).

| variant, order | iterations | accepted in order (each re-tested on the new base) | final / FlyDSL base |
|---|---|---|---|
| r1, gain | 4 | `prepoll=q5+q6+q7`; `hold.q7=64` (the FlyDSL-tuned 192 re-tuned); `q5_kpre` | **0.948** |
| r1, random | 6 | `prepoll=q7`; `q3_batch=4 & prepoll=q5+q6+q7`; `prepoll=q6+q7 & hold.q6=64`; `inline=default & hold.q7=64`; `prepoll=q5+q6+q7`; `q5_kpre` | **0.949** |
| r4s, gain | 2 | `prepoll=q5+q6+q7` | **0.963** |
| r4s, random | 3 | `prepoll=q6`; `q4_umax=ir & prepoll=q5+q6+q7` | **0.959** |

Both orders converge on the same core (no hint pre-polls in Q5-Q7, Q5's key rows issued before its head poll, Q7's
hold re-tuned); every rejected value was re-timed on each new base (decision 15): e.g. `q4_scan=lds` and
`pbar=compiler` passed the WIN rule against the controls on later bases (0.995-0.996 / cur) but were not the best arm,
`put_nt` lost 2.4-8 % on every base, `hold.q7=320` 4 %, `mailbox_impl=asm` 3 %.

### 7.2 Long context (stack base, L 131,072; job cl3; start = `prepoll=q5+q6+q7`)

Same space plus `q4_thr` (0 / 8192 / 16384), 12 rounds. On the `prepoll` base the layer is **0.973 (r1) / 0.976
(r4s)** of the FlyDSL chain at 131,072 tokens. No further arm passed the WIN rule against the controls' spread
([0.995, 1.003]): the best were `pbar=compiler` 0.997 / cur (r1) and `q3_batch=16` 0.995, `q4_chunks=16` 0.997,
`q3_switch=0` 0.998 / cur (r4s); `q4_thr=8192` 0.9996-0.9997 / cur. With the hint pre-polls gone, Q4's chunk spread
and Q3's path switch no longer move the chain end at 128K (they did on the default base, 6.2): rejected on base
`prepoll`, kept as knobs (decision 15).

## 8. FlyDSL inefficiency catalogue -> knobs -> verdicts (on base X)

| catalogue item (`reverse/FLYDSL-STAGES.md`) | knob(s) | verdict (base: served stack, L 1024 unless stated) |
|---|---|---|
| hard-spin polls, a throttle hint before every payload poll (Q5 / Q6 / Q7 `pre_poll`) | `prepoll` (skip the hint), `pause`, `pbar` | **prepoll WIN 0.95-0.97 everywhere (exact)**; pause / pbar neutral |
| dense `vmcnt(0)` drains (Q5's key rows loaded after the poll + barrier) | `q5_kpre` (sel polled per thread, key loads issued before the head poll) | **WIN** after prepoll (r1 0.987 / cur) |
| Q4 s_nop-padded DPP chains (338 of 547 wait states) | `q4_umax` (asm / ir / readlane), `q4_scan` (dpp / lds) | near-neutral at served L (Q4 is under budget: qdense skips it); `q4_umax=ir` in the r4s random pick |
| poll / mailbox instruction form | `mailbox_impl` (fly / asm / irf) | fly = FlyDSL shape (default); asm LOSS 1.009-1.030; irf neutral |
| no `nt` on streams | `kv_nt` (K / V cache loads), `put_nt` (mailbox stores) | kv_nt neutral; put_nt LOSS 2-8 % |
| Q3's small-context path slower at its top end (P8) | `q3_switch` | r4s 32K-max WIN 0.984-0.985; r1 neutral / LOSS at max |
| Q4's two-level step at long L (P8) | `q4_chunks`, `q4_thr` | 16 chunks WIN at 128K / max (r4s), LOSS at 32K alone; thr in the long-L climb |
| Q3 batch width | `q3_batch` | in the r1 random pick (4); neutral alone |
| scalar vs vector table loads | `tbl` | neutral |
| Q7's probability poll in one batch | `q7_pp` | split LOSS 1.3 % |
| inlining / FMA contraction forms | `inline`, `fma`, `fmax` | neutral (bitwise in every case) |
| FlyDSL-tuned holds | `sched.hold.q5/q6/q7/q2` (climb) | `hold.q7` 192 -> 64 WIN at r1 with the port |

### 8.1 WaveScope ATT (job n4b2, n01-09; plain build, r4s, L 1024; `parse_kernel_trace.py` of one traced dispatch)

`mono` = the FlyDSL stages, `rk` = the five functions at defaults, `pickp` = `prepoll=q5+q6+q7,q5_kpre`. Ratios vs
`mono` (stall-cycle ratios are per type, shares are of the traced wave's stall cycles):

| arm | instructions | traced cycles | barrier-stall cycles | VMEM-wait cycles | VMEM-load-stall cycles | shares barrier / VMEM-wait / VMEM-load |
|---|---|---|---|---|---|---|
| `mono` | 1 | 1 | 1 | 1 | 1 | 50.5 % / 26.6 % / 13.3 % |
| `rk` | 0.976 | 1.069 | 0.962 | 0.952 | 1.346 | 46.3 % / 24.1 % / 17.1 % |
| `pickp` | 0.970 | 1.031 | **0.647** | 1.383 | 1.453 | 32.5 % / 36.5 % / 19.3 % |

Read: the port's traced wave issues 2.4 % fewer instructions (poll-loop trip counts are timing-dependent, so this
is not attributed to a code difference; the ISA comparison is section 3.2); `prepoll` removes a third of the barrier stall (the waves no longer park behind a hint poll at the
workgroup barrier) and moves the wait to the payload loads themselves (VMEM-wait up), which is the intended
exchange: the chain end arrives earlier (6.1 / 6.3). The traced-cycle totals of a single dispatch of a persistent,
poll-driven kernel depend on which CTA and which producer timing the trace caught (the `rk` total is 1.069 here
against 0.991 over 24 rounds in situ); the in-situ timings are the verdicts, ATT is the structural view.

## 9. Not done here (named, not narrowed)

- **T2 / T4 for N4b** (served teacher-forced KL with P11's harness, RULER): not run, because N4b gives no gain on the
  climbed served set (section 5 re-test); it stays default-off and passes T0. If a later base brings its r4s bf16
  form ahead of the exact chain, T2 / T4 are the gate before any serving use. N4a (`qdense=3`) is already in the
  served stack (P13) and the port reproduces it bitwise.
- **r4s Q3 placement** (section 5, side finding): FlyDSL's Q3 / Q4 with the rocKE Q5-Q7 is 0.6 % faster at r4s than
  all five functions on the served stack. Not yet taken into the e2e arm (`arms.d/w8c.sh` links all five at every
  narrow width); the ISA / stamp difference of Q3 r4s inside the layer is the next item. (Fix round F8: resolved on
  both nodes with separately compiled copies, explained in the ISA, and run end to end as `full3_w8c57`.)
- **WaveScope ATT before / after** (method step 6): run (job n4b2, n01-09, r4s, L 1024, one traced dispatch per
  build; section 8.1). One dispatch per arm is a structural view, not a timing: the in-situ timings above are the
  verdicts.
- **Wide / grouped variants** (r8g4, r16g4, ...): no function (the served stack keeps FlyDSL's there through
  `ATOM_Q38_EXTERN_MISSING=mono`; e2e log `extern-missing` counts them).
- **Rust direct backend levers** for these stages: the functions are inlined into FlyDSL's COMGR build (hybrid path);
  per-stage direct scopes apply once the stages run in a rocKE layer kernel (WP3). (Fix round F7: the concrete
  blockers, with file references.)
- **ATOM-side guard**: a function set whose `qdense` disagrees with the schedule's deadlocks the layer (alola rule
  210); the sidecar records `stage_knobs`, ATOM does not check it yet (an additive ATOM-truemono check is a follow-up).

## Verification (independent verifier, W8, 2026-11-01)

The verifier tried to refute the report, reading the code, the job logs (`hackathon-prep/logs/q38/w8c/*.console`,
`w8-c-*.out`) and the cluster results (`results/w8-c-*`, `qk-w8c-e2`), and re-ran the decisive checks itself: two
new jobs on the **other** node (n07-17, GPU bus 101, health ok): `w8-vWP2c-x1` (68243998) and `w8-vWP2c-x2`
(68244085). The stages were fresh snapshots of the current trees (`rocke-w8-vWP2c`, `atom-w8-vWP2c`, `w8-vWP2c-h`).
Absolute numbers are in the private companion. Nothing was deleted, no git state was changed, and no rocKE or ATOM
file was edited. The local outputs are new files under `hackathon-prep/qwen38/truemono/WP2c/verify_wp2c/`.

**Verdict: PARTIAL.** Every WP2c gate in D0 holds and reproduces: bitwise parity at the served L and at
32K / 128K / max, the `q4_tie` mutant across chunk boundaries, and `qdense` as a knob. The stage-level timing claims
also reproduce on the second node. Three summary claims are stronger than the evidence, and two 9.1 discipline items
were never exercised (listed below).

### V1. Confirmed

- **.ll parity, extended.** The verifier regenerated the default functions locally from the current package and
  ran `ll_cmp.py` against the `rv-dump4` `.pre.ll` for **all 35** (stage, variant) pairs: Q3-Q7 x r1, r2, r3, r4,
  r2s, r3s and r4s. The report covered 25: Q3 / Q4 at only r1 and r4s. The float multiset matches in 35 of 35. The
  spot tables (Q4 r1, Q7 r4s) also show equal DPP, permlane, readlane, barrier, LDS-atomic and fence counts. What
  differs is only the mailbox load / store type spelling (i64 vs v2i32, i128 vs v4i32, the same widths) and
  readfirstlane counts.
- **Bitwise on MI355X, with must-fail mutants (x1).** Defaults, r1 and r4s, L 1024 and 32,771, `--solo`: all
  bitwise and deterministic. `q5_scale`, `q7_gate`, `q3_relu` and `q4_rank` each differ in 4 of 4 cells. The
  `atom_check` layout constants are equal. The builder's own runs agree: a1 / a2 / v1 / v3 bitwise, `q4_tie` caught
  in 15 / 15, 35 / 35 and 20 / 20 cells, and v3's `absched` bitwise with nonzero payloads on the stack schedule.
  The v2 `absched` "AB_QSA FAIL" was a harness bug (the NUMA payload probe read zeros); v3 re-ran it with the fix.
- **Q4 chunk-mailbox exception.** The non-bitwise abx cells (only `q4_chunks` != 8 arms, all non-default) start at
  the second-half offset +2048 of each row, inside `[2048, isc)`. `sel`, `isc` and `h_out` are equal there. The
  report states this correctly.
- **Existing builds byte-identical.** The rocKE working tree matches HEAD. The only rocKE files modified in the
  WP2c window outside `qsa2/` and `test_qwen38_qsa2.py` belong to other agents (large_batch, stages2,
  truemono_tail). `qsa2/lower.py` subclasses the lowerer with no global registration or monkeypatching, so no other
  build sees the `q38.*` ops. `test_qwen38_qsa2.py`: 32 passed (local CPU).
- **Tuned-best-base check (the decisive one).** The climb re-tuned a schedule lever, `qsa.r1 hold.q7` 192 -> 64,
  but only on port arms; the FlyDSL base kept 192. The verifier timed the FlyDSL stages with the same override in
  the same job (x1, stackmoe, L 1024, 24 rounds, controls [0.998, 1.001] / [0.999, 1.003]):

  | arm | r1 / base | r4s / base |
  |---|---|---|
  | rocKE defaults (`rk`) | 0.999 noise | **0.980 WIN** |
  | FlyDSL + `hold.q7=64` | 1.000 noise | 1.000 noise (the override does not touch the r4s cell) |
  | pick without the hold | **0.975 WIN** | **0.963 WIN** |
  | pick (`prepoll=q5+q6+q7`, `q5_kpre`, r1 `hold.q7=64`) | **0.948 WIN** | **0.963 WIN** |

  The hold is worth nothing to the FlyDSL chain. It pays about 2.6 % only together with the port's pre-poll
  removal, so the base was the tuned best and the 0.948 is a port gain. The default r4s WIN (0.980), which the
  builder measured only on n01-09, also holds on n07-17.
- **Long contexts on the second node (x2, stack base, 16 rounds, controls within [0.998, 1.003]).** These were
  single-node in the report.

  | L | r1 `rk` | r1 `prepoll` | r4s `rk` | r4s `prepoll` |
  |---|---|---|---|---|
  | 32,768 | **0.993 WIN** | **0.961 WIN** | **0.992 WIN** | **0.979 WIN** |
  | 131,072 | 1.011 **LOSS** | **0.982 WIN** | **0.993 WIN** | **0.978 WIN** |
  | 262,136 | 0.999 noise | **0.969 WIN** | **0.992 WIN** | **0.978 WIN** |

  "With prepoll it wins at every L" reproduces on both nodes. The disclosed default loss at r1 / 131K reproduces
  and is larger here: 1.011 vs 1.006.
- **Both-node confirmation of the climbed picks.** cf1 (n07-17) shows 0.947 / 0.948 (r1) and 0.960-0.964 (r4s),
  with controls in range.
- **End to end in the same job as the hybrid.** `qk-w8c-e2` (68243486) ran `hybrid`, `full3` / `full0` and
  `full3_w8c` / `full0_w8c` in one job. The server log shows the QSA functions linked on the narrow builds and
  FlyDSL's on the wide ones. Only the `_w8c` arm carries `qsa.r1 hold.q7=64`, and per x1 that is not a FlyDSL gain.

### V2. Overstated, or not exercised

1. **K=3 "tok/s per request 1.010" is not robust.** Per pass, `full3_w8c` / `full3` request tok/s is 0.970 /
   1.002 / 0.999, and the median of the per-pass values gives 1.000. Only the report's aggregate statistic gives
   1.010; pass 1 has one slow request. The robust K=3 metric is the
   median ms per round: 0.993 / 0.992 / 0.989 per pass. The K=0 tok/s ratio is consistent (1.013 / 1.020 / 1.035).
2. **The DESIGN 9.1 E2E win rule was not applied.** That rule asks for 6 rotated pairs and an A-vs-A of the pinned
   hybrid; this run was 3 quick-track passes with no A-vs-A. In e1, `full3`'s own per-pass median round spread
   about 5 % (pass 3), against a claimed effect under 1 %. The quick-track number satisfies decision 14 (same job
   as the hybrid), but no e2e WIN can be declared from it.
3. **"r4s 0.956-0.964 climbed" includes 0.956**, which comes from an arm the climb did not accept (cl2 it2: 0.997
   of cur, no WIN). Accepted or confirmed r4s sets give 0.959-0.964 (x1: 0.963).
4. **9.1 clock sampling was not done.** No `amd-smi` SCLK / power sampling or throttled-round drop exists in
   `bench_qsa.py` / `climb_qsa.py`. The tight controls make throttling unlikely, but the item was not exercised.
5. **Decision 16 asks for more than multiset parity.** It names instruction order, waitcnt placement, registers /
   spills and MFMA interleave. `ll_cmp` compares operation multisets only. `rv_diff` gives counts and shows
   undocumented deltas: Q4 r1 has 308 FlyDSL s_nops vs 556 for the port, with M5 357 vs 88. The two sum about the
   same, which looks like a stage-split boundary shift, but the report does not state it. The register / scratch
   use of the default port's layer kernel is not recorded. The first (`asm`) port's layer carried 68 bytes of
   private scratch that the FlyDSL build does not.
6. **Report inconsistencies.**
   - Section 2 step 6 says "WaveScope: not run", but sections 8.1 / 9 show it ran.
   - The front summary says "10 must-fail mutants", but 9 numeric + 5 T0 = 14.
   - Rust direct-backend levers (method step 4) were not climbed. Section 9 justifies this because the hybrid
     path is COMGR-only.
7. **Some small verdicts are below the timing resolution.** W9's P16 lane (alola rule 215) measured a ~1 %
   per-code-object timing lottery on qsa r1 extern-linked builds: recompiled controls with identical `.text` ran
   0.988-1.012. Both 9.1 controls here sit on the base (`base2` reuses its code object; `base_rc` is one
   recompile), so an arm's own code-object luck is uncontrolled. These effects are within that band: the r4s
   "FlyDSL Q3 / Q4 is 0.6 % faster" side finding, N4b's "0.9-1.5 % behind" verdict, the long-L `q4_chunks` /
   `q3_switch` / `pbar` rejections near 0.995-0.999, and the r1 default at 131K (1.006 / 1.011). They are not
   resolved until they are re-timed with several separately loaded copies per arm. The headline effects are well
   outside the band and reproduced across fresh compiles in 4+ jobs: `prepoll` / pick at 0.95-0.98, and the r4s
   default at 0.980.
8. **Git note.** At HEAD, `copy-rocm-lib` has a commit "Qwen38 rocKE monokernel work" (2026-11-01 15:55, the
   user's identity). It includes `qsa2/` and was made 2 minutes after this report's last write. The verifier
   cannot attribute it to the builder; the user owns git.

## Fix round (W8, 2026-11-02)

The verifier's PARTIAL listed nine issues. Each one is addressed below, re-run where a measurement was missing.
Ratios only here; absolute numbers in the private companion (section 9). No git state was changed, no file was deleted
(the pre-fix copies of the three edited harness files stay under `WP2c/orig_fix/`), and no rocKE or ATOM file was
edited: the functions are the same `qsa2/` package, staged unchanged as `rocke-w8-fx1`.

New harness files (`hackathon-prep/qwen38/truemono/WP2c/`):
- `clocks.py`: DESIGN 9.1 SCLK / power sampling.
- `bench_rc.py`: lottery-resolved in-situ timing.
- `isa_deep.py`, `deep_summary.py`: decision-16 ISA parity beyond counts.
- `e2e91.py`: the DESIGN 9.1 E2E rule.
- `rc_table.py`, `rc_abs.py`: tables (ratios / private absolutes).
- `submit_fx.sh`: every fix-round submission, recorded.

Additive edits:
- `bench_qsa.py` and `climb_qsa.py` now sample clocks (`--no-clocks` turns it off).
- `wp2c_job.sh` gained the phases `benchrc` and `isa2`.

Quick track: the new arm file `truemono-quick/arms.d/w8c2.sh` adds the A-vs-A twins `hybrid_b`, `full3_b`, `full0_b`
and `full3_w8c57`; `w8c.sh` is unchanged.

Jobs:

| job | node | what |
|---|---|---|
| `w8-fx-rc1` 68244316 | n01-09 | served L, lottery-resolved |
| `w8-fx-rc1b` 68244446 | n07-17 | the same, other node |
| `w8-fx-rc2` 68244276 | n07-17 | L 131,072 |
| `w8-fx-rc3` 68244381 | n01-09 | 32K / 128K / max |
| `w8-fx-isa` 68244272 | CPU | deep ISA, 7 variants x 7 configurations |
| `qk-w8fx-e3` 68244368 | n01-09 | K=3, 9.1 E2E rule |
| `qk-w8fx-e3b` 68244529 | n07-17 | K=3 on the resolved picks |
| `qk-w8fx-e0` 68244674 | n07-17 | K=0, 9.1 E2E rule |

### F1. The K=3 request tok/s claim (issue 1)

Corrected in the front summary and in section 6.4. Per pass, e2's `full3_w8c` / `full3` request tok/s is 0.970 / 1.002 /
0.999, so the median of the per-pass values is 1.000: no gain. The 1.010 was a pooled aggregate that one slow request
moved. The K=3 metric that holds up is the median ms per round (0.993 / 0.992 / 0.989 per pass). F2 applies the
9.1 rule to both metrics.

### F2. The DESIGN 9.1 E2E rule (issue 2)

The setup:
- One server at a time.
- Six passes, each a seeded permutation of the arms rotated per pass (= 6 rotated pairs).
- In the same job, two A-vs-A twins: the pinned hybrid twice (`hybrid`, `hybrid_b`) and the comparator twice
  (`full3` / `full3_b`, or `full0` / `full0_b`).
- The noise band is the union of both twins' per-pair ratios.
- WIN = the median ratio is below the band's minimum AND at least 5 of 6 pairs are below 1 (`e2e91.py`).
- Ratios are oriented so that below 1 is better: time per round and per token as arm / ref, request tok/s as
  ref / arm. The table quotes tok/s the other way round, as arm / ref.

| job, arm (functions) | vs | K=3 ms per round: median, pairs < 1, band min | verdict | request tok/s (arm / ref), median | verdict |
|---|---|---|---|---|---|
| e3 (n01-09), `full3_w8c` (e2's picks: r1 five functions with `prepoll`, `q5_kpre`, `hold.q7=64`; r4s five with `prepoll`) | `full3` | 0.9925, 6/6, 0.9946 | **WIN** | 1.014 (4/6) | NEUTRAL (band in ref / arm: 0.983-1.023) |
| e3, `full3_w8c57` (the same, FlyDSL's Q3 / Q4 at r4s) | `full3` | 0.9946, 6/6, 0.9946 | WIN (at the band edge) | 1.013 (5/6) | NEUTRAL |
| e3, `full3_w8c57` | `full3_w8c` | 1.0045, 2/6 | NEUTRAL | 1.004 | NEUTRAL |
| e3b (n07-17), `full3_w8c57` with `q5_kpre` at r4s too (= the lottery-resolved layer picks, F6) | `full3` | 0.9967, 5/6, 0.9970 | WIN (at the band edge) | 1.012 (4/6) | NEUTRAL (band 0.986-1.015) |
| every `_w8c` arm | `hybrid` | 0.936-0.939 | **WIN** | 1.06-1.07 | **WIN** |

K=0 (job e0, n07-17, 6 pairs; twins `hybrid` / `hybrid_b` and `full0` / `full0_b`):

| arm | vs | request tok/s (arm / ref), median, pairs better | band (ref / arm) | verdict | per-token time, median, pairs < 1 | verdict |
|---|---|---|---|---|---|---|
| `full0_w8c` (r1: five functions, `prepoll`, `q5_kpre`, `hold.q7=64`) | `full0` | **1.015**, 6/6 | 0.997-1.004 | **WIN** | 0.986, 5/6 (band 0.998-1.053) | **WIN** |
| `full0_w8c` | `hybrid` | **1.116**, 6/6 | | **WIN** | 0.898 | **WIN** |

K=0 GSM8K correct counts: `full0_w8c` 27-30 (mean 29.0), `full0` 29-31 (29.8), `full0_b` 28-31 (29.3), `hybrid`
28-32 (29.8), `hybrid_b` 29-31 (29.8). The arms vary from pass to pass with the same prompts, so serving is not
run-to-run deterministic here. The 0.8-count lower mean of `full0_w8c` is within that spread (standard error about
0.5 per arm). The functions are bitwise to the stages at every layer cell gated (section 4, F6).

GSM8K correct counts (32 prompts per pass): `full3_w8c` 28-30, `full3_w8c57` 28-31, `full3` 28-31, `hybrid` 28-31,
`hybrid_b` 28-31. That is within the twins' own spread; the functions are bitwise to the stages, so the count can only
move through the sampled requests. The acceptance length per round is 2.96-3.13 in every arm.

Reading: under the 9.1 rule, the functions are an e2e WIN on K=3 ms per round against the consolidated stack, by
0.3-0.75 %, on both nodes. On request tok/s the twins' spread (1.5-2 %) is wider than the effect, so tok/s per request
is NEUTRAL against the full stack and a WIN only against the hybrid. The r4s Q3 / Q4 placement (F8) is 0.4 % of one
QSA layer type, which is far below e2e resolution: `full3_w8c57` vs `full3_w8c` is NEUTRAL.

### F3. Clock sampling (issue 4)

`clocks.py` reads the job GPU's sysfs `pp_dpm_sclk` (the `*` level) and `hwmon` power every 50 ms, with amd-smi as
a fallback. It records each round's SCLK median / minimum and power median. A round is dropped when its median SCLK
is below 0.97 of the job's 95th-percentile clock, and verdicts are taken on the kept rounds (both views are kept in
the JSON). `bench_qsa.py`, `climb_qsa.py` and `bench_rc.py` all use it.

In every fix-round timing job (rc1, rc1b, rc2, rc3: 224 rounds) SCLK was at the top level in every sampled round, so
0 rounds were dropped. Power medians per cell are in the private companion. The earlier verdicts were not re-run for
clocks alone: the fix-round re-timing (F6) of the same arms, with clocks sampled, reproduces them.

### F4. The climbed r4s range (issue 3)

Corrected in the front summary to 0.959-0.964 (accepted or confirmed sets). 0.956 came from cl2 iteration 2, an arm
with no WIN against cur (0.997 / cur), which the climb did not accept.

Lottery-resolved (F6), the r4s picks on the served stack are:

| r4s arm | n01-09 | n07-17 |
|---|---|---|
| all five functions, `prepoll` + `q5_kpre` | 0.965 | 0.965 |
| `prepoll` only | 0.969 | 0.969 |
| FlyDSL's Q3 / Q4 + the rocKE Q5-Q7 with `prepoll` + `q5_kpre` | **0.961** | **0.961** |

### F5. Decision-16 parity beyond operation counts (issue 5)

`isa_deep.py` compares, per stage, the FlyDSL build and the build with the functions linked. Both come from the same
GPU-less `rv_dump` job (`w8-fx-isa`) and are split per stage by `rv_isa`. It reports:
- the instruction ORDER: the difflib ratio of the opcode streams and of the class streams, where 1 = the same
  sequence;
- the WAITCNT PLACEMENT: each wait as (counter, vmem / LDS ops since the last wait, the class of the next
  instruction), compared as sequences;
- per-stage and kernel REGISTERS, spills and scratch;
- the MFMA INTERLEAVE: count, gaps, and MFMAs right after a `(0)` wait;
- s_nop cycles, with the end-of-text padding separated out.

Configurations: `plain` (no schedule, default functions), `stk` (the served schedule + its MoE externs, functions
`numa=1,qdense=3`), `pk` / `pp` / `ppir` (the climbed sets), `pk57` / `pp57` (the same with FlyDSL's Q3 / Q4). Seven
narrow variants each, 49 pairs. Tables: `WP2c/fix_local/deep_summary.md`; JSON per pair under
`fix_local/res/w8-fx-isa-fx1/`.

| item | result |
|---|---|
| **The Q4 / M5 s_nop gap the verifier saw** | Not code. A gfx950 code object ends with about 260 `s_nop 0` of padding after the last block. `rv_isa split` gives them to whichever stage the block layout puts last. FlyDSL lays M5 out last (M3 with the stack's MoE externs). With the rocKE Q4 linked, LLVM places Q4's body after M5 at r1 (Q4 = 33 instructions in place plus the rest at the end), so Q4 gets the padding. Without it, Q4 r1 has 558 vs 547 executed s_nop cycles and M5 has 193 / 193. Q4's DPP-chain nops are FlyDSL's to within 2 % in every variant. |
| **Registers** | Default port, plain: r1 VGPR 246 / 246 and SGPR 103 / 103; r4s VGPR 228 -> 224 and SGPR 102 -> 99 (fewer). Served (`stk` / `pk`): VGPR within 1, except r3 at 216 vs 250 (fewer). |
| **Spills / scratch** | The 68-byte private segment is a reserved frame with no scratch instruction (alola rule 210). It appears on both sides: plain FlyDSL at r2 / r2s, the port at r3; served, both at r2 / r3 / r4, and the port alone at r1. SGPR spills: served FlyDSL has 2 at r1 / r4s where the port has 0; the port has 2 at r4 (`pk`, `pp`) and 16 at r2 when FlyDSL's Q3 / Q4 are mixed with the rocKE Q5-Q7 (`pp57`, `pk57`). These are SGPR-to-VGPR-lane spills, and r2 is not a K=3 / K=0 CONC-1 width. No VGPR spills on either side. The first (`asm`) port's 68 bytes are gone in the default port at r1 (plain). |
| **MFMA interleave** | Q3-Q7 have no MFMA on either side. Whole-kernel MFMA counts and MFMAs right after a `(0)` wait are equal in all 49 pairs, e.g. served r4s 198 / 198 and 94 / 94. The non-QSA stages keep an order ratio of 0.89-1.00. |
| **Instruction order (Q3-Q7)** | Q5: 0.83-0.93 at r1 and 0.28-0.92 at the other widths. Q3: 0.82-0.84 plain and 0.63-0.68 on the served dense path. Q4: 0.63 at r1 plain and 0.26-0.29 elsewhere (0.66-0.67 with `q4_umax=ir`; the same operation multiset, a different loop structure; branches 94 vs 105 at r1). Q6: 0.69-0.78. Q7: 0.42-0.76. |
| **Waitcnt** | Plain: Q3, Q4 and Q6 within 2 of FlyDSL's count, Q5 +1 to +6, Q7 +1 to +9. Served dense Q3: +3 to +11 (r4s 59 vs 48, of which 16 placements FlyDSL lacks). Served Q7 at r3: +39 to +48 (order 0.42-0.43); r3 is not a CONC-1 width. Placement ratio 0.31-0.99. |

Causes found in the ISA:
- **(a) Q7's cross-wave sums.** FlyDSL derives one LDS base from `0x10ae0 + 4 tid` and reads pairs at 512-byte
  offsets (`ds_read2st64_b32`). The port forms each address as `4 tid + C` with C above the 16-bit DS offset field,
  so each read gets its own `v_add` and no read2 pairing. The result is 5 more `ds_read_b32` and 6 more `lgkmcnt`
  waits at r1.
- **(b) Hint polls, by design.** They load only the tag dword (`buffer_load_dword` where FlyDSL uses `_dwordx2`).
- **(c) Integer index arithmetic.** FlyDSL's signed floor-division idiom gives `s_ashr` / `s_cselect` / `s_subb`
  sequences; the port uses unsigned shifts. Same values, different SALU / VALU streams.
- **(d) The served dense Q3 (`qdense=3` key-only path) at r4s.** The port uses 4 `s_load_dword` where FlyDSL uses
  3 `s_buffer_load_dword`, and adds an early wait group: +11 `s_waitcnt`. That fits F8: FlyDSL's Q3 / Q4 under the
  rocKE Q5-Q7 is 0.4 % faster at r4s. With FlyDSL's Q3 / Q4 (`*57`), Q3 and Q4 are again identical in order
  (0.99-1.00) and in waits.

So parity by construction holds at the operation level and bitwise. In the ISA it holds in registers and MFMA
interleave, and the s_nop budget is FlyDSL's once the padding is removed. Instruction order and part of the waitcnt
placement differ for causes (a), (c) and (d).

Two knobs follow from the diff and are not built yet:
- `lds_base` split: one region base plus small DS offsets, for (a).
- dense Q3 scalar loads in FlyDSL's `s_buffer_load` form, for (d).

Both are named in "Not done (fix round)".

### F6. Sub-1 % verdicts, re-timed with separately loaded copies (issue 7)

`bench_rc.py` builds EVERY arm, the FlyDSL base included, 4 times as separately compiled and loaded code objects.
Copy 0 is the arm. Copies 1-3 are the same IR under other kernel names, using no-op schedule levers (`pause=0` /
`pause.<a1|a2|q1|q2|c1>=0`). All 4 copies of every arm had one identical `.text` sha, so the lottery is placement,
not code.

Every copy is checked bitwise before timing, and all copies are timed in shuffled rounds (24 at L 1024, 16 at long
L) with clocks sampled. The band is the medians of the base's own copies (its second graph and its 3 recompiles).
- Arm verdicts: WIN = the worst copy median of the arm is below the band minimum by 0.002, and at least 3/4 of the
  pooled rounds are below 1.
- `X:Y` comparisons: BETTER / WORSE only when every copy median of X clears every copy median of Y by 0.002.
  Otherwise UNRESOLVED.

Every arm's numerics passed on both nodes. Pooled ratios to the FlyDSL base; the bracket is the copies' range:

| item (verifier's list) | n01-09 | n07-17 | resolved verdict |
|---|---|---|---|
| r4s, FlyDSL Q3 / Q4 + rocKE Q5-Q7 (`pick57`) vs all five (`pick`), L 1024 | 0.9956 (0.9605-0.9619 vs 0.9653-0.9661) | 0.9965 (0.9610-0.9616 vs 0.9636-0.9664) | **`pick57` BETTER on both nodes**: the side finding is real, 0.4 % |
| r1, the same comparison | 1.0094 | 1.0077 | `pick57` WORSE on both nodes: keep all five at r1 |
| N4b `tile` bf16 vs its exact comparator `pick57`, r1 | 1.0060 | 1.0084 | N4b WORSE on both nodes (0.6-0.8 %): the "behind at r1" verdict is real |
| N4b `tile` bf16 vs `pick57`, r4s | 0.9996 | 1.0000 | UNRESOLVED on both (a tie); `tile` f32 WORSE (1.0034, n01-09) |
| `hold.q7=64` on the FlyDSL stages (`flyhold`) vs base, r1 | 1.0038 (copies 0.998-1.008, bimodal) | 0.9988 | NEUTRAL on both: the hold is not a FlyDSL gain |
| the hold on the port (`pick` vs no hold), r1 | 0.974 | - | BETTER |
| r1 default port at L 131,072 | 1.0068 (1.0065-1.0070) | 1.0092 (1.0088-1.0104) | **LOSS on both nodes, real** (0.7-0.9 %); at 32K 0.999 and at max 1.001 (NEUTRAL) |
| r4s default port at long L | 0.989 / 0.991 / 0.993 (32K / 128K / max) | 0.990 (128K) | WIN at every L |
| long-L rejections on base `prepoll` (cl3: `pbar=compiler`, `q3_batch=16`, `q3_switch=0`), 128K | - | r1 0.997 / 1.003 / 0.998; r4s 1.001 / 0.999 / 0.997 | UNRESOLVED: within the lottery, and the rejections stand |
| `q4_chunks=16,q4_thr=8192` (`thr`) on base `prepoll`, r1 128K | 0.990 BETTER | 0.982 BETTER | **cl3's rejection is overturned at long L** (cl3 timed one compile) |
| `thr` on the r1 pick (`pkthr` vs `pick`) | 32K 0.997 BETTER, 128K 0.993 BETTER, max 0.988 BETTER | L 1024: 1.008 WORSE | L-dependent: a win at 32K-max, a 0.8 % loss at the served L |
| `q4_chunks=16` alone, r1 | 128K unresolved, max 0.997 BETTER, 32K 1.082 WORSE | 128K 0.987 BETTER | L-dependent, as section 6.2 said |
| `q5_kpre` at r4s (`pk` vs `pp`) | L 1024 unresolved; 128K 1.004 and max 1.005 WORSE | L 1024 unresolved | neutral at the served L, a small loss at long L |
| `q4_umax=ir` at r4s (`ppir` vs `pp`), L 1024 | 0.9957 | 0.9988 | UNRESOLVED |

What changes:
- **The served r4s pick becomes FlyDSL's Q3 / Q4 with the rocKE Q5-Q7** (`prepoll` + `q5_kpre`), resolved on both
  nodes. Its e2e run is F2's e3b.
- **The r1 pick stays all five functions** (`prepoll`, `q5_kpre`, `hold.q7=64`).
- `q4_chunks=16,q4_thr=8192` is a knob that wins at L >= 32K and loses at the served L. A function serves every L of
  its variant, so it stays out of the served pick: rejected on base `pick` at L 1024, accepted on base `prepoll` at
  32K-max (decision 15), and to be re-tested on every new base.
- The r1 default loss at 128K is real; the climbed exact set wins there by 2.7 % (`pick` 0.973).

### F7. Report inconsistencies and the Rust direct backend (issue 6)

Fixed in place:
- section 2 step 6 now says WaveScope ran (section 8.1);
- the front summary now says 14 must-fail mutants (9 numeric + 5 T0) and names them.

Rust direct backend levers (method step 4): still not climbed, now with the concrete blockers.
1. The functions are LLVM-IR device functions that FlyDSL links into its kernel module before LLVM optimization
   (`ATOM-truemono .../kernels/extern_stages.py`, docstring; MLIR internalizes and inlines them), and COMGR compiles
   the whole FlyDSL kernel. The direct backend turns serialized rocKE IR into a complete code object
   (`rocke/platform/python/rocke/runtime/code_object.py`, `build_direct_rocke_ir_hsaco`); it has no step that links
   into a FlyDSL module.
2. Even as a rocKE kernel, the bodies use this package's own `q38.*` ops (`q38.intr`, `q38.ifr` / `q38.yield`,
   `q38.poll`, `q38.bin`), which only `qsa2/lower.py` lowers. The direct parser accepts only the `arith.` / `math.` /
   `vector.` / `memref.` / `tile.` / `gpu.` / `scf.` / `cf.` dialects (`rocke-codeobj/src/rir/parse.rs:51`) and
   refuses any other op (`parse.rs:57-66`).

The route is a rocKE layer kernel for Q3-Q7 (WP3's `layers=1` form, as WP2a built for the HC runs) with the `q38`
ops expressed in direct-lowered ops (`tile.dpp_op` already covers the DPP moves / max). The per-stage direct scopes
can then be climbed as WP2a did. Named in "Not done (fix round)".

### F8. The r4s Q3 / Q4 placement, followed up (issue 8)

- **Layer:** resolved BETTER on both nodes (F6).
- **Cause (F5 (d)):** the served dense Q3's scalar loads and its 11 extra waits at r4s.
- **e2e:** new arm `full3_w8c57` (`arms.d/w8c2.sh`; `ATOM_Q38_EXTERN_MISSING=mono` keeps FlyDSL's Q3 / Q4 for r4s
  only). The server log shows "no q3 / q4 function for build variant r4s ... runs the FlyDSL stage" and the rocKE
  Q5-Q7 r4s lines in every run; the expect file asserts both.
- **9.1 rule (F2):** a WIN on K=3 ms per round vs `full3` at the band edge, on both nodes. NEUTRAL vs `full3_w8c`:
  0.4 % of one layer type is below what 6 e2e pairs resolve.
- **Recommendation:** the r4s placement follows the layer verdict. The served function set for K=3 is `full3_w8c57`
  with `q5_kpre` at r4s (e3b's arm).

### F9. Git (issue 9)

The fix round changed no git state. The HEAD commit "Qwen38 rocKE monokernel work" is the user's.

### Updated verdict

**PASS for the port gates; the speed claims are restated under the lottery-resolved and E2E rules.**
- Bitwise parity and mutants hold, re-confirmed in every fix-round arm on both nodes.
- Served L, in situ, resolved on both nodes:
  - r1: 0.947-0.949 (all five functions);
  - r4s: 0.961 (FlyDSL's Q3 / Q4 + the rocKE Q5-Q7) and 0.965 (all five);
  - defaults: r1 parity, r4s 0.986-0.987.
- Long L: the climbed sets win at every L. The r1 default loses 0.7-0.9 % at 128K, a real and disclosed loss.
- E2E (9.1 rule): K=3 ms per round WIN vs the consolidated stack (0.992-0.997) and vs the hybrid (0.936-0.939);
  request tok/s NEUTRAL vs the stack and WIN vs the hybrid. K=0 request tok/s WIN vs the consolidated stack (1.015,
  6/6 pairs, both twins' bands cleared) and vs the hybrid (1.116).

### Not done (fix round)

- **The two parity knobs from F5:** the `lds_base` region split (Q7 read pairing) and dense Q3 scalar loads in
  FlyDSL's form. Then re-time Q3 r4s (all five functions) against `pick57`.
- **The Rust direct backend** for Q3-Q7 (F7): needs the WP3 rocKE layer kernel and direct-lowered forms of the `q38`
  ops.
- **A per-L choice** of `q4_chunks` / `q4_thr`: the function is per variant only. Using it at long contexts would need
  a length-dependent switch inside Q4 (the knob already has one: `q4_thr`) that does not move the code at the served
  L, which `pkthr` does today (F6).
- Carried over unchanged: wide / grouped widths, and the ATOM-side `qdense` guard.
