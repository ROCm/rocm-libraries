# WP2c: QSA Q3-Q7 as rocKE stage functions (W8, wave 2, 2026-11-01)

D0 section 5 (wave 2b, WP2c) and DESIGN 5 / 7.8 / 9.1 / 12.3-12.4; PLAN decisions 13, 15, 16. This file gives ratios,
shares and counts only. Absolute device times and tok/s: `hackathon-prep/private-perf/qwen38-true-mono/wave2/WP2c.md`.
No git state was changed, no file was deleted, nothing in `dev/ATOM` or `dev/rocm-libraries` was touched; ATOM-truemono
was not edited (the ports link through its existing `ATOM_Q38_EXTERN_STAGES`).

**Verdict: gate met for the bitwise port (served L AND 32K / 128K / max, every narrow width, ties, mutants); the
port beats the FlyDSL stages in situ on the served stack once its catalogued knobs are climbed; N4a / N4b pass T0.**

- **Port.** Q3, Q4, Q5, Q6, Q7 of the FlyDSL mono are rocKE stage functions (new package
  `kernels/gfx950/qwen38_mono/qsa2/`), linked in place of the FlyDSL stages for every narrow variant
  (r1-r4, r2s-r4s), plain and on ATOM's `numa=1` mailbox layout.
- **Parity by construction (decision 16).** Per stage and variant, the f32 / f64 operation multiset, every DPP /
  permlane / readlane, every LDS access, atomic and barrier of the rocKE function equals the FlyDSL stage's final
  `.ll` (21 of 21 stage-variant pairs checked); after the `fly` mailbox (section 3.2) the ISA memory forms match too.
- **Bitwise numerics.** 85 + 35 + 161 + 55 in-situ cells (the layer kernel with the functions linked vs the FlyDSL
  build from one snapshot and tag) are bitwise equal in every byte of the scratch (both halves), the QSA caches and
  the layer output: 7 variants, contexts 639 to 262,136 (the maximum), planted Q4 ties across chunk boundaries,
  `numa=1`, the served stack's schedule, every exact knob of the climbs. 10 must-fail mutants differ where they must.
- **Speed in situ (DESIGN 9.1, two controls, WIN rule).** Defaults on the served stack (`stackmoe` = the stack's
  schedule + its rocKE fused MoE): r1 0.999 (parity), r4s **0.980 WIN**. Climbed (bitwise): r1 **0.948**, r4s
  **0.956-0.963** of the FlyDSL stage's layer, the WIN repeated on the other node (section 6.3).
- **End to end (decision 14, quick track, 3 passes, same job as the hybrid; section 6.4).** The consolidated full
  stack plus the functions with the climbed picks (`numa=1,qdense=3,prepoll=q5+q6+q7`, r1 `q5_kpre`, `qsa.r1`
  `hold.q7=64`): K=3 median ms per round **0.991** of the full stack (0.939 of the hybrid), tok/s per request 1.010;
  K=0 tok/s per request **1.017** of the full stack (1.114 of the hybrid); GSM8K correct within the arms' spread.
  A first job with `prepoll` only gave 0.990 / 1.013.
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
6. **WaveScope**: not run (section 9); the stamped stage views (section 6.2) stand in for the stage-level picture.

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
  narrow width); the ISA / stamp difference of Q3 r4s inside the layer is the next item.
- **WaveScope ATT before / after** (method step 6): run (job n4b2, n01-09, r4s, L 1024, one traced dispatch per
  build; section 8.1). One dispatch per arm is a structural view, not a timing: the in-situ timings above are the
  verdicts.
- **Wide / grouped variants** (r8g4, r16g4, ...): no function (the served stack keeps FlyDSL's there through
  `ATOM_Q38_EXTERN_MISSING=mono`; e2e log `extern-missing` counts them).
- **Rust direct backend levers** for these stages: the functions are inlined into FlyDSL's COMGR build (hybrid path);
  per-stage direct scopes apply once the stages run in a rocKE layer kernel (WP3).
- **ATOM-side guard**: a function set whose `qdense` disagrees with the schedule's deadlocks the layer (alola rule
  210); the sidecar records `stage_knobs`, ATOM does not check it yet (an additive ATOM-truemono check is a follow-up).
