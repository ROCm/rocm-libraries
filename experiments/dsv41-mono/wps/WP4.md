# WP4: small-batch stage work on the V4.1 mono (rows <= 48), Path B

W13, 2026-11-03 / 04: round 1, then the fix round ("r2"). MI355X (gfx950), TP2 no-EP, DSpark 5, the B0 recipe (bf16
KV, fp8 index plane, `--level 0` + FULL graphs). This file has ratios only. Absolute numbers are in
`hackathon-prep/private-perf/dsv41-mono/WP4-abs.md`, `raw/wp4/` and `raw/wp4r/`.

## Verdict

**Every WP4 item is now built or closed with a measured reason (§1). All new code is default-off. Every bitwise
item passes its in-place gate on both nodes, and every must-fail mutant is caught. The D0 expectation (C1 1.22,
C2 1.18, C4 1.12, C8 1.06) is NOT reached, and the 3x target is far out of reach for this WP.** The best measured
configuration takes about 1.5-2 % off ATOM's GPU decode step at C1 (PENDING_FINAL for the combination). End to end, that is inside the served
noise.

The best configuration is `Tu`:
- the rocKE GEMV family plus the rocKE routed `ug` with `sop` / `ra`;
- `askip`, `poll1` and `xq`.

The tuner's best (§5) is the same set without `ug`.

Per CONC, best WP4 configuration vs ATOM's best served TP2 path. ATOM's best path is mono + DSpark at C1-C8;
native ties mono at C8 / C16. Served values are per-request decode tok/s from fresh servers, with rotated passes
on both nodes and a permutation test against the A-vs-A twin:

| CONC | D0 expected | best WP4, GPU decode step (HTL) | best WP4, served | verdict |
|---|---|---|---|---|
| C1 | 1.22 | PENDING_FINAL | PENDING_FINAL | miss |
| C2 | 1.18 | PENDING_FINAL | PENDING_FINAL | miss |
| C4 | 1.12 | PENDING_FINAL | PENDING_FINAL | miss |
| C8 | 1.06 | PENDING_FINAL | PENDING_FINAL | miss |
| C16 | (WP6) | not measured | every arm inside the twin (§3.6) | reference only |
| TP1 | must run | - | - | not addressed here: TP1 is WP7. Every WP4 build refuses TP != 2 and keeps ATOM's stage. |

## 1. Status of every D0 §4 WP4 item

All knobs default off. With every knob unset, the traced K1 / K2 IR equals pristine df86c482. This was checked
locally after every edit of this round, and on the cluster for `.pre.ll` and `.text` of K1 s6 ix0 / ix1, K1 s12,
K2 s6 / s48 and draft K2 s5.

The gate column means: in place, every byte a launch writes (scratch included) is compared, with LDS poisoned
between the reference and the candidate. The last column is the GPU decode step vs ATOM (HTL), against the mean
of the base and its A-vs-A twin.

| item | knob | status | gate | GPU decode step vs ATOM |
|---|---|---|---|---|
| 4.1 K2 per layer kind | `ATOM_DSV41_RK_K2SPLIT=1` | built (r1) | PASS, both nodes (r1) | tie |
| 4.2 FP8 GEMV family on the `stage_wqkv` core | `ATOM_DSV41_RK_EXTERN=wqkv=;wqb=;woa=;wob=;shared=` (`eA`) | **built**: rocKE `stage_wqb`, `stage_woa`, `stage_wob`, `stage_shared`, plus P2's `stage_wqkv` | **PASS**: 0 differing launches; 5 of 5 mutants caught | **0.986-0.995 at C1** (5 jobs, both nodes); 0.987-1.001 at C2 / C4; 0.993-1.004 at C8 |
| 4.2 `sop` / `ra` on those stages | function set `sop2ra` | built | PASS | tie with `eA` |
| 4.3 K2b MXFP4 MoE port | `ATOM_DSV41_RK_EXTERN=ug=` | **built (routed ug)**: rocKE `stage_ug` is ATOM's routed gate / up GEMV (MXFP4 x MXFP8 scaled MFMA) plus its silu / MXFP8-requant epilogue. Groups 1 and 2, rows 1-16. | **PASS**: 0 diffs at rows 1-11, groups 1 and 2, target and draft, both nodes; 5 of 5 mutants caught | default form: tie (0.995-1.006); `sop=1,ra=1` alone (`uGs`): 0.989 at C1, tie above. `sop=1,ra=1` with `eA` (`eUs`): **0.981 at C1** (DSpark), **0.978-0.986 at 1-4 rows** (non-speculative, both nodes), 0.990-0.999 at C2-C8 |
| 4.3 MoE down / combine | - | not ported. The `dsop` lever applies the port's one code-generation change to ATOM's own down stage. | `dsop` PASS | tie (in situ 0.997-0.999) |
| 4.4 one launch a layer | `ATOM_DSV41_RK_LAYER=1` | **built** for the non-indexer layers: K1 + K2 merged with a grid-wide hand-over | **PASS**: 18,432 launches a rank, 0 diffs. Mutants: `nowb` 1,888 / 1,888; `nosync` 33 / 1,408 (a race, caught when it bites) | **loss: 1.054-1.069 at C1**, 1.044-1.051 at C2, 1.025-1.037 at C4, 1.018-1.026 at C8; non-speculative 1-4 rows **1.071-1.097**. Served C1 0.952 (p < 0.001). |
| 4.4 fold compressor / candidates / Engram / meta | - | not built (§4: the fold's own mechanism is the `layer=1` loss) | - | - |
| 4.5 holds | `ATOM_DSV41_RK_LEV=hold.<stage>=N`, `shold`, `qhold` | built | PASS | tie (0.994-1.001) |
| 4.5 spread / prefetch | `pfx` (wo_b weights issued before wo_a) | built | PASS | **loss**: 1.017-1.022 at C1 / C2 |
| 4.5 XCD homing | `xcd` (one tile); `xq` (XCD-homed ug queues past one tile) | built | PASS. With `xq`, only its own queue counters differ (9-18 scratch bytes in the K2 launches above 16 rows); every output is identical. | tie alone (0.995-1.003); part of the tuner's best |
| 4.5 prepoll / poll shape | `poll1` (one do-while poll loop); `PAUSE=2` | built | PASS (`poll1`); PAUSE is bitwise by construction | tie alone (0.994-1.006; PAUSE=2 0.996-1.003); part of the tuner's best |
| 4.5 NUMA dual copies | `ATOM_DSV41_RK_NUMA=1` | built (r1) | PASS on **both** nodes. n01-09: 12,652 + 19,200 launches, 0 diffs; `put_bit` 16,462 / 16,462. | tie. Served tie (C1 1.010, C8 1.002). |
| 4.5 consumer-homed single copy | `ATOM_DSV41_RK_NUMA=4` / `=5` | **built** | PASS: 12,400 + 19,200 / 12,394 launches, 0 diffs. `put_bit` 14,920 / 14,920. `home_flip` deadlocks: caught. | **loss**: 1.008-1.024 / 1.011-1.041 |
| 4.5 epoch tags | `ATOM_DSV41_RK_CCL` epoch form | built by WP5, not duplicated here | WP5 | WP5 |
| 4.6, bitwise part: dead attention splits | `askip`, `pskip` | built (`askip` fixed in r2: §3.4) | PASS. `askip`: 12,376 + 19,200 launches, 0 diffs; `skip_live` 2,598 / 2,598 K2 launches. | `askip` **0.992-0.995** at C2 / C8, 0.995-0.999 at C1; `pskip` tie |
| 4.6 row-scaled attention above 12 rows | `ATOM_DSV41_RK_LEV=asplit` (TOLERANCE) | built: 32 splits x 32 keys above 12 rows | **PASS (accuracy)**: GSM8K and passkey inside the twin (§3.4) | tie (0.998-1.004) |
| 4.5 MALL prefetch | - | W14 T1 owns it: `w14/T1-mall-prefetch.md` | - | ceiling 0.06-0.07 of K2 at 6 rows (perfect predictor) |
| 4.7 Rust direct backend | - | **not applicable** to these stages (§4) | - | - |

## 2. What was built in this round

ATOM (`dev/ATOM-dsv41`, `atom/models/deepseek_v41/mono/kernels/`; additive, default off):

- `rk_lev.py` / `rk_lev_dev.py`: `ATOM_DSV41_RK_LEV=<lever>[^<lever>..]`.
  - Code-generation levers: `usop`, `dsop`, `ura`, `gsop`, `gsop2`, `gra`, `askip`, `pskip`, `poll1`, `asplit`.
  - Schedule levers: `hold.<stage>=N`, `shold`, `qhold`, `pfx`, `xcd`, `xq`, `ua=N`, `nop=N` (a recompiled no-op
    control).
  - A lever is captured at build time and enters the build key and the kernel name (`_lv<hash>`).
  - Every lever was checked to change the traced code where it applies.
  - The gate (`ATOM_DSV41_RK_LEV_CHECK=<dir>`) runs ATOM's build and the lever build from one snapshot. It
    compares every byte a launch can write (26-86 MB).
  - Mutants (`ATOM_DSV41_RK_LEV_MUTANT`): `opsel`, `scale_byte`, `bit`, `chain_drop`, `skip_live`.
- `rk_poison.py`: between reference and candidate, 256 CTAs fill all 160 KiB of LDS on every CU with
  `ds_write_b32` asm. gfx950 keeps LDS contents across launches, so without it a candidate that skips an LDS write
  reads the reference's data.
- `rk_extern.py`: stages `wqb`, `woa`, `wob`, `shared` and `ug`, beside P2's `wqkv`. The call sites in `moe.py`,
  `attn_pre.py`, `attn_post.py` and `layer_post.py` choose in Python, so with the knob off the traced code is
  ATOM's.
- `rk_layer.py` (4.4): `ATOM_DSV41_RK_LAYER=1`, one kernel a layer.
  - It holds K1's and K2's bodies with a union LDS.
  - The grid-wide hand-over is `vmcnt(0)`, barrier, `buffer_wbl2 sc1`, an agent-scope arrival counter, then
    `buffer_inv sc1`.
  - Refused for indexer layers, timeline / debug builds, and with NUMA, PAUSE, EXTERN, CCL or PREFETCH.
- `rk_numa.py`: mode 4 (a consumer-homed single copy where the consumer CTA is static, dual copies otherwise) and
  mode 5 (homed single copy, copy 0 elsewhere). New mutant: `home_flip`.
- `rk_attn2.py` (4.6): `asplit`, the decode attention in 32 splits of 32 keys above 12 rows.
- `rk_bench.py`: in-situ per-launch A/B timing of variants inside a running engine.

rocKE (`library/kernels/gfx950/dsv41_mono/`, all new files; `qwen38_mono` untouched):
- 4.2, on `stage_wqkv`'s core with its `WqkvKnobs` and mutants: `stage_wqb.py`, `stage_woa.py`, `stage_wob.py`,
  `stage_shared.py`.
- 4.3: `stage_ug.py`.

Each one re-implements the ATOM stage it replaces bit for bit. It is lowered as an `lds="arg"` device function
and linked by `ATOM_DSV41_RK_EXTERN`.

Harness (new files, `hackathon-prep/dsv41/wp4r/`):
- job, submit and held-release scripts;
- `h/wp4_run.sh`: gate kinds `k2split | numa | lev | ext | layer`; phases `gate..gate5`, `bench`, `htl`, `htl2`
  (non-speculative), `tune`, `served`, `acc`;
- `h/tune.py` (the surrogate tuner);
- `srv_pool.py` (served pooling and the permutation test);
- `lt.py` / `lt.sh` (local trace and identity).

## 3. Gates and measurements

### 3.1 Bitwise gates (both nodes)

Offline engine, eager. DSpark seeds over the C1 / C2 / C4 / C8 groups, plus non-speculative runs. Counts are per
rank:

| gate | node | launches | differing |
|---|---|---|---|
| all levers at once (`usop dsop ura gsop2 gra askip pfx shold qhold xcd poll1`) | n07-17 | 12,216 + 12,812 + 19,200 | 0 |
| all levers | n01-09 | 12,302 + 19,200 | 0 |
| each lever alone (12 levers, plus `pskip`, `hold`, `ua=2`, `ua=3`) | n01-09 | 5,000-7,400 each | 0 |
| `askip` (fixed) | n01-09 | 12,376 + 19,200 | 0 |
| `xq` | n01-09 | 6,676 | 0 outside its own queue counters |
| extern `wqkv wob wqb` | n01-09 | 12,296 + 19,200 | 0 |
| extern, all five GEMV stages | n01-09 | 11,878 + 19,200 | 0 |
| extern `ug`, groups 1 (rows 1-6, 10, 15; target and draft) | n01-09 | 12,400 + 10,220 + 23,040 | 0 |
| extern `ug`, groups 2 (rows 7-11) | n07-17 | 11,520 + 15,360 | 0 |
| extern `ug` with `sop=1,ra=1` | n01-09 / n07-17 | 9,624 / 7,680 | 0 |
| extern, all six stages | n01-09 / n07-17 | 10,226 / 7,680 | 0 |
| best combination (six stages + `askip poll1 xq`) | n07-17 / n01-09 | 11,786 + 23,040 / PENDING_FINAL | 0 outside the `xq` counters |
| `layer=1` | n01-09 | 4,224 + 3,328 + 7,680 + 3,200 | 0 |
| NUMA 1 | n07-17 / **n01-09** | 12,382 + 19,200 / 12,652 + 19,200 | 0 |
| NUMA 4 / NUMA 5 | n07-17 | 12,400 + 19,200 / 12,394 | 0 |

Mutants must fail. Counts are launches with a diff over launches that ran the mutated code:

| mutant | caught |
|---|---|
| lever `opsel` | 3,986 / 3,986 and 3,636 / 3,636 |
| lever `scale_byte` | 2,036 / 2,036 (the K2 launches) |
| lever `chain_drop` | 3,470 / 3,470 |
| lever `skip_live` (fixed `askip`) | 2,598 / 2,598 (the K2 launches) |
| extern GEMV `put_bit`, `chain_drop`, `scale_byte`, `row`, `opsel` | every launch that runs a mutated stage (e.g. 2,472 of 3,814 launches: the ones that run it) |
| extern `ug` `put_bit`, `chain_drop`, `scale_byte`, `row`, `opsel` | every K2 launch with a linked `ug`: 1,654-1,863 of 5,202-5,540 launches (DSpark, n01-09), 3,803-3,840 of 7,680 (groups 2, n07-17). `row` misses only 1-pick launches, where it changes nothing. |
| NUMA `put_bit` (with state restore) | 16,468 / 16,468 (n07-17), 16,462 / 16,462 (n01-09) |
| NUMA `bit` | 3,556 / 3,556 |
| NUMA 4 / 5 `home_flip` | deadlock, caught on both modes (the consumer polls a copy nobody wrote) |
| layer `nowb` / `nosync` | 1,888 / 1,888; 33 / 1,408 |

### 3.2 In-situ per-launch timing (`rk_bench`, n01-09)

Each launch is timed as ATOM's build and as each variant, interleaved, in one running engine. Recompiling an
unchanged K1 under a new name moves its time by about 1 %, a "code-object lottery": every K1 variant, K2-only
levers included, sits at 0.988-0.995 against the twin's 1.003. So K1 ratios below 1 % are not attributable, and
every later HTL job carries a recompiled no-op (`nop=N`) as its control.

K2, median per-layer ratio vs ATOM at 6 / 24 / 48 rows:

| variant | 6 rows | 24 rows | 48 rows |
|---|---|---|---|
| twin | 1.004 | 1.003 | 1.003 |
| `usop` | 0.994 | 1.009 | 1.014 (loss at 24-48) |
| `dsop` | 0.999 | 0.999 | 0.997 |
| `ura` | 1.003 | 0.998 | 1.001 |
| `gsop` | 0.998 | 0.999 | 1.000 |
| `gra` | 0.998 | 0.997 | 0.999 |
| `pfx` | 1.005 | 0.998 | 1.001 |
| `xcd` | 1.007 | 0.999 | 1.000 |
| all levers together | 1.010 | 1.006 | 1.012 |
| extern `wqkv wob wqb` | 0.998 | 0.998 | 1.002 |

### 3.3 GPU decode step (HTL: FULL graphs, fresh engine per arm-run, rotated passes)

Ratio arm / mean(ATOM, ATOM twin), per pass. Eight jobs: n01-09 `r3c r4b r8a r5c r9e(non-spec) r6d(tuner)`,
n07-17 `r3a r9d`.

| arm | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| A-vs-A twin | 0.997-1.007 | 0.994-1.003 | 0.995-1.006 | 0.997-1.003 |
| recompiled control `nop` | 0.998-1.022 | 0.993-1.006 | 0.997-1.012 | 0.996-1.004 |
| `eA`, extern GEMV family | **0.986-0.995** | 0.987-1.001 | 0.983-1.000 | 0.993-0.998 |
| `uG`, extern `ug` | 1.001-1.006 | 1.003-1.004 | 1.003-1.004 | 0.995-1.003 |
| `eU`, all six (`ug` default) | 0.994-1.000 | 0.994-1.004 | 0.997-1.014 | 0.995-1.003 |
| `uGs`, extern `ug` sop / ra alone | 0.989 | 0.991-1.001 | 0.991-1.003 | 0.995-0.998 |
| **`eUs`, all six (`ug` sop / ra), n07-17** | **0.981** | 0.993-0.999 | 0.990-0.996 | 0.994-0.996 |
| `askip` (fixed) | 0.995-0.999 | 0.992-0.993 | 0.998-0.999 | 0.993-0.994 |
| `asplit` / `asplit pskip askip` | 1.002-1.004 / 0.995-1.003 | 0.998 / 0.995-1.001 | 0.998-1.000 | 0.998-1.003 / 1.000-1.003 |
| `layer=1` | 1.054-1.069 | 1.044-1.051 | 1.025-1.037 | 1.018-1.026 |
| `layer=1`, no L2 write-back (`nowb`, incorrect: diagnostic) | 1.007-1.022 | 0.969-0.987 | 0.907-0.917 | 0.997-1.006 |
| `layer=1`, no arrival wait (`nosync`, incorrect: diagnostic) | 1.012-1.018 | 1.003-1.007 | 0.999-1.004 | 1.011-1.018 |
| `poll1` | 0.996-1.001 | 0.994-0.998 | 0.997-1.001 | 0.996-1.006 |
| holds (`wob 16, shared 32, wqb 16`) | 0.995 | 0.994-0.998 | 0.999-1.001 | 0.997-1.000 |
| `xq` | 0.999-1.003 | 0.996-0.998 | 0.996-0.997 | 0.995-1.001 |
| `pfx` | 1.020-1.022 | 1.017-1.019 | 1.005-1.006 | 0.998-1.004 |
| PAUSE=2 | 1.001-1.003 | 0.998-1.002 | 0.996-0.998 | 1.002-1.003 |
| NUMA 4 / NUMA 5 | 1.022-1.024 / 1.041 | 1.008-1.010 / 1.020-1.022 | 1.009-1.011 / 1.012-1.014 | 1.012-1.017 / 1.011-1.015 |
| n07-17: `usop dsop ura gsop2 gra` | 0.993 | 1.005 | 1.015 | 1.008 |
| all levers at once | 1.050-1.056 | 1.072-1.073 | 1.032-1.037 | 1.028-1.043 |
| best combination `Tu` (six stages + `askip poll1 xq`) | PENDING_FINAL | | | |

Non-speculative (1-4 rows, `htl2`), both nodes:

| arm | C1 (1 row) | C2 | C4 |
|---|---|---|---|
| `eA` | 0.986-0.995 | 0.991-0.993 | 0.983-0.991 |
| `eUs` | 0.982-0.986 | 0.978-0.986 | 0.981-0.984 |
| `layer=1` | 1.093-1.097 | 1.085-1.092 | 1.071-1.081 |

Reading:
- The rocKE GEMV family is the one consistent single win: 1 % at C1 on five jobs, below the `nop` control on every
  pass.
- The routed `ug` port in ATOM's code-generation form is a tie. In its op_sel / read-ahead form, stacked on the
  family, it takes the win to about 2 % at C1 (DSpark) and at 1-4 rows (non-speculative), 0.5-1 % at C2-C8. The same change in ATOM's own stage (`usop` + `ura`) was a tie /
  loss, so the rocKE function's schedule, not the `op_sel` alone, carries it.
- `askip` saves 0.5-0.8 % at C2 / C8, where score waves run several splits.
- Everything else is a tie or a loss. Stacking every lever loses: the `pfx`, `usop` and `xcd` losses add up.

### 3.4 Attention: the `askip` fix and the tolerance arm

**`askip` was a no-op in the first r2 jobs.** Its helper was a plain Python function, so `if split < written` on
traced values became a Python truth test and every DMA was still issued. Its `skip_live` mutant went uncaught
twice, the second time with the LDS poison, and that exposed the bug. The helper is now traced: the traced K2
differs from the old one, `skip_live` changes the code, and the fixed lever passes its gate with the mutant caught
(§3.1). A scan of every WP4 helper found no other untraced runtime condition.

**`asplit`** (32 splits x 32 keys above 12 rows) is not bitwise. K2 output tolerance vs ATOM over 690 launches
above 12 rows (gate `t0`):

| output | median max-abs / max-ref | median rel-L2 | worst rel-L2 |
|---|---|---|---|
| `ffn_out` | 1.1e-4 | 1.1e-3 | 7.5e-2 |
| `res_ffn` | 5.0e-4 | 1.1e-3 | 1.4e-2 |

At 12 rows and below its builds are bitwise: 23,040 non-speculative launches, 0 diffs.

Accuracy (served, n01-09, GSM8K 1319 3-shot chat, CONC 16):

| run | flexible | strict | passkey 8K / 24K |
|---|---|---|---|
| ATOM | 0.9121 | 0.1198 | 20 / 20 and 20 / 20 |
| twin | 0.8999 | 0.1236 | 20 / 20 and 20 / 20 |
| `asplit` | 0.9045 | 0.1198 | 20 / 20 and 20 / 20 |

The `asplit` scores sit inside the A-vs-A range: **PASS** (tolerance class). The timing is a tie, so it stays
default off.

### 3.5 `layer=1` cost split (diagnostic arms)

The two halves of the hand-over were removed one at a time. Without the arrival wait, C1 is still 1.012-1.018.
Without the L2 write-back, C1 is 1.007-1.022, but C4 is 0.907-0.917. That build is incorrect (1,888 / 1,888
launches differ), so its time also reflects different data. At C1 both halves cost about the same. The
`buffer_wbl2` dominates at C2 / C4, where more K1 output crosses XCDs.

### 3.6 Served (both nodes, 4 passes each, fresh server per arm-run, CONC 1 / 2 / 4 / 8 / 16)

Per-request decode tok/s, per pass arm / mean(ATOM, twin). Each cell gives the mean over 8 passes, then the
permutation p of the arm vs the twin:

| arm | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| twin (its own ratios) | 0.998 [0.975, 1.021] | 1.006 [0.940, 1.064] | 1.003 [0.973, 1.038] | 0.993 [0.945, 1.022] | 0.998 [0.959, 1.031] |
| `eA` | 1.016 [0.970, 1.050], p 0.12 | 1.025, p 0.55 | 0.998, p 0.63 | 0.989, p 0.76 | 0.994, p 0.91 |
| `poll1` | 1.008, p 0.35 | 1.023, p 0.60 | 0.996, p 0.65 | 0.995, p 0.89 | 0.988, p 0.64 |
| NUMA 1 | 1.010, p 0.32 | 1.029, p 0.53 | 0.986, p 0.19 | 1.002, p 0.48 | 1.006, p 0.69 |
| `layer=1` | **0.952, p < 0.001** | 1.002, p 0.92 | 0.968, p 0.06 | 0.985, p 0.78 | 1.021, p 0.34 |
| native (ATOM, no mono) | 0.614 | 0.725 | 0.905 | 0.984 | 1.017 |
| `T` / `Tu` (best combination) | PENDING_FINAL | | | | |

The served cells see a GPU-step result only where it is large (`layer=1` at C1). A 1-2 % GPU-step win is inside
the served spread: the twin alone spans 0.975-1.021 at C1.

## 4. Items closed without a build, and why

- **4.4 fold of compressor / candidate table / Engram / per-layer meta.** `layer=1` measures the fold's own
  mechanism. Replacing the K1 -> K2 launch boundary with an in-kernel grid hand-over costs 2-7 % of the step at
  DSpark widths and 7-10 % at 1-4 rows, on both nodes (§3.3, §3.5). Each folded launch would need the same
  grid-wide hand-over, because its producers and consumers run on different CTAs. The kernels it would fold are
  small: from the C1 kernel trace, the compressor is about 0.017 of the step, Engram about 0.007 and step_begin
  about 0.008. The fold was built, measured as a loss, and not extended.
- **4.7 Rust direct backend.** The direct compiler emits whole code objects (rocKE IR -> HSACO, no LLVM). The
  WP4 stages are LLVM device functions linked into ATOM's FlyDSL kernels; `device_function.py` is listed as "hybrid
  path only, n/a" in `qwen38-true-mono/research/rust-direct.md`. No standalone rocKE kernel on the served
  small-batch path exists to auto-select it for.
- **MALL expert prefetch.** W14 T1 owns it (`w14/T1-mall-prefetch.md`):
  - MALL hits read about 1.11-1.14x faster than HBM for a floor-quality stream.
  - With a perfect predictor, the ceiling is about 0.06-0.07 of K2 at 6 rows and 0.03 at 48 rows.
  - The early-router predictor recalls 0.72-0.77 of the union at top-6.
- **C16** belongs to WP6 (wide rows); the served C16 cell above is reference only. **TP1** belongs to WP7.

## 5. Path A re-test (D0.1) and the surrogate tuner

**Path A re-test.** D0.1's rule: switch to a persistent body only if its loop-body cost x is <= 0.99 per layer
at 6 rows and <= 1.00 at 1 row, on both nodes. The WP4 bodies now exist. The Path A element they can measure is
the layer boundary inside one launch (`layer=1`, K1 -> K2 by a grid-wide hand-over instead of a graph launch
boundary):

| width | n01-09 | n07-17 | rule |
|---|---|---|---|
| 6 rows (C1 DSpark) | 1.054-1.064 | 1.068-1.069 | needs <= 0.99 |
| 1 row (C1 non-speculative) | 1.095-1.097 | 1.093-1.097 | needs <= 1.00 |

**Outcome: the rule fails on both nodes at both widths. Path B stays the plan of record** and Path A stays a
default-off knob. A persistent pass would replace every launch boundary of the step with such a hand-over.

**Tuner.** `h/tune.py` fits a Bayesian linear surrogate over the lever indicators with pairwise interactions and
picks the next evaluation by lower confidence bound. Around that:
- combination sweep: every pair of open levers is tried with the incumbent;
- lookback: the incumbent's members are re-tested one at a time (drop / swap);
- the priors come from §3;
- a replicated base every few evaluations estimates the noise.

Each evaluation is a fresh-engine HTL run (C1 / C2 / C4 / C8). The objective is the mean GPU-step ratio vs the
interleaved base. The space was 23 lever values in 21 groups (`ug` default form included; its `sop` / `ra` form
was found later and joins in `Tu`); the run made 30 evaluations on n01-09, with base-pair noise 0.0015.

Result:
- The tuner accepted `eA + poll1` (0.993, replicated 0.992), then **`eA + poll1 + askip + xq` (0.984 / 0.988,
  replicated)**.
- 24 further pair and lookback evaluations did not beat it. Every third lever added (`dsop`, `gra`, `gsop`,
  `gsop2`, holds, `K2SPLIT`, `ura`, `usop`, `ua=2`, `pfx`, `qhold`, `pskip`) scored 0.986-1.003.
- Lookback dropping one member: without `poll1` 0.992, without `askip` 0.995, without `eA` 0.994, without `xq`
  0.999. Each member carries part of the win.
- Final configuration `T` = that set; `Tu` = `T` with the `ug` sop / ra function (§3.3, §3.6).

## 6. Jobs (all `rocke` partition, atom-dev nightly_202610171506)

| job | node | what |
|---|---|---|
| wp4r3a | n07-17 | knob-off identity dump; lever gates + mutants; NUMA 1 / 4 / 5 gates + mutants; HTL (cut by the time limit) |
| wp4r3c | n01-09 | lever + extern gates and mutants; in-situ bench; HTL 10 arms x 2 |
| wp4r4a | n01-09 | layer gates + mutants; five-stage extern gates + mutants; lever singles; NUMA second-node gate + mutants |
| wp4r4b | n01-09 | HTL 9 arms x 2 (`layer=1`, `eA`, `poll1`, `xq`, holds) |
| wp4r5c | n01-09 | fixed `askip` gate + `skip_live`; `asplit` tolerance; HTL 6 arms x 2; GSM8K + passkey |
| wp4r6d | n01-09 | tuner, 30 evaluations |
| wp4r7s1, wp4r7s7b | n01-09, n07-17 | served, 7 arms x 4 passes x 5 CONC each |
| wp4r8a, wp4r8b, wp4r8c | n01-09, n01-09, n07-17 | `ug` gates + mutants (groups 1 and 2); HTL 7 arms x 2 |
| wp4r9d, wp4r9e | n07-17, n01-09 | Path A: HTL `layer=1` + diagnostics; non-speculative 1-4 row HTL |
| wp4r10n01, wp4r10n07 | n01-09, n07-17 | best combination: gate, HTL, served 5 arms x 4 passes |

Failed or superseded (not used): wp4r6a / wp4r6b (a tuner driver bug: `setsid` forked under
`start_new_session`, so 32 engines started at once); wp4r5a / wp4r5b / wp4r7s7 (the n07-17 `/tmp` was lost and
the node reseeded); wp4r9a-c (resubmitted with more arms).
