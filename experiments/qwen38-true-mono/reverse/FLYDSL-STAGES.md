# FlyDSL mono stages, reverse-engineered (PLAN decision 16)

What FlyDSL's compiled code does in every stage of the Qwen3.8 mono layer kernels, where it leaves time, and which knobs
a rocKE port should search over. Every claim below comes from the final ISA / LLVM IR of the builds the served stacks
use, split per stage by the toolkit in `hackathon-prep/qwen38/truemono/reverse/`. Counts are static instruction counts
in a stage's code (not time); shares are the mono K=3 round shares of `RP/GAP_TABLE.md` section 3. Ratios only in this
tree.

Data: job `rv-dump4` (68240929, q38 container, GPU-less, FlyDSL 0.3.4.1, ROCm 10 COMGR), 81 debug + 54 plain layer
builds = 3 sets x 3 kinds x 9 variants:

- `mono`: plain FlyDSL mono on ATOM-truemono (no levers). These are the FlyDSL originals of every stage.
- `stkfly`: the current best stack's schedule (`picksI` + `qsa:qdense=3`, `sched.txt` from `gen_tuned.py`) on FlyDSL
  stage bodies.
- `stk`: the stack as served, with `stkfly` plus the rocKE fused MoE / B3 stage functions linked (`atom-truemono-cs7`,
  `rocke-qk-cs7`, `h-qk-cs7`).
- Variants: r1, r2, r3, r4, r2s, r3s, r4s, r8g4, r16g4.

The job also compiled 21 rocKE / Triton comparison kernels. Generated reports are in
`hackathon-prep/qwen38/truemono/reverse/results/rv-dump4-report/`:

- `STAGES.md`: per stage, every build's metrics plus the findings with evidence.
- `PAIRS.md` / `pairs.json`: mono vs stkfly, stkfly vs stk, head, and the .ll level.
- `catalogue.json` and `identity.txt`.

Raw splits are in `hackathon-prep/logs/q38/rv-dump4/` (`split.tgz`, `dumps_ll.tgz`, `fns_stk.tgz`).

## 0. The toolkit (how a wave-2 port uses it)

| step | tool | what it does |
|---|---|---|
| dump | `rv_dump.py --kind K --variant V --debug 1` (in `rv_job.sh`) | builds one mono layer compile-only with FlyDSL's IR dump on. It keeps `NN_llvm_ir.ll` (`.pre.ll`: the last MLIR->LLVM stage; extern rocKE stage functions are still `declare`d), the code object (`.hsaco`, with the externs linked), and the resources. `--debug 1` adds DWARF line tables and per-stage markers (below). |
| identity | `rv_identity.py DUMPS [ROCKE]` | checks that a debug (marked) build has the same `.text` as the plain build, and the same IR once debug metadata is stripped |
| split | `rv_isa.py split X.hsaco --out X.split.json` | disassembles the code object, gives every instruction its stage from the line table (`llvm-symbolizer --inlines`), and computes per-stage metrics |
| split (rocKE / FlyTape) | `--pyfunc REGEX [--pyfunc-skip F,G] [--remap OLD=NEW]` / `--registry markers.json` / `--map regex=stage` / `--labels-from Y.split.json` | rocKE builds (`ROCKE_DEBUG_LOC=1`): the label is the outermost enclosing Python function. FlyTape (`rv_ft.py`): the label is the phase marker. `--labels-from` transfers labels to a build without debug info by alignment. |
| .ll | `rv_ll.py profile X.pre.ll` | per-stage IR profile (opcodes, intrinsics, memory ops by address space and type, inline asm). `canon()` gives a canonical text (no value numbering or metadata) for structural diffs. |
| diff | `rv_diff.py A.split.json B.split.json [--group moe=m3+m4+m5]` | per-stage side-by-side metrics, a `matches` / `differs` verdict with reasons, and per-stage `findings` (inefficiency, evidence line, knob candidates) |
| catalogue | `rv_catalogue.py --split D --dumps D --fns FN --out R` | `STAGES.md`, `PAIRS.md`, and the .ll-level FlyDSL stage vs linked rocKE function |
| LB MoE | `rv_lb.py --rocke LB.split.json --ft FT.split.json` | phase-by-phase rocKE large-batch MoE vs the FlyTape tiled tape (`MOE-LB.md`) |
| rocKE / Triton kernels | `rv_rocke.py --recipe R --debug 0/1` | the head (`mx_decode_gemv`), the LB MoE (`moe_lb`, `moe_lb2`), and the mono's Triton head (`triton.compile` for gfx950 without a GPU) |
| jobs | `rv_job.sh` (GPU-less container: `truemono/P6/submit_cpu.sh --gpus 0`), `rv_ft_job.sh` (1 GPU), `rv_lb2_job.sh` (current `large_batch/*.py` overlaid on a rocKE stage); `sync_rv.sh <sfx> [overlay]` stages the tools as `rv-h-<sfx>` | the whole layer dump takes about 4 min on 64 CPUs |
| tests | `test_rv.py` (39 CPU tests), `mut_rv.py` (24 mutants, all killed) | |

A port's loop:

1. Build the rocKE stage into the layer kernel with `ROCKE_DEBUG_LOC=1`.
2. Run `rv_isa.py split --pyfunc 'stage_<x>\.py'` (or `--map`).
3. Run `rv_diff.py` against the FlyDSL stage from this catalogue (same kind and variant).
4. Parity is reached when the verdict is `matches`, or `differs` only where intended.
5. Each finding's knob goes into the search space.

**Stage markers (how the split works).** FlyDSL's location capture records the full Python call chain, but MLIR's LLVM
translation keeps only the innermost and the outermost frame. So `rv_dump.py`, in its own process with ATOM untouched,
runs each stage (`kernel._run`, `sched.stage_enter` / `stage_leave`) inside a trampoline that is the FlyDSL tracing
boundary (`tracing_context(func=tramp)`). The stage is encoded in the trampoline's line number
(`rv_common.marker_line`), which survives as the outermost frame of every location.

**Validity:**

- **IR:** the marked builds' IR equals the plain builds' after stripping debug metadata in 54 of 54 builds.
- **`.text`:** identical in 51 to 52 of 54. The ones that differ are the known non-reproducible `delta_ple` r2s / r3s / r4s
  builds (alola rule 137; same sizes); their splits describe one of the build's own variants.
- **Attribution:** 0.77-0.92 of a build's instructions get their stage directly from a marker. The rest, mostly loop
  control and the per-stage prologue, take the stage of the code around them.
- **rocKE and FlyTape:** rocKE `ROCKE_DEBUG_LOC=1` builds have the plain `.text` in 14 of 14, and FlyTape tapes with
  markers in 2 of 2.

## 1. What every FlyDSL stage has in common (the cross-cutting inefficiencies)

These hold in every kind and variant unless noted. Evidence addresses belong to the build named next to them (or in
`STAGES.md`); addresses differ between builds.

1. **MFMA K loops with no LDS read-ahead and one serial accumulator.** In A1, M1, B1 and Q1, every one of the 20
   `v_mfma_scale_f32_16x16x128_f8f6f4` sits right after `s_waitcnt lgkmcnt(0)`, and 18-19 of 20 accumulate into the
   previous MFMA's destination. The pattern is:

   ```
   ds_read_b128 v[14:17]; ds_read_b128 v[18:21]; ds_read2_b32 v[10:11]   (next A fragment + scales)
   s_waitcnt lgkmcnt(0)
   v_mfma_scale ... v[6:9], ..., v[6:9], v10, v3
   ```

   (`mono_delta_r1` `0x4804`-`0x4810`, a1). C1 is the same: 7 of 12 MFMAs after the wait, 11 of 12 chained.
   - **Knobs:** LDS operand read-ahead by one step (double-buffered fragments; same MFMA order, so bitwise) and
     `sched_group_barrier` interleave.
   - Several accumulators would change the summation order, so they belong to the tolerance class only.
2. **Scale VGPRs rewritten right before the scaled MFMA.** This affects 9 of 20 MFMAs in A1 / M1 / B1 / Q1, 11 of 12 in
   C1, 5-16 of 20-40 in M4, 5-20 of 15-60 in M5, and 10 of 40 in P1. The scale operands come from a fresh
   `ds_read2_b32` or a VALU shift each step (`mono_delta_ple_r1` `0xe8d0: ds_read2_b32 v[10:11]` in M1, `0xb98c: v_ashrrev_i32_e32 v6, s8, v26` in C1). This is the pattern P-L3 measured at about 8x slower MFMAs on gfx950.
   - **Knob:** `sop` (op_sel scales: 4 K steps' scale bytes in one dword, picked with op_sel), already a rocKE lever in
     `large_batch/gemm.py` and `moe_lb.py`.
3. **Hard-spin mailbox polls.** Every hand-off poll is a 5-60 instruction loop over `buffer_load_dword{x2,x4} sc1`,
   `s_waitcnt vmcnt(0)` and a compare, with no `s_sleep`. Each stage has 1-10 of them (Q4 has 10).
   - **Knobs:** a pause (`s_sleep 1`), one polling wave that broadcasts through LDS (P9: unthrottled sc1 polls starved
     producers), deadline polls (`rocke.helpers.mailbox.poll`).
4. **No `nt` on any weight stream.** Every FlyDSL weight load is a plain `buffer_load_dwordx4` (A1 24, M1 27, B1 20,
   Q1 20, M4 20, Q7 17, P1 23, P2 19 at r1). Only mailbox data carries `sc1`. The rocKE fused MoE that beats FlyDSL's
   M3-M5 streams its expert weights with `nt` (section 3).
   - **Knob:** `nt` / cache policy per weight stream.
5. **DPP reduction chains padded with `s_nop`.**
   - M4 and M5 spend 100 of their 193 / 196 `s_nop` wait states between consecutive `v_max_u32_dpp` (the dependent DPP
     chain of a wave max).
   - Q4 spends 338 of 547 on `v_max_u32_dpp` and `v_add_u32_dpp` chains.
   - A1 and B3 have `v_mov_b32` -> `v_mov_b32_dpp` hazards.
   - **Knobs:** interleave independent reductions (rows, two keys) to fill the wait states; `xshfl` forms (rocKE's B3
     uses `v_add_f32_dpp` fused reductions and `ds_swizzle`: 38 `v_mov_b32_dpp` against FlyDSL's 50); a `readlane` tree.
6. **Dense full drains.** The ratio of full `vmcnt(0)` waits to vector loads is high in A2 / M2 (7-8 for 13), B3 (9 /
   23), Q2 (7 / 13), Q3 (11 / 30), Q5 (10 / 12), Q6 (4 / 10) and Q7 (12 / 40).
   - Most sit inside polls (one per retry) or right after a stage's first loads, so the rest of the stage's loads land
     serially.
   - **Knobs:** issue all of a stage's independent loads before the first poll / wait, and use counted `vmcnt(N)`
     waits.
7. **Registers, not spills, are the limit.**
   - No build spills, apart from one tiny case: some `qsa` builds (mono r2 / r2s, stk r2 / r3 / r4) declare 68 B of
     private segment but have no scratch instruction (a reserved frame object).
   - VGPRs run 176 (delta r1) to 254 (qsa r16g4) for the whole layer, one wave per SIMD.
   - A stage's own register reach is far lower: the highest VGPR index touched is 51-97 in M3 / C1 / M4 / M5 at delta r1.
     The whole layer pays for its widest stage (A1 / M1 175, Q7 245).
   - **Knob:** per-stage register budget in a persistent kernel (fresh ids per stage, lazy resources, P9 / WP-L2c).
8. **Stage code is contiguous.**
   - Each stage is a single run of code (`runs=1`) in every narrow build.
   - LLVM does not interleave stages, except for a 113-166 instruction piece of A1 that the `stkfly` schedule pulls
     into B1 / Q3.
   - So a stage A/B in the persistent kernel is a clean code-region A/B.
9. **What the schedule levers change in the code** (`mono` vs `stkfly`, .ll opcode-sequence similarity 0.89-0.996 per
   stage):
   - holds and prio add `s_sleep` / `s_setprio`;
   - `numa=1` doubles the mailbox stores (`vmem_store` 2 -> 4 per stage);
   - `dyn.q6` / `dyn.q7` and `qdense` grow Q3-Q7 (Q4 VALU 3610 -> 4037).
   - The stage bodies are otherwise the same code.

## 2. Per stage

Columns, all at r1 / r4s, from `mono_delta_r1` / `mono_delta_r4s` (Q stages from `mono_qsa_*`, P stages from
`mono_delta_ple_r1`):

- instructions;
- vector loads / stores;
- LDS ops;
- MFMAs;
- VALU;
- full `vmcnt(0)` / all waits;
- CTA barriers;
- polls;
- `s_nop` wait states.

The stage laid out last in a code object also carries about 260 never-executed alignment `s_nop` after `s_endpgm`. That
is M5 in most builds (`mono_delta_r1`: 261 of its 1389) and A1 in delta r4s (267 of 2422). They count as instructions
but not as `s_nop` wait states (`pad_nops` in the split).

| stage (share) | what the code does | counts r1 (r4s) | inefficiencies (evidence) | knob candidates for the port |
|---|---|---|---|---|
| A1 (0.085) | HC norm + down + inject. 24 weight `dwordx4` per lane (one shot), 20 scaled MFMAs over LDS operands, then a VALU tail: norm of 4 streams, pack, chain, in-CTA reduce. | 2134 (2422); ld 25; LDS 164; MFMA 20; VALU 1637; vm0 3/108; bar 6; nop 68 | VALU tail is 0.77 of the stage's instructions (DESIGN 1.3 "tail-bound"); serial LDS->MFMA chain (20/20 after lgkmcnt(0), 18 chained); scale rewrite 9/20; no nt | read-ahead; `sop`; P3 `spread` (already searched); split tail over more waves (bitwise-ordered reduce); `xpre` entry preload |
| A2 (0.029) | HC up: 3 MFMAs; 13 (37) loads; 2-4 polls on the HC mailboxes | 529 (1144); vm0 7/30 (7/49); polls 2; nop 44 | 7 drains for 13 loads; hard spins (`loop at 0x52f0`) | issue loads before the poll; poll pause; `hold.a2` (already a lever) |
| B1 (0.048) | QKVZ projection: 20 scaled MFMAs, 20 weight `dwordx4` | 399 (523); LDS 55; vm0 5/36; polls 2 | MFMA serial chain (20/20 lgkmcnt(0), 19 chained); scale rewrite 9/20; no nt; drain before barrier in r4s / wide builds | read-ahead; `sop`; nt; `hold.b1` / `prio.b1` |
| B3 (0.076) | gated delta rule on VALU (no MFMA): 214 `v_fmac`, DPP reductions, state snapshot stores | 1372 (1999); ld 23 (35); st 11 (35); VALU 924; vm0 9/103; polls 4 | 9 drains for 23 loads; 50 `v_mov_b32_dpp` (fused DPP adds in rocKE); 10 narrow `dword` loads | already ported (rocKE B3, section 3) |
| C1 (0.054) | out_proj: 12 scaled MFMAs, 25 loads | 406 (504); vm0 5/31; polls 3 | scale rewrite 11/12 (worst); 11/12 chained | `sop`; read-ahead; poll pause |
| M1 (0.073) | as A1 on the MLP HC: issues all weights before polling C1 | 2195 (2257); ld 35; LDS 164; MFMA 20; VALU 1649 | as A1; polls 2 (44-48 instruction loops reloading 3 `dwordx4 sc1`) | as A1 + poll shape |
| M2 (0.014) | as A2 | 486 (892); vm0 8/27 | 8 drains / 13 loads, 4 hard spins | as A2 |
| M3 (0.045) | router: 9 loads, VALU dots (no MFMA), 82 LDS ops | 916 (1023); VALU 666; polls 1 | hard spin | already ported (fused MoE) |
| M4 (0.171) | routed + shared gate_up: 20 (40) MFMAs, 48 (88) loads; key max over DPP | 986 (1925); nop 193 (347); vm0 8/53 | DPP max chains: 100 nop cycles; scale rewrite 5-16; no nt on expert weights | already ported (fused MoE) |
| M5 (0.096) | down + combine + HC write: 15 (60) MFMAs | 1389 (2196); nop 196 (313); vm0 9/50 | DPP max chains; drain before barrier (r4s); no nt (wide) | already ported (fused MoE) |
| Q1 (0.015) | fused QSA projection: 20 scaled MFMAs | 333 (411); vm0 6/35 | as B1 | as B1 |
| Q2 (0.004) | q / k norm + RoPE (no LDS, no MFMA) | 344 (352); vm0 7/16 | 7 full drains for 13 loads | already ported (`stage_q2`, bitwise) |
| Q3 (0.006) | index scores, f64 dot on VALU | 1242 (1328); VALU 910; ld 30 (20 narrow `dword`) | narrow loads; 11 drains | `q3=mfma` (N10, tolerance); wide loads |
| Q4 (0.009) | top-512 selection: 6051 instructions, 33 CTA barriers, 369 LDS ops, 10 polls | 6051 (6051); nop 547 | DPP max/add chains (338 nop cycles); 33 barriers | chunk spread (G5); interleaved reductions; fewer barrier phases |
| Q5 (0.017) | scores over the selected list; unrolls with rows | 1191 (4046); vm0 10/77 (25/238) | 10 drains for 12 loads at r1 | load batching; `qdense` (P13, tolerance) |
| Q6 (0.007) | softmax per head | 399 (442); vm0 4/21; nop 55 | drains; DPP chains | poll pause; interleave |
| Q7 (0.031) | attention output on VALU (no MFMA): 1653 VALU, 40 loads | 2098 (2569); vm0 12/55; polls 5 | 12 drains / 40 loads; VGPR reach 245 (the qsa layer's widest stage); no nt | MFMA form (tolerance); register budget per stage |
| P1 (0.005 with P2) | PLE projection: 40 scaled MFMAs | 434; nop 52 (48 after MFMA) | scale rewrite 10/40; 23/40 after lgkmcnt(0); no nt | `sop`; read-ahead |
| P2 | PLE gate / conv / residual on VALU | 1813; VALU 1494; vm0 5/31 | no nt (19 plain `dwordx4`); hard spin | nt; poll pause |

The wide builds (r8g4 / r16g4) repeat the narrow bodies per row pass:

- M4 r16g4 has 160 MFMAs, of which 85 come after `lgkmcnt(0)` and 74 have a rewritten scale, plus 1264 nop cycles.
- M5 r16g4 has 210 MFMAs (79 rewritten scales).
- Q5 r16g4 is 13171 instructions with 91 full drains.

So the per-stage knobs scale with rows there.

## 3. What makes the existing rocKE stages win (FlyDSL original vs the rocKE stage, same layer, same schedule)

`stkfly` vs `stk` (FlyDSL stage bodies vs the linked rocKE functions, identical schedule), and the head:

- **Fused MoE (rocKE `moe_fused` for M3-M5; 0.78-0.89 of the mono MoE).** Delta r1 has 4545 rocKE instructions
  against FlyDSL's 3359; r4s has 5856 against 5375.
  - **Expert weights streamed with `nt`.** rocKE has 45 (r1) / 65 (r4s) `buffer_load_dwordx4 nt sc1` and 22 / 35
    `dword nt sc1` scale loads. FlyDSL has 35 / 75 plain `dwordx4` and 16 / 38 plain `dword`.
  - **One stage instead of three.** CTA barriers drop from 12 to 8 at r1 and from 12 to 9 at r4s. The three FlyDSL code
    runs become one run, so there is no M3->M4->M5 boundary inside a CTA.
  - **More independent MFMA work per weight load.** rocKE has 60 MFMAs at r1 (90 at r4s): 18 of 60 chained on the
    previous accumulator and 26 after `lgkmcnt(0)`. FlyDSL's M4 + M5 have 35 MFMAs, of which 27 are chained.
  - **The down outputs leave through a private mailbox.** Stores are `sc1 dwordx2`: 20 at r1, 41 at r4s, against 2 in
    FlyDSL.
  - **Kept 1:1 from FlyDSL:** the router's VALU dot structure and the poll shape (hard spins, 11 vs 8 polls).
  - **At the .ll level:** the rocKE function uses `llvm.fmuladd` (232 vs 0) and `<4 x i32>` buffer loads (57 vs
    FlyDSL's i128 61) (`PAIRS.md` ".ll level").
- **B3 (rocKE `stage_b3`, tuned knobs, 0.963 of the best mono layer).**
  - **Fewer instructions:** 1339 vs 1410 at r1, 1894 vs 2074 at r4s.
  - **Fewer loads:** 12 vs 23 at r1, because 10 narrow `buffer_load_dword` become 2.
  - **Fewer full drains:** 7 vs 9 `vmcnt(0)` at r1 and 6 vs 9 at r4s; `lgkmcnt(0)` 26 vs 39 at r4s.
  - **Fewer SALU:** 116 vs 159 at r1.
  - **Fused DPP reductions:** 14 to 21 `v_add_f32_dpp` and `ds_swizzle` replace part of the `v_mov_b32_dpp` + add pairs
    (38 vs 50).
  - **State snapshot stored `nt`:** 16 / 32 `buffer_store_dword nt` against plain stores.
  - **At the .ll level:** FlyDSL's 64 `s.buffer.load.i32` of the per-step table (r4s) become vector loads; `fmuladd`
    246 vs 0.
- **Head (rocKE `mx_decode_gemv` vs the mono's Triton `_mxfp4_head`, logits r4).**
  - **Triton loads weights a byte at a time:** 68 `global_load_ubyte nt` and 32 `global_load_ushort` per program;
    rocKE uses 40 `buffer_load_dwordx4`. This is the PMC fetch excess the head ledger saw (1.23x vs 1.00x of the head's
    bytes).
  - **Triton spills SGPRs:** 41 (logits r4) / 46 (argmax) against 0. Its VGPRs are 159 vs 112.
  - **Triton does far less MFMA work per program:** 16 MFMAs (MFMA gap mean 25, max 253) against rocKE's persistent
    panel loop with 80 MFMAs (gap mean 10).
  - **rocKE stages `x` in LDS** (20-83 KB).
  - rocKE still has hard-spin polls and no `nt`. Those are open knobs; the ledger found `cpol=sc0` mixed.

The common thread is that rocKE wins by:

1. streaming weights `nt`;
2. removing stage boundaries and barriers;
3. using wide loads and fused DPP reductions instead of narrow loads and mov + op pairs;
4. reducing scalar and drain overhead.

None of these wins used a parametric search over FlyDSL's remaining patterns from section 1: LDS read-ahead, `sop`,
poll pause, and drain placement. Those are the open headroom for the wave-2 ports.

## 4. Port checklist per wave-2 stage (A1 / A2 / M1 / M2, B1, C1, Q1, Q3-Q7, P1 / P2)

1. **Parity.** Diff the rocKE stage against this catalogue's FlyDSL stage at the same kind and variant
   (`rv_diff.py`). The port is at parity when instruction classes are within 10 %, the memory forms and cache bits
   are the same, and the waitcnt mix, MFMA count / chain, barriers and polls match.
2. **Knob space.** Expose every hard-coded choice the findings name as a default-off knob:
   - `read-ahead` (LDS operand prefetch depth 0/1/2);
   - `sop` (op_sel scales);
   - `nt` per weight stream;
   - poll `pause` / `pollwave`;
   - `drain` placement (loads before the poll);
   - reduction form (`xshfl`: dpp mov+op / fused dpp / swizzle / readlane tree; interleave);
   - per-stage register budget;
   - barrier phases (Q4).
3. **Searchable order.** Make the knob order searchable (PLAN 15). Let the tuner climb with the losers retained
   (PLAN 13).
4. **Re-run the split after every accepted change.** A finding that disappears is the proof the knob acted (the
   evidence address in `STAGES.md` is gone).
