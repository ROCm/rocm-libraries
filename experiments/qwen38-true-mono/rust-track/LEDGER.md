# Rust track (RT): the direct code-object compiler for the Qwen3.8 true monokernel (lever ledger)

User decision 6 (PLAN.md, 2026-10-30): the Rust direct compiler is a committed track; bitwise parity with COMGR is
not required (accuracy-gated: per-kernel tolerance vs an fp32 reference with must-fail mutants, determinism); goal:
beat COMGR on every true-mono stage kernel and become the mono's default code-object backend. Decisions 9-11: the
compiler core is ours to change (isel, scheduler, regalloc, waitcnt / hazards, encoders, layout, per-XCD paths).

Ratios only (direct or variant time / COMGR time; < 1 is faster). Absolute numbers:
`hackathon-prep/private-perf/qwen38-true-mono/rust-track/`. No git state was changed anywhere. Tree:
`C:/Users/vanantha/Documents/dev/copy-rocm-lib-rust` (plain copy; R1/R2 overlay + this track's changes).

## Milestone 1: coverage (2026-10-30)

### What was added

Compiler (crate `platform/rust/rocke-codeobj`; ISA rows in `isa/gfx950.isa`, tables regenerated):

| op | lowering | notes |
|---|---|---|
| `math.exp` (f32) | LLVM's `lowerFEXP` with fast FMA: `ph = x log2e`, FMA residual + low part, `v_rndne`, `v_exp_f32`, `v_ldexp`, underflow 0 / overflow +inf selects | B3 decay, MoE router |
| `arith.cvt_scalef32_pk_{fp8,bf8}_f32` | `v_cvt_scalef32_pk_*` (VOP3 0x235 / 0x236), tied old, op_sel[3] = high half | MXFP8 packs (B3, MoE) |
| `tile.mfma_scale_mx` | `v_mfma_scale_f32_16x16x128_f8f6f4` (new row) and the 32x32x64 one; formats in cbsz/blgp, scale byte selects in the `v_mfma_ld_scale_b32` op_sel / op_sel_hi | fused MoE (fp8 x fp4) |
| `ptr<T,constant>` from `tile.int_to_ptr` | loads through it are `invariant`: SMEM (`s_load`) when uniform even after global writes | Q2 / B3 tables |
| kernel symbols with `-` | accepted (ELF writer and parser) | fused MoE r4 (`combinebase-1`) |
| `arith.maximum3_f32 {abs}` | `v_maximum3_f32` + VOP3 abs (new `M_ABS*` modifier set) | MXFP8 amax |
| `tile.dpp_op` / `tile.mov_dpp` | `v_mov_b32_dpp` / `v_max_u32_dpp`; masked or bound_ctrl-clear forms tied to `old` (new rows) | B3 xor-4 exchange, MoE wave umax scan, Q2 row shifts |
| `arith.cvt_f32_to_f64` / `cvt_f64_to_f32`, `math.sqrt` (f64) | `v_cvt_f64_f32` / `v_cvt_f32_f64`; LLVM's `lowerFSQRTF64` (rsq + Goldschmidt, 2^256 scaling, zero / +inf class) | Q2 norm |
| `tile.global_store_cp` / `global_load_cp` | GLOBAL with sc0 / sc1 / nt (sc0 added to the load/store rows), saddr or per-lane address | asm puts / loads |
| `tile.bcnt`, `tile.s_ff1`, `tile.memtime`, `tile.s_nop`, `tile.compiler_fence`, `tile.ds_{or,write,read}_b32` | one instruction each (raw-address DS rows are fences) | MoE slots, probes, CCL |

Python (`platform/python/rocke/runtime/direct_lift.py`, hooked into `helpers/compile.py` for
`code_object_backend="direct"` only, on a copy; `ROCKE_DIRECT_LIFT=0` disables): the asm idioms of
`helpers.device_ops`, `helpers.mailbox` and the stage modules are lifted to the ops above (pointer moves, clocks,
getreg, sleeps / nops, cache-bit stores and loads, scoped atomics, wbl2 / inv fences, B3's bank-masked row shifts,
the DPP umax scan, maximum3, MXFP8 asm converts, B3's `s_load` table, LDS raw ops, and `mailbox.asm_poll_template`
polls, rebuilt and compared exactly before lifting, as `tile.tagged_poll`). An unrecognized template stays and the
kernel falls back to COMGR as before.

### Gates

| gate | result |
|---|---|
| crate build + examples, 0 warnings; `cargo test --lib` / all targets / 1 thread (CPUONLY job rt-cpu-v1) | 447 / 447 each |
| `isa_tables.py check` (both ISAs), ISA tool tests | ok, 25 passed |
| census vs the R2 crate: 2,029 gfx950 + 18 gfx1151 IRs | 2,017 / 2,017 and 18 / 18 objects byte-identical, 12 same errors, 0 unexpected |
| new unit tests (`target::gfx9::q38`, words checked against `llvm-mc -mcpu=gfx950`) | 18 / 18 (local) |

### Coverage census (`rt_census.py`, local compile-only, the lifted IR; `rt_builds.py` labels)

| kernel family | builds | direct compile |
|---|---|---|
| `mx_decode_gemv` head, logits and fused argmax, rows 4 and 16, shipped and `irops` | 8 | all |
| `stage_b3` B3, r1 / r2s / r3s / r4s / r4 / r4g2, mono and tuned knobs, shipped and IR-form mailboxes | 24 | all |
| `moe_fused` fused MoE suffix, rows 1 and 4, shipped and IR-form | 4 | all |
| `stage_q2` Q2 (stage-function kernel), r1 / r2 / r4 / r4s, asm and ir mailboxes | 8 | all |
| toy chain `best_spec(4,8,l2)`, `best_spec(8,0)`, shipped (asm) and `irops` | 4 | all |
| probe P2 (head in a persistent launch), default and `pimpl=ir` | 2 | all |
| probe P4 a2a, P5 host writer, CCL link-bandwidth probe | 3 | all |
| probes P4 counter / ticket / xl / boundary, P5 kv / scalar / host, CCL all-reduce / all-gather / argmax | 10 | refused: the bounded asm poll loops (P4 `p4c`/`p4p`, P5 `p5h`, CCL `cclp`/`cclf`), CCL's `s_add_u32`/`s_addc_u32` pair and an i64 `lshr` (open) |

## Milestone 2: numerics and paired timing (in progress)

Harnesses: `rt_b3.py` (ATOM B3 rig: bitwise vs the mono layer's B3, determinism, then shuffled interleaved graph
timing with byte-copy and fresh-recompile controls), `rt_fmoe.py` (fused_ab cell: FlyDSL prefix + suffix, MoE part
= pair - pre per round), jobs `jobs/rt_gpu.sh` (container), `jobs/rt_cpu.sh` / `rt_cpu_submit.sh` (crate gates).
Harness additions (2026-10-31): `rt_head.py` (lm_head GEMV, logits + fused argmax checks), placements (every arm
loaded 3-4 times as separate modules / graphs, all replayed per round, mean per round: byte-identical arms differed
by up to 1.5% with one graph each), the pooled-COMGR view (`vs_pooled`: arm / mean of C, C2, R per round),
byte-identical lever arms measured once (`same_as`), `RT_DROP_FAILING=1` (screening: an arm failing numerics leaves
the timing instead of stopping it), `rt_fmoe.py --stamps` (phase-stamped builds, per-phase ratios vs C),
`rt_design.py` (screening designs and the surrogate fit), `rt_mutants.py` lever mutants (the lever arm must fail).

### Plain direct object (no levers) vs COMGR

| kernel | n01-09 | n07-17 | verdict |
|---|---|---|---|
| B3 r1 tuned | 0.991 (rt-g5), 0.987 (rt-s1, pooled) | 0.986 (rt-g3) | WIN on both nodes |
| B3 r4s tuned | 0.990 (rt-g5), 0.999 (rt-s1) | 0.996 (rt-g3) | neutral (win on one run only) |
| fused MoE r1 | 1.017 (rt-g5), 1.022 (rt-s1) | 1.022 (rt-g3) | LOSS |
| fused MoE r4 | 1.047 (rt-s1) | 1.044 (rt-g3) | LOSS |
| head fused r4 (m 1/2/4) | 1.010 (rt-s1) | 1.004-1.006 (rt-g3) | LOSS / neutral |
| head fused r16 | - | 1.007-1.013 (rt-g3) | LOSS / neutral |

Numerics: every direct object bitwise equal to COMGR's (B3 vs the mono layer; fused MoE vs the FlyDSL layer and C;
head logits and argmax), deterministic.

### Lever screening rt-s1 (decision 13: all 12 levers, losers included; n01-09, 16 rounds, vs pooled COMGR)

Design: 2^(12-7) resolution IV (32 runs; main effects clear of two-lever interactions, interactions aliased in
chains of 4-5), the same 32 arms for every kernel; byte-identical arms measured once. Surrogate: ridge over main
effects + all pairs (`rt_design.py fit`). Every arm passed numerics (bitwise).

| kernel | D | best measured arm (levers) | strongest main effects (lever on - off) |
|---|---|---|---|
| B3 r1 tuned | 0.987 | 0.976 (dppmirror, dppcombine, dppcombinef, ldspairs, nopause, xorcmp) | dppcombine -0.0024, ldspairs -0.0021, dppcombinef -0.0017, nopause -0.0017, dppmirror -0.0016 |
| B3 r4s tuned | 0.999 | 0.989 (dppmirror, ldsquads, ldspairs, vmemfirst, wgfence) | dppmirror -0.0056, ldspairs -0.0034 |
| fused MoE r1 | 1.022 | 0.998 (nofence, dppmirror, dppcombinef, ldsquads, xorcmp, wgfence) | wgfence -0.0064, ldsquads / xorcmp -0.0047, dppmirror -0.0043, nofence -0.0035, vmemfirst +0.0035 |
| fused MoE r4 | 1.047 | 1.014 (all 12) | nofence -0.0164, xorcmp -0.0047, ldsquads -0.0040, wgfence -0.0035, minmax -0.0022 |
| head fused r4 | 1.010 | 1.005-1.006 (S03 / S08) | nofence +0.0174 (loses), vmemfirst -0.0026 to -0.0034, storewait -0.0010 |

Findings: `nofence`, an R2 solo loser, is the largest single win on the fused MoE r4 (and a 1.7% loss on the head):
lever sets must be per kernel family. Interactions screened so far are small (|pair| <= 0.001); pairs aliased in
chains are separated by the next design. No lever is pruned: `storewait`, `minmax` and `vmemfirst` change no
byte of some kernels (measured once, `same_as`), which is the only exclusion (per kernel, byte-identical objects).

### Compiler levers added in this milestone (all default-off, `ROCKE_DIRECT_LEVERS`)

| lever | what | gate |
|---|---|---|
| `dppcombinef` | f32 add / mul / max and u32 max fold their DPP move (with `dppcombine`) | GPU mutants (add-as-mul, max-as-add, mul-as-add) |
| `ldspairs` | dword LDS loads / stores of one base within 1 KiB -> `ds_read2_b32` / `ds_write2_b32` | 2 unit tests; GPU mutant offsets swapped: killed |
| `minmax` | compare + selects of the same two values -> `v_max` / `v_min` | unit test; GPU mutant max/min swapped: killed |
| `xorcmp` | slt(x ^ S, y ^ S) -> ult(x, y) (S = sign bit) | GPU mutant order reversed: killed |
| `wgfence` | workgroup fences as LLVM non-tgsplit: release waits stores + LDS DMA + LDS only (`WaitRelease`), acquire nothing | unit test + CPU mutant: killed; GPU mutant survives (a race the numerics do not hit) |
| `litreuse` | straight-line blocks reuse literal registers and SGPR copies as loop blocks do | unit test + CPU mutant: killed |
| `litshare` | the existing one-register-per-literal pass (debug flag) as a lever | existing unit tests |
| `bitop3` | trees of and / or / xor / not over <= 3 values -> `v_bitop3_b32` (gfx950; encoding checked vs llvm-mc) | 2 unit tests + CPU mutant: killed; GPU mutant pending |
| `selxor` | select(c, x^A, x^B) -> x ^ select(c, A, B); (x ^ s) & -2^j -> x ^ (s & -2^j) when x has j low zeros (folds into the selected constants) | GPU mutants pending |
| `subadd` | k1 - (x + k2) -> (k1 - k2) - x | GPU mutant pending |

With `selxor` + `bitop3` + `subadd` + `litreuse` the router's monotone key takes cmp, cndmask, sub, bitop3 per
element as COMGR does (was 8 instructions).

Mutants rt-m6 (n07-17): killed maximum3 abs, ldsquads offset, minmax swap, xorcmp order, ldspairs offsets, both
lift mutants (and in rt-g3 the cvt / mx-scale / rowshift ones); survivors retargeted to kernels whose bytes the
lever changes (dppmirror half-mirror and dppcombine operand order: the toy chain has no such code).

### Screening rounds 2-3 and the first two-node wins (2026-10-31)

rt-s2 (n01-09): 2^(17-11) resolution IV (64 runs) over 17 levers + phase stamps of the fused MoE. rt-s3 (n01-09):
2^(20-14) resolution IV over 20 levers (+ `loadorder`, `scalesep`, `dsrebase`) plus the s2 finalists (F: measured
best and the surrogate's top two; G: F + scalesep + dsrebase + loadorder). rt-c1 (n07-17): the same F / G / D arms,
32 rounds (second-node confirmation), and the 14 GPU mutants (all killed). All arms bitwise; none dropped.

| kernel | D (n01-09 / n07-17) | best lever set (n01-09 / n07-17) | verdict |
|---|---|---|---|
| B3 r1 tuned | 0.992 / 0.983 | G0 0.970 / 0.972; G1 0.977 / 0.967 | WIN on both nodes |
| B3 r4s tuned | 0.993 / 0.998 | G1 0.979 / 0.979 | WIN on both nodes |
| fused MoE r1 | 1.022 / 1.016 | G1 0.975 / 0.976 | WIN on both nodes |
| fused MoE r4 | 1.051 / 1.046 | G1 1.002 / 1.009 | neutral / LOSS (was 1.05) |
| head fused r4 (m 1/2/4) | 1.006-1.007 / 1.008-1.009 | F 1.005 / 1.005 | neutral / LOSS |
| head fused r16 (m 1/8/16) | 1.006-1.010 / 1.009-1.012 | F 1.006-1.007 / 1.006-1.007 | neutral |

Lever sets (ROCKE_DIRECT_LEVERS):
- B3 r1 G0: storewait, nofence, dppmirror, dppcombine, dppcombinef, ldspairs, minmax, xorcmp, selxor, subadd,
  scalesep, dsrebase, loadorder.
- B3 r4s G1: dppcombine, dppcombinef, dppmirror, ldspairs, ldsquads, litreuse, litshare, minmax, nopause, selxor,
  storewait, subadd, vmemfirst, wgfence, xorcmp, scalesep, dsrebase, loadorder.
- fused MoE r1 G1: dppmirror, ldsquads, litreuse, litshare, minmax, nofence, selxor, subadd, wgfence, xorcmp,
  scalesep, dsrebase, loadorder.

Phase stamps (rt-s2 `fst`, medians over CTAs and 64 launches, ratio vs C): the plain direct object's fused MoE r4
spends 1.31x C's time in the gate/up MFMA phase and 1.2-1.4x in down / combine; r1's router runs 0.78x. The causes
found in the ISA and fixed as levers: every block-scaled MFMA copied its two scales into a fresh VGPR pair (two moves
and a VALU-to-MFMA wait per MFMA: `scalesep` reads them in place; the encoder takes two scale operands), and every LDS
read past the 16-bit offset field added its whole offset (`dsrebase`: one 64 KiB-aligned base per block and address,
the rest in the field, as COMGR's rebased pointers). Together with the s2 set they move fused MoE r1 from 1.02 to 0.975.

Levers added (default-off): `loadorder` (vmemfirst + the load read first issues first; loads read only past the
block keep program order: their in-block readers are the loop's yield copies), `scalesep`, `dsrebase`, and the
per-kernel lever attribute (`KernelDef.attrs["direct_levers"]`, joined to ROCKE_DIRECT_LEVERS for that compile only;
not serialized; `platform/tests/runtime/test_direct_targets.py`). Gates: crate v11 459+ tests, census byte-identical
vs R2 (2,017 + 18), gfx950 ISA cross-check vs llvm-mc 0 mismatches (v_bitop3 and the two-scale MFMA rows included),
CPU mutants 19 / 19 killed, GPU mutants 14 / 14 killed (rt-c1).

Lever notes from the data (decision 13: nothing pruned): `nofence` helps the fused MoE (-0.9 to -1.6%) and costs
the head +0.9 to +4.3%; `loadorder` costs the head r4 +0.2% alone and +1.2% with the F sets, the surrogate gives it
-0.7% on r16 (model noise there is high: LOO rms 0.018); `vmemfirst` helps the head (-0.1 to -1.5%) and costs fused MoE
r1 +0.2-0.4%. Lever sets are therefore per kernel family (the attribute above).

### Refinement rt-s4 / rt-c2 (both nodes, 32 rounds) and the remaining B3 / head builds rt-c3 (n07-17)

| kernel | best set | n01-09 | n07-17 | verdict |
|---|---|---|---|---|
| B3 r1 tuned | G0 (above) | 0.975 | 0.973 | WIN (both nodes, twice) |
| B3 r4s tuned | G1 (above) | 0.981 | 0.976 | WIN (both nodes, twice) |
| fused MoE r1 | G1 (above) | 0.970 | 0.976 | WIN (both nodes, twice) |
| fused MoE r4 | all 20 levers (S63) / G1 | 0.995 / 1.005 | 1.008 / 1.009 | neutral |
| head fused r4 | F0 (s2) | 1.005 | 1.004-1.005 | neutral |
| head fused r16 | F1 / S08 | 1.008 | 1.008 | neutral |
| B3 r2s / r3s / r4 / r4g2 tuned | B3 r4s G1 set | 0.962 / 0.966 / 0.986 / 0.989 (rt-s6) | 0.960 / 0.965 / 0.990 / 0.988 (D 0.986 / 0.975 / 1.017 / 1.009) | WIN (both nodes) |
| B3 r1 / r4s mono | B3 r4s G1 set | 0.992 / 0.950 (rt-s6) | 0.984 / 0.948 | WIN (both nodes) |
| head logits r4 (m 1/2/4) | D / F sets | - | 1.001-1.008 | neutral |
| head logits r16 (m 1/8/16) | D | - | 1.008-1.013 (lever sets 1.01-1.04) | neutral / LOSS |

(Head fused rows 1 and 8 are not valid `MxDecodeGemvSpec`s: rows_max is 4 or 16.)

Fused MoE r4 phase stamps with the best sets: issue + gate/up together equal C's (the loads issue faster, the MFMAs
wait longer), router / route are 5-9% faster, and the residue is in down / combine (+10% / +20%, which include
waiting on other CTAs' data).

### Coverage: every true-mono kernel compiles directly (2026-10-31)

General inline asm (`rir/isel_asm.rs`): gfx950 assembles a `tile.inline_asm` template into machine instructions on
virtual registers (SALU / VALU of plain syntax, GLOBAL loads and stores with saddr or `off` and sc0 / sc1 / nt /
offset, s_waitcnt (kept as fencing waits), s_memrealtime, s_nop / s_sleep, local labels and branches, loops one level
deeper), operands from the constraint string (=v / =&v / =s / =&s, v / s inputs, tied digits, ~{memory} as a
scheduling fence; an `s` operand of a divergent value is refused); SGPR outputs are uniform; a `sideeffect` asm is
kept when its results are unused. Any other syntax is refused (COMGR fallback; `ROCKE_DIRECT_STRICT=1` makes a refusal
an error for validation runs). Plus i64 logical right shifts by constants. P4's counter poll assembles word for word as
COMGR's (unit test against COMGR's words); CPU mutants (waitcnt bits, branch targets, sc1, lshr carry) killed.

| family | builds | direct |
|---|---|---|
| probes P4 counter / ticket / xl / boundary, P5 kv / scalar / host | 7 | all (were refused) |
| CCL all-reduce TP2 / TP4, all-gather TP2, argmax TP2 | 4 | all (were refused) |

Census vs R2 still 2,017 / 2,017 + 18 / 18 byte-identical (the census corpus has none of these kernels).
GPU validation of the asm kernels through their own suites (P4 / P5 numerics + negative controls + mutants, the CCL
multi-process tests) with both backends: `jobs/rt_asm.sh`, pending.

### GPU validation of the asm-carrying kernels (rt-asm1..5, n01-09, 2 GPUs for CCL)

Every kernel of the P4 and P5 benches and of the CCL multi-process suite compiles directly in strict mode
(`rt_census_asm.py`: P4 39, P5 126, CCL 818 kernel variants). The suites' own gates, both backends:

| suite | COMGR | direct |
|---|---|---|
| P4 (sync primitives, ticket, xl): numerics, negative controls, mutants | PASS | PASS |
| P5 (coherence): positives | PASS | PASS |
| P5 negative controls (writer without write-back must read stale) | PASS | FAIL by default, PASS with `acqfence` |
| CCL (bitwise all-reduce / all-gather / argmax, tag wrap, stress, dropped peers, aborts, stale, silent, coherence, lm_bitwise, lm_canon, lm_mutants) | PASS | PASS (after the lift fix below) |

Two findings, both fixed or levered:
- The asm lifting (`direct_lift.py`) dropped the compiler-barrier meaning of templates clobbering memory and the
  waits they end with: the CCL protocol raced (lm_mutants hung: a rank timed out; with lifting off, the assembled
  templates passed). A lifted memory-clobbering template now sits between compiler fences and keeps its trailing
  waits (`ROCKE_DIRECT_LIFT_FENCE=0` restores the old form for experiments). No B3 / fused MoE template clobbers
  memory (0 fences added there); the head gets 2.
- The direct compiler's acquire fence writes L2 back before invalidating (a release's `buffer_wbl2`), stronger than
  LLVM's gfx940 lowering; P5's negative controls then read fresh data. Lever `acqfence` lowers an acquire-only fence
  as LLVM does (wait, `buffer_inv`): P5 passes every control. Proposed default (no stage kernel has an acquire fence;
  census objects with one change bytes).

Other additions: v_bfi_b32, s_buffer_load_dword, s_dcache_inv rows (cross-checked against llvm-mc: 0 mismatches),
SMEM glc, the assembler's global atomics / buffer_inv / scalar loads / fixed-VGPR outputs (`{v[a:b]}`, raw `vN` names),
the template's instruction order kept (a scheduling barrier after each), `s` operands of values this analysis cannot
prove uniform read through `v_readfirstlane` (LLVM accepted them only as uniform), `ROCKE_DIRECT_STRICT=1`.
Gates v17: crate tests, census byte-identical vs R2, CPU mutants of every new piece killed.

### Toy chain paired timing (rt-t1 n01-09 / rt-t1c n07-17, 24 rounds, `jobs/rt_gpu.sh` step `r2`)

`r2_gpu.py time --what toy`: ratios vs the shipped COMGR object A; controls A2 (byte copy) and R (recompile).
I = COMGR on the lifted irops IR (the same IR the direct arms compile), D = direct no levers, DB = direct with the
B3 set, DM = direct with the MoE r1 set.

| toy shape | I | D | DB | DM | controls band | verdict DB |
|---|---|---|---|---|---|---|
| d4w8l2 | 1.024 / 1.037 | 1.088 / 1.104 | 0.988 / 0.994 | 0.998 / 1.001 | 0.987-1.015 / 0.979-1.026 | neutral (both) |
| d8w0 | 0.996 / 0.996 | 1.025 / 1.022 | 0.988 / 0.989 | 0.993 / 0.994 | 0.992-1.011 / 0.992-1.009 | WIN (both nodes) |

Plain direct loses (2-10%); the B3 set turns it into a two-node win on d8w0 and a neutral on d4w8l2, where the
lifted IR itself costs COMGR 2-4% (I) so DB is 3.5-4% faster than COMGR on the same IR.

## Never-regress track (PLAN decision 18, 2026-11-02; W-RNR)

User: "rust should only improve, not deprove anything". (a) A serving guard: direct only where a same-job paired A/B on
both rocke nodes shows direct <= COMGR; (b) every case where direct is slower is a compiler defect, fixed in the crate.
Ratios only (direct / COMGR of the same KernelDef, one process, shuffled interleaved rounds, the harnesses' DESIGN 9.1
verdicts; < 1 is faster). Absolutes and job index: `hackathon-prep/private-perf/qwen38-true-mono/rust-track/rnr/`.
Harness: `hackathon-prep/qwen38/truemono/RNR/`. No git state was changed. **Base of every verdict below: the
copy-rocm-lib crate `rnr-v8` (the WP-R3 crate + the levers of this section; the HC confirmation ran on `rnr-v7`,
whose objects for those sets are the same), backend defaults `acqfence` + `bufoffany`.**

### Serving guard (part a)

- `qwen38_mono/persist/rust_guard.py` (new) + the DB `persist/rust_backend_db.json` (TuningDB format, family
  `qwen38_mono.codeobj`, one record per key and node: ratio, verdict, the timed lever set, base, job). `select(key)` is
  direct only when both nodes' records are `WIN`, or `TIE` with ratio <= 1, for one lever set; anything missing,
  losing, a tie above 1 or `HOLD` (measured, not booked) is COMGR. `guard_kernel(kernel, key)` sets the backend mark
  (an explicit COMGR mark for the losers, so `ROCKE_CODEOBJ_BACKEND=direct` no longer moves them) and the booked
  `direct_levers`.
- Hooks (new code paths, the old behaviour unchanged): `persist.knobs.rust_family(kernel, family, key=...)` (the DB
  decides) and `rust_scope_backend(..., guard_keys=...)`; `ROCKE_DIRECT_GUARD=1` (default off) also gates the D0.6
  family defaults (`b3`, `moe_r1`) and framework `auto` by every recorded key of the family.
- Gates: `library/tests/test_qwen38_rust_guard.py` (14 tests: regressing families fall back to COMGR, the rule's
  must-fail cases, IR / LLVM text unchanged, COMGR objects byte-identical with the guard (compiled), the env switch
  ignored for guarded COMGR kernels); `RNR/guard_mutants.py`: 10 / 10 must-fail mutants of the rule and hooks killed;
  `test_qwen38_wpr3.py` unchanged and passing (D0.6 behaviour with the guard off).

### Compiler levers added (part b; all default-off, `ROCKE_DIRECT_LEVERS` / `dlevers` scopes)

| lever | what LLVM does better (found in the ISA diff) | code |
|---|---|---|
| `dsfold` | constants nested in `(x + c) * m` / shifts / subs of an LDS index go to the DS offset field; the remainder is one value (CSE) | `rir/opt.rs` `split_deep` |
| `addrfold` | a buffer voffset (also under its `select(c, x, 2^30)` guard, `_cp` forms too) rebuilt as one shared base + a constant per access (was 3 SALU + a fused add-shift per load) | `rir/opt.rs` `fold_voffset` |
| `dppcombinei` | i32 max / min and u32 min fold their DPP move (`v_max_i32_dpp`; new ISA rows) | `target/gfx9/dpp.rs` |
| `minmax3` | `max(max(a,b),c)` -> `v_max3_i32/u32`, min3, clamps -> `v_med3_i32/u32` (new rows) | `target/gfx9/minmax3.rs` (new) |
| `ldspairsfar` | dword LDS reads past `ds_read2`'s 8-bit offsets pair through one rebased address per 1 KiB window | `target/gfx9/lds_pairs.rs` `merge_far` |
| `ldsduos` | two 8-byte-aligned dword LDS reads -> `ds_read_b64` | `rir/isel.rs` `find_dword_duos` |
| `packfar` | packed f32 (`v_pk_mul/add/fma_f32`) for lanes that are not adjacent (SLP), up to 24 apart | `target/gfx9/pack.rs` `pair_far` |
| `fmac` | `v_fma_f32 d,a,b,d` -> VOP2 `v_fmac_f32` (code size) | `target/gfx9/mod.rs` `shrink_fmac` |
| `masksall` | a uniform compare's lane mask evaluated once in straight-line blocks and MFMA kernels too | `target/gfx9/masks.rs` |
| `u16zext` | the zero extension of a `buffer_load_ushort` / `ds_read_u16` result is the register itself (no `v_and`, which drained `vmcnt(0)` in the LB stream) | `rir/isel.rs` |
| `i16vec` | element extract / insert of an i16 vector (the LB x-scale reads) | `rir/isel.rs` |
| `ldsalias` | LDS accesses of different allocations do not order each other in the list scheduler | `rir/post.rs` |
| `vmlat60` / `vmlat100` | VMEM load latency 60 / 100 slots in the scheduler (model 180) | `rir/post.rs` |
| (coverage, default) | `v_cvt_scalef32_pk_fp4_f32` row: the LB fp4 requant's inline asm assembles (was a refusal) | `isa/gfx950.isa` |

Gates (CPU job `rnr-cpu-v8`, 68246983): 0 warnings; `cargo test` lib / all / 1 thread 495 each (14 new: 10 lever tests in
`target/gfx9/q38.rs`, 3 in `minmax3.rs`, 1 in `lds_pairs.rs`); `isa_tables check` both ISAs; census vs the frozen pre-RNR crate:
**no levers 2,017 + 18 objects byte-identical, the booked B3 and MoE r1 sets byte-identical** (2,017 each), each new lever's
changed objects listed (fmac 751, addrfold 378, packfar 620, vmlat60 652, ...; 0 status changes, 0 refusal changes);
lever-scope gate for the old 23 and the 14 new levers: scoped on == env, env scoped off == plain (2,029 each); the 9 new ISA
rows cross-checked against llvm-mc: 276 match, 0 mismatch. Every timed lever object below is bitwise equal to COMGR
in its harness (rt_hc vs COMGR and FlyDSL, rt_fmoe, rt_head, pl4b_bench numerics + mutants).

### Root causes, fixes, before / after (n01-09 / n07-17)

| regression | root cause (direct vs COMGR ISA, same IR) | fix | before | after (booked) |
|---|---|---|---|---|
| HC A1/A2/M1/M2 (24 framework kernels) | +7-10 % VALU / +40 % SALU: LDS index constants not in DS offsets (130 `v_add_lshl` + 77 `v_add`), per-load scalar voffset chains, amax `v_max_i32` chains (no dpp / max3), far scale reads unpaired, unpacked f32 lanes; `loadfirst` helps some widths and hurts others | `dsfold`, `addrfold`, `dppcombinei`, `minmax3`, `ldspairsfar` (+ per width `loadfirst` on/off, `fmac`, `vmlat60`, `masksall`, `packfar`) | D 1.03-1.12, B3+loadfirst 1.00-1.03 | **19 of 24 direct**: 0.972-0.998 on both nodes (screen rr-hc5 and confirmation rr-hc6, 32 rounds, all <= 1); 5 stay COMGR (below) |
| fused MoE rows 4 | `nofence` (in the r1 set) lets the scheduler pull stores and phase code apart; the RNR levers remove VALU / SALU | the MoE r1 set without `nofence` + RNR levers (`M1Nx`) | M1 1.002-1.005 / 1.002-1.009 | **0.993 TIE / 0.996 WIN** (rr-fl4, 32 rounds) |
| large-batch MoE (`moe_lb2`) | `nofence` hoisted each MFMA result's LDS staging store right behind its MFMA: 263 sites of `s_nop 7; s_nop 3` (12 wait states); `buffer_load_ushort` + `v_and 0xffff` forced `vmcnt(0)` in the stream; MFMA scales copied per MFMA (old lever `scalesep` was not in the tried scopes); fp4 refused (asm) | `M1Nx` (no `nofence`, `scalesep`, `u16zext`, RNR levers); fp4 row | dD 1.014-1.075, dlev sets 1.01-1.07 | **parity: 0.994-1.011 / 0.996-1.010**, fp4 compiles and is bitwise; COMGR kept (coordinator: both-node WINs first) |
| head `mx_decode_gemv` | main loop matches COMGR with `loadfirst` (WP-R3); the epilogue (argmax fold) carries +68 instructions: unvectorized LDS (b32 vs b64), SGPR copies, constant VGPRs | `ldsduos`, `litreuse` (fused r4: instruction count to COMGR's + 5) | 1.002-1.005, fused r16 1.009-1.011 | 1.002-1.004 (fused r4, logits r16), fused r16 1.008-1.010, logits r4 0.998-1.000: **COMGR kept** |
| toy d4w8l2 | (not re-measured) | - | B3 set 0.988 / 0.994 TIE | the guard books it (TIE <= 1 on both nodes, WP-R3 record) |
| fw Q2 (not on the list; 4-11 %) | no RNR lever changes its object (aliases); the polls (one dwordx4 tagged poll with a per-lane exit) and framework SALU differ | open | 1.045 / 1.114 (r1 / r4s, n07) | COMGR |

HC kernels still COMGR: `delta mlp r4s` (ZLP 0.992 / 0.995, chosen in its own confirmation run: HOLD until a fresh
confirmation), `qsa mlp r2s` (screen and confirmation disagree: 0.979 / 0.976 then 1.006 / 0.995), `qsa mlp r4s` (screen
1.008 on n01), `delta_ple attn r4s` (no set <= 1: best 1.003 / 1.003), `delta_ple mlp r4s` (screen 1.015 / 1.011, confirmation
0.998 / 1.000: noisy). The booked lever set per HC kernel is in the DB (`rust_guard table`).

### Base-tagged lever verdicts (decision 15: every reject stays in the pools)

- `nofence`: WIN on fused MoE r4 at base plain (rt-s1, -1.6 %); **LOSS at base M1N** (fused r4 0.9959 without vs 0.9995 with,
  n07) and on `moe_lb2` (dNx vs dN: 1.000-1.011 vs 1.006-1.075); neutral on HC at base N6f (+0.1-0.2 %).
- `masksall` alone: LOSS on HC attn at base B3L (+0.6-1.0 %); TIE / small WIN at base N6f (qsa attn r4s 0.9955 on n01);
  in `moe_lb2` it frees SALU but adds SGPR spills (v_readlane +481): LOSS at base dNx (1.007-1.024).
- `ldspairsfar` alone: LOSS at base B3L (1.019); part of every winning HC set (removing it at base N6f costs 0.4-0.5 %).
- `loadfirst`: per width: removing it wins on attn r1 / qsa (0.9907-0.9935), loses on mlp r3s (1.0018).
- `fmac`: neutral at base B3L; in the HC `ZLv` / `N6` sets.
- `vmlat60` / `vmlat100`, `ldsalias`: no change on HC (aliases) and `moe_lb2`; small WIN with `ZLv` on HC attn.
- `scalesep` (old lever): the main gain on `moe_lb2` (dS 1.007-1.026 from dD 1.014-1.075).
- Head: `F2` (ldspairs, ldsquads, dpp, fmac, minmax3, dsfold, addrfold) LOSS at base F (+0.1-0.2 %); `ldsduos` neutral; `ldsduos`
  + `litreuse` neutral (-0.04 %).
