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

