# The Rust direct code-object compiler and the Q38 true monokernel (research, 2026-10-29)

Read-only analysis. No git state was changed: the rust branch was read with `git archive` / `git show` into temp
dirs, merges were tried with `git merge-file -p` into temp files, and the only build was one `cargo check` of the
crate at the branch tip (clean, offline, no dependencies). Ratios only; absolute numbers stay in
`hackathon-prep/private-perf/rocke-direct-ct/` and `copy-rocm-lib-agents/*-out/private/`.

Names used below:

- R = `dnn-providers/hip-kernel-provider/rocke`, C = `R/platform/rust/rocke-codeobj/src`.
- **mono tree** = this working tree, branch `users/vanantha/rocke-monokernels` at 89209ac7665 (base 9b4fee7d990).
- **rust branch** = `users/vanantha/rocke/rust-compiler` at 23c165f8b3f. Its own work is 8 commits,
  3abfd5dc95a..23c165f8b3f, on top of e7d4664bda9; 9b4fee7d990..e7d4664bda9 is 186 upstream commits that the mono
  tree does not have.

## 0. Bottom line

1. The direct compiler is a real, gated gfx950 backend: rocKE IR text to HSACO in Rust, no LLVM. On the census it
   compiles a median 27-31x faster than COMGR (minimum 4.5x), byte-deterministic across thread counts and across
   Windows and Linux builds, and its code runs at about COMGR's speed: all 305 convolution cases at 0.986 of COMGR,
   183 attention-forward entries at 1.003, GEMM median 0.992. Family ratios range from 0.28 to about 1.06.
2. **It does not compile any Q38 kernel today, and it does not fall back for them either.** Every Q38 build uses an op
   name the Rust parser does not know (`tile.tagged_poll`, `tile.buffer_load_cp`/`buffer_store_cp`,
   `tile.int_to_ptr`, `tile.mfma_scale_mx`, `math.exp`, ...). That is a `ROCKE_DIRECT_PARSE` error, which
   `compile_kernel` raises instead of falling back to COMGR. The other blocker is `tile.inline_asm`, with about
   250 sites in 8 representative builds. The compiler accepts only the empty `"=v,0"` pin and three `ds_read`
   forms. An asm statement with zero or several results fails validation as `ROCKE_DIRECT_MALFORMED`, which does
   not fall back either.
3. What it could give a sync-bound persistent kernel beyond LLVM is real, but most of it has to be built. The parts
   that exist today:
   - a poll loop expanded after instruction selection (`flag_wait_eq`), which no IR pass can hoist;
   - an exact, per-register `s_waitcnt` model that you can read and change;
   - `s_barrier_bare` without a drain;
   - sc1/nt bits on global accesses;
   - spills that never cross an EXEC join;
   - a hazard recognizer that covers every instruction (nothing is hidden in asm);
   - deterministic output and fast compiles.

   What has to be built:
   - the Q38 ops (about 4-6 engineer-weeks for all of them; section 2);
   - kernarg preload;
   - branch relaxation, which a whole-round kernel needs;
   - register pinning across stages;
   - a source-line map so WaveScope still has blame.
4. Recommendation: overlay only the rust branch's own commits, not the whole branch: 137 new files, 43 straight
   copies, 24 three-way merges, of which 18 are clean. Leave out every file the direct path does not need, including
   `lower_llvm.py`. Add it as a default-off knob (`code_object_backend=direct`) with an automatic COMGR fallback.
   Make `mx_decode_gemv` (the head) and the toy chain the first direct targets. Make direct a lever in M5 (RSI),
   not a dependency of M1-M4.

## 1. The direct compiler

### 1.1 Pipeline

The compiler is a Rust `cdylib`, `rocke_codeobj_ffi`, with no dependencies (`Cargo.lock` lists only the crate). It
is about 66k lines over 111 files: `rir/` (42 files, shared with gfx1151), `target/gfx9/` (44 files, gfx950), the ELF
writer, and the ISA descriptions `isa/gfx950.isa` (286 instructions), `isa/gfx11.isa` and `gfx11.overlay`. The
stages, from `R/platform/dsl_docs/runtime/direct_compiler.md`, checked against the sources:

| Stage | What it does | Files (under C) |
|---|---|---|
| Parse | Parses `ck.dsl.ir/v1` text. Input is cut into chunks parsed on several threads, found by a SIMD line scanner (AVX-512 or AVX2). Repeated constants are hash-consed, and a session can parse an input while it is still being written. On gfx950: early constants, splat-mask selects become scalar selects, and `nontemporal` becomes the `nt` bit. | `rir/parse.rs`, `parse_par.rs`, `scan.rs`, `hashcons.rs`, `stream.rs`, `target/gfx9/{early_constants,mask_splat,cpol}.rs` |
| Validate | Checks operand, result and attribute types for every op, and the limits: region nesting, vector width, LDS, workgroup size, uniformity. | `rir/validate.rs` |
| LDS layout | Lets allocations whose lifetimes do not overlap share space. | `rir/analysis.rs` |
| Optimize | Simplification and folding; CSE, including repeated LDS loads; LICM of **pure** ops only (loads, asm and syncs are not pure: `OpKind::is_pure`); known bits; `unroll=true` loops; `noalias`/`readonly` facts. | `rir/opt.rs`, `unroll.rs`, `alias.rs` |
| Auto-unroll | Unrolls small constant-trip loops, kept only if the result spills no more and keeps occupancy, or if a cost model predicts a fifth fewer cycles. | `rir/unroll.rs`, `target/gfx9/unrolled.rs` |
| gfx950 IR passes | Fuses `if` blocks and widens their stores. Reuses repeated loads: within one straight-line run, and also past barriers for memory of `noalias readonly` arguments. | `target/gfx9/{widen,ro_loads}.rs` |
| Uniformity | Assigns each value to SGPRs (wave-uniform) or VGPRs (per-lane). | `rir/analysis.rs` |
| isel | Op-by-op selection, the BURS-style tables generated from the `.isa` descriptions, and gfx950 hooks: MFMA (plain, block-scaled 32x32x64 f8f6f4, BF16x3), LDS DMA with M0, transposing LDS reads, fp8/bf8 conversions, per-wave atomics, lane ops and cache-policy bits. `burs` (a tree-pattern selector for i32 address math) is a prototype and off by default. | `rir/isel.rs`, `rir/burs.rs`, `rir/gen_burs.rs`, `target/gfx9/{isel_hooks,mfma,lds_dma,lds_tr,fp8,atomics,lanes}.rs` |
| Machine rewrites | Peephole rules proven on gfx950, VALU fusion, zero-accumulator MFMA, M0 hoist, packed f32, `v_lshl_add_u64`, operand legalization, literal and copy reuse. | `rir/{post,peephole}.rs`, `target/gfx9/{zero_acc,m0_hoist,rules,select,wide_add,pack,masks,legalize}.rs` |
| Scheduling + regalloc | Schedules each candidate window and allocates it, then keeps the widest window that keeps occupancy, or a wider one when the cost model wins. AGPRs are used for accumulators. Other searches: bank-wide windows, SGPR relief, clause-aware allocation. | `rir/{mod,regalloc,mma_order,fma_mix}.rs`, `target/gfx9/{sched,agpr,agpr_spill,bank_wide,trade,latency,relief,clauses}.rs` |
| Spill | VGPRs spill to free AGPRs, then to LDS slots that cost no occupancy, then to scratch; SGPRs spill to VGPR lanes. A spill store runs under its definition's EXEC; reloads never move across an EXEC write. `verifyspill` re-checks each round. | `rir/{spill,reload}.rs`, `target/gfx9/{spill,agpr_spill}.rs` |
| Waitcnt | Per-register scoreboard on vmcnt/lgkmcnt, merged over predecessor blocks. LDS DMA ranges are tracked as their own keys. Explicit waits retire what they cover. `s_barrier` (`tile.sync`) waits for everything, the LDS-only barrier waits for lgkmcnt alone, and `s_barrier_bare` waits for nothing. | `target/gfx9/waitcnt.rs` |
| Hazards | LLVM's `GCNHazardRecognizer` predicates for gfx950: H1-H14, the MFMA/AGPR/DOT rules M1-M9, and the permlane swaps. Waits are `s_nop`s counted over every path into a block. | `target/gfx9/hazards.rs` |
| Encode + ELF | Encoders generated from the tables, which refuse operands the hardware ignores. Writes the kernel descriptor (AGPR offset), the MessagePack metadata note, symbols and hashes. | `rir/{asm,isa}.rs`, `target/gfx9/asm.rs`, `hsaco_writer.rs`, `elf.rs` |

Tables: `tools/isa_tables.py` generates `rir/gen_isa.rs`, `gen_burs.rs`, `gen_opcodes.rs` and
`target/gfx9/gen_isa.rs` from the `.isa` files. `check` fails on a stale table.

### 1.2 gfx950 coverage

Supported: every op of the kernel families rocKE builds for gfx950. That means:

- MFMA: f16/bf16/f32/f64/fp8/bf8, `16x16x128_f8f6f4`, the block-scaled `32x32x64` atoms, and the xf32 (BF16x3)
  emulation, with AGPR accumulators;
- LDS DMA, `ds_read_tr*`, fp8/bf8 conversions, packed f32;
- `permlane16/32_swap`, DPP xor/quad/dpp8, `ds_bpermute`/`ds_swizzle`, `readlane`/`readfirstlane`;
- `buffer_*` and `global_*` access, `global_atomic_add`(_f32, _pk_f16);
- `memref.fence`, which becomes `buffer_wbl2` / `buffer_inv` with the needed waits;
- the stream-K flag ops `global_flag_wait_eq` and `global_flag_store`;
- `s_setprio`, `sched_barrier` (mask 0 is a fence; other masks are ignored), and explicit `tile.s_waitcnt`.

Targets are gfx1151 and gfx950 only. The code object is built for xnack and sramecc `any`.

### 1.3 What it refuses or ignores

- **`tile.inline_asm`:** only the empty template with `"=v,0"` (a register pin), and on gfx950 `ds_read_b32/b64/b128`
  with `"=v,v"`. Any other template with one result is `ROCKE_DIRECT_UNSUPPORTED`, which falls back. Zero or several
  results fail validation's `arity(1,1)` as `ROCKE_DIRECT_MALFORMED`, which does **not** fall back.
- **`tile.iglp_opt` and `sched_group_barrier`** are accepted and do nothing: there are no scheduling groups. A
  `sched_barrier` with mask 0 is a scheduling fence.
- **`agpr_alloc`:** only `[0, 0]`.
- **Unknown op names** are `ROCKE_DIRECT_PARSE`, which does **not** fall back.
- **One kernel per code object.** `compile_module` and the LLVM-IR entry points are COMGR-only.
- **No DWARF and no line tables:** `ROCKE_DEBUG_LOC` does nothing, and profilers and rocgdb see no source lines.
- **No branch relaxation:** a branch past the SOPP simm16 range (about ±128 KiB of code) fails with
  `ROCKE_DIRECT_ENCODE` (`rir/asm.rs` "branch out of range"). The g950i report lists it as still open.
- The LDS swizzle runs on wave32 only. A kernel without arguments keeps its kernarg pointer. Deep `scf.if` nesting
  can run out of SGPRs with an unhelpful diagnostic.
- **Silently ignored:** the `inreg` parameter attribute (kernarg preload). The parser reads only `noalias` and
  `readonly` from parameter attributes, so a preload build compiles with plain `s_load` kernargs.

### 1.4 Determinism and numerics

- The output depends only on the IR text and the target. Thread counts, session vs whole vs streamed input, and the
  SIMD scan level change no byte. Nothing is cached. In the compile-time campaign, every one of the 2,013 objects
  was stable across samples and byte-identical between the Linux and Windows builds.
- **Numerics against COMGR:** every rounding the IR states is kept. LLVM can fuse, for example the last FMA of a
  chain plus its f16 convert into `v_fma_mixlo_f16`, which gives up to 1 ULP of difference. `fmax`/`fmin` quiet
  signaling NaNs, as `maxnum` requires, and no rewrite assumes there are no NaNs or signed zeros. Float atomics are
  order-dependent under either compiler. On hardware, every changed object ran bitwise or within the float64
  reference in the g950 campaign.
- **Q38 gates:** the Q38 gates are "bitwise vs the mono" (FlyDSL/LLVM objects). Kernels that are mostly bf16 with
  explicit `arith.fma` should match. Each stage still needs its own bitwise check, because LLVM's fusions are the
  one known source of 1-ULP differences.

### 1.5 Gates and tools

From `R/platform`, with `PYTHONPATH=python`:

- `cargo test --locked`;
- `python tools/isa_tables.py check` and `check --isa gfx950`;
- `pytest tests/runtime/test_code_object.py test_code_object_elf.py test_direct_gemm.py test_direct_targets.py
  test_direct_session.py`;
- `ctest -R "rocke_native_(gemm|direct)_code_object"`.

The crate examples (`cargo run --release --example <name>`):

| Example | Use |
|---|---|
| `target_compile <gfx> <out> [--max-vgprs N] <in.ir>...` | `.hsaco` plus a `.s` listing per input. `--max-vgprs` forces spills. |
| `bench950 <dir>` / `bench_corpus` | Compile time over a corpus. Fails if any object differs from the corpus's `.hsaco`, so it doubles as the byte-identity gate. |
| `hazard_corpus <report.tsv> <disasm>...` | Strips COMGR's `s_nop`s, puts its own back, and fails anywhere it waits less than COMGR. |
| `isa_xcheck [--target gfx950]` + `tools/isa_xcheck.py` | Encoding variants of every table instruction, compared against `llvm-mc`, offline. |
| `mir_dump`, `rir_compile`, `unroll_stats`, `lds_conflicts`, `mb_hsaco`, `model_check` | Machine IR after isel, optimized IR, unroll stats, a static LDS bank-conflict report, a microbenchmark packager, cost-model checks. |

Change rule: any object a change did not mean to change stays byte-identical. Every changed object gets an
on-silicon check against COMGR's object for the same IR, inputs and launch shape. Allocation and spill changes are
also checked at forced VGPR budgets, with `verifyspill`, and with `ROCKE_DIRECT_THREADS=1`.

### 1.6 Recorded status against COMGR (ratios; ours over COMGR, below 1 is faster)

- **Compile time, gfx950 census on alola CPUONLY** (`private-perf/rocke-direct-ct/out2/tables.md`, 2,013 kernels):
  median speedup 27x, minimum 4.6x, total-time ratio 37x; 32x and 42x when Python lowering is included. Earlier run
  (1,230 kernels): 31x median, 4.5x minimum, 47x total. Slowest direct compile against its COMGR compile: 1/12.
- **Runtime**, from the g950n end check, with nulls at 0.996-1.001:
  - convolution, all 305 cases: 0.986;
  - attention forward, 183 entries: 1.003;
  - conv families from 0.278 (grouped dgrad) to 1.029 (wgrad);
  - attn dense 1.049, tuning 0.966, unified route 1.046.

  From g950h: GEMM median 0.992. The G1 (norm/elementwise) family is at 1.000.
- **What is on the branch:** the crate at 23c165f8b3f equals `copy-rocm-lib-agents/stage3-after-g950n` byte for byte
  apart from line endings, and includes g950o's clause preference. g950p is **not** on the branch: `ranges.rs`,
  `lds_rebase.rs`, `lds_reuse.rs`, `prefer_facts`/`plain_twin`. g950p brought attention tuning to 0.957, unified to
  0.990 and dense to 1.065, and the named VGPR-limit kernels to 0.78-1.13. Ask the user whether to take g950p's crate
  from the agents workspace when overlaying.
- **One-off check:** one `cargo check --locked --offline` of the branch crate passed clean.

## 2. What the Q38 code uses, and what the direct compiler does with it

### 2.1 Method

I built representative kernels in Python, without compiling them, and walked their IR:

- toy chain `best_spec(4,8,layers=2)` and `(8,0)`;
- B3 r1 default and r4s tuned;
- fused MoE r1 `BEST_KNOBS[0]`;
- the MoE suffix r4;
- `mx_decode_gemv` head, `nt` and fused-argmax variants;
- an `mx_mfma` probe.

I then serialized four of them with the rust branch's `ir_serialize.py`, through a temp overlay of
`platform/python` (it imports cleanly against the mono tree), and diffed op names against `C/rir/ast.rs`.

Sizes:

| Build | Ops | Serialized IR |
|---|---:|---:|
| toy chain | 2.3k | 174 KiB |
| B3 r1 | 3.0k | 214 KiB |
| fused MoE r1 | 8.5k | 612 KiB |
| fused-argmax head | 2.5k | 196 KiB |

A whole round (one body per layer kind plus head plus draft) is likely a few MiB. The census's largest direct
compile was 2.6 MB of IR, at a 1/7.5 to 1/12 compile-time ratio against COMGR.

The ops the Rust parser does not know, per build:

- toy chain: `buffer_store_cp`, `int_to_ptr`, `tagged_poll`;
- B3: `math.exp`;
- fused MoE: `cvt_scalef32_pk_fp8_f32`, `math.exp`, `buffer_load_cp`, `buffer_store_cp`, `int_to_ptr`,
  `mfma_scale_mx`, `tagged_poll`;
- head: `cvt_scalef32_pk_bf16_fp4`, `buffer_load_cp`, `buffer_store_cp`, `permlane16_swap`, `tagged_poll`.

Every other op name in these builds is known.

### 2.2 Op-by-op table

Effort is in engineer-days, including the ISA cross-check, tests and a hardware bitwise check. "Fallback first"
means adding the name in `rir/ast.rs` with a `ROCKE_DIRECT_UNSUPPORTED` refusal, about 0.5 d for all of them, so
`compile_kernel` falls back instead of raising.

**Mailbox, poll and memory model** (`helpers/mailbox.py`, `qwen38_mono/mailbox_ir.py`, `device_ops.py`):

| Q38 use | Today | Work and effort | Rust files to extend |
|---|---|---|---|
| `tile.tagged_poll` (`mb_impl=ir`; aux sc1; pause `s_nop 0` / `s_sleep 1`; 1-16 specs of 2 or 4 dwords) | PARSE, no fallback | New leaf modeled on `flag_wait_eq`: the buffer loads with cpol, explicit `vmcnt(0)`, one `v_cmp_ne` per tag, OR the lane masks, `s_and exec`, branch on scc (wave-uniform exit, as the asm does), and the pause. Register it as a fence in `ro_loads`, `alias`, `loadorder`, `lane_cse` and `unroll`. 2-3 d | `rir/ast.rs`, `validate.rs`, `isel.rs` (beside `flag_wait_eq`), `target/gfx9/ro_loads.rs`, `cpol.rs` |
| `tile.buffer_load_cp` / `buffer_store_cp` (aux 0/2/16/18 = none/nt/sc1/sc1+nt; 1, 2 or 4 dwords) | PARSE | Reuse the `BufferLoadVec` and store isel. Add `m=sc0,sc1,nt` to the MUBUF rows (the MUBUF encoding already has the sc0, sc1 and nt fields) and an `M_SC0` bit. 1-2 d | `isa/gfx950.isa`, then `tools/isa_tables.py generate --isa gfx950`; `target/gfx9/cpol.rs`, `asm.rs`, `rir/isel.rs` |
| `tile.int_to_ptr` (i64 to global, or to `constant` for `s_load` tables) | PARSE | Global: a register-class move (`Val::D`), 0.5 d. The constant-space form is an `s_load` path for uniform loads through the pointer (`SLoadB*` rows exist), 1 d | `rir/ast.rs`, `validate.rs`, `isel.rs`, `analysis.rs` (uniformity) |
| `tile.global_load_atomic` / `global_store_atomic` (scope, ordering) | PARSE | Reuse `flag_store` and `memory_fence`, plus sc0/sc1 by scope. 1 d | `rir/isel.rs`, `target/gfx9/isel_hooks.rs` (`fence`) |
| asm mailbox put `global_store_dwordxN ... sc1 [nt]` (`v,v,s`, no result) | MALFORMED | Use `mb_impl=ir` (`buffer_store_cp`), or add a `cpol` attribute to `memref.global_store_vN`: global sc1 and nt already exist, sc0 does not. 0.5 d | `target/gfx9/cpol.rs`, `rir/isel.rs` |
| asm poll loop (`${:uid}` labels, multiple results) | MALFORMED | Do not port it; direct builds must use `mb_impl=ir`. 0 d | none |
| `ptr_from_i64` asm `s_mov_b64 "=s,s"` / `v_mov_b64` | UNSUPPORTED (falls back) | Use `mailbox_ir.ptr_from_i64_ir` (`int_to_ptr`), or accept the template as an SGPR pin. 0.5 d | `rir/isel.rs` (InlineAsmIdentity) |
| `fresh` (`; fresh $0`, `"=v,0"` / `"=s,0"`) | UNSUPPORTED | Treat comment-only templates as the existing pin and add an SGPR pin. Probably not needed: the direct compiler rematerializes one-dword values from constants and SGPRs, which is the spill `fresh` works around in LLVM. 0.5 d | `rir/isel.rs`, `validate.rs`, `unroll.rs` (pin rule) |
| `s_sleep N` (holds, `head_sleep`, `pause=s_sleep1`) | MALFORMED (no result) | New op `tile.s_sleep` plus an `SSleep` SOPP row. 0.5 d | `isa/gfx950.isa`, `rir/ast.rs`, `isel.rs` |
| `s_nop N` (`device_ops.s_nop`, MFMA tail nops) | MALFORMED | Not needed once MFMAs are IR ops, since the hazard pass inserts nops. Keep an op for timing experiments. 0.25 d | `rir/isel.rs` |
| `s_memrealtime` / `s_memtime` (stamps, phase timing) | MALFORMED/UNSUPPORTED | New op, an SMEM row on lgkmcnt. 0.5 d | `isa/gfx950.isa`, `rir/isel.rs`, `target/gfx9/waitcnt.rs` (an SMEM result) |
| `s_getreg_b32 hwreg(HW_REG_XCC_ID / HW_ID)` (NUMA IOD calibration, placement) | UNSUPPORTED | New op, an `SGetregB32` SOPK row (hazard h13 already models setreg/getreg). 0.5 d | `isa/gfx950.isa`, `rir/isel.rs` |
| `buffer_wbl2` / `buffer_inv` sc1 or sc0+sc1, `release_l2` / `acquire_l2` (bulk hand-off) | MALFORMED | Use `memref.fence` (lowered by `isel_hooks::fence` to the same instructions), or add a `tile.cache_op` op (`BufferWbl2` and `BufferInv` rows exist). 0.5 d | `rir/isel.rs` |
| `atomic_add_scoped` (`global_atomic_add ... sc0 sc1` + `vmcnt(0)`, fused-argmax counter) | UNSUPPORTED | Use `memref.global_atomic_add` (`GAtomicAddU32Rtn` with glc exists); add sc1 for system scope. 0.5 d | `target/gfx9/atomics.rs`, `cpol.rs` |
| `ds_or_b32` + `lgkmcnt(0)` (MoE expert bitmaps) | MALFORMED | A `tile.lds_atomic_or` op and a `DsOrB32` row. 0.5 d | `isa/gfx950.isa`, `rir/isel.rs` (like `lds_atomic_add`) |
| `s_ff1_i32_b32` (fused MoE slot scan) | UNSUPPORTED | Op plus a row (only the b64 row exists). 0.5 d | `isa/gfx950.isa`, `rir/isel.rs` |
| asm `s_load_dwordx4 ... ; s_waitcnt lgkmcnt(0)` (B3 table) | MALFORMED/UNSUPPORTED | The constant-pointer `int_to_ptr` path above | as above |
| asm `global_load_dwordx4` L2 touch + asm `s_waitcnt vmcnt(0)` consuming them (toy `prefetch=l2_touch`) | UNSUPPORTED/MALFORMED | A `tile.prefetch` op: a load whose result is dead but which is never removed, with an optional drain point. 1 d | `rir/ast.rs`, `opt.rs` (DCE), `isel.rs` |
| `tile.s_barrier_bare` (fence compiler/workgroup), `tile.sync_lds_only`, `sched_barrier` mask 0, `s_setprio` | supported | none. Do not use `b.sync()`: like LLVM, `tile.sync` drains every counter. | - |
| `memref.global_*_typed/vN`, `tile.buffer_load_vN`, `buffer_rsrc`, `smem_*`, DPP xor, swizzle, bpermute, `permlane32_swap`, `readlane`, `readfirstlane`, `global_ptr_to_i64` | supported | none | - |

**Compute** (`mx_decode_gemv.py`, `mx_mfma.py`, `moe_fused.py`, `stage_b3.py`, `mx_quant.py`, `core/arch/mfma_scale_mx.py`):

| Q38 use | Today | Work and effort | Rust files to extend |
|---|---|---|---|
| `tile.mfma_scale_mx`, **16x16x128**, fmt_a 0 (fp8) / fmt_b 4 (fp4), `opsel_b` 0 or 2 (MoE: 150 sites) | PARSE | Only the 32x32x64 scaled form exists (`mfma::scaled`, scale bytes taken from the low byte). Needed: the `v_mfma_scale_f32_16x16x128_f8f6f4` row (the same VOP3PX2 form with `v_mfma_ld_scale_b32`, op 0x2d) and opsel to scale-byte select. Either map the op or have the builder emit `tile.mma` with scales. Hazards m1-m7 exist. 2-3 d plus a bitwise MoE gate | `isa/gfx950.isa`, `target/gfx9/mfma.rs`, `rir/ast.rs` |
| `tile.mma mfma_f32_16x16x32_bf16` (head) | supported | none | - |
| `arith.cvt_scalef32_pk_bf16_fp4` (byte 0-3; head: 960 sites) | PARSE | A gfx950 VOP3 row and selection; check numerics against the OCP decode. 1 d | `isa/gfx950.isa`, `target/gfx9/fp8.rs` |
| `arith.cvt_scalef32_pk_fp8_f32` / `_bf8_f32` (`hi`; MXFP8 quant) and the asm forms in `mx_quant.py` (`cvt="asm"`, op_sel) | PARSE / UNSUPPORTED | The same row family. Builds should use `cvt=intrinsic`. 1 d | as above |
| `tile.permlane16_swap` (head: 40-120 sites) | PARSE | The `VPermlane16SwapB32` row and hazard H14 already exist; add selection beside `permlane32_swap`. 0.5 d | `rir/ast.rs`, `target/gfx9/lanes.rs` |
| asm DPP `v_max_u32_dpp row_shr/row_bcast` scan (`wave_umax_dpp`, MoE), B3 `v_mov_b32_dpp row_shr/shl bank_mask` sibling exchange | UNSUPPORTED | A general `tile.dpp {ctrl,row_mask,bank_mask,bound_ctrl}` op on `VMovB32Dpp` plus DPP VALU forms. H7 is modeled. 1-2 d | `isa/gfx950.isa`, `rir/isel.rs`, `target/gfx9/hazards.rs` |
| asm `v_maximum3_f32 \|a\|,\|b\|,\|c\|` (MXFP8 amax) | UNSUPPORTED | An op with `maximum` NaN semantics (unlike `fmax3`) plus a row. 0.5-1 d | `isa/gfx950.isa`, `rir/isel.rs` |
| asm MFMA chains (`mx_mfma` `mfma_asm*`, probes only) | UNSUPPORTED | Keep them on COMGR; probes are not on the serving path. 0 d | - |
| `math.exp` f32 (B3 gates, MoE router) | PARSE | Reproduce LLVM's non-`afn` `llvm.exp.f32` expansion so results stay bitwise with the mono. The `exp2` and `ldexp` pieces exist. 1-2 d | `rir/isel.rs` |
| `math.sqrt` f64 (Q2 stage) | not parsed as an f64 op | LLVM's f64 sqrt expansion (`v_rsq_f64`, Newton steps, scaling), bitwise, 1-2 d. Or keep Q2 on COMGR. | `rir/isel.rs`, `isa/gfx950.isa` |
| `math.exp2_fast`, `rcp_fast`, `rsqrt`, `log2`, `fma`, `fdiv` f32, `trunc_f32_to_bf16`, `fabs`, `fmax` | supported | none | - |

**Kernel ABI and composition:**

| Q38 use | Today | Work and effort | Rust files to extend |
|---|---|---|---|
| `inreg` kernarg preload (`kernarg_preload=8`; it removed the stage-entry lgkmcnt(0), which was 0.44 of entry stalls) | silently ignored (falls back to `s_load`) | Descriptor fields (`kernarg_preload` length and offset), user-SGPR layout, gfx950's 256-byte backward-compatible entry, and precolored SGPRs. 3-4 d | `hsaco_writer.rs`, `rir/parse.rs` (param attr), `regalloc.rs` (precolor), `target/gfx9/mod.rs` |
| `device_function.py` (an LLVM `define void` for FlyDSL linking) | n/a | Hybrid path only. The true mono inlines everything into one IRBuilder, so nothing is needed. | - |
| Branches past about ±128 KiB (one body per layer kind inside a runtime layer loop, plus head and draft) | `ROCKE_DIRECT_ENCODE` | Branch relaxation: `s_getpc_b64` / `s_add_u32` / `s_addc_u32` / `s_setpc_b64` through a reserved SGPR pair, plus iteration over block sizes. **Needed for a whole-round kernel.** 2-3 d | `rir/asm.rs` (~l.492), `target/gfx9/asm.rs`, `rir/mod.rs` (layout) |
| `ROCKE_DEBUG_LOC` for WaveScope / Coz blame | no DWARF | A sidecar map from PC to IR op (`MInst` has no origin field today) that WaveScope can read in place of line tables. 2-3 d | `rir/mir.rs`, `isel.rs`, `hsaco_writer.rs` |

### 2.3 Total

- Fallback hygiene: about 0.5-1 d. Op names, plus `inline_asm` arity failures turned into UNSUPPORTED.
- Everything the serving builds need on `mb_impl=ir` and `cvt=intrinsic`: about 15-25 d of compiler work, plus about
  5 d of hardware bitwise gates.
- Branch relaxation, kernarg preload and the line map: about 7-10 d more.
- About 4-6 engineer-weeks in all for one person working in Opus-agent style. The g950 agents landed comparable
  slices at about 3-6 ops per agent-day.

## 3. Extra control for a persistent, sync-bound kernel

The hybrid's layer kernels are bound by sync and latency:

- 2.3x their issued-bytes floor;
- CTAs poll 0.60-0.75 of the time;
- hand-off hops take 0.26-0.55 of the critical path;
- B3 at r2s spent 0.39 of its time in `s_barrier` and 0.15 in vmcnt waits;
- the mono head spent 0.45-0.47 at `vmcnt(0)` plus barrier.

The table shows what the direct compiler controls that LLVM/COMGR does not, and whether it exists today.

| Control | LLVM/COMGR today | Direct compiler | State |
|---|---|---|---|
| **Poll loops never hoisted or merged** | IR has no while loop. The `ir` poll is a self-contained block structure with a side-effecting pause asm, and correctness relies on that asm and on `convergent`. Device functions that were not convergent hung stamped builds. | Polls are leaf ops expanded after isel (`flag_wait_eq` is the template). LICM moves only pure ops, and loads, asm and syncs are not pure. | Template exists; `tagged_poll` needs building (2-3 d). Gate: `ro_loads` must treat the poll as a fence. **Never mark mailbox or scratch arguments `noalias readonly`**: the direct compiler then reuses loads across barriers. |
| **Exact `s_waitcnt` placement** | `SIInsertWaitcnts` is opaque. The asm poll forces `vmcnt(0)` every retry. Inline asm hides counter effects. | A per-register scoreboard we own (`target/gfx9/waitcnt.rs`). `s_barrier_bare` waits for nothing. Explicit waits retire what they cover. LDS DMA ranges are their own keys. | Exists. New levers to build (1-3 d each): see the list after the table. |
| **No vmcnt(0) drain before barriers** | Avoided only by using `s_barrier_bare(fence="compiler")`. | Same semantics: `tile.sync` drains everything, `s_barrier_bare` drains nothing, and a "workgroup" fence adds only the fence's own waits. | Exists. |
| **Cache-policy bits per access** | Through asm or `buffer_*_cp` aux bits (Python only). | `cpol.rs`: sc1 from agent-scope atomics and flags, nt from `nontemporal`, on global accesses. | Partly exists. Missing: sc0 (system scope), cpol on MUBUF, and a per-op `cpol` attribute (1-2 d). This covers `state_nt`, the largest B3 lever (0.9695), and `put_nt`. |
| **Instruction order and placement of prefetch versus poll** | The machine scheduler may move loads across plain IR; ordering needs `sched_barrier` or asm. | Fence-flagged instructions (`SWaitcntExplicit`, `BufferInv`/`Wbl2`, `SBarrierBare`, `SSetprio`, `SchedBarrier`, LDS DMA) are never crossed by the window scheduler. A poll leaf would be a fence, so weights issued before it stay before it. | Exists for the fence set. Add the poll and prefetch ops. |
| **Register pinning across stages** (tag, layer, mailbox bases, NUMA copy offsets kept in fixed SGPRs for the whole run) | Not expressible; the `fresh` asm only prevents merging. | The allocator already precolors hardware inputs (`regalloc.rs`, "precolored groups first"). Extending that to an IR `reg` attribute or reserved SGPR ranges is straightforward. | Needs building, 3-5 d. Also gives kernarg preload its SGPRs. |
| **Exec-join spill miscompiles** (FlyTape's `lint_isa.py` target) | Possible; needs an ISA lint of every build. | A spill store runs under its definition's EXEC; a reload never crosses an EXEC write; `verifyspill` recomputes every round. | Exists by construction. Keep the lint as a gate anyway. |
| **Hazards inside "asm"** (learning: an asm MFMA right after a VALU write needs `s_nop`) | LLVM does not hazard-check asm bodies. | Once asm idioms are IR ops, every instruction goes through `hazards.rs`. `hazard_corpus` proves it never waits less than COMGR. | Exists for IR ops. |
| **Determinism and build history** (memory note: `iglp_opt(1)` code depends on earlier compiles in the same process) | History-dependent; needs fresh processes and a job-local comgr cache. | Byte-deterministic, no cache, thread-count invariant. `iglp_opt` is a no-op. | Exists. |
| **Compile time of a whole-round kernel and of RSI loops** | COMGR's slowest census compiles are 1-3 orders of magnitude above the median, and its tail is history-dependent. | Median 27-31x faster, minimum 4.5x. The largest census kernel (2.6 MB IR) compiled at 1/7.5. Hill climbs and exhaustive combination sweeps over whole-round builds get about an order of magnitude more candidates per hour. | Exists. A several-MiB whole-round kernel is untested (`bench950` on the first M3 IR). |
| **Code size** | LLVM can inline-expand and unroll; there is no layout control. | No DWARF. Auto-unroll is size-bounded and reverts when occupancy drops. The `.s` listing per object (`target_compile`) gives exact byte counts. | Exists. Branch relaxation is **missing**: a hard blocker above about ±128 KiB jumps. |
| **Occupancy and AGPR use at 512 threads, one CTA per CU** | `waves_per_eu` hints. | Every kernel keeps room for one workgroup per CU. Spills go to AGPRs, then free LDS, then scratch. The mono already runs at 228-246 VGPRs. | Exists. Check B3 and MoE for spills first, as with COMGR. |

New waitcnt levers that the direct compiler would make possible:

- Counted vmcnt for "poll first, then weights": issue the poll loads, then the weight loads, then
  `vmcnt(#weights)`. This retires only the poll, because vmcnt completes in order. LLVM can compute that count too,
  but the asm or block-structured poll keeps it from doing so.
- A scalar-memory poll (`s_load ... glc`, which counts on lgkmcnt) that never drains the weight stream on vmcnt.
  This needs hardware validation of scalar-cache coherence, and note that the toy's `step_load=smem` lost.
- Waits per IR op, as a knob.

What the direct compiler does **not** buy: LLVM's `iglp_opt` and sched-group interleaving (a no-op here), and DWARF.
Profiling and bisecting a direct object would need the line map. The g950 campaigns show no general speedup over
COMGR on ordinary kernels (about 1.00 overall). The value for Q38 is control of the sync path, plus compile speed
for search.

## 4. Integration into the mono working tree (no git state changes)

### 4.1 Do not overlay the whole branch

A 3-way merge of the 41 files both sides changed since 9b4fee7d990 (temp files, `git merge-file -p`, with ours as the
mono tree, base 9b4fee7d990, theirs 23c165f8b3f) gives:

- **18 clean, 22 conflicted, 118 conflict hunks**, plus 1 identical file.
- Almost all of the conflicts are GDN decode: the mono commit carries an older copy of upstream's GDN knobs and load
  order (#13789, #13821, #13961). The worst file is `library/kernels/gfx950/gdn_decode.py` with 25 hunks. Others:
  `dispatch/gdn/gfx950.py` 9, `common.py` 7, `ALGORITHM.md` 10, GDN tests 1-9 each, and three test files that both
  sides added: `test_gdn_decode_knobs.py` 3, `..._knobs_gfx950_numeric.py` 8, `..._load_order.py` 8.
- Outside GDN: `core_types.cpp` 2, `ir.h` 1, `lower_llvm.py` 1, `rocke_representative_ir_sha256.json` 6, the
  `target_intrinsics_emit` pair 1 each.

None of these files is needed for the direct compiler.

### 4.2 Overlay the rust commits only

Use base e7d4664bda9, theirs 23c165f8b3f, ours the mono tree; 204 files in all.

| Class | Count | Action |
|---|---:|---|
| new on the rust side | 137 | Copy straight. This is the crate (111 files), `runtime/code_object.py`, `code_object_elf.py`, `direct_session.py`, `helpers/direct_gemm.py`, `tools/isa_tables.py`, `isa_xcheck.py`, `isa_burs.py`, `tests/runtime/test_{code_object,code_object_elf,direct_gemm,direct_session,direct_targets}.py`, `tests/tools/*`, `dsl_docs/runtime/{direct_compiler,code_object_compiler}.md`, and the C++ FFI header and tests. |
| mono equals base | 43 | Copy theirs. Includes `core/ir_serialize.py`, `helpers/compile.py`, `runtime/__init__.py`, `runtime/comgr.py`, `rocke/__init__.py`, the example and driver import renames, and the `attention_bwd` and conv library files. |
| 3-way, clean | 18 | `helpers/__init__.py`, `core/backend.py`, `core/isa/backend.py`, `dsl_docs/reference/env_flags.md`, `platform/AGENTS.md`, `CMakeLists.txt`, several C++ files, the golden JSON, `run_portable_ir_gates.py`, `kernel_cache.py`, `reproduce_kernel.py`. |
| 3-way, conflicts | 6 (11 hunks) | `lower_llvm.py` 2, `lower_llvm_internal.h` 3, `conv_direct_grouped.py` 2, `test_direct_kernel_cache.py` 2, `instance_conv_direct_grouped.h` 1, `tests/CMakeLists.txt` 1. |

The conflicts come from the rust commits' COMGR-path address-split work (`_global_gep`, index-range facts) and the
hipTensor conv changes, which depend on upstream code the mono tree lacks. In `lower_llvm.py` one hunk is the mono's
`constant`-space `int_to_ptr` load against theirs' `_global_gep`. **The direct path needs none of them.**

Minimal overlay, about 160 files:

- **Copy:** the crate; `runtime/{code_object,code_object_elf,direct_session}.py`; `runtime/__init__.py`;
  `runtime/comgr.py`; `helpers/compile.py`; `helpers/direct_gemm.py`; `core/ir_serialize.py`; `rocke/__init__.py`;
  the drivers and examples that import from `runtime.code_object`; `tools/isa_*.py`; the direct runtime and tool
  tests; the two `dsl_docs/runtime` pages.
- **Merge (all clean):** `helpers/__init__.py`, `core/backend.py`, `core/isa/backend.py`, `env_flags.md`.
- **Do not overlay:**
  - `lower_llvm.py`: keep the mono's; the direct path never lowers to LLVM;
  - every C++ file and the CMake/CTest files (user rule: no C++ work);
  - the `attention_bwd`, conv and portable-IR golden changes;
  - the GDN files.

Checked: a temp copy of the mono `platform/python` with exactly this Python overlay imports, resolves
`code_object_backend="direct"`, and serializes the toy, B3, fused-MoE and head kernels.

Procedure (when the user approves; still no git commands that change state):

- Copy files with plain `cp` from `git show 23c165f8b3f:<path>` output, or from a `git archive` extraction in `/tmp`.
- For the 3-way files, run `git merge-file -p <mono> <base e7d4664bda9> <tip>` and write the result over the file.
- Track the overlay in a manifest, `experiments/qwen38-true-mono/overlay/rust-direct.manifest`, listing each path
  with its source sha and class. This makes it reversible with `git checkout -- <paths>` by the user and keeps it
  visible in `git status`.
- Normalize line endings: `.gitattributes` and CRLF. The branch's files are LF in the repo, so copy from
  `git show`, not from a Windows checkout.

### 4.3 Building the library on alola

There is no cargo on login or compute nodes. Skill rule 100 is a private rustup on a CPUONLY job, and a toolchain is
already installed at `~/rocke-sdpa-claude/d950-rust/rustup` (Rust 1.99.1, used by
`copy-rocm-lib-agents/parent-out/ct950/stage/build_so.sh`). The crate has no dependencies, so `--locked --offline`
builds without crates.io.

1. Sync the overlaid tree with `cluster_sync.sh rocke rocke-q38direct`. Never scp a CRLF tree.
2. Run a CPUONLY job (`cluster_job.sh --arch cpu --cpus 16 --mem 32`) that runs from node-local `/tmp` and keeps
   `CARGO_TARGET_DIR` out of the synced tree:

   ```bash
   export RUSTUP_HOME=$HOME/rocke-sdpa-claude/d950-rust/rustup
   export PATH=$RUSTUP_HOME/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin:$PATH
   export CARGO_HOME=$JOBTMP/cargo CARGO_TARGET_DIR=$JOBTMP/target
   cargo build --release --locked --offline --manifest-path platform/rust/rocke-codeobj/Cargo.toml --lib
   cargo test  --release --locked --offline --manifest-path platform/rust/rocke-codeobj/Cargo.toml --lib -q
   cp $CARGO_TARGET_DIR/release/librocke_codeobj_ffi.so $OUT/so/ && sha256sum ...
   ```

   If that toolchain is missing, use the rule-100 bootstrap
   (`curl https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path`).
3. In GPU jobs (gfx950 `rocke` partition or MARKHAM), set `ROCKE_CODEOBJ_DIRECT_LIB=$OUT/so/librocke_codeobj_ffi.so`.
   `rocke.runtime.code_object` loads it with ctypes and checks `rocke_codeobj_ffi_abi_version`. When the variable is
   unset it searches the platform library path for `rocke_codeobj_ffi`, and a failed search is not cached.
4. Select the backend with `code_object_backend="direct"` per call, or `ROCKE_CODEOBJ_BACKEND=direct` per process.
   `artifact.code_object_backend` and `artifact.direct_fallback` record what happened. `compiler_identity("direct")`
   includes the library's path, size and mtime, so cache keys change when the library is rebuilt.
5. Gate the Linux `.so` against the Windows DLL by byte identity of a small corpus (`bench950 <dir>`), as the
   compile-time campaign did.

### 4.4 Staged recommendation

| When | Step | Gate |
|---|---|---|
| Now (M0/M1, beside the layer port, never blocking it) | Apply the minimal overlay. Add the fallback hygiene: op names plus UNSUPPORTED instead of PARSE or MALFORMED for every Q38 op and asm arity. Add a framework knob, `FrameworkKnobs.code_object_backend`, default `"comgr"`, passed to `compile_kernel`. It changes no IR, so the off state is byte-identical, and it is recorded in the artifact and bench provenance, not in the kernel name. A build that falls back must say so: a silent fallback is a test failure, as with `backend="python"`. | `cargo test`, `isa_tables check` (both ISAs), the direct runtime pytest, and the targeted Q38 tests with direct=on: they must all fall back cleanly. |
| M1-M2 | First direct targets: (a) `mx_decode_gemv`, which needs `buffer_load_cp`, `cvt_scalef32_pk_bf16_fp4`, `permlane16_swap`, and `tagged_poll` and the scoped atomic for the fused argmax, about 6-8 d; (b) the toy chain with `mb_impl=ir`, which needs `tagged_poll`, `buffer_store_cp`, `int_to_ptr`, `s_getreg`, `s_sleep`, about 5 d. They exercise the mailbox and poll path at hop scale and give the first direct-versus-COMGR hop ratio. | Bitwise against the COMGR object (same IR, inputs, launch) and against the mono head; `hazard_corpus` on the COMGR disassembly; `target_compile --max-vgprs`, `verifyspill`, `ROCKE_DIRECT_THREADS=1` identity; exec-join ISA lint; shuffled paired A/B against COMGR with a byte-copy null. |
| M2-M3 | B3 (`math.exp`, DPP ops, cpol `state_nt`) and fused MoE (`mfma_scale_mx` 16x16x128 with opsel, `ds_or`, `ff1`, `v_maximum3`, MXFP8 converts). Then branch relaxation before the first 48-layer M3 kernel, and the line map before M5 needs blame. | Per-stage bitwise rig gates, as for any lever. |
| M5 (RSI) | Direct as a lever family: poll-first counted vmcnt, SMEM poll, per-op cpol, SGPR pinning, kernarg preload, waits per op. Each is a default-off knob kept even when it loses, hill-climbed with lookback. Compile speed makes exhaustive combination sweeps practical. | As above, plus WaveScope captures from the line map. |

Stop rule: if direct loses to COMGR on the toy chain and the head after the levers (ratio above about 1.00 with the
nulls tight), keep it as an off knob and do not port B3 and MoE. The Q38 gain has to come from sync control, not
from general code quality, where the two compilers are at parity.

## 5. Open questions for the user

1. Take g950p's crate (`copy-rocm-lib-agents/g950p` tree, `ranges.rs`, `lds_rebase.rs`, `lds_reuse.rs`) instead of
   the branch tip? It is newer and better on attention, but it is not on any branch.
2. Make `mb_impl=ir`, `cvt=intrinsic` and `int_to_ptr` pointers mandatory for direct builds, so that none of the asm
   idioms get ported?
3. Is 4-6 weeks of compiler work in scope beside the stage ports (G1-G8 in DESIGN.md), or only the head and toy
   pilot first?
