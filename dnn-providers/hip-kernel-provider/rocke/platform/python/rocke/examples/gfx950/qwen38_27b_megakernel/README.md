<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Qwen3.8-27B decode megakernel in rocKE (gfx950 / MI355X) — work in progress

This folder holds a rocKE port of the FlyDSL decode megakernel that AMD published for Qwen3.8-27B. It also holds
**A1**, the first speed-up on top of the port. A1 is a full forward pass (64 layers, 8 token rows) in one
persistent launch. It matches the FlyDSL kernel's results and is 0.8–3.3 % faster than it.

This is a work-in-progress branch for the team to read, reproduce and optimize. It is not a library kernel.

| File | What it is |
|---|---|
| `a1_kernel.py` | The kernel: one rocKE IR builder, `build_full(spec)`. The spec `p2c_full_a1` is A1. |
| `build_a1.py` | CPU-only build: rocKE IR → LLVM IR → link OCML → comgr `-O3` → `kernel.hsaco`. |
| `bench_a1.py` | GPU timing: A1 against ATOM's FlyDSL kernel, in one process, on identical inputs. |
| `KNOWN_ISSUES.md` | Three compiler/ISA traps we hit while porting. |

## 1. What the kernel does

Background reading:

- AMD's blog [Bringing Ultra-Fast, Interactive LLM Inference to AMD Instinct MI355X GPUs](https://rocm.blogs.amd.com/software-tools-optimization/megakernel-intro/README.html)
  (2026-10-20). It describes the megakernel, speculative decoding with a 7-token drafter, MXFP4 weights and the
  results.
- The FlyDSL source we ported is in [proaditya/ATOM `qwen38-megakernels`](https://github.com/proaditya/ATOM/tree/qwen38-megakernels)
  at `atom/models/qwen38_27b/mono/kernels/`. The kernel is `verify_m8.py` plus the `gen_replay_records.py`
  transform (kernel name `qwen_m8_replay`). We used commit `352229d6`. The same kernels, with SGLang and vLLM
  plugins, are in [proaditya/aiter `aitermk-qwen38`](https://github.com/proaditya/aiter/tree/aitermk-qwen38/aitermk).
- The DSL itself is [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL).

**One launch = one verify pass of the target model.** Speculative decoding drafts 7 tokens. The target model then
checks the last accepted token plus the 7 drafts, so 8 rows, in one pass. The launch shape is:

- grid 256 workgroups × 256 threads, one workgroup per CU, all resident at once;
- one kernel argument: the address of a pointer table (65 rows × 45 slots of 64-bit pointers);
- LDS 159,792 B per workgroup, no scratch;
- output: fp32 logits `[8 × 248,320]`.

**The model** has 64 layers in a repeating D/D/D/A pattern: 48 Gated DeltaNet (GDN) layers, 16 gated-softmax
attention layers, and an MLP in every layer. Hidden size 5120, MLP width 17,408. Every GEMM uses the scaled MFMA
`v_mfma_scale_f32_16x16x128_f8f6f4`, with MXFP4 weights × MXFP8 activations.

**Stages in one layer.** Each workgroup has a fixed role, decided by its block index:

1. **Fold + RMSNorm.** Sum the previous layer's 3 fp32 Down partials plus the residual, normalize, quantize to MXFP8.
2. **GDN layer.** QKV GEMM + causal conv1d + the beta/alpha (BA) projections. Then the delta-rule state update.
   Blocks 0–191 run the GDN core. Blocks 192–255 run the core tail and prefetch the next Up weights into L2. Then a
   gated norm with z, quantized to MXFP8.
3. **Attention layer.** Q/gate/K/V GEMM. Then Q/K norm, partial RoPE and the new K/V rows. Then split-context
   partials (flash-decoding style, running max and denominator). Then the split reduce × sigmoid gate.
4. **Output projection.** An atomic bf16 "mixer" sum, then the post-attention norm owners.
5. **MLP.** Up (gate+up GEMM, `SiLU(gate)·up`). Then Down (3 fp32 partial planes).
6. **Layer 63 only.** Final fold and RMSNorm, then the LM head streamed to fp32 logits.

**Synchronization.** Stages hand data to each other through memory, using point-to-point counter atomics.
- One workgroup per XCD copies global counters into XCD-local release words.
- Full grid barriers appear only around the attention partials and the final fold.
- Every poll is bounded. A timeout sets the failure word `native[289]`, and the kernel exits instead of hanging.

## 2. How it was ported

The port is a **faithful, op-by-op transliteration**. Each FlyDSL stage became rocKE IR with the same:
- memory operations and cache bits;
- wait counts, barriers and atomics;
- float operations in the same order;
- MFMA sequence.

Keeping the float order exact matters for a later step: a correct port can then be compared bit for bit against
FlyDSL, once the atomic summation order is pinned (see §3).

The order of work:

1. **Stages in isolation.** Each stage was built alone, checked against its own oracle, and compared with
   FlyDSL's LLVM IR for that stage.
2. **Composition.** The stages were joined into one builder. Then every isolation knob was removed, leaving one
   kernel with FlyDSL's ABI. The builder still has two diagnostic "twin" knobs, `pin` and `mxpin` (§3).
3. **rocKE extensions.** rocKE has no generic intrinsic op. So `a1_kernel.py` adds two IR ops: a generic LLVM
   intrinsic call and a raw LLVM instruction. They are lowered by rocKE's python backend, which is why
   `build_a1.py` uses `backend="python"`. FlyDSL's `pow`/`cos`/`sin`/`exp2` are OCML calls, so `build_a1.py`
   links `ocml.bc`, the way FlyDSL's serializer does.
4. **Two deviations from the reference kernel**, both about memory ordering:
   - the residual1 store is write-through (`sc0 sc1`);
   - the mixer zeroing stores are write-through (`global_store_short … sc0 sc1`).

   Our comparison arm, called "FlyDSL reference" below, is ATOM's FlyDSL kernel with the same two store changes.
   So rocKE-vs-FlyDSL comparisons are like for like.
5. **Compiler workarounds.** Every wave-uniform decision is an SGPR value (`readfirstlane`). The lane-0 counter
   publications narrow and restore EXEC inside one asm block. Without this rule, the compiler produced wrong code;
   see `KNOWN_ISSUES.md` #1.
6. **Three exact-order speed-ups kept from the port, none changing float order:**
   - Down's activation loaded with one full-lane x4 load + `row_ror:8` DPP;
   - GDN QKV epilogue operands issued before the fold wait;
   - fold polls that read the counter and the failure word under one `vmcnt(0)`.

## 3. How parity was checked (method)

All gates were frozen before the data they judge was collected.

- **Stage oracles.** Each stage's output is checked against an interval oracle derived from its inputs.
  - Example: a dot product passes if its error is within `7.85e-4 × Σ|a·w|`.
  - "Tight" outliers are counted against frozen per-layer limits.
  - Quantization, padding and copies must be exact.
  - GDN replay records and state must follow the delta rule.
  - Attention partials must hold exact sentinel values when empty.
- **W4: full-kernel stage check.** The assembled kernel runs real prompts at contexts 1K, 8K and 32K. Every stage of
  every layer is dumped and checked against the stage oracles. That is 27,414 checked units per arm.
  - The check includes speculative accept lengths k = 1…8.
  - Each k is committed through ATOM's real state-commit path, and the following round is checked too.
  - FlyDSL runs through the same harness as a control and must pass the same checks.
- **G2: whole-model agreement.** A noise band was frozen first, from FlyDSL against itself, across processes. Then:
  - **W1:** rocKE's own launch-to-launch spread must stay inside the band;
  - **W2:** rocKE-vs-FlyDSL differences (logits, flip rate, last-layer error) must stay inside the band;
  - **W3:** over 24 windows of 256 greedy steps, rocKE's first token difference from FlyDSL must not come earlier
    than FlyDSL's own self-differences (one-sided Mann–Whitney, p ≥ 0.05).
- **Pinned twins.** The kernel is not bit-reproducible, because the atomic summation order changes from launch to
  launch. So each timed kernel has a twin that differs only in a pinned atomic order: the bf16 mixer and the fp32
  BA split-K sum. The twins of rocKE and FlyDSL must repeat bit for bit and match each other on every consumed
  output. They are never timed.
- **Six-cell timing.** Cells are contexts 1K/8K/32K × modes cooperative/graph.
  - Three arms run in the same job: FlyDSL reference / previous rocKE / candidate.
  - ≥ 200 interleaved launches per arm per process, and ≥ 3 fresh processes.
  - Identical-arm (A/A) displacement must stay ≤ 0.25 %.
  - Parity means rocKE/FlyDSL ≤ 1.01 in all six cells.
  - Adopting a speed-up needs > 1 % gain in a pre-registered target cell, with no cell worse than −1 %.

Result for the port before A1: parity met, rocKE/FlyDSL reference = 0.989–0.991 in all six cells.

## 4. The A1 result

**Problem.** In the attention partials loop, the compiler's `vmcnt` waits that guard the pre-loop Q loads also
counted the 16 K/V prefetch loads just issued for the next tile. The compiler cannot see that those are
prefetches, because they are inline-asm DMAs. So every pass waited for its own prefetch.

**Fix.** A1 adds one compiler-visible `s_waitcnt vmcnt(0)` right after the Q loads (knob `a1`). LLVM then drops
the 8 in-loop Q guards. Float values and order are unchanged.

**Trace check.** An instruction-level thread trace at 32K confirmed the mechanism: with A1, the long attention
splits finish sooner, and the in-loop stall on the K/V prefetch almost disappears.

**Full gates on the A1 object.**
- W4: 27,414 / 27,414.
- G2: pass.
- Six-cell timing: 0 pinned mismatches; identical-arm (A/A) noise within the 0.25 % limit.

| cell | A1 gain vs the port | A1 / FlyDSL reference |
|---|---|---|
| 1K cooperative | +0.00 % | 0.991 |
| 1K graph | +0.04 % | 0.992 |
| 8K cooperative | +0.69 % | 0.984 |
| 8K graph | +0.71 % | 0.985 |
| 32K cooperative | **+2.33 %** | 0.967 |
| 32K graph | +2.35 % | 0.967 |

The gain grows with context, because only the attention partials loop changed.

## 5. Tried and parked

Each line below is one idea, described in words only. None of these changes is in this branch; "built" means built
as an off-by-default switch on the port, outside this branch.

- **SB** (one-round-trip polls + S1 + S2 together): passed W4, but was slower at 1K. Rejected.
- **FB** (batch the fold's first chunk/row-sum poll reads): a synthetic overlap test predicted less than the 1 % bar,
  so parked; never timed as a full kernel.
- **A-SKIP** (skip the O/l rescale when the running max is unchanged): too few critical 32K waves could skip to pass
  the pre-set bar. Parked.
- **S1:** prime Up's first three weight steps from waves 1–3 during the output stage. Built, not screened.
- **S2:** hoist Up's in-loop scale loads, removing compiler over-waits. Built, not screened.
- **S3:** drop a redundant pre-loop failure-word read at three poll sites. Possibly no gain.
- **S4 / G-FWD:** GDN finalize reads the core tile from LDS instead of global memory. Expected gain near zero.
- **A-PV32:** pair the PV MFMA blocks. Cannot keep the exact float order, so excluded.
- **G-VEC:** wider GDN state access. Changes the reduction order, so excluded.
- **G-BCAST:** remove the delta broadcasts but keep the sum.
- **G-NT:** streaming hints on GDN state reads.
- **Hand-off placement:** put signal words on the reader's die. A single reader gained a little under load
  (marginal). Many readers on both dies: untested.
- **Profiler basics:** LDS bank conflicts and request sizes look healthy. The clearest exposed stall is the Up
  compiler wait (S2).

### Next: screen, then climb

A plan to search combinations of the parked switches, all relative to A1:

1. **Freeze.** Build each exact-order switch, off by default, and prove it is byte-identical to A1 when off.
   Candidates: S2, PV operand LDS batching, independent layer-head pointer loads, weight non-temporal hints, S1,
   S3, S4, G-NT, G-BCAST, Up epilogue bank layout.
2. **Screen.** Test each switch alone against A1 in the same job: W4 at 8K first, then timing at 1K/8K/32K
   cooperative. Keep a switch only if its gain beats noise in every process. The noise margin is
   `max(0.10 %, 2 × A/A)`, with a Bonferroni-corrected test across all switches.
3. **Rescue.** Test only pairs listed before any data, with a mechanism behind them (today: S1+S2).
4. **Climb.** With ≤ 4 survivors, try every subset. With more, climb greedily from A1, comparing each step with
   the current best in the same job.
5. **Qualify.** The final combination must pass leave-one-out (removing any member must not help). Then the full
   suite on its exact hash: W4 1K/8K/32K → G2 → six-cell timing, with > 1 % in a target cell.

## 6. Reproduce: build → run → time

**Requirements:**
- one MI355X (gfx950);
- the ATOM Qwen3.8-27B megakernel image, built from `blog_benchmark/Dockerfile` in
  [proaditya/ATOM `qwen38-megakernels`](https://github.com/proaditya/ATOM/tree/qwen38-megakernels) (see its
  `blog_benchmark/README.md`);
- the Qwen3.8-27B checkpoint that the ATOM blog benchmark uses;
- a folder of `*.jsonl` prompt files, each row holding a `"prompt"` key.

**Build (CPU only).** Any machine with Docker works. `$ROCKE` is this rocKE tree
(`dnn-providers/hip-kernel-provider/rocke`). `$ATOM_IMAGE` is the image above.

```bash
docker run --rm --network=none -v "$ROCKE":/rocke:ro -v "$PWD/a1_build":/out "$ATOM_IMAGE" bash -c \
  'cd /tmp && PYTHONPATH=/rocke/platform/python /opt/venv/bin/python \
     /rocke/platform/python/rocke/examples/gfx950/qwen38_27b_megakernel/build_a1.py /out'
```

The output prints `sha256`, `kernel_ll_sha256` and two `matches_reference_*` flags. With the image's ROCm 10.0
toolchain the expected values are:

- LLVM IR `612296e3afb932f64f20628636b06034643b97fb8327d0f17843aefc3dd66257`;
- code object `1f710cd33adee782cb2abd7304f4e6d778f7f36d8e1ac896df00d70dc9dd4a9e` (114,016 B; 316 VGPRs incl.
  60 AGPRs, 106 SGPRs, LDS 159,792 B, no scratch).

A different compiler or `ocml.bc` gives a different code object. Matching LLVM IR still means the rocKE side is
identical. Outside the image, set `LLVM_BIN` (and `OCML_BC`, `ROCKE_COMGR_LIB` if needed); see `build_a1.py`.

**Run and time (GPU).**

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v "$ROCKE":/rocke:ro -v "$PWD/a1_build":/a1:ro -v "$MODEL_DIR":/model:ro -v "$PROMPTS_DIR":/prompts:ro \
  -v "$PWD/results":/results "$ATOM_IMAGE" bash -c \
  'for p in 1 2 3; do MODEL_DIR=/model PROMPTS_DIR=/prompts A1_OBJECT_DIR=/a1 RESULTS_DIR=/results \
     /opt/venv/bin/python /rocke/platform/python/rocke/examples/gfx950/qwen38_27b_megakernel/bench_a1.py \
     --process $p; done'
```

Each process prints one line per context with the cooperative and graph medians for both arms and the ratio
`a1_over_flydsl`. It writes `results/timing-p<k>.json`.

Notes:
- The FlyDSL arm here is ATOM's **shipped** build, not the FlyDSL reference of §2, step 4. So expect ratios
  near, but not equal to, the table in §4.
- `bench_a1.py` checks only the failure word, finiteness, the top token and drift against the first launch. It is
  a timing tool, not the W4/G2 correctness gates of §3.
