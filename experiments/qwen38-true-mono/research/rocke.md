# rocKE survey for the true monokernel (condensed; 2026-10-29)

Tree: `copy-rocm-lib/dnn-providers/hip-kernel-provider/rocke` at 89209ac7665 ("Q38 monokernel work"; R below;
P = R/platform/python/rocke, L = R/library). `dev/rocm-libraries` is on another branch and lacks this work.

## Pipeline
- spec -> `build_*(spec, arch)` -> `KernelDef` (P/core/ir.py `IRBuilder`) -> LLVM `.ll` (P/core/lower_llvm.py,
  flavors llvm20/22/23) -> comgr HSACO (P/helpers/compile.py, -O3) -> `hipModuleLaunchKernel` (P/runtime/launcher.py
  `KernelLauncher`, `time_launches`). Imports: library may import platform, never reverse.
- Engines: C++ (`rocke_engine`) and Python must be byte-identical, EXCEPT builds using `ir.PYTHON_ONLY_OPCODES`
  (mailbox `ir`, mfma_scale_mx opsel, int_to_ptr, tagged_poll, inreg kernarg preload, f64 sqrt): force
  `backend="python"`, distinct name suffix, excluded from the gate. User rule: skip C++ work; Python-only OK.
- Env: `ROCKE_BACKEND=python`, `ROCKE_LLVM_FLAVOR` (q38 ROCm10 container = llvm23; venv torch2.12 = llvm22),
  `ROCKE_COMGR_LIB`, `AMD_COMGR_CACHE_DIR` node-local, `ROCKE_DEBUG_LOC=1` for WaveScope.
- Composition: everything inlines into one IRBuilder. `L/kernels/gfx950/hiptensor_mono/inline.py::inline_kernel_body`
  copies a tuned KernelDef body into another builder (renames values/LDS, binds params, block_id -> task coords).
  `qwen38_mono/device_function.py` lowers a KernelDef as an LLVM device function (alwaysinline + convergent default).

## Tuning / search
- `P/dispatch/tuning/axes.py::KnobAxis`, `space.py::KnobSpace` (stable tuning_id), `walk.py`, `testing.py`.
- `P/dispatch/tuning/search/`: DE/JDE, Hyperband, TPE, GP-EI, HillClimb, Thompson Portfolio, LightGBM priors,
  racing/noise models, TuningDB, checkpoint/resume. `P/dispatch/gemm/selector.py` LightGBM+Origami selector.
- Runbook: `R/platform/dsl_docs/optimization/optimization_runbook.md` (Step 0 exhaustive lever list, The Loop: one
  lever at a time with ISA diff, lever catalogue A..Q). gfx950 notes: `.../optimization/arch/gfx950.md`.
- WaveScope: `R/platform/dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_trace.py`.

## Q38 pieces present (L/kernels/gfx950/qwen38_mono/, README.md is the index)
- `framework.py` (T9): `build_layer_kernel(StageSpec(stage, emit(b, ctx, p, t), lds, prologue))` along a
  `TaskTable` -> one KernelDef with ATOM's ABI (`step_ptr`, `layer`, 41 x i64 args, kernarg 16+8k), grid 256x512,
  LDS arena (`P/helpers/lds_arena.py`), phase barriers w/o vmcnt drain, mailbox helpers on `LayerCtx`, stamps,
  multi-layer launches (`layers=N`, parity slots), `FrameworkKnobs` (kernarg_preload, step_first, stamps).
  Only measured on `toy_chain.py` (best_spec 0.92-0.93 hop-only / ~1.01 with weights vs FlyDSL twin; stacked vs
  stacked 0.95-0.99). No real-layer numeric A/B yet (T11 next).
- `atom_layout.py`: ATOM mono schedule as data: ORDERS per kind (delta: a1 a2 b1 b3 c1 m1..m5; qsa: a1 a2 q1..q7
  c1 m1..m5; delta_ple: p1 p2 + delta), TASKS, BASE (CTA offsets), REGIONS/SC scratch, dataflow `_FLOW`,
  `atom_table`, `atom_stack` (cross-layer WAR validation).
- Stages: `stage_b3.py` (GDN B3, bitwise mono, tuned/NUMA knobs, rows<=4 + r4g2), `moe.py` (M3/M4/M5 split,
  bitwise), `moe_fused.py` (expert-major fused M3+M4+M5, NUMA, rows<=4), `stage_q2.py` (QSA Q2), `stage_adapt.py`
  (wrap hybrid stage bodies as framework stages), `stage_fn.py`, `stamps.py`, `mailbox_ir.py`, `abi.py`, `mono_layout.py`.
- Helpers: `P/helpers/mailbox.py` (tagged u32 payload+tag, tag = step*64+layer+1; put/put2 sc1; poll; NUMA copies),
  `device_ops.py` (s_sleep, memrealtime, hw_id, buffer_wbl2/inv, scoped atomics, ptr_from_i64), `task_schedule.py`,
  `lds_arena.py`, `mx_quant.py` (MXFP8 activation quant).
- `L/kernels/gfx950/mx_decode_gemv.py`: MXFP4 decode GEMV (bf16 / mxfp8 / bf16q8 activations), persistent split,
  epilogues logits / keys / fused argmax (last-arriver tagged slots). Served as the lm_head.
- `L/kernels/gfx950/mx_mfma.py` + `P/core/arch/mfma_scale_mx.py`: scaled MFMA 16x16x128 / 32x32x64 any f8f6f4 format
  pair with opsel; converts `cvt_scalef32_pk_fp8_f32`, `cvt_scalef32_pk_bf16_fp4`; `dpp_row_mirror`, `permlane16_swap`.
- `L/kernels/gfx950/gdn_decode.py`: production GDN/KDA decode (dispatch `L/dispatch/gdn/`, knob registry axes/rules).
- Example `P/examples/gfx950/persistent_mailbox/` (hop RTT microbench + case study).

## Other relevant current-tree kernels
- bf16 GEMV: `P/instances/common/gemm_skinny.py` (split-K warp/LDS/grid, last-arriver). Norms: rmsnorm2d etc.
- Router top-k: `topk_softmax.py` (K<=32). `moe_sorting.py`. `moe_fused_mega_fp8.py` (E=8 only). No W4A4 (no fp4
  activation quant convert). Attention decode: `attention_unified.py`, split-KV, `lightning_indexer.py` (scores
  only); no large-k top-k / gather sparse decode (QSA Q3-Q7 unported). Sampling: greedy argmax only.
- Persistent prior art: `L/kernels/gfx950/hiptensor_mono/task_queue.py` (ticket DAG queue, static/phase/dynamic),
  `tensor_network_mono/circuit_kernel.py:400-466` (grid barriers full / last / xcd with epochs), `P/helpers/persistent.py`.
- Sync ops: fence(scope), global_flag_store/wait_eq, global_atomic_add (i32/f32), LDS atomics, s_barrier_bare,
  sync_lds_only, s_setprio, sched_group_barrier, DPP/swizzle/bpermute/permlane, readfirstlane, inline_asm.
  No CAS/xchg/min/max/i64 atomics (inline_asm workaround). No multi-GPU/IPC.

## Gaps vs a whole-pass rocKE monokernel
- Stage emitters missing: HC A1/A2/M1/M2, B1 (+conv), C1, Q1, Q3-Q7 (indexer scorer, radix top-512, gather sparse
  attention), P1/P2, head-in-kernel, draft front/meta, sampling/verify-accept.
- Rows > 4 (grouped r8g4/r12g4/r16g4, wide) for MoE and B3.
- Reusable striped grid sync (FlyTape-style) helper; i64/CAS atomics; ISA lint for exec-join spills (FlyTape
  `scripts/lint_isa.py`).
- Code size / compile time of a whole-pass kernel: emit one body per layer kind and loop (FlyTape traces one copy
  per run), not 48 inlined layers.
