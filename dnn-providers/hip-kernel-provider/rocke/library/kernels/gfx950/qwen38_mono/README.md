# qwen38_mono (gfx950)

Python-authored family for the Qwen3.8 per-layer decode monokernel. So far it
holds the pieces of the persistent-kernel mailbox layer that need
Python-engine-only opcodes; the layer kernels themselves are not here yet.

| module | what |
|---|---|
| `mailbox_ir.py` | the `ir` mailbox implementation (`buffer_store_cp` puts, `tile.tagged_poll` polls, `int_to_ptr` runtime pointers). Importing it registers `MailboxKnobs(impl="ir")` with `rocke.helpers.mailbox`. |
| `device_function.py` | hybrid path: lower a `KernelDef` as an LLVM device function (`define void`, C ABI: pointers as `i64`, LDS as a module global or an `lds_base` argument, inline knob), written as `.ll` / `.bc` / `.json`, for linking into the FlyDSL mono kernel (`flydsl.compiler.extern_link`). Python engine only. |
| `mono_layout.py` | the ATOM mono layout constants a stage function is built against (`from_atom`: read-only import of ATOM's `layout` / `config` / `qsa`). |
| `stage_fn.py` | the stage-function ABI, build and generator CLI (`python -m kernels.gfx950.qwen38_mono.stage_fn --out DIR`). ATOM's `ATOM_Q38_EXTERN_STAGES` links the output in place of a stage body. `StageKnobs`: `xshfl` (`dpp2`, the mono's VALU-only form, default / `dpp` / `lds`), `tbl` (`smem`, the mono's scalar loads of the per-step QSA table through a constant-space pointer, default / `vmem`), `mailbox_impl`, `fn` (device-function knobs; default `lds="arg"`, `inline="always"`: the mono must inline a stage). |
| `stage_q2.py` | Q2 of the QSA layer as a stage function, bit-identical to the mono's (rows 1-4, plain and seq builds). |
| `stage_b3.py` | B3 of the DeltaNet layer (gated delta rule decode: b/a dots, gates, q/k norms, f32 VK recurrence with snapshots, sibling core exchange, gated RMSNorm, MXFP8 pack) as ONE emitter for the isolated kernel (`build_b3_kernel`), the hybrid function (`lower_b3_function`, registry `b3`, prologue `b3pre` for `pre="dma"`) and the layer framework (`b3_stage_spec`). `B3Knobs()` = the mono's B3, bitwise; `tuned_knobs` / `TUNED_SPECS` = the in-situ climb's best per variant (bitwise too). CLI: `python -m kernels.gfx950.qwen38_mono.stage_b3 --out DIR --knobs tuned`, then `ATOM_Q38_EXTERN_STAGES=b3=DIR,b3pre=DIR ATOM_Q38_EXTERN_MISSING=mono`; it emits only `VERIFIED_VARIANTS` (the narrow ones and `r4g2`; others need `--unverified`). Sidecars carry `stage_knobs` and, on the body, `needs` (`["b3pre"]` for `pre="dma"`): ATOM refuses such a body without its prologue and drops both to FlyDSL when the prologue file is missing under `MISSING=mono`. |
| `framework.py` | the rocKE persistent layer-kernel framework (roadmap T9): `build_layer_kernel` composes stage emitters `StageSpec(stage, emit(b, ctx, p, t), lds, prologue)` along a `rocke.helpers.task_schedule.TaskTable` into one `KernelDef` with ATOM's ABI (`step_ptr`, `layer`, 41 i64 args: kernarg offset `16 + 8 k`), one LDS arena (`rocke.helpers.lds_arena`) with phase barriers that do not drain `vmcnt`, mailbox helpers on `LayerCtx`, harness-format stage stamps, and multi-layer launches (`layers=N`, parity scratch slots). `StageSpec(per_cta=True)` runs a self-guarded emitter once per CTA with its task index (the stage-function ABI); `FrameworkKnobs`: `step_first` (step load before the `entry` hook's loads), `step_load` (`smem`: a losing lever), `kernarg_preload=N` (`inreg` parameters, Python engine only: `ir.PYTHON_ONLY_PARAM_ATTRS`), `stamps_lds` / `lds_touch` (LDS residency of builds whose stages allocate their own LDS). `stage_kernel` wraps one stage as a standalone kernel (isolated A/B, or a device function for the hybrid path). Mirrored ops only. |
| `stamps.py` | per-CTA stage stamps in the ATOM `ATOM_Q38_STAMPS` format (the harness timeline decoder reads rocKE and FlyDSL layer kernels alike). Multi-layer launches: each layer's points are flushed under its layer id; point 0 of a later layer is that CTA's previous flush end (not a launch entry), so decode later layers from the previous layer's end (`toy_chain_bench.decode_launch` / `rebase_stamps`; the rebased points also feed the harness decoder). The framework barriers between a layer's flush and the next layer's stamp reset. |
| `atom_layout.py` | ATOM's mono schedule (`TASKS`, `BASE`, `ORDERS`) and scratch layout (`REGIONS`, `SC`) as rocKE data, plus each stage's dataflow, so `TaskTable.validate` / `validate_stack` check the real schedules (cross-layer write-after-read of a multi-layer launch). |
| `stage_adapt.py` | path b's plug for the tracks' stage bodies: `stage_fn_factory(name, layout)` wraps a registered hybrid stage-function body (`stage_fn.register_stage`: q2, b3, m3..m5), unchanged, as a framework `StageSpec(per_cta=True)` (every CTA calls it once with its round-0 task, the ABI's `task`; tag / step / layer / the 41 args / QSA `l0..l3` (`qsa_lens`, ATOM `qsa_len`) from the framework). `register_stage_fns` registers them as `build_layer`'s default emitters; `adapted_knobs` drops the framework arena's one-CTA-per-CU padding (the bodies allocate their own LDS); self-placed MoE bodies refuse non-ATOM placements; a stage module that fails to import is reported (`RuntimeWarning`, `import_errors()`, quoted in the factory's `KeyError`), not dropped silently. A q2 build lowers `llvm.sqrt.f64` (Python engine only, added to the kernel lowering). |
| `moe.py` | the MoE stages M3 (router), M4 (routed + shared gate_up), M5 (down + combine + HC write) as inline emitters over a `MoeCtx`, bitwise the mono's for every knob (`MoeKnobs`: expert-union dedupe with a re-sliced expert-major map, codegen / schedule levers, routing and placement levers; defaults = the mono's schedule, byte-identical IR when off); `build_moe_suffix` is the standalone split-launch suffix (`m3 m4 m5` or `m4 m5`) with the mono kernel's signature. Python engine only (`_pyonly`). Harness and lever ledger: `hackathon-prep/qwen38/moe/`, `hackathon-prep/reports/qwen38-rocke/ledger/moe.md`. |
| `stage_moe.py` | M3 / M4 / M5 as hybrid stage functions (`stage_fn` ABI, `lds="arg"`, `alwaysinline` by default), generator CLI with `--knobs`; `--knobs best` writes the serving set `BEST` per layer kind (`<stage>_<kind>_<variant>`, only the cells that beat the mono; ATOM resolves `{kind}`, `ATOM_Q38_EXTERN_MISSING=mono` keeps the rest); `m4` only together with `m5` (sidecar `requires` / `pair`, enforced by ATOM); refuses a knob set whose LDS arena exceeds the mono kernel's, and refuses to run without that check unless told the arena. |
| `moe_fused.py` | the expert-major FUSED MoE (M3 + M4 + M5 in one body, design `hackathon-prep/reports/qwen38-rocke/MOE_FUSED_DESIGN.md`): a shared group streams the shared expert during the router window, the routed CTAs stream one byte line each (gate_up units, then down units of the same experts; a line per IOD), the gate_up results go through the mono's own `dn` mailbox, the down results through a private `ys` mailbox in the dead `ss` region under the private tag `tag ^ 2^31` (never the launch tag), rank-ordered combine on CTAs 0..159; knob `numa=1` addresses the mailboxes of ATOM's `numa=1` schedule lever (IOD-local polls of dual copies; `lg_copy` / `xchg_home` / `ys_home` `numa2` put the router logits / exchange / down outputs in the NUMA area too); bitwise the mono's (`FusedKnobs`, `FusedPlan` line arithmetic with a host model, `build_fused_suffix` standalone). Python engine only. Harness and ledger: `hackathon-prep/qwen38/moe-fused/`, `ledger/fused-moe.md`. |
| `stage_moe_fused.py` | the fused MoE as hybrid stage functions: `m3` = the fused body (self-placed from the block id), `m4` / `m5` empty; the three sidecars require each other (one `pair` id), so ATOM refuses a partial link; LDS arena checked against the mono's. Sidecars of `numa=1` bodies say `"mailboxes": "numa1"` (ATOM keeps its numa=1 lever for the build and refuses it without), bodies using the NUMA area name its header (`"numa_hdr"`). Generator: `python -m kernels.gfx950.qwen38_mono.stage_moe_fused --out DIR --variants ... [--knobs best (default) / JSON]`. |
| `direct_pilot.py` | Rust direct code-object compiler pilot (true-mono R1): `representative_builds()` (toy chain, B3, fused MoE, MoE suffix, MXFP4 head, scaled-MFMA probe) and `neutralize_census` (every construct past the first that the direct compiler refuses). Framework knob `FrameworkKnobs.code_object_backend` (`""` default = process default, `"comgr"`, `"direct"`): marks the built `KernelDef` (`rocke.helpers.compile.request_code_object_backend`), changes no IR or name; a build the direct compiler does not lower is refused (`ROCKE_DIRECT_UNSUPPORTED`) and COMGR compiles it, recorded in `KernelArtifact.direct_fallback`. Tests: `library/tests/test_qwen38_direct_pilot.py`. |
| `toy_chain.py` | the framework's toy chain (GEMV levels with a reduction between them, exact int32 math): hop modes `a2a` (mono style) / `reduce_task` / `last_arriver`, consume `wave0` / `waves`, reduce `swizzle` / `dpp`, 2-layer stack (seeds from `h_in[4 (layers - 1)]`), idle-CTA L2 prefetch (asm loads drained in batches: their destinations stay allocated), `early=entry` (g0 weights issued at the kernel entry; with framework `step_first` the step load goes first), placement. `best_spec(depth, work, layers)` = the T9 gate's best knobs per shape. LDS exchange words alternate copies along each CTA's stage order (`_pb`: per-kind stage counter; `_parity_guards` adds a CTA barrier where a placement cannot alternate). Bench and case study: `library/benchmarks/gfx950/qwen38_mono/` (`toy_chain_case_study.md`). |
| `persist/` | the true-mono persistent-framework platform (WP1 of `experiments/qwen38-true-mono/D0.md`): `knobs` (`PersistKnobs`: P6's R1-R5, `inline`, `args`, D0.8's sync defaults; `EdgeKnobs` / `EdgePlan`: per-edge protocol, NUMA polarity, poll implementation, placement and queue; `RustScope`: per-stage direct-backend lever sets), `sync` (P4's primitives: striped grid sync with abort-on-timeout, per-XCD-start residency ticket, deadline polls, per-CTA arrivals, the launch-fatal error word), `kv` (P5's in-launch coherence protocols with the markers of the static check), `lint` (static checks: polls that never reload, in-launch KV acquire, fused staging over a polled mailbox, the spill / scratch gate), `isa_gate` + `isa_dist` (WP0a's pooled ISA gate for rocKE builds), `driver` (the persistent layer loop: R1-R5), `parity` (the P4 / P5 probe kernels on these primitives: LLVM IR identical), `micro` / `ptoy` (hardware-gate kernels). The framework takes the knobs through `FrameworkKnobs(persist=, edges=, err_arg=, rust_scope=)`, all off by default (existing builds byte-identical). Tests: `library/tests/test_qwen38_persist.py`; hardware gates: `hackathon-prep/qwen38/truemono/WP1/`. |
| `ccl/` | multi-GPU in-kernel collectives over IPC peer mailboxes (host side: `rocke.runtime.peer`): `layout` (`CclKnobs`, `AllReduceSpec` / `AllGatherSpec` / `ArgmaxSpec`, arena map, error codes, `select_knobs`), `wire` (asm emitters: `saddr` pushes to 64-bit peer bases, per-lane flag stores, bounded tagged-pair / flag polls with `s_memrealtime` deadlines and the abort word), `collectives` (one-shot / two-shot all-reduce, all-gather, argmax merge; epoch and slot base advanced by the last CTA, no host reset between launches; launch header, on by default; a wave that records an error stops publishing; optional grid sync between chained collectives; test hooks and must-fail mutants), `ref` (bit-exact numpy references), `war` (slot-reuse checker), `lint` (peer-address and receive-arena lint). Python engine (`collectives.backend_for`); mirrored ops only (inline asm + plain IR). Suite, benchmark and case study: `platform/python/rocke/examples/gfx950/ccl/`; CPU tests `library/tests/test_qwen38_ccl.py`, `platform/tests/test_peer_runtime.py`. |

The mirrored (both-engine) part lives in `rocke.helpers.mailbox` and
`rocke.helpers.device_ops` (inline asm only). Microkernels, harness and
hop-latency results: `platform/python/rocke/examples/gfx950/persistent_mailbox/`.

## cpp-parity exemption

The repo rule (`rocke/AGENTS.md`, "the #1 invariant") is that both engines
emit the same LLVM-IR bytes for every kernel family. Builds of this family
that use an opcode in `rocke.core.ir.PYTHON_ONLY_OPCODES` (decisions D1b and
D4 of the Qwen3.8 roadmap) cannot meet it, because the C++ engine does not
lower those opcodes. They are handled as follows:

* They carry a distinct kernel-name suffix (`mbir` from
  `MailboxKnobs.suffix()`).
* They compile with `compile_kernel(..., backend="python")`, chosen
  explicitly (`mailbox_ir.backend_for(kernel)`; the example harness's
  `compile_for` does the same). A silent cpp->python fallback is a test
  failure: the tests assert the backend that ran.
* They are excluded from the C++ byte-identity gate. Every build that uses
  only mirrored ops (`impl="asm"`, the default) keeps it; see
  `platform/tests/test_mailbox_helpers.py::test_day1_cpp_engine_byte_identical_for_asm_impl`.
* Nothing in `rocke.helpers` emits these opcodes (`platform/AGENTS.md`: no
  helper without a C++ port path), and a test scans the helper sources for
  that.

The layer framework (`framework.py`, `stamps.py`, `toy_chain.py`) and its
helpers (`rocke.helpers.task_schedule`, `rocke.helpers.lds_arena`) emit
mirrored ops only and keep the byte-identity invariant:
`library/tests/test_qwen38_layer_framework.py::test_toy_cpp_engine_byte_identical`
lowers lever builds with `backend="both"`. A framework kernel inherits the
exemption above only through a stage emitter that uses a Python-only opcode.

Tests: `library/tests/test_qwen38_moe.py` (MoE plans, knobs, builds, compiles),
`library/tests/test_qwen38_layer_framework.py`, `library/tests/test_qwen38_stage_adapt.py`,
`platform/tests/test_task_schedule.py`, `library/tests/test_mailbox_ir.py`, `library/tests/test_qwen38_stage_fn.py`
(device functions and stage functions, CPU; the GPU proofs run in the Qwen3.8
harness: `hackathon-prep/qwen38/hybrid/`, documented in
`hackathon-prep/reports/qwen38-rocke/HYBRID.md`).

Device functions (`device_function.py`) are a lowering mode of the Python
engine with no C++ counterpart: like the opcodes above they are outside the
byte-identity gate, and kernel lowering is untouched (a test checks that
lowering a device function leaves a later kernel lowering byte-identical).
`math.sqrt` on f64 lowers only there (`llvm.sqrt.f64`, the FlyDSL mono's own
square root): `uses_python_only_ops` reports it as `math.sqrt.f64`
(`ir.PYTHON_ONLY_TYPED_OPS`), so a layer kernel carrying `stage_q2`'s body is
routed to the Python engine up front.

Device functions are `convergent` by default (`DeviceFunctionKnobs.convergent`):
a stage body holds barriers and cross-lane ops, and a non-convergent call may be
threaded under a divergent branch of the caller before inlining (seen: the mono's
stamped `if tid == 0` blocks around the B3 call split wave 0 over two copies of
B3's `s_barrier`s and hung the layer).

## Known limits

* Status (T9 gate): the framework's overhead is measured on the toy chain
  only, against a FlyDSL twin of the toy built from the mono's idioms (not
  mono's kernel), with the mono's own entry preload (`a1_preload`) and
  the stacked FlyDSL twin as baselines (`toy_chain_case_study.md`). There is
  no real-stage numeric A/B of adapted stages against the mono's mailbox
  outputs (rows 1 / rows 4 seq / MTP) and no round ratio yet: the T11 numeric
  A/B is the next gate, before any real-layer performance claim.
* The NUMA lever (`numa_copies=2` + `numa_flip_arg`) depends on a
  timing-based host calibration (`calibrate_iod_home`); `kernarg_preload`
  and f64 `math.sqrt` are Python-engine only.

* `MailboxKnobs(impl="ir")` polls through the `tile.tagged_poll` op, whose
  loads LLVM may hoist out of the loop when the polled buffer is a `noalias`
  kernel parameter (a spin that never reloads; true-mono P2 finding 1). Use
  `impl="irf"` (the op's `reload="launder"` variant) or the asm poll in new
  builds; `persist.lint.spin_without_reload` finds the hazard in a
  disassembly.
* `fence_impl="ir"` (an LLVM `fence` around non-atomic ready stores and
  polls) relies on backend behaviour, not on the LLVM memory model; the
  default `asm` fence does not.
* A runtime layer id passed to `mailbox.launch_tag` is not checked on the
  device. The launcher must call `mailbox.check_layer` (layer <= 62) for
  every layer, the drafter's 48 included.
