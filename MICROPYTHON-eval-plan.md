# Evaluating MicroPython as the embedded interpreter for `ck-dsl-provider`

**Status:** Phase 0 (feasibility spike) — started 2026-06-01
**Branch / worktree:** `users/dahawkin/ck-dsl-provider-micropython`
**Nature:** One of several **parallel candidates** being investigated. NOT a committed direction.

## Objective
Determine whether the provider can run the CK DSL **compile path** with **zero dependency on
any Python installation** using an embedded MicroPython, and quantify the cost/risk vs. the
parallel candidates (static-CPython embed, C++ rewrite, etc.).

## Motivation
- The provider must **not depend on a system Python installation**.
- Embedding **full CPython** carries its own risks: artifact size, security surface, the
  `ctypes`/OpenSSL/stdlib `.so` tail, and frozen-build + LTO/bitcode toolchain fragility.
- The current provider is a **POC**; rewriting the provider's C++ embedding glue + tests
  entirely is acceptable.

## Guiding constraints
- **`ck_dsl` is what must run** under MicroPython — it's vendored Composable Kernel upstream
  (`projects/composablekernel/python/ck_dsl`). Every edit to it is fork-maintenance burden, so
  this plan **prefers MicroPython-side shims over editing `ck_dsl`** and measures residual
  `ck_dsl` edits as an explicit cost.
- Provider C++ glue + tests are **accepted as a full rewrite** (POC) — scheduled *last*, after
  the make-or-break risks are retired.
- **POC target = conv implicit GEMM** (user direction, 2026-06-01; supersedes the brief FMHA
  retarget below). Chosen because the provider **already has the full C++ integration** for it —
  `ConvImplicitGemmPayload`, the bridge `compile("conv_implicit_gemm", payload, arch)`, and
  integration tests that validate accuracy vs `CpuFpReferenceConvolution` on gfx1151. So Phase 2
  (embed MicroPython in the provider) and end-to-end accuracy validation are comparatively cheap.
  Elementwise stays the cheapest *mechanism* proof (G1 stepping stone); conv is the real target.
  We still do *not* make all of `ck_dsl` MicroPython-clean — only the conv compile closure.

## Why the prior "MicroPython impossible" note doesn't end this
A prior spike concluded full-CPython is required. Re-verified against the live tree: the real
costs are (1) the CPython C-API embedding glue, (2) `dataclasses` (228×, 181 `frozen=True`) +
runtime `typing`/`lru_cache`/`pathlib` across `ck_dsl`, (3) `ctypes`→`libamd_comgr`. The user
accepts (1) as a rewrite; (2) becomes a shim-fidelity question; (3) becomes the **make-or-break
FFI gate** below. The language itself is clean (no `match`/`async`/metaclasses), so MicroPython's
*compiler* is not the blocker.

---

## Phase 0 — FFI / comgr go-no-go (THE make-or-break spike) 🔴
Cheapest test of the hardest blocker. If it fails, MicroPython is rejected and we stop — before
touching the provider or ck_dsl.

**Steps**
1. Build the MicroPython **Unix port** with `ffi` (`MICROPY_PY_FFI`) + `uctypes` enabled.
2. Standalone (no provider, no ck_dsl): reimplement *only* the comgr 3-stage compile
   (`SOURCE → BC → RELOCATABLE → EXECUTABLE`) from `runtime/comgr.py` against
   `libamd_comgr.so` using MicroPython `ffi`/`uctypes`.
3. Feed it a tiny hand-written LLVM IR string; require a valid `.hsaco` blob out.
4. Stress exactly the worrying surface: `byref` out-params (`create_data_set`/`get_data`/
   `action_data_count`), `c_char_p`/`c_void_p` arrays (option lists), opaque-handle
   `Structure`s, `create_string_buffer` blob read-back.

**Gate G0**
- PASS: `.hsaco` produced and loadable → proceed to Phase 1.
- FAIL: `ffi`/`uctypes` can't model the out-param/array surface → **STOP, MicroPython rejected**;
  write up why for the bake-off.
- ALSO RECORD: MicroPython `ffi` is **Unix-port only** → if Windows is in scope, flag as a
  likely hard dead-end now.

---

## Phase 1 — Can MicroPython *run* the ck_dsl compile slice? 🟠
1. Pin the **exact import closure** of the one exercised compile path; trim the eager
   `ck_dsl/__init__.py` + `helpers/__init__.py` pull-ins (`subprocess`/`ast`/`inspect`/`benchmark`).
2. Stand up MicroPython-side **shims**, validated against the slice only: `dataclasses`
   (`frozen=True`, `field(default_factory=...)`, `replace()`), runtime `typing.Generic/Protocol`,
   `functools.lru_cache`, `pathlib`. Wire `comgr.py`/`hip_module.py` to the Phase-0 `ffi` layer.
3. Success = run the compile entry for **one conv kernel** under MicroPython and **byte-compare
   emitted LLVM IR** vs. CPython for the same input (equivalence oracle).

**Gate G1**
- PASS: IR matches with shims + ≤ small, enumerable `ck_dsl` edits → proceed.
- FAIL/ESCALATE: shim surface explodes or hits runtime-semantics walls (descriptors,
  `__set_name__`, metaclass behaviour) → record shim-vs-fork cost, escalate decision.

---

## Phase 2 — Embed MicroPython in the provider, one path end-to-end 🟡
1. Rewrite the provider embedding layer on MicroPython's `mp_*` embed API: replace
   `EmbeddedInterpreter.cpp`, `CompileServiceBridge.cpp`, `PythonError.cpp`; reproduce the
   `py::dict` payload contract (`ConvImplicitGemmPayload.cpp`) as `mp_obj_t`; rewrite error
   translation. Swap `Python3::Python`/`pybind11::embed` out of CMake.
2. Rewrite affected unit/integration tests.
3. Wire one path end-to-end: provider → MicroPython → ck_dsl slice → comgr → `.hsaco` → launch
   one conv on **gfx1151**, validate accuracy vs the existing `CpuFpReferenceConvolution` harness.

**Gate G2**
- PASS: conv compiles + runs + matches reference on gfx1151 with no Python install present
  (verify via isolated `sys.path` / no `PYTHONPATH`).

---

## Phase 3 — Bake-off evidence report 🟢
Head-to-head table for the parallel investigation:
- Shipped **artifact size** & whether any external `.so` tail remains (target: none).
- **Startup / per-compile latency** vs CPython embed.
- **Python-install independence** (proven by Phase 2 isolation test).
- **`ck_dsl` fork footprint** (lines changed in vendored upstream + shim LOC).
- **Portability** (Unix-only `ffi`; Windows verdict).
- **Risk register** (shim fidelity, upstream drift, multi-plugin embedding interactions).

---

## Recommendation
Fund **Phase 0 only** as the first commit — highest information-per-dollar; protects against
sinking Phase 1/2 effort into a dead end. Decide Phase 1 after seeing G0.

---

## Phase 0 working log
_(to be filled in as the spike progresses)_

### Environment facts (2026-06-01)
- libamd_comgr: `/opt/rocm-7.2.4/lib/libamd_comgr.so` (+ `.so.3`); also `.so.2` in `/lib/x86_64-linux-gnu`. ROCm 7.2.4.
- MicroPython: none on PATH → building Unix port from source.
- Build prereqs present: gcc, cc, make, git, pkg-config; **libffi 3.4.6** (pkg-config OK).
- uctypes: standard in the Unix port; ffi (`MICROPY_PY_FFI`) is a headline Unix-port feature — confirm `import ffi` after build.

### comgr API surface to reproduce (from `runtime/comgr.py`)
All functions return `c_int` status. Opaque handles `_Handle{uint64}` are **structs passed by
value** in ctypes; under SysV AMD64 a single-eightbyte INTEGER struct == bare `uint64`, so model
them as `"Q"` in `ffi`. **Zero `CFUNCTYPE` callbacks** → the MicroPython ffi-callback GC bug is N/A.

The 3-stage chain: SOURCE(LLVM_IR) → COMPILE_SOURCE_TO_BC → CODEGEN_BC_TO_RELOCATABLE →
LINK_RELOCATABLE_TO_EXECUTABLE → extract EXECUTABLE bytes (HSACO).

Entry points + the FFI mechanics each exercises:
- `create_data_set(out _DataSet*)`, `create_data(kind, out _Data*)`, `create_action_info(out*)`
  → **out-param**: pointer to a uint64 slot, read back (uctypes/bytearray + addressof).
- `destroy_data_set(set)`, `release_data(data)`, `destroy_action_info(info)`,
  `data_set_add(set, data)` → **struct(s)-by-value** as `"Q"`.
- `set_data(data, size, char*)`, `set_data_name(data, char*)`,
  `action_info_set_isa_name(info, char*)`, `action_info_set_language(info, int)`.
- `action_info_set_option_list(info, char**, count)` → **array of char\***: bytearray of N*8
  pointer slots, each = addressof(option bytes); keep refs alive.
- `action_data_count(set, kind, out size_t*)`, `action_data_get_data(set, kind, idx, out _Data*)`.
- `get_data(data, in/out size_t*, char* | NULL)` → **two-call size-then-read**; NULL first, then
  bytearray buffer; read back via the bytearray.
- `do_action(action_kind, info, in_set, out_set)`.
- `status_string(status, out char**)` → out `char**`; spike may report numeric status only.

Make-or-break for G0: (a) `ffi` passes single-eightbyte structs-by-value correctly as `"Q"`;
(b) `uctypes.addressof` on bytes/bytearray supports out-params + the char* pointer array.

### Findings (2026-06-01)
- MicroPython 3.4.0 (commit `44a569b`), Unix port `build-standard`, ~785KB text. `ffi` exposes
  `{open, func, as_bytearray, callback}`; `uctypes` exposes `{addressof, bytearray_at, bytes_at,
  struct, sizeof, ...}`. libffi 3.4.6.
- Spike `spike/comgr_ffi_spike.py` reimplements `comgr.py`'s 3-stage chain with **no ctypes**:
  handles → `"Q"`, out-params/buffers/`char**` array → bytearrays via `uctypes.addressof`,
  two-call `get_data` with NULL=`0`.
- Run (`LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib`): produced **3520-byte HSACO**, `magic==\x7fELF`,
  `readelf` → ELF64, OS/ABI **AMD HSA**, Machine **AMD GPU**, targeted gfx1151. Compile is
  host-side (no GPU needed); blob is structurally valid (load/launch deferred to Phase 2).
- **Confirmed:** (1) single-eightbyte INTEGER struct-by-value == bare `uint64` holds through
  MicroPython `ffi`/libffi — all handle calls succeeded; (2) `uctypes.addressof` on bytes/bytearray
  covers out-params + the `char**` option array; (3) zero callbacks → ffi-callback GC bug N/A.

### ctypes footprint across ck_dsl (scoping G0's claim)
Heavy but concentrated in `runtime/`: `comgr.py` (50, **compile — ported by G0**),
`hip_module.py` (96, **launch**), `torch_module.py` (4, not used). All other ctypes is under
`examples/` (standalone bench/tune scripts, not provider code).

The provider drives **compile** in Python (`compile_service.compile` → `helpers.compile.compile_kernel`
→ `comgr.py`) and does GPU **load/launch in C++** (`hipModuleLoadData`/`hipModuleLaunchKernel`),
NOT via `hip_module.py` — stated in `compile_service.py`'s own docstring. So `hip_module.py`'s 96
launch-side ctypes refs are **never called** on the provider path. `hip_module.py` is reachable only
because `comgr.py` imports 4 lib-resolution helpers from it (`_IS_WINDOWS, _LazyFn, _add_dll_dir,
_candidate_lib_paths`); a small refactor extracting those into their own module keeps the launch
ctypes out of the MicroPython scope entirely (Phase 1 task).
`hip_module.py` is not harder than comgr if ever needed: no callbacks; all 3 handle structs are
single `void*`; buffers are `(c_ubyte*n).from_buffer_copy` = bytearray. Only multi-field struct is
`hipDeviceProp_t` (device query, off-path).

### G0 verdict: **PASS** (scoped: compile-path FFI / comgr)
The FFI/comgr make-or-break blocker is cleared. MicroPython `ffi`/`uctypes` can drive the in-process
comgr compile end-to-end. The prior "MicroPython impossible" claim is refuted on the FFI axis.
**Caveat (unchanged):** MicroPython `ffi` is Unix-port only → Windows remains blocked at the *port*
level (not the ABI level — Win x64 would also pass a single-eightbyte struct in a register).
**Next:** proceed to Phase 1 (can MicroPython actually *run* the ck_dsl compile slice — dataclasses/
typing/lru_cache/pathlib shim fidelity), the real remaining uncertainty.

## Phase 1 working log

### Import-closure analysis (CPython ground truth, 2026-06-01)
Probes: `spike/phase1_closure.py` (full), `spike/phase1_minimal_closure.py` (heavy package
`__init__`s neutered via sys.modules namespace stubs), `spike/phase1_trace.py` (who-imports-what).
- Full closure of provider compile path: **120 ck modules + 68 non-ck top-level packages.**
- Neutering eager `__init__` bloat (`ck_dsl`/`helpers`/`instances`/`runtime`) cut ck modules to
  **21 (elementwise) / 34 (+conv)**, but non-ck only fell to 57 → the stdlib breadth is pulled by
  the **leaf compile-core modules**, not the `__init__` bloat.
- Elementwise smoke compiles cleanly with the stubs (valid gfx1151 HSACO). **Conv does NOT stay
  minimal**: `conv → helpers.pipeline → helpers.schedule → analysis.ir → analysis/__init__ →
  analysis.report → `from ..helpers import KernelArtifact`` (package-level export) — real coupling
  into `analysis` + package-level `helpers`. So elementwise is the clean first slice; conv needs
  the coupling untangled (or `analysis`/`helpers` partially shimmed).

### Causal map → the closure collapses (key finding)
`spike/phase1_trace.py` shows the 57 non-ck packages are mostly **cascade from removable roots**:
- `inspect ← dataclasses` → pulls `ast, dis, tokenize, token, linecache, opcode`. A clean
  dataclasses shim (no `inspect`) deletes all of them.
- `subprocess, tempfile ← helpers.compile` (only the **hipcc fallback**, not the comgr path) →
  pull `shutil → bz2, lzma`, `locale → re`, `weakref`, `signal/select/selectors/fcntl`. Making
  those two imports lazy deletes the whole cascade.
- `urllib ← pathlib`, `ipaddress ← urllib` → a minimal pathlib shim deletes both.
- `glob ← hip_module` → the 4-helper extraction deletes it. `ctypes ← comgr` → Phase-0 ffi.
- `json ← arch.target`, `re` → MicroPython built-ins.

### Core-path shim surface (final inventory)
Real shims: **`dataclasses` (big: frozen/field/replace), `typing` (medium), `enum` (small),
`functools.lru_cache` (small), `pathlib` (minimal)**. Built-in in MicroPython: `json, math,
struct, re`. Two small ck_dsl edits: lazy `subprocess`/`tempfile` in `helpers/compile.py`;
extract comgr's 4 lib helpers out of `hip_module.py`. **New risk:** MicroPython `re` is a
SUBSET — must verify the actual patterns in `ir/lower_*/arch/elementwise/compile` parse under it.

### Remaining Phase-1 steps (to reach G1)
1. Build the shim package (dataclasses, typing, enum, functools.lru_cache, pathlib) for MicroPython.
2. Drop in the Phase-0 comgr-ffi as `runtime/comgr` replacement; extract the 4 hip_module helpers.
3. Provide trimmed `__init__`s (or direct leaf imports) for the elementwise slice.
4. Run `compile_smoke`-equivalent under MicroPython; **byte-compare emitted LLVM IR vs CPython**
   (equivalence oracle) → G1. Then attempt the conv slice (untangle analysis/helpers coupling).

### G1 BLOCKER (2026-06-01): MicroPython erases class annotations → dataclasses unrunnable
Make-or-break test of the vendored `udataclasses` (`spike/mp1/test_udataclasses.py`) failed —
and failed **identically under CPython and MicroPython**, exposing a fundamental wall:
- **MicroPython does not expose class `__annotations__`** (`hasattr(C, "__annotations__") == False`):
  bare-annotation fields (`name: int`, no default) are parsed and **discarded**; the field
  name/order does not exist at runtime.
- **`udataclasses` discovers fields from `cls.__dict__`** — only attributes carrying a value
  (`field()` or a default). Bare-annotation required fields are invisible to it.
- Combined: for a required field declared `x: int` (ck_dsl's normal style) the field metadata is
  **gone after parsing** under MicroPython. NO shim — udataclasses or hand-written — can recover
  what the interpreter threw away.

Blast radius on the chosen elementwise slice (concrete): `ElementwiseSpec.op` is bare/required;
`core/ir.py` = **97** bare no-default fields, `core/arch/target.py` = **53**. So even the
simplest slice cannot construct its core IR/spec objects. This is a MicroPython *language-runtime*
limitation, independent of FFI (G0 passed) and of shim fidelity.

**Escape routes (all expensive — escalated to user):**
1. **Rewrite ck_dsl** so every required field carries an explicit `field()`/default (lands in
   `__dict__`). ~150+ fields on the core path alone, hundreds across 228 dataclasses. Invasive,
   perpetual fork of vendored CK; could be automated by a source transform but still a fork.
2. **Patch the MicroPython compiler** to emit/retain class `__annotations__`, then a small
   annotation-reading dataclasses shim works and **ck_dsl stays 100% unmodified**. One change to
   a component we already vendor/build, but non-trivial C compiler work + a MicroPython fork.
3. **Source-to-source transform at bundle build** (annotations → `field()` defaults) — automated
   fork-in-a-build-step; keeps CK source readable but fragile and still a maintained transform.

**Verdict:** G1 is NOT cleared. udataclasses (the chosen approach) cannot work as-is. The decision
between routes 1/2/3 is strategic and changes MicroPython's cost vs the parallel candidates.

### G1 UPDATE (2026-06-01): ck_dsl is changeable → dataclasses SOLVED
User confirmed ck_dsl can be modified (CK team in the loop; changes to support a new interpreter
are expected). That selects **Route 1** and unblocks dataclasses. Two findings refined the approach:
- `udataclasses` is a poor fit: even with `= field()` it generates a **keyword-only `__init__`**
  (`ES("copy")` fails) and sorts fields **alphabetically**, so adopting it would force converting
  every positional dataclass construction in ck_dsl to keyword — far more invasive than needed.
- **Switched to a custom shim** (`spike/mp1/shims/dataclasses.py`, ~150 lines, no `exec`):
  declaration order via a field-creation counter; positional+keyword `__init__`; required fields;
  `default`/`default_factory`; frozen (eq/hash/immutability); `eq`/`repr`/`fields`/`replace`/
  `asdict`/`astuple`; single-level field inheritance. (Ordered fields stored as an explicit tuple,
  since MicroPython dict `.values()` is unordered.) **Validated byte-identical to CPython** across
  all of the above on both interpreters (`spike/mp1/test_dc_custom.py`).
- **Required ck_dsl change (the Route-1 cost):** every dataclass field declared with an explicit
  `name: T = field(...)` (required) or `field(default=...)` (optional) so it lands in `__dict__`
  (MicroPython erases bare annotations). Mechanical + automatable; CK team adopts it. Backward-
  compatible with CPython dataclasses. Blast radius (core path): ir.py 97 + arch/target.py 53 +
  more across 228 dataclasses — automate with an AST transform.

**Dataclasses no longer blocks G1.** Remaining to reach G1: other small shims (typing/enum/
lru_cache/pathlib), apply the field() transform to the elementwise slice, wire comgr-ffi +
trimmed `__init__`s, run elementwise under MicroPython, byte-compare LLVM IR vs CPython. Open
risks still to hit: MicroPython `re` subset; runtime `typing` usage; any other language gaps.

### POC retarget: FMHA forward + backward (scoping, 2026-06-01)
Entry points exist: `build_fmha_fwd_mfma(spec: FmhaMfmaSpec, arch)` (instances/common/fmha_mfma.py)
and `build_fmha_bwd(spec: FmhaBwdSpec, arch)` (instances/common/fmha_bwd.py); also the unified
attention builders (`build_unified_attention_2d/3d`, `UnifiedAttentionProblem/2DSpec/3DSpec`).
**No new language-level blockers:** zero real `match`/`case` statements and zero `async def`/
`await` anywhere in ck_dsl (the grep hits were the word "match" in comments and "async DMA"/
`use_async_kv` identifiers). Generators (`yield`) are used but MicroPython supports them.
FMHA's incremental cost over elementwise is **breadth, not new blockers**: a much larger import
closure (helpers/attention, mfma_attention, mfma_attention_bwd, fusion_*, pipeline, distribution,
mfma_gemm_inner, …) → more `= field()` conversions and more modules that must import cleanly. The
same open risks apply (re subset, runtime typing). **Confidence checklist for the FMHA POC:**
(1) G0 FFI ✓, (2) dataclasses mechanism ✓, (3) FMHA language features ✓ clean, (4) elementwise-G1
run [pending — flushes re/typing], (5) FMHA fwd+bwd closure imports + builds under the shims
[pending]. Declare confidence only after (4) and (5).

### POC retarget #2: conv implicit GEMM (supersedes FMHA, 2026-06-01)
Switched the POC target back to **conv implicit GEMM** because the provider already integrates it
(C++ `ConvImplicitGemmPayload` + bridge `compile("conv_implicit_gemm", payload, arch)` +
accuracy-validating integration tests on gfx1151) — far cheaper to take end-to-end than FMHA.
Entry points: `build_implicit_gemm_conv(spec, arch)`, `is_valid_spec(spec, arch)`,
`_conv_spec_from_payload` (instances/common/conv_implicit_gemm.py; ConvProblem/ImplicitGemmConvSpec).
Already known from Phase-1 closure work: conv is heavier than elementwise — it couples
`conv → helpers.pipeline → schedule → analysis.ir → analysis/__init__ → report →
`from ..helpers import KernelArtifact`` (package-level), so the analysis/helpers coupling must be
untangled (trimmed `__init__`s or partial shims). No new language blockers (the whole-ck_dsl scan
already found zero real match/case and zero async def/await). **Confidence checklist (conv):**
(1) G0 FFI ✓ (2) dataclasses ✓ (3) language clean ✓ (4) elementwise-G1 run [pending] (5) conv
closure imports + builds under shims, untangling analysis/helpers coupling [pending]. Then Phase 2
reuses the existing conv provider integration for end-to-end accuracy on gfx1151.

### G1a ACHIEVED (2026-06-01): MicroPython runs elementwise codegen → IR matches CPython
MicroPython runs ck_dsl's **full elementwise `lower_kernel_to_llvm`** and emits LLVM IR that is
**semantically identical to CPython** — same length (3037), same checksum, and `sort`-identical
line-for-line. The ONLY difference is the position of one `declare` line: declares come from the
`_INTRINSIC_DECLS` dict (core/lower_llvm.py) collected unordered, and MicroPython iterates
dict/set in a different order. Byte-identity needs a one-line determinism fix (sort the emitted
declares). The `re`-subset and runtime-`typing` risks are **cleared for this slice** — lower_llvm
uses `re` and it ran correctly under MicroPython.

Harness = a processed `ck_dsl` **bundle** (`spike/mp1/`, gitignored copy) built by
`spike/mp1/build_bundle.py`; run by `spike/mp1/run_g1.py` under both interpreters.

**Shims written** (`spike/mp1/shims/`, all validated): `dataclasses` (custom, order-preserving),
`typing`, `functools` (lru_cache), `pathlib` (str subclass, no `__new__`), `__future__`,
`itertools` (product/accumulate). Built-in in MicroPython: `re`/`json`/`math`/`struct`/`collections`.

**ck_dsl source transforms needed** (automated in build_bundle.py — these are the Route-1 changes
the CK team would adopt): (1) every dataclass field → explicit `= field(...)`; (2) PEP-448
star-unpacking in list/tuple displays `[a,*b,c]` → `[a]+list(b)+[c]` (MicroPython lacks it,
~62 sites); (3) `open(x)` → `open(str(x))` (MicroPython open rejects str subclasses); (4)
`os.environ.get(...)` → `os.getenv(...)` (MicroPython `os` has no `environ`, is read-only).
Plus the determinism fix (sort declares) for byte-identity.

**MicroPython language/runtime limits hit & handled:** no class `__annotations__`; no PEP-448
displays; `str` has no `__new__` and `open()` rejects str subclasses; builtin `os` is read-only and
lacks `environ`; no `__future__`/`itertools`/`dataclasses`/`typing`/`functools`/`pathlib` modules.
None fatal — all handled by shims + automated transforms.

**Confidence: high for the mechanism.** Remaining to full conv-POC confidence: (G1b) feed the
MicroPython IR to the Phase-0 comgr-ffi → HSACO (mechanism already proven in G0); the declares
determinism fix for byte-identity; then the **conv** slice (untangle analysis/helpers coupling,
apply the same transforms, re-run — conv may surface more `re` patterns / stdlib).

### G1 CONV ACHIEVED (2026-06-01) — CONFIDENCE GATE MET
MicroPython runs the **full conv implicit-GEMM codegen** (`build_implicit_gemm_conv` +
`lower_kernel_to_llvm`, `spike/mp1/run_conv.py`) and emits **43,542 bytes of LLVM IR identical to
CPython** — same length, same checksum, `sort`-identical (only ~10 `declare` lines reordered, the
same dict-iteration nondeterminism as elementwise). Notes:
- The feared `analysis`/`helpers` coupling **resolved itself**: trimming `analysis/__init__` cut
  the import chain to `report.py` (which was the only thing doing `from ..helpers import ...`).
- Conv needed a **larger GC heap** — MicroPython unix default is 2 MB; conv OOM'd until run with
  `-X heapsize=1024M`. The embedded interpreter must be configured with an adequate heap (trivial).
- No new language/stdlib blockers beyond the elementwise set; the same shims + 4 transforms suffice.

**Every confidence-checklist item is green** (G0 FFI; dataclasses; language; elementwise IR==CPython;
conv IR==CPython). **Verdict: confident MicroPython can run the conv compile path.** Trigger met to
commit to the conv POC.

### G1b ACHIEVED (2026-06-01): conv → HSACO end-to-end in MicroPython, byte-identical to CPython
`spike/mp1/run_g1b.py` runs the WHOLE conv compile path in one MicroPython process with no CPython:
conv DSL → `lower_kernel_to_llvm` (43,542 B IR) → `comgr_ffi.build_hsaco_from_llvm_ir` → **8,808 B
HSACO** (valid ELF64, OS/ABI AMD HSA, Machine AMD GPU, gfx950). `spike/mp1/comgr_ffi.py` is the
module form of the Phase-0 comgr-ffi (the `runtime/comgr` replacement). Run with
`LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ... -X heapsize=1024M`.
**Strengthening check (`spike/mp1/compile_ll.py`):** compiling BOTH the MicroPython IR and the
CPython IR through comgr yields **byte-identical HSACOs** — so the declare-ordering diff is purely
cosmetic (no codegen impact); MicroPython produces the *exact same compiled conv kernel* as CPython.
The declares-determinism fix is therefore optional (nice-to-have for IR-text diffs, not needed for
HSACO equality).

### WINDOWS feasibility + Phase-2 architecture decision (2026-06-01)
**MicroPython `ffi` (`modffi.c`) is `ports/unix` ONLY** — not in the embed port nor the Windows port
(uses POSIX dlopen/libffi). So any design where MicroPython drives comgr via `ffi` is a **Windows
dead-end**. The G0/G1b spike used `ffi` for comgr as a Linux convenience to prove the chain — NOT the
shipping architecture.

Two architectures:
- **Arch B (spike-style):** MicroPython does codegen + comgr-via-ffi. ❌ Windows (no ffi).
- **Arch A (CHOSEN):** MicroPython does **codegen only** → returns LLVM IR text; **C++ calls comgr**
  (links `amd_comgr`). ✅ Windows. Also cleaner (comgr lives in C++ with the HIP/launch code).

**Why Arch A is Windows-viable, with evidence:**
- The Windows-critical piece is already proven: **G1a's `run_g1.py` codegen uses ZERO ffi** (just
  `sys` + ck_dsl + pure-Python shims) and produced IR byte-identical to CPython — that IS the Arch-A
  MicroPython side.
- The **embed port is portable C** (no dlopen/pthread/unistd) → embeds in the provider DLL on Windows.
- Codegen path OS touches = only `os.getenv` (transformed) + a `sys.modules` torch check — portable;
  `arch_specs.json` via `open()` or frozen.
- `amd_comgr.dll` ships with the Windows ROCm/HIP SDK; C++ calling it mirrors the existing
  `hipModuleLoadData` launch path that already runs on Windows.
- **Freeze the ck_dsl bundle into the MicroPython binary** (frozen modules, all ports) → single
  self-contained, no-filesystem, no-Python-install artifact on Linux AND Windows (the original goal).

**Arch A changes vs current provider:** `compile_service.compile` returns LLVM IR text (+ metadata)
instead of HSACO; add a **C++ comgr wrapper** (IR→HSACO, cross-platform); `HipModule`/launch unchanged.

**Windows unknowns to verify in Phase 2 (not testable from this Linux box):** build embed port on
Windows (MSVC/clang) into the DLL; C++ `amd_comgr.dll` calls under Windows HIP SDK; frozen-modules
build (`mpy-cross` + freeze the transformed bundle); `__file__` absent under frozen → embed/guard
`arch_specs.json` (already noted in [[ck-dsl-embed-cpython]]).

### Remaining for the conv POC (post-confidence) — built on Arch A
1. (Optional) sort the emitted declares in `core/lower_llvm.py` for byte-identical IR *text* — NOT
   needed for HSACO equality (proven byte-identical above).
2. Phase 2 (Arch A): build MicroPython embed port as a static lib; replace
   EmbeddedInterpreter/CompileServiceBridge/PythonError on MicroPython's `mp_*` API; configure GC
   heap (conv needs >2 MB); marshal the existing conv `py::dict` payload as `mp_obj_t`. **MicroPython
   returns LLVM IR text** (not HSACO).
3. Phase 2 (Arch A): add a **C++ comgr wrapper** (IR text → HSACO via `amd_comgr`, cross-platform) —
   the new step replacing Python's comgr; feed its HSACO into the unchanged `HipModule`/launch path.
4. End-to-end: reuse the existing conv provider integration + `CpuFpReferenceConvolution` accuracy
   oracle on gfx1151.
5. Package the ck_dsl bundle build (the 4 transforms + trimmed __init__s) as a real build step
   (ideally **frozen modules** for a self-contained no-Python-install artifact on Linux+Windows);
   land the field()/star-unpacking/etc. transforms as upstream CK changes. NOTE: the comgr-ffi swap
   is NO LONGER part of the bundle under Arch A — comgr moves to C++.
