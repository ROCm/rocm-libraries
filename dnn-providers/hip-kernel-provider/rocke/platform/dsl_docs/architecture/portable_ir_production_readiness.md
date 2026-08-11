# Portable-IR prototype: production readiness review

Assessment of the portable-IR Python-to-C++ bridge against production
requirements. Companion documents: [`portable_ir_schema.md`](portable_ir_schema.md)
(the as-built wire schemas and VM contract) and
[`../../python/rocke/portable_ir/portable_ir_scaling_plan.md`](../../python/rocke/portable_ir/portable_ir_scaling_plan.md)
(the forward rollout plan). This document covers what the prototype does today,
what it does not, and what must close before the implementation story starts.

Everything quantitative below was measured on the configuration in
[Method](#method), not carried over from earlier runs. Where a number is
reproduced from a prior session it is re-measured and labelled as such.

## Verdict

**The core technical risk is retired; the delivery risk is not.**

The record-and-replay design does what it claims. The C++ recipe VM reproduces
the Python front end exactly, including for *rolled* parametric recipes replayed
at shapes the roller never sampled — verified at the HSACO byte level, which is
the artifact that actually ships. That was the open question, and the answer is
affirmative for all four kernel families examined.

What is not ready is everything around that core. The shared library the JIT
path depends on **does not link from a clean tree**, there is **no CI wiring of
any kind**, and rolling generalizes over input shapes far better than over
tile/warp geometry — of seven axes that roll across the four families, only one
is a tile-geometry axis. None of these are design flaws; they are unfinished
engineering, and each has a bounded fix.

Separately, measuring at the HSACO level exposed something the existing `.ll`
gate cannot see: on gfx942 only 32 of the 46 "parity-passing" kernels compile to
a binary at all, and one kernel exhausts host memory inside LLVM. Both are
pre-existing defects in the kernel set rather than portable-IR regressions, but
they bound what a parity number means and they constrain how a CI job must be
built.

Recommendation: proceed to the implementation story, with the four P0 items in
[Gap analysis](#gap-analysis) treated as entry criteria rather than work items
inside it.

## Method

| | |
|---|---|
| Host | AMD Instinct MI355X node, 4 visible GPUs |
| ROCm / comgr | 7.2.0, comgr reports `(7, 2)` |
| LLVM flavor | `llvm22`, auto-resolved from the installed comgr and pinned on every path |
| Target | `gfx950` (parity matrix additionally covers `gfx942`) |
| Host compiler | GNU 13.3.0, CMake `Release`, `-DROCKE_BUILD_PYENV=OFF` |
| Device | Not required for the gates below; comgr compiles for the target ISA on the host |

The `rocke_engine` CPython extension is not built in this environment, so
builder-side `backend='cpp'` falls back to the Python lowerer. That is the
intended configuration here: the Python lowerer is the *oracle* being compared
against, so it must be the reference on the Python side.

## What works

### 1. The concrete replay path, re-measured

`parity_matrix` lowers each production `KernelDef` three ways — native Python,
`ir_export` → C import → C lower, and record → CBOR → C recipe VM → C lower —
and demands byte-identical `.ll`.

```
kernels tested: 56   (build-skipped: 12)
  gfx950:  engine 46/46 byte-identical   recipe 46/46 byte-identical   (lower-skip 10)
  gfx942:  engine 46/46 byte-identical   recipe 46/46 byte-identical   (lower-skip 10)
```

Both backend paths agree with Python exactly, on both architectures. Concrete
recipes carry `exact_names`, so the VM reproduces Python's SSA names verbatim —
this is text equality, not merely semantic equivalence.

Carrying those kernels through comgr to object code — which `parity_matrix` does
not do — gives the same answer wherever a binary can be produced at all:

| Arch | Compared at HSACO | comgr deterministic | Engine path | Recipe path |
|---|---|---|---|---|
| gfx950 | 45 | 45/45 | 45/45 identical | 45/45 identical |
| gfx942 | 32 | 32/32 | 32/32 identical | 32/32 identical |

The determinism column compiles the *same* `.ll` twice and compares; it is the
control that makes the other two columns meaningful. Every kernel that compiles
produces byte-identical HSACO on both backend paths.

The shortfall from 46 is not a parity failure — it is kernels that no path can
compile, discussed in [Compilability](#ll-parity-overstates-what-is-shippable).

### 2. The rolled path, proven at the HSACO level

This is the new result and the one that matters most, because it is the first
evidence that the *parametric* artifact — not just a replayed concrete trace —
produces the correct binary. The gate is
`rocke.portable_ir.drivers.roll_hsaco_parity`:

```
author time : record 2 concrete traces -> roller -> ONE parametric recipe -> CBOR
run time    : CBOR -> C DOM decode -> C recipe VM expand at spec -> C lower -> comgr -> HSACO
oracle      : pure Python build -> Python lower -> comgr -> HSACO
```

Across four families and seven axes, **22/22 verification points produced
byte-identical HSACO, 8 of them at held-out axis values the roller never
sampled**:

| Family | Axis | Kind | Points (held-out in bold) | Compression | HSACO |
|---|---|---|---|---|---|
| `gemm_universal` | `tile_n` | tile geometry | 32, 64, **128**, **256** | 10.0x | 4/4 identical |
| `conv_implicit_gemm` | `K` | input shape | 64, 128, **256** | 3.0x | 3/3 identical |
| `conv_implicit_gemm` | `N` | input shape | 8, 16, **32** | 3.0x | 3/3 identical |
| `attention_dense` | `seqlen_kv` | input shape | 512, 1024, **2048** | 3.0x | 3/3 identical |
| `attention_dense` | `num_query_heads` | input shape | 64, 128, **256** | 3.0x | 3/3 identical |
| `fused_moe` (gather) | `hidden` | input shape | 512, 1024, **2048** | 4.2x | 3/3 identical |
| `fused_moe` (gather) | `tokens` | input shape | 32, 64, **128** | 3.0x | 3/3 identical |

Held-out points are the substance of the claim. Reproducing the two traces the
roller was shown proves only that replay works; producing the correct binary at
`seqlen_kv=2048` from a recipe inferred at 512 and 1024 is what demonstrates the
recipe generalized over the axis.

Two internal consistency checks fall out of the data. Kernels that are the same
configuration reached from different sweeps hash identically — the
`attention_dense` default appears as `sha=f5c4c2fa911741e4` from both the
`seqlen_kv` and `num_query_heads` sweeps — and HSACO differs at every distinct
axis value, so the comparison is not vacuously succeeding on a
spec-insensitive kernel.

### 3. Rolled `.ll` is byte-equal too (was alpha-equivalent)

A concrete recipe replays Python's SSA names verbatim, because its binds *are*
the finished names. A rolled recipe cannot do that — one instruction expands
many times, so every expansion has to draw a fresh name — and it originally
emitted the engine's positional default:

```
-  %tid7 = call i32 @llvm.amdgcn.workitem.id.x()
-  %div8 = sdiv i32 %tid7, 64
+  %v7 = call i32 @llvm.amdgcn.workitem.id.x()
+  %v8 = sdiv i32 %v7, 64
```

Only the prefix differed. Both engines name values `%<prefix><counter>` off the
same counter, and the counters were already in lockstep — across the four
families, 4,954 renamed values showed zero index drift — so supplying the
prefix was enough to close the gap. Two changes did it:

- **Per-op prefix.** Python picks the prefix per op via `result_name_hint`
  (`%mul`, `%tid`, `%acc32`, ~38 distinct ones in the attention kernel).
  `RecordingIRBuilder` now captures that hint where it is passed and stores it
  on the instruction, and the VM feeds it to `rocke_b_op` as the result-name
  hint. Mirroring Python's table in C++ was the alternative and was rejected:
  the hint lives at ~38 Python call sites, so a copy would silently drift as ops
  are added, whereas a recorded hint travels with the data and needs no C++
  change for a new op. Recipes without the field fall back to `v`, the previous
  behavior.
- **Fan lane names.** Loop-carry fans were the remaining 8 values in the GEMM:
  Python names them positionally (`%acc_m0_n0` … `%acc_m0_n3`) while the roller
  minted `%__fa0_0`. That is roller business, not VM business — the roller now
  derives Python's lane prefix from the sampled traces and parameterizes on it,
  so it extends to lane counts never sampled. Fans whose names are not simply
  the lane index (a 2-D `acc_m{m}_n{n}`, say) keep the synthetic naming and stay
  alpha-equivalent.

Rolled `.ll` is now byte-identical at all 22 verification points, held-out ones
included, so a checksum comparison is sufficient and comparing the rolled path
no longer requires a comgr compile. The driver still scores `EXACT`,
`ALPHA-EQ`, and `DIFFER` separately and fails on anything below `EXACT`, so a
naming regression surfaces instead of passing quietly on the HSACO gate.

### 4. The gate discriminates (negative control)

An all-pass result is only meaningful if the comparison can fail. Replaying the
GEMM recipe at spec value *v* and comparing against Python built at *v′* gives a
clean diagonal:

| py \ vm | 32 | 64 | 128 |
|---|---|---|---|
| **32** | identical | differ | differ |
| **64** | differ | identical | differ |
| **128** | differ | differ | identical |

Off-diagonal cells differ in both HSACO and alpha-normalized `.ll`. The gate
detects a wrong-spec replay.

### 5. Compile-time characteristics

Median over 5 iterations at three axis points each, front end only plus the
common comgr stage:

| Family | Python front end | Recipe VM front end | Speedup | comgr (common) | Cold JIT, Python | Cold JIT, recipe |
|---|---|---|---|---|---|---|
| GEMM | 1.38 ms | 0.32 ms | 4.3x | 1.8 ms | 3.2 ms | 2.1 ms |
| Attention (dense) | 20.25 ms | 5.74 ms | 3.5x | 3.2 ms | 23.5 ms | 9.0 ms |

The VM front end is 3.5–4.3x faster than the Python front end, but comgr is
common to both paths and dominates for small kernels, so end-to-end cold JIT
improves by a more modest 1.5–2.6x. These figures exclude CPython interpreter
startup and module import, which the C path avoids entirely — that, rather than
front-end microseconds, is the larger practical argument for the replay path in
a deployed provider.

Artifact sizes favor the parametric form: attention is 506 KiB as one recipe
versus 1518 KiB as three concrete traces; GEMM is 16.2 KiB versus 162.8 KiB
across four shapes.

## What does not work

### The shared library does not link from a clean tree — P0

Reproduced from scratch in this session. `librocke_core.a` builds cleanly, then
the `--whole-archive` shared link fails:

```
/usr/bin/ld: core/librocke_core.a(rocke_build_id.cpp.o): in function `rocke_build_id':
  multiple definition of `rocke_build_id'; core/librocke_core.a(build_id.cpp.o): first defined here
/usr/bin/ld: ... multiple definition of `rocke_engine_version' ...
```

`cpp/core/build_id.cpp` and `cpp/core/rocke_build_id.cpp` both define
`rocke_build_id()` and `rocke_engine_version()`. The collision is invisible in a
static archive — the linker picks one member — and only surfaces when
`--whole-archive` pulls in every object to build the `.so`. Since the entire
online JIT path loads that `.so` via ctypes, **nothing in the JIT path works
from a clean checkout without intervention.**

Every measurement in this document required
`-Wl,--allow-multiple-definition`, which silently picks whichever definition the
linker sees first. That is acceptable for an assessment and unacceptable to
ship: build-ID provenance is exactly the thing you do not want resolved
arbitrarily. The fix is to delete the stale duplicate (`.gitignore` drops
`build*`, so `rocke_build_id.cpp` is the intended survivor) and its header, then
drop the workaround flag.

### No CI wiring — P0

The repository has 41 workflows under `.github/workflows/`. **None reference
rocke.** Every gate described here is a script somebody has to remember to run.
Concretely, the byte-identity contract is unenforced: any change to the Python
lowerer can silently desynchronize the two engines and no automated check will
notice.

This also makes a claim in the scaling plan false today:

```334:336:dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/portable_ir/portable_ir_scaling_plan.md
- CI runs the parity matrix and the device replay gate
  (`tests/portable_ir/test_portable_ir.py`, `drivers/gpu_replay.py`) on every
  kernel change.
```

The same section points at a `run_*_demo.sh` set as the HSACO-tier gate; those
scripts were removed during the port and no longer exist. Both statements should
be corrected to describe intent rather than status.

### Rolling generalizes over shapes far better than over geometry — P1

The most consequential *technical* limitation, and it cuts against the stated
assumption that shapes and tile/warp geometry are equally available knobs. Of
the seven axes that roll, **exactly one is a tile-geometry axis** (`gemm_universal
:: tile_n`). Every conv, attention, and MoE success is an input-shape axis. Eight
probed axes were refused:

| Family | Axis | Roller refusal |
|---|---|---|
| `gemm_universal` | `tile_m` | shorter-at-larger-axis — and in fact **non-monotonic**: 90 → 81 → 114 ops over 16/32/64 (see below) |
| `gemm_universal` | `tile_k` | verify failed: k-atom constant not affine in `tile_k` |
| `conv_implicit_gemm` | `tile_n` | no run candidate: trace lengths 59 vs 83 do not segment |
| `conv_implicit_gemm` | `tile_m` | no run candidate: trace lengths 160 vs 184 do not segment |
| `conv_implicit_gemm` | `C` | non-affine constant 6 vs 7 (spatial product) |
| `attention_dense` | `head_size` | merge conflict on `tile.smem_alloc` |
| `attention_dense` | `block_n` | non-affine constant 8 vs 4 |
| `fused_moe` (gather) | `hidden` at 128 | opcode change: `global_load` vs `global_load_vN` |

These refusals are informative rather than mysterious, and several are inherent
rather than roller bugs. `attention_dense :: block_n` is the clearest: with
`seqlen_kv` fixed, the KV block count is `seqlen_kv / block_n`, so 64 → 128 takes
a constant from 8 to 4. That is a *division* by the axis, and the roller's
solver infers linear relationships only. No amount of roller work makes a
reciprocal affine; supporting it requires either a richer intexpr grammar or
treating `block_n` as a separate recipe family. Likewise `fused_moe :: hidden` at
128 changes the *opcode* (vectorization width flips `global_load` to
`global_load_vN`) — a structural change, not a parametric one, and it rolls
cleanly once sampling starts at 512 where the width is stable.

`gemm_universal :: tile_m` was later measured properly and is worth separating from
the rest: it is not merely shrinking but **non-monotonic** (90 → 81 → 114 ops over
16/32/64), because `tile.mma` scales cleanly (2 → 4 → 8) while the load path
re-vectorizes underneath it (`memref.global_load_vN` 3 → 2 → 3, address arithmetic
33 → 28 → 42). There is therefore no sampling window in which it is affine —
closing it needs opcode/vector-width selection (`static_if`), not a better solver.

**Since this assessment, the constants-only axes roll together.** `src/roll_nd.py`
covers an axis *cross product* in one recipe (conv `N`×`K`, attention
`seqlen_kv`×`num_query_heads`, MoE `hidden`×`tokens`, gated by
`drivers/roll_nd_coverage.py`), so the count above is now per family rather than
per family-times-axis. The refusals in the table stand, and `C` and `tile_m` were
re-probed as *pairs* and refused again.

The safety property holds throughout: **every refusal is a refusal, not a wrong
answer.** The roller verifies against the Python oracle at sampled and held-out
points and falls back to concrete per-shape recipes when inference fails. Across
every probe in this assessment, the roller never emitted an incorrect recipe. A
refusal costs compression, never correctness.

One boundary on that claim, found while extending the roller: the oracle
(`recipes_equiv`) compares **programs, not kernel names**. A recipe whose
`kernel_name_fmt` does not track its axes passes every oracle check while emitting
the wrong symbol — demonstrated with a kernel named after a derived quantity, where
`roll()` reports success and the oracle agrees at every point. The `.ll`/HSACO
gates do catch it (the symbol is in the compared text) and `roll_nd` checks names
explicitly, so no shipped artifact is affected; but "the oracle passed" is not by
itself sufficient grounds to ship a rolled recipe.

### `.ll` parity overstates what is shippable — P1

`parity_matrix` reports 46/46 on both architectures, but that measures *agreement
between engines*, not *compilability*. Pushing the same kernels through comgr
shows a large gap on gfx942:

| Arch | `.ll` parity | Reaches HSACO | LLVM fatal error | Refused at Python lowering |
|---|---|---|---|---|
| gfx950 | 46/46 | 45 | 1 | 10 |
| gfx942 | 46/46 | 32 | 14 | 10 |

The 14 gfx942 casualties are real kernels — `batched_gemm`, `grouped_gemm`,
`conv_implicit_gemm`, `conv_implicit_gemm_wgrad`, `flatmm`, `gemm_multi_d`,
`gemm_multi_abd`, `mfma_gemm`, `moe_gemm_fused`, `moe_fused_mega` and others —
whose emitter default config uses gfx950-only MFMA intrinsics. LLVM rejects them
outright:

```
LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.mfma.f32.32x32x16.f16
```

Both engines produce identical IR, and that IR then fails to codegen. Parity is
genuine; shippability is not. The 10 "refused at Python lowering" cases are
correct behavior by contrast — gfx1151/gfx1201 WMMA kernels declining to lower
for an MI-class target, caught cleanly as `NotImplementedError`.

Two consequences for the acceptance plan. First, the honest headline for the
concrete path is "45 on gfx950, 32 on gfx942 byte-identical as *binaries*", not
46/46. Second, a CI gate asserting only `.ll` equality would stay green through
a change that made every gfx942 kernel uncompilable, so the gate should carry
kernels to HSACO.

### One kernel exhausts host memory in LLVM — P1 (pre-existing)

`moe_fused_mega_fp8` cannot be compiled on either architecture. Attribution
matters here, so it was isolated: building the `KernelDef` and lowering through
the **pure Python stack**, with no portable-IR code involved, produces a modest
1,784-line / 97 KiB `.ll`, and comgr at `-O3` then consumes unbounded memory:

```
built KernelDef ok
PYTHON lower ok: 1784 lines, 97 KiB of .ll
now comgr -O3 (pure python stack, no portable-IR involved)...
LLVM ERROR: out of memory
```

Observed at ~1.5 TB resident before being capped. **This is a pre-existing
defect in the Python stack and its interaction with the LLVM backend, not a
portable-IR regression** — the replay path never gets a chance to run. It is
recorded here because it is a live hazard for any CI job that compiles the full
kernel set: uncapped, a single kernel can take down the runner. Any HSACO gate
should set `RLIMIT_AS` and isolate each compile in a subprocess.

That isolation is needed regardless: LLVM reports fatal errors through
`abort()`, so an unsupported intrinsic or a failed allocation kills the whole
process rather than returning an error. A sweep that compiles many kernels in
one process stops at the first bad one and loses every result after it.

### No HSACO cache — P1

The JIT path recompiles from CBOR on every request. comgr is 1.8–3.2 ms and
dominates the small-kernel budget, so a persistent cache keyed by
`(family, spec values, arch)` is the single highest-leverage runtime
optimization and is currently absent.

### Bundle key hygiene — P2

Kernel names embed spec values that the recipe parametrizes over, so a rolled
recipe can carry a name asserting a shape it no longer fixes. The family-key
concept exists in `rocke.bundle/v1` but nothing enforces the separation between
the stable family key and the varying spec values.

## Gap analysis

| # | Gap | Sev | Evidence | Fix |
|---|---|---|---|---|
| 1 | Shared library does not link | P0 | Reproduced this session from a clean build | Delete duplicate `build_id.cpp` + header; drop `--allow-multiple-definition` |
| 2 | No CI wiring | P0 | 41 workflows, 0 reference rocke | Add parity matrix + roll/HSACO gate to PR CI |
| 3 | Rolled path ungated | P0 | Was untested before this session | Land `roll_hsaco_parity` as a CI gate |
| 4 | Docs assert gates that do not run | P0 | Scaling plan lines 334–336; stale `run_*_demo.sh` | Correct to intent, not status |
| 5 | Tile/warp geometry rolls for 1 of 4 families | P1 | 7 rolled axes, 1 geometric; 8 refusals | Extend roller; accept per-geometry recipe families where non-affine |
| 5b | `.ll` gate hides uncompilable kernels | P1 | gfx942: 46/46 `.ll` but 32 reach HSACO | Carry CI gates through to HSACO |
| 5c | `moe_fused_mega_fp8` exhausts host memory in LLVM | P1 | Reproduced on the pure Python stack | Pre-existing; cap `RLIMIT_AS` and fork-isolate compiles in CI |
| 6 | No HSACO cache | P1 | comgr 1.8–3.2 ms per compile, unavoidable today | Cache on `(family, spec, arch)` |
| 7 | Provider `ArtifactStore` path not integrated | P1 | Marked "pending integration" in the plan | Wire recipe expansion behind a C-JIT flag |
| 8 | Bundle key hygiene unenforced | P2 | Names embed parametrized spec values | Enforce family key vs spec separation |
| 9 | Non-linear axes unsupported | P2 | `block_n`, conv `C` refusals | Richer intexpr grammar, or separate families |
| 10 | Oracle cannot see kernel names | P2 | `roll()` + `recipes_equiv` accepts a recipe emitting a wrong symbol | Compare `kernel_name_fmt` in `recipes_equiv` (`roll_nd` already checks it) |
| 11 | Tuning axes need opcode selection, not affine fits | P2 | `tile_m` non-monotonic 90 → 81 → 114 | `static_if` over vector width / opcode (P3 in the scaling plan) |

## Success criteria and acceptance test plan

Proposed definition of done for the implementation story. Criteria 1–5 are
enforceable in CI today given gaps 1–3; 6–7 need the cache and provider work.

| # | Criterion | Gate | Status |
|---|---|---|---|
| 1 | Clean-tree build produces a loadable `librocke.so` with no linker overrides | CI build job | **Failing** (gap 1) |
| 2 | Concrete recipe path byte-identical `.ll` vs Python, all kernels × arches | `parity_matrix` | **Passing** 46/46 × 2 |
| 2b | Concrete recipe path byte-identical **HSACO**, every kernel that compiles | HSACO sweep | **Passing** 45/45 gfx950, 32/32 gfx942 |
| 3 | Rolled recipe path byte-identical HSACO, incl. held-out points | `roll_hsaco_parity` | **Passing** 22/22, 8 held-out |
| 4 | Rolled `.ll` byte-identical to Python | `roll_hsaco_parity` | **Passing** 22/22 |
| 5 | Device numerics match the Python-built kernel | `gpu_replay` | Passing previously; not re-run here |
| 6 | Cold JIT within budget on a cache miss; cache hit avoids comgr | new bench | **Not implemented** (gap 6) |
| 7 | Provider serves a recipe-backed kernel behind the C-JIT flag | integration test | **Not implemented** (gap 7) |
| 8 | Roller never emits an incorrect recipe; refusals degrade to concrete | oracle in `roll()` | **Holding** across all probes, with the kernel-name caveat above (gap 10) |
| 9 | One recipe covers a family's non-reduction axis **cross product** | `roll_nd_coverage` | **Passing** 4/4 families, every grid point + holdout |

Two notes on the plan. Criteria 3 and 4 now both hold, so CI can gate the rolled
path on a `.ll` checksum and treat the HSACO compile as the slower confirming
run rather than the only usable signal. Criterion 8 is a safety property rather than a coverage target — the
roller's value is that a gap in it is a compression loss, so CI should record
refusals as informational and only fail on a *verified-wrong* recipe.

Coverage targets to agree before implementation: which kernel families must roll
(all four demonstrated here, or the full `SUPPORT_MATRIX.md` set), and whether
tile/warp geometry rolling is required for release or deferred given gap 5.

## Reproducing

```bash
cd dnn-providers/hip-kernel-provider/rocke/platform
cmake -S . -B /tmp/rocke_online/core -DCMAKE_BUILD_TYPE=Release -DROCKE_BUILD_PYENV=OFF
cmake --build /tmp/rocke_online/core --target rocke_core -j"$(nproc)"
# --allow-multiple-definition is the gap-1 workaround; remove once the duplicate is deleted
c++ -shared -fPIC -Wl,--allow-multiple-definition \
    -Wl,--whole-archive /tmp/rocke_online/core/librocke_core.a -Wl,--no-whole-archive \
    -lm -o /tmp/rocke_online/librocke.so

cd ..
export PYTHONPATH=platform/python:library
export ROCKE_ONLINE_LIB=/tmp/rocke_online/librocke.so
python3 -m rocke.portable_ir.drivers.roll_hsaco_parity     # rolled path, HSACO
python3 -m rocke.portable_ir.drivers.parity_matrix --flavor llvm22   # concrete path, .ll
```
