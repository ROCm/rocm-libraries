# Attention dispatcher — agent guide

This folder owns the **path-level dispatch** for unified attention: which kernel
family (2D tiled prefill vs 3D split-KV decode) and which specialized candidate
handles a given `AttentionRequest`.

## What the dispatcher decides (and what it does NOT)

**Decides:** kernel path (`"2d"` or `"3d"`), candidate name, algorithm tag,
and spec identity `(path, head_size, block_size)`.

**Does NOT decide, on the unified path:** CTA geometry (`num_warps`, `tile_size`),
`num_segments`, `waves_per_eu`, or any other performance knob. Those live in
`builders/common/attention_spec_builder.py` and
`kernels/common/attention_unified.py`. The parity identity with C++ is
`(path, head_size, block_size)` only — C++ reads `num_segments` as a parameter
passed from Python, it does not recompute it.

**Standalone candidates are a bounded exception.** A candidate that owns its own
kernel module builds that kernel's own spec here, tuning included:
`gfx950.py::_dense_spec` resolves `block_n`, the persistent decision and the CTA
count, and `gfx942.py::_dense_spec` resolves those plus `waves_per_eu`. Those specs
are consumed only by their own builder and never enter the C++ parity identity. One
rule governs the exception: **any value the kernel bakes into its `kernel_name` must
be resolved from the kernel's own policy function**, not pinned here — `gfx942.py`
calls `kernels.gfx942.attention_dense._tuned_waves_per_eu` for exactly that reason.
A number pinned in the factory drifts away from the policy, the name tag and the
compiled binary then disagree, and the name-keyed launcher cache serves the wrong
HSACO.

## Candidate registry — priority table

| priority | candidate | declared arches | module | scope |
|---|---|---|---|---|
| 3 | `attention_gfx942_dense` | gfx942 | `gfx942.py` | bf16/fp16 D64/D128 dense prefill, default **and** persistent grids (opt-in only) |
| 3 | `attention_gfx950_dense` | gfx950 | `gfx950.py` | bf16/fp16 dense persistent prefill (opt-in only) |
| 5 | `attention_gfx942_dense_pipe` | gfx942 | `gfx942.py` | fp16 2D prefill flash |
| 5 | `attention_gfx950_d256` | gfx950 | `gfx950.py` | bf16 D256 2D prefill |
| 5 | `attention_gfx1250_wmma` | gfx1250 | `gfx1250.py` | fp16 WMMA FMHA forward (opt-in only) |
| 5 | `attention_d256_decode` | gfx942, gfx950 | `generic.py` | bf16 D256 3D decode |
| 10 | `attention_unified_2d` | all | `generic.py` | generic 2D prefill fallback |
| 10 | `attention_unified_3d` | all | `generic.py` | generic 3D decode fallback |

Lower priority number = higher precedence. Generic candidates (10) remain the
fallback for everything a specialized candidate does not claim.

Three candidates are **opt-in only** and never win under `algorithm="auto"`:
`attention_gfx942_dense`, `attention_gfx950_dense` and `attention_gfx1250_wmma`.
Registering a kernel makes it reachable; making it an arch's default is a
separate decision that wants benchmark evidence, so none of them silently
displaces the unified path its arch routes to today.

**Tier 3 is reserved for opt-in candidates.** Because they outrank every other
tier, that opt-in check is the only thing keeping them off the default path — a
tier-3 candidate whose `support()` forgets it would silently claim all traffic
for its arch. `Capability` cannot express the check (it constrains the request's
*selector*, not its shape), so it stays in the predicate and every tier-3
candidate has to carry it.

## Layout

One module per architecture, each owning its candidates and exporting a
`register(registry)`; `__init__.py` holds the request type, the registry
assembly, and the entry points. `common.py` holds what every candidate shares
(request, spec, gates) and imports none of the arch modules, so assembly order
does not matter. `generic.py` is for candidates declaring more than one arch --
the two unified paths and `d256_decode` -- which is not the same as portable:
each still lists its arches explicitly, because there is no family wildcard.

The two generic candidates declare every known arch on purpose: they select a
*path*, and `attention_unified` picks the concrete backend downstream from the
running device (wave64 MFMA on gfx942/gfx950, wave32 WMMA on gfx1250, the
arch-neutral scalar kernel elsewhere). They are an explicit, tested exception to
the wave-size consistency invariant.

## How to add a new specialized candidate

Follow the `_make_d256_decode_candidate()` pattern in `generic.py`:

1. **Add a cohort predicate** in `kernels/common/attention_unified.py` —
   a pure function of `UnifiedAttentionProblem` that returns `True` for the
   target shape family. This is the single source of truth for membership;
   import it lazily inside the factory to keep `dispatch/` arch-neutral.

2. **Declare a `Capability`** on the candidate: the explicit `arches` it serves,
   its `dtypes`, any head-size or block-size bounds as `ShapeRange`, and the
   `supports_features` it can handle (`causal`, `sliding_window`, `sinks`).
   This is required — `register()` rejects a candidate without one. Anything the
   capability declares must not be re-checked in `support()`.

3. **Add a factory function** `_make_<name>_candidate()` in the module for the
   arch it serves (`generic.py` if it declares more than one).
   The `support()` closure carries only what is left, in order: request errors →
   `_selector_matches` → `supports_native_unified_attention` → cohort predicate →
   path check (`select_path() == "2d"` or `"3d"`).

4. **Register** it from that module's `register(registry)`. A new arch module
   also needs one line in `__init__.py` to join the assembly loop.

   Point `build` at the real builder if the candidate has one. The unified
   paths do not: they return an `AttentionSpec` naming a *path*, and no builder
   consumes that. A standalone kernel like `attention_gfx1250_wmma` returns its
   builder's own spec instead, which is why it can declare `build`, a real
   grid, and a real signature where the unified candidates declare none.

5. **Add CPU-only dispatch tests** in
   `tests/dispatch/attention/test_<name>_wiring.py`. Cover: registration,
   spec_id, algorithm, priority ordering, rejection gates (wrong arch/dtype/
   cohort/path), and routing for each target arch. Use `_PinnedArch` context
   manager to avoid GPU dependency. Call `candidate.admits(req)`, not
   `candidate._supports(req)` — the latter skips the capability prefilter and is
   no longer a complete verdict, which is what the underscore is there to say.
   The coverage invariants in
   `test_declared_coverage.py` apply to the new candidate automatically.

6. **No C++ changes needed.** The dispatcher is Python-only.

## Engine-level selection: the ranker seam

`dispatch_attention(req, *, ranker=None)` chooses among the candidates that
support `req` in two stages:

1. `CandidateRegistry.supported(req)` filters to eligible candidates, already
   sorted ascending by `(priority, name)`.
2. A **ranker** — `Callable[(request, supported) -> reordered]` — reorders them
   best-first; `dispatch_attention` takes `ranked[0]`.

When no ranker is supplied, the named default `priority_ranker` (in
`attention.py`) is used: it is an identity pass, so the registered priority order
wins (behavior-preserving). A heuristic ranker is a **drop-in replacement** that
scores candidates against problem metadata (or offline benchmark data) and sorts
by score — no change to the registry or candidates. Safety invariant enforced by
the registry: a ranker may reorder or drop candidates but **cannot introduce one
the request does not support** (raises `ValueError`).

This is the **engine-level** half of the intended hierarchical design. The
**per-engine** half is each candidate's `select_spec` (today thin: it records
`(path, head_size, block_size, …)` and defers geometry). A future phase can move
the `_select_*` / `_enable_*` heuristics from
`builders/common/attention_spec_builder.py` into per-engine `select_spec`s so an
engine owns both "am I eligible?" and "how do I tune myself."

Coverage: `tests/dispatch/attention/test_ranker.py`.

## Additive registration (open/closed)

Adding a candidate must not change any existing candidate's `supports()` verdict
or `select_spec()` output. This is an executable invariant in
`tests/dispatch/attention/test_additive_registration.py`: it seeds a **fresh**
`CandidateRegistry` from `attention_candidates()`, registers a throwaway example
engine into that copy (never the shipped singleton), and asserts every
pre-existing candidate's behavior is byte-identical with and without it.

## Per-engine spec_fn (geometry ownership)

The long-run goal is the GEMM shape: each engine builds its own kernel spec
(`platform/python/rocke/dispatch/gemm/bf16_rcr.py` — one `_spec_*` per candidate),
instead of one shared `_tiled_spec_from_problem` cascade of `if` branches.

Migration is incremental — one cohort at a time.

1. **Extract** the cohort's branch from `_tiled_spec_from_problem`
   (`builders/common/attention_spec_builder.py`) into a named
   `_spec_<cohort>(problem)` — a self-contained builder (resolves its own arch /
   spec class). Pure move: byte-identical, no value change.
2. The cascade branch **delegates** to it (`return _spec_<cohort>(problem)`), so
   the shared function shrinks by one branch.
3. If the cohort has a matching dispatch candidate, that candidate **documents
   ownership** of the `spec_fn` (a docstring linkage). Some cohorts have no
   candidate yet (they ride the generic `unified_2d`) — those spec_fns are
   ORPHANS awaiting a future engine; note that in the spec_fn docstring. Either
   way, geometry stays in the **builder layer** — do NOT move it into a candidate.
   The dispatcher still decides only `(path, head_size, block_size)`, and the C++
   parity identity is unchanged (see the top of this doc).
4. **Test** byte-identity + non-interference (see the
   `test_per_engine_spec_fns.py` -- table-driven, one entry per cohort), then
   GPU-verify the cohort's arch (kernel name / built spec unchanged vs pre-change).

Migrated so far (all builder-layer spec_fns in
`builders/common/attention_spec_builder.py`; `_tiled_spec_from_problem` and
`_tiled_3d_spec_from_problem` are now clean arch dispatchers):
- `_spec_gfx942_fp16_flash` — owned by `attention_gfx942_dense_pipe`.
- `_spec_gfx942_bf16_flash` — ORPHAN (no dispatch candidate yet; routed via the
  generic `unified_2d`). Needs a future `gfx942_bf16` engine.
- `_spec_generic_2d_non_gfx950` — the non-flash 2D residual for EVERY non-gfx950
  arch (gfx942, gfx1201, gfx1151, ...), built from the shared
  `_base_2d_generic_fields` only. ORPHAN.
- `_spec_gfx950_generic` — the shared `_base_2d_generic_fields` plus the
  gfx950-only schedule tail + the D256 gfx950 fast-route override folded in (kept
  behind the `_kau.` module handle for test-steering). The 2D `_spec_field_names`
  guards are gone -- the per-arch split replaced them.
- `_spec_generic_3d` — the shared gfx942/gfx950 3D split-KV fallthrough (one
  function; the `_gfx942_3d_*` helpers self-gate, so no arch split).

Remaining: gfx1250 (2D + 3D) -- still inline early-returns in both cascades,
DEFERRED (no gfx1250 hardware to GPU-verify this pass). The `_kau.` D256
indirection must be preserved by any code that touches the gfx950 override.

## Multi-engine benchmarking: `attention_sweep_space`

`attention_sweep_space(req)` returns the deduped `select_spec` of every candidate
that supports `req` — the "evaluate multiple engines for one problem" primitive.
It can time 2D/3D paths from the **prefill** harness only; the dedicated decode
benchmarks have no sweep lane. The prefill benches
(`benchmarks/gfx{942,950}/attention/prefill/benchmark_prefill2d_live.py`) consume
it via the opt-in `--variants sweep` lane — a shared helper
(`benchmarks/common/attention_sweep.py:run_sweep`) that times each launched path
the registry offers and records which engine names mapped to it. Contract tests:
`tests/dispatch/attention/test_sweep_space.py`.

Framework-phase caveat: because geometry is deferred (see below), engines that
route to the same launched path collapse to one timed entry. The decode benches
(`benchmark_decode_live.py`) do **not** yet have a sweep lane — they lack a
variant-loop; adding one is a mechanical follow-up reusing the same shared
`run_sweep` helper.

## DEFERRED — production wiring + heuristic selection

The registry is currently **not load-bearing for GPU execution**. Production runs
through `run_unified_attention_torch` (`kernels/common/attention_unified.py`),
which calls `problem.select_path()` + `_tiled_spec_from_problem()` **directly**,
bypassing `dispatch_attention` / `ATTENTION_REGISTRY`. The dispatcher is exercised
only by these CPU tests and the benches' `prod` / `sweep` lanes.

Two later increments make it load-bearing:

- **Thin routing (low risk):** route `run_unified_attention_torch`'s path choice
  through the registry. The 2d-vs-3d decision is already the same pure
  `select_path()` both sides use, so this is byte-identical by construction;
  geometry still comes from `_tiled_spec_from_problem`.
- **Full engine-owned geometry (larger):** each engine's `select_spec` produces the
  tuned spec, absorbing the relevant `_select_*` branches. If C++ selection parity
  is required, geometry parity is the "separate, larger effort" noted in the
  `attention.py` module docstring ("DEFERRED — arch-tuned block geometry").

Heuristic-driven selection (a real scoring ranker) is independent of the above and
is a drop-in via the ranker seam once a scoring signal exists.

## How to tune `num_segments` for a new cohort

See the worked example:
`benchmarks/gfx942/attention/decode/TUNING_D256.md`

## Testing (CPU-only, no GPU)

```bash
PYTHONPATH=library:platform/python python -m unittest discover \
    -s library/tests/dispatch/attention -p "test_*.py" -v
```

All dispatch tests complete in < 1 s.
