# rocKE PR Review Guidelines

The **engineering-judgment** layer for reviewing (and self-reviewing) a rocKE change:
the substantive mistakes that recur in rocKE PRs and need cross-file, cross-engine, or
cross-stage reasoning a generic or purely mechanical review won't catch (see
[Automated use](#automated-use) for how an LLM reviewer with repo context applies them).

This is the **review ("what to catch")** companion to
[`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md), the **authoring ("what to do")** process +
Definition of Done. An author works that DoD; a reviewer confirms it was met and then applies
the judgment below. The two docs are meant to be read together and cross-reference each other.

## Read (and enforce) these first — this doc does not restate them

A review MUST also confirm the change complies with the documents below. They are a
precondition, not an alternative, to what follows.

| Concern | Authoritative doc |
|---|---|
| Authoring process & **Definition of Done** (per change type: new kernel / knob / feature / perf), the local check runner, and the per-arch test matrix | [`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md) |
| Canonical agent-rules index / compliance / hard-invariant entry point (routes to `platform/AGENTS.md`) | [`AGENTS.md`](../../../AGENTS.md) |
| Compliance, export control, hard invariants (and internal-tracker nomenclature) | [`platform/AGENTS.md`](../../AGENTS.md) |
| The two-engine contract: byte-identity, operand/eval order, the golden snapshot, the differential gate as *definition of done*, the `ruff --fix`/F841 emitter hazard | [`platform/dsl_docs/development/invariants.md`](invariants.md), [`platform/dsl_docs/development/engine_contributing.md`](engine_contributing.md) |
| C++ / Python style & formatting | [`style/CPP_STYLE.md`](../../../style/CPP_STYLE.md), [`style/PYTHON_STYLE.md`](../../../style/PYTHON_STYLE.md) |
| Merge policy: branch name, title, test-in-diff, forbidden files, `pre-commit` | [`../../../docs/LIBRARIES_PR_BOT_FAQ.md`](../../../../../../docs/LIBRARIES_PR_BOT_FAQ.md), [`../../../CONTRIBUTING.md`](../../../../../../CONTRIBUTING.md) |
| Start-here routing for the rocKE tree | [`README.md`](../../../README.md) |

### Which sections apply — route by changed path

Applying all five sections to every diff is wasteful and invites speculative findings in
sections with no changed code. Route by what the diff actually touches; compliance applies to
every diff regardless.

| Changed path | Apply |
|---|---|
| `platform/cpp/**`, `core/**`, `helpers/**` | the byte-identity notes below + §3 |
| `**/dispatch/**`, `*selector*` | §2 |
| `library/kernels/**`, `platform/python/rocke/instances/**` | §1 + §3 + docs-discoverability |
| `**/tests/**`, `**/parity_*.py`, `**/*verify*.py` | §4 |
| `**/*.md` | compliance + docs-discoverability |
| *any diff* | compliance ([`platform/AGENTS.md`](../../AGENTS.md)) |

From that DoD ([`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md)), the steps a diff-only read misses —
check each explicitly: **on-GPU numeric** test (not spec-only geometry), **dispatcher
registration**, **end-user visibility** (the support matrix), **docs discoverability**
([`instances/index.md`](../instances/index.md)), **algorithm docs for a new
pipeline** (the note below), and promote-to-`platform` for reusable optimizations. Run the local
check runner green on each affected arch — a separate run
per device, as the numeric lane only covers the visible GPU (see the *Local testing matrix* in
[`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md)).

**Docs discoverability means the algorithm, not just the catalog row.** A *new pipeline* — a
distinct schedule or data-movement strategy, not merely a knob — must be documented in its arch's
builder docs (`library/builders/<arch>/<op>/ALGORITHM.md` for what it computes and *why it is
shaped that way*, plus the `README.md`), per Process D in
[`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md). This is easiest to miss exactly when the pipeline
lands *inside an existing kernel module* — a second self-contained builder/lowering body in the
same `.py` reads like a variant of an already-documented family, so the diff looks doc-complete
while a reader who did not write it cannot find what the new schedule does or when it is selected.
The smell: a diff that adds a self-contained builder or lowering body but touches no
`builders/**/{ALGORITHM,README}.md`.

Four review-time notes that sit *on top of* the byte-identity contract and are easy to get wrong:

- **A green representative-golden is not cross-engine proof.** The golden is a Python-drift
  snapshot; cross-engine agreement is the separate differential gate ([`check_byte_identity.py`](../../tools/check_byte_identity.py)),
  which builds the C++ engine. Adding a golden entry never proves the two engines agree.
- **Green fast-CI does not prove the C++ engine compiles or matches** — it is frequently not
  built in the default job. Review the C++ side by hand, and run the differential gate when a
  shared layer moved.
- **Guard new logic so already-covered shapes emit identical IR** (`if <special-case> else
  <original>`). Adding an unconditional op to a hot path changes the IR for every shape and
  breaks byte-identity for the common case; the golden/gate staying green is then the proof the
  change is a no-op for covered shapes.
- **Grade a divergence by who consumes it now.** A shared-layer change that landed in only one
  engine is real debt to resync; whether it is a *live* break or a *latent* one depends on
  whether the consuming path is built and served today. Establish that with evidence (trace the
  callers, confirm the engine is built), not assumption — the severity follows from the answer.

---

## 1. Never silently narrow coverage

A change can be perfectly correct and still be a regression if it quietly routes a real shape
onto a slow or unsupported path. Silence is the danger: the output is right, so nothing fails,
and the gap is invisible until someone profiles production.

- **Classify every new rejection or support gate.** Is it a real *hardware limit* (LDS budget,
  thread cap, unsupported dtype) or an *implementation shortcut* (divisibility, power-of-two,
  even-shape)? Shortcuts are `NOT_YET_IMPLEMENTED`, not permanent contracts.

  ```python
  # Bad: a convenience constraint that reads like a hardware rule and silently
  #      sends non-divisible GQA ratios to the slow path forever.
  if num_q_heads % block_m != 0:
      return False, "unsupported"          # why? forever? nobody can tell

  # Good: name the kind of limit, and leave a path forward.
  if num_q_heads % block_m != 0:
      return False, "NOT_YET_IMPLEMENTED: pad-block-q for non-divisible GQA (see #<issue>)"
  ```

- **A preflight support gate must reject exactly what the spec constructor rejects**, per arch.
  If the cheap gate admits a shape the constructor later refuses, you get a cryptic late
  failure; if it rejects one the constructor would accept, you get a needless fallback.
- **Fallbacks must be loud or measured, never silent.**
- **Gate scope must equal verified scope.** If a gate *opens* a path for a broad set of shapes,
  the configs you did not measure are unverified — test them or scope the gate down to what you
  proved. "Routes to the fast path" is not "produces correct numbers."
- **`is_valid_spec` / `supports_*` is functional-only.** It answers one question — *can this
  kernel be emitted and run correctly?* Reject only for emit-impossible/undefined configs
  (impossible tile/atom geometry, LDS/register budget, missing MMA-op / dtype / arch mismatch,
  mutually-incompatible flags, a boundary mode with no safe IR path). Keep *opinions* out —
  performance heuristics ("prefer d128"), benchmark-coverage gaps, policy ("prod only enables
  GQA-8"), and re-checking routing the dispatcher already does all belong in dispatch/selectors,
  not the validator. State the hardware/algorithm invariant in `reason`, not the workload.
  (Detail + the reject / don't-reject lists: [`authoring_model.md`](../architecture/authoring_model.md) §2.)

*(Authoring side: the Core DoD in [`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md) requires an
on-GPU **numeric** test — "not spec-only geometry, that passes green and lets a regression
ship." This section is what a reviewer checks that requirement actually caught.)*

---

## 2. Dispatcher & spec integration

The dispatcher picks a kernel by building a **spec** from the problem and routing on it.
Anything that affects the built kernel must live *on the spec*, and everything that decides
*which* spec is chosen must stay internally consistent. This is the richest source of subtle,
hard-to-catch rocKE bugs.

- **Knobs and features live on the `KernelSpec`, not in `os.environ`.** A new tuning axis must
  be a spec field — with validation, a selector, a `kernel_name` tag, and a cache-key entry — so
  the dispatcher can route and cache-key on it. An environment variable is invisible to the
  routed spec, cannot participate in candidate selection, and is not captured by the launcher
  cache key.

  ```python
  # Bad: a knob the dispatcher can never see or cache-key on.
  if os.environ.get("ROCKE_USE_FAST_PATH"):
      spec.num_warps = 2

  # Good: a spec field the selector sets; routing and cache-key see it.
  spec.num_warps = _select_num_warps(problem)   # validated, tagged, in the cache key
  ```

- **The cache key must stay faithful to what was actually built.** If a hot path recomputes the
  launcher key from the general heuristics but an override changed the real spec elsewhere, the
  key lies about the artifact and you get wrong-kernel reuse. Key on the override *predicate*
  (one bit that flips exactly when the override applies), not on duplicated flags — and never by
  folding the override back into the regenerable heuristics.
- **A cohort is one predicate, shared.** The condition that defines a cohort must be a single
  function called by *both* the candidate's `supports()` and the spec-builder override — never
  two hand-copied condition lists that can drift apart.

  ```python
  # Bad: the same cohort spelled out twice; edit one, forget the other -> the two
  #      disagree about what the cohort is.
  def supports(req):    return req.arch == "gfx942" and req.head_size == 64 and req.use_sinks and ...
  def _spec_from(prob): if prob.head_size == 64 and prob.use_sinks and ...: tune()

  # Good: one predicate, both call it.
  def _sink_cohort(p): return p.head_size == 64 and p.use_sinks and p.sliding_window == 0 and ...
  def supports(req):    return req.arch == "gfx942" and _sink_cohort(req)
  def _spec_from(prob): if _sink_cohort(prob): tune()
  ```

- **Route a cohort by keeping spec-builder, launch-meta, and cache-key in agreement.** A
  spec-only override is not enough: if the kernel is built for one geometry but launched with
  another, it produces garbage even though it compiled.
- **A kernel is not "done" until it is reachable through the dispatch that serves it.** Register
  it and exercise it *through* the dispatcher in a harness — a hand-built spec in a benchmark
  proves kernel coverage, not that routing selects it, and not that the served path uses it.
- **Fix the chooser, not the artifact.** When a *selector* produces the wrong spec, change the
  selection layer, not the kernel emitter. Changing which spec is chosen leaves every other
  shape provably unaffected; editing the kernel risks all of them.
- **Hand-pinned overrides live *above* the machine-regenerable heuristics.** A config a human
  found and wants kept must not live inside a selector table an autotuner/sweep rewrites — it
  gets clobbered on regen (or silently depends on the sweep covering that cohort). Keep it as an
  explicit override layer on top of the heuristics.

*(Authoring side: dispatcher registration is required by
[`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md) §A "Dispatcher wiring" and §C "Dispatcher feature
list"; new knobs go through its Process B. Its Process B lets an unwired knob stay env/flag-gated
"until a sweep proves a win" — treat that as a **temporary diagnostic override**, not a shipping
knob, per the spec-field rule above.)*

---

## 3. Single-source & blast radius

- **Name a repeated concept once.** The same predicate pasted at N sites is a latent
  correctness bug — the day someone updates one site and forgets another, the code is subtly
  wrong only on the missed path. One named helper makes a partial update impossible. (This is
  the same failure as the duplicated-cohort case above.)
- **One builder per family; no bespoke shape kernels.** Ship one `build_<family>(spec)`; a new
  module/function/branch keyed to a single shape class — `build_attention_sq8192.py`, or
  `if spec.seqlen_q == 8192: return build_other(...)` inside a body — is the smell. Shape-specific
  behavior belongs in spec flags or dispatch/heuristics; a harness may hardcode shapes but must
  call the same public `build_*` / `run_*_torch` the dispatcher uses, never a forked IR path.
  (Detail: [`authoring_model.md`](../architecture/authoring_model.md) §1.)
- **Name by operation + algorithm, not workload — and hoist a hard-coded shape to a classified
  constant.** A builder/module/spec-class name states the *operation + tiling*
  (`build_unified_attention_2d_tiled`), never runtime geometry (`build_attention_d128_sq4096`,
  `prefill_modelA_sq4096`, `_build_..._d256_lean`). `kernel_name()` **may** carry codegen knobs that
  change emitted IR (`dtype`, `head_size`, `block_n`, tile/warp, `ragged`, `persistent`) but
  **not** workload identity (`batch`, `seqlen`, `M/N/K`, model codenames) — those are `b.param`
  runtime args. When a body still pins one shape (e.g. `head_size == 256`), lift it to a named
  constant whose comment classifies it — **validation gate** (widen once another shape is proven)
  vs **algorithmic limit** (the schedule bakes it, e.g. a per-thread stride derived from that
  dim) per §1 — and derive extents from `spec` where the schedule allows. Reviewer cue: a
  workload dimension in a *name* (vs a codegen knob in `kernel_name()`) is the tell; check the
  name and the guard tell the same story. (Detail:
  [`authoring_model.md`](../architecture/authoring_model.md).)
- **Arch-independent logic belongs in `common/`,** not forked across `gfx942/` and `gfx950/`.
  Forked copies drift; fix one, miss the other. Conversely, when logic *is* arch-specific, cover
  every arch you enabled — don't fix on the arch you happen to be sitting on and assume the
  other inherits it.
- **Make the change easy, then make the easy change.** If a one-line behavioral change requires
  edits in twenty files, the code is not shaped for the change — a refactor-first step is
  missing. Prefer a *thought-out structural change that makes life easier for the next author*
  over an ad-hoc patch that fights the current structure. Land the refactor as a proven no-op
  first (the differential gate stays green), then the real change lands small and reviewable.
- **Judge blast radius honestly.** A large file count can be *inherent* (the same logical change
  across two engines × two arches × several stages) — that is fine — or *inflated* (unrelated
  fixes, regenerated data blobs) — split that out. Unrelated fixes belong in their own PR.
- **Removing dead code means removing the symbol *and* every stale reference** — comments,
  docstrings, flag names — in the same commit. A comment pointing at a deleted symbol misleads
  the next reader.
- **`platform` must not depend on `library`.** The DSL engine (`platform/`) knows nothing about
  specific kernels; `library/` sits on top of it. A `platform` test that imports `library`
  creates a dependency cycle and leaks kernel concerns into the engine. Keep the arrow one-way;
  tests for library code live under `library/tests/`. (This is the dependency *direction* — distinct
  from the Python↔C++ layer twinning the invariants doc governs.)

*(Authoring side: the one-way `library → platform` rule is invariant #2 in
[`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md); promoting a reusable optimization down to
`platform/` is its Process G.)*

---

## 4. Verification honesty (green ≠ correct)

The byte-identity gate proves the *engine* is correct (the two backends agree). It does **not**
prove a *kernel* is numerically correct.

- **Parity is not correctness.** Both engines can emit the *same wrong* output and pass every
  parity gate. Numeric correctness needs an independent oracle — an fp32 reference on real
  silicon (see [`platform/AGENTS.md` §"Hardware requirements for numeric tests"](../../AGENTS.md)), not a rocKE-vs-rocKE comparison. When
  the incumbent has no kernel for the case, write the fp32 reference in torch; do not settle for
  "two rocKE configurations agree."
- **A skip is not a pass.** A harness that *skips* when a dependency isn't built (a missing
  binding, an unbuilt engine, a `skipUnless`) reports green without ever running. Confirm the
  check actually executed in an environment where the thing under test is built.
- **A new behavior needs a test that would have failed before the change.** Asserting an
  always-present substring, or adding an empty test file to satisfy a gate, is coverage theater.
- **Get load-bearing numbers from the tool, not arithmetic.** LDS budgets, occupancy, CU counts
  drive design decisions; a hand-derived number off by a factor flips the conclusion. Read
  ground truth from comgr / the profiler.
- **Classify a bottleneck with dynamic hardware counters, never static instruction counts.** An
  opcode histogram shows instruction *mix*, not where *cycles are lost* (bank conflicts, memory
  latency, scheduler bubbles are runtime phenomena). A wrong diagnosis yields the wrong
  optimization.
- **Lock performance provenance.** Label the device (SKUs of one arch differ in CU count, so
  absolute microseconds are device-specific), make the *ratio* the headline, and measure on the
  path that actually ships — not a re-transcribed wrapper.
- **Absolute numbers are internal; a relative factor is publishable.** Raw measured figures
  (TFLOP/s, GB/s, tokens/s, latency in ms, absolute throughput) are never public — they live
  only in the internal, access-controlled perf-data store. A **relative** factor that carries
  no absolute figure — `1.2x` lift, `0.9x` regression, "~15% faster" — **may** appear in the PR
  description or commit message, and is the *preferred* way to convey impact in public
  artifacts. So a reviewer asks the author for the relative delta, never a raw number; flagging
  a bare `1.2x` as a compliance breach is wrong. (Policy: the Public Data Policy, "Performance
  numbers: raw vs. relative"; `platform/AGENTS.md` §Compliance.)

*(Authoring side: [`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md) provides the single check runner,
`tools/run_checks.py`. When a change leans on it being green, confirm the **on-GPU numeric lane
actually ran** — a device-less run reports `ALL PASSED` with the numeric lane skipped, and a
skip is not a pass.)*

---

## 5. Reusable design principles

These apply well beyond the case that surfaced them.

- **Measure the space, not just the selection.** A benchmark that runs only the production
  dispatch is blind to its own routing mistakes. The gap between *best-achievable* (a sweep) and
  *shipped* (the dispatch) is the routing bug, quantified — so measure configs the chooser did
  not pick.
- **Repair > loud-reject > silent-slow > cryptic-fail.** Given an infeasible input, prefer to
  repair it deterministically; else reject with a categorized reason; never fall silently to a
  slow path or die with a cryptic downstream error.
- **Coverage = union(curated, generated); the rest is a blind spot.** Anything in neither the
  curated menu nor the generator's output is unmeasured and unknown — closed upstream by
  hypothesis (hardware reasoning, profiling a bottleneck), not by re-running the same bench.
- **Separate "what ships" from "what's achievable."** Keep both per-shape figures — in the
  internal perf-data store — so their delta tells a kernel team whether the next win is a
  faster kernel or better routing; the public artifact carries only the relative delta, not
  the raw pair.
- **A baked heuristic is a snapshot that drifts.** Offline-tuned winners frozen into static
  selector tables go stale as hardware, shapes, and the compiler change. Be explicit about
  whether you shipped a snapshot or a live search, and keep a reproducible path to regenerate
  the snapshot rather than hand-patching it forever.

*(Authoring side: the mandatory **Step 0 exhaustive lever sweep** — "measure the space" as a
required authoring step — is Process C/D in [`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md).)*

## Automated use

This doc is dual-audience: a human reviewer, and an LLM reviewer prompted with it. The rules are
the same for both — what differs is *how much* each can check. When this file backs an automated
reviewer, that reviewer follows the tiering, severity ladder, and output contract below.

### What is checkable, and by whom

| Tier | Examples | Who can check |
|---|---|---|
| From the diff alone | test-in-diff, an `os.environ` knob, a workload size in a name, a model/customer name (compliance) | an agent, from the patch |
| With repo context | duplicated predicate/cohort, missing C++ twin, preflight-vs-constructor gate mismatch, an arch fork of `common/` logic | an agent, with a grep/read across files |
| Human or CI only | on-GPU numeric parity, whether the C++ engine builds and matches, bottleneck class, occupancy/LDS budgets | a person, a built C++ engine, real silicon, a profiler |

An automated pass owns tiers 1–2 and must **defer, not guess,** on tier 3 — flag "needs on-GPU /
built-C++ / profiler verification," never assert a tier-3 result from static text.

### Severity ladder

| Severity | Triggers |
|---|---|
| Critical | a compliance breach (a raw/absolute perf number — TFLOP/s, GB/s, latency, absolute throughput — or a customer/model name in-repo, PR, or commit; a **relative** factor like `1.2x`/`0.9x` is allowed), silent coverage narrowing, a cache key that lies about the built artifact |
| High | a missing on-GPU numeric test, a knob in `os.environ` instead of on the spec, a *live* cross-engine divergence on a served path |
| Medium | a duplicated predicate/cohort, inflated blast radius, a preflight/constructor gate mismatch |
| Minor | naming (a workload token in a name), a stale reference to deleted code, a docs-discoverability gap |

### Output contract

- **Cap the findings.** Report only the few that survive verification; an unbounded list reliably
  buries the two real findings under a dozen speculative ones.
- **One finding per comment, anchored on a changed line.**
- **Every finding is `problem → evidence (file:line) → fix`.** No evidence, no finding.
- **Post nothing if nothing survives verification.** Silence is a valid — and common — result.

---

*Scope: this is judgment guidance, deliberately not a mechanical checklist. The authoring
**Definition of Done** and the local check runner live in
[`KERNEL_AUTHORING.md`](../../../KERNEL_AUTHORING.md); the mechanical gates (branch/title/test-in-diff/
forbidden files/formatting) and the hard invariants are enforced by the PR bot, `pre-commit`,
and the docs linked at the top. When a recurring mistake here becomes mechanically checkable,
move it into a lint/hook — and keep it listed here as context the automated reviewer still
reads (see [Automated use](#automated-use)).*
