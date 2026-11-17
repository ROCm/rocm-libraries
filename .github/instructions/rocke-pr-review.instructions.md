---
applyTo: "dnn-providers/hip-kernel-provider/rocke/**"
excludeAgent: "cloud-agent"
---

# rocKE PR review rules (Copilot code review)

Apply these rules to changed lines under `dnn-providers/hip-kernel-provider/rocke/`.
Each rule says what to flag and what not to flag. Name the rule in your comment.

## Rule 1 — No tracking IDs, PR numbers or GitHub handles in code

- Flag Jira-style IDs (`AICK-1234`, `SWDEV-123`), PR or issue numbers (`#10583`,
  `PR 10583`) and GitHub handles (`@user`) in source code, comments, docstrings, string
  literals, kernel names, test names and test file names.
- Suggest a descriptive reference instead: what the code does or guards against.
  Tracker IDs belong in the commit message and PR description only.
- Do not flag `#` used as a comment marker or in prose such as "number of batches".

## Rule 2 — No absolute performance numbers in code

- Flag measured latency, throughput, TFLOP/s, MFU or bandwidth values (for example
  `<N> us`, `<N> TFLOP/s`, `<N> TB/s`) in comments, docstrings or string literals.
  Comments stay qualitative ("faster for long sequences").
- Do not flag tile or shape sizes (`16x32`, `(4, 16, 8)`, `D128`), register or byte
  counts, tolerances, or other numbers that are not measured performance.

## Rule 3 — Kernel and dispatch knobs belong on the KernelSpec, not in environment variables

- Flag new `os.environ`, `os.getenv` or `getenv` reads in library, dispatch, selector,
  builder or kernel code that change which kernel or configuration is selected or built.
- The knob belongs as a field on the spec (validator, selector, kernel-name tag and cache
  key) or as an explicit caller argument.
- Do not flag environment reads in tests, benchmarks or standalone tools that only
  configure the harness and never change what the dispatcher selects.

## Rule 4 — At a trust boundary, reject invalid input; never normalize it

- Flag `getattr(obj, "field", default)`, masking (`x & 0xFF`), clamping (`min`/`max`)
  or coercion applied to a request, spec or IR field that decides behavior, when it turns
  missing or malformed input into a different valid value instead of raising.
- In the comment, say what the expression rewrites and what the caller gets instead of
  an error. Validate at the last gate before the value is used.
- A refactor must keep failure modes: if the old code raised on missing or invalid input,
  the new code must too.

## Rule 5 — Python and C++ implementations must agree

- Attention selection exists in two engines and both are live: Python
  (`library/kernels/common/attention_unified.py`, the `_enable_*` and `_select_*`
  functions) and C++ (`platform/cpp/instances/common/attention_unified_selectors.cpp`,
  where each function is marked `/* Python: <name>(problem). */`).
- For changes in either engine, flag disagreement with its counterpart in selectors,
  lowering, atoms or instances.
- Do not flag families with no C++ twin, such as GDN and KDA under
  `library/dispatch/gdn/`: Python-only changes there are expected.

## Rule 6 — Anything that changes the built kernel must reach its identity

- The compiled-binary key is derived from the spec (`KernelId.compile_key` hashes the spec
  fields). Flag two reachable configurations that emit different IR under the same
  compilation key. Identify the value that differs but is absent from the key.
- Where `kernel_name()` encodes codegen knobs, flag a new IR-changing spec field that is
  not reflected there.
- For a cohort override (a predicate that switches a spec flag on for some problems),
  flag a spec builder, launch metadata (grid/block) and cache key that do not all read
  the same predicate.

## Rule 7 — A new kernel or flag must be reachable on the shipped path

- Flag a new kernel builder, spec flag or tuned configuration that no selector,
  `_enable_*` predicate, spec builder or dispatch candidate turns on, so only a hand-built
  spec in a test or benchmark exercises it.
- Flag tests or benchmarks that prove a new kernel only by constructing its spec directly;
  ask for a test that reaches it through the dispatcher or selector for a real problem.
- Flag prose claiming production attention goes through the Python dispatcher:
  `run_unified_attention_torch` selects specs itself and does not call
  `dispatch_attention`.

## Rule 8 — A change inside shared structure reaches every member

- Flag a change placed in a branch, set, table or helper shared by several archs
  (gfx942, gfx950, gfx1250) or kernel families when the PR describes only one of them.
- In the comment, list the other members the change reaches and ask what it does to each
  (routing, segment counts, cache key, spec class). Ask for validation evidence for
  members the PR does not discuss.
- Also flag comments or docstrings on the shared structure that the change makes wrong.

## Rule 9 — A fixed bug is a class, not one site

- When the PR fixes a wrong condition, guard or proxy at one site, search for the same
  pattern in the C++ twin, the other arch's helper and sibling predicates deciding the same
  invariant. Flag any copy the PR leaves unfixed, with its path.

## Rule 10 — Shapes the PR does not target must emit identical IR

- Flag new IR-emitting code in a builder or lowering path that runs for every shape
  instead of behind the spec flag or predicate the PR adds. Untargeted shapes must emit
  byte-identical IR.
- Add goldens for new cases; update existing golden or reference IR entries
  (`tests/golden/`, `*_ir_sha256.json`) only for intentional emission changes.
  Flag updates that do not explain which shapes changed and why.
- Goldens check baseline stability; Python/C++ parity checks agreement; GPU tests
  against an independent reference check numerical correctness for tested cases.

## Rule 11 — No silent fallback

- Flag a path that, when the fast kernel is unsupported or fails, quietly falls back to a
  slower kernel, a different dtype (for example fp8 to fp16) or a different algorithm
  without raising, warning or recording the fallback.
- Prefer, in order: have the caller repair the request, reject it loudly, fall back
  with a visible warning. A silent slow path and a cryptic failure are both defects.
- Do not flag fallbacks that are documented, tested and reported to the caller.
