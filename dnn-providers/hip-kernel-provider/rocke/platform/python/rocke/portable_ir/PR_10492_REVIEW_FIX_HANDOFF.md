# PR #10492 correctness follow-up handoff

## Purpose and state

This handoff preserves the bounded correctness follow-up from the review of PR
#10492, `fix(rocke): harden JIT recipe record and replay`, at head
`5ba022a4e6b8f5c1c7355b3db2205b04efe22db1` on 2026-08-06.

The earlier scalar-validation handoff is complete. Its decisions and regression
evidence now live in `DECISION_LOG.md`; they are not repeated here. This file
contains only the two unresolved correctness findings below. It should be
removed after both fixes are verified and their outcomes are added to the
decision log.

## Scope

Implement only:

1. semantic parity between the Python recipe expander and the C recipe VM for
   the compile-time expressions and loops changed by this robustness PR;
2. safe, explicit binding for single-result `emit.out` declarations.

Do not widen this follow-up into complete recipe-schema validation, a generic
schema walker, opcode contract validation, provider dispatch integration, or
rolling redesign.

## Finding 1: restore Python-oracle and C-VM semantic parity

### Confirmed behavior

The C VM now rejects:

- division or modulo by zero;
- `LONG_MIN / -1` and `LONG_MIN % -1`;
- unknown or malformed `spec_str_eq` operands;
- non-positive `static_for` and rolled-list steps;
- loop-increment overflow.

The Python expander in `utils/recipe_expand.py` still:

- maps division and modulo by zero to `0`;
- returns false for an unknown string spec;
- converts a zero step to `1`;
- can fail to terminate for a negative step;
- uses Python floor division and modulo for negative operands, while C truncates
  division toward zero and derives the remainder from that quotient.

Confirmed review repro: Python evaluated `{"div": [1, 0]}` as `0`; the
standalone C replay path rejected the same expression with `integer division by
zero`.

This is a correctness issue because `src/roll.py` uses `expand_recipe()` plus
`recipes_equiv()` as the device-free proof that a rolled recipe reproduces its
independently recorded concrete traces. The oracle and deployed replay engine
must implement one specialization language.

### Implementation plan

In `python/rocke/portable_ir/utils/recipe_expand.py`:

1. Replace the inline division/modulo lambdas with checked helpers.
2. Implement integer division without a float conversion: divide absolute
   values, apply the quotient sign for truncation toward zero, and compute
   modulo as `a - quotient * b`.
3. Derive `LONG_MIN` and `LONG_MAX` from the native C `long` width and reject
   the same `LONG_MIN / -1` and `LONG_MIN % -1` cases as the C VM.
4. Raise `ExpandError` for zero divisors.
5. Require `spec_str_eq` to contain exactly two strings and require its spec
   name to exist in `spec_str`.
6. In `_expand_name_list()`, `_expand_iter_list()`, and `static_for`, reject
   steps less than or equal to zero. Remove the `or 1` coercion.
7. Before each loop increment, reject `iv > LONG_MAX - step`, matching the C
   VM's overflow check.

In `python/rocke/portable_ir/tests/test_roller.py`:

1. Add focused unit cases for zero division/modulo, signed division/modulo,
   signed overflow, malformed and unknown string predicates, non-positive
   steps, and loop-increment overflow.
2. Retain positive cases for declared true/false string comparisons and normal
   increasing loops.
3. Assert failures are `ExpandError`; no test should rely on a timeout to prove
   that a loop does not terminate.

Do not change the C VM unless a test proves that its current checked semantics
are wrong. The main implementation work for this finding belongs in the Python
oracle.

## Finding 2: reject unsafe single-result bindings

### Confirmed behavior

The multi-result `emit.outs` path requires every result to have a nonempty,
distinct bind. The single-result `emit.out` path still calls
`rv_bind_name(..., "r")`, so a missing bind silently defaults to `r`.

Confirmed review repro: a concrete recipe with two side-effecting inline-
assembly operations and `out: {"type": "i32"}` replayed successfully and
produced two definitions named `%r`. `llvm-as` rejected the result with
`multiple definition of local value named 'r'`.

### Implementation plan

In `cpp/portable_ir/recipe_vm.cpp`:

1. Require `emit.out.bind` to be a nonempty string, matching the existing
   multi-result rule.
2. Resolve placeholders before checking the final bind.
3. For concrete recipes using exact SSA names, reject a result bind that would
   redefine an existing SSA-producing register. Preserve intentional register
   rebinding for parametric expansion, where exact SSA naming is disabled.
4. Keep `alias` behavior unchanged; aliases rebind the VM register table without
   defining a new LLVM SSA value.

In `tests/portable_ir/recipe_vm_replay.cpp`:

1. Reject a missing, empty, or non-string single-result bind.
2. Reject two concrete single-result operations with the same explicit bind.
3. Retain a positive single-result replay case and the existing multi-result
   coverage.
4. Lower the positive result and verify its expected distinct SSA definition.
   Keep the hermetic unit independent of an external assembler; use `llvm-as`
   on the standalone replay output as an additional verification step when the
   tool is available.

## Expected implementation footprint

Production and focused test changes should normally be limited to:

- `cpp/portable_ir/recipe_vm.cpp`;
- `python/rocke/portable_ir/utils/recipe_expand.py`;
- `python/rocke/portable_ir/tests/test_roller.py`;
- `tests/portable_ir/recipe_vm_replay.cpp`.

After verification, update `DECISION_LOG.md` with the final decisions and exact
regressions, then remove this handoff. Changes to another file need a concrete
reason tied to one of the two findings.

## Explicit non-goals

- numeric-cast range validation;
- add, subtract, or multiply overflow checks;
- vector-count or shared-memory-dimension validation;
- generic opcode arity or operand-contract validation;
- nested list-of-map attribute support;
- multi-axis rolling inference;
- new region forms;
- JSON/CBOR parser changes or a recipe schema-version change;
- provider `ArtifactStore` or dispatch integration;
- GPU execution claims.

## Verification plan

From `dnn-providers/hip-kernel-provider/rocke/platform`:

Set `ROCKE_REVIEW_BUILD` to a writable out-of-tree build directory using the
platform's normal environment mechanism, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=python:../library \
python3 -m unittest discover -s python/rocke/portable_ir/tests -v

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=python:../library \
python3 -m rocke.portable_ir.drivers.record_coverage

cmake -S . -B "$ROCKE_REVIEW_BUILD" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DROCKE_BUILD_PYENV=OFF \
  -DROCKE_SANITIZE=ON
cmake --build "$ROCKE_REVIEW_BUILD" \
  --target rocke_portable_ir_recipe_vm_replay \
           rocke_portable_ir_dom_decoders \
           rocke_portable_ir_replay_cli -j
ctest --test-dir "$ROCKE_REVIEW_BUILD" \
  -R 'rocke_portable_ir_(recipe_vm_replay|dom_decoders)' \
  --output-on-failure
```

If LeakSanitizer reports only that it cannot run under the ptraced runner, run
the two test executables directly with `ASAN_OPTIONS=detect_leaks=0`. Keep
AddressSanitizer and UndefinedBehaviorSanitizer enabled.

Also run the CPU-only recipe parity matrix for `gfx942` and `gfx950` against a
fresh shared library. This is required because both fixes sit on the path used
to certify and replay parametric recipes. GPU compilation or launch is not
required for this follow-up and must not be claimed unless separately executed.

```bash
PYTHONPATH=python:../library \
python3 -c 'import sys; from rocke.portable_ir.src import online; online.build_lib(sys.argv[1])' \
  "$ROCKE_REVIEW_BUILD/librocke.so"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=python:../library \
ROCKE_ONLINE_LIB="$ROCKE_REVIEW_BUILD/librocke.so" \
python3 -m rocke.portable_ir.drivers.parity_matrix \
  --arches gfx942,gfx950
```

Finally, from the repository root:

```bash
git diff --check
git status --short
```

## Acceptance criteria

The follow-up is complete when all of the following are true:

- Python and C produce the same values or the same rejection for the focused
  integer-expression and loop cases;
- malformed or unknown string predicates are rejected by both paths;
- non-positive Python expansion loops fail immediately with `ExpandError`;
- every single-result declaration requires a nonempty bind;
- repeated concrete single-result SSA definitions are rejected before lowering;
- valid recorded recipes retain byte-identical replay for `gfx942` and `gfx950`;
- the full Python portable-IR suite, recorder coverage gate, focused C++ tests,
  DOM tests, sanitizer run, parity matrix, and diff check pass;
- the final decisions and regression evidence are added to `DECISION_LOG.md`;
- this handoff is removed after the decision-log update.
