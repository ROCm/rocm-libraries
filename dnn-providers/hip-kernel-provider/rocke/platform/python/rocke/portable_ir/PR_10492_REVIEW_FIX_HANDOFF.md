# PR #10492 review-fix handoff

## Purpose

This handoff preserves the bounded follow-up from the review of PR #10492,
`fix(rocke): harden JIT recipe record and replay`, at head
`c48abb03beb4a4687102c0601d25489825bf3233`.

Implement only the two confirmed recipe-VM validation fixes below, add their
focused regressions, and correct the stale PR-description wording. Do not widen
this work into general recipe-schema validation.

## Current behavior to fix

### 1. Reject invalid `spec_str_eq` operands and references

File:
`cpp/portable_ir/recipe_vm.cpp`, in `rv_int()`.

The integer-spec path rejects an unknown spec, but the string predicate path
does not. Given this expression:

```json
{"spec_str_eq": ["typo", "f16"]}
```

`rv_spec_str()` returns null and `rv_int()` silently evaluates the predicate to
false. A recipe with no declaration named `typo` therefore returns `ROCKE_OK`
and can select the wrong `static_if` arm.

Required behavior:

- require `spec_str_eq` to contain exactly two string values;
- reject an unknown string-spec name with `rv_fail()`;
- preserve ordinary true and false comparisons for a declared, supplied string
  spec;
- return `ROCKE_ERR_VALUE` and no kernel for malformed or unknown references.

Keep the implementation local to `rv_int()`. Do not introduce a general schema
walker or change the representation of runtime specs.

### 2. Reject wrong kinds for `const_f32` and `scf_for` flags

File:
`cpp/portable_ir/recipe_vm.cpp`, in `rv_exec_instr()`.

Two scalar instruction paths currently read values without checking their DOM
kind:

- `const_f32` ignores a failed `rocke_jnum()` call, so a missing or string
  `fval` becomes `0.0`;
- `scf_for` reads `unroll` and `elide_trailing_barrier` through `.b` whenever
  the field exists, even if the node is not `JD_BOOL`.

Required behavior:

- require `const_f32.fval` to be numeric before calling
  `rocke_b_const_f32()`;
- when present, require `scf_for.unroll` and
  `scf_for.elide_trailing_barrier` to be booleans;
- retain the existing defaults when either optional flag is absent;
- return `ROCKE_ERR_VALUE` and no kernel for wrong-kind values.

Use direct checks at the decode sites. Do not refactor unrelated instruction
decoding.

## Focused regression plan

Add cases only to:
`tests/portable_ir/recipe_vm_replay.cpp`.

Extend the existing `check_rejected()` coverage with:

1. `spec_str_eq` referencing an undeclared string spec;
2. a non-string `spec_str_eq` spec name;
3. a non-string `spec_str_eq` comparison literal;
4. `const_f32` with a nonnumeric `fval`;
5. `const_f32` with a missing `fval`;
6. `scf_for.unroll` with a non-boolean value;
7. `scf_for.elide_trailing_barrier` with a non-boolean value.

Retain or add small positive checks only where needed to prove that:

- a declared string spec still produces both normal true and false predicate
  results;
- valid boolean flags and absent optional flags continue to replay.

Do not add a new test binary or a generic malformed-recipe framework.

## PR-description correction

The PR description currently asks reviewers to review a "single commit," while
the PR contains multiple commits. Replace that sentence with commit-count-neutral
wording such as:

> Review the complete diff on top of
> `users/yraparti/rocke-jit-compilation-prototype`.

Do not squash, reorder, or otherwise rewrite commits as part of this follow-up.

## Expected implementation footprint

Production and test changes should be limited to:

- `cpp/portable_ir/recipe_vm.cpp`;
- `tests/portable_ir/recipe_vm_replay.cpp`.

This handoff document is the only additional repository artifact. The PR-body
wording is an external metadata update, not a source change.

## Explicit non-goals

Do not include any of the following:

- numeric-cast range validation;
- add, subtract, or multiply overflow checks;
- vector-count or shared-memory-dimension validation;
- changes to single-result default bind behavior;
- new result-count or opcode-arity validation;
- changes to recorder coverage or rolling inference;
- changes to JSON/CBOR parsing or the recipe schema version;
- provider `ArtifactStore` or dispatch integration;
- broad decoder refactoring;
- unrelated documentation cleanup.

The existing decision log already records numeric casts and add/subtract/multiply
overflow as known boundaries. Leave those boundaries unchanged in this fix.

## Verification

From `dnn-providers/hip-kernel-provider/rocke/platform`:

```bash
cmake -S . -B /tmp/rocke-pr10492-fix -DCMAKE_BUILD_TYPE=Debug \
  -DROCKE_BUILD_PYENV=OFF
cmake --build /tmp/rocke-pr10492-fix \
  --target rocke_portable_ir_recipe_vm_replay \
           rocke_portable_ir_dom_decoders -j
ctest --test-dir /tmp/rocke-pr10492-fix \
  -R 'rocke_portable_ir_(recipe_vm_replay|dom_decoders)' \
  --output-on-failure

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=python:../library \
python3 -m unittest discover -s python/rocke/portable_ir/tests -v
```

If LeakSanitizer reports only that it cannot run under a ptraced test runner,
rerun the two C++ tests with `ASAN_OPTIONS=detect_leaks=0`; keep AddressSanitizer
and UndefinedBehaviorSanitizer enabled.

Finally run from the repository root:

```bash
git diff --check
git status --short
```

GPU execution is not required for these reject-path-only changes. Do not claim
fresh GPU or full parity-matrix coverage unless it is separately run and
retained.

## Acceptance criteria

The follow-up is complete when all of the following are true:

- every malformed case listed above returns `ROCKE_ERR_VALUE`;
- every rejected call leaves the output kernel null;
- valid string predicates, numeric `const_f32`, valid boolean flags, and absent
  optional flags retain their current behavior;
- the focused C++ tests and existing 30-test Python portable-IR suite pass;
- the diff contains no production or test changes outside the two expected
  files;
- the PR description no longer states that the diff is a single commit;
- the known boundaries and non-goals above remain untouched.
