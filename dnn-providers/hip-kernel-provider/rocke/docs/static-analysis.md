<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Advisory static analysis: reading the reports

Three tools look at rocke Python before you push: **radon** measures how hard
the code is to follow, **pyright** checks types, and **vulture** finds dead code.
They are *advisory*: none of the three can fail a push, whatever they find. This
page explains what each number in the reports means, how to read each line, and
what to do about the most common findings. The examples are real output for
`library/kernels/gfx942/attention_tiled_2d.py`.

## Running the checks

They run as pre-push hooks from the repo-root `.pre-commit-config.yaml`
(`rocke-radon`, `rocke-pyright`, `rocke-vulture`) on the rocke `.py` files a
push changes. CI runs only the pre-commit stage, so it never runs them. Enable
once per clone:

```bash
pre-commit install --hook-type pre-push
```

Two things to know before you do:

- The first push after enabling installs the three tool environments, which
  needs network access once.
- The config sets no default stages, so this also runs the repo's other hooks
  (black, the whitespace fixers, bandit and so on) on the pushed files at push
  time, and those *can* fail a push. The three rocke checks cannot.

Run them by hand on any files, or on all of rocke (radon and pyright take tens of
seconds there, vulture a few):

```bash
pre-commit run rocke-radon   --hook-stage pre-push --files <paths>
pre-commit run rocke-pyright --hook-stage pre-push --all-files
```

To see all three reports for a whole branch without installing the pre-push hook
(this also runs the repo's other hooks):

```bash
pre-commit run --hook-stage pre-push --from-ref origin/develop --to-ref HEAD
```

| Setting | Where |
|---|---|
| pyright rules and import roots | `rocke/pyrightconfig.json` |
| radon report thresholds, vulture confidence | top of `rocke/tools/advisory_checks.py` |
| Tool versions | `additional_dependencies` in `.pre-commit-config.yaml` |

Absolute numbers on existing code are mostly a map of where complexity lives.
What matters in a PR is whether *your change* made a number worse. Run the
check on `develop` and on your branch and compare.

## radon: complexity and maintainability

```text
library/kernels/gfx942/attention_tiled_2d.py
    SLOC 4224  comment lines 2174 (33% of LOC)  functions 68
    MI 0.0 (C)  avg CC 13.1 (C)  Halstead V 26140  D 22.0  bugs 8.7

Most complex functions (CC >= 11, top 15):
  F  208  .../attention_tiled_2d.py:1314  build_unified_attention_2d_tiled  (4386 lines)
  F  171  .../attention_tiled_2d.py:3996  build_unified_attention_2d_tiled._emit_kv_body  (1412 lines)
  F  150  .../attention_tiled_2d.py:483  UnifiedAttention2DTiledSpec.__post_init__  (435 lines)
  ...
```

### Cyclomatic complexity (CC): the number that matters most

**What it measures:** how many independent paths run through one function,
which is roughly the number of test cases needed to exercise every branch. A
function with no branches scores 1, and every decision point adds to it:

| Construct | Adds |
|---|---|
| `if`, `elif`, `x if c else y` | +1 each |
| `for`, `while` (and an `else:` on a loop) | +1 each |
| each `except` clause; `else:` on a `try` | +1 each |
| `assert` | +1 |
| `a and b` / `a or b` | +1 per extra operand (`a and b and c` = +2) |
| comprehension | +1 per `for`, +1 per `if` inside it |
| `with`, `finally`, plain `else:` of an `if` | 0 |

| Rank | CC | Reading |
|---|---|---|
| A | 1–5 | simple |
| B | 6–10 | fine |
| C | 11–20 | getting hard to hold in your head; the report lists from here |
| D | 21–30 | hard to test every path |
| E | 31–40 | very hard to change safely |
| F | 41+ | effectively untestable branch by branch |

**Nested functions are scored separately.** Kernel builders do most of their
work in nested helpers, so the report lists them as `outer.inner`
(`build_unified_attention_2d_tiled._emit_kv_body`). An outer function's score
counts only its own branches, not those of the helpers inside it. Its
`(N lines)` span does include them: `build_unified_attention_2d_tiled` spans
4386 lines, of which `_emit_kv_body` alone is 1412.

**Why kernel builders score high:** almost every branch in a builder is a
compile-time switch on the spec (`if PV_FP8_MFMA:`, `if causal:`), choosing
which instructions to emit. A builder with CC 208 is really a family of kernel
variants in one function. That is a real cost, since each flag combination is a
different kernel to verify. It is also expected for this code, so an F on an
existing builder is not by itself a reason to refactor.

**What to act on:**
- **A function your PR pushed up a rank**, especially into D or worse. That is
  usually a new spec flag threaded through an existing function. Consider
  emitting the new variant from its own helper.
- **New functions above C.** Aim for B or better in new non-emitter code
  (dispatch, tools, tests).
- **Validation and naming code** like `__post_init__` (CC 150) and `kernel_name`
  (CC 50). These are long `if` chains over spec fields, and a table of rules
  often expresses them in a fraction of the branches.
- **Refactoring emitter code** (`core`, `helpers`, `instances`,
  `library/kernels`) must keep the emitted IR identical. Moving code into a
  helper is safe only if the builder calls happen in the same order. Run the
  byte-identity gate afterwards (`python tools/run_checks.py --steps gate`).

### Maintainability index (MI)

**What it measures:** a 0–100 score per file, combining size (SLOC), total
complexity, Halstead volume (below) and comment density. Higher is better.
Ranks: **A** 20–100, **B** 10–19, **C** 0–9.

**Large files always score 0.** The formula starts at 171 points, subtracts
penalties for size, and adds up to 50 back for comments. For this kernel, SLOC
costs 135 points and Halstead volume 53, and the 33% comment share earns 46
back. That leaves about 29 points, and every point of cyclomatic complexity in
the file costs another 0.23, so a few hundred branches take it to 0. MI
therefore says nothing about quality in big kernel files. Use it to compare
small and medium files (tools, dispatch, tests), where it does move.

### Halstead metrics

These count *operators* (`+`, `=`, calls, keywords) and *operands* (names,
constants) across the whole file:

- **V (volume):** how much information the code contains, about
  `length × log2(vocabulary)`. It grows with size; 26140 here.
- **D (difficulty):** how error-prone the code is to write and read. It rises
  with the variety of operators in use and with how often the same names are
  reused. As a rough guide, under 10 reads easily and 20+ is dense.
- **bugs:** `V / 3000`, a 1970s estimate of delivered defects. Read it only as
  "volume, rescaled". 8.7 does not mean this file has 9 bugs.

### Raw counts

- **SLOC:** lines with code on them; LOC (the total) also counts blank and
  comment lines.
- **comment lines:** `#` comments plus docstring and multi-line string lines,
  shown as a share of LOC.
- **functions:** functions, methods and nested functions in the file.

## pyright: type checking

```text
.../attention_tiled_2d.py:3142: error: Argument of type "tuple[Value, list[Value]] | Value" cannot be assigned to parameter "a" of type "Value" in function "div" (reportArgumentType)
pyright: 81 error(s), 0 warning(s)
     55  reportArgumentType
     16  reportCallIssue
      5  reportOptionalSubscript
      5  reportGeneralTypeIssues
```

Each line is `file:line: severity: first line of the message (rule)`. The tally
at the end groups by rule: a large count under one rule is usually a single
recurring pattern, not many separate bugs. The kernel code and the tests have
different dominant patterns, so there are two lists.

**In kernel code** (`library/kernels`, `library/builders`) most errors come from
three patterns:

1. **Lists pre-filled with `None`.** `Q_reg = [[None] * QK_K_ITERS ...]`
   (`attention_tiled_2d.py:3713`) gives pyright the type `list[list[None]]`.
   Every later `Q_reg[a][k] = <Value>` is then an error ("`Value` is not
   assignable to `None`"), and so is every read passed where a `Value` is
   expected. Fix: build the list with a comprehension or `append`, or annotate
   it (`Q_reg: list[list[Value | None]]`) and check for `None` where it is read.
2. **Loop context managers that return two shapes.** `with b.scf_for(...) as x:`
   gives the loop index alone, or `(index, iter_args)` when the loop carries
   values (`platform/python/rocke/core/ir.py:5503`). pyright cannot tell which,
   so `x` is `Value | tuple[Value, list[Value]]` and every use errors. This is
   best fixed once, in platform (typed overloads, or separate entry points for
   the two shapes), which would clear it across the library.
3. **Optional values used without a check.** `X | None` passed or indexed where
   `X` is required (`reportOptionalSubscript`, `reportOptionalMemberAccess`).
   These are the ones worth reading one by one: some are real "this can be
   `None` on a path nobody tested" bugs.

**In tests**, one pyright limitation produces the largest group of errors in the
whole tree:

4. **Mixed-type dictionaries unpacked into typed keywords.**
   `registered_attention_combos(req, **_SAMPLE)`, where `_SAMPLE` holds both
   strings and integers: pyright types every value as `int | str` and checks it
   against each keyword parameter, so every parameter narrower than that reports
   an error. These are false positives. Fix: annotate the dictionary as a
   `TypedDict`, or pass the keywords explicitly.

**Switched off on purpose:** `reportPossiblyUnboundVariable`. Builders define names
inside spec-flag branches and use them inside matching branches later. That is
correct at runtime, but pyright cannot connect the two `if`s (67 false positives
in this kernel alone).

**`reportMissingImports` warnings:** `torch`, `aiter`, `pandas`, `matplotlib`
and the compiled `rocke_engine` module are not installed in the hook
environment, so their types are unknown and checks on them are skipped. A
missing *rocke* module usually means a file adds a directory to `sys.path` at
runtime; add that directory to `extraPaths` in `pyrightconfig.json`.

**Silencing one line:** `# pyright: ignore[reportArgumentType]`. Name the rule,
so the comment does not hide other errors on that line.

## vulture: dead code

```text
.../attention_tiled_2d.py:3230: unused function '_issue_kv_fp8_async_load' (60% confidence)
.../attention_tiled_2d.py:3073: unused variable 'valid1' (60% confidence)
.../attention_tiled_2d.py:5284: unsatisfiable 'if' condition (100% confidence)
vulture: 5 finding(s)
```

**How it decides:** vulture reads all of rocke (library, platform, tools,
cmake) and reports a definition whose *name* appears nowhere else. It then
prints only the files you asked about. Test files are scanned, so a helper used
only by a test is not reported, but findings *in* tests are not shown: fixtures
and `test_*` functions look unused by design.

**Confidence:** 100% means certain (unreachable code, an `if False:` branch).
60% covers unused functions, methods, classes, attributes and variables, which
are probably dead but worth checking first.

**Matching is by name only**, which cuts both ways:
- *Missed dead code:* a dead `m_f` is not reported if any other file uses a
  variable called `m_f`.
- *False positives:* anything reached without its name in the source is
  reported as unused. That includes functions looked up through a registry or
  `getattr`, entry points listed in `pyproject.toml` `[project.scripts]`, pytest
  hooks, and public platform API that no caller uses yet.

**Before deleting anything in emitter code:** the IR builder is side-effecting.
`x = b.const_i32(8)` emits an instruction even if `x` is never read. When
vulture reports an unused variable, drop the *name* (`b.const_i32(8)` or
`_x = ...`), never the call, unless you have confirmed the call emits nothing.
`loop.results[0]`, for example, is a plain property. Then run the byte-identity
gate. This is the same hazard behind AGENTS.md's "never `ruff --fix` emitter
code" rule.

**Deliberate leftovers:** `if False:` blocks kept for parity with another arch
are reported as unsatisfiable; that is expected.

**Silencing one definition:** add a `noqa` comment with vulture's code. The
existing `# noqa: F841` markers on ABI parameters already silence vulture too.

| Code | Finding |
|---|---|
| `V101` | unused attribute |
| `V102` | unused class |
| `V103` | unused function |
| `V104` | unused import (`F401` also works) |
| `V105` | unused method |
| `V106` | unused property |
| `V107` | unused variable (`F841` also works) |
| `V201` | unreachable code |

```python
def _registry_entry(b):  # noqa: V103 - reached via KERNELS[name]
```
