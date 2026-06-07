# Mutation campaign — recommendations

## Slice scope vs. campaign total

- **131 total survivors** across the project at slice start.
- This slice triaged and resolved survivors in two areas: `Tensile.TensileLogic.*` validators and `Tensile.Common.Utilities`.
- **111 mutants KILLED** by new/strengthened tests (all reverts clean, base rc 0 / mutant rc 1).
- **3 lines pragma'd** (`# pragma: no mutate`) covering 10 intentionally-unhelpful logging mutants.
- **4 mutants ruled genuinely equivalent** (no test can kill them; see below).
- The remaining survivors toward the 131 total are **outside this slice** (modules not yet processed) — see next-slice notes.

## Functions with remaining (un-killed) survivors

After this slice, the triaged functions have **no surviving killable mutants left**. The only mutants left alive in-slice are by design:

| Function | Module | Mutant | Disposition |
|---|---|---|---|
| `_cu_count_from_path` | ValidWorkGroupMappingXCC | `mutmut_9` | equivalent (regex `cu`->`CU` under `re.IGNORECASE`) |
| `_validateWorkGroupMappingXCC` | ValidWorkGroupMappingXCC | `mutmut_14` | equivalent (missing-key default `-1`->`+1`; both early-accept) |
| `SpinnyThing.increment` | Common.Utilities | `mutmut_1` | equivalent (unused `value` param default `1`->`2`) |
| `isRhel8` | Common.Utilities | `mutmut_14` | equivalent (`open(f,"r")`->`open(f,)`; `"r"` is the default) |
| `SpinnyThing.increment` | Common.Utilities | `mutmut_4` | pragma'd (cosmetic `\b` write) |
| `ceilDivide` | Common.Utilities | `mutmut_6/7/8/9` | pragma'd (negative-register log string/print) |
| `ceilDivide` | Common.Utilities | `mutmut_17/18/19/20` | pragma'd (divide-by-zero log string/print) |

Everything else in-slice was `missing-assertion-strength` and is now killed.

## Equivalent-mutant patterns worth a `do_not_mutate` rule

These recur and are provably unkillable. Codifying rules would cut triage noise on future runs:

1. **`re.IGNORECASE` literal-case flips.** Mutating ASCII letters inside a regex string literal that is compiled/used with `re.IGNORECASE` (e.g. `cu`->`CU`) is always equivalent. Rule: skip char-case string mutations on regex literals when the pattern is used with `IGNORECASE`.

2. **Redundant-default / open-mode drops.** `open(path, "r")` -> `open(path,)` is equivalent because `"r"` is the default mode. Rule: skip dropping an argument whose value equals the call's documented default (notably `open` mode `"r"`).

3. **Unused-parameter default changes.** `def increment(self, value=1)` where `value` is never read in the body — any default-value mutation is equivalent. Rule: skip default-value mutation on parameters with no read in the function body.

4. **Both-branches-accept guard defaults.** `dict.get(key, -1)` -> `get(key, +1)` where every downstream check (`>0`, power-of-two via `&`, divisibility) accepts both sentinels. Hard to auto-detect; flag manually rather than rule-ban.

Patterns 1–3 are mechanical and good `do_not_mutate` candidates. Pattern 4 is value-dependent; leave to triage.

## Intentionally-unhelpful (pragma) pattern

The 10 pragma'd mutants all share one shape: **error/diagnostic strings and stdout writes that carry no behavioral contract** (return value and persisted state identical). Two sub-shapes:
- `print(...)` -> `print(None)`
- format-literal vandalism (`XX...XX` wrap, case-flip) on a log line.

Where the diagnostic **is** a real operator-facing contract (the validators' `Error: Validation failed: ... index: N` lines), we treated the same mutation shape as `missing-assertion-strength` and pinned the exact text via `capsys` instead of pragma'ing. The judgment call: pragma only when no consumer (human or test) depends on the string. The `ceilDivide` / `SpinnyThing.increment` writes are pure terminal cosmetics with no caller contract, hence pragma.

## Design-smell refactor candidates

1. **Repeated `print(f"Error: ...: {e} (file: {filepath}, index: {solution['SolutionIndex']})")` across validators** (`_validateWorkGroup`, `_validateMatrixInstruction`, `_validateWorkGroupMappingXCC`, `_report_xcc_failure`). Four+ near-identical diagnostic emitters generated a large fraction of in-slice survivors. Extract a single `_reportValidationFailure(kind, exc, filepath, solution)` helper. One pinned test then covers the format for all callers; mutants collapse to one site.

2. **`solution.get('SolutionIndex', '?')` scattered at every call site.** Centralize index extraction (`_solutionIndex(solution)`); removes the family of key-name / default mutants (`mutmut_15`–`mutmut_22` on `_report_xcc_failure`) that all probe the same `.get` call.

3. **`ceilDivide` mixes computation with two error-logging branches** that return the same value (`0`). The negative-register and divide-by-zero branches are log-and-return-0 noise. Consider raising or returning a sentinel the caller checks, rather than print-and-continue — would turn pragma'd cosmetic mutants into real, killable behavioral contracts.

4. **`ProgressBar` / `SpinnyThing` write raw control bytes to `sys.stdout`** inline. The display layer (`\r`, `\b`, format strings) is interleaved with state math, so cosmetic mutations land in the same functions as the real arithmetic. Splitting render-string construction (pure, testable) from the `stdout.write` (cosmetic, pragma-able) would cleanly separate killable from unkillable.

## Next-slice notes

- **Remaining survivors are in untouched modules.** This slice cleared `TensileLogic` validators + `Common.Utilities`. The path to draining the 131 is to enumerate survivors by module and take the next-densest cluster.
- **Prioritize by survivor density per function**, not file order. `_report_xcc_failure` (14) and `_validateWorkGroupMappingXCC` (13) dominated this slice — diagnostic-heavy functions are survivor magnets. Grep next modules for repeated `print(f"Error...` / `.get(key, default)` to predict clusters.
- **Apply refactor #1 before the next run if touching validators elsewhere** — collapsing the shared diagnostic emitter will pre-empt an entire mutant family rather than killing them one-by-one.
- **Pre-seed `do_not_mutate` for patterns 1–3** so the next run doesn't re-surface known-equivalent regex-case / default-arg / open-mode mutants.
- **`capsys` exact-string pinning is the workhorse** for the `print(None)` / format-vandalism family on contract-bearing diagnostics. Default to it; reserve pragma for genuinely contract-free cosmetics.
- Slice suite is green (184 passed, 70 snapshots) with the 3 pragmas applied; safe baseline for the next slice.
