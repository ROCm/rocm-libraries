# rocKE SLO linter

Measures how big, how low-level and how copy-pasted each rocKE kernel file is.
No LLM, no network, no GPU. Python 3.10+ stdlib only.

## Run it

```
python tools/slo/slo.py report   [--json FILE] [--scope DIR]
python tools/slo/slo.py check    [--base REF] [--json|--junit|--markdown FILE] [--github]
python tools/slo/slo.py baseline --update
```

- `report`: every number, no pass/fail. `--json -` writes the JSON to stdout.
  `--scope DIR` (repeatable, rocke-relative) measures other folders instead of the configured scope.
- `check`: compares with `baseline.json`. Lists what got worse (warning) and what got better.
  `--base REF` limits it to files changed since `REF` and follows renames (needs git).
- `baseline --update`: saves today's numbers. Warns if any number is looser than before.

Warn only by default: `check` exits 0 even with warnings. `--fail-on increase` makes "worse"
exit 1. Nothing else changes. CI snippets and the full contract: `CI.md`.

The tool finds the rocke folder from its own location (two folders up), so it works from a
source tree or an installed copy. `--root` points it somewhere else.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Ran fine. Warnings do not change this unless `--fail-on increase`. |
| 1 | Something got worse, and `--fail-on increase` was given. |
| 2 | The tool could not run: bad args, a file that does not parse, git missing when asked. |

## Checks

| ID | Check | Target for new files |
|---|---|---|
| SLO101 | File SLOC | 600 or less |
| SLO102 | Longest function | 120 lines or less |
| SLO103 | Raw IR builder calls per 100 SLOC | 15 or less |
| SLO104 | Helper-call ratio | report only |
| SLO105 | Bool fields per frozen dataclass | 8 or less |
| SLO106 | Copy-paste overlap with any other file | 40% or less |
| SLO107 | Repeated low-level patterns (14 kinds, see `slo_patterns.py`) | may not rise |

Old files are compared to their own baseline numbers, not to the targets. A new file over a
target is a warning. SLO104 is report only: it never fails, even with `--fail-on increase`.
The targets are proposals. Limits and scope live in `slo_config.py`, so owners change them
there, not in the code.

## How each number is counted

- **SLOC**: lines minus blank, comment-only and docstring lines.
- **Longest function**: `def` line to last line. Nested functions are counted on their own.
- **Raw IR ops**: calls on the IR builder, plus bare `const_*()` calls. The builder is found by
  type, not only by the name `b`: parameters annotated `IRBuilder`, `x = IRBuilder(...)`,
  aliases of those, and `self.<attr>` set from them. Renaming the builder does not hide its calls.
- **Helper-call ratio**: `helper / (helper + raw)`. Helpers are names imported from
  `rocke.helpers`, plus methods on objects those names build.
- **Bool fields**: fields annotated `bool` on `@dataclass(frozen=True)` classes. The worst class
  in the file counts.
- **Overlap**: distinct lines two files share, divided by the distinct lines of the smaller file.
  Lines under 14 characters and comment lines are ignored. Files under 200 SLOC are skipped.
  A second, advisory score (`overlap_norm`) replaces variable names first, so a copy with renamed
  variables still shows up.

## Scope

`library/kernels/**` and `platform/python/rocke/instances/**`. Tests, examples, benchmarks and
builders are skipped.

## Tests

```
pytest tools/slo/tests
```

- Small examples with hand-counted answers for every metric and pattern.
- One run on the real tree that must parse every file and finish in under 10 seconds.
- Compare, exit codes, every output format, and `--base` on a throwaway git repo.
- Break it on purpose: on a copy of the real tree, add raw ops, grow the longest function and
  paste a file. Each must give the right check id, and `--fail-on increase` must exit 1.

`library/tests/test_slo_ratchet.py` runs `check` against the committed baseline. It only warns.
