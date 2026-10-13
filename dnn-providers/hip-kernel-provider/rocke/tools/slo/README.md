# rocKE SLO linter

Measures how big, how low-level and how copy-pasted each rocKE kernel file is.
No LLM, no network, no GPU. Python 3.10+ stdlib only.

## Run it

```
python tools/slo/slo.py report [--json FILE] [--scope DIR]
```

`report` prints every number, with no pass/fail. `--json -` writes the JSON to stdout.
`--scope DIR` (repeatable, rocke-relative) measures other folders instead of the configured scope.
The tool finds the rocke folder from its own location (two folders up), so it works from a
source tree or an installed copy. `--root` points it somewhere else.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Ran fine. |
| 2 | The tool could not run: bad args, or a file that does not parse. |

## Checks

| ID | Check | Target for new files |
|---|---|---|
| SLO101 | File SLOC | 600 or less |
| SLO102 | Longest function | 120 lines or less |
| SLO103 | Raw IR builder calls per 100 SLOC | 15 or less |
| SLO104 | Helper-call ratio | report only |
| SLO105 | Bool fields per frozen dataclass | 8 or less |
| SLO106 | Copy-paste overlap with any other file | 40% or less |
| SLO107 | Repeated low-level patterns (14 kinds, see `slo_patterns.py`) | count only |

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

Small examples with hand-counted answers for every metric and pattern, plus one run on the real
tree that must parse every file and finish in under 10 seconds.
