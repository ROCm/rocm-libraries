# TensileLite client diagnostics

This is the reference pattern for emitting failure diagnostics in tensilelite.
New diagnostics should copy it rather than reach for a bare `std::cout`.

It exists because a real failure (a gfx12 benchmark exiting `1`) reached CI as
nothing but `ClientWriter Benchmark Process exited with code 1` plus a stray
`two` on stdout — no config, no GPU, no phase, no solution, no error. The model
below makes that class of dead-end impossible.

## Principles

1. **Every failure carries context.** What failed, *where* (config, GPU, phase,
   solution, kernel), the error itself, and a concrete next step.
2. **One stable, greppable tag.** Every machine-readable line starts with
   `[tensilelite:diag]`. Tooling filters on it; a stray debug print can never
   masquerade as a diagnostic.
3. **Survives truncation and the process boundary.** The client runs as a
   subprocess; its diagnostics go to `stderr`, which CI captures verbatim. The
   one-line record stays intact even when a log is clipped.
4. **No bare debug tokens.** If it is worth printing on a failure path, it is
   worth a `Diagnostic`. `std::cout << "two"` is the anti-pattern.

## Format (hybrid)

Each `emit()` writes two things to `stderr`:

- a single **logfmt** line — greppable and machine-parseable:

  ```
  [tensilelite:diag] level=ERROR cat=solution-failed config=rotate_mode1_gfx12.yaml gpu=gfx1201 phase=enqueue problem_idx=0 solution_idx=7 solution=Cijk_..._00 kernel=Cijk_... exception=St13runtime_error msg="launchKernels failed" next="rerun with --log-level=Debug"
  ```

- a **banner** block — the same fields, laid out for a human reading the log.

Values containing spaces, `=`, `"`, or newlines are quoted and escaped in the
logfmt line.

## API

`client/include/Diagnostic.hpp`, header-only, `namespace TensileLite::Client`.

```cpp
Diagnostic(Diagnostic::Severity::Error, "solution-failed")
    .field("problem_idx", problemIdx)
    .field("solution", solution->name())
    .field("msg", err.what())
    .next("rerun with --log-level=Debug")
    .emit();
```

- `Severity` is `Fatal` (about to exit nonzero), `Error` (one unit failed, run
  continues), or `Warning`.
- The first argument after severity is a short, stable **category** you can grep
  for (`solution-failed`, `client-fatal`, `run-summary`).
- `field(key, value)` accepts anything streamable to `std::ostream`.
- `next(advice)` is the recommended "what do I do now" field.

### Ambient context — set once, attached to every diagnostic

`config`, `gpu`, and `phase` are filled automatically:

- `g_diagConfig` / `g_diagArch` (`Diagnostic.hpp`) are set once in `runClient`
  after argument parsing and after `GetHardware`.
- `phase` is the active `ScopedTimer` category (`TimingInstrumentation.hpp`).
  Wrap a step in a `ScopedTimer("name")` and any diagnostic thrown inside it
  reports `phase=name` for free. This is why the timer instrumentation and the
  diagnostics share one phase variable.

## The two patterns

**Fatal — report and exit.** `main` wraps `runClient` in a top-level
`catch (std::exception&) / catch (...)`, emits a `client-fatal` diagnostic, and
returns `2`. This covers setup-phase throws (library load, code-object load,
HIP init) that previously aborted with no context.

**Per-operation — report and continue.** The per-solution `catch` in the
benchmark loop widens to `std::exception`, emits a `solution-failed` diagnostic
with the solution and problem identity, still reports `Validation INVALID`, and
lets the run proceed to the next solution.

A `run-summary` diagnostic is emitted before any nonzero return so the tail of
the log states how many errors occurred.

## Exit codes

- `0` — success.
- `1..255` — `listeners.error()`: the run completed but N units failed (see the
  `solution-failed` and `run-summary` diagnostics).
- `2` — a fatal/uncaught error (see the `client-fatal` diagnostic). Note `2`
  also overlaps `listeners.error()==2`; the emitted `cat=` disambiguates.

## Reading a failed run

```
grep '\[tensilelite:diag\]' <client-or-ci-log>
```

`ClientWriter` (Python) points here on any nonzero client exit.

## Python side (same model, same tag)

`Tensile/Diagnostics.py` is the Python counterpart of `Diagnostic.hpp`: the same
`[tensilelite:diag]` tag, the same hybrid logfmt+banner output, written to
`stderr`. A failure that crosses the Python/C++ boundary reads identically on
both sides.

```python
from Tensile.Diagnostics import Diagnostic

Diagnostic(Diagnostic.FATAL, "run-failed") \
    .field("config", config) \
    .field("msg", err) \
    .next("rerun: Tensile <config> <output_dir> --use-cache") \
    .emit()
```

Applied where the harness previously swallowed failures:

- `Tensile/Tests/common/test_config.py` — the subprocess wrapper no longer uses
  `check=True` (which dumped the full, unreadable argv). It checks the child
  return code, emits a `test-config-phase-failed` diagnostic (phase, config,
  exit code), and raises a `CalledProcessError` with a short command.
- `test_config_build.py` / `test_config_run.py` — the `_build` / `_run` helpers
  emit `build-failed` / `run-failed` before the exception propagates, so the
  child names the config and phase even amid a traceback.
- `Tensile/ClientWriter.py` — emits `client-exit-nonzero` /
  `client-process-failed` pointing at the client's `[tensilelite:diag]` lines.

## Extending to the host library

The same shape applies to `tensilelite-host`. Today the host uses ad-hoc
`std::cerr` banners (e.g. the PCI-chip-id messages in `Tensile/hip/HipUtils.hpp`)
— those are the next candidates to route through a host-side equivalent of this
record so library failures get the same tagged, contextual treatment.
