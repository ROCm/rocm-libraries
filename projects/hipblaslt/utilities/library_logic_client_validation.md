# Validation coverage: `LibraryLogic` and `LibraryClient` YAML blocks

Scope: top-level YAML blocks consumed by `LibraryLogic.main()` and `ClientWriter.main()` in `tensilelite/Tensile/`. Investigation is read-only and based on tracing the load path from `Tensile.Tensile()` → `executeStepsInConfig()` → the two `main()` functions, plus the keys those functions actually read.

## Summary

### Headline gaps

- Neither block goes through any schema validation step today. The YAML payload is loaded by `LibraryIO.read` (`Tensile/Tensile.py:560`) and handed straight to the consumer as a dict (for `LibraryLogic`) or a list of dicts (for `LibraryClient`). No equivalent of `checkParametersAreValid` runs.
- `LibraryLogic` has an authoritative key set: `defaultAnalysisParameters` at `Tensile/Common/GlobalParameters.py:588-594`. It lists exactly five keys with known default values and types. `assignParameterWithDefault` (`Tensile/Common/Utilities.py:322-326`) silently *drops any key in the user YAML that is not in the defaults dict* — typos and stale keys vanish without warning. This is the same coverage gap as `GlobalParameters`, only narrower.
- `LibraryClient` has *no* authoritative key set. The list-of-dicts payload is scanned for three hardcoded literal strings (`ActivationArgs`, `FactorDimArgs`, `ICacheFlush`) in `Tensile/ClientWriter.py:161-167`. Unknown keys are silently ignored. Value shapes are loose; downstream constructors do a small amount of `printWarning`-only validation on the elements.
- `LibraryLogic` is used by virtually every real test config under `Tensile/Tests/common/`. `LibraryClient` is used by exactly one test (`gemm/xfp32.yaml`, with an empty body); the three keys it documents are in practice consumed from `BenchmarkFinalParameters` inside `BenchmarkProblems`, not from `LibraryClient` itself.
- A small number of analysis-parameter keys referenced in `LibraryLogic.py` (`SmoothOutliers` at line 174, `BranchPenalty` at lines 849/858/860) are **not** in `defaultAnalysisParameters`. Because `assignParameterWithDefault` iterates only over keys present in the defaults dict, those references would `KeyError` if the code paths are ever entered — they look like dead/legacy parameter knobs.

### Recommended scope additions for the type-validator plan

- **Include `LibraryLogic`.** It has a stable five-key schema, real-world usage, and silent typo-drop behaviour. Type, enum and unknown-key checks all have a clear authoritative source (`defaultAnalysisParameters`).
- **Defer or skip `LibraryClient`.** Its in-tree usage is essentially nil and its in-`LibraryClient` schema is just three optional list-valued keys. A typo-only check is the only thing realistic to bolt on, and the cost/benefit is poor compared to other blocks.

---

## LibraryLogic

### Q1: What goes inside it?

Authoritative source: `Tensile/Common/GlobalParameters.py:588-594`.

```python
defaultAnalysisParameters = {
    "ScheduleName": "Tensile",
    "DeviceNames": "fallback",
    "ArchitectureName": "gfx000",
    "LibraryType": "GridBased",
    "SolutionImportanceMin": 0.01,  # = 0.01=1% total time saved by keeping this solution
}
```

Keys recognised after merge with defaults at `Tensile/LibraryLogic.py:1444-1447`:

| Key | Default | Value type observed | Notes |
|-----|---------|---------------------|-------|
| `ScheduleName` | `"Tensile"` | `str` | Used as filename prefix at `LibraryLogic.py:1501,1510`. Real configs use `"aquavanjaram"`, `"aldebaran"`. |
| `DeviceNames` | `"fallback"` (str) | `str` *or* `list[str]` | Default is a bare string; every real config supplies a `list` of device-id strings. Passed straight through to `LibraryIO.createLibraryLogic` (`LibraryLogic.py:1511`). |
| `ArchitectureName` | `"gfx000"` | `str` | Real values are `"gfx942"`, `"gfx90a"`, etc. |
| `LibraryType` | `"GridBased"` | `str` | Branch-tested against `"FreeSize"` / `"Prediction"` at `LibraryLogic.py:101, 200, 207, 223`; the writer at `LibraryIO.py:639-656` recognises `"FreeSize"`, `"Prediction"`, and treats anything else as `"Matching"` (using the value as a distance metric). `"GridBased"` therefore ends up in the catch-all branch. |
| `SolutionImportanceMin` | `0.01` | `float` | Used at `LibraryLogic.py:588`. |

Other `inputParameters["…"]` accesses inside `LibraryLogic.py` (`SmoothOutliers` line 174, `BranchPenalty` lines 849/858/860) refer to keys that are not in `defaultAnalysisParameters`. Because `assignParameterWithDefault` only iterates the defaults dict, those keys are never populated; the code paths that touch them would `KeyError` if hit. Treat them as legacy.

Real snippets:

```yaml
# Tensile/Tests/common/gemm/xfp32.yaml:87-91
LibraryLogic:
    ScheduleName: "aquavanjaram"
    DeviceNames: ["Device 0049", "Device 0050"]
    ArchitectureName: "gfx942"
    LibraryType: "GridBased"
```

```yaml
# Tensile/Tests/common/groupedgemm/grouped_gemm.yaml:216-220
LibraryLogic:
    ScheduleName: "aldebaran"
    DeviceNames: ["Device 0050", "Device 0051", "Device 0052", "Device 0054", "Device 0062", "Device 7400", "Device 740c"]
    ArchitectureName: "gfx90a"
    LibraryType: FreeSize
```

`SolutionImportanceMin` is not set in any of the surveyed common-test YAMLs; everyone takes the default. Likewise `ScheduleName` and `LibraryType` are not always provided (the in-repo configs that omit them silently take defaults).

### Q2: What validation exists today?

Load path: `Tensile.Tensile()` (`Tensile/Tensile.py:481`) → `LibraryIO.read(configPaths[0])` (`Tensile.py:560`) → dict stored in `config["LibraryLogic"]` → unwrapped at `Tensile.py:139-145` → `LibraryLogic.main(config, …)` (`LibraryLogic.py:1567`) → `generateLogic(config, …)` (`LibraryLogic.py:1427`) → per-key copy via `assignParameterWithDefault` (`LibraryLogic.py:1444-1447`).

- **Unknown-key detection: none.** `assignParameterWithDefault` (`Tensile/Common/Utilities.py:322-326`) iterates over `defaultAnalysisParameters` and pulls each key from `config` if present, else from defaults. Any key in the YAML that is *not* in `defaultAnalysisParameters` is silently ignored. There is no symmetric loop walking the user's keys to catch typos.
- **Type checks: none.** Values are `deepcopy`'d as-is. `DeviceNames` is documented as a default-`str` but real usage is `list[str]`; nothing rejects either shape, and downstream consumers (`LibraryIO.createLibraryLogic`) just thread it through into the output document.
- **Enum / range checks: none on entry.** `LibraryType` is implicitly enum-like (`"FreeSize" | "Prediction" | "GridBased" | "Matching"`-style distance metric) but no input-time check enforces the set. An unknown value falls through into the `Matching` branch (`LibraryIO.py:651-656`) and becomes the literal `distance` metric label — which may or may not be valid downstream, but no error is raised here.
- **Cross-field / consistency checks: none.**
- **Behaviour on a malformed entry:** typo → silent drop → fall back to default; wrong type for `SolutionImportanceMin` → would surface as a comparison `TypeError` deep inside `LogicAnalyzer` (`LibraryLogic.py:588`); wrong type for `ScheduleName` → would surface as a `str.format` error (`LibraryLogic.py:1501`). No early/clear error message in any case.

Net: this is a **coverage gap** analogous to `GlobalParameters`, just bounded to five keys.

### Q3: What validation could we do?

- **Unknown-key detection (highest value).** The default dict is a closed five-key set. A symmetric loop `for k in userYaml: if k not in defaultAnalysisParameters: warn`-or-error catches typos like `ScheculeName`, `LibrarType` that are silently dropped today.
- **Type checks.** Pin each of the five keys to a concrete expected Python type — at minimum: `ScheduleName: str`, `ArchitectureName: str`, `LibraryType: str`, `SolutionImportanceMin: (int, float)`. `DeviceNames` is the awkward one (default is `str`, real usage is `list[str]`) — declare it `Union[str, list[str]]` to match reality.
- **Enum check on `LibraryType`.** Allowed strings inferable from `LibraryIO.py:639-656` and `LibraryLogic.py:101,200`: `{"FreeSize", "Prediction", "GridBased", "Matching"}` plus any distance-metric labels that fall into the catch-all. Recommend enforcing `{"FreeSize", "Prediction", "GridBased", "Matching"}` (or whatever the canonical set is — defer to the maintainer; the code accepts more than it advertises).
- **Range check on `SolutionImportanceMin`.** It is interpreted as a fraction in `[0, 1]`; values outside that range silently distort solution culling. A bounds check would be cheap.
- **Don't bother with cross-field validation.** The five keys are independent.

---

## LibraryClient

### Q1: What goes inside it?

Authoritative source: there is none. The block is consumed by ad-hoc string lookups at `Tensile/ClientWriter.py:159-167`. No defaults dict, no dataclass, no schema.

```python
# Tensile/ClientWriter.py:159-167
if len(config) > 0:
  for lc in config[0:]:
    if "ActivationArgs" in lc:
      activationEnums = lc["ActivationArgs"]
      break
    if "FactorDimArgs" in lc:
      factorDimEnums = lc["FactorDimArgs"]
    if "ICacheFlush" in lc:
      icacheFlushArgs = lc["ICacheFlush"]
```

The expected shape is a `list` of single-key dicts. Recognised keys, taken from the literal-string lookups above and a comment block at `ClientWriter.py:151-157`:

| Key | In-code default (when key absent or `LibraryClient: null`) | Value type | Notes |
|-----|------------------------------------------------------------|------------|-------|
| `ActivationArgs` | `[[{'Enum': 'relu'}]]` (`ClientWriter.py:149`) | `list[list[dict]]` — outer list of activation settings, each inner list is dicts like `{Enum: relu}` | Used to construct `SolutionStructs.Solution.ActivationArgs(problemType, …)` (`Solution.py:399`). `ActivationType` enum members are the allowed values (e.g. `none`, `relu`, `gelu`, …). |
| `FactorDimArgs` | `[0]` (`ClientWriter.py:150`) | `list[int]` of `0`/`1` | Validated inside `FactorDimArgs.__init__` (`Solution.py:352-364`): casts each element to `int`, `printWarning` if not in `{0, 1}`. |
| `ICacheFlush` | `[False]` (`ClientWriter.py:158`) | `list[bool]` | Consumed verbatim — passed downstream into `writeClientConfig` (`ClientWriter.py:179`). |

There is also a wart in the loop logic worth flagging: the `if "ActivationArgs" … : break` short-circuits before `FactorDimArgs`/`ICacheFlush` are tested in the *same* list element, so callers must put each key in a separate `- …` block. Real-world examples don't exercise this; only `gemm/xfp32.yaml:93` actually has a `LibraryClient:` block in `Tensile/Tests/common/`, and it is empty:

```yaml
# Tensile/Tests/common/gemm/xfp32.yaml:93
LibraryClient:
```

All other `ActivationArgs:` / `FactorDimArgs:` / `ICacheFlush:` occurrences under `Tensile/Tests/common/` are inside `BenchmarkProblems → BenchmarkFinalParameters`, consumed by `BenchmarkStructs.py:195-210` — a *different* code path. Example:

```yaml
# Tensile/Tests/common/gemm/large_size.yaml:58-63
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [2048, 65536, 1, 16384]
        - BiasTypeArgs: ['b']
        - ActivationArgs:
          - [Enum: relu]
```

### Q2: What validation exists today?

Load path: `Tensile.Tensile()` (`Tensile/Tensile.py:481`) → `LibraryIO.read` (`Tensile.py:560`) → `config["LibraryClient"]` → `Tensile.py:161-175` → `ClientWriter.main(config, …)` (`ClientWriter.py:92`) → loop at `ClientWriter.py:159-167`.

- **Unknown-key detection: none.** Only literal strings `ActivationArgs`, `FactorDimArgs`, `ICacheFlush` are tested with `in`. A typo like `Activatoin` is silently ignored; the defaults stand.
- **Type checks on the block itself: none.** `config` is treated as iterable with `len(config) > 0` and `for lc in config[0:]`; if a user provided `LibraryClient: {ActivationArgs: …}` (a dict instead of a list) the iteration would yield the dict's string keys and the `if "ActivationArgs" in lc` check would test substring containment in the string — silently malformed.
- **Type checks on the values:** delegated to downstream constructors. `FactorDimArgs.__init__` (`Solution.py:352-364`) does `int(fdim)` casts and warns if outside `{0, 1}`. `ActivationArgs.__init__` (`Solution.py:399-422`) validates `{Enum: …}` dict shape and warns/exits on mis-shapes. `ICacheFlush` is unvalidated — used as a plain list.
- **Enum / range checks:** only via the warn-only downstream constructors above.
- **Cross-field / consistency checks: none.** A guard exists at `ClientWriter.py:168` (`isForAll = problemType["ActivationType"] in ['all', 'hipblaslt_all']`) so `ActivationArgs` is *used* only when the problem type permits it, but a user-provided `ActivationArgs` for a non-`all` problem type is silently dropped, not flagged.
- **Behaviour on a malformed entry:** silent default in most cases; downstream `printWarning` or `printExit` for a few specific value-shape errors at construction time. No early/clear error message at YAML-load time.

Net: a coverage gap, but a low-traffic one — the block is barely used in the test corpus, and its three keys also exist in the (separately-validated) `BenchmarkFinalParameters` path.

### Q3: What validation could we do?

- **Unknown-key detection (only cheap win).** Walk each `lc` dict's keys and flag anything outside `{"ActivationArgs", "FactorDimArgs", "ICacheFlush"}`. This is a small, contained typo guard.
- **Type checks.** Could enforce that the block is a `list` of single-key `dict`s; that `ActivationArgs` is `list[list[dict]]`; that `FactorDimArgs` is `list[int]`; that `ICacheFlush` is `list[bool]`. Of the four, the outer `list` shape is the only one that protects against truly silent corruption — the inner shapes are already trapped (or warned on) by the downstream constructors.
- **No good case for enum/range checks here.** `FactorDimArgs` already warns on `{0,1}`. `ActivationArgs` already exits on missing `Enum`. `ICacheFlush` is a free `list[bool]`.
- **Reason to leave it alone (mostly):** practically zero in-tree usage; the same key names *do* have a validated home inside `BenchmarkFinalParameters`. The validator could either skip this block entirely or apply the cheap unknown-key + outer-shape check and stop there. Recommend not investing further until there is a real test that exercises a non-empty `LibraryClient:` body.
