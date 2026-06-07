# Parametric-Chaos Run-2 Validation Report

Generated: 2026-06-07  
Branch set: 20 unique branch IDs (PublicInputSurfaceDeep)  
Validation methods used: z3 (bounded SAT / UNSAT), CrossHair (bounded symbolic), pytest pass-check (CPU-only)

---

## Per-unit table

| branch_id (12) | file:line | classification | solver_status | confirmed | reified? |
|---|---|---|---|---|---|
| `0643ca620d99` | Tensile/BenchmarkProblems.py:302 | derived-local | SAT | yes | yes (test_pchaos_BenchmarkProblems_L302_char.py) |
| `09380ac263b6` | Tensile/Toolchain/Validators.py:237 | runtime-dependent | UNKNOWN | yes | no |
| `099093bf09ab` | Tensile/BenchmarkProblems.py:557 | solver-backed-under-assumptions | SAT | yes | yes (test_pchaos_BenchmarkProblems_L557_char.py) |
| `0d3cd6b0f663` | Tensile/BenchmarkProblems.py:304 | fully-static | SAT | yes | yes (test_pchaos_BenchmarkProblems_L304_char.py) |
| `3ae422d17a07` | Tensile/ClientWriter.py:366 | runtime-dependent | SAT | yes | no |
| `6647a7e665fa` | Tensile/BenchmarkProblems.py:586 | solver-backed-under-assumptions | SAT | yes | yes (test_pchaos_BenchmarkProblems_L586_char.py) |
| `6869457874b8` | Tensile/LibraryIO.py:701 | solver-backed-under-assumptions | SAT | yes | yes |
| `6ff09bcdcc57` | Tensile/BenchmarkProblems.py:657 | solver-backed-under-assumptions | SAT | no | yes (test_pchaos_BenchmarkProblems_L657_char.py) |
| `83e4a1ea64ad` | Tensile/BenchmarkProblems.py:133 | runtime-dependent | UNKNOWN | no | no |
| `85cf4fadab76` | Tensile/ClientWriter.py:798 | runtime-dependent | UNKNOWN | yes | no |
| `8e797886ed0f` | Tensile/BenchmarkProblems.py:740 | solver-backed-under-assumptions | SAT | yes | yes (test_pchaos_BenchmarkProblems_L740_char.py) |
| `8fc5b4598eb9` | Tensile/Toolchain/Validators.py:226 | solver-backed-under-assumptions | SAT | yes | no |
| `927cbfe5d810` | Tensile/ClientWriter.py:787 | runtime-dependent | UNKNOWN | no | no |
| `9a47d378ae60` | Tensile/Toolchain/Validators.py:236 | solver-backed-under-assumptions | SAT | yes | yes |
| `bfe92c77b1f3` | Tensile/Toolchain/Validators.py:73 | runtime-dependent | UNKNOWN | no | no |
| `c03b6953169e` | Tensile/Toolchain/Validators.py:97 | runtime-dependent | SAT | yes | yes (test_pchaos_Validators_L97_char.py) |
| `cc98dba04c70` | Tensile/Toolchain/Validators.py:86 | runtime-dependent | SAT | yes | no |
| `d2f6f0df95db` | Tensile/LibraryIO.py:689 | solver-backed-under-assumptions | SAT | yes | no |
| `e278ed047bbb` | Tensile/ClientWriter.py:574 | runtime-dependent | UNKNOWN | yes | no |
| `ffb27402fcf8` | Tensile/Toolchain/Validators.py:195 | runtime-dependent | SAT | yes | no |

---

## Summary counts

| metric | count |
|---|---|
| Total branches inventoried | 20 |
| SAT | 14 |
| UNSAT | 0 |
| UNKNOWN | 6 |
| Witnesses confirmed | 16 |
| Tests reified | 9 |

---

## UNKNOWN branches — explicit statements

**`09380ac263b6` — Tensile/Toolchain/Validators.py:237 (`_exeExists(Path(file))`)**  
Solver status: UNKNOWN. Confirmed: yes.  
Rationale: Re-read REAL source: Validators.py:237 is `if _exeExists(Path(file)): return file` else (L238) raise FileNotFoundError; _exeExists (L200-210) == `os.access(file, os.X_OK)`. Predicate is a pure filesystem probe over CLI input `file`, no symbolic structure -> z3 genuinely unknown for the concrete predicate; claim self-classifies runtime-dependent and does NOT over-assert SAT. Re-executed all 3 claim

**`83e4a1ea64ad` — Tensile/BenchmarkProblems.py:133 (`not os.path.isfile(cachePath)`)**  
Solver status: UNKNOWN. Confirmed: no.  
Rationale: Checker independently re-established the claim against REAL code. (1) Read Tensile/BenchmarkProblems.py:121-138 in-container: line 133 is verbatim `if not os.path.isfile(cachePath):` / `return None`, the cache-miss guard-return inside _readCacheIfValid. Predicate = `not os.path.isfile(cachePath)`. (2) Claim is solver_status=unknown, classification=runtime-dependent, witnesses by VALUE (z3 triviall

**`85cf4fadab76` — Tensile/ClientWriter.py:798 (`not os.path.isfile(clientExe)`)**  
Solver status: UNKNOWN. Confirmed: yes.  
Rationale: Independently re-established the claim against real code. Read ClientWriter.py:795-806: getClientExecutablePath() sets clientExe = globalParameters.get("PrebuiltClient") then guards `if not os.path.isfile(clientExe):` -> raises FileNotFoundError; else returns clientExe. Matches claimed predicate exactly. Re-executed both polarities against the ACTUAL predicate in-container (tl-char): TRUE clientEx

**`927cbfe5d810` — Tensile/ClientWriter.py:787 (`not os.path.exists(metaDataFilePath)`)**  
Solver status: UNKNOWN. Confirmed: no.  
Rationale: Re-read ClientWriter.py:787 in-container: predicate is exactly `if not os.path.exists(metaDataFilePath):` (else-branch of `if problemTypeDict`). Re-executed both claimed examples against the REAL os.path.exists in tl-char: TRUE example /tmp/lib_absent/library/gfx90a/metadata.yaml left absent -> not os.path.exists==True (printExit branch taken, matches claim); FALSE example /tmp/lib_present/library

**`bfe92c77b1f3` — Tensile/Toolchain/Validators.py:73 (`Path(defaultPath).exists()`)**  
Solver status: UNKNOWN. Confirmed: no.  
Rationale: Re-established against REAL code in tl-char (docker exec -w /work/projects/hipblaslt/tensilelite). (1) sed confirms line 73 is `if Path(defaultPath).exists():` inside _windowsSearchPaths. (2) defaultPath = DEFAULT_ROCM_BIN_PATH_WINDOWS = Path(C:/Program Files/AMD/ROCm) (Validators.py:35); inspect.getsource shows defaultPath bound directly to the constant, NO public input (CLI/YAML/env/arith) flows

**`e278ed047bbb` — Tensile/ClientWriter.py:574 (`os.path.exists(sourceDir)`)**  
Solver status: UNKNOWN. Confirmed: yes.  
Rationale: Read real code in-container: ClientWriter.py:574 in writeClientConfigIni is exactly `assert os.path.exists(sourceDir), f"sourceDir={sourceDir} does not exist"`, matching the claim. Re-executed both witnesses in-container (docker exec tl-char python3): FALSE example sourceDir='/nonexistent/path/source' -> os.path.exists=False -> AssertionError 'sourceDir=/nonexistent/path/source does not exist' (re

---

## Runtime-dependent branches (never silently asserted)

The following branches have predicate truth values that depend on runtime state (filesystem, OS type, environment variables) and are **not** fully captured by any CLI/YAML/global-parameter public input:

- `09380ac263b6` Tensile/Toolchain/Validators.py:237 — `_exeExists(Path(file))`
- `3ae422d17a07` Tensile/ClientWriter.py:366 — `os.name != "nt"`
- `83e4a1ea64ad` Tensile/BenchmarkProblems.py:133 — `not os.path.isfile(cachePath)`
- `85cf4fadab76` Tensile/ClientWriter.py:798 — `not os.path.isfile(clientExe)`
- `927cbfe5d810` Tensile/ClientWriter.py:787 — `not os.path.exists(metaDataFilePath)`
- `bfe92c77b1f3` Tensile/Toolchain/Validators.py:73 — `Path(defaultPath).exists()`
- `c03b6953169e` Tensile/Toolchain/Validators.py:97 — `os.environ.get("ROCM_PATH")`
- `cc98dba04c70` Tensile/Toolchain/Validators.py:86 — `not os.name == "nt"`
- `e278ed047bbb` Tensile/ClientWriter.py:574 — `os.path.exists(sourceDir)`
- `ffb27402fcf8` Tensile/Toolchain/Validators.py:195 — `os.name == "nt"`
