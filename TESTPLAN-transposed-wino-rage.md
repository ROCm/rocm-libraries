# Test plan — registering `TransposedConvWinoRageRxS<2,3>` in the Find solver lists

Covers the change on branch `users/dahawkin/register-transposed-wino-rage`
(commit "fix(miopen): register TransposedConvWinoRageRxS in Find solver lists").

## 1. What changed

Two lines in `projects/miopen/src/mlo_dir_conv.cpp`, each adding
`miopen::solver::conv::TransposedConvWinoRageRxS<2, 3>` beside the existing
`TransposedConvWinoFuryRxS<2, 3>` entry:

| Container | Line | Covers |
|---|---|---|
| `GetWindogradSolvers()` | 123 | Forward and backward-data |
| `GetWindogradWrWSolvers()` | 183 | Backward-weights |

Plus one CHANGELOG entry. No solver logic, applicability, kernel, or perf-db
change.

## 2. Why a gfx950 machine is required

`ConvWinoRageRxSCommon::IsApplicable` (`src/solver/conv/conv_wino_rage_RxS.cpp:317-322`)
rejects any device that is not `gfx942*`, `gfx950*`, or `gfx120*`. The unit-test
params for this solver (`GetTestParams()`,
`test/gtest/unit_conv_solver_ConvWinoRageRxS.cpp:73-81`) declare
`Gpu::gfx94X | Gpu::gfx950 | Gpu::gfx120X`, so on any other device the GPU
suites skip rather than fail.

`TransposingSolver` (`src/include/miopen/utility/transposing_solver.hpp`) reports
`MayNeedWorkspace() == true` and `IsDynamic() == false`, so the solver is
reachable only through Find, never immediate mode. Its `IsApplicable` also
returns false when no tensor layout differs from the `NCDHW` target
(`transposing_solver.hpp:960-964`), so **only NHWC problems can select it**. An
NCHW problem will correctly report not-applicable; that is not a failure.

## 3. Facts the tests assert against

Numeric solver ids, read from the built library on this branch:

| Solver | Id |
|---|---|
| `ConvWinoRageRxS<2-3>` | 180 |
| `TransposedConvWinoFuryRxS<2-3>` | 190 |
| `TransposedConvWinoRageRxS<2-3>` | 191 |

Confirm these on the gfx950 machine before relying on them — ids shift when the
registry in `src/solver.cpp` changes.

## 4. Build

```bash
cd <worktree>
cmake -G Ninja -S projects/miopen -B build-miopen \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DMIOPEN_BACKEND=HIP \
  -DGPU_TARGETS=gfx950 \
  -DBUILD_TESTING=On
ninja -C build-miopen > build.log 2>&1 || { tail -100 build.log; false; }
```

Build the driver as well (`-DMIOPEN_BUILD_DRIVER=On`, the default) — sections 6
and 7 need `bin/MIOpenDriver`.

Two build notes carried over from the x86/gfx1151 dev box, both **expected to
disappear on a gfx950 host with a matching ROCm**. If either reappears, it is a
pre-existing environment problem, not this change:

- Vendored hipconv cdna4 kernels fail to compile against ROCm 7.2.4 clang
  (`__builtin_amdgcn_ds_read_tr16_b64_v4i16` address-space mismatch in
  `src/hipconv/src/arch/cdna4/direct/direct_l1/transpose_weights.h:345`).
  Workaround if needed: `-DMIOPEN_USE_HIPCONV=Off`. Only do this for the
  functional runs in sections 5-6; section 7 perf numbers must be taken with
  hipconv at its default `On`, because `ConvHipConv` is a competing NHWC solver.
- `-Wmissing-noreturn` fires as an error in `src/include/miopen/pooling/solvers.hpp`
  under newer clang.

## 5. Level 1 — enumeration and registry (fast, no GPU dependency on the result)

These are the direct regression guards for the defect.

### 5.1 Find-list/registry drift guard

```bash
ninja -C build-miopen test_solver_find_list_registry
./build-miopen/bin/test_solver_find_list_registry
```

Expect: `PASSED`. This proves the newly listed solver resolves in the solver Id
registry and `solver::Id::GetAlgo()` does not throw. Already verified green
off-target.

Note the coverage gap: this test only catches Find-list entries missing from the
registry. It cannot catch the reverse — a registered solver missing from a Find
list — which is exactly the defect being fixed. Closing that is story S5 and is
out of scope here.

### 5.2 Enumeration proof

The solver must now appear in the Find lists. Build the throwaway probe:

```bash
cat > /tmp/dump_find_ids.cpp <<'EOF'
#include <miopen/mlo_internal.hpp>
#include <cstdio>
int main()
{
    for(const auto& id : GetAllFindSolverDbIds())
        std::printf("%s\n", id.c_str());
}
EOF

cd build-miopen
/opt/rocm/lib/llvm/bin/clang++ -std=c++20 -x hip -U__HCC__ \
  -D__HIP_PLATFORM_AMD__=1 -DMIOPEN_BETA_API=1 --offload-arch=gfx950 \
  -I../projects/miopen/src/include -Iinclude -I../projects/miopen/include \
  -isystem /opt/rocm/include \
  -isystem _deps/nlohmann_json-src/include \
  -isystem _deps/sqlite3-src -isystem _deps/bzip2-src \
  /tmp/dump_find_ids.cpp -Llib -lMIOpen -Wl,-rpath,$PWD/lib -o /tmp/dump_find_ids
/tmp/dump_find_ids | grep -c 'TransposedConvWinoRageRxS<2-3>'
```

Expect: `2` — one from `GetWindogradSolvers()`, one from
`GetWindogradWrWSolvers()`. On `develop` the same command prints `0`. Already
verified off-target (both the `2` and the `0`).

## 6. Level 2 — correctness on gfx950

### 6.1 Solver unit tests

```bash
ninja -C build-miopen test_unit_conv_solver_ConvWinoRageRxS
./build-miopen/bin/test_unit_conv_solver_ConvWinoRageRxS \
  --gtest_filter='*TransposedWinoRageRxS*'
```

Covers Fwd, Bwd, and WrW FP16 NHWC suites plus the CPU device-applicability
suite, defined at `unit_conv_solver_ConvWinoRageRxS.cpp:218-268`. Shapes include
grouped (`g=2`), `g=1`, asymmetric `R!=S` (3x5), and — for WrW — degenerate
`H=W=1` spatial dims that exercise the `GetGroupConvLayout` layout-string path.

Expect: all pass, none skipped. A skip means the arch gate did not see gfx950 —
check `rocminfo`.

Run the non-transposed suite too, to prove no collateral damage:

```bash
./build-miopen/bin/test_unit_conv_solver_ConvWinoRageRxS
```

These tests call `RunTest(solver{})` directly and therefore already passed before
the change. They are the control, not the proof.

### 6.2 The actual proof — Find selects the solver

This is the criterion the change exists for: an NHWC convolution issued through
the public Find API must now be able to land on solver 191. Use the driver, FP16,
with a shape drawn from the unit tests.

```bash
export MIOPEN_FIND_MODE=NORMAL          # full find, no FindDb short-circuit
export MIOPEN_LOG_LEVEL=6

./build-miopen/bin/MIOpenDriver convfp16 \
  -I NHWC -O NHWC -f NHWC \
  -n 1 -c 40 -H 20 -W 20 -k 20 -y 3 -x 3 \
  -p 1 -q 1 -u 1 -v 1 -g 2 \
  -F 1 -S -1 -V 1 -t 1 -i 1 2>&1 | tee find-fwd.log

grep -i 'TransposedConvWinoRageRxS' find-fwd.log
```

Expect: the solver appears in the enumerated/benchmarked set, and verification
reports OK. Repeat with `-F 2` (bwd-data) and `-F 4` (wrw); `-F 4` is what
exercises the second registration.

Run the identical three commands against a `develop` build. There the grep must
return nothing — that contrast is the regression evidence.

### 6.3 Forced-selection run

Pin Find to the solver so nothing else can win, which turns "was enumerable" into
"was executable and numerically correct":

```bash
MIOPEN_FIND_MODE=NORMAL \
MIOPEN_DEBUG_FIND_ONLY_SOLVER=191 \
./build-miopen/bin/MIOpenDriver convfp16 \
  -I NHWC -O NHWC -f NHWC \
  -n 1 -c 20 -H 20 -W 20 -k 20 -y 3 -x 3 \
  -p 1 -q 1 -u 1 -v 1 -g 1 \
  -F 1 -S -1 -V 1 -t 1 -i 1
```

Expect: runs and verifies OK. On `develop` the same command must fail to find any
solution — the solver is registered, so the env var parses, but Find never
enumerates it.

Confirm the numeric id first (`MIOPEN_DEBUG_FIND_ONLY_SOLVER` also accepts the
string `TransposedConvWinoRageRxS<2-3>`, which is safer against id drift).

### 6.4 Workspace agreement

The solver reports a non-zero workspace (inner solver + transpose buffers,
`transposing_solver.hpp:974-989`), and `GetWorkSpaceSizeWinograd`
(`src/convolution.cpp:98-104`) takes the max across the Find list. Adding a
solver to that list can therefore raise the API-reported workspace.

Check that the API number is at least what the chosen solution asks for, on the
same shape as 6.2. `MIOPEN_ENABLE_LOGGING_CMD=1` plus `MIOPEN_LOG_LEVEL=6` prints
both the requested workspace and the per-solution `workspace_sz`.

Expect: API-reported size >= the selected solution's `workspace_sz`, and no
`Not enough workspace` throw. An increase here is correct behaviour, not a
regression — but note the delta, because it is user-visible allocation growth.

### 6.5 Kill switches still work

```bash
MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3=0 <driver cmd from 6.2>
MIOPEN_DEBUG_CONV_WINOGRAD=0             <driver cmd from 6.2>
```

Expect: in both cases the solver disappears from the log and the run still
completes on some other solver. This proves the new list entry respects the
existing disable paths.

### 6.6 NCHW is unaffected

```bash
./build-miopen/bin/MIOpenDriver convfp16 \
  -n 1 -c 40 -H 20 -W 20 -k 20 -y 3 -x 3 -p 1 -q 1 -g 2 \
  -F 1 -S -1 -V 1 -t 1 -i 1
```

Expect: verification OK, and `TransposedConvWinoRageRxS` absent — the
transposing wrapper is not applicable when there is no layout difference. Its
presence here would indicate a real bug.

## 7. Level 3 — performance and selection impact

The point of the fix is that a previously unreachable solver can now win. Two
things to measure, both on gfx950 with hipconv at its default `On`:

1. **Does it ever win?** For the NHWC shapes in 6.2, run untuned Find (`-S -1`,
   `-t 1`) and record the winning solver and time before and after the change.
   Non-selection is an acceptable outcome and not a test failure — the fix
   restores an option, it does not promise a win. Record it either way.
2. **Does anything get slower?** Adding a solver to a Find list costs
   enumeration and, on a cold cache, compilation time. Compare wall time of the
   first Find call before and after on the same shape.

Beyond these ad-hoc shapes, this is the acceptance criterion the corpus-replay
harness (backlog Epic F / S2) exists to answer properly: replay the 31-model NHWC
corpus and diff per-shape winning solver and time against a `develop` baseline.
That harness does not exist yet, so treat section 7 as indicative rather than
exhaustive.

## 8. Regression sweep

Before raising the PR, on gfx950:

```bash
# Winograd solver unit tests as a family
for t in build-miopen/bin/test_unit_conv_solver_Conv*Wino*; do
  echo "== $t"; "$t" || echo "FAILED: $t"
done

# Find-list guard
./build-miopen/bin/test_solver_find_list_registry
```

Expect: no new failures relative to a `develop` build on the same machine. Take
the `develop` baseline first — several of these suites have pre-existing skips
and it is easy to misread a skip as a regression.

## 9. Exit criteria

| # | Criterion | Blocking |
|---|---|---|
| 1 | `test_solver_find_list_registry` passes | Yes |
| 2 | Enumeration probe (5.2) prints `2` on the branch and `0` on `develop` | Yes |
| 3 | `*TransposedWinoRageRxS*` unit suites pass on gfx950, none skipped | Yes |
| 4 | Find selects or at least enumerates solver 191 for an NHWC fwd, bwd, and wrw problem (6.2) | Yes |
| 5 | Forced-selection run (6.3) verifies OK | Yes |
| 6 | API-reported workspace >= selected solution's `workspace_sz`; no throw (6.4) | Yes |
| 7 | Both kill switches suppress the solver (6.5) | Yes |
| 8 | NCHW path unchanged (6.6) | Yes |
| 9 | No new failures in the Winograd unit-test family vs. `develop` (8) | Yes |
| 10 | Perf delta recorded for the exercised NHWC shapes (7) | No — record only |

## 10. Known gaps

- No automated test asserts that a registered transposing solver is enumerable
  through Find. That is the defect class this change represents, and section 5.2
  covers it only manually. Story S5 in the backlog owns closing it.
- Sections 6.2 and 7 use hand-picked shapes. Corpus-wide evidence needs the
  Epic F replay harness.
- `gfx942` and `gfx120X` also pass the solver's arch gate. The change is
  arch-independent, so gfx950 coverage is sufficient for merge, but a gfx942 run
  of section 6.1 is cheap insurance if a machine is available.
