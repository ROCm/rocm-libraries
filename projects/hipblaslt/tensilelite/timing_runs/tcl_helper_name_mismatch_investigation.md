# TensileCreateLibrary helper name/object mismatch investigation (all shipped logic)

Findings only, no code change ([DECISIONS.md](DECISIONS.md), topic 8, "Investigation (findings
only, no fix)", plan step P5). The question comes from
[python_kernel_bench_setup_investigation.md](python_kernel_bench_setup_investigation.md),
"Not recommended: reuse `TensileCreateLibrary.generateKernelHelperObjects`": the function dedupes
helper kernels on names predicted by the `*KernelNames` functions, and those names do not match the
objects it then builds. Can that drop a helper, duplicate one, or pair a name with the wrong object
for any real configuration?

Code state: commit `2083e6254bf` (the Python under `tensilelite/` is identical to `55e29436e9a`).
Logic: every `.yaml` under `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full` (3,313
files, all architectures). Every count below comes from running the real TensileCreateLibrary code
on these files; code reading was used to explain the counts. Inferences are marked as such.

## Summary

- **One real configuration drops helpers: the `gfx1250-strict` build.** It is missing
  `Cijk_SB_PostGSU1_VW1`, `_VW2` and `_VW4` (the `SingleBuffer` output-conversion kernel for
  `F8BS`/`BBS`-style problems: float workspace, bf16 output, no bias). The one `SingleBuffer`
  solution in `gfx1250-strict/gfx1250-strict/Equality/gfx1250-strict_Cijk_Alik_Bljk_F8BS_BH_UserArgs.yaml`
  calls for them (the benchmark path would emit them); the library's helper list does not declare
  them (checked on the real output, see "Confirming the drop"). `gfx1250-strict` is always built as
  its own TensileCreateLibrary run (it is a stepping of `gfx1250`), so this applies to the default
  `all` build too. Cause: the predicted name uses the solution's `GlobalSplitU`, so a
  `MultipleBuffer` solution with `GlobalSplitU: 0` and the `SingleBuffer` solution with
  `GlobalSplitU: 0` predict the same `..._PostGSU0_VW*` names while needing different objects
  (`PostGSU16` and `PostGSU1`). The `MultipleBuffer` one comes first, so the `SingleBuffer` objects
  are never built.
  **At run time the missing kernels cannot be requested** (from the host code, not run): this
  solution is persistent data-parallel (`StreamKForceDPOnly: 1`), so it has no accumulation buffer,
  GSU is fixed at 0 and the output-conversion kernel is never launched for it (see "Runtime reach").
  The shipped library is not broken today; the defect is the order-dependent dedupe, which would
  break a library as soon as a `SingleBuffer` solution that does use the conversion kernel sits
  behind a same-name `MultipleBuffer` one.
- **No other configuration drops or adds a helper.** All 18 single-arch builds except
  `gfx1250-strict`, `gfx942 --experimental`, `gfx950 --enable-gemm-a2a-fusion`, the default
  multi-arch `all` build (16 archs in one run) and every logic file built on its own (3,305 files
  that reach the helper step): the helper set equals the set the objects call for.
- **No duplicates reach `Kernels.cpp`, but most helper objects are built more than once.** The final
  `dict.fromkeys` dedupes on kernel name, so the output never repeats a name. Before it, 1,807
  objects are built for 265 kept on `gfx942` (2,189 for 427 in the `all` build), because the proxy
  names differ between solutions that need the same objects.
- **Positional mispairing happens but never selects the wrong set.** `itertools.compress` pairs
  3 predicted names with 2 objects (`ComplexFloat`) or 1 object (`ComplexDouble`) in 34 calls on
  `gfx942`, 4 on `gfx950`, 2 each on `gfx1250`/`gfx1250-strict` (692 summed over per-file builds).
  In every one of them the mask is all `True`, so all objects are kept whatever the pairing. A
  partially-visited mask together with a length mismatch, the case that would keep the wrong object,
  occurs 0 times.
- **`MultipleBufferSingleKernel` with `AdaptiveGemmGSUA` is not reachable today**: no shipped
  solution has `AdaptiveGemmGSUA != 0` (0 of the 682,765 solutions of the `all` build, the 2,868 of
  `gfx1250-strict`, and the gfx1102 / experimental / GEMM+A2A captures).
- **Related, not caused by the name prediction: two different BetaOnly kernels share one name.**
  `Cijk_I_GA` (gfx950: `I8IS` with float compute and `I8II` with int32 compute) and, in the `all`
  build, also `Cijk_I8_GA` (gfx950 `I8I8S` float compute versus gfx1170 int32 compute). The name
  omits the compute type, so the name dedupe keeps whichever comes first and the other problem type
  is left with a beta-only kernel of the wrong `D`/`beta` types. Which one is kept depends on the
  build: the gfx950-only build keeps the float version, the `all` build keeps the int32 one (from
  gfx1101). **Latent at run time**: every solution that produces these names is `MultipleBuffer` or
  `MultipleBufferSingleKernel` with `AdaptiveGemmGSUA: 0`, and the host never launches a beta-only
  kernel for those, so the wrong kernel is not reached today. It would be if a `SingleBuffer` (or
  adaptive) solution of one of these types were added. The benchmark path dedupes by name the same
  way, but a benchmark step has one `ProblemType`, so it cannot hit this.

## Method

Scripts are throwaway, in `/tmp/tl_plan/P5/` (not in the repo).

1. **Capture** (`capture.py <arch> <out.pkl> <jobs> [TCL flags]`). Runs the real
   `TensileCreateLibrary.Run.run()` with the same arguments as the CMake build
   (`--architecture=<arch>`, `--cxx-compiler=/opt/rocm/llvm/bin/clang++`, lazy loading on,
   experimental off unless asked), so logic-file selection, parsing, merging and solution order are
   the real ones. Two populations appear below. TCL's **unique solutions** are the solution list
   after the identity dedupe at `Run.py:1203-1204` ("Number of unique solutions" in the log).
   `run()` then reduces it to the **kernel list** with `generateKernelObjectsFromSolutions`
   (dedupe on `getKeyNoInternalArgs`, `Run.py:884-895`) and passes that to
   `generateKernelHelperObjects` (`Run.py:1515-1516`). Two wrappers are installed by monkeypatching
   (no repo edit):
   - `generateKernelObjectsFromSolutions`: records the unique-solution list (before kernel dedupe),
     for the per-file analysis and a second replay check.
   - `generateKernelHelperObjects`: calls the real function and records its output (kernel name +
     SHA-1 of header and source of every object, in order). Then, for every kernel and every helper
     type, records the raw predicted names (`kernelObjectNameCallables()`) and the names and source
     hashes of the objects `initHelperKernelObjects(kernel, type, ...)` returns. Then it stops the
     run (no code objects are built).

   Per-kernel data is memoized on the fields the helper code reads (canonical `ProblemType` state,
   `GlobalSplitU`, `GlobalSplitUAlgorithm`, `isStreamK`, `StreamKAtomic`, `_GlobalAccumulation`,
   `AdaptiveGemmGSUA`, `ActivationFused`, `WavefrontSize`, `ISA`, `GlobalSplitUPGR`). The memo was
   checked against computing every kernel without it on gfx1103 (1,019 kernels, 0 differences), and
   every replay below reproduces the real output exactly.
2. **Replay** (`analyze.py`, `report.py`). Re-runs the `generateKernelHelperObjects` loop on the
   recorded strings (same callable order, same `"Enum"` sort with `reverse=True`, same `visited` set,
   same `itertools.compress`, same `dict.fromkeys` and final sort) and checks it against the real
   output: **identical name, source hash and order for every capture below**, both on the kernel list
   and on the unique-solution list. All per-configuration statistics (required, kept, built, length
   mismatch, partial mask, kernel indices) are computed on the kernel list, which is what the real
   function sees. The replay then compares with the **required set**: the union, by
   name, of the objects `initHelperKernelObjects` returns for every kernel, i.e. what the benchmark
   path (`writeBenchmarkFiles`, which builds every object and dedupes by name) would emit for the
   same solutions. It counts missing and extra names, names whose objects have more than one source,
   objects built before dedupe, calls where the name and object lists differ in length, calls with a
   partially-visited mask, and selected positions whose object does not match the predicted name.
   "Needed but skipped" counts object names a solution needed that were not built at that point
   (whether or not a later solution built them).
3. **Per logic file**: the same replay on each file's unique solutions in their real order, which is what
   a TensileCreateLibrary run with `--logic-filter` naming that one file does (kernel dedupe within
   a file never changes the result: a repeated kernel has the same names and they are all visited).
4. **Mechanism statistics** (`stats.py`), and the drill-down scripts `dig*.py` for the examples.
5. **Confirming the drop** (`check_symbols.py`): runs the real `gfx1250-strict` build up to
   `generateKernelHelperObjects` and greps the header text of the real helper list (what
   `writeHelpers` writes to `Kernels.h`/`Kernels.cpp`) for `__global__` declarations.

Configurations captured: one run per architecture directory (gfx908, gfx90a, gfx942, gfx950,
gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153, gfx1170, gfx1171,
gfx1200, gfx1201, gfx1250, gfx1250-strict), `gfx942 --experimental` (adds the 16
`aquavanjaram/gfx942/Experimental` files), `gfx950 --enable-gemm-a2a-fusion` (adds the one
GEMM+A2A file filtered by default), and the default `all` group: the base architectures from
`cmake/tensilelite_supported_architectures.cmake` in one run, minus `gfx1250-strict`, which
TensileCreateLibrary always builds as a separate run because it shares `gfx1250`'s ISA
(`isaCollisionFreeGroups`). Together the runs select all 3,313 logic files, and 3,305 of them reach
the helper step with at least one solution: 3,288 in the default builds, plus the 16 experimental
files and the GEMM+A2A file. The other 8 are the one-solution files
`*_Cijk_Alik_Bljk_B8BS_BH_BiasS_HAS_SAV_UserArgs.yaml` and
`*_Cijk_Alik_Bljk_B8F8BS_BH_BiasS_HAS_SAV_UserArgs.yaml` of gfx1170, gfx1171, gfx1200 and gfx1201:
their solution is parsed, and then removed by the identity dedupe at `Run.py:1203-1204` because an
equal solution comes from another file of the same arch (`count_yaml.py`).

**Limitation, gfx1170/gfx1171.** The installed ROCm 7.2 compiler does not accept these targets, so
`makeIsaInfoMap` reports `SupportedISA` false and solution construction asserts. For these two (and
for the `all` run, which includes them) `capture_capsub.py` substitutes gfx1200's probed
capabilities. The helper inputs come from the logic files. Every solution entry in the 154 logic
files of each arch is parsed into the library (raw YAML entries 33,972 / 38,153 = "Number of
solutions parsed" in the log; `count_yaml.py`); the identity dedupe then leaves 33,748 / 37,925
unique solutions, and 152 files per arch reach the helper step, the same two-file pattern as gfx1200
and gfx1201 with the real compiler (see above). Fields the generator derives from the capabilities
(and the `Valid` flag, which is not checked before the helper step) could still differ with a
compiler that supports these targets.

## Results

### Per configuration

"Solutions" is TCL's unique solutions (before kernel dedupe); "Kernels" is the list
`generateKernelHelperObjects` receives, and every column to its right is computed on it.
"Required" is the number of distinct helper names the kernels call for; "Kept" is what
`generateKernelHelperObjects` returns. "Built" is objects built before the final name dedupe.
"Len mismatch" and "Partial mask" count `itertools.compress` calls (calls where at least one name
was new).

| Configuration | Logic files | Solutions | Kernels | Required | Kept | Missing | Built | Len mismatch | Partial mask |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gfx908 | 55 | 58 | 58 | 69 | 69 | 0 | 84 | 0 | 1 |
| gfx90a | 108 | 24,597 | 9,670 | 88 | 88 | 0 | 236 | 0 | 3 |
| gfx942 | 844 | 351,748 | 232,710 | 265 | 265 | 0 | 1,807 | 34 | 9 |
| gfx942 `--experimental` | 860 | 355,260 | 236,222 | 265 | 265 | 0 | 1,807 | 34 | 9 |
| gfx950 | 720 | 84,262 | 84,159 | 186 | 186 | 0 (1 name, 2 sources) | 277 | 4 | 5 |
| gfx950 `--enable-gemm-a2a-fusion` | 721 | 84,263 | 84,160 | 186 | 186 | 0 (1 name, 2 sources) | 277 | 4 | 5 |
| gfx1100 | 46 | 1,836 | 1,722 | 57 | 57 | 0 | 96 | 0 | 4 |
| gfx1101 | 60 | 1,554 | 1,358 | 66 | 66 | 0 | 117 | 0 | 4 |
| gfx1102 | 46 | 1,223 | 1,019 | 56 | 56 | 0 | 80 | 0 | 4 |
| gfx1103 | 46 | 1,223 | 1,019 | 56 | 56 | 0 | 80 | 0 | 4 |
| gfx1150 | 47 | 1,472 | 1,290 | 59 | 59 | 0 | 83 | 0 | 4 |
| gfx1151 | 50 | 2,968 | 2,537 | 59 | 59 | 0 | 86 | 0 | 4 |
| gfx1152 | 46 | 1,223 | 1,019 | 56 | 56 | 0 | 80 | 0 | 4 |
| gfx1153 | 47 | 1,269 | 1,061 | 56 | 56 | 0 | 86 | 0 | 4 |
| gfx1170 (caps substituted) | 152 | 33,748 | 32,373 | 114 | 114 | 0 | 243 | 0 | 13 |
| gfx1171 (caps substituted) | 152 | 37,925 | 36,504 | 114 | 114 | 0 | 195 | 0 | 12 |
| gfx1200 | 155 | 84,019 | 77,816 | 108 | 108 | 0 | 192 | 0 | 10 |
| gfx1201 | 152 | 51,930 | 48,098 | 108 | 108 | 0 | 207 | 0 | 9 |
| gfx1250 | 284 | 2,933 | 2,933 | 173 | 173 | 0 | 203 | 2 | 4 |
| **gfx1250-strict** | 278 | 2,868 | 2,868 | 171 | **168** | **3** | 195 | 2 | 4 |
| `all` (16 base archs, one run) | 2,964 | 682,765 | 534,327 | 427 | 427 | 0 (2 names, 2 sources each) | 2,189 | 34 | 27 |

Logic files are those with at least one solution reaching the helper step. In every row the replay
equals the real output, no extra name is produced, and every kept object has the source the
required set calls for (where a name has one source).

**Per logic file** (each file built on its own: 3,288 default files, the 16 experimental ones and
the GEMM+A2A one, 3,305 in total): 0 files with a missing, extra or wrong-source helper; 0 names
with two sources; 0 calls with both a length mismatch and a partial mask; 0 "needed but skipped"
names. Summed over files: 9,036 redundant object builds, 692 length-mismatch calls, 659
partial-mask calls.

### Confirming the drop

On the real `gfx1250-strict` helper list (168 objects, 1,287 declared `__global__` kernels):
`Cijk_SB_PostGSU1_VW1`, `Cijk_SB_PostGSU1_VW2`, `Cijk_SB_PostGSU1_VW4` and `Cijk_SB_GG_PostGSU1_VW1`
are absent; `Cijk_SB_PostGSU16_VW1` and `Cijk_SB_PostGSU2_VW1` are declared. No other object emits
a `PostGSU1` kernel for this name: `PostGSU16` objects emit GSU 16, 8, 4, 2 only
(`KernelWriterConversion.__init__`, `gsuKernels`, `KernelWriterConversion.py:73`).

The sequence in the real kernel-list order ("Kernel index" is the position in the list
`generateKernelHelperObjects` receives):

| Kernel index | Logic file (under `gfx1250-strict/`) | `GlobalSplitU` | Algorithm | Predicted names | Objects |
| ---: | --- | ---: | --- | --- | --- |
| 1426 | `gfx1250-strict/Equality/gfx1250-strict_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` | 0 | MultipleBuffer | `Cijk_SB_PostGSU0_VW{1,2,4}` (new) | `Cijk_SB_PostGSU16_VW{4,2,1}` (built) |
| 1811–1823, 2486 | five MX `F4BS`/`F8BS` files | 0 | MultipleBuffer | same, visited | not built (already present) |
| 2492 | `gfx1250-strict/Equality/gfx1250-strict_Cijk_Alik_Bljk_F8BS_BH_UserArgs.yaml` | 0 | **SingleBuffer** | same, visited | `Cijk_SB_PostGSU1_VW{4,2,1}`: **never built** |

The `Alik_Bljk_F8BS_BH_UserArgs` logic file exists only under `gfx1250-strict/` (three copies:
`gfx1250-strict`, `_96cu`, `_192cu`), and only the `gfx1250-strict/gfx1250-strict` copy has the
`SingleBuffer` solution; no gfx1250 (non-strict) logic file has one, so the gfx1250 build has
nothing that calls for `PostGSU1`. Built on its own, the `F8BS_BH_UserArgs` file is fine (its
`MultipleBuffer` solutions have `GlobalSplitU` 1 and -1, which predict different names).

#### Runtime reach (from the code, not run: no gfx1250 hardware here)

The `SingleBuffer` solution is solution 1 of that file: `GlobalSplitU: 0`, `StreamK: 3`,
`StreamKForceDPOnly: 1` (the only `StreamKForceDPOnly: 1` in the gfx1250 and gfx1250-strict
logic). The generator makes it persistent data-parallel: `Solution.py:2325-2328` sets
`GlobalSplitU = 0`, `SupportUserGSU = False` and `GlobalSplitUAlgorithm = "SingleBuffer"`
(non-StreamK), and the `_GlobalAccumulation` assignment (`Solution.py:2170-2181`) leaves it `None`
for persistent data-parallel solutions, so the library gets `globalAccumulation = 0`
(`Contractions.py:811`; the capture confirms `_GlobalAccumulation` `None` for this solution). On the
host:

- GSU is 0: `calculateAutoGSU` returns `sizeMapping.globalSplitU` unless it is -1
  (`ContractionSolution.cpp:2170`), and a user GSU is rejected because `SupportUserGSU` is off
  (`checkInternalArgumentsSupport`, `ContractionSolution.cpp:5620`). The persistent data-parallel
  branch also throws for GSU > 1 (`ContractionSolution.cpp:5141`).
- The output-conversion launch needs `(gsu > 1 && globalAccumulation && globalAccumulation != 3)`
  or a parallel reduction (`ContractionSolution.cpp:5279`; `getAccumulation` returns the solution's
  `globalAccumulation` when `AdaptiveGemmGSUA` is 0). Both GSU and `globalAccumulation` are 0,
  and `resolvePersistentSettings` leaves the reduction at `none` for a non-StreamK persistent
  solution (`ContractionSolution.cpp:5014-5033`; default in `ContractionSolution.hpp:521`).

So no problem can make this solution ask for `Cijk_SB_PostGSU1_VW*`. The conversion objects
`initConversionKernelObjects` builds for it are themselves unused: it keys on
`GlobalSplitUAlgorithm` only, not on `_GlobalAccumulation`, so the benchmark path emits kernels
that nothing launches. The drop is real in the build output but harmless for this logic. It is
not harmless in general: a `SingleBuffer` solution with `GlobalSplitU` > 1 or -1 (`_GlobalAccumulation:
SingleBuffer`, `globalAccumulation = 1`) launches `..._PostGSU1_VW<vw>` whenever the chosen GSU is
above 1 (`gsu = 1` for `globalAccumulation == 1` before `outputConversionKernelName`), and the same
ordering would leave it without its kernel.

### Mechanisms

The name functions and object builders disagree in four ways
(`KernelHelperNaming.py`; `generateKernelHelperObjects` is `TensileCreateLibrary/Run.py:897`):

| Difference | Names | Objects | Effect in shipped logic |
| --- | --- | --- | --- |
| `PostGSU` number | `KernelWriterConversion.kernelName` with the solution's `GlobalSplitU` (`PostGSU0`, `PostGSU1`, `PostGSU-1`, `PostGSU7`, ...) | `GlobalSplitUPGR` (16), or 1 for `SingleBuffer` (`KernelHelperNaming.py:171`, `:185`) | Solutions that need the same objects predict different names (most of the redundant builds). Solutions that need different objects can predict the same name: the gfx1250-strict drop. |
| `_GG` tag | solution's `GroupedGemm` | `GroupedGemm = False` (each object emits both variants) | More redundant builds only. |
| Vector widths | `[1, 2]` if `isDouble()` else `[1, 2, 4]` (`:51`) | by `numBytes()`: `[1]` if > 8, `[1, 2]` if > 4, else `[1, 2, 4]` (`:162`) | `ComplexFloat`: 3 names / 2 objects; `ComplexDouble`: 3 names / 1 object. After the sort the names are `VW4, VW2, VW1` against objects `VW2, VW1` (or `VW1`), so every position is paired with the wrong object, and `compress` ignores the surplus mask entries. Harmless only because the mask is all-`True` in every such call. |
| `MultipleBufferSingleKernel` | returns `None` (no names) | objects when `AdaptiveGemmGSUA != 0` (`:172`) | Such objects would never be built; no shipped solution has `AdaptiveGemmGSUA != 0`. |

`BetaOnly`, `Reduction`, `ActivationEnumHeader` and `ActivationFunction` names match their objects
apart from `_GG`.

Why the lucky outcomes hold, from the data:

- The length mismatch only occurs for complex types, whose problem types have no bias list, so all
  names of one call are new together or visited together. The partial masks (27 calls in the `all`
  build: 19 conversion, 7 beta-only, 1 reduction) come from problem types sharing some bias-typed
  names (e.g. an `HSS` problem type with bias types `[H, S]` whose `BiasS` names an earlier problem
  type already claimed: mask `BiasH` new, `BiasS` visited) and from beta-only lists
  that repeat one name per bias type when `BetaOnlyUseBias` is off (`Cijk_B_GA`, `Cijk_B_GA`). In
  all of them the name and object lists have the same length and the same order (vector width
  outer, bias type inner), so `compress` keeps the right objects.
- Predicted names that map to more than one object list are common (e.g. 41 of 2,089 on gfx942, 59
  of 192 on gfx1200): bias-typed names shared between problem types with different bias lists,
  `Cijk_*_GA` beta-only names, and `PostGSU<n>` names shared by solutions with different objects.
  They only drop something when the first solution to claim the name does not build the object a
  later solution needs, and no other solution builds it later. "Needed but skipped" is 0 in every
  configuration except gfx1250-strict (3, never built).

### Same name, different source (`BetaOnly`)

`KernelWriterBetaOnly.kernelName` (`KernelWriterBetaOnly.py:301`) encodes `DestDataType`, grouped /
batched, bias type and `_GA`, but not `ComputeDataType`, while the kernel's `D` and `beta` types come
from `ComputeDataType` (`KernelWriterBetaOnly.py:42`). The host name (`betaOnlyKernelName`,
`ContractionSolution.cpp:4010`) omits it too. Found:

| Build | Name | Sources (first solution) | Kept |
| --- | --- | --- | --- |
| gfx950 | `Cijk_I_GA` | `gfx950_Cijk_Ailk_Bjlk_I8IS_BH_HAS_Bias_HAS_SAV_UserArgs` (`float * D`, `float beta`); `gfx950_Cijk_Ailk_Bljk_I8II_BH_AI_SAV_UserArgs` (`int32_t * D`, `int32_t beta`) | float version |
| `all` | `Cijk_I_GA` | `navi32_Cijk_Ailk_Bljk_I8II_BH_UserArgs` (int32) first; the gfx950 `I8IS` file (float) | int32 version |
| `all` | `Cijk_I8_GA` | `gfx1170_Cijk_Ailk_Bjlk_I8BH_SHB_HA_S_SCD_SAV_UserArgs` (int32 compute) first; `gfx950_Cijk_Ailk_Bjlk_I8I8S_BH_HAS_Bias_HAS_SAV_UserArgs` (float compute) | int32 version |

The diff of the two `Cijk_I_GA` sources is exactly the `D` pointer type, the `beta` type and the
`SCALAR_ZERO`/multiply casts (`float` versus `int32_t`). Both the TensileCreateLibrary path and the
benchmark path keep the first object per name, so this is a helper naming collision, independent of
the name prediction. The gfx1170 side of the `Cijk_I8_GA` collision comes from the
capability-substituted capture.

Runtime reach (from the code, not run): the only beta-only launch is in `solve()` under
`gsu > 1 && globalAccumulation != 2 && globalAccumulation != 3` (`ContractionSolution.cpp:5127`; the
grouped-GEMM call at `:5416` is commented out), so `MultipleBuffer` (2) and
`MultipleBufferSingleKernel` (3) solutions never launch it. Every solution that produces a
colliding name, in every capture, is one of those with `AdaptiveGemmGSUA: 0` and not StreamK
(counts are unique solutions, kernels in parentheses): gfx950 `Cijk_I_GA` 11 (11) `MultipleBuffer`
(GSU 2); in the `all` build `Cijk_I_GA` 3,705 (2,824) `MultipleBuffer` (GSU -1 or 2–30) and
1,566 (1,444) `MultipleBufferSingleKernel`, `Cijk_I8_GA` 2,827 (2,396) `MultipleBuffer` and
4,909 (4,000) `MultipleBufferSingleKernel`. So the wrong-typed kernel is never launched with shipped logic;
it would be if a `SingleBuffer` solution with GSU > 1 or -1 (or an adaptive one) of one of these
problem types were added.

## Conclusion

- **Drop: yes, in one shipped configuration, harmless at run time today.** The `gfx1250-strict`
  library lacks `Cijk_SB_PostGSU1_VW{1,2,4}` (and their `_GG` variants), which one `SingleBuffer`
  solution calls for in the build. That solution is persistent data-parallel and never launches the
  conversion kernel, so nothing requests the missing kernels. The cause is the `PostGSU` number in
  the predicted name; it depends on solution order, so any logic change can add or remove such drops
  without warning, including ones a launching `SingleBuffer` solution would hit.
- **Duplicate: no duplicate names reach the output** (name dedupe at the end). Objects are built
  redundantly (up to ~7× on gfx942), a cost, not a correctness issue.
- **Mispair: the positional pairing is wrong for every complex-type call** (3 names, 1–2 objects),
  but no shipped configuration hits the combination (length mismatch with a partially-visited mask)
  that would select the wrong objects. `MultipleBufferSingleKernel` + `AdaptiveGemmGSUA` would drop
  helpers but is not used by shipped logic.
- **Separately**, the `BetaOnly` name omits the compute type, so `Cijk_I_GA` (gfx950 and `all`) and
  `Cijk_I8_GA` (`all`) each stand for two different kernels and one problem type gets the other's.
  Latent: all solutions involved are `MultipleBuffer`/`MultipleBufferSingleKernel`, for which the
  host never launches a beta-only kernel.

Not fixed as part of this plan (DECISIONS.md topic 8). A fix direction, for whoever picks this up:
dedupe on the names of the built objects (or on the F2 input key from the benchmark path) instead
of predicted names, and add the compute type to the beta-only kernel name on both the Python and the
host side.

## Not measured / open items

- Runtime reach of the dropped `gfx1250-strict` kernels and of the beta-only collisions is derived
  from the host code, not run (no gfx1250 hardware here; the gfx950 beta-only case was not run
  either).
- gfx1170/gfx1171 with a compiler that supports them (capabilities were substituted, see Method).
- Builds with an explicit `--logic-filter` other than one file, or arch subsets other than the ones
  listed. The per-file and per-arch results bracket them, but order-dependent drops like the
  gfx1250-strict one can appear in any combination that puts a `MultipleBuffer` solution with the
  same `GlobalSplitU` value before a `SingleBuffer` one of a problem type with the same conversion
  name.
