# TensileLite Codegen Pipeline

This note maps the path from an input TensileLite YAML file to saved GPU kernel
artifacts. It covers both flows that create kernels:

- The tuning flow driven by `Tensile/bin/Tensile <config.yaml> <output-dir>`.
- The device-library packaging flow driven by `Tensile/bin/TensileCreateLibrary
  <logic-dir> <output-dir> HIP`, which consumes library-logic YAMLs and emits
  installable code objects plus runtime selection data.

## Executive Flow

```text
Input config YAML
  -> Tensile.Tensile()
  -> executeStepsInConfig()
  -> BenchmarkProblems.main()
     -> BenchmarkProcess
     -> Solution objects
     -> KernelWriterAssembly + rocisa
     -> .s assembly
     -> .o object
     -> .co assembly code object
     -> helper HIP .hsaco code object
     -> temporary benchmark TensileLibrary.{yaml,dat}
     -> tensilelite-client benchmark CSV
  -> LibraryLogic.main()
     -> 3_LibraryLogic/*.yaml
  -> ClientWriter.main()
     -> TensileCreateLibrary subprocess
     -> 4_LibraryClient/library/<gfx>/*.co, *.hsaco, *.yaml/*.dat
```

For a packaged device library, the middle of the flow starts at
`3_LibraryLogic/*.yaml`:

```text
Library-logic YAMLs
  -> TensileCreateLibrary.run()
  -> LibraryIO.parseLibraryLogicFile()
  -> Solution objects + MasterSolutionLibrary
  -> KernelWriterAssembly + rocisa
  -> .s assembly
  -> .o object
  -> .co assembly code object
  -> helper HIP .hsaco code object
  -> final TensileLibrary_*.{dat,yaml}, lazy-library files, mapping files
```

## Top-Level Entrypoint

`Tensile/bin/Tensile` imports `Tensile.Tensile` and calls `Tensile.main()`.
The real CLI implementation is in `Tensile/Tensile.py:Tensile(userArgs)`.

The entrypoint:

1. Parses command-line options and the input config path.
2. Reads the input YAML or JSON through `LibraryIO.read()`.
3. Handles alternate config format if `--alternate-format` is used.
4. Restores and assigns `globalParameters`.
5. Validates the ROCm toolchain.
6. Builds the assembly and source toolchain wrappers.
7. Chooses target ISA from `--gpu-targets`, YAML `GlobalParameters.ISA`, or
   local GPU detection.
8. Builds `isaInfoMap` with architecture, assembly, and register capabilities.
9. Calls `executeStepsInConfig()`.

`executeStepsInConfig()` is the phase dispatcher:

- `BenchmarkProblems` section -> `BenchmarkProblems.main()`.
- `LibraryLogic` section -> `LibraryLogic.main()`, unless `--build-only`.
- `LibraryClient` section -> `ClientWriter.main()`, unless `--build-only`.

`--build-only` is important: it still generates and compiles kernels in
`BenchmarkProblems`, but then skips benchmarking, library logic generation, and
client-library generation.

## Input Config Structure

A normal tuning YAML has these top-level sections:

- `GlobalParameters`: process-wide switches such as architecture, library
  format, code object version, cache behavior, validation, and client options.
- `BenchmarkProblems`: list of problem type plus parameter groups. This is the
  only section required to generate candidate kernels.
- `LibraryLogic`: optional analysis parameters used to convert benchmark CSVs
  into selection logic.
- `LibraryClient`: optional client validation section after logic is generated.

Within one `BenchmarkProblems` entry:

- Entry `[0]` is the `ProblemType` config. It becomes a
  `SolutionStructs.Problem.ProblemType`.
- Later entries are problem-size group configs. Each group is parsed by
  `BenchmarkStructs.BenchmarkProcess`.
- `BenchmarkCommonParameters` become constant solution parameters.
- `ForkParameters` define cartesian-product solution parameters.
- `ForkParameters.Groups` can inject grouped parameter choices into the
  permutation stream.
- `UseCustomMainLoopSchedule` can appear in `ForkParameters` or
  `BenchmarkCommonParameters` to control custom main-loop scheduling. The
  canonical parameter name in this codebase is `UseCustomMainLoopSchedule`;
  `UseCustomMainloopScheduling` is not a registered key.
  Conceptually, "false" maps to `0` and "true" maps to `1`, but the YAML
  should use integers:

  ```yaml
  ForkParameters:
    - UseCustomMainLoopSchedule: [-1]  # auto: use CMS only when supported
    # - UseCustomMainLoopSchedule: [0] # disable
    # - UseCustomMainLoopSchedule: [1] # require CMS
  ```
- `BenchmarkFinalParameters[0].ProblemSizes` becomes a `ProblemSizes` object.
- `CustomKernels` bypasses normal fork generation and loads metadata for named
  custom assembly kernels.

## BenchmarkProblems Phase

Source: `Tensile/BenchmarkProblems.py`.

### 1. Parse Problem and Step Config

`BenchmarkProblems.main()` iterates each `BenchmarkProblems` YAML entry and
builds a `ProblemType` object for naming and output-file selection.

For each problem-size group, `_benchmarkProblemType()` creates a
`BenchmarkProcess`:

- `BenchmarkProcess.__init__()` creates the `ProblemType`.
- `getConfigParameters()` flattens common parameters and fork parameters,
  validates parameter names and values, expands `Groups`, and builds
  `ProblemSizes`, `BiasTypeArgs`, `ActivationArgs`, `FactorDimArgs`, and
  `ICacheFlush` state.
- `convertParametersToSteps()` currently creates one final `BenchmarkStep`.
- `constructForkPermutations` lazily enumerates the cartesian product of fork
  parameters and group overrides.

### 2. Generate Solution Objects

For forked solutions, `_generateForkedSolutions()` parallelizes
`_generate_single_solution()` over every fork permutation.

For each permutation:

1. Start with `{"ProblemType": problemType.state, "ISA": targetIsa}`.
2. Merge constant parameters from `BenchmarkCommonParameters`.
3. Merge one fork-parameter permutation.
4. Expand `MatrixInstruction` into derived MI parameters when it uses the
   9-field form.
5. Run matrix-instruction validation.
6. Construct `SolutionStructs.Solution.Solution`.

The `Solution` constructor fills defaults from `defaultSolution`, validates
parameter types, chooses ISA and code object version, and calls
`Solution.assignDerivedParameters()`. This is where many implicit kernel
parameters are derived or rejected.

This is also where custom main-loop scheduling is resolved. The YAML-facing
parameter is `UseCustomMainLoopSchedule`, with integer values:

- `-1`: auto. Use CMS if a registered custom schedule supports this solution;
  otherwise fall back to the normal scheduler.
- `0`: disable CMS even if a registered custom schedule would match.
- `1`: require CMS. If no registered custom schedule supports the solution, the
  solution is rejected.

Use `0` and `1` in YAML, not `false` and `true`. The type checker intentionally
distinguishes Python `bool` from `int`, because bools serialize differently in
logic/msgpack paths.

For custom kernels, `_generateCustomKernelSolutions()` loads custom-kernel
metadata, checks that the custom kernel's problem type matches the config, and
then constructs `Solution` objects.

At this point a valid `Solution` still represents tuning metadata. It is not a
saved kernel yet.

### 3. Select Unique Kernel Work Items

`writeBenchmarkFiles()` turns solutions into kernel work items:

- `solution.getKernels()` currently returns the solution itself with
  `Kernel: True` set.
- Duplicate kernels are filtered using `getKeyNoInternalArgs()`.
- Helper kernel objects are collected with `initHelperKernelObjects()` for
  support code such as activation helpers or other non-GEMM helper kernels.
- A `KernelWriterAssembly` is created for assembly generation.

Then `writeSolutionsAndKernels()` does the actual kernel generation and code
object build.

## Kernel Generation and Save Path

Source: `Tensile/TensileCreateLibrary/Run.py`, shared by the tuning flow and
the packaging flow.

### 1. Prepare Output Directories

`writeSolutionsAndKernels()` prepares:

```text
<sourcePath>/
  library/<gfx>/
  build_tmp/<SOURCE_STEM>/assembly/
  build_tmp/<SOURCE_STEM>/code_object_tmp/
```

`library/<gfx>/` is the final location for generated code objects in that
particular source tree. Target-feature variants such as xnack are kept in file
names where needed, while the directory name uses the base gfx target.

The tuning flow passes a cache-specific source directory:

```text
<out>/1_BenchmarkProblems/<problem>_<idx>/00_Final/caches/<hash>/source/
```

The packaging flow usually passes the requested output directory:

```text
<out>/
```

### 2. Generate Assembly Text

Only assembly GEMM kernels are supported in TensileLite here:

```python
assert numKernels == numAsmKernels, "Only assembly kernels are supported in TensileLite"
```

For each unique assembly kernel, `processKernelSource()`:

1. Calls `kernelWriterAssembly.setRocIsa(data, outOptions)`.
2. Computes the assembly file base name with `getKernelFileBase()`.
3. Calls `KernelWriterAssembly.getSourceFileString(kernel)`.
4. Calls `KernelWriter.getHeaderFileString(kernel)`.
5. Returns a `KernelCodeGenResult` with source text, kernel name, target ISA,
   wavefront size, occupancy, and prefetch metadata.

`KernelWriterAssembly.getSourceFileString()` chooses one of two paths:

- Custom kernel: `_getCustomKernelSource()` reads
  `Tensile/CustomKernels/<kernelName>.s`.
- Generated kernel: `_getKernelSource()` from `KernelWriter`.

Generated-kernel `_getKernelSource()`:

1. Calls `_initKernel()`.
2. Initializes rocisa with target ISA and wavefront size.
3. Reads asm, arch, and register capabilities from rocisa.
4. Builds per-kernel writer state, VGPR/SGPR pools, labels, code modules, and
   component selections.
5. Calls `kernelBody()` or `kernelBodySubtile()`.
6. Produces a rocisa module containing the generated instruction stream.
7. Optionally runs the StinkyTofu optimization pipeline.
8. Emits final assembly text.

The generated string is the assembly source for one kernel.

### 2a. Custom Main-Loop Scheduling

Custom main-loop scheduling, usually abbreviated CMS in names/comments, is a
special path inside assembly kernel generation. It does not replace the
YAML-to-solution pipeline or the `.s -> .o -> .co` toolchain pipeline. It
replaces the default placement of main-loop instruction modules once a solution
has already been selected for generated assembly. It does not rewrite
hand-written `CustomKernels/<name>.s` source files.

The support check happens during `Solution.assignDerivedParameters()`:

1. The incoming `UseCustomMainLoopSchedule` value is read from the solution
   state.
2. If the value is `-1` or `1`, the solution calls
   `Components.CustomSchedule.hasCustomSchedule(state)`.
3. `hasCustomSchedule()` immediately rejects CMS matching unless:
   `UseCustomMainLoopSchedule` is truthy, matrix instructions are enabled, the
   ISA is `gfx950` (`IsaVersion(9,5,0)`), and the problem is not mixed-input
   width.
4. It then walks the `_SCHEDULE_REGISTRY` populated by `@RegisterSchedule`
   functions in `Tensile/Components/CustomSchedule.py`.
5. A registered schedule matches on a narrow set of parameters such as dtype,
   layout, macro tile, `DepthU`, PGR/PLR, direct-to-LDS mode, wave-separated
   global reads, global/local read widths, `MatrixInstruction`, `MIWaveGroup`,
   `LDSTrInst`, and `TransposeLDS`.
6. If `UseCustomMainLoopSchedule == 1` and no schedule matches, the solution is
   rejected. If the value was `-1`, the solution silently falls back to
   `UseCustomMainLoopSchedule = 0`.
7. CMS is rejected with `TailloopInNll=True` and with `UseSubtileImpl=True`.

The final resolved value is persisted in solution metadata, participates in
kernel naming through the `CMS` tag, and is copied into runtime solution
metadata as `customMainLoopScheduling`.

When CMS is enabled, `KernelWriter.kernelBody()` still generates the same broad
categories of code: global reads, local writes, local reads, pack/convert code,
swap/reset pointer code, MFMA code, waits/barriers, and loop-close code. The
difference is that it does not immediately feed those modules through the
default `_makeSubIterSchedule()` path. Instead it accumulates per-iteration
modules:

```text
LRCodeAAllIters / LRCodeBAllIters
PackCodeAAllIters / PackCodeBAllIters
LRSwapAAllIters / LRSwapBAllIters
LWSwapAAllIters / LWSwapBAllIters
MfmaCodeAllIters
globalReadA / globalReadB
globalReadIncA / globalReadIncB
localWriteA / localWriteB
loop-close code
```

After the loop body modules are collected, the writer calls
`customMainLoopSchedule(...)`. That function:

1. Strips comments from the instruction modules.
2. Calls `hasCustomSchedule(kernel)` again to retrieve the selected
   `ScheduleInfo`.
3. Optionally reorders the MFMA stream with `ScheduleInfo.mfmaReorder`.
4. Builds an `idMap` that binds schedule keys such as `LRA0`, `LRB1`,
   `PackA0`, `GRA`, `GRB`, `GRIncA`, `LWA`, `LWB`, `SYNC`, and `SNOP` to the
   actual generated instruction lists.
5. Validates the schedule with `CMSValidator.isValid()`. Validation checks that
   the schedule references legal keys, has the expected number of instructions,
   orders positions monotonically, and satisfies dependency/synchronization
   rules.
6. Emits a `MAINLOOP` macro. The macro iterates over MFMA issue positions and
   injects the registered instruction streams at the MFMA indices listed in
   `ScheduleInfo.optSchedule`.
7. Adds guards for code-path-specific streams and for loop variants such as
   `useGR`, `usePLR`, `useGRInc`, and `useLoop`.
8. Emits wait-count variants for main-loop, no-global-load, and no-local-load
   paths using `nglshift`, `nllshift`, and `nllZeroDscnt`.

`KernelWriterAssembly.simdSpecDispatch()` then emits the actual unrolled loop.
For one code path it repeatedly invokes `MAINLOOP(0)`. For multiple code paths,
it reads the SIMD id, branches to the matching specialized loop body, invokes
`MAINLOOP(<id>)`, and jumps to the common loop end.

`noLoadLoopBody()` also changes under CMS. Instead of generating a separate
default no-load loop body, it invokes the same `MAINLOOP` macro with different
boolean arguments:

```text
MAINLOOP(ID, useGR, usePLR, useGRInc, useLoop)
```

That lets the registered CMS schedule describe the full main-loop family while
masking global reads, local-read prefetch, global-read increments, or loop
counter updates for no-load/no-global-load variants.

When CMS is disabled or falls back to `0`, the writer uses the normal path:

- `makeSchedule()` and the selected `SIA` component schedule global reads,
  local writes, and increments into loop iterations.
- `_makeSubIterSchedule()` interleaves local reads, pointer code, waits,
  pack/convert code, and MFMA/MAC code around each sub-iteration.
- The normal `closeLoop()` and no-load loop generation paths are used.

`UsePLRPack` is special with CMS. In the non-CMS path, YAML can request it and
derived-parameter logic filters it down to supported cases. In the CMS path,
`UsePLRPack` is treated as schedule-owned state: `Solution.assignDerivedParameters()`
temporarily clears it before schedule matching, and individual custom schedule
functions may set it internally. Setting `UsePLRPack` from YAML is not a way to
change a CMS schedule.

### 3. Write `.s` and Assemble `.o`

`writeAssembly()` writes the source string to:

```text
<sourcePath>/build_tmp/<SOURCE_STEM>/assembly/<kernelBase>.s
```

Then `Assembler.__call__()` invokes the ROCm compiler in assembler mode:

```text
amdclang++ -x assembler --target=amdgcn-amd-amdhsa \
  -mcode-object-version=<version> -c -mcpu=<gfx> \
  -mwavefrontsize64 | -mno-wavefrontsize64 \
  <kernelBase>.s -o <kernelBase>.o
```

The object is saved beside the temporary assembly:

```text
<sourcePath>/build_tmp/<SOURCE_STEM>/assembly/<kernelBase>.o
```

Unless `KeepBuildTmp` is set, the `.s` file is removed after assembling.

### 4. Link and Bundle Assembly Code Objects

`buildAssemblyCodeObjectFiles()` groups kernels by ISA, then links the `.o`
files into raw code objects and optionally bundles/compresses them.

For normal non-lazy benchmark builds, objects for one arch are linked to:

```text
<assemblyTmp>/TensileLibrary_<gfx>.co.raw
```

Then the bundler writes:

```text
<sourcePath>/library/<gfx>/TensileLibrary_<gfx>.co
```

For lazy library builds, solutions can carry `solution._state["codeObjectFile"]`.
Those kernels are grouped by that value instead, producing:

```text
<sourcePath>/library/<gfx>/<codeObjectFile>.co
```

This is the main "saved kernel" artifact for generated assembly GEMM kernels.
It contains one or more linked assembly kernels for a target architecture.

### 5. Generate and Save Helper-Kernel Code Objects

`writeHelpers()` writes helper HIP source and headers:

```text
<sourcePath>/Kernels.cpp
<sourcePath>/Kernels.h
```

`buildSourceCodeObjectFiles()` then:

1. Compiles `Kernels.cpp` with HIP device-only compilation into a temporary
   object.
2. Lists offload targets in that object with the bundler.
3. Unbundles each target into `.hsaco.raw`.
4. Moves each final helper code object into:

```text
<sourcePath>/library/<gfx>/Kernels.so-000-<gfx>[...].hsaco
```

These helper `.hsaco` files are separate from the GEMM assembly `.co` file.

### 6. Save a Benchmark-Time Library File

Back in `writeBenchmarkFiles()`, the tuning flow creates a
`MasterSolutionLibrary.BenchmarkingLibrary`, applies names, and writes:

```text
<sourcePath>/library/<gfx>/TensileLibrary.yaml
```

or:

```text
<sourcePath>/library/<gfx>/TensileLibrary.dat
```

depending on `globalParameters["LibraryFormat"]`.

This file maps benchmark-time problem selection to the solutions built into the
code objects.

### 7. Write Client Parameters and Run Benchmark

`writeClientConfig()` writes:

```text
<sourcePath>/ClientParameters.ini
```

The INI contains:

- `library-file=<sourcePath>/library/<gfx>/TensileLibrary.{yaml,dat}`
- one or more `code-object=<sourcePath>/library/<gfx>/*.co`
- problem sizes
- data types and problem type metadata
- benchmark and validation options
- output CSV path

Helper `.hsaco` files are generated into the same `library/<gfx>/` directory,
but they are not part of the `codeObjectFiles` list returned by
`writeSolutionsAndKernels()`.

`runClient()` executes the prebuilt `tensilelite-client` with that config. The
client benchmarks every candidate solution over the configured problem sizes and
writes:

```text
<out>/1_BenchmarkProblems/<problem>_<idx>/Data/00_Final.csv
```

`LibraryIO.writeSolutions()` also writes the solution metadata used by later
analysis:

```text
<out>/1_BenchmarkProblems/<problem>_<idx>/Data/00_Final.yaml
```

After a successful non-build-only run, `BenchmarkProblems.main()` copies those
files into:

```text
<out>/2_BenchmarkData/<problem>_<idx>.csv
<out>/2_BenchmarkData/<problem>_<idx>.yaml
```

## LibraryLogic Phase

Source: `Tensile/LibraryLogic.py`.

`LibraryLogic.main()` calls `generateLogic()` with:

```text
benchmarkDataPath = <out>/2_BenchmarkData
libraryLogicPath  = <out>/3_LibraryLogic
```

`generateLogic()`:

1. Reads every benchmark `.csv` in `2_BenchmarkData`.
2. Requires a matching `.yaml` solution metadata file.
3. Parses the solution metadata with `LibraryIO.parseSolutionsFile()`.
4. Groups benchmark records by `ProblemType`.
5. Calls `analyzeProblemType()` to select the best solution for exact/range
   problem sizes according to the requested library type and performance metric.
6. Calls `LibraryIO.createLibraryLogic()`.
7. Writes each logic file to `3_LibraryLogic`.

The resulting library-logic YAML is a list-style schema:

```text
0. MinimumRequiredVersion
1. ScheduleName
2. ArchitectureName or {Architecture, CUCount}
3. DeviceNames
4. ProblemType
5. Solutions
6. IndexOrder
7. ExactLogic
8. RangeLogic
9. Optional tile-selection data
10. PerfMetric
11. LibraryType
```

This file is not a code object. It is the persisted decision table that says
which solution should handle which problem shape.

## LibraryClient Phase

Source: `Tensile/ClientWriter.py`.

`ClientWriter.main()` consumes `3_LibraryLogic` and creates `4_LibraryClient`.
The key step is a subprocess call built by `getBuildClientLibraryScript()`:

```text
Tensile/bin/TensileCreateLibrary \
  --architecture=<gfx> \
  --code-object-version=<version> \
  --cxx-compiler=<assembler path> \
  --library-format=<yaml|msgpack> \
  <out>/3_LibraryLogic \
  <out>/4_LibraryClient \
  HIP
```

After `TensileCreateLibrary` writes code objects and library files,
`ClientWriter.main()`:

1. Finds generated `.co` and `.yaml` files in
   `<out>/4_LibraryClient/library/<gfx>/`.
2. Parses the logic files again to create problem metadata.
3. Writes client parameter INIs.
4. Runs the client in validation mode.

## TensileCreateLibrary Packaging Flow

Source: `Tensile/TensileCreateLibrary/Run.py`.

This is the path used by CMake/device-library builds and also by
`ClientWriter.main()`.

### 1. Read Arguments and Logic Files

`Tensile/bin/TensileCreateLibrary` imports `TensileCreateLibrary.run()`.
`parseArguments()` requires:

```text
LogicPath OutputPath RuntimeLanguage
```

Common options include:

- `--architecture=<gfx>` or `all`
- `--logic-filter=<glob>`
- `--library-format=yaml|msgpack`
- `--code-object-version=4|5|default`
- `--no-lazy-library-loading`
- `--keep-build-tmp`
- `--no-compress`

`run()` filters logic files by extension, architecture, experimental directory
policy, and optional target predicates.

### 2. Parse Logic into Libraries and Solutions

`generateLogicDataAndSolutions()` parses each logic file with
`LibraryIO.parseLibraryLogicFile()`.

`parseLibraryLogicFile()` reads YAML and calls `parseLibraryLogicData()`, which:

1. Normalizes older/list-style logic into dictionary form.
2. Recreates `ProblemType`.
3. Recreates every saved solution as a `Solution` object.
4. Forces derived-parameter reassignment for old logic files.
5. Builds a `MasterSolutionLibrary` with exact/range/lazy selection nodes.

`generateLogicDataAndSolutions()` merges per-architecture master libraries,
reindexes solutions deterministically, applies fallback libraries when present,
and collects each original solution for code generation.

For lazy-library entries, it also sets:

```text
solution._state["codeObjectFile"] = <lazy-library-name>
```

That value controls which final `.co` file receives the solution's kernel.

### 3. Generate and Build Kernels

The packaging path calls:

```text
generateKernelObjectsFromSolutions()
generateKernelHelperObjects()
writeSolutionsAndKernelsTCL()
```

`writeSolutionsAndKernelsTCL()` uses the same core primitives as the tuning
flow:

- `processKernelSource()` for rocisa assembly text generation.
- `writeAssembly()` for `.s` files.
- `Assembler.__call__()` for `.s -> .o`.
- `buildAssemblyCodeObjectFiles()` for `.o -> .co`.
- `writeHelpers()` and `buildSourceCodeObjectFiles()` for helper `.hsaco`.

The output root is the requested `OutputPath`, so final artifacts go under:

```text
<OutputPath>/library/<gfx>/
```

### 4. Write Runtime Selection Files

After kernels are built, `passPostKernelInfoToLibrary()` copies post-codegen
metadata such as CU occupancy and unrolled-loop math clocks back into the
solution-library objects.

Then `run()` writes per-architecture runtime files:

- Non-lazy:

  ```text
  <OutputPath>/library/<gfx>/TensileLibrary_<gfx>.{dat,yaml}
  <OutputPath>/library/<gfx>/TensileLibrary_<gfx>.co
  ```

- Lazy:

  ```text
  <OutputPath>/library/<gfx>/TensileLibrary_lazy_<gfx>.{dat,yaml}
  <OutputPath>/library/<gfx>/<lazy-library-name>.{dat,yaml}
  <OutputPath>/library/<gfx>/<lazy-library-name>.co
  <OutputPath>/library/<gfx>/TensileLiteLibrary_lazy_<gfx>_Mapping.dat
  ```

- Helper kernels:

  ```text
  <OutputPath>/library/<gfx>/Kernels.so-000-<gfx>[...].hsaco
  ```

The runtime loads the library metadata to select a solution, then loads the
matching code object for the selected kernel.

## Important Output Directories

For `Tensile/bin/Tensile config.yaml tensile-out`:

```text
tensile-out/
  1_BenchmarkProblems/
    <problem>_<idx>/
      00_Final/
        caches/<hash>/source/
          ClientParameters.ini
          Kernels.cpp
          Kernels.h
          library/<gfx>/
            TensileLibrary_<gfx>.co
            TensileLibrary.{yaml,dat}
            Kernels.so-000-<gfx>[...].hsaco
          build_tmp/...                 # only kept with KeepBuildTmp
        build/run.sh
      Data/
        00_Final.csv
        00_Final.yaml
  2_BenchmarkData/
    <problem>_<idx>.csv
    <problem>_<idx>.yaml
  3_LibraryLogic/
    <schedule>_<problem>.yaml
  4_LibraryClient/
    library/<gfx>/
      *.co
      *.hsaco
      *.yaml or *.dat
```

For `TensileCreateLibrary logic-dir device-lib HIP`:

```text
device-lib/
  library/<gfx>/
    TensileLibrary_<gfx>.co                 # non-lazy default co
    <lazy-library-name>.co                  # lazy co groups
    TensileLibrary_lazy_<gfx>.{dat,yaml}
    TensileLibrary_<gfx>.{dat,yaml}
    <lazy-library-name>.{dat,yaml}
    TensileLiteLibrary_lazy_<gfx>_Mapping.dat
    Kernels.so-000-<gfx>[...].hsaco
```

## Source Map

Primary control flow:

- `Tensile/bin/Tensile`: CLI shim.
- `Tensile/Tensile.py:Tensile()`: reads config, resolves toolchain and ISA.
- `Tensile/Tensile.py:executeStepsInConfig()`: dispatches phases.
- `Tensile/BenchmarkProblems.py:main()`: iterates benchmark problem groups.
- `Tensile/BenchmarkProblems.py:_benchmarkProblemType()`: generates, builds,
  and benchmarks one problem group.
- `Tensile/LibraryLogic.py:main()` and `generateLogic()`: creates
  `3_LibraryLogic`.
- `Tensile/ClientWriter.py:main()`: creates and validates `4_LibraryClient`.

Config and solution parsing:

- `Tensile/BenchmarkStructs.py:BenchmarkProcess`: turns YAML problem-group
  config into benchmark steps.
- `Tensile/BenchmarkStructs.py:constructForkPermutations`: lazy cartesian
  product of fork parameters.
- `Tensile/SolutionStructs/Problem.py:ProblemType`: normalized problem type.
- `Tensile/SolutionStructs/Problem.py:ProblemSizes`: exact/range problem sizes.
- `Tensile/SolutionStructs/Solution.py:Solution`: validated/derived kernel
  parameter state.
- `Tensile/SolutionStructs/Solution.py:Solution.assignDerivedParameters()`:
  resolves `UseCustomMainLoopSchedule` from `-1/0/1` into the final CMS on/off
  state or rejects unsupported forced-CMS solutions.
- `Tensile/LibraryIO.py`: YAML/msgpack read-write, solution metadata, and
  library-logic parsing.

Kernel codegen and code object build:

- `Tensile/TensileCreateLibrary/Run.py:processKernelSource()`: one-kernel
  source generation wrapper.
- `Tensile/TensileCreateLibrary/Run.py:writeAssembly()`: writes `.s`.
- `Tensile/TensileCreateLibrary/Run.py:writeSolutionsAndKernels()`: benchmark
  flow codegen/build.
- `Tensile/TensileCreateLibrary/Run.py:writeSolutionsAndKernelsTCL()`:
  packaging flow codegen/build.
- `Tensile/KernelWriterAssembly.py:KernelWriterAssembly`: assembly kernel
  writer.
- `Tensile/KernelWriter.py:_getKernelSource()`: initializes writer state and
  emits generated kernel assembly.
- `Tensile/KernelWriter.py:makeSchedule()` and `_makeSubIterSchedule()`:
  default main-loop scheduling path used when CMS is off.
- `Tensile/Components/CustomSchedule.py`: registered custom main-loop
  schedules, CMS matching, `ScheduleInfo`, and `customMainLoopSchedule()`.
- `Tensile/Components/CMSValidator.py`: validation for custom schedule
  instruction counts, ordering, dependencies, and synchronization.
- `Tensile/Components/`: modular codegen components used by the writer.
- `rocisa/`: instruction/module assembly generation backend.
- `Tensile/Toolchain/Component.py:Assembler`: `.s -> .o`.
- `Tensile/Toolchain/Assembly.py:buildAssemblyCodeObjectFiles()`: `.o -> .co`.
- `Tensile/Toolchain/Source.py:buildSourceCodeObjectFiles()`: helper
  `Kernels.cpp -> .hsaco`.

Packaging:

- `Tensile/bin/TensileCreateLibrary`: CLI shim.
- `Tensile/TensileCreateLibrary/ParseArguments.py`: device-library CLI
  arguments.
- `Tensile/TensileCreateLibrary/Run.py:run()`: reads logic, generates code
  objects, writes runtime libraries.
- `Tensile/TensileCreateLibrary/Run.py:generateLogicDataAndSolutions()`:
  recreates solutions and master libraries from logic YAMLs.

## Practical Debugging Notes

- To inspect generated assembly, run with `KeepBuildTmp: True` or
  `--keep-build-tmp`; otherwise temporary `.s`, `.o`, and build directories are
  removed after code object creation.
- In a benchmark run, start at
  `1_BenchmarkProblems/<problem>_<idx>/00_Final/caches/<hash>/source/` to see
  the exact generated source tree used by the client.
- In a device-library build, start at `library/<gfx>/` under the requested
  output path. That directory contains the final runtime artifacts.
- If a candidate disappears before benchmarking, check solution rejection in
  `Solution.assignDerivedParameters()` and kernel generation errors surfaced by
  `removeInvalidSolutionsAndKernels()`.
- If a kernel is generated but not loadable, check that the client INI points to
  the matching `TensileLibrary.{yaml,dat}` and every required `.co`/`.hsaco`.
- Lazy-library builds route kernels into code objects by `codeObjectFile`; if a
  selected solution cannot find its code object, inspect lazy-library names,
  mapping files, and `solution._state["codeObjectFile"]`.
