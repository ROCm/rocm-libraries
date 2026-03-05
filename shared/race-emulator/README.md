# Race Emulator

A CPU-side emulator for AMD GPU assembly that detects race conditions (missing `s_waitcnt`, `s_barrier`, etc.) without requiring a GPU. It is currently used by hipblaslt and tensilelite to validate GEMM kernel assembly.

## Motivation
A race occurs when a GPU thread reads an ambiguous value from a register, local memory (LDS), or global memory. When a race occurs, the behaviour of a GPU program, even for a fixed initial state, can be nondeterministic.

Races (also called race 'conditions') can be difficult to avoid, particularly in complex direct-to-assembly code generation systems that attempt to eek out optimal performance. Races can also lie dormant through many iterations of a program, only to be triggered when the GPU enters a specific state, which makes them difficult to debug.

This developer tool aims to serve as an early detection system for races, and to provide useful diagnostics that a developer can use to find and eliminate them. It is currently aimed at developers working directly with AMD GPU assembly, although it can be integrated into any stack that emits AMD GPU assembly. Diagnostics are at the assembly level, see the examples below for a demonstration.

This project is currently in early stage development, and might change direction dramatically.

Feedback and suggestions from more folks with expertise in LLVM assembly parsing, AMD GPU emulators, and AMD GPU tools such as rocgdb would be fantastically useful!

## Goals

Given AMD GPU assembly code, detect all intra-workgroup races. Some examples are:

1) A single thread issues a load from global memory to a register, but does not wait for the load to complete before using the register.
2) Two threads in different waves (but the same workgroup) write to the same address in LDS, without using a workgroup barrier to specify their relative order. The LDS address is subsequently read and used by a third thread.

## Maybe goals

- Make the analysis value sensitive. For example in case (2) above if the two writing threads write the same value, it could be considered race free, because the value that the third thread reads is not ambiguous (it doesn't matter which thread wrote as the value is the same).
- Detect inter-workgroup race conditions in global memory. For example, atomic writes for some variants of split-k GEMMs.

## Non-goals
- With its current design of using an emulator, a non-goal is to emulate all AMD GPU programs. We will add instructions as needed for different use cases.

## Current status

The majority of the effort so far has gone into emulating instructions. For gfx942 (MI300X) there are currently 157+ instructions partially or fully supported. With this set of instructions, the emulator has numerically validated runs of

- hipBlasLt gemm kernels (f32 -> f32 and bf16 -> f32)
- a HipKittens gemm kernel (bf16 -> f32)
- a few simple HIP programs.

Being able to numerically validate all vector instructions is not necessary for detecting race conditions, but it provides useful proof that all instructions are correctly emulated. The project also has unit tests for all the instructions it emulates, individually.

Below we present two simple examples of races that are currently detectable. There are more in the test directory.

### Case 1: Single thread

The following assembly is modified from a hip program where each thread adds two integers and writes the result to global memory:

```c++
int tid = threadIdx.x;
auto v1 = data[threadIdx.x];
auto v2 = data[threadIdx.x+17];
auto v3 =  v1 + v2;
```

The modification that introduces a race is to increase the `vmcnt` wait value from 0 to 1. As a result, the thread only waits for `v1` to be loaded before performing the add, and so the value for second operand may not yet be in register.

```
s_load_dwordx2 s[0:1], s[0:1], 0x0
v_lshlrev_b32_e32 v0, 2, v0
s_waitcnt lgkmcnt(0)
global_load_dword v1, v0, s[0:1]
global_load_dword v2, v0, s[0:1] offset:68
s_waitcnt vmcnt(1) ; <--- SHOULD WAIT FOR BOTH!
v_add_u32_e32 v1, v2, v1
global_store_dword v0, v1, s[0:1]
s_endpgm
```

The race-emulator detects the race, with diagnostic message

```asm
VGPR race detected on line 8. Conflicting events:

5     |     global_load_dword v1, v0, s[0:1]
6 --> |     global_load_dword v2, v0, s[0:1] offset:68
7     |     s_waitcnt vmcnt(1) ; <--- SHOULD WAIT FOR BOTH!
8 --> |     v_add_u32_e32 v1, v2, v1
9     |     global_store_dword v0, v1, s[0:1]
```

It highlights the 2 lines that are involved in the race with `-->`.

### Case 2: Threads in different waves

Race conditions of this sort can arise when `s_barrier` is not used to synchronize threads in different waves (subgroups). In the following example the LDS is used as a shared memory for threads to exchange data.

```asm
  ; Each thread loads a distinct 4 bytes from global to a vector register.
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  v_lshlrev_b32_e32 v0, 2, v0
  v_sub_u32_e32 v2, 0, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v1, v0, s[0:1]
  s_waitcnt vmcnt(0)

  ; Each thread writes its 4 bytes to LDS.
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)

  ;  s_barrier <--- MISSING BARRIER
  ; Each thread reads from LDS, from an address written by another wave.
  ds_read_b32 v1, v2 offset:1020
  s_waitcnt lgkmcnt(0)
  global_store_dword v0, v1, s[0:1]
  s_endpgm
```

The error message provided by `race-emulator` is

```asm
LDS race in byte 512 detected. Race between a pair in:

Wave 2 Lane 0:
11     |   ; Each thread writes its 4 bytes to LDS.
12 --> |   ds_write_b32 v0, v1
13     |   s_waitcnt lgkmcnt(0)

Wave 1 Lane 63:
16     |   ; Each thread reads from LDS, from an address written by another wave.
17 --> |   ds_read_b32 v1, v2 offset:1020
18     |   s_waitcnt lgkmcnt(0)
```
Above, race-emulator has detected that a thread in wave 2 is writing to the same address that a thread in wave 1 is reading from, and that the order is not specified. This means that the value read in race 1 is ambiguous.

## Implementation of the core race detection logic

Every byte address of LDS, and every vector register of every wave, contains (in the emulator) a data structure that tracks the memory operations it is currently involved in. Every time a memory instruction (ds_read, global_store, buffer_load, etc.) or a barrier-related operation (s_endpgm, s_barrier, s_waitcnt, etc.) is executed in the emulator, the data structures of the relevant LDS bytes and registers are updated. If an LDS address or vector register is read by _any_ instruction, and the data structure determines that the value contained is ambiguous, a C++ exception is thrown immediately (design note: we initially started using LLVM/MLIR error propagation, but it added too much bloat for our liking so we switched to exceptions). In _theory_ an exception should only be thrown when an ambiguous value is written to global memory, but for now, for simplicity, that's not the design.

Note: One thing that needs completition is adding support for detecting race conditions in scalar registers.

## Usage

Currently, this project does not have a tool that works as simply as `./race-emulator my-kernel.s`. This is because some kernel-specific work must always be done to initialize the arguments for the kernel. We could automate this to some extent by dumping the kernel arguments from a GPU run (or recording them before hipModuleLaunchKernel is called), and then reusing those. However for numerical validation in the emulator, we'd still need to substitute pointer arguments with CPU pointers. For a full example, I suggest seeing the end-to-end tests in `tests/e2e_hip_general.cpp`.

## Alternative approaches

This project is still early stage, so it might make sense to pivot to a new design. Some alternative design approaches for developing a race detection tool I have considered are outlined below. They are presented in the order of the number of dependencies required.

### Level 0
---
Does not use LLVM for parsing assembly (or anything else). Does not use an external emulator. Does not run on, or require, a GPU. i.e. completely standalone.

**Pros**: Full control over the implementation.

**Cons**: Implement everything from scratch.

This is the approach currently taken in this project.


### Level 1
---
Use LLVM to parse the assembly, and then run a custom emulator on the LLVM Machine IR.

**Pros**: Robust parsing.

**Cons**: I personally didn't know enough about LLVM's internals to get started in this direction.

### Level 2
---

Like level 1, but additionally use an existing emulator. For example, there is the [FFM project](https://statics.teams.cdn.office.net/evergreen-assets/safelinks/2/atp-safelinks.html) for emulating MI450, which might be open sourced, and might be backported to previous architectures.

**Pros**: No need to reimplement an emulator, contribute to making FFM more robust.

**Cons**: Unknowns. Is FFM flexible enough to allow us to add in our hooks to track the necessary information to detect race conditions? Unknown timelines for when this will be usable.


### Level 3
---
Execute on real hardware. The code would need to be instrumented to record global/LDS reads and writes, as well as s_barriers and s_waitcnts, and stream these back to CPU, for every wave. A host program would then analyze the recorded accesses, and detect race conditions as a post processing step. I suspect that this is the approach taken by NVidia's [racecheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool) tool. Perhaps [this](https://umr.readthedocs.io/en/main/index.html) open source AMD tool could serve as a starting point.

**Pros**: Would not need to emulate the full GPU ISA. Potentially faster runtime than emulation.

**Cons**: Unknowns. Is this possible? Will it work in all cases, even when there are no registers to spare for instrumentation? Would the focus be on HIP code, LLVM IR, or AMD GPU assembly?


## Integration status

race-emulator is integrated into rocm-libraries in two places. Both
require building hipblaslt with `-DHIPBLASLT_ENABLE_RACE_EMULATOR=ON`
(see [Enabling race-emulator in hipblaslt](#enabling-race-emulator-in-hipblaslt)
below).

1. **tensilelite-client**: Race checking can be enabled in Tensile YAML test
   files by setting `CheckForRaces: 1` in `GlobalParameters`. The emulator
   runs on the generated kernel assembly after the first warmup iteration.
   See `Tensile/Tests/common/gemm/race_check_gfx942.yaml` for an example.

2. **hipblaslt (direct assembly)**: When custom assembly kernels are loaded
   via the `HIPBLASLT_CUSTOM_ASM_DIR` environment variable, the race emulator
   automatically validates each kernel on first invocation. No code changes
   are needed — race checking is built into the direct assembly dispatch path.
   Note: this approach will change when the more robust custom assembly
   approach is integrated into hipblaslt.

If race checking is requested at runtime (e.g. `CheckForRaces: 1` in a YAML
file) but the build was compiled without `HIPBLASLT_ENABLE_RACE_EMULATOR`,
the client will throw a runtime error rather than silently skipping the check.

## Building

race-emulator has no dependencies beyond the C++ standard library (C++20).
GoogleTest is required for tests, and is fetched automatically if not found.

The examples below show typical configurations. Flags like `-DGPU_TARGETS`,
compiler paths, and optional dependency flags will vary depending on your
hardware and environment.

### Standalone (for development and testing)

Build and test the library in isolation, without any other rocm-libraries
components. This is the fastest way to iterate on race-emulator itself:

```bash
cmake -S shared/race-emulator -B build -G Ninja \
  -DRACE_EMULATOR_BUILD_TESTING=ON
ninja -C build
ctest --test-dir build
```

Run a specific subset of tests:

```bash
ctest --test-dir build -R ParserTest
```

### Enabling race-emulator in hipblaslt

Race-emulator support in hipblaslt is controlled by the CMake option
`HIPBLASLT_ENABLE_RACE_EMULATOR` (default `OFF`). When enabled,
race-emulator is pulled in via `add_subdirectory` and the compile
definition `HIPBLASLT_HAS_RACE_EMULATOR` is set for both
tensilelite-client and the hipblaslt library. Add this flag to any
hipblaslt cmake command to enable race checking:

```
-DHIPBLASLT_ENABLE_RACE_EMULATOR=ON
```

When the flag is `OFF`, hipblaslt builds and runs normally without any
race-emulator dependency. If race checking is then requested at runtime
(e.g. via `CheckForRaces: 1` in a YAML file, or via the
`HIPBLASLT_CUSTOM_ASM_DIR` environment variable), a runtime error is
thrown with a message indicating that the build does not support race
checking.

### As part of hipblaslt

**Tensilelite client** (for running `Tensile.sh`):

```bash
cmake --preset tensilelite \
  -S projects/hipblaslt -B build/tensilelite -G Ninja \
  -DGPU_TARGETS=gfx942 \
  -DHIPBLASLT_ENABLE_LLVM=1 \
  -DHIPBLASLT_ENABLE_RACE_EMULATOR=ON \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DBoost_ROOT=/path/to/boost
ninja -C build/tensilelite
```

**hipblaslt-bench**:

```bash
cmake -S projects/hipblaslt -B build/hipblaslt -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx942 \
  -DHIPBLASLT_ENABLE_RACE_EMULATOR=ON \
  -DHIPBLASLT_ENABLE_BLIS=0
ninja -C build/hipblaslt hipblaslt-bench
```

Note: `-DHIPBLASLT_ENABLE_BLIS=0` is only needed if BLIS is not installed.

To also build race-emulator's own tests within a hipblaslt build above, add
`-DRACE_EMULATOR_BUILD_TESTING=ON` to the cmake command.

### As part of the full monorepo superbuild

race-emulator is registered as a supported component. It is built
automatically when `ROCM_LIBS_ENABLE_COMPONENTS` includes `race-emulator`
(or `all`).

## Running with race checking

The instructions below require hipblaslt to have been built with
`-DHIPBLASLT_ENABLE_RACE_EMULATOR=ON`.

### Tensile.sh

Add `CheckForRaces: 1` to the `GlobalParameters` section of a Tensile YAML
file. For example:

```yaml
GlobalParameters:
  NumElementsToValidate: -1
  CheckForRaces: 1
```

Then run as usual:

```bash
./Tensile.sh my_test.yaml outputdir \
  --prebuilt-client=tensilelite/client/tensilelite-client
```

The race emulator will run on each generated kernel. If a race is detected,
the client will report the diagnostic and exit with a non-zero status.

### hipblaslt-bench (direct assembly)

The `examples/` directory contains `simple_gemm.cpp`, a minimal HIP GEMM
kernel, and `simple_gemm.s`, its compiled gfx942 assembly. To regenerate the
assembly from the source (e.g. for a different target):

```bash
hipcc --cuda-device-only -S --offload-arch=gfx942 -O3 simple_gemm.cpp -o simple_gemm.s
```

Set the environment variables to point at the directory containing the `.s`
file, then run hipblaslt-bench:

```bash
export HIPBLASLT_CUSTOM_ASM_DIR=/path/to/examples
export HIPBLASLT_ENABLE_DIRECT_ASSEMBLY=1
./hipblaslt-bench -m 128 -n 64 -k 256 -r f32_r --verify --alpha 1 --beta 1
```

On success, the output includes:

```
[DirectAssembly] Running race emulator on: /path/to/examples/simple_gemm.s
[DirectAssembly] Race emulator completed: SUCCESS
```

To demonstrate race detection, edit `simple_gemm.s` and change the
`s_waitcnt vmcnt(0)` on line 73 to `s_waitcnt vmcnt(1)`. This causes the
thread to proceed without waiting for the second global load to complete.
Re-running hipblaslt-bench will produce a diagnostic like:

```
VGPR race detected on line 74. Conflicting events:

66     |        global_load_dword v7, v[4:5], off
67 --> |        global_load_dword v3, v[8:9], off
68     |        s_add_i32 s2, s2, -1

73     |   s_waitcnt vmcnt(1)
74 --> |        v_fmac_f32_e32 v6, v3, v7
75     |        s_cbranch_scc0 .LBB0_3
```

To disable custom assembly dispatch:

```bash
export HIPBLASLT_ENABLE_DIRECT_ASSEMBLY=0
```
