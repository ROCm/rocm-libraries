# # AI Rules for rocRoller

This file provides guidance to AI agents when working with code in this repository.
(`CLAUDE.md` and `AGENTS.md` at the repo root both point here — this is the single source of truth.)

## Overview

RocRoller is a software library for generating optimized AMDGPU assembly kernels.
It transforms high-level kernel specifications (`Command`) through a dual-graph IR (`KernelGraph` with `ControlGraph` + `CoordinateGraph`) into optimized GPU assembly, then assembles to binary via AMD Comgr and executes via HIP.

See `docs/src/DesignOverview.md` for the full design writeup; the pipeline stages are summarized below.

## Architecture: the compilation pipeline

1. **`Command`** (`lib/include/rocRoller/Operations`) — user-facing description of tensor/scalar operations
   (load, store, `T_Execute` elementwise ops, `T_Mul`, etc.). Operations reference each other by
   `Operations::OperationTag`. Runtime-provided values (pointers, scalars) are `CommandArgument`s.
2. **`KernelGraph`** (`lib/include/rocRoller/KernelGraph`) — `Command::translate` lowers a `Command` into a
   `KernelGraph`, a pair of graphs plus a mapping between them:
   - `CoordinateGraph`: a hypergraph of `Dimension` nodes (sizes/strides, loop indices, tiles) connected by
     edges (e.g. `Flatten`, `Split`) describing how indexes transform into each other.
   - `ControlGraph`: `Operation` nodes (e.g. `LoadVGPR`, `Multiply`, `ForLoop`, `If`) describing control flow
     and dependencies; control nodes can nest their own sub-`ControlGraph` as a body.
   - The graph is iteratively rewritten by passes in `lib/include/rocRoller/KernelGraph/Transforms` (and
     `lib/source/KernelGraph`) — e.g. `LowerTile`, `AddLDS`, `AddPrefetch`, `AddStreamK`, `FuseLoops`,
     `AddLDSBarriers` — each transform takes a `KernelGraph` and returns a rewritten one. This is where most
     scheduling/performance work happens.
   - `Expression` (`lib/include/rocRoller/Expression.hpp`) represents scalar/index math as a `std::variant`
     tree, used throughout the coordinate/control graphs; `ExpressionTransformations` simplifies/rewrites them.
3. **Code generation** — `KernelGraph::generate` walks the (fully lowered) graph and emits `Instruction`s
   (`lib/include/rocRoller/InstructionValues`, `lib/include/rocRoller/CodeGen`) via a `Context`
   (`lib/include/rocRoller/Context.hpp`), which owns register allocators and a GPU architecture description.
   Instructions are scheduled (`lib/include/rocRoller/Scheduling`: `Scheduler` implementations, `IObserver`s
   for e.g. wait-count/hazard tracking) into `ScheduledInstructions`.
4. **Assembly + execution** — `Assembler` (`lib/include/rocRoller/Assemblers`) turns scheduled instructions
   into object code via AMD Comgr; `ExecutableKernel` loads/launches it via HIP. `CommandKernel`
   (`CommandSolution.hpp`) ties this whole pipeline together: `Command` → `KernelGraph` →
   `ScheduledInstructions` → `ExecutableKernel`, and exposes `launchKernel`.

Supporting subsystems: `GPUArchitecture` (per-arch instruction/capability info, generated at build time by
`GPUArchitectureGenerator` and queried at runtime via `GPUArchitectureLibrary`), a `Component`
plugin/factory system for swapping architecture-specific implementations, a YAML `Serialization`
framework (used for `Command`/`KernelGraph`/`Expression` debugging and caching), and a coroutine-based
`Generator<T>` for lazy instruction sequences.

## Build & Test Commands

```bash
# Configure + build (see CMakePresets.json for other presets: asan, precheckin, amd-mrisa, coverage, docs)
cmake --preset default:release -B build -S . [-DROCROLLER_ENABLE_FETCH=ON]
cmake --build build -j

# Run all tests via CTest (from the build dir); -LE GPU to skip GPU tests on CPU-only machines
ctest
ctest -LE GPU

# Catch2 (preferred for new tests) — supports name/regex and tags
./test/rocroller-tests-catch "<test-name-or-regex>"
./test/rocroller-tests-catch --list-tests
./test/rocroller-tests-catch --list-tags

# GTest (legacy tests)
./test/rocroller-tests --gtest_filter="<test-name-or-regex>"
./test/rocroller-tests --gtest_filter="-*GPU_*"   # exclude GPU tests

# Format code before submitting a PR
./scripts/fix-format

# Performance testing / benchmarking
./scripts/rrperf --help
./scripts/rrperf autoperf --help
```

- New Catch2 tests go in `test/catch/` and must be added to `test/catch/CMakeLists.txt`.
- Tests needing a `Context` but no GPU: inherit `GenericContextFixture`. Tests needing a real GPU: inherit
  `CurrentGPUContextFixture` and prefix the test name with `GPU_` (so it's excluded from CPU-only filters).
- Use `Settings::set()` to override options within a test; the context fixture calls `Settings::reset()` after.
- Set `OMP_NUM_THREADS` to roughly `[NUM_PHYSICAL_CORES/2, NUM_PHYSICAL_CORES)` for multi-threaded tests —
  oversubscription slows them down.
- Full list of `ROCROLLER_*` env vars (logging, debugging, e.g. `ROCROLLER_SAVE_ASSEMBLY`,
  `ROCROLLER_LOG_LEVEL`, `ROCROLLER_BREAK_ON_THROW`): see `lib/include/rocRoller/Utilities/Settings.hpp`
  and the "Logging and Debugging" section of `README.md`.

## Conventions

- **Default layout is column-major.** N (non-transposed) = column-major, T (transposed) = row-major.

## Git Workflow

- **Main branch**: `develop` (use for PRs, not `main` or `master`)
- **Branch naming**: `users/<username>/<feature-name>`

## File Structure Conventions

- `Foo.hpp`: Class/concept definitions with declaration-only functions
- `Foo_impl.hpp`: Short inlinable function definitions (included at bottom of `Foo.hpp`)
- `Foo_fwd.hpp`: Forward declarations, type aliases (e.g., `FooPtr = std::shared_ptr<Foo>`), minimal includes
- `Foo.cpp`: Longer function definitions

## Coding Style

- **Formatting**: Follows `clang-format` version 13 (`./scripts/fix-format`)
- **Functions**: Static/free functions start with uppercase; instance functions start with lowercase
- **Variables**: Private members start with `m_`; public members do not
- **Naming**: camelCase (not snake_case)
- **Macros/CMake**: `UPPER_CASE` with `ROCROLLER_` prefix
- **C++ Standard**: C++20 is available but prefer C++17 modern practices unless explicitly reviewed
- **Memory**: Use `std::make_unique`/`std::make_shared` instead of `new`; use `std::vector` for arrays
- **Type aliases**: Use `using` instead of `typedef`

### Adding New Tests

- **Preferred**: Add Catch2 tests in `test/catch/` → builds `rocroller-tests-catch`
- **Legacy**: GTest tests in `test/unit/` → builds `rocroller-tests`

### Debugging and Error Handling

- Use `Log::debug()` etc. from `Utilities/Logging.hpp` for debug output.
- Prefer `AssertFatal` with `ShowValue()` over `assert`.
