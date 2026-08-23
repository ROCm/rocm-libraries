# [hipBLASLt] Overlap helper-kernel compile with the assembly build and library writing

## Summary

The helper-kernel HIP source (`Kernels.cpp`) was compiled serially after the
assembly-kernel generation and link, so its full duration was added to the build
time. This change runs the helper compile (the link step that produces
`Kernels.so-000-<arch>.hsaco`) concurrently with the rest of the run, which does
not depend on it.

`writeHelpers` only needs `kernelHelperObjs` (already available) and the static
headers copied by `copyStaticFiles`. So the helper compile is launched on a
worker thread up front and joined only at the end of the run, after the master
solution libraries are written. It therefore overlaps both the assembly-kernel
generation/link and the end-of-run solution-library serialization
(`passPostKernelInfoToLibrary` and the `writeMsl` master-library writes).

The helper compile is on the critical path only if it outlasts everything else
it overlaps; otherwise it is effectively free. Build time is never worse than
before.

## Output is unchanged

The single `Kernels.so-000-<arch>.hsaco` per arch is still produced by the same
`buildSourceCodeObjectFiles` call. No runtime or loader changes are required.

## Why a worker thread (and not joblib)

- This is a single long-running background task, not a map over a collection.
  `ParallelMap2` (joblib) is for fanning a function out over many items; here
  there is exactly one job to run off the main thread, so a thread is the right
  primitive.
- The work is I/O-bound and releases the GIL: it is mostly an `amdclang`
  subprocess call plus file moves, so a thread genuinely runs concurrently with
  the main thread while it drives the assembly pool and writes libraries.
- It must overlap `ParallelMap2` itself (used by both the assembly generation and
  the library writing). The assembly path uses joblib's loky backend, which
  spins up a separate process pool. A thread in the main process composes cleanly
  with that pool instead of nesting a second competing pool.
- No pickling. loky runs in separate processes and would require pickling the
  task and its toolchain arguments (compiler, bundler, paths); a thread shares
  memory and passes them by reference, and avoids fork-with-threads hazards.

## Performance

Measured on gfx90a (2765 assembly kernels, 34 helper objects expanding to 222
helper kernels, 64 build jobs):

- Sequential (before): total 92.8s
- Overlap with the assembly build only: total 70.9s
- Overlap with the assembly build and the library writing (this change): total
  66.4s (about 28 percent faster than sequential)

Phase breakdown for this change: assembly generation 24.2s, helper compile 28.9s,
pass-kernel-info 2.1s, master-library writes 7.9s. The helper compile overlaps
all of these, so it no longer contributes to the critical path.

The win scales with how much independent work is available to overlap the helper
compile with. It is never a regression.

## Correctness and equivalence

All 222 helper kernels were verified to be ISA-identical to the sequential
build, kernel for kernel.

Note: whole-file code-object hashes are non-deterministic in the baseline itself
(amdclang embeds a content/build-id that varies run to run), so per-kernel ISA
comparison is the correct equivalence check rather than a file hash diff.

## Testing

- Ran TensileCreateLibrary for gfx90a before and after the change; compared the
  per-kernel disassembly of the resulting helper code object: identical for all
  222 kernels.
- Confirmed no errors and that the expected `Kernels.so-000-gfx90a.hsaco` is
  produced.
