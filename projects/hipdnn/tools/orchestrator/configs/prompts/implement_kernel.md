You are making `TEST_ENGINE` execute a hipDNN graph.

`TEST_ENGINE` is an engine in hipDNN's hip-kernel-provider at
`${vars.test_engine_dir}`. It is a skeleton: the engine, its plan builders and its
kernel compilation path all work, and its one kernel translation unit is a set of
empty stubs. Nothing it claims to support actually computes anything.

Your job this round: make it run **this graph** correctly, and make it **beat**
`${inputs.reference_engine}` -- the engine it duplicates -- on that graph.

- Graph:        ${inputs.graph}
- Operations:   ${steps.profile.outputs.op_tokens_text}
- Target arch:  ${inputs.arch}
- Reference:    ${inputs.reference_engine}
- Speed target: median must be at most ${inputs.perf_target_ratio} x the reference median
- Attempt:      ${loop.attempt} of ${loop.max_iterations}
- Operator notes: ${inputs.notes}

# Read the previous round first

`${loop.feedback_path}` accumulates one section per failed round. **Read it before you
change anything.**

When a round fails, the note names the step that failed and the directory its logs are
in, for example `step 'build' exited 1 (logs: .../kernel_cycle/iter-00/build)`. Open
`stdout.log` and `stderr.log` in that directory. That is where the compiler
diagnostic, the failing test names or the timing table actually are; the one-line note
is not the evidence. If a previous round left partly-working code in place, you are
fixing it, not starting over.

# Work out what is missing before you write anything

1. **Read the graph.** Node types, tensor UIDs and how they connect, data types,
   layouts, strides, which tensors are inputs, outputs and intermediates. A virtual
   tensor is an intermediate that is never written to memory -- the operations around
   it have to fuse.
2. **Read what TEST_ENGINE already has.** `${vars.test_engine_dir}/plans/` holds the
   plan it can build and `${vars.test_engine_dir}/kernels/` holds the source that plan
   compiles. The engine is deliberately narrow -- one plan builder, one graph shape,
   one kernel entry point -- so in the common case nothing is missing except the kernel
   body. Establish that first; it is a very different round of work from adding a plan,
   and guessing wrong wastes the round.
3. **Read the reference.** `${vars.mlops_engine_dir}` is `${inputs.reference_engine}`,
   the engine TEST_ENGINE was ported from. Its plans and kernels are the specification
   for what each entry point computes and the baseline this run is timed against.
   TEST_ENGINE's plan is a port of its plan, so the structure already has a worked
   example. It supports far more shapes than TEST_ENGINE does; that breadth is not
   something to reproduce.
4. **Confirm the engine will accept the graph at all.**
   `${vars.test_engine_dir}/plans/` carries the applicability checks that decide
   whether TEST_ENGINE claims a graph. An engine that declines this graph never runs,
   and the benchmark step fails with "engine does not support this graph" no matter
   how good the kernel is.

**Do not widen the engine.** It plans exactly one graph shape on purpose. Adding
fusion, training, backward, or a second plan builder is out of scope even when the
reference engine has them: none of it is benchmarked, none of it is in the suite this
run gates on, and every extra plan is another path that can report success on work
nobody measured. If the graph genuinely needs a shape the engine cannot plan, say so in
`summary` and stop rather than building it.

# Kernel entry points and their signatures

Do not guess signatures. They are fixed by the call sites, and a mismatch is silent
corruption rather than a compile error: `hipModuleLaunchKernel` reads one pointer per
parameter the *kernel* declares, so a kernel declaring the wrong number or the wrong
types reads the wrong memory and the failure surfaces as a wrong result somewhere else.

For every kernel you write, the authority is the plan that launches it:

- `_compiledProgram->getKernel("<name>")` names the entry point you must define.
- The matching `->launch(stream, ...)` gives that entry point's parameter list, in
  order. Plans usually have several launch sites -- fused-activation and plain paths,
  one-pass and multi-pass variants -- and every one of them has to work.
- The plan's `compile()` and its `*KernelCompileOptions.hpp` give the `-D` macros the
  source is instantiated with: data types, workgroup shape, vector width, layout,
  variant selection. Your kernel is a template driven by those macros, not a kernel
  specialised to the shapes in this one graph.

# Header dependencies -- the part that is easy to get wrong

Kernel sources are embedded at configure time into **one map shared by the whole
`hip_kernel_provider_impl` target**, keyed by file basename
(`${vars.repo_root}/dnn-providers/hip-kernel-provider/src/cmake/KernelEmbedding.cmake`).
Two consequences that pull in opposite directions:

- You **may** `#include "<basename>"` for any header another engine already embeds --
  `VectorTypes.hpp`, `HipKernelActivation.hpp`, `HipKernelMath.hpp` and the
  per-operation helpers next to the `HIP_MLOPS_ENGINE` kernels -- with no CMake change
  at all. hiprtc resolves includes out of the shared map.
- You **must not** add a basename to TEST_ENGINE's embedding list that is already
  embedded elsewhere. Duplicate basenames emit a duplicate symbol and the provider
  stops building.

Check `${vars.mlops_engine_dir}/kernels/CMakeLists.txt` for what is already embedded
before adding anything to `${vars.test_engine_dir}/kernels/CMakeLists.txt`. A helper
that is not already embedded goes either inline in your kernel file or into a new,
uniquely-named header you add to TEST_ENGINE's list.

# Correctness requirements

The integration suite runs `${steps.profile.outputs.gtest_filter}` against
TEST_ENGINE, and it does not stop at the shapes in your graph.

- Handle the whole shape family the plans admit, not the one case in the graph file.
  Ranks, layouts and data types the applicability check accepts are all in scope, and
  extents that do not divide evenly by the workgroup or vector width are ordinary
  inputs rather than edge cases. Guard every global memory access.
- Every graph output must be written; every declared input must be read as declared.
  Respect strides and layout instead of assuming contiguity or one memory order.
- Use the accumulation precision the reference uses. Reductions in fp16 accumulate
  visible error long before the comparison tolerance is reached.
- Put a `__syncthreads()` between every shared-memory write and the dependent read.
- Kernel sources must compile as standalone hiprtc translation units: no host headers,
  no host-only types, no include hiprtc cannot resolve from the embedded map.

Passing by narrowing is not passing. The suite counts how many cases it *passes*, not
just how many it fails, against what TEST_ENGINE accepted before you started -- so
tightening an applicability check until the hard cases skip fails the round rather
than ending it.

# Speed target -- read this before you decide how to implement

After the suite passes, the graph is timed on TEST_ENGINE and on
`${inputs.reference_engine}`, back to back, same iteration count, same machine state.
The round ends only when

    TEST_ENGINE median <= ${inputs.perf_target_ratio} x ${inputs.reference_engine} median

**A port of the reference is a guaranteed failure.** Copying
`${inputs.reference_engine}`'s kernels, or rewriting the same algorithm with the same
memory access pattern, lands at a ratio of about 1.0 by construction. That passes
every correctness gate and then fails this one, every iteration, until the budget runs
out. Read the reference to learn what the operation *is* and which entry points do
what; do not treat it as the implementation to reproduce.

Beating it means doing something it does not do. The reference is a generic kernel
that has to serve every rank, layout, data type and shape the engine claims. This
graph is one shape. Things worth measuring your way against:

- Vector width: is the access pattern using the widest load and store the layout,
  alignment and element count actually permit?
- Workgroup shape and grid mapping: does the plan's choice suit this occupancy, and
  how much of each wavefront is doing useful work at the tail?
- Passes over memory: a bandwidth-bound operation is priced in bytes moved. Count the
  reads and writes of the large tensors and see whether any of them can be fused away.
- Reuse: per-channel parameters are tiny and read by every element of their channel.
  Where do they live across the spatial loop -- registers, LDS, or a reload per
  element?
- Layout: a channels-last and a channels-first traversal are not the same kernel. The
  generic path picks conservatively for both.

You must not buy speed with correctness: the integration suite runs before the
benchmark and every case it passed before has to still pass. Skipping work the graph
declares, narrowing what the engine accepts, or dropping the higher-precision
accumulator are not optimisations, and the gates ahead of this one catch all three.

The failure feedback names the exact budget in milliseconds and how far over it you
are, so treat each round as "cut this many ms", not "try to be faster".

# Scope

You may change anything under `${vars.test_engine_dir}`, and
`${vars.container_file}` for registering a new plan builder.

You may **not** change the integration test suite, the test bundles, the per-engine
TOML configuration, or `HIP_MLOPS_ENGINE`. Those define how this work is measured, and
they are hashed before and after you run; editing them fails the round. If a test
looks wrong, say so in `summary` and leave it alone.

Do not run the build or the tests yourself. The orchestrator builds, runs the suite
and benchmarks the graph against `HIP_MLOPS_ENGINE` immediately after you finish, and
feeds you the result. Time spent building here is time not spent on the kernel.

# Output contract

Write a JSON object to exactly this path, and nothing else that matters to stdout:

    ${step.result_file}

Schema (all keys required):

```json
{
  "engine_ops": ["<operation family each graph node maps to in TEST_ENGINE>"],
  "entry_points": ["<every __global__ entry point you defined or changed>"],
  "changed_files": ["<absolute path of every file you created or modified>"],
  "new_plans": ["<plan family you added, if any; [] when the plans already existed>"],
  "addressed_feedback": ["<each issue from the feedback file you fixed, and how; [] on the first attempt>"],
  "summary": "<what this round changed and why, in a few sentences>"
}
```

`changed_files` must not be empty, and the files must exist when you finish. The
orchestrator re-hashes the engine directory: a round that reports work but wrote
nothing to disk is failed on the evidence, not on the report.
