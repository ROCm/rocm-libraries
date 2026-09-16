# Runbook: author a kernel for a hipDNN graph

The **only ordered workflow**. [SKILL.md](SKILL.md) owns entry and completion; the
linked pages own contracts. Each gate requires a current observation; a previous
session's log, a sibling graph's result or a proposed command is not evidence.

## Paths and interpreters

Resolve absolute paths first. These are explicit arguments, not implicit tool inputs:

```bash
REPO=/absolute/path/to/rocm-libraries
HIPDNN="$REPO/projects/hipdnn"
INSTALL=/absolute/path/to/hipdnn-install      # find_package root; produced by step 2
WORK=/absolute/path/to/scratch                # kernel source + harness, outside product source
GRAPH=/absolute/path/to/graph.json-or-.bin    # or a samples/ source path
ARCH=gfx942
```

Keep authored experiment material — kernel drafts, harness, logs — under the
workspace's WIP/evidence roots, not in product source, until the kernel is handed to
integration. Do not run GPU work on a login host; see [device-envelope.md](device-envelope.md).

## 1. Entry

Record [SKILL.md](SKILL.md)'s entry contract: the graph and its form, target
architecture(s) and local availability, intended launch ABI or "unbound", dtypes,
layouts, and the shape envelope requested. State whether the request is "serve this
graph" or "serve this operation family"; the second is a scope decision, not an
assumption you may make.

Write the dtypes, the layouts — derived from strides, never assumed — and the shape
envelope down as a **claim**, not a note. That claim is what decides the
runtime-versus-`-D` split at step 6 and step 7, and it is the thing step 8's coverage
is measured against; an envelope nobody wrote down cannot be reported as
under-covered.

**Gate:** all five entry-contract facts recorded — the graph and its form, a named
architecture with its local availability, a stated ABI position, and the written
dtype/layout/shape-envelope claim. The fifth, a capable reference for this graph, is
the one this step cannot settle alone: name the reference you intend here, step 2's
gate turns that name into an `isApplicable` answer for *this* graph, and step 4
settles it. The other four are blocked here when missing, never defaulted.

## 2. Build and install hipDNN, and run a minimal driver against it

`$INSTALL` is declared in the Paths block and produced by no other step, yet three
later gates are observable only from a program built against it: step 4 must *ask* a
reference whether it can execute this graph, step 7 needs hipRTC compiling your
source, and step 8 needs both. Do this before reading the graph — a reference that
declines this graph changes the plan, and it is cheaper to learn that now than at
step 4.

If you already have an install, this step is a check, not a skip: an old install or a
copied command is not an observation.

**Build and install.** Use the workspace skills rather than a build system of this
skill's own. `.claude/skills/hipdnn-build` configures and builds `$HIPDNN` under a
ROCm toolchain into its `build/` directory; `.claude/skills/hipdnn-install` installs
that build tree into the branch's install prefix beneath the workspace WIP root. Take
the invocation and the prefix from those two pages. `$INSTALL` is then that prefix —
the directory whose `lib/cmake/` holds `hipdnn_frontend/`, `hipdnn_data_sdk/` and
`hipdnn_test_sdk/`, which is what [harness.md](harness.md)'s `find_package` calls
resolve against.

**Those two pages carry no feature flags, and the defaults are off.** They configure a
toolchain and a prefix, nothing more. The flags that decide whether your graph's
operation exists in the build at all are owned by
[hipdnn-superbuild](../hipdnn-superbuild/SKILL.md)'s option table — read it and set what
your graph needs. `HIPDNN_ENABLE_SDPA` is the one that bites: it defaults **OFF**
(`projects/hipdnn/CMakeLists.txt:55`), and with it off the SDPA API is `#ifdef`-compiled
out of the frontend, so an attention graph does not fail loudly — the plan **declines**.
`HIPDNN_ENABLE_KERNEL_INGESTOR` is off by default too, and also gates
`hipdnn_validate_descriptors`, which is why that binary is usually missing later.

A decline you caused by building without the flag is indistinguishable, at the API, from
a decline the reference genuinely owes you. **Resolve that before believing either.**

**The minimal driver.** The smallest program that makes the later gates observable —
not the harness, which comes at step 8. Built against `$INSTALL`, it does two things:

1. Loads `$GRAPH`, serializes it with `to_binary()`, constructs a
   `hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor`, calls `isApplicable`
   (`hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp:38`)
   and **prints the answer**. `execute`
   (`hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp:56-58`)
   takes the same bytes plus the variant pack; its call shape is in
   [harness.md](harness.md) under "The oracle" and is not repeated here.
2. Compiles one trivial HIP source string through hipRTC once per `$ARCH`, in the
   production envelope ([device-envelope.md](device-envelope.md)), and prints per
   architecture the compile result and the program log.

The test SDK is reachable from an install — it installs its targets, its headers, its
generated version header and its export set (`projects/hipdnn/test_sdk/CMakeLists.txt:54`,
`:57`, `:68`, `:74`). The integration tests' harness is not; [harness.md](harness.md)
says what that costs you.

**Gate:** the driver's own printed output — an `isApplicable` answer for *this* graph,
and a hipRTC compile result for each `$ARCH`, **with the feature flags this build was
configured with recorded beside them**. A build that exited zero is not this gate, and
neither is an install you inherited. Step 4 rests on the first half, step 7 on the
second, step 8 on both. A driver that was not run reports NOT RUN rather than a bare
pass.

`isApplicable` returning false is a real observation and feeds step 4's decline
handling — **but only once you have confirmed the operation was compiled in.** Check the
flags against the configure log's `hipDNN: SDPA support disabled` / `enabled` line
(`projects/hipdnn/CMakeLists.txt:57-60`) before recording a decline. A decline from a
feature that was never built is a fact about your build, not about the reference, and
carrying it into step 4 produces a BLOCKED report that is internally consistent and
wrong.

## 3. Read the graph and specify the operation

Follow [graph-analysis.md](graph-analysis.md). Enumerate nodes, node types, tensor
UIDs, dims, strides, dtypes and virtual status; reconstruct edges from the
`*_tensor_uid` fields; then write the operation specification — the mathematics, the
conventions it depends on, and the disposition of every schema field.

Do not begin from the operation's popular name. `SDPA_FWD` names a family whose
masking, scaling, GQA, layout and deprecated-field precedence differ per graph, and
at least one precedence rule in tree is a live trap (`Knowledge/hipdnn/sdpa-mask-attribute-precedence.md`).

**Gate:** every node classified, every tensor classified input/output/virtual with
its UID resolved and its role derived as [graph-analysis.md](graph-analysis.md)
describes, every matched schema field consumed, rejected or proven inert — per
enumerator where the field is an enum — and the specification written down.
An unresolved semantic question blocks the affected path.

## 4. Establish the oracle

Confirm a hipDNN reference can execute *this* graph before authoring anything —
[harness.md](harness.md), and the decline list in
`$REPO/dnn-providers/integration-tests/README.md` ("What the reference
executors cannot verify"). Paged KV, varlen, ragged offsets, block-sparse masks,
sink tokens, dropout, FP8 descale and softmax statistics are declined by both CPU
and GPU references; **CPU is not a fallback for any of them.**

If the graph carries a declined feature, the choice is: narrow the graph to a
representable form and state the narrowing, supply an independently trusted
reference and state its provenance, or record BLOCKED. Fabricated golden data and a
reference derived from the kernel under test are both disqualifying.

**Gate:** a named, capable, independent oracle, or an explicit BLOCKED.

## 5. Mine prior art and pin the device envelope

Use [prior-art.md](prior-art.md) to find how the operation is actually implemented —
in-tree first (`projects/composablekernel`, `projects/miopen`, `projects/hipblaslt`,
`projects/rocprim`, `projects/rocwmma`), then external semantic references. Record
what you took from where, and what you deliberately did not take.

Use [device-envelope.md](device-envelope.md) to pin the compile envelope and the
target device's facts. When the target architecture is not the local one, obtain its
facts from that device, not from memory — and obtain them by **running the probe, not
by describing one**: `device_probe.py --mode early --arch <exact gfx token>
--sweep-root <dir>` for an architecture this host might hold, or the
`alola-gpu-test` path in [device-envelope.md](device-envelope.md) for one it does
not. Record what came back. Three outcomes, all acceptable, all reported:

- **observed** — a utility ran and the device answered. The facts are that device's.
- **unobserved** — exit 3, `ProbeUnavailable`: no inspection utility could be run at
  all. That is a statement about the tooling, not about the host, and it does **not**
  fail this gate. Carry it forward as a stated limitation.
- **device-absent** — a utility ran and contradicted the request. That is a real
  observation too, and the architecture then needs the scheduler rather than this host.

Where a fact could not be observed, say so and name the documented source used
instead — the ROCm per-architecture specification table [prior-art.md](prior-art.md)
links is the intended one. What this gate forbids is a device fact reported as
observed when no probe ran.

**Gate:** a named algorithm with a source, and per target architecture a recorded
compile envelope, a device-fact set, and the probe attempt with its outcome —
observed, unobserved or device-absent. An unobserved outcome passes this gate as a
stated limitation; facts taken from documentation instead are labelled as such.

## 6. Decompose

Decide, per [graph-analysis.md](graph-analysis.md): one kernel, a legal fusion, or an
ordered sequence of launches. Fusion is preferred where the producer/consumer edge is
elementwise-local or already tiled together; it is not preferred where it forces a
global synchronization inside a kernel.

A multi-launch decomposition is a legitimate answer whenever the launch ABI is unbound —
the normal case, [SKILL.md](SKILL.md) — and not a fallback to apologize for; on a bound
ABI it is a signal you are on the wrong branch, because there `prepare()` issues exactly
one launch at a grid the installed handler chose.
Record for each launch: its inputs and outputs by UID, its grid/block shape, and any
scratch buffer it needs. Scratch is the decomposition's cost: an intermediate that
the graph marks virtual has no caller-supplied buffer, so a multi-launch plan needs
workspace, and whether the downstream launch ABI can supply it is an integration
constraint that must be raised now, not discovered later.

Record the generalization decision from [SKILL.md](SKILL.md): which quantities are
runtime arguments and which are `-D` specializations, with a reason for each
specialization.

**Gate:** a written decomposition with per-launch contracts, scratch requirements and
the generalization decision.

## 7. Author the kernel

Write HIP source inside the hipRTC envelope ([device-envelope.md](device-envelope.md)).
Non-negotiables:

- The entry point is `extern "C" __global__`. Its argument list is normally **yours to
  choose**, because the integration writes the handler's `launch()` around it. Only when
  the ABI is bound — the `hiprtc_file` drop-in branch, a pack that already exists and
  already ships a handler — is the list that handler's `launch` argument list verbatim, in
  order, and then **wrong arity or order is diagnosed nowhere**: hipRTC compiles it,
  symbol lookup resolves it, and the launch passes whatever it has into whatever you
  declared. On that branch the same handler also fixes the grid and block shape and issues
  exactly one launch, with the same silence on a mismatch — take the geometry and the
  launch count from its `prepare()` the way you take the list from its `launch()`
  ([SKILL.md](SKILL.md)).
- Index through the graph's **strides**, not an assumed contiguous layout. Layout is
  not an enum in the schema; it exists only as the stride pattern.
- Accumulate in `float` regardless of storage dtype, and cast on store — the in-tree
  precedent is `ConvFwd.cpp`.
- Use `int64_t` index arithmetic wherever an offset can exceed 2^31.
- Guard every compile-time-bound macro with `#ifndef <NAME>` / `#error`. An unbound
  token otherwise compiles against whatever it happens to mean and fails only in the
  numbers.
- One source, many variants: resolve a dtype tag to a device type with a two-level
  macro paste, never a one-level one.

**Gate:** hipRTC compiles the source for every target architecture, with the compile
log captured. A compile is not a correctness result and must not be reported as one.

## 8. Prove it

Build and run the harness of [harness.md](harness.md): identical seeded inputs to
both sides, hipDNN reference on one, your kernel on the other, per-output comparison
at a tolerance whose provenance you state.

Three assertions the harness must make, because each corresponding failure is
otherwise indistinguishable from success:

1. The reference actually executed — not declined, not skipped.
2. Your kernel actually launched and wrote its output — sentinel-fill outputs before
   launch and confirm they changed.
3. The comparison covered every graph output, including ones you did not expect to
   be interesting.

Run the shape set you claim, not one point of it, plus the boundary cases the
specification implies: a non-contiguous stride, a dimension of 1, a
non-tile-multiple size, and the smallest and largest shapes in the claimed envelope —
across every dtype and every architecture claimed at step 1.

**When the numbers do not match.** The expected first run, not an exception. Work in
this order and stop at the first thing that explains it:

1. **Reduce.** Smallest failing shape, one failing dtype, one failing output. A
   mismatch you can read by hand is a different problem from the one you cannot.
2. **Hand-compare one element** against the step-3 operation specification — the
   mathematics you wrote down, not the kernel you wrote.
3. **Sentinel-check.** Confirm the kernel wrote at all, and for a multi-launch plan
   that each scratch intermediate was written before it was read; [harness.md](harness.md)
   describes both fills.
4. **Re-check the precedence traps** [graph-analysis.md](graph-analysis.md) names —
   mask precedence, a scalar arriving by a different route than you assumed, a
   window or offset convention off by one — against *this* graph, not the family.
5. **Run the contiguous variant** of the same shape. Still wrong means arithmetic;
   suddenly right means indexing, and the strides are where to look.
6. **Only then tolerance**, and only with a named cause — accumulation order, a
   reference that does not split input and output dtypes, a genuinely wider
   intermediate. A number chosen because it made the test pass is a defect report in
   disguise ([harness.md](harness.md)).

**Gate:** the coverage run, reported against the coverage claimed. Per output, a pass
at a stated tolerance on named shapes on a named device, with the three assertions
above satisfied; and each boundary case, dtype and architecture claimed either run
and passed, or named NOT RUN. Coverage you did not run does not fail this gate — an
unreported gap does. A skipped reference, an unlaunched kernel or an all-zero
comparison is a failed gate.

## 9. Report and hand off

Report: the operation specification, the decomposition, the kernel source and its
entry-point signature, the compile options per architecture, the harness invocation,
the numeric results per output and shape, and the does-not-prove list from
[SKILL.md](SKILL.md).

For integration, state the four handover facts
[hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) consumes at its
RUNBOOK step 1: the entry-point signature, the bundle's file set (sources plus every
header they include), the compile-time macros the source requires bound and their legal
values, and the launch geometry and workspace the kernel assumes. Whether the kernel must
guard its own bounds follows from that geometry and must be stated. If those conflict
with an existing pack's registered handler, say so — that conflict is a rebuild decision
for the integration skill, not something to paper over by changing the kernel's
mathematics.

**Gate:** the report is complete and every claim in it traces to an observation in
stages 2-8.
