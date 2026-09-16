You are authoring a hipRTC-compilable HIP kernel that computes a hipDNN graph.

Execute the skill at `${vars.skills_dir}/hipdnn-kernel-authoring/SKILL.md`. Its
`RUNBOOK.md` is the only ordered workflow; read it and follow it. This prompt does not
replace it -- it tells you what this particular run is, where to put things, and what
will be measured after you finish.

- Graph:          ${inputs.graph}
- Operations:     ${steps.profile.outputs.op_tokens_text}
- Target arch:    ${inputs.arch}
- Engine to be:   ${inputs.engine_name} (stage two; you are not building it)
- Source kind:    ${inputs.kernel_source_kind}
- Attempt:        ${loop.attempt} of ${loop.max_iterations}
- Operator notes: ${inputs.notes}

# Read the previous round first

`${loop.feedback_path}` accumulates one section per failed round. **Read it before you
change anything.** When a round fails, the note names the step that failed and the
directory its logs are in. Open `stdout.log` in that directory: the hipRTC compiler
diagnostic, the list of unresolved entry points, or the failing contract checks are
there. The one-line note is not the evidence.

## You are probably not starting from nothing

This is attempt ${loop.attempt}. Every attempt in this run shares one directory, and a
previous attempt's work is still on disk -- it is not cleaned between rounds. **Before
you design anything, look:**

```
ls ${vars.kernel_dir}          # kernel sources a previous attempt wrote
ls ${run.dir}                  # its harness, build tree, logs, reports
```

If a kernel is already there, **your job is to finish or repair it, not to replace it.**
Read it, read any `*-report.json` and `logs/` beside it, and find out how far it got. A
previous round that ran out of time may have left a kernel that already compiles and
already passes its own numerics -- in which case the missing piece is the contract file
below, and rewriting the kernel from scratch throws away work that was already correct
and spends this attempt's budget re-deriving it.

Start over only when the evidence says the existing kernel is wrong, and say in
`summary` what you found and why you kept, repaired or replaced it.

## Budget

Your wall-clock budget is finite and a round that overruns it is killed with no partial
credit. **Write the contract file as soon as the kernel and its harness results exist**,
then improve things and rewrite it if you learn more. A complete result that is later
refined beats a perfect result that never got written down.

# Where things go

Everything you write goes under `${run.dir}`:

- `${vars.kernel_dir}` -- the kernel sources and any headers.
- A correctness harness, built wherever you like under `${run.dir}`, linked against the
  installed hipDNN at `${vars.ingestor_install_dir}` (use it as the `find_package` root;
  it was installed before you started).

**You may not edit anything in the checkout.** `${vars.repo_root}` is readable -- read
the existing packs, the reference kernels, the schemas, whatever you need -- but a single
byte changed anywhere under `dnn-providers/` fails this round. The kernel reaches the
product tree in stage two, through an agent whose scope allows it. If you believe a
product file is wrong, say so in `summary` and leave it alone.

# The ABI is yours to choose

The answer to "what is the launch ABI" is **unbound**, and that is the normal answer, not
a fallback. Stage two writes the dispatch handler's `launch()` from the parameter list
you declare. So choose the parameter list that makes the kernel clean, write it down
exactly, and stage two will be held to it.

Do not copy a signature from `PointwiseNative.cpp` or `ConvNative.cpp`. Those are
reference scaffolds -- `PointwiseAdd` computes one element under
`if(blockIdx.x == 0 && threadIdx.x == 0)` -- and their ABIs were chosen for toys.

# What the orchestrator checks after you finish

These run automatically. Knowing them is not cheating; they are the contract.

**It compiles your sources itself.** It loads hipRTC directly, compiles every file in
`bundle.sources` with `--offload-arch=${inputs.arch}` and `-D<NAME>=<first legal value>`
for every entry of `required_defines`, and then asks the resulting code object whether
each `entry_points[].name` is in it. Three consequences:

- Every entry point must be `extern "C" __global__`. A C++-linkage kernel compiles, and
  its symbol is then `_Z13something...`, which is not the string the descriptor will hold
  and not what `getKernel()` will ask for. This is checked, and it is checked against the
  literal name you declare.
- Every source you list must compile standalone as a hipRTC translation unit: no host
  headers, no host-only types, and no include hipRTC cannot resolve. Headers you list in
  `bundle.headers` are offered to hipRTC by basename and may only be `.h`, `.hpp` or
  `.cuh`.
- `entry_points[].source_file` must be one of `bundle.sources`. An entry point whose
  source is not in the bundle is compiled by nothing.

**It proves every required macro is guarded.** For each entry of `required_defines` it
compiles again with that one macro dropped, and **requires that compile to fail**. So
every macro you declare needs, at the top of the source that uses it:

```c
#ifndef HKP_SOMETHING
#error "HKP_SOMETHING must be defined"
#endif
```

A macro whose absence still compiles is not a specialization axis. It is a token that
silently means whatever it happens to mean, and the failure shows up only in the numbers.

**It runs your harness and reads the report.** It does not read your opinion of the
result. `numerics.harness_command` is the argv it runs and `numerics.report_path` is the
JSON it then opens. The report file is deleted before the launch, so a stale one reads as
a failure. The harness must be a real compiled executable: an interpreter or a shell as
`harness_command[0]` is refused outright.

Your harness must write exactly this, and the orchestrator asserts on all of it:

```json
{
  "reference": "<the hipDNN reference executor you compared against>",
  "reference_executed": true,
  "kernel_launched": true,
  "outputs_compared": ["<name of every graph output compared>"],
  "mismatched_outputs": ["<the ones that failed; [] is the goal>"],
  "per_output": [
    {"name": "...", "max_abs_err": 0.0, "max_rel_err": 0.0,
     "rtol": 1e-5, "atol": 1e-6, "pass": true, "sentinel_changed": true}
  ],
  "shapes": [{"label": "...", "dims": [1, 2, 3, 4]}],
  "device": "${inputs.arch}",
  "seed": 0
}
```

Three of those fields exist because the corresponding failure is otherwise
indistinguishable from success:

- `reference_executed` -- a reference that declined or skipped is an unmeasured bucket.
  Check the decline list in `dnn-providers/integration-tests/README.md` before you
  build the harness, not after: paged KV, varlen, ragged offsets, block-sparse masks,
  sink tokens, dropout, FP8 descale and softmax statistics are declined by both the CPU
  and GPU references, and CPU is not a fallback for any of them.
- `kernel_launched` and `sentinel_changed` -- fill every output buffer with a sentinel
  before the launch and confirm it changed. A kernel that never ran leaves both sides
  comparing the harness's own fill, which matches perfectly.
- `outputs_compared` -- every graph output, including ones you did not expect to be
  interesting.

Run the shape set you intend to claim, not one point of it, plus the boundaries the
specification implies: a non-contiguous stride, a dimension of 1, a non-tile-multiple
extent, and the smallest and largest shapes in the envelope.

# What stage two needs from you, and cannot recover

Stage two writes `graph_match`, the metadata, the matchers and `launch()` from this
contract. Anything you leave out is a constraint the pack will claim by accident.

- `admits` is the raw material of `graph_match`. A precondition you do not state is a
  shape the engine will accept and compute wrongly.
- `launch_geometry.guards_own_bounds` decides who owes the bounds check. If `prepare()`
  computes the grid from the shape, the final block is partially populated and **the
  kernel must guard `index >= total` itself**, with `int64_t` arithmetic. A constant grid
  is the exception, not the rule.
- Anything derived, conditional or computed belongs in the handler, not in a descriptor
  define: the descriptor's substituter does literal replacement and nothing else.

# Output contract

Write a JSON object to exactly this path, and nothing else that matters:

    ${step.result_file}

```json
{
  "graph": "<absolute path to the graph you were given>",
  "arch": "${inputs.arch}",
  "operation_spec": "<the mathematics, the conventions it depends on, and the disposition of every matched schema field>",
  "field_dispositions": [
    {"field": "<schema field>", "disposition": "consumed|rejected|inert", "why": "..."}
  ],
  "tensors": [
    {"uid": 0, "role": "input|output|virtual", "dims": [], "strides": [], "dtype": "FLOAT"}
  ],
  "decomposition": [
    {"launch": "<name>", "entry_point": "<symbol>", "inputs": [0, 1], "outputs": [2],
     "grid": "<formula>", "block": "<formula>", "workspace_bytes": "<formula or 0>"}
  ],
  "entry_points": [
    {"name": "<the extern \"C\" symbol>",
     "signature": "<the full declaration, verbatim, as it appears in the source>",
     "source_file": "<absolute path, one of bundle.sources>",
     "params": [
       {"name": "a", "type": "const float*", "role": "tensor_in|tensor_out|workspace|scalar", "uid": 0}
     ]}
  ],
  "bundle": {"sources": ["<absolute path>"], "headers": ["<absolute path>"]},
  "required_defines": [
    {"name": "HKP_X", "legal_values": ["FLOAT", "HALF"], "why": "...", "guarded": true}
  ],
  "compile_options": ["<any option beyond --offload-arch and the -D set above>"],
  "launch_geometry": {"grid": "<formula>", "block": "<formula>", "guards_own_bounds": true},
  "workspace_bytes": "<formula, or \"0\">",
  "admits": {"dtypes": [], "layouts": [], "rank": 4, "min_extents": {}, "max_extents": {},
             "divisibility": [], "alignment": "<statement>"},
  "abi": "unbound",
  "numerics": {
    "harness_command": ["<absolute path to your harness>", "--out", "<absolute path>"],
    "report_path": "<the same absolute path>",
    "tolerance": {"<output name>": {"rtol": 1e-5, "atol": 1e-6}},
    "tolerance_provenance": "<where these numbers come from - not 'chosen'>"
  },
  "does_not_prove": ["<untested shape, dtype, architecture; every reference decline>"],
  "summary": "<what this round did, in a few sentences>"
}
```

`params` and `signature` are the same statement twice, and they are compared: the
parameter count parsed out of `signature` must equal the length of `params`. Every
output UID in `tensors` must appear as some entry point's `tensor_out` parameter.
`does_not_prove` may not be empty -- a kernel proved on one device has limits, and an
empty list claims it has none.
