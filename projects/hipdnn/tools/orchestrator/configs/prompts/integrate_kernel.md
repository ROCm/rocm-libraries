You are landing a proved kernel as a **new** generic-kernel-ingestor pack: its native
symbols, its descriptors, its registration with the shared test suite, and the graphs
that verify it.

Execute the skill at `${vars.skills_dir}/hipdnn-kernel-integration/SKILL.md`. Its
`RUNBOOK.md` is the only ordered integration workflow -- six steps, each with a named
artifact. Read the table at the top of it before you start step 1 and budget for all six:
**nobody discovers step 5 at step 5.** This prompt tells you what this run is and what
will be measured; it does not replace the runbook.

- Engine name:      ${inputs.engine_name}
- Pack class:       ${steps.identity.outputs.pack_class}
- Descriptor slug:  ${steps.identity.outputs.descriptor_slug}
- Registration fn:  ${steps.identity.outputs.register_symbol}
- Native source:    ${steps.identity.outputs.native_file}
- Descriptor dir:   ${steps.identity.outputs.descriptor_dir}
- Engine TOML:      ${steps.identity.outputs.engine_toml}
- CTest target:     ${steps.identity.outputs.external_test_target}
- Census filter:    ${steps.identity.outputs.census_filter}
- Source kind:      ${inputs.kernel_source_kind}
- Target arch:      ${inputs.arch}
- Graph:            ${inputs.graph}
- Attempt:          ${loop.attempt} of ${loop.max_iterations}
- Operator notes:   ${inputs.notes}

Those names are derived once, from the engine name, and written to
`${vars.identity_file}`. Use them exactly. They are what the validator, the CMake target,
the TOML lookup and `--test-engine` are all checked against, and a second spelling
anywhere is a run that validates one engine and tests another.

# Read these two files before you do anything else

**`${vars.authoring_file}`** is your input. It is the handover contract from the agent
that authored and proved the kernel: the entry-point signatures, the bundle's file set,
the macros the source requires bound with their legal values, the launch geometry and
workspace, and the admitted-shape envelope. You must echo its sha256 back in your own
contract, so read it rather than re-deriving anything from the tree.

**`${loop.feedback_path}`** accumulates one section per failed round, naming the step
that failed and the directory its logs are in. Open that directory's `stdout.log`. The
failing check names and the compiler diagnostics are there; the one-line note is not the
evidence.

## You are probably not starting from nothing

This is attempt ${loop.attempt}, and nothing a previous attempt wrote has been reverted.
Unlike the authoring stage, your output lands in the **checkout**, so a previous round's
work is sitting in the product tree right now. Before you generate or write anything:

```
git -C ${vars.repo_root} status --short        # everything a previous attempt changed
ls ${vars.ingestor_dir}                        # the pack, its symbols, its descriptors
ls ${vars.provider_tests_dir}                  # the bundle cases
```

If `${vars.integration_file}` already exists, a previous attempt got as far as writing a
contract; read it and treat it as a claim to verify, not as truth.

**Repair what is there rather than regenerating over it.** Re-running the generator on a
tree that already holds a half-finished pack can quietly reintroduce placeholders that a
previous round had already filled in -- run
`generate.py --check-placeholders` first and let its output tell you what is genuinely
missing. Say in `summary` what you found and whether you kept, repaired or regenerated
it.

## Budget

A round that overruns its budget is killed with no partial credit, and the work above is
what the next attempt inherits. Land a coherent tree and write the contract as soon as
the pieces exist; refine afterwards.

# This is the create path. There is no pack to attach to.

Only `ConvNative.cpp` and `PointwiseNative.cpp` exist under
`${vars.ingestor_dir}/packs/`. Both are **reference scaffolds, not integration targets**:
`PointwiseAdd` computes one element under `if(blockIdx.x == 0 && threadIdx.x == 0)` at
grid 1x1x1, and `ConvFwd` is a naive direct convolution serving 6 of 1218
`ConvolutionFwd` bundle cases. They exist to exercise the ingestor path end to end and to
be read as worked examples. Hanging this kernel off either inherits a matcher, a geometry
and an ABI all chosen for a toy.

So: new symbols, a new pack, a new row in `IngestorPacks.cpp`. That is ordinary work with
a rebuild in it, and the orchestrator does the rebuild after you finish.

# Use the generator. Do not hand-write what it emits.

```
${steps.generator_venv.outputs.python} ${vars.generator_dir}/generate.py \
    --config <your config.yaml> --output-dir <an EMPTY directory under ${run.dir}> --dry-run
${steps.generator_venv.outputs.python} ${vars.generator_dir}/generate.py \
    --config <your config.yaml> --output-dir <the same directory> --force
```

`--force` is not optional after the first run: generating a second time into a non-empty
output directory is refused, and every iteration on a bundle or a metadata value hits
that. The generator derives the native symbol namespace from the engine name, so that
name is not a free label.

`${vars.generator_dir}/README.md` owns the config schema.
`${vars.generator_dir}/configs/scale_add.yaml` is the closest worked example for an
`embedded_source` direct-load pack; `configs/binary_ops.yaml` shows the multi-pack shape
with per-pack discriminators; `configs/hiprtc_dropin.yaml` shows the `hiprtc_file`
`bundle`/`defines` form. Put your config under `${run.dir}` so it is run evidence.

It emits, under `--output-dir`:

- `descriptors/<slug>/` -- the KMD, UED, UDD, UHD, UMD and KDP JSONs.
- `packs/<Class>Native.cpp` -- your hook bodies, as stubs marked `FILL THIS OUT`.
- `tests/Test<Name>Packs.cpp` -- **complete, not a stub**; it exercises
  `discoverDescriptorSets()` and the provider's real loading path.
- `tests/Test<Name>Matchers.cpp` -- a stub.
- `fragments/` -- six CMake/C++ text files that are *splice instructions*, not shipped
  files. You apply each one to its real consumer by hand.

**A fragment file is not evidence that its splice was applied.** All six consumers, and
both halves of the registration:

| Fragment | Consumer |
|---|---|
| `cmake_descriptor_files.txt` | `${vars.provider_dir}/CMakeLists.txt`, the `HIPDNN_DESCRIPTOR_FILES` list |
| `cmake_ingestor_kernels.txt` | `${vars.ingestor_dir}/IngestorKernels.cmake`, `HIPDNN_INGESTOR_PACK_KERNELS` |
| `cmake_target_sources.txt` | `${vars.ingestor_dir}/CMakeLists.txt`, `target_sources` |
| `cmake_test_sources.txt` | the provider's engine-test `CMakeLists.txt`, `target_sources` |
| `ingestor_packs.hpp.txt` | `${vars.ingestor_dir}/IngestorPacks.hpp`, the declaration |
| `ingestor_packs.cpp.txt` | `${vars.ingestor_dir}/IngestorPacks.cpp`, the `s_packs` row |

The last two are both required and neither is optional. A pack declared in the header but
missing from the `s_packs` table links fine into the plugin and silently vanishes from
the statically-linked unit-test binary, with no error either way.

# The seam: write launch() first, then check it against the kernel

`${vars.authoring_file}` names each entry point's parameters, in order, with types and
roles. Your `launch()` passes exactly those, in exactly that order.

**Wrong arity or wrong order is diagnosed nowhere.** hipRTC compiles the kernel,
`getKernel(entry_point)` resolves it, and `hipModuleLaunchKernel` reads one pointer per
parameter the *kernel* declared. A short argument list reads whatever is next in memory;
two same-typed pointers swapped is a wrong answer with no diagnostic. The orchestrator
compares your `launch_arg_order` against the kernel's parameter names, element by
element, and that comparison is the only place in the toolchain where this is caught.

Two other seam decisions, both yours:

- **Handler-supplied vs descriptor-bound defines.** Your handler adds the defines every
  kernel it prepares needs, built by hand next to the compile call. Descriptor-bound
  defines are appended after the handler's and overwrite them by key. Anything derived,
  conditional or computed is the handler's, because the descriptor's substituter does
  literal replacement and nothing else. Do not grow the substituter to avoid a handler
  change.
- **Geometry is the handler's, so the guard is the kernel's.** The authoring contract
  states `launch_geometry.guards_own_bounds`; honour it. If `prepare()` computes the grid
  from the shape, the final block is partially populated and the kernel must guard
  itself.

Use `buildIngestorKernelCode` in `prepare()` rather than calling the compiler directly.
It is the one place source loading and path containment are handled for every
`kernel_source.kind`. `ConvNative.cpp` calls
`_kernelCompiler.compile(kernel.source.sourceFile, options)` itself, which serves
`embedded_source` only -- which is why a `hiprtc_file` descriptor under the conv pack
throws at plan-build time no matter how correct the descriptor is. This run's source kind
is `${inputs.kernel_source_kind}`.

# Registration and graphs are half the job

**Register with the testing system.** Add an `add_external_integration_test_target` entry
beside the provider's existing ones in `${vars.provider_dir}/src/CMakeLists.txt`. Spell its
TARGET_NAME with the CMake PROJECT_NAME variable exactly as its siblings do, so the
registered CTest name comes out as
`${steps.identity.outputs.external_test_target}`. Supply `ENGINE_NAME`, a `TEST_CONFIG`
TOML at `${steps.identity.outputs.engine_toml}`, and the `INSTALL_SUBDIR` /
`INSTALL_TEST_FILE` / `TEST_CATEGORIES_YAML` arguments its siblings use -- an engine
registered only in the build tree is not exercisable against an install, which is where a
customer meets it. `dnn-providers/integration-tests/README.md` documents every argument.

**Add graphs as bundles, not C++.** `BUILD_CPP_GRAPH_TESTS` is OFF by default and CMake
enforces bundles for new graph-verification coverage. Use the dedup-aware importer:

```
${steps.generator_venv.outputs.python} \
    ${vars.repo_root}/dnn-providers/integration-tests/migration-scripts/import_graph.py \
    --graph <graph.json> \
    --bundle-dir ${vars.repo_root}/dnn-providers/integration-tests/integration-test-bundles/
```

It prints the generated case id to stderr. **Those ids are what you report**, and the
orchestrator runs exactly them against your engine and requires each to pass. Cover what
your pack admits, from the authoring contract's envelope: each dtype, each layout, the
boundary shapes, and at least one case per specialization axis so more than one of your
candidates is actually selected.

# Scope

You may add and edit under:

- `${vars.ingestor_dir}` and the provider's engine-test tree
- `${vars.provider_dir}/CMakeLists.txt` and `${vars.provider_dir}/src/CMakeLists.txt`
- `${vars.provider_dir}/config/` -- your own new TOML only
- the bundle tree -- **new cases only**

You may **not** modify or delete: the integration test suite sources, its CMake, its
category YAMLs, any existing bundle case, any other engine's TOML, `HIP_MLOPS_ENGINE`,
the `Pointwise`/`ConvFwd` packs and their descriptors, or the generator's own code and
templates. All of it is hashed before and after you run. Adding a bundle case is allowed;
appending a case to an existing `sweep.json` is allowed; **editing a case that is already
there is not**, and it is separately detected and separately reported. If a case looks
wrong, say so in `summary` and leave it alone.

# Build and test what you write

You have cmake, ninja, clang and the ROCm runtime on your PATH, and the build tree is
at `${vars.ingestor_build_dir}`. Use them. An edit you have not compiled is a guess,
and a round that spends an agent session producing a guess and then fails on a typo is
the most expensive way to find a typo.

The cheap loop, in the order that pays:

```
cmake --build ${vars.ingestor_build_dir} --parallel 16 --target <the one target you touched>
${vars.build_bin}/hip_kernel_provider_tests --gtest_filter=<YourSuite>.*
```

Build the target you changed, not the world, and run the focused suite rather than the
whole matrix. A full rebuild-and-run cycle inside this step is time not spent on the
pack, and the orchestrator is going to do the authoritative pass immediately after you
finish anyway.

That last point is the one to keep hold of: **your build is for catching your own
mistakes, not for producing the verdict.** The orchestrator reconfigures, rebuilds,
installs and runs every gate against the INSTALL tree after you return, and its result
is the one the loop reads. A green run in your session and a red gate afterwards is a
difference worth understanding rather than arguing with -- it usually means you tested
the build tree and the gate tested the install.

# What the orchestrator checks after you finish

- `generate.py --check-placeholders --emitted-root ${vars.ingestor_dir} --emitted-root ${vars.provider_tests_dir}`
  must exit 0. It derives the file set from your config, so it fails both when a hook body
  still says `FILL THIS OUT` **and** when an emitted file was never spliced anywhere.
- Your contract is checked against the tree: every declared hook symbol must appear in
  your native source, both registration splices must be present, the TOML and every
  descriptor file must exist, and `launch_arg_order` must equal the kernel's parameter
  order.
- `hipdnn_validate_descriptors <descriptor dir> --expect-engine ${inputs.engine_name} --json`
  must find your engine.
- `hip_kernel_provider_tests --gtest_filter=${steps.identity.outputs.census_filter}` must
  run at least one case and pass, in a fresh process, because registration and discovery
  are memoized. That filter spans every generated test source your engine owns, and the
  gate also requires at least one suite per source: a generated test file that contributes
  no suite has not run, and a test that cannot run cannot pass.
- `ctest -N` from the **install** prefix must list exactly one entry named
  `${steps.identity.outputs.external_test_target}`, and its command line must name your
  engine.
- The shared suite, pinned to your engine with `--verification-mode gpu` and filtered to
  exactly your reported bundle case ids, must pass at least as many cases as you declared,
  with zero failures and fewer skips than selections.
- The same suite naming an engine that does not exist must **fail** with
  `Error: Engine '<name>' is not loaded.` and exit 1. That is the control that makes the
  positive run mean anything: because unservable cases skip, a run in which your engine
  was never loaded at all otherwise looks exactly like a run in which it served
  everything.

A green run is not coverage. Report the passed count and the case names.

# Output contract

Write a JSON object to exactly this path, and nothing else that matters:

    ${step.result_file}

```json
{
  "engine_name": "${inputs.engine_name}",
  "pack_class": "${steps.identity.outputs.pack_class}",
  "descriptor_slug": "${steps.identity.outputs.descriptor_slug}",
  "generator_config": "<absolute path to the config you wrote>",
  "generated_dir": "<absolute path to the generator's output directory>",
  "kernel_source_kind": "${inputs.kernel_source_kind}",
  "native_file": "${steps.identity.outputs.native_file}",
  "hooks": [
    {"role": "graph_match|graph_criterion|kernel_match|score|workspace|dispatch",
     "symbol": "<the C++ symbol>", "implemented": true, "reason_if_declined": ""}
  ],
  "registration": {
    "symbol_scope_function": "${steps.identity.outputs.register_symbol}",
    "packs_table_row": "<the s_packs row you added, verbatim>",
    "header_decl": true
  },
  "cmake_splices": [
    {"fragment": "cmake_descriptor_files.txt", "applied_to": "<absolute path of the file you edited>"}
  ],
  "descriptor_files": ["<absolute path of every descriptor JSON now in the provider tree>"],
  "external_test_target": "${steps.identity.outputs.external_test_target}",
  "engine_toml": "${steps.identity.outputs.engine_toml}",
  "bundle_case_ids": ["<every case id import_graph.py printed>"],
  "bundle_gtest_filter": "<a gtest filter selecting exactly those ids, colon-separated>",
  "entry_point_launched": "<the entry point your handler launches>",
  "launch_arg_order": ["<kernel parameter names, in the order launch() passes them>"],
  "consumed_authoring_sha256": "<sha256 of ${vars.authoring_file}>",
  "changed_files": ["<absolute path of every file you created or modified>"],
  "addressed_feedback": ["<each issue from the feedback file you fixed, and how; [] on the first attempt>"],
  "summary": "<what this round changed and why>"
}
```

`hooks` must contain at least `graph_match` and `dispatch`, both implemented. A hook you
decline needs a `reason_if_declined`; a hook you declare implemented must have its symbol
present in the native source. `changed_files` and `bundle_case_ids` may not be empty --
the engine directory is re-hashed, so a round that reports work but wrote nothing is
failed on the evidence rather than on the report.
