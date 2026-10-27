# Engine onboarding

This page checks that an engine on the user's branch can be trained, then hands off to
[generate.md](generate.md) for a smoke round. It does not create engines: for a new engine,
start with `hipdnn-ingestor-engine`. Corpus operation and facet details are in
[corpus-facets.md](corpus-facets.md).

## What kind of engine it is

Read it from the checkout (`Grep`/`Glob`/`Read`). An engine can be trained only if a model of
it is read through a binding; [generate.md](generate.md), "What the engine is", defines the
two kinds. In short:

- **Descriptor engine**: a UED built into the plugin (not under `tests`, `test_descriptors`,
  `fixtures`, `integration_tests` or `descriptor-packaging/examples/`) whose `name` is the
  engine; two that claim one name are an error. Its directory is the descriptor directory;
  its provider is the `dnn-providers/<provider>` it lives under. Its role maps are its
  bindings (`sort_kernel_catalog` is L2, `predict_engine` is L1; L2 is trained first), keyed
  by gfx arch or `default`; its archs are the role-map gfx keys, else the `arch` lists of its
  KDPs. Its corpus operation is what its graph matcher (`graph_match.native`) accepts; by the
  hip-kernel-provider's naming convention the symbol's middle segment names it, arch prefix
  dropped (`hipkernel.gfx950_conv_fwd.graph_match` serves `conv_fwd`), and where the name
  says nothing, a census run does ([corpus-facets.md](corpus-facets.md)).
- **Opaque engine**: a `HIPDNN_REGISTER_ENGINE` name with no UED, implemented in its
  provider's code (`grep -rn <NAME> dnn-providers`). It is L1 only. It can be trained when it
  overrides `getPrediction` and binds UHD ids through the plugin SDK (`EngineModelBinding` /
  `resolveDeclaredEnginePredictions`): per arch, or per metric under `default` for every
  arch. Those ids are what a model must be trained under. With no binding there is nothing
  to train for: add one first (declare the ids, override `getPrediction`). Its archs are the
  ones its ids are declared for, and its operations those of the plan builders it registers,
  whose `isApplicable` checks also say which devices it runs on.
- **Closed shape set**: listed with `coverage: pack` in
  `projects/hipdnn/tools/corpus_gen/operations/engines.json`. It serves exactly its pack's
  shapes; it is trained on all of them and judged by recall. A new pack-coverage engine adds
  its entry there on its branch.
- **Engine id**: the FNV-1a hash of the name, printed in hex by `hipdnn_list_engines`;
  `uhd_gen --engine-id` takes it as a signed decimal ([data.md](data.md)).

## What the branch must have

1. **A binding.** An opaque engine: the declared ids and `getPrediction` above. A descriptor
   engine: a UED whose `name` is the engine name. It declares `id`, `graph_match`, the role maps
   (`"sort_kernel_catalog": { "gfx942": "<uhd uuid>" }`, or keyed `default`), `metadata`, and
   `knobs`. A role map value is one id or a list (one UHD per metric). A UED that binds its
   model by a scalar `"heuristic": X` key is refused by the loader (the engine does not
   register): rewrite it as `"sort_kernel_catalog": {"default": X}`, and X's UHD `kind` /
   `payload` as `"adapter": "native"`, `"objective"`, `"native": {"symbol": ...}`.
2. **A graph matcher** (the UED's `graph_match.native` symbol, registered by the provider),
   any **UMDs** its kernels need, and a **KMD** whose fields cover every knob. Knob
   parity is two subset checks at load
   (`projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp`):
   every `$kernel.*` axis a model reads must be in the UED `knobs` **and** in the KMD fields;
   otherwise the loader drops the model and ranks by priority, then id.
3. **Registration.** After a full build, `hipdnn_list_engines --plugin-dir <abs plugin dir>`
   prints the engine name. Descriptors reach the plugin directory through the `ALL` descriptor
   staging and packaging targets; a build restricted to a target list can register fewer
   engines than exist. The engine's packs must be staged through the normal packaging path,
   under the descriptor root its build uses.
4. **Applicability.** The engine serves the corpus graphs: corpus_gen offers every proposed
   problem to it on the GPU, and keeps what it serves.
5. **Correctness verdicts.** Every timed candidate carries `numerically_valid` (true, false or
   null) and a `validation` string; a bench path with no correctness check fails collection.
   A `false` row blocks installation until the kernel or UMD changes.
6. **Work counts for `tflops`.** The `tflops` label is `graph.flops / (avgTimeMs × 1e9)` from
   the engine's published `graph.flops`. An engine that publishes none trains `time`, or a
   metric-less ranker.
7. **Everything on the branch.** The build compiles the checkout alone: a pack the engine
   needs must be merged into it.
8. **The UHD tools.** The branch carries `projects/hipdnn/tools/corpus_gen/operations/` and
   `projects/hipdnn/tools/uhd_gen/`; an engine branch cut without them merges a branch that
   has them.

## Workflow

1. Identify the engine (above): kind, roles, metrics, archs, the operation.
2. Check the device's arch is among the engine's archs ([generate.md](generate.md), step 0).
   Build ([data.md](data.md), "The tools"), then
   `<build>/bin/hipdnn_list_engines --plugin-dir <abs plugin dir>`: the engine and its id.
3. A descriptor engine: validate its descriptors with the loader's own validation:
   ```bash
   <build>/bin/hipdnn_validate_descriptors <per-arch root> --expect-engine <engine>
   ```
   (`projects/hipdnn/tools/IngestorGenerator/ValidateDescriptors.cpp`): undeclared knobs,
   duplicate knob tuples, dangling uuids, bad arch keys, and a real load of every role-bound
   UHD. Every `VIOLATION:` line is the branch's to fix.
4. Choose the corpus operation and `--keep` facets ([corpus-facets.md](corpus-facets.md)).
   The operation follows what the engine serves (its graph matcher, or its plan builders),
   not its name.
5. A smoke shape set ([generate.md](generate.md) step 3 with `--count 200`): exit 0 with graphs
   admitted on every operation.
6. A smoke round through [generate.md](generate.md): collect, train and score per role, then
   Stage V ([validate.md](validate.md)). Walk every failure with [triage.md](triage.md):
   - **build**: the first compiler error with file:line;
   - **registration**: engine found and its id; validator findings;
   - **applicability**: graphs proposed vs admitted;
   - **kernels**: per kernel id + knob tuple, failures by reason (fault, timeout on two GPUs,
     numerically invalid, `config_not_applicable`);
   - **installation**: invalid markers, knob parity, metric/objective.

## Decision rules

- Fix what the round attributes to the branch, rebuild, and rerun the failing step.
- Every kernel `config_not_applicable` usually means a descriptor layout problem (the tree is
  one engine's folder, or `kpack/` is not at the per-arch root), not the kernels.
- A kernel that fails on one GPU only is the device; the same kernel id + knob tuple failing
  on two GPUs is the kernel.
- A smoke round passes when graphs are admitted and collected, the model trains and is
  scored held out, and Stage V's identity, available, metric and selector-revision gates
  pass ([generate.md](generate.md), "Tier"). Then move to `standard`.

## Report

The engine as identified (kind, roles, metrics per role, archs and where they were read, the
operation and `--keep` facets with the reason), the device, the checklist above with
pass/fail per item, the validator's findings, and the smoke round's outcome.

## Common failures

| Symptom | Fix |
|---|---|
| no UED with that `name` built into the plugin | the UED is under a test tree, or its `name` differs (case included) |
| UED with a scalar `heuristic` key | port the UED and its UHD to role maps and `adapter` (above) |
| opaque engine binds no UHD id (`Engine does not provide predictions`) | add a binding in the provider first (above) |
| engine not listed after the build | packs not staged through the packaging targets, not built for this arch, or a relative `--plugin-dir` |
| corpus_gen exit 2, zero admitted graphs | the engine does not run on this arch (`HIPDNN_LOG_LEVEL=info` shows its refusal), wrong operation, or over-narrow `--keep`; re-check the graph matcher |
| corpus_gen exit 3 | fewer graphs than `--count` without the engine shown to serve no more: widen facets or accept the recorded shortfall |
| collection refuses the bench response | the engine's bench path emits no `numerically_valid` verdict |
| `tflops` requested, no work count | publish `graph.flops`, or train `time` |
