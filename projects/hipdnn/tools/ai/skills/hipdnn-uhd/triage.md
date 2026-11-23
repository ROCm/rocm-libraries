# Triage

Every failure lands in one of three classes, and the class decides who acts:

| Class | Who acts | Typical evidence |
|---|---|---|
| **Device** | rerun the graph, or use another GPU of the device | GPU fault or hang with no kernel attributable; it does not repeat |
| **Engine** | the engine developer changes the branch | the same kernel id and knob tuple fails every time; compile errors; descriptors the loader refuses; nothing admitted |
| **Tools** | fix the command, or report the tool bug | argparse errors, Python tracebacks from `uhd_gen`, a reused output directory |

A fault attributable to one kernel id is **Engine** even when it looks like a GPU fault: rerun
that graph on another GPU of the device (`HIP_VISIBLE_DEVICES`); if the same kernel fails
there too, it is the kernel.

## Where the evidence is

- `hipdnn_corpus_gen`: stderr (per operation: engine queries, problems, constraint
  admissions, regimes short) and `<set>/<operation>/manifest.json` `reports`.
- A collection or training run (`uhd_gen generate`): `generation_manifest.json`
  (`failed_graphs`, `commands` with each call's argv, return code and descriptor
  environment) and the per-graph bench logs beside it.
- Validation: the rows document (`validate_l1.py` / `validate_l2.py`) and `decide.py`'s
  report, which names every failing gate's problems.
- The runtime: `HIPDNN_LOG_LEVEL=info` on a single `hipdnn_bench --graph <file>` call
  reproduces any one graph; on `hipdnn_corpus_gen --count 1` it shows why the engine refuses
  the problems it is offered.

## The engine's evidence (stop; the branch must change)

| Evidence | What the developer changes |
|---|---|
| a compile error in the build | fix it on the branch |
| engine missing from `hipdnn_list_engines`, or `is not registered by any loaded plugin` | the UED `name` is wrong, or its descriptors are not built and staged for this arch (see [onboarding.md](onboarding.md)) |
| `VIOLATION:` from `hipdnn_validate_descriptors <per-arch root> --expect-engine <engine>`; the loader's `the model is not used and kernels rank by priority, then id` or `the model is not used -- ranking degrades to static_order` | fix the UED/KMD/UMD (undeclared knob, dangling uuid, bad arch key, duplicate tuple), then retrain |
| corpus_gen `Generated 0 problems` / exit 2 | the engine does not run on this arch, the wrong operation or `--keep` facets, or its graph matcher or UMDs admit nothing on this arch; rerun one operation with `HIPDNN_LOG_LEVEL=info --count 1` into a scratch directory and quote the engine's refusal (its plan builder's log line) |
| `deterministic catalog: all N problem(s)…` | every problem has one candidate: nothing for an L2 to rank; train the L1 |
| `the benchmark produced no successful valid timings` | every kernel failed to build or run, or was numerically invalid |
| rows with `numerically_valid: false`; promote refuses an invalid marker | fix the kernel or narrow the UMD; retraining cannot fix it |
| `timing response must carry a numerical-validation verdict` | the engine's bench path performs no correctness check |
| `the sweep did not time exactly the catalog it reported` | enumeration and sweep disagree for this engine |
| the same kernel id + knob tuple fails on two GPUs | fix or withdraw that kernel |
| a model ranks on a `$kernel.*` axis missing from the UED `knobs` or the KMD fields | declare the knob; the loader otherwise drops the model silently |

## The device's evidence (retry)

| Evidence | Action |
|---|---|
| `Memory access fault by GPU node`, `HSA_STATUS_ERROR_*`, `hipErrorLaunchFailure` / `hipErrorIllegalAddress`, a GPU reset, with no kernel attributable | rerun the graph; if it follows the GPU, use another |
| a `hipdnn_bench` call that never returns | the validation scripts record it (`timed_out`) and continue; rerun that graph alone on another GPU |
| out-of-memory kill | close other GPU work, or shard the collection |
| `Could not query device string: Stream device query failed`, `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, `rocminfo` lists no gfx agent; corpus_gen exits 2 with no engine refusal | the GPU is not visible to the process: in a container, pass `/dev/kfd`, `/dev/dri` and the `video` and `render` groups ([data.md](data.md)) |

## The tools' evidence (fix the command)

| Evidence | Fix |
|---|---|
| `: error: argument …`, `unrecognized arguments`, `the following arguments are required` | the command does not match this checkout's `uhd_gen`; check `--help` |
| `--output-dir must not exist; generation never overwrites a previous run` | write each run to a new directory |
| `generate requires a true problem holdout: 0 < --eval-fraction < 1` | pass a fraction in range, `--eval-benchmarks`, or `--recall` for a closed shape set |
| corpus_gen refuses a pack-coverage engine | pass `--kdp-root` |
| every graph fails collection, the engine's load error naming a kernel file or archive under `<plugin dir>/arch_content/` | the build staged no kernels for this arch ([data.md](data.md), "Kernel files") |
| a Python traceback from `uhd_gen` | a tool bug; report it with the command and the checkout's SHA |

## Harmless noise

`Plugin path 'hipdnn_plugins/heuristics/' is invalid` means no heuristic plugins are
installed, and needs no action. corpus_gen's `no declared archetype was admitted for ...;
that combination is exploration only` is not a failure either: the engine served none of the
operation's named workloads for that combination, which the search still explores. Report it
when the engine is meant to serve those workloads.

## Silent failures: green runs with useless results

| Symptom | Likely cause |
|---|---|
| every engine declines every graph | a CMake reconfigure emptied `lib/hipdnn_plugins/engines/arch_content/`; rebuild the descriptor staging targets |
| fewer engines registered than expected | a build restricted to a target list skipped descriptor staging |
| every config `config_not_applicable` | one engine's folder handed in as the tree instead of the per-arch root that owns `kpack/` |
| validation passes but the model has no effect | a benchmark winner record outranks the model: run with an empty `HIPDNN_CACHE_DIR` |
| an opaque model installs but is never read | it was trained under an id other than the one the provider declares |
| a candidate dropped into `HIPDNN_DESCRIPTOR_RUNTIME_DIR` "validates" | the runtime keeps the build's document for an id the build defines; validate with `HIPDNN_DESCRIPTOR_DIR` on a copy ([validate.md](validate.md)) |
| the model loads but scores nothing (static order) | a dotless feature name, or a `$kernel.*` axis missing from the UED knobs / KMD fields |
| regret suspiciously low | a holdout leak: `eval_problem_keys` intersect `training_problem_keys` |
| the model splits on `device.*` | device identity used as a feature |
| generate trained on fewer rows than collected | the dataset holds several revisions or L2s; a run reads one |
| held-out score jumps between rounds | different slices, or a drawn `holdout` instead of fixed `eval_benchmarks`; compare on the same rows |
| a new set returns far fewer graphs than asked with exclusions | the corpus_gen in the build predates walking through excluded configurations; rebuild it |

## Report

Per failure: the step, the class, the evidence line quoted, the graph and kernel when there is
one, and the action taken or the change the engine developer has to make.
