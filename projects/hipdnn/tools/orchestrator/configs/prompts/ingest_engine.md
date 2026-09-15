You are running the production half for an ingestor engine that now exists: proving it
from the **installed** tree, accounting for a corpus, and stating exactly what the
evidence does and does not establish.

Execute the skill at `${vars.skills_dir}/hipdnn-ingestor-engine/SKILL.md`. Its `RUNBOOK.md`
owns the ordered workflow. **The create path is not yours** -- stages 1 through 3 of that
runbook are already done: the pack, its symbols, its descriptors, its registration and its
bundle graphs were landed and gated in the previous stage. You own its stages **4, 5 and
7**: build/pack/install and the host boundary, baseline device proof from the
installation, and the final corpus proof with runtime reconciliation. Stage 6, tuning, is
gated on `tune` below.

- Engine name:       ${inputs.engine_name}
- Install prefix:    ${vars.ingestor_install_dir}
- Build directory:   ${vars.ingestor_build_dir}
- Descriptor dir:    ${steps.identity.outputs.descriptor_dir}
- CTest target:      ${steps.identity.outputs.external_test_target}
- Census suite:      ${steps.identity.outputs.census_suite}
- Target arch:       ${inputs.arch}
- Corpus:            ${inputs.corpus_dir}
- Dialect:           direct_load / ${inputs.kernel_source_kind}
- Tuning:            ${inputs.tune}
- Attempt:           ${loop.attempt} of ${loop.max_iterations}
- Operator notes:    ${inputs.notes}

# Read these first

- **`${vars.integration_file}`** -- what the previous stage landed: the hooks, the
  descriptors, the registration, the bundle case ids and the launch ABI.
- **`${vars.authoring_file}`** -- the kernel's admitted-shape envelope, which is what
  predicts which corpus graphs should be served and which should decline.
- **`${loop.feedback_path}`** -- one section per failed round, naming the failing step and
  its log directory. Open that directory. The corpus join report, with the specific
  unaccounted graph names, is in it.

# The corpus is the job, and the denominator is the directory

`${inputs.corpus_dir}` is the corpus. **Every file under it must end this round with
exactly one attributable outcome.** The orchestrator globs that directory itself and joins
your report against it by resolved path. A graph you do not mention is `unaccounted` and
fails the round; a graph you mention that is not in the directory is `unjoined` and also
fails it; naming one twice is `duplicates`.

The five outcomes, and they are not interchangeable:

| Outcome | Means |
|---|---|
| `served` | The engine ran the graph. Name the engine actually observed, not the one you expected |
| `declined` | The engine's matcher refused it, with the reason. This is a **result**, not a gap |
| `error` | It was attempted and something went wrong. Blocks the gate |
| `missing` | No timing row and no decline reason -- the outcome is not known. Blocks the gate |
| `ambiguous` | More than one outcome joins to it. Blocks the gate |

A missing timing row is not a decline reason. Do not reconstruct a runtime reason from
offline policy: if the evidence does not carry it, the outcome is `missing`, and saying so
is the correct answer. `${vars.skills_dir}/hipdnn-ingestor-engine/workloads.md` owns the
ledger's identity and join rules.

A new pack's served slice will be thin. That is expected and is not something to fix by
widening a matcher: report the slice honestly.

# Prove it from the installation, not the build tree

The install at `${vars.ingestor_install_dir}` is current as of the previous stage. If you
change anything that affects it, rebuild and reinstall -- changed installed artifacts
invalidate the previous stage's evidence, and an old install certifies nothing.

Two traps the runbook names explicitly:

- **The provider's installed CTest root is `${vars.ingestor_ctest_root}`, not the install
  prefix.** Pointing `ctest` at the prefix finds nothing and exits 0.
- **`--output-on-failure` hides passing suites' case counts, and an all-skip suite still
  reports CTest PASS.** Run verbose and record selected, served, skipped and failed
  counts with the observed reasons.

Resolve the exact UED name to the engine id from the installation. A prefix match or a
registry listing is not dispatch attribution.

# rocKE is ON in this build, and that is a question you must answer

This flow configures with `HIPKERNELPROVIDER_ENABLE_ROCKE=ON` alongside
`HIPDNN_ENABLE_KERNEL_INGESTOR=ON`. State what that did for this engine. The honest answer
for a direct-load HIP pack is normally "nothing": rocKE is always the *packaged* dialect,
`kind: rocke` descriptors are lowered by `hkp_pack` into `kind: kpack`, and
`ROCKE_BUILDER` is stubbed and fatal in
`${vars.ingestor_dir}/IngestorKernelCode.hpp`. Note also that
`HIPKERNELPROVIDER_ENABLE_ROCKE` is **not** the production producer switch --
`HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE` is, it additionally requires
`ROCKE_WHEEL_DIR` and a resolvable comgr, and it is off here.

Whatever the answer is, put it in `rocke.why`. An unexamined build switch is how a run
acquires a capability nobody checked, and "it was on" is not a finding.

The packaged census does not apply either: it covers **packaged** engines only, keys off
`HKP_CENSUS_TEST_SUITES`, and this is a direct-load engine whose inventory evidence is its
own generated suite `${steps.identity.outputs.census_suite}` against the arch-independent
descriptor tree. Say that rather than reporting a census that was never registered.

# Tuning

`tune` is `${inputs.tune}`.

When it is `off`, do not run knob sweeps and do not regenerate the selection. Record
`tuning.performed: false` with `tuning.why` naming the switch. Tuning is the fourth arrow
of `graph -> kernel -> integrate -> optimize`; a correct, slow, fully-accounted engine is
this flow's success, and a speed claim this run did not measure is its failure.

When it is `on`, follow runbook stage 6: propose bounded candidates, keep separate install
and output trees for every arm, never mutate the baseline and call it a comparison, then
repeat stages 3-5 with the final config and a new empty generation destination. An
isolation arm does not certify the regenerated shipping set.

# Scope

Same boundaries as the previous stage. You may work under the ingestor engine tree, the
provider's CMake and your own engine's TOML, and you may add bundle cases. You may not
modify the integration test suite, its category YAMLs, any existing bundle case, another
engine's TOML, `HIP_MLOPS_ENGINE`, the `Pointwise`/`ConvFwd` packs, or the generator's own
code and templates. All of it is hashed.

You may build and install in this stage -- unlike the previous one -- because regenerating
or tuning obliges a rebuild before any gate can mean anything. The orchestrator re-runs
the installed-tree gates after you finish regardless.

# What the orchestrator checks after you finish

- Your contract describes a real installation: the prefix and the final descriptor root
  both exist and the root holds descriptors, and `rocke` carries a stated disposition.
- `device_probe.py --mode installed` against `${vars.ingestor_install_dir}`. A missing or
  invisible installation fails it even when early feasibility passed.
- `hipdnn_validate_descriptors --expect-engine ${inputs.engine_name} --json` against the
  current tree.
- The operation-wide suite, pinned to your engine with `--verification-mode gpu`, from the
  install: zero failures **and** more than zero passes. A large skip count is expected; a
  zero pass count is what "every selected case skipped" looks like, and the exit code
  reports it as success.
- The corpus join described above.

# Output contract

Write a JSON object to exactly this path, and nothing else that matters:

    ${step.result_file}

```json
{
  "install_prefix": "${vars.ingestor_install_dir}",
  "final_descriptor_root": "<absolute path to the descriptor tree the gates must read>",
  "engine_name": "${inputs.engine_name}",
  "stage_completed": 7,
  "dialect": "direct_load",
  "kernel_source_kind": "${inputs.kernel_source_kind}",
  "rocke": {
    "enabled_in_build": true,
    "used": false,
    "why": "<what HIPKERNELPROVIDER_ENABLE_ROCKE=ON did for this engine, and why>"
  },
  "corpus": {
    "root": "${inputs.corpus_dir}",
    "inputs": [
      {"graph": "<absolute path>", "outcome": "served|declined|error|missing|ambiguous",
       "engine": "<the engine actually observed, or \"\">",
       "reason": "<the runtime reason, verbatim where one exists>"}
    ]
  },
  "candidates_selected": ["<every kernel candidate a pass was actually served by>"],
  "tuning": {"performed": false, "why": "<why, naming the tune switch>"},
  "suite": {"selected": 0, "passed": 0, "skipped": 0, "failed": 0,
            "case_names": ["<the cases that passed>"]},
  "does_not_prove": ["<architectures not run, dtypes not covered, shapes declined, performance silence>"],
  "changed_files": ["<absolute path of every file you created or modified; [] is legitimate>"],
  "summary": "<what this round established>"
}
```

`corpus.inputs` must have one entry per file under `${inputs.corpus_dir}`.
`candidates_selected` may not be empty: if every pass came from one kernel, your other
variants are unexercised no matter how many cases ran, and naming them is how that becomes
visible. `does_not_prove` may not be empty.
