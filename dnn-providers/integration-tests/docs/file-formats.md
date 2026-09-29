# Integration Test File Formats

Every file a developer reads or writes when working with the cross-provider
integration suite: what it is, where it lives, who writes it, and the rules the
harness enforces when it reads it.

- To **run** the suite and read its output, see [Running the Tests](running-tests.md).
- To **add** a bundle or **update** claims, see [Adding Tests and Updating Claims](adding-tests.md).

## Map

```
dnn-providers/integration-tests/
  integration-test-bundles/                 # the bundle tree (the test data)
    {tier}/                                 # quick | standard | comprehensive | full
      {Op}/{Layout}/{DataType}/{Name}/      # a single-graph bundle
        {Name}.json                         #   graph
        {Name}.meta.json                    #   metadata sidecar         (required)
        {Name}.support.json                 #   support-claim sidecar    (required)
        {Name}.tensors.dvc                  #   golden-data pointer      (optional)
        {Name}.tensor<uid>.bin              #   golden tensors, via DVC  (not in git)
      {Op}/{Topology}/                      # a template-sweep bundle
        graph.template.json                 #   topology with ${case.*} placeholders
        sweep.json                          #   case matrix (+ required per-case metadata)
        support.json                        #   support-claim sidecar    (required)
        golden/{CaseId}/tensors.dvc         #   golden-data pointer      (optional, per case)
        golden/{CaseId}/tensor<uid>.bin     #   golden tensors, via DVC  (not in git)
  test_categories.yaml                      # CTest tiers for this project's own binaries
  test_categories_external.yaml             # CTest tiers for pre-registered, non-GTest tests

dnn-providers/<provider>/
  config/<ENGINE_NAME>.toml                 # per-engine tolerances, validator overrides, skips
  test_categories_integration.yaml          # CTest tiers for the lane (hip-kernel-provider: one
                                            #   <ENGINE_NAME>_test_categories_integration.yaml per engine)
```

Terms (full list in the [index](README.md#terms)):

- A **bundle** is one graph plus everything that describes it. It is either a
  **single-graph bundle** (one `{Name}.json`) or a **sweep** (one
  `graph.template.json` expanded once per case in `sweep.json`). "Bundle" covers
  both.
- A **case** is one registered test. A single-graph bundle is one case; a sweep is
  one case per `cases[]` entry.

Every bundle carries its graph, its **metadata**, and a **support-claim
sidecar**. Golden data is the only optional part.

## Discovery and test names

The harness walks the bundle root (`--golden-data-dir`, default
`<exe>/../lib/integration-test-bundles/`) and registers:

- **Single-graph bundles:** every `.json` file, except `graph.template.json`,
  `sweep.json`, companion files whose last dotted segment is `meta` or `support`
  (`Small.meta.json`, `Small.support.json`), and anything under a sweep root.
  Other dotted names still count as graphs (`resnet50.v2.json`).
- **Sweeps:** every directory that contains *both* `graph.template.json` and
  `sweep.json`. Each `cases[].id` becomes one test.

GTest names come from the path, not from anything inside the files. Each path
segment is sanitized (any character other than `[A-Za-z0-9_]` becomes `_`):

| Bundle kind | Suite | Test | Example full name |
|---|---|---|---|
| Single graph | relative directory, segments joined by `_` | file stem | `quick_BatchnormFwdInference_nchw_fp32_Small.Small` |
| Sweep | sweep directory, segments joined by `_` | `cases[].id` | `quick_RMSNorm_Default.2_1_1_1_bfp16_nchw_fa8ec9` |

So the tier directory is always the suite prefix (`quick_*`, `standard_*`, …),
which is what the tier YAML files match on. Two bundles that sanitize to the same
full name abort registration with `Bundle name collision`, naming both paths.
Keep every bundle under a tier directory: a graph placed directly in the data
root registers under the root folder's name and matches no tier.

Discovery imposes no folder schema beyond that; the
`{tier}/{Op}/{Layout}/{DataType}/{Name}` convention below is a convention, kept
by the tools and the verifier, not by registration.

## Single-graph bundle — `{Name}.json`

One concrete graph: the hipDNN frontend graph serialized to JSON. The top level
carries graph-wide data types and two arrays:

- `nodes` — each with a `type` (for example `RMSNormAttributes`,
  `SdpaAttributes`), `inputs` / `outputs` mapping each port to a tensor `uid`
  (`null` for an unused optional port), and op-specific `attributes`.
- `tensors` — each with `uid`, `name`, `dims`, `strides`, `data_type`,
  `virtual`, and optionally a baked scalar `value` / `value_type` or
  `is_runtime_pass_by_value`.

Excerpt from `integration-test-bundles/quick/SdpaFwd/bhsd/fp16/hd128_causal_mha/Small/Small.json`:

```json
{
  "nodes": [
    {
      "type": "SdpaAttributes",
      "inputs":  { "q_tensor_uid": 0, "k_tensor_uid": 1, "v_tensor_uid": 2, "attn_mask_tensor_uid": null },
      "outputs": { "o_tensor_uid": 3, "stats_tensor_uid": null },
      "attributes": { "causal_mask": true, "attn_scale_value": 0.08838834764831843 }
    }
  ],
  "tensors": [
    { "uid": 0, "name": "Q", "dims": [2, 32, 2048, 128], "strides": [8388608, 262144, 128, 1], "data_type": "half", "virtual": false }
  ]
}
```

**Written by:** `--capture-bundles` (from a C++ graph test), a per-op golden-data
generator under `reference-data-scripts/`, or `import_graph.py` when a graph does
not fit an existing sweep. Rarely by hand.

**Tensor names matter.** `validator_overrides` in the engine TOML match on the
tensor *name* (`RMSNorm_0::Y`), and names survive capture where uids may not.

**Path convention:** `{tier}/{Op}/{Layout}/{DataType}/{Name}/{Name}.json`, where
`Layout` is `nchw`/`nhwc`/`ncdhw`/`ndhwc` (SDPA uses `bhsd`) and `DataType` is
`fp16`/`fp32`/`bfp16`/`fp8`/`int8` (SDPA writes bfloat16 as `bf16`).

## Template-sweep bundle — `graph.template.json` + `sweep.json`

Use a sweep when one topology is tested across many shapes, dtypes, or layouts.
The template holds what never varies (node types, wiring, uids, `virtual`
flags); each sweep case holds what does.

### `graph.template.json`

The same JSON as a single-graph bundle, except that any JSON **value** may be a
placeholder string of exactly the form `"${case.<path>}"`. The whole value is
replaced — there is no substitution inside a longer string.

```json
{
  "io_data_type": "${case.io_data_type}",
  "tensors": [
    { "uid": 1, "name": "RMSNorm_0::Y", "data_type": "${case.data_type}",
      "dims": "${case.dims}", "strides": "${case.strides}", "virtual": false }
  ]
}
```

Resolution rules (`IntegrationTestBundle.hpp`):

- Inside any object that has an integer `uid` (a tensor), a placeholder is
  looked up first in that uid's entry in the case's `values.tensors[]`, then in
  the case-wide `values`.
- Placeholders named `${case.dims}`, `${case.strides}` or `${case.data_type}`
  inside such an object **must** resolve from the per-tensor entry; a case-wide
  fallback for those three is an error. Other names (`${case.io_data_type}`)
  may fall back.
- `<path>` may be a dotted path into a nested object in `values`.
- A placeholder with no value makes that case fail to load (it is dropped from
  registration with an `ERROR` log line). A `values` key that no placeholder
  used is a warning (`Unused sweep value '…'`).

### `sweep.json`

```json
{
  "version": 1,
  "cases": [
    {
      "id": "small_fp32_nchw",
      "values": {
        "io_data_type": "float",
        "tensors": [
          { "uid": 0, "data_type": "float", "dims": [2, 3, 4, 5], "strides": [60, 20, 5, 1] }
        ]
      },
      "golden": { "path": "golden/small_fp32_nchw/tensors.dvc" },
      "metadata": {
        "format_version": 1,
        "generator": "generate_batchnorm_fwd_golden.py",
        "reference_source": "PyTorch 2.12.0+rocm7.2",
        "seed": 42
      }
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `cases[]` | yes | One test per entry. |
| `cases[].id` | yes | Non-empty string, unique within the sweep. Becomes the GTest test name. |
| `cases[].values` | no | Placeholder values; `values.tensors[]` holds per-tensor values keyed by `uid` (each `uid` must exist in the template, once). |
| `cases[].metadata` | **yes** | Same schema as a [metadata sidecar](#metadata-sidecar--namemetajson). A case without it fails to load. |
| `cases[].golden` | no | `{"path": "golden/<CaseId>/tensors.dvc"}`. Absent or `null` means a graph-only case. Present without a string `path` is an error. |
| `cases[].tensor_patches` | no | Structural per-tensor edits applied after placeholder expansion; see below. |

Case ids are handles, not descriptions. `import_graph.py` generates them as
`{shape}_{dtype}_{layout}_{≤3 salient attrs}[_{hash6}]`; the `hash6` suffix is
added only when the readable part is not already unique.
**Do not author cases by hand** — generate them with `import_graph.py` (see
[Adding Tests](adding-tests.md#add-a-bundle)). Editing a field of an existing
case, such as adding its `golden` pointer, is fine. Do not rename ids casually:
support-claim sidecars, and miopen-provider's `ffm-quick` tier, name cases by id.

#### `tensor_patches`

For a case that differs from the template by more than a substitution — for
example a tensor that must become runtime pass-by-value and therefore must not
carry a baked `value`:

```json
"tensor_patches": [
  { "uid": 4, "set": { "is_runtime_pass_by_value": true }, "remove": ["value", "value_type"] }
]
```

`set` upserts fields, then `remove` erases them. The `uid` must exist in the
expanded graph. A tensor left with `is_runtime_pass_by_value: true` and a
`value` fails validation at load.

## Metadata sidecar — `{Name}.meta.json`

Provenance and run guards for a single-graph bundle. **Every single-graph bundle
has one.** A sweep case carries the same object inline as `cases[].metadata`,
required on every case. Parsed by `src/harness/BundleMetadata.hpp`.

| Field | Type | Effect |
|---|---|---|
| `format_version` | int | **Required, must be `1`.** Otherwise the whole metadata object is rejected. |
| `enforcement_level` | string | `applicability`, `buildable`, or `full` (the default when absent). How far a run must get for a support claim to count as confirmed; see [Support claims](#support-claim-sidecars--namesupportjson-and-supportjson). Any other value rejects the metadata. |
| `minimum_vram_mb` | int | Skip the case when the device has less VRAM. |
| `gpu_architecture` | string | Golden data is arch-locked: skip the case unless the device's base arch token matches (`gfx942` matches `gfx942:sramecc+:xnack-`, not `gfx940`). Absent means the data is portable. |
| `seed` | int | Seed for generated inputs. |
| `inputs` | object | Per-input overrides keyed by tensor uid as a string (`"3"`); non-numeric keys are skipped with a warning. |
| `generator`, `generator_version`, `generated_at`, `reference_source`, `reference_source_hash`, `reference_strategy`, `rocm_version`, `operation`, `generation_command`, `notes` | string | Provenance only; no effect on the run. |

A bundle without its metadata is an error. Required fields:

- `format_version` (`1`) — the harness rejects the metadata without it.
- `generator` and `reference_source`, non-empty — required by
  `reference-data-scripts/verify_golden_bundles.py`, so a reviewer can tell
  where the graph and its expected values came from.

Generators may write extra keys (`config`, `input_range`, …); the harness
ignores keys it does not know.

## Golden data — `.tensors.dvc` and `.bin`

Golden tensors are optional. A bundle without them is **graph-only** and is
verified against the GPU or CPU reference executor instead. See
[Verification modes](running-tests.md#verification-modes).

- **Single graph:** `{Name}.tensor<uid>.bin`, one per tensor, all listed in one
  `{Name}.tensors.dvc` pointer.
- **Sweep:** `golden/{CaseId}/tensor<uid>.bin`, listed in
  `golden/{CaseId}/tensors.dvc`, referenced from the case's `golden.path`.

The `.bin` files are raw tensor bytes in the tensor's declared `data_type`, and
are never committed (`*.bin` is git-ignored). Git tracks only the `.dvc`
pointer, a YAML list of outputs with their md5 and size:

```yaml
outs:
- path: Small.tensor0.bin
  md5: 8510f215d557db558ee2f131ed05d2e3
  size: 33554432
  hash: md5
  remote: golden-data
```

Two DVC remotes are configured in the repository's `.dvc/config`:

| Remote | URL | Used by |
|---|---|---|
| `storage` (default) | `s3://therock-dvc/rocm-libraries` | legacy ops (pointers with no `remote:` key) |
| `golden-data` | `s3://therock-dvc/rocm-libraries/hipdnn/golden-data` | new hipDNN ops (pointers carry `remote: golden-data`) |

Because each output names its own remote, a bare `dvc pull` fetches everything.
Both remotes allow anonymous reads; pushing needs AWS credentials. The DVC
commands are in [Adding Tests](adding-tests.md#golden-data-with-dvc).

## Support-claim sidecars — `{Name}.support.json` and `support.json`

A support claim is a checked-in promise that a named **engine accepts** a graph
on a given **arch** and **platform**. It asserts acceptance, not correctness —
correctness is the output comparison's job. Claims turn "the engine quietly
stopped supporting this graph" from a skip nobody reads into a test failure.

| Bundle kind | Sidecar | Shape |
|---|---|---|
| Single graph `dir/Small.json` | `dir/Small.support.json` | `claims: { ENGINE: { arch: [platforms] } }` |
| Sweep `dir/sweep.json` | `dir/support.json` (one file for the whole sweep) | `claims: { ENGINE: [ { cases: [ids], support: { arch: [platforms] } } ] }` |

Single graph (`quick/BatchnormFwdInference/nchw/bfp16/Small/Small.support.json`,
one of its two engines shown), in the exact on-disk form:

```json
{
  "claims": {
    "HIP_MLOPS_ENGINE": {
      "gfx1151": [
        "windows"
      ],
      "gfx90a": [
        "linux"
      ],
      "gfx942": [
        "linux"
      ]
    }
  },
  "version": 1
}
```

Sweep (`quick/Layernorm/Variant2/support.json`, case list shortened):

```json
{
  "claims": {
    "HIP_MLOPS_ENGINE": [
      {
        "cases": [
          "2_2_3_2_2_bfp16_ncdhw_normalized_dim_count1",
          "2_2_3_2_2_bfp16_ncdhw_normalized_dim_count2"
        ],
        "support": {
          "gfx1151": [
            "windows"
          ],
          "gfx942": [
            "linux"
          ]
        }
      }
    ]
  },
  "version": 1
}
```

A bundle nothing is claimed for yet (see
[Adding Tests](adding-tests.md#when-no-engine-accepts-the-graph-yet)):

```json
{
  "claims": {},
  "version": 1
}
```

Rules:

- `version` must be `1`.
- **Engine** is the engine name the run passes as `--test-engine`
  (`MIOPEN_ENGINE`, `HIPBLASLT_ENGINE`, `HIP_MLOPS_ENGINE`, `ASM_SDPA_ENGINE`, …).
- **Arch** is the device's *base* token — the part of `gcnArchName` before the
  first `:` — and matches exactly. `gfx942` covers `gfx942:sramecc+:xnack-`; it
  does not cover `gfx940`, and there are no family wildcards.
- **Platform** is `linux` or `windows`.
- In a sweep sidecar a case id may appear in at most one group per engine. A
  case named in no group is simply unclaimed.
- **Every bundle has a sidecar.** A sidecar claims nothing about engines, archs
  or platforms it does not list; the absence of a claim is not a claim of
  non-support. A bundle that no engine accepts yet carries the empty sidecar
  above. A bundle with **no** sidecar is invisible to the claim machinery: no
  verdict, not even `unclaimed_support`.
- Sidecars are **machine-written** by `--write-support-claims`. Its form is
  stricter than JSON validity: sorted keys, 2-space indent, one array element
  per line, trailing newline, LF line endings; platforms and case ids sorted;
  one group per identical `support` map, groups ordered by their first case id.
  The verifier rejects anything that is not `json.dumps(…, indent=2,
  sort_keys=True)` output, and the writer rewrites anything not in its own form
  on its next run. So hand edits are limited to *retracting* a claim or writing
  an empty sidecar, in exactly that form.

`scripts/verify_support_claims.py` checks: the schema; canonical form; that every
claimed sweep case id exists in the sibling `sweep.json`; that no id repeats per
engine; that graphs with non-empty claims do not declare an unknown
`enforcement_level`; and that no sidecar is orphaned (a `X.support.json` with no
`X.json`, or a `support.json` outside a sweep root). It is registered as the
`verify-support-claims` pre-commit hook, but the repository's pre-commit
configuration currently excludes `integration-test-bundles/` from every hook,
so **run it by hand** after changing sidecars or sweeps:

```bash
python dnn-providers/integration-tests/scripts/verify_support_claims.py
```

How claims are checked during a run, and every verdict a run can report, is in
[Running the Tests](running-tests.md#support-claim-summary). The harness
lifecycle behind it is in
[Support Claim Enforcement](support-claim-enforcement.md).

## Per-engine test config — `config/<ENGINE_NAME>.toml`

Each provider owns one TOML file per engine, passed to the binary as
`--test-config` (CTest lanes pass it for you). It changes how *that engine* is
tested without recompiling or touching the bundles.

| Provider | Files |
|---|---|
| miopen-provider | `config/MIOPEN_ENGINE.toml` |
| hipblaslt-provider | `config/HIPBLASLT_ENGINE.toml` |
| hip-kernel-provider | `config/HIP_MLOPS_ENGINE.toml`, `config/ASM_SDPA_ENGINE.toml` |

```toml
[meta]
version = 1

[[tolerance_overrides]]
filters = ["Smoke/IntegrationGpuConvWrw3dBfp16.Correctness/14"]
atol = 1.19
rtol = 0.2

[[validator_overrides]]
filters       = ["*LayernormBackward*"]
tensors       = ["*::DSCALE", "*::DBIAS"]
validator     = "rms"
rms_threshold = 1e-4

[[test_skips]]
archs     = ["gfx90a", "gfx10", "gfx11", "gfx12"]   # optional; omit to match every arch
platforms = ["windows"]                             # optional; omit to match both
filters   = ["*ConvFwdBiasActiv*"]
reason    = "ROCm/rocm-libraries#6979 - no engine has an applicable solution"
```

| Table | Rules |
|---|---|
| `[meta]` | `version = 1` is required; a missing or unsupported version is a load error, not a silent ignore. |
| `[[tolerance_overrides]]` | `filters`, `atol` and `rtol` are all required. **Later entries win** when several match. |
| `[[validator_overrides]]` | Applies when a `filters` glob matches the test **and** a `tensors` glob matches the output tensor's label (its name, or `uid=N` when unnamed). `validator` is `allclose`, `allclose_matching_infinities` or `rms`; `rms_threshold` is required and positive for `rms`, forbidden for the other two. **Later entries win.** Allclose is the default everywhere; this table is the only thing that changes it, and is meant for outputs where per-element comparison is the wrong question, not for buying slack. See the validators below. |
| `[[test_skips]]` | `filters` and `reason` are required. **The first matching entry wins** — the opposite order from the two tables above. `archs` is a substring match against the raw `gcnArchName` (so `gfx11` covers `gfx1100` and `gfx1151`); `platforms` is `linux` / `windows`. The skip message is `[arch <current gcnArchName>] <reason>`. |

`filters` are globs matched against the full GTest name — the same string
`--gtest_filter` matches, but not `--gtest_filter` syntax: `:` does not separate
alternatives and a leading `-` does not negate. Use one array element per
pattern, and only `*` and `?` as wildcards: Linux matches with `fnmatch`
(case-sensitive), Windows with `PathMatchSpecA` (case-insensitive).

The two non-default validators:

- **`rms`** — compares the aggregate relative-RMS error against
  `rms_threshold`. For long reductions (layernorm/RMSNorm backward
  `dscale`/`dbias`), whose elements can cancel toward zero so per-element
  relative error is unbounded.
- **`allclose_matching_infinities`** — grades exactly as `allclose` at the same
  atol/rtol, except that an element infinite with the *same sign* in both the
  reference and the engine output compares equal (NaN, opposite-signed
  infinities and finite-versus-infinite still fail). For outputs whose correct
  value is infinite on both sides, such as the log-sum-exp of a fully masked
  SDPA row. It is **host-only**: a tensor it selects fails with
  `Validator override NOT APPLICABLE ON DEVICE` whenever its comparison runs on
  the device — under `auto`, every run where the GPU reference produced the
  expected values. Run such a config with `--validator cpu`, or narrow the
  `tensors` glob.

Neither is defined for integer outputs: a glob that catches one fails that
tensor with a message naming the glob; narrow it to float outputs.

The TOML applies to bundle and C++ graph tests alike. It never applies to
`hipdnn_golden_data_tests`: an engine's config describes how far that engine may
drift and cannot skip or loosen a check on our own golden data. Full schema:
`src/harness/TestSettings.hpp`.

A `test_skips` entry is how a *known* engine limitation is recorded. It skips in
`SetUp()`, before the support-claim query, so a claim on a skipped case is
neither enforced nor refreshed on that arch.

## Tier YAML files — `test_categories*.yaml`

These map GTest name patterns to CTest labels and timeouts. They are parsed at
configure time by `shared/ctest/parse_test_categories.py`; `shared/ctest/README.md`
is the reference for the format. Three scopes exist and are easy to confuse:

| File | Governs |
|---|---|
| `dnn-providers/integration-tests/test_categories.yaml` | This project's own binaries (`hipdnn_integration_tests_unit_tests`, `hipdnn_gpu_ref_tests`, `hipdnn_golden_data_tests`) |
| `dnn-providers/integration-tests/test_categories_external.yaml` | Pre-registered non-GTest CTest tests (the Python verifiers, test-name validation) |
| `dnn-providers/<provider>/test_categories_integration.yaml` (hip-kernel-provider: `HIP_MLOPS_ENGINE_test_categories_integration.yaml`, `ASM_SDPA_ENGINE_test_categories_integration.yaml`) | `hipdnn_integration_tests` **run against that provider's engine** — the lane's bundle suites |

A provider's own `test_categories.yaml` governs its native `*_plugin_tests`
binaries, not the shared suite.

```yaml
test_categories:
  quick:
    description: "quick-tier bundle sweeps plus always-built C++ tests"
    test_patterns: ["quick_*", "Smoke/*", "Quick/*"]
    exclude: ["*DISABLED*"]
    labels: ["quick", "pre-commit", "smoke"]

exclude_gpu:
  exclude_gpu_gfx110X_windows:
    test_patterns: ["*"]
    labels: ["quick", "standard", "ex_gpu_gfx110X"]

execution_settings:
  default_timeout: 300
  category_timeouts:
    quick: 1200
```

- `test_patterns` / `exclude` are GTest globs; `/` and `_` are literal, so
  bundle suites (`quick_*`) and C++ suites (`Smoke/*`) need separate patterns.
- Each category becomes one CTest test named `<prefix>_<category>_suite` carrying
  its `labels`.
- Tier inclusion is written out, not inherited: in the provider files the
  `standard` category lists the quick patterns *and* the standard ones. Labels
  do not cascade on their own; check the file you are editing. What each
  provider tier selects is tabulated in
  [Running the Tests](running-tests.md#test-tiers).
- Keep `Smoke/*` and `Quick/*` in a provider's quick tier even on a default
  build: the always-built C++ tests register under those prefixes, and the
  C++ graph tests do too when a branch builds them.
- `exclude_gpu_<arch>[_windows|_linux]` adds a per-arch variant suite
  (`<prefix>_<category>_<arch>_suite`, label `ex_gpu_<arch>`) carrying that
  arch's negative patterns. A match-everything pattern (`"*"`) cannot be
  expressed as a gtest filter, so that variant is registered `DISABLED` and
  never launched. Variants are generated per `ex_gpu_*` label declared in the
  file, not for the arch you are building; you select one with
  `ctest -L ex_gpu_<arch>`.
- miopen-provider's `ffm-quick` lists exact test names (sweep case ids).
  Regenerating a sweep can rename those ids and silently shrink that tier.

## Where each format is enforced

| Format | Parsed / checked by | When it fails |
|---|---|---|
| Graph `.json` | `IntegrationTestBundle.hpp` | Case dropped from registration with an `ERROR` log line; the run stays green with one test fewer |
| `graph.template.json` + `sweep.json` | `BundleDiscovery.hpp`, `IntegrationTestBundle.hpp` | A malformed `sweep.json` or a bad/duplicate id aborts registration; a bad case (missing placeholder value, missing metadata, bad `golden.path`) is dropped like a bad graph |
| `.meta.json` / `cases[].metadata` | `BundleMetadata.hpp`; `reference-data-scripts/verify_golden_bundles.py` | Required for every bundle and every sweep case, with `generator` and `reference_source`; a missing file or block is an error |
| `.support.json` / `support.json` | `SupportClaims.cpp` at run time; `scripts/verify_support_claims.py` by hand | Required for every bundle; a missing sidecar is an error. A broken claim fails the test at run time |
| Golden `.bin` / `.dvc` | `IntegrationTestBundle.hpp`; `reference-data-scripts/verify_golden_bundles.py` | Falls back to a reference in `auto` mode; fails in `golden` mode |
| Engine `.toml` | `TestSettings.hpp` | Binary exits 1 at startup |
| `test_categories*.yaml` | `shared/ctest/parse_test_categories.py` | CMake **warning** at configure; that target's tier suites are not generated, so `ctest -L <tier>` quietly runs less. Check the configure log or `ctest -N -L <tier>` |

The only load problem that registers a *failing* test is a tensor marked
`is_runtime_pass_by_value` that still carries a baked `value`.
