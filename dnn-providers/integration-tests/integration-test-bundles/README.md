# Integration Test Bundles

Pre-computed reference tensors for integration tests. Bundles are **optional** —
tests can run without them using `--verification-mode gpu` or
`--verification-mode cpu`. Bundles enable golden-comparison mode, which is more
sensitive but requires data to be fetched first.

Binary data is stored in S3 via [DVC](https://dvc.org) — git only tracks small
`.dvc` pointer files. The tree now supports both RFC 0011 bundle kinds:

- **Single-graph bundles**: one `{Name}.json` plus one `{Name}.tensors.dvc`
  pointer for that bundle's tensor blobs.
- **Template-sweep bundles**: one `graph.template.json`, one `sweep.json`, and
  one `golden/{CaseId}/tensors.dvc` pointer per expanded case.

| Key            | Value                                |
|----------------|--------------------------------------|
| Default remote | `storage` → `s3://therock-dvc/rocm-libraries` (legacy shared store; existing ops) |
| hipDNN remote  | `golden-data` → `s3://therock-dvc/rocm-libraries/hipdnn/golden-data` (new ops, per-op subdirs) |
| Tracking       | Per-bundle or per-sweep-case `.dvc` tracking (`.json` stays in git for PR review) |
| Naming spec    | [RFC 0011 Section 4.1](../../../projects/hipdnn/docs/rfcs/0011_GoldenReferenceValidation.md) |

### DVC Remote Layout (`golden-data`)

New hipDNN golden data uses a single `golden-data` remote rooted at
`s3://therock-dvc/rocm-libraries/hipdnn/golden-data`. DVC stores objects
content-addressed under `<remote-url>/files/md5/`, so all ops on this remote share
one store:

```
s3://therock-dvc/rocm-libraries/hipdnn/golden-data/files/md5/<ab>/<rest...>
```

**Per-op organization is in the git tree, not in S3.** Each op is a top-level
folder under `integration-test-bundles/{Tier}/` (e.g. `SdpaFwd`, `BatchnormFwdInference`).
Push/pull is **scoped by repo path** so you operate on one op at a time, even
though the bytes all land in the same content-addressed S3 store:

```bash
# push only SDPA-forward bundles, to the golden-data remote:
dvc push -r golden-data --recursive \
  dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd \
  dnn-providers/integration-tests/integration-test-bundles/standard/SdpaFwd
```

**Each pointer declares its own remote (so the default `dvc pull` just works).**
A bundle on the `golden-data` remote sets a per-output `remote:` key in its
`.tensors.dvc`:

```yaml
outs:
- path: Small.tensor0.bin
  remote: golden-data        # <- DVC fetches/pushes this output from golden-data
  md5: 437ad9f6e2b1db28ca900503e09f7a63
  size: 524288
  hash: md5
```

With this key, a **bare `dvc pull` (no `-r`)** pulls golden-data outputs from
golden-data and everything else from the default `storage` remote — in one pass.
This is why **CI needs no change**: the existing `dvc pull` step already fetches
golden-data bundles. The remote travels *with the data pointer*, not in a separate
workflow file every consumer must remember to flag.

**Why a single remote (not one remote per op):**
- DVC remotes are global names in `.dvc/config`; a `.tensors.dvc` pointer records
  md5/size plus an optional per-output `remote:`. The pull/push location is that
  per-output remote (falling back to the repo default).
- One remote + the per-output `remote:` key keeps **CI to a single bare `dvc pull`**,
  and every new op is just a new folder — no `.dvc/config` edit, no CI change.
- Per-op isolation is **logical** (git folder taxonomy + path-scoped push), not a
  separate S3 keyspace. Cross-op dedup is not expected (different ops produce
  different tensors), so a shared store under the root is acceptable.
- Trade-off: because all ops share `…/golden-data/files/md5/`, you cannot prune one
  op's S3 objects in isolation with `dvc gc --cloud` (gc keys off what the repo
  references across the whole remote). If a future op needs independent S3
  lifecycle / gc, promote it to its own remote (`…/golden-data/<op>`) at that time.

> The legacy `storage` remote (bare `rocm-libraries` root) remains the default.
> Existing ops (e.g. Batchnorm) keep working unchanged; their pointers have no
> `remote:` key so they resolve to `storage`. New hipDNN ops add `remote: golden-data`
> to their pointers (and push with `-r golden-data`).

## Folder Convention

Single-graph bundle (the `.tensors.dvc`/`.bin` files are optional — a
`{Name}.json` with no tensor files is a valid graph-only bundle, verified
against a live GPU/CPU reference executor instead of golden comparison):

```
integration-test-bundles/{Tier}/{Operation}/{Layout}/{DataType}/{Name}/
    {Name}.json              # graph description (committed to git)
    {Name}.tensors.dvc       # optional — DVC pointer for this bundle's .bin files
    {Name}.tensor0.bin       # optional — binary tensor data (DVC-tracked)
    {Name}.tensor1.bin
    ...
```

Template-sweep bundle (each case's `golden/{CaseId}/` directory is likewise
optional per case):

```
integration-test-bundles/{Tier}/{Operation}/{TopologyName}/
    graph.template.json      # invariant graph topology with ${case.*} placeholders
    sweep.json               # case matrix + per-case metadata/optional golden paths
    golden/{CaseId}/tensors.dvc   # optional
    golden/{CaseId}/tensor0.bin   # optional
    golden/{CaseId}/tensor1.bin
    ...
```

Single-graph segments:

| Segment     | Allowed values                                | Example            |
|-------------|-----------------------------------------------|--------------------|
| `Tier`      | `quick`, `standard`, `comprehensive`, `full`  | `quick`            |
| `Operation` | PascalCase op name                            | `BatchnormFwdInference` |
| `Layout`    | `nchw`, `nhwc`, `ncdhw`, `ndhwc`              | `nhwc`             |
| `DataType`  | `fp16`, `fp32`, `bfp16`, `fp8`, `int8`        | `fp16`             |
| `Name`      | Descriptive name (PascalCase or snake_case)   | `Small`, `resnet50_layer3` |

Template-sweep segments:

| Segment        | Allowed values                               | Example     |
|----------------|----------------------------------------------|-------------|
| `Tier`         | `quick`, `standard`, `comprehensive`, `full` | `quick`     |
| `Operation`    | PascalCase op name                           | `BatchnormFwdInference` |
| `TopologyName` | Stable graph skeleton name                   | `Inference` |

### `tensor_patches` (per-case structural tensor edits)

A `sweep.json` case can carry an optional `tensor_patches` array to make
structural edits to a tensor after `graph.template.json` placeholders are
expanded, but before the graph is sealed into a flatbuffer. This covers cases
that differ from the template by more than a placeholder substitution — e.g. a
case where a tensor is runtime-pass-by-value and must not also carry a baked
`value`/`value_type`.

Each entry targets one tensor by `uid` and supports two operations:

```json
"tensor_patches": [
  {
    "uid": 4,
    "set": { "is_runtime_pass_by_value": true },
    "remove": ["value", "value_type"]
  }
]
```

- `set` (object): upserts the given fields on the target tensor.
- `remove` (array of strings): erases the given fields from the target tensor.
- `uid` must match a tensor present in the expanded graph, or loading the case
  fails with an error.

`set` is applied before `remove` for a given patch entry. A tensor marked
`is_runtime_pass_by_value: true` that still carries `value`/`value_type` after
patching fails validation at load time — use `remove` to drop the baked
fallback fields when converting a tensor to runtime pass-by-value.

## Pull Data Locally

Bundles are optional. Skip this step if you only need computed (GPU/CPU)
verification.

```bash
# From the repo root:

# Pull everything. A bare pull resolves each output to its own remote:
# golden-data outputs come from golden-data (via the per-output 'remote:' key),
# all other outputs from the default 'storage' remote.
dvc pull

# Pull only one op's quick-tier bundles (sufficient for smoke tests)
dvc pull dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd
```

CI needs no special flag: the existing `dvc pull` step in `therock-ci-linux.yml`
already fetches golden-data bundles, because each golden-data pointer carries its
own `remote: golden-data` key (see the DVC Remote Layout section above).

> **Note:** DVC commands must be run from the repo root (`rocm-libraries/`), not from a subdirectory.

## Verify Bundles

```bash
python dnn-providers/integration-tests/reference-data-scripts/verify_golden_bundles.py \
  dnn-providers/integration-tests/integration-test-bundles
```

The verifier checks graph JSON parsing, tensor byte sizes, metadata sidecars, and
output-tensor NaN/Inf rejection. Real `.bin` payloads are optional unless the
bundle includes `<Name>.tensors.dvc`; when that pointer exists, run `dvc pull`
if only the pointer file is present.

For `quick/BatchnormFwdInference/nchw/fp32/Small/Small.json`, the verifier
prints the advisory test name
`quick_BatchnormFwdInference_nchw_fp32_Small.Small`.

## Add a New Bundle

```bash
# 1. Create the bundle directory
mkdir -p dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 2. Copy your files in
cp resnet50_layer3.json        dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3/
cp resnet50_layer3.tensor*.bin dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 3. Author a single per-bundle DVC pointer listing every .bin, then let DVC fill in the hashes.
#    DVC cannot generate a multi-file pointer itself, so we write the `outs:` list and `dvc commit`.
#    For NEW hipDNN ops, tag each output with `remote: golden-data` so a bare
#    `dvc pull` (and CI) fetches it from the golden-data remote.
BUNDLE=dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3
{ echo "outs:"; for f in "$BUNDLE"/*.tensor*.bin; do
    echo "- path: $(basename "$f")"; echo "  remote: golden-data"; done; } \
    > "$BUNDLE/resnet50_layer3.tensors.dvc"
dvc commit -f "$BUNDLE/resnet50_layer3.tensors.dvc"

# 4. Git-add the .json (for PR review) and the .tensors.dvc pointer
git add "$BUNDLE/resnet50_layer3.json" "$BUNDLE/resnet50_layer3.tensors.dvc"

# 5. Commit and push
#    NEW hipDNN ops push to the golden-data remote; legacy 'storage' ops omit -r.
git commit -m "Add ConvFwd resnet50_layer3 bundle"
dvc push -r golden-data    # or: dvc push   (legacy 'storage' remote)
git push
```

### SDPA forward bundles (regen-driven)

SDPA-forward bundles are produced by a generator, not copied in. Regenerate,
record pointers, and push to `golden-data`:

```bash
# 1. Regenerate all SDPA-fwd bundles (quick + standard tiers)
cd dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd
bash generate_golden_data.sh                 # Tier A (bf16/fp16, nomask/causal, hd128/hd192)
GENERATE_TIER_B=1 bash generate_golden_data.sh   # + FP8/GROUP (not yet CI-validated)
cd -

# 2. Record .bin hashes into the per-bundle pointers + cache (from repo root).
#    `dvc commit` preserves the existing per-output `remote: golden-data` key, so
#    pointers stay routed to golden-data across regenerations. (For a brand-new
#    bundle dir, add the key once — see step 3 of "Add a New Bundle".)
dvc commit -f --recursive \
  dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd \
  dnn-providers/integration-tests/integration-test-bundles/standard/SdpaFwd

# 3. Push tensor data to the golden-data remote
dvc push -r golden-data --recursive \
  dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd \
  dnn-providers/integration-tests/integration-test-bundles/standard/SdpaFwd

# 4. Stage .json + .meta.json + .tensors.dvc (NOT .bin — gitignored), commit, git push
git add dnn-providers/integration-tests/integration-test-bundles/{quick,standard}/SdpaFwd
git commit -m "Update SDPA forward golden bundles"
git push
```

> `dvc push` uploads from the **local cache**, not the working tree — always
> `dvc commit` (or `dvc add`) first so regenerated `.bin` are cached, or push will
> silently skip them.

## Update an Existing Bundle

```bash
BUNDLE=dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3

# 1. Overwrite the files in the bundle directory
cp new_tensors/*.bin "$BUNDLE/"

# 2. Re-record the .bin hashes in the existing per-bundle pointer.
#    (If the set of tensor files changed, re-author the `outs:` list as in step 3 of "Add a New Bundle".)
dvc commit -f "$BUNDLE/resnet50_layer3.tensors.dvc"

# 3. Stage the updated pointer, plus the .json if it also changed
git add "$BUNDLE/resnet50_layer3.tensors.dvc" "$BUNDLE/resnet50_layer3.json"

# 4. Commit and push
git commit -m "Update ConvFwd resnet50_layer3 tensors"
dvc push
git push
```

Old data remains in S3 by content hash. Reverting the git commit restores the
old `.dvc` pointer, and `dvc pull` fetches the previous version.

## Remove a Bundle

```bash
# 1. Remove DVC tracking for the bundle
dvc remove dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3/resnet50_layer3.tensors.dvc

# 2. Delete the bundle directory
rm -rf dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 3. Commit
git commit -m "Remove ConvFwd resnet50_layer3 bundle"
git push
```

## Revert DVC to Git (Emergency)

If DVC tracking needs to be rolled back:

### Single bundle

```bash
# Pull the data if not on disk, then remove DVC tracking and re-add to git
dvc pull dnn-providers/integration-tests/integration-test-bundles/quick/BatchnormFwdInference/nchw/fp32/Small/Small.tensors.dvc
dvc remove dnn-providers/integration-tests/integration-test-bundles/quick/BatchnormFwdInference/nchw/fp32/Small/Small.tensors.dvc
git add -f dnn-providers/integration-tests/integration-test-bundles/quick/BatchnormFwdInference/nchw/fp32/Small/*.bin
git commit -m "Revert Small bundle from DVC to git tracking"
```

### All bundles (nuclear)

```bash
# Find and revert the DVC migration commit
git log --oneline -- "*.dvc" | head -5
git revert <migration-commit-hash>
```

## How It Works

Each bundle's `.bin` files are tracked together by a single `<Name>.tensors.dvc`
pointer. `.json` files stay in git.

```
On disk (your checkout)           In git                     In S3
---------------------------------  -------------------------  -------------------------
resnet50_layer3/
  resnet50_layer3.json             resnet50_layer3.json       (not in S3)
  resnet50_layer3.tensors.dvc      resnet50_layer3.tensors.dvc (not in S3)
  resnet50_layer3.tensor0.bin      (tracked by .tensors.dvc)  ab/cd1234...  (tensor0)
  resnet50_layer3.tensor1.bin      (tracked by .tensors.dvc)  ef/gh5678...  (tensor1)
```

- `.json` graph descriptions are committed to git — visible in PR diffs
- One `.tensors.dvc` per bundle records the md5/size of every `.bin` (multiple `outs:`)
- `.bin` tensor data is stored in S3 by content hash via DVC
- Identical files are stored once regardless of path
- Old versions persist — revert a `.tensors.dvc` pointer to restore previous data
- `dvc push` uploads only new/changed files
- `dvc pull` downloads only what is missing from local cache

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `dvc pull` auth error | Run `aws sts get-caller-identity`. Reads are anonymous; writes need AWS credentials. |
| `.tensors.dvc` exists but no data on disk | `dvc pull path/to/Name.tensors.dvc` |
| `.bin` accidentally committed to git | `git rm --cached path/to/file.bin`, then re-record with `dvc commit path/to/Name.tensors.dvc` |
| Added/removed tensor files in a bundle | Re-author the `outs:` list in `Name.tensors.dvc`, then `dvc commit -f` it |
| Tests can't find bundle data | `dvc pull` then `dvc status` to check for drift; or run with `--verification-mode gpu` to skip bundle comparison |
