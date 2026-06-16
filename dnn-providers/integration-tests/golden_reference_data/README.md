# Golden Reference Data

Pre-computed reference tensors for integration tests. Binary data is stored
in S3 via [DVC](https://dvc.org) — git only tracks small `.dvc` pointer files.

| Key            | Value                                |
|----------------|--------------------------------------|
| Remote         | `s3://therock-dvc/rocm-libraries`    |
| Tracking       | Per-bundle `.bin` tracking — one `<Name>.tensors.dvc` per bundle (`.json` stays in git for PR review) |
| Naming spec    | [RFC 0011 Section 4.1](../../../projects/hipdnn/docs/rfcs/0011_GoldenReferenceValidation.md) |

## Folder Convention

```
golden_reference_data/{Tier}/{Operation}/{Layout}/{DataType}/{Name}/
    {Name}.json              # graph description (committed to git)
    {Name}.tensors.dvc       # DVC pointer tracking all of this bundle's .bin files (committed to git)
    {Name}.tensor0.bin       # binary tensor data (DVC-tracked)
    {Name}.tensor1.bin
    ...
```

| Segment     | Allowed values                                | Example            |
|-------------|-----------------------------------------------|--------------------|
| `Tier`      | `quick`, `standard`, `comprehensive`, `full`  | `quick`            |
| `Operation` | PascalCase op name                            | `BatchnormFwdInference` |
| `Layout`    | `nchw`, `nhwc`, `ncdhw`, `ndhwc`              | `nhwc`             |
| `DataType`  | `fp16`, `fp32`, `bfp16`, `fp8`, `int8`        | `fp16`             |
| `Name`      | Descriptive name (PascalCase or snake_case)   | `Small`, `resnet50_layer3` |

## Pull Data Locally

```bash
# From the repo root:

# Pull all bundles
dvc pull

# Pull only quick-tier bundles (sufficient for smoke tests)
dvc pull dnn-providers/integration-tests/golden_reference_data/quick/
```

The Linux CI workflow (`therock-ci-linux.yml`) runs `dvc pull` automatically. Other pipelines (Windows, superbuild) may need it wired in separately — see backlog.

> **Note:** DVC commands must be run from the repo root (`rocm-libraries/`), not from a subdirectory.

## Add a New Bundle

```bash
# 1. Create the bundle directory
mkdir -p dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 2. Copy your files in
cp resnet50_layer3.json        dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/
cp resnet50_layer3.tensor*.bin dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 3. Author a single per-bundle DVC pointer listing every .bin, then let DVC fill in the hashes.
#    DVC cannot generate a multi-file pointer itself, so we write the `outs:` list and `dvc commit`.
BUNDLE=dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3
{ echo "outs:"; for f in "$BUNDLE"/*.tensor*.bin; do echo "- path: $(basename "$f")"; done; } > "$BUNDLE/resnet50_layer3.tensors.dvc"
dvc commit -f "$BUNDLE/resnet50_layer3.tensors.dvc"

# 4. Git-add the .json (for PR review) and the .tensors.dvc pointer
git add "$BUNDLE/resnet50_layer3.json" "$BUNDLE/resnet50_layer3.tensors.dvc"

# 5. Commit and push
git commit -m "Add ConvFwd resnet50_layer3 bundle"
dvc push
git push
```

## Update an Existing Bundle

```bash
BUNDLE=dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3

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
dvc remove dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/resnet50_layer3.tensors.dvc

# 2. Delete the bundle directory
rm -rf dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 3. Commit
git commit -m "Remove ConvFwd resnet50_layer3 bundle"
git push
```

## Revert DVC to Git (Emergency)

If DVC tracking needs to be rolled back:

### Single bundle

```bash
# Pull the data if not on disk, then remove DVC tracking and re-add to git
dvc pull dnn-providers/integration-tests/golden_reference_data/quick/BatchnormFwdInference/nchw/fp32/Small/Small.tensors.dvc
dvc remove dnn-providers/integration-tests/golden_reference_data/quick/BatchnormFwdInference/nchw/fp32/Small/Small.tensors.dvc
git add -f dnn-providers/integration-tests/golden_reference_data/quick/BatchnormFwdInference/nchw/fp32/Small/*.bin
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
| Tests can't find reference data | `dvc pull` then `dvc status` to check for drift |
