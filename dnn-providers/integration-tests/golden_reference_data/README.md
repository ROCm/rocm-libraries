# Golden Reference Data

Pre-computed reference tensors for integration tests. Binary data is stored
in S3 via [DVC](https://dvc.org) — git only tracks small `.dvc` pointer files.

| Key            | Value                                |
|----------------|--------------------------------------|
| Remote         | `s3://therock-dvc/rocm-libraries`    |
| Tracking       | Per-bundle (one `.dvc` file per bundle) |
| Auto-stage     | Enabled (`dvc add` auto-stages `.dvc` and `.gitignore` for git) |
| Naming spec    | [RFC 0011 Section 4.1](../../../projects/hipdnn/docs/rfcs/0011_GoldenReferenceValidation.md) |

## Folder Convention

```
golden_reference_data/{Tier}/{Operation}/{Layout}/{DataType}/{Name}/
    {Name}.json              # graph description
    {Name}.tensor0.bin       # binary tensor data
    {Name}.tensor1.bin
    ...
```

| Segment     | Allowed values                                | Example            |
|-------------|-----------------------------------------------|--------------------|
| `Tier`      | `quick`, `standard`, `comprehensive`, `full`  | `quick`            |
| `Operation` | PascalCase op name                            | `BatchnormFwdInference` |
| `Layout`    | `nchw`, `nhwc`, `ncdhw`, `ndhwc`              | `nhwc`             |
| `DataType`  | `fp16`, `fp32`, `bfp16`, `fp8`, `int8`        | `fp16`             |
| `Name`      | snake_case descriptive name                   | `resnet50_layer3`  |

## Pull Data Locally

```bash
# Pull all bundles
dvc pull

# Pull only quick-tier bundles (sufficient for smoke tests)
dvc pull dnn-providers/integration-tests/golden_reference_data/quick/
```

CI runs `dvc pull` automatically. You only need this for local development.

## Add a New Bundle

```bash
# 1. Create the bundle directory
mkdir -p dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 2. Copy your files in
cp resnet50_layer3.json        dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/
cp resnet50_layer3.tensor*.bin dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 3. DVC-track the bundle (auto-stages .dvc and .gitignore for git)
dvc add dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 4. Commit and push
git commit -m "Add ConvFwd resnet50_layer3 bundle"
dvc push
git push
```

## Update an Existing Bundle

```bash
# 1. Overwrite the files in the bundle directory
cp new_tensors/*.bin dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 2. Re-track (updates the hash in the .dvc file)
dvc add dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3/

# 3. Commit and push
git commit -m "Update ConvFwd resnet50_layer3 tensors"
dvc push
git push
```

Old data remains in S3 by content hash. Reverting the git commit restores the
old `.dvc` pointer, and `dvc pull` fetches the previous version.

## Remove a Bundle

```bash
# 1. Remove DVC tracking
dvc remove dnn-providers/integration-tests/golden_reference_data/quick/ConvFwd/nhwc/fp16/resnet50_layer3.dvc

# 2. Delete the data
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
dvc pull dnn-providers/integration-tests/golden_reference_data/quick/BatchnormFwdInference/nchw/fp32/Small.dvc
dvc remove dnn-providers/integration-tests/golden_reference_data/quick/BatchnormFwdInference/nchw/fp32/Small.dvc
git add -f dnn-providers/integration-tests/golden_reference_data/quick/BatchnormFwdInference/nchw/fp32/Small/
git commit -m "Revert Small bundle from DVC to git tracking"
```

### All bundles (nuclear)

```bash
# Find and revert the DVC migration commit
git log --oneline -- "*.dvc" | head -5
git revert <migration-commit-hash>
```

## How It Works

Each bundle is tracked independently. `dvc add` hashes every file by MD5 and
stores it in S3 by that hash. Git only sees the `.dvc` pointer file.

```
On disk (your checkout)         In git              In S3
------------------------------  ------------------  -------------------------
resnet50_layer3/                resnet50_layer3.dvc  ab/cd1234...  (json)
  resnet50_layer3.json            outs:              ef/gh5678...  (tensor0)
  resnet50_layer3.tensor0.bin       - md5: ab12..    ij/kl9012...  (tensor1)
  resnet50_layer3.tensor1.bin         path: ...
```

- Identical files are stored once regardless of path
- Old versions persist — revert a `.dvc` pointer to restore previous data
- `dvc push` uploads only new/changed files
- `dvc pull` downloads only what is missing from local cache

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `dvc pull` auth error | Run `aws sts get-caller-identity`. Reads are anonymous; writes need AWS credentials. |
| `.dvc` file exists but no data on disk | `dvc pull path/to/bundle.dvc` |
| `.bin` accidentally committed to git | `git rm --cached path/to/file.bin` then `dvc add` the bundle |
| `dvc add` says "already tracked by Git" | `git rm -r --cached path/to/bundle/` first, then `dvc add` |
| Tests can't find reference data | `dvc pull` then `dvc status` to check for drift |
