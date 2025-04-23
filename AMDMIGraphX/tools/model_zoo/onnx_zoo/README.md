# ONNX Zoo model tester

Helper script to test [`ONNX Zoo models`](https://onnx.ai/models/) which have test data with [`test_runner.py`](../../test_runner.py)

## Getting the repository

> [!IMPORTANT]
> Make sure to enable git-lfs.

```bash
git clone https://github.com/onnx/models.git --depth 1
```

## Running the tests

> [!IMPORTANT]
> The argument must point to a folder, not a file.

```bash
# VERBOSE=1 DEBUG=1 # use these for more log
# ATOL=0.001 RTOL=0.001 TARGET=gpu # are the default values
./test_models.sh models/validated
```

You can also pass multiple folders, e.g.:

```bash
./test_models.sh models/validated/text/machine_comprehension/t5/ models/validated/vision/classification/shufflenet/
```

## Results

Result are separated by dtype: `logs/fp32` and `logs/fp16`

### Helpers

```bash
# Something went wrong
grep -HRL PASSED logs
# Runtime error
grep -HRi RuntimeError logs/
# Accuracy issue
grep -HRl FAILED logs
```

## Cleanup

If at any point something fails, the following things might need cleanup:
- Remove `tmp_model` folder
- `git lfs prune` in `models`