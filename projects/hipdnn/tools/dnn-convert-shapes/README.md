# dnn-convert-shapes

Convert MIOpen driver shape files to hipDNN JSON graph files.

## Overview

`dnn-convert-shapes` translates MIOpen driver invocations into hipDNN JSON graph files. It accepts two input modes:

- **Shape files**: `.txt` files containing one MIOpen driver invocation per line (batch conversion).
- **Inline arguments**: a single MIOpen driver argument string passed directly on the CLI via `--args`.

It supports convolution and batchnorm operations, 2-D and 3-D convolutions, forward/backward/wgrad directions, and NCHW/NHWC layouts.

The generated JSON graphs are consumed by [`dnn-benchmark`](../dnn-benchmarking/README.md) to run and benchmark the shapes against hipDNN providers.

## Installation

A virtual environment is recommended to isolate the tool from system Python packages.

```bash
# Create and activate a venv
python3 -m venv .venv
source .venv/bin/activate

# Standard install
pip install .

# Editable install (for development)
pip install -e .
```

## Usage

### Convert shape files

```bash
# Convert one or more shape files (output goes next to each input file)
dnn-convert-shapes graphs/shapes.txt graphs/shapes_3D.txt

# Write output to a specific directory
dnn-convert-shapes shapes.txt --outdir graphs/generic_convolutions/
```

### Inline argument conversion

```bash
# Convert a single inline MIOpen driver invocation
dnn-convert-shapes --args 'convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 1 -q 0 -F 1'

# Inline args with explicit output path
dnn-convert-shapes --args 'convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1' \
  --output graphs/my_conv.json
```

### Output naming

Each converted graph is written as `<stem>_conv_<direction>_n<N>c<C>H<H>W<W>_....json`. Duplicate shapes within a file get a numeric suffix. Lines beginning with `#` and blank lines are skipped. A leading repeat-count column (e.g. `5  ./bin/MIOpenDriver ...`) is stripped automatically.

## Supported Operations

| Operation | Data Type |
|-----------|-----------|
| `conv` | float |
| `convfp16` | half |
| `convbfp16` | bfloat16 |
| `bnorm` | float |
| `bnormfp16` | half |
| `bnormbfp16` | bfloat16 |

## Exit Codes

- `0` -- Shape file mode: file was readable and processing finished, even if individual lines failed to parse (per-line failures are reported as warnings on stderr but do not abort the run). Inline `--args` mode: the single invocation converted successfully.
- `1` -- Fatal error: input file not found, no shape files provided, output path could not be written, or (inline `--args` mode only) the argument string failed to parse or used an unsupported operation.
