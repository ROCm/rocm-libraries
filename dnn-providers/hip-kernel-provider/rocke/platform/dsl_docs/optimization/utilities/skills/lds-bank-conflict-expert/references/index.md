# Reference index

## Choose the input

Provide:

- an exact registered target;
- one supported `ds_*` opcode;
- the opcode's wave size; and
- access records with stable IDs, lane indices, LDS byte addresses, access widths,
  active state, and optional logical coordinates.

Do not convert byte addresses to elements or dwords before calling the predictor.
If a layout starts in another unit, perform that conversion explicitly and retain
the resulting byte address in the request.

`scripts/predict.py` accepts a JSON object with `target`, `opcode`, `wave_size`,
and `accesses`, plus optional `coordinate_axes`. Each access object uses the
fields described in [model-contract.md](model-contract.md). Pass `-` as the input
path to read the request from standard input.

## CLI invocation

Use the Python interpreter from an environment where `rocke` is importable. From
this skill's directory, read a request file or standard input with:

```text
python scripts/predict.py request.json
python scripts/predict.py -
```

The provider test artifact carries a copy under `tests/analysis/`. From
`bin/hip_kernel_provider`, with that directory on `PYTHONPATH`, invoke it with:

```text
python tests/analysis/lds-bank-conflict-expert/predict.py request.json
```

Both copies are invoked through Python and do not require executable permissions.

## Choose the reference

- Read [model-contract.md](model-contract.md) for access, result, diagnostic, and
  serialization semantics.
- Read [gfx90a.md](gfx90a.md) for the initial `gfx90a` profile scope.
- Read [gfx942.md](gfx942.md) for the explicit `gfx942` profile scope.
- Read [comparison.md](comparison.md) before comparing target profiles.
- Read [validation-boundaries.md](validation-boundaries.md) before describing what
  a prediction proves.

## Choose the output

- Use the [CLI invocation](#cli-invocation) above for canonical JSON suitable for
  review, fixtures, and downstream analysis.
- Consume the production Python API directly when integrating with another rocKE
  tool. Keep analysis downstream of the serialized semantic boundary.

An unsupported target or operation is a result to report and correct. It is not a
reason to guess a profile or rewrite the request.
