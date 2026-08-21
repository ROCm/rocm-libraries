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

## Choose the reference

- Read [model-contract.md](model-contract.md) for access, result, diagnostic, and
  serialization semantics.
- Read [gfx90a.md](gfx90a.md) for the `gfx90a` profile scope.
- Read [gfx950.md](gfx950.md) for the width- and direction-sensitive `gfx950`
  profile scope.
- Read [comparison.md](comparison.md) before comparing target profiles.
- Read [validation-boundaries.md](validation-boundaries.md) before describing what
  a prediction proves.

## Choose the output

- Use `scripts/predict.py` for canonical JSON suitable for review, fixtures, and
  downstream analysis.
- Consume the production Python API directly when integrating with another rocKE
  tool. Keep analysis downstream of the serialized semantic boundary.

An unsupported target or operation is a result to report and correct. It is not a
reason to guess a profile or rewrite the request.
