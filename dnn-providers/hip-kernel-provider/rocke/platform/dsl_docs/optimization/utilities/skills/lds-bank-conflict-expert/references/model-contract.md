# Model contract

## Request

Call the production predictor with an exact target, opcode, wave size, and a
sequence of access records. An access is the fundamental record because one lane
may contribute multiple logical elements or vector fragments.

Each access has:

- a stable non-negative `access_id`;
- a lane within the selected wave;
- a non-negative `lds_byte_address`;
- a positive `access_width_bytes`;
- an active flag; and
- optional integer coordinates whose axis names are supplied separately by the
  result contract.

Use unique IDs even when multiple accesses share a lane or byte address. Preserve
inactive accesses when the caller needs them displayed; the predictor excludes
them from collision grouping.

## Result

Treat the returned semantic document as authoritative. It contains:

- independent schema and profile versions;
- the exact selected profile identity;
- the normalized request;
- stable access records and classifications;
- conflict and broadcast groups referencing access IDs;
- counts and maximum multiplicity; and
- stable diagnostics.

Conflict multiplicity is the number of accesses in a semantic conflict group. It
is not a performance estimate or a profiler counter prediction.

## Classification

- `normal`: active and not a member of a reported group;
- `conflict`: member of a distinct-address conflict group;
- `broadcast`: member of a supported same-address broadcast group; and
- `inactive`: retained for presentation but excluded from prediction.

Do not collapse broadcasts into conflicts. Do not infer normality from an empty
group list if prediction failed or the request was unsupported.

## Serialization

Use the production canonical JSON functions. They provide deterministic ordering,
JSON primitives only, and strict schema-version validation. Do not serialize
dataclass internals manually or use tuple-keyed JSON objects. Analyze an in-memory
result and its canonical JSON round trip with identical group and classification
semantics.
