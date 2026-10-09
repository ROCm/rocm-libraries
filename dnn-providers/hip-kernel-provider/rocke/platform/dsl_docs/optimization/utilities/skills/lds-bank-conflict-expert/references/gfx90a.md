# gfx90a profile

Select `gfx90a` explicitly. The profile identity remains `gfx90a` even when a
later target shares one or more prediction rules.

## Scope

- Use wave64 with the currently registered `ds_read_b32`, `ds_read_b64`,
  `ds_read_b128`, `ds_write_b32`, `ds_write_b64`, and `ds_write_b128`
  operations.
- Supply byte addresses and access widths exactly as issued by the layout being
  analyzed. Align each address to its access width (4, 8, or 16 bytes), and keep
  the full access within the 65,536-byte LDS capacity.
- Treat broadcasts, lane-phase separation, and wave-partition separation as
  opcode-scoped semantics returned by the predictor.
- Treat unavailable opcodes or wave sizes as unsupported, and access-width
  mismatches or misaligned addresses as invalid request data. Do not approximate
  either with the nearest supported operation.

Query the production profile through prediction and use its errors for operations
outside this reviewed set. Update this reference together with any reviewed profile
expansion.

## Review checklist

For a representative layout, include:

1. a known distinct-address conflict;
2. a nearby no-conflict case;
3. a same-address broadcast;
4. a lane-phase-separated case where applicable;
5. a wave-partition-separated case where applicable; and
6. an unsupported-input case.

Use the semantic result's group IDs and multiplicity in reports. Do not convert
them into timing, throughput, or a physical-bank claim.
