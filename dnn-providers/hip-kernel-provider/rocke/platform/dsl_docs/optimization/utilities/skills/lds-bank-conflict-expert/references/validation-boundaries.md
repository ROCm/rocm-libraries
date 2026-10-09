# Validation boundaries

## What prediction establishes

Within a registered profile's documented scope, prediction establishes the
model's deterministic classification of supplied logical LDS accesses. It can
identify semantic broadcasts, distinct-address conflict groups, inactive records,
and unsupported or invalid inputs without a GPU.

## What prediction does not establish

Do not use a result as proof of:

- physical bank count, bank width, port topology, or arbitration mechanism;
- instruction latency, stall cycles, throughput, or achieved performance;
- behavior of an unregistered opcode, width, alignment, or wave mode;
- correctness of the caller's lane-to-address extraction; or
- current hardware validation merely because a profile exists.

Collision periods and lane phases are scoped prediction rules. Multiple physical
explanations may produce the same observed collision behavior.

## Validation layers

Keep these layers distinct:

1. Validate request construction against the originating layout or instruction.
2. Validate predictor golden cases on CPU.
3. Validate canonical serialization and deterministic ordering.
4. Validate downstream analysis against result group membership and multiplicity.
5. Validate hardware behavior separately when suitable public methodology and
   authorized hardware are available.

Model-level multi-access grouping tests validate the defined group-cardinality
semantics. They do not retroactively prove how a hardware counter scales for the
same group.

## Public reporting

State the target, opcode, wave size, address units, profile version, and scoped
result. Keep private evidence, raw profiler records, internal locations, job
metadata, and software performance figures out of public artifacts.
