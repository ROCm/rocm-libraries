# Replace artificial FP32 padding snapshot inputs

## Status

Accepted

## Context

The FP32/XF32 snapshot sweep varies MacroTile independently of MIWaveTile and
MIWaveGroup. It also supplies MIInputPerThread and LocalReadVectorWidth without
deriving a Solution. Most combinations do not describe a kernel, and the XF32
row uses the native FP32 MatrixInstK value.

## Decision

Remove those four snapshot entries. Extend the existing FP32/XF32 codegen test
with representative, self-consistent MatrixInstruction configurations and
explicit expected padding for A and B. Derive the remaining parameters through
Solution, check the read width, and require assembly generation to succeed.

## Consequences

The regression tests protect padding choices for realizable configurations
instead of pinning arbitrary helper inputs. Keep the focused cost, offset, and
guard tests; leave the other datatype snapshots unchanged. This replacement
does not change the padding algorithm or establish GPU numerical correctness.
