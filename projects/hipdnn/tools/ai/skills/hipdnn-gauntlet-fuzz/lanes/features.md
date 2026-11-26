# Lane: feature group (`FEAT<k>`)

**Trigger:** one lane per optional feature group in the brief (bias and dBias, dropout, padding with seq_len, ragged offsets, scale
tensors, extra outputs such as stats, ...). The dispatch names the group. **Agent:** `scout`. **Output:** `R/lanes/FEAT<k>.md`, ideas.

## Method

1. The feature's semantics (brief section 1) and every attribute it adds: shapes, broadcasts, dtypes, strides, values.
2. Which combinations the matcher accepts, with rule ids; which kernels serve them (from the brief, or `R/lanes/KMAP.md` if present).
3. How the harness builds, fills and grades the feature (brief section 9): what it refuses, which reference serves it, how inputs are
   drawn (a Philox keep mask, explicit masks, seq_len tensors).
4. Crossings with other features (padding with dropout, bias with GQA, ragged with stats), with the mask, and with the edge shapes.

## Look for

Broadcast shapes on each axis, unaligned pitches, overlapping rows, extreme but legal values (probabilities near 0 and 1, offsets
near their type limits, all-padding and no-padding rows), dtype mixes the matcher allows, and a feature that changes which kernel
runs. Mark amplifiers (`1/(1-p)`, large bias) against the judgeability bound.
