# Lane: backend and serialization path (`BACK`)

**Trigger:** always. **Agent:** `scout`. **Output:** `R/lanes/BACK.md`, ideas.

Find values that survive the frontend but reach the plugin changed (brief section 3).

## Look for

- Optional vs zero: a field that is absent vs explicitly 0 or default, and which the engine reads.
- Narrowing: int64 to int32 dims, strides and offsets; float to half for scalars; pass-by-value scalars and their declared types.
- Uid handling: collisions, virtual tensors, tensors the engine does not expect, unusual stride orders, `override_shape`.
- Engine selection and applicability caches: the same graph twice, then a near-identical one, in one process; plan and workspace
  reuse across cases (a batch-dependent failure is the symptom).
- Plugin version and schema floors: a property the engine does not read reaching it.

## Ideas

`Expect: BACKEND-MANGLE` with the line where the value changes. Cache and reuse ideas give a graph sequence, not one graph, and
usually `Place: hand-probe` or a `--batch-size` experiment: say which.
