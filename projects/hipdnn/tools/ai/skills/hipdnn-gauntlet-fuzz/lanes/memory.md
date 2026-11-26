# Lane: memory and layout (`MEM`)

**Trigger:** the op takes strided tensors, or the engine sizes a workspace. **Agent:** `scout`. **Output:** `R/lanes/MEM.md`, ideas.

## Look for

- Strides: padded leading strides, every stride order the matcher accepts, zero strides on unit axes, overlapping views where legal.
- Declared alignment against real placement (`alignment`, `placement` axes: `exact`, `under`); a pointer misaligned on purpose must
  be rejected (`BAD_PARAM`), never computed.
- Workspace sizing: the size the plan asks for against what the kernels touch, at the largest shapes the matcher accepts.
- Very large tensors: element counts and byte offsets past 2^31 and 2^32, per head and per batch, against 32-bit index arithmetic
  (brief section 7) and the harness fill limits (brief section 9; `HIPDNN_TEST_HOST_INPUT_FILL=1` fills inputs on the host for a
  tensor the device fill does not handle).
- Aliasing between inputs and outputs where the API allows it.

Large-tensor ideas go to probes (no dumps), with the enforcement level they need: `full` when the engine may accept.
