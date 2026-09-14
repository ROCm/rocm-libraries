You are writing a HIP RTC kernel that satisfies a hipDNN graph.

**This is a loop exercise.** Its purpose is to produce a first kernel with real,
findable defects, then repair it from review feedback. Follow the attempt rules below
exactly -- they override the usual "write it properly the first time" instinct.

# Inputs

- Graph JSON:        ${inputs.graph}
- Target arch:       ${inputs.arch}
- Write the kernel to: ${vars.kernel_dir}/kernel.hip
  (create the directory if it does not exist)
- Attempt:           ${loop.attempt} of ${loop.max_iterations}
- Previous kernel (empty on the first attempt): ${loop.previous.generate.outputs.kernel_path}
- Review feedback:   ${loop.feedback_path}

**Read the feedback file first.** Its contents decide which rule below applies.

# Attempt rules

## If the feedback file has no issues in it yet (first attempt)

Write the most naive kernel that produces the right answer *for exactly the shapes in
this graph and nothing else*. Specifically:

- Hard-code every dimension, stride and element count as literal integer constants read
  straight out of the graph JSON. No parameters, no runtime shape arguments, no
  `#define`s computed from inputs.
- Assume the tensors are contiguous and row-major, whatever the graph's `strides` say.
- Use one fixed block size and compute the grid as if the element count divides evenly
  by it. Do not emit a bounds guard.
- Accumulate in the storage type. Do not widen to fp32 for accumulation.
- Ignore any layout, padding or transposition subtleties; the straight-line
  interpretation is enough for this attempt.

Do not annotate the file with apologies or "TODO: handle general case" comments. Write
it as if this were the finished kernel. The reviewer must find the defects by reading
the code, not by reading a confession.

In `assumptions`, record the shortcuts you took, one per entry.

## If the feedback file lists critical issues (every later attempt)

Now write it properly. Edit the existing kernel at the path above -- keep what was
correct and fix what was named:

- Address **every** critical issue in the feedback by name. Do not defer any.
- Handle the full declared shape range, including shapes that do not divide evenly by
  your block size. Guard every global memory access.
- Respect the declared strides and layout exactly. Do not assume contiguity.
- Use the compute and accumulation types the graph specifies; where it implies a wider
  accumulator (fp16 in, fp32 accumulate), use it.
- Keep it a standalone hiprtc translation unit: no host headers, no host-only types, no
  includes hiprtc cannot resolve.

List each issue you fixed in `addressed_issues`, saying what you changed.

# Output contract

Write a JSON object to exactly this path:

    ${step.result_file}

```json
{
  "kernel_path": "<absolute path to the .hip file you wrote>",
  "entry_point": "<name of the __global__ kernel function>",
  "launch_config": {
    "grid": "<how the caller should compute grid dimensions>",
    "block": "<block dimensions>",
    "shared_bytes": "<dynamic shared memory required, or 0>"
  },
  "assumptions": ["<shortcuts taken, or interpretations chosen>"],
  "addressed_issues": ["<critical issues from feedback you fixed; [] on first attempt>"]
}
```

`kernel_path` must exist when you finish; the orchestrator checks it.
