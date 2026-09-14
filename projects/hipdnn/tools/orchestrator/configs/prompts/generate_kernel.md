You are writing a HIP RTC kernel that satisfies a hipDNN graph.

# Task

Read the graph and produce a single HIP source file, compilable with hiprtc, whose
kernel performs every operation the graph describes, for the stated architecture.

- Graph JSON:        ${inputs.graph}
- Target arch:       ${inputs.arch}
- Write the kernel to: ${vars.kernel_dir}/kernel.hip
  (create the directory if it does not exist)
- Attempt:           ${loop.attempt} of ${loop.max_iterations}
- Additional constraints from the operator: ${inputs.notes}

# Prior attempts

Previous kernel (empty on the first attempt): ${loop.previous.generate.outputs.kernel_path}

Review feedback so far is in ${loop.feedback_path}. **Read that file before writing
anything.** If it contains issues from a previous iteration, you are fixing the existing
kernel at the path above, not starting over: keep what was correct, address every
critical issue by name, and say in `assumptions` which ones you addressed and how.

# Required reading before you write code

- The graph JSON itself: node types, tensor UIDs and how they connect, data types,
  layouts, strides, alignment, and which tensors are inputs vs outputs vs intermediates.
- Do not invent operations the graph does not contain, and do not drop ones it does.
- Where the graph is ambiguous, pick the interpretation hipDNN uses elsewhere in
  ${vars.hipdnn_root} and record the choice in `assumptions`.

# Correctness requirements

- Every graph output tensor must be written; every graph input must be read as declared.
- Respect declared strides and layouts. Do not assume contiguity unless the graph says so.
- **Handle the declared shape family, not just the shapes in this graph file.** The graph
  carries one concrete set of dimensions; the kernel must be correct for every shape the
  graph's declared types, layouts and rank admit, including shapes that do not divide
  evenly by your block/tile size. Specialising to the literal dimensions in the file is a
  correctness defect, not an optimisation. Guard every global memory access.
- Use the exact compute and accumulation types the graph specifies. Where the graph
  implies a higher-precision accumulator (e.g. fp16 in, fp32 accumulate), use it.
- The kernel must compile as a standalone hiprtc translation unit: no host headers, no
  host-only types, no external includes hiprtc cannot resolve.

# Output contract

Write a JSON object to exactly this path, and nothing else to stdout that matters:

    ${step.result_file}

Schema (all keys required):

```json
{
  "kernel_path": "<absolute path to the .hip file you wrote>",
  "entry_point": "<name of the __global__ kernel function>",
  "launch_config": {
    "grid": "<how the caller should compute grid dimensions>",
    "block": "<block dimensions>",
    "shared_bytes": "<dynamic shared memory required, or 0>"
  },
  "assumptions": ["<each interpretation you had to choose, one per entry>"],
  "addressed_issues": ["<critical issues from feedback you fixed; [] on first attempt>"]
}
```

`kernel_path` must be a file that exists when you finish. The orchestrator checks it;
a path that was never written fails this step and the attempt is retried.
