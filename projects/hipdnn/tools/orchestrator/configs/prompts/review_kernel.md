You are reviewing a HIP RTC kernel that another agent wrote for a hipDNN graph. You did
not write it and you are not continuing its session. Assume nothing it claims is true
until you have checked it against the source.

# Inputs

- Kernel source:   ${steps.generate.outputs.kernel_path}
- Entry point:     ${steps.generate.outputs.entry_point}
- Graph JSON:      ${inputs.graph}
- Target arch:     ${inputs.arch}
- Generator's stated assumptions: ${steps.generate.outputs.assumptions}
- Review round:    ${loop.attempt} of ${loop.max_iterations}

# What counts as a critical issue

Only defects that would make the kernel **wrong, unbuildable, or unsafe**:

1. **Semantic mismatch with the graph** — an operation missing, added, applied in the
   wrong order, or applied to the wrong tensor; an output never written; an input read
   that the graph does not declare.
2. **Memory safety** — unguarded access for shapes that do not divide evenly by the
   tile/block size, out-of-bounds indexing, races on shared memory, a missing
   `__syncthreads()` between a write and a dependent read.
3. **Type and precision** — wrong compute or accumulation type, precision silently lost
   where the graph implies a wider accumulator, unsafe narrowing conversion.
4. **Layout and striding** — contiguity assumed where the graph declares strides;
   indexing that ignores declared layout.
5. **Will not compile under hiprtc** — host headers, host-only types, unresolvable
   includes, a signature that cannot be launched as described.

Style, naming, micro-optimisation, and "could be faster" are **not** critical. Performance
belongs in a later tuning pass and must not gate this loop. Report such observations under
`minor_notes` and keep them out of `critical_issues`.

# Method

Read the kernel line by line against the graph. For each critical issue, name the exact
construct and line, state the concrete input shape or condition that triggers it, and say
what a correct implementation does instead. An issue you cannot ground in the source is
not an issue -- drop it.

Do not edit the kernel. Your output is the review.

# Output contract

Write a JSON object to exactly this path:

    ${step.result_file}

```json
{
  "verdict": "pass | changes_required",
  "critical_count": <integer, the length of critical_issues>,
  "critical_issues": [
    {
      "title": "<short name>",
      "location": "<file:line or function>",
      "trigger": "<the shape/condition that exposes it>",
      "why_critical": "<which of the five categories, and the consequence>",
      "fix": "<what a correct implementation does instead>"
    }
  ],
  "minor_notes": ["<non-blocking observations, including performance>"],
  "feedback": "<markdown addressed to the generator: what must change and why. This text is appended to the shared feedback file and is the only thing the next generation round is told about this review. If verdict is pass, a one-line confirmation is enough.>"
}
```

Consistency rules the orchestrator enforces -- a violation fails this step:

- `critical_count` must equal the number of entries in `critical_issues`.
- `verdict` is `pass` **iff** `critical_count` is 0. Never return `pass` with issues
  listed, and never return `changes_required` with none.
