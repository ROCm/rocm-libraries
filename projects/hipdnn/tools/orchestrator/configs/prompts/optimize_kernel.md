You are making an existing hipDNN kernel-ingestor engine faster.

- Engine:           ${inputs.engine_name}
- Graph:            ${inputs.graph}
- Arch:             ${inputs.arch}
- Attempt:          ${loop.attempt} of ${loop.max_iterations}
- Operator notes:   ${inputs.notes}

The engine is already integrated and already passes its tests. Your job is speed, and
the run ends when it still passes everything it passed before — not when a number
improves. That is not permission to skip the optimisation; it is the acknowledgement
that a round which measured honestly and found nothing is a result, and a round that
got faster by getting wronger is not.

# Read these first

- **`${vars.authoring_file}`** — the kernel's admitted-shape envelope and its launch
  ABI. You may change how the kernel computes; changing WHAT it admits is a different
  job with a different flow, and the matcher and descriptors are gated against this
  contract.
- **`${loop.feedback_path}`** — one section per failed round, naming the failing step
  and its log directory. Open that directory; the one-line note is not the evidence.
- **The previous round's benchmark**, if there was one:
  `${loop.previous.bench_read.outputs.feedback}`

## You are probably not starting from nothing

This is attempt ${loop.attempt}, and nothing a previous attempt wrote has been reverted.
Before you change anything:

```
git -C ${vars.repo_root} status --short     # what previous attempts already changed
ls ${vars.ingestor_dir}                     # the pack, its kernels, its descriptors
```

A previous round may have landed a change that was correct but unmeasured, or measured
and rejected. Read the feedback and the benchmark summary before you re-derive a
conclusion somebody already paid for.

# What the numbers mean

Each round runs `dnn-benchmark` on this engine alone and flattens one row for you:

- **`kernel_median_ms`** — GPU kernel time. This is the number you are optimising.
- **`host_median_ms`** — submit plus drain. It has a floor of roughly 0.03 ms on this
  class of device; if the two are close, you are measuring the launch path and not the
  kernel.
- **`oracle_speedup`** — hipDNN auto-tuned the plan selection and this is
  `heuristic / tuned`. **It is the yardstick, and it tells you where the win is not.**
  A value near 1.00 means the heuristic is already picking the best compiled plan, so
  there is nothing to win in selection and the work is in the kernel itself. A value
  meaningfully above 1.00 means the engine has good variants it is not choosing, and
  the cheapest win available is in ranking or knobs rather than in new code.
- **`measurable`** — 0 means the graph is too small for any of this to mean anything.
  The round fails on it rather than reporting noise as progress.

An optimisation that only moves `host_median_ms` has not made the kernel faster.

# What you may change

Under `${vars.ingestor_dir}`: the kernel sources, the pack's dispatch and launch
geometry, knob defaults, descriptor selection metadata. You may add bundle cases.

**You may not** widen the admitted shape envelope, change the launch ABI the authoring
contract declares, edit another engine, or touch the integration test suite, its
category YAMLs or any existing bundle case. The scope guard fails the round on any of
those, and it is checked, not trusted.

# Optimise like an engineer, not like a search

State a hypothesis before you change code, and say in `summary` whether the
measurement supported it. "Tried things until the number moved" is how a run produces
a kernel nobody can reason about and a speed-up that evaporates on the next shape.

Reasonable places to look, in rough order of how often they pay:

1. **Memory access** — coalescing, vectorised loads, avoiding strided reads the layout
   does not require, LDS staging and bank conflicts.
2. **Launch geometry** — block size and tile shape against occupancy and the shapes the
   envelope actually admits, not against the one graph in front of you.
3. **Arithmetic** — removing redundant address math from inner loops, hoisting
   invariants, the accumulator type the graph implies.
4. **Selection** — only when `oracle_speedup` says the headroom is there.

Every one of those is a claim you can check with a measurement. Make the change, run
the round, and let the number decide.

# Output contract

Write a JSON object to exactly this path:

    ${step.result_file}

It is the same integration contract shape the engine already has — the gates re-check
it against the checkout, so it must describe the tree as it now stands. Carry forward
the fields that have not changed, and make sure `changed_files` lists everything you
touched this round.

In `summary`, say what you hypothesised, what you changed, what the measurement did,
and — if the numbers did not move — say that plainly. A round that reports an honest
null result is worth more than one that claims a win the next round cannot reproduce.
