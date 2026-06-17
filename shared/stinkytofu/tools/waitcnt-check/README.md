# waitcnt-check

A standalone validator for `s_wait_*` (wait-count) insertion in **STIR**
(StinkyTofu IR) dumps. It parses a STIR file, simulates the per-counter
in-flight queues across the control-flow graph, and reports any consumer
that reads the result of an asynchronous memory op that has **not** been
drained by a preceding `s_wait_*`.

It is an independent cross-check of the compiler's
`StinkyWaitCntInsertionPass`: the dataflow model here mirrors
`shared/stinkytofu/src/transforms/asm/waitcnt/WaitDataflow.cpp`, so a STIR
that the pass waited correctly should validate clean.

## What it checks

Each tracked asynchronous memory op completes on one hardware counter, and
each counter is drained by its own wait instruction:

| Counter | Producers                                            | Wait op             |
| ------- | ---------------------------------------------------- | ------------------- |
| DS      | `ds_read` / `ds_load` / `ds_write` / `ds_store` / `ds_atomic` | `s_wait_dscnt`      |
| Buffer  | vector `buffer` / `global` / `flat` load + store     | `s_wait_loadcnt`    |
| KM      | scalar `s_load` / `s_buffer_load`                    | `s_wait_kmcnt`      |
| Tensor  | `tensor_load_to_lds`                                 | `s_wait_tensorcnt`  |

The validator models, per counter:

- **Per-predecessor queues.** Each block keeps one in-flight queue per CFG
  predecessor (not a single union), so a consumer at a join computes the
  strictest required wait as the `min` over paths. Queues are capped at the
  hardware in-flight window (64) so loops converge.
- **RAW** (consumer reads a register/LDS token still in flight),
  **WAR-on-LDS** and **barrier** ordering against the DS FIFO, plus the
  conservative fallbacks used by the reference pass for untagged memtokens.
- The **tensor** counter only needs to drain at a barrier (or when there is
  a single wave); see `--num-waves`.

When it finds a consumer whose producer is still in flight, it reports a
`MISSING` violation: the wait the compiler should have emitted is absent (or
too lenient / placed too late).

## Usage

```bash
python3 waitcnt_check.py [options] <file.stir>
```

By default only violations are printed. Example:

```bash
python3 shared/stinkytofu/tools/waitcnt-check/waitcnt_check.py dump_function-after.stir
```

### Options

| Option              | Description                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------- |
| `--num-waves N`     | NumWaves tile config. `0` (default) = multi-wave: the tensor counter only drains at barriers. `1` = tensor drains at every consumer. |
| `-v`, `--verbose`   | Also print the `Wait Instructions` listing (what each `s_wait_*` drained) and `Notes`.       |
| `--quiet`           | Print only the one-line summary per function.                                                |
| `-h`, `--help`      | Show help.                                                                                   |

### Exit codes

| Code | Meaning                                   |
| ---- | ----------------------------------------- |
| `0`  | PASS — no violations                      |
| `1`  | FAIL — at least one violation             |
| `2`  | Could not read or parse the input file    |

## Reading a violation

Only the **first** violation per counter/queue is reported (one diagnostic
per `s_wait_*` kind), each followed by a dump of the live queue state:

```
  MISSING: Consumer v_wmma_scale_f32_16x16x128_f8f6f4 @ line 1913 in ^label_LoopBeginL
           uses async producer ds_load_b128 @ line 1805 (s_wait_dscnt)
           still in-flight (queue depth=28, needs wait s_wait_dscnt 27)
    Queue dump for s_wait_dscnt at consumer ... in ^label_LoopBeginL:
      required (drains ALL 2 undrained producer(s) this consumer uses): s_wait_dscnt 26
      live queues (DS): 2 per-pred queue(s)
        queue[0] pred=^label_openLoopL_1 depth=28
          idx 0: ds_load_b128 @ line 1805 (drain with s_wait_dscnt 27)  <-- undrained producer
          idx 1: ds_load_b64  @ line 1806 (drain with s_wait_dscnt 26)  <-- undrained producer
          ...
```

- **`needs wait <op> N`** in the headline is per producer: the keep value
  that drains just the op named there (for an op at index `i` in a queue of
  size `n`, that is `n - i - 1`).
- **`required: <op> N`** is the single wait that drains **every** undrained
  producer the consumer reads — the `min` of the per-producer values. This is
  the value the compiler should have emitted before the consumer.
- Each queue line shows its source predecessor and, per op, the keep value
  that would drain it; `<-- undrained producer` marks the offending ops.

## Notes

- The tool is dependency-free (Python 3 standard library only).
- It reads a STIR text dump as produced by the StinkyTofu asm pipeline; it
  does not require building the compiler.
