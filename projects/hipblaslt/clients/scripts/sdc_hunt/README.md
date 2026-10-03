# fast_check SDC hunt

An on-demand recipe for chasing intermittent silent data corruption (SDC), such as
ROCM-31901 and ROCM-31905, with fast_check. fast_check checks every element of D
on every launch and names the bad elements, solution and iteration, which makes it
the cheapest way to tell whether an environment change makes a failure appear.

`fast_check_sdc_hunt.py` runs hipblaslt-test's fast_check cases under each
combination of the requested environment axes and appends one JSON line per run
to a results file. Each line records the environment the run saw:

| Axis | How it varies | Recorded as |
| --- | --- | --- |
| Concurrency | `--load none gemm cotenant:<CUs> 'command:<cmd>'` runs a second GPU workload alongside: another GEMM (`hipblaslt-bench`), the CU-occupying [cotenant](../cotenant/README.md) kernel, or any command, such as a copy loop or the EDP helper | `load` |
| XNACK | `--xnack unset 0 1` sets `HSA_XNACK` (0 is the only supported mode on MI455X) | `HSA_XNACK`, plus the amdgpu `noretry` parameter |
| Preemption | the amdgpu `cwsr_enable` module parameter; changing it needs a driver reload, so run once per setting and driver (before and after the ROCM-31285 fix) | `cwsr_enable`, `amdgpu_version` |
| Placement | every buffer a failing case reports crossing a 4 GiB boundary | `buffers_crossing_4gib` |

If failures line up with buffers that cross a 4 GiB boundary, they are carry-drop
address defects (class C on AIHPBLAS-4988), and the placement cases from
AIHPBLAS-4994 reproduce them deterministically.

## Running

```bash
TEST=build/release/clients/hipblaslt-test
clients/scripts/sdc_hunt/fast_check_sdc_hunt.py --test-bin $TEST \
    --xnack unset 0 --load none gemm cotenant:64 --runs 3 \
    --results sdc_$(hostname).jsonl
```

The default filter, `*sdc_hunt*`, selects the `matmul_fast_check_sdc_hunt*` cases
in `clients/tests/data/matmul_gtest.yaml`: the Achilles GEMM sizes and data types,
each launched and checked 20 times on the heuristic's first 4 solutions. Their
category, `sdc_hunt`, is in no tier, so they run only when asked for. Pass
`--filter` to run any other fast_check cases. For Stream-K solutions, export
`TENSILE_SOLUTION_SELECTION_METHOD=2` before running (it is recorded).

The script exits non-zero if any run failed. A failing run's full output is saved
next to the results file, under a name that includes the invocation's start time,
and named in the record's `log` field.

A run is not counted as clean unless its load really ran: the script waits for the
cotenant's `READY` (up to `--load-ready-seconds`), or gives other loads
`--load-settle-seconds`, and records a load that exits before the run as
`load_failed`. A load that stops during the run is recorded as
`load_ran_throughout: false` and fails the run. The cotenant's kernel log is
kept next to the results when a run fails. A run in which no test ran, usually a stale binary or
a mistyped `--filter`, also fails.

This is for dedicated hardware, not CI: the runs hold the GPU for a long time, and
on gfx1250 a page fault can leave the GPU unusable until it is reset (ROCM-32049).
