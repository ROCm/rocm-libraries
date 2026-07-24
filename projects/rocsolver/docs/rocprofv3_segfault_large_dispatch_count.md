# rocprofv3 segfault with large kernel dispatch counts

## Summary

rocprofv3 1.1.0 crashes with SIGSEGV inside `aqlprofile_pmc_iterate_data` when the
profiled application launches more than approximately 16 000 kernel dispatches in a
single run. The failure is reproducible with counter collection (`-i`), independent
of the number of counters requested, and affects both the `-f csv` and default output
modes.

## Environment

| Component | Value |
|---|---|
| Tool | `rocprofv3` 1.1.0 (`git_revision: c2d94761`) |
| ROCm | 7.2.3 |
| GPU | AMD MI300A (`gfx942`) |
| OS kernel | Linux 6.8.0-31-generic |
| Host kernel (rocprofv3 log) | Linux 5.15.0-151-generic |

## Reproduction steps

Profile `rocsolver_syevd` (FP32, `--uplo L`) for increasing matrix sizes using
`rocsolver-bench` with `--iters 1` (which internally runs 3 calls: 1 hot + 2 cold):

```bash
cat > /tmp/pmc.txt << 'EOF'
pmc: SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_TRANS_F32 SQ_INSTS_VALU_MFMA_MOPS_F32 TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum GRBM_GUI_ACTIVE
EOF

for N in 512 1024 2048; do
  rocprofv3 -i /tmp/pmc.txt -d /tmp/out_n${N} -f csv \
    -- rocsolver-bench -f syevd --uplo L -n ${N} --perf 1 --iters 1 -r s
  echo "n=${N}: exit $?"
done
```

## Observed behaviour

| n | Dispatches (total) | Outcome |
|---|---|---|
| 512 | ~8 000 | **Success** — CSV written in ~4 s |
| 1024 | ~16 000 | **Success** — CSV written in ~8 s |
| 2048 | ~31 000 | **SIGSEGV** — no output written |

The crash happens after the workload exits, during rocprofv3's counter-data
serialisation phase. The fault address (`0x40000000008`, `0x204000000008`) is
consistently far outside any mapped region, suggesting a buffer that was sized for
the n=1024 dispatch count is overflowed or dereferenced past its end at n=2048.

Crash stack (representative, addresses vary per run):

```
*** SIGSEGV (@0x40000000008) received by PID ... ***
    aqlprofile_pmc_iterate_data
    (librocprofiler-sdk-tool.so internals)
```

The crash is not sensitive to:
- Number of counters in the input file (reproduced with as few as 1 counter)
- Output format (`-f csv` vs default)
- `ROCPROFV3_PMC_BUFFER_SIZE` environment variable
- Using `/opt/rocm-7.2.3/bin/rocprofv3` vs `/usr/bin/rocprofv3` (same binary)

## Workaround

Use `rocprof` (v1) with `--stats`, which handles tens of thousands of dispatches
without issue and produces an output CSV with all the same hardware counters plus
per-dispatch GPU timing (`BeginNs`, `EndNs`, `DurationNs`):

```bash
cat > /tmp/pmc_v1.txt << 'EOF'
pmc : SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_TRANS_F32
pmc : SQ_INSTS_VALU_MFMA_MOPS_F32 TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum
EOF

rocprof -i /tmp/pmc_v1.txt -o out_n2048.csv --stats \
  rocsolver-bench -f syevd --uplo L -n 2048 --perf 1 --iters 1 -r s
# "31327 contexts collected" — exits cleanly
```

Note that rocprof v1 splits counters across multiple passes automatically (one `pmc`
line per pass). Cross-validation against rocprofv3 for n=512 and n=1024 confirms that
both tools produce identical FLOPs and within ~2% HBM bytes.

## Counter name compatibility

All counter names used above (`SQ_INSTS_VALU_*`, `TCC_EA0_*`) are accepted by both
rocprofv3 and rocprof v1 on gfx942 without modification.

## Notes for rocprofv3 maintainers

- The fault address pattern (`0x4000_0000_0008`) suggests a pointer offset calculation
  that overflows or wraps at a fixed dispatch-count boundary between ~16 000 and
  ~31 000 dispatches.
- `--kernel-iteration-range` does not accept range syntax (`1:10000`); it expects a
  single integer and is not a viable workaround for limiting data volume.
- `--collection-period` was also tested and hung indefinitely at n=2048 (no output,
  required SIGKILL).
- The issue appears in both the counter-collection path and the output-serialisation
  path — no partial output is written when the crash occurs.
