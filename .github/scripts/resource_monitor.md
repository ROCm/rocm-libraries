# CPU and RAM monitoring for Linux CI commands

`resource_monitor.py` runs a command, samples resource use every five seconds,
and preserves the command's exit status. It uses Python's standard library.
The output is a JSON summary plus a CSV time series with the same filename stem.
When `GITHUB_STEP_SUMMARY` is set, it also writes a table into the Actions job
summary. A CSV row is flushed after every sample so partial data remains if a
later CI step fails.

```bash
python3 .github/scripts/resource_monitor.py \
  --phase "hipRAND hardware test" \
  --output parity/resources-hardware.json -- \
  ctest --test-dir build -L '^standard$' --output-on-failure
```

Use the same command prefix around a rocJITsu test or a hardware test. The
`hiprand-rocjitsu-parity.yml` workflow records its build and test phases and
uploads the files through its existing `parity/` artifact step. Other workflows
can upload the selected output directory with `actions/upload-artifact`.

## Measurement scope

The default `--scope auto` uses cgroup CPU and memory counters inside a job
container. This measures the container's CPU use and RAM charge, including
processes and page cache. CPU is reported in cores: `1.0` means one CPU core
was fully occupied during the sampling interval. RAM is the sampled cgroup
memory usage, so it may include cached files from earlier steps in the same
container.

On a hardware runner without a job container, auto mode samples the command's
process tree through `/proc`. This includes its child processes, such as tests
started by `ctest`. Summed process RSS can count shared pages more than once.
Short-lived processes between samples may be missed in the time series. The
final CPU time also uses the operating system's child-process accounting.

Use `--scope cgroup` to require cgroup counters or `--scope process` to force
process-tree sampling. `--interval 1` gives finer peaks at the cost of more
sampling overhead. The monitor works with CPU rocJITsu emulation and physical
GPU hardware jobs; it measures CPU and RAM, not GPU utilization or VRAM.

## Output fields

The JSON contains `duration_seconds`, `cpu_seconds`, `average_cpu_cores`,
`peak_cpu_cores`, `peak_memory_bytes`, `peak_process_rss_bytes`, `exit_code`,
and `scope`. The CSV contains timestamped CPU, RAM, and process RSS samples.
The recorded peaks are sampling peaks, so they may be lower than brief spikes.
