# Stall-gate timing reproducer

Checks whether HIP event timing is accurate with and without the hipDNN `StallGate`.
Tracks issue #12863.

## What it does

Each output row is 50 samples (after 3 warmups) of one workload, mode, and delay.

Workloads:

- `empty`: START then STOP, no work.
- `spin Nus`: one single-thread kernel that spins for N µs and times itself with
  `wall_clock64()`. Each N (0, 2, 5, 10, 20, 50, 100, 200, 1000) is a separate row.
- `memset N`: one `hipMemsetAsync` of 4 KiB to 128 MiB.

Modes (`mode` column):

- `unstalled`: START, work, STOP.
- `stalled`: `arm()`, START, work, STOP, `release()`. The GPU runs nothing until the
  release, so host time between START and the launch is excluded.

Delay (`delay` column): `0`, or a 200 µs host busy-wait after START. It stands in
for hipDNN host code. The output has one block per delay.

The spin kernel is the oracle. The event span contains the whole kernel, so a
correct event time is never shorter than the kernel's own time.

## Run it

You need a ROCm or HIP SDK install with `hipcc`, and a git checkout of this branch.

1. Get only the needed files:

   ```bash
   git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout set --no-cone /projects/hipdnn/data_sdk/include/ /projects/hipdnn/tools/stall_gate_repro/
   git checkout users/sareeder/stall-gate-repro
   ```

2. Build. Set `--offload-arch` to your GPU (for example `gfx1101`, `gfx1151`, `gfx90a`).

   Linux:

   ```bash
   hipcc -std=c++17 -O2 --offload-arch=gfx1101 -I projects/hipdnn/data_sdk/include \
       projects/hipdnn/tools/stall_gate_repro/stall_gate_repro.cpp -o stall_gate_repro -lpthread
   ```

   Windows (`hipcc` is in `%HIP_PATH%\bin`):

   ```bat
   hipcc -std=c++17 -O2 --offload-arch=gfx1101 -I projects/hipdnn/data_sdk/include ^
       projects/hipdnn/tools/stall_gate_repro/stall_gate_repro.cpp -o stall_gate_repro.exe
   ```

3. Run it, and save the output:

   ```bash
   ./stall_gate_repro 50 0 > stall_gate_repro.txt      # args: [reps=50] [device=0]
   ```

   The run takes a few seconds. Keep other GPU work off the device during the run.

4. Post `stall_gate_repro.txt` on issue #12863. Include the OS, the driver version,
   and the ROCm or HIP SDK version.

## Read the output

All times are in µs.

| Column | Meaning |
|---|---|
| `ev_min`, `ev_med`, `ev_max` | Min, median, max of the event elapsed time |
| `dev_med` | Median kernel self-timed duration (spin rows only) |
| `d_med`, `d_min` | Median and min of event minus kernel time, per sample (spin rows only) |
| `neg` | Samples with a negative event time |
| `viol` | Samples where the event time is shorter than the kernel time |
| `declined=N timedOut=N` | Printed only when nonzero: N stalled samples ran unstalled, or the watchdog released them |

How to judge the result:

- `viol > 0` or a negative `d_min` on a stalled row: the stalled timestamps are wrong.
- `d_med` changes with spin length: the error is not a constant offset, so the
  ranking of raw values is also unreliable.
- Stalled rows stay positive and `d_med` stays constant: the gate is accurate.
  Only near-empty spans need a clamp.
- `stall gate usable: no`, or `declined=` on stalled rows: the device does not
  support the gate, so stalled rows are really unstalled.

## Reference: Linux, MI210 (gfx90a)

- `neg` and `viol` are 0 on every row.
- Stalled `d_med` is 7.84 µs for every spin length.
- The 200 µs host delay adds about 205 µs unstalled and nothing stalled.
- An empty span reads 4.48 µs.
