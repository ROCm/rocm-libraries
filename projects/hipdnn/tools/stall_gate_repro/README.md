# Stall-gate timing reproducer

Checks whether HIP event timing is accurate with and without the hipDNN `StallGate`.
Tracks issue #12863.

## What it does

The program times these workloads, 50 samples each:

- An empty span (START then STOP).
- A spin kernel of 0 to 1000 µs. The kernel times itself with `wall_clock64()`.
- `hipMemsetAsync` of 4 KiB to 128 MiB.

Each workload runs unstalled and stalled. Each runs with no host delay and with a
200 µs host delay after START.

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

All times are in µs. `delta = event - device` exists only for spin rows.

| Column | Meaning |
|---|---|
| `ev_min`, `ev_med`, `ev_max` | Event elapsed time |
| `dev_med` | Kernel self-timed duration |
| `d_med`, `d_min` | Event minus kernel time |
| `neg` | Samples with a negative event time |
| `viol` | Samples where the event time is shorter than the kernel time |

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
