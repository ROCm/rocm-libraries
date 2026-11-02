# Cotenant benchmarking

Run `hipblaslt-bench` (or any command) with LDS contention from a background
"cotenant" kernel.

The standalone launcher and `hipblaslt-bench --cotenant-cus` are Linux only; the launcher is not
built or installed on Windows, and the bench rejects `--cotenant-cus` there.

```bash
hipblaslt-cotenant --cus 64 -- hipblaslt-bench -m 4096 -n 4096 -k 4096
```

All options must come **before** `--`; everything after it is the command to run
(the kernel binary is found automatically, so no path is needed).

## Usage

The hipblaslt clients build produces `hipblaslt-cotenant-kernel` and copies
`hipblaslt-cotenant` beside it; both install into `bin/` next to `hipblaslt-bench`:

- **Installed:** `hipblaslt-cotenant --cus 64 -- hipblaslt-bench ...` (on `PATH`).
- **Build tree:** `<build>/clients/hipblaslt-cotenant --cus 64 -- ...`.

The launcher uses the kernel beside it; pass `--binary PATH` to use another one.
Like `hipblaslt-bench`, the kernel runs only on the build's `GPU_TARGETS`; on
another GPU it exits before `READY` and the launcher reports that it did not
become resident.

`hipblaslt-bench` itself needs the ROCm runtime libraries on the loader path; if
it fails to start with a `libomp.so` error, export
`LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/llvm/lib`.

## How it works

`hipblaslt-cotenant-kernel` launches `--cus` persistent, compute-free workgroups.
By default, each reserves the entire per-CU LDS as dynamic shared memory, so
each lands on a distinct CU and leaves no LDS for a GEMM workgroup.

`--max-occupancy N` (1–64, default 1) follows TensileLite's LDS-based occupancy
control: each workgroup reserves approximately `LDS per CU / N`. Use 2 for half
the LDS or 4 for a quarter:

```bash
hipblaslt-cotenant --cus 64 --max-occupancy 4 -- hipblaslt-bench -m 4096 -n 4096 -k 4096
```

The allocation rounds **down** to a 256-byte granule, adding one granule if
the result would allow more than `N` workgroups per CU. For example, 64 KiB / 3
rounds to 21,760 bytes, allowing three workgroups based on LDS alone.
The reservation must fit the per-block limit. Allocation granularity and other
resources can still lower the achievable occupancy.
The log reports the reserved bytes and HIP's theoretical maximum blocks per CU.
This flag changes only LDS reservation; register usage and block size stay fixed.
For values above 1, workgroups may share CUs, so `--cus` is not a guarantee of
distinct CUs. `READY` confirms workgroup residency, not distinct-CU placement.
The wave-slot limit also caps occupancy (8 blocks per CU on gfx950 with 256-thread
blocks), so values above that reserve less than `LDS per CU / N` per CU in
effect; the launcher logs a warning when the reported limit is below `N`.

Written for mi300 and mi350 architectures; it might not work
correctly on other targets.

To confirm the kernel is actually executing (not just that a GPU context
exists), each workgroup increments a system-scope atomic counter in host-pinned
memory at entry; the host waits until all `N` have reported, then logs `READY`.
This is what the launcher waits for, so the command starts against full
residency without polling driver internals or guessing a settle time.

`hipblaslt-cotenant`:

1. launches `--cus` workgroups and waits for their `READY` marker,
2. runs the command after `--` under that contention,
3. kills the cotenant when the command exits or the script is interrupted.

Pass `--cus 0` to run the command with no cotenant at all — the uncontended
baseline. Otherwise `--cus` must be at least 1 and at most the device CU count
(reported by `rocminfo`). This bound applies at every max-occupancy setting.
Very high contention can stall a GEMM, either because too few CUs remain or because
the LDS left on each CU is smaller than the GEMM workgroups need. On an MI355X
(256 CUs, f16 4096³), `--cus 128` and `--cus 200` at max occupancy 3 and `--cus 256`
at max occupancy 6 ran, while `--cus 255` at max occupancy 1 and `--cus 256` at
max occupancy 3 stalled. The outcome depends on the GEMM kernel. If a run stalls,
reduce the workgroup count or increase max occupancy to reserve less LDS per
workgroup. This applies to both launcher and same-process use.

Useful flags: `--device N` (sets `HIP_VISIBLE_DEVICES`), `--wait` (max seconds to
wait for `READY`), `--grace` (extra settle time after residency, default 0).

## Same-process benchmarking

To run the same cotenant in `hipblaslt-bench` on a separate, nonblocking stream:

```bash
hipblaslt-bench --cotenant-cus 64 --cotenant-max-occupancy 4 -m 4096 -n 4096 -k 4096
```

`--cotenant-cus` defaults to 0 (disabled); otherwise it must not exceed the
selected device's CU count. `--cotenant-max-occupancy` defaults to 1 and uses
the same LDS calculation and 1–64 range as the launcher. Use the benchmark's
`--device` option to select the GPU.

For each candidate solution, the cotenant starts after setup and waits for all
workgroups to become resident (30-second timeout). It stays active through warmup,
skip-slow screening, and fixed or adaptive timing, then stops before validation
and cleanup. Skipping a solution also stops its cotenant. Startup and shutdown
are outside the GEMM timing interval.

The in-process kernel checks a host-pinned stop flag once per wave after
every 256 sleeps. After measurement, a one-thread kernel on the GEMM stream
sets that flag. Shutdown waits for the next check, outside GEMM timing. If that
kernel cannot be launched or does not run within 30 seconds, the host sets the
flag directly. If the cotenant still does not stop, the benchmark prints an
error and exits with status 1 rather than continue under unintended contention.
The standalone kernel sleeps until its process is killed. Both share the
same kernel source and LDS reservation logic.
