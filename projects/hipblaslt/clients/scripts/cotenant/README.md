# Cotenant benchmarking

Run `hipblaslt-bench` (or any command) with LDS contention from a background
"cotenant" kernel.

Linux only: the kernel uses POSIX APIs (`unistd.h`, `pause()`, `usleep()`,
`getpid()`) and is not built or installed on Windows.

```bash
hipblaslt-cotenant --cus 64 -- hipblaslt-bench -m 4096 -n 4096 -k 4096
```

All options must come **before** `--`; everything after it is the command to run
(the kernel binary is found automatically, so no path is needed).

## Usage

`hipblaslt-cotenant` and its kernel `hipblaslt-cotenant-kernel` install into
`bin/` next to `hipblaslt-bench`:

- **Installed:** `hipblaslt-cotenant --cus 64 -- hipblaslt-bench ...` (on `PATH`).
- **Build tree:** `<build>/clients/hipblaslt-cotenant --cus 64 -- ...`.
- **Source checkout:** `clients/scripts/cotenant/hipblaslt-cotenant --cus 64 -- ...`
  builds the kernel on first use into `~/.cache/hipblaslt/cotenant`. Point
  `--binary` at a prebuilt kernel to skip that.

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

Written for mi300 and mi350 architectures; it might not work
correctly on other targets.

To confirm the kernel is actually executing (not just that a GPU context
exists), each workgroup increments a system-scope atomic counter in host-pinned
memory at entry; the host waits until all `N` have reported, then logs `READY`.
This is what the launcher waits for, so the command starts against full
residency without polling driver internals or guessing a settle time.

`hipblaslt-cotenant`:

1. builds the kernel on first use (arch auto-detected via `rocminfo`, override
   with `--arch`; compiler defaults to `hipcc`, override with `HIPCC=...`),
2. launches `--cus` workgroups and waits for their `READY` marker,
3. runs the command after `--` under that contention,
4. kills the cotenant when the command exits or the script is interrupted.

Pass `--cus 0` to run the command with no cotenant at all — the uncontended
baseline. Otherwise `--cus` must be at least 1 and less than the device CU count
(reported by `rocminfo`). This bound applies at every max-occupancy setting.

Useful flags: `--device N` (sets `HIP_VISIBLE_DEVICES`), `--wait` (max seconds to
wait for `READY`), `--grace` (extra settle time after residency, default 0).
