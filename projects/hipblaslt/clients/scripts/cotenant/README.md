# Cotenant benchmarking

Run `hipblaslt-bench` (or any command) while a fixed number of CUs are occupied
by a background "cotenant" kernel, to measure GEMM performance under CU
contention.

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

`hipblaslt-cotenant-kernel` is a persistent, compute-free kernel. It reserves the
**entire** per-CU LDS as dynamic shared memory for a single block, sized from a
runtime device query (no per-architecture constants). This does two things at
once: only one cotenant block fits per CU (so a grid of `N` workgroups occupies
exactly `N` CUs), and **zero LDS remains** for anything else — so the hardware
cannot co-schedule a benchmarked-kernel workgroup onto a cotenant-occupied CU
either. `N` is therefore an exact count of CUs *removed* from the benchmarked
command, not merely "one competing block per CU".

Reserving only *half* the LDS still pins the cotenant to one block per CU, but
leaves ~half the LDS free — enough for a low-LDS kernel (e.g. a shallow-`K` GEMM)
to co-reside on a cotenant CU and run unaffected, silently under-reporting
contention. Full-LDS reservation closes that gap; on gfx942/gfx950 it is
permitted because a single block may take all of the per-CU LDS.

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
2. launches it on `--cus` CUs and waits for its `READY` marker,
3. runs the command after `--` under that contention,
4. kills the cotenant when the command exits or the script is interrupted.

Pass `--cus 0` to run the command with no cotenant at all — the uncontended
baseline. Otherwise `--cus` must be at least 1 and less than the device CU count
(reported by `rocminfo`); occupying every CU would leave none for the benchmark,
so that is rejected.

Useful flags: `--device N` (sets `HIP_VISIBLE_DEVICES`), `--wait` (max seconds to
wait for `READY`), `--grace` (extra settle time after residency, default 0).
