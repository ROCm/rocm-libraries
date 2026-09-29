# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Poor-man's sampling profiler: repeatedly attach gdb to a running process and record
the main thread's backtrace. The target is stopped while gdb attaches, so samples are
unbiased in program position even though wall time stretches.

Usage: python gdb_sample.py <out.txt> <max_samples> <interval_s> -- <cmd...>
The target runs <interval_s> between attaches; pick it so samples span the whole run
(native runtime / interval < max_samples), or late phases go unsampled.
"""
import subprocess
import sys
import time


def main():
    out_path, max_samples, interval = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    cmd = sys.argv[sys.argv.index("--") + 1 :]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    time.sleep(0.05)
    n = 0
    with open(out_path, "w") as out:
        while proc.poll() is None and n < max_samples:
            r = subprocess.run(
                [
                    "gdb",
                    "-nx",
                    "-batch",
                    "-p",
                    str(proc.pid),
                    "-ex",
                    "set pagination off",
                    "-ex",
                    "set print frame-arguments none",
                    "-ex",
                    "set print frame-info short-location",
                    "-ex",
                    "bt",
                ],
                capture_output=True,
                text=True,
            )
            frames = [line for line in r.stdout.splitlines() if line.startswith("#")]
            if frames:
                out.write(f"=== sample {n}\n" + "\n".join(frames) + "\n")
                n += 1
            time.sleep(interval)
    result = proc.communicate()[0].decode(errors="replace").strip()
    print(f"samples={n} target_output={result}")


if __name__ == "__main__":
    main()
