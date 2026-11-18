#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stand-in for tensilelite-gpu-probe: usage is fake_gpu_probe.py <event log> [exit codes...].

It appends one JSON line {"probe": true} to the event log, which the fake client also appends to,
so the tests can check the order of probes and launches. Its exit code is the entry of the code
list that matches its run number (0 when the list is exhausted).
"""

import json
import sys


def main():
    log_path, codes = sys.argv[1], [int(code) for code in sys.argv[2:]]
    with open(log_path, "a+") as log:
        log.seek(0)
        runs = sum(1 for line in log if json.loads(line).get("probe"))
        log.write(json.dumps({"probe": True}) + "\n")
    return codes[runs] if runs < len(codes) else 0


if __name__ == "__main__":
    sys.exit(main())
