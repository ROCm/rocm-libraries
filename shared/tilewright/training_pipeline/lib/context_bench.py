#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipblaslt-bench under one execution context (lib/contexts.py).

    context_bench.py <context.json> --device D --yaml FILE ...

The generated `hipblaslt-bench` wrapper of a context runs this; it replaces
itself with the bench (or hipblaslt-cotenant around it), so stage03 watches
and stops the bench process as it would the bench itself.
"""
import json
import os
import sys
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib.contexts import bench_command  # noqa: E402


def main(argv) -> int:
    if len(argv) < 1:
        print("usage: context_bench.py <context.json> <bench args...>", file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[0]).read_text())
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    try:
        cmd, env_changes = bench_command(spec, argv[1:], stamp)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env.update(env_changes)
    sys.stdout.flush()
    os.execvpe(cmd[0], cmd, env)
    return 127


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
