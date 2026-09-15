"""Stand-in for an agent step: writes an artifact and a result file, fast.

Usage: emit.py <result_file> <iteration> <artifact_path> <countdown_from>

Exists so the MCP integration path can be exercised end to end in seconds
instead of hours of agent sessions. The result deliberately carries three
shapes the server must treat differently: a number the loop condition reads,
a real file under a `path` output, and a path-looking string under a `string`
output that must never be promoted to an artifact.
"""

import json
import os
import sys


def main() -> int:
    result_file, iteration, artifact_path, countdown_from = sys.argv[1:5]
    iteration_index = int(iteration)
    remaining = max(0, int(countdown_from) - iteration_index)

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as handle:
        handle.write(f"iteration {iteration_index}, remaining {remaining}\n")

    payload = {
        "remaining": remaining,
        "produced": artifact_path,
        "looks_like_a_path": "D:/not/a/real/artifact.txt",
        "summary": f"iteration {iteration_index} left {remaining} to go",
    }
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    print(f"emit: iteration={iteration_index} remaining={remaining}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
