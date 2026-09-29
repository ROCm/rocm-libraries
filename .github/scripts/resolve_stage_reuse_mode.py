#!/usr/bin/env python3


import os
from pathlib import Path

event = os.environ["EVENT_NAME"]
branch = (
    os.environ["PR_BASE_REF"] if event == "pull_request" else os.environ["REF_NAME"]
)
requested = os.environ["REQUESTED_MODE"]
baseline = os.environ["BASELINE_RUN_ID"]

if branch.startswith("release/"):
    mode, reason = "off", "release branch"
elif requested == "reuse-stage" and not baseline:
    mode, reason = "off", "no baseline run ID"
else:
    mode, reason = requested, "requested mode"

print(
    f"Stage reuse: requested={requested} effective={mode} "
    f"target={branch} reason={reason}"
)
with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
    print(f"mode={mode}", file=output)
