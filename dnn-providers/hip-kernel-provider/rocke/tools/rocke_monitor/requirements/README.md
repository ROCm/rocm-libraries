<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Python environments

The toolkit (`rk`) and the worker tools use only Python 3.10+ and its standard library, except the orchestrator, which
runs the Claude session through the Claude Agent SDK. The SDK and the checks have environments of their own, which the
image carries (none is installed on the bare machine). Each of the checks' environments is installed from its lock
(`*.lock.txt`: the exact versions the checks run with), so a new pytest or numpy release never changes what the checks
collect, skip or report; the `*.txt` files hold the lower bounds the code needs.

| Environment | File | Used by | Created by |
|---|---|---|---|
| `/opt/rkmon/venv` (`paths.venv`) | `checks.lock.txt` (bounds: `checks.txt`) | the review checks, merge gates, develop sanity, drift checks, the probes (`PROBE_PYTHON`) and the worker's `run_pytest` | the image |
| `/opt/rkmon/venv-gen` | `gen.lock.txt` (bounds: `gen.txt`) | the verifier, for suites outside the rocKE tree that need Jinja2 and PyYAML | the image |
| `/opt/rkmon/venv-claude` | none: the SDK's release that the build settings' `claude_agent_sdk` gives (else the newest), its wheel checked against the SHA-256 they give; its dependencies come from the package index | the orchestrator (`rkmon_lib/orchestrator.py`), and `claude`, the Claude Code CLI the SDK carries | the image (`docker/install_sdk.sh`) |
| a GPU PyTorch environment | none | a verifier's GPU experiments only | by hand: PyTorch's ROCm wheels depend on the GPU and the ROCm release, so the package does not pin them |

The checks venv never has torch: CI's lanes run rocKE's tests without it, and a test that imports it must skip there
as it does here. The image build fails when torch is importable in it, and `rk doctor` checks it.
