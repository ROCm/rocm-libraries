# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-exports ``kernels.common.conv_wgrad_workspace_reduce`` (workspace-reduce kernel for two-stage deterministic wgrad).

Backward-compatible re-export for callers still importing from the old
``rocke.instances.common`` location. New code should import directly from
``kernels.common.conv_wgrad_workspace_reduce`` instead.
"""

from kernels.common.conv_wgrad_workspace_reduce import *  # noqa: F401,F403
