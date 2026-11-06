# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A local web editor for StinkyTofu IR checkpoint bundles."""

from .server import Workspace, WorkspaceError, make_server

__all__ = ["Workspace", "WorkspaceError", "make_server"]
