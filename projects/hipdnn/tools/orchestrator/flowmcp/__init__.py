# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""MCP server package wrapping the orchestrator's `Engine`.

Named `flowmcp` rather than `mcp` on purpose: the orchestrator root is on
`sys.path` for every entry point and every test, so a package named `mcp` here
would shadow the installed SDK.

Only `flowmcp.server` imports that SDK. Everything else in this package is
stdlib, or stdlib plus `runner`, so the hermetic test suite exercises the real
logic with nothing extra installed. This module therefore imports nothing.
"""
