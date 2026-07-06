# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Examples - demonstrations of how a CONSUMER wires the rocke.benchmark.perf primitives.

These are NOT part of the shipped tool surface. They show how a developer, an
agent, or an external perf framework would drive the primitives for a given
workflow. Each example is a thin consumer; the reusable logic lives in the
`rocke.benchmark.perf` primitives (and the local store in `rocke.benchmark.perf.tool`).
"""
