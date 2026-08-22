# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Examples - reference integrations that wire the rocke.benchmark.perf primitives.

These ship in the repo as the reference for how a consumer (a developer,
or the external perf framework) drives the primitives for a given workflow - e.g.
`profile_gemm_sweep` over rocKE's GEMM sweep. Each is a thin consumer; the reusable
logic lives in the `rocke.benchmark.perf` primitives (and the local store in
`rocke.benchmark.perf.tool`). They do the produce + local-store side only; the
*system* work (which GPUs run, scheduling, at-scale runs, mass data storage) is the
external framework's.
"""
