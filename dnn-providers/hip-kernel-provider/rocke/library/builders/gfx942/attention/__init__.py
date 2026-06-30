# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 (CDNA3 / MI300X) unified-attention 2D tiled SDPA-fwd examples.

A torch-reference-only parity + benchmark harness for the gfx942 narrow /
wide (flash-regime) tiled attention kernel in
``rocke.instances.gfx942.attention_tiled_2d``. Unlike the gfx950 attention
harness it has **no** Triton/AITER dependency: the oracle is a fp32 torch
reference, so the example runs on any box with torch + a gfx942 GPU.

Ships the canonical shapes we validate (``shapes.json``) and the definitive
correctness + perf check (``final_shapes_check.py``); see ``README.md``. The
exhaustive sweep / probe / regression harnesses that found & gated the levers
are archived under ``~/attention-archive/``.
"""
