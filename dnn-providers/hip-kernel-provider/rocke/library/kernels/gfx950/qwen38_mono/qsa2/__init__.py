# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""QSA Q3-Q7 of the Qwen3.8 mono layer as rocKE stage functions (true-mono WP2c).

Default knobs = the FlyDSL mono's stages, bitwise; every other knob is off by
default. Nothing here changes an existing build: the package is imported only
by its own generator (``python -m kernels.gfx950.qwen38_mono.qsa2.gen``).
"""
