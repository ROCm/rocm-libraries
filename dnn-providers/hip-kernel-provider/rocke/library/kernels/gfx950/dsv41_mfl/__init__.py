# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocKE stage functions of monofly's DeepSeek-V4.1 layer kernels, the layer lane (monofly-study dsv41/ports/layer):
the stages outside the attention / MoE lanes' (``stage_norm``: NormQuant). monofly's device ABI: ``..dsv41_mf.mfabi``."""
