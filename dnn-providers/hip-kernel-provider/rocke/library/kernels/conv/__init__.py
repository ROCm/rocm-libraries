# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convolution AOT support: kernel identities and names, AOT build jobs, and
the direct-conv variant registry shared by the AOT cache and the dispatcher;
plus the split-K cost model (:mod:`kernels.conv.split_k_planner`) the conv
split-K resolver plans with, the implicit-GEMM helper modules (dgrad interior
and box planning, output-plane band order), the per-output-channel forward
bias every conv forward family adds (:mod:`kernels.conv.conv_bias`) and the
3-D Winograd family."""
