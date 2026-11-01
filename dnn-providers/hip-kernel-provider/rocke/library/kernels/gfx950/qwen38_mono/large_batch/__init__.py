# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Large-batch (rows 5-128, TP1 / TP2) pieces of the Qwen3.8 true rocKE monokernel (``DESIGN-LB.md``).

* :mod:`.bytes_model` (WP-L0c): per-stage HBM bytes of one decode step per rank (host only).
* :mod:`.gdn_recur` (probe P-L1): the GDN recurrence at large batch as rocKE kernels (the LB recurrence with its
  knobs, FlyTape's ``gdn_recur`` re-implemented, the bytes-floor streamer); Python engine only, ``q38lb_`` names.
* :mod:`.gdn_recur_ref` (P-L1): fp64 / float32 references and the exact-order emulation of the LB kernel (host only).
"""
