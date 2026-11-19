# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host builders for gfx942 chunkwise Kimi Delta Attention (KDA) prefill.

``kda_chunk_prep``, ``kda_chunk_scan``, and ``kda_chunk_fused`` compile and
launch the kernels in ``kernels.gfx942.kda_chunkwise``. These are the
chunkwise KDA prefill path (per-channel gate plus delta-rule write), not the
Gated DeltaNet K5/K6 scan.

Two host lanes share the same packed layout:

* Torch GPU experiments: ``kda_chunk_fused.check`` /
  ``launch_packed`` (needs ROCm torch). Used by
  ``library/tests/test_kda_chunkwise_gfx942_numeric.py``.
* External ``run_manifest`` / ``benchmark.remote_test``: numpy pack +
  oracle in ``hostpack.py``, registered by importing ``manifest.py``.
  The fused example's ``--no-verify --output-dir`` writes a manifest
  with ``kind=kda_chunk_fused_bf16`` and ``runner_module`` pointing here.
  Point ``remote_test.config.ARCHES`` at
  ``builders.gfx942.kda.kda_chunk_fused`` and pass ``--m B --n H --k T``.
"""
