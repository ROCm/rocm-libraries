# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collected sweep results -> the dataset a UHD is trained on (RFC 0019.13 §8.3).

Adds what the per-shard CSVs cannot carry: the §8.3 checks, the failure encoding, and the
`tflops`/`gbs` rates derived from engine-reported flops and bytes. Importable without the
training stack.
"""
