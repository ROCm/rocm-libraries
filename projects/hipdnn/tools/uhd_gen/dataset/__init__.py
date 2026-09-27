# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collected sweep results -> the dataset a UHD is trained on (RFC 0019.13 §8.3).

Stage three of the pipeline in ../README.md: `export-benchmarks` writes the per-shard CSVs, this
publishes the Parquet dataset from them, and `train` reads that. What it adds between the two is
everything the CSV cannot carry -- the §8.3 checks, the failure encoding the published dataset
uses, and the `tflops`/`gbs` rates derived from the operation's declared flops and bytes.

Deliberately importable without the training stack. See `__main__.py`.
"""
