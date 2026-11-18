<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# TensileLite Perf YAMLs

This directory contains curated TensileLite performance and tuning input YAMLs.
They are intended for manual tuning and benchmarking, not for automatic CI test collection.

Directory layout is grouped by architecture first, then workload family. Files under
`tuning/` are broader search or sweep configs; sibling YAMLs are curated runnable configs.

`finetune` denotes parameters that should eventually be derived by
TensileLite/StinkyTofu, but are specified in YAML until that derivation is
smart enough to choose them automatically.

`gfx1250_96wgp/` is the 96-WGP best-kernel set (or 192-WGP too).
