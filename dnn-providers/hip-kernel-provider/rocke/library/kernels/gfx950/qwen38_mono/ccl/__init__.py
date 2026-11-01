# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""In-kernel collectives (CCL) of the Qwen3.8 mono family: IPC peer
mailboxes + all-reduce / all-gather / argmax merge (DESIGN-CCL WP-C1/C2).

Modules: ``layout`` (knobs, specs, arena map, error codes), ``wire`` (asm
emitters of the wire forms and bounded polls), ``collectives`` (kernel
builders), ``ref`` (CPU references, bit-exact), ``war`` (slot-reuse checker),
``lint`` (peer-address lint). Host runtime: ``rocke.runtime.peer``.
"""
