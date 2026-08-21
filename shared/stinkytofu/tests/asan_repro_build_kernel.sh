#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Runs InsertAsanCheckPass on the asan-repro kernel body, strips
# stinkytofu-opt's own auto-generated (empty/placeholder) kernel descriptor
# -- which it always synthesizes for whatever function it recognizes in the
# input, uniquely identifiable by its `.section .rodata,#alloc` opener vs.
# kernel_descriptor.s's plain `.rodata` -- then appends the real hand-written
# descriptor from kernel_descriptor.s. See kernel_body.s's header comment for
# why the two are kept in separate files.
set -euo pipefail

STINKYTOFU_OPT="$1"
KERNEL_BODY_S="$2"
KERNEL_DESCRIPTOR_S="$3"
OUTPUT_S="$4"

"${STINKYTOFU_OPT}" --arch gfx1250 "${KERNEL_BODY_S}" \
    --from-label region_start --to-label region_end \
    --InsertAsanCheckPass --emit-asm \
  | sed -n '/^\.amdgcn_target/,$p' \
  | sed '/^\.section \.rodata,#alloc$/,/^\.end_amdgpu_metadata$/d' \
  > "${OUTPUT_S}"
cat "${KERNEL_DESCRIPTOR_S}" >> "${OUTPUT_S}"
