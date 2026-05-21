---
description: Build hipDNN with providers through the rocm-libraries superbuild
argument-hint: "[preset] [clean] [ROCM_PATH=<path>] [CLANG_PATH=<path>] [GPU_TARGETS=<arch>] [SHA=<commit>]"
allowed-tools: Bash, Read, Grep, Glob
---

Use the `hipdnn-superbuild` skill to configure and build hipDNN through the rocm-libraries superbuild.

Parse `$ARGUMENTS` as the user-provided build options. Preserve the skill's default linked-install behavior assumptions and follow the active workspace build-output safety rules.
