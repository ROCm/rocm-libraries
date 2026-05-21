---
description: Run tests against an existing hipDNN superbuild
argument-hint: "[component: hipdnn|miopen|hipblaslt|hip-kernel|integration-tests|all] [scope: unit|integration|all] [ROCM_PATH=<path>] [--filter=<gtest_pattern>] [--verbose] [--keep-going]"
allowed-tools: Bash, Read, Grep, Glob
---

Use the `hipdnn-superbuild-test` skill to run tests against an existing hipDNN superbuild.

Parse `$ARGUMENTS` as the user-provided component, scope, filter, and execution options. Keep full test output in logs and report only concise tails on failure.
