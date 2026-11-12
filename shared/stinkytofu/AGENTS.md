# AI Agent Guidance

This file provides guidance for AI coding agents when working on StinkyTofu.

## DAG scheduler

Rules for changing the CDNA5 DAG scheduler are in
[`src/transforms/asm/dag/AGENTS.md`](src/transforms/asm/dag/AGENTS.md). Read them before you edit
the scheduler.

## Build and test

Build, test, FileCheck, and clang-tidy commands are in [`README.md`](README.md) and
[`tests/README.md`](tests/README.md). Activate the project venv first.

## Verification harness

For the order in which to verify a change (CPU checks first, GPU last) and how the agent scaffolding is
kept honest across model generations, see [`harness/HARNESS.md`](harness/HARNESS.md). This
file wins on any conflict.
