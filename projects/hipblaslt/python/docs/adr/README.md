# Architecture Decision Records for the Python bindings

This directory holds the decisions about the hipBLASLt Python bindings that should outlive the project that made them. Each one is a short file in [Michael Nygard's format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions), the same form as the TensileLite characterization ADRs (`tensilelite/tensilelite/Tests/unit/characterization/adr/`).

Most of these decisions were made in the bindings' design review, Review A ([AIHPBLAS-5274](https://amd-hub.atlassian.net/browse/AIHPBLAS-5274)), which numbers them. Each ADR names the decision it records.

## When to write one

Write an ADR when a decision would surprise a future maintainer, or when undoing it would cost real work: how the bindings find hipBLASLt, how they identify a solution, what a binding is allowed to accept. A routine implementation choice isn't an ADR; a code comment or the PR description covers it.

An ADR lands in the PR that implements its decision.

## Format

- One file per decision, named `NNNN-short-slug.md`, numbered in order.
- A title line, `# ADR NNNN: <title>`, then `Status:` and `Decision:` lines (the Review A decision number, or "none").
- The sections **Context**, **Decision** and **Consequences**.
- **Status** is `Proposed`, `Accepted`, or `Superseded by NNNN`.
- **Append-only.** Don't rewrite an accepted ADR. To change a decision, add a new ADR that supersedes it, and change only the old one's `Status` line.

## Index

- [0001: The package version is hipBLASLt's version](0001-version-source.md) (Decision 1)
- [0002: How Windows finds the DLLs](0002-windows-dll-loading.md) (Decision 5)
- [0003: CI builds against an installed ROCm, and checks enums against the PR's headers](0003-ci-header-strategy.md) (Decisions 6, 11 and 13)
