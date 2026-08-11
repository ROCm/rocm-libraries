# The ROCm interfaces layer

Status: proposed design, prototype-backed. This tree builds standalone
(`cmake -S interfaces`) and is not yet wired into the root ROCm build. Every capability
described here is real code in this tree, and every claim below is backed by a named test
you can run.

This is the documentation set for the interfaces layer: a thin, versioned boundary that
lets a ROCm math library change how it is implemented without breaking the callers linked
against it.

## Start here

If you read one page, read [02-why-a-stable-boundary.md](02-why-a-stable-boundary.md). It
shows the two failures this layer exists to prevent - a caller welded to your internals,
and a provider `.so` leaking 170 libstdc++ symbols into the process - and why a versioned
boundary is the fix.

Then read the chapters in order (02 is also step 2 below):

| Read | Chapter | What you get |
| --- | --- | --- |
| 1 | [01-architecture.md](01-architecture.md) | The three layers (loader / runtime / protocols), how one call flows, why tables grow but never break. |
| 2 | [02-why-a-stable-boundary.md](02-why-a-stable-boundary.md) | The threat model. What breaks without a versioned boundary, shown failing-case first. |
| 3 | [03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md) | The normative contract: version-node registry, symbol-map idioms, SONAME rules, the reusable non-vacuity proof recipe. |
| 4 | [04-hardening.md](04-hardening.md) | Each hardening step, threat before fix, every claim citing the ctest that proves it. |
| 5 | [05-extending.md](05-extending.md) | How-to recipes: add a provider, add a version node, add an ABI proof, change a public API. |
| 6 | [provider-protocols.md](provider-protocols.md) | The provider protocol specification (the C ABI tables, per domain). This is the conceptual chapter 06; it keeps its original unnumbered filename, which is why the numbered chapters jump from 05 to 07. |
| 7 | [07-status-and-roadmap.md](07-status-and-roadmap.md) | What is done, what is committed next, what is aspirational. |

## Who this is for

- **Interfaces-layer maintainers** are the primary audience. Chapters 01, 03, 04, and 05
  are the durable core: the design, the contract, the invariants, and how to extend them
  without archaeology.
- **Library integrators** (a rocBLAS or rocRAND author wondering what adopting this costs)
  should read 02, then 01 and 05.
- **A reviewer of PR #10272** can approve from 02 (why), 04 (what is proven), and
  [07-status-and-roadmap.md](07-status-and-roadmap.md) (what is and is not claimed).

## The reference layer

The chapters above are the human-friendly path. Beneath them sit three supporting reference
documents that predate this set. They differ in status and none is a settled, normative
source of truth; the chapters link into them where the detail matters and do not rewrite
them here.

- [rocblas-provider-clusters.md](rocblas-provider-clusters.md) - the rocBLAS narrowing map:
  how 1,213 public declarations classify into provider primitives. It is directional input,
  not an adopted provider ABI (see its own opening note).
- [audit-findings.md](audit-findings.md) - the initial ABI audit (hipSOLVER facade,
  enum coupling, RAND visibility, the rocRAND header defect). It is a findings record from
  that audit, not a contract.
- [api-change-process.md](api-change-process.md) - a compatibility pointer only: its content
  has been absorbed into [03](03-abi-and-versioning-contract.md) and
  [05](05-extending.md), and the stub remains so existing links do not break.

The provider protocol specification (provider-protocols.md) is the one genuinely normative
reference; it now appears as chapter 06 in the reading order above.

## Writing these docs

The house style is in [STYLE.md](STYLE.md): threat-first, example-first, every claim cites
a test. Follow it when you add or edit a chapter.
