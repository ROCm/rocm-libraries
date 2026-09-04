# The ROCm interfaces layer

Status: proposed design with a working noncanonical rocBLAS implementation. This tree builds
standalone (`cmake -S interfaces`) or through the default-off root option
`ROCM_LIBS_ENABLE_INTERFACES`. Executable
capabilities here name the CTest that proves them; intended contracts and planned work are
marked as such and cite the implementation-status note in
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md#implementation-status-prototype)
or [07-status-and-roadmap.md](07-status-and-roadmap.md).

This is the documentation set for the interfaces layer: a thin, versioned boundary that
lets a ROCm math library change how it is implemented without breaking the callers linked
against it.

## Start here

If you read one page, read [02-why-a-stable-boundary.md](02-why-a-stable-boundary.md). It
shows the two failures this layer exists to prevent - a caller locked to a single provider
it cannot replace without relinking (one implementation welded to one package), and a
provider `.so` leaking roughly 174 libstdc++ symbols into the process - and why a versioned
boundary is the fix.

Then read the chapters in order (02 is also step 2 below):

| Read | Chapter | What you get |
| --- | --- | --- |
| 1 | [01-architecture.md](01-architecture.md) | The three layers (loader / runtime / protocols), how one call flows, why tables grow but never break. |
| 2 | [02-why-a-stable-boundary.md](02-why-a-stable-boundary.md) | The threat model. What breaks without a versioned boundary, shown failing-case first. |
| 3 | [03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md) | The normative contract: version-node registry, symbol-map idioms, SONAME rules, the reusable non-vacuity proof recipe. |
| 4 | [04-hardening.md](04-hardening.md) | Each hardening step, threat before fix, with evidence matched to each claim's status. |
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
  how 1,219 public callables classify into provider primitives, including six grouped-GEMM
  callables that remain bridge-only. It is directional input, not an adopted provider ABI
  (see its own opening note).
- [audit-findings.md](audit-findings.md) - the initial ABI audit (hipSOLVER facade,
  enum coupling, RAND visibility, the rocRAND header defect). It is a findings record from
  that audit, not a contract.
- [api-change-process.md](api-change-process.md) - a compatibility pointer only: its content
  has been absorbed into [03](03-abi-and-versioning-contract.md) and
  [05](05-extending.md), and the stub remains so existing links do not break.

The provider protocol specification (provider-protocols.md) states the proposed target
provider contract and is only partially implemented; it now appears as chapter 06 in the
reading order above. Where that spec and the current prototype differ, the
implementation-status note in
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md#implementation-status-prototype)
is authoritative (see the status banner at the top of that spec).

## Writing these docs

The house style is in [STYLE.md](STYLE.md): threat-first, example-first, evidence matched
to each claim's status (see rule 3 in STYLE). Follow it when you add or edit a chapter.
