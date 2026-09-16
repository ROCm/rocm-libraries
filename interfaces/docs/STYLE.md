# Doc style guide

These docs have one voice. It is the reason a newcomer can read `02` and understand why the
layer exists, and a maintainer can read `05` and add a provider without archaeology. Follow
these rules when you add or edit a chapter.

## The voice, in five rules

1. **Threat before mechanism.** Never explain how something works before the reader knows
   what breaks without it. Open a hardening section with the failure, then the fix. If you
   cannot name the failure, you do not yet understand why the code exists.
2. **Example before abstraction.** Show the concrete case first - a real symbol, a real
   `.map`, a real `nm -D` line - then generalize. `rocblas_sgemm@@ROCBLAS_ABI_5` teaches
   more than "a versioned symbol."
3. **Back every claim with the right kind of evidence, and name it.** Not every claim can
   cite a CTest; match the evidence to the claim's status:
   - **Executable capability** (the layer does X today) -> name the `ctest` that would fail
     if it stopped, by name not count (`abi03_interpose_hazard`, not "26 tests pass").
   - **Intended contract** (a rule a provider or loader must obey, not yet fully enforced)
     -> cite the implementation-status note in `03` or the roadmap entry in `07`, and say
     which half runs today.
   - **One-time observation** (a measurement, not an asserted invariant) -> give the dated
     evidence and say it is not enforced by CTest (for example the local `__tsan_` symbol
     count).
   - **Design direction** (proposed, not built) -> label it ASPIRATIONAL and link to `07`;
     never let it read as a fact.
   A threat-model "Proven by" cell may therefore name a CTest, a `03`/`07` citation, or
   "architecture design" - but the surrounding prose must not then claim every entry is a
   CTest.
4. **Separate the proven from the proposed.** The layer is a proposed design, not adopted
   code. Say so. Keep DONE, COMMITTED-NEXT, and ASPIRATIONAL visibly apart (see `07`). Never
   let a plan read as a fact.
5. **Say what you do not claim.** A reviewer trusts a doc that draws its own boundaries.
   State what is out of scope as plainly as what is in it.

## Register

- Second person, active voice, present tense. "You call it once. It hands you back a table."
- Short sentences carry the argument; tables carry the exactness. Use prose for the *why*, a
  table for a registry, a contract, or a symbol map.
- Spec-register (dense, normative) is correct in `03` and in `provider-protocols.md` (the
  proposed target provider contract). `rocblas-provider-clusters.md` is directional input,
  not an adopted ABI - keep its dense register but do not treat it as normative.
- ASCII only. No smart quotes, em dashes, or Unicode symbols - use `-`, `->`, and plain
  quotes.

## Mechanics

- Diagrams are Mermaid (renders on GitHub). Reach for one when a data flow or a resolution
  order is clearer seen than read; do not diagram the trivial.
- Link generously between chapters and into the reference layer. A reader should never have
  to guess where the precise version lives.
- Do not add comments to code samples. The surrounding prose is the explanation; in these
  docs a code comment is a smell. (This is a house rule for the doc samples, not a
  repository-wide policy - the tree's own C/C++ sources carry substantive comments.)
- Keep the reference layer as reference. Do not rewrite `provider-protocols.md` (proposed
  target contract, partially implemented) or `rocblas-provider-clusters.md` (directional
  input, not an adopted ABI) into this voice - link to them.

## Review checklist

Before you land a doc change, confirm:

- [ ] Every hardening/capability section opens with the threat, not the mechanism.
- [ ] Every executable-capability claim names a real `ctest` (verified against
      `tests/CMakeLists.txt`); intended-contract, observation, and design-direction claims
      instead cite `03`/`07` or are labeled accordingly (see rule 3).
- [ ] No brittle counts stand in for names.
- [ ] Proposed vs proven is unambiguous; the Status line is present.
- [ ] Out-of-scope items are stated.
- [ ] ASCII only; no code comments; Mermaid diagrams render.
- [ ] Cross-links resolve, including into the reference layer.

There is no CI gate on these yet - this checklist is enforced by review. If the docs start
to drift, promoting the checklist to a check alongside `check_api_policy.py` is the next
step (tracked in [07-status-and-roadmap.md](07-status-and-roadmap.md)).
