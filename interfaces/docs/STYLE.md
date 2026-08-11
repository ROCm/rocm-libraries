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
3. **Every claim cites a test.** If you assert the layer does X, name the `ctest` that would
   fail if it stopped doing X. Cite the test **name**, not a count - names do not drift,
   counts do. "26 tests pass" rots; `abi03_interpose_hazard` does not.
4. **Separate the proven from the proposed.** The layer is a proposed design, not adopted
   code. Say so. Keep DONE, COMMITTED-NEXT, and ASPIRATIONAL visibly apart (see `07`). Never
   let a plan read as a fact.
5. **Say what you do not claim.** A reviewer trusts a doc that draws its own boundaries.
   State what is out of scope as plainly as what is in it.

## Register

- Second person, active voice, present tense. "You call it once. It hands you back a table."
- Short sentences carry the argument; tables carry the exactness. Use prose for the *why*, a
  table for a registry, a contract, or a symbol map.
- Spec-register (dense, normative) is correct in `03` and in the reference layer
  (`provider-protocols.md`, `rocblas-provider-clusters.md`). Everywhere else, prefer the
  example-first register.
- ASCII only. No smart quotes, em dashes, or Unicode symbols - use `-`, `->`, and plain
  quotes.

## Mechanics

- Diagrams are Mermaid (renders on GitHub). Reach for one when a data flow or a resolution
  order is clearer seen than read; do not diagram the trivial.
- Link generously between chapters and into the reference layer. A reader should never have
  to guess where the precise version lives.
- Do not add comments to code samples. The surrounding prose is the explanation; a code
  comment is a smell (and the repo bans them outside Python docstrings).
- Keep the reference layer as reference. Do not rewrite `provider-protocols.md` or
  `rocblas-provider-clusters.md` into this voice - they are normative and already precise.
  Link to them.

## Review checklist

Before you land a doc change, confirm:

- [ ] Every hardening/capability section opens with the threat, not the mechanism.
- [ ] Every capability claim names a real `ctest` (verified against `tests/CMakeLists.txt`).
- [ ] No brittle counts stand in for names.
- [ ] Proposed vs proven is unambiguous; the Status line is present.
- [ ] Out-of-scope items are stated.
- [ ] ASCII only; no code comments; Mermaid diagrams render.
- [ ] Cross-links resolve, including into the reference layer.

There is no CI gate on these yet - this checklist is enforced by review. If the docs start
to drift, promoting the checklist to a check alongside `check_api_policy.py` is the next
step (tracked in [07-status-and-roadmap.md](07-status-and-roadmap.md)).
