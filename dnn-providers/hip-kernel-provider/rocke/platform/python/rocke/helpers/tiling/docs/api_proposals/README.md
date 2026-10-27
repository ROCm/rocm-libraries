# Tiling-API proposals

Each file here is **one proposed addition/change to the tiling API**, written when authoring a kernel hit
friction the API should remove. **One proposal per file** so they can be triaged and processed individually
(accept → implement in `helpers/tiling`, defer, or reject).

- Written by the **Tiling Kernel Architect** (`.claude/team_members/tiling_expert.md`) via the
  **`/rocke-tiling-api`** skill. A proposal is a *suggestion*, not a change — the API itself is not edited here.
- Name files for the capability, e.g. `PROPOSAL_lds_swizzle_policy.md`, `PROPOSAL_kloop_pipeliner.md`.
- Copy `_TEMPLATE.md` to start. Keep each focused; if a proposal's rationale is deep, put the derivation in a
  design doc (`../`) and link it.
- When a proposal is implemented, note the commit/PR and move it to a `done/` subfolder (or delete — these are
  working proposals, not an archive).
