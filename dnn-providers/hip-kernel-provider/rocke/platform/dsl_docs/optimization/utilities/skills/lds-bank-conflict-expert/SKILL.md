---
name: lds-bank-conflict-expert
description: Predict and analyze LDS access conflicts with an exact rocKE architecture profile and a serializable semantic result. Use when examining lane-to-LDS byte-address mappings, checking broadcasts or conflict multiplicity, comparing supported gfx profiles, or producing deterministic conflict JSON without a GPU or profiler.
---

# LDS Bank Conflict Expert

Use the production predictor as the single source of conflict semantics. Do not
derive a second bank formula in this skill or in downstream analysis.

## Workflow

1. Collect the exact target, opcode, wave size, and one access record per logical
   LDS access. Give every access a stable ID, lane, byte address, and width.
2. Read [references/index.md](references/index.md) for input routing and supported
   outputs. Read [references/model-contract.md](references/model-contract.md) when
   constructing access records or interpreting conflict groups.
3. Read the matching architecture reference. Start with
   [references/gfx90a.md](references/gfx90a.md) for `gfx90a`; reject a target that
   has no registered profile.
4. Run `scripts/predict.py` to produce canonical semantic JSON. Treat diagnostics
   as part of the result, not as optional console commentary.
5. Analyze classifications, group membership, multiplicity, and diagnostics from
   that semantic document without recomputing them.
6. Read [references/comparison.md](references/comparison.md) before comparing
   profiles and [references/validation-boundaries.md](references/validation-boundaries.md)
   before making architecture or validation claims.

## Required boundaries

- Select exact profile identities. Never substitute `gfx90a` for an unknown target.
- Keep prediction CPU-only and independent of profiler output or private data.
- Distinguish broadcast, distinct-address conflict, inactive access, and invalid or
  unsupported input.
- Report empirical collision semantics only within the documented opcode, wave,
  and address scope. Do not infer physical organization from the model.
- Use public target names and qualitative semantics. Do not add internal links,
  paths, raw measurements, job metadata, or software performance figures.
