# Lane: frontend bypass (`FRONT`)

**Trigger:** always. **Agent:** `scout`. **Output:** `R/lanes/FRONT.md`, ideas.

Find graphs the frontend builds but should reject, or that mean something else further down.

## Look for

- Checks the frontend does not make (brief section 2): shapes, strides, dtypes, attribute ranges, attribute combinations.
- Attribute combinations that validate but are interpreted differently downstream (mask alignment, scale vs scale tensor, a
  probability with no seed).
- Defaults that differ between the frontend and the backend or the plugin.
- Shapes or strides only some layers validate; graphs built through the raw backend API, deserialized or JSON graphs, where frontend
  `validate()` does not run at all.

## Ideas

`Expect: FRONTEND-GAP`, naming the layer that misses the check in `Why:` and what then happens (the engine accepts and computes
something, crashes, or declines). These usually become `Place: hand-probe` (a hand-built graph or a migration-scripts bundle): say
which.
