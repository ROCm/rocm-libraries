---
id: pattern-catalog-exhausted
title: "Catalog exhausted — family table cannot move the bottleneck"
type: pattern
tags: [escape-hatch, routing]
symptoms: [catalog-exhausted]
candidate_techniques:
  - technique-algorithm-break
  - technique-fusion
  - technique-persistent-streamk
related:
  - process-escape-hatch
  - process-optimization-loop
  - family-overview
sources: [project-rocke]
---

# Catalog exhausted

The one-lever loop is spinning: Step 0 ceiling still misses, probes name the
same limiter, and the next change is a tile or pipeline already tried.

Do **not** pick another row from the family table. Open
`process-escape-hatch`, pass the stall test, then prototype with
`technique-algorithm-break`.

Query: `python3 scripts/query.py --symptom catalog-exhausted`
