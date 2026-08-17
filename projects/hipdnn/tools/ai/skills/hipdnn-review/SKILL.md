---
name: hipdnn-review
description: "DEPRECATED — superseded by hipdnn-pr-quality. Reviewing hipDNN PRs and diffs is now the review-assist action of the hipdnn-pr-quality overlay. This stub remains only so the old name still resolves and redirects."
---

# hipDNN Review (deprecated)

**This skill is deprecated.** It has been superseded by `hipdnn-pr-quality`, which folds hipDNN code
review into a single PR-quality overlay on the shared `rocm-pr-quality` base skill.

When invoked:

1. Tell the user, in your own words, that `hipdnn-review` is deprecated and has been replaced by
   `hipdnn-pr-quality`. Do not silently proceed as if it were still current.
2. Then carry out the review using `hipdnn-pr-quality` (its **review-assist** action), which preserves
   the hipDNN review checks (correctness, API/cuDNN compatibility, provider behavior, RAII/resource
   ownership, code reuse, and ASIC/multi-arch coverage) on top of the base.

There is nothing else to do here. This stub exists only to redirect callers of the old name.
