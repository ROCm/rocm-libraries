---
name: pr-summary
description: "DEPRECATED — superseded by hipdnn-pr-quality. Drafting PR titles and bodies is now the author-assist action of the hipdnn-pr-quality overlay. This stub remains only so the old name still resolves and redirects."
---

# PR Summary (deprecated)

**This skill is deprecated.** It has been superseded by `hipdnn-pr-quality`, which folds PR-description
authoring into a single PR-quality overlay on the shared `rocm-pr-quality` base skill.

When invoked:

1. Tell the user, in your own words, that `pr-summary` is deprecated and has been replaced by
   `hipdnn-pr-quality`. Do not silently proceed as if it were still current.
2. Then carry out the request using `hipdnn-pr-quality` (its **author-assist** action drafts and
   revises PR titles and bodies), not the old `pr-summary` behavior.

There is nothing else to do here. This stub exists only to redirect callers of the old name.
