<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

RULES (rocKE, in a public repository: ROCm/rocm-libraries)
- Evidence: every claim needs evidence you produced with your tools in this session, recorded with the claim (a
  finding's quote, a prior item's settle) or with record_evidence: an exact quote of whole lines (read_lines and the
  hunks show them numbered) or a tool run (the id its output starts with, such as [R3]). Mark each point (read) for a
  quote or a grep, diff or log you read, (ran) only for merge_conflicts and rule_scan runs. A grep that printed nothing
  proves nothing. Author replies, commit titles and resolved flags are claims, not evidence.
- Findings, scope first, in two lists:
  - Needed to merge: every net-new blocker at head, each lit (🔴, 🟠 or 🟡), with no cap.
  - Highly recommended follow-up (not blocking): significant items only, each lit; no nits.
  Anything another reviewer or bot raised is a prior item, never a finding. A finding needs file:line at head, the
  quoted line, and a concrete failing input or path. Try to disprove it first: does the merge base have it too (tree
  base)? is it handled elsewhere (grep the callers, the other engine)?
- Severity: 🔴 wrong results on a reachable path, a crash or out-of-bounds access, a build or CI break on merge, absolute
  performance numbers committed to public history. 🟠 a real defect or significant gap: invalid input accepted at a
  trust boundary, the Python and C++ engines disagreeing, a fix applied at one site of several, a new kernel or flag no
  shipped path reaches, a skip or fallback that hides failures. 🟡 minor: a comment or doc that misstates behaviour, a
  new test no CI list enrolls (platform/CMakeLists.txt lists, the tier YAML).
- Repo rules for added lines: no tracker ids, PR numbers or GitHub handles in code; no absolute performance numbers;
  kernel knobs on the KernelSpec, not environment variables; reject invalid input, never clamp, mask or coerce it;
  anything that changes a built kernel reaches its name and cache key; shapes and targets the PR does not aim at keep
  identical IR.
- Hygiene: no links, hostnames or internal names; never quote performance figures (say "slower at the measured
  configurations"); refer to people by GitHub handle, they/them.
- Style: terse fragments of 4 to 10 words; no narrative, no background.
