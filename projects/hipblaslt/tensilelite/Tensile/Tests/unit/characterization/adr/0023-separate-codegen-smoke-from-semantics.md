# Separate code-generation smoke coverage from semantic assertions

## Status

Accepted

## Context

ADR 0019 added a SHA-256 digest of every opcode kind in each config-driven
kernel. The public coverage lane then reported 30 saved-result mismatches. The
digest also omitted operands, instruction counts, and instruction order, so it
was broader than the behavior named by each test without fully describing that
behavior.

The set-cover configurations were selected to execute otherwise unreached code.
They are useful coverage and generation-status checks, but they do not define a
stable instruction sequence. Treating those cases as semantic snapshots made
compiler changes fail unrelated coverage tests.

## Decision

Remove the opcode-set digest from the shared saved-result format. Keep
set-cover cases as explicitly named smoke tests that assert the selected problem
group produces kernels. Cases without known emitter failures require every
kernel to succeed; known-failure cases allow a later fix to turn them green.
They no longer create `.ambr` snapshots.

Add `required_source_patterns` to the shared emit assertion for tests that name
specific generated behavior. Each pattern has a human-readable description and
must occur in at least one successfully emitted kernel. Use it first for the
S11a integer-to-float conversion. The two gfx1250 cases whose broad snapshots
changed in the public lane remain generation smoke tests: their named
intermediate instructions are not present in the final emitted assembly, so a
source assertion would misrepresent what those tests observe.

## Consequences

Compiler changes no longer require re-recording the 75 set-cover cases. Those
tests continue to protect reachability and generation status, but make no claim
about instruction semantics. A named behavior is protected only when its test
supplies a focused source assertion; remaining configuration-driven tests must
gain such assertions as their intended behavior is reviewed.
