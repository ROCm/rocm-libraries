# Remove set-cover basename snapshots

## Status

Accepted

## Context

The 75 set-cover emit cases were selected to reach otherwise uncovered code.
Their three saved-result files recorded only each generated kernel's
content-derived basename and emitter return code. A basename is computed from
solution state before assembly emission, so it can change after unrelated
derivation updates while remaining blind to an incorrect instruction sequence.
Re-recording those hashes after every such change would preserve neither the
coverage target nor the emitted semantics.

The emitter also carries scheduler state within a process. A process-wide
one-time warm-up made logic-driven results depend on which test happened to run
first, while the config-driven path had already moved to per-call self-warming.

## Decision

Remove the three set-cover basename snapshot files. Keep every selected YAML
problem group and assert its exact generated-kernel count, plus successful
emitter status except for cases that intentionally admit known failures. Add
narrow source-pattern assertions to representative MX-fp6, dot2, and swizzled
addressing cases where the final assembly exposes a stable named behavior.

Capture the ordered rejection reasons in the set-cover derivation snapshots so
cases targeting different guards cannot pass with identical base-state output.
Make both logic- and config-driven assembly harnesses perform one throwaway
warm-up for each call, eliminating worker-order dependence.

## Consequences

The set-cover suite now distinguishes solution-count changes, rejection guards,
and representative emitted instructions without snapshotting compiler identity.
Adding or removing an accepted kernel is a readable table update. A source
pattern is added only where the test can name a stable semantic observable;
the remaining configurations continue to be explicitly scoped reachability and
generation-status checks.
