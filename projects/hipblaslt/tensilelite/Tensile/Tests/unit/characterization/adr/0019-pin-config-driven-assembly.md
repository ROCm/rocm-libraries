# Record emitted assembly in config-driven saved results

## Status

Accepted

## Context

The config-driven characterization tests saved each generated kernel's name and
emitter return code. The name is derived from solution parameters before source
emission, so it does not change when an emitter produces a different instruction
sequence for the same solution.

A review probe replaced the integer-to-float conversion in the S11a
alpha-before-load-C path with a same-register move. The S11a emission and saved
result tests both still passed because the changed assembly was discarded before
the assertion.

Full assembly text is too large for these saved results. Register allocation,
labels, and the order of independent instructions also vary between otherwise
equivalent runs.

## Decision

Add a SHA-256 digest of the emitted opcode set to every saved result produced by
`assert_config_emits_golden`. The set keeps register allocation, labels,
instruction counts, and independent scheduler reorderings from changing the
digest. Keep the kernel name and emitter return code so changes in solution
selection and generation status remain visible.

The format migration also refreshes the 15 basename-only mismatches reported by
the stack's public coverage job. The affected nodes retain their kernel counts
and emitter return codes under the current supported compiler.

Four derivation-rejection tests previously saved only the integer `0`, which
could not distinguish the intended rejection from an earlier unrelated one.
Those tests now run solution generation serially and compare the complete
multiset of rejection diagnostics instead of saving the solution count.

## Consequences

Adding or removing an instruction kind now fails the affected test even when
the solution name and emitter return code stay constant. Reviewers must inspect
the generated assembly before accepting a digest change; the digest detects a
change but does not explain it. Register operands, instruction counts, and
instruction ordering remain outside this saved result because they can vary
between runs.

The saved results remain compact, but they are specific to the normalized output
of the supported assembler. After an intentional toolchain or code-generation
change, update only the affected nodes and record the reason before accepting
the new values.
