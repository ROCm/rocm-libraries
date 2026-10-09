# Profile comparison

Compare profiles by running the same normalized access set through each exact
target profile. Preserve the opcode, wave size, access widths, active state, and
byte addresses so the target is the only changed input.

Report:

- selected target and profile version;
- supported versus unsupported status;
- access classifications;
- conflict and broadcast group membership; and
- maximum semantic multiplicity.

Keep profile identity separate from behavior. Shared results do not make one
target an alias or fallback for another. Different results establish a scoped
prediction difference for that request; they do not establish a general hardware
or performance comparison.

When profiles cover different operation sets, report the coverage difference.
Do not fill a missing result with another profile's output. When comparing a
saved result, preserve its profile version so a later rule revision is visible.
