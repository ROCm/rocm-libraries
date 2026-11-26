# Lane: probe violations (`PROBE`)

**Trigger:** any probe `expect` was violated, or a probe row ended UNEXERCISED. **Agent:** `task`. **Output:** `R/triage/PROBE.md`,
findings.

For each violated family, decide which is wrong: the engine, the expectation, or the oracle. Report the decision, not the count.

## Method

1. Read the rule that should make the engine decline the graph (matcher source, coverage doc rule id). No such rule in this engine
   (an `expect` copied from another engine): the expectation is wrong.
2. Replay the violating graph at **full** enforcement (`HIPDNN_TEST_HOST_INPUT_FILL=1` fills inputs on the host when the device
   fill does not handle a very large tensor).
3. Decide:
   - **engine** (`applicability` or `defect`): it accepts what its rule forbids, or accepts and computes wrongly;
   - **expectation** (`INFO`, fix the space on both branches): it accepts, computes correctly, the rule allows it;
   - **oracle** (`harness-gap`): UNEXERCISED with the reason read from the outcome (an accepted case at `buildable` enforcement is
     always UNEXERCISED: the family must not run at `buildable` if the engine may accept it).
4. For an expectation fix, write the corrected family (`expect`, enforcement) in `Fix:` with the rule id.
