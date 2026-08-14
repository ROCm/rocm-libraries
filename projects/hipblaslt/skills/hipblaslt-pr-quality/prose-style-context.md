# Prose and Style Context

## Purpose

Use this context when writing pull-request descriptions, review comments, handoffs, design notes, and engineering summaries for any change in rocm-libraries, whether the code is C++ or Python and whether it touches production code, build files, or tests.

The writing must help a busy engineer understand the change without first reading the code or earlier pull requests. Technical accuracy is required, but repository knowledge is not assumed.

## Reader model

Write for a high-school senior who:

- has written novice Python and C++;
- understands functions, classes, lists, dictionaries, pointers, files, exceptions, tests, and command-line tools;
- has not seen rocm-libraries or the changed code;
- may not know GPU terminology, ROCm conventions, math-library or kernel concepts, acronyms, or processes; and
- will stop reading if the explanation is vague, inflated, or unnecessarily long.

An experienced engineer should still find the text precise. Plain language must not remove important technical facts.

## Priority order

When goals conflict, use this order:

1. Correctness
2. Clarity
3. Necessary context
4. Concision
5. Elegance

Do not shorten a sentence by replacing an explanation with jargon.

## Length limits

- Aim for about 200 words for a pull-request description.
- Use 201–350 words when definitions materially improve understanding.
- Never exceed 300 words without explicit approval.
- Prefer sentences under 25 words. Treat 30 words as a warning.
- Keep most paragraphs to two through four sentences.

These are readability limits, not targets. A complete 160-word description is better than a padded 200-word description.

## Sentence-level gate

Apply every question below to every sentence. If any answer is no, rewrite the sentence.

1. Can the reader identify what object, function, file, or process the sentence is about?
2. If a name or acronym appears for the first time, is it defined in plain language?
3. Are the remaining terms common to a Python programmer? If not, are they explained?
4. Does the sentence make sense without reading an earlier pull request?
5. Does it express one main idea with a clear subject and verb?
6. Is it concrete about an input, output, behavior, failure, or consequence?
7. Does the reader know why this fact matters?
8. Could a simpler verb replace an abstract noun phrase?
9. Does it avoid slogans, sales language, and clever phrasing?
10. Does it describe the current net change rather than discarded development history?
11. Is every factual claim supported by the diff, a test result, or an authoritative source?
12. Could the target reader paraphrase it correctly after one reading?

## Explain concepts before identifiers

Introduce the idea first, then give the code name.

Bad:

> `clusterEnabled` distinguishes the disabled unit cluster from changes to either axis.

Good:

> A graphics processor may run neighboring groups of work together as a cluster. `clusterEnabled` accepts `[x, y]`, the cluster size in two directions. It returns `False` for `[1, 1]` and `True` when either value is greater than 1.

Use backticks for literal identifiers, commands, values, and paths. Do not expect an identifier to explain itself.

## Terminology rules

- Define an acronym on first use: “Windows Subsystem for Linux (WSL).”
- Treat code-specific names such as `PAP`, `TDM`, `StreamK`, and `F32X` as labels, then explain what behavior they control or what is being checked.
- Explain domain terms with a short concrete phrase: “tile, or block of matrix work,” “kernel, or function that runs on the GPU.”
- Define a technique the first time the change depends on it, in one plain sentence. Examples: mutation testing makes temporary source changes and checks whether tests detect them; a template is C++ code the compiler stamps out once per type; an RAII type releases a resource automatically when it goes out of scope.
- Prefer “saved expected result” and optionally introduce “snapshot” in parentheses.
- Prefer “source-code version” over “revision” when Git knowledge is unnecessary.
- Prefer “automated GitHub checks” over an unexplained “CI.”
- Prefer “tests that record current behavior” over an unexplained “characterization tests.”
- Prefer “the set of functions and types other code calls” over an unexplained “API,” and name what compatibility is at stake when the binary interface (ABI) can change.

Avoid compressed review shorthand unless it is a literal project term and is explained. Each word below tends to replace a concrete fact with a vibe; if you use one, make sure the sentence still states the input, output, behavior, or measurement it stands for. Warning words include:

Review and workflow shorthand:

- surface
- contract
- pin
- wiring
- provenance
- fail-closed
- golden
- survivor
- harden
- durable
- robust
- comprehensive
- surgical
- idiomatic
- canonical
- first-class
- source of truth
- end-to-end
- battle-tested

Vague-improvement words (say what changed and by how much):

- clean up
- clean, cleaner
- simplify
- streamline
- optimize
- improve, enhance
- refactor (when used as the whole explanation)
- modernize
- tweak
- fix up
- properly, correctly (as the only description of the new behavior)
- gracefully

Hand-waving qualifiers (delete or replace with a fact):

- just, simply, merely
- obviously, clearly, of course
- basically, essentially
- various, several, a number of (give the count)
- etc., and so on (name the remaining items or stop the list)
- as needed, if necessary (state the condition)
- should (say what it does, or what is unverified)

Reliability and scale claims that need evidence:

- scalable
- performant, fast, efficient (give the measurement)
- lightweight
- production-ready
- thread-safe (name what is now safe under which access)
- backward-compatible (name the interface and the consumers)

Also avoid marketing language such as “powerful,” “seamless,” “best-in-class,” “game-changing,” “blazing-fast,” “rock-solid,” “elegant,” or “cutting-edge.”

## Concrete language

State observable behavior instead of broad intent.

Bad:

> This hardens LibraryIO serialization contracts.

Good:

> The tests verify that an omitted output format selects YAML and that an unknown format prints the current error.

Bad:

> The workflow preserves provenance and enforces strict semantics.

Good:

> The script records the source-code version and requires the unchanged tests to pass before and after the temporary edit.

Bad:

> Coverage was improved.

Good:

> The tests executed 95.54% of `Utilities.py`.

Bad:

> This refactors the allocator for robustness.

Good:

> `TensorBuffer` now frees its device memory in a destructor instead of a manual `free()` call, so an early return no longer leaks the buffer.

Bad:

> Improved kernel performance.

Good:

> The change tiles the inner loop by 64 elements, which reduced runtime on the `sgemm` benchmark from 1.8 ms to 1.2 ms on gfx942.

## Pull-request structure

Keep the repository-required headings. Each section has one job.

### `JIRA ID`

Provide the exact issue key. Keep the issue in the body when the title intentionally omits it.

### `Motivation`

In two through four sentences:

1. Explain what the component does.
2. Identify the missing or unsafe behavior.
3. State the practical consequence.

Do not begin with the implementation or a list of files.

### `Technical Details`

Describe the current net diff. Use bullets only for parallel, independently reviewable facts.

- Name exact inputs and expected outputs when useful.
- Give a small example when it explains more than a label.
- State whether production code, configuration, saved expected results, or device behavior changes.
- Link the direct stack dependency.

Do not describe files removed during an earlier decomposition if they are absent from the current diff.

### `Test Plan`

State what should be run and what behavior each group checks. Use exact file or module names when they help the reader reproduce the work.

### `Test Result`

Separate completed evidence from pending automation.

- Report exact pass, skip, and saved-result counts.
- Do not call an environment failure a test failure.
- Say “Automated GitHub checks are pending” when that is the current state.
- Do not claim a run occurred unless evidence exists.

### `Submission Checklist`

Preserve the repository-required checklist item.

### `Risk level`

State both the level and the reason. Name what can change.

Bad:

> Low risk.

Good:

> Low (1/5): test-only changes; product and device behavior are unchanged.

## Link policy

Use a link when a necessary explanation would make the body long or distract from the change.

- Link to authoritative repository documentation before external summaries.
- Use descriptive link text instead of a raw URL.
- Keep the core purpose, behavior, evidence, and risk in the pull-request body.
- Do not make readers follow a link merely to understand the first paragraph.
- Do not link to stale branches, private paths, temporary files, or unverified documents.

A link provides depth. It does not replace the explanation.

## Formatting

- Use the minimum number of headings required by the repository template.
- Use bullets for parallel facts, not for ordinary prose.
- Avoid nested lists unless the hierarchy is essential.
- Avoid bold emphasis as a substitute for organization.
- Use active voice when it names the responsible code clearly.
- Keep code identifiers in backticks.
- Use numerals for exact inputs, counts, versions, and percentages.
- Keep punctuation and capitalization consistent across bullets.

## Evidence and scope discipline

- Describe what the pull request contains now.
- Distinguish changes to tests, build files, and configuration from changes to production behavior.
- Distinguish current behavior from behavior judged to be correct.
- Do not imply that passing tests prove correctness when they only record existing behavior.
- Do not use a single metric, such as line coverage, mutation score, or a benchmark delta, as a quality claim by itself. State what was measured, on what input, and on what hardware or platform.
- Do not hide pending checks behind a broad statement such as “all tests pass.”
- Do not claim device coverage for host-only tests, or claim a result on one GPU architecture holds on another.
- When the change can affect the set of functions and types other code calls, or the binary interface, say so and name the affected consumers.
- Do not add historical explanation unless it helps a reviewer evaluate the current diff.

## Drafting workflow

1. List every technical noun and acronym in the draft.
2. Mark which ones the reader model may not know.
3. Define, replace, or remove each unfamiliar term.
4. Put the component’s purpose before its implementation details.
5. Replace broad claims with an input, output, example, or measured result.
6. Apply the sentence-level gate one sentence at a time.
7. Read the description as an independent entry point, not as one layer of a known stack.
8. Remove repetition and development-history residue.
9. Confirm every number and test claim against recorded evidence.
10. Check the required headings, issue key, dependency, and word count.

## Final acceptance test

Before delivering prose, a reviewer should be able to answer all five questions without opening the diff:

1. What does the affected component do?
2. What specific behavior was not adequately checked or documented?
3. What exactly does this pull request add or change?
4. What evidence has passed, and what is still pending?
5. What production or device behavior can this change affect?

If any answer is missing, the prose is not ready.
