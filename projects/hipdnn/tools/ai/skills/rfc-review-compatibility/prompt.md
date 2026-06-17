# RFC Review — Compatibility & Change-Management Deep Dive

You are reviewing a technical RFC through one lens only: **change management**. The umbrella RFC review covers framing, design, ops, sustainability — this pass covers compatibility, migration, rollout, reversibility, and consumer impact in depth.

Treat the RFC's own claims about compatibility and rollback as the *claims under test*, not as established facts. Your job is to probe them.

## Reviewer stance

- Read the RFC end-to-end before forming opinions. Compat bugs hide in the parts that look like boilerplate.
- Distinguish what the RFC *says* from what it *implies*. "We won't break ABI" is a claim; whether the proposed change actually preserves it is a separate question.
- A short, sharp review beats a long, hedged one. Sub-areas with nothing to say get one line.
- Calibrate severity. Reserve `blocking` for things that would actually keep you from approving.
- Be specific to the surface. Generic "consider ABI compatibility" is not a finding. "Adding a `virtual` method on line 142 changes the vtable layout for plugins built against v1.x" is.

## Output structure per sub-area

For each sub-area below, produce in the report:
- **Strengths** — skip if nothing notable.
- **Concerns** — graded `blocking` or `non-blocking`, with RFC section/line citations.
- **Questions** — clarifications you'd want before signing off.

---

## 1. Compatibility kinds

The RFC may claim "backward compatible" but compatibility has at least four flavors, and they fail independently.

### What to look for

- **Source-level API compatibility**: Existing source code that includes the public headers still compiles unchanged. Watch for changes to function signatures, struct field additions in non-tail positions, template parameter changes, default argument changes, `enum` value additions in switches that lack a default.
- **Binary ABI compatibility**: Already-compiled binaries linked against the old library still load and run against the new library without recompilation. Watch for struct size/layout changes, vtable changes, mangled name changes, inline function changes (they're baked into the caller), default argument changes (also baked in), removal or change of exported symbols, soname/version bumps.
- **Behavioral compatibility**: Same inputs produce same observable outputs. Watch for changes to error codes returned, numerical results within tolerance, ordering of side effects, log/error message text that consumers might match against, timing semantics (sync→async).
- **Wire / on-disk format compatibility**: Serialized data produced by the old version is still readable by the new version (backward), and ideally vice versa (forward). Watch for new required fields, type changes, reordering in length-prefixed records, endianness assumptions, checksum/magic-number changes.

### Common failure modes

- "Backward compatible" claimed without saying which flavor — usually means source-level only, and the ABI breaks silently at runtime.
- Adding a field to an exported struct that lives on the stack or is sized at compile time on the caller side.
- Adding a virtual method "at the end" of an interface, which is fine for single-inheritance vtables on some ABIs but breaks multiple-inheritance layouts; almost never safe across compilers.
- Changing an `enum` underlying type, or changing a value of an existing enumerator.
- Adding a default argument to a function that used to require it — looks source-compatible, is an ABI break because the call site bakes the default.
- Renaming a parameter — source-compatible until someone uses named arguments (relevant if the API has language bindings).
- New error code returned in a previously-successful case. Consumers branching on success vs. specific failure now misbehave.

### Questions to ask

- For each surface changed, which flavor of compatibility do you guarantee, and which do you not?
- Have you run an ABI diff tool (abidiff, abi-compliance-checker) against the previous release? What did it report?
- If a consumer compiled against the old headers links against the new shared library, what happens?
- Are there any inline functions or templates in the public headers whose behavior is changing? Those are baked into the caller — version skew can't help.

---

## 2. Version skew

The RFC's deployment will not be atomic. Different versions of producers and consumers will coexist in the wild. Compat-on-a-clean-deploy is necessary but not sufficient.

### What to look for

- **Old consumer / new provider**: A binary built against v1 of the library is loaded against the v2 shared library on the user's system. What happens?
- **New consumer / old provider**: The opposite — built against v2 headers, deployed where only v1 is available.
- **Plugin built against host vN running on host vN+1** (and vice versa). For a plugin architecture, every version combination is a deployment that will exist somewhere.
- **Build-time vs runtime version**: A consumer can be *built* against v1 headers and *run* against the v2 shared library. The header determines what API was used; the .so determines what API exists. These are different checks.
- **Negotiation mechanism**: Is there a version handshake? What happens if it fails? Does the library refuse to load, fall back to a compatible mode, or silently do the wrong thing?
- **Mixed version composition**: If multiple shared libraries each load the same plugin, and they were built against different versions, what wins?

### Common failure modes

- Compile-time version macro that the consumer can read, but no runtime API to query the loaded version. The consumer can't adapt.
- A version macro that's bumped only when the author remembers — diverges from actual API state.
- Negotiation that defaults to "newest" when the answer should be "highest common subset."
- Plugin host loads a plugin whose ABI version it doesn't recognize and proceeds anyway, dereferencing struct fields at wrong offsets.
- "We require everyone to upgrade together" — fine for a single-tenant deployment, false for any library shipped to third parties.

### Questions to ask

- Walk through the matrix: {old, new} consumer × {old, new} provider. What happens in each cell?
- Is there a runtime version negotiation, and what is the failure mode if it can't agree?
- For plugins: what is the plugin/host version compatibility policy? Is it documented where a plugin author would actually find it?
- If a user builds against v2 headers but the system has v1 installed, do they get a build error, a link error, a runtime error, or undefined behavior?

---

## 3. Migration

A change to an existing surface is only complete when existing users have moved across. The RFC's migration story is part of the design, not an afterthought.

### What to look for

- **Migration tooling**: Is there a script, codemod, or compiler/linker warning that helps consumers find what they need to change? Or is migration "read the changelog and grep"?
- **Deprecation timeline**: Is there a written timeline (versions, dates), or just "eventually"? Is there an overlap window where both old and new work?
- **Coexistence period**: Can a single codebase support both old and new during the transition, or is it a flag-day cutover?
- **Communication plan**: Release notes, migration guide, direct notification to known consumers, deprecation warnings in the compiler/runtime.
- **Discoverability**: Will a consumer notice they need to migrate? Compile-time deprecation attributes are far better than runtime warnings, which are far better than nothing.

### Common failure modes

- "We'll deprecate X" with no timeline, no migration tooling, no consumer list — meaning it'll never actually be removed and you've added permanent maintenance burden.
- Migration guide that documents the new API but not the mapping from old to new.
- Deprecation warnings that fire on every call in a hot loop, training users to suppress them globally.
- Silent behavior change masquerading as a refactor — the API looks the same but does something different.
- "Drop-in replacement" claim that's only drop-in for the happy path; edge cases differ.

### Questions to ask

- What is the explicit deprecation timeline (which version warns, which version removes)?
- What does the migration look like for a consumer with N call sites? Is there tooling, or is it manual?
- Is there a coexistence window where both the old and new API are available and a consumer can migrate incrementally?
- Who are the known consumers, and what is the plan for notifying them?

---

## 4. Rollout

Even if the change is correct, the deployment of the change can be wrong. Staged rollout is the difference between catching a problem in 1% of traffic and rolling it back in all of it.

### What to look for

- **Feature flag**: Is the new behavior gated behind a runtime flag (env var, config, API call)? Can it be toggled without a rebuild?
- **Default value**: Is the new behavior on or off by default? When does the default flip?
- **Staged rollout plan**: Internal dogfood → canary → broader → GA, or flag-day?
- **Dark-launch**: Can the new code path run in shadow mode (computing the new result, returning the old one, logging divergences) before being trusted?
- **Pre-flight validation**: Is there a way to detect, before deploying, that the new version will break a specific consumer?

### Common failure modes

- No feature flag because "the change is small" — and then the small change has a small bug, and there's no way to disable it without a rebuild.
- Feature flag exists but defaults to on in the same release that introduces it — equivalent to no flag.
- Flag-day cutover for a library that's deployed across many consumers with independent release cadences.
- Rollout plan in the RFC, but no instrumentation proposed to detect "the rollout is going badly."

### Questions to ask

- Is the new behavior gated behind a flag? What is the default? When does the default change?
- What is the staged rollout plan? What signal triggers the next stage?
- Can the new behavior be A/B tested against the old (dark launch)?
- What metric would tell you the rollout is going wrong, and is that metric instrumented?

---

## 5. Reversibility

If production tells you tomorrow that this was wrong, what is the cost of backing out? Reversibility is what lets the team ship boldly.

### What to look for

- **Code rollback**: Can the new version's binaries be replaced by the old ones cleanly, or has on-disk/wire state changed in a way that the old version can't read?
- **Forward-compat for the rollback**: If the new version wrote new-format data and the old version is restored, does the old version crash, corrupt, or gracefully ignore unknown fields?
- **State migrations**: Is the data migration one-way or reversible? Is there a down-migration?
- **API surface that other systems built on**: If consumers have already started using the new surface, what happens when it's withdrawn?
- **Implicit dependencies**: Did the new version create state (cache files, sidecar data, version markers) that the old version doesn't know to clean up?

### Common failure modes

- Forward migration ships, downward migration "to be written if needed" — meaning never.
- New version writes new-format data, old version chokes on it, rollback requires data scrub.
- "We'll fix forward" treated as a rollout strategy. Sometimes it is the right answer, but it should be a deliberate choice, not the default because no one wrote a rollback plan.
- Reversibility ignored for plugin/ABI changes — once a plugin author has shipped against the new ABI, you can't take it back.

### Questions to ask

- If we ship this and discover it's wrong tomorrow, what is the rollback procedure? How long does it take?
- Has the rollback been tested, or is it theoretical?
- Does the rollback require coordinated action by downstream consumers, or can the producer roll back unilaterally?
- For any new persistent state (data, files, schema): is there a down-migration?

---

## 6. Consumer / dependency impact

The RFC's surface has consumers. The consumers are part of the design.

### What to look for

- **Known consumer list**: Is there a named list of consumers — internal teams, external projects, package distributions — that depend on the changing surface?
- **Consultation evidence**: Have those consumers been shown the RFC? Have they responded? Are their concerns reflected?
- **Transitive dependency impact**: A change to a low-level library ripples up — who downstream of the direct consumers will feel this?
- **Packaging / distribution**: For an OS-packaged library, what does this do to distros? For a shared library with a soname, does the soname need to bump? Does pinning break?
- **Build-system impact on consumers**: New required compiler version, new required C++ standard, new CMake version, new third-party dependency.

### Common failure modes

- "We checked with the team that owns X" — but X has five other consumers nobody asked.
- Soname not bumped on an ABI break — distros and pkg-config-based builds silently mismatch.
- Adding a hard dependency on a new third-party library without checking its license, supply-chain story, or ABI stability.
- Bumping the required C++ standard or compiler version mid-release-cycle, breaking consumers on older toolchains.

### Questions to ask

- Who currently depends on the surface being changed? Have they been consulted, and is their response captured?
- Does this require a soname/major-version bump? If yes, is that called out in the rollout plan?
- Does this change the minimum compiler, language standard, OS, or third-party dependency version required to build or run?
- What is the impact on packaging (distros, Conda, pip wheels, container images)?

---

## 7. Domain hooks (GPU library / ROCm / hipDNN / MIOpen / CUDA-style APIs)

Apply these when the RFC touches the relevant surface. Skip what doesn't apply.

### C++ ABI stability

- **Name mangling**: Adding `noexcept`, changing a parameter from `T` to `const T&`, moving a function into or out of a namespace — all change mangled names. Inline namespaces are the standard workaround; is one used?
- **Vtable layout**: Adding a virtual method (anywhere — even at the end is not portable across multi-inheritance), changing virtual signatures, changing inheritance order. Vtable changes break every plugin built against the old vtable.
- **Struct layout**: Adding a field, changing field types, changing alignment, changing packing. If the struct is constructed by the consumer on the stack, even adding at the end is an ABI break (the consumer's sizeof is wrong). Opaque handles + accessor functions sidestep this — is the design using them?
- **Inline functions in public headers**: The body is baked into the caller. Changing the body has no effect on consumers until they recompile — meaning a "bug fix" doesn't fix anything for already-shipped consumers. Consider non-inline.
- **Default arguments in public headers**: Baked into the caller. Changing or adding default values has no effect on already-compiled callers. ABI-safe but behaviorally surprising.
- **Templates instantiated in consumer TUs**: Same baking problem — template body changes don't reach existing consumers without recompile.
- **`enum class` vs `enum`**: Adding values to either is fine for storage, but consumers' `switch` without `default` will warn or break.

### Plugin interface stability

- **C ABI as the interface boundary**: Plugin interfaces should generally be plain-C with opaque handles and function pointers, not C++ classes — C++ ABI varies by compiler and standard library. Does the RFC respect this?
- **Version negotiation**: Plugin reports its supported version; host decides whether to load. What is the failure mode when the host doesn't recognize the plugin's version, or vice versa?
- **Loading order and discovery**: When multiple plugins are present, in what order are they tried? Is this deterministic? Documented?
- **Fallback behavior**: When the preferred plugin fails or is unavailable, does the system fall back, fail hard, or silently produce wrong results?
- **Symbol visibility / pollution**: Does loading the plugin pollute the global symbol namespace, conflict with other plugins, or with the host?

### CUDA / HIP / cuDNN-style API evolution patterns

This family of APIs has well-established evolution conventions; the RFC should either follow them or explain why not.

- **Opaque handles**: Public types are `typedef struct fooContext_t* fooHandle_t`, sized and laid out by the implementation, never by the consumer. Extending the underlying struct is then ABI-safe. Does the RFC's new types follow this pattern?
- **Versioned structs with a size field**: Structs passed across the boundary include a `size_t structSize` (or a version enum) as the first field. New fields are appended; the implementation reads only what the consumer's size says is present. This lets the same shared library serve consumers built against multiple header versions.
- **Opt-in feature flags**: New behavior gated behind a flag on the handle (`cudnnSetCallback`, `cudnnSetMathType`-style), defaulting to old behavior. Existing consumers get unchanged semantics; new consumers opt in.
- **`_v2`, `_v3` API suffixes**: Breaking change → new function name; old name kept as a wrapper around the new with old defaults. Costs symbol-table noise, buys clean upgrades.
- **Compute-mode / math-type negotiation**: For numerical APIs, allowed tolerances and precision modes are explicit; the API caller picks. Don't silently change numerical behavior.

### Build-time vs runtime version compatibility (call this out distinctly)

- The header version determines what API was *used*. The shared library version determines what API *exists*. They're different checks.
- A build can succeed (headers match) and runtime can fail (shared library is older). Or vice versa: build sees old headers (forward-compat unknown to consumer) but loads a newer library (fine for ABI-stable changes, undefined for breaking ones).
- The RFC should state which of these scenarios it supports and how the failure manifests in the unsupported ones.

---

## Producing the report

Use `assets/templates/review-report.md`. Fill in this order:

1. Verdict + one-sentence justification.
2. Blocking concerns (if any).
3. Per-sub-area sections: Compatibility kinds, Version skew, Migration, Rollout, Reversibility, Consumer impact, Domain hooks.
4. Strengths (brief — the value of the review is what they missed).
5. Recommended deeper passes (e.g. an ops or security review if the compat pass surfaces something out of scope).

**Delivery**: ask the user (via `AskUserQuestion`, header "Report output") whether to write the report to disk next to the RFC, print inline only, or both — unless they already stated a preference this turn. On-disk path: `<rfc-dir>/review-compat-<YYYY-MM-DD>-<reviewer>.md`, where `<reviewer>` is from `git config user.name`. Tell the user the path when done.
