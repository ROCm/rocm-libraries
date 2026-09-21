# RFC Security Review — Detailed Checklist

You are doing a focused security review of an RFC. The goal is the review a security-minded engineer with codebase context would give: catch the realistic threats, name the unstated trust assumptions, and ask the questions that surface hidden exposure. Avoid OWASP-checklist-recitation; the value is engineering judgment about *this* design in *this* domain.

## Reviewer stance

- **Start with the threat model.** Every other lens is meaningless without one. If the RFC doesn't state one, infer the most realistic one for the domain and label it as inferred. Then judge the design against it.
- **Calibrate to the domain.** A numerical GPU library's realistic attacker is not a remote network adversary; it is a malicious model file, a co-tenant on shared hardware, or a compromised dependency. A web service's attacker is anyone with a URL. Don't borrow threats from the wrong category.
- **Resist invented threats.** "What if the attacker had quantum capability" is noise. If the realistic surface is narrow, say so plainly — that *is* the review.
- **Distinguish stated trust from implied trust.** The bugs are almost always in things the RFC treats as trusted without saying why.
- **A short, sharp review beats a long, hedged one.** A sub-area with nothing to say is one line: "Crypto — N/A, no crypto in scope."
- **Cite specifics.** Every concern names the RFC section or line; otherwise the author can't act on it.

## Domain hooks: ROCm / hipDNN / MIOpen / GPU library context

The user's primary codebase is a C++ GPU library shipped to downstream consumers. Calibrate the threat model accordingly:

- **The realistic attacker is usually not "evil on the network."** It is most often:
  - A **malicious model file or serialized graph** that triggers OOB read/write, integer overflow in shape arithmetic, or arbitrary code through parser/deserializer bugs.
  - A **memory-safety bug** (UAF, buffer overflow, integer overflow) reachable from user-controlled input — tensor dimensions, layout descriptors, kernel selection parameters.
  - **Shared-host scenarios** (multi-tenant GPU): isolation between processes / containers, GPU memory residue, side channels via timing or contention.
  - **Supply chain** on dependencies and build artifacts — downstream consumers trust your binaries, so a compromised dep or unsigned release propagates.
  - **Plugin / loadable-backend trust** — a backend or plugin loaded into your process runs *as* your process. Same address space, same privileges. Treat plugin discovery and loading as a trust decision.
- **C++ memory safety** in the implementation language is structurally part of the threat model, not an afterthought. Integer arithmetic on shapes (`n*c*h*w*sizeof(T)`) overflows; format-string-style bugs in logging happen; OOB reads in workspace sizing happen.
- **File I/O paths** matter — model loading, **MIOpen kernel-DB / perf-DB caches**, config files, temp directories. Ask: who can write to these paths? Whose permissions and umask apply? What if the cache is symlinked?
- **Privilege model is usually flat** (user-space library). "Least privilege" mostly means: do not require root, do not ship setuid helpers, do not write outside the user's expected directories, do not need CAP_SYS_*.
- **Telemetry, crash dumps, error messages** — anything that leaves the host. What is included? Opt-in or opt-out? Is the user told? Does it contain model fragments, paths, hostnames, GPU IDs that could fingerprint a tenant?

When the RFC is *not* in this domain, ignore these hooks and apply judgment appropriate to its actual surface (e.g., a build-system RFC is mostly supply-chain and build-integrity; a public-API RFC is mostly trust-boundary and input validation).

---

## The areas

For each area below, produce three things in the report:
- **Strengths** — what the RFC does well on this axis (skip if nothing notable).
- **Concerns** — issues, graded `blocking` or `non-blocking`, with RFC section/line citations.
- **Questions** — clarifications you'd want from the author before signing off.

If an area genuinely does not apply, write one line and move on.

---

### 1. Threat model

**This is the most important section. Read it first, write it first.** An RFC with no threat model is the most common defect in security review — and once you supply one, half the other concerns become obvious.

**What to look for**
- Does the RFC name the realistic attacker — who, with what capability, after what asset?
- Is the attacker model honest about the deployment context (shared host, multi-tenant cluster, single-user workstation, CI runner, downstream OEM customer)?
- Are non-threats explicitly out of scope? ("Physical access to the host" is usually out; saying so prevents scope creep.)
- Does the RFC distinguish between threats the design *prevents*, threats it *detects*, and threats it *accepts*?

**Common failure modes**
- No threat model at all. Author treats security as "we'll handle it later."
- Threat model copy-pasted from a generic template ("CIA triad, defense in depth") with no specifics.
- Threat model that names a flashy attacker (nation-state, APT) and ignores the realistic one (a fuzzed input, a careless dep update).
- Implicit trust assumption never named — e.g., "we assume the model file is well-formed" but the design loads arbitrary user-supplied files.
- Threats and mitigations conflated — "we mitigate X by doing Y" without ever describing what X actually is.

**Questions to ask**
- Who is the realistic attacker for this feature? What capability do they have? What are they trying to achieve?
- What threats are explicitly out of scope, and why is that defensible?
- What in this design must remain trusted, and on what basis?

---

### 2. Trust boundaries

**What to look for**
- Map every boundary the design crosses: process, container, network, user, privilege, host, tenant.
- At each boundary: what crosses it (data, code, control), in which direction, and what validates it on entry?
- Where does *untrusted* input become *trusted*? That transition is where validation has to be airtight.
- Does the design rely on an upstream component's validation? Is that reliance stated?

**Common failure modes**
- Trust boundary moved without saying so — e.g., a library that previously only consumed in-process data now opens a socket or reads a file.
- Validation done on one side of the boundary and assumed on the other (classic TOCTOU surface).
- "The caller is responsible for sanitizing inputs" — fine, but then say so loudly in the API docs, not buried in an RFC paragraph.
- A plugin or shared-library boundary treated as a real isolation boundary. It is not — code in your address space is you.

**Questions to ask**
- What is the trust boundary between [component A] and [component B] in this design?
- If a malicious actor controls [boundary input], what can they do?
- Does any existing trust boundary move as a result of this RFC?

---

### 3. Attack surface

**What to look for**
- Enumerate *new* surface: input formats, network endpoints, files read, IPC sockets, env vars consumed, command-line flags, dlopen targets, callback hooks.
- For each: what code path does it reach, and is that code path reviewed for hostile input?
- New parsers and deserializers are the highest-risk additions — every one is a potential vuln.
- Reductions in surface count too — if the RFC removes an endpoint, that's a strength.

**Common failure modes**
- A "config file" introduced that turns out to be a Turing-complete DSL.
- A serialization format chosen because it was convenient, not because it is safe to parse hostile input (pickle, eval, YAML with object deserialization, arbitrary protobuf with `Any`).
- New env var or file path read with no documentation of who controls it.
- A debug-only endpoint introduced "just for now" with no plan to remove it.

**Questions to ask**
- What new inputs does this design accept? Where does each come from?
- Is the parser for [format X] hardened, or are we relying on it not being targeted?
- What environment variables, config files, or sockets does this read or open?

---

### 4. Input validation & injection

**What to look for**
- Every parser is a vuln until proven otherwise. Look at the choice of parser/library and whether the design constrains what it accepts.
- Every concatenation into a shell command, SQL query, file path, log line, format string, or HTML/JS context is a potential injection.
- Tensor / shape arithmetic on user-controlled dimensions: integer overflow, negative values cast to size_t, multiplication producing nonsense sizes that then get passed to `malloc`.
- Path traversal: any user-supplied filename joined into a directory path needs canonicalization and a containment check.
- Length checks: every buffer interaction needs a bound that comes from the *reader*, not from the input.

**Common failure modes**
- `snprintf(buf, sizeof(buf), user_string)` — format-string bug.
- `system("foo " + user_input)` — command injection. Same with `popen`, `execlp` with shell.
- `n * c * h * w * sizeof(T)` overflowing to a small number, then allocating a tiny buffer and writing the intended size into it.
- Negative shape value cast to size_t, becomes huge, allocator returns nullptr or commits the host.
- "We validate at the API boundary" — and then internal functions are called from elsewhere and skip validation.
- Deserializing pickle, untrusted YAML, untrusted XML with external entities, protobuf `Any` without type-allowlisting.

**Questions to ask**
- What validates [user-supplied field X] and at what point?
- For shape/size arithmetic on user input — what prevents overflow on a 4-billion-element tensor?
- If the input parser receives a deliberately malformed [format Y], what happens?

---

### 5. Cryptography

If the RFC has no crypto in scope, this is one line: "Crypto — N/A." Don't invent crypto concerns.

**What to look for**
- Algorithm choice: standard, current, with a documented reason for selecting it.
- Key management: where do keys come from, how are they stored, rotated, destroyed? Hardcoded keys are a blocker.
- RNG source: `/dev/urandom` or platform CSPRNG, not `rand()`, not seeded from time.
- Constant-time operations where they matter (comparing MACs, comparing secrets).
- Don't roll your own. Custom protocols, custom modes, "lightweight" hashes — almost always wrong.

**Common failure modes**
- ECB mode. MD5/SHA1 for anything other than non-security checksums. DES.
- Reusing a nonce / IV.
- Keys in source, in environment variables logged to crash dumps, in command-line args visible in `ps`.
- Comparing secrets with `memcmp` or `strcmp` (timing leak).
- TLS with cert verification disabled "for testing."

**Questions to ask**
- Why was [algorithm/library] chosen? Is it on a deprecation path?
- Where do keys live at rest? Who can read them? How are they rotated?
- Is any custom cryptographic construction proposed? If so, who reviewed it?

---

### 6. Authentication & authorization

If the design has no auth surface (e.g., an in-process library with no privilege boundary), this is one line.

**What to look for**
- Who can call what, and how is that enforced (not just intended)?
- AuthN failure mode: fail closed, not fail open. A check that throws-but-is-caught is fail-open.
- AuthZ checks at every entry point, not just the "main" one — admin endpoints, debug endpoints, IPC entry points.
- Distinction between authentication (who you are) and authorization (what you can do). Conflating them is a frequent design smell.
- Session/token lifetime, revocation, replay protection.

**Common failure modes**
- "Internal endpoint, no auth needed" — until it's exposed.
- AuthZ check on the UI but not on the API.
- Bearer tokens with no expiry and no revocation.
- Role check on a single chokepoint that turns out to be skippable.

**Questions to ask**
- What identity is required to invoke [operation X]? What enforces it?
- What is the behavior on auth failure — denied, prompted, logged, alerted?
- How is privilege revoked when a user/system loses access?

---

### 7. Secrets handling

**What to look for**
- How secrets enter the system (env, file, KMS, vault, prompt) and whether the path is documented.
- Where secrets live at rest — memory only, on disk, in a config, in source.
- What logs them — explicitly or accidentally via error messages, debug dumps, telemetry, crash reports.
- Rotation: is there a procedure, and is it tested?
- Memory hygiene: secrets zeroed after use, not landing in core dumps.

**Common failure modes**
- Secrets logged on error ("auth failed for token=abcdef…").
- Secrets included in crash reports / telemetry / bug reports.
- Secrets in source for "test" or "default" — these always escape to production.
- Secrets passed as command-line args (visible in `ps`, in shell history, in process accounting).
- Long-lived secrets where short-lived would do.

**Questions to ask**
- What secrets does this design need, and what is the lifecycle of each?
- What prevents a secret from appearing in logs, error messages, or telemetry?
- How is a compromised secret rotated, and how long until the rotation propagates?

---

### 8. Supply chain

**What to look for**
- New direct dependencies: who maintains, license, release cadence, known CVE history. New transitive dependencies pulled in by the direct ones.
- Build provenance: are artifacts reproducible? Signed? Built in a known environment?
- Dep update policy: is there one, or do deps drift?
- Pinning vs floating: pinned versions reduce supply-chain risk but defer it; floating gets fixes faster but lets compromised releases in.
- For C++ libraries shipped to downstream consumers: every dep you add becomes a dep your consumers inherit. Be honest about that cost.
- Build-time tools count too — a compromised codegen tool produces compromised binaries.

**Common failure modes**
- "We pull in this header-only library, no risk" — until it links transitively to ten other things.
- Vendored copy of upstream library, no plan to update, no record of which version it was.
- Build pulls deps from network at build time, no hash pinning.
- No SBOM, no way for downstream to know what's in the binary.
- New dep maintained by a single person with no continuity plan.

**Questions to ask**
- What new direct and transitive dependencies does this RFC add?
- How are dep updates triaged, and how quickly can a critical CVE be picked up?
- Are the build artifacts reproducible? Is there an SBOM?

---

### 9. Privacy

**What to look for**
- What user/host/tenant data the design observes, processes, transmits, retains.
- What appears in logs, metrics, traces, telemetry, error messages, crash dumps, bug reports.
- Whether data crossing a host boundary is opt-in or opt-out, and whether the user is meaningfully told.
- Data retention: how long, where, who can access.
- Inadvertent fingerprinting: hostnames, MACs, GPU UUIDs, model names, file paths can identify a tenant even without PII.

**Common failure modes**
- Telemetry opt-out buried in a config file no one reads.
- Crash dumps containing the model being processed, full file paths, environment.
- Error messages that quote the failing input verbatim — useful for debugging, also a privacy/secret-leak channel.
- Logs at default verbosity include identifiers a co-tenant or log aggregator shouldn't see.
- "Anonymized" data that isn't (small population + a few fields = identifiable).

**Questions to ask**
- What data leaves the host as a result of this design? Under what consent model?
- What appears in default-verbosity logs that wouldn't be appropriate to publish?
- What is the retention policy for [collected data], and who can query it?

---

### 10. Audit & forensics

If the design has no privilege boundary or sensitive operation, this is one line.

**What to look for**
- Can you tell, after the fact, who did what? With what input? When?
- Audit log integrity — append-only, signed, on a separate system, not editable by the actor being audited.
- Granularity: enough to investigate a real incident, not so much it becomes noise (or itself a privacy problem).
- Are admin actions logged? Are *failed* actions logged? Failure logs are often more useful than success logs.

**Common failure modes**
- Logs go to a file the operating user can `rm`.
- Auth-failure events not logged, so brute force is invisible.
- High-cardinality logs (every request body) that overflow retention and get truncated.
- Audit logs that contain the secrets that were being audited.

**Questions to ask**
- After an incident, what record would let us reconstruct what happened?
- What admin or privileged actions does this design enable, and how are they audited?
- Who can modify or delete the audit trail?

---

## Producing the report

Use `assets/templates/review-report.md`. Fill sections in this order:

1. **Verdict** + one-sentence justification (`approve` | `approve-with-comments` | `needs-revision` | `block`).
2. **Threat model summary** — call it out at the top. If the RFC didn't state one, supply your inferred model and label it `(inferred)`.
3. **Blocking concerns** — the bar for approval. Use sparingly.
4. Non-blocking concerns and questions, grouped by area. Skip any area with nothing to say.
5. **Strengths** — brief.
6. **Recommended deeper passes** — only if warranted.

**Delivery**: before writing, ask the user via `AskUserQuestion` (header: "Report output") whether to write the report on disk next to the RFC, print inline only, or both — unless they already stated a preference this turn. Options:

1. **On disk next to the RFC** — `<rfc-dir>/review-security-<YYYY-MM-DD>-<reviewer>.md`
2. **Inline in chat only**
3. **Both**

If writing to disk, use `git config user.name` as `<reviewer>` and tell the user the path when done.
