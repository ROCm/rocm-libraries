# PLAN — enable the deferred deeper-layer tools via a Docker image (hermetic, human-free)

**Scope, not implementation.** The parametric-chaos pipeline ran all three rollout surfaces with the
*optional* deeper-layer tools (CodeQL, ACTS, PICT, Daikon, Atheris) marked **unavailable** — the
stdlib fallbacks (intra-function `ast` def-use for slicing; built-in pairwise for covering arrays)
satisfied every deliverable, so no host install and no human step were needed. This plan removes the
"please install a tool" blocker by **baking the tools into a Docker image** instead of the host, so
the pipeline stays reproducible and runnable with zero human intervention.

Grounding facts (measured 2026-06-07 in `tl-char`): AlmaLinux 8.10, Python 3.12.10, ROCm/AMD clang
23, cmake 3.27.9, pip 26.1, **no JDK**, outbound HTTPS to github works (HTTP/2 200).

---

## Goal (one sentence)

Produce a prebuilt, reproducible container that makes CodeQL / PICT / Atheris (and, where they add
real value for *Python*, ACTS / Daikon) available to the pchaos workflow phases, so the
`preflight.json` `optional_tools` flip to `available` and each tool-gated phase upgrades from its
stdlib fallback to the real tool — with the same schemas, the same add-only/driver-gates-and-commits
discipline, and the same "deterministic helpers are the source of truth" rule.

### Achieved when
- A committed `env/Dockerfile.tools` builds an image (call it `tl-pchaos-tools`) from public sources,
  no click-through/manual download in the build path.
- `docker exec tl-pchaos-tools <tool> --version` succeeds for each enabled tool (receipt captured).
- The workflow preflight, run against that image, reports the enabled tools `available=true`.
- For each enabled tool, ONE end-to-end smoke proves it produces *real* output on a real pchaos
  input (not a fabricated stub), and the result is cross-checked against the stdlib fallback where
  both exist (e.g. CodeQL slice ⊇ ast def-use slice; PICT cases cover all stdlib pairs).
- No regression to the methodology-A gate; everything add-only; nothing pushed.

### Non-goals
- Replacing the stdlib fallbacks — they stay as the always-available floor and the cross-check oracle.
- Putting these tools in the ROCm `tl-char` image (see architecture decision below).
- Lifting coverage — these tools sharpen *analysis quality* (slices, covering strength, fuzz-found
  witnesses), not the methodology-A number, which remains a regression guard.

---

## Architecture decision: a SEPARATE tools image, not a `tl-char` layer

**Recommend a dedicated `tl-pchaos-tools` image (or a standalone Layer C with no ROCm dependency),
mounting the worktree at `/work`, NOT extending `tl-char`.** Rationale:

- The deferred tools are **source/solver/domain analyzers** — none need `rocisa`, ROCm, or a GPU.
  CodeQL slices source (`--build-mode=none`); PICT/ACTS consume a domain model; Atheris fuzzes
  extracted *pure helpers*; Daikon (if used) traces pure helpers. Keeping them off the multi-GB ROCm
  base keeps the image small, rebuildable, and decoupled from the rocisa build dance.
- Adding a JDK + CodeQL bundle + LLVM/libFuzzer to the ROCm image risks the carefully-pinned rocisa
  runtime (LD_LIBRARY_PATH, nanobind) for no benefit.
- The phases that DO need the real Tensile modules (Reify pin-tests, the real-entry `Tensile.Tensile`
  path) already run in `tl-char` and stay there. Only the *tool-gated* sub-steps run in the tools
  image. The workflow already passes a container name per phase implicitly via the `docker exec`
  prompt; generalize it to `args.toolsContainer` (default = `tl-char`, so today's behavior is
  unchanged) and route CodeQL/PICT/Atheris/Daikon steps to `tl-pchaos-tools`.

Base image: `almalinux:8` (matches tl-char's OS, keeps query/abi expectations consistent) **or**
`ubuntu:24.04` (newer JDK/clang, simpler apt). Recommend **ubuntu:24.04** for the tools image — its
apt has a current `temurin`/`openjdk`, `cmake`, `clang`, and `pictt` is trivial to build; the OS
match with tl-char doesn't matter for source-level tools.

---

## Per-tool scope (value for THIS Python pipeline, install path, caveats)

| Tool | Pipeline phase it upgrades | Value (Python) | Hermetic install in Docker | Effort | Recommend |
| --- | --- | --- | --- | --- | --- |
| **CodeQL (Python)** | Slice (interprocedural backward slice → public inputs) | **HIGH** — beats stdlib intra-function def-use; resolves cross-function/`globalParameters` derivations the `ast` seed misses | download official CLI bundle + `codeql/python-all` pack from GitHub releases (public, no click-through); `codeql database create --language=python --build-mode=none` | M | **YES (1st)** |
| **PICT (Microsoft, MIT)** | Combinatorial (covering arrays w/ constraints, higher strength) | **MEDIUM-HIGH** — real n-wise + constraint exclusion vs our built-in pairwise; OSS, tiny | `git clone microsoft/pict && cmake && make` (we have cmake/clang) | **S** | **YES (2nd)** |
| **Atheris (Google, Apache-2.0)** | Validate (coverage-guided fuzz of extracted pure helpers / parsers) | **MEDIUM** — finds witnesses/counter-examples Hypothesis' random search misses | `pip install atheris` (needs clang+LLVM/libFuzzer; 3.12 wheels exist but may build from source) | M | **YES (3rd)** |
| **ACTS (NIST)** | Combinatorial (alt covering-array engine) | **LOW once PICT exists** — same need as PICT | **license-gated**: NIST requires an email/click-through to download the jar → NOT hermetic | — | **SKIP** (PICT satisfies "ACTS/PICT") |
| **Daikon (UW, MIT)** | Validate (dynamic invariant detection) | **LOW for Python** — mature front-ends are Java (Chicory)/C; the Python path is experimental/thin | needs JDK; Python tracing is the weak link, not the install | L | **DEFER/DROP**; get dynamic-invariant value from Hypothesis observed examples instead |

Honest takeaways baked into the table: **PICT replaces ACTS** (same deliverable, OSS, hermetic), and
**Daikon is low-value for a Python target** — don't spend the JDK+frontend effort there; if dynamic
invariants are wanted, mine them from Hypothesis-generated examples on the extracted pure helpers.

---

## Dockerfile sketch (`env/Dockerfile.tools` → image `tl-pchaos-tools`)

```dockerfile
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv git curl unzip ca-certificates \
      cmake clang build-essential openjdk-21-jre-headless \
 && rm -rf /var/lib/apt/lists/*

# --- mandatory pchaos pip layer (same as tl-char, so solve/verify work here too) ---
RUN pip3 install --no-cache-dir --break-system-packages \
      z3-solver crosshair-tool hypothesis pysmt

# --- CodeQL CLI + Python query pack (public release; pin the version) ---
ARG CODEQL_VERSION=2.20.3
RUN curl -fsSL -o /tmp/codeql.zip \
      https://github.com/github/codeql-cli-binaries/releases/download/v${CODEQL_VERSION}/codeql-linux64.zip \
 && unzip -q /tmp/codeql.zip -d /opt && rm /tmp/codeql.zip
ENV PATH=/opt/codeql:$PATH
RUN codeql pack download codeql/python-all   # cache the standard Python libraries/queries

# --- PICT (Microsoft, MIT) ---
RUN git clone --depth 1 https://github.com/microsoft/pict /tmp/pict \
 && cmake -S /tmp/pict -B /tmp/pict/build && cmake --build /tmp/pict/build \
 && install /tmp/pict/build/cli/pict /usr/local/bin/pict && rm -rf /tmp/pict

# --- Atheris (Google, Apache-2.0); clang already present ---
RUN CLANG=$(which clang) pip3 install --no-cache-dir --break-system-packages atheris || \
    echo "atheris build failed on 3.12 — leave unavailable, fallback stays Hypothesis"
```

Notes: pin every version (CodeQL release tag, PICT commit) for reproducibility; the Atheris line is
tolerant (its 3.12 build is the one real risk — if it fails the preflight just keeps Atheris
`unavailable` and Validate uses Hypothesis, exactly as today). Build context needs no `./artifacts/`
(unlike tl-char) — this image is ROCm-free, so it builds anywhere.

---

## Workflow integration (small, backward-compatible)

1. **Parameterize the tool container.** Add `args.toolsContainer` to `parametric-chaos-characterize.mjs`
   (default `'tl-char'`). Route the CodeQL/PICT/Atheris/Daikon sub-steps' `docker exec` to it; leave
   Reify pin-tests and the deterministic helpers on `tl-char`.
2. **Preflight already classifies optional tools** — no change needed; with the tools image it simply
   reports them `available`, and the existing phase prompts already say "use the real tool if
   available, else the documented fallback."
3. **Slice phase:** when CodeQL available, build the DB once (preflight or an Inventory sub-step) and
   have each Slice agent query the interprocedural backward slice; **keep the `ast` def-use as the
   cross-check** (CodeQL slice must be a superset; log any symbol only the fallback found).
4. **Combinatorial phase:** when PICT available, emit a `.pict` model from `domain_model.json` +
   harvested constraints and parse its output into the SAME `covering_array/{model.json,cases.csv}`
   schema; **assert PICT cases cover every stdlib pairwise pair** (no silent regression in coverage).
5. **Validate phase:** when Atheris available, fuzz the extracted pure helper for each unit; any
   counter-example becomes a (verified, adversarially-checked) witness → reified add-only test.
6. **Unchanged invariants:** schemas identical; driver still re-runs the deterministic helpers +
   `finalize.py` for ground-truth counts (Lesson B); add-only; driver owns gate + commit; never push.

---

## Verification / trust (doer ≠ checker, measure-don't-inflate)

- Per tool: a captured `--version` receipt + ONE real-input smoke committed under
  `parametric-chaos/_tooling/<tool>/`.
- **Cross-check against the fallback, never replace blindly:** CodeQL slice ⊇ ast slice; PICT cases ⊇
  stdlib pairs; Atheris counter-examples re-run adversarially against the real predicate before
  reification (same Verify discipline as the SAT witnesses).
- The methodology-A gate stays the regression guard; run it after any tool-enabled run.

---

## Time bound + complexity (eating our own dog food — see `orchestration-plan` shared core)

- **Image build:** one-time, target < 20 min (CodeQL download ~hundreds of MB is the long pole; cache
  the layer). Not on the per-run critical path.
- **CodeQL `database create` (Python, build-mode=none):** ~O(LOC) scan; on the 34k-LOC codegen
  residue expect single-digit minutes — **run it ONCE in preflight**, not per-unit, so no Slice agent
  approaches the ~180 s watchdog. Bound: DB build < 5 min on the largest surface; per-unit query < 10 s.
- **PICT:** covering-array generation is fast (sub-second for ≤ ~12 params); n-wise growth is
  polynomial in params at fixed strength — cap params per branch (already `--max-params 12`).
- **Atheris:** time-boxed fuzzing — set an explicit `-max_total_time` (e.g. 30 s/helper) so a Validate
  unit stays under the watchdog; fuzzing is unbounded by nature, so the time box IS the bound.

---

## Recommended rollout order (minimum-credible first)

1. **PICT** — smallest, OSS, immediate `covering_array/` upgrade; proves the tools-image pattern.
2. **CodeQL (Python)** — biggest analysis upgrade (interprocedural slicing); highest value.
3. **Atheris** — fuzz the pure helpers; accept that the 3.12 build may keep it optional.
4. **ACTS** — **skip** (PICT satisfies the deliverable; NIST license isn't hermetic).
5. **Daikon** — **defer/drop** for Python; revisit only if a Java/C target appears.

"Minimum credible" deliverable = items 1–2 (PICT + CodeQL): that alone flips the two most valuable
`optional_tools` to available and upgrades Slice + Combinatorial with real cross-checked output.

---

## Preconditions / open questions for the human (none block scoping)

- **CodeQL license:** free for automated analysis of open-source code; confirm it's acceptable for
  this internal repo's use before shipping the image widely (the *plan* and local use are fine).
- Whether to fold the tools image into `env/` build docs alongside the existing two-layer Dockerfile,
  or keep it as a clearly-separate `Dockerfile.tools`.
