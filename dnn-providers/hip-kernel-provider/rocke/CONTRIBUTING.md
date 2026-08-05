# Contributing to rocke

Conventions and automation for the rocke engine
([platform/](platform/) + [library/](library/)). Pairs with the process map in
[library/ENGINEERING_PROCESS.md](library/ENGINEERING_PROCESS.md) and the hard
rules in [platform/AGENTS.md](platform/AGENTS.md).

---

## Table of Contents

- [Commit conventions](#commit-conventions)
- [Branch naming](#branch-naming)
- [Definition of Done gates](#definition-of-done-gates)

---

## Commit conventions

rocke uses [Conventional Commits](https://www.conventionalcommits.org/) with a
**required scope**, matching existing history (`feat(rocke):`, `test(rocke):`,
`fix(rocke):`, `perf(attention):`, `feat(rocke/attention):`, `docs(rocke):`).

```
<type>(<scope>): <subject>
```

- **type** — one of `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`,
  `ci`, `build`, `style`.
- **scope** — required. Use `rocke`, `hip-kernel-provider`, or a sub-area like
  `rocke/attention`.
- **subject** — imperative, lower-case, no trailing period.

Examples:

```
feat(rocke): add gfx950 mask-phase-split 2D lever
test(rocke/attention): add on-GPU numeric parity for d128 ring
fix(rocke): correct i64 paged-KV offset for >2GiB caches
```

Merge/revert/fixup commits are exempt.

## Branch naming

`users/<github-username>/<kebab-branch-name>` (e.g. `users/anarao/gfx950-mask-split`).

## Definition of Done gates

Run [`tools/run_checks.py`](tools/run_checks.py) on demand before opening a PR.
See [library/ENGINEERING_PROCESS.md](library/ENGINEERING_PROCESS.md) and
[DEFINITION_OF_DONE.md](DEFINITION_OF_DONE.md) for the full checklist. The stages:

1. Relative-path guard.
2. **Byte-identity gate at both llvm20 and llvm22.**
3. Platform instance parity (`.py`/`.c` emitter pairs).
4. pytest (`platform/tests library/tests`).
5. On-GPU numeric parity — needs a HIP device; the lane that catches correctness
   regressions a spec-only test would miss. Only exercises the arch of the GPU on
   your machine, so run it on each arch you can.
