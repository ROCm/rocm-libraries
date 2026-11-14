---
name: hipdnn-aiter-asm-sdpa-fwd-sync
description: "Update the forward-pass ASM kernels of hip-kernel-provider's ASM SDPA engine from an AITER release or commit. Asks which GPU archs to update, plans a tiered diff of AITER's prebuilt ASM forward objects and CSV tables, applies it with generated provenance, then walks the applicability, build and GPU validation work the diff implies. Ports ASM kernels only, never CK. Use when asked to update, refresh, bump or check the AITER ASM SDPA forward kernels, to add a GPU arch AITER ships forward ASM kernels for, or to verify the vendored forward kernels and their provenance."
argument-hint: "[check|update|verify] [<aiter-ref>] [--arch <archs>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN AITER ASM SDPA Forward Sync

This skill keeps the forward kernels of the ASM SDPA engine in step with AITER. The engine lives in `dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/`. Its kernels are AITER's prebuilt ASM code objects (`.co`) and the CSV tables that describe them, vendored under `asm/asm_kernels/<arch>/<module>/`. The build turns each CSV into a kernel registry (`asm/asm_kernels/codegen.py`) and packs the objects into one `.kpack` archive per arch (`asm/asm_kernels/pack.py`).

Run it on each new AITER release, or when a GPU arch needs forward kernels.

**Scope: forward only.** The backward kernels (`*/fmha_v3_bwd/`), `SdpaBwd*` code and backward provenance are out of scope. The helper script refuses to write anywhere but a tracked forward module directory and the pin file.

## Rules

1. **ASM kernels only, never CK.** Port only the prebuilt objects and CSV tables under AITER's `hsa/<arch>/<module>/`. Never port a CK kernel, CK host code, FlyDSL or Triton code. When AITER's `fmha_fwd_v3()` returns -1 for a case, or its Python gate `can_impl_fmha_v3_fwd` rejects it, AITER hands it to CK; the engine must decline that case, not look for another kernel. The script enforces the parts it can see: it fetches only allowlisted `hsa/` modules, rejects any object with a CK symbol, and never copies upstream source.
2. **Ask which GPU archs to update.** Before planning, run `archs`, show the user the archs AITER ships tracked forward ASM modules for, and ask which ones to update. Never pick archs for the user. If the request already names archs, check each against that list and confirm.
3. **Nothing changes the checkout until the user approves the plan report.** `plan` only writes a report. Run `apply` only after the user has read the report and approved its scope.
4. **One approved code edit at a time.** Tier 2 and tier 3 findings need engine changes. Propose each change, with the finding it answers, and wait for approval.
5. **Provenance is generated.** `aiter_fwd_pins.json` and each module's `SOURCE.md` are written by `apply` and `mark-ported`. Never edit them by hand; change exclusions with `plan --exclude`, `--drop-exclude` or `--include`.
6. **Verified means run on that GPU.** A kernel counts as validated only after its GPU tests pass on that arch. Say which archs were not run.
7. No Jira IDs in code, comments, CSVs or provenance files.

## Inputs

- **Mode**: `check` (is there anything to pull?), `update` (pull a release), or `verify` (is the tree consistent?). Infer it from the request.
- **AITER ref** (`<aiter-ref>`): a release tag such as `v0.1.27`, or a commit. Default: the newest release, from `latest`.
- **GPU archs** (`<archs>`): comma-separated, such as `gfx942,gfx950,gfx1250`. Always asked for; see rule 2.
- **Baseline** (`<baseline-commit>`): needed only on the first update of a vendored module that has no pin yet. It is the AITER commit the vendored files came from, and it also seeds the host-code commit (see below). Take it from the module's `SOURCE.md` or the provenance table in `asm/asm_kernels/README.md`, and confirm it with the user. Modules with a pin use their pinned commits and ignore `--baseline`. The gfx942 and gfx950 `fmha_v3_fwd` modules are pinned.

## Two pinned commits per module

Each module's pin records two AITER commits:

- `commit`: where the vendored objects and CSVs came from. `apply` sets it.
- `abi_commit`: the AITER code the engine mirrors by hand: the kernel argument structs, their packing, the launch grid and block size, the dispatch rules, the MI308 detection, and `hsa/codegen.py`, from which the vendored `asm/asm_kernels/codegen.py` is derived. The dispatch rules live in the C++ host code (`csrc/cpp_itfs/mha_fwd.cu`) and in AITER's Python dispatch gate, `can_impl_fmha_v3_fwd` in `aiter/ops/mha.py` (a batch and a varlen copy, each with the helpers it calls, such as `is_fmha_v3_fp8`, and the caller's code it reads, such as `swa`). Plans diff that upstream code from this commit, so a change stays in every later report until it is ported. `apply` advances it only when none of that code changed or went missing. After porting a change, run `mark-ported --module <module>`.

The engine holds one copy of that host code and launches every arch's objects with it, so a ported change reaches every arch that vendors the module. A plan that leaves such an arch out reports `host_code_shared_with_unplanned_archs`. A module with no `abi_commit` of its own, such as a newly added arch, is diffed from the `abi_commit` the other archs of the module record (the oldest, if they differ). `mark-ported` records the host code for every arch of the module together, and refuses while their objects come from different commits.

While `abi_commit` lags `commit`, plans report `abi_unported` at tier 3.

## Helper script

Default to the copy bundled with the skill you were invoked from: `<skill-directory>/scripts/aiter_fwd_sync.py`, where `<skill-directory>` is this skill's directory. Use the `projects/hipdnn/tools/ai/skills/hipdnn-aiter-asm-sdpa-fwd-sync/scripts` copy in a checkout only when developing this skill.

Run it from inside the rocm-libraries checkout to update, or pass `--repo-root`. It needs `git` and network access to github.com. It keeps a blobless partial clone of AITER in `~/.cache/hipdnn-aiter-fwd-sync` (change it with `--cache`) and reads files through sparse checkouts.

| Command | What it does | Writes |
|---|---|---|
| `latest` | Newest AITER release tag and its commit | Nothing |
| `archs --ref <aiter-ref>` | GPU archs with tracked forward ASM modules at that ref, what the tree holds, and forward modules that are not ported | Nothing |
| `plan --ref <aiter-ref> --arch <archs> --report <report-dir>/plan.json` | Compares the tree with the ref and classifies every change into a tier | The report (`.json` and `.md`) only |
| `apply --report <report-dir>/plan.json` | Applies an approved report | The planned module directories and `aiter_fwd_pins.json` |
| `mark-ported --module <module>` | Records that the engine mirrors the host code of the module's pinned commit, for every arch that vendors the module; refuses while those archs hold objects from different commits. Repeat `--module` for several modules | `aiter_fwd_pins.json` and the `SOURCE.md` of each arch of the module |
| `verify [--arch <archs>]` | Checks pins, manifests, CSV rows against objects, kernel symbols, CK symbols | Nothing |

`<report-dir>` is any directory outside the checkout, for example a temporary directory. Exit codes: 0 success, 1 `verify` found problems, 2 the script could not run.

`plan` options:

- `--baseline <baseline-commit>`: see Inputs.
- `--exclude <module> <match> <reason>`: leave matching upstream rows out. `<module>` is `<arch>/<module-name>`, such as `gfx942/fmha_v3_fwd`. `<match>` is `column=v1,v2`, with `;` between columns, such as `bf16_cvt=1,2`.
- `--include <module> <kernel-name>`: keep a row a rule would leave out. `<kernel-name>` is the row's `knl_name`.
- `--drop-exclude <module> <match>`: remove the pinned rule whose match is exactly `<match>`.

An `--exclude` with the same match as a pinned rule replaces it.

On a module's first update, upstream rows that existed at the baseline but are not in the tree are excluded automatically, with the reason "not vendored at baseline". That keeps the deliberate subset the tree already holds. Rows that are new since the baseline are always surfaced for a decision.

## Workflow: check

1. Run `latest` and report the newest release.
2. Run `archs --ref <aiter-ref>`, and ask which archs to check (rule 2).
3. Run `plan` for those archs and summarize the markdown report: the highest tier per module, and the findings that need a decision. Point to [tiers.md](tiers.md) for what each tier means.
4. Change nothing. Offer the update.

```bash
python3 <skill-directory>/scripts/aiter_fwd_sync.py latest
python3 <skill-directory>/scripts/aiter_fwd_sync.py archs --ref <aiter-ref>
python3 <skill-directory>/scripts/aiter_fwd_sync.py plan --ref <aiter-ref> --arch <archs> --report <report-dir>/plan.json
```

## Workflow: update

1. **Branch.** Work on a branch off `develop` named `users/<github-user>/aiter-fwd-<aiter-ref>`, where `<github-user>` is the user's GitHub login. Make sure the checkout has no uncommitted changes under `asm/asm_kernels/`.
2. **Ref and archs.** Resolve the ref (`latest` if none was given), run `archs`, and ask which archs to update (rule 2).
3. **Plan.** Run `plan`. Add `--baseline <baseline-commit>` when any chosen module has no pin. Show the user the markdown report.
4. **Decide the scope with the user.** Use [tiers.md](tiers.md). For each row added with a new value, decide whether to keep it or exclude it, and for each excluded row record a reason. Re-run `plan` with `--exclude` and `--include` until the report is what the user wants, then get explicit approval.
5. **Apply.** Run `apply --report <report-dir>/plan.json`, then `verify`. Both must succeed.
6. **Engine work.** Work through each finding with its page: tier 0 and 1 in [data-changes.md](data-changes.md), tier 2 in [new-values.md](new-values.md), tier 3 in [structural.md](structural.md). A large tier 3 item, such as a new arch, can go in its own PR; agree that with the user. Once every `reference_changed` diff for a module is ported, run `mark-ported --module <module>` and `verify`.
7. **Provenance outside the pin.** `apply` replaces each forward module's `SOURCE.md` with a generated one. The other forward provenance names no commit; it points at the `<arch>/fmha_v3_fwd` pins in `asm/asm_kernels/aiter_fwd_pins.json`. The forward row of `asm/asm_kernels/README.md` names both pinned fields, `asm/AsmKernelPath.hpp` cites `commit`, and `asm/SdpaFwdKernelArgs.hpp` cites `abi_commit`. Update the README's forward row and layout notes only when the archs or the object directories change. If a padding struct was ported, update the provenance comment of `asm/SgprPadding.hpp`, which the backward kernels share.
8. **Build and test.** Follow [validation.md](validation.md).
9. **PR.** Follow [pr.md](pr.md).

```bash
python3 <skill-directory>/scripts/aiter_fwd_sync.py plan --ref <aiter-ref> --arch <archs> --baseline <baseline-commit> --report <report-dir>/plan.json
python3 <skill-directory>/scripts/aiter_fwd_sync.py apply --report <report-dir>/plan.json
python3 <skill-directory>/scripts/aiter_fwd_sync.py verify
python3 <skill-directory>/scripts/aiter_fwd_sync.py mark-ported --module <module>
```

## Workflow: verify

Run `verify`, optionally with `--arch <archs>`. Report each `FAIL` line. Before the first update there is no pin file, so `verify` reports the vendored modules as unpinned; the first `update` fixes that.

## Completion and handoff

Report:

- The AITER ref and commit, the archs updated and the archs the user chose not to update.
- The highest tier per module, each exclusion and inclusion with its reason, and the forward modules not ported.
- Each engine change made for a tier 2 or tier 3 finding, and each CK fallback case that became a decline.
- `verify` output, the build result, and the unit tests run.
- GPU validation per arch: run and passed, run and failed, or not run. A forward kernel on an arch with no GPU run is unvalidated ([validation.md](validation.md)).
- The branch and the PR draft. Nothing is pushed or posted without approval.
