---
name: hipdnn-aiter-asm-sdpa-fwd-sync
description: "Update the forward and backward ASM kernels of hip-kernel-provider's ASM SDPA engine from an AITER release or commit. Asks which GPU archs and directions to update, plans a tiered diff of AITER's prebuilt ASM objects and CSV tables (fmha_v3_fwd, fmha_v3_bwd and the other tracked modules), applies it with generated provenance, then walks the applicability, build and GPU validation work the diff implies. Ports ASM kernels only, never CK. Use when asked to update, refresh, bump or check the AITER ASM SDPA forward or backward kernels, to add a GPU arch AITER ships ASM SDPA kernels for, or to verify the vendored kernels and their provenance."
argument-hint: "[check|update|verify] [<aiter-ref>] [--arch <archs>] [--direction fwd|bwd|all]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN AITER ASM SDPA Sync

This skill keeps the forward and backward kernels of the ASM SDPA engine in step with AITER. The engine lives in `dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/`. Its kernels are AITER's prebuilt ASM code objects (`.co`) and the CSV tables that describe them, vendored under `asm/asm_kernels/<arch>/<module>/`. The build turns each CSV into a kernel registry (`asm/asm_kernels/codegen.py`) and packs the objects into one `.kpack` archive per arch (`asm/asm_kernels/pack.py`).

Run it on each new AITER release, or when a GPU arch needs kernels.

**Scope: forward and backward.** The script tracks these modules:

| Direction | Modules | Engine code |
|---|---|---|
| Forward (`fwd`) | `fmha_v3_fwd`, `fmha_fwd_bf16`, `fmha_fwd_bf16_varlen` | `SdpaFwd*` |
| Backward (`bwd`) | `fmha_v3_bwd` | `SdpaBwd*` |

`--direction fwd`, `bwd` or `all` (the default) limits `archs`, `plan` and `verify` to one direction. The helper script refuses to write anywhere but a tracked module directory and the pin file.

## Rules

1. **ASM kernels only, never CK.** Port only the prebuilt objects and CSV tables under AITER's `hsa/<arch>/<module>/`. Never port a CK kernel, CK host code, FlyDSL or Triton code. When AITER's `fmha_fwd_v3()` or `fmha_v3_bwd()` returns -1 for a case, or its Python layer (`aiter/ops/mha.py`, including the `can_impl_fmha_v3_fwd` and `can_impl_fmha_v3_bwd` gates) routes the case to CK, the engine must decline that case, not look for another kernel. The script enforces the parts it can see: it fetches only allowlisted `hsa/` modules, rejects any object with a CK symbol, and never copies upstream source.
2. **Ask which GPU archs to update.** Before planning, run `archs`, show the user the archs AITER ships tracked ASM modules for, and ask which archs and directions to update. Never pick them for the user. If the request already names archs, check each against that list and confirm.
3. **Nothing changes the checkout until the user approves the plan report.** `plan` only writes a report. Run `apply` only after the user has read the report and approved its scope.
4. **One approved code edit at a time.** Tier 2 and tier 3 findings need engine changes. Propose each change, with the finding it answers, and wait for approval.
5. **Provenance is generated.** `aiter_pins.json` and each module's `SOURCE.md` are written by `apply` and `mark-ported`. Never edit them by hand; change exclusions with `plan --exclude`, `--drop-exclude` or `--include`.
6. **Verified means run on that GPU.** A kernel counts as validated only after its GPU tests pass on that arch. Say which archs were not run.
7. No Jira IDs in code, comments, CSVs or provenance files.

## Inputs

- **Mode**: `check` (is there anything to pull?), `update` (pull a release), or `verify` (is the tree consistent?). Infer it from the request.
- **AITER ref** (`<aiter-ref>`): a release tag such as `v0.1.29`, or a commit. Default: the newest release, from `latest`.
- **GPU archs** (`<archs>`): comma-separated, such as `gfx942,gfx950,gfx1250`. Always asked for; see rule 2.
- **Direction** (`<direction>`): `fwd`, `bwd` or `all`. Default `all`.
- **Baselines**: needed only on the first update of a module that has no pin yet. Modules with a pin use the pinned commits and ignore `--baseline`; the gfx942 and gfx950 `fmha_v3_fwd` modules are pinned. See "Baselines for a first update".

## Two pinned commits per module

Each module's pin records two AITER commits:

- `commit`: where the vendored objects and CSVs came from. `apply` sets it.
- `abi_commit`: the AITER code the engine mirrors by hand: the kernel argument structs, their packing, the launch grid and block size, the dispatch rules, the MI308 detection, and `hsa/codegen.py`, from which the vendored `asm/asm_kernels/codegen.py` is derived. The dispatch rules live in the C++ host code (`csrc/cpp_itfs/mha_fwd.cu` and `mha_bwd.cu`) and in AITER's Python gates in `aiter/ops/mha.py`: `can_impl_fmha_v3_fwd` (a batch and a varlen copy) for forward, and `can_impl_fmha_v3_bwd`, `_flash_attn_backward` and `_flash_attn_varlen_backward` for backward. Plans diff that upstream code from this commit, so a change stays in every later report until it is ported. `apply` advances it only when none of that code changed or went missing. After porting a change, run `mark-ported --module <module>`.

While `abi_commit` lags `commit`, plans report `abi_unported` at tier 3.

## Baselines for a first update

`plan --baseline [<scope>=]<baseline-commit>` names the AITER commit an unpinned module's vendored files came from. It also seeds the host-code commit, unless `--abi-baseline [<scope>=]<abi-commit>` names a different one. `<scope>` is `fwd`, `bwd` or `<arch>/<module-name>`; without a scope the commit applies to every planned module. Repeat either option for several scopes; the most specific scope wins.

Take the commits from the provenance table in `asm/asm_kernels/README.md` and the provenance comments of the engine's argument structs, and confirm them with the user:

| Direction | `--baseline` (objects) | `--abi-baseline` (host code) |
|---|---|---|
| Forward, gfx942 and gfx950 | `17d4a33b6f9535e820353ebc6217769efc3766d6` | same |
| Backward, gfx942 and gfx950 | `cdcfa833bdf554ca75594c90dde4316ea9b50199` (v0.1.13) | `17d4a33b6f9535e820353ebc6217769efc3766d6` |

`asm/SdpaFwdKernelArgs.hpp`, `asm/SdpaBwdKernelArgs.hpp` and `asm/SgprPadding.hpp` name `9522048` instead: a commit on an AITER side branch whose argument and padding structs are identical to those at `17d4a33`. The engine's MI308 detection and backward packing follow `17d4a33`, so `17d4a33` is the right host-code baseline for both directions. Because the backward host-code baseline is older than its objects, the first backward plan reports `abi_unported` and diffs every mirrored function from `17d4a33`.

On a module's first update, upstream rows that existed at the baseline but are not in the tree are excluded automatically, with the reason "not vendored at baseline". So is an upstream CSV that existed at the baseline but is not in the tree, such as the backward `fmha_bwd_dq_shuffle.csv`. That keeps the deliberate subset the tree already holds. Rows and CSVs that are new since the baseline are always surfaced for a decision. Keep an excluded row or CSV with `--include`.

## Helper script

Default to the copy bundled with the skill you were invoked from: `<skill-directory>/scripts/aiter_fwd_sync.py`, where `<skill-directory>` is this skill's directory. Use the `projects/hipdnn/tools/ai/skills/hipdnn-aiter-asm-sdpa-fwd-sync/scripts` copy in a checkout only when developing this skill.

Run it from inside the rocm-libraries checkout to update, or pass `--repo-root` (before or after the subcommand). It needs `git` and network access to github.com. It keeps a blobless partial clone of AITER in `~/.cache/hipdnn-aiter-fwd-sync` (change it with `--cache`) and reads files through sparse checkouts.

| Command | What it does | Writes |
|---|---|---|
| `latest` | Newest AITER release tag and its commit | Nothing |
| `archs --ref <aiter-ref> [--direction <direction>]` | GPU archs with tracked ASM modules at that ref, what the tree holds, and modules that are not ported | Nothing |
| `plan --ref <aiter-ref> --arch <archs> [--direction <direction>] --report <report-dir>/plan.json` | Compares the tree with the ref and classifies every change into a tier | The report (`.json` and `.md`) only |
| `apply --report <report-dir>/plan.json` | Applies an approved report, for the modules it holds | The planned module directories, `aiter_pins.json` and every pinned module's `SOURCE.md` |
| `mark-ported --module <module>` | Records that the engine mirrors the host code of the module's pinned commit. Repeat `--module` for several | `aiter_pins.json` and every pinned module's `SOURCE.md` |
| `verify [--arch <archs>] [--direction <direction>]` | Checks pins, manifests, CSV rows against objects, kernel symbols, CK symbols | Nothing |

`<report-dir>` is any directory outside the checkout, for example a temporary directory. Exit codes: 0 success, 1 `verify` found problems, 2 the script could not run.

`plan` options:

- `--baseline` and `--abi-baseline`: see "Baselines for a first update".
- `--exclude <module> <match> <reason>`: leave matching upstream rows out. `<module>` is `<arch>/<module-name>`, such as `gfx942/fmha_v3_fwd`. `<match>` is `column=v1,v2`, with `;` between columns, such as `bf16_cvt=1,2`. The column `csv` matches the CSV file by its path in the module; `csv=<file>.csv` alone leaves the whole file out, header included.
- `--include <module> <name>`: keep a row (`<name>` is its `knl_name`) or a whole CSV (`<name>` is its path, ending in `.csv`) that a rule or the baseline would leave out.
- `--drop-exclude <module> <match>`: remove the pinned rule whose match is exactly `<match>`.

An `--exclude` with the same match as a pinned rule replaces it.

The report lists, per module, how many objects are added, removed, changed and unchanged, and the row count of every CSV before and after.

## Workflow: check

1. Run `latest` and report the newest release.
2. Run `archs --ref <aiter-ref>`, and ask which archs and directions to check (rule 2).
3. Run `plan` for those archs and summarize the markdown report: the highest tier per module, and the findings that need a decision. Point to [tiers.md](tiers.md) for what each tier means.
4. Change nothing. Offer the update.

```bash
python3 <skill-directory>/scripts/aiter_fwd_sync.py latest
python3 <skill-directory>/scripts/aiter_fwd_sync.py archs --ref <aiter-ref>
python3 <skill-directory>/scripts/aiter_fwd_sync.py plan --ref <aiter-ref> --arch <archs> --direction <direction> --report <report-dir>/plan.json
```

## Workflow: update

1. **Branch.** Work on a branch off `develop` named `users/<github-user>/aiter-<direction>-<aiter-ref>`, where `<github-user>` is the user's GitHub login and `<direction>` is `fwd`, `bwd` or `sync` for both. Make sure the checkout has no uncommitted changes under `asm/asm_kernels/`.
2. **Ref, archs and direction.** Resolve the ref (`latest` if none was given), run `archs`, and ask which archs and directions to update (rule 2).
3. **Plan.** Run `plan`. Add `--baseline` (and `--abi-baseline` where the host code is older than the objects) when any chosen module has no pin. Show the user the markdown report.
4. **Decide the scope with the user.** Use [tiers.md](tiers.md). For each row or CSV added with a new value, decide whether to keep it or exclude it, and for each exclusion record a reason. Re-run `plan` with `--exclude` and `--include` until the report is what the user wants, then get explicit approval.
5. **Apply.** Run `apply --report <report-dir>/plan.json`, then `verify`. Both must succeed.
6. **Engine work.** Work through each finding with its page: tier 0 and 1 in [data-changes.md](data-changes.md), tier 2 in [new-values.md](new-values.md), tier 3 in [structural.md](structural.md). A large tier 3 item, such as a new arch or a new backward pipeline stage, can go in its own PR; agree that with the user. Once every `reference_changed` diff for a module is ported, run `mark-ported --module <module>` and `verify`.
7. **Provenance outside the pin.** `apply` replaces each updated module's `SOURCE.md` with a generated one. The other forward provenance names no commit; it points at the `<arch>/fmha_v3_fwd` pins in `asm/asm_kernels/aiter_pins.json`. The forward row of `asm/asm_kernels/README.md` names both pinned fields, `asm/AsmKernelPath.hpp` cites `commit`, and `asm/SdpaFwdKernelArgs.hpp` cites `abi_commit`. Update the README's forward row and layout notes only when the archs or the object directories change. The backward provenance names its commit: after a backward update, update the backward row of the README and the backward line in the header comment of `asm/AsmKernelPath.hpp` to the new tag and commit. After `mark-ported`, update the commit in the provenance comments of `asm/SdpaBwdKernelArgs.hpp` and the packing comments in `plans/SdpaBwdPlan.cpp`. If a padding struct was ported, update the provenance comment of `asm/SgprPadding.hpp`, which both directions share.
8. **Build and test.** Follow [validation.md](validation.md).
9. **PR.** Follow [pr.md](pr.md).

```bash
python3 <skill-directory>/scripts/aiter_fwd_sync.py plan --ref <aiter-ref> --arch <archs> --direction <direction> --baseline <scope>=<baseline-commit> --abi-baseline <scope>=<abi-commit> --report <report-dir>/plan.json
python3 <skill-directory>/scripts/aiter_fwd_sync.py apply --report <report-dir>/plan.json
python3 <skill-directory>/scripts/aiter_fwd_sync.py verify
python3 <skill-directory>/scripts/aiter_fwd_sync.py mark-ported --module <module>
```

For example, the first backward update of gfx942 and gfx950, keeping gfx950's dq_shuffle table:

```bash
python3 <skill-directory>/scripts/aiter_fwd_sync.py plan --ref v0.1.29 --arch gfx942,gfx950 --direction bwd \
    --baseline bwd=cdcfa833bdf554ca75594c90dde4316ea9b50199 \
    --abi-baseline bwd=17d4a33b6f9535e820353ebc6217769efc3766d6 \
    --include gfx950/fmha_v3_bwd fmha_bwd_dq_shuffle.csv \
    --report <report-dir>/plan.json
```

## Workflow: verify

Run `verify`, optionally with `--arch <archs>` and `--direction <direction>`. Report each `FAIL` line.

- Before a module's first update it has no pin, so `verify` reports it as unpinned; the first `update` fixes that.
- Every object must be named by a CSV row. A CSV with only a header, such as gfx942's `fmha_bwd_dq_shuffle.csv`, is valid and names none.

## Completion and handoff

Report:

- The AITER ref and commit, the archs and directions updated, and the ones the user chose not to update.
- The highest tier per module, each exclusion and inclusion with its reason, and the modules not ported.
- Each engine change made for a tier 2 or tier 3 finding, and each CK fallback case that became a decline.
- `verify` output, the build result, and the unit tests run.
- GPU validation per arch and direction: run and passed, run and failed, or not run. A kernel on an arch with no GPU run is unvalidated, and so is a backward row no test case reaches ([validation.md](validation.md)).
- The branch and the PR draft. Nothing is pushed or posted without approval.
