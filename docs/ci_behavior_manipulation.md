# CI Behavior Manipulation

TheRock CI is controlled by [`configure_ci.py`](../.github/scripts/therock_configure_ci.py), where it controls push, pull request, workflow dispatch and schedule CI behavior.

## Default behavior for push and pull request

TheRock CI will determine [Linux targets](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/workflows/therock-ci.yml#L94) and [Windows targets](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/workflows/therock-ci.yml#L107) and [file changes](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/scripts/therock_matrix.py#L7-L33), then run build and tests accordingly on file changes.

Example: a change made to `projects/rocfft` will only run `FFT` builds and tests.

For [CI changes](https://github.com/ROCm/rocm-libraries/blob/f4ddefcbc17bf36889295f4a97ee40ebcf1b7cdc/.github/scripts/therock_configure_ci.py#L114-L121), we run all build and smoke tests.

## Pull request behavior

Here are additional labels that manipulate the CI behavior. The labels we provide are:

- `skip-therockci`: The CI will skip all builds and tests

## Label-gated cmake flags for multi-arch CI

A label can also make the multi-arch presubmits build TheRock with a non-default cmake flag, for one pull request only. This is how you get CI coverage — including GPU tests — for a code path that is off by default.

Honored by `therock-multi-arch-ci.yml` and `therock-multi-arch-ci-asan.yml`. **Not** honored by the nightlies, by `workflow_dispatch`, or by push builds: labels are read only for `pull_request` events, so a dispatch always builds with the default configuration.

The map lives in [`therock_multiarch_label_flags.py`](../.github/scripts/therock_multiarch_label_flags.py) and is flat — a label maps straight to a list of cmake options, with no per-project key, because multi-arch always builds all of TheRock for this repository:

```python
LABEL_GATED_THEROCK_FLAGS: dict[str, list[str]] = {
    "ci:miopen-hipdnn-wrapper": [
        "-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON",
    ],
}
```

Currently no labels are listed. Add each new one as a bullet here when you add it to the map.

### Only `-DTHEROCK_FLAG_*` options are allowed

The script rejects anything else, and the prefix is not a formality. `therock_declare_flag` *adds* it: declaring `NAME MIOPEN_ENABLE_HIPDNN_WRAPPER` creates the cache variable `THEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER`. That prefixed name is the superbuild knob you set; the flag machinery is what forwards the unprefixed name down into the subprojects listed in `SUB_PROJECTS`. Setting the unprefixed name at the top level does nothing at all — subproject arguments are an explicit allowlist — and you get a green build with the flag off.

`-DTHEROCK_ENABLE_*` is rejected for a different reason: TheRock generates its own `THEROCK_ENABLE_*` options *after* these are spliced in, and cmake takes the last `-D`, so such an option would be silently overridden.

Adding a new flag therefore takes two pull requests, in order:

1. In `ROCm/TheRock`, declare the flag in `FLAGS.cmake` with `therock_declare_flag(... SUB_PROJECTS <project>)`. If the flag is Linux-only, make it a no-op on Windows there — the release workflow's Windows leg gets the same options and there is no per-platform key.
2. In this repository, add the map entry, a bullet under this section, and the label itself to the repository's label list.

> [!WARNING]
> The TheRock commit used by a run is pinned at merge-base time. If the `FLAGS.cmake` entry lands upstream *after* your branch's base, the pinned TheRock will not know the flag, cmake will ignore it, and **the build goes green with the flag off**. Merge or rebase onto a newer base and confirm the `Resolve TheRock ref` job picked a commit that contains the declaration.

### A label has an effect if and only if it is a key in the map

There is no naming convention and none is enforced. `ci:<project>-<feature>` is suggested for new labels, matching the existing style, but nothing checks it. Avoid bare `ci:` names that collide with the already-crowded namespace (`ci:asan`, `ci:ccache`, `ci:smoke`, `ci:debug`, `ci:gpu:gfx942`, `ci:testonly`).

Both multi-arch workflows read the same map, but each only reacts to labels on the events it subscribes to. Adding a mapped label starts a fresh run that builds with the flag on; removing it starts one that builds with the flag off. Adding or removing any *unmapped* label runs the short `Resolve label-gated cmake flags` job, which reports `label_relevant=false`, and everything expensive is skipped — nothing about the build configuration changed, so there is nothing to rebuild.

A label already applied keeps taking effect on later pushes. On a `synchronize`, `opened` or `reopened` event the gate does not apply and the full label set of the pull request is still consulted, so a sticky label still injects its flag.

### Caveats

**The gated build replaces the normal one.** A labeled pull request has no green flag-off signal, because both configurations would collide in the same artifact store. If you need the baseline too, put it in a second pull request — separate runs are namespaced by run ID and never collide.

**Stage reuse is forced off.** Reuse is keyed on changed file paths only; cmake flags are not part of the key, so a flag-on run could otherwise inherit stages that were built flag-off. The release workflow switches `stage_reuse_mode` to `dry-run` whenever a flag is active, and passing `prebuilt_stages` or `baseline_run_id` together with a gated label is a hard error rather than a silently wrong build.

**Re-running an old run replays its original payload.** The labels a re-run sees are the ones that were on the pull request when that run was *first* triggered, so applying a label and then re-running a stale failed run gives you a flag-off build that looks fine. Push a commit or re-apply the label instead. The job's step summary always names the exact label set and flags it acted on, which is the way to confirm what a given run actually built.

## Workflow dispatch behavior

For `workflow_dispatch`, you are able to trigger CI in [GitHub's therock-ci.yml workflow page](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml). To trigger a workflow dispatch, click "Run workflow" and fill in the fields accordingly.
