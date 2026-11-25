# MIOpen changelog fragments

For user-facing changes, add one uniquely named fragment per entry to this directory instead of editing `../CHANGELOG.md`. For example, `12345-reduce.resolved.md`:

```markdown
* [Reduce] Fixed incorrect results for noncontiguous output tensors.
```

Use a PR or issue number plus a short slug, or a unique lowercase slug if no number exists yet. In each section, the assembler sorts numbered fragments by number, then unnumbered fragments by filename. Write one `* ` bullet as it should appear in the changelog; indent continuation lines by at least two spaces. Internal-only changes do not need an entry. Pre-commit CI validates fragment format on PRs into `develop`; for a release-branch backport, run `pre-commit run changelog-fragments --files <fragment-path>` locally. Reviewers decide whether a note is needed.

| Filename suffix | Changelog heading |
|---|---|
| `.added.md` | Added |
| `.changed.md` | Changed |
| `.deprecated.md` | Deprecated |
| `.removed.md` | Removed |
| `.resolved.md` | Resolved Issues |
| `.optimized.md` | Optimized |
| `.known.md` | Known Issues |

## Release checklist

Before cutting or tagging a release, check the exact source ref in a clean checkout:

```sh
python3 .github/scripts/changelog_fragments.py --project projects/miopen --check-empty
```

If `develop` has pending fragments, dispatch the workflow and merge its PR. For a separate release branch with pending backport fragments, use `--fold` on a branch based on that release ref and open a regular PR there; the bot only targets `develop`. Recheck the final source ref before shipping. When finalizing a release, remove the `(Unreleased) ` prefix from its heading and add a newly marked heading on the branch where development continues. The assembler refuses to write unless the newest release heading starts with `## (Unreleased) `.

## Opt in another project

Add the directory name under `projects/` to the [project registry](../../../.github/scripts/changelog-projects.json), create a `changelog.d/` directory with a contributor guide, and mark the newest `CHANGELOG.md` section with the `## (Unreleased) ` prefix. Update the project's contributor guidance and give its fragments the same CODEOWNERS as its changelog. The shared **Changelog fragments** workflow runs weekly and on demand, using a separate bot PR for each registered project. No project-specific workflow or App setup is needed.
