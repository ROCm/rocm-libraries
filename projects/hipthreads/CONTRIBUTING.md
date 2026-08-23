<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to hipThreads">
  <meta name="keywords" content="ROCm, contributing, hipThreads">
</head>

# Contributing to hipThreads #

We welcome contributions to hipThreads.
Please follow these details to help ensure your contributions will be successfully accepted.

## Issue Discussion ##

Please use the GitHub Issues tab to notify us of issues.

* Use your best judgement for issue creation.
  If your issue is already listed, upvote the issue and comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue.
  If we evaluate your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible, including the GPU architecture, ROCm version, and any build or run output, so we can collect information about your configuration.
    This helps reduce the time required to reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change meets the acceptance criteria, or to discuss an idea or feature request pertaining to the library.

## Acceptance Criteria ##

hipThreads is a C++-style concurrency library for AMD GPUs.
It implements `std::thread`-like primitives (`hip::wthread`, `hip::mutex`, `hip::lock_guard`, `hip::condition_variable`, ...) that run inside GPU kernels, so existing `std::thread` CPU code can be ported to the GPU with minimal changes.
It is built on [HIP](https://github.com/ROCm/HIP) and [libhipcxx](https://github.com/ROCm/libhipcxx).

Code in hipThreads should keep the public interface compatible with the C++ Standard Library threading facilities wherever practical, so that familiar `std::`-style concurrency code ports to the GPU with minimal changes.
New primitives should mirror their standard-library counterparts in naming and semantics.

Contributions should be in scope for hipThreads — that is, threading and concurrency primitives and their direct supporting code.
Functionality that is not specific to threading/concurrency (general-purpose device utilities, containers, `std::`-style types, or parallel algorithms) is usually better suited to [libhipcxx](https://github.com/ROCm/libhipcxx) or [rocThrust](https://github.com/ROCm/rocThrust).
If you're unsure whether a change fits, open an issue to discuss it before raising a pull request.

In order for a change to be accepted, it must build cleanly, pass the test suite, and undergo a code review.

The GitHub "Issues" tab may also be used to discuss ideas surrounding particular features or changes before raising pull requests.

## Code Structure ##

The deliverable is a static library `libhipthreads.a` plus public headers under `inc/hip/`.
Only `src/hip/thread.cxx` is compiled into the library, everything else is header-only.

* Public umbrella headers (extensionless, libc++ style) live directly in `inc/hip/`: `thread`, `mutex`, `condition_variable`, `pseudo_mutex`, `pseudo_condition_variable`, `thread_config`.
* Implementation headers live under `inc/hip/__thread/`, `inc/hip/__mutex/`, and `inc/hip/__condition_variable/`, with low-level helpers in `inc/hip/__clib/` and `inc/hip/__support/`.
* The persistent scheduler kernel and host-side plumbing live in `src/hip/thread.cxx`.
* Tests live in `test/` (a libc++-style **lit** suite plus standalone unit executables).
* Examples are standalone CMake projects under `examples/`, each laid out as a series of `stepN-*` directories showing an incremental CPU to GPU port.
  The examples are also used as integration tests.

See `README.md` for a deeper description of the architecture and build/test workflow, and the full documentation at [https://rocm.docs.amd.com/projects/hipThreads/en/latest/](https://rocm.docs.amd.com/projects/hipThreads/en/latest/).

## Coding Style ##

C and C++ code should be formatted using `clang-format`.
The repository ships a `.clang-format` (LLVM base, 4-space indent, 120 column limit, access modifiers indented).
Please format the code you change before opening a pull request.

Prefer formatting **only the lines you changed**, rather than reformatting whole files (which creates noisy diffs and can reformat unrelated code).
`git clang-format` makes this easy — it formats just the lines in your staged changes:

```bash
# Stage your changes first, then format only the changed lines:
git add <files>
git clang-format

# Or format the changed lines relative to a base branch (e.g. before pushing):
git clang-format develop
```

The reformatting is left unstaged so you can review it before committing.
To format a specific file in full (for example, a brand-new file you are adding), use:

```bash
clang-format -style=file -i <path-to-source-file>
```

Documentation files (Markdown and reStructuredText) are not subject to a column limit.
Instead, write one sentence per line, following the [LLVM coding standards](https://llvm.org/docs/CodingStandards.html#source-code-width).
This keeps diffs small and readable, since editing a sentence changes only its line rather than rewrapping a whole paragraph.

## Branching and Release Model ##

hipThreads follows the branching and release model used across the ROCm libraries.

* **`develop` is the integration branch.**
  All pull requests target `develop`.
  It always reflects the latest accepted work and is the branch that is integrated against.
* **Release branches are cut as `release/rocm-X.Y`** at the ROCm code-freeze date, where `X` and `Y` are the ROCm release major and minor numbers.
  Once a release branch is cut it receives only bug fixes and release-critical changes (cherry-picked from `develop`), not new features.
* **Releases are tagged `rocm-X.Y.Z`**, following the tag convention used across `rocm-libraries`, where `Z` is the patch number.

## Pull Request Guidelines ##

When opening a pull request, please follow these guidelines:

* **Target the `develop` branch.**
  Open your pull request against `develop`, our integration branch (see [Branching and Release Model](#branching-and-release-model)).
* **Keep PRs small and focused.**
  Smaller PRs are easier and faster to review, and tend to surface fewer bugs.
  It is perfectly fine, and encouraged, to break a larger feature into a series of smaller PRs, even if some of them do not add user-visible functionality on their own (for example, a PR that only restructures or prepares the code ahead of the change that builds on it).
* **One concern per PR.**
  Keep each PR scoped to a single feature or fix.
  If you spot an unrelated issue while working, open a separate PR (or issue) for it rather than bundling it in.
* **Write a descriptive title and body.**
  Explain *what* the change does and *why*, link any related issues, and call out anything reviewers should pay special attention to or any decisions you'd like feedback on.
* **Review your own PR first.**
  Read through the diff as if you were the reviewer before requesting review — this catches typos, leftover debugging code, and unintended changes.
* **Say how you verified it.**
  Note the tests you added or ran (and on which GPU architecture), so reviewers know the change has been exercised.
* **Keep it green.**
  Make sure the code builds cleanly and the test suite passes before requesting review, and respond to review feedback by pushing follow-up commits to the same branch.

### Branch Naming ###

Name your topic branches `users/<username>/<short-description>`, for example `users/jdoe/fix-detach-race`.
This namespaces work by author and keeps the branch list readable.

### Deliverables ###

#### Tests and benchmarks ####

New changes should include **test coverage**.
Tests live in the `test/` directory; each `test/*.cxx` file is auto-discovered and compiled as a HIP executable, and the suite is also runnable through `lit`.
Tests should cover the functionality added to the public API.
If you modify existing behavior, update the affected tests to match.

#### Changes record ####

All noticeable changes are recorded in the `CHANGELOG.md` file.
For every release we annotate the additions, fixes, changes, deprecations, and/or optimizations introduced in that release.
When opening a PR, add the meaningful changes it introduces to the appropriate sections under the latest unreleased entry.

### Process ###

After you create a PR, you can take a look at a diff of the changes you made using the PR's "Files" tab.

PRs must pass through the code review described in the [Acceptance Criteria](#acceptance-criteria) section before they can be merged.

During code reviews, another developer will take a look through your proposed change.
If any modifications are requested (or further discussion about anything is needed), they may leave a comment.
You can follow up and respond to the comment, and/or create comments of your own if you have questions or ideas.
When a modification request has been completed, the conversation thread about it will be marked as resolved.

To update the code in your PR (e.g. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.
