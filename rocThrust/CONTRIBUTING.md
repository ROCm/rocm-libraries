<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to rocThrust">
  <meta name="keywords" content="ROCm, contributing, rocThrust">
</head>

# Contributing to rocThrust #

We welcome contributions to rocThrust.  Please follow these details to help ensure your contributions will be successfully accepted.

## Issue Discussion ##

Please use the GitHub Issues tab to notify us of issues.

* Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration. This helps reduce the time required to
    reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

## Acceptance Criteria ##

rocThrust is a version of the Thrust parallel algorithms library that has been ported to [HIP](https://github.com/ROCm/HIP) and [ROCm](https://www.github.com/ROCm/ROCm).
This allows the library to be used on AMD GPU devices.

Code in rocThrust should perform only the work that's necessary to support the Thrust API - any algorithmic work should be passed off to the backend system.
On AMD platforms, this backend is [rocPRIM](https://github.com/ROCm/rocPRIM).

In order to prevent performance regressions, when a pull request is created, a number of automated checks are run. These checks:
* test the change on various OS platforms (Ubuntu, RHEL, etc.)
* run on different GPU architectures (MI-series, Radeon series cards, etc.)
* run benchmarks to check for performance degredation

In order for change to be accepted:
* it must pass all of the automated checks
* it must undergo a code review

The GitHub "Issues" tab may also used to discuss ideas surrounding particular features or changes before raising pull requests.

## Code Structure ##

Thrust library code is located in the /thrust/ directory. The majority of the code required for porting the library to hip is located
in /thrust/system/hip/. HIP tests are located in the /test/ directory, while the original Thrust cuda tests can be found in /testing/.

## Coding Style ##

C and C++ code should be formatted using `clang-format`. Use the clang-format version for Clang 9, which is available in the `/opt/rocm` directory. Please do not use your system's built-in `clang-format`, as this is an older version that will have different results.

To format a file, use:

```
/opt/rocm/hcc/bin/clang-format -style=file -i <path-to-source-file>
```

To format all files, run the following script in rocThrust directory:

```
#!/bin/bash
git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/hcc/bin/clang-format  -style=file -i
```

Also, githooks can be installed to format the code per-commit:

```
./.githooks/install
```

## Pull Request Guidelines ##

Our code contribution guidelines closely follows the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/).

When you create a pull request, you should target the default branch. Our current default branch is the **develop** branch, which serves as our integration branch.
Releases are cut to release/rocm-rel-x.y, where x and y refer to the release major and minor numbers.

### Deliverables ###

New changes should include test coverage. HIP tests are located in the /test/ directory, while the original Thrust cuda tests can be found in /testing/.

### Process ###

After you create a PR, you can take a look at a diff of the changes you made using the PR's "Files" tab.

PRs must pass through the checks and the code review described in the [Acceptance Criteria](#acceptance-criteria) section before they can be merged.

Checks may take some time to complete. You can view their progress in the table near the bottom of the pull request page. You may also be able to use the links in the table
to view logs associated with a check if it fails.

During code reviews, another developer will take a look through your proposed change. If any modifications are requested (or further discussion about anything is
needed), they may leave a comment. You can follow up and respond to the comment, and/or create comments of your own if you have questions or ideas.
When a modification request has been completed, the conversation thread about it will be marked as resolved.

To update the code in your PR (eg. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.