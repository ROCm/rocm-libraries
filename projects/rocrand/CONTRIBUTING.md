<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to rocRAND">
  <meta name="keywords" content="ROCm, contributing, rocRAND">
</head>

# Contributing to rocRAND #

We welcome contributions to rocRAND. Please follow these details to help ensure your contributions will be successfully accepted.

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

The rocRAND project provides functions that generate pseudorandom and quasirandom numbers. The rocRAND library is implemented in the [HIP](https://github.com/ROCm/HIP) programming language and optimized for AMD's latest discrete GPUs. It is designed to run on top of AMD's [ROCm](https://rocm.docs.amd.com/) runtime, but it also works on CUDA-enabled GPUs.

Performance and statistical accuracy are important in rocRAND. 
To protect against regressions, when a pull request is created, a number of automated checks are run. These checks:
- test the change on various OS platforms (Ubuntu, RHEL, etc.)
- run on different GPU architectures (MI-series, Radeon series cards, etc.)
- run benchmarks to check for performance degredation

In order for change to be accepted:
- it must pass all of the automated checks
- it must undergo a code review

The GitHub "Issues" tab may also be used to discuss ideas surrounding particular features or changes, before raising pull requests.

## Code Structure ##

Library code is located inside the library/ directory. It contains two subdirectories:
* include/ contains header files that can be included in projects using rocRAND.
There is a separate header for each RNG engine and statistical distribution.
* src/ contains the RNG implementations

Other directories in rocRAND include:
* benchmark/ code for benchmarking engines and distributions is located here
* docs/ contains files used to generate the rocRAND API documentation
* python/ contains examples that demonstrate how rocRAND can be loaded and using in Python
* scripts/ contains utility scripts for things like formating code and tuning config
* test/ contains tests for the rocRAND API
* tools/ contains code that helps generate the precomputed constants used by some generators

## Coding Style ##

C and C++ code should be formatted using `clang-format`. Use the clang-format version for Clang 9, which is available in the `/opt/rocm` directory. Please do not use your system's built-in `clang-format`, as this is an older version that will have different results.

To format a file, use:

```
/opt/rocm/hcc/bin/clang-format -style=file -i <path-to-source-file>
```

To format all files, run the following script in rocRAND directory:

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

Additional things to consider when creating a pull request:
* ensure code builds successfully
* check to make sure that your change does not break existing test cases
* ensure that new code has test coverage
  * new unit tests should integrate within the existing [googletest framework](https://github.com/google/googletest/blob/master/googletest/docs/Primer.md)
  * tests must have good code coverage
  * code must also have benchmark tests, and performance must be acceptable to maintainers

### Process ###

After you create a PR, you can take a look at a diff of the changes you made using the PR's "Files" tab.

PRs must pass through the checks and the code review described in the [Acceptance Criteria](#acceptance-criteria) section before they can be merged.

Checks may take some time to complete. You can view their progress in the table near the bottom of the pull request page. You may also be able to use the links in the table
to view logs associated with a check if it fails.

During code reviews, another developer will take a look through your proposed change. If any modifications are requested (or further discussion about anything is
needed), they may leave a comment. You can follow up and respond to the comment, and/or create comments of your own if you have questions or ideas.
When a modification request has been completed, the conversation thread about it will be marked as resolved.

To update the code in your PR (eg. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.
