<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to hipFFT">
  <meta name="keywords" content="ROCm, contributing, hipFFT">
</head>

# Contributing to hipFFT #

We welcome contributions to hipFFT.  Please follow these details to help ensure your contributions will be successfully accepted.

## Issue Discussion ##

Please use the GitHub Issues tab to notify us of issues.

* Use your best judgment for issue creation. If your issue is already listed, upvote the issue and
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

When a contribution is submitted via a pull request, a number of automated checks are run in order to verify compilation correctness and prevent performance regressions.

These checks include:

* Building and testing the change on various OS platforms (Ubuntu, RHEL, etc.)
* Running on different AMD GPU architectures (MI-series, Radeon series cards, etc.)
* Running on different NVIDIA GPU architectures (V100, A100, etc)
* Running benchmarks to check for performance degradation

In order for a submission to be accepted:
* It must pass all of the automated checks
* It must undergo a code review

Users can visualize our continuous integration infrastructure in: `hipFFT/.jenkins`.

The GitHub "Issues" tab may also be used to discuss ideas surrounding particular features or changes before raising pull requests.

## Code Structure ##

In a broad view, hipFFT library is structured as follows:

        ├── docs/: contains hipFFT documentation
        ├── library/:  contains main source code and headers
        │   ├── src/amd_detail/    : for porting to AMD devices
        │   ├── src/nvidia_detail/ : for porting to NVIDIA devices
        ├── clients/:
        │   ├── bench/   : contains benchmarking code
        │   ├── samples/ : contains examples
        │   ├── tests/   : contains our test infrastructure
        ├── shared/: contains important global headers and those for linking to other applications

## Coding Style ##

* All public APIs are C89 compatible; all other library code should use c++17.
* Our minimum supported compiler is clang 3.6.
* Avoid CamelCase: rule applies specifically to publicly visible APIs, but is encouraged (not mandated) for internal code.

* C and C++ code should be formatted using `clang-format`. You can use the clang-format version available in `hipFFT/.clang-format`.

    To format a C/C++ file, use:

    ```
    clang-format -style=file -i <path-to-source-file>
    ```
* Python code should use:

    ```
    yapf --style pep8
    ```

## Pull Request Guidelines ##

Our code contribution guidelines closely follow the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/).

This repository follows the [git flow](http://nvie.com/posts/a-successful-git-branching-model/) workflow, which dictates a /master branch where releases are cut, and a /develop branch which serves as an integration branch for new code.

Note that a [git extension](https://github.com/nvie/gitflow) has been developed to ease the use of the 'git flow' methodology, but requires manual installation by the user.

The following guidelines apply:

* When you create a pull request, you should target the default branch. Our current default branch is the **develop** branch.
* Note that releases are cut to release/rocm-rel-x.y, where x and y refer to the release major and minor numbers.
* Ensure code builds successfully.
* Do not break existing test cases
* Code must also have benchmark tests, and performance must approach the compute bound limit or memory bound limit.

### Deliverables ###

New changes should include test coverage. Our testing infrastructure is located in `clients/tests/`, and can be used as a reference.

The following guidelines apply:

* New functionality will only be merged with new unit tests.
* New unit tests should integrate within the existing [googletest framework](https://github.com/google/googletest/blob/master/googletest/docs/Primer.md).
* Tests must have good code coverage.


### Process ###

All pull requests must pass through the checks and the code review described in the [Acceptance Criteria](#acceptance-criteria) section before they can be merged.

Once a contribution is ready to be submitted, consider the following:

* Before you create a PR, ensure that all files have been gone through the clang formatting: clang-format -i <changed_file>

* While creating a PR, you can take a look at a `diff` of the changes you made using the PR's "Files" tab, and verify that no unintentional changes are being submitted.

* Checks may take some time to complete. You can view their progress in the table near the bottom of the pull request page. You may also be able to use the links in the table to view logs associated with a check if it fails.

* During code reviews, another developer will take a look through your proposed change. If any modifications are requested (or further discussion about anything is needed), they may leave a comment. You can follow up and respond to the comment, and/or create comments of your own if you have questions or ideas.

* When a modification request has been completed, the conversation thread about it will be marked as resolved.

* To update the code in your PR (eg. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.

* Once your contribution is approved, we will use the *squash merge* option from GitHub to integrate it to the corresponding branch.

## Code License ##

All code contributed to this project will be licensed under the license identified in the [LICENSE.md](https://github.com/ROCm/hipFFT/blob/develop/LICENSE.md). Your contribution will be accepted under the same license.
