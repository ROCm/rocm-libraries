# Contributing to the ROCm Libraries

Thank you for contributing! This guide outlines the development workflow, contribution standards, and best practices when working in the monorepo.

## Getting Started

### Option A: Clone the Monorepo

```bash
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
```

### Option B: Clone the Monorepo with Sparse-Checkout

To limit your local checkout to only the project(s) you work on and improve performance with a large codebase, you can configure sparse-checkout prior to cloning:

```bash
git clone --no-checkout https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas shared/tensile
git checkout develop # or the branch you are starting from
```

## Working on Multiple Projects

If your work involves changing projects or introducing new projects, you can update your sparse-checkout environment:

```bash
git sparse-checkout set projects/hipsparse projects/rocsparse
```

This keeps your working directory clean and fast, as you won't need to clone the entire monorepo.

---

## Directory Structure

- `.github/`: CI workflows, scripts, and configuration files for synchronizing repositories during the migration period.
- `docs/`: Documentation, including this guide and other helpful resources.
- `projects/<name>/`: Each folder corresponds to a ROCm library that was previously maintained in its own GitHub repository and released as distinct packages.
- `shared/<name>/`: Shared components that existed in their own repository, used as dependencies by multiple libraries, but do not produce distinct packages in previous ROCm releases.

Further changes to the structure may be made to improve development efficiency and minimize redundancy.

---

## Making Changes

### From a Developer's Perspective

You can continue working inside your project's folder as you did before the monorepo migration.
This process is intended to remain as familiar as possible, though some adjustments may be made to improve efficiency based on feedback.

#### Example: hipblaslt Developer

```bash
cd projects/hipblaslt
# Edit, build, test as usual
```

---

## Keeping Your Branch in Sync

To stay up to date with the latest changes in the monorepo:

```bash
git fetch origin
git rebase origin/develop
```

Avoid using git merge to keep history clean and maintain a linear progression.

---

## Branching Model

We are transitioning to trunk-based development.
Until the switch is fully implemented, we will continue to sync changes to individual repositories following their existing development model (e.g., `develop` -> `staging` -> `mainline` -> `release`).
However, once trunk-based development is in place, feature branches will be created directly from the default branch, develop.

## Pull Request Guidelines

### 1. Branch Naming and Forks

When creating a branch for your work, use the following convention to make branch names informative and consistent: `users/<github-husername>/<branch-name>`.

Try to keep branch names descriptive yet concise to reflect the purpose of the branch. For example, referencing the GitHub Issue number if the pull request is related.

The build and test infrastructure has some tasks where pull requests from forks have fewer privileges than pull requests from branches within this repo. Thus, branches in this repo are encouraged but you are welcome to use forks and their potential gaps.

### 2. Opening the PR

Once you're ready:

```bash
git push origin branch-name-like-above
```

### 3. Auto-Labeling and Review Routing

The monorepo uses automation to assign labels and reviewers based on the changed files. Reviewers are designated via the top-level CODEOWNERS file.

### 4. Fanout to Subrepos

To streamline the transition to the monorepo, existing checks will be leveraged. If your PR in the monorepo modifies files from a previously standalone repository, the system will automatically create or update child PRs in those repositories. The results from these child PR checks will be reflected back into the monorepo PR.

Automated jobs will handle synchronization and tracking tasks during this transition period.

You don’t need to maintain the individual repositories once you’re onboarded to the monorepo—just focus on the monorepo PR.

### 5. Tests and CI

Eventually, existing infrastructure will be updated to directly point to the monorepo, and references to the old repositories will be removed. Until that point, the fanout sequence will be deployed.


---

- 💬 [Start a discussion](https://github.com/ROCm/rocm-libraries/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-libraries/issues)

Happy contributing!
