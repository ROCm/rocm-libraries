# Contributing to the ROCm Libraries

Thank you for contributing! This guide outlines the development workflow, contribution standards, and best practices when working in the monorepo.

---

## Getting Started

### 1. Clone the Monorepo

```bash
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
```

### 2. Enable Sparse-Checkout (Optional but Recommended)

To limit your local checkout to only the project(s) you work on and improve performance with a large codebase:

```bash
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas shared/tensile
```

You can add more folders later with:

```bash
git sparse-checkout set --add projects/hipblas
```

This keeps your working directory clean and fast, as you won't need to clone the entire monorepo.

After setting or updating the sparse paths, make sure to update your working directory to reflect the new layout:

```bash
git checkout develop  # or the current branch you're working on
```

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

## Working on Multiple Projects

You can work on multiple projects simultaneously by adjusting your sparse-checkout:

```bash
git sparse-checkout set projects/hipsparse projects/rocsparse
```

This allows you to focus on multiple libraries at once without the need to checkout the entire monorepo.

---

## Keeping Your Branch in Sync

To stay up to date with the latest changes in the monorepo:

```bash
git fetch origin
git rebase origin/develop
```

Avoid using git merge to keep history clean and maintain a linear progression.

---

## Pull Request Guidelines

### 1. Auto-Labeling and Review Routing

The monorepo uses automation to assign labels and reviewers based on the changed files. Reviewers are designated via the top-level CODEOWNERS file.

### 2. Fanout to Subrepos

To streamline the transition to the monorepo, existing checks will be leveraged. If your PR in the monorepo modifies files from a previously standalone repository, the system will automatically create or update child PRs in those repositories. The results from these child PR checks will be reflected back into the monorepo PR.

Automated jobs will handle synchronization and tracking tasks during this transition period.

You don’t need to maintain the individual repositories once you’re onboarded to the monorepo—just focus on the monorepo PR.

### 3. Tests and CI

Eventually, existing infrastructure will be updated to directly point to the monorepo, and references to the old repositories will be removed.

---

### Branching Model

We are transitioning to trunk-based development. Until the switch is fully implemented, we will continue to sync changes to individual repositories following their existing development model (e.g., develop -> staging -> mainline -> release). However, once trunk-based development is in place, feature branches will be created directly from main.

---

## Submitting a PR

Once you're ready:

```bash
git push origin feature/my-change
```

Open a PR in GitHub. The labels and reviewer routing will be handled automatically based on your changes.

---

- 💬 [Start a discussion](https://github.com/ROCm/rocm-libraries/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-libraries/issues)

Happy contributing!
