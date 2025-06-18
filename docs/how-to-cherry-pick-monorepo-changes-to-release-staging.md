# How to cherry-pick monorepo changes into release-staging branches

When a project has been migrated into the ROCm monorepo, day-to-day work happens on the monorepo’s `develop` branch.  
Down-stream teams, however, still consume the original (pre-monorepo) repositories, particularly their `release-staging/rocm-rel-x.y` branches, through a variety of mechanisms.
This document explains how to move a change from the monorepo into those release-staging branches while guaranteeing that every commit on a release-staging branch also exists in the monorepo.  

## 1. Land the change in the monorepo

1. Create a pull request in `ROCm/rocm-libraries` that targets `develop`.  
2. When merging, choose **Squash & Merge** (if the change can be represented as a single logical commit).  
   Why? A single commit is easier to cherry-pick later.

Result: The commit is now on `ROCm/rocm-libraries:develop`.

## 2. Wait for the automatic “fan-out” sync

Every ~15 minutes, a CI job copies new commits from the monorepo back into the corresponding standalone repositories.

After merging your PR:

1. Monitor the CI job or simply wait ~15 minutes.  
2. Go to the original repo.
3. Pull the propagated commit.

```
$ git checkout develop
$ git pull origin develop
$ git log
commit 3aa5b75e...  ← note this SHA
Author:   John Doe
Date:     2025-06-12
    [rocm-libraries] fix: add function foo()
```

4. Write down the SHA (`3aa5b75e` in this example).

## 3. Cherry-pick into the release-staging branch

1. Create a local branch based on the release-staging branch:

```
$ git checkout -b cherry-pick-foo-rel-7.0 origin/release-staging/rocm-rel-7.0
```

2. Cherry-pick the commit you noted:

```
$ git cherry-pick 3aa5b75e
```

3. Resolve any merge conflicts (rare if the branch is close to develop).
4. Push the branch and open a PR that targets  
   `release-staging/rocm-rel-7.0`.

5. Request reviews, obtain approvals, and merge.

## FAQ

Q : Can I cherry-pick multiple commits at once?  
A : Yes, but prefer a squash merge in the monorepo so you only need to pick one.

Q : What if the auto-sync hasn’t copied the commit?  
A : Verify the CI status in `rocm-libraries`. If failed, ask the infra team; the commit will re-sync after a successful run.

Q : Can I push directly to the release-staging branch?  
A : No. Always go through a PR so CI and reviewers can validate the cherry-pick.

## Summary

In short:

1. Merge change to monorepo `develop`.  
2. Wait for auto-sync to original repo `develop`.  
3. Cherry-pick to `release-staging/rocm-rel-7.0`.  

Following this process keeps release branches perfectly in sync with the monorepo history while allowing critical fixes to flow to down-stream consumers.
