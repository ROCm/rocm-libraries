#!/usr/bin/env python3
"""
Utility functions for naming conventions and templates in monorepo fanout automation.

Example Usage:

    from monorepo_utils import FanoutNaming

    # Static method when PR title is needed
    pr_title = FanoutNaming.compute_pr_title(args.pr, entry.name)

    # Static method when PR body is needed
    pr_body = FanoutNaming.compute_pr_body(args.pr, entry.name, args.repo)

    # Static method when only branch name is needed
    branch = FanoutNaming.compute_branch_name(args.pr, entry.name)

    # Instance method to create a FanoutNaming object
    naming = FanoutNaming(
        pr_number=args.pr,
        monorepo=args.repo,
        category=entry.category,
        name=entry.name,
        subrepo="ROCm/rocBLAS"
    )

"""

from dataclasses import dataclass

@dataclass
class FanoutNaming:
    pr_number: int          # pull request number in the monorepo
    monorepo: str           # monorepo in org/repo format
    category: str           # category of the subrepo (e.g., projects, shared)
    name: str               # name of the subrepo in category/name format
    subrepo: str            # subrepo in org/repo format

    @property
    def branch_name(self) -> str:
        return self.compute_branch_name(self.pr_number, self.name)

    @staticmethod
    def compute_branch_name(pr_number: int, name: str) -> str:
        return f"monorepo-pr/{pr_number}/{name}"

    @property
    def pr_title(self) -> str:
        return f"[MONOREPO AUTO-FANOUT] PR #{self.pr_number} to {self.name}"

    @property
    def prefix(self) -> str:
        return f"{self.category}/{self.name}"

    @property
    def pr_body(self) -> str:
        return (
            f"This is an automated PR for subtree `{self.prefix}` "
            f"originating from monorepo PR [#{self.pr_number}](https://github.com/{self.monorepo}/pull/{self.pr_number}). "
            f"PLEASE DO NOT MERGE OR TOUCH THIS PR, AUTOMATED WORKFLOWS FROM THE MONOREPO ARE USING IT."
        )

    @property
    def subrepo_full_url(self) -> str:
        return f"https://github.com/{self.subrepo}.git"
