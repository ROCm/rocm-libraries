#!/usr/bin/env python3

"""
Sync Patches to Subrepositories
-------------------------------

This script is part of the monorepo synchronization system. It runs after a monorepo pull request
is merged and applies relevant changes to the corresponding sub-repositories using Git patches.

- Uses the merge commit of the monorepo PR to extract subtree changes.
- Generates one patch per changed file under each subtree.
- Applies each patch as a separate commit in the subrepo.
- Squashes all commits per subtree into one before pushing.
- Uses the repos-config.json file to map subtrees to sub-repos.
- Assumes this script is run from the root of the monorepo.

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --subtrees  : A newline-separated list of subtree paths in category/name format (e.g., projects/rocBLAS)
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:
    python pr_merge_sync_patches.py --repo ROCm/rocm-libraries --pr 123 --subtrees "$(printf 'projects/rocBLAS\nprojects/hipBLASLt\nshared/rocSPARSE')" --dry-run --debug
"""

import argparse
import os
import subprocess
import logging
import tempfile
import re
from typing import Optional, List
from pathlib import Path
from github_cli_client import GitHubCLIClient
from config_loader import load_repo_config
from repo_config_model import RepoEntry

logger = logging.getLogger(__name__)

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apply subtree patches to sub-repositories.")
    parser.add_argument("--repo", required=True, help="Full repository name (e.g., org/repo)")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--subtrees", required=True, help="Newline-separated list of changed subtrees (category/name)")
    parser.add_argument("--config", required=False, default=".github/repos-config.json", help="Path to the repos-config.json file")
    parser.add_argument("--dry-run", action="store_true", help="If set, only logs actions without making changes.")
    parser.add_argument("--debug", action="store_true", help="If set, enables detailed debug logging.")
    return parser.parse_args(argv)

def get_subtree_info(config: List[RepoEntry], subtrees: List[str]) -> List[RepoEntry]:
    """Return config entries matching the given subtrees in category/name format."""
    requested = set(subtrees)
    matched = [
        entry for entry in config
        if f"{entry.category}/{entry.name}" in requested
    ]
    missing = requested - {f"{e.category}/{e.name}" for e in matched}
    if missing:
        logger.warning(f"Some subtrees not found in config: {', '.join(sorted(missing))}")
    return matched

def _run_git(args: List[str], cwd: Optional[Path] = None) -> str:
    """Run a git command and return stdout."""
    cmd = ["git"] + args
    logger.debug(f"Running git command: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()

def _clone_subrepo(repo_url: str, branch: str, destination: Path) -> None:
    """Clone a specific branch from the given GitHub repository into the destination path."""
    _run_git([
        "clone",
        "--branch", branch,
        "--single-branch",
        f"https://github.com/{repo_url}",
        str(destination)
    ])
    logger.debug(f"Cloned {repo_url} into {destination}")

def _configure_git_user(repo_path: Path) -> None:
    """Configure git user.name and user.email for the given repository directory."""
    _run_git(["config", "user.name", "assistant-librarian[bot]"], cwd=repo_path)
    _run_git(["config", "user.email", "assistant-librarian[bot]@users.noreply.github.com"], cwd=repo_path)

def _apply_patch(repo_path: Path, patch_path: Path) -> None:
    """Apply a patch file to the working tree."""
    _run_git(["apply", str(patch_path)], cwd=repo_path)
    logger.info(f"Applied patch to working tree at {repo_path}")

def _stage_changes(repo_path: Path) -> None:
    """Stage all changes in the repository."""
    _run_git(["add", "."], cwd=repo_path)
    logger.debug(f"Staged all changes in {repo_path}")

def _extract_commit_message_from_patch(patch_path: Path) -> str:
    """Extract and clean the original commit message from the patch file,
    removing '[PATCH]' and trailing PR references like (#NN) from the title."""
    with open(patch_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    commit_msg_lines = []
    in_msg = False
    for line in lines:
        if line.startswith("Subject: "):
            subject = line[len("Subject: "):].strip()
            # Remove leading "[PATCH]" if present
            if subject.startswith("[PATCH]"):
                subject = subject[len("[PATCH]"):].strip()
            # Remove trailing PR refs like (#NN)
            subject = re.sub(r"\s*\(#\d+\)$", "", subject)
            commit_msg_lines.append(subject + "\n")
            in_msg = True
        elif in_msg:
            if line.startswith("---"):
                break
            commit_msg_lines.append(line)
    return "".join(commit_msg_lines).strip()

def _format_commit_message(monorepo_url: str, pr_number: int, merge_sha: str, original_msg: str) -> str:
    """Prepend a sync annotation to the original commit message."""
    annotation = f"[rocm-libraries] {monorepo_url}#{pr_number} (commit {merge_sha[:7]})\n\n"
    return annotation + original_msg

def _commit_changes(repo_path: Path, message: str, author_name: str, author_email: str) -> None:
    """Commit staged changes with the specified author and message."""
    _run_git([
        "commit",
        "--author", f"{author_name} <{author_email}>",
        "-m", message
    ], cwd=repo_path)
    logger.debug(f"Committed changes with author {author_name} <{author_email}>")

def _set_authenticated_remote(repo_path: Path, repo_url: str) -> None:
    """Set the push URL to use the GitHub App token from GH_TOKEN env."""
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN environment variable is not set")
    remote_url = f"https://x-access-token:{token}@github.com/{repo_url}.git"
    _run_git(["remote", "set-url", "origin", remote_url], cwd=repo_path)

def _push_changes(repo_path: Path, branch: str) -> None:
    """Push the commit to origin of branch."""
    _run_git(["push", "origin", branch], cwd=repo_path)
    logger.debug(f"Pushed changes from {repo_path} to origin")

def generate_file_level_patches(prefix: str, merge_sha: str, output_dir: Path) -> List[Path]:
    """
    Generate one patch file per changed file in the given subtree prefix from a merge commit.
    Returns list of Path objects to the generated patch files.
    """
    # Get changed files in that subtree for the merge commit
    files_str = _run_git([
        "diff-tree", "--no-commit-id", "--name-only", "-r", merge_sha, "--", prefix
    ])
    files = [line.strip() for line in files_str.splitlines() if line.strip()]
    patch_files = []

    for changed_file in files:
        # Generate patch for each file relative to subtree prefix
        # Note: format-patch does not have an easy single-file option, so fallback to git diff output to patch file
        patch_path = output_dir / (changed_file.replace("/", "_") + ".patch")
        diff_output = _run_git([
            "diff", f"{merge_sha}^!", "--", changed_file
        ])
        # Write patch to file, but rewrite paths to be relative to subtree prefix by removing prefix/
        # Since changed_file is under prefix, strip prefix + slash from paths in patch header
        adjusted_patch = _adjust_patch_paths(diff_output, prefix)
        patch_path.write_text(adjusted_patch, encoding="utf-8")
        patch_files.append(patch_path)

    logger.debug(f"Generated {len(patch_files)} file-level patches for prefix '{prefix}'")
    return patch_files

def _adjust_patch_paths(patch_text: str, prefix: str) -> str:
    """
    Remove prefix/ from the paths in the patch text to make it relative for application in subrepo.
    """
    # Typical patch headers:
    # --- a/prefix/filename
    # +++ b/prefix/filename
    # We replace 'a/prefix/' and 'b/prefix/' with 'a/' and 'b/'
    lines = patch_text.splitlines()
    adjusted_lines = []
    prefix_slash = prefix.rstrip("/") + "/"
    for line in lines:
        if line.startswith("--- a/") and line[6:].startswith(prefix_slash):
            adjusted_lines.append("--- a/" + line[6+len(prefix_slash):])
        elif line.startswith("+++ b/") and line[6:].startswith(prefix_slash):
            adjusted_lines.append("+++ b/" + line[6+len(prefix_slash):])
        else:
            adjusted_lines.append(line)
    return "\n".join(adjusted_lines) + "\n"

def resolve_patch_author(client: GitHubCLIClient, repo: str, pr: int) -> tuple[str, str]:
    """Determine the appropriate author for the patch
    Returns: (author_name, author_email)"""
    pr_data = client.get_pr_by_number(repo, pr)
    body = pr_data.get("body", "") or ""
    match = re.search(r"Originally authored by @([A-Za-z0-9_-]+)", body)
    if match:
        username = match.group(1)
        logger.debug(f"Found originally authored username in PR body: @{username}")
    else:
        username = pr_data["user"]["login"]
        logger.debug(f"No explicit original author, using PR author: @{username}")
    name, email = client.get_user(username)
    return name or username, email

def apply_patches_and_squash(entry: RepoEntry, monorepo_url: str, monorepo_pr: int,
                            patch_paths: List[Path], author_name: str, author_email: str,
                            merge_sha: str, dry_run: bool = False) -> None:
    """
    Clone the subrepo, apply multiple file-level patches each as a commit,
    then squash all new commits into one before pushing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subrepo_path = Path(tmpdir) / entry.name
        _clone_subrepo(entry.url, entry.branch, subrepo_path)
        if dry_run:
            logger.info(f"[Dry-run] Would apply {len(patch_paths)} patches to {entry.url} as {author_name} <{author_email}>")
            return

        _configure_git_user(subrepo_path)

        # Get current HEAD commit (before applying patches)
        base_commit = _run_git(["rev-parse", "HEAD"], cwd=subrepo_path)

        for patch_path in patch_paths:
            logger.debug(f"Applying patch {patch_path.name} to {entry.name}")
            _apply_patch(subrepo_path, patch_path)
            _stage_changes(subrepo_path)
            original_commit_msg = _extract_commit_message_from_patch(patch_path)
            commit_msg = _format_commit_message(monorepo_url, monorepo_pr, merge_sha, original_commit_msg)
            _commit_changes(subrepo_path, commit_msg, author_name, author_email)

        # Squash all commits since base_commit into one
        logger.debug(f"Squashing commits since {base_commit} into one")

        # Create a combined commit message from all individual commit messages
        combined_msg_lines = []
        for patch_path in patch_paths:
            orig_msg = _extract_commit_message_from_patch(patch_path)
            combined_msg_lines.append(orig_msg)
            combined_msg_lines.append("\n---\n")
        combined_commit_msg = f"[rocm-libraries] {monorepo_url}#{monorepo_pr} (commit {merge_sha[:7]})\n\n" + "".join(combined_msg_lines).strip()

        # Perform squash via git reset + commit
        _run_git(["reset", "--soft", base_commit], cwd=subrepo_path)
        _run_git(["commit", "-m", combined_commit_msg, "--author", f"{author_name} <{author_email}>"], cwd=subrepo_path)

        _set_authenticated_remote(subrepo_path, entry.url)
        _push_changes(subrepo_path, entry.branch)
        logger.info(f"Applied and squashed patches, then pushed to {entry.url} as {author_name} <{author_email}>")

def main(argv: Optional[List[str]] = None) -> None:
    """Main function to apply patches to sub-repositories."""
    args = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO
    )
    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    subtrees = [line.strip() for line in args.subtrees.splitlines() if line.strip()]
    relevant_subtrees = get_subtree_info(config, subtrees)
    merge_sha = client.get_squash_merge_commit(args.repo, args.pr)
    logger.debug(f"Merge commit for PR #{args.pr} in {args.repo}: {merge_sha}")
    for entry in relevant_subtrees:
        prefix = f"{entry.category}/{entry.name}"
        logger.debug(f"Processing subtree {prefix}")
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_dir = Path(tmpdir)
            patch_files = generate_file_level_patches(prefix, merge_sha, patch_dir)
            if not patch_files:
                logger.info(f"No changed files detected for subtree {prefix}, skipping.")
                continue
            author_name, author_email = resolve_patch_author(client, args.repo, args.pr)
            apply_patches_and_squash(entry, args.repo, args.pr,
                                     patch_files, author_name, author_email,
                                     merge_sha, args.dry_run)

if __name__ == "__main__":
    main()
