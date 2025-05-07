#!/usr/bin/env python3
"""
GitHub Checks API Client Utility
--------------------------------
This client allows creating or updating GitHub check runs using the REST API.
It authenticates using a GitHub token provided in the GH_TOKEN environment variable.
This client exists because the GitHub CLI does not support creating check runs directly.
"""

import os
import requests
import logging

logger = logging.getLogger(__name__)

class GitHubAPIClient:
    def __init__(self) -> None:
        """Initialize the GitHub API client."""
        token = os.getenv("GH_TOKEN")
        if not token:
            raise EnvironmentError("GH_TOKEN environment variable not set.")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        self.api_base = "https://api.github.com"

    def _get_json(self, url: str, error_msg: str) -> dict:
        """Helper method to perform a GET request and return JSON response."""
        logger.debug(f"GET {url}")
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            logger.error(f"{error_msg}: {response.status_code} {response.text}")
            response.raise_for_status()
        return response.json()

    def _request_json(self, method: str, url: str, json: dict, error_msg: str) -> dict:
        """Helper method to perform a request and return JSON response."""
        response = requests.request(method, url, headers=self.headers, json=json)
        if not response.ok:
            logger.error(f"{error_msg}: {response.status_code} {response.text}")
            response.raise_for_status()
        return response.json()

    def _check_run_payload(self, check_name: str, status: str, details_url: str, conclusion: str, summary: str, head_sha: str) -> dict:
        """Create the payload for a check run with potentially empty fields safely coerced to strings."""
        payload = {
            "name": check_name,
            "head_sha": head_sha,
            "status": status,
            "details_url": str(details_url or ""),
            "conclusion": str(conclusion or "neutral"),
            "output": {
                "title": str(check_name or ""),
                "summary": str(summary or "")
            }
        }
        return payload

    def get_pr_by_head_branch(self, repo_url: str, branch_name: str) -> dict | None:
        """Fetch the pull request associated with a specific head branch."""
        org = repo_url.split('/')[0]  # Extract the organization name from repo_url
        full_head_ref = f"{org}:{branch_name}"
        url = f"{self.api_base}/repos/{repo_url}/pulls?head={full_head_ref}&state=open"
        prs = self._get_json(url, f"Failed to fetch PR by branch name in {repo_url}")
        return prs[0] if prs else None

    def get_pr_checks(self, repo_url: str, pr_number: int) -> list:
        """Fetch check runs associated with a pull request."""
        # the api has checks assigned to SHA, so map PR to a SHA first
        pr_url = f"{self.api_base}/repos/{repo_url}/pulls/{pr_number}"
        pr = self._get_json(pr_url, f"Failed to fetch PR #{pr_number} in {repo_url}")
        sha = pr["head"]["sha"]
        checks_url = f"{self.api_base}/repos/{repo_url}/commits/{sha}/check-runs"
        data = self._get_json(checks_url, f"Failed to fetch check runs for commit {sha} in {repo_url}")
        return data.get("check_runs", [])

    def create_synthetic_check(self, repo_url: str, pr_number: int, check_name: str,
                               status: str, details_url: str, conclusion: str, summary: str) -> None:
        """Create a synthetic check run for the monorepo pull request."""
        pr_url = f"{self.api_base}/repos/{repo_url}/pulls/{pr_number}"
        pr = self._get_json(pr_url, f"Failed to get PR for synthetic check in {repo_url}")
        head_sha = pr["head"]["sha"]
        payload = self._check_run_payload(check_name, status, details_url, conclusion, summary, head_sha)
        url = f"{self.api_base}/repos/{repo_url}/check-runs"
        self._request_json("POST", url, payload, f"Failed to create synthetic check '{check_name}'")
        logger.info(f"Created synthetic check '{check_name}' for PR #{pr_number} in {repo_url}")

    def upsert_check_run(self, repo_url: str, check_name: str, pr_number: int,
                         status: str, details_url: str, conclusion: str, summary: str) -> None:
        check_runs = self.get_pr_checks(repo_url, pr_number)
        existing = next((check for check in check_runs if check["name"] == check_name), None)
        """Create or update a check run in a repository."""
        if existing:
            check_run_id = existing["id"]
            url = f"{self.api_base}/repos/{repo_url}/check-runs/{check_run_id}"
            payload = self._check_run_payload(check_name, status, details_url, conclusion, summary, existing["head_sha"])
            self._request_json("PATCH", url, payload, f"Failed to update check '{check_name}'")
            logger.info(f"Updated check '{check_name}' on PR #{pr_number} in {repo_url}")
        else:
            self.create_synthetic_check(repo_url, pr_number, check_name, status, details_url, conclusion, summary)
