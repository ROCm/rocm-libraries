#!/usr/bin/env python3
"""
GitHub Checks API Client Utility
--------------------------------
This client allows creating or updating GitHub check runs using the REST API.
It authenticates using a GitHub token provided in the GH_TOKEN environment variable.

Only functionality for upserting (create or update) check runs is implemented,
as needed for reflecting sub-repo checks in monorepo PRs.
"""

import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubChecksAPIClient:
    def __init__(self):
        token = os.getenv("GH_TOKEN")
        if not token:
            raise EnvironmentError("GH_TOKEN environment variable not set.")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        self.api_base = "https://api.github.com"

    def get_pr_checks(self, repo_url: str, pr_number: int) -> list:
        """
        Fetch check runs associated with a pull request.
        """
        url = f"{self.api_base}/repos/{repo_url}/pulls/{pr_number}/check-runs"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            logging.error(f"Failed to fetch PR checks: {response.status_code} {response.text}")
            response.raise_for_status()
        check_runs = response.json().get("check_runs", [])
        logging.debug(f"Fetched {len(check_runs)} check runs for PR {pr_number} in {repo_url}")
        return check_runs

    def create_synthetic_check(self, repo_url: str, pr_number: int, check_name: str, status: str, conclusion: str, summary: str) -> None:
        """
        Create a synthetic check run for the monorepo pull request.
        """
        url = f"{self.api_base}/repos/{repo_url}/check-runs"
        payload = {
            "name": check_name,
            "head_sha": pr_number,  # Use the PR's head commit SHA here if needed
            "status": status,
            "conclusion": conclusion,
            "output": {
                "title": check_name,
                "summary": summary
            }
        }
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 201:
            logging.info(f"Successfully created synthetic check '{check_name}' for PR {pr_number} in {repo_url}")
        else:
            logging.error(f"Failed to create synthetic check: {response.status_code} {response.text}")
            response.raise_for_status()

    def upsert_check_run(self, repo_url: str, check_name: str, pr_number: int, status: str, conclusion: str, summary: str) -> None:
        """
        Create or update a check run in a repository.
        """
        check_runs = self.get_pr_checks(repo_url, pr_number)
        existing_check = next((check for check in check_runs if check["name"] == check_name), None)
        if existing_check:
            check_run_id = existing_check["id"]
            url = f"{self.base_url}/repos/{repo_url}/check-runs/{check_run_id}"
            payload = {
                "status": status,
                "conclusion": conclusion,
                "output": {
                    "title": check_name,
                    "summary": summary
                }
            }
            response = requests.patch(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                logging.info(f"Successfully updated check '{check_name}' for PR {pr_number} in {repo_url}")
            else:
                logging.error(f"Failed to update check run: {response.status_code} {response.text}")
                response.raise_for_status()
        else:
            self.create_synthetic_check(repo_url, pr_number, check_name, status, conclusion, summary)
