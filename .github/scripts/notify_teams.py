#!/usr/bin/env python3
"""
Microsoft Teams Notification Script for CI Failures

This script extracts error context from build/test logs and sends
notifications to Microsoft Teams via webhook.

Usage:
    python notify_teams.py \
        --project miopen \
        --failure-stage build \
        --log-path TheRock/build/logs \
        --webhook-url "$WEBHOOK_URL" \
        [--pr-number NUM] \
        [--pr-title "TITLE"] \
        [--job-name "Test name"] \
        [--dry-run]

Note: The GitHub Actions run URL is constructed automatically from
GITHUB_REPOSITORY and GITHUB_RUN_ID environment variables. For local
testing, these environment variables are optional and will use placeholder
values if not set.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


class ErrorExtractor:
    """Extracts and categorizes errors from log files"""

    # Error patterns for build failures
    # Note: Order matters - more specific patterns should come first
    BUILD_ERROR_PATTERNS = {
        "Compilation Issue": r"(compil|undefined reference|unresolved external|linker|link.*error)",
        "Docker Issue": r"(docker|container|image)",
        "Configuration Issue": r"(cmake.*error|configuration.*error|configuration.*fail)",
        "Timeout": r"(timeout|timed out)",
        "Permission Issue": r"(permission|access denied)",
        "Memory Issue": r"(out of memory|oom|memory)",
    }

    # Error patterns for test failures (includes build patterns + test-specific)
    TEST_ERROR_PATTERNS = {
        "Test Failure": r"(test.*fail|failed.*test|assertion|gtest)",
        "Timeout": r"(timeout|timed out)",
        "Memory Issue": r"(out of memory|oom|memory)",
        "GPU/Driver Issue": r"(gpu|device|driver)",
    }

    # Error context extraction limits
    MAX_CONTEXT_LINES = 10
    MAX_OUTPUT_LINES = 7

    def __init__(self, log_path: str, failure_stage: str):
        self.log_path = Path(log_path)
        self.failure_stage = failure_stage

    def extract(self) -> Tuple[str, str]:
        """
        Extract error context and categorize the issue type.

        Returns:
            Tuple of (error_log, issue_type)
        """
        if self.log_path.is_dir():
            return self._extract_from_directory()
        if self.log_path.is_file():
            return self._extract_from_file()
        return (
            "Error details not found in logs. Check the GitHub Actions run for full output.",
            f"{self.failure_stage.title()} Failure",
        )

    def _extract_from_directory(self) -> Tuple[str, str]:
        """Extract errors from a directory of log files"""
        error_log = ""

        if self.log_path.exists():
            try:
                # Use grep to search for errors across all files in directory
                result = subprocess.run(
                    [
                        "grep",
                        "--recursive",
                        "--ignore-case",
                        "--extended-regexp",
                        "(error|failed|fatal|exception)",
                        str(self.log_path),
                        "--after-context",
                        "3",
                        "--before-context",
                        "2",
                        "--max-count=5",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                error_log = result.stdout
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
            ):
                # Fallback if grep fails or not available (Windows)
                error_log = self._scan_directory_python()

        if not error_log:
            error_log = "Error details not found in logs. Check the GitHub Actions run for full output."

        # Categorize based on full output BEFORE limiting to 7 lines
        issue_type = self._categorize_error(error_log)

        # Then limit to 7 lines for the error context
        error_lines = error_log.split("\n")[:7]
        error_log_limited = "\n".join(error_lines)

        return (error_log_limited, issue_type)

    def _extract_from_file(self) -> Tuple[str, str]:
        """Extract errors from a single log file"""
        error_log = ""

        if self.failure_stage == "test" and self.log_path.exists():
            # For test logs, read from the file directly
            try:
                with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    last_lines = lines[-100:] if len(lines) > 100 else lines

                    # Search for failure patterns
                    error_lines = []
                    for i, line in enumerate(last_lines):
                        if re.search(
                            r"(error|failed|fatal|exception|assertion)",
                            line,
                            re.IGNORECASE,
                        ):
                            # Include context: 2 lines before and 3 lines after
                            start = max(0, i - 2)
                            end = min(len(last_lines), i + 4)
                            error_lines.extend(last_lines[start:end])
                            if len(error_lines) >= self.MAX_CONTEXT_LINES:
                                break

                    error_log = "".join(error_lines[: self.MAX_CONTEXT_LINES])
            except (IOError, UnicodeDecodeError):
                error_log = f"Test failed: Could not read log file at {self.log_path}"
        elif self.log_path.exists():
            # For build logs from a single file
            return self._extract_from_directory()  # Use grep approach

        if not error_log:
            error_log = "Error details not found in logs. Check the GitHub Actions run for full output."

        # Categorize based on full output BEFORE limiting to 7 lines
        issue_type = self._categorize_error(error_log)

        # Then limit to 7 lines for the error context
        error_lines = error_log.split("\n")[:7]
        error_log_limited = "\n".join(error_lines)

        return (error_log_limited, issue_type)

    def _scan_directory_python(self) -> str:
        """Fallback method to scan directory using Python (for Windows compatibility)"""
        error_log = ""
        max_files = 10
        files_scanned = 0

        for log_file in self.log_path.rglob("*.log"):
            if files_scanned >= max_files:
                break

            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    matches = re.finditer(
                        r"(error|failed|fatal|exception)", content, re.IGNORECASE
                    )
                    for match in list(matches)[:5]:
                        start = max(0, match.start() - 100)
                        end = min(len(content), match.end() + 100)
                        error_log += content[start:end] + "\n"
                        if len(error_log) > 500:
                            return error_log
                files_scanned += 1
            except (IOError, UnicodeDecodeError):
                continue

        return error_log

    def _categorize_error(self, error_log: str) -> str:
        """Categorize the error based on patterns in the log"""
        patterns = (
            self.TEST_ERROR_PATTERNS
            if self.failure_stage == "test"
            else self.BUILD_ERROR_PATTERNS
        )

        for issue_type, pattern in patterns.items():
            if re.search(pattern, error_log, re.IGNORECASE):
                return issue_type

        # Default issue type
        return f"{self.failure_stage.title()} Failure"

    def _limit_output(self, error_log: str) -> str:
        """Limit error output to MAX_OUTPUT_LINES"""
        error_lines = error_log.split("\n")[: self.MAX_OUTPUT_LINES]
        return "\n".join(error_lines)


class TeamsNotifier:
    """Sends notifications to Microsoft Teams via webhook"""

    def __init__(self, webhook_url: str, dry_run: bool = False):
        self.webhook_url = webhook_url
        self.dry_run = dry_run

    def send_notification(
        self,
        project: str,
        platform: str,
        failure_stage: str,
        run_url: str,
        error_context: str,
        issue_type: str,
        pr_number: Optional[str] = None,
        pr_title: Optional[str] = None,
        job_name: Optional[str] = None,
    ) -> bool:
        """
        Send notification to Teams webhook.

        Returns:
            True if successful, False otherwise
        """
        # Validate webhook URL
        if not self.webhook_url or self.webhook_url in ["", "test"]:
            print(
                f"Warning: Webhook URL not set for project '{project}', skipping Teams notification"
            )
            return True  # Exit gracefully

        # Format the message
        message = self._format_message(
            project,
            platform,
            failure_stage,
            run_url,
            error_context,
            issue_type,
            pr_number,
            pr_title,
            job_name,
        )

        if self.dry_run:
            print("DRY RUN MODE - Would send the following notification:")
            print("=" * 80)
            print(json.dumps(message, indent=2))
            print("=" * 80)
            return True

        # Send the notification
        try:
            import urllib.request

            data = json.dumps(message).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status in [200, 202]:
                    print(
                        f"Successfully sent Teams notification for {project} with response status: {response.status}"
                    )
                    return True
        except Exception as e:
            print(f"Error sending Teams notification: {e}")
            return False

    def _format_message(
        self,
        project: str,
        platform: str,
        failure_stage: str,
        run_url: str,
        error_context: str,
        issue_type: str,
        pr_number: Optional[str] = None,
        pr_title: Optional[str] = None,
        job_name: Optional[str] = None,
    ) -> dict:
        """Format the Teams message payload for Power Automate webhook"""

        # Platform name capitalization
        platform_name = platform.title()
        failure_label = f"{failure_stage.title()} Failure"

        # Build PR info section
        pr_info = ""
        if pr_title and pr_number:
            pr_info = f"\n**PR #{pr_number}:** {pr_title}\n"

        # Build job name section
        job_info = ""
        if job_name:
            job_info = f"\n**Job:** {job_name}\n"

        # Clean error context (remove excessive whitespace, limit length)
        error_snippet = error_context.strip()
        if len(error_snippet) > 1000:
            error_snippet = error_snippet[:1000] + "..."

        # Build the text message with proper markdown (no escaping needed for Power Automate)
        text = (
            f"{platform_name} {failure_label}\n\n"
            f"{pr_info}\n\n"
            f"{job_info}\n\n"
            f"**Issue Type:** {issue_type}\n\n"
            f"**Error Context:**\n```\n{error_snippet}\n```\n\n"
            f"[View Full Run]({run_url})"
        )

        # Create simple JSON payload for Power Automate webhook
        message = {"text": text}

        return message


def main():
    parser = argparse.ArgumentParser(
        description="Send Microsoft Teams notifications for CI failures"
    )
    parser.add_argument(
        "--project", required=True, help="Project name (e.g., miopen, rocblas)"
    )
    parser.add_argument(
        "--failure-stage",
        required=True,
        choices=["build", "test"],
        help="Stage of failure (build or test)",
    )
    parser.add_argument(
        "--log-path",
        required=True,
        help="Path to log file or directory containing logs",
    )
    parser.add_argument(
        "--webhook-url", required=True, help="Microsoft Teams webhook URL"
    )
    parser.add_argument(
        "--pr-number", default="", help="Pull request number (optional)"
    )
    parser.add_argument(
        "--pr-title", nargs="*", default=[], help="Pull request title (optional)"
    )
    parser.add_argument(
        "--job-name", default="", help="Job/test name (optional, for test failures)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the message instead of sending it"
    )

    args = parser.parse_args()

    # Detect platform automatically
    detected_platform = platform.system().lower()

    # Construct GitHub Actions run URL from environment variables
    # Use defaults for local testing when environment variables are not set
    github_repository = os.getenv("GITHUB_REPOSITORY", "local/testing")
    github_run_id = os.getenv("GITHUB_RUN_ID", "0")

    run_url = f"https://github.com/{github_repository}/actions/runs/{github_run_id}"

    if github_repository == "local/testing":
        print("Warning: Running in local test mode (GITHUB_REPOSITORY not set)")
        print(f"Using placeholder run URL: {run_url}")

    # Extract error context
    extractor = ErrorExtractor(args.log_path, args.failure_stage)
    error_context, issue_type = extractor.extract()

    # Handle pr_title which may be a list due to nargs='*'
    pr_title = " ".join(args.pr_title) if args.pr_title else None

    # Send notification
    notifier = TeamsNotifier(args.webhook_url, args.dry_run)
    success = notifier.send_notification(
        project=args.project,
        platform=detected_platform,
        failure_stage=args.failure_stage,
        run_url=run_url,
        error_context=error_context,
        issue_type=issue_type,
        pr_number=args.pr_number if args.pr_number else None,
        pr_title=pr_title,
        job_name=args.job_name if args.job_name else None,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
