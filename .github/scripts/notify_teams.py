#!/usr/bin/env python3
"""
Microsoft Teams Notification Script for CI Failures

This script extracts error context from build/test logs and sends
notifications to Microsoft Teams via webhook. It supports automatic
routing to the correct Teams channel based on PR title tags or
component names.

Usage (Multi-project routing - recommended):
    python notify_teams.py \
        --failure-stage test \
        --log-path ./test_logs/test_output.log \
        --webhook-urls '{"miopen": "URL1", "hipdnn": "URL2"}' \
        --pr-number NUM \
        --pr-title "[miopen] Fix bug" \
        --component-name "miopen" \
        [--job-name "Test name"] \
        [--dry-run]

Usage (Legacy single-project mode):
    python notify_teams.py \
        --project miopen \
        --failure-stage build \
        --log-path TheRock/build/logs \
        --webhook-url "$WEBHOOK_URL" \
        --pr-number NUM \
        [--pr-title "TITLE"] \
        [--job-name "Test name"] \
        [--dry-run]

Routing Logic (when using --webhook-urls):
    1. Single PR title tag (e.g., "[miopen]") -> routes to that project
    2. Multiple tags (e.g., "[miopen][hipdnn]") -> uses component name to disambiguate
    3. No tags -> falls back to component name matching
    4. No match -> exits with success (0), no notification sent

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
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class ChannelRouter:
    """Routes notifications to the appropriate Teams channel based on PR title tags or component"""

    # Project configuration - order matters! First match wins.
    # Projects are checked in insertion order (Python 3.7+ guarantees this).
    PROJECT_CONFIG = {
        "miopen": {
            "title_tags": ["[miopen]"],
            "component_patterns": ["miopen"],
        },
        "hipdnn": {
            "title_tags": ["[hipdnn]", "[miopenprovider]"],
            "component_patterns": ["hipdnn"],
        },
        # Future projects can be added here:
        # "rocblas": {
        #     "title_tags": ["[rocblas]"],
        #     "component_patterns": ["rocblas"],
        # },
    }

    @staticmethod
    def validate_webhook_url(url: str) -> bool:
        """Validate that the URL is a legitimate Teams webhook URL.

        Supports both:
        - Legacy Office 365 webhooks: https://<tenant>.webhook.office.com/webhookb2/...
        - Power Automate workflows: https://<id>.environment.api.powerplatform.com/...

        Returns False for empty strings or non-Teams URLs.
        """
        if not url or not url.strip():
            return False

        # Legacy Office 365 webhook format
        office365_pattern = r"^https://[a-zA-Z0-9-]+\.webhook\.office\.com/webhookb2/.*"

        # Power Automate workflow format
        power_automate_pattern = r"^https://[a-zA-Z0-9-]+\.[\w.]*\.api\.powerplatform\.com(:\d+)?/powerautomate/.*"

        return bool(
            re.match(office365_pattern, url) or re.match(power_automate_pattern, url)
        )

    @staticmethod
    def sanitize_pr_title(title: str) -> str:
        """Sanitize PR title to prevent injection attacks"""
        # Remove potentially dangerous characters while preserving readability
        return re.sub(r'[<>&"\']', "", title)

    def _get_matching_projects_from_title(self, pr_title: str) -> List[str]:
        """Extract all matching project names from PR title tags.

        Returns:
            List of project names that have matching tags in the PR title.
        """
        matches = []
        if not pr_title:
            return matches

        pr_title_lower = pr_title.lower()
        for project, config in self.PROJECT_CONFIG.items():
            for tag in config["title_tags"]:
                if tag.lower() in pr_title_lower:
                    matches.append(project)
                    break  # Found match for this project, move to next
        return matches

    def _get_project_from_component(self, component_name: str) -> Optional[str]:
        """Match component name to a project.

        Returns:
            Project name if matched, None otherwise.
        """
        if not component_name:
            return None

        component_lower = component_name.lower()
        for project, config in self.PROJECT_CONFIG.items():
            for pattern in config["component_patterns"]:
                if pattern.lower() in component_lower:
                    return project
        return None

    def determine_target_channel(
        self, pr_title: str, component_name: str
    ) -> Optional[str]:
        """Determine which channel to notify based on PR title and component.

        Routing logic:
        1. If PR title has exactly one project tag, use that project
        2. If PR title has multiple project tags, use failing component to pick one
        3. If PR title has no tags, fall back to component matching
        4. If no match, return None

        Returns:
            str: The project name to notify, or None if no match found.
        """
        # Sanitize inputs
        pr_title = self.sanitize_pr_title(pr_title) if pr_title else ""

        # Get all projects tagged in PR title
        title_matches = self._get_matching_projects_from_title(pr_title)

        if len(title_matches) == 1:
            # Single tag - use it directly
            project = title_matches[0]
            logger.info(f"Matched project '{project}' via single PR title tag")
            return project

        if len(title_matches) > 1:
            # Multiple tags - use component to disambiguate
            logger.info(
                f"Multiple project tags found in PR title: {title_matches}. "
                f"Using component '{component_name}' to disambiguate."
            )
            component_project = self._get_project_from_component(component_name)
            if component_project and component_project in title_matches:
                logger.info(
                    f"Matched project '{component_project}' via component disambiguation"
                )
                return component_project
            else:
                # Component doesn't match any tagged project - use first tag
                project = title_matches[0]
                logger.warning(
                    f"Component '{component_name}' doesn't match any tagged project. "
                    f"Falling back to first tag: '{project}'"
                )
                return project

        # No tags found - fall back to component matching
        component_project = self._get_project_from_component(component_name)
        if component_project:
            logger.info(f"Matched project '{component_project}' via component pattern")
            return component_project

        # No match found
        logger.info(
            f"No matching project found for PR title '{pr_title}' and component '{component_name}'. "
            f"Supported projects: {', '.join(self.PROJECT_CONFIG.keys())}. Skipping notification."
        )
        return None


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
            "Error details not found in logs. Check the previous steps in GitHub Actions",
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
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Send Microsoft Teams notifications for CI failures"
    )
    parser.add_argument(
        "--project",
        default="",
        help="Project name (e.g., miopen, rocblas). Optional if using --webhook-urls with auto-routing.",
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
    # New: JSON object mapping project names to webhook URLs
    parser.add_argument(
        "--webhook-urls",
        default="",
        help='JSON object mapping project names to webhook URLs (e.g., \'{"miopen": "url1", "hipdnn": "url2"}\')',
    )
    # Legacy: single webhook URL (for backwards compatibility)
    parser.add_argument(
        "--webhook-url",
        default="",
        help="Microsoft Teams webhook URL (legacy, use --webhook-urls for multi-project support)",
    )
    parser.add_argument("--pr-number", required=True, help="Pull request number")
    parser.add_argument(
        "--pr-title", nargs="*", default=[], help="Pull request title (optional)"
    )
    parser.add_argument(
        "--component-name",
        default="",
        help="Component/job name for routing fallback when using --webhook-urls",
    )
    parser.add_argument(
        "--job-name", default="", help="Job/test name (optional, for test failures)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the message instead of sending it"
    )

    args = parser.parse_args()

    # Handle pr_title which may be a list due to nargs='*'
    pr_title = " ".join(args.pr_title) if args.pr_title else ""

    # Determine webhook URL and project based on arguments
    webhook_url = ""
    project = args.project

    if args.webhook_urls:
        # New multi-project routing mode
        try:
            webhook_urls = json.loads(args.webhook_urls)
            if not isinstance(webhook_urls, dict):
                logger.error("--webhook-urls must be a JSON object")
                sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --webhook-urls: {e}")
            sys.exit(1)

        # Determine target channel using ChannelRouter
        router = ChannelRouter()
        target_project = router.determine_target_channel(pr_title, args.component_name)

        # Exit early if no matching project found
        if target_project is None:
            logger.info("No notification sent - project not supported")
            sys.exit(0)  # Exit with success

        # Check if webhook URL exists for the matched project
        if target_project not in webhook_urls:
            logger.warning(
                f"Project '{target_project}' matched but no webhook URL provided in --webhook-urls"
            )
            sys.exit(0)  # Exit with success

        webhook_url = webhook_urls[target_project]
        project = target_project

        # Check for empty webhook URL (secret not set)
        if not webhook_url or not webhook_url.strip():
            logger.warning(
                f"Webhook URL for project '{target_project}' is empty (secret may not be set)"
            )
            sys.exit(0)  # Exit with success

        # Validate webhook URL format
        if not ChannelRouter.validate_webhook_url(webhook_url):
            logger.error(
                f"Invalid Teams webhook URL format for project '{target_project}'. "
                f"Expected: https://<tenant>.webhook.office.com/webhookb2/... or "
                f"https://<id>.environment.api.powerplatform.com/powerautomate/..."
            )
            sys.exit(1)

    elif args.webhook_url:
        # Legacy single webhook mode
        webhook_url = args.webhook_url
        if not project:
            logger.error("--project is required when using --webhook-url")
            sys.exit(1)
    else:
        logger.error("Either --webhook-urls or --webhook-url must be provided")
        sys.exit(1)

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

    # Send notification
    notifier = TeamsNotifier(webhook_url, args.dry_run)
    success = notifier.send_notification(
        project=project,
        platform=detected_platform,
        failure_stage=args.failure_stage,
        run_url=run_url,
        error_context=error_context,
        issue_type=issue_type,
        pr_number=args.pr_number if args.pr_number else None,
        pr_title=pr_title if pr_title else None,
        job_name=args.job_name if args.job_name else None,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
