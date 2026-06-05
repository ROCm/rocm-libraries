# Test Fixtures for Teams Notification Script

This directory contains sample log files for testing the `notify_teams.py` script locally without requiring a full CI build.

## Files

### sample_build_error.log
Sample build failure log containing compilation errors. Used to test:
- Error pattern detection for compilation issues
- Build failure categorization
- Error context extraction

### sample_test_error.log
Sample test failure log containing GTest output. Used to test:
- Test failure pattern detection
- Test-specific error categorization
- GTest assertion parsing

## Usage

Test the notification script with these fixtures:

```bash
# Test build failure notification
python3 .github/scripts/notify_teams.py \
  --project miopen \
  --failure-stage build \
  --log-path test/fixtures/sample_build_error.log \
  --webhook-url "test" \
  --dry-run

# Test test failure notification
python3 .github/scripts/notify_teams.py \
  --project miopen \
  --failure-stage test \
  --log-path test/fixtures/sample_test_error.log \
  --webhook-url "test" \
  --pr-number "123" \
  --pr-title "Fix convolution backward pass" \
  --dry-run
```
Replace "test" in `--webhook-url` with the actual webhook url if you want to test sending to teams

The `--dry-run` flag will print the formatted message without actually sending it to Teams.
