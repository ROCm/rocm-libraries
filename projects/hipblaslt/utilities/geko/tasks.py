# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Developer task runner for GEKO.

Run ``invoke --list`` to see available tasks. Common usage:

    invoke install            # editable install (pip install -e .)
    invoke test               # run the test suite
    invoke test --skip-slow   # quick run, forwards args to pytest
    invoke lint               # flake8
    invoke format             # black + isort
    invoke build              # build sdist + wheel
    invoke clean              # remove build/test artifacts
"""

from invoke.tasks import task

GEKO_DIRS = "geko scripts tests"


@task(help={"editable": "Install in editable mode (pip install -e)."})
def install(c, editable=True):
    """Install geko (and the ``geko`` console script) into the active environment."""
    flag = "-e " if editable else ""
    c.run(f"pip install {flag}.")


@task(
    help={
        "skip_slow": "Skip @pytest.mark.slow tests.",
        "skip_geko_bin": "Skip @pytest.mark.geko_bin subprocess tests.",
        "args": "Extra arguments forwarded to pytest (quote them).",
    }
)
def test(c, skip_slow=False, skip_geko_bin=False, args=""):
    """Run the pytest suite under tests/."""
    extra = []
    if skip_slow:
        extra.append("--skip-slow")
    if skip_geko_bin:
        extra.append("--skip-geko-bin")
    if args:
        extra.append(args)
    c.run(f"python3 -m pytest tests/ {' '.join(extra)}".rstrip())


@task
def lint(c):
    """Run flake8 over the geko package."""
    c.run("flake8 geko")


@task(help={"check": "Only check formatting; do not modify files."})
def format(c, check=False):
    """Format code with black and sort imports with isort."""
    chk = " --check" if check else ""
    c.run(f"black --line-length=100{chk} {GEKO_DIRS}")
    c.run(f"isort --profile=black{' --check-only' if check else ''} {GEKO_DIRS}")


@task
def build(c):
    """Build an sdist and wheel into dist/."""
    c.run("python3 -m build")


@task
def clean(c):
    """Remove build, packaging, and test caches."""
    c.run(
        "rm -rf build dist *.egg-info geko.egg-info .pytest_cache .tox htmlcov "
        ".coverage coverage.xml coverage.json"
    )
    c.run(r"find . -type d -name __pycache__ -prune -exec rm -rf {} +")
