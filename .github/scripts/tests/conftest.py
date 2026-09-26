"""
pytest configuration for .github/scripts/tests/.

Stubs out modules that are only available in CI (e.g. modules from a TheRock
checkout) so the full test suite can be collected and run locally without those
dependencies present.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# amdgpu_family_matrix lives in TheRock/build_tools/github_actions/ and is only
# available when TheRock is checked out alongside the monorepo. When it is not
# present, stub it out so therock_configure_ci and its tests can be imported.
_THEROCK_ACTIONS = (
    Path(__file__).parents[2] / "TheRock" / "build_tools" / "github_actions"
)
if not _THEROCK_ACTIONS.exists():
    sys.modules.setdefault("amdgpu_family_matrix", MagicMock())
