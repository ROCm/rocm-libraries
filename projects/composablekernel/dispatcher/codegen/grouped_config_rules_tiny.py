"""Rule set "tiny": a minimal subset of the "tests" rule set.

This module exposes the uniform ``get_configs`` entry point expected by
``unified_grouped_conv_codegen.get_default_configs``. The selection logic lives
in ``grouped_config_rules_testing`` (``get_tiny_configs`` /
``_select_tiny_configs``); this module is a thin wrapper so the rule set is
addressable by name from the codegen dispatcher.
"""

from grouped_config_rules_testing import get_tiny_configs as get_configs  # noqa: F401
