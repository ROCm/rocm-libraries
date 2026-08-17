# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import importlib.util


def test_extension_module_is_not_public_top_level_import():
    assert importlib.util.find_spec("hipdnn_frontend_python") is None


def test_import_alias_loads_frontend_module():
    import hipdnn_frontend as hipdnn

    assert hipdnn.__name__ == "hipdnn_frontend"
    assert hasattr(hipdnn, "Graph")
