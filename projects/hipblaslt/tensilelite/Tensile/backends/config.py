# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared backend configuration parsing and validation."""

from Tensile.Common import printExit


def parse_backend_config(raw_backend):
    """Parse and validate Backend configuration from YAML.
    
    Enforces a strict contract: Backend must be a dict with 'Name' (non-empty string) 
    and optional 'Config' (dict). All inputs are normalized to canonical form:
    {"Name": <lowercase string>, "Config": <dict or empty dict>}
    
    Args:
        raw_backend: Raw Backend value from YAML (typically dict or None)
        
    Returns:
        Canonical dict: {"Name": name, "Config": config}
        
    Raises:
        Calls printExit (which raises SystemExit) on any validation failure
    """
    if raw_backend is None:
        # Default backend
        return {"Name": "tensile", "Config": {}}
    
    if not isinstance(raw_backend, dict):
        printExit(
            "Invalid backend configuration: 'Backend' must be a dictionary with "
            "key 'Name' and optional key 'Config'."
        )
    
    if "Name" not in raw_backend:
        printExit(
            "Invalid backend configuration: 'Backend' must contain key 'Name'."
        )
    
    name = raw_backend.get("Name")
    if not isinstance(name, str) or not name.strip():
        printExit(
            "Invalid backend configuration: 'Backend.Name' must be a non-empty string."
        )
    
    config = raw_backend.get("Config", {})
    if config is None:
        config = {}
    if not isinstance(config, dict):
        printExit(
            "Invalid backend configuration: 'Backend.Config' must be a dictionary."
        )
    
    return {
        "Name": name.strip().lower(),
        "Config": config,
    }
