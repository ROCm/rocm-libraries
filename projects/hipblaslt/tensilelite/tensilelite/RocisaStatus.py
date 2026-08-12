# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Detect how the ``rocisa`` package is installed.

This lives inside the ``Tensile`` package (rather than in the tensilelite-root
``tasks.py``) so it ships with the installed test artifacts and can be imported
without pulling in ``invoke`` — see
``Tensile/Tests/unit/test_rocisa_install_status.py``.
"""


def _rocisa_install_status():
    """Return the install status of rocisa.

    Returns one of:
        "absent"       – not installed at all
        "editable"     – installed as editable (pip install -e)
        "non-editable" – installed normally (pip install, wheel, tox)

    Reads PEP 610 direct_url.json metadata; it does NOT import rocisa, so the
    import-time staleness check is never triggered here.
    """
    import json
    from importlib import metadata

    try:
        raw = metadata.distribution("rocisa").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return "absent"
    if not raw:
        return "non-editable"
    try:
        if json.loads(raw).get("dir_info", {}).get("editable"):
            return "editable"
    except (ValueError, AttributeError):
        pass
    return "non-editable"
