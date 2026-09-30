################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.ensurePath``.

These pin the exact behavior of the ``except OSError`` branch: when
``os.makedirs`` raises a (non-FileExists) OSError, ``ensurePath`` re-raises a
plain ``OSError`` carrying the message ``Failed to create directory "<path>" ``.
"""

import os

import pytest

from Tensile.Common.Utilities import ensurePath

pytestmark = pytest.mark.unit


def _path_that_triggers_oserror(tmp_path):
    """Return a path whose creation raises a (non-FileExists) OSError.

    Creating a directory underneath a regular file raises
    NotADirectoryError (a subclass of OSError, not FileExistsError), which
    drives ``ensurePath`` into its ``except OSError`` re-raise branch.
    """
    f = tmp_path / "afile"
    f.write_text("")
    return os.path.join(str(f), "sub")


def test_ensurepath_oserror_reraises_exact_message_and_type(tmp_path):
    """Kills mutmut_2 (None message), mutmut_4/5/6 (message text changes),
    mutmut_3 (`/` -> TypeError instead of OSError).

    Pins: a plain OSError is raised whose message is exactly
    'Failed to create directory "<path>" ' (note: capital F and trailing
    space, with the path interpolated via %s).
    """
    bad = _path_that_triggers_oserror(tmp_path)
    with pytest.raises(OSError) as excinfo:
        ensurePath(bad)
    # type must be exactly OSError (mutmut_3 raises TypeError, mutmut_6's
    # invalid %S conversion raises ValueError -- neither is caught here).
    assert type(excinfo.value) is OSError
    assert str(excinfo.value) == 'Failed to create directory "%s" ' % (bad)


def test_ensurepath_oserror_message_is_nonempty_and_capitalized(tmp_path):
    """Reinforces mutmut_2 (None message), mutmut_4 (XX wrapper),
    mutmut_5 (lowercase 'failed'), mutmut_6 (uppercased text).

    Pins the literal prefix/suffix of the original message.
    """
    bad = _path_that_triggers_oserror(tmp_path)
    with pytest.raises(OSError) as excinfo:
        ensurePath(bad)
    msg = str(excinfo.value)
    assert msg.startswith('Failed to create directory "')
    assert msg.endswith('" ')
    assert bad in msg
