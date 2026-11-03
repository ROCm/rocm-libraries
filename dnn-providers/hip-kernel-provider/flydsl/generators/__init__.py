# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""AOT kernel-object generators for the FlyDSL provider.

Run a generator as a module from the ``flydsl/`` directory so the shared
helpers resolve as a package:

    python -m generators.gen_rmsnorm --arch gfx1151

See ``../REGEN.md`` for the Python environment these need.
"""
