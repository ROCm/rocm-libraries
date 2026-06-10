# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer.subtile.module_builder — C++ rocisa module-builder facade.

Thin re-export of the ``_tensile_writer.subtile.rocisa_builder`` nanobind
submodule. Unlike the other ``tensile_writer.subtile`` modules, which expose
*data-only* decisions (geometry, offset plans, scheduling, instType), this one
exposes ``ModuleBuilder`` — a C++ class that constructs genuine ``rocisa``
``Module`` objects by driving the ``rocisa`` Python API.

It is the foundation for moving the subtile emit loops
(``InstructionEmitter`` / ``SubtileGREmit`` / ``SubtileLREmit`` /
``SubtileScaleEmit``) into C++ in later slices. Construction is backend-routed
through rocisa (StinkyTofu is not used for gfx950 subtile, which emits via the
rocisa string/Module path).

Boundary contract: ``ModuleBuilder`` owns no writer state. VGPR/SGPR indices,
sgpr/label *names*, and ``writer.states`` scalars are resolved on the Python
side and passed in as ints/strings; the builder only assembles rocisa Items
from those primitive inputs. See
``cpp_migration/docs/rocisa_module_builder_boundary.md``.

Importing this module requires ``rocisa`` to be installed (it is a hard
dependency of the subtile emit path, exactly like ``Kernel.py``).
"""

from tensile_writer import _tensile_writer as _ext

_builder = _ext.subtile.rocisa_builder

ModuleBuilder = _builder.ModuleBuilder

__all__ = [
    "ModuleBuilder",
]
