# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.register``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/register.cpp``.
``RegisterPool`` is a stateful allocator; logicalIR uses ``StinkyRegister``
plus its own internal allocation (see ``StinkyRegister`` / ``RegType`` in
``shared/stinkytofu/python_module/src/python_bindings.cpp``) but it is NOT
a 1:1 replacement for ``RegisterPool``. Dummies only.

Note: nanobind exposes ``RegisterPool`` as the primary class and
``RegisterPool::Register`` as an inner class. Python-side consumers only
reference ``RegisterPool`` by name; the inner ``Register`` is accessed as
``RegisterPool.Register``. We attach it as a class attribute to preserve
that shape.
"""

from __future__ import annotations

from ._dummy import make_dummy_class

_P = "rocisa.register"


RegisterPool = make_dummy_class(f"{_P}.RegisterPool")
RegisterPool.Register = make_dummy_class(f"{_P}.RegisterPool.Register")
