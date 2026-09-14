# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python port of the runtime's ``$kernel.<field>`` define substituter.

The authority is C++:
``plugin_sdk/include/hipdnn_plugin_sdk/ingestor/KernelDefineSubstitution.hpp``,
tested by ``plugin_sdk/tests/ingestor/TestKernelDefineSubstitution.cpp``. This
module exists so the GENERATOR refuses a ``defines`` value the loader would
refuse, at the moment it is authored, instead of shipping a bundle that drops
its pack with one ``LOG_ERROR`` on someone else's machine. Every rule below is
a mirror of a rule there, and ``tests/test_kernel_defines.py`` re-runs the C++
test table against this implementation so the two can be diffed by a reader.

**Literal token replacement, not an expression language.** Operators,
arithmetic, comparisons, conditionals, calls, defaults, nesting and ``$graph``
are errors rather than half-working substitutions -- ``"$kernel.a + 1"`` must
not quietly become ``"2 + 1"`` in a ``-D`` flag. Anything derived or conditional
belongs in the pack's dispatch handler, which already builds
``KernelCompileOptions`` by hand.

Two entry points, because the two checks see different inputs:

* :func:`validate_kernel_define_template` asks the KMD "could this ever
  resolve?". It is the generator-time check, and it mirrors the loader's
  set-resolution check -- schema-level, because a kernel may legally omit a
  field the KMD defaults.
* :func:`substitute_kernel_define` resolves against one kernel's COMPLETED
  metadata, mirroring what the runtime does at ``prepare()``.
"""

#: Prefix of every bound token. Nothing else may follow a ``$``.
KERNEL_DEFINE_TOKEN_PREFIX = "$kernel."

#: Characters that only ever appear in a define value because someone is
#: writing an expression: arithmetic, comparison, logic, a conditional, a call,
#: or a default. They are refused in a value that binds a token, which is the
#: only place they could be mistaken for something this substituter evaluates.
#: A define with no token is opaque text and is passed through untouched, so
#: ``-DLIMIT=-1`` remains authorable.
EXPRESSION_OPERATOR_CHARS = frozenset("+-*/%=<>!&|^~?:()")

#: The KMD types that have a pinned spelling as ``-D`` flag text. ``float`` and
#: ``int_list`` are absent deliberately -- see :func:`render_metadata_value`.
RENDERABLE_KMD_TYPES: tuple[str, ...] = ("bool", "int", "string")


class KernelDefineError(Exception):
    """A ``defines`` value the runtime substituter would refuse.

    Carries the same sentence the C++ side puts in its ``error`` out-parameter,
    so an author who hits the check here and the check there reads one message.
    """


def _is_identifier_start(character: str) -> bool:
    return character.isascii() and (character.isalpha() or character == "_")


def _is_identifier_char(character: str) -> bool:
    return _is_identifier_start(character) or (
        character.isascii() and character.isdigit()
    )


def _scan_kernel_define_template(template_text: str, on_field) -> str:
    """Walk ``template_text`` once, handing each ``$kernel.<field>`` token to
    ``on_field``, which returns the field's replacement text or raises
    :class:`KernelDefineError`. Literal runs are copied verbatim."""
    if "$" not in template_text:
        # Byte-identical passthrough, and the only path a define without a
        # token takes: an operator character here is ordinary text, not an
        # expression.
        return template_text

    out: list[str] = []
    position = 0
    length = len(template_text)
    while position < length:
        character = template_text[position]
        if character in EXPRESSION_OPERATOR_CHARS:
            raise KernelDefineError(
                f"value '{template_text}' uses '{character}', but a bound define "
                f"is literal token replacement -- it evaluates no operators, "
                f"comparisons, conditionals, calls or defaults. Put derived "
                f"values in the pack's dispatch handler"
            )
        if character != "$":
            out.append(character)
            position += 1
            continue

        if not template_text.startswith(KERNEL_DEFINE_TOKEN_PREFIX, position):
            raise KernelDefineError(
                f"value '{template_text}' contains a '$' that does not begin a "
                f"'{KERNEL_DEFINE_TOKEN_PREFIX}<field>' token; no other binding "
                f"source exists"
            )

        name_begin = position + len(KERNEL_DEFINE_TOKEN_PREFIX)
        name_end = name_begin
        while name_end < length and _is_identifier_char(template_text[name_end]):
            name_end += 1
        if name_end == name_begin or not _is_identifier_start(
            template_text[name_begin]
        ):
            raise KernelDefineError(
                f"value '{template_text}' has a '{KERNEL_DEFINE_TOKEN_PREFIX}' "
                f"with no field name after it"
            )

        out.append(on_field(template_text[name_begin:name_end]))
        # Single pass: a replacement that itself contains a '$' is literal text,
        # never rescanned. Nesting is not a feature.
        position = name_end
    return "".join(out)


def render_metadata_value(value, field_name: str) -> str:
    """One metadata value as the text of a ``-D`` flag.

    The spelling of each accepted type is pinned, because two kernel variants
    that differ only in a bound field and render the same flag text compile to
    the same binary and silently become one kernel.

    ``float`` and ``int_list`` are refused: there is no single spelling of
    either. ``1.0`` and ``1`` are different types in device code and C++'s
    ``std::to_chars(1.0)`` and Python's ``repr(1.0)`` disagree about which one
    to emit; a list has no separator that is right for every macro.
    """
    # bool before int: Python's bool IS an int, and a `flag` rendering as `True`
    # here and `1` in the runtime is exactly the divergence this port prevents.
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        raise KernelDefineError(
            f"metadata field '{field_name}' is a float, which cannot be bound "
            f"into a define: '1' and '1.0' are different types in device code "
            f"and no spelling is right for both. Emit it from the pack's "
            f"dispatch handler instead"
        )
    if isinstance(value, (list, tuple)):
        raise KernelDefineError(
            f"metadata field '{field_name}' is an int_list, which cannot be "
            f"bound into a define: no separator is right for every macro. Emit "
            f"it from the pack's dispatch handler instead"
        )
    raise KernelDefineError(
        f"metadata field '{field_name}' has an unknown metadata type"
    )


def _render_declared_type(kmd_type: str, field_name: str) -> None:
    """The schema-level half of :func:`render_metadata_value`: reject a
    DECLARED type nothing can render, reusing the value-level wording so the
    generator-time message and the runtime message agree."""
    if kmd_type == "float":
        render_metadata_value(0.0, field_name)
    elif kmd_type == "int_list":
        render_metadata_value([], field_name)
    elif kmd_type not in RENDERABLE_KMD_TYPES:
        render_metadata_value(None, field_name)


def substitute_kernel_define(template_text: str, metadata: dict) -> str:
    """Resolve every ``$kernel.<field>`` against ``metadata``, which must be the
    kernel's COMPLETED metadata -- KMD defaults already filled in, so a field
    the descriptor omits still resolves.

    Raises :class:`KernelDefineError` naming what the author must change.
    """

    def on_field(field_name: str) -> str:
        if field_name not in metadata:
            raise KernelDefineError(
                f"value '{template_text}' names metadata field '{field_name}', "
                f"which this kernel does not carry"
            )
        return render_metadata_value(metadata[field_name], field_name)

    return _scan_kernel_define_template(template_text, on_field)


def validate_kernel_define_template(
    template_text: str, field_types: dict, schema_name: str
) -> None:
    """Check that ``template_text`` could resolve for ANY kernel of an engine
    whose KMD declares ``field_types`` (``{name: kmd type}``): every token names
    a declared field, and every such field has a renderable type.

    Deliberately schema-level, not value-level -- it runs where a kernel may
    legally omit a field the KMD defaults, so asking for the value would reject
    a legal descriptor.

    Raises :class:`KernelDefineError`.
    """

    def on_field(field_name: str) -> str:
        if field_name not in field_types:
            raise KernelDefineError(
                f"value '{template_text}' names metadata field '{field_name}', "
                f"which metadata schema '{schema_name}' does not declare"
            )
        _render_declared_type(field_types[field_name], field_name)
        return ""

    _scan_kernel_define_template(template_text, on_field)
