# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``CustomOpAttributes``.

A plugin's own operation: hipDNN carries the tensors in, the tensors out and
an opaque byte payload to whichever plugin claims ``custom_op_id``, and the
plugin alone says what the op computes. The features come from
``custom_op_attributes.fbs``. The serializer
(``utilities/json/CustomOpAttributes.hpp``) writes the four fields at the
top of the node, an empty list as ``[]``; its reader requires all four. What
hipDNN itself does with the op is the transport, so the features are the
transport's, by group:

- ``inputs`` (exclusive): ``none``, ``one`` or ``multi`` tensors in.
- ``outputs`` (exclusive): the same, out.
- ``data`` (exclusive): ``empty`` or ``present``, the payload.

The API takes any count of tensors, none included (the backend's tests
finalize a custom op with no inputs and with no outputs), and any payload.
No dtype, shape or layout group: those are the plugin's to define, and no
provider or reference executor in this tree implements a custom op (the GPU
reference rejects one, and the CPU reference registers no plan).

The API requires a non-empty ``custom_op_id`` (the frontend's and the
backend's validation) and every uid a tensor of the graph (the backend's
``fromNode``). A graph that breaks these, or whose lists are not of uids and
bytes, raises.
"""

from __future__ import annotations

from support_features import Feature, FeatureError

NODE_TYPE = "CustomOpAttributes"
SCHEMA_FILE = "custom_op_attributes.fbs"
SCHEMA_TABLE = "CustomOpAttributes"

OP_ID = "custom_op_id"
INPUTS, OUTPUTS = "input_tensor_uids", "output_tensor_uids"
DATA = "data"

FEATURES = (
    Feature("inputs:none", (INPUTS,), "no tensor in"),
    Feature("inputs:one", (INPUTS,), "one tensor in"),
    Feature("inputs:multi", (INPUTS,), "several tensors in"),
    Feature("outputs:none", (OUTPUTS,), "no tensor out"),
    Feature("outputs:one", (OUTPUTS,), "one tensor out"),
    Feature("outputs:multi", (OUTPUTS,), "several tensors out"),
    Feature("data:empty", (DATA,), "no payload"),
    Feature("data:present", (DATA,), "an opaque payload for the plugin"),
)

NOT_FEATURES: dict[str, str] = {
    OP_ID: "names a plugin's operation: an open set, each its own op, whose "
    "features only that plugin defines",
}


def arity(count: int) -> str:
    """The value of the ``inputs`` or ``outputs`` group for ``count`` tensors."""
    if count == 0:
        return "none"
    return "one" if count == 1 else "multi"


def checked_uids(node: dict, port: str, tensors_by_uid: dict) -> list[int]:
    """The uids of ``port``, each a tensor of the graph, or raise."""
    uids = node.get(port)
    if not isinstance(uids, list):
        raise FeatureError(f"{port} {uids!r} is not a list of uids")
    for uid in uids:
        # type() rather than isinstance: a bool is an int, and True finds uid 1.
        if type(uid) is not int:
            raise FeatureError(f"{port} uid {uid!r} is not an int")
        if uid not in tensors_by_uid:
            raise FeatureError(f"{port} uid {uid} is not a tensor of the graph")
    return uids


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    op_id = node.get(OP_ID)
    if not isinstance(op_id, str) or not op_id:
        raise FeatureError(f"{OP_ID} {op_id!r} is not a non-empty string")
    inputs = checked_uids(node, INPUTS, tensors_by_uid)
    outputs = checked_uids(node, OUTPUTS, tensors_by_uid)
    data = node.get(DATA)
    if not isinstance(data, list) or any(
        type(byte) is not int or not 0 <= byte <= 255 for byte in data
    ):
        raise FeatureError(f"{DATA} {data!r} is not a list of bytes")

    return {
        f"inputs:{arity(len(inputs))}",
        f"outputs:{arity(len(outputs))}",
        "data:present" if data else "data:empty",
    }
