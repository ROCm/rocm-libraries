# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph validation for supported operations."""

from typing import Any, Dict, List, Optional, Set, Tuple

from ..common.exceptions import GraphLoadError

_REQUIRED_TOP_LEVEL_KEYS: List[str] = ["tensors", "nodes"]

# Supported operation types - accepts any hipDNN operation
SUPPORTED_NODE_TYPES: Set[str] = {
    "ConvolutionFwdAttributes",
    "ConvolutionBwdAttributes",
    "ConvolutionWrwAttributes",
    "MatmulAttributes",
    "PointwiseAttributes",
    "BatchnormAttributes",
    "BatchnormBackwardAttributes",
    "BatchnormInferenceAttributes",
}

# Stat tensor keys per batchnorm node type — all must share the same data_type.
# Keys are optional; only those present in the node are compared.
_STAT_TENSOR_KEYS: Dict[str, List[Tuple[str, str]]] = {
    "BatchnormAttributes": [
        ("outputs", "mean_tensor_uid"),
        ("outputs", "inv_variance_tensor_uid"),
        ("inputs", "running_mean_tensor_uid"),
        ("inputs", "running_var_tensor_uid"),
    ],
    "BatchnormInferenceAttributes": [
        ("inputs", "mean_tensor_uid"),
        ("inputs", "inv_variance_tensor_uid"),
    ],
    "BatchnormBackwardAttributes": [
        ("inputs", "mean_tensor_uid"),
        ("inputs", "inv_variance_tensor_uid"),
    ],
}

# Required keys per node type, as (section, key) tuples.
# section is "inputs", "outputs", "parameters", or None (top-level node).
# Derived from the C++ from_json implementations in data_sdk/utilities/json/.
_REQUIRED_NODE_FIELDS: Dict[str, List[Tuple[str, str]]] = {
    "ConvolutionFwdAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "w_tensor_uid"),
        ("outputs", "y_tensor_uid"),
        ("parameters", "pre_padding"),
        ("parameters", "post_padding"),
        ("parameters", "stride"),
        ("parameters", "dilation"),
        ("parameters", "conv_mode"),
    ],
    "ConvolutionBwdAttributes": [
        ("inputs", "dy_tensor_uid"),
        ("inputs", "w_tensor_uid"),
        ("outputs", "dx_tensor_uid"),
        ("parameters", "pre_padding"),
        ("parameters", "post_padding"),
        ("parameters", "stride"),
        ("parameters", "dilation"),
        ("parameters", "conv_mode"),
    ],
    "ConvolutionWrwAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "dy_tensor_uid"),
        ("outputs", "dw_tensor_uid"),
        ("parameters", "pre_padding"),
        ("parameters", "post_padding"),
        ("parameters", "stride"),
        ("parameters", "dilation"),
        ("parameters", "conv_mode"),
    ],
    "MatmulAttributes": [
        ("inputs", "a_tensor_uid"),
        ("inputs", "b_tensor_uid"),
        ("outputs", "c_tensor_uid"),
    ],
    "BatchnormInferenceAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "mean_tensor_uid"),
        ("inputs", "inv_variance_tensor_uid"),
        ("inputs", "scale_tensor_uid"),
        ("inputs", "bias_tensor_uid"),
        ("outputs", "y_tensor_uid"),
    ],
    "BatchnormAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "scale_tensor_uid"),
        ("inputs", "bias_tensor_uid"),
        ("inputs", "epsilon_tensor_uid"),
        ("outputs", "y_tensor_uid"),
    ],
    "BatchnormBackwardAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "dy_tensor_uid"),
        ("inputs", "mean_tensor_uid"),
        ("inputs", "inv_variance_tensor_uid"),
        ("inputs", "scale_tensor_uid"),
        ("outputs", "dx_tensor_uid"),
        ("outputs", "dscale_tensor_uid"),
        ("outputs", "dbias_tensor_uid"),
    ],
}


class GraphValidator:
    """Validates that a graph contains valid operations.

    Accepts any hipDNN operation type that can be deserialized and executed.
    """

    def __init__(self, supported_types: Optional[Set[str]] = None) -> None:
        """Initialize validator with supported operation types.

        Args:
            supported_types: Set of node type names that are supported.
                           Defaults to SUPPORTED_NODE_TYPES.
        """
        self._supported_types = (
            supported_types
            if supported_types is not None
            else SUPPORTED_NODE_TYPES.copy()
        )

    def validate(self, graph_json: Dict[str, Any]) -> None:
        """Validate top-level graph structure and node required fields.

        Args:
            graph_json: The parsed graph JSON dictionary.

        Raises:
            GraphLoadError: If required top-level keys are missing, the graph
                            contains no operation nodes, any node is missing
                            required fields, or a node references an unknown
                            tensor UID.
        """
        missing_keys = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in graph_json]
        if missing_keys:
            raise GraphLoadError(
                f"Graph is missing required top-level keys: {', '.join(missing_keys)}"
            )

        nodes = graph_json["nodes"]
        if not nodes:
            raise GraphLoadError("Graph contains no operation nodes")

        uid_to_dtype: Dict[int, str] = {
            t["uid"]: t.get("data_type", "float")
            for t in graph_json["tensors"]
            if "uid" in t
        }

        for node in nodes:
            self._validate_node_fields(node, uid_to_dtype)

    def _validate_node_fields(
        self, node: Dict[str, Any], uid_to_dtype: Dict[int, str]
    ) -> None:
        """Check that a node has all required fields for its type.

        Args:
            node: A single node dictionary from graph JSON.
            uid_to_dtype: Map of tensor UID to data_type string.

        Raises:
            GraphLoadError: If a required field is missing, references an
                unknown tensor UID, or batchnorm stat tensors have mismatched
                data types.
        """
        node_type = node.get("type", "")
        node_name = node.get("name", node_type)

        if node_type not in self._supported_types:
            raise GraphLoadError(
                f"Node '{node_name}' has unknown type '{node_type}'. "
                f"Supported types: {', '.join(sorted(self._supported_types))}"
            )

        required = _REQUIRED_NODE_FIELDS.get(node_type)
        if required is None:
            # Supported type with no per-field requirements defined above (e.g.
            # PointwiseAttributes).  Field validation is intentionally deferred
            # to the hipDNN backend for these types.
            return

        known_uids = uid_to_dtype.keys()
        missing = []
        bad_uids = []
        for section, key in required:
            container = node.get(section, {})
            if key not in container:
                missing.append(f"{section}.{key}")
            else:
                uid = container[key]
                if isinstance(uid, int) and uid not in known_uids:
                    bad_uids.append(f"{section}.{key}={uid}")

        if missing:
            raise GraphLoadError(
                f"Node '{node_name}' ({node_type}) is missing required fields: "
                + ", ".join(missing)
            )

        if bad_uids:
            raise GraphLoadError(
                f"Node '{node_name}' ({node_type}) references unknown tensor UID(s): "
                + ", ".join(bad_uids)
            )

        self._validate_stat_tensor_types(node, node_name, node_type, uid_to_dtype)

    def _validate_stat_tensor_types(
        self,
        node: Dict[str, Any],
        node_name: str,
        node_type: str,
        uid_to_dtype: Dict[int, str],
    ) -> None:
        """Check that all batchnorm stat tensors share the same data type.

        Args:
            node: A single node dictionary from graph JSON.
            node_name: Display name for error messages.
            node_type: Node type string.
            uid_to_dtype: Map of tensor UID to data_type string.

        Raises:
            GraphLoadError: If stat tensors have mismatched data types.
        """
        stat_keys = _STAT_TENSOR_KEYS.get(node_type)
        if stat_keys is None:
            return

        # Collect (key, dtype) for each stat tensor that is present in this node.
        found: List[Tuple[str, str]] = []
        for section, key in stat_keys:
            uid = node.get(section, {}).get(key)
            if isinstance(uid, int) and uid in uid_to_dtype:
                found.append((key, uid_to_dtype[uid]))

        if len(found) < 2:
            return

        ref_key, ref_dtype = found[0]
        for key, dtype in found[1:]:
            if dtype != ref_dtype:
                raise GraphLoadError(
                    f"Node '{node_name}' ({node_type}) stat tensor type mismatch: "
                    f"'{ref_key}' is '{ref_dtype}' but '{key}' is '{dtype}'. "
                    "All stat tensors (mean, inv_variance, running) must have the same data type."
                )

    def get_supported_types(self) -> Set[str]:
        """Get the set of supported operation types.

        Returns:
            Set of supported node type names.
        """
        return self._supported_types.copy()
