# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch CUDA tensor management for graph execution."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..graph.tensor_info import TensorInfo

# Map data type strings to torch dtypes
TORCH_DTYPE_MAP = {
    "float": torch.float32,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "double": torch.float64,
    "int8": torch.int8,
    "int32": torch.int32,
    "uint8": torch.uint8,
}

# Map data type strings to numpy dtypes (for random generation)
NUMPY_DTYPE_MAP = {
    "float": np.float32,
    "half": np.float16,
    "bfloat16": np.float32,  # numpy doesn't have bfloat16
    "double": np.float64,
    "int8": np.int8,
    "int32": np.int32,
    "uint8": np.uint8,
}


class PyTorchCudaBufferManager:
    """Manages PyTorch CUDA tensor allocation for graph execution.

    This class handles:
    - Allocating CUDA tensors for all non-virtual tensors
    - Filling input tensors with random data
    - Zeroing output tensors
    - Providing tensors for graph execution
    """

    def __init__(
        self,
        tensor_infos: List[TensorInfo],
        device: str = "cuda:0",
        graph_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize buffer manager with tensor metadata.

        Args:
            tensor_infos: List of TensorInfo objects describing tensors.
            device: CUDA device to use (e.g., "cuda:0").
            graph_json: Optional graph JSON used to derive dependent synthetic
                inputs, such as standalone SDPA backward O/LSE tensors.
        """
        self._tensor_infos = tensor_infos
        self._tensor_info_by_uid = {tensor.uid: tensor for tensor in tensor_infos}
        self._device = torch.device(device)
        self._graph_json = graph_json or {}
        self._tensors: Dict[int, torch.Tensor] = {}
        self._host_data: Dict[int, np.ndarray] = {}

    def allocate_all(self) -> None:
        """Allocate CUDA tensors for all non-virtual tensors."""
        for tensor_info in self._tensor_infos:
            if tensor_info.is_virtual:
                continue

            dtype = TORCH_DTYPE_MAP.get(tensor_info.data_type.lower(), torch.float32)
            if tensor_info.is_pass_by_value:
                value = torch.tensor(
                    [tensor_info.value], dtype=dtype, device=self._device
                )
                self._tensors[tensor_info.uid] = value
                self._host_data[tensor_info.uid] = np.asarray([tensor_info.value])
                continue

            if tensor_info.strides:
                tensor = torch.empty_strided(
                    tensor_info.dims,
                    tensor_info.strides,
                    dtype=dtype,
                    device=self._device,
                )
            else:
                tensor = torch.empty(
                    tensor_info.dims,
                    dtype=dtype,
                    device=self._device,
                )
            self._tensors[tensor_info.uid] = tensor

    @staticmethod
    def _host_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Copy a tensor to numpy using the validation representation."""
        host = tensor.detach().cpu()
        if host.dtype == torch.bfloat16:
            return host.to(dtype=torch.float32).numpy()
        return host.numpy()

    @staticmethod
    def _node_uid(node: Dict[str, Any], key: str) -> Optional[int]:
        for section_name in ("inputs", "outputs"):
            value = (node.get(section_name) or {}).get(key)
            if value is not None:
                return int(value)
        return None

    @staticmethod
    def _node_param(node: Dict[str, Any], key: str, default: Any = None) -> Any:
        for section_name in ("parameters", "attributes", "inputs", "outputs"):
            section = node.get(section_name) or {}
            if key in section:
                return section[key]
        return default

    def _set_tensor_from_value(self, uid: int, value: torch.Tensor) -> None:
        tensor = self._tensors.get(uid)
        if tensor is None:
            return
        tensor.copy_(value.to(dtype=tensor.dtype, device=tensor.device))
        self._host_data[uid] = self._host_numpy(tensor)

    def _fill_sdpa_backward_prerequisites(self) -> None:
        """Populate standalone SDPA backward O/LSE inputs from Q/K/V tensors."""
        for node in self._graph_json.get("nodes") or []:
            if node.get("type") != "SdpaBackwardAttributes":
                continue

            q_uid = self._node_uid(node, "q_tensor_uid")
            k_uid = self._node_uid(node, "k_tensor_uid")
            v_uid = self._node_uid(node, "v_tensor_uid")
            o_uid = self._node_uid(node, "o_tensor_uid")
            stats_uid = self._node_uid(node, "stats_tensor_uid")
            if None in (q_uid, k_uid, v_uid, o_uid, stats_uid):
                continue

            o_info = self._tensor_info_by_uid.get(o_uid)
            stats_info = self._tensor_info_by_uid.get(stats_uid)
            if o_info is None or stats_info is None:
                continue
            if o_info.is_output or stats_info.is_output:
                continue

            q_base = self._tensors.get(q_uid)
            k_base = self._tensors.get(k_uid)
            v_base = self._tensors.get(v_uid)
            if q_base is None or k_base is None or v_base is None:
                continue

            dropout_probability = self._node_param(node, "dropout_probability", 0.0)
            if float(dropout_probability or 0.0) != 0.0:
                continue

            with torch.no_grad():
                q = q_base.to(dtype=torch.float32)
                k = k_base.to(dtype=torch.float32)
                v = v_base.to(dtype=torch.float32)

                q_heads = int(q.shape[-3])
                kv_heads = int(k.shape[-3])
                if q_heads != kv_heads:
                    if kv_heads == 0 or q_heads % kv_heads != 0:
                        raise ValueError(
                            f"Unsupported SDPA GQA head counts: q_heads={q_heads}, "
                            f"kv_heads={kv_heads}"
                        )
                    repeat = q_heads // kv_heads
                    k = k.repeat_interleave(repeat, dim=-3)
                    v = v.repeat_interleave(repeat, dim=-3)

                scale_uid = self._node_uid(node, "scale_tensor_uid")
                if scale_uid is not None and scale_uid in self._tensors:
                    scale_value = float(
                        self._tensors[scale_uid].detach().reshape(-1)[0].item()
                    )
                else:
                    scale = self._node_param(node, "attn_scale_value", None)
                    scale_value = (
                        1.0 / (float(q.shape[-1]) ** 0.5)
                        if scale is None
                        else float(scale)
                    )

                scores = torch.matmul(q, k.transpose(-2, -1)) * scale_value

                mask_uid = self._node_uid(node, "attn_mask_tensor_uid")
                if mask_uid is not None and mask_uid in self._tensors:
                    scores = scores + self._tensors[mask_uid].to(dtype=torch.float32)

                if bool(self._node_param(node, "causal_mask", False)):
                    length_q = scores.shape[-2]
                    length_k = scores.shape[-1]
                    causal = torch.ones(
                        length_q,
                        length_k,
                        dtype=torch.bool,
                        device=scores.device,
                    ).tril()
                    scores = scores.masked_fill(~causal, float("-inf"))

                stats = torch.logsumexp(scores, dim=-1)
                probs = torch.exp(scores - stats.unsqueeze(-1))
                output = torch.matmul(probs, v)

                self._set_tensor_from_value(o_uid, output)
                self._set_tensor_from_value(stats_uid, stats)

    def fill_inputs_random(self, seed: Optional[int] = None) -> None:
        """Fill input tensor buffers with random data.

        Args:
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        for tensor_info in self._tensor_infos:
            if tensor_info.is_output or tensor_info.is_virtual:
                continue

            if tensor_info.is_pass_by_value:
                continue

            tensor = self._tensors.get(tensor_info.uid)
            if tensor is None:
                continue

            # Generate random data on CPU (for reproducibility with numpy)
            np_dtype = NUMPY_DTYPE_MAP.get(tensor_info.data_type.lower(), np.float32)
            data = np.random.uniform(0.0, 1.0, tensor_info.dims).astype(np_dtype)

            # Copy to CUDA tensor, then store the logical value that is actually
            # present after device dtype conversion (BF16 is represented as FP32
            # in numpy because numpy has no native bfloat16 dtype).
            tensor.copy_(torch.from_numpy(data))
            self._host_data[tensor_info.uid] = self._host_numpy(tensor)

        self._fill_sdpa_backward_prerequisites()

    def zero_outputs(self) -> None:
        """Zero output tensor buffers."""
        for tensor_info in self._tensor_infos:
            if not tensor_info.is_output:
                continue

            tensor = self._tensors.get(tensor_info.uid)
            if tensor is not None:
                tensor.zero_()

    def get_tensors(self) -> Dict[int, torch.Tensor]:
        """Get mapping of tensor UIDs to CUDA tensors.

        Returns:
            Dictionary mapping tensor UID to torch.Tensor on CUDA.
        """
        return self._tensors

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        """Copy output tensor data from CUDA to numpy array.

        Args:
            uid: Tensor UID.

        Returns:
            Numpy array with output data, or None if tensor not found.
        """
        tensor = self._tensors.get(uid)
        if tensor is None:
            return None

        return self._host_numpy(tensor)

    def get_input_data(self, uid: int) -> Optional[np.ndarray]:
        """Get the host-side input data for a tensor.

        Args:
            uid: Tensor UID.

        Returns:
            Numpy array with input data, or None if not found.
        """
        return self._host_data.get(uid)

    def get_output_tensors(self) -> List[TensorInfo]:
        """Get list of output tensor infos.

        Returns:
            List of TensorInfo objects for output tensors.
        """
        return [ti for ti in self._tensor_infos if ti.is_output]

    def cleanup(self) -> None:
        """Free all tensors."""
        self._tensors.clear()
        self._host_data.clear()
        # Let PyTorch handle CUDA memory cleanup via garbage collection
        torch.cuda.empty_cache()

    def __enter__(self) -> "PyTorchCudaBufferManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup tensors."""
        self.cleanup()
