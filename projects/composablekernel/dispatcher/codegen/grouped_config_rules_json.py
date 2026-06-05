#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
"json" rule set for Grouped Convolution Tile Configurations.

Unlike the other rule sets, this rule set loads the kernel configurations 
directly from the reference JSON config files in ``configs/grouped_conv/<variant>/profiler/*.json``.

Selected via ``get_default_configs(rule_set="json")``.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow importing sibling codegen modules
# ---------------------------------------------------------------------------
_CODEGEN_DIR = Path(__file__).parent.resolve()
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

log = logging.getLogger(__name__)

# Root directory holding the reference JSON config files.
_CONFIG_ROOT = _CODEGEN_DIR / "configs" / "grouped_conv"

# Config-set subdirectory to read from (the full reference set).
_CONFIG_SUBSET = "profiler"


# =============================================================================
# JSON loaders
# =============================================================================
def load_depthwise_configs_from_json(
    data: dict,
    arch: str = "gfx942",
    instance_id: Optional[int] = None,
):
    """Load depthwise convolution configs from parsed JSON data.

    Args:
        data: Parsed JSON config data
        arch: Target GPU architecture
        instance_id: If specified, load only the instance with this ID

    Returns:
        List of DepthwiseConvKernelConfig objects
    """
    from unified_grouped_conv_codegen import DepthwiseConvKernelConfig

    ndim_spatial = data["ndim_spatial"]
    layout = data["layout"]
    datatype = data["datatype"]

    instances = data["instances"]
    if instance_id is not None:
        instances = [inst for inst in instances if inst["id"] == instance_id]
        if not instances:
            raise ValueError(f"Instance ID {instance_id} not found in depthwise config")

    configs = []
    for inst in instances:
        config = DepthwiseConvKernelConfig(
            tile_h=inst["tile_h"],
            tile_w=inst["tile_w"],
            filt=inst["filt"],
            str_h=inst["str_h"],
            str_w=inst["str_w"],
            pad_h=inst["pad_h"],
            pad_w=inst["pad_w"],
            nbatch=inst["nbatch"],
            sub_h=inst["sub_h"],
            sub_w=inst["sub_w"],
            in_vec=inst["in_vec"],
            out_vec=inst["out_vec"],
            ndim_spatial=ndim_spatial,
            arch=arch,
            layout=layout,
            datatype=datatype,
        )
        configs.append(config)

    log.info(
        f"Loaded {len(configs)} depthwise configs "
        f"(layout={layout}, dtype={datatype})"
    )
    return configs


def load_configs_from_json(
    config_path: Path,
    arch: str = "gfx942",
    instance_id: Optional[int] = None,
):
    """Load kernel configurations from a JSON config file.

    Args:
        config_path: Path to JSON config file
        arch: Target GPU architecture
        instance_id: If specified, load only the instance with this ID

    Returns:
        List of GroupedConvKernelConfig objects
    """
    from unified_grouped_conv_codegen import (
        GroupedConvVariant,
        GroupedConvTraitConfig,
        GroupedConvKernelConfig,
        TileConfig,
        StreamKConfig,
        StreamKReductionStrategy,
    )

    with open(config_path, "r") as f:
        data = json.load(f)

    variant_map = {
        "forward": GroupedConvVariant.FORWARD,
        "fwd": GroupedConvVariant.FORWARD,
        "forward_depthwise": GroupedConvVariant.FORWARD_DEPTHWISE,
        "bwd_data": GroupedConvVariant.BACKWARD_DATA,
        "bwd_weight": GroupedConvVariant.BACKWARD_WEIGHT,
    }
    variant = variant_map.get(data["variant"])
    if variant is None:
        raise ValueError(f"Unknown variant: {data['variant']}")

    if variant == GroupedConvVariant.FORWARD_DEPTHWISE:
        return load_depthwise_configs_from_json(data, arch, instance_id)

    ndim_spatial = data["ndim_spatial"]
    layout = data["layout"]
    datatype = data["datatype"]

    instances = data["instances"]
    if instance_id is not None:
        instances = [inst for inst in instances if inst["id"] == instance_id]
        if not instances:
            raise ValueError(f"Instance ID {instance_id} not found in {config_path}")

    configs = []
    for inst in instances:
        # Map specialization to pipeline constraints
        # Specializations like filter1x1_stride1_pad0 don't change the pipeline config
        # but are tracked in the trait for kernel naming and runtime checks

        trait = GroupedConvTraitConfig(
            pipeline=inst["pipeline"],
            scheduler=inst["scheduler"],
            epilogue=inst["epilogue"],
            pad_m=True,
            pad_n=True,
            pad_k=True,
            double_smem_buffer=inst.get("double_smem_buffer", False),
            num_groups_to_merge=inst.get("num_groups_to_merge", 1),
            split_image=inst.get("split_image", False),
            explicit_gemm=inst.get("explicit_gemm", False),
            two_stage=inst.get("two_stage", False),
            specialization=inst.get("specialization", "default"),
            streamk_config=StreamKConfig(
                streamk_enabled=inst.get("streamk_enabled", False),
                strategy=StreamKReductionStrategy(inst.get("streamk_reduction_strategy", "TREE")),
                streamk_persistent=inst.get("streamk_persistent", False)
            ) if inst.get("streamk_enabled", False) else StreamKConfig()
        )

        # compv2/basic_v2 (GemmPipelineAGmemBGmemCRegV2) is not compatible with
        # CK Tile's GroupedConvolutionBackwardWeightKernel. The builder maps
        # PipelineVersion::V2 to GemmPipelineAgBgCrMem (i.e. "mem"), not to
        # GemmPipelineAGmemBGmemCRegV2. Skip if any config somehow has compv2.
        if variant == GroupedConvVariant.BACKWARD_WEIGHT and trait.pipeline in ("compv2", "basic_v2"):
            log.info(f"Skipping instance {inst['id']}: compv2/basic_v2 pipeline not compatible with CK Tile bwd_weight")
            continue

        config = GroupedConvKernelConfig(
            tile=TileConfig(
                tile_m=inst["tile_m"],
                tile_n=inst["tile_n"],
                tile_k=inst["tile_k"],
                warp_m=inst["warp_m"],
                warp_n=inst["warp_n"],
                warp_k=inst["warp_k"],
                warp_tile_m=inst["warp_tile_m"],
                warp_tile_n=inst["warp_tile_n"],
                warp_tile_k=inst["warp_tile_k"],
            ),
            trait=trait,
            variant=variant,
            ndim_spatial=ndim_spatial,
            arch=arch,
            layout=layout,
            vector_size_a=inst["vector_size_a"],
            vector_size_b=inst["vector_size_b"],
            vector_size_c=inst["vector_size_c"],
            num_wave_groups=inst.get("num_wave_groups", 1),
        )
        # Tag each GEMM config with its file's concrete datatype so that
        # generate_all emits the kernel only for that datatype (otherwise an
        # untagged config is compiled for every datatype).
        config.datatype = datatype
        configs.append(config)

    log.info(
        f"Loaded {len(configs)} configs from {config_path} "
        f"(variant={data['variant']}, layout={layout}, dtype={datatype})"
    )
    return configs


# =============================================================================
# Unified rule-set entry point
# =============================================================================
def get_configs(
    arch: str,
    variants: List,
    ndims: List[int],
    datatypes: List[str],
) -> List:
    """Build all configs for the "json" rule set by loading them from the
    reference JSON config files.

    Unified rule-set entry point used by
    ``unified_grouped_conv_codegen.get_default_configs``. For each requested
    variant, every ``configs/grouped_conv/<variant>/profiler/*.json`` file is
    loaded (forward includes the depthwise ``ngchw`` files), filtered to the
    requested ``ndims`` and ``datatypes``.
    """
    from unified_grouped_conv_codegen import GroupedConvVariant

    # Map variant enum -> on-disk config directory name.
    variant_dirs = {
        GroupedConvVariant.FORWARD: "forward",
        GroupedConvVariant.FORWARD_DEPTHWISE: "forward",
        GroupedConvVariant.BACKWARD_DATA: "backward_data",
        GroupedConvVariant.BACKWARD_WEIGHT: "backward_weight",
    }

    requested_dirs = []
    for v in variants:
        d = variant_dirs.get(v)
        if d is not None and d not in requested_dirs:
            requested_dirs.append(d)

    datatype_set = set(datatypes) if datatypes else None
    ndim_set = set(ndims) if ndims else None

    configs: List = []
    for d in requested_dirs:
        config_dir = _CONFIG_ROOT / d / _CONFIG_SUBSET
        if not config_dir.is_dir():
            log.warning(f"JSON config dir not found: {config_dir}")
            continue
        for json_path in sorted(config_dir.glob("*.json")):
            with open(json_path, "r") as f:
                meta = json.load(f)
            if datatype_set is not None and meta.get("datatype") not in datatype_set:
                continue
            if ndim_set is not None and meta.get("ndim_spatial") not in ndim_set:
                continue
            configs.extend(load_configs_from_json(json_path, arch=arch))

    log.info(f"json rule set: loaded {len(configs)} configs from {_CONFIG_ROOT}")
    return configs
