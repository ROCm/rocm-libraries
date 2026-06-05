
from typing import List

def _classify_config(cfg) -> str:
    """Classify a config into a feature category for stratified test selection.

    The datatype tag is folded into the category so that the stratified subset keeps
    at least one config per datatype per feature category.
    """
    from unified_grouped_conv_codegen import (
        DepthwiseConvKernelConfig,
    )
    dt = getattr(cfg, "datatype", None) or "fp16"
    if isinstance(cfg, DepthwiseConvKernelConfig):
        return f"depthwise:{dt}"
    tr = cfg.trait
    if getattr(tr.streamk_config, "streamk_enabled", False):
        return f"streamk:{dt}"
    if tr.split_image:
        return f"split_image:{dt}"
    if tr.num_groups_to_merge > 1:
        return f"merged_groups:{dt}"
    if tr.two_stage:
        return f"two_stage:{dt}"
    if tr.explicit_gemm:
        return f"explicit_gemm:{dt}"
    return f"regular:{dt}"


def _select_test_configs(configs) -> List:
    """Select ~20% of configs with stratified sampling for test builds.

    Guarantees:
      1. At least 1 config from each feature category.
      2. Every (pipeline, scheduler) combo per variant is represented.

    Selection: every 5th config (indices 4, 9, 14, ...) from each category,
    matching awk 'NR % 5 == 0' convention.
    """
    from collections import defaultdict
    from unified_grouped_conv_codegen import (
        GroupedConvKernelConfig
    )

    categories = defaultdict(list)
    for cfg in configs:
        cat = _classify_config(cfg)
        categories[cat].append(cfg)

    selected_ids = set()

    # Take ~20% from each category (minimum 1)
    for cat, cat_configs in categories.items():
        cat_selected = False
        for i, cfg in enumerate(cat_configs):
            if (i + 1) % 5 == 0:
                selected_ids.add(id(cfg))
                cat_selected = True
        # Ensure minimum 1 per category
        if not cat_selected:
            selected_ids.add(id(cat_configs[0]))

    # Ensure pipeline/scheduler coverage per (variant, datatype) (GEMM only).
    gemm_configs = [c for c in configs if isinstance(c, GroupedConvKernelConfig)]
    variant_combos = defaultdict(set)
    variant_covered = defaultdict(set)
    for c in gemm_configs:
        vkey = (c.variant, getattr(c, "datatype", None) or "fp16")
        combo = (c.trait.pipeline, c.trait.scheduler)
        variant_combos[vkey].add(combo)
        if id(c) in selected_ids:
            variant_covered[vkey].add(combo)

    for vkey, required in variant_combos.items():
        variant, dt = vkey
        missing = required - variant_covered[vkey]
        for combo in missing:
            for c in gemm_configs:
                if c.variant == variant and \
                        (getattr(c, "datatype", None) or "fp16") == dt and \
                        (c.trait.pipeline, c.trait.scheduler) == combo:
                    selected_ids.add(id(c))
                    break

    return [c for c in configs if id(c) in selected_ids]

def get_configs(
    arch: str,
    variants: List,
    ndims: List[int],
    datatypes: List[str]
  ) -> List:
    """Build all available configs for the "test" rule set.

    Unified rule-set entry point used by
    ``unified_grouped_conv_codegen.get_default_configs``. 
    Trims down the profiler config set using the rules defined in ``_select_test_configs``.
    """

    from grouped_config_rules import get_configs as get_profiler_configs

    all_configs = get_profiler_configs(arch, variants, ndims, datatypes)
    test_configs = _select_test_configs(all_configs)
    return test_configs