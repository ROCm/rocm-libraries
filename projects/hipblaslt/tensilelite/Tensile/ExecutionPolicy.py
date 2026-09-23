# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dependency-free execution policy, input normalization, and legacy translation."""

from dataclasses import dataclass
from enum import Enum


class TileProcessingStrategy(str, Enum):
    NONE = "None"
    DATA_PARALLEL = "DataParallel"
    STREAM_K = "StreamK"


class WorkAssignment(str, Enum):
    STATIC_GRID = "StaticGrid"
    DYNAMIC_WORK_QUEUE = "DynamicWorkQueue"
    HYBRID = "Hybrid"


class UnsupportedExecutionPolicy(ValueError):
    """Known parameter values form an unsupported execution-policy candidate."""


@dataclass(frozen=True)
class ExecutionPolicy:
    strategy: TileProcessingStrategy
    assignment: WorkAssignment

    @property
    def persistent(self):
        return self.strategy != TileProcessingStrategy.NONE

    @property
    def data_parallel(self):
        return self.strategy == TileProcessingStrategy.DATA_PARALLEL

    @property
    def stream_k(self):
        return self.strategy == TileProcessingStrategy.STREAM_K


def resolve_policy(state):
    """Validate the public pair without importing configuration or codegen."""
    try:
        strategy = TileProcessingStrategy(state.get("TileProcessingStrategy", "None"))
    except (ValueError, TypeError) as exc:
        raise ValueError("TileProcessingStrategy must be 'None', DataParallel, or StreamK") from exc
    try:
        assignment = WorkAssignment(state.get("WorkAssignment", "StaticGrid"))
    except (ValueError, TypeError) as exc:
        raise ValueError("WorkAssignment must be StaticGrid, DynamicWorkQueue, or Hybrid") from exc
    if strategy == TileProcessingStrategy.NONE:
        assignment = WorkAssignment.STATIC_GRID
    elif strategy == TileProcessingStrategy.DATA_PARALLEL and assignment != WorkAssignment.STATIC_GRID:
        raise UnsupportedExecutionPolicy(f"{strategy.value} supports WorkAssignment=StaticGrid only")
    policy = ExecutionPolicy(strategy, assignment)
    if "_PersistentLoop" in state and state["_PersistentLoop"] != policy.persistent:
        raise ValueError("_PersistentLoop disagrees with the derived execution policy")
    return policy


def _policy(state):
    # Raw dictionaries are accepted at test/legacy entry points; generated state
    # has canonical selectors and never reconstructs legacy scheduling modes.
    if "TileProcessingStrategy" not in state and ("StreamK" in state or "StreamKForceDPOnly" in state):
        state = normalize_execution_policy(state)
    return resolve_policy(state)


def isPersistent(state):
    return _policy(state).persistent


def isDataParallel(state):
    return _policy(state).data_parallel


def isStreamK(state):
    return _policy(state).stream_k


def hasStaticAssignment(state):
    return isPersistent(state) and _policy(state).assignment == WorkAssignment.STATIC_GRID


def hasDynamicAssignment(state):
    return _policy(state).assignment == WorkAssignment.DYNAMIC_WORK_QUEUE


def hasHybridAssignment(state):
    return _policy(state).assignment == WorkAssignment.HYBRID


def requiresPartialReduction(state):
    return isStreamK(state) and not state.get("StreamKAtomic", 0)

# Input normalization and legacy/prebuilt compatibility.

ALIASES = {
    "StreamKXCCMapping": "PersistentXCCMapping",
    "StreamKWorkStealing": "WorkQueueStealing",
}
SELECTORS = {"StreamK", "StreamKForceDPOnly", "TileProcessingStrategy", "WorkAssignment"}


def normalize_execution_policy_with_defaults(config, library_defaults):
    """Merge file defaults before global defaults erase selector provenance.

    A solution can override either spelling of a selector or shared control.
    Partial selector overrides retain the other dimension from the file; only
    old/new spellings requested at the same tier must agree. Shared controls
    retain their inherited values and are validated against the resulting pair.
    """
    result = dict(library_defaults)
    result.update(config)
    legacy_selectors = {"StreamK", "StreamKForceDPOnly"}
    canonical_selectors = SELECTORS - legacy_selectors
    requested_legacy = legacy_selectors.intersection(config)
    requested_canonical = canonical_selectors.intersection(config)
    if requested_legacy:
        # Recover a missing legacy companion from canonical file defaults only
        # at this input boundary, before the normalizer removes old selectors.
        missing = legacy_selectors - result.keys()
        if missing and canonical_selectors.intersection(library_defaults):
            if "StreamK" in missing:
                fallback = normalize_execution_policy({
                    key: value for key, value in library_defaults.items() if key in canonical_selectors
                })
                result["StreamK"] = (
                    0 if fallback["TileProcessingStrategy"] == "None" else
                    {"StaticGrid": 3, "DynamicWorkQueue": 4, "Hybrid": 5}[
                        fallback["WorkAssignment"]]
                )
            if "StreamKForceDPOnly" in missing:
                # An explicit mode supplies both canonical dimensions; an
                # incomplete assignment-only default need not validate alone.
                result["StreamKForceDPOnly"] = int(
                    library_defaults.get("TileProcessingStrategy") == "DataParallel"
                )
        for key in canonical_selectors - requested_canonical:
            result.pop(key, None)
    elif requested_canonical and legacy_selectors.intersection(library_defaults):
        missing = canonical_selectors - requested_canonical
        if missing:
            fallback_config = {
                key: value for key, value in library_defaults.items() if key in SELECTORS
            }
            for key in requested_canonical:
                fallback_config.pop(key, None)
            if "TileProcessingStrategy" in requested_canonical:
                fallback_config.pop("StreamKForceDPOnly", None)
            fallback = normalize_execution_policy(fallback_config)
            for key in missing:
                result[key] = fallback[key]
        for key in legacy_selectors:
            result.pop(key, None)

    for old, new in ALIASES.items():
        if old in config and new not in config:
            result.pop(new, None)
        elif new in config and old not in config:
            result.pop(old, None)
    if (requested_legacy or requested_canonical) and "_PersistentLoop" not in config:
        result.pop("_PersistentLoop", None)
    explicit = set(config) | (set(result) & (SELECTORS | set(ALIASES) | set(ALIASES.values())))
    return normalize_execution_policy(result, explicit_keys=explicit)



def _translate_legacy_streamk_selectors(config, explicit, result):
    """Validate legacy selectors and write their canonical equivalents to result."""
    mode = config.get("StreamK", 0)
    force = config.get("StreamKForceDPOnly", 0)
    if type(mode) is not int or mode not in (0, 3, 4, 5):
        raise ValueError("Legacy StreamK must be 0, 3, 4, or 5; modes 1 and 2 are retired")
    # The legacy flag also accepted YAML booleans; mode numbers stay integers.
    if type(force) not in (int, bool) or force not in (0, 1):
        raise ValueError("StreamKForceDPOnly must be 0 or 1")
    if force and (mode != 3 or config.get("StreamKAtomic", 0)):
        raise UnsupportedExecutionPolicy("StreamKForceDPOnly requires non-atomic StreamK=3")
    strategy = "DataParallel" if force else "None" if mode == 0 else "StreamK"
    assignment = {0: "StaticGrid", 3: "StaticGrid", 4: "DynamicWorkQueue", 5: "Hybrid"}[mode]
    for key, value in (("TileProcessingStrategy", strategy), ("WorkAssignment", assignment)):
        if key == "WorkAssignment" and strategy == "None" and key in explicit:
            # Keep the supplied spelling for enum validation; the resolver
            # canonicalizes this inactive dimension after validating it.
            result[key] = config[key]
            continue
        if key in explicit and config[key] != value:
            raise ValueError(f"Conflicting legacy StreamK and {key}={config[key]} (expected {value})")
        result[key] = value


def normalize_execution_policy(config, explicit_keys=None, regenerate=True):
    """Translate before defaults/naming, retaining only canonical selectors.

    ``explicit_keys`` lets merge paths distinguish requested values from inherited
    defaults. ``regenerate=False`` preserves the layout of a prebuilt artifact.
    """
    result = dict(config)
    explicit = set(config if explicit_keys is None else explicit_keys)
    if "PersistentLoop" in explicit:
        raise ValueError("PersistentLoop is derived from TileProcessingStrategy and WorkAssignment; it is not a tuning parameter")
    legacy = bool(explicit & {"StreamK", "StreamKForceDPOnly"})
    if legacy:
        _translate_legacy_streamk_selectors(config, explicit, result)
    for old, new in ALIASES.items():
        if old in explicit:
            if new in explicit and config[old] != config[new]:
                raise ValueError(f"Conflicting {old} and {new}")
            result[new] = config[old]
        result.pop(old, None)
    result.pop("StreamK", None)
    result.pop("StreamKForceDPOnly", None)
    result.setdefault("TileProcessingStrategy", "None")
    result.setdefault("WorkAssignment", "StaticGrid")
    policy = resolve_policy(result)
    result["WorkAssignment"] = policy.assignment.value
    result["_PersistentLoop"] = policy.persistent
    for option in ("StreamKAtomic", "StreamKFixupTreeReduction", "DebugStreamK"):
        if not policy.stream_k:
            if not legacy and option in explicit and result.get(option, 0):
                raise UnsupportedExecutionPolicy(f"{option} requires TileProcessingStrategy=StreamK")
            result[option] = 0
    if result.get("WorkQueueStealing", 0) and (not policy.stream_k or policy.assignment.value == "StaticGrid"):
        if not legacy or policy.persistent:
            raise UnsupportedExecutionPolicy("WorkQueueStealing requires StreamK with DynamicWorkQueue or Hybrid")
    if not policy.persistent:
        for option in ("PrefetchAcrossPersistent", "ReuseAcrossPersistent", "DebugPersistentKernelLoopForever"):
            if not legacy and option in explicit and result.get(option, 0):
                raise UnsupportedExecutionPolicy(f"{option} requires a persistent TileProcessingStrategy")
            result[option] = False if option == "DebugPersistentKernelLoopForever" else 0
        result["PersistentXCCMapping"] = 0
        result["WorkQueueStealing"] = 0
    support = dict(result.get("InternalSupportParams", {}))
    version = support.get("PersistentLoopArgsVersion", 0)
    outer_version = support.get("KernArgsVersion", 3)
    if type(version) is not int or version not in (0, 1):
        raise ValueError("Unsupported PersistentLoopArgsVersion")
    if type(outer_version) is not int or outer_version not in (0, 1, 2, 3):
        raise ValueError("Unsupported KernArgsVersion")
    custom = result.get("CustomKernel")
    handwritten = (not custom.get("generated", False)
                   if isinstance(custom, dict) and custom.get("name")
                   else bool(result.get("CustomKernelName")))
    if version == 1 and not policy.data_parallel:
        # Selector overrides inherit the source's generated layout. Recompute
        # it below, while preserving explicit and prebuilt layout contracts.
        if not regenerate or handwritten or "InternalSupportParams" in explicit:
            raise ValueError("PersistentLoopArgsVersion=1 requires DataParallel/StaticGrid")
    if version == 1 and outer_version != 3 and (not regenerate or handwritten):
        raise ValueError("PersistentLoopArgsVersion=1 requires KernArgsVersion=3")
    if regenerate and not handwritten:
        # Native whole-tile scheduling and the verified outer protocol are
        # generator capabilities. Regenerating known older logic upgrades both;
        # handwritten/prebuilt artifacts retain the layout they declare.
        support["PersistentLoopArgsVersion"] = 1 if policy.data_parallel else 0
        if policy.data_parallel:
            support["KernArgsVersion"] = 3
    if regenerate and (legacy or support.get("PersistentLoopArgsVersion", 0) != version
                       or support.get("KernArgsVersion", outer_version) != outer_version):
        result["AssignedDerivedParameters"] = False
        result["AssignedProblemIndependentDerivedParameters"] = False
    support.setdefault("PersistentLoopArgsVersion", 0)
    result["InternalSupportParams"] = support
    return result
