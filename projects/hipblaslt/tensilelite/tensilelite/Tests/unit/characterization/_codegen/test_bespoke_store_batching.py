# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compiled D-store batching for the two beta-kept slowdown recipes.

Feature: rebalance the VGPR-budgeted batches in refineOccupancy only for
BespokeNarrowing. Restoring budget-sized full batches plus a tiny remainder
breaks both balance and minimum-tail tests, with beta zero and nonzero.
Preservation: keep stock batching and compiled bytes unchanged. Applying the
rebalance to stock, or changing its shared store budget, breaks the stock pins.

The fixtures are unchanged one-solution general recipes. The narrowing facts
reproduce their keep-beta bespoke siblings (provenance.json). All instructions
come from production-assembled objects; no GPU, mocks or comment assertions.
"""
import ast
import copy
import inspect
import textwrap
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bespoke_codegen_common import assembly_digest, compile_main, variant

pytestmark = pytest.mark.unit
DATA = Path(__file__).parent / "data/test_data/_logic/bespoke_store_batching"
RECIPES = (
    ("bf16_mt192x160.yaml", "001a4ad86324", (192, 160), (6144, 1024), 15),
    ("bf16_mt256x128.yaml", "8e1cce2c4de9", (256, 128), (4096, 2048), 16),
)
PATHS = ((0, "NonEdge"), (1, "NonEdge"), (0, "Then"), (1, "Then"))
# Independently observed stock layout on c4a20925321: beta zero has more VGPRs
# available on the non-edge path; beta nonzero and edge keep the 6-element cap.
STOCK_BATCHES = {
    "001a4ad86324": {(0, "NonEdge"): (8, 7), (1, "NonEdge"): (6, 6, 3),
                     (0, "Then"): (6, 6, 3), (1, "Then"): (6, 6, 3)},
    "8e1cce2c4de9": {(0, "NonEdge"): (8, 8), (1, "NonEdge"): (6, 6, 4),
                     (0, "Then"): (6, 6, 4), (1, "Then"): (6, 6, 4)},
}


def store_batches(instructions, addresses, beta, region, mode="GSU1", stores_per_element=1):
    """Count compiled D stores between batch barriers, on activation-none.

Use ELF labels only to locate the production beta/edge paths (not comments).
Stock dispatches activation at each entry, so inspect its none arm rather than
concatenating all mutually exclusive activations. Local labels inside that arm
must not split a batch. Each dwordx4 D store is one eight-bf16 store element.
"""
    prefix = f"label_GW_B{beta}_FD0_VW8_{mode}_"
    # Unreferenced fallthrough Then labels can be omitted from the ELF; the
    # preceding NonEdgeEnd contains only the edge dispatch, before activation.
    entry = prefix + region
    start = addresses[entry] if entry in addresses else addresses[prefix + "NonEdgeEnd"]
    # Elided lower-VW alternatives can alias Else to Then. The next *distinct*
    # global-write address bounds the remaining eight-wide edge path.
    end = (addresses[prefix + "NonEdgeEnd"] if region == "NonEdge" else
           min(pc for name, pc in addresses.items() if name.startswith("label_GW_") and pc > start))
    none = [pc for name, pc in addresses.items()
            if "Activation_None" in name and start <= pc < end]
    if none:
        start = min(none)
        end = min(pc for name, pc in addresses.items()
                  if "Activation_" in name and start < pc < end)
    code = [text for pc, text in instructions if start <= pc < end]
    batches, stores, loads = [], 0, 0
    for text in code:
        if text.startswith("s_barrier") and stores:
            batches.append(stores)
            stores = 0
        if text.startswith("buffer_store_dwordx4 "):
            stores += 1
        if text.startswith("buffer_load_dwordx4 "):
            loads += 1
    if stores:
        batches.append(stores)
    assert batches, f"beta={beta} {region}: compiled path has no D stores"
    assert all(size % stores_per_element == 0 for size in batches), "partial store element lost"
    return tuple(size // stores_per_element for size in batches), loads


@pytest.fixture(scope="module", params=RECIPES, ids=[row[1] for row in RECIPES])
def compiled_pair(request, tmp_path_factory):
    recipe, identity, tile, shape, elements = request.param
    out = tmp_path_factory.mktemp(identity)
    pair = {}
    for arm in ("stock", "bespoke"):
        path = DATA / recipe
        if arm == "bespoke":
            path = variant(path, out / "bespoke.yaml",
                           problem={"Activation": False, "ActivationType": "none", "UseBias": 0,
                                    "BiasDataTypeList": [], "UseScaleAlphaVec": 0},
                           solution={"BespokeNarrowing": True,
                                     "AssertSizeEqual": {0: shape[0], 1: shape[1], 2: 1, 3: 4096},
                                     "AssertFree0ElementMultiple": 256,
                                     "AssertFree1ElementMultiple": 256,
                                     "AssertSummationElementMultiple": 256})
        compiled = out / arm
        solution, _, instructions, text_sha = compile_main(path, compiled)
        assert (solution["MacroTile0"], solution["MacroTile1"]) == tile
        assert solution["ProblemType"]["UseBeta"] is True
        raw = subprocess.run(["nm", "--defined-only", "--format=posix",
                              str(next(compiled.glob("*.o")))],
                             capture_output=True, text=True, check=True).stdout
        addresses = {parts[0]: int(parts[2], 16) for line in raw.splitlines()
                     if len(parts := line.split()) >= 3 and parts[1].lower() == "t"}
        # MT256x128 divides the exact shape: its bespoke edge path is eliminated.
        paths = PATHS if arm == "stock" or identity == "001a4ad86324" else PATHS[:2]
        layouts = {key: store_batches(instructions, addresses, *key) for key in paths}
        for (beta, region), (batches, loads) in layouts.items():
            assert sum(batches) == elements, f"{identity} {arm} beta={beta} {region}: D elements lost"
            assert loads == (elements if beta else 0), \
                f"{identity} {arm} beta={beta} {region}: wrong number of C loads"
        # MB partial writes store eight f32 values per element: two dwordx4
        # instructions, not the single dwordx4 of the final bf16 D path.
        partial_regions = ("NonEdge", "Then") if identity == "001a4ad86324" else ("NonEdge",)
        partials = {region: store_batches(instructions, addresses, 0, region, "MB", 2)
                    for region in partial_regions}
        for region, (batches, loads) in partials.items():
            assert sum(batches) == elements, f"{identity} {arm} MB {region}: partial elements lost"
            assert loads == 0, f"{identity} {arm} MB {region}: partial writes unexpectedly load C"
        pair[arm + "_partials"] = partials
        pair[arm] = (layouts, text_sha, assembly_digest(instructions))
    return recipe, identity, pair


@pytest.mark.parametrize("beta", [0, 1], ids=["beta_zero", "beta_nonzero"])
def test_bespoke_d_store_batches_are_balanced(compiled_pair, beta):
    _, identity, pair = compiled_pair
    unbalanced = {region: batches for (path_beta, region), (batches, _) in pair["bespoke"][0].items()
                  if path_beta == beta and max(batches) - min(batches) > 2}
    assert not unbalanced, f"{identity} beta={beta}: unbalanced D-store batches {unbalanced} (even step=2)"


@pytest.mark.parametrize("beta", [0, 1], ids=["beta_zero", "beta_nonzero"])
def test_bespoke_d_store_tail_is_not_smaller_than_stock(compiled_pair, beta):
    _, identity, pair = compiled_pair
    tiny_tails = {key[1]: (batches, min(STOCK_BATCHES[identity][key]))
                  for key, (batches, _) in pair["bespoke"][0].items()
                  if key[0] == beta and len(batches) > 1
                  and min(batches) < min(STOCK_BATCHES[identity][key])}
    assert not tiny_tails, (f"{identity} beta={beta}: bespoke D-store tail below stock minimum "
                            f"with batch loop remaining (batches, stock minimum): {tiny_tails}")


def test_bespoke_mb_partial_write_layout_is_balanced(compiled_pair):
    _, identity, pair = compiled_pair
    expected = (8, 7) if identity == "001a4ad86324" else (8, 8)
    partials = {region: layout[0] for region, layout in pair["bespoke_partials"].items()}
    assert partials and all(batches == expected for batches in partials.values()), \
        f"{identity}: MB partial-write batches changed: {partials} (want {expected})"


def test_stock_mb_partial_write_layout_is_unchanged(compiled_pair):
    _, identity, pair = compiled_pair
    expected = (14, 1) if identity == "001a4ad86324" else (14, 2)
    partials = {region: layout[0] for region, layout in pair["stock_partials"].items()}
    assert partials and all(batches == expected for batches in partials.values()), \
        f"{identity}: stock MB partial-write batches changed: {partials} (want {expected})"


@pytest.mark.parametrize("key", PATHS, ids=[f"beta{beta}_{region}" for beta, region in PATHS])
def test_stock_d_store_batch_layout_is_unchanged(compiled_pair, key):
    _, identity, pair = compiled_pair
    got = pair["stock"][0][key][0]
    assert got == STOCK_BATCHES[identity][key], f"{identity} {key}: stock D-store batches changed: {got}"


@pytest.mark.parametrize("digest", ["text_sha256", "assembly_sha256"], ids=["machine_text", "compiled_assembly"])
def test_stock_store_kernel_compiled_bytes_are_unchanged(compiled_pair, digest):
    recipe, _, pair = compiled_pair
    pins = json.loads((DATA / "stock_assembly_digests.json").read_text())["recipes"][recipe]
    _, text_sha, assembly_sha = pair["stock"]
    got = {"text_sha256": text_sha, "assembly_sha256": assembly_sha}[digest]
    assert got == pins[digest], f"{recipe}: stock {digest} changed"


@pytest.fixture(scope="module")
def production_batch_consumer():
    """Execute the emitter's actual slice/index code, without ISA emission.

    Keeping the consumer prefix from production prevents this test from
    silently modelling an old fixed-size loop after its slicing is corrected.
    Only the backend emission is replaced with recording the sliced elements.
    """
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    method = KernelWriterAssembly.globalWriteElementBatch
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    emit = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_emit_batch_loop")
    loop = copy.deepcopy(next(node for node in emit.body if isinstance(node, ast.For)))
    stop = next(index for index, node in enumerate(loop.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "elementsThisBatch"
                        for target in node.targets))
    loop.body = loop.body[:stop + 1] + ast.parse("captured.append(elementsThisBatch)").body
    tree = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    return compile(tree, inspect.getfile(method), "exec"), method.__globals__


def refined_store_batches(dtype, budget, n, consumer, *, first_row=None):
    """Real pools, real occupancy/refinement, and the real consumer slices."""
    from rocisa.enum import RegisterType
    from rocisa.register import RegisterPool
    from tensilelite.Common.DataType import DataType
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = object.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(
        c=SimpleNamespace(numVgprValu=n), version=(9, 5, 0), doubleVgpr=False,
        regCaps={"MaxVgpr": 32, "PhysicalMaxVgpr": 32,
                 "MaxSgpr": 102, "PhysicalMaxSgpr": 800},
        archCaps={"DeviceLDS": 65536, "LdsGranularity": 256, "MaxWavesPerSimd": 8})
    for name, kind, size in (("vgprPool", RegisterType.Vgpr, 32),
                             ("sgprPool", RegisterType.Sgpr, 8),
                             ("agprPool", RegisterType.Accvgpr, 0)):
        pool = RegisterPool(size, kind, defaultPreventOverflow=False, printRP=False)
        if size:
            pool.add(0, size, "batch-property")
        setattr(writer, name, pool)
    kernel = {"ProblemType": {"DataType": DataType(dtype), "DestDataType": DataType(dtype)},
              "BespokeNarrowing": True, "NumElementsPerBatchStore": budget,
              "StoreRemapVectorWidth": 0, "CompactLoopStore": False,
              "EnableMatrixInstruction": True, "NumThreads": 256,
              "LdsNumBytes": 0, "ISA": (9, 5, 0)}
    ss = SimpleNamespace(numVgprsPerElement=1, align=1,
                         optSingleColVgpr=True, optSharedColVgpr=False,
                         cfg=SimpleNamespace(numElementsPerBatchLimitedBySgprs=32,
                                             numTempSgprPerBatch=2, numMaskSgprPerBatch=1,
                                             numMaskSgprPerElement=2))
    element = (list(range(n)) if first_row is None else
               [(0 if index < first_row else 1, index, 0, 0) for index in range(n)])
    max_vgprs, _ = writer.setOccupancy(kernel)
    batch_size, _, count, num_sgprs = writer.refineOccupancy(
        kernel, first_row is not None, element, 0, 1, max_vgprs, ss)
    captured = []
    code, globals_ = consumer
    exec(code, globals_, {"self": writer, "kernel": kernel, "ss": ss, "element": element,
                         "numElementsPerBatch": batch_size, "numBatchesCLS": count,
                         "numBatches": count, "captured": captured})
    return captured, batch_size, num_sgprs


@pytest.mark.parametrize("dtype,step", [("h", 2), ("b", 2), ("s", 1)], ids=["half", "bf16", "f32"])
def test_bespoke_store_batches_balance_any_element_count(production_batch_consumer, dtype, step):
    # Fix: distribute the remainder across batches, not just one final tail.
    # Break: revert to repeated ceil(n/count), which fails n=128, NEPB=14.
    unbalanced = []
    for input_budget in range(2, 33):
        # The existing half/bf16 rule rounds the budget DOWN before rebalance.
        # Both bounds use that effective budget.
        budget = input_budget - input_budget % step
        for n in range(1, 257):
            batches, batch_size, num_sgprs = refined_store_batches(
                dtype, input_budget, n, production_batch_consumer)
            sizes = tuple(map(len, batches))
            context = (f"{dtype}: n={n}, NEPB={budget} (input={input_budget}), batches={sizes}, "
                       f"allocated batch_size={batch_size}, numSgprs={num_sgprs}")
            assert [value for batch in batches for value in batch] == list(range(n)), context
            assert sizes and min(sizes) > 0 and max(sizes) <= budget, context
            # Per-batch allocation must price the largest emitted batch, not
            # the smallest one, even when the input budget would fit both.
            assert max(sizes) <= batch_size, context
            if len(sizes) > 1:
                # One-batch paths intentionally retain the original cap.
                assert batch_size - max(sizes) < step, context
            assert num_sgprs == 3 + 2 * batch_size, context
            assert len(sizes) <= (n + budget - 1) // budget, context
            if max(sizes) - min(sizes) > step:
                unbalanced.append((n, input_budget, sizes))
    examples = unbalanced[:3] + [case for case in unbalanced if case[:2] == (128, 14)]
    assert not unbalanced, (f"{dtype}: {len(unbalanced)} unbalanced emitted batch lists "
                            f"(rounding step={step}); (n, input NEPB, batches): {examples}")


@pytest.mark.parametrize("dtype", ["h", "b"], ids=["half", "bf16"])
def test_odd_atomic_first_row_cap_keeps_fixed_batch_bounds(production_batch_consumer, dtype):
    # Preservation: the three-element atomic row cap is applied AFTER even
    # rounding. Rebalancing either allocation or slices despite this odd cap
    # breaks the literal bounds or exceeds the three-element register budget.
    batches, batch_size, num_sgprs = refined_store_batches(
        dtype, 14, 8, production_batch_consumer, first_row=3)
    sizes = tuple(map(len, batches))
    assert batch_size == 3 and max(sizes) <= batch_size, (batch_size, sizes)
    assert sizes == (3, 3, 2), sizes
    assert len(batches) == 3
    assert num_sgprs == 9
    assert [element for batch in batches for element in batch] == [
        (0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 3, 0, 0),
        (1, 4, 0, 0), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0)]
