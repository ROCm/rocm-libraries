# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import numpy as np

import hipdnn_frontend as hipdnn


def run_autotune():
    """
    Demonstrates engine discovery, knob introspection and autotuning a graph.
    This is the Python equivalent of samples/autotune/AutotuneSample.cpp

    Tunes a convolution forward graph over every applicable engine, prints the
    ranking, and then executes the graph -- autotune() leaves the winning plan
    active, so no further selection call is needed.
    """

    print("=" * 70)
    print("Autotune Test")
    print("=" * 70)

    # Dimensions
    n, c, h, w = 8, 16, 32, 32
    k, r, s = 16, 3, 3
    stride, pad, dil = 1, 1, 1

    out_h = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1

    print(f"\nInput dimensions: N={n}, C={c}, H={h}, W={w}")
    print(f"Filter dimensions: K={k}, C={c}, R={r}, S={s}")
    print(f"Output dimensions: N={n}, K={k}, H={out_h}, W={out_w}")

    print("\nCreating hipdnn handle...")
    handle = hipdnn.create_handle()

    graph = hipdnn.Graph()
    graph.set_name("autotune_graph")
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")
    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("w")

    conv_attrs = hipdnn.ConvFpropAttributes()
    conv_attrs.set_name("conv_fprop_node")
    conv_attrs.set_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    y = graph.conv_fprop(x, weight, conv_attrs)
    y.set_name("y")
    y.set_output(True)

    print("\nValidating graph...")
    validation_result = graph.validate()
    if not validation_result.is_good():
        print(f"✗ Graph validation failed: {validation_result.get_message()}")
        return

    print("Building operation graph...")
    build_result = graph.build_operation_graph(handle)
    if not build_result.is_good():
        print(f"✗ build_operation_graph failed: {build_result.get_message()}")
        return

    # 1. Discover the engines that can run this graph, and their knobs.
    print("\nDiscovering engines...")
    configs = graph.get_engine_configs()
    for config in configs:
        print(
            f"  engine {config.engine_id} ({config.engine_name}): "
            f"{len(config.knobs)} knob(s), "
            f"workspace estimate {config.estimated_workspace_size} B, "
            f"exhaustive={config.supports_exhaustive}"
        )
        for knob in config.knobs:
            # knob.constraint describes the legal values, so a KnobSweepAxis can
            # be generated from it instead of hardcoding candidates.
            print(
                f"      {knob.knob_id}: type={knob.value_type.name}, "
                f"default={knob.default_value}, constraint={knob.constraint}"
            )

    if not configs:
        print("✗ No engine supports this graph")
        return

    # 2. Collect autotune candidates. add_all_engines() takes every engine; use
    #    add_engine(id, [KnobSetting(...)]) or add_engine_sweep() for finer control.
    print("\nAdding all engines as autotune candidates...")
    add_result = graph.add_all_engines()
    if not add_result.is_good():
        print(f"✗ add_all_engines failed: {add_result.get_message()}")
        return

    # 3. Allocate device buffers plus workspace. The pre-compile estimate is a
    #    lower bound, so give the budget headroom; plans needing more are skipped.
    estimated_workspace = graph.get_estimated_max_workspace_size()
    workspace_size = max(2 * estimated_workspace, 1 << 20)
    print(f"Workspace estimate: {estimated_workspace} B, budget {workspace_size} B")

    tensors = {
        x: np.random.randn(n, c, h, w).astype(np.float32),
        weight: np.random.randn(k, c, r, s).astype(np.float32),
        y: np.zeros((n, k, out_h, out_w), dtype=np.float32),
    }
    buffers = {}
    for tensor, data in tensors.items():
        buffer = hipdnn.DeviceBuffer(data.nbytes)
        buffer.copy_from_host(data.tobytes())
        buffers[tensor] = buffer
    workspace_buffer = hipdnn.DeviceBuffer(workspace_size)

    # 4. Tune. The variant pack may be keyed by tensor or by tensor UID.
    config = hipdnn.AutotuneConfig()
    config.mode = hipdnn.TuneMode.STANDARD
    config.strategy = hipdnn.AutotuneStrategy.RUN_UNTIL_STABLE
    config.warmup_iterations = 2
    config.max_iterations = 20

    print("\nAutotuning...")
    results = graph.autotune(
        handle,
        {tensor: buffer.ptr() for tensor, buffer in buffers.items()},
        workspace_buffer.ptr(),
        workspace_size=workspace_size,
        config=config,
    )

    print("\n" + "=" * 50)
    print("Autotune Results")
    print("=" * 50)
    for result in sorted(results, key=lambda r: (r.rank < 0, r.rank)):
        if result.succeeded:
            print(
                f"  #{result.rank} engine {result.engine_id} ({result.engine_name}): "
                f"min {result.min_time_ms:.6f} ms, avg {result.avg_time_ms:.6f} ms, "
                f"stddev {result.stddev_ms:.6f} ms over {result.iterations_run} iter(s), "
                f"workspace {result.workspace_size} B"
            )
            for setting in result.knob_settings:
                print(f"      {setting.knob_id}={setting.value}")
        else:
            print(
                f"  -- engine {result.engine_id} ({result.engine_name}) failed: "
                f"{result.error_message}"
            )

    winners = [result for result in results if result.succeeded]
    if not winners:
        print("\n✗ No engine benchmarked successfully")
        return

    # 5. The winning plan is already active: execute uses it directly.
    print(
        f"\nActive plan after autotune: {graph.get_plan_name()} "
        f"(engine {graph.get_execution_plan_engine_id()})"
    )
    exec_result = graph.execute(
        handle,
        {tensor.get_uid(): buffer.ptr() for tensor, buffer in buffers.items()},
        workspace_buffer.ptr(),
    )
    if not exec_result.is_good():
        print(f"✗ Execution failed: {exec_result.get_message()}")
        return

    y_result = np.frombuffer(buffers[y].copy_to_host(), dtype=np.float32).reshape(
        n, k, out_h, out_w
    )
    print(f"First 10 output values: {y_result.flatten()[:10]}")

    print("\n" + "=" * 70)
    print("SUCCESS: Autotuning completed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_autotune()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
