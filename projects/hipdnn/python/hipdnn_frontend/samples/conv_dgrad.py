# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import numpy as np
import hipdnn_frontend as hipdnn


def run_convolution_dgrad():
    """
    Demonstrates building and executing a convolution backward data gradient graph using hipdnn_frontend.
    This is the Python equivalent of samples/convolution/ConvDgrad.cpp
    Computes dx (input gradient) given dy (output gradient) and weights.
    """

    print("=" * 70)
    print("Convolution Backward Data (Dgrad) Test")
    print("=" * 70)

    # Define dimensions matching the C++ sample
    n = 16  # Batch size

    # Input (dx dimensions)
    c = 16  # Number of input channels
    h = 16  # Height
    w = 16  # Width

    # Filter dimensions
    k = 16  # Number of output channels (dy channels)
    r = 3  # Filter height
    s = 3  # Filter width

    # Convolution parameters
    u = 1  # Height stride
    v = 1  # Width stride
    padH = 1  # Height padding
    padW = 1  # Width padding
    dilH = 1  # Height dilation
    dilW = 1  # Width dilation

    # Calculate output gradient dimensions (dy dimensions)
    outH = (h + 2 * padH - dilH * (r - 1) - 1) // u + 1
    outW = (w + 2 * padW - dilW * (s - 1) - 1) // v + 1

    print(f"\nInput gradient (dx) dimensions: N={n}, C={c}, H={h}, W={w}")
    print(f"Filter dimensions: K={k}, C={c}, R={r}, S={s}")
    print(f"Output gradient (dy) dimensions: N={n}, K={k}, H={outH}, W={outW}")
    print(
        f"Conv params: stride=({u},{v}), padding=({padH},{padW}), dilation=({dilH},{dilW})"
    )

    # Create a handle for backend operations
    print("\nCreating hipdnn handle...")
    handle = hipdnn.create_handle()

    # Create a graph
    graph = hipdnn.Graph()
    graph.set_name("convolution_backward_data_graph")
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    print("Graph created with FLOAT data type")

    # Create tensors
    print("\nCreating tensors...")

    # Output gradient (dy) - this is the input to backward pass
    dy = hipdnn.Tensor.create([n, k, outH, outW], hipdnn.DataType.FLOAT)
    dy.set_name("output_gradient_dy")
    print(
        f"  Output gradient tensor (dy): shape={[n, k, outH, outW]}, uid={dy.get_uid()}"
    )

    # Weight tensor
    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("weight")
    print(f"  Weight tensor: shape={[k, c, r, s]}, uid={weight.get_uid()}")

    # Set convolution dgrad attributes
    conv_attributes = hipdnn.ConvDgradAttributes()
    conv_attributes.set_name("conv_backward_data_node")
    conv_attributes.set_pre_padding([padH, padW])
    conv_attributes.set_post_padding([padH, padW])
    conv_attributes.set_stride([u, v])
    conv_attributes.set_dilation([dilH, dilW])

    print("\nBuilding convolution backward data operation...")
    # Perform convolution backward data (dgrad)
    dx = graph.conv_dgrad(dy, weight, conv_attributes)

    # Mark the output tensor
    if dx:
        dx.set_name("input_gradient_dx")
        dx.set_output(True)
        print(f"Input gradient tensor (dx) created: uid={dx.get_uid()}")

    # Validate the graph
    print("\nValidating graph...")
    validation_result = graph.validate()
    if validation_result.is_good():
        print("✓ Graph validation successful!")
    else:
        print(f"✗ Graph validation failed: {validation_result.get_message()}")
        return

    # Build the operation graph
    print("\nBuilding operation graph...")
    build_result = graph.build_operation_graph(handle)
    if build_result.is_good():
        print("✓ Operation graph built successfully")
    else:
        print(f"✗ Failed to build operation graph: {build_result.get_message()}")
        return

    # Create execution plans
    print("\nCreating execution plans...")
    plan_result = graph.create_execution_plans()
    if plan_result.is_good():
        print("✓ Execution plans created successfully")
    else:
        print(f"✗ Failed to create execution plans: {plan_result.get_message()}")
        return

    # Check support
    print("\nChecking backend support...")
    support_result = graph.check_support()
    if support_result.is_good():
        print("✓ Graph operations are supported by backend")
    else:
        print(f"✗ Backend support check failed: {support_result.get_message()}")
        return

    # Build plans
    print("\nBuilding execution plans...")
    build_plans_result = graph.build_plans()
    if build_plans_result.is_good():
        print("✓ Execution plans built successfully")
    else:
        print(f"✗ Failed to build plans: {build_plans_result.get_message()}")
        return

    # Prepare test data
    print("\n" + "=" * 50)
    print("Preparing Test Data")
    print("=" * 50)

    # Initialize with random values
    dy_data = np.random.uniform(0.0, 1.0, [n, k, outH, outW]).astype(np.float32)
    weight_data = np.random.uniform(0.0, 1.0, [k, c, r, s]).astype(np.float32)

    # Calculate expected input gradient size
    dx_shape = [n, c, h, w]
    dx_data = np.zeros(dx_shape, dtype=np.float32)

    print(f"\nOutput gradient (dy) shape: {dy_data.shape}")
    print(f"Weight shape: {weight_data.shape}")
    print(f"Input gradient (dx) shape: {dx_shape}")
    print(f"\nFirst 10 dy values: {dy_data.flatten()[:10]}")
    print(f"First 10 weight values: {weight_data.flatten()[:10]}")

    # Allocate device memory
    print("\nAllocating device memory...")
    dy_buffer = hipdnn.DeviceBuffer(dy_data.nbytes)
    weight_buffer = hipdnn.DeviceBuffer(weight_data.nbytes)
    dx_buffer = hipdnn.DeviceBuffer(dx_data.nbytes)

    total_bytes = dy_data.nbytes + weight_data.nbytes + dx_data.nbytes
    print(f"✓ Allocated {total_bytes} bytes of device memory")

    # Copy data to device
    print("\nCopying data to device...")
    dy_buffer.copy_from_host(dy_data.tobytes())
    weight_buffer.copy_from_host(weight_data.tobytes())
    print("✓ Data copied to device")

    # Create variant pack mapping tensor UIDs to device pointers
    print("\nPreparing variant pack...")
    variant_pack = {
        dy.get_uid(): dy_buffer.ptr(),
        weight.get_uid(): weight_buffer.ptr(),
        dx.get_uid(): dx_buffer.ptr(),
    }

    print(f"Variant pack created with {len(variant_pack)} tensor mappings")

    # Get workspace size
    workspace_size = graph.get_workspace_size()
    print(f"\nWorkspace size required: {workspace_size} bytes")

    # Allocate workspace if needed
    workspace_buffer = None
    workspace_ptr = 0
    if workspace_size > 0:
        workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
        workspace_ptr = workspace_buffer.ptr()
        print(f"✓ Allocated {workspace_size} bytes of workspace memory")

    # Execute the graph
    print("\n" + "=" * 50)
    print("Executing graph...")
    print("=" * 50)

    exec_result = graph.execute(handle, variant_pack, workspace_ptr)

    if exec_result.is_good():
        print("✓ Graph executed successfully!")
    else:
        print(f"✗ Graph execution failed: {exec_result.get_message()}")
        return

    # Copy results back to host
    print("\nCopying results back to host...")
    dx_bytes = dx_buffer.copy_to_host()
    dx_result = np.frombuffer(dx_bytes, dtype=np.float32).reshape(dx_shape)
    print("✓ Results copied to host")

    # Display results
    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)

    print(f"\nFirst 10 dx values: {dx_result.flatten()[:10]}")

    # Simple validation - check that output is not all zeros
    if np.all(dx_result == 0):
        print("\n✗ WARNING: Input gradient (dx) is all zeros!")
    else:
        print(f"\n✓ Convolution backward data produced non-zero gradients")
        print(f"  dx min: {np.min(dx_result):.6f}")
        print(f"  dx max: {np.max(dx_result):.6f}")
        print(f"  dx mean: {np.mean(dx_result):.6f}")
        print(f"  dx std: {np.std(dx_result):.6f}")

    print("\n" + "=" * 70)
    print("SUCCESS: Convolution backward data (dgrad) completed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_convolution_dgrad()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
