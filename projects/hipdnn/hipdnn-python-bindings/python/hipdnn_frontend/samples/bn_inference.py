# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import numpy as np
import hipdnn_frontend as hipdnn

def run_batch_norm_inference():
    """
    Demonstrates building and executing a batch normalization inference graph using hipdnn_frontend.
    """
    
    print("Creating batch normalization inference graph...")
    
    # Define input dimensions
    n, c, h, w = 16, 16, 16, 16  # Batch size, channels, height, width
    print(f"Input dimensions: N={n}, C={c}, H={h}, W={w}")

    # Create a handle for backend operations
    print("\nCreating hipdnn handle...")
    handle = hipdnn.Handle()
    #print(f"Handle created: {handle}")

    # Create a graph
    graph = hipdnn.Graph()
    graph.set_name("batch_norm_inference_graph")
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    print("Graph created with FLOAT data type")

    # Create tensors
    print("\nCreating tensors...")
    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("input")
    print(f"  Input tensor: shape={[n, c, h, w]}, uid={x.get_uid()}")
    
    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale.set_name("scale")
    print(f"  Scale tensor: shape={[1, c, 1, 1]}, uid={scale.get_uid()}")
    
    bias = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    bias.set_name("bias")
    print(f"  Bias tensor: shape={[1, c, 1, 1]}, uid={bias.get_uid()}")
    
    mean = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    mean.set_name("mean")
    print(f"  Mean tensor: shape={[1, c, 1, 1]}, uid={mean.get_uid()}")
    
    inv_variance = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    inv_variance.set_name("inv_variance")
    print(f"  Inv variance tensor: shape={[1, c, 1, 1]}, uid={inv_variance.get_uid()}")

    # Set attributes for batch normalization inference
    bn_attributes = hipdnn.BatchnormInferenceAttributes()
    
    print("\nBuilding batch normalization operation...")
    # Perform batch normalization inference
    y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, bn_attributes)
    
    # Mark the output tensor
    if y:
        y.set_name("output")
        y.set_output(True)
        print(f"Output tensor created: uid={y.get_uid()}")
    
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
    # Uses FALLBACK heuristic mode by default
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

    print(f"  Input tensor: shape={[n, c, h, w]}, uid={x.get_uid()}")
    print(f"  Output tensor: uid={y.get_uid()}")
    print(f"  Scale tensor: shape={[1, c, 1, 1]}, uid={scale.get_uid()}")
    print(f"  Bias tensor: shape={[1, c, 1, 1]}, uid={bias.get_uid()}")
    print(f"  Mean tensor: shape={[1, c, 1, 1]}, uid={mean.get_uid()}")
    print(f"  Inv variance tensor: shape={[1, c, 1, 1]}, uid={inv_variance.get_uid()}")

    # Build plans
    print("\nBuilding execution plans...")
    build_plans_result = graph.build_plans()
    if build_plans_result.is_good():
        print("✓ Execution plans built successfully")
    else:
        print(f"✗ Failed to build plans: {build_plans_result.get_message()}")
        return

    # Get workspace size
    # print("\nQuerying workspace requirements...")
    # workspace_result, workspace_size = graph.get_workspace_size()
    # if workspace_result.is_good():
    #     print(f"✓ Workspace size: {workspace_size} bytes")
    # else:
    #     print(f"✗ Failed to get workspace size: {workspace_result.get_message()}")
    #     return

    # Prepare input data
    print("\n" + "="*50)
    print("Preparing data for execution...")
    print("="*50)
    
    # Create host arrays with random data
    x_data = np.random.randn(n, c, h, w).astype(np.float32)
    scale_data = np.random.randn(1, c, 1, 1).astype(np.float32)
    bias_data = np.random.randn(1, c, 1, 1).astype(np.float32)
    mean_data = np.random.randn(1, c, 1, 1).astype(np.float32)
    inv_variance_data = np.abs(np.random.randn(1, c, 1, 1).astype(np.float32)) + 0.1  # Ensure positive
    
    print(f"Created random input tensors")
    print(f"  x shape: {x_data.shape}, dtype: {x_data.dtype}")
    print(f"  scale shape: {scale_data.shape}, dtype: {scale_data.dtype}")
    print(f"  bias shape: {bias_data.shape}, dtype: {bias_data.dtype}")
    print(f"  mean shape: {mean_data.shape}, dtype: {mean_data.dtype}")
    print(f"  inv_variance shape: {inv_variance_data.shape}, dtype: {inv_variance_data.dtype}")
    
    # Allocate device memory
    print("\nAllocating device memory...")
    x_buffer = hipdnn.DeviceBuffer(x_data.nbytes)
    scale_buffer = hipdnn.DeviceBuffer(scale_data.nbytes)
    bias_buffer = hipdnn.DeviceBuffer(bias_data.nbytes)
    mean_buffer = hipdnn.DeviceBuffer(mean_data.nbytes)
    inv_variance_buffer = hipdnn.DeviceBuffer(inv_variance_data.nbytes)
    y_buffer = hipdnn.DeviceBuffer(x_data.nbytes)  # Output same size as input
    
    print(f"✓ Allocated {x_data.nbytes + scale_data.nbytes * 4 + x_data.nbytes} bytes of device memory")
    
    # Copy data to device
    print("\nCopying data to device...")
    x_buffer.copy_from_host(x_data.tobytes())
    scale_buffer.copy_from_host(scale_data.tobytes())
    bias_buffer.copy_from_host(bias_data.tobytes())
    mean_buffer.copy_from_host(mean_data.tobytes())
    inv_variance_buffer.copy_from_host(inv_variance_data.tobytes())
    print("✓ Data copied to device")
    
    # Allocate workspace if needed
    workspace_buffer = None
    # if workspace_size > 0:
    #     print(f"\nAllocating workspace of {workspace_size} bytes...")
    #     workspace_buffer = hipdnn.DeviceBuffer(workspace_size)
    #     print("✓ Workspace allocated")
    
    # Create variant pack mapping tensor UIDs to device pointers
    print("\nPreparing variant pack...")
    variant_pack = {
        x.get_uid(): x_buffer.ptr(),
        scale.get_uid(): scale_buffer.ptr(),
        bias.get_uid(): bias_buffer.ptr(),
        mean.get_uid(): mean_buffer.ptr(),
        inv_variance.get_uid(): inv_variance_buffer.ptr(),
        y.get_uid(): y_buffer.ptr()
    }
    
    print(f"Variant pack created with {len(variant_pack)} tensor mappings")
    
    # Execute the graph
    print("\n" + "="*50)
    print("Executing graph...")
    print("="*50)
    
    workspace_ptr = workspace_buffer.ptr() if workspace_buffer else 0
    exec_result = graph.execute(handle, variant_pack, workspace_ptr)
    
    if exec_result.is_good():
        print("✓ Graph executed successfully!")
    else:
        print(f"✗ Graph execution failed: {exec_result.get_message()}")
        return
    
    # Copy results back to host
    print("\nCopying results back to host...")
    y_bytes = y_buffer.copy_to_host()
    y_data = np.frombuffer(y_bytes, dtype=np.float32).reshape(x_data.shape)
    print("✓ Results copied to host")
    
    # Display some results
    print("\n" + "="*50)
    print("Execution Results")
    print("="*50)
    print(f"Output shape: {y_data.shape}")
    print(f"Output dtype: {y_data.dtype}")
    print(f"First 10 output values: {y_data.flat[:10]}")
    print(f"Output min: {y_data.min():.6f}, max: {y_data.max():.6f}, mean: {y_data.mean():.6f}")
    
    # Verify batch normalization formula (optional)
    print("\nVerifying batch normalization formula (first element):")
    expected = (x_data.flat[0] - mean_data.flat[0]) * inv_variance_data.flat[0] * scale_data.flat[0] + bias_data.flat[0]
    actual = y_data.flat[0]
    print(f"  Expected: {expected:.6f}")
    print(f"  Actual:   {actual:.6f}")
    print(f"  Difference: {abs(expected - actual):.9f}")
    
    print("\n" + "="*50)
    print("Batch normalization inference complete!")
    print("="*50)

if __name__ == "__main__":
    try:
        run_batch_norm_inference()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
