# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import numpy as np
import hipdnn_frontend as hipdnn

def run_batch_norm_inference():
    """
    Demonstrates building and executing a batch normalization inference graph using hipdnn_frontend.
    Uses exact test values from TestCpuFpReferenceBatchnormFp64::BatchnormFwdInferenceSanityValidationNchw
    to verify Python bindings correctness.
    """
    
    print("=" * 70)
    print("Batch Normalization Inference Test")
    print("Using values from C++ test: BatchnormFwdInferenceSanityValidationNchw")
    print("=" * 70)
    
    # Define input dimensions matching the C++ test
    n, c, h, w = 1, 1, 2, 2  # Batch size=1, channels=1, height=2, width=2
    print(f"\nInput dimensions: N={n}, C={c}, H={h}, W={w}")

    # Create a handle for backend operations
    print("\nCreating hipdnn handle...")
    handle = hipdnn.Handle()

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

    # Prepare test data matching the C++ test
    print("\n" + "="*50)
    print("Preparing Test Data (from C++ test)")
    print("="*50)
    
    # Input data: x = [1, 2, 3, 4]
    x_data = np.array([[[[1.0, 2.0],
                         [3.0, 4.0]]]], dtype=np.float32)
    
    # Scale = 2.0
    scale_data = np.array([[[[2.0]]]], dtype=np.float32)
    
    # Bias = 0.5
    bias_data = np.array([[[[0.5]]]], dtype=np.float32)
    
    # Mean = 2.5
    mean_data = np.array([[[[2.5]]]], dtype=np.float32)
    
    # Inv variance = 0.8
    inv_variance_data = np.array([[[[0.8]]]], dtype=np.float32)
    
    # Expected output from C++ test
    expected_output = np.array([[[[-1.9, -0.3],
                                   [1.29, 2.90]]]], dtype=np.float32)
    
    print(f"\nTest Values:")
    print(f"  Input: {x_data.flatten()}")
    print(f"  Scale: {scale_data.flatten()[0]}")
    print(f"  Bias: {bias_data.flatten()[0]}")
    print(f"  Mean: {mean_data.flatten()[0]}")
    print(f"  Inv_variance: {inv_variance_data.flatten()[0]}")
    print(f"\nExpected Output: {expected_output.flatten()}")
    
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
    
    workspace_ptr = 0  # No workspace needed for this test
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
    
    # Display and verify results
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    
    print(f"\nActual output: {y_data.flatten()}")
    print(f"Expected output: {expected_output.flatten()}")
    
    # Compute differences
    differences = np.abs(y_data.flatten() - expected_output.flatten())
    max_diff = np.max(differences)
    
    print(f"\nElement-wise comparison:")
    print("-" * 50)
    for i, (actual, expected, diff) in enumerate(zip(y_data.flatten(), 
                                                     expected_output.flatten(), 
                                                     differences)):
        print(f"  Element[{i}]: Actual={actual:10.8f}, Expected={expected:10.8f}, Diff={diff:.9e}")
    
    # Check tolerance
    tolerance = 1e-6
    print(f"\nTolerance check (tolerance = {tolerance}):")
    if max_diff < tolerance:
        print(f"✓ TEST PASSED! Maximum difference ({max_diff:.9e}) is within tolerance")
    else:
        print(f"✗ TEST FAILED! Maximum difference ({max_diff:.9e}) exceeds tolerance")
    
    # Manual verification of batch norm formula for first element
    print("\n" + "="*50)
    print("Manual Formula Verification (first element)")
    print("="*50)
    print("Formula: y = scale * (x - mean) * inv_variance + bias")
    print(f"         y = {scale_data.flat[0]} * ({x_data.flat[0]} - {mean_data.flat[0]}) * {inv_variance_data.flat[0]} + {bias_data.flat[0]}")
    
    manual_result = scale_data.flat[0] * (x_data.flat[0] - mean_data.flat[0]) * inv_variance_data.flat[0] + bias_data.flat[0]
    print(f"         y = {manual_result:.8f}")
    print(f"Actual output[0]: {y_data.flat[0]:.8f}")
    print(f"Difference: {abs(manual_result - y_data.flat[0]):.9e}")
    
    print("\n" + "="*70)
    if max_diff < tolerance:
        print("SUCCESS: Python bindings produce correct results!")
    else:
        print("FAILURE: Results do not match C++ test values")
    print("="*70)

if __name__ == "__main__":
    try:
        run_batch_norm_inference()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
