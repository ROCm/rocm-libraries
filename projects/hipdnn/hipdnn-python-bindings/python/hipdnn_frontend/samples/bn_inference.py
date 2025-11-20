# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import numpy as np
import hipdnn_frontend as hipdnn

def run_batch_norm_inference():
    """
    Demonstrates building a batch normalization inference graph using hipdnn_frontend.
    
    Note: This sample creates and validates the graph structure. Actual execution
    would require handle management which is not exposed in the current Python bindings.
    """
    
    print("Creating batch normalization inference graph...")
    
    # Define input dimensions
    n, c, h, w = 16, 16, 16, 16  # Batch size, channels, height, width
    print(f"Input dimensions: N={n}, C={c}, H={h}, W={w}")

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
    #bn_attributes.set_name("bn_inference_node")
    # Note: compute_data_type is set on the graph, not on the attributes
    
    print("\nBuilding batch normalization operation...")
    # Perform batch normalization inference
    # The batchnorm_inference method takes all tensors as arguments
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

    # # Check if the graph has duplicate tensor IDs
    # print("\nChecking for duplicate tensor IDs...")
    # duplicate_check = graph.checkNoDuplicateTensorIds()
    # if duplicate_check.is_good():
    #     print("✓ No duplicate tensor IDs found")
    # else:
    #     print(f"✗ Duplicate tensor IDs detected: {duplicate_check.get_message()}")

    # # Sort the graph topologically
    # print("\nPerforming topological sort...")
    # sort_result = graph.topologicallySortGraph()
    # if sort_result.is_good():
    #     print("✓ Graph successfully sorted")
    # else:
    #     print(f"✗ Topological sort failed: {sort_result.get_message()}")

    # # Build flatbuffer operation graph
    # print("\nBuilding flatbuffer operation graph...")
    # build_result = graph.buildFlatbufferOperationGraph()
    # if build_result.is_good():
    #     print("✓ Flatbuffer operation graph built successfully")
    # else:
    #     print(f"✗ Failed to build flatbuffer graph: {build_result.get_message()}")

    # Create execution plans
    print("\nCreating execution plans...")
    plan_result = graph.create_execution_plans()
    if plan_result.is_good():
        print("✓ Execution plans created successfully")
    else:
        print(f"✗ Failed to create execution plans: {plan_result.get_message()}")

    # Check support
    print("\nChecking backend support...")
    support_result = graph.check_support()
    if support_result.is_good():
        print("✓ Graph operations are supported by backend")
    else:
        print(f"✗ Backend support check failed: {support_result.get_message()}")

    # Build plans
    print("\nBuilding execution plans...")
    build_plans_result = graph.build_plans()
    if build_plans_result.is_good():
        print("✓ Execution plans built successfully")
    else:
        print(f"✗ Failed to build plans: {build_plans_result.get_message()}")

    # # Get workspace size
    # print("\nQuerying workspace requirements...")
    # workspace_result, workspace_size = graph.get_workspace_size()
    # if workspace_result.is_good():
    #     print(f"✓ Workspace size: {workspace_size} bytes")
    # else:
    #     print(f"✗ Failed to get workspace size: {workspace_result.get_message()}")

    print("\n" + "="*50)
    print("Graph construction complete!")
    print("Note: Actual execution would require handle management")
    print("which is not exposed in the current Python bindings.")
    print("="*50)

    # Demonstrate creating input data (even though we can't execute)
    print("\nExample input data shapes that would be used for execution:")
    print(f"  x: np.ndarray with shape {(n, c, h, w)}")
    print(f"  scale: np.ndarray with shape {(1, c, 1, 1)}")
    print(f"  bias: np.ndarray with shape {(1, c, 1, 1)}")
    print(f"  mean: np.ndarray with shape {(1, c, 1, 1)}")
    print(f"  inv_variance: np.ndarray with shape {(1, c, 1, 1)}")

if __name__ == "__main__":
    run_batch_norm_inference()
