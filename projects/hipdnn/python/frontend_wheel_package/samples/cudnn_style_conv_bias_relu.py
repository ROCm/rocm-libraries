# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Convolution + bias + ReLU written the way cuDNN's Python frontend is used.

Apart from the import line and the device buffers, this is the cudnn.pygraph
API: graph-level data types, keyword op calls, build(), and an execute() keyed
by tensor that raises on failure. It needs a real engine (for example the MIOpen
provider), not the test stub, and checks the result against a NumPy reference.
"""

import numpy as np

import hipdnn_frontend as cudnn


def reference_conv_bias_relu(x, w, b, padding, stride, dilation):
    n, c, h, wd = x.shape
    k, _, r, s = w.shape
    out_h = (h + 2 * padding[0] - dilation[0] * (r - 1) - 1) // stride[0] + 1
    out_w = (wd + 2 * padding[1] - dilation[1] * (s - 1) - 1) // stride[1] + 1
    padded = np.pad(x, ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2))
    y = np.zeros((n, k, out_h, out_w), dtype=np.float64)
    for i in range(r):
        for j in range(s):
            rows = slice(
                i * dilation[0], i * dilation[0] + stride[0] * out_h, stride[0]
            )
            cols = slice(
                j * dilation[1], j * dilation[1] + stride[1] * out_w, stride[1]
            )
            y += np.einsum("nchw,kc->nkhw", padded[:, :, rows, cols], w[:, :, i, j])
    return np.maximum(y + b, 0.0).astype(np.float32)


def strides_of(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return strides


def device_buffer(array):
    buffer = cudnn.DeviceBuffer(array.nbytes)
    buffer.copy_from_host(array.tobytes())
    return buffer


def main():
    rng = np.random.default_rng(0)
    padding, stride, dilation = [1, 1], [1, 1], [1, 1]
    x_host = rng.uniform(-1.0, 1.0, (2, 8, 16, 16)).astype(np.float32)
    w_host = rng.uniform(-1.0, 1.0, (16, 8, 3, 3)).astype(np.float32)
    b_host = rng.uniform(-1.0, 1.0, (1, 16, 1, 1)).astype(np.float32)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    X = graph.tensor(
        name="X",
        dim=x_host.shape,
        stride=strides_of(x_host.shape),
        data_type=x_host.dtype,
    )
    W = graph.tensor(
        name="W",
        dim=w_host.shape,
        stride=strides_of(w_host.shape),
        data_type=w_host.dtype,
    )
    B = graph.tensor(
        name="B",
        dim=b_host.shape,
        stride=strides_of(b_host.shape),
        data_type=b_host.dtype,
    )

    conv_output = graph.conv_fprop(
        image=X, weight=W, padding=padding, stride=stride, dilation=dilation
    )
    bias_output = graph.bias(name="bias", input=conv_output, bias=B)
    Y = graph.relu(name="relu", input=bias_output)
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.build([cudnn.heur_mode.FALLBACK])

    expected = reference_conv_bias_relu(
        x_host, w_host, b_host, padding, stride, dilation
    )
    x_dev, w_dev, b_dev = (
        device_buffer(x_host),
        device_buffer(w_host),
        device_buffer(b_host),
    )
    y_dev = cudnn.DeviceBuffer(expected.nbytes)
    workspace = cudnn.DeviceBuffer(max(graph.get_workspace_size(), 1))

    graph.execute({X: x_dev, W: w_dev, B: b_dev, Y: y_dev}, workspace)
    cudnn.hip_device_synchronize()

    actual = np.frombuffer(y_dev.copy_to_host(), dtype=np.float32).reshape(
        expected.shape
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    print(f"engine: {graph.get_plan_name()}")
    print(f"max abs error: {np.max(np.abs(actual - expected)):.3e}")
    print("SUCCESS: cuDNN-style conv + bias + relu matches the NumPy reference")


if __name__ == "__main__":
    main()
