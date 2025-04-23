# MIVisionX Neural Network Tests

## Script to run neural network tests

```
python runNeuralNetworkTests.py --help
```

usage: 

```
runNeuralNetworkTests.py [--profiler_mode PROFILER_MODE]
                         [--profiler_level PROFILER_LEVEL]
                         [--miopen_find MIOPEN_FIND]
                         [--test_info TEST_INFO]

Arguments:
  -h, --help            show this help message and exit
  --profiler_mode       NN Profile Mode - optional (default:0 [range:0 - 9])
  --profiler_level      NN Profile Batch Size in powers of 2 - optional (default:7 [range:1 - N])
  --miopen_find         MIOPEN_FIND_ENFORCE mode - optional (default:1 [range:1 - 5])
  --test_info           Show test info - optional (default:no [options:no/yes])
```

Test Info:
```
--profiler_mode     - NN Profile Mode: optional (default:0 [range:0 - 9])
    --profiler_mode 0 -- Run All Tests
    --profiler_mode 1 -- Run caffe2nnir2openvx No Fuse flow
    --profiler_mode 2 -- Run caffe2nnir2openvx Fuse flow
    --profiler_mode 3 -- Run caffe2nnir2openvx FP16 flow
    --profiler_mode 4 -- Run onnx2nnir2openvx No Fuse flow
    --profiler_mode 5 -- Run onnx2nnir2openvx Fuse flow
    --profiler_mode 6 -- Run onnx2nnir2openvx FP16 flow
    --profiler_mode 7 -- Run nnef2nnir2openvx No Fuse flow
    --profiler_mode 8 -- Run nnef2nnir2openvx Fuse flow
    --profiler_mode 9 -- Run nnef2nnir2openvx FP16 flow
--profiler_level    - NN Profile Batch Size in powers of 2: optional (default:7 [range:1 - N])
--miopen_find       - MIOPEN_FIND_ENFORCE mode: optional (default:1 [range:1 - 5])
```
