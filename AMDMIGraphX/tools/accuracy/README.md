# AMD MIGraphX Accuracy checker
## Instructions
First ensure requirements and MIGraphX's python library are installed. Refer to MIGraphX instructions at the root directory to install the python library.
Use the command below to install remaining dependencies:
```
pip install -r requirements.txt
```
The accuracy checker will compare outputs from MIGraphX and onnx runtime. Therefore, an onnx file is required argument.
Example usage is below:
```
python accuracy_checker.py --onnx [path to onnx_file]
```

The output of the checker will either report as `PASSED` or `FAILED`. For detailed information,
the `--verbose` flag can be passed in to the command line which shows the mismatched elements between MIGraphX and onnx runtime.

By default, the tolerance is set to `1e-3`, but this can be changed by passing in `--tolerance [tolerance]`.
If the tolerance value is increased, then less accurate results from MIGraphX will be accepted.

For models that support variable batch sizes, use `--batch [batch_size]` to modify the batch size.

Random values are assigned to the model's inputs. However, they can be set to only contain 1s if the `--fill1` flag is passed in.
This is useful for verifying models such as bert which use integer datatypes.

By default, the CPU Execution Provider is used when running onnx runtime. If building onnx runtime with a different version, specify the provider using `--provider`.
