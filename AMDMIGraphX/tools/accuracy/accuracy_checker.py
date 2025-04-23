#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
import argparse
import numpy as np
import migraphx
import onnxruntime as ort
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'MIGraphX accuracy checker. Use to verify onnx files to ensure MIGraphX\'s output \
                                                  is within tolerance of onnx runtime\'s expected output.'
    )
    file_args = parser.add_argument_group(title='file type arguments')
    file_args.add_argument('--onnx', type=str, help='path to onnx file')
    file_args.add_argument('--tf', type=str, help='path to tf pb file')
    parser.add_argument('--provider',
                        type=str,
                        default='CPUExecutionProvider',
                        help='execution provider for onnx runtime \
                                (default = CPUExecutionProvider)')
    parser.add_argument('--batch',
                        type=int,
                        default=1,
                        help='batch size (if specified in onnx file)')
    parser.add_argument('--fill1',
                        action='store_true',
                        help='fill all arguments with a value of 1')
    parser.add_argument('--fill0',
                        action='store_true',
                        help='fill all arguments with a value of 0')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='quantize MIGraphX model to fp16')
    parser.add_argument('--argmax',
                        action='store_true',
                        help='use argmax for accuracy')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='show verbose information (for debugging)')
    parser.add_argument('--tolerance',
                        type=float,
                        default=1e-3,
                        help='accuracy tolerance (default = 1e-3)')
    parser.add_argument('--input-dim',
                        type=str,
                        action='append',
                        help='specify input parameter dimension \
                                with the following format --input-dim input_name:dim0,dim1,dim2...'
                        )
    parser.add_argument('--target',
                        type=str,
                        default='gpu',
                        help='target to compile and run MIGraphX on')

    parser.add_argument('--ort-run',
                        dest="ort_run",
                        action='store_true',
                        default=False,
                        help='only perform an onnxruntime run')

    parser.add_argument('--ort-logging',
                        dest="ort_logging",
                        action='store_true',
                        default=False,
                        help='Turn on ort VERBOSE logging via session options')

    parser.add_argument(
        '--disable-offload-copy',
        dest="offload_copy",
        action='store_false',
        default=True,
        help=
        'Disable offload copying (user must handle copy to and from device)')

    parser.add_argument(
        '--disable-fast-math',
        dest="fast_math",
        action='store_false',
        default=True,
        help='Disable fast math optimizations (etc: rewrite_gelu)')

    parser.add_argument('--exhaustive_tune',
                        dest="exhaustive_tune",
                        action='store_true',
                        default=False,
                        help='Enable exhaustive tuning for solutions')

    args = parser.parse_args()

    return args, parser


# taken from ../test_runner.py
def check_correctness(gold_outputs,
                      outputs,
                      rtol=1e-3,
                      atol=1e-3,
                      use_argmax=False,
                      verbose=False):
    if len(gold_outputs) != len(outputs):
        print('Number of outputs {} is not equal to expected number {}'.format(
            len(outputs), len(gold_outputs)))
        return False

    out_num = len(gold_outputs)
    ret = True

    if not use_argmax:
        for i in range(out_num):
            if not np.allclose(gold_outputs[i], outputs[i], rtol, atol):
                ret = False
                if verbose:
                    with np.printoptions(threshold=np.inf):
                        print('\nOutput {} is incorrect ...'.format(i))
                        print('Expected value: \n{}'.format(gold_outputs[i]))
                        print('\n......\n')
                        print('Actual value: \n{}\n'.format(outputs[i]))
                        diff = gold_outputs[i] - outputs[i]
                        max_diff = np.max(np.abs(diff))
                        print(f'Max Difference: {max_diff}')
                else:
                    print('Outputs do not match')
                    break
    else:
        golden_argmax = np.argmax(gold_outputs)
        actual_argmax = np.argmax(outputs)
        if actual_argmax != golden_argmax:
            ret = False
            print('\nOutput argmax is incorrect ...')
            if verbose:
                print('Expected argmax value: \n{}'.format(golden_argmax))
                print('......')
                print('Actual argmax value: \n{}\n'.format(actual_argmax))
    return ret


def get_np_datatype(in_type):
    datatypes = {
        'double_type': np.float64,
        'float_type': np.float32,
        'half_type': np.half,
        'int64_type': np.int64,
        'uint64_type': np.uint64,
        'int32_type': np.int32,
        'uint32_type': np.uint32,
        'int16_type': np.int16,
        'uint16_type': np.uint16,
        'int8_type': np.int8,
        'uint8_type': np.uint8,
        'bool_type': bool
    }
    return datatypes[in_type]


def main():
    args, parser = parse_args()

    use_onnx = True
    if args.onnx == None:
        use_onnx = False
    if not use_onnx and args.tf == None:
        print('Error: please specify either an onnx or tf pb file')
        parser.print_help()
        sys.exit(-1)

    model_name = args.onnx

    batch = args.batch

    custom_inputs = args.input_dim

    input_dims = {}
    if custom_inputs != None:
        for input in custom_inputs:
            input_dim = ''.join(input.split(':')[:-1])
            dims = [int(dim) for dim in input.split(':')[-1].split(',')]
            input_dims[input_dim] = dims

    if use_onnx:
        if not input_dims:
            model = migraphx.parse_onnx(model_name, default_dim_value=batch)
        else:
            model = migraphx.parse_onnx(model_name,
                                        default_dim_value=batch,
                                        map_input_dims=input_dims)
    else:
        model_name = args.tf

        if not input_dims:
            model = migraphx.parse_tf(model_name, batch_size=batch)
        else:
            model = migraphx.parse_tf(model_name,
                                      batch_size=batch,
                                      map_input_dims=input_dims)

    if (args.fp16):
        migraphx.quantize_fp16(model)

    if args.verbose:
        print(model)

    if not args.ort_run:
        model.compile(
            migraphx.get_target(args.target),
            offload_copy=args.offload_copy,
            fast_math=args.fast_math,
            exhaustive_tune=args.exhaustive_tune,
        )

    params = {}
    test_inputs = {}
    for name, shape in model.get_parameter_shapes().items():
        if args.verbose:
            print(f'Parameter {name} -> {shape}')
        in_shape = shape.lens()
        in_type = shape.type_string()
        if not args.fill1 and not args.fill0:
            test_input = np.random.rand(*(in_shape)).astype(
                get_np_datatype(in_type))
        elif not args.fill0:
            test_input = np.ones(in_shape).astype(get_np_datatype(in_type))
        else:
            test_input = np.zeros(in_shape).astype(get_np_datatype(in_type))
        test_inputs[name] = test_input
        migraphx_arg = migraphx.argument(test_input)
        if not args.offload_copy:
            migraphx_arg = migraphx.to_gpu(migraphx_arg)
        params[name] = migraphx_arg

    if not args.ort_run:
        if not args.offload_copy:
            pred_migx = np.array(migraphx.from_gpu(model.run(params)[-1]))
        else:
            pred_migx = np.array(model.run(params)[-1])

    if use_onnx:
        sess_op = ort.SessionOptions()

        if args.ort_logging:
            sess_op.log_verbosity_level = 0
            sess_op.log_severity_level = 0

        sess = ort.InferenceSession(model_name,
                                    sess_options=sess_op,
                                    providers=[args.provider])

        ort_params = {}
        for input in sess.get_inputs():
            ort_params[input.name] = test_inputs[input.name]

        try:
            pred_fw = sess.run(None, ort_params)[-1]
        except Exception as e:
            if any(input_dims):
                print(
                    'Error: custom input dim may not be compatible with onnx runtime'
                )
            raise e
    else:
        import tensorflow as tf

        def load_tf_graph(model_name):
            with tf.io.gfile.GFile(model_name, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            with tf.compat.v1.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def)
            return graph

        graph = load_tf_graph(model_name)
        is_nhwc = False
        graph_ops = []
        for op in graph.get_operations():
            graph_ops.append(op.name)
            if 'Conv' in op.node_def.op:
                if 'NHWC' in op.get_attr('data_format').decode('utf-8'):
                    is_nhwc = True
        graph_ops_set = set(graph_ops)
        tf_dict = {}

        for name in test_inputs.keys():
            # graph.get_operations() adds 'import/' to the op name
            tf_name = f'import/{name}'
            if tf_name not in graph_ops_set:
                continue
            x = graph.get_tensor_by_name(f'{tf_name}:0')
            tf_input = test_inputs[name]
            # transpose input for NHWC model
            if tf_input.ndim == 4 and is_nhwc:
                tf_dict[x] = np.transpose(tf_input, (0, 2, 3, 1))
            else:
                tf_dict[x] = tf_input

        # assume last node in graph is output
        # TODO: let user specify op name for output
        y = graph.get_tensor_by_name(f'{graph_ops[-1]}:0')

        with tf.compat.v1.Session(graph=graph) as sess:
            y_out = sess.run(y, feed_dict=tf_dict)
            pred_fw = y_out

    if not args.ort_run:
        is_correct = check_correctness(pred_fw, pred_migx, args.tolerance,
                                       args.tolerance, args.argmax,
                                       args.verbose)
        verbose_string = ' Rerun with --verbose for detailed information.' \
                if not args.verbose else ''
        if is_correct:
            print('PASSED: MIGraphX meets tolerance')
        else:
            print('FAILED: MIGraphX is not within tolerance.' + verbose_string)


if __name__ == '__main__':
    main()
