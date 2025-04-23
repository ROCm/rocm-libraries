# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

from argparse import ArgumentParser
import random


def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")

    common_group = parser.add_argument_group(
        'common', 'common-related options')
    # Data-related
    common_group.add_argument('--image-dataset-path', '-d', type=str,
                              help='image folder files')
    common_group.add_argument('--batch-size', '-b', type=int, default=10,
                              help='number of examples for each iteration')
    common_group.add_argument('--display', action="store_true",
                              help='--display:to display output from the pipeline')
    common_group.add_argument('--no-display', dest='display', action="store_false",
                              help='--no-display:to not display output from the pipeline')
    # case when none of the above is specified
    parser.set_defaults(display=False)
    common_group.add_argument('--max-height', '-mh', type=int, default=1000,
                              help='maximum height set during decoding')
    common_group.add_argument('--max-width', '-mw', type=int, default=1000,
                              help='maximum width set during decoding')
    common_group.add_argument('--color-format', '-c', type=int, default=1,
                              help='color format used during decoding')

    common_group.add_argument('--print_tensor', action="store_true",
                              help='--print_tensor: to print tensor output from the pipeline')
    common_group.add_argument('--no-print_tensor', dest='print_tensor', action="store_false",
                              help='--no-print_tensor: to not print tensor output from the pipeline')
    # case when none of the above is specified
    parser.set_defaults(print_tensor=False)

    common_group.add_argument('--classification', action="store_true",
                              help='--classification: to use for classification')
    common_group.add_argument('--no-classification', dest='classification', action="store_false",
                              help='--no-classification: to use for detection pipeline')
    # case when none of the above is specified
    parser.set_defaults(classification=True)

    common_group.add_argument('--rocal-gpu', default=False, action="store_true",
                              help='--use_gpu to use gpu')
    common_group.add_argument('--no-rocal-gpu', dest='rocal-gpu', action="store_false",
                              help='--no-rocal-gpu to use cpu backend')
    
    common_group.add_argument('--one-hot-encode', action="store_true",
                              help='--one-hot-encode: to use for one-hot-encoding of labels')
    common_group.add_argument('--no-one-hot-encode', dest='one-hot-encode', action="store_false",
                              help='--no-one-hot-encode: to used when we do not want to one hot encode the labels')
    # case when none of the above is specified
    parser.set_defaults(one_hot_encode=False)

    common_group.add_argument('--NHWC', action='store_true',
                              help='run input pipeline NHWC format')
    common_group.add_argument('--no-NHWC', dest='NHWC', action='store_false',
                              help='run input pipeline NCHW format')
    parser.set_defaults(NHWC=True)  # case when none of the above is specified

    common_group.add_argument('--fp16', default=False, action='store_true',
                              help='run input pipeline fp16 format')

    common_group.add_argument('--local-rank', type=int, default=0,
                              help='Device ID used by rocAL pipeline')
    common_group.add_argument('--world-size', '-w', type=int, default=1,
                              help='number of partitions to split the dataset')
    common_group.add_argument('--num-threads', '-nt', type=int, default=1,
                              help='number of CPU threads used by the rocAL pipeline.')
    common_group.add_argument('--num-epochs', '-e', type=int, default=1,
                              help='number of epochs to run')
    common_group.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                              help='manually set random seed')

    # unit_test.py related options
    python_unit_test = parser.add_argument_group(
        'python-unittest', 'python-unittest-related options')
    python_unit_test.add_argument('--reader-type', '-r', type=str, default="file",
                                  help='Reader used for reading and decoding the images')
    python_unit_test.add_argument('--augmentation-name', '-aug_name', type=str, default="resize",
                                  help='refer python unit test for all augmentation names ')
    python_unit_test.add_argument('--output-file-name', '-f', type=str, default="",
                                  help='file name to save the augmentation outputs')
    python_unit_test.add_argument('--interpolation-type', '-i', type=int, default=1,
                                  help='interpolation type used for resize and crop')
    python_unit_test.add_argument('--scaling-mode', '-sm', type=int, default=0,
                                  help='scaling mode type used for resize')

    # audio_unittests.py related options
    audio_unit_test = parser.add_argument_group(
        'audio-python-unittest', 'audio-python-unittest-related options')
    audio_unit_test.add_argument('--audio_path', type=str, default="",
                                  help='audio files path')
    audio_unit_test.add_argument('--file_list_path', type=str, default="",
                                  help='file list path')
    audio_unit_test.add_argument('--test_case', type=int, default=None,
                                  help='test case')
    audio_unit_test.add_argument('--qa_mode', type=int, default=1,
                                  help='enable qa mode to compare audio output with ref outputs')
    # coco_reader.py related options
    coco_reader = parser.add_argument_group(
        'coco-pipeline', 'coco-pipeline-related options')
    coco_reader.add_argument('--json-path', '-json-path', type=str,
                               help='coco dataset json path')
    # caffe_reader.py related options
    caffe_pipeline = parser.add_argument_group(
        'caffe-pipeline', 'caffe-pipeline-related options')
    caffe_pipeline.add_argument('--detection', '-detection', type=str,
                                help='detection')
    # video_pipeline.py related options
    video_pipeline = parser.add_argument_group(
        'video-pipeline', 'video-pipeline-related options')
    video_pipeline.add_argument('--video-path', '-video-path', type=str,
                                help='video path')
    video_pipeline.add_argument('--sequence-length', '-sequence-length', type=int,
                                help='video path')
    # web_dataset_reader.py related options
    web_dataset_reader = parser.add_argument_group(
        'webdataset-pipeline', 'webdataset-pipeline-related options')
    web_dataset_reader.add_argument('--index-path', '-index-path', type=str,
                               help='web dataset index path')

    return parser.parse_args()
