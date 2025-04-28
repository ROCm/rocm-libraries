# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.generic import ROCALNumpyIterator
import amd.rocal.fn as fn
import amd.rocal.types as types

import os
import cv2
from parse_config import parse_args


def draw_patches(image, idx, args=None):
    # image is expected as a numpy array
    if not args.NHWC:
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/numpy_reader/" + str(idx) + ".png", image)


def main():
    args = parse_args()
    # Args
    data_path = args.image_dataset_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    num_threads = args.num_threads
    random_seed = args.seed
    local_rank = args.local_rank
    world_size = args.world_size
    output_layout = types.NHWC if args.NHWC else types.NCHW

    try:
        path = "output_folder/numpy_reader/"
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as error:
        print(error)

    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads,
                        device_id=local_rank, seed=random_seed, rocal_cpu=rocal_cpu)

    with pipeline:
        numpy_reader_output = fn.readers.numpy(
            file_root=data_path, shard_id=local_rank, num_shards=world_size, output_layout=types.NHWC)
        resize_output = fn.resize(
            numpy_reader_output, resize_width=400, resize_height=400, output_layout=output_layout)
        pipeline.set_outputs(resize_output)

    pipeline.build()

    cnt = 0
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline)
    for epoch in range(args.num_epochs):
        print("epoch:: ", epoch)
        for i, [batch] in enumerate(numpyIteratorPipeline):
            if i == 0 and args.print_tensor:
                print(batch)
            for img in batch:
                draw_patches(img, cnt, args)
                cnt += 1
        numpyIteratorPipeline.reset()
    print("##############################  NUMPY READER SUCCESS  ############################")


if __name__ == '__main__':
    main()
