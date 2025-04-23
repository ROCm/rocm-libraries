# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

import random
from amd.rocal.plugin.pytorch import ROCALClassificationIterator

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import cv2
import os


def draw_patches(img, idx, device=True, layout="NCHW"):
    # image is expected as a tensor, bboxes as numpy
    import cv2
    if device is False:
        image = img.cpu().numpy()
    else:
        image = img.numpy()
    if layout == "NCHW":
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/file_reader/" +
                str(idx)+"_"+"train"+".png", image * 255)


def main():
    if len(sys.argv) < 3:
        print('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    try:
        path = "output_folder/file_reader/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    rocal_cpu = True if sys.argv[2] == "cpu" else False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    local_rank = 0
    world_size = 1

    image_classification_train_pipeline = Pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu)

    with image_classification_train_pipeline:
        jpegs, _ = fn.readers.file(file_root=data_path)
        decode = fn.decoders.image_slice(jpegs, output_type=types.RGB,
                                         file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        res = fn.resize(decode, resize_width=224, resize_height=224,
                        output_layout=types.NCHW, output_dtype=types.UINT8)
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(res,
                                        output_layout=types.NCHW,
                                        output_dtype=types.FLOAT,
                                        crop=(224, 224),
                                        mirror=flip_coin,
                                        mean=[0.485 * 255, 0.456 *
                                              255, 0.406 * 255],
                                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        image_classification_train_pipeline.set_outputs(cmnp)

# There are 2 ways to get the outputs from the pipeline
# 1. Use the iterator
# 2. use the pipe.run()

# Method 1
    # use the iterator
    image_classification_train_pipeline.build()
    imageIteratorPipeline = ROCALClassificationIterator(
        image_classification_train_pipeline)
    cnt = 0
    for i, it in enumerate(imageIteratorPipeline):
        print(it)
        print("************************************** i *************************************", i)
        for img in it[0]:
            cnt += 1
            draw_patches(img[0], cnt, device=rocal_cpu, layout="NCHW")
    imageIteratorPipeline.reset()
    print("*********************************************************************")

# Method 2
    iter = 0
    # use pipe.run() call
    output_data_batch = image_classification_train_pipeline.run()
    print("\n Output Data Batch: ", output_data_batch)
    # length depends on the number of augmentations
    for i in range(len(output_data_batch)):
        print("\n Output Layout: ", output_data_batch[i].layout())
        print("\n Output Dtype: ", output_data_batch[i].dtype())
        for image_counter in range(output_data_batch[i].batch_size()):
            image = output_data_batch[i].at(image_counter)
            image = image.transpose([1, 2, 0])
            cv2.imwrite("output_images_iter" + str(i) + str(image_counter) +
                        ".jpg", cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
