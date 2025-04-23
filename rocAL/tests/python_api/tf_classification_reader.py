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

from amd.rocal.plugin.tf import ROCALIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types
import os
import amd.rocal.fn as fn
import tensorflow as tf
import numpy as np
from parse_config import parse_args


def draw_patches(img, idx, device, args=None):
    import cv2
    args = parse_args()
    # converting to numpy for opencv display since iterator returns tf.tensor
    img = img.numpy()
    if not args.NHWC:
        img = img.transpose([0, 1, 2])
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/tf_reader/classification/" +
                str(idx) + "_" + "train" + ".png", image)


def main():
    args = parse_args()
    # Args
    image_path = args.image_dataset_path
    rocal_cpu = False if args.rocal_gpu else True
    device = "cpu" if rocal_cpu else "gpu"
    batch_size = args.batch_size
    one_hot_labels = 1 if args.one_hot_encode else 0
    num_threads = args.num_threads
    tensor_layout = types.NHWC if args.NHWC else types.NCHW
    tf_record_reader_type = 0
    feature_key_map = {
        'image/encoded': 'image/encoded',
        'image/class/label': 'image/class/label',
        'image/filename': 'image/filename'
    }
    try:
        path = "output_folder/tf_reader/classification/"
        is_exist = os.path.exists(path)
        if not is_exist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,
                    device_id=args.local_rank, seed=2, rocal_cpu=rocal_cpu)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        inputs = fn.readers.tfrecord(path=image_path, reader_type=tf_record_reader_type, user_feature_key_map=feature_key_map,
                                     features={
                                         "image/encoded": tf.io.FixedLenFeature((), tf.string, ""),
                                         "image/class/label": tf.io.FixedLenFeature([1], tf.int64, -1),
                                         "image/filename": tf.io.FixedLenFeature((), tf.string, "")
                                     }
                                     )
        jpegs = inputs["image/encoded"]
        images = fn.decoders.image(
            jpegs, user_feature_key_map=feature_key_map, output_type=types.RGB, path=image_path)
        resized = fn.resize(images, resize_width=300,
                            resize_height=300, output_layout=tensor_layout)
        if one_hot_labels == 1:
            labels = inputs["image/class/label"]
            _ = fn.one_hot(labels, num_classes=1000)
        pipe.set_outputs(resized)
    # Build the pipeline
    pipe.build()
    # Dataloader
    image_iterator = ROCALIterator(pipe, device=device)
    cnt = 0
    # Enumerate over the Dataloader
    for i, ([images_array], labels_array) in enumerate(image_iterator, 0):
        if args.print_tensor:
            print("\n", i)
            print("lables_array", labels_array)
            print("\n\nPrinted first batch with", (batch_size), "images!")
        for element in list(range(batch_size)):
            cnt += 1
            draw_patches(images_array[element], cnt, device, args=args)
        break
    image_iterator.reset()

    print("##############################  TF CLASSIFICATION  SUCCESS  ############################")


if __name__ == "__main__":
    main()
