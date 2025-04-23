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
import tensorflow as tf
import amd.rocal.fn as fn
import numpy as np
import os
from parse_config import parse_args


def get_onehot(image_labels_array, num_classes):
    one_hot_vector_list = []
    for label in image_labels_array:
        one_hot_vector = np.zeros(num_classes)
        if label[0] != 0:
            np.put(one_hot_vector, label[0] - 1, 1)
        one_hot_vector_list.append(one_hot_vector)

    one_hot_vector_array = np.array(one_hot_vector_list)

    return one_hot_vector_array


def get_weights(num_bboxes):
    weights_array = np.zeros(num_bboxes)
    for pos in list(range(num_bboxes)):
        np.put(weights_array, pos, 1)
    return weights_array


def draw_patches(img, idx, bboxes, device, args=None):
    import cv2
    args = parse_args()
    # converting to numpy for opencv display since iterator returns tf.tensor
    img = img.numpy()
    if not args.NHWC:
        img = img.transpose([0, 1, 2])
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image = cv2.normalize(image, None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
            (loc_[2])), int((loc_[3]))), color, thickness)
        cv2.imwrite("output_folder/tf_reader/detection/" +
                    str(idx) + "_" + "train" + ".png", image)


def main():
    args = parse_args()
    # Args
    image_path = args.image_dataset_path
    num_classes = 91
    rocal_cpu = False if args.rocal_gpu else True
    device = "cpu" if rocal_cpu else "gpu"
    batch_size = args.batch_size
    num_threads = args.num_threads
    tensor_layout = types.NHWC if args.NHWC else types.NCHW
    tf_record_reader_type = 1
    feature_key_map = {
        'image/encoded': 'image/encoded',
        'image/class/label': 'image/object/class/label',
        'image/class/text': 'image/object/class/text',
        'image/object/bbox/xmin': 'image/object/bbox/xmin',
        'image/object/bbox/ymin': 'image/object/bbox/ymin',
        'image/object/bbox/xmax': 'image/object/bbox/xmax',
        'image/object/bbox/ymax': 'image/object/bbox/ymax',
        'image/filename': 'image/filename'
    }
    try:
        path = "output_folder/tf_reader/detection"
        is_exist = os.path.exists(path)
        if not is_exist:
            os.makedirs(path)
    except OSError as error:
        print(error)

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,
                    device_id=args.local_rank, seed=2, rocal_cpu=rocal_cpu)
    with pipe:
        inputs = fn.readers.tfrecord(path=image_path, reader_type=tf_record_reader_type, user_feature_key_map=feature_key_map,
                                     features={
                                         'image/encoded': tf.io.FixedLenFeature((), tf.string, ""),
                                         'image/class/label': tf.io.FixedLenFeature([1], tf.int64,  -1),
                                         'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                                         'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                                         'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                                         'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                                         'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                                         'image/filename': tf.io.FixedLenFeature((), tf.string, "")
                                     }
                                     )
        jpegs = inputs["image/encoded"]
        _ = inputs["image/class/label"]
        decoded_images = fn.decoders.image_random_crop(jpegs, user_feature_key_map=feature_key_map, output_type=types.RGB,
                                                       random_aspect_ratio=[
                                                           0.8, 1.25],
                                                       random_area=[0.1, 1.0],
                                                       num_attempts=100, path=image_path)
        resized = fn.resize(decoded_images, resize_width=300,
                            resize_height=300, output_layout=tensor_layout)
        pipe.set_outputs(resized)
    pipe.build()
    image_iterator = ROCALIterator(pipe, device=device)

    cnt = 0
    for i, ([images_array], bboxes_array, labels_array, num_bboxes_array) in enumerate(image_iterator, 0):
        print("ROCAL augmentation pipeline - Processing batch %d....." % i)

        for element in list(range(batch_size)):
            cnt += 1
            if args.print_tensor:
                print("Processing image %d....." % element)
            features_dict = {
                "image": images_array[element],
                "true_image_shape": np.array([len(images_array[element]), len(images_array[element, 0]), len(images_array[element, 0, 0])])
            }
            labels_dict = {
                "num_groundtruth_boxes": num_bboxes_array[element],
                "groundtruth_boxes": bboxes_array[element],
                "groundtruth_classes": get_onehot(labels_array[element], num_classes),
                "groundtruth_weights": get_weights(num_bboxes_array[element])
            }
            processed_tensors = (features_dict, labels_dict)
            if args.print_tensor:
                print("\nPROCESSED_TENSORS:\n", processed_tensors)
            draw_patches(images_array[element], cnt,
                         bboxes_array[element], device, args=args)
        print("\n\nPrinted first batch with", (batch_size), "images!")
        break
    image_iterator.reset()

    print("##############################  TF DETECTION  SUCCESS  ############################")


if __name__ == "__main__":
    main()
