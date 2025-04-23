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

import os, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import argparse
import sys

from amd.rocal.plugin.tf import ROCALIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

def main():
    parser = argparse.ArgumentParser(
        description='Tensorflow pets training arguments')
    parser.add_argument(
        '-b',
        '--backend',
        type=str,
        help='run rocal on cpu/gpu; optional; default: cpu',
        default='cpu',
        required=False)
    parser.add_argument(
        '-d',
        '--device-id',
        type=int,
        help='GPU device ID; optional; default: 0',
        default=0,
        required=False)
    parser.add_argument(
        '-bs',
        '--batch-size',
        type=int,
        help='batch size to run training; optional; default: 8',
        default=8,
        required=False)
    parser.add_argument(
        '-c',
        '--num-classes',
        type=int,
        help='Number of classes in the dataset; ; optional; default: 37',
        default=37,
        required=False)
    parser.add_argument(
        '-l',
        '--learning-rate',
        type=float,
        help='Learning rate for training; optional; default: 0.005',
        default=0.005,
        required=False)
    parser.add_argument(
        '-dir',
        '--records-dir',
        type=str,
        help='Path for tf records; optional',
        default='tf_pets_records/',
        required=False)
    
    try:
        args = parser.parse_args()
    except BaseException:
        sys.exit()

    train_records_dir = args.records_dir + 'train/'
    val_records_dir = args.records_dir + 'val/'

    num_classes = args.num_classes
    learning_rate = args.learning_rate
    train_batch_size = args.batch_size
    rocal_cpu = True if (args.backend == "cpu") else False
    device = "cpu" if rocal_cpu else "gpu"
    device_id = args.device_id

    print("\n-----------------------------------------------------------------------------------------")
    print('TF records (train) are located in %s' % train_records_dir)
    print('TF records (val) are located in %s' % val_records_dir)
    print("-----------------------------------------------------------------------------------------\n")

    image_size = [128, 128, 3]
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_size,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes)
    ])

    model.summary()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['acc'])

    TFRecordReaderType = 0
    featureKeyMap = {
        'image/encoded': 'image/encoded',
        'image/class/label': 'image/object/class/label',
        'image/filename': 'image/filename'
    }

    trainPipe = Pipeline(batch_size=train_batch_size, num_threads=8, rocal_cpu=rocal_cpu, device_id=device_id, prefetch_queue_depth=6,
                         tensor_layout=types.NHWC, mean=[0, 0, 0], std=[255, 255, 255], tensor_dtype=types.FLOAT)
    with trainPipe:
        inputs = fn.readers.tfrecord(path=train_records_dir, reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
                                     features={
                                         'image/encoded': tf.io.FixedLenFeature((), tf.string, ""),
                                         'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                                         'image/filename': tf.io.FixedLenFeature((), tf.string, "")
                                     }
                                     )
        jpegs = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        images = fn.decoders.image(
            jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=train_records_dir)
        resized = fn.resize(
            images, resize_width=image_size[0], resize_height=image_size[1])
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmn_images = fn.crop_mirror_normalize(resized, crop=(image_size[1], image_size[0]),
                                              mean=[127.5, 127.5, 127.5],
                                              std=[127.5, 127.5, 127.5],
                                              mirror=flip_coin,
                                              output_dtype=types.FLOAT,
                                              output_layout=types.NHWC)
        trainPipe.set_outputs(cmn_images)
    trainPipe.build()

    valPipe = Pipeline(batch_size=train_batch_size, num_threads=8,
                       rocal_cpu=rocal_cpu, device_id=device_id, prefetch_queue_depth=6, tensor_layout=types.NHWC, tensor_dtype=types.FLOAT)
    with valPipe:
        inputs = fn.readers.tfrecord(path=val_records_dir, reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
                                     features={
                                         'image/encoded': tf.io.FixedLenFeature((), tf.string, ""),
                                         'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                                         'image/filename': tf.io.FixedLenFeature((), tf.string, "")
                                     }
                                     )
        jpegs = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        images = fn.decoders.image(
            jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=val_records_dir)
        resized = fn.resize(
            images, resize_width=image_size[0], resize_height=image_size[1])
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmn_images = fn.crop_mirror_normalize(resized, crop=(image_size[1], image_size[0]),
                                              mean=[127.5, 127.5, 127.5],
                                              std=[127.5, 127.5, 127.5],
                                              mirror=flip_coin,
                                              output_dtype=types.FLOAT,
                                              output_layout=types.NHWC)
        valPipe.set_outputs(cmn_images)
    valPipe.build()

    trainIterator = ROCALIterator(trainPipe, device=device)
    valIterator = ROCALIterator(valPipe, device=device)

    # Create the metrics
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    epoch = 0
    train_batches = math.ceil(len(trainIterator) / train_batch_size)
    val_batches = math.ceil(len(valIterator) / train_batch_size)
    while epoch < 10:
        print('Epoch :', epoch + 1)
        accuracy_metric.reset_state()
        pbar = tf.keras.utils.Progbar(target=train_batches, stateful_metrics=[])
        step = 0
        for ([train_image_ndArray], train_label_ndArray) in trainIterator:
            train_label_ndArray = train_label_ndArray - 1
            with tf.GradientTape() as tape:
                prediction = model(train_image_ndArray, training=True)
                loss = loss_fn(train_label_ndArray, prediction)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables))
                accuracy_metric.update_state(train_label_ndArray, prediction)
                results = {'loss': loss, 'train_acc': accuracy_metric.result()}
                step += 1
                pbar.update(step, values=results.items(), finalize=False)
        pbar.update(step, values=results.items(), finalize=True)
        trainIterator.reset()
        accuracy_metric.reset_states()
        pbar = tf.keras.utils.Progbar(target=val_batches, stateful_metrics=[])
        step = 0
        for ([val_image_ndArray], val_label_ndArray) in valIterator:
            val_label_ndArray = val_label_ndArray - 1
            prediction = model(val_image_ndArray, training=False)
            accuracy_metric.update_state(val_label_ndArray, prediction)
            results = {'val_acc': accuracy_metric.result()}
            step += 1
            pbar.update(step, values=results.items(), finalize=False)
        pbar.update(step, values=results.items(), finalize=True)
        valIterator.reset()
        epoch += 1


if __name__ == '__main__':
    main()

######################################## NO CHANGES IN CODE NEEDED ########################################
