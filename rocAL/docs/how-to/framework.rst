.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

.. _framework:

********************************************************************
ML Framework Integration
********************************************************************

rocAL improves the pipeline efficiency by preprocessing the data and parallelizing the data loading on the CPU and running trainings on the GPU. To separate the data loading from the training, rocAL provides TensorFlow and PyTorch iterators and readers as a plugin. The integration process with PyTorch and TensorFlow is described in the sections below.

.. _pytorch:

PyTorch Integration
===========================

This section demonstrates how to use rocAL with PyTorch for training. Follow the steps below to get started. 

Build PyTorch Docker
--------------------------------

Build a rocAL PyTorch docker by following the steps here.

Create Data-loading Pipeline
----------------------------------------

Follow these steps:

1. Import libraries for `rocAL <https://github.com/ROCm/rocAL/blob/master/docs/examples/pytorch/toynet_training/train.py#L28>`_.

.. code-block:: python
   :caption: Import libraries

    from amd.rocal.plugin.pytorch import ROCALClassificationIterator
    from amd.rocal.pipeline import Pipeline
    import amd.rocal.fn as fn
    import amd.rocal.types as types


2. See a rocAL pipeline for PyTorch below. It reads data from the dataset using a fileReader and uses image_slice to decode the raw images. The other required augmentation operations are also defined in the `pipeline <https://github.com/ROCm/rocAL/blob/master/docs/examples/pytorch/toynet_training/train.py#L38>`_.

.. code-block:: python
   :caption: Pipeline for PyTorch

    def trainPipeline(data_path, batch_size, num_classes, one_hot, local_rank, world_size, num_thread, crop, rocal_cpu, fp16):
        pipe = Pipeline(batch_size=batch_size, num_threads=num_thread, device_id=local_rank, seed=local_rank+10, 
                    rocal_cpu=rocal_cpu, tensor_dtype = types.FLOAT16 if fp16 else types.FLOAT, tensor_layout=types.NCHW, 
                    prefetch_queue_depth = 7)
        with pipe:
            jpegs, labels = fn.readers.file(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
            rocal_device = 'cpu' if rocal_cpu else 'gpu'
            # decode = fn.decoders.image(jpegs, output_type=types.RGB,file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
            decode = fn.decoders.image_slice(jpegs, output_type=types.RGB,
                                                        file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
            res = fn.resize(decode, resize_x=224, resize_y=224)
            flip_coin = fn.random.coin_flip(probability=0.5)
            cmnp = fn.crop_mirror_normalize(res, device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mirror=flip_coin,
                                            image_type=types.RGB,
                                            mean=[0.485,0.456,0.406],
                                            std=[0.229,0.224,0.225])
            if(one_hot):
                _ = fn.one_hot(labels, num_classes)
            pipe.set_outputs(cmnp)
        print('rocal "{0}" variant'.format(rocal_device))
        return pipe


3. Import libraries for PyTorch.

.. code-block:: python
   :caption: Import libraries for PyTorch

   import torch.nn as nn
   import torch.nn.functional as F
   import torch.optim as optim


4. Call the training pipeline with rocAL classification data `loader <https://github.com/ROCm/rocAL/blob/master/docs/examples/pytorch/toynet_training/train.py#L78>`_.

.. code-block:: python
   :caption: Call the training pipeline

   Def get_pytorch_train_loader(self):
           print(“in get_pytorch_train_loader function”)   
           pipe_train = trainPipeline(self.data_path, self.batch_size, self.num_classes, self.one_hot, self.local_rank, 
                                       self.world_size, self.num_thread, self.crop, self.rocal_cpu, self.fp16)
           pipe_train.build()
           train_loader = ROCALClassificationIterator(pipe_train, device=”cpu” if self.rocal_cpu else “cuda”, device_id = self.local_rank)


5. Run the `training script <https://github.com/ROCm/rocAL/blob/master/docs/examples/pytorch/toynet_training/train.py#L179>`_.

.. code-block:: python
   :caption: Run the training pipeline

    # Training loop
        for epoch in range(10):  # loop over the dataset multiple times
            print(“\n epoch:: “,epoch)
            running_loss = 0.0

            for i, (inputs,labels) in enumerate(train_loader, 0):

                sys.stdout.write(“\r Mini-batch “ + str(i))
                # print(“Images”,inputs)
                # print(“Labels”,labels)
                inputs, labels = inputs.to(device), labels.to(device)


6. To see and run a sample training script, refer to `rocAL PyTorch example <https://github.com/ROCm/rocAL/tree/master/docs/examples/pytorch>`_.

.. _tensorflow:

TensorFlow Integration
===============================

This section demonstrates how to use rocAL with TensorFlow for training. Follow the steps below to get started. 

Build TensorFlow Docker
--------------------------------------

Build a rocAL TensorFlow docker by following the steps here.

Create Data-loading Pipeline
----------------------------------------

Follow these steps:

1. Import libraries for `rocAL_pybind <https://github.com/ROCm/rocAL/blob/master/docs/examples/tf/pets_training/train.py#L22>`_.

.. code-block:: python
   :caption: Import libraries

    from amd.rocal.plugin.tf import ROCALIterator
    from amd.rocal.pipeline import Pipeline
    import amd.rocal.fn as fn
    import amd.rocal.types as types


2. See a rocAL pipeline for TensorFlow below. It reads data from the TFRecords using TFRecord Reader and uses ``fn.decoders.image`` to decode the raw `images <https://github.com/ROCm/rocAL/blob/master/docs/examples/tf/pets_training/train.py#L128>`_.

.. code-block:: python
   :caption: Pipeline for TensorFlow

    trainPipe = Pipeline(batch_size=TRAIN_BATCH_SIZE, num_threads=1, rocal_cpu=RUN_ON_HOST, tensor_layout = types.NHWC)
        with trainPipe:
            inputs = fn.readers.tfrecord(path=TRAIN_RECORDS_DIR, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
            features={
                'image/encoded':tf.io.FixedLenFeature((), tf.string, ""),
                'image/class/label':tf.io.FixedLenFeature([1], tf.int64,  -1),
                'image/filename':tf.io.FixedLenFeature((), tf.string, "")
                }
                )
            jpegs = inputs["image/encoded"]
            images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=TRAIN_RECORDS_DIR)
            resized = fn.resize(images, resize_x=crop_size[0], resize_y=crop_size[1])
            flip_coin = fn.random.coin_flip(probability=0.5)
            cmn_images = fn.crop_mirror_normalize(resized, crop=(crop_size[1], crop_size[0]),
                                                mean=[0,0,0],
                                                std=[255,255,255],
                                                mirror=flip_coin,
                                                output_dtype=types.FLOAT,
                                                output_layout=types.NHWC,
                                                pad_output=False)
            trainPipe.set_outputs(cmn_images)
    trainPipe.build()


3. Import libraries for `TensorFlow <https://github.com/ROCm/rocAL/blob/master/docs/examples/tf/pets_training/train.py#L174>`_.

.. code-block:: python
   :caption: Import libraries for TensorFlow

    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
    import tensorflow_hub as hub
    Call the train pipeline
    trainIterator = ROCALIterator(trainPipe)  
    Run the training Session
    i = 0
        with tf.compat.v1.Session(graph = train_graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            while i < NUM_TRAIN_STEPS:


                for t, (train_image_ndArray, train_label_ndArray) in enumerate(trainIterator, 0):
                    train_label_one_hot_list = get_label_one_hot(train_label_ndArray)


4. To see and run a sample training script, refer to `rocAL TensorFlow example <https://github.com/ROCm/rocAL/tree/master/docs/examples/tf/pets_training>`_.

.. __resnet50:

Run Resnet50 classification training with rocAL
=======================================================

#. Ensure you have downloaded ``ILSVRC2012_img_val.tar`` (6.3GB) and ``ILSVRC2012_img_train.tar`` (138 GB) files and unzip into ``train`` and ``val`` folders
#. Build `rocAL Pytorch docker <https://github.com/ROCm/rocAL/blob/master/docker/README.md>`_ 

    * Run the docker image

    .. code-block:: shell 

        sudo docker run -it -v <Path-To-Data-HostSystem>:/data -v /<Path-to-GitRepo>:/dockerx -w /dockerx --privileged --device=/dev/kfd --device=/dev/dri --group-add video --shm-size=4g --ipc="host" --network=host <docker-name>

    .. note:: 
        Refer to the `docker <https://github.com/ROCm/rocAL#docker>`_ page for prerequisites and information on building the docker image. 

    Optional: Map localhost directory on the docker image

    * Option to map the localhost directory with imagenet dataset folder to be accessed on the docker image.
    * Usage: ``-v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH}``

#. To see and run a sample training script, refer to `rocAL Imagenet example <https://github.com/ROCm/rocAL/tree/master/docs/examples/pytorch/imagenet_training>`_.


