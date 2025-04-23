# Exporting Frozen Graphs in Tensorflow 1

## Description
This example demonstrates how to export a frozen graph protobuf in Tensorflow 1.X that can be used as input to MIGraphX. Specifically, this is an example of exporting a frozen protobuf of a tensorflow BERT model. 

## How to Use this Example


In order to support bert from tensorflow's official [repository](https://github.com/google-research/bert), a serving_input_fn for the estimator must be implemented in [run_classifier.py](https://github.com/google-research/bert/blob/master/run_classifier.py). In this script, insert the following function after importing all libraries and setting up flags:

```
#...
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# insert function here
def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    }, default_batch_size=1)()
    return input_fn
```

Since we are passing dynamic shape placeholders in the serving_input_fn, the default_batch_size value will essentially determine the resulting shape in the graph.

For inference, we will focus on the "probabilities" layer's output, and we can name this layer by modifying the following [line](https://github.com/google-research/bert/blob/master/run_classifier.py#L608):

```
probabilities = tf.nn.softmax(logits, axis=-1, name="output")

```

Next, we need to export the saved model after training:
```
def main(_):
# ...
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n" 
        writer.write(output_line)
        num_written_lines += 1 
    assert num_written_lines == num_actual_predict_examples

# insert code here
  if FLAGS.do_train:  # optional to attach export to train flag
      estimator._export_to_tpu = False
      estimator.export_savedmodel('saved_models', serving_input_fn)
# ...
```

Run bert with the suggested arguments:
```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \ # change to appropriate size that fits on GPU
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
```

When running, search for the following lines in the output:
```
INFO:tensorflow:Restoring parameters from /tmp/model.ckpt-1603
INFO:tensorflow:SavedModel written to: saved_models/temp-1564086017/saved_model.pb
```

Note the ID followed by "temp-" (in this case, 1564086017). A directory should exist under saved_models/ that is named with the ID.

We also need to record the name of the output layer in bert. This can be done by inspecting the saved model.
```
saved_model_cli show --dir saved_models/1564086017 --tag_set serve --signature_def serving_default
```
The output should look like this:
```
The given SavedModel SignatureDef contains the following input(s):
  inputs['input_ids'] tensor_info:
      dtype: DT_INT32
      shape: (1, 128)
      name: input_ids_1:0
  inputs['input_mask'] tensor_info:
      dtype: DT_INT32
      shape: (1, 128)
      name: input_mask_1:0
  inputs['label_ids'] tensor_info:
      dtype: DT_INT32
      shape: (1)
      name: label_ids_1:0
  inputs['segment_ids'] tensor_info:
      dtype: DT_INT32
      shape: (1, 128)
      name: segment_ids_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['probabilities'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 2)
      name: loss/output:0
Method name is: tensorflow/serving/predict

```
Here the output name is given as "loss/output:0", but we will strip the ":0" from the end, as we are concerned with the node only.

We will use tensorflow's freeze graph utility script and the information gathered above to create the frozen protobuf file. 

```
CKPT_NUM=1603
MODEL_ID=1564086017
OUT_NAME=loss/output

cd /path/to/tensorflow
python tensorflow/python/tools/freeze_graph.py \
  --input_graph=/tmp/mrpc_model/graph.pbtxt \
  --input_binary=false \
  --input_checkpoint=/tmp/mrpc_model/model.ckpt-${CKPT_NUM} \
  --input_saved_model_dir=/path/to/bert/saved_models/${MODEL_ID} \
  --output_graph=/tmp/frozen_bert.pb \
  --output_node_names=${OUT_NAME}
```

The final output should be a frozen protobuf that is compatible with MIGraphX.