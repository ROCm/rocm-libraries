## Running Pets Training Example

### Building the required TF Rocm docker

* Use the instructions in the [docker section](https://github.com/ROCm/rocAL/docker) to build the required [Tensorflow docker](https://github.com/ROCm/rocAL/docker/rocal-with-tensorflow.dockerfile)
* Upgrade pip to the latest version.

### Running the training

* To setup dataset, run

```shell
bash download_and_preprocess_dataset.sh
```

* To run this example, just execute:

```shell
python3 train.py
```
