#!/bin/bash

pip install cloud-tpu-client==0.10 torch==1.12.0 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl
pip install datasets==1.11.0

git clone https://github.com/huggingface/transformers.git
cd transformers; git checkout tags/v4.0.1; pip install -e .

export PYTHONPATH=$PYTHONPATH:$PWD/transformers/src:$PWD/src





