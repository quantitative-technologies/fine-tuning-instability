#!/bin/bash

# Our experiments were done with torch-1.12.0, datasets-1.11.0, transformers tag v4.20.1 and the torch_xla-1.12 wheel below.

pip install torch==1.12.0
pip install datasets==1.11.0

# Check for TPU
if [ "$1" == 'TPU' ]
then
    printf "\nUsing TPU. Installing torch_xla...\n"
    pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl
else
    printf "\nNot using TPU\n"
fi

# Additional dependency for our library
pip install aenum

git clone https://github.com/huggingface/transformers.git
cd transformers; git checkout tags/v4.20.1; pip install -e .; cd ..

export PYTHONPATH=$PYTHONPATH:$PWD/transformers/src:$PWD/src





