# fine-tuning-instability


## Fine-Tuning Instability for Large Language Models


This repository contains the implementation for the `AdamWL2SP` optimizer, as 
described in this [blog post](https://quantitative-technologies.github.io/fine-tuning-instability/).
`AdamWL2SP` is the adaptive moment estimation (`Adam`) optimizer with decoupled 
weight decay and $L^2-\mathrm{SP}$ regularization.

The optimizer is implemented in [`src/transformers_fine_tuning/optim/adamwl2sp.py`](https://github.com/quantitative-technologies/fine-tuning-instability/blob/master/src/transformers_fine_tuning/optim/adamwl2sp.py) and is based on 
the PyTorch implementation of [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

[`src/transformers_fine_tuning/transformers/trainer_optimizer_init.py`](https://github.com/quantitative-technologies/fine-tuning-instability/blob/master/src/transformers_fine_tuning/transformers/trainer_optimizer_init.py) 
is a subclass of `Trainer` from the [🤗](https://huggingface.co) `transformers` library that 
facilitates custom optimizers, such as our `AdamWL2SP`. It is not strictly 
necessary but we prefer the design to `Trainer`. (In case of further interest, 
see [Passing optimizer to Trainer constructor does not work #18635](https://github.com/huggingface/transformers/issues/18635#issue-1339386290).)

The example script `fine-tune.py` demonstrates using our code to fine-tune 
ALBERT on the RTE task, using optimizers such as `AdamW` from `torch`, or our 
custom `AdamWL2SP` optimizer. The hyperparameters are set the same as were used
in our experiments. The model, optimizer, task, random seeds and hyperparameters
can be modified by setting the appropriate global variables in the script. 

It will work with `CPU`, `GPU` via `cuda` and `TPU` via `torch_xla`, with optional
concurrency if multiple `TPU` cores are available.

Note that this is not the actual script that was used to run our experiments, which
performs additional tracking of the metrics. For given seeds the results will
not reproduce those reported in the blog. However, a series of fine-tuning runs
with `fine-tune.py` should produce qualitatively similar results.

## Install

The python module itself `transformers_fine_tuning` is in the subdirectory [`src/transformers_fine_tuning`](https://github.com/quantitative-technologies/fine-tuning-instability/tree/master/src/transformers_fine_tuning).

In order to prepare an environment to run the example python script 
`fine-tune.py`, clone this repository and run

```console
source setup.sh
```

in the console, which will install dependencies and set up environment variables.

## Usage

For fine-tuning without concurrency simply run in the console:

```console
python fine-tune.py
```

It will automatically use the `TPU` processor if one is available.

For multi-core `TPU` environments, concurrent training can be done as
follows. For example, if 8 cores are available:

```console
python transformers/examples/pytorch/xla_spawn.py --num_cores 8 fine-tune.py
```

For fixed random seeds, concurrent training will not replicate a non-concurrent
training run.


