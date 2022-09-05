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

