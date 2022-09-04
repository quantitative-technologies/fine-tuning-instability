import numpy as np

import torch
from torch import nn
import torch_xla.core.xla_model as xm
from datasets import load_dataset, load_metric
from transformers import AutoModelForPreTraining, AutoModelForSequenceClassification, AutoTokenizer, Trainer, set_seed
from transformers.trainer_pt_utils import get_parameter_names

from transformers_fine_tuning.optim.adamwl2sp import AdamWL2SP
from transformers_fine_tuning.transformers.trainer_optimizer_init import TrainerOptimizerInit
from transformers_fine_tuning.transformers.training_args_l2sp import TrainingArgumentsL2SP

# This must be True to use adamw_l2sp, because the custom optimizer is not natively supported by Trainer
USE_OPTIMIZER_INIT = True

MODEL_NAME = 'albert-large-v2'
TASK = 'rte'
MAX_DATA_SIZE = None
MAX_SEQ_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
OPTIMIZER = 'adamw_torch' # For AdamWL2SP Regularization use: 'adamw_l2sp'
ADAM_EPSILON = 1e-6
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
WEIGHT_DECAY1 = 0.5
WEIGHT_DECAY2 = 0.0
SEED = 10013
DATA_SEED = 20000
OUTPUT_DIR = 'output'


def main():
    train_args = TrainingArgumentsL2SP(
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        seed=SEED,
        data_seed=DATA_SEED,
        optim=OPTIMIZER,        
        adam_epsilon=ADAM_EPSILON,
        adam_beta1 = ADAM_BETA1,
        adam_beta2 = ADAM_BETA2,
        weight_decay1=WEIGHT_DECAY1,
        weight_decay2=WEIGHT_DECAY2,
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        do_eval=True,
        full_determinism=True
    )

    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sp_model = AutoModelForPreTraining.from_pretrained(MODEL_NAME)
    sp_model = sp_model.to(xm.xla_device())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_datasets = load_dataset("glue", TASK)
    metric = load_metric("glue", TASK)

    def compute_metrics(p):
        preds = p.predictions
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples['sentence1'], examples['sentence2'])
        )
        return tokenizer(*args, padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True
    )

    # Warning affects randomness
    train_dataset = raw_datasets["train"]
    if MAX_DATA_SIZE is not None:
        train_dataset = train_dataset.select(range(MAX_DATA_SIZE))
    eval_dataset = raw_datasets["validation"]

    # for index in random.sample(range(len(train_dataset)), 3):
    #     print(f"Sample {index} of the training set: {train_dataset[index]}.")
    def adamw_init(model, train_args):
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [
            name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": train_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters,
                                 lr=train_args.learning_rate,
                                 betas=(train_args.adam_beta1,
                                        train_args.adam_beta2),
                                 eps=train_args.adam_epsilon)

    def adamw_l2sp_init(model, train_args):
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [
            name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "param_names": [n for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decays": (train_args.weight_decay1, train_args.weight_decay2),
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "param_names": [n for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decays": (0.0, 0.0),
            },
        ]

        # Get the SP model parameters
        sp_params = list(sp_model.named_parameters())

        return AdamWL2SP(
            optimizer_grouped_parameters,
            sp_params,
            lr=train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            eps=train_args.adam_epsilon
        )

    OPTIMIZER_INIT = adamw_init if OPTIMIZER == 'adamw_torch' else adamw_l2sp_init if OPTIMIZER == 'adamw_l2sp' else None

    if USE_OPTIMIZER_INIT:
        trainer = TrainerOptimizerInit(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            optimizers_init=(OPTIMIZER_INIT, None))
    else:
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer)

    trainer.train()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
