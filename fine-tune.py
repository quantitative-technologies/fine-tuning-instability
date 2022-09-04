import logging
import numpy as np
import sys

import torch
from torch import nn
import datasets
from datasets import load_dataset, load_metric
import transformers
from transformers import AutoModelForPreTraining, AutoModelForSequenceClassification, AutoTokenizer, Trainer, set_seed
from transformers.trainer_pt_utils import get_parameter_names
from transformers_fine_tuning.optim.adamwl2sp import AdamWL2SP
from transformers_fine_tuning.transformers.trainer_optimizer_init import TrainerOptimizerInit
from transformers_fine_tuning.transformers.training_args_l2sp import TrainingArgumentsL2SP

# This must be True to use adamw_l2sp, because the custom optimizer is not natively supported by Trainer
USE_OPTIMIZER_INIT = True

# Hyperparameter settings from Mosbach, et. al. On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines
MODEL_NAME = 'albert-large-v2'
TASK = 'rte'
MAX_DATA_SIZE = None
MAX_SEQ_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
LR_SCHEDULE = 'linear'
WARMUP_RATIO = 0.1
OPTIMIZER = 'adamw_torch' # For AdamWL2SP Regularization use: 'adamw_l2sp'
ADAM_EPSILON = 1e-6
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
WEIGHT_DECAY1 = 0.5
WEIGHT_DECAY2 = 0.0
MAX_GRAD_NORM = 1.0
SEED = 10013
DATA_SEED = 20003
OUTPUT_DIR = 'output'
LOG_LEVEL = 'passive'
# This will be ignored if torch_xla is installed
DEVICE = 'cuda'  # Use 'cpu' for no cuda (slow!)


def main():
    # hyperparameters and other args for fine-tuning
    train_args = TrainingArgumentsL2SP(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        seed=SEED,
        data_seed=DATA_SEED,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULE,
        warmup_ratio=WARMUP_RATIO,
        optim=OPTIMIZER,        
        adam_epsilon=ADAM_EPSILON,
        adam_beta1 = ADAM_BETA1,
        adam_beta2 = ADAM_BETA2,
        weight_decay1=WEIGHT_DECAY1,
        weight_decay2=WEIGHT_DECAY2,
        max_grad_norm=MAX_GRAD_NORM,
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        log_level=LOG_LEVEL,
        do_eval=True,
        full_determinism=True,
        no_cuda=DEVICE != 'cuda'
    )

    # Set seed here before model initialization
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sp_model = AutoModelForPreTraining.from_pretrained(MODEL_NAME)
    # We need to manually place sp_model on the device to get accelerated learning
    try:
        import torch_xla.core.xla_model as xm
        sp_model = sp_model.to(xm.xla_device())
    except ModuleNotFoundError as e:
        sp_model = sp_model.to(DEVICE)

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

    train_dataset = raw_datasets["train"]
    if MAX_DATA_SIZE is not None:
        train_dataset = train_dataset.select(range(MAX_DATA_SIZE))
    eval_dataset = raw_datasets["validation"]

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
    
     # Setup logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

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
