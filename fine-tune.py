import math
import numpy as np
import random
import site

#site.addsitedir('/content/transformers/src')
import torch
from torch import nn
import torch_xla.core.xla_model as xm
from datasets import load_dataset, load_metric
from transformers import AutoModel, AutoModelForPreTraining, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import has_length

from fine_tune.optim.adamwl2sp import AdamWL2SP
from fine_tune.transformers.trainer_optimizer_init import TrainerOptimizerInit
from fine_tune.transformers.training_args_l2sp import TrainingArgumentsL2SP

USE_OPTIMIZER_INIT = True

MODEL_NAME = 'albert-large-v2'
TASK = 'rte'
MAX_DATA_SIZE = None
MAX_SEQ_LENGTH = 128
EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 4e-5
WEIGHT_DECAY1 = 200.0
WEIGHT_DECAY2 = 0.0
SEED = 10000
DATA_SEED = 20000
OPTIMIZER = 'adamw_l2sp'
ADAM_EPSILON = 1e-6
OUTPUT_DIR = 'output'

train_args = TrainingArgumentsL2SP(
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    seed=SEED,
    data_seed=DATA_SEED,
    optim=OPTIMIZER,
    weight_decay1=WEIGHT_DECAY1,
    weight_decay2=WEIGHT_DECAY2,
    adam_epsilon=ADAM_EPSILON,
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

# for p in sp_model.parameters():
#     p.requires_grad = False

torch.save(sp_model, 'sp_model.pt')

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
    batched=True)

# Warning affects randomness
train_dataset = raw_datasets["train"]
if MAX_DATA_SIZE is not None:
    train_dataset = train_dataset.select(range(MAX_DATA_SIZE))
eval_dataset = raw_datasets["validation"]

# for index in random.sample(range(len(train_dataset)), 3):
#     print(f"Sample {index} of the training set: {train_dataset[index]}.")
def adamw_init(model, train_args):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
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
                             betas=(train_args.adam_beta1, train_args.adam_beta2),
                             eps=train_args.adam_epsilon)

def adamw_l2sp_init(model, train_args):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
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

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

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

# Create adamw_torch optimizer manually

if False:
    # Get the # of training samples
    train_dataloader = trainer.get_train_dataloader()

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = train_args.train_batch_size * train_args.gradient_accumulation_steps * train_args.world_size

    len_dataloader = len(train_dataloader)
    num_update_steps_per_epoch = len_dataloader // train_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_examples = trainer.num_examples(train_dataloader)
    if train_args.max_steps > 0:
        max_steps = train_args.max_steps
        num_train_epochs = train_args.max_steps // num_update_steps_per_epoch + int(
            train_args.max_steps % num_update_steps_per_epoch > 0
        )
        # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
        # the best we can do.
        num_train_samples = train_args.max_steps * total_train_batch_size
    else:
        max_steps = math.ceil(train_args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(train_args.num_train_epochs)
        num_train_samples = trainer.num_examples(train_dataloader) * train_args.num_train_epochs

    lr_scheduler = get_scheduler(
        train_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=train_args.get_warmup_steps(max_steps),
        num_training_steps=max_steps,
    )

    trainer.optimizer, trainer.lr_scheduler = optimizer, lr_scheduler
trainer.train()

sp_model2 = torch.load('sp_model.pt')

assert sp_model.state_dict().__str__() == sp_model2.state_dict().__str__()

compare_models(sp_model, sp_model2)
