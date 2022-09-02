import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import Trainer 
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


class TrainerOptimizerInit(Trainer):
    """
    Args:
        optimizers_init (`Tuple[Callable[[Union[PreTrainedModel, nn.Module], TrainingArguments], torch.optim.Optimizer], 
                                torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple containing (1) a function that is
            used to create an optimizer from the `model` and `args`, and (2) the scheduler to use. Will default to an 
            instance of [`AdamW`] on your model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled 
            by `args`.
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers_init: Tuple[Callable[[Union[PreTrainedModel, nn.Module], TrainingArguments], torch.optim.Optimizer], 
                               torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(model=model, 
                         args=args, 
                         data_collator=data_collator, 
                         train_dataset=train_dataset, 
                         eval_dataset=eval_dataset, 
                         tokenizer=tokenizer, 
                         model_init=model_init, 
                         compute_metrics=compute_metrics, 
                         callbacks=callbacks,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)

        self.optimizer_init, self.lr_scheduler = optimizers_init

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can subclass and override 
        this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            if self.optimizer_init is None:
                # fall back to original behaviour
                return super().create_optimizer()

            self.optimizer = self.optimizer_init(opt_model, self.args)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer