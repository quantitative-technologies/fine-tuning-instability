from aenum import extend_enum
from dataclasses import dataclass, field

from transformers.training_args import OptimizerNames, TrainingArguments

# Add support for our custom optimizer
extend_enum(OptimizerNames, 'ADAMW_L2SP', "adamw_l2sp")


@dataclass
class TrainingArgumentsL2SP(TrainingArguments):
    """
    In addition to TraininngArguments:

    Args:
        weight_decay1 (`float`, *optional*, defaults to 0):
            The weight decay to apply (if not zero) to the source layers in [`AdamWL2SP`] optimizer.
        weight_decay2 (`float`, *optional*, defaults to 0):
            The weight decay to apply (if not zero) to the novel layers in [`AdamWL2SP`] optimizer.
        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
            The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, adamw_l2sp or adafactor.

    """
    weight_decay1: float = field(default=0.0, metadata={"help": "Weight decay for L2-SP regularization, for source parameters."})
    weight_decay2: float = field(default=0.0, metadata={"help": "Weight decay for L2-SP regularization, for novel parameters."})
