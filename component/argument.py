from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})


@dataclass
class QLoRAArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    task_type: str = field(default="", metadata={"help": "预训练任务：[sft, pretrain]"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    add_nosie: Optional[bool] = field(default=False, metadata={"help": "add noise"})
    task: Optional[str] = field(default="", metadata={"help": "the task name RE/NER"})
    noise_var: Optional[float] = field(default=1e-5, metadata={"help": "lora rank"})
    noise_gamma: Optional[float] = field(default=1e-6, metadata={"help": "1e-4 (default), eps for adversarial copy training."})
    adv_step_size: Optional[float] = field(default=1e-3, metadata={"help": "1 (default), perturbation size for adversarial training."})
    project_norm_type: Optional[str] = field(default="inf", metadata={"help": "the task name RE/NER"})
    useKL: Optional[bool] = field(default=False, metadata={"help": "use KL"})
    noise_rate: Optional[float] = field(default=0.8, metadata={"help": "noise rate"})
    # local_rank: Optional[int] = field(default=2, metadata={"help": "lora alpha"})
