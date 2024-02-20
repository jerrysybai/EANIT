import os
os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true" 
# os.environ["WORLD_SIZE"] = "1"

from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)
import argparse
from loguru import logger
from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict
from torch.nn import functional as F
import numpy as np

from component.collator import SFTDataCollator
from component.dataset import  ChatGLM2SFTDataset
# from utils.dataset_new import SFTDataset, SFTDataset_all
from utils.dataset_ins import SFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer
from component.loss import TargetLMLoss
from component.llama_model import Llama_seq2seq
from utils.metrics import get_metrics
from utils.NEFT import AT_llama


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    # cls = bnb.nn.Linear4bit
    cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/qlora/qwen-7b-sft-qlora.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # logger.add(join(training_args.output_dir, 'train.log'))
    # logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')
    # 下面的设置至关重要，否则无法多卡训练
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # device_map = "auto"
    # # if we are in a distributed setting, we need to set the device map and max memory per device
    # if os.environ.get('LOCAL_RANK') is not None:
    #     local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    #     device_map = {'': local_rank}
    # p = os.environ.get('LOCAL_RANK')
    training_args.ddp_find_unused_parameters = False
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    device_map = "auto"

    # 加载模型
    if args.add_nosie:
        model = AT_llama.from_pretrained(
            args.model_name_or_path,
            # device_map={'':torch.cuda.current_device()},
            device_map= device_map,
            # load_in_4bit=True,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            ),
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            ),
            trust_remote_code=True,
        )
    
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    # ChatGLMTokenizer不需要设置，仅设置其他tokenizer
    elif tokenizer.__class__.__name__ != 'ChatGLMTokenizer':
        assert tokenizer.eos_token_id is not None
        assert tokenizer.bos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        
    # 指加载训练集
    if model.config.model_type == 'chatglm':
        train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length)
    else:
        train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length, is_train=True,type = args.task)
        eval_dataset = SFTDataset(args.eval_file, tokenizer, args.max_seq_length, is_train=True, type = args.task)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
    # 找到所有需要插入adapter的全连接层
    target_modules = find_all_linear_names(model)
    # 初始化lora配置
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # model = get_peft_model(model, config)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, '').eval()
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    # 初始化损失函数
    loss_func = TargetLMLoss(args, ignore_index=-100)

    compute_metrics = get_metrics(tokenizer, args.task)

    # 初始化Trainer
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func,
        compute_metrics = compute_metrics,
    )
    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    if False:
        train_result = trainer.train()
        # 保存最好的checkpoint
        final_save_path = join(training_args.output_dir, 'final')
        trainer.save_model(final_save_path)  # Saves the tokenizer too
        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    if True:
        trainer.evaluate()


if __name__ == "__main__":
    main()


