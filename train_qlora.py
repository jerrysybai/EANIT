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
import os
from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict
from torch.nn import functional as F
import numpy as np

from component.collator import SFTDataCollator
from component.dataset import SFTDataset, ChatGLM2SFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer
from component.loss import TargetLMLoss
from component.llama_model import Llama_seq2seq


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
    cls = bnb.nn.Linear4bit
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


def NEFTune(model, noise_alpha=5):
    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            # during training, we add noise to the embedding
            # during generation, we don't add noise to the embedding
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha/torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
            else:
                return orig_embed(x)
        return new_func
    ##### NOTE: this is for a LLaMA model ##### 
    ##### For a different model, you need to change the attribute path to the embedding #####
    model.base_model.model.model.embed_tokens.forward = noised_embed(model.base_model.model.base_model.embed_tokens, noise_alpha)
    return model

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

    training_args.ddp_find_unused_parameters = False
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
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
        # train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length, args.max_seq_length, 50, path = "data/RE")
        train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length)
        eval_dataset = SFTDataset(args.eval_file, tokenizer, args.max_seq_length)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    # # 部分tokenizer没有pad_token_id
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # # 部分tokenizer的pad_token_id与eos_token_id相同，如InternLM，会导致无法计算eos_token_id的loss。将pad_token_id设为unk_token_id
    # if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.unk_token_id is not None:
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # # 如果两者相同，模型训练时不会计算eos_token_id的loss
    # if tokenizer.pad_token_id == tokenizer.eos_token_id:
    #     raise Exception('pad_token_id should not be equal to eos_token_id')

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
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    # 初始化损失函数
    loss_func = TargetLMLoss(ignore_index=-100)

    def compute_metrics(pred_o):
        
        labels = np.array(pred_o.label_ids)
        preds = np.array(pred_o.predictions)
        labels = np.where(labels>0, labels, 0)
        preds = np.where(preds>0, preds, 0)
        label_all = []
        pred_all = []
        cor_tot = 0
        for i in range(preds.shape[0]):
            pred = preds[i].tolist()
            label = labels[i].tolist()
            response = tokenizer.decode(pred)
            response = response.strip().replace(tokenizer.eos_token, "").replace("<unk>", "").strip().split("; ")
            label = tokenizer.decode(label)
            label = label.strip().replace(tokenizer.eos_token, "").replace("<unk>", "").strip().split("; ")
            
            labels_sub = []
            preds_sub = []
            
            for l in label:
                if l != "None":
                    labels_sub.append(l)
                if l == "None":
                    label_all.append((i,"None","None"))
                    continue
                l_list = l.split(": ")
                if len(l_list) != 2:
                    continue
                label_all.append((i,l_list[0].replace(" ", ""),l_list[1].replace(" ", "")))
            for r in response:
                if r != "None":
                    preds_sub.append(r)
                if r == "None":
                    pred_all.append((i,"None","None"))
                    continue
                r_list = r.split(": ")
                if len(r_list) != 2:
                    continue
                if (i,r_list[0].replace(" ", ""),r_list[1].replace(" ", "")) not in pred_all:
                    pred_all.append((i,r_list[0].replace(" ", ""),r_list[1].replace(" ", "")))
            for pre_label in preds_sub:
                for it_label in labels_sub:
                    if pre_label.find(it_label) != -1:
                        cor_tot += 1
                        
        ner_tot_recall = len(label_all)
        tot_pred_tot = len(pred_all)
        
        cor_tot = 0
        for item in pred_all:
            if item in label_all:
                cor_tot += 1
        p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0 
        r = cor_tot / ner_tot_recall 
        f1_tot = 2 * (p * r) / (p + r) if cor_tot > 0 else 0.0
        ad = {'f1':  f1_tot, 'precision': p, 'recall': r}
        print(ad)    
        return {'f1':  f1_tot, 'precision': p, 'recall': r}

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
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()


