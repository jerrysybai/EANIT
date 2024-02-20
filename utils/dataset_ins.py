import json
from typing import Any
from loguru import logger
from torch.utils.data import Dataset
import os
import random
from transformers import AutoTokenizer, BitsAndBytesConfig

def levenshtein_distance(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    # 初始化距离矩阵
    matrix = [[0 for n in range(len_str2)] for m in range(len_str1)]

    for i in range(len_str1):
        matrix[i][0] = i

    for j in range(len_str2):
        matrix[0][j] = j

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,      # 删除
                               matrix[i][j - 1] + 1,      # 插入
                               matrix[i - 1][j - 1] + cost)  # 替换

    return matrix[len_str1 - 1][len_str2 - 1]/len_str1

class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_length, is_train = True, type = "NER", noise_rate = 0.8, sample_rate = None):
        self.type = type
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_len = max_seq_length
        self.is_train = is_train
        self.padding = True
        self.noise_rate = noise_rate
        if is_train:
            logger.info('noise rate: {}'.format(noise_rate))
        logger.info('Loading data: {}'.format(path))
        with open(path, encoding="utf-8") as f:
            self.example = json.load(f)
        if sample_rate is not None:
            self.example = self.example[:int(len(self.example) * sample_rate)]
        logger.info("there are {} data in dataset".format(len(self.example)))
           
    def __len__(self):
        return len(self.example)
    def __getitem__(self, index):
    # def __call__(self, index):
        data = self.example[index]
        instruction = data["instruction"]
        adv_instruction = data["adv_instruction"]
        question = data["question"]
        format = data["format"]
        option = data["option"]
        inputs = data["input"]
        output = data["output"]
        
        
        prob = random.uniform(0,1)
        if prob < self.noise_rate and self.is_train:
            all_input = "Instruction: " + adv_instruction + " \n" +"Question: "+ question + " \n" + "Format: "+ format + " \n" +"Option: "+ option + " \n"+ "Text: " + inputs+ " \n" + "Answer:"
            adv_input = "Instruction: " + adv_instruction + " \n" +"Question: "+ question + " \n" + "Format: "+ format + " \n" +"Option: "+ option + " \n"+ "Text: " + inputs+ " \n" + "Answer:"
        else:
            all_input = "Instruction: " + instruction + " \n" +"Question: "+ question + " \n" + "Format: "+ format + " \n" +"Option: "+ option + " \n"+ "Text: " + inputs+ " \n" + "Answer:"
            adv_input = "Instruction: " + instruction + " \n" +"Question: "+ question + " \n" + "Format: "+ format + " \n" +"Option: "+ option + " \n"+ "Text: " + inputs+ " \n" + "Answer:"
        output = " "+ output
        
        model_inputs = self.tokenizer(
                all_input,
                max_length=self.max_len,
                padding=self.padding,
                truncation=True,
            )
        
        noise_inpts = self.tokenizer(
                adv_input,
                max_length=self.max_len,
                padding=self.padding,
                truncation=True,
            )
        
        labels = self.tokenizer(
                output,
                max_length=self.max_len,
                padding=self.padding,
                truncation=True,
            )
        
        tokenized_input = model_inputs["input_ids"] + [self.eos_token_id]
        tokenized_noise = noise_inpts["input_ids"] + [self.eos_token_id]
        tokenized_label = labels["input_ids"][1:] + [self.eos_token_id]
        
        input_ids = tokenized_input + tokenized_label
        input_ids = input_ids[:self.max_len]
        test_input_ids = tokenized_input[:self.max_len]
        
        target_mask = [0] * min(self.max_len, len(tokenized_input)) + [1] * max(min(self.max_len - len(tokenized_input), len(tokenized_label)), 0)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        
        noise_ids = tokenized_noise + tokenized_label
        noise_ids = noise_ids[:self.max_len]
        noise_mask = [0] * min(self.max_len, len(tokenized_noise)) + [1] * max(min(self.max_len - len(tokenized_noise), len(tokenized_label)), 0)
        assert len(noise_ids) == len(noise_mask)
        

        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        noise_att_mask = [1] * len(noise_ids)
        assert len(noise_ids) == len(noise_mask) == len(noise_att_mask)
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
            'noise_ids': noise_ids,
            'noise_att_mask': noise_att_mask,
            'noise_mask': noise_mask,
            'test_input': test_input_ids,
            'label_ids': tokenized_label,
            'mention_pos':(len(tokenized_input)-1, len(tokenized_noise)-1),
            # "ins_len": ins_len
        }
        return inputs




# tokenizer = AutoTokenizer.from_pretrained(
#         "/cto_labs/baishengyuan/noise_llm_data/Llama-7B",
#         trust_remote_code=True,
#         # llama不支持fast
#         use_fast=False
#     )
# tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
# a = SFTDataset("/home/baishengyuan/project/noise_llm/code/noise_dataset/train.json", tokenizer, 512)
# a(1)