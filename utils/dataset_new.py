import json
from typing import Any
from loguru import logger
from torch.utils.data import Dataset
import os
import random
from transformers import AutoTokenizer, BitsAndBytesConfig
# from noise_data import replace_instruction, random_pad_instruction, cut_instruction, cut_mix_instruction, opposit_instruction

global_instructions = {
    "NER":"Please list all entity words in the text that fit the category.Output format is \"type1: word1; type2: word2\". \n",
    "RE":"Given a phrase that describes the relationship between two words, extract the words and the lexical relationship between them. The output format should be \"relation1: word1, word2; relation2: word3, word4\". \n",
    "RE_class": "Find the relationship between {0} and {1} in text from following options? \n",
    "TXT_class": "Filter text types from the following options. \n",
    "ABSA": "For a given text, identify its emotional polarity and association with specified aspects, extract the words and the emotional between them. The output format should be \"emotional1: word1, word2; emotional2: word3, word4\". \n",
}

Noise_op = ["cut", "cut mix", "random pad", "replace", "opposit"]


class Data_prepare:
    def __init__(self, type, instruction, noise_instruction) -> None:
        self.type = type
        self.instruction = instruction
        self.noise_instruction = noise_instruction
        
    def process(self) -> None:
        if self.type == None :
            pass
        else:
            getattr(self, "prepare_data_" + self.type)()
        
    def prepare_data_NER(self, data_list, labels):
        example = []
        labels_str = ', '.join(labels)
        instruction = self.instruction
        instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
        noise_instruction = self.noise_instruction
        noise_instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
        for idx, instance in enumerate(data_list ):
            
            kv_pairs = []

            for entity in instance['entities']:
                if entity['type'] == 'NA' or entity['type'] == '':
                    continue
                kv_pair = [entity['name'], entity['type']]
                kv_pairs.append(kv_pair)

            if len(kv_pairs) > 0:
                label = " " + "; ".join(["{}: {}".format(v, k) for (k, v) in kv_pairs])
            else:
                label = " None"
                
            example.append({
                    "id": str(idx),
                    "sentence": instance['sentence'],
                    "label": label,
                    "ground_truth": label,
                    "instruction": instruction,
                    "noise_instruction":noise_instruction
                })
        return example
    
    def prepare_data_ABSA(self, data_list, labels):
        return self.prepare_data_RE(data_list, labels)
    
    def prepare_data_RE(self, data_list, labels):
        labels.append("None")
        example = []
        labels_str = ', '.join(labels)
        instruction = self.instruction
        instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
        noise_instruction = self.noise_instruction
        noise_instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
        for idx, instance in enumerate(data_list ):
            
            relation_pairs = []
            ground_truth_pairs = []

            for relation in instance['relations']:
                if relation['type'] == 'NA' or relation['type'] == '':
                    ground_truth_pairs.append([relation['head']['name'], 'NA', relation['tail']['name']])
                    continue
                relation_pair = [relation['head']['name'], relation['type'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)

            if len(relation_pairs) > 0:
                label = ' ' + "; ".join("{}: {}, {}".format(r, h, t) for (h, r, t) in relation_pairs)
            else:
                label = ' None'

            if len(ground_truth_pairs) > 0:
                ground_truth = ' ' + "; ".join("{}: {}, {}".format(r, h, t) for (h, r, t) in ground_truth_pairs)
            else:
                logger.error("******Error item: {}******".format(instance))
                raise Exception('Dataset Error:{}, No ground truth!'.format("dataset_name"))
            example.append({
                    "id": str(idx),
                    "sentence": instance['sentence'],
                    "label": label,
                    "ground_truth": ground_truth,
                    "instruction": instruction,
                    "noise_instruction":noise_instruction
                })
        return example
    
    def prepare_data_RE_class(self, data_list, labels):
        labels.append("None")
        example = []
        labels_str = ', '.join(labels)
        for idx, instance in enumerate(data_list ):
            
            relation_pairs = []
            ground_truth_pairs = []
            
            for relation in instance['relations']:
                if relation['type'] == '':
                    ground_truth_pairs.append([relation['head']['name'], 'NA', relation['tail']['name']])
                    continue
                relation_pair = [relation['head']['name'], relation['type'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)
                
            assert len(relation_pairs) == 1

            if len(relation_pairs) == 1:
                label = ' ' + relation_pairs[0][1]
            else:
                label = ' None'
            # get instruction  
            instruction = self.instruction
            noise_instruction = self.noise_instruction   
            instruction = instruction.format(relation_pairs[0][1], relation_pairs[0][2])
            noise_instruction = noise_instruction.format(relation_pairs[0][0], relation_pairs[0][2])
            instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
            noise_instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
            
            # if len(ground_truth_pairs) == 1:
            #     ground_truth = ' ' + relation_pairs[0][0]
            # else:
            #     logger.error("******Error item: {}******".format(instance))
            #     raise Exception('Dataset Error:{}, No ground truth!'.format("dataset_name"))
            
            example.append({
                    "id": str(idx),
                    "sentence": instance['sentence'],
                    "label": label,
                    "ground_truth": label,
                    "instruction": instruction,
                    "noise_instruction":noise_instruction
                })
        return example
    
    def prepare_data_TXT_class(self, data_list, labels):
        example = []
        labels_str = ', '.join(labels)
        instruction = self.instruction
        instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
        noise_instruction = self.noise_instruction
        noise_instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
        for idx, instance in enumerate(data_list ):
            label = ' ' + instance['label_text']
            example.append({
                    "id": str(idx),
                    "sentence": instance['text'],
                    "label": label,
                    "ground_truth": label,
                    "instruction": instruction,
                    "noise_instruction":noise_instruction
                })
        return example
        

class SFTDataset(Dataset):
    # 读取一个数据集下的数据
    # 
    def __init__(self, path, tokenizer, max_seq_length, noise_tp = "cut", is_train = True, type = "NER"):
        self.type = type
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.train_path = os.path.join(path, 'train.json')
        self.test_path = os.path.join(path, 'test.json')
        self.label_path = os.path.join(path, 'labels.json')
        self.noise_tp = noise_tp
        self.max_len = max_seq_length
        logger.info('Loading data: {}'.format(path))
        if is_train:
            instances, labels = self._load_dataset(self.train_path, self.label_path)
        else:
            instances, labels = self._load_dataset(self.test_path, self.label_path)
        self.instruction = global_instructions[type]
        self.noise_instruction = self.get_noise_instruction(self.instruction)
        self.prepare_data = Data_prepare(type, self.instruction, self.noise_instruction)
        self.example = getattr(self.prepare_data, "prepare_data_" + type)(instances, labels)
        logger.info("there are {} data in dataset".format(len(self.example)))
        self.padding = True
    
    def get_noise_instruction(self, instruction):
        return instruction

    
    def _load_dataset(self, dataset_path, labels_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            if self.type == "TXT_class":
                lines = task_f.readlines()
                instances = []
                for line in lines:
                    instances.append(json.loads(line))
            else:
                s = task_f.read()
                instances = json.loads(s)
        with open(labels_path, encoding="utf-8") as labels_f:
            labels = json.load(labels_f)
        return instances, labels
    
    
    def __len__(self):
        return len(self.example)
    
    def __getitem__(self, index):
        data = self.example[index]
        instruction = data["instruction"].format(data["sentence"])
        noise_instruction = data["noise_instruction"].format(data["sentence"])
        label = data["label"]

        model_inputs = self.tokenizer(
                instruction,
                max_length=self.max_len,
                padding=self.padding,
                truncation=True,
            )
        
        noise_inpts = self.tokenizer(
                noise_instruction,
                max_length=self.max_len,
                padding=self.padding,
                truncation=True,
            )

        labels = self.tokenizer(
                label,
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
            'mention_pos':(len(tokenized_input)-1, len(tokenized_noise)-1)
        }
        return inputs



class SFTDataset_all(SFTDataset):
    # 读取一个文件夹下所有数据集的数据
    def __init__(self, path, tokenizer, max_seq_length, noise_tp = "cut", is_train = True, type = "NER"):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.noise_tp = noise_tp
        self.instruction = global_instructions[type]
        self.noise_instruction = self.get_noise_instruction(self.instruction)
        self.prepare_data = Data_prepare(type, self.instruction, self.noise_instruction)
        self.example = []
        path_all = os.listdir(path)
        for data_set in path_all[:]:
            dataset_path = os.path.join(path, data_set)
            train_path = os.path.join(dataset_path, 'train.json')
            test_path = os.path.join(dataset_path, 'test.json')
            label_path = os.path.join(dataset_path, 'labels.json')
            self.max_len = max_seq_length
            logger.info('Loading data: {}'.format(path))
            if is_train:
                instances, labels = self._load_dataset(train_path, label_path)
            else:
                instances, labels = self._load_dataset(test_path, label_path)
            self.example += getattr(self.prepare_data, "prepare_data_" + type)(instances, labels)
            # if type == "NER":
            #     self.example += self.prepare_data_NER(instances, labels)
            # else:
            #     self.example += self.prepare_data_RE(instances, labels)
            logger.info("there are {} data in dataset".format(len(self.example)))
        self.padding = True
           
    
