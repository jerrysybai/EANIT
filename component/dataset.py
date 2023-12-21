import json
from loguru import logger
from torch.utils.data import Dataset
import os
import random

class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        # random.shuffle(data_list)
        self.data_list = data_list
        print(self.get_max_len())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']

        # 收集多轮对话
        utterances = []
        promot_text = []
        for x in conversation:
            utterances.append(x['human'])
            promot_text = x['human'].split("\nOption: ")
            utterances.append(x['assistant'])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids
        
        promot_text[0] = promot_text[0] + "\nOption:"
        promot = self.tokenizer(promot_text[0], add_special_tokens=False).input_ids
        text = self.tokenizer(promot_text[1], add_special_tokens=False).input_ids
        promot_mask = [0] + [1] * len(promot) + [0] * len(text) + [0] + [0] * len(utterances_ids[1]) + [0]

        # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        input_ids = [self.bos_token_id]
        test_input = [self.bos_token_id]
        label_ids = []
        target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
                test_input += (utterances_id + [self.eos_token_id])
            else:
                target_mask += [1] * (len(utterances_id) + 1)
                label_ids += (utterances_id + [self.eos_token_id])
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        promot_mask = promot_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask) == len(promot_mask)
        out_mask = [-100] * self.tokenizer.vocab_size
        for ids in input_ids:
            out_mask[ids] = 0
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
            'test_input':test_input,
            'label_ids':label_ids,
            'out_mask': out_mask,
            'promot_mask':promot_mask
        }
        return inputs
    
    def get_max_len(self):
        max_len = 0
        for i in range(len(self.data_list)):
            data = self.data_list[i]
            data = json.loads(data)
            conversation = data['conversation']

            # 收集多轮对话
            utterances = []
            for x in conversation:
                utterances.append(x['human'])
                utterances.append(x['assistant'])
            utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

            # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
            input_ids = [self.bos_token_id]
            test_input = [self.bos_token_id]
            label_ids = []
            target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
            for i, utterances_id in enumerate(utterances_ids):
                input_ids += (utterances_id + [self.eos_token_id])
                if i % 2 == 0:
                    target_mask += [0] * (len(utterances_id) + 1)
                    test_input += (utterances_id + [self.eos_token_id])
                else:
                    target_mask += [1] * (len(utterances_id) + 1)
                    label_ids += (utterances_id + [self.eos_token_id])
            if len(input_ids) > max_len:
                max_len = len(input_ids)
        return max_len
    
class SFTDataset_class(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, label2id, is_test = False):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        self.label2id = label2id
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        # random.shuffle(data_list)
        self.data_list = data_list
        self.is_test = is_test

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']

        # 收集多轮对话
        utterances = []
        label_ids = []
        for x in conversation:
            utterances.append(x['human'])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids
        for x in conversation:
            if self.is_test :
                label_ids = [0]
            else:
                label_ids.append(self.label2id[x["assistant"].strip()])


        # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        input_ids = [self.bos_token_id]
        test_input = [self.bos_token_id]
        labels = []
        
        
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            if i % 2 == 0:
                test_input += (utterances_id + [self.eos_token_id])
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids)  == len(attention_mask)
        assert len(label_ids) == 1
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids[0]
        }
        return inputs
    
class SFTDataset1(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, max_source_length, max_target_length, type="RE", path = None):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        path_0 = os.listdir(path)
        self.data = []
        for data_set in path_0[:]:
            dataset_path = os.path.join(path, data_set)
            labels_path = os.path.join(dataset_path, 'labels.json')
            file = os.path.join(dataset_path, 'train.json')
            logger.info('Loading data: {}'.format(file))
            data_list, labels = self._load_dataset(file, labels_path)
            self.data += self.prepare_data(data_list, labels)
        logger.info("there are {} data in dataset".format(len(self.data)))  
        random.shuffle(self.data)  
        # labels_path = os.path.join(path, 'labels.json')
        # data_list, self.labels = self._load_dataset(file, labels_path)
        # logger.info("there are {} data in dataset".format(len(data_list)))
        # self.data_list = data_list
        # self.type = type
        # self.path = path
        # self.data = self.prepare_data()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = True

    def prepare_data(self, data_list, labels):
        example = []
        labels_str = ', '.join(labels)
        instruction = "Given a phrase that describes the relationship between two words, extract the words and the lexical relationship between them. The output format should be \"relation1: word1, word2; relation2: word3, word4\". \n"
        instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
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
                    "instruction": instruction
                })
        return example
     
    def get_instruction(self, instance):
        # "instructions \n options \n {0} \n Answer: "
        instruction = instance["instruction"]
        content = instance['sentence']
        # TODO, support few shot
        # add few shot samples
        try:
            instruction = instruction.format(content)
        finally:
            return instruction   
        
    def __len__(self):
        return len(self.data)
    
    def _load_dataset(self, dataset_path, labels_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)
        with open(labels_path, encoding="utf-8") as labels_f:
            labels = json.load(labels_f)

        return instances, labels

    def __getitem__(self, index):
        
        data = self.data[index]
        instruction = data["instruction"].format(data["sentence"])
        label = data["label"]
        
        # task_input = self.tokenizer.bos_token + instruction[:self.max_source_length-2] + self.tokenizer.eos_token
        # task_input = self.tokenizer.bos_token + instruction[:self.max_source_length-2] + self.tokenizer.eos_token
        # label = label + self.tokenizer.eos_token
        
        model_inputs = self.tokenizer(
                instruction,
                max_length=self.max_source_length,
                padding=self.padding,
                truncation=True,
            )
        
        labels = self.tokenizer(
                label,
                max_length=self.max_target_length,
                padding=self.padding,
                truncation=True,
            )
        
        tokenized_input = [self.bos_token_id] + model_inputs["input_ids"] + [self.eos_token_id]
        tokenized_label = labels["input_ids"] + [self.eos_token_id]
        
        input_ids = tokenized_input[:self.max_source_length] + tokenized_label[:self.max_target_length]
        target_mask = [0] * min(self.max_source_length, len(tokenized_input)) + [1] * min(self.max_target_length, len(tokenized_label))
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        
        labels_id = [-100] * len(tokenized_input[:self.max_source_length]) + tokenized_label[:self.max_target_length]
        
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask) == len(labels_id)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
            'labels':labels_id
        }
        return inputs
 
        
        # conversation = data['conversation']

        # # 收集多轮对话
        # utterances = []
        # for x in conversation:
        #     utterances.append(x['human'])
        #     utterances.append(x['assistant'])
        # utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        # input_ids = [self.bos_token_id]
        # target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
        # for i, utterances_id in enumerate(utterances_ids):
        #     input_ids += (utterances_id + [self.eos_token_id])
        #     if i % 2 == 0:
        #         target_mask += [0] * (len(utterances_id) + 1)
        #     else:
        #         target_mask += [1] * (len(utterances_id) + 1)
        # assert len(input_ids) == len(target_mask)
        # # 对长度进行截断
        # input_ids = input_ids[:self.max_seq_length]
        # target_mask = target_mask[:self.max_seq_length]
        # attention_mask = [1] * len(input_ids)
        # assert len(input_ids) == len(target_mask) == len(attention_mask)
        # inputs = {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'target_mask': target_mask
        # }
        # return inputs


class ChatGLM2SFTDataset(SFTDataset):

    def __getitem__(self, index):
        """
        基本沿袭ChatGLM2的指令微调的格式，做了小修改，多轮对话如下。
        """
        # 每条数据格式为: [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']
        input_format = '[Round {}]\n\n问：{}\n\n答：'
        target_format = '{}'

        # 收集多轮对话
        utterances = []
        for i, x in enumerate(conversation):
            human = input_format.format(i+1, x['human'])
            assistant = target_format.format(x['assistant'])
            utterances += ([human, assistant])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # 每条数据格式为: [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        input_ids = []
        target_mask = []  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += utterances_id
            # input部分
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id))
            # target部分
            else:
                input_ids += [self.eos_token_id]
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs

