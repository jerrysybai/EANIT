from typing import Any, Dict, List
import torch


class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length, is_test = False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.is_test = is_test

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch]
        test_len = [len(x['test_input']) for x in batch]
        label_len = [len(x['label_ids']) for x in batch]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        batch_max_testlen = min(max(test_len), self.max_seq_length)
        batch_max_labellen = min(max(label_len), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch, labels_batch, test_batch, normal_pos_batch, noise_pos_batch, ins_mask = [], [], [], [], [], [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            test_input = x["test_input"]
            label_ids = x["label_ids"]
            mention_pos = x["mention_pos"]
            # ins_len = x["ins_len"]
            
            normal_pos = [mention_pos[0] + i for i in range(len(label_ids))][:150]
            noise_pos = [mention_pos[1] + i for i in range(len(label_ids))][:150]
            pospad_len = 150 - len(normal_pos)
            normal_pos = normal_pos + [normal_pos[-1]] * pospad_len
            noise_pos = noise_pos + [noise_pos[-1]] * pospad_len
 
            padding_len = batch_max_len - len(input_ids)
            test_padlen = batch_max_testlen - len(test_input)
            label_padlen = batch_max_labellen - len(label_ids)
            
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            test_input =  [0] * test_padlen + test_input
            label_ids = label_ids + [0] * label_padlen

            # truncate
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]
            test_input = test_input[:self.max_seq_length]
            label_ids = label_ids[:self.max_seq_length]
            # ins = [1]*ins_len + [0] * (len(input_ids) - ins_len)


            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)
            labels_batch.append(label_ids)
            test_batch.append(test_input)
            normal_pos_batch.append(normal_pos)
            noise_pos_batch.append(noise_pos)
            # ins_mask.append(ins)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        test_batch = torch.tensor(test_batch, dtype=torch.long)
        normal_batch = torch.tensor(normal_pos_batch, dtype=torch.long)
        noise_batch = torch.tensor(noise_pos_batch, dtype=torch.long)
        # ins_mask_batch = torch.tensor(ins_mask, dtype=torch.long)
        
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'target_mask': target_mask_batch,
            'test_labels':labels_batch,
            'test_ids':test_batch,
            'noise_pos' : noise_batch,
            'normal_pos': normal_batch,
            # 'ins_mask':ins_mask_batch
        }
        return inputs
