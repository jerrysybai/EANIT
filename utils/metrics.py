import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def get_metrics(tokenizer, type):
    
    
    def compute_metrics_re(pred_o):
        labels = np.array(pred_o.label_ids)
        preds = np.array(pred_o.predictions)
        labels = np.where(labels>0, labels, 0)
        preds = np.where(preds>0, preds, 0)
        label_all = []
        pred_all = []
        label_all_full = []
        pred_all_full = []
        cor_tot = 0
        for i in range(preds.shape[0]):
            pred = preds[i].tolist()
            label = labels[i].tolist()
            response = tokenizer.decode(pred)
            response = response.strip().replace(tokenizer.eos_token, "").replace("<unk>", "").strip().split("; ")
            label = tokenizer.decode(label)
            label = label.strip().replace(tokenizer.eos_token, "").replace("<unk>", "").strip().split("; ")
            for l in label:
                l_list = l.split(": ")
                if len(l_list) != 2:
                    continue
                label_all_full.append((i,l_list[0],l_list[1]))
                ents = l_list[1].split(", ")
                if len(ents) != 2:
                    continue
                label_all.append((i,l_list[0],ents[0],ents[1]))
            for r in response:
                r_list = r.split(": ")
                if len(r_list) != 2:
                    continue
                pred_all_full.append((i,r_list[0],r_list[1]))
                r_ents = r_list[1].split(", ")
                if len(r_ents) != 2:
                    continue
                pred_all.append((i,r_list[0],r_ents[0], r_ents[1]))

        for item in pred_all:
            p = (item[0],item[1],item[3],item[2])
            if item in label_all or p in label_all:
                cor_tot += 1                
        ner_tot_recall = len(label_all)
        tot_pred_tot = len(pred_all)
        p1 = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0 
        r1 = cor_tot / ner_tot_recall 
        f1_tot1 = 2 * (p1 * r1) / (p1 + r1) if cor_tot > 0 else 0.0
        ad = {'f1_1':  f1_tot1, 'precision': p1, 'recall': r1}
        print(ad) 
        
        cor_tot = 0
        for item in pred_all_full:
            if item in label_all_full:
                cor_tot += 1
        p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0 
        r = cor_tot / ner_tot_recall 
        f1_tot = 2 * (p * r) / (p + r) if cor_tot > 0 else 0.0
        ad = {'f1':  f1_tot, 'precision': p, 'recall': r}
        print(ad)    
        return {'f1':  f1_tot, 'precision': p, 'recall': r}
    
    
    def compute_metrics_ner(pred_o):
        labels = np.array(pred_o.label_ids)
        preds = np.array(pred_o.predictions)
        labels = np.where(labels>0, labels, 0)
        preds = np.where(preds>0, preds, 0)
        label_all = []
        pred_all = []
        labels_sub = []
        preds_sub = []
        cor_tot = 0
        for i in range(preds.shape[0]):
            pred = preds[i].tolist()
            label = labels[i].tolist()
            response = tokenizer.decode(pred)
            response = response.strip().replace(tokenizer.eos_token, "").replace("<unk>", "").strip().split("; ")
            label = tokenizer.decode(label)
            label = label.strip().replace(tokenizer.eos_token, "").replace("<unk>", "").strip().split("; ")
            for l in label:
                if l != "None":
                    labels_sub.append((i,l))
                if l == "None":
                    label_all.append((i,"None","None"))
                    continue
                l_list = l.split(": ")
                if len(l_list) != 2:
                    continue
                label_all.append((i,l_list[0].replace(" ", ""),l_list[1].replace(" ", "")))
            for r in response:
                if r != "None":
                    preds_sub.append((i,r))
                if r == "None":
                    pred_all.append((i,"None","None"))
                    continue
                r_list = r.split(": ")
                if len(r_list) != 2:
                    continue
                
                pred_all.append((i,r_list[0].replace(" ", ""),r_list[1].replace(" ", "")))
                        
        ner_tot_recall = len(label_all)
        tot_pred_tot = len(pred_all)
        
        cor_tot = 0
        for item in pred_all:
            if item in label_all:
                cor_tot += 1
        p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0 
        r = cor_tot / ner_tot_recall 
        f1_tot = 2 * (p * r) / (p + r) if cor_tot > 0 else 0.0
        
        cor_tot = 0
        for item in preds_sub:
            if item in labels_sub:
                cor_tot += 1
        p1 = cor_tot / len(preds_sub) if len(preds_sub) > 0 else 0 
        r1 = cor_tot / len(labels_sub) 
        f1 = 2 * (p1 * r1) / (p1 + r1) if cor_tot > 0 else 0.0
        
        ad = {'f1_none':  f1_tot, 'precision': p, 'recall': r, 'f1':  f1, 'precision1': p1, 'recall1': r1}
        print(ad)    
        return {'f1_none':  f1_tot, 'precision': p, 'recall': r, 'f1':  f1, 'precision1': p1, 'recall1': r1}
    
    def compute_metrics_re_class(pred_o):
        labels = np.array(pred_o.label_ids)
        preds = np.array(pred_o.predictions)
        labels = np.where(labels>0, labels, 0)
        preds = np.where(preds>0, preds, 0)
        label_all = []
        pred_all = []
        label2id = {}
        for i in range(preds.shape[0]):
            pred = preds[i].tolist()
            label = labels[i].tolist()
            response = tokenizer.decode(pred)
            response = response.strip().replace(tokenizer.eos_token, "").replace("<unk>", "").strip()
            label = tokenizer.decode(label)
            label = label.strip().replace(tokenizer.eos_token, "").replace("<unk>", "").strip()
            if label not in label2id:
                label2id[label] = len(label2id)
            label_all.append(label)
            pred_all.append(response)
            
        for i in range(len(label_all)):
            label_all[i] = label2id[label_all[i]]
            if pred_all[i] not in label2id:  
                pred_all[i] = 0 if label_all[i]!=0 else 1
            else:
                pred_all[i] = label2id[pred_all[i]]
        f1 = f1_score(label_all, pred_all, average='micro')
        precision = precision_score(label_all, pred_all, average='micro')
        recall = recall_score(label_all, pred_all, average='micro')
        f1_m = f1_score(label_all, pred_all, average='macro')
        precision_m = precision_score(label_all, pred_all, average='macro')
        recall_m = recall_score(label_all, pred_all, average='macro')
        ad = {'micro-f1':  f1, 'micro-precision': precision, 'micro-recall': recall, 'macro-f1':  f1_m, 'macro-precision': precision_m, 'macro-recall': recall_m}
        print(ad)    
        return {'micro-f1':  f1, 'micro-precision': precision, 'micro-recall': recall, 'macro-f1':  f1_m, 'macro-precision': precision_m, 'macro-recall': recall_m}
    
    if type=="RE" or type == "ABSA":
        return compute_metrics_re
    elif type == "NER":
        return compute_metrics_ner
    elif type == "RE_class" or type == "TXT_class":
        return compute_metrics_re_class       
    
