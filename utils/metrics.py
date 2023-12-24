import numpy as np

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
    
    return compute_metrics_re if type=="RE" else compute_metrics_ner