import nltk 
import json 
from collections import Counter 
import argparse 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm 
import numpy as np 
from transformers import  AutoModel, AutoTokenizer
import torch
from numpy.linalg import norm
import csv 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import os 

def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--target",
        default='agnews',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--topN",
        default=50,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--dr_model",
        default='arxiv_ckpt',
        type=str,
        help="The dense retrieval model",
    )

    parser.add_argument(
        "--model",
        default='arxiv_ckpt',
        type=str,
        help="The model for generating token embeddings",
    )

    parser.add_argument(
        "--round",
        default=0,
        type=int,
        help="The round of iteration",
    )


    parser.add_argument(
        "--loc",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--glo",
        default=1,
        type=int,
    )

    args = parser.parse_args()
    return args



def load_document(args, n_classes, unlabeled_pseudo = None, unlabeled_label = None):
    cnter = [Counter()  for _ in range(n_classes)]
    total_cnt = Counter() 
    lemmatizer = WordNetLemmatizer()

    stop_words = {x :1 for x in set(stopwords.words('english'))}
    r = args.round
    path = f"/datasets/{args.target}_openws/{args.dr_model}_N{args.topN}_loc{args.loc}_global{args.glo}/{args.target}_{args.dr_model}_train_top{args.topN}_round{r}.jsonl"
    with open(path,  'r') as f:
        for lines in tqdm(f):
            lines = json.loads(lines)
            idx = int(lines["_id"])
            text = lines["text"].strip()
            doc_idx = int(lines["docid"])
            if unlabeled_pseudo is not None and unlabeled_label is not None and unlabeled_pseudo[doc_idx] != idx :
                continue
            words = [w for w in word_tokenize(text) if not w in stop_words]
            words=[word.lower() for word in words if word.isalpha() and len(word) > 2]
            for w in words:
                cnter[idx][lemmatizer.lemmatize(w)] += 1
                total_cnt[lemmatizer.lemmatize(w)] += 1
    return total_cnt, cnter

def load_pred_data(dataset = 'agnews', ckpt = '', n_iter = 0):

    path = f"{dataset}/{ckpt}_{n_iter}"
    print(f"load path {path}")

    unlabeled_pred = np.load(f"{path}/unlabeled_pred.npy")
    unlabeled_feat = np.load(f"{path}/unlabeled_feat.npy")
    unlabeled_label = np.load(f"{path}/unlabeled_label.npy")
    unlabeled_pseudo = np.load(f"{path}/unlabeled_pseudo.npy")

    return   unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo
            
            
if __name__ == "__main__":
    args = get_arguments()
    print("Loading Text")
    text = []
    label = []
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)
    unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo = load_pred_data(dataset = args.target, ckpt = args.dr_model + f"_{args.topN}", n_iter = 0)
    
    model.eval()
    model.cuda()

    if args.round == 0:
        with open(f"datasets/{args.target}/classes_full.txt", 'r') as f:
            qtext = list(map(lambda x:x.strip().lower(), f.readlines()))
            id2label = [_ for _ in range(len(qtext))]
            print(qtext)
    else:
        with open(f"datasets/{args.target}/{args.dr_model}_N{args.topN}_loc{args.loc}_global{args.glo}/classes_round{args.round}.txt", 'r') as f:
            qtext = list(map(lambda x:x.strip(), f.readlines()))
            id2label = [_ for _ in range(len(qtext))]
            print(qtext)
    n_classes = len(id2label)
    total_cnt, cnter = load_document(args, n_classes ,unlabeled_pseudo=unlabeled_pseudo, unlabeled_label=unlabeled_label)
    word_cnter = np.zeros([len(total_cnt), n_classes])
    idx_to_word = {}
    for i, w in enumerate(total_cnt):
        for j in range(n_classes):
            word_cnter[i][j] = cnter[j][w]
        idx_to_word[i] = w 
    # TF 
    tf = np.sum(word_cnter, axis = -1)/np.sum(word_cnter)
    # idf
    idf = np.log10( (len(total_cnt)/(np.sum(word_cnter, axis = -1)) )) 
    print(tf.shape, idf.shape)
    os.makedirs(f"datasets/{args.target}/{args.dr_model}_N{args.topN}_loc{args.loc}_global{args.glo}/", exist_ok = True)
    os.makedirs(f"datasets/{args.target}_openws/{args.dr_model}_N{args.topN}_loc{args.loc}_global{args.glo}/", exist_ok = True)
    f_out = open(f"datasets/{args.target}/{args.dr_model}_N{args.topN}_loc{args.loc}_global{args.glo}/classes_round{args.round+1}.txt", 'w')
    f_out_score = open(f"datasets/{args.target}_openws/{args.dr_model}_N{args.topN}_loc{args.loc}_global{args.glo}/classes_round{args.round+1}_openws.csv", 'w')
    
    
    csv_writer = csv.writer(f_out_score, delimiter = '\t')
    csv_writer.writerow(["Class", "Rank", "Word", "score"])
    lemmatizer = WordNetLemmatizer()
    for i in range(n_classes):
        score1 = Counter()
        score2 = Counter()
        score = Counter()
        rank1 = 1
        rank2 = 1
        sim_score = []
        tf = word_cnter[:, i] / np.sum(word_cnter, axis = 1)
        tf_idf_score = (word_cnter[:, i]) * idf * np.sqrt(tf)
        idx = np.argsort(tf_idf_score)[::-1]

        with torch.no_grad():
            seq = torch.LongTensor([tokenizer.encode(qtext[i])]).cuda()
            emb = model(seq)
            all_layer_outputs = emb.hidden_states[2].detach().cpu()            

            label_embed = np.mean(tensor_to_numpy(all_layer_outputs[-1].squeeze(0))[1: -1], axis = 0)

        for (j, id) in enumerate(idx[:50]):
            with torch.no_grad():
                seq = torch.LongTensor([tokenizer.encode(idx_to_word[id])]).cuda()
                emb = model(seq)
                all_layer_outputs = emb.hidden_states[2].detach().cpu()
                token_embed = np.mean(tensor_to_numpy(all_layer_outputs[-1].squeeze(0))[1: -1], axis = 0)
            print("class:", qtext[i], 'rank:', j, "token:", idx_to_word[id],  \
                "cossim:", np.dot(label_embed, token_embed)/(norm(label_embed) * norm(token_embed)))

            sim_score.append(np.dot(label_embed, token_embed)/(norm(label_embed) * norm(token_embed)))
        rank_2 = np.argsort(sim_score)[::-1]
        for j in range(50):
            word_name = idx_to_word[idx[j]]
            if word_name in qtext[i] or lemmatizer.lemmatize(word_name, pos='v') in qtext[i] or lemmatizer.lemmatize(word_name, pos='n') in qtext[i]:
                continue 
            else:   
                score1[word_name] = 1/(1+j) 
                rank1 += 1
            word_name = idx_to_word[ idx[rank_2[j]] ] 
            if word_name in qtext[i] or lemmatizer.lemmatize(word_name, pos='v') in qtext[i] or lemmatizer.lemmatize(word_name, pos='n') in qtext[i]:
                continue 
            else:   
                score2[word_name] = 1/(1+j)
                rank2 += 1
        if args.loc:
            for x in score1:
                score[x] += score1[x]
        if args.glo:
            for x in score2:
                score[x] += score2[x]
        max_word = score.most_common()[:10][0][0]
        f_out.write(f"{qtext[i]} {max_word}\n")
        
        cnt = 1
        for w in score.most_common()[:10]:
            word = w[0]
            score = w[1]
            csv_writer.writerow([qtext[i], cnt, word, score])
            cnt += 1



    
    