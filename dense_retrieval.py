from re import L
import faiss 
import argparse
import json
from tqdm import trange, tqdm
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer
import torch 
import os 
import csv 
## load embedding
def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default='news_corpus',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--model",
        default='simcse',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--type",
        default='unlabeled',
        type=str,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--gpuid",
        default=0,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--N",
        default=20,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--round",
        default=1,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--prompt_id",
        default=1,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--dual_reg",
        default=1,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
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

def load_pred_data(dataset = 'mesh', ckpt = '', n_iter = 0):
    path = f"{dataset}/{ckpt}_{n_iter}"

    unlabeled_pred = np.load(f"{path}/unlabeled_pred.npy")
    unlabeled_feat = np.load(f"{path}/unlabeled_feat.npy")
    unlabeled_label = np.load(f"{path}/unlabeled_label.npy")
    unlabeled_pseudo = np.load(f"{path}/unlabeled_pseudo.npy")
    return  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo

args = get_arguments()
text = []
label = []

print("Model Name:", args.model)

print("Loading Text")
with open(f"../datasets/{args.dataset}/{args.type}.jsonl", 'r') as f:
    for lines in f:
        lines = json.loads(lines)
        text.append(lines["text"])
        label.append(lines["_id"])
    print("corpus size:", len(text),)

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model)
model = model.to(f"cuda:{args.gpuid}")

text_tmp = []
if args.round == 0:
    with open(f"datasets/{args.dataset}/classes_full.txt", 'r') as f:
        qtext = list(map(lambda x:x.strip().lower(), f.readlines()))
        id2label = [_ for _ in range(len(qtext))]
        print(qtext)
else:
    with open(f"datasets/{args.dataset}/{args.model}_N{args.N}_loc{args.loc}_global{args.glo}/classes_round{args.round}.txt", 'r') as f:
        qtext = list(map(lambda x:x.strip(), f.readlines()))
        id2label = [_ for _ in range(len(qtext))]
        print(qtext, tokenizer.tokenize(qtext[0]))
print("Query Embedding")

q_embeddings = []
num_iter = len(qtext)//args.batch_size if len(qtext) % args.batch_size == 0 else (len(qtext)//args.batch_size + 1)
for i in trange(num_iter):
    inputs = tokenizer(qtext[i*args.batch_size:(i+1)*args.batch_size], max_length = 40 if args.round > 0 else 16, padding=True, truncation=True, return_tensors="pt").to(f"cuda:{args.gpuid}")
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]
        embeddings = embeddings.squeeze(1)
        q_embeddings.append(embeddings.cpu().numpy())
q_embeddings = np.concatenate(q_embeddings, axis = 0)


print("Loading Passage Embedding")
with open(f"datasets/{args.dataset}/embedding_{args.model}_{args.type}.pkl", 'rb') as handle:
    passage_embedding = pickle.load(handle)

print("Calculating FAISS")
dim = q_embeddings.shape[1]
faiss.omp_set_num_threads(32)
cpu_index = faiss.IndexFlatIP(dim)
cpu_index.add(passage_embedding)    

dev_D, dev_I = cpu_index.search(q_embeddings, args.N)
os.makedirs(f"datasets/{args.dataset}/{args.model}_N{args.N}_loc{args.loc}_global{args.glo}", exist_ok = True)
print(f"round = {args.round}, {args.model}")
file_name = f"datasets/{args.dataset}/{args.model}_N{args.N}_loc{args.loc}_global{args.glo}/{args.dataset}_{args.model}_{args.type}_top{topN}_round{args.round}.jsonl"

visited = {}

total = np.zeros(dev_I.shape[1])
acc = np.zeros(dev_I.shape[1])
with open(file_name, 'w') as f, open(f"datasets/{args.dataset}/{args.model}_N{args.N}_loc{args.loc}_global{args.glo}/acc_{args.model}_round{args.round}.csv", "w") as f_out:
    if args.round == 0:
        writer = csv.writer(f_out, delimiter = '\t')
        writer.writerow(["Class Name", "Accuracy"])
        for i in range(dev_I.shape[0]):
            acc_class =  0
            for j in range(args.N):
                data = {"_id": int(i),"label": int(label[dev_I[i][j]]),  "text": text[dev_I[i][j]], "docid": int(dev_I[i][j]), "sim": "{:.4f}".format(dev_D[i][j])}
                f.write(json.dumps(data) + '\n')
                total[j] += 1
                if int(label[dev_I[i][j]]) == int(i):
                    acc[j] += 1
                    acc_class+= 1
            writer.writerow([qtext[i], acc_class/args.N])
        
        writer.writerow([f'AVG@{args.N}', np.cumsum(acc)[-1]/np.cumsum(total)[-1], [np.cumsum(acc)[k]/np.cumsum(total)[k] for k in range(10, topN, 10)]])
    else:
        writer = csv.writer(f_out, delimiter = '\t')
        writer.writerow(["Class Name", "Accuracy"])
        unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo = load_pred_data(dataset = args.dataset, ckpt = args.model, n_iter = 0)
        for i in range(dev_I.shape[0]):
            acc_class =  0
            cnt_class = 0
            for j in range(args.N):
                data = {"_id": int(i), "label": int(label[dev_I[i][j]]),  "text": text[dev_I[i][j]], "docid": int(dev_I[i][j]), "sim": "{:.4f}".format(dev_D[i][j])}
                f.write(json.dumps(data) + '\n')
                total[j] += 1
                if int(label[dev_I[i][j]]) == int(i):
                    acc[j] += 1
                    acc_class += 1
                cnt_class += 1
            writer.writerow([qtext[i], acc_class/cnt_class])
        writer.writerow([f'AVG@{args.N}', np.cumsum(acc)[-1]/np.cumsum(total)[-1], [np.cumsum(acc)[k]/np.cumsum(total)[k] for k in range(10, topN, 10)]])
           