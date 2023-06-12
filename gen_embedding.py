import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import argparse
import json
from tqdm import trange
import numpy as np
import pickle
# Tokenize input texts
def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default='mesh',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--model",
        default='arxiv_ckpt',
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
        default=64,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--gpuid",
        default=0,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )
    args = parser.parse_args()
    return args

args = get_arguments()
text = []
label = []

print("Model Name:", args.model )

with open(f"datasets/{args.dataset}/{args.type}.jsonl", 'r') as f:
    for lines in f:
        lines = json.loads(lines)
        text.append(lines["text"])
        label.append(lines["_id"])

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained(args.model )
model = AutoModel.from_pretrained(args.model )

model = model.to(f"cuda:{args.gpuid}")

embedding = []

num_iter = len(text)//args.batch_size if len(text) % args.batch_size == 0 else (len(text)//args.batch_size + 1)
for i in trange(len(text)//args.batch_size + 1):
    inputs = tokenizer(text[i*args.batch_size:(i+1)*args.batch_size], padding=True, truncation=True, max_length = 256, return_tensors="pt").to(f"cuda:{args.gpuid}")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]
        embeddings = embeddings.squeeze(1)
        embedding.append(embeddings.cpu().numpy())
    
embedding = np.concatenate(embedding, axis = 0)
print(embedding.shape)


'''Save the embedding'''
with open(f"../datasets/{args.dataset}/embedding_{args.model}_{args.type}.pkl", 'wb') as handle:
    pickle.dump(embedding, handle, protocol=4)

'''Plot the TSNE graph'''
def plot_embedding(embedding, label, args, name= ""):
    import matplotlib
    from MulticoreTSNE import MulticoreTSNE as TSNE
    tsne = TSNE(n_jobs=16, n_components=2)
    Y = tsne.fit_transform(embedding)
    with open(f"datasets/{args.dataset}/embedding_{args.model}{name}_tsne.pkl", 'wb') as handle:
        pickle.dump(Y, handle, protocol=4)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colormap = ["red",  "limegreen", "darkorange", "blueviolet", "blue","magenta", "olive", "teal",  "grey", "cyan", 'black']
    pseudo = [colormap[x%11] for x in label]
    plt.figure(figsize = [12.5, 8], dpi = 150)
    plt.scatter(Y[:, 0], Y[:, 1], c = pseudo, s = 10, alpha = 0.7)
    plt.tight_layout()
    plt.savefig(f"datasets/{args.dataset}/tsne_{args.model}{name}.pdf")

plot_embedding(embedding, label, args)
   

    



