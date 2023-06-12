import faiss 
import numpy as np 
import os 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import json 


def save_data(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, mutual_info_bald = None, dataset = 'agnews', n_iter = 0, ckpt = ''):
    
    path = f"{dataset}/{ckpt}_{n_iter}"
    os.makedirs(path, exist_ok = True)
    
    with open(f"{path}/train_pred.npy", 'wb') as f:
        np.save(f, train_pred)
    
    with open(f"{path}/train_feat.npy", 'wb') as f:
        np.save(f, train_feat)
    
    with open(f"{path}/train_label.npy", 'wb') as f:
        np.save(f, train_label)

    with open(f"{path}/unlabeled_pred.npy", 'wb') as f:
        np.save(f, unlabeled_pred)

    with open(f"{path}/unlabeled_feat.npy", 'wb') as f:
        np.save(f, unlabeled_feat)
    
    with open(f"{path}/unlabeled_label.npy", 'wb') as f:
        np.save(f, unlabeled_label)
    
    with open(f"{path}/unlabeled_pseudo.npy", 'wb') as f:
        np.save(f, unlabeled_pseudo)
   


def load_pred_data(dataset = 'mesh', ckpt = '', n_iter = 0):
    # os.makedirs(f"{dataset}/{n_labels}", exist_ok = True)
    # with open(f"{dataset}/{n_labels}/train_pred.npy", 'rb') as f:

    path = f"{dataset}/{ckpt}_{n_iter}"
    train_pred = np.load(f"{path}/train_pred.npy")

    train_feat = np.load(f"{path}/train_feat.npy")

    train_label = np.load(f"{path}/train_label.npy")

    unlabeled_pred = np.load(f"{path}/unlabeled_pred.npy")

    unlabeled_feat = np.load(f"{path}/unlabeled_feat.npy")

    unlabeled_label = np.load(f"{path}/unlabeled_label.npy")
    
    unlabeled_pseudo = np.load(f"{path}/unlabeled_pseudo.npy")

    current_val = -np.max(unlabeled_pred, axis = -1)
    idx = np.argsort(current_val)

    unlabel_correct = [1 if x == y else 0 for (x, y) in zip(unlabeled_pseudo, unlabeled_label)]

    return train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo

