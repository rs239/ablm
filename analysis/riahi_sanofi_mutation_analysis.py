#!/usr/bin/env python

import pandas as pd
import torch
import numpy as np
import h5py
import sys
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools, csv
import  copy, multiprocessing, traceback, logging, pickle, traceback, time
import scipy.stats, sklearn.decomposition, sklearn.preprocessing, sklearn.covariance
from scipy.stats import describe
from scipy import sparse
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as PathEffects
import random
from sklearn.linear_model import Ridge

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim as opt
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append('..')
from model import AntibodyNetMultiAttn, DesautelsPredictor
from utils import evaluate_spearman
# from torch.utils.tensorboard import SummaryWriter
from dataloader import create_dataloader_desautels

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def preprocess_sanofi_spreadsheet(args, ab_name):
    df = pd.read_excel(f"{args.datadir}/ddg_all.xlsx", sheet_name=ab_name, header=None)

    seq_H = ''.join( [a.strip() for a in open(f"{args.datadir}/{ab_name}_VH.fasta","rt").readlines()[1:]])
    seq_L = ''.join( [a.strip() for a in open(f"{args.datadir}/{ab_name}_VL.fasta","rt").readlines()[1:]])

    print("Flag 329.10 H:", seq_H)
    print("Flag 329.10 L:", seq_L)

    score = {'H': defaultdict(dict) , 'L': defaultdict(dict)}
    for i,c in enumerate(df.iloc[:,0]):
        #print ("Flag 1 ", i, c)
        if pd.isnull(c) or c.strip()=="":
            continue
        elif c.startswith(ab_name):
            chain = 'L' if ': LC' in c else 'H'
            method = c.split('(')[1].replace(')','').strip()
        elif c == 'wild_aa':
            edits = [a.strip() for a in df.iloc[i,:].tolist()]
        elif len(c) in [2,3,4]:
            vals = df.iloc[i,:].tolist()
            posn = int(c[1:])

            if ab_name == 'm396':
                if chain == 'L':
                    posn = posn -1 #sanofi's LC numbering is off by one for m396 light chain?

            wt_seq = seq_H if chain == 'H' else seq_L
            for j, a in enumerate(edits[1:]):
                a1 = a.strip()[0]
                if len(a.strip())!=1: continue
                try:
                    v = float(vals[j+1])
                except:
                    continue
                edit_seq = (wt_seq[:(posn-1)] + a1 + wt_seq[(posn):]).strip()
                score[chain][edit_seq][method] = v

    categories = "TopNetTree,Flex ddG,MOE,SAAMBE-3D"
    if ab_name == '80R':
        categories = "TopNetTree,Flex ddg,MOE,SAAMBE-3D"

    vH, vL = [], []
    for seq, vals in score['H'].items():
        vH.append([seq] + [vals[a] for a in categories.split(",")])
    for seq, vals in score['L'].items():
        vL.append([seq] + [vals[a] for a in categories.split(",")])
    dfH = pd.DataFrame(vH, columns="seq,TopNetTree,Flex ddG,MOE,SAAMBE-3D".split(","))
    dfL = pd.DataFrame(vL, columns="seq,TopNetTree,Flex ddG,MOE,SAAMBE-3D".split(","))

    dfH.to_csv(f"{args.datadir}/{ab_name}_mutations_VH.csv", index=False)
    dfL.to_csv(f"{args.datadir}/{ab_name}_mutations_VL.csv", index=False)

    
def create_data_features(args, ab_name):
    from protein_embedding import ProteinEmbedding
    from embed import reload_models_to_device

    # f = h5py.File('{}/{}.h5'.format(args.datadir, ab_name), 'w')
    # if args.is_baseline:
    #     embed_dim = 20
    # else:
    #     embed_dim = 6169

    with h5py.File('{}/{}.h5'.format(args.datadir, ab_name), 'w') as f:

        for chain_type in ['H', 'L']:
            df = pd.read_csv(os.path.join(args.datadir, '{}_mutations_V{}.csv'.format(ab_name, chain_type)))

            reload_models_to_device()
            k=50
            feats_list = []
            for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
                seq = row['seq']
                prot = ProteinEmbedding(seq, chain_type)
                prot.embed_seq('beplerberger')
                prot.create_cdr_mask()
                kmut_matr_h = prot.create_kmut_matrix(num_muts=k, embed_type='beplerberger')
                cdr_embed = prot.create_cdr_embedding(kmut_matr_h, sep = False, mask = True)

                feats_list.append(cdr_embed)
                # dset[idx, :, :] = cdr_embed
                del cdr_embed, kmut_matr_h, prot

            out_tens = pad_sequence(feats_list, batch_first=True)

            dset = f.create_dataset(chain_type, out_tens.shape)
            dset[:,:,:] = out_tens

            del out_tens



def ridge_regression(args):
    device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

    # load pretrained model:
    if not args.is_baseline:
        pretrained_H = AntibodyNetMultiAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
        pretrained_L = AntibodyNetMultiAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
        best_arch_path = '/net/scratch3/scratch3-3/chihoim/model_ckpts/best_pretrained'
        H_best_path = os.path.join(best_arch_path, 'bb_feat256_H_best_newnum.pt')
        L_best_path = os.path.join(best_arch_path, 'bb_feat256_L_best_newnum.pt')
        checkpoint_H = torch.load(H_best_path, map_location=device)
        checkpoint_L = torch.load(L_best_path, map_location=device)
        pretrained_H.load_state_dict(checkpoint_H['model_state_dict'])
        pretrained_L.load_state_dict(checkpoint_L['model_state_dict'])
        pretrained_H.eval()
        pretrained_L.eval()

    alpha = 0.01
    print("Regression Alpha Term:", alpha)

    for ab_name in ['m396','80R','CR3022']:
        # load the h5py dataset
        dataset_path = '/data/cb/rsingh/work/antibody/ci_data/raw/riahi_sanofi/{}.h5'.format(ab_name)
        with h5py.File(dataset_path, "r") as f:
            dataset_H_all = torch.FloatTensor(np.array(f['H'])).to(device)
            dataset_L_all = torch.FloatTensor(np.array(f['L'])).to(device)

        # load the labels:
        score_cols = ['TopNetTree','Flex ddG','MOE','SAAMBE-3D']
        for chain_type in ['H', 'L']:
            if chain_type == 'H':
                print("Processing {} Heavy Chain...".format(ab_name))
                dataset = dataset_H_all
                pretrained_model = pretrained_H
            else:
                print("Processing {} Light Chain...".format(ab_name))
                dataset = dataset_L_all
                pretrained_model = pretrained_L

            df_label = pd.read_csv('/data/cb/rsingh/work/antibody/ci_data/raw/riahi_sanofi/{}_mutations_V{}.csv'.format(ab_name, chain_type))
            labels_all = df_label[score_cols].to_numpy()

            # sample test and train: 50/50
            shuffled_idx = list(range(dataset.shape[0]))
            random.shuffle(shuffled_idx)
            mid_idx = int(len(shuffled_idx)/2)
            train_idx, test_idx = sorted(shuffled_idx[:mid_idx]), sorted(shuffled_idx[mid_idx:])

            # create features:
            task = 0
            with torch.no_grad():
                X_all, _ = pretrained_model(dataset, dataset, None, None, task=task, task_specific=True)
                X_all = X_all.detach().cpu().numpy()

            # x = pretrained_model.project(dataset[:,:,-2204:-4])
            # X_all = pretrained_model.recurrent(torch.cat([x, dataset[:,:,-4:]], dim=-1), task).detach().cpu().numpy()

            X_train, X_test = X_all[train_idx, :], X_all[test_idx, :]
            y_train, y_test = labels_all[train_idx, :], labels_all[test_idx, :]

            # perform Ridge Regression
            clf = Ridge(alpha=alpha)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            for j in range(y_pred.shape[-1]):
                spear = evaluate_spearman(y_pred[:, j], y_test[:, j])
                print("Spearman Rank Scores for {}: {}".format(score_cols[j], spear))
            test_score = clf.score(X_test, y_test)
            print("Overall Regression Score: {:.2f}".format(test_score))
            print()

class RiahiDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels   # DataFrame
        self.score_cols = ['TopNetTree','Flex ddG','MOE','SAAMBE-3D']
        self.label_vals = self.labels[self.score_cols]


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        row_vals = self.label_vals.iloc[idx]
        input_feat = torch.FloatTensor(self.features[idx, :, :])
        label = torch.FloatTensor(row_vals)

        return input_feat, label

def create_riahi_dataloader(ab_name, chain_type, batch_size):
    dataset_path = '/data/cb/rsingh/work/antibody/ci_data/raw/riahi_sanofi/{}.h5'.format(ab_name)
    with h5py.File(dataset_path, "r") as f:
        features = torch.FloatTensor(np.array(f[chain_type]))

    df_label = pd.read_csv('/data/cb/rsingh/work/antibody/ci_data/raw/riahi_sanofi/{}_mutations_V{}.csv'.format(ab_name, chain_type))
    dataset = RiahiDataset(features, df_label)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def train_riahi(args):
    batch_size = 16
    chain_type = 'H'
    ab_name = 'm396'
    num_epochs = 300
    lr = 0.01
    device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

    # build dataloader
    dataloader = create_riahi_dataloader(ab_name, chain_type, batch_size)
    print("Size of Riahi Dataset:", len(dataloader.dataset))

    # build model, loss fn, optimizer:
    model = DesautelsPredictor(input_dim=512, mid_dim = 64, embed_dim = 256, hidden_dim=256).to(device)
    loss_fn = nn.MSELoss()
    optimizer = opt.SGD(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[200, 250], gamma=0.1)

    # load pre-trained models here:
    pretrained = AntibodyNetMultiAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    best_arch_path = '/net/scratch3/scratch3-3/chihoim/model_ckpts/best_pretrained'
    best_path = os.path.join(best_arch_path, 'bb_feat256_{}_best_newnum.pt'.format(chain_type))
    checkpoint = torch.load(best_path, map_location=device)
    pretrained.load_state_dict(checkpoint['model_state_dict'])
    pretrained.eval()

    # load WT feature
    with open('/data/cb/rsingh/work/antibody/ci_data/raw/riahi_sanofi/{}_WT_{}.p'.format(ab_name, chain_type), 'rb') as f:
        WT_feat = pickle.load(f).to(device)

    for i in range(num_epochs):
        
        model.train()
        total_loss = 0

        for k, (input_batch, label_batch) in enumerate(dataloader):

            # create the WT features batch:
            WT_feat_batch = WT_feat.repeat(input_batch.shape[0], 1, 1)

            input_batch, label_batch, WT_feat_batch = input_batch.to(device), label_batch.to(device), WT_feat_batch.to(device)

            # convert k x 6169 --> k x 256
            with torch.no_grad():
                feat, _ = pretrained(input_batch, input_batch, None, None, task=0, return2=True)
                feat_wt, _ = pretrained(WT_feat_batch, WT_feat_batch, None, None, task=0, return2=True)

            pred_batch = model(feat, feat_wt, chain_type, 'riahi')

            # update the parameters (loss, backprop, etc.)
            loss = loss_fn(pred_batch, label_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if k == len(dataloader) - 1:
                print("TRAIN Epoch {}, Batch {}, Loss: {}".format(i, k, total_loss/(k+1)))
                # writer.add_scalar('Loss/train', total_loss/(k+1), i)

        scheduler.step()

        # saving models:
        model_savepath = "/net/scratch3/scratch3-3/chihoim/model_ckpts/riahi_pretrained"
        if (i+1)%50 == 0 and model_savepath != "":
            raw_model = model.module if hasattr(model, "module") else model
            model_sum = {'epoch':i, 'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}
            savepath = os.path.join(model_savepath, 'riahi_epoch{}.pt'.format(i+1))
            torch.save(model_sum, savepath)

    # writer.flush()


def train_desautels(device_num, num_epochs, batch_size, exec_type, lr, traintest_split):
    pred_cols = ['FoldX_Average_Whole_Model_DDG', 'FoldX_Average_Interface_Only_DDG',
            'Statium', 'Sum_of_Rosetta_Flex_single_point_mutations',
            'Sum_of_Rosetta_Total_Energy_single_point_mutations']

    device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")
    chain_type = 'H'

    # load the model/optimizer
    model = DesautelsPredictor(input_dim=512, mid_dim = 64, embed_dim = 256, hidden_dim=256).to(device)
    checkpoint_model = torch.load('/net/scratch3/scratch3-3/chihoim/model_ckpts/riahi_pretrained/riahi_epoch300.pt', map_location=device)
    loss_fn = nn.MSELoss()
    optimizer = opt.SGD(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # freezing the LSTM layers
    for param in model.attention_H.parameters():
        param.requires_grad = False
    for param in model.attention_L.parameters():
        param.requires_grad = False

    # load pre-trained models here:
    pretrained = AntibodyNetMultiAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    best_arch_path = '/net/scratch3/scratch3-3/chihoim/model_ckpts/best_pretrained'
    best_path = os.path.join(best_arch_path, 'bb_feat256_{}_best_newnum.pt'.format(chain_type))
    checkpoint = torch.load(best_path, map_location=device)
    pretrained.load_state_dict(checkpoint['model_state_dict'])
    pretrained.eval()

    print("Building dataloader...")
    # load the dataloader
    train_loader_des, val_loader_des, test_loader_des = create_dataloader_desautels( 
                                                batch_size=batch_size, traintest_split = traintest_split)
    print("Done building dataloader!")

    # load WT features:
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/WT_feats/WT_{}.p'.format(chain_type), 'rb') as f:
        WT_feat = pickle.load(f).to(device)


    X_train, y_train = [],[]
    X_test, y_test = [],[]

    if exec_type == 'train':
        # with autograd.detect_anomaly():
        for i in range(num_epochs):

            # train the model
            model.train()
            total_loss = 0

            for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(train_loader_des):

                # predict the label using the model
                feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

                # create the WT features batch:
                WT_H_feat_batch = WT_feat.repeat(feat_H_batch.shape[0], 1, 1)

                # project all feature dimensions down to 256:
                with torch.no_grad():
                    feat_h, _ = pretrained(feat_H_batch, feat_H_batch, None, None, task=0, return2=True)
                    feat_wt_h, _ = pretrained(WT_H_feat_batch, WT_H_feat_batch, None, None, task=0, return2=True)

                pred_batch = model(feat_h, feat_wt_h, chain_type, 'desautels')

                # update the parameters (loss, backprop, etc.)
                loss = loss_fn(pred_batch, label_batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if k == len(train_loader_des) - 1:
                    print("TRAIN Epoch {}, Batch {}, Loss: {}".format(i, k, total_loss/(k+1)))

            scheduler.step()

            # print("********************")

            # validate the model
            model.eval()
            total_loss = 0

            for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(val_loader_des):

                # predict the label using the model
                feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

                # create the WT features batch:
                WT_H_feat_batch = WT_feat.repeat(feat_H_batch.shape[0], 1, 1)

                # project all feature dimensions down to 50:
                with torch.no_grad():
                    feat_h, _ = pretrained(feat_H_batch, feat_H_batch, None, None, task=0, return2=True)
                    feat_wt_h, _ = pretrained(WT_H_feat_batch, WT_H_feat_batch, None, None, task=0, return2=True)

                    pred_batch = model(feat_h, feat_wt_h, chain_type, 'desautels')

                # update the parameters (loss, backprop, etc.)
                loss = loss_fn(pred_batch, label_batch)
                total_loss += loss.item()

                if k == len(val_loader_des) - 1:
                    print("VAl Epoch {}, Batch {}, Loss: {}".format(i, k, total_loss/(k+1)))

            print("----------------------------------------------")



    # feed the model through the test set
    model.eval()
    losses = []
    labels_total, preds_total = [], []

    for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(test_loader_des): #, total=len(test_loader_des):

        # predict the label using the model
        feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

        # create the WT features batch:
        WT_H_feat_batch = WT_feat.repeat(feat_H_batch.shape[0], 1, 1)

        # project all feature dimensions down to 50:
        with torch.no_grad():
            feat_h, _ = pretrained(feat_H_batch, feat_H_batch, None, None, task=0, return2=True)
            feat_wt_h, _ = pretrained(WT_H_feat_batch, WT_H_feat_batch, None, None, task=0, return2=True)

            pred_batch = model(feat_h, feat_wt_h, chain_type, 'desautels')

        #-----------------------------------------------------

        labels_total.append(label_batch)
        preds_total.append(pred_batch)

        # update the parameters (loss, backprop, etc.)
        loss = loss_fn(pred_batch, label_batch)
        losses.append(loss.item())

    save_dict = dict()


    # Calculate Average Loss over the TEST SET:
    print("TEST SET Average LOSS: {}".format(sum(losses)/len(losses)))

    # calculate spearman rank for each value:
    labels_total = torch.cat(labels_total, dim=0)
    preds_total = torch.cat(preds_total, dim=0)

    for j in range(labels_total.shape[-1]):
        spear = evaluate_spearman(preds_total[:, j], labels_total[:, j])
        print("Spearman Rank Scores for {}: {}".format(pred_cols[j], spear))
        save_dict[pred_cols[j]] = (preds_total[:, j], labels_total[:, j])

    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/plots/train_riahi_infer_desautels_spl{}.p'.format(traintest_split), 'wb') as p:
        pickle.dump(save_dict, p)


    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details", type=str, choices = ["1_preprocess_m396_data", 
                    "2_create_m396_features", "3_run_m396_analysis", "4_preprocess_80R_data", "5_create_80R_features", "6_run_80R_analysis", 
                    "7_preprocess_CR3022_data", "8_create_CR3022_features", "9_run_CR3022_analysis", "ridge_regression", "train_riahi", "train_desautels"])
    parser.add_argument("--datadir", help="output directory (can set to '.')", type=str, default="/data/cb/rsingh/work/antibody/ci_data/raw/riahi_sanofi/")
    parser.add_argument("--is_baseline", help="Are we training a baseline model.", type=bool, default=False)
    parser.add_argument("--device_num", help="GPU device for computation.", type=int, default=0)
    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')


    args = parser.parse_args()
    assert args.mode is not None
    
    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}
    args.extra = extra_args
    
    pd.set_option('use_inf_as_na', True)

    if args.mode == "1_preprocess_m396_data":
        preprocess_sanofi_spreadsheet(args, "m396")

    elif args.mode == "2_create_m396_features":
        create_data_features(args, "m396")

    elif args.mode == "4_preprocess_80R_data":
        preprocess_sanofi_spreadsheet(args, "80R")

    elif args.mode == "5_create_80R_features":
        create_data_features(args, "80R")

    elif args.mode == "7_preprocess_CR3022_data":
        preprocess_sanofi_spreadsheet(args, "CR3022")

    elif args.mode == "8_create_CR3022_features":
        create_data_features(args, "CR3022")

    elif args.mode == "ridge_regression":
        ridge_regression(args)

    elif args.mode == "train_riahi":
        train_riahi(args)

    elif args.mode == "train_desautels":
        for s in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
            train_desautels(0, 100, 32, "train", lr=0.01, traintest_split=s)
            print()

    else:
        assert False
        
