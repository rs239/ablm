import os
import pickle
from tqdm import tqdm
import argparse
import random
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import h5py
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn



class MultiDataset(Dataset):
    def __init__(self, dset1, dset2, labels, device):
        self.dset1 = pad_sequence(dset1, batch_first=True)#.to('cuda:{}'.format(device))
        self.dset2 = pad_sequence(dset2, batch_first=True)#.to('cuda:{}'.format(device))
        self.labels = torch.FloatTensor(labels)#.to('cuda:{}'.format(device))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.dset1[idx,:,:], self.dset2[idx,:,:], self.labels[idx]

def my_collate(data):
    temp1, temp2, temp3 = [], [], []
    mask1, mask2 = [], []

    for x1_path, x2_path, tms in data:
        with open(x1_path, 'rb') as f1:
            # print(x1_path)
            x1_data = pickle.load(f1)
        with open(x2_path, 'rb') as f2:
            # print(x2_path)
            x2_data = pickle.load(f2)

        x1_data.cpu()
        x2_data.cpu()

        # pre-padding step 1:
        temp1.append(torch.flip(x1_data, [0]))
        temp2.append(torch.flip(x2_data, [0]))

        mask1.append(torch.zeros(x1_data.shape[0]))
        mask2.append(torch.zeros(x2_data.shape[0]))

        # post-padding(original):
        # temp1.append(x1_data)
        # temp2.append(x2_data)

        temp3.append(tms)

        del x1_data, x2_data

    out1 = pad_sequence(temp1, batch_first=True)
    out2 = pad_sequence(temp2, batch_first=True)

    out_mask1 = pad_sequence(mask1, padding_value=1, batch_first=True)
    out_mask2 = pad_sequence(mask2, padding_value=1, batch_first=True)

    # pre-padding step 2:
    out1, out2 = torch.flip(out1, [1]), torch.flip(out2, [1])
    out_mask1, out_mask2 = torch.flip(out_mask1, [1]).type(torch.bool), torch.flip(out_mask2, [1]).type(torch.bool)

    # print(out1.shape, out_mask1.shape)
    # print(out2.shape, out_mask2.shape)
    # assert False

    ########### adding an extra dimension for features
    # out1 = torch.cat((out1, torch.zeros(1, out1.shape[1], out1.shape[2])), dim=0)
    # out2 = torch.cat((out2, torch.zeros(1, out2.shape[1], out2.shape[2])), dim=0)
    # out_mask1 = torch.cat((out_mask1, torch.zeros(out_mask1.shape[0], 1)), dim=-1)
    # out_mask2 = torch.cat((out_mask2, torch.zeros(out_mask2.shape[0], 1)), dim=-1)
    ###########


    out3 = torch.Tensor(temp3)

    return out1, out2, out3, out_mask1, out_mask2


def create_dataloader_sabdab(data_dir, batch_size, emb_type, mut_type, chain_type, region):
    # parent_dir = '../data/processed/sabdab'
    parent_dir = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab'

    embed_dir = os.path.join(parent_dir, data_dir, emb_type)

    if region == 'whole':
        data_path = os.path.join(parent_dir, 'pair_scores', 'set1_{}_pair_scores_100k.csv'.format(chain_type))
    else:
        data_path = os.path.join(parent_dir, 'pair_scores', 'set1_CDR{}_{}_pair_scores_100k.csv'.format(region, chain_type))

    data = pd.read_csv(data_path)
    data = data.iloc[:100000]

    print(data.shape[0])

    data_list = []
    for idx, row in data.iterrows():
        p1_id, p2_id, tms, rmsd = row['pdb_id1'], row['pdb_id2'], row['tm_score'], row['rmsd']

        p1 = '{}_{}_{}_k100.p'.format(p1_id, mut_type, chain_type)
        p2 = '{}_{}_{}_k100.p'.format(p2_id, mut_type, chain_type)

        # original embeddings:
        # p1 = 'sabdab_{}_{}_orig.p'.format(p1_id, chain_type)
        # p2 = 'sabdab_{}_{}_orig.p'.format(p2_id, chain_type)

        emb1 = os.path.join(embed_dir, p1)
        emb2 = os.path.join(embed_dir, p2)

        data_list.append((emb1, emb2, tms))

    # randomly split dataset into train, val, test (70-15-15):
    train_idx = int(len(data_list)*0.7)
    val_idx = int(len(data_list)*0.85)

    train_loader = DataLoader(data_list[:train_idx], batch_size=batch_size, pin_memory=False,
                              num_workers = 4, collate_fn=my_collate, shuffle=True)
    val_loader = DataLoader(data_list[train_idx:val_idx], batch_size=batch_size, pin_memory=False,
                              num_workers = 4, collate_fn=my_collate, shuffle=True)
    test_loader = DataLoader(data_list[val_idx:], batch_size=batch_size, pin_memory=False,
                              num_workers = 4, collate_fn=my_collate, shuffle=True)

    return train_loader, val_loader, test_loader


def create_dataloader_libra(data_dir, batch_size, emb_type, mut_type, chain_type):
    # parent_dir = '../data/processed/libraseq'
    parent_dir = '/data/cb/rsingh/work/antibody/ci_data/processed/libraseq'

    embed_dir = os.path.join(parent_dir, data_dir, emb_type)
    data_path = os.path.join(parent_dir, 'libraseq_pairs_{}_Set1_100k.csv'.format(chain_type))

    df = pd.read_csv(data_path, keep_default_na=False)
    df = df.iloc[:100000]

    data_list = []
    for index, row in df.iterrows():
        p1 = '{}_{}_{}_k100.p'.format(row['source_seq1'], mut_type, chain_type)
        p2 = '{}_{}_{}_k100.p'.format(row['source_seq2'], mut_type, chain_type)

        # original embeddings:
        # p1 = '{}_{}_orig.p'.format(row['source_seq1'], chain_type)
        # p2 = '{}_{}_orig.p'.format(row['source_seq2'], chain_type)

        tms = float(row['score'])

        emb1 = os.path.join(embed_dir, p1)
        emb2 = os.path.join(embed_dir, p2)

        data_list.append((emb1, emb2, tms))

    train_idx = int(len(data_list)*0.7)
    val_idx = int(len(data_list)*0.85)

    train_loader = DataLoader(data_list[:train_idx], batch_size=batch_size, pin_memory=False,
                              num_workers = 4, collate_fn=my_collate, shuffle=True)
    val_loader = DataLoader(data_list[train_idx:val_idx], batch_size=batch_size, pin_memory=False,
                              num_workers = 4, collate_fn=my_collate, shuffle=True)
    test_loader = DataLoader(data_list[val_idx:], batch_size=batch_size, pin_memory=False,
                              num_workers = 4, collate_fn=my_collate, shuffle=True)

    return train_loader, val_loader, test_loader



class DesautelsDataset(Dataset):
    def __init__(self, features_H, features_L, labels):
        self.features_H = features_H
        self.features_L = features_L
        self.labels = labels   # DataFrame
        self.scores_cols = ['FoldX_Average_Whole_Model_DDG', 'FoldX_Average_Interface_Only_DDG',
                            'Statium', 'Sum_of_Rosetta_Flex_single_point_mutations',
                            'Sum_of_Rosetta_Total_Energy_single_point_mutations']
        self.label_vals = self.labels[self.scores_cols]

        # standardize label scores (x - mean) / std
        self.label_vals = (self.label_vals - self.label_vals.mean()) / self.label_vals.std()


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        row_vals = self.label_vals.iloc[idx]
        feature_H = torch.FloatTensor(self.features_H[idx, :, :])
        feature_L = torch.FloatTensor(self.features_L[idx, :, :])
        feature_id = row['Antibody_ID'] # unncessary for now
        label = torch.FloatTensor(row_vals)

        return feature_H, feature_L, label

def create_dataloader_desautels(batch_size, traintest_split = 0.2, onehot=False, embed_type='bb'):
    np.random.seed(1) # for reproducability
    dataset_path = '../data/processed/desautels'

    if onehot:
        dataset_path = os.path.join(dataset_path, 'desautels_onehot.h5')
    else:
        if embed_type in ['ours_bb', 'ours_protbert', 'ours_esm1b']:
            dataset_path = os.path.join(dataset_path, 'desautels_cdrembed_{}.h5'.format(embed_type))
        else:
            dataset_path = os.path.join(dataset_path, 'desautels_{}.h5'.format(embed_type))

    score_cols = ['FoldX_Average_Whole_Model_DDG', 'FoldX_Average_Interface_Only_DDG',
                    'Statium', 'Sum_of_Rosetta_Flex_single_point_mutations',
                    'Sum_of_Rosetta_Total_Energy_single_point_mutations']

    start = time.time()
    with h5py.File(dataset_path, "r") as f:
        dataset_H_all, dataset_L_all = f['H'], f['L']

        # load the label:
        label_path = os.path.join(dataset_path, "Desautels_m396_seqs.csv")
        invalid_idx_path = os.path.join(dataset_path, "invalid_idx.p")
        labels_df = pd.read_csv(label_path)

        # identify all rows with NaN score values, and remove them
        extra_invalid = labels_df[labels_df['Statium'].isnull()].index.tolist()

        with open(invalid_idx_path, 'rb') as f:
            invalid_idx2 = pickle.load(f)

        invalid_idx = sorted(list(set(extra_invalid) | set(invalid_idx2)))

        all_idxs = list(range(len(dataset_H_all)))
        valid_idxs = [i for i in all_idxs if i not in invalid_idx]

        # FOR TESTING ONLY!
        # valid_idxs = valid_idxs[:1000]
        
        if embed_type == 'bb':
            dataset_H_all = dataset_H_all[:, :, -2200:]
            dataset_L_all = dataset_L_all[:, :, -2200:]

        dataset_H_all = dataset_H_all[valid_idxs, :, :]
        dataset_L_all = dataset_L_all[valid_idxs, :, :]
        labels_df = labels_df.iloc[valid_idxs]
        labels_df = labels_df.reset_index(drop=True)

        # check for NANs:
        if np.any(np.isnan(dataset_H_all)):
            print("Dataset H contains NaNs!")
            raise ValueError
        if np.any(np.isnan(dataset_L_all)):
            print("Dataset L contains NaNs!")
            raise ValueError

        # valid_idx = valid_idx[:1000] # For testing. COMMENT OUT when running actual experiment!
        # dataset_H_all, dataset_L_all, labels_df = dataset_H_all[:2000], dataset_L_all[:2000], labels_df.head(2000)
        print(f"Dataset Size: {len(dataset_H_all)}. (1000 is for testing)")

        assert dataset_H_all.shape[0] == dataset_L_all.shape[0] == labels_df.shape[0]

        # calculate the splits here:
        dataset_len = dataset_H_all.shape[0]
        train_size = int(traintest_split * dataset_len)
        all_idx = list(range(dataset_len))
        np.random.shuffle(all_idx)
        trainval_idx = all_idx[:train_size]
        trainval_split = int(0.9*train_size)

        train_idx, val_idx = sorted(trainval_idx[:trainval_split]), sorted(trainval_idx[trainval_split:])
        test_idx = sorted(all_idx[train_size:])
        
        # split the datasets into train / val / test
        features_H_train, features_L_train = dataset_H_all[train_idx, :, :], dataset_L_all[train_idx, :, :]
        labels_df_train = labels_df.iloc[train_idx].reset_index(drop=True)
        features_H_val, features_L_val = dataset_H_all[val_idx, :, :], dataset_L_all[val_idx, :, :]
        labels_df_val = labels_df.iloc[val_idx].reset_index(drop=True)
        features_H_test, features_L_test = dataset_H_all[test_idx, :, :], dataset_L_all[test_idx, :, :]
        labels_df_test = labels_df.iloc[test_idx].reset_index(drop=True)

        end = time.time()
        print("Time it took to load the H5 files: {:.2f} seconds".format(end-start))

        train_data = DesautelsDataset(features_H_train, features_L_train, labels_df_train)
        val_data = DesautelsDataset(features_H_val, features_L_val, labels_df_val)
        test_data = DesautelsDataset(features_H_test, features_L_test, labels_df_test)

        print("Train/Test split: {}, {}".format(len(train_data)+len(val_data), len(test_data)))



    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers = 4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers = 4)

    return train_loader, val_loader, test_loader