import os
import argparse
from tqdm import tqdm
import glob
import time
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import sys
# sys.path.append('..')

from ..base_config import *
# import base_config
from ..abmap.utils import get_boolean_mask, find_sequence
from mutate import generate_mutation
from embed import embed_sequence, reload_models_to_device #, load_model
from model import MultiTaskLossWrapper

import torch

print("all imports were successful!")
assert False



def create_covabdab_embeddings(args):
    reload_models_to_device(args.device_num)

    df = pd.read_csv("/data/cb/rsingh/work/antibody/ci_data/raw/covabdab/neutralize_sarscov2.csv")
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    out_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/covabdab/"
    chain_type = 'H'

    # load valid ids:
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/covabdab/valid_ids.txt', 'r') as f:
        valid_ids_list = f.read().splitlines()

    k=100
    invalids = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # if index < 1710:
        #     continue

        seq_id = row['Name']
        if seq_id not in valid_ids_list:
            continue

        seq = row['VH or VHH'] if chain_type == 'H' else row['VL']

        if ' ' in seq:
            invalids.append(seq_id)
            continue

        prot_embed = ProteinEmbedding(seq, chain_type, dev=args.device_num)
        try:
            prot_embed.create_cdr_mask()
            # print(prot_embed.cdr_mask)
            # print("HAW"+prot_embed.seq+"HEE")
        except:
            invalids.append(seq_id)
            continue

        for embed_type in ['beplerberger']:
            out_path = os.path.join(out_dir, 'cdrembed_maskaug4', embed_type)
            prot_embed.embed_seq(embed_type=embed_type)
            kmut_matr_h = prot_embed.create_kmut_matrix(num_muts=k, embed_type=embed_type)
            cdr_embed = prot_embed.create_cdr_embedding(kmut_matr_h, sep=False, mask=True)

            # save the embedding first:
            with open(os.path.join(out_path, "{}_cat2_{}_k{}.p".format(seq_id, chain_type, k)), 'wb') as fc:
                pickle.dump(cdr_embed, fc)

    print(invalids)



def create_covabdab_features(args):
    reload_models_to_device(args.device_num)

    df = pd.read_csv("/data/cb/rsingh/work/antibody/ci_data/raw/covabdab/neutralize_sarscov2.csv")
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    covabdab_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/covabdab"
    embed_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/covabdab/cdrembed_maskaug4/{}".format(args.embed_type)
    out_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/covabdab/covabdab_features_120122/{}".format(args.embed_type)

    # Load our pretrained AbNet Model
    from model import AntibodyNetMultiAttn
    model = AntibodyNetMultiAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    pretrained_path = "../model_ckpts/091522_bb_newnum_{}/beplerberger_epoch50.pt".format(args.chain_type)
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Loaded the pre-trained model!")

    k=100

    with open(os.path.join(covabdab_dir, 'invalids.txt'), 'r') as f:
        invalids_list = f.read().splitlines()

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        seq_id, seq_h = row['Name'], row['VH or VHH']

        # if the sequence id is invalid, skip
        if seq_id in invalids_list:
            continue

        # prot_embed = ProteinEmbedding(seq_h, 'H', dev=1)
        # prot_embed.create_cdr_mask()

        # --------------------------------------
        # FOR PROTBERT ONLY:

        # embed_type = args.embed_type
        # prot = ProteinEmbedding(seq_h, 'H')
        # try:
        #     prot.embed_seq(embed_type = args.embed_type)
        # except:
        #     continue
        # with open(os.path.join(out_dir, embed_type, '{}_H_orig.p'.format(seq_id)), 'wb') as f:
        #     pickle.dump(prot.embedding, f)
        # continue
        # --------------------------------------

        # load the embedding:
        with open(os.path.join(embed_dir, "{}_cat2_H_k100.p".format(seq_id)), 'rb') as p:
            cdr_embed = pickle.load(p)

        cdr_embed = cdr_embed.to(device)
        cdr_embed = torch.unsqueeze(cdr_embed, dim=0) # create the batch dimension (singleton)

        # create and save STRUCTURE specific feature:
        with torch.no_grad():
            feat_task, _ = model(cdr_embed, cdr_embed, None, None, 0, task_specific=True)
        assert feat_task.shape[-1] == 512
        with open(os.path.join(out_dir, "{}_struc_H.p".format(seq_id)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_task), p)

        # create and save FUNCTION specific feature:
        with torch.no_grad():
            feat_task, _ = model(cdr_embed, cdr_embed, None, None, 1, task_specific=True)
        assert feat_task.shape[-1] == 512
        with open(os.path.join(out_dir, "{}_func_H.p".format(seq_id)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_task), p)

        # create and save INTERMEDIATE n' x k feature, before Transformer:
        with torch.no_grad():
            feat_interm, _ = model(cdr_embed, cdr_embed, None, None, None, return2=True)
        assert feat_interm.shape[-1] == 256
        with open(os.path.join(out_dir, "{}_interm_H.p".format(seq_id)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_interm), p)


if __name__ == "__main__":
    print("running covabdab analysis!")